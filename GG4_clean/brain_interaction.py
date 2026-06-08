
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, solve, inv, lstsq
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from GG4 import Brain
from BMI_and_Hand import BMI_and_Hand
from typing import Tuple
from scipy.optimize import minimize
import time

'''

In neuroscience some typical control objectives are:

Suppress seizure activity — drive high-amplitude oscillations to near zero
Maintain a target firing rate — hold neural population activity at a desired level*    -> can proabably try to implement this.
Desynchronise — break up synchronised oscillations (relevant for Parkinson's)
Entrain to a rhythm — lock neural activity to a target frequency
State switching — drive the network from one attractor state to another


'''


RANDOM_SEED    = 42
T_IDENT        = 500     # identification experiment length
T_CONTROL      = 500    # closed-loop control length        # estimating with longer controlled period to see effects of parameter drift
LATENT_DIM     = 6   # latent state dimension for LDS fit
PRBS_PERIOD    = 8       # PRBS switching period (timesteps)
SSI_HORIZON    = 20     # block rows for N4SID

# LQR cost matrices — tune these
Q_LQR = np.eye(LATENT_DIM) * 0.01  # state cost
R_LQR = np.eye(2) * 2.0          # input cost (2 inputs)

# -----------------------------------------------------------------------------
# STAGE 1: PRBS identification experiment
# -----------------------------------------------------------------------------


#prbs is like white noise so should hit all modes of the system
def generate_prbs(T, p, period=8, seed=0) ->np.ndarray:
    """Generate PRBS signal switching every `period` steps, values in [0,1]."""
    rng = np.random.default_rng(seed)
    U = np.zeros((T, p))
    current = rng.integers(0, 2, size=p).astype(float)
    for t in range(T):
        if t % period == 0:
            current = rng.integers(0, 2, size=p).astype(float)
        U[t] = current
    return U

def run_identification_experiment(brain, T, p, prbs_period=8) -> Tuple[np.ndarray,np.ndarray]:
    """Run open-loop PRBS experiment, collect observations."""
    U_ident = generate_prbs(T, p, period=prbs_period)
    Y_ident = np.zeros((T, 16))   # 16 observation channels

    for t in range(T):
        Y_ident[t] = np.array(brain.measure())
        brain.next_state(U_ident[t])

    return U_ident, Y_ident


# -----------------------------------------------------------------------------
# STAGE 2: Deterministic N4SID with known inputs
# -----------------------------------------------------------------------------

def deterministic_n4sid(Y, U, n, i=15) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    N4SID with known inputs: jointly identify A, B, C.

    Parameters
    ----------
    Y : (T, m)  observations
    U : (T, p)  known inputs
    n : int     latent state dimension
    i : int     block rows (horizon)

    Returns
    -------
    A : (n, n)
    B : (n, p)
    C : (m, n)
    """
    T, m = Y.shape
    _, p = U.shape

    cols = T - 2 * i + 1
    if cols <= 0:
        raise ValueError(f"T={T} too short for i={i}")

    # Build block Hankel matrices for Y and U
    def block_hankel(X, i, cols):
        d = X.shape[1]
        H = np.zeros((2 * i * d, cols))
        for k in range(2 * i):
            H[k*d:(k+1)*d, :] = X[k:k+cols, :].T
        return H

    HY = block_hankel(Y, i, cols)   # (2im, cols)
    HU = block_hankel(U, i, cols)   # (2ip, cols)

    Y_past   = HY[:i*m, :]
    Y_future = HY[i*m:, :]
    U_past   = HU[:i*p, :]
    U_future = HU[i*p:, :]

    # Stack past: [Y_past; U_past; U_future]
    W_past = np.vstack([Y_past, U_past, U_future])

    # Oblique projection removing input contribution
    # Project Y_future onto W_past, removing U_future
    # Standard N4SID: O_i = Y_future / U_future (along W_past)
    combined = np.vstack([W_past, U_future])
    X_proj, _, _, _ = lstsq(combined.T, Y_future.T, rcond=None)
    O_i = X_proj.T @ combined - X_proj.T[:, -i*p:] @ U_future

    # SVD for observability matrix
    Ug, s, _ = svd(O_i, full_matrices=False)
    Γ = Ug[:, :n] @ np.diag(np.sqrt(s[:n]))

    # C from first m rows
    C = Γ[:m, :]

    # A via shift invariance
    Γ_up   = Γ[:(i-1)*m, :]
    Γ_down = Γ[m:, :]
    A, _, _, _ = lstsq(Γ_up, Γ_down, rcond=None)

    # Stabilise A
    rho = np.abs(np.linalg.eigvals(A)).max()
    if rho >= 0.99:
        A = A * (0.99 / rho)

    # Normalise C columns
    col_norms = np.linalg.norm(Γ, axis=0, keepdims=True)
    Γ = Γ / np.maximum(col_norms, 1e-10)
    C = Γ[:m, :]

    # Estimate initial states via least squares
    # x_{t+1} = A x_t + B u_t  =>  stack and solve for B
    # Use the identified C to get state estimates: x_hat = C^+ y
    C_pinv = np.linalg.pinv(C)
    x_hat = (C_pinv @ Y.T).T   # (T, n) rough state estimates

    # B via regression: x_{t+1} - A x_t = B u_t
    residuals = x_hat[1:] - (A @ x_hat[:-1].T).T   # (T-1, n)
    # Min norm: B = residuals.T @ U[:-1] @ (U[:-1].T @ U[:-1])^{-1}
    B, _, _, _ = lstsq(U[:-1], residuals, rcond=None)
    B = B.T   # (n, p)

    return A, B, C


# -----------------------------------------------------------------------------
# STAGE 3: LQR design
# -----------------------------------------------------------------------------

def design_lqr(A, B, Q, R):
    """
    Discrete-time LQR via DARE.
    Returns gain K such that u = -K x minimises Σ (x^T Q x + u^T R u).
    """
    try:
        P = solve_discrete_are(A, B, Q, R)
        K = inv(R + B.T @ P @ B) @ B.T @ P @ A
    except Exception as e:
        print(f"DARE failed ({e}), using simple gain")
        K = np.zeros((B.shape[1], A.shape[0]))
    return K


# -----------------------------------------------------------------------------
# Adaptive state Estimation Classes
# -----------------------------------------------------------------------------

class OnlineKalmanFilter:
    """Causal Kalman filter for real-time state estimation."""

    #kalman filter is a special case of the luenburger observer with optimally chosen gain

    def __init__(self, A, B, C, Q, R):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        n = A.shape[0]
        self.x = np.zeros(n)
        self.P = np.eye(n) * 1.0

    def update(self, y, u):
        """One step: predict then update. Returns estimated state."""
      
        x_pred = self.A @ self.x + self.B @ u
        P_pred = self.A @ self.P @ self.A.T + self.Q

       
        inn = y - self.C @ x_pred
        S   = self.C @ P_pred @ self.C.T + self.R
        S   = 0.5 * (S + S.T)

        K = P_pred @ self.C.T @ inv(S)
        self.x = x_pred + K @ inn
        self.P = (np.eye(len(self.x)) - K @ self.C) @ P_pred
        self.P = 0.5 * (self.P + self.P.T)

        return self.x.copy()
    
#----------------------------------------------------------------
#            adaptive kalman filter that has adative state estimation of B using RLS, also anneals Q/R ratio
#--------------------------------------------------------------------------
class AdaptiveKalmanFilter:
    """
    Kalman filter with RLS on B and annealing Q/R ratio.
    As B estimate improves (prediction error decreases), 
    Q/R ratio decreases — filter trusts model more.
    """
    def __init__(self, A, B, C, Q, R, lam=0.99, anneal_rate=0.995, min_QR_ratio=0.01):
        '''
        params
        ----------
        lam -> controls forgetting factor ( how much weight we give to old data as compared to new data when estimating B)

        anneal_rate -> rate that Q/R ratio decrease, rate of increase in confidence in model

        min_QR_ratio -> to stop Q collapsing and the model completely ignoring process noise.
        
        
        '''
        self.A = A
        self.B = B.copy()
        self.C = C
        self.Q = Q.copy()
        self.R = R.copy()


        self.Q_base = Q.copy()
        self.R_base = R.copy()   # reference R to scale against
        
        n, p = B.shape
        self.x = np.zeros(n)
        self.P = np.eye(n)
        
        # RLS state for B
        self.lam = lam                          # forgetting factor
        self.P_rls = np.eye(p) * 100.0          # RLS covariance — starts large (uncertain)
        
        # Annealing state
        self.anneal_rate  = anneal_rate         # how fast Q/R shrinks per step
        self.min_QR_ratio = min_QR_ratio        # floor on Q/R
        self.qr_ratio     = 1.0                 # starts at 1 — equal trust
        
        # Prediction error tracking for confidence estimate
        self.pred_error_ema = None              # exponential moving average
        self.pred_error_decay = 0.99

    def update(self, y, u):
        """One step: RLS update on B, anneal Q, Kalman predict+update."""

        # -- 1. Predict ----------------------------------------------------
        x_pred = self.A @ self.x + self.B @ u
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # -- 2. Innovation -------------------------------------------------
        inn = y - self.C @ x_pred
        S   = self.C @ P_pred @ self.C.T + self.R
        S   = 0.5 * (S + S.T)

        # -- 3. Kalman update ----------------------------------------------
        K = P_pred @ self.C.T @ inv(S)
        self.x = x_pred + K @ inn
        self.P = (np.eye(len(self.x)) - K @ self.C) @ P_pred
        self.P = 0.5 * (self.P + self.P.T)

        # -- 4. RLS update on B --------------------------------------------
        # Residual: what the model didn't predict
        # x_new ≈ A x_old + B u  →  residual = x_new - A x_old
        # Use filtered state as proxy for true state
        residual = self.x - self.A @ (self.x - K @ inn)   # (n,)

        # RLS gain
        Pu = self.P_rls @ u                                # (p,)
        denom = self.lam + u @ Pu
        K_rls = Pu / denom                                 # (p,)

        # B update: each row of B updated independently
        B_error = residual - self.B @ u                    # (n,)
        self.B = self.B + np.outer(B_error, K_rls)         # (n, p)

        # RLS covariance update
        self.P_rls = (self.P_rls - np.outer(Pu, Pu) / denom) / self.lam

        # -- 5. Track prediction error for confidence ----------------------
        pred_err = float(np.linalg.norm(inn))
        if self.pred_error_ema is None:
            self.pred_error_ema = pred_err
        else:
            self.pred_error_ema = (self.pred_error_decay * self.pred_error_ema 
                                   + (1 - self.pred_error_decay) * pred_err)

        # -- 6. Anneal Q/R ratio -------------------------------------------
        # As prediction error decreases, anneal faster
        # If error is growing, slow down annealing (model getting worse)
        self.qr_ratio = max(
            self.min_QR_ratio,
            self.qr_ratio * self.anneal_rate
        )
        self.Q = self.Q_base * self.qr_ratio

        return self.x.copy()

    @property
    def current_B(self):
        return self.B.copy()

    @property
    def confidence(self):
        """0 = no confidence, 1 = full confidence in model."""
        return 1.0 - self.qr_ratio / 1.0
    
#----------------------------------------------------
# online EM for adaptive state estimation
#---------------------------------------------------
class OnlineEM:
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        window:    int   = 50,
        em_iters:  int   = 3,
        update_A:  bool  = False,
        update_C:  bool  = False,
    ):
        self.A = A.copy()
        self.B = B.copy()
        self.C = C.copy()
        self.Q = Q.copy()
        self.R = R.copy()

        n = A.shape[0]
        self.x = np.zeros(n)
        self.P = np.eye(n)

        self.window    = window
        self.em_iters  = em_iters
        self.update_A  = update_A
        self.update_C  = update_C

        self.Y_buf = None
        self.U_buf = None
        self.t     = 0

        self.x_window_start = np.zeros(n)
        self.P_window_start = np.eye(n)

        self._kf = OnlineKalmanFilter(A, B, C, Q, R)

    def update(self, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        m = len(y)
        p = len(u)

        if self.Y_buf is None:
            self.Y_buf = np.zeros((self.window, m))
            self.U_buf = np.zeros((self.window, p))

        if self.t % self.window == 0:
            self.x_window_start = self._kf.x.copy()
            self.P_window_start = self._kf.P.copy()

        idx = self.t % self.window
        self.Y_buf[idx] = y
        self.U_buf[idx] = u
        self.t += 1

        self._kf.A = self.A
        self._kf.B = self.B
        self._kf.C = self.C
        self._kf.Q = self.Q
        self._kf.R = self.R
        x_hat   = self._kf.update(y, u)
        self.x  = x_hat
        self.P  = self._kf.P.copy()

        if self.t >= self.window and self.t % self.window == 0:
            self._run_em_step()

        return x_hat

    def _run_em_step(self):
        idx = self.t % self.window
        if idx == 0:
            Y_win = self.Y_buf.copy()
            U_win = self.U_buf.copy()
        else:
            Y_win = np.vstack([self.Y_buf[idx:], self.Y_buf[:idx]])
            U_win = np.vstack([self.U_buf[idx:], self.U_buf[:idx]])
            
        Y_T = Y_win.T
        U_T = U_win.T
        T_w = self.window
        m   = Y_T.shape[0]
        n   = self.A.shape[0]
        p   = self.B.shape[1]

        A, B, C, Q, R = (
            self.A.copy(), self.B.copy(), self.C.copy(),
            self.Q.copy(), self.R.copy(),
        )

        for _ in range(max(1, self.em_iters)):

            # -- forward Kalman -------------------------------------------
            x_f = np.zeros((T_w, n))
            P_f = np.zeros((T_w, n, n))
            x   = self.x_window_start.copy()
            P   = self.P_window_start.copy()
            ll  = 0.0

            for t in range(T_w):
                x_pred = A @ x + B @ U_T[:, t]
                P_pred = A @ P @ A.T + Q
                inn    = Y_T[:, t] - C @ x_pred
                S      = C @ P_pred @ C.T + R
                S      = 0.5 * (S + S.T)
                K      = P_pred @ C.T @ inv(S)
                x      = x_pred + K @ inn
                P      = (np.eye(n) - K @ C) @ P_pred
                P      = 0.5 * (P + P.T)
                x_f[t] = x
                P_f[t] = P
                sign, logdet = np.linalg.slogdet(S)
                ll += -0.5 * (logdet + inn @ solve(S, inn) + m * np.log(2 * np.pi))

            # -- backward RTS ---------------------------------------------
            x_s     = np.zeros((T_w, n))
            P_s     = np.zeros((T_w, n, n))
            P_cross = np.zeros((T_w - 1, n, n))
            x_s[-1] = x_f[-1]
            P_s[-1] = P_f[-1]

            for t in range(T_w - 2, -1, -1):
                x_pred     = A @ x_f[t] + B @ U_T[:, t + 1]
                P_pred     = A @ P_f[t] @ A.T + Q
                P_pred     = 0.5 * (P_pred + P_pred.T)
                G          = P_f[t] @ A.T @ inv(P_pred)
                x_s[t]     = x_f[t] + G @ (x_s[t + 1] - x_pred)
                P_s[t]     = P_f[t] + G @ (P_s[t + 1] - P_pred) @ G.T
                P_s[t]     = 0.5 * (P_s[t] + P_s[t].T)
                P_cross[t] = P_s[t + 1] @ G.T

            # -- sufficient statistics ------------------------------------
            Exx      = sum(P_s[t] + np.outer(x_s[t], x_s[t]) for t in range(T_w))
            Exx_past = sum(P_s[t] + np.outer(x_s[t], x_s[t]) for t in range(T_w - 1))
            Exx_fut  = sum(P_s[t+1] + np.outer(x_s[t+1], x_s[t+1]) for t in range(T_w - 1))
            Exx_lag  = sum(P_cross[t] + np.outer(x_s[t+1], x_s[t]) for t in range(T_w - 1))

            # -- B update -------------------------------------------------
            R_xu = sum(np.outer(x_s[t+1] - A @ x_s[t], U_T[:, t]) for t in range(T_w - 1))
            UUt  = U_T[:, :-1] @ U_T[:, :-1].T + 1e-8 * np.eye(p)
            B    = R_xu @ inv(UUt)

            # -- A update (optional) --------------------------------------
            if self.update_A:
                BU_xt = sum(np.outer(B @ U_T[:, t], x_s[t]) for t in range(T_w - 1))
                A_new = (Exx_lag - BU_xt) @ inv(Exx_past)
                rho   = np.abs(np.linalg.eigvals(A_new)).max()
                if rho < 0.99:
                    A = A_new

            # -- C update (optional) --------------------------------------
            if self.update_C:
                Eyx      = sum(np.outer(Y_T[:, t], x_s[t]) for t in range(T_w))
                C        = Eyx @ inv(Exx)
                col_norms = np.linalg.norm(C, axis=0, keepdims=True)
                C        = C / np.maximum(col_norms, 1e-10)

            # -- Q update -------------------------------------------------
            BU     = B @ U_T[:, :-1]
            EBU_xt = sum(np.outer(BU[:, t], x_s[t]) for t in range(T_w - 1))
            Tp     = T_w - 1
            Q_new  = (
                Exx_fut / Tp
                - A @ Exx_lag.T / Tp
                - Exx_lag @ A.T / Tp
                + A @ Exx_past @ A.T / Tp
                + B @ U_T[:, :-1] @ U_T[:, :-1].T @ B.T / Tp
                + (EBU_xt @ A.T + A @ EBU_xt.T) / Tp
            )
            Q_new        = 0.5 * (Q_new + Q_new.T)
            eigv, eigvec = np.linalg.eigh(Q_new)
            y_var        = float(np.mean(np.diag(Y_T @ Y_T.T / T_w)))
            eigv         = np.clip(eigv, 1e-8, y_var)
            Q            = eigvec @ np.diag(eigv) @ eigvec.T

            # -- R update -------------------------------------------------
            Eyy          = Y_T @ Y_T.T / T_w
            R_new        = Eyy - C @ Exx @ C.T / T_w
            R_new        = 0.5 * (R_new + R_new.T)
            eigv, eigvec = np.linalg.eigh(R_new)
            R            = eigvec @ np.diag(np.maximum(eigv, 1e-8)) @ eigvec.T

        self.A, self.B, self.C, self.Q, self.R = A, B, C, Q, R

    @property
    def current_B(self) -> np.ndarray:
        return self.B.copy()

    @property
    def confidence(self) -> float:
        return float(np.trace(self.Q))


def refine_with_em(Y, U, A_init, B_init, C_init, n_iter=30):
    """
    EM refinement of N4SID estimates using known inputs.
    This is easier than blind EM — inputs are known so B update is exact.
    """
    from scipy.linalg import solve_discrete_lyapunov

    m, T = Y.T.shape
    n = A_init.shape[0]
    p = B_init.shape[1]

    A, B, C = A_init.copy(), B_init.copy(), C_init.copy()
    Q = np.eye(n) * 0.1
    R = np.eye(m) * 0.5
    mu0 = np.zeros(n)
    P0 = solve_discrete_lyapunov(A, Q)

    Y_T = Y.T   # (m, T)
    U_T = U.T   # (p, T)

    ll_hist = []

    for it in range(n_iter):
        # -- E-step: RTS smoother with known inputs -----------------------
        # Forward Kalman
        x_f = np.zeros((T, n))
        P_f = np.zeros((T, n, n))
        x = mu0.copy()
        P = P0.copy()
        ll = 0.0

        for t in range(T):
            x_pred = A @ x + B @ U_T[:, t]
            P_pred = A @ P @ A.T + Q
            inn = Y_T[:, t] - C @ x_pred
            S = C @ P_pred @ C.T + R
            S = 0.5 * (S + S.T)
            K = P_pred @ C.T @ np.linalg.inv(S)
            x = x_pred + K @ inn
            P = (np.eye(n) - K @ C) @ P_pred
            P = 0.5 * (P + P.T)
            x_f[t] = x
            P_f[t] = P
            sign, logdet = np.linalg.slogdet(S)
            ll += -0.5 * (logdet + inn @ np.linalg.solve(S, inn) + m * np.log(2 * np.pi))

        ll_hist.append(ll)

        # Backward RTS
        x_s = np.zeros((T, n))
        P_s = np.zeros((T, n, n))
        P_cross = np.zeros((T-1, n, n))
        x_s[-1] = x_f[-1]
        P_s[-1] = P_f[-1]

        for t in range(T-2, -1, -1):
            x_pred = A @ x_f[t] + B @ U_T[:, t+1]
            P_pred = A @ P_f[t] @ A.T + Q
            P_pred = 0.5 * (P_pred + P_pred.T)
            G = P_f[t] @ A.T @ np.linalg.inv(P_pred)
            x_s[t] = x_f[t] + G @ (x_s[t+1] - x_pred)
            P_s[t] = P_f[t] + G @ (P_s[t+1] - P_pred) @ G.T
            P_s[t] = 0.5 * (P_s[t] + P_s[t].T)
            P_cross[t] = P_s[t+1] @ G.T

        # -- M-step -------------------------------------------------------
        Exx      = sum(P_s[t] + np.outer(x_s[t], x_s[t]) for t in range(T))
        Exx_past = sum(P_s[t] + np.outer(x_s[t], x_s[t]) for t in range(T-1))
        Exx_fut  = sum(P_s[t+1] + np.outer(x_s[t+1], x_s[t+1]) for t in range(T-1))
        Exx_lag  = sum(P_cross[t] + np.outer(x_s[t+1], x_s[t]) for t in range(T-1))

        # A update: accounts for known inputs
        BU_xt = sum(np.outer(B @ U_T[:, t], x_s[t]) for t in range(T-1))
        A = (Exx_lag - BU_xt) @ np.linalg.inv(Exx_past)

        # B update: exact since inputs are known
        R_xu = sum(np.outer(x_s[t+1] - A @ x_s[t], U_T[:, t]) for t in range(T-1))
        UUt = U_T[:, :-1] @ U_T[:, :-1].T + 1e-8 * np.eye(p)
        B = R_xu @ np.linalg.inv(UUt)

        # C update
        Eyx = sum(np.outer(Y_T[:, t], x_s[t]) for t in range(T))
        C = Eyx @ np.linalg.inv(Exx)

        # Q update
        BU = B @ U_T[:, :-1]
        EBU_xt = sum(np.outer(BU[:, t], x_s[t]) for t in range(T-1))
        Tp = T - 1
        Q_new = (Exx_fut/Tp - A @ Exx_lag.T/Tp - Exx_lag @ A.T/Tp
                 + A @ Exx_past @ A.T/Tp + B @ U_T[:,:-1] @ U_T[:,:-1].T @ B.T/Tp
                 + (EBU_xt @ A.T + A @ EBU_xt.T)/Tp)
        Q_new = 0.5 * (Q_new + Q_new.T)
        eigv, eigvec = np.linalg.eigh(Q_new)
        Q = eigvec @ np.diag(np.clip(eigv, 1e-8, None)) @ eigvec.T

        # R update
        Eyy = Y_T @ Y_T.T / T
        R_new = Eyy - C @ Exx @ C.T / T
        R_new = 0.5 * (R_new + R_new.T)
        eigv, eigvec = np.linalg.eigh(R_new)
        R = eigvec @ np.diag(np.clip(eigv, 1e-8, None)) @ eigvec.T

        # Stabilise A
        rho = np.abs(np.linalg.eigvals(A)).max()
        if rho >= 0.99:
            A = A * (0.99 / rho)

        print(f"  EM iter {it+1:3d} | LL = {ll:.2f}")

    return A, B, C, Q, R, ll_hist

# -----------------------------------------------------------------------------
# STAGE 5: Closed-loop control experiment
# -----------------------------------------------------------------------------

def run_lqr(brain, A, B, C, K, Q_kal, R_kal, T, x_target=None):
    """
    Closed-loop LQR control.
    u_t = clip(-K (x_hat_t - x_target), 0, 1)
    """
    p = B.shape[1]
    if x_target is None:
        x_target = np.zeros(A.shape[0])

    kf = OnlineKalmanFilter(A, B, C, Q_kal, R_kal)

    Y_ctrl = np.zeros((T, C.shape[0]))
    U_ctrl = np.zeros((T, p))
    X_est  = np.zeros((T, A.shape[0]))

    u = np.zeros(p)
    for t in range(T):
        y = np.array(brain.measure())
        Y_ctrl[t] = y

        # Estimate state
        x_hat = kf.update(y, u)
        X_est[t] = x_hat

        # LQR control law
        u = -K @ (x_hat - x_target)
        u = np.clip(u, 0, 1)   # hard constraint
        U_ctrl[t] = u

        brain.next_state(u)

    return Y_ctrl, U_ctrl, X_est


def run_uncontrolled(brain, T):
    """Open-loop baseline — no input."""
    m = len(brain.measure())
    Y_open = np.zeros((T, m))
    for t in range(T):
        Y_open[t] = np.array(brain.measure())
        brain.next_state()
    return Y_open

#===========================================================

#   control policy to make it reach a target (setpoint regulation) -> unused at the moment

#===========================================================


def compute_steady_state(A, B, C, y_target):
    """
    Find (x_ss, u_ss) such that x_ss = A x_ss + B u_ss and C x_ss = y_target.
    Solves the combined linear system:
        [I-A   -B ] [x_ss]   [0       ]
        [C      0 ] [u_ss] = [y_target]
    """
    n, p = B.shape[1], B.shape[1]
    n = A.shape[0]
    p = B.shape[1]
    m = C.shape[0]

    # Stack the system
    M = np.block([
        [np.eye(n) - A,  -B          ],
        [C,               np.zeros((m, p))]
    ])
    rhs = np.concatenate([np.zeros(n), y_target])

    # Minimum norm solution
    sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
    x_ss = sol[:n]
    u_ss = sol[n:]

    return x_ss, u_ss


def run_lqr_controlled_with_target(brain, A, B, C, K, Q_kal, R_kal, T, y_target_arr):
    if y_target_arr.ndim == 1:
        get_target = lambda t: y_target_arr
    else:
        get_target = lambda t: y_target_arr[t, :]

    p = B.shape[1]
    kf = OnlineKalmanFilter(A, B, C, Q_kal, R_kal)
    Y_ctrl = np.zeros((T, C.shape[0]))
    U_ctrl = np.zeros((T, p))
    X_est  = np.zeros((T, A.shape[0]))

    y_prev = None
    x_ss, u_ss = np.zeros(A.shape[0]), np.zeros(p)
    u = np.zeros(p)

    for t in range(T):
        y_t = get_target(t)

        # Recompute steady state only when target changes
        if y_prev is None or not np.allclose(y_t, y_prev):
            x_ss, u_ss = compute_steady_state(A, B, C, y_t)  # compute states and inputs at steady state on our target
            u_ss = np.clip(u_ss, 0, 1)
            y_prev = y_t.copy()

        y = np.array(brain.measure())
        Y_ctrl[t] = y
        x_hat = kf.update(y, u)
        X_est[t] = x_hat

        u = u_ss - K @ (x_hat - x_ss)
        u = np.clip(u, 0, 1)
        U_ctrl[t] = u

        brain.next_state(u)

    return Y_ctrl, U_ctrl, X_est


#----------------------------------------------------------------------------------

# MPC ->finite-horizon optimisation 

#---------------------------------------------------------------------------------

def build_prediction_matrices(A, B, C, N) -> Tuple[np.ndarray,np.ndarray]:  # need these for MPC quadratic problem framing
    """
    Build F (free response) and Phi (forced response) matrices.
    Y = F x0 + Phi U
    
    Y   : (N*m,)   stacked predicted outputs
    U   : (N*p,)   stacked inputs
    F   : (N*m, n)
    Phi : (N*m, N*p)  lower block triangular
    """
    n = A.shape[0]
    m = C.shape[0]
    p = B.shape[1]

    F   = np.zeros((N * m, n))
    Phi = np.zeros((N * m, N * p))

    A_pow = np.eye(n)
    for k in range(N):
        A_pow = A_pow @ A if k > 0 else A
        F[k*m:(k+1)*m, :] = C @ A_pow

        for j in range(k + 1):
            # Phi[k, j] = C A^{k-j} B
            A_kj = np.linalg.matrix_power(A, k - j)
            Phi[k*m:(k+1)*m, j*p:(j+1)*p] = C @ A_kj @ B

    return F, Phi

def mpc_step_qp(A, B, C, F, Phi, x0, y_target_seq, Q_mpc, Q_terminal, R_mpc, N): # solves mpc as a quadratic program with box constraints on inputs instead of gradient based
    """
    Solve MPC as a QP.
    
    min  1/2 U^T H U + f^T U
    s.t. 0 <= U <= 1
    
    y_target_seq : (N, m)  target over horizon
    """
    n = A.shape[0]
    m = C.shape[0]
    p = B.shape[1]

    # Stack Q_mpc into block diagonal  (N*m, N*m)
    Q_big = np.kron(np.eye(N), Q_mpc)   # block diagonal
    Q_big[(N-1)*m : N*m, (N-1)*m : N*m] = Q_terminal
    R_big = np.kron(np.eye(N), R_mpc)

    # Reference vector  (N*m,)
    r = y_target_seq.flatten()           # stack target over horizon

    # Free response
    y_free = F @ x0                      # (N*m,)

    # QP matrices
    # cost = 1/2 (Phi U + y_free - r)^T Q (Phi U + y_free - r) + 1/2 U^T R U
    #      = 1/2 U^T (Phi^T Q Phi + R) U + (y_free - r)^T Q Phi U + const
    H = Phi.T @ Q_big @ Phi + R_big      # (N*p, N*p)
    f = Phi.T @ Q_big @ (y_free - r)    # (N*p,)

    H = 0.5 * (H + H.T)                 # ensure symmetric

    # Solve with scipy — box constraints 0 <= u <= 1
    bounds = [(0.0, 1.0)] * (N * p)
    result = minimize(
        lambda u: 0.5 * u @ H @ u + f @ u,
        x0=np.zeros(N * p),
        jac=lambda u: H @ u + f,        # exact gradient — much faster
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-8},
    )

    U_opt = result.x.reshape(N, p)
    return U_opt[0], U_opt

def mpc_step_warm(A, B, C,x0, y_target:np.ndarray, N, Q_mpc, R_mpc, Q_terminal, u_prev, u_last=None):
    """
    MPC step with warm start.
    u_prev : (N, p) previous solution — shift by one and pad to warm start.
    Returns (u_first, u_sequence_for_next_step)

    parameters
    -------------
    y_target 
        should be T,neurons matrix of the target function

    N  mpc prediction horizon


    """
    n = A.shape[0]
    p = B.shape[1]

    # Shift previous solution: drop first step, repeat last, use as initial guess
    u_warm = np.vstack([u_prev[1:], u_prev[-1:]])   # (N, p)
    u0 = u_warm.flatten()

    def cost(u_flat):
        U = u_flat.reshape(N, p)
        x = x0.copy()
        total = 0.0
        R_rate = np.eye(p) * 5.0   # penalty on input changes

        u_previous = u_last if u_last is not None else np.zeros(p)  # handles not previous step at first step
        for k in range(N):
            y = C @ x
            e = y - y_target
            total += e @ Q_mpc @ e + U[k] @ R_mpc @ U[k]

            # Penalise rate of change of input
            du = U[k] - u_previous
            total += du @ R_rate @ du
            u_previous = U[k]

            x = A @ x + B @ U[k]
        e_terminal = C @ x - y_target
        total += e_terminal @ Q_terminal @ e_terminal
        return total

    bounds = [(0.0, 1.0)] * (N * p)

    result = minimize(cost, u0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 100, 'ftol': 1e-6})

    u_opt = result.x.reshape(N, p)
    return u_opt[0], u_opt   # first input + full sequence for next warm start


def run_mpc(brain, A, B, C, Q_kal, R_kal, y_target_arr:np.ndarray, T, N=10,Q_mpc = None, Q_terminal = None, R_mpc = None):
    '''
    parameter
    ----------
    y_target 
        full target output array (T, m) (time,neurons)    
    '''
    p = B.shape[1]
    m = C.shape[0]
    n = A.shape[0]

    if R_mpc is None:
        R_mpc      = np.eye(p) * 10.0  ## higher R to penalise rapid change in inputs more
    if Q_mpc is None:
        Q_mpc = np.eye(m) * 1.0
    if Q_terminal is None:
        Q_terminal = np.eye(m) * 10.0

    kf = OnlineKalmanFilter(A, B, C, Q_kal, R_kal) # old without adaptive state estimation on B
    # kf = AdaptiveKalmanFilter(A, B, C, Q_kal, R_kal, lam= 0.99, anneal_rate=0.995,min_QR_ratio=0.01) # doesnt work yet

    Y_mpc = np.zeros((T, m))
    U_mpc = np.zeros((T, p))
    X_est = np.zeros((T, n))

     # Precompute prediction matrices once for mpc quadratic program
    F, Phi = build_prediction_matrices(A, B, C, N)

    u = np.zeros(p)
    u_prev_sequence = np.zeros((N, p))   # warm start storage

    # Handle both (m,) fixed and (T, m) time-varying targets
    if y_target_arr.ndim == 1:
        get_target_seq = lambda t: y_target_arr
    else:
        def get_target_seq(t):
            end = min(t + N, T)
            seq = y_target_arr[t:end]
            if len(seq) < N:
                seq = np.vstack([seq, np.tile(seq[-1], (N - len(seq), 1))])
            return seq   # (N, m)

    for t in range(T):
        y_target = get_target_seq(t)
        y = np.array(brain.measure())
        Y_mpc[t] = y

        x_hat = kf.update(y, u)
        X_est[t] = x_hat

        u, u_prev_sequence = mpc_step_qp(
            A, B, C,F,Phi, x_hat, y_target,
            Q_mpc,Q_terminal, R_mpc, 
            N
        )
        U_mpc[t] = u
        brain.next_state(u)

        if t % 50 == 0:
            print(f"  t={t:3d} | ||y - y_target|| = {np.linalg.norm(y - y_target):.4f}")

    return Y_mpc, U_mpc, X_est

#===========================================================================
# -- 6. Target Generation
#===========================================================================
def generate_point_to_point_targets(T, m, n_points=6, hold_steps=None, amplitude=4.0, seed=42):
    """
    Generate point-to-point targets — holds at each point for hold_steps
    then jumps to the next. Classic Fitts' task structure.

    Parameters
    ----------
    T          : total timesteps
    m          : number of observation channels
    n_points   : number of distinct target points
    hold_steps : timesteps per target (default T // n_points)
    amplitude  : scale of targets
    seed       : random seed

    Returns
    -------
    y_target : (T, m)
    """
    rng = np.random.default_rng(seed)
    hold_steps = hold_steps or T // n_points

    # Generate distinct target vectors
    targets = rng.uniform(-amplitude, amplitude, size=(n_points, m))

    y_target = np.zeros((T, m))
    for t in range(T):
        idx = min(t // hold_steps, n_points - 1)
        y_target[t] = targets[idx]

    return y_target


def generate_random_walk_targets(T, m, step_std=0.05, amplitude=4.0, seed=42):
    """
    Generate a smooth random walk target — drifts continuously,
    clipped to [-amplitude, amplitude].

    Parameters
    ----------
    T        : total timesteps
    m        : number of observation channels
    step_std : std of each random step (smaller = smoother)
    amplitude: clip range
    seed     : random seed

    Returns
    -------
    y_target : (T, m)
    """
    rng = np.random.default_rng(seed)
    y_target = np.zeros((T, m))
    y_target[0] = rng.uniform(-amplitude * 0.5, amplitude * 0.5, size=m)

    for t in range(1, T):
        step = rng.normal(0, step_std, size=m)
        y_target[t] = np.clip(y_target[t-1] + step, -amplitude, amplitude)

    return y_target





# -----------------------------------------------------------------------------
# STAGE 6: Evaluation plots
# -----------------------------------------------------------------------------


#look at impulse response and controllability
def plot_impulse_response(A, B, C, n_steps=30):
    """
    Plot the impulse response of the identified system.
    Computes Markov parameters: CB, CAB, CA²B, ..., CA^{n-1}B
    
    Shows how each input channel affects each output channel over time.
    """
    m = C.shape[0]
    p = B.shape[1]
    n = A.shape[0]

    # Compute Markov parameters  (n_steps, m, p)
    markov = np.zeros((n_steps, m, p))
    A_pow = np.eye(n)
    for k in range(n_steps):
        markov[k] = C @ A_pow @ B
        A_pow = A_pow @ A

    # -- Plot -----------------------------------------------------------------
    fig, axes = plt.subplots(p, 1, figsize=(14, 4 * p), sharex=True)
    if p == 1:
        axes = [axes]

    t = np.arange(n_steps)
    for inp in range(p):
        ax = axes[inp]
        for ch in range(m):
            ax.plot(t, markov[:, ch, inp], alpha=0.5, linewidth=1, label=f"ch{ch}")
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_ylabel(f"Output response")
        ax.set_title(f"Impulse response — input u{inp}")
        ax.legend(fontsize=6, ncol=4, loc='upper right')

    axes[-1].set_xlabel("Timestep")
    plt.suptitle("System Impulse Response (Markov Parameters)", fontsize=12)
    plt.tight_layout()
    plt.show()

    # -- Print cumulative reachability rank ------------------------------------
    print("\nCumulative reachable output rank vs horizon:")
    for k in [1, 5, 10, 20, n_steps]:
        if k > n_steps:
            continue
        # Stack Markov parameters up to step k
        M = np.hstack([markov[j] for j in range(k)])   # (m, k*p)
        s = np.linalg.svd(M, compute_uv=False)
        rank = np.sum(s > 1e-3)
        print(f"  Horizon {k:3d}: rank={rank}, singular values={np.round(s[:min(6,len(s))], 3)}")

    # -- DC gain ---------------------------------------------------------------
    G_dc = C @ np.linalg.inv(np.eye(n) - A) @ B
    U_dc, s_dc, _ = np.linalg.svd(G_dc)
    print(f"\nDC gain singular values: {np.round(s_dc, 4)}")
    print(f"Reachable output directions at steady state: {np.sum(s_dc > 1e-3)}")
    for i in range(len(s_dc)):
        top_ch = np.argsort(np.abs(U_dc[:, i]))[::-1][:4]
        print(f"  Direction {i}: gain={s_dc[i]:.4f}, strongest channels={top_ch.tolist()}")



def plot_identification(U_ident, Y_ident, title="Identification Experiment"):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(U_ident, alpha=0.7)
    axes[0].set_ylabel("Input u(t)")
    axes[0].set_title(title)
    axes[0].legend([f"u{i}" for i in range(U_ident.shape[1])], loc="upper right", fontsize=8)

    axes[1].imshow(Y_ident.T, aspect="auto", interpolation="nearest",
                   extent=[0, len(Y_ident), Y_ident.shape[1]-0.5, -0.5])
    axes[1].set_ylabel("Channel")
    axes[1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.show()


def plot_eigenvalues(A, title="Identified A eigenvalues"):
    eigs = np.linalg.eigvals(A)
    fig, ax = plt.subplots(figsize=(5, 5))
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    ax.scatter(eigs.real, eigs.imag, color='tab:blue', zorder=5)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
    print(f"Eigenvalue magnitudes: {np.abs(eigs).round(3)}")
    print(f"Spectral radius: {np.abs(eigs).max():.4f}")


def plot_controlled_vs_uncontrolled(Y_open, Y_ctrl, U_ctrl):
    T = min(len(Y_open), len(Y_ctrl))
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # Observation power (mean across channels)
    axes[0].plot(np.linalg.norm(Y_open[:T], axis=1), label="Uncontrolled", color='black')
    axes[0].plot(np.linalg.norm(Y_ctrl[:T], axis=1), label="Controlled", color='tab:blue')
    axes[0].set_ylabel("||y(t)||")
    axes[0].set_title("Observation norm: controlled vs uncontrolled")
    axes[0].legend()

    # A few observation channels
    for ch in range(min(3, Y_open.shape[1])):
        axes[1].plot(Y_open[:T, ch], alpha=0.4, color=f'C{ch}', linestyle='--')
        axes[1].plot(Y_ctrl[:T, ch], alpha=0.8, color=f'C{ch}', label=f"y{ch}")
    axes[1].set_ylabel("Observations")
    axes[1].legend(fontsize=8)

    # Control inputs
    axes[2].plot(U_ctrl[:T], alpha=0.8)
    axes[2].set_ylabel("u(t)")
    axes[2].set_xlabel("Timestep")
    axes[2].set_title("Control inputs")
    axes[2].legend([f"u{i}" for i in range(U_ctrl.shape[1])], fontsize=8)

    plt.tight_layout()
    plt.show()

    # Summary stats
    norm_open = np.linalg.norm(Y_open[:T], axis=1).mean()
    norm_ctrl = np.linalg.norm(Y_ctrl[:T], axis=1).mean()
    print(f"Mean ||y|| uncontrolled: {norm_open:.4f}")
    print(f"Mean ||y|| controlled:   {norm_ctrl:.4f}")
    print(f"Change: {(norm_ctrl - norm_open)/norm_open * 100:.1f}%")
    print(f"Mean control effort ||u||: {np.linalg.norm(U_ctrl, axis=1).mean():.4f}")



def plot_comparison(Y_open, Y_ctrl, U_ctrl, y_target_arr, channels_to_plot:np.ndarray=np.asarray([0,1,2]), label="Controlled"):
    T = len(Y_open)
    n_ch = min(len(channels_to_plot), Y_open.shape[1])

    fig, axes = plt.subplots(n_ch + 2, 1, figsize=(14, 4 + 3 * n_ch), sharex=True)

    # -- Observation norm -----------------------------------------------------
    axes[0].plot(np.linalg.norm(Y_open, axis=1), color='black', label='Uncontolled', alpha=0.7)
    axes[0].plot(np.linalg.norm(Y_ctrl, axis=1), color='tab:blue', label=label, alpha=0.8)
    axes[0].plot(np.linalg.norm(y_target_arr, axis=1), color='red', linestyle='--', label='Target', alpha=0.6)
    axes[0].set_ylabel("||y(t)||")
    axes[0].set_title(f"Observation norm — {label} vs Uncontrolled")
    axes[0].legend(fontsize=8)

    # -- One subplot per channel -----------------------------------------------
    for i,ch in enumerate(channels_to_plot):
        ax = axes[i + 1]
        ax.plot(Y_open[:, ch], color='black', alpha=0.3, linestyle='--', label='Uncontrolled')
        ax.plot(Y_ctrl[:, ch], color=f'C{ch}', alpha=0.9, label=f"y{ch}")
        if y_target_arr.ndim == 1:
            ax.axhline(y_target_arr[ch], color='red', linestyle='--', alpha=0.6, label='target')
        else:
            ax.plot(y_target_arr[:, ch], color='red', linestyle='--', alpha=0.6, label='target')
        ax.set_ylabel(f"ch{ch}")
        ax.legend(fontsize=7, loc='upper right')

    # -- Inputs ---------------------------------------------------------------
    axes[-1].plot(U_ctrl, alpha=0.8)
    axes[-1].set_ylabel(f"u(t) — {label}")
    axes[-1].set_xlabel("Timestep")
    axes[-1].set_ylim(-0.05, 1.05)
    axes[-1].legend([f"u{i}" for i in range(U_ctrl.shape[1])], fontsize=8)

    plt.suptitle(f"{label} Control", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()

    # Summary stats
    tracking_error = np.linalg.norm(Y_ctrl - y_target_arr, axis=1)
    baseline_error = np.linalg.norm(Y_open - y_target_arr, axis=1)
    print(f"\n{label} summary:")
    print(f"  Mean tracking error:     {tracking_error.mean():.4f}")
    print(f"  Baseline tracking error: {baseline_error.mean():.4f}")
    print(f"  Mean control effort:     {np.linalg.norm(U_ctrl, axis=1).mean():.4f}")
# -----------------------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------------------

def plot_pca_dimensionality(Y):
    """SVD/PCA on observation matrix to estimate number of latent states."""
    U, s, Vt = np.linalg.svd(Y - Y.mean(axis=0), full_matrices=False)
    
    var_explained = s**2 / np.sum(s**2)
    cumvar = np.cumsum(var_explained)
    
    # Participation ratio — effective number of dimensions
    PR = np.sum(s**2)**2 / np.sum(s**4)
    
    # 95% variance threshold
    n_95 = np.searchsorted(cumvar, 0.95) + 1

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    axes[0].plot(s**2 / s[0]**2, 'o-', markersize=4)
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Normalised variance")
    axes[0].set_title("Scree plot")
    axes[0].axvline(PR, color='red', linestyle='--', label=f'PR={PR:.1f}')
    axes[0].legend()

    axes[1].plot(cumvar, 'o-', markersize=4)
    axes[1].axhline(0.95, color='red', linestyle='--', label='95%')
    axes[1].axvline(n_95 - 1, color='orange', linestyle='--', label=f'n={n_95}')
    axes[1].set_xlabel("Component")
    axes[1].set_ylabel("Cumulative variance")
    axes[1].set_title("Cumulative explained variance")
    axes[1].legend()

    axes[2].plot(np.log(s), 'o-', markersize=4)
    axes[2].set_xlabel("Component")
    axes[2].set_ylabel("log singular value")
    axes[2].set_title("Log singular values")

    plt.suptitle(f"PCA dimensionality — PR={PR:.1f}, 95% var at n={n_95}", fontsize=12)
    plt.tight_layout()
    plt.show()
    
    print(f"Participation ratio:       {PR:.1f}")
    print(f"95% variance components:   {n_95}")
    print(f"Current LATENT_DIM:        {LATENT_DIM}")

if __name__ == "__main__":

    # -- 1. Identification ----------------------------------------------------
    print("=== Stage 1: PRBS Identification ===")
    brain_ident = Brain(random_seed=RANDOM_SEED)
    p = brain_ident.input_dim   # 2
    print('input dimension from brain: ',p)

    

    U_ident, Y_ident = run_identification_experiment(
        brain_ident, T=T_IDENT, p=p, prbs_period=PRBS_PERIOD
    )

    # plot_pca_dimensionality(Y_ident)
    print(f"Collected {T_IDENT} timesteps, {Y_ident.shape[1]} channels, {p} inputs")
    plot_identification(U_ident, Y_ident)

    # -- 2. System identification ---------------------------------------------
    print("\n=== Stage 2: N4SID System Identification ===")
    A, B, C = deterministic_n4sid(Y_ident, U_ident, n=LATENT_DIM, i=SSI_HORIZON)

    print(f"A: {A.shape}, B: {B.shape}, C: {C.shape}")
    plot_eigenvalues(A)

    # Noise covariance estimates for Kalman filter
    m = C.shape[0]
    Q_kal = np.eye(LATENT_DIM) * 0.1
    R_kal = np.eye(m) * 0.01

    print("\n=== Stage 2b: EM Refinement ===")    # em initilialised with n4sid
    A, B, C, Q_kal, R_kal, ll_hist = refine_with_em(
        Y_ident, U_ident, A, B, C, n_iter=30
    )
    # Q_kal and R_kal now come directly from EM rather than being guessed


    plot_impulse_response(A, B, C)



    # -- 3. LQR design --------------------------------------------------------
    print("\n=== Stage 3: LQR Design ===")
    K = design_lqr(A, B, Q_LQR, R_LQR)
    print(f"LQR gain K: {K.shape}")
    print(f"K norm: {np.linalg.norm(K):.4f}")

    # Check closed-loop stability
    A_cl = A - B @ K
    eigs_cl = np.linalg.eigvals(A_cl)
    print(f"Closed-loop spectral radius: {np.abs(eigs_cl).max():.4f}")
    if np.abs(eigs_cl).max() < 1.0:
        print("Closed-loop system is stable")
    else:
        print("WARNING: Closed-loop system is unstable — increase R_LQR")

    # -- 4. Baseline: uncontrolled ---------------------------------------------
    print("\n=== Stage 4: Uncontrolled Baseline ===")
    brain_open = Brain(random_seed=RANDOM_SEED + 1)
    Y_open = run_uncontrolled(brain_open, T=T_CONTROL)

   #======================================================================
   # testing
   #======================================================================

   # How many output directions can your 2 inputs actually reach?
    # Look at CB — maps inputs directly to output first markov parameter 1 step after delta
    
    

    # DC gain — steady state output per unit input
    G_dc = C @ np.linalg.inv(np.eye(A.shape[0]) - A) @ B   # (m, p)    C (sI-A)^-1 B
    U_dc, s_dc, Vt_dc = np.linalg.svd(G_dc)
    print(f"DC gain singular values: {s_dc}")
    print(f"Effective output rank: {np.sum(s_dc > 1e-3)}")

    # Output directions reachable by your inputs
    print(f"\nTop reachable output directions (columns of U_dc):")
    for i in range(len(s_dc)):
        print(f"  Direction {i}: gain={s_dc[i]:.4f}, top channels={np.argsort(np.abs(U_dc[:, i]))[::-1][:4]}")

    #there are 2 principle directions of out output space that we can reach with our 2 inputs

    dir0 = U_dc[:, 0] 
    dir1 = U_dc[:, 1] 


    def target_achievability(y_target_vec, U_dc, s_dc):
        """What fraction of the target lies in the reachable subspace.
        
        params
        ------
        y_target_vec : (m,) target output vector
        U_dc : (m, m) left singular vectors of DC gain
        s_dc : (m,) singular values of DC gain
        
        
        """
        # Project target onto reachable directions
        proj = U_dc[:, :len(s_dc)] @ (U_dc[:, :len(s_dc)].T @ y_target_vec)
        achievable = np.linalg.norm(proj) / (np.linalg.norm(y_target_vec) + 1e-12)
        print(f"  Target norm:      {np.linalg.norm(y_target_vec):.3f}")
        print(f"  Achievable norm:  {np.linalg.norm(proj):.3f}")
        print(f"  Achievability:    {achievable*100:.1f}%")
        return achievable


    #new experiment, pulse target on subset of channels
    brain1 = Brain(random_seed=RANDOM_SEED) # need fresh brain # 17:44 changed to 2 # same brain as estimated for
    # pulse channels 0 2
    target1 = np.zeros((T_CONTROL, m))
    target1[:T_CONTROL//2, :2] = 4.0  # step

    #pulse all
    target2= np.zeros((T_CONTROL, m))
    target2[:T_CONTROL//2, :] = 4.0  # step

    #one high
    target3 = np.zeros((T_CONTROL, m))
    target3[:, 14] = 4.0  # step on channel 0 only


    # Scale to desired amplitude #--> should be in achievable output space     #some of Dir0 is negative, 
    target_dir0 = np.zeros((T_CONTROL, m))
    target_dir0[:T_CONTROL//2, :] = dir0 * 3   # scale up direction 0 for first half
    # second half stays at zero — return to baseline

     # Scale to desired amplitude #--> should be in achievable output space
    target_dir0 = np.zeros((T_CONTROL, m))
    target_dir0[:T_CONTROL//2, :] = dir0 * 3   # scale up direction 0 for first half
    # second half stays at zero — return to baseline

    # What outputs can you actually achieve with u in [0,1]?
    # At steady state: y = G_dc @ u_ss where u_ss in [0,1]^p
    # The reachable set is the image of the unit box under G_dc

    # Corner points of input box
    u_corners = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=float)
    y_corners = (G_dc @ u_corners.T).T   # (4, m)


    y_max = G_dc @ np.array([1.0, 1.0])   #  output space with both inputs high
    y_u0 = G_dc @ np.array([1.0, 0.0])    #  outpit space with u0 hig
    y_u1 = G_dc @ np.array([0.0, 1.0])    # output space with u1 high


    print("Reachable output corners:")
    for i, uc in enumerate(u_corners):
        print(f"  u={uc} → y norm={np.linalg.norm(y_corners[i]):.3f}")


    # u=[1,1] gives maximum output — use as"high" target
    #step in output space of two inputs high
    target_max = np.zeros((T_CONTROL, m))
    target_max[:T_CONTROL//2, :] = y_max * 0.8   # 80% of maximum to avoid saturation

    target5 = np.zeros((T_CONTROL, m))
    target5[:T_CONTROL//2, :2] = 2
    Q_focused_t5 = np.eye(m) 
    Q_focused_t5[:2, :2] = 10.0  # focus on tracking error in first 2 channels for target5  

    #sine in direction of y_u0
    target6 = np.zeros((T_CONTROL, m))
    target6[:, :] = np.sin(np.linspace(0,24* np.pi, T_CONTROL))[:, None] * y_u0 * 0.5

    # piecewise
    target7 = np.zeros((T_CONTROL, m))
    target7[:T_CONTROL//8, :] = y_u0 * 0.5
    target7[T_CONTROL//8:T_CONTROL//4, :] = y_u0 * 1.2 + y_u1 * 0.5
    target7[T_CONTROL//4:T_CONTROL//2, :] = y_u1 * 0.5
    target7[T_CONTROL//2:T_CONTROL//4*3, :] = y_max * 0.5
    target7[T_CONTROL//4*3:, :] = 0.0


    #another piecewise
    target8 = np.zeros((T_CONTROL, m))
    target8[:T_CONTROL//8, :] = y_u0 * 0.5
    target8[T_CONTROL//8:T_CONTROL//4, :] =  y_u1 * 0.5
    target8[T_CONTROL//4:T_CONTROL//2, :] = y_u1 * 0.5
    target8[T_CONTROL//2:T_CONTROL//4*3, :] = y_max * 0.5
    target8[T_CONTROL//4*3:, :] = 0.0


    #pure noise target
    noise_std1 = 0.5
    target9 =  np.random.randn(T_CONTROL, m) * noise_std1


    #noisy target -->noisy step
    noise_std2 = 0.5
    target10 = target_max + np.random.randn(T_CONTROL, m) * noise_std2

    #other stress testing inputs
    # 1. Infeasible step - jumps outside constraint bounds
    target_infeasible = np.ones((T_CONTROL, m)) * y_max * 1.5

    #infeasible target small output direction
    target_infeasible2 = np.ones((T_CONTROL, m)) * y_u1 * 1.5

    # 2. High-frequency switching - faster than prediction horizon
    horizon_length = 20
    dt = 1.0  # 
    t = np.arange(T_CONTROL)
    target_switching = np.sign(np.sin(2 * np.pi * t / (dt * horizon_length))) [:,None]* y_max* 0.5

    # 3. Ramp that accelerates beyond what the horizon covers
    # Ramp hits constraint_max halfway through the simulation
    scale = y_max / (0.5 * T_CONTROL * dt)**2  
    target_ramp = np.outer(t**2, np.ones(m)) * scale



    #==================================================================================================

    '''     Runs to put into notebook'''

    #interesting --> can see saturation
    # y ,u ,x = run_mpc(
    #         brain1, A, B, C, Q_kal, R_kal,target6, T=T_CONTROL, N=50,Q_mpc = None,Q_terminal = None)
        
    #     plot_comparison(Y_open, y, u, target6,channels_to_plot=np.asarray([0,7,3,14]), label = 'mpc')

    #also interesting
    # y ,u ,x = run_mpc(
    #     brain1, A, B, C, Q_kal, R_kal,target6, T=T_CONTROL, N=50,Q_mpc = None,Q_terminal = None)
    
    # plot_comparison(Y_open, y, u, target6,channels_to_plot=np.asarray([0,7,3,14]), label = 'mpc')


    # can see penalised control effort
    # y ,u ,x = run_mpc(
    #     brain1, A, B, C, Q_kal, R_kal,target7, T=T_CONTROL, N=50,Q_mpc = None,Q_terminal = None,R_mpc = np.eye(p)*1000)
    
    # plot_comparison(Y_open, y, u, target7,channels_to_plot=np.asarray([0,7,3,14]), label = 'mpc')

    #numerical issues with large Q/R ratios in mpc quadratic program — need to tune optimization parameters to get convergence
    #  y ,u ,x = run_mpc(
    #         brain1, A, B, C, Q_kal, R_kal,target7, T=T_CONTROL, N=50,Q_mpc = np.eye(m)*100,Q_terminal = np.eye(m)*100000,R_mpc = np.eye(p))
        
    #     plot_comparison(Y_open, y, u, target7,channels_to_plot=np.asarray([0,7,3,14]), label = 'mpc')




    #=================================================================================================



    #testing achievability of targets

    print(' dir0 : ',dir0)
    print('dir1 : ',dir1)
    print('target1')
    target_achievability(target1[T_CONTROL//4], U_dc, s_dc)
    print('target2')
    target_achievability(target2[T_CONTROL//4], U_dc, s_dc)
    print('target3')
    target_achievability(target3[T_CONTROL//4], U_dc, s_dc)
    print('target_dir0')
    target_achievability(target_dir0[T_CONTROL//4], U_dc, s_dc)
    print('target_max')
    target_achievability(target_max[T_CONTROL//4], U_dc, s_dc)
    print('target5')
    target_achievability(target5[T_CONTROL//4], U_dc, s_dc)
    print('target6')
    target_achievability(target6[T_CONTROL//4], U_dc, s_dc)
    print('target7')
    target_achievability(target7[T_CONTROL//4], U_dc, s_dc)
    print('target8')
    target_achievability(target8[T_CONTROL//4], U_dc, s_dc)
    print('target9')
    target_achievability(target9[T_CONTROL//4], U_dc, s_dc)
    print('target10')
    target_achievability(target10[T_CONTROL//4], U_dc, s_dc)
    print('target_infeasible')
    target_achievability(target_infeasible[T_CONTROL//4], U_dc, s_dc)
    print('target_switching')
    target_achievability(target_switching[T_CONTROL//4], U_dc, s_dc)
    print('target_ramp')
    target_achievability(target_ramp[T_CONTROL//4], U_dc, s_dc)

    #if a target is perfectly achievble ten in theory we do not need to apply special weighting in out mpc

    #Q_focused scales how much we care about tracking error in each channel, 
    #Q_terminal weights error at the end of the horizon, higher Q_terminal encourages the controller to settle faster 
    #we can weight our mpc to care more about certain channels

    #
    #Q_focused =

    #good Q,Qt,R settings ive found
    #Q = 10,   Qt = 100, R = 
    



    if input() == 'y': #its the ration between R_mpc and Q_mpc that matters  :  higher Q/R better control but u clips
        y ,u ,x = run_mpc(
            brain1, A, B, C, Q_kal,R_kal, y_target_arr = target_switching, T=T_CONTROL, N=horizon_length,Q_mpc = np.eye(m)*10,Q_terminal = np.eye(m)*100,R_mpc = np.eye(p)*3000) #horizon length is defined by target switching  #
        
        plot_comparison(Y_open, y, u, y_target_arr = target_switching,channels_to_plot=np.asarray([0,7,3,14]), label = 'mpc')

        
    CB = C @ B  # (m, p)
    print("First Markov parameters CB (m x p):")
    print(CB)
    print("\nPer channel, per input:")
    for ch in range(m):
        print(f"  ch{ch:2d}:  u0={CB[ch,0]:+.4f}  u1={CB[ch,1]:+.4f}  sign={'OPPOSITE' if CB[ch,0]*CB[ch,1] < 0 else 'same'}")
        



    #====== Performance plots ====================================================

    # tracking_error_mpc = np.linalg.norm(Y_ctrl - y_target_arr, axis=1)
    # tracking_error_lqr = np.linalg.norm(Y_lqr  - y_target_arr, axis=1)
    # baseline_error      = np.linalg.norm(Y_open - y_target_arr, axis=1)
    # fig, ax = plt.subplots(figsize=(14, 4))
    # ax.plot(baseline_error,      color='black',    label='Uncontrolled')
    # ax.plot(tracking_error_lqr,  color='tab:green', label='LQR')
    # ax.plot(tracking_error_mpc,  color='tab:blue',  label='MPC')
    # ax.axvline(T_CONTROL//2, color='gray', linestyle='--', alpha=0.5, label='Target switch')
    # ax.set_ylabel("||y - y_target||")
    # ax.set_xlabel("Timestep")
    # ax.set_title("Tracking error: LQR vs MPC vs Uncontrolled")
    # ax.legend()
    # plt.tight_layout()
    # plt.show()

    # print(f"\nMean tracking error:")
    # print(f"  Uncontrolled: {baseline_error.mean():.4f}")
    # print(f"  LQR:          {tracking_error_lqr.mean():.4f}")
    # print(f"  MPC:          {tracking_error_mpc.mean():.4f}")
    # print(f"\nComputation time:")
    # print(f"  LQR: {t_lqr:.2f}s ({t_lqr/T_CONTROL*1000:.2f}ms/step)")
    # print(f"  MPC: {t_mpc:.2f}s ({t_mpc/T_CONTROL*1000:.2f}ms/step)")
    # print(f"  MPC is {t_mpc/t_lqr:.0f}x slower")




    ''' interaction with BMI and Hand'''

#same brain seed as before
# bmi = BMI_and_Hand(Brain(random_seed=RANDOM_SEED + 1))

