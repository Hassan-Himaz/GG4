
from __future__ import annotations

import numpy as np
from numpy.linalg import svd, solve, inv, lstsq
from scipy.linalg import solve_discrete_lyapunov
from typing import Tuple


#code structure

#dataclasses
# - - -
#all the functions mostly in order of when they are applied in main workflow

# - - 

#main estimation workflow  - > def estimat.....






#scratch build

#----------dataclassses---------------------------------------------------

class LDSParams:
    def __init__(self, A, B, C, Q, R, mu0, P0):
        self.A   = A
        self.B   = B
        self.C   = C
        self.Q   = Q
        self.R   = R
        self.mu0 = mu0
        self.P0  = P0

    def copy(self):
        return LDSParams(
            A=self.A.copy(), B=self.B.copy(), C=self.C.copy(),
            Q=self.Q.copy(), R=self.R.copy(),
            mu0=self.mu0.copy(), P0=self.P0.copy()
        )

#---------------step 1 n4sid for initial A and C estimation---------------

def _block_hankel(Y: np.ndarray, i: int) -> np.ndarray:
    """Build a 2i-block-row Hankel matrix from output sequence Y (m, T).

    Returns H of shape (2*i*m, T - 2*i + 1).
    """
    m, T = Y.shape
    cols = T - 2 * i + 1
    if cols <= 0:
        raise ValueError(f"Sequence too short for i={i}: need T > 2i, got T={T}")
    H = np.zeros((2 * i * m, cols))
    for row_block in range(2 * i):
        H[row_block * m:(row_block + 1) * m, :] = Y[:, row_block:row_block + cols]
    return H


def _oblique_projection(Y_future: np.ndarray, Y_past: np.ndarray) -> np.ndarray:
    """Oblique projection of Y_future onto row-space of Y_past.

    O_i = Y_future / Y_past  (oblique projection along orthogonal complement)
    Computed as: O_i = Y_future @ Y_past^T @ (Y_past @ Y_past^T)^+ @ Y_past
    """
    # Least-squares: min ||Y_future - X Y_past||_F  =>  X = Y_future Y_past^T (Y_past Y_past^T)^+
    # Then O_i = X @ Y_past
    X, _, _, _ = lstsq(Y_past.T, Y_future.T, rcond=None)
    return (X.T @ Y_past)


def stochastic_subspace_identification(
    Y: np.ndarray,
    n: int,
    i: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate A and C from output data via SSI-COV / N4SID projection.

    Parameters
    ----------
    Y : (m, T)  output observations
    n : int     desired state dimension
    i : int     number of block rows (≥ n/m, rule of thumb: 2x-4x that)

    Returns
    -------
    A : (n, n)
    C : (m, n)
    """
    m, T = Y.shape

    # 1. Build block-Hankel and split into past / future halves
    H = _block_hankel(Y, i)                    # (2im, cols)
    Y_past   = H[:i * m, :]                    # (im,  cols)
    Y_future = H[i * m:, :]                    # (im,  cols)

    # 2. Oblique projection  O_i  =  Y_future / Y_past
    O_i = _oblique_projection(Y_future, Y_past)   # (im, cols)

    # 3. SVD of O_i  →  select n leading components  →  observability matrix Γ
    U, s, Vt = svd(O_i, full_matrices=False)

    U1 = U[:, :n]
    S1 = np.diag(s[:n])
    Γ  = U1 @ np.sqrt(S1)

    # 4. Extract C from first m rows of Γ
    C = Γ[:m, :]                               # (m, n)

    # 5. Extract A via shift invariance:  Γ_{1..i-1} A = Γ_{2..i}
    Γ_up   = Γ[:(i - 1) * m, :]               # ((i-1)m, n)
    Γ_down = Γ[m:,           :]               # ((i-1)m, n)
    A, _, _, _ = lstsq(Γ_up, Γ_down, rcond=None)   # (n, n)

    # normalising A 
    max_radius = 0.99
   
    """Rescale A so spectral radius <= max_radius.""" # to keep A stable
    rho = np.abs(np.linalg.eigvals(A)).max()
    if rho >= max_radius:
        A = A * (max_radius / rho)
   
    col_norms = np.linalg.norm(Γ, axis=0, keepdims=True)
    Γ = Γ / np.maximum(col_norms, 1e-10)
    C = Γ[:m, :]


    return A, C


#------------------step 2 initialising B,Q,R


def initialise_BQR(observation:np.ndarray,
                   A_mat:np.ndarray,
                   C_mat:np.ndarray,
                   InputDim:int,
                   initial_QR_ratio:float,
                   percentage_output_variance_explained_by_R:float,
                   rank_sufficient_factor:float = 0.5,
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    initialise B Q R

    params
    ---------

    observation: (m, T)  output observations
    A_mat: (n, n)  state transition matrix from N4SID
    C_mat etsimated from N4SID
    

    retunrs
    ---------
    B Q R in a tuple of numpy arrays
    '''
    
    # runs a naive kalman filter to get estimate of states, 
    # then uses SVD of residuals to get initial estimate of B
    #we set initial Q/R ratio high as we do not trust our model yet

    A = A_mat
    C = C_mat
    Y = observation
    p = InputDim


    m, T = Y.shape
    n = A.shape[0]

    # ---- naive Kalman (no inputs) to collect innovations ----
    # Use identity covariances to bootstrap
    Q0 = np.eye(n) * 1e-2
    R0 = np.eye(m) * 1e-1
    x  = np.zeros(n)
    P  = np.eye(n)
    innovations = np.zeros((T, m))

    for t in range(T):
        # Predict
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q0

        # Update
        S = C @ P_pred @ C.T + R0
        K = P_pred @ C.T @ inv(S)
        inn = Y[:, t] - C @ x_pred
        innovations[t] = inn
        x = x_pred + K @ inn
        P = (np.eye(n) - K @ C) @ P_pred

    # ---- Residual SVD for B initialisation ----
    # Run initial RTS smoother to get state estimates
    U_zero = np.zeros((p, T))  # no inputs yet
    # Use innovations-based x_smooth from the Kalman pass above

    # Compute state residuals: r_t = x_{t+1} - A x_t
    # x_smooth is (T, n) from the Kalman filter above
    x_smooth_init = np.zeros((T, n))
    x = np.zeros(n)
    P = np.eye(n)
    x_all = np.zeros((T, n))
    for t in range(T):
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q0
        S = C @ P_pred @ C.T + R0
        K = P_pred @ C.T @ inv(S)
        inn = Y[:, t] - C @ x_pred
        x = x_pred + K @ inn
        P = (np.eye(n) - K @ C) @ P_pred
        x_all[t] = x

    # State residuals
    residuals = x_all[1:] - (A @ x_all[:-1].T).T   # (T-1, n)
    R_mat = residuals.T                               # (n, T-1)

    # SVD — leading directions span input subspace
    U_svd, s_svd, _ = svd(R_mat, full_matrices=False)
    B = U_svd[:, :p]   # (n, p)  — already unit norm columns

    # average covariance in y to define scale of noise covariance R
    # average covariance matrix of Y  (m, m)
    R_empirical = np.cov(Y)   # Y is (m, T), np.cov treats rows as variables
    R = R_empirical * percentage_output_variance_explained_by_R + np.eye(m)*rank_sufficient_factor  

    var_Q_scale  = np.trace(R) / m
    Q_rand       = np.random.randn(n, n)
    Q_rand       = Q_rand @ Q_rand.T / n
    Q            = (Q_rand + np.eye(n)) * initial_QR_ratio * var_Q_scale




    return B, Q, R


#--------------step 3 state augmented kalman filter for E step----------------


def kalman_filter_augmented(
    Y: np.ndarray,
    params: LDSParams,
    tau: np.ndarray,          # (p, T-1)  per-timestep input scales (Laplace mixing vars)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Augmented state Kalman filter.  z_t = [x_t; u_t]

    Augmented dynamics:
        z_{t+1} = A_aug z_t + w_t
        y_t     = C_aug z_t + v_t

    where:
        A_aug = [[A, B],      C_aug = [C, 0]
                 [0, 0]]

    u_t has no dynamics (random walk with zero mean, variance tau_t).
    The Laplace prior is handled via the diagonal prior covariance tau.
    """
    m, T = Y.shape
    n = params.A.shape[0]
    p = params.B.shape[1]
    nz = n + p   # augmented state dim

    # ---- Build augmented system matrices ----
    A_aug = np.zeros((nz, nz))
    A_aug[:n, :n] = params.A        # x dynamics
    A_aug[:n, n:] = params.B        # input drives x
    # u block is zero — u_t modelled as zero-mean each step, variance from tau

    C_aug = np.zeros((m, nz))
    C_aug[:, :n] = params.C         # only x is observed

    # ---- Augmented noise covariances ----
    # Q_aug: process noise on [x; u]
    # x  block: params.Q
    # u  block: diag(tau_t) — changes per timestep (Laplace scale mixture)
    # cross:    zero

    x_f  = np.zeros((T, nz))
    P_f  = np.zeros((T, nz, nz))
    inns = np.zeros((T, m))
    ll   = 0.0

    # Initial augmented state and covariance
    mu0_aug = np.zeros(nz)
    mu0_aug[:n] = params.mu0

    P0_aug = np.zeros((nz, nz))
    P0_aug[:n, :n] = params.P0
    P0_aug[n:, n:] = np.eye(p)     # prior on u at t=0

    x = mu0_aug.copy()
    P = P0_aug.copy()

    for t in range(T):
        # Build time-varying Q_aug using tau_t
        tau_t = tau[:, t] if t < T - 1 else tau[:, -1]
        Q_aug = np.zeros((nz, nz))
        Q_aug[:n, :n] = params.Q
        Q_aug[n:, n:] = np.diag(tau_t)   # Laplace scale mixture variance

        # Predict
        x_pred = A_aug @ x
        P_pred = A_aug @ P @ A_aug.T + Q_aug
        P_pred = 0.5 * (P_pred + P_pred.T)

        # Innovation
        inn = Y[:, t] - C_aug @ x_pred
        S   = C_aug @ P_pred @ C_aug.T + params.R
        S   = 0.5 * (S + S.T)

        try:
            np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            S += np.eye(m) * 1e-8

        sign, logdet = np.linalg.slogdet(S)
        ll += -0.5 * (logdet + inn @ solve(S, inn) + m * np.log(2 * np.pi))

        K = P_pred @ C_aug.T @ inv(S)
        x = x_pred + K @ inn
        P = (np.eye(nz) - K @ C_aug) @ P_pred
        P = 0.5 * (P + P.T)

        x_f[t] = x
        P_f[t] = P
        inns[t] = inn

    return x_f, P_f, inns, ll


#-----------------------------------rts smoother from augmented filter for E step-----------------------


def rts_smoother_augmented(
    Y: np.ndarray,
    params: LDSParams,
    tau: np.ndarray,          # (p, T-1)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """RTS smoother over augmented state z_t = [x_t; u_t]."""
    m, T = Y.shape
    n  = params.A.shape[0]
    p  = params.B.shape[1]
    nz = n + p

    x_f, P_f, inns, ll = kalman_filter_augmented(Y, params, tau)

    x_s     = np.zeros((T, nz))
    P_s     = np.zeros((T, nz, nz))
    P_cross = np.zeros((T - 1, nz, nz))

    x_s[-1] = x_f[-1]
    P_s[-1] = P_f[-1]

    for t in range(T - 2, -1, -1):
        tau_t = tau[:, t] if t < T - 1 else tau[:, -1]
        Q_aug = np.zeros((nz, nz))
        Q_aug[:n, :n] = params.Q
        Q_aug[n:, n:] = np.diag(tau_t)

        A_aug = np.zeros((nz, nz))
        A_aug[:n, :n] = params.A
        A_aug[:n, n:] = params.B

        x_pred = A_aug @ x_f[t]
        P_pred = A_aug @ P_f[t] @ A_aug.T + Q_aug
        P_pred = 0.5 * (P_pred + P_pred.T)

        G = P_f[t] @ A_aug.T @ inv(P_pred)

        x_s[t] = x_f[t] + G @ (x_s[t + 1] - x_pred)
        P_s[t] = P_f[t] + G @ (P_s[t + 1] - P_pred) @ G.T
        P_s[t] = 0.5 * (P_s[t] + P_s[t].T)

        P_cross[t] = P_s[t + 1] @ G.T

    # Split augmented state back into x and u
    x_smooth = x_s[:, :n]
    u_smooth = x_s[:, n:]           # (T, p)  — posterior mean of u_t
    Px_smooth = P_s[:, :n, :n]
    Pu_smooth = P_s[:, n:, n:]      # (T, p, p)  — posterior variance of u_t

    return x_smooth, u_smooth, Px_smooth, Pu_smooth, P_cross, inns, ll



#--------------------step 4 EM m step

def _m_step_augmented(
    Y: np.ndarray,
    u_smooth: np.ndarray,   # (T, p)  posterior mean of u
    Pu_smooth: np.ndarray,  # (T, p, p)  posterior variance of u
    x_smooth: np.ndarray,   # (T, n)
    Px_smooth: np.ndarray,  # (T, n, n)
    P_cross: np.ndarray,    # (T-1, nz, nz)  full augmented cross-cov
    params: LDSParams,
    min_var: float = 1e-8,
) -> LDSParams:
    """
    M-step using augmented smoother output.
    Key difference from original: u uncertainty (Pu_smooth) enters
    the sufficient statistics, so B update accounts for input estimation error.
    """
    m, T = Y.shape
    n = params.A.shape[0]
    p = params.B.shape[1]

    # ---- Sufficient statistics with input uncertainty ----
    # E[x_t x_t^T]
    Exx      = sum(Px_smooth[t] + np.outer(x_smooth[t], x_smooth[t]) for t in range(T))
    Exx_past = sum(Px_smooth[t] + np.outer(x_smooth[t], x_smooth[t]) for t in range(T-1))

    # E[x_{t+1} x_t^T] — from augmented P_cross, x block only
    nz = n + p
    Exx_lag = sum(
        P_cross[t][:n, :n] + np.outer(x_smooth[t+1], x_smooth[t])
        for t in range(T-1)
    )

    # E[u_t u_t^T] — includes posterior variance (key augmentation benefit)
    Euu = sum(Pu_smooth[t] + np.outer(u_smooth[t], u_smooth[t]) for t in range(T-1))

    # E[x_{t+1} u_t^T] — cross term from augmented smoother
    Exu = sum(
        P_cross[t][:n, n:] + np.outer(x_smooth[t+1], u_smooth[t])
        for t in range(T-1)
    )

    # E[x_t u_t^T]
    Exu_same = sum(np.outer(x_smooth[t], u_smooth[t]) for t in range(T-1))

    # ---- Update A and B jointly ----
    # [A | B] = E[x_{t+1} [x_t; u_t]^T] @ E[[x_t; u_t][x_t; u_t]^T]^{-1}
    Exu_past = sum(
        P_cross[t][:n, n:] + np.outer(x_smooth[t+1], u_smooth[t])   
        for t in range(T-1)
    )

    # Build joint sufficient stat matrix E[[x;u][x;u]^T]  (n+p, n+p)
    E_joint_past = np.zeros((n+p, n+p))
    E_joint_past[:n, :n] = Exx_past
    E_joint_past[:n, n:] = Exu_same
    E_joint_past[n:, :n] = Exu_same.T
    E_joint_past[n:, n:] = Euu

    # RHS: E[x_{t+1} [x_t; u_t]^T]
    E_rhs = np.hstack([Exx_lag, Exu_past])   # (n, n+p)

    AB_new = E_rhs @ inv(E_joint_past)        # (n, n+p)
    A_new  = AB_new[:, :n]
    B_new  = AB_new[:, n:]

    # Stabilise A
    eigvals_A, eigvecs_A = np.linalg.eig(A_new)
    rho = np.abs(eigvals_A).max()
    if rho >= 0.95:
        scale = np.where(np.abs(eigvals_A) >= 0.95, 0.95 / np.abs(eigvals_A), 1.0)
        A_new = np.real(eigvecs_A @ np.diag(eigvals_A * scale) @ inv(eigvecs_A))

    # ---- Update C ----
    Eyx = sum(np.outer(Y[:, t], x_smooth[t]) for t in range(T))
    C_new = Eyx @ inv(Exx)
    col_norms = np.linalg.norm(C_new, axis=0, keepdims=True)
    C_new = C_new / np.maximum(col_norms, 1e-10)

    # ---- Update Q ----
    # Now includes input uncertainty in the residual
    Exx_fut = sum(Px_smooth[t+1] + np.outer(x_smooth[t+1], x_smooth[t+1]) for t in range(T-1))
    Tp = T - 1

    Q_new = (  Exx_fut / Tp
             - A_new @ Exx_lag.T / Tp
             - Exx_lag @ A_new.T / Tp
             + A_new @ Exx_past @ A_new.T / Tp
             + B_new @ Euu @ B_new.T / Tp          # uses E[uu^T] not just u u^T
             - (Exu_past @ B_new.T + B_new @ Exu_past.T) / Tp )

    Q_new = 0.5 * (Q_new + Q_new.T)
    eigv, eigvec = np.linalg.eigh(Q_new)
    y_var = float(np.mean(np.diag(Y @ Y.T / T)))
    eigv  = np.clip(eigv, min_var, y_var)
    Q_new = eigvec @ np.diag(eigv) @ eigvec.T

    # ---- Update R ----
    Eyy   = Y @ Y.T / T
    Eyx_c = C_new @ Exx @ C_new.T / T
    R_new = Eyy - Eyx_c
    R_new = 0.5 * (R_new + R_new.T)
    eigv, eigvec = np.linalg.eigh(R_new)
    R_new = eigvec @ np.diag(np.maximum(eigv, min_var)) @ eigvec.T

    return LDSParams(A=A_new, B=B_new, C=C_new, Q=Q_new, R=R_new,
                     mu0=x_smooth[0].copy(), P0=Px_smooth[0].copy())



# --------------------step 5 EM loop------------------

def update_tau(
    u_smooth: np.ndarray,     # (T, p)
    Pu_smooth: np.ndarray,    # (T, p, p)
    lam: float,               # Laplace rate parameter (sparsity)
) -> np.ndarray:
    """
    M-step update for tau (Laplace scale mixture variances).

    For Laplace prior p(u) = (λ/2) exp(-λ|u|), the EM update for
    the mixing variance τ_t,j is:

        τ_t,j = sqrt(E[u_t,j^2]) / λ

    where E[u_t,j^2] = u_smooth[t,j]^2 + Pu_smooth[t,j,j]
    (posterior second moment including uncertainty).
    """
    T, p = u_smooth.shape
    tau = np.zeros((p, T))
    for t in range(T):
        for j in range(p):
            E_u2 = u_smooth[t, j]**2 + Pu_smooth[t, j, j]   # E[u^2] = mean^2 + var
            tau[j, t] = np.sqrt(E_u2) / lam + 1e-8           # floor to avoid zero
    return tau


def run_em_augmented(
    Y: np.ndarray,
    params: LDSParams,
    lam_sparse: float = 0.1,        # Laplace sparsity parameter
    n_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = True,
    freeze_params_iters: int = 25,  # freeze A,B,C,Q,R for first N iters
) -> tuple[LDSParams, np.ndarray,np.ndarray, list[float]]:
    """
    EM with augmented state [x; u] and Laplace prior on u.

    E-step: RTS smoother over augmented state, tau held fixed
    M-step: update tau (sparsity scales) + model params


    returns
    ---------
    params:
        final LDSParams after EM

    U_final: 
        (T, p) final smoothed input estimates

    ll_history: 
        list of log-likelihood values across EM iterations
    """
    m, T = Y.shape
    n = params.A.shape[0]
    p = params.B.shape[1]

    ll_history = []
    params = params.copy()

    # Initialise tau — start large (permissive, low effective penalty)
    # so early iters can find nonzero inputs before sparsity kicks in
    tau = np.ones((p, T)) * (np.trace(params.Q) / n) / lam_sparse

    for it in range(n_iter):
        # ── E-step ──────────────────────────────────────────────────────────
        x_smooth, u_smooth, Px_s, Pu_s, P_cross, inns, ll = rts_smoother_augmented(Y, params, tau)

        ll_history.append(ll)
        if verbose:
            print(f"  EM iter {it+1:3d}  |  LL = {ll:.4f}  |  "
                  f"u_nnz = {np.sum(np.abs(u_smooth) > 1e-3)}")

        # Convergence
        if it > 0:
            delta = abs(ll - ll_history[-2]) / (abs(ll_history[-2]) + 1e-12)
            if delta < tol:
                if verbose:
                    print(f"  Converged at iter {it+1} (Δ={delta:.2e})")
                break

        # ── M-step: tau update (always) ──────────────────────────────────
        tau = update_tau(u_smooth, Pu_s, lam=lam_sparse)

        # ── M-step: model params (frozen early on) ───────────────────────
        if it >= freeze_params_iters:
            params = _m_step_augmented(
                Y, u_smooth, Pu_s, x_smooth, Px_s, P_cross, params
            )

    U_final = u_smooth.T    # (p, T) to match original convention
    return params, U_final, tau, ll_history



#------------------main estimation workflow------------------

def estimate_latent_and_input(
    observation: np.ndarray,
    LatentDim: int,
    InputDim: int,
) -> tuple[np.ndarray, np.ndarray]:

    T, m = observation.shape
    Y    = observation.T
    
    if T < 2 * (LatentDim + 2) + 1:
        # Too short for SSI — return zeros
        print('too short for ssi to work')
        return np.zeros((T, LatentDim)), np.zeros((T, InputDim))
    
    #need to go through each step now
    #step 1: n4sid

    A_init, C_init = stochastic_subspace_identification(Y, n=LatentDim, i=20)
    B_init, Q_init, R_init = initialise_BQR(Y, A_init, C_init, InputDim, initial_QR_ratio=0.1, percentage_output_variance_explained_by_R=0.5)
    init_params = LDSParams(A=A_init, B=B_init, C=C_init, Q=Q_init, R=R_init,
                            mu0=np.zeros(LatentDim), P0=np.eye(LatentDim))

    
    params, U, tau_final,ll_hist = run_em_augmented(Y = Y,params=init_params,
                                                    lam_sparse = 0.1,
                                                    n_iter = 200,
                                                    tol = 1e-4,
                                                    verbose = True,
                                                    freeze_params_iters = 25)
    

    #final smooth to get latents and inputs
    # with converged params and tau
    x_smooth, u_smooth, Px_smooth, Pu_smooth, _, _, _ = \
        rts_smoother_augmented(Y, params, tau_final)

    latent_states = x_smooth    # (T, n)
    inputs        = u_smooth    # (T, p)

    return latent_states, inputs