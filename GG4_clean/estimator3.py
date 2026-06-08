"""
GG4 High-Fidelity State and Input Estimator
============================================
Pipeline:
  1. Stochastic Subspace Identification (SSI)  →  A, C
  2. B initialisation via SVD of residual
  3. RTS smoother                              →  smoothed states + covariances
  4. Hidden input estimation: regularised OLS
  5. Sparse input prior (LASSO / ISTA)
  6. EM refinement (E-step: RTS + inputs; M-step: parameters)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd, solve, inv, lstsq
from scipy.linalg import solve_discrete_lyapunov


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────
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

class SmootherResult:
    def __init__(self, x_smooth, P_smooth, P_cross, x_filter, P_filter, innovations, log_likelihood):
        self.x_smooth      = x_smooth
        self.P_smooth      = P_smooth
        self.P_cross       = P_cross
        self.x_filter      = x_filter
        self.P_filter      = P_filter
        self.innovations   = innovations
        self.log_likelihood = log_likelihood

# ─────────────────────────────────────────────────────────────────────────────
# Stabilising A matrix for EM initilaisation and Kalman filter + rts smoother for E step
# ─────────────────────────────────────────────────────────────────────────────

def _stabilise_A(A: np.ndarray, max_radius: float = 0.99) -> np.ndarray:
    """Rescale A so spectral radius <= max_radius."""
    rho = np.abs(np.linalg.eigvals(A)).max()
    if rho >= max_radius:
        A = A * (max_radius / rho)
    return A


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Stochastic Subspace Identification
# ─────────────────────────────────────────────────────────────────────────────


#   -> old n4sid

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

    # normalising A and 
    A = _stabilise_A(A)
        
    col_norms = np.linalg.norm(Γ, axis=0, keepdims=True)
    Γ = Γ / np.maximum(col_norms, 1e-10)
    C = Γ[:m, :]


    return A, C


    





# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Initialise B from SVD of residual
# ─────────────────────────────────────────────────────────────────────────────

def initialise_B(
    Y: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    p: int,
    lam: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialise B, Q, R from the one-step prediction residual.

    Strategy
    --------
    1. Run a naive forward pass (no input) to get innovations ε_t = y_t - C x̂_t.
    2. Treat ε as a proxy for the unmeasured-input contribution.
    3. SVD of the lagged residual covariance Σ_ε gives the dominant input directions.

    Returns
    -------
    B   : (n, p)
    Q   : (n, n)  initial process noise estimate
    R   : (m, m)  initial observation noise estimate
    """
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

    # ---- Initial noise covariances ----
    inn_cov = (innovations.T @ innovations) / T        # (m, m)  empirical innovation cov
    R = np.diag(np.diag(inn_cov)) + lam * np.eye(m)   # diagonal + regularisation

   
    y_var = float(np.mean(np.diag(inn_cov)))
    Q = np.eye(n) * y_var * 1.5

    return B, Q, R


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Kalman filter + RTS smoother
# ─────────────────────────────────────────────────────────────────────────────

def kalman_filter(
    Y: np.ndarray,
    U: np.ndarray,
    params: LDSParams,
) -> SmootherResult:
    """Forward Kalman filter.

    Parameters
    ----------
    Y : (m, T)  observations
    U : (p, T)  input estimates (can be zeros on first pass)

    Returns partial SmootherResult with filter quantities filled.
    """
    m, T = Y.shape
    n    = params.A.shape[0]

    x_f  = np.zeros((T, n))
    P_f  = np.zeros((T, n, n))
    inns = np.zeros((T, m))
    ll   = 0.0

    x = params.mu0.copy()
    P = params.P0.copy()

    for t in range(T):
        # Predict
        u_t    = U[:, t]
        x_pred = params.A @ x + params.B @ u_t
        P_pred = params.A @ P @ params.A.T + params.Q

        # Innovation
        inn = Y[:, t] - params.C @ x_pred
        S   = params.C @ P_pred @ params.C.T + params.R

        # Symmetrise S to avoid numerical drift
        S = 0.5 * (S + S.T)

        try:
            S_chol = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            S += np.eye(m) * 1e-8
            S_chol = np.linalg.cholesky(S)

        # Log-likelihood contribution
        sign, logdet = np.linalg.slogdet(S)
        ll += -0.5 * (logdet + inn @ solve(S, inn) + m * np.log(2 * np.pi))

        # Update
        K = P_pred @ params.C.T @ inv(S)
        x = x_pred + K @ inn
        P = (np.eye(n) - K @ params.C) @ P_pred
        P = 0.5 * (P + P.T)

        x_f[t]  = x
        P_f[t]  = P
        inns[t] = inn

    return SmootherResult(
        x_smooth=x_f.copy(),
        P_smooth=P_f.copy(),
        P_cross=np.zeros((T - 1, n, n)),
        x_filter=x_f,
        P_filter=P_f,
        innovations=inns,
        log_likelihood=ll,
    )


def rts_smoother(
    Y: np.ndarray,
    U: np.ndarray,
    params: LDSParams,
    causal:bool = False,
) -> SmootherResult:
    """Rauch-Tung-Striebel smoother.

    Runs forward Kalman, then backward RTS pass.
    Computes lag-1 smoother covariances P_{t+1,t|T} needed for the M-step.

    parameters
    ----------

    causal 
        if you set causal to true then the smoother skips the backwards pass -> kalman filter

    """
    m, T = Y.shape
    n    = params.A.shape[0]

    # ---- Forward pass ----
    res = kalman_filter(Y, U, params)
    x_f = res.x_filter       # (T, n)
    P_f = res.P_filter        # (T, n, n)

    x_s     = np.zeros((T, n))
    P_s     = np.zeros((T, n, n))
    P_cross = np.zeros((T - 1, n, n))   # P_{t+1,t|T}

    x_s[-1] = x_f[-1]
    P_s[-1] = P_f[-1]


    if causal:
        return SmootherResult(
            x_smooth=x_f,
            P_smooth=P_f,
            P_cross=np.zeros((T - 1, n, n)),
            x_filter=x_f,
            P_filter=P_f,
            innovations=res.innovations,
            log_likelihood=res.log_likelihood,
        )

    # ---- Backward pass ----
    for t in range(T - 2, -1, -1):
        u_next  = U[:, t + 1]
        x_pred  = params.A @ x_f[t] + params.B @ u_next
        P_pred  = params.A @ P_f[t] @ params.A.T + params.Q
        P_pred  = 0.5 * (P_pred + P_pred.T)

        # Smoother gain
        G = P_f[t] @ params.A.T @ inv(P_pred)   # (n, n)

        x_s[t] = x_f[t] + G @ (x_s[t + 1] - x_pred)
        P_s[t] = P_f[t] + G @ (P_s[t + 1] - P_pred) @ G.T
        P_s[t] = 0.5 * (P_s[t] + P_s[t].T)

        # Lag-1 cross-covariance:  P_{t+1,t|T} = P_{t+1|T} G_t^T
        P_cross[t] = P_s[t + 1] @ G.T

    return SmootherResult(
        x_smooth=x_s,
        P_smooth=P_s,
        P_cross=P_cross,
        x_filter=x_f,
        P_filter=P_f,
        innovations=res.innovations,
        log_likelihood=res.log_likelihood,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Regularised OLS input estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_inputs_ols(
    x_smooth: np.ndarray,
    params: LDSParams,
    lam: float = 1e-1,
) -> np.ndarray:
    """Recover hidden inputs via Tikhonov-regularised OLS.

    At each time step:
        u_t = (B^T B + λI)^{-1} B^T (x_{t+1|T} - A x_{t|T})

    Parameters
    ----------
    x_smooth : (T, n)
    lam      : L2 regularisation weight

    Returns
    -------
    U : (p, T-1)  input estimates for t = 0 … T-2
    """
    T, n = x_smooth.shape
    p    = params.B.shape[1]
    BtB  = params.B.T @ params.B + lam * np.eye(p)   # (p, p)
    Bt   = params.B.T                                  # (p, n)

    U = np.zeros((p, T - 1))
    for t in range(T - 1):
        residual = x_smooth[t + 1] - params.A @ x_smooth[t]   # (n,)
        U[:, t]  = solve(BtB, Bt @ residual)

    return U


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Sparse input estimation (ISTA / LASSO)
# ─────────────────────────────────────────────────────────────────────────────

def _soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    """Element-wise soft-thresholding operator."""
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def estimate_inputs_sparse(
    x_smooth: np.ndarray,
    params: LDSParams,
    lam: float = 1e-2,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> np.ndarray:
    """Recover sparse hidden inputs via ISTA (proximal gradient for LASSO).

    Minimises over all t:
        ½ ||x_{t+1} - A x_t - B u_t||² + λ ||u_t||₁

    Each timestep is solved independently (parallelisable); ISTA runs per step.

    Returns
    -------
    U : (p, T-1)
    """
    T, n = x_smooth.shape
    p    = params.B.shape[1]
    B    = params.B

    # Lipschitz constant L = λ_max(B^T B)
    BtB = B.T @ B
    L   = float(np.linalg.eigvalsh(BtB).max())
    if L < 1e-12:
        return np.zeros((p, T - 1))
    step = 1.0 / L

    U = np.zeros((p, T - 1))
    for t in range(T - 1):
        r = x_smooth[t + 1] - params.A @ x_smooth[t]   # (n,)
        u = np.zeros(p)
        for _ in range(max_iter):
            grad  = BtB @ u - B.T @ r                  # (p,)
            u_new = _soft_threshold(u - step * grad, step * lam)
            if np.linalg.norm(u_new - u) < tol:
                u = u_new
                break
            u = u_new
        U[:, t] = u

    return U


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — EM refinement
# ─────────────────────────────────────────────────────────────────────────────

def _m_step(
    Y: np.ndarray,
    U: np.ndarray,
    res: SmootherResult,
    params: LDSParams,
    update_B: bool = True,
    freeze_Q: bool = False,
    freeze_R: bool = False,
    min_var: float = 1e-8,
) -> LDSParams:
    """M-step: closed-form parameter updates given smoothed sufficient statistics.

    Uses the standard LDS M-step with input term.

    Sufficient statistics:
        E1  = Σ_t E[x_t]                    (sum of smoothed means)
        Exx = Σ_t E[x_t x_t^T]             (includes P_smooth + x_smooth x_smooth^T)
        Exx_lag = Σ_t E[x_{t+1} x_t^T]     (P_cross + outer products)
    """
    m, T = Y.shape
    n    = params.A.shape[0]
    p    = params.B.shape[1]

    xs   = res.x_smooth    # (T, n)
    Ps   = res.P_smooth    # (T, n, n)
    Pc   = res.P_cross     # (T-1, n, n)

    # ---- Sufficient statistics ----
    # Σ_t x_t x_t^T  (t = 0..T-1)
    Exx      = sum(Ps[t] + np.outer(xs[t], xs[t]) for t in range(T))

    # Σ_t x_t x_t^T  (t = 0..T-2)
    Exx_past = sum(Ps[t]     + np.outer(xs[t],     xs[t])     for t in range(T - 1))

    # Σ_t x_{t+1} x_t^T
    Exx_lag  = sum(Pc[t]     + np.outer(xs[t + 1], xs[t])     for t in range(T - 1))

    # Input contribution: Σ_t B u_t x_t^T  (t = 0..T-2)
    BU_xt    = sum(np.outer(params.B @ U[:, t], xs[t]) for t in range(T - 1))

    # ---- Update C ----
    # C = (Σ y_t x_t^T) @ (Σ x_t x_t^T)^{-1}
    Eyx = Y @ xs / 1.0           # (m, n), but we need sum form
    Eyx = sum(np.outer(Y[:, t], xs[t]) for t in range(T))   # (m, n)
    C_new = Eyx @ inv(Exx)

    col_norms = np.linalg.norm(C_new, axis=0, keepdims=True)
    C_new = C_new / np.maximum(col_norms, 1e-10)

    # ---- Update A ----
    # A = (Exx_lag - B Σ u_t x_t^T) @ Exx_past^{-1}
    A_new = (Exx_lag - BU_xt) @ inv(Exx_past)

    eigvals_A, eigvecs_A = np.linalg.eig(A_new)
    rho = np.abs(eigvals_A).max()
    if rho >= 0.95:
        scale = np.where(np.abs(eigvals_A) >= 0.95, 0.95 / np.abs(eigvals_A), 1.0)
        eigvals_clipped = eigvals_A * scale
        A_new = np.real(eigvecs_A @ np.diag(eigvals_clipped) @ np.linalg.inv(eigvecs_A))
    

    # ---- Update B ----
    if update_B:
        # Residual after A update: r_t = E[x_{t+1}] - A x_t
        # B = (Σ r_t u_t^T) @ (Σ u_t u_t^T)^{-1}
        R_xu = sum(np.outer(xs[t + 1] - A_new @ xs[t], U[:, t]) for t in range(T - 1))
        UUt  = U @ U.T + 1e-8 * np.eye(p)                     # (p, p)
        B_new = R_xu @ inv(UUt)
    else:
        B_new = params.B.copy()

    # ---- Update Q ----


    if freeze_Q == True:
        Q_new = params.Q.copy()
    else:

        # Q = (1/(T-1)) Σ_t E[(x_{t+1} - A x_t - B u_t)(x_{t+1} - A x_t - B u_t)^T]
        # Expanded:  (1/(T-1)) [ Σ E[x_{t+1}x_{t+1}^T]
        #                       - A_new Σ E[x_t x_{t+1}^T]
        #                       - Σ B u_t E[x_{t+1}]^T
        #                       + (A_new Exx_past A_new^T + B UU^T B^T + cross terms) ]
        # Using the cleaner form: Q = Exx_fut/T' - A Exx_lag^T/T' - A Exx_lag/T' * A^T ... 
        # Standard result:
        Exx_fut = sum(Ps[t + 1] + np.outer(xs[t + 1], xs[t + 1]) for t in range(T - 1))
        Tp = T - 1  # number of transitions

        # Cross terms involving inputs
        BU     = params.B @ U                                    # (n, T-1)  B u_t for each t
        EBU_xt = sum(np.outer(BU[:, t], xs[t]) for t in range(Tp))   # (n, n)  Σ Bu_t x_t^T

        Q_new  = (  Exx_fut / Tp
                - A_new @ Exx_lag.T / Tp
                - Exx_lag @ A_new.T / Tp
                + A_new @ Exx_past @ A_new.T / Tp
                + B_new @ U @ U.T @ B_new.T / Tp
                + (EBU_xt @ A_new.T + A_new @ EBU_xt.T) / Tp )


        Q_new   = 0.5 * (Q_new + Q_new.T)
        eigv, eigvec = np.linalg.eigh(Q_new)
    
        # after computing Q_new Cap Q size:
        y_var = float(np.mean(np.diag(np.asarray(Y) @ np.asarray(Y).T / Y.shape[1])))
        eigv = np.clip(eigv, min_var, y_var)
        Q_new = eigvec @ np.diag(eigv) @ eigvec.T

    # ---- Update R ----
    if freeze_R:
        R_new = params.R.copy()
    else:

        # R = (1/T) Σ_t [y_t y_t^T - C E[x_t] y_t^T]
        Eyy   = Y @ Y.T / T                                        # (m, m)
        Eyx_c = C_new @ Exx @ C_new.T / T
        R_new = Eyy - Eyx_c
        R_new = 0.5 * (R_new + R_new.T)
        eigv, eigvec = np.linalg.eigh(R_new)
        R_new = eigvec @ np.diag(np.maximum(eigv, min_var)) @ eigvec.T

    # ---- Initial state ----
    mu0_new = xs[0].copy()
    P0_new  = Ps[0].copy()

        
    return LDSParams(A=A_new, C=C_new, B=B_new, Q=Q_new, R=R_new,
                    mu0=mu0_new, P0=P0_new)


def run_em(
    Y: np.ndarray,
    params: LDSParams,
    n_iter: int = 100,
    sparse: bool = True,
    lam_input: float = 1e-2,
    lam_ols: float = 1e-2,
    update_B: bool = True,
    tol: float = 1e-4,
    verbose: bool = True,
    causal:bool = False,
) -> tuple[LDSParams, np.ndarray, list[float]]:
    """EM loop over E-step (RTS + input estimation) and M-step (parameter update).

    Parameters
    ----------
    Y          : (m, T)  observations
    params     : initial LDSParams
    n_iter     : maximum EM iterations
    sparse     : use ISTA (L1) rather than ridge (L2) for input estimation
    lam_input  : sparsity / regularisation weight for input estimator
    lam_ols    : ridge weight if sparse=False
    update_B   : whether to re-estimate B in the M-step
    tol        : convergence threshold on relative LL change
    verbose    : print LL per iteration

    Returns
    -------
    params_final : refined LDSParams
    U_final      : (p, T-1)  final input estimates
    ll_history   : list of log-likelihoods
    """
    m, T = Y.shape
    p    = params.B.shape[1]
    ll_history: list[float] = []
    params = params.copy()

    # Initialise inputs to zero
    U = np.zeros((p, T - 1))

    freeze_Q_iters: int = 25  # freeze Q for first 20 iters
    freeze_R_iters: int = 25  # freeze R for first 20 iters

    for it in range(n_iter):
        # ── E-step ──────────────────────────────────────────────────────────
        # Pad U to length T for the smoother (last column duplicated)
        U_full = np.hstack([U, U[:, -1:]])   # (p, T)
        res    = rts_smoother(Y, U_full, params,causal=causal)

        # Re-estimate inputs from smoothed states
        if sparse:
            lam_warmup = lam_input * 0.1   # very low threshold for first 10 iters  this allows to grab any non-zero inputs  gives the B estimation something to work with
            lam_this_iter = lam_warmup if it < 10 else lam_input
            U = estimate_inputs_sparse(res.x_smooth, params, lam=lam_this_iter)
        else:
            U = estimate_inputs_ols(res.x_smooth, params, lam=lam_ols)

        ll = res.log_likelihood
        ll_history.append(ll)

        if verbose:
            print(f"  EM iter {it + 1:3d}  |  LL = {ll:.4f}")

        # Convergence check
        if it > 0:
            delta = abs(ll - ll_history[-2]) / (abs(ll_history[-2]) + 1e-12)
            if delta < tol:
                if verbose:
                    print(f"  Converged at iter {it + 1} (Δ={delta:.2e})")
                break

        # ── M-step ──────────────────────────────────────────────────────────
        update_B_this_iter = update_B and (it >= 25)   # freeze for first 25 iters   so B doesnt grow, greedy allocation of explained variance in residuals to inputs
        freeze_Q = it < freeze_Q_iters
        freeze_R = it < freeze_R_iters
        params = _m_step(Y, U, res, params,
                         update_B=update_B_this_iter,
                         freeze_Q=freeze_Q,
                         freeze_R=freeze_R)
     
       
        

    return params, U, ll_history


# ─────────────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ─────────────────────────────────────────────────────────────────────────────

def fit(
    Y: np.ndarray,
    n: int,
    p: int,
    ssi_i: int           = 20,
    lam_B: float         = 1e-2,
    lam_input: float     = 0.9,          
    sparse_inputs: bool  = True,
    em_iter: int         = 50,
    em_tol: float        = 1e-4,
    verbose: bool        = True,
    causal: bool = False,
) -> tuple[LDSParams, np.ndarray, SmootherResult, list[float]]:
    """Full GG4 pipeline: SSI → B init → RTS → sparse inputs → EM.

    Parameters
    ----------
    Y               : (m, T)  observation matrix
    n               : latent state dimension
    p               : hidden input dimension
    ssi_i           : SSI block rows (≥ n // m + 1)
    lam_B           : regularisation for B initialisation
    lam_input       : sparsity weight for input estimation (ISTA λ)
    sparse_inputs   : use L1 (True) or L2 (False) for input prior
    em_iter         : max EM iterations
    em_tol          : EM convergence tolerance on relative LL change
    verbose         : print progress

    Returns
    -------
    params    : final LDSParams
    U         : (p, T-1)  estimated inputs
    smoother  : final SmootherResult
    ll_hist   : log-likelihood per EM iteration
    """
    m, T = Y.shape
    assert T > 2 * ssi_i, f"Need T > 2*ssi_i={2*ssi_i}, got T={T}"

    # if m < n:
    #     # Reduce latent dim to match observable subspace
    #     n = m
    #     print(f"Warning: m={m} < n requested, reducing to n={m}")

    # ── Stage 1: SSI ────────────────────────────────────────────────────────
    if verbose:
        print("Stage 1: Stochastic Subspace Identification")
    A, C = stochastic_subspace_identification(Y, n=n, i=ssi_i)

    eigs = np.linalg.eigvals(A)
    
    print(f"A eigenvalues (magnitude): {np.abs(eigs)}")
    print(f"A eigenvalues (angle deg): {np.angle(eigs, deg=True)}")
    print(f"Spectral radius: {np.abs(eigs).max():.4f}")
        

    # ── Stage 2: Initialise B ────────────────────────────────────────────────
    if verbose:
        print("Stage 2: Initialising B via residual SVD")
    B, Q, R = initialise_B(Y, A, C, p=p, lam=lam_B)

    mu0 = np.zeros(n)
    P0  = solve_discrete_lyapunov(A, Q)

    params = LDSParams(A=A, C=C, B=B, Q=Q, R=R, mu0=mu0, P0=P0)

    # ── Stage 3 + 4/5: Initial RTS + input estimation ───────────────────────
    if verbose:
        print("Stage 3: Initial RTS smoother pass")
    U_full = np.zeros((p, T))
    res    = rts_smoother(Y, U_full, params, causal = causal)


    # Residual SVD for B — use smoothed states
    residuals = res.x_smooth[1:] - (params.A @ res.x_smooth[:-1].T).T  # (T-1, n)
    R_mat = residuals.T                                                    # (n, T-1)
    U_svd, _, _ = svd(R_mat, full_matrices=False)
    params.B = U_svd[:, :p]   # overwrite B with better estimate

    if verbose:
        print("Stage 4/5: Initial input estimation (sparse={})".format(sparse_inputs))
    if sparse_inputs:
        U = estimate_inputs_sparse(res.x_smooth, params, lam=lam_input)
    else:
        U = estimate_inputs_ols(res.x_smooth, params, lam=lam_input)



    print('ihdahsdhadvhadva')
    # adaptive regularisation
    residuals = np.diff(res.x_smooth, axis=0) - (params.A @ res.x_smooth[:-1].T).T
    Btr_norms = np.array([float(np.linalg.norm(params.B.T @ residuals[t])) for t in range(len(residuals))])
    # Set lam to be 20% of the median non-zero B^T r norm
    adaptive_lam = float(np.percentile(Btr_norms, 80)) * lam_input

    print(f"Btr_norms 80th pct: {float(np.percentile(Btr_norms, 80)):.4f}")
    print(f"Btr_norms max: {float(np.max(Btr_norms)):.4f}")
    print(f"adaptive_lam: {adaptive_lam:.4f}")

    print(f"Residual std: {residuals.std():.4f}")
    print(f"Residual max: {np.abs(residuals).max():.4f}")
    print(f"lam_input: {lam_input}")
    print(f"B norm: {np.linalg.norm(params.B):.4f}")




    # ── Stage 6: EM ──────────────────────────────────────────────────────────
    if verbose:
        print("Stage 6: EM refinement")

    params, U, ll_hist = run_em(
        Y, params,
        n_iter=em_iter,
        sparse=sparse_inputs,
        lam_input=adaptive_lam,
        verbose=verbose,
        tol=em_tol,
        causal = causal,
    )

    # Final smoother pass with converged parameters
    U_full = np.hstack([U, U[:, -1:]])
    final_res = rts_smoother(Y, U_full, params,causal = causal)

    return params, U, final_res, ll_hist


def festimate_latent_and_input(
    observation: np.ndarray,
    LatentDim: int,
    InputDim: int,
    ssi_i: int | None   = None,
    lam_B: float        = 0.5,
    lam_input: float    = 1.2,
    sparse_inputs: bool = True,
    em_iter: int        = 200,
    em_tol: float       = 1e-4,
    n_restarts: int     = 3,
    verbose: bool       = False,
    causal:bool         = True,
) -> tuple[np.ndarray, np.ndarray]:

    T, m = observation.shape
    Y    = observation.T

    if ssi_i is None:
        ssi_i = min(2 * LatentDim, (T - 1) // 4)
        ssi_i = max(ssi_i, LatentDim + 2)

    
    if T < 2 * (LatentDim + 2) + 1:
        # Too short for SSI — return zeros
        print('too short for ssi to work')
        return np.zeros((T, LatentDim)), np.zeros((T, InputDim))

    params, U, final_res, ll_hist = fit(
        Y,
        n             = LatentDim,
        p             = InputDim,
        ssi_i         = ssi_i,
        lam_B         = lam_B,
        lam_input     = lam_input,
        sparse_inputs = sparse_inputs,
        em_iter       = em_iter,
        em_tol        = em_tol,
        verbose       = verbose,
        causal        = causal,
    )

    latent_states = final_res.x_smooth
    inputs        = U.T
    inputs        = np.vstack([inputs, inputs[-1:, :]])

    return latent_states, inputs