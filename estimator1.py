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

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.linalg import svd, solve, inv, lstsq
from scipy.linalg import solve_discrete_lyapunov


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LDSParams:
    """Linear Dynamical System parameters.

    Model:
        x_{t+1} = A x_t + B u_t + w_t,   w_t ~ N(0, Q)
        y_t     = C x_t       + v_t,      v_t ~ N(0, R)
    """
    A:   np.ndarray          # (n, n)  state transition
    C:   np.ndarray          # (m, n)  observation matrix
    B:   np.ndarray          # (n, p)  input matrix
    Q:   np.ndarray          # (n, n)  process noise covariance
    R:   np.ndarray          # (m, m)  observation noise covariance
    mu0: np.ndarray          # (n,)    initial state mean
    P0:  np.ndarray          # (n, n)  initial state covariance

    def copy(self) -> LDSParams:
        return LDSParams(**{k: v.copy() for k, v in self.__dict__.items()})


@dataclass
class SmootherResult:
    """Output of the RTS smoother."""
    x_smooth:   np.ndarray   # (T, n)   E[x_t | y_{1:T}]
    P_smooth:   np.ndarray   # (T, n, n) Var[x_t | y_{1:T}]
    P_cross:    np.ndarray   # (T-1, n, n) E[x_{t+1} x_t^T | y_{1:T}] - lag-1 smoother cov
    x_filter:   np.ndarray   # (T, n)   filtered means (forward pass)
    P_filter:   np.ndarray   # (T, n, n) filtered covariances
    innovations: np.ndarray  # (T, m)   y_t - C x̂_{t|t-1}
    log_likelihood: float


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Stochastic Subspace Identification
# ─────────────────────────────────────────────────────────────────────────────

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
    i : int     number of block rows (≥ n/m, rule of thumb: 2×–4× that)

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
    U1 = U[:, :n]                              # (im, n)
    S1 = np.diag(s[:n])                        # (n,  n)
    Γ  = U1 @ np.sqrt(S1)                      # (im, n)  observability matrix

    # 4. Extract C from first m rows of Γ
    C = Γ[:m, :]                               # (m, n)

    # 5. Extract A via shift invariance:  Γ_{1..i-1} A = Γ_{2..i}
    Γ_up   = Γ[:(i - 1) * m, :]               # ((i-1)m, n)
    Γ_down = Γ[m:,           :]               # ((i-1)m, n)
    A, _, _, _ = lstsq(Γ_up, Γ_down, rcond=None)   # (n, n)

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

    # ---- SVD of lagged innovation outer product ----
    # Form  Σ_{ε,lag1} = (1/T-1) Σ_t ε_{t+1} ε_t^T
    eps_future = innovations[1:, :]   # (T-1, m)
    eps_past   = innovations[:-1, :]  # (T-1, m)
    Sigma_lag  = (eps_future.T @ eps_past) / (T - 1)   # (m, m)

    # Map to state space via C^+: state residual ≈ C^+ ε
    # Residual state dynamics: r_{t+1} ≈ B u_t  →  SVD of C^+ Σ_lag C^{+T}
    C_pinv  = np.linalg.pinv(C)                        # (n, m)
    Sigma_x = C_pinv @ Sigma_lag @ C_pinv.T            # (n, n)

    U, s, _ = svd(Sigma_x)
    B = U[:, :p] * np.sqrt(s[:p])                      # (n, p)  leading input directions

    # ---- Initial noise covariances ----
    inn_cov = (innovations.T @ innovations) / T        # (m, m)  empirical innovation cov
    R = np.diag(np.diag(inn_cov)) + lam * np.eye(m)   # diagonal + regularisation

    # Q from steady-state Lyapunov approximation given R
    # APA^T - P + Q = 0 with P from innovations
    P_ss = C_pinv @ inn_cov @ C_pinv.T
    try:
        Q = P_ss - A @ P_ss @ A.T
        # Project to nearest PSD
        eigvals, eigvecs = np.linalg.eigh(Q)
        Q = eigvecs @ np.diag(np.maximum(eigvals, lam)) @ eigvecs.T
    except Exception:
        Q = np.eye(n) * lam

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
) -> SmootherResult:
    """Rauch-Tung-Striebel smoother.

    Runs forward Kalman, then backward RTS pass.
    Computes lag-1 smoother covariances P_{t+1,t|T} needed for the M-step.
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
    lam: float = 1e-2,
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

    # ---- Update A ----
    # A = (Exx_lag - B Σ u_t x_t^T) @ Exx_past^{-1}
    A_new = (Exx_lag - BU_xt) @ inv(Exx_past)

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
    Q_new   = eigvec @ np.diag(np.maximum(eigv, min_var)) @ eigvec.T

    # ---- Update R ----
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

    # ---- Enforce A stability: clip spectral radius to < 1 ----
    eigvals_A, eigvecs_A = np.linalg.eig(A_new)
    rho = np.abs(eigvals_A).max()
    if rho >= 1.0:
        A_new = A_new * (0.99 / rho)

    return LDSParams(A=A_new, C=C_new, B=B_new, Q=Q_new, R=R_new,
                     mu0=mu0_new, P0=P0_new)


def run_em(
    Y: np.ndarray,
    params: LDSParams,
    n_iter: int = 50,
    sparse: bool = True,
    lam_input: float = 1e-2,
    lam_ols: float = 1e-2,
    update_B: bool = True,
    tol: float = 1e-4,
    verbose: bool = True,
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

    for it in range(n_iter):
        # ── E-step ──────────────────────────────────────────────────────────
        # Pad U to length T for the smoother (last column duplicated)
        U_full = np.hstack([U, U[:, -1:]])   # (p, T)
        res    = rts_smoother(Y, U_full, params)

        # Re-estimate inputs from smoothed states
        if sparse:
            U = estimate_inputs_sparse(res.x_smooth, params, lam=lam_input)
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
        params = _m_step(Y, U, res, params, update_B=update_B)

    return params, U, ll_history


# ─────────────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ─────────────────────────────────────────────────────────────────────────────

def fit(
    Y: np.ndarray,
    n: int,
    p: int,
    ssi_i: int           = 20,
    lam_B: float         = 1e-3,
    lam_input: float     = 1e-2,
    sparse_inputs: bool  = True,
    em_iter: int         = 50,
    em_tol: float        = 1e-4,
    verbose: bool        = True,
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

    # ── Stage 1: SSI ────────────────────────────────────────────────────────
    if verbose:
        print("Stage 1: Stochastic Subspace Identification")
    A, C = stochastic_subspace_identification(Y, n=n, i=ssi_i)

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
    res    = rts_smoother(Y, U_full, params)

    if verbose:
        print("Stage 4/5: Initial input estimation (sparse={})".format(sparse_inputs))
    if sparse_inputs:
        U = estimate_inputs_sparse(res.x_smooth, params, lam=lam_input)
    else:
        U = estimate_inputs_ols(res.x_smooth, params, lam=lam_input)

    # ── Stage 6: EM ──────────────────────────────────────────────────────────
    if verbose:
        print("Stage 6: EM refinement")
    params, U, ll_hist = run_em(
        Y, params,
        n_iter=em_iter,
        sparse=sparse_inputs,
        lam_input=lam_input,
        verbose=verbose,
        tol=em_tol,
    )

    # Final smoother pass with converged parameters
    U_full = np.hstack([U, U[:, -1:]])
    final_res = rts_smoother(Y, U_full, params)

    return params, U, final_res, ll_hist


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Ground truth system
    n_true, m_true, p_true, T = 4, 6, 2, 500

    A_true = 0.95 * np.linalg.qr(rng.standard_normal((n_true, n_true)))[0]
    C_true = rng.standard_normal((m_true, n_true))
    B_true = rng.standard_normal((n_true, p_true))
    Q_true = 0.01 * np.eye(n_true)
    R_true = 0.1  * np.eye(m_true)

    # Sparse inputs: events at ~5% of timesteps
    U_true = np.zeros((p_true, T))
    event_times = rng.choice(T, size=int(0.05 * T), replace=False)
    U_true[:, event_times] = rng.standard_normal((p_true, len(event_times)))

    # Simulate
    x = rng.multivariate_normal(np.zeros(n_true), Q_true)
    Y = np.zeros((m_true, T))
    for t in range(T):
        Y[:, t] = C_true @ x + rng.multivariate_normal(np.zeros(m_true), R_true)
        x = A_true @ x + B_true @ U_true[:, t] + rng.multivariate_normal(np.zeros(n_true), Q_true)

    print("=== GG4 Estimator — smoke test ===")
    params, U_est, smoother, ll_hist = fit(
        Y, n=n_true, p=p_true, ssi_i=20, lam_input=5e-3,
        sparse_inputs=True, em_iter=30, verbose=True,
    )

    print(f"\nFinal LL        : {ll_hist[-1]:.4f}")
    print(f"EM iterations   : {len(ll_hist)}")
    print(f"Input sparsity  : {np.mean(np.abs(U_est) < 1e-4):.2%} zeros")
    print(f"A shape         : {params.A.shape}")
    print(f"B shape         : {params.B.shape}")
    print(f"C shape         : {params.C.shape}")