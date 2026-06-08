"""
system_id.py
============
Full pipeline for identifying LDS matrices from a Brain interface.

Workflow
--------
1. collect_data        — Schroeder multisine input, one long trial
2. n4sid               — oblique-projection subspace ID (data-driven init)
3. select_latent_dim   — BIC on singular values to pick state dimension
4. identify_and_refine — N4SID init + RTS-EM refinement
5. validate            — one-step-ahead VAF on held-out data

Usage
-----
    brain = Brain(random_seed=0)
    result = run_full_pipeline(brain)
    print(result.lds)          # fitted matrices
    print(result.vaf)          # validation VAF
"""

import numpy as np
import scipy.linalg as la
from dataclasses import dataclass
from typing import Optional
from scipy.signal import StateSpace, dlsim
from scipy.optimize import minimize

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LDSParams:
    A:    np.ndarray   # (d, d)   state transition
    B:    np.ndarray   # (d, M)   input matrix
    C:    np.ndarray   # (N, d)   observation matrix
    Q:    np.ndarray   # (d, d)   process noise covariance
    R:    np.ndarray   # (N, N)   observation noise covariance
    mu_0: np.ndarray   # (d,)     initial state mean
    P_0:  np.ndarray   # (d, d)   initial state covariance

    def __repr__(self):
        d, N, M = self.A.shape[0], self.C.shape[0], self.B.shape[1]
        eigs = np.linalg.eigvals(self.A)
        return (
            f"LDSParams(d={d}, N={N}, M={M})\n"
            f"  A eigenvalues : {np.round(eigs, 3)}\n"
            f"  |eigs|        : {np.round(np.abs(eigs), 3)}\n"
            f"  stable        : {bool(np.all(np.abs(eigs) < 1))}"
        )


@dataclass
class PipelineResult:
    lds:          LDSParams
    X_smooth:     np.ndarray   # (T_train, d) smoothed states on training data
    Y_train:      np.ndarray   # (T_train, N)
    U_train:      np.ndarray   # (T_train, M)
    Y_val:        np.ndarray   # (T_val, N)
    U_val:        np.ndarray   # (T_val, M)
    vaf:          float        # validation VAF (%)
    latent_dim:   int
    num_lags:     int
    ll_history:   np.ndarray   # log-likelihood per EM iteration
    singular_vals: np.ndarray  # from projected Hankel SVD
    ic_scores:    np.ndarray   # BIC scores per candidate dimension


# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA COLLECTION
# ──────────────────────────────────────────────────────────────────────────────

def collect_data(
    brain,
    n_samples:    int = 2000,
    n_settle:     int = 200,
    val_fraction: float = 0.2,
    amplitude:    float = 0.45,   # keeps signal in [0.05, 0.95] after centering
    n_freqs:      int  = 80,
    seed:         int  = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Probe the brain with a Schroeder multisine and collect (Y, U).

    Schroeder multisine: deterministic signal with flat power spectrum and
    minimised crest factor via phase staggering  φ_k = -k(k-1)/F * π.
    Strictly better than uniform random for exciting all modes within a
    bounded amplitude constraint.

    The signal is centred at 0.5 and scaled so it stays in [0, 1].

    Returns
    -------
    Y_train, U_train, Y_val, U_val  — all (T, dim), time-first convention
    """
    rng      = np.random.default_rng(seed)
    U_dim    = brain.input_dim
    Y_dim    = len(np.array(brain.measure()))
    t        = np.arange(n_samples)

    # Build Schroeder multisine base signal
    freqs  = np.linspace(1.0 / n_samples, 0.49, n_freqs)
    k      = np.arange(n_freqs)
    phases = -(k * (k - 1) / n_freqs) * np.pi

    u_base = np.zeros(n_samples)
    for i in range(n_freqs):
        u_base += np.cos(2 * np.pi * freqs[i] * t + phases[i])

    # Normalise to [-amplitude, +amplitude] then shift to [0.5-amp, 0.5+amp]
    u_base = u_base / np.max(np.abs(u_base)) * amplitude + 0.5

    # Build input matrix: each channel is a shifted/inverted version
    # so channels are uncorrelated (essential for B identifiability)
    U_data = np.zeros((n_samples, U_dim))
    for ch in range(U_dim):
        shift = int(ch * n_samples / U_dim)
        sign  = (-1) ** ch
        U_data[:, ch] = np.clip(
            0.5 + sign * (np.roll(u_base, shift) - 0.5),
            0.0, 1.0
        )

    # Settle the system to a steady state before recording
    print(f"  Settling ({n_settle} steps)...")
    for _ in range(n_settle):
        brain.next_state(np.zeros(U_dim) + 0.5)

    # Collect data
    Y_data = np.zeros((n_samples, Y_dim))
    print(f"  Collecting {n_samples} samples...")
    for k in range(n_samples):
        Y_data[k, :] = np.array(brain.measure())
        brain.next_state(U_data[k, :])

    # Train / validation split
    split   = int(n_samples * (1 - val_fraction))
    Y_train = Y_data[:split]
    U_train = U_data[:split]
    Y_val   = Y_data[split:]
    U_val   = U_data[split:]

    print(f"  Done. Train: {Y_train.shape}, Val: {Y_val.shape}")
    return Y_train, U_train, Y_val, U_val
import numpy as np


def collect_hybrid_data(
    brain,
    n_samples:           int = 2500,
    n_settle:            int = 200,
    val_fraction:        float = 0.2,
    amplitude:           float = 0.45,
    freq_min:            float = 0.001, # Default to broadband for Stage 1
    freq_max:            float = 0.49,
    num_trials:          int = 5,       # Number of ensemble averaging passes
    seed:                int = 0,
    **kwargs                            # Safely catches old unused args like impulse_spacing
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Replaces the old impulse train with a Band-Targeted Schroeder Multisine 
    and Ensemble Averaging to mathematically crush measurement noise.
    """
    rng   = np.random.default_rng(seed)
    U_dim = brain.input_dim
    Y_dim = len(np.array(brain.measure()))
    
    # Divide total requested samples into identical repeating trials
    n_samples_per_trial = n_samples // num_trials
    
    # Dynamically scale the number of frequencies based on trial length
    n_freqs = min(80, n_samples_per_trial // 4) 
    
    # ---------------------------------------------------------
    # 1. Build the Targeted Schroeder Multisine
    # ---------------------------------------------------------
    t = np.arange(n_samples_per_trial)
    freqs = np.linspace(freq_min, freq_max, n_freqs)
    
    # Schroeder phase formula minimizes peaking (keeps crest factor low)
    k = np.arange(n_freqs)
    phases = -(k * (k - 1) / n_freqs) * np.pi

    u_base = np.zeros(n_samples_per_trial)
    for i in range(n_freqs):
        u_base += np.cos(2 * np.pi * freqs[i] * t + phases[i])

    # Scale to precisely hit the amplitude cap, centered at 0.5
    u_base = (u_base / np.max(np.abs(u_base))) * amplitude + 0.5

    # Shift and invert channels to ensure they are uncorrelated (orthogonal)
    U_trial = np.zeros((n_samples_per_trial, U_dim))
    for ch in range(U_dim):
        shift = int(ch * n_samples_per_trial / U_dim)
        sign  = (-1) ** ch
        U_trial[:, ch] = np.clip(
            0.5 + sign * (np.roll(u_base, shift) - 0.5),
            0.0, 1.0
        )

    # ---------------------------------------------------------
    # 2. Settle the System
    # ---------------------------------------------------------
    print(f"  Settling ({n_settle} steps) at baseline 0.5...")
    for _ in range(n_settle):
        brain.next_state(np.zeros(U_dim) + 0.5)

    # ---------------------------------------------------------
    # 3. Collect Ensemble Averaged Data
    # ---------------------------------------------------------
    Y_all_trials = np.zeros((num_trials, n_samples_per_trial, Y_dim))
    
    print(f"  Collecting {num_trials} identical trials of {n_samples_per_trial} steps (Total: {n_samples})...")
    for trial in range(num_trials):
        for k in range(n_samples_per_trial):
            Y_all_trials[trial, k, :] = np.array(brain.measure())
            brain.next_state(U_trial[k, :])

    # The Magic: Average across trials to destroy zero-mean Gaussian noise
    Y_averaged = np.mean(Y_all_trials, axis=0)

    # ---------------------------------------------------------
    # 4. Train / Validation Split (On the Averaged Data)
    # ---------------------------------------------------------
    split   = int(n_samples_per_trial * (1 - val_fraction))
    Y_train = Y_averaged[:split]
    U_train = U_trial[:split]
    Y_val   = Y_averaged[split:]
    U_val   = U_trial[split:]

    return Y_train, U_train, Y_val, U_val
def collect_hybrid_dataa(
    brain,
    n_samples:       int   = 2000,
    n_settle:        int   = 200,
    val_fraction:    float = 0.2,
    multisine_amp:   float = 0.45,
    impulse_amp:     float = 0.45,  # Max amplitude for the strike
    impulse_spacing: int   = 50,    # Number of time steps between strikes
    seed:            int   = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Probe the system with a hybrid signal:
    First half: Schroeder multisine for continuous broadband excitation.
    Second half: Sparse impulse train to capture high-frequency transient impacts.
    """
    rng      = np.random.default_rng(seed)
    U_dim    = brain.input_dim
    Y_dim    = len(np.array(brain.measure()))
    
    n_multi    = n_samples // 2
    n_impulses = n_samples - n_multi
    
    # ---------------------------------------------------------
    # 1. Build the Schroeder multisine (First Half)
    # ---------------------------------------------------------
    t_multi = np.arange(n_multi)
    n_freqs = 80
    freqs   = np.linspace(1.0 / n_multi, 0.49, n_freqs)
    k       = np.arange(n_freqs)
    phases  = -(k * (k - 1) / n_freqs) * np.pi

    u_base = np.zeros(n_multi)
    for i in range(n_freqs):
        u_base += np.cos(2 * np.pi * freqs[i] * t_multi + phases[i])

    u_base = u_base / np.max(np.abs(u_base)) * multisine_amp + 0.5

    U_multi = np.zeros((n_multi, U_dim))
    for ch in range(U_dim):
        shift = int(ch * n_multi / U_dim)
        sign  = (-1) ** ch
        U_multi[:, ch] = np.clip(
            0.5 + sign * (np.roll(u_base, shift) - 0.5),
            0.0, 1.0
        )

    # ---------------------------------------------------------
    # 2. Build the Sparse Impulse Train (Second Half)
    # ---------------------------------------------------------
    U_impulse = np.zeros((n_impulses, U_dim)) + 0.5  # Baseline at 0.5
    
    for ch in range(U_dim):
        # Stagger the start times for each channel so they don't all hit at once
        start_idx = rng.integers(0, impulse_spacing)
        
        # Place a 1-step impulse at regular, sparse intervals
        strike_indices = np.arange(start_idx, n_impulses, impulse_spacing)
        
        # Randomly flip the direction of the impulse (positive or negative strike)
        directions = rng.choice([-1, 1], size=len(strike_indices))
        U_impulse[strike_indices, ch] = np.clip(
            0.5 + (impulse_amp * directions), 
            0.0, 1.0
        )

    # Combine the two signals
    U_data = np.vstack([U_multi, U_impulse])

    # ---------------------------------------------------------
    # 3. Settle and Collect
    # ---------------------------------------------------------
    print(f"  Settling ({n_settle} steps)...")
    for _ in range(n_settle):
        brain.next_state(np.zeros(U_dim) + 0.5)

    Y_data = np.zeros((n_samples, Y_dim))
    print(f"  Collecting {n_samples} hybrid samples...")
    for k in range(n_samples):
        Y_data[k, :] = np.array(brain.measure())
        brain.next_state(U_data[k, :])

    # Train / validation split
    split   = int(n_samples * (1 - val_fraction))
    Y_train = Y_data[:split]
    U_train = U_data[:split]
    Y_val   = Y_data[split:]
    U_val   = U_data[split:]

    return Y_train, U_train, Y_val, U_val

# ──────────────────────────────────────────────────────────────────────────────
# 2. INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_time_first(Z: np.ndarray, name: str) -> np.ndarray:
    """Auto-transpose if array looks like (dim, T) instead of (T, dim)."""
    if Z.ndim == 1:
        return Z.reshape(-1, 1)
    if Z.shape[0] < Z.shape[1]:
        print(f"  [warn] {name} shape {Z.shape} looks transposed — fixing.")
        return Z.T
    return Z


def _build_hankel(Z: np.ndarray, num_lags: int) -> np.ndarray:
    """Z: (T, dim) → Hankel (num_lags*dim, T-num_lags+1)"""
    T, dim = Z.shape
    T_h    = T - num_lags + 1
    H      = np.zeros((num_lags * dim, T_h))
    for i in range(num_lags):
        H[i*dim:(i+1)*dim, :] = Z[i : i + T_h, :].T
    return H


def _jitter(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Symmetrise and add scaled jitter for numerical stability."""
    M = 0.5 * (M + M.T)
    scale = np.mean(np.abs(np.diag(M))) + 1e-12
    return M + np.eye(len(M)) * eps * scale


def _check_dimensions(Y: np.ndarray, U: np.ndarray, lds: LDSParams,
                      context: str = "") -> None:
    """
    Raise a clear ValueError if any LDSParams matrix has the wrong shape.
    Called after n4sid and after each EM M-step to catch mismatches early
    rather than letting them propagate into the Kalman filter as cryptic
    matmul errors.
    """
    T, N = Y.shape
    M    = U.shape[1]
    d    = lds.A.shape[0]

    errors = []
    if lds.A.shape   != (d, d):  errors.append(f"A: {lds.A.shape}   != ({d},{d})")
    if lds.B.shape   != (d, M):  errors.append(f"B: {lds.B.shape}   != ({d},{M})")
    if lds.C.shape   != (N, d):  errors.append(f"C: {lds.C.shape}   != ({N},{d})")
    if lds.Q.shape   != (d, d):  errors.append(f"Q: {lds.Q.shape}   != ({d},{d})")
    if lds.R.shape   != (N, N):  errors.append(f"R: {lds.R.shape}   != ({N},{N})")
    if lds.mu_0.shape != (d,):   errors.append(f"mu_0: {lds.mu_0.shape} != ({d},)")
    if lds.P_0.shape  != (d, d): errors.append(f"P_0: {lds.P_0.shape}  != ({d},{d})")

    if errors:
        raise ValueError(
            f"Dimension mismatch in {context}:\n" + "\n".join(errors)
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. N4SID — ONE-SHOT SUBSPACE IDENTIFICATION
# ──────────────────────────────────────────────────────────────────────────────


def n4sid(
    Y:         np.ndarray,
    U:         np.ndarray,
    LatentDim: int,
    num_lags:  Optional[int] = None,
) -> tuple[np.ndarray, LDSParams]:
    """
    ARMORED N4SID: Designed to survive adversarial dynamic range traps.
    Uses massive Hankel windows and Tikhonov Regularized Regression.
    """
    Y = _ensure_time_first(Y, 'Y')
    U = _ensure_time_first(U, 'U')

    T, N = Y.shape
    M    = U.shape[1]
    d    = LatentDim

    # ---------------------------------------------------------
    # COUNTERMEASURE 1: Massive Hankel Window
    # ---------------------------------------------------------
    if num_lags is None:
        num_lags = max(10, T // 3)  # Look at 1/3 of the entire dataset at once

    # Ensure T_h is large enough
    min_lags = int(np.ceil((d + 1) / max(N, 1)))
    max_lags = T // 3
    num_lags = int(np.clip(num_lags, min_lags, max_lags))

    T_h = T - num_lags + 1

    H_y = _build_hankel(Y, num_lags)  
    H_u = _build_hankel(U, num_lags)  

    # Oblique projection
    Qu, _ = np.linalg.qr(H_u.T, mode='reduced')
    O_hat = H_y - (H_y @ Qu) @ Qu.T   

    # SVD
    Ug, Sg, _ = np.linalg.svd(O_hat, full_matrices=False)
    
    actual_d = min(d, Ug.shape[1])
    if actual_d < d:
        print(f"  [warn] SVD rank={actual_d} < Requested={d}. Truncating.")
        d = actual_d

    Gamma = Ug[:, :d] * np.sqrt(np.maximum(Sg[:d], 0))  
    
    A_sub, _, _, _ = la.lstsq(Gamma[:-N, :], Gamma[N:, :])
    A_sub = A_sub[:d, :d]    

    X_sub = (np.linalg.pinv(Gamma) @ O_hat).T  

    # ---------------------------------------------------------
    # COUNTERMEASURE 2: Tikhonov Regularization (Ridge Regression)
    # ---------------------------------------------------------
    # Prevents stiff dynamic ranges from causing the B matrix to explode
    n_reg = T_h - 1
    LHS   = np.hstack([X_sub[1:],   Y[:n_reg]])   
    RHS   = np.hstack([X_sub[:-1],  U[:n_reg]])   
    
    alpha = 1e-6  # Microscopic friction to stabilize the solver
    Theta = la.solve(RHS.T @ RHS + alpha * np.eye(RHS.shape[1]), RHS.T @ LHS)
    Theta = Theta.T                               

    A = Theta[:d, :d]    
    B = Theta[:d, d:]    
    C = Theta[d:, :d]    

    # Noise covariances
    X_pred    = (A @ X_sub[:-1].T + B @ U[:n_reg].T).T
    state_res = X_sub[1:] - X_pred
    obs_res   = Y[:T_h] - (C @ X_sub.T).T

    Q_raw = np.cov(state_res.T) if n_reg > d else np.eye(d) * 1e-4
    R_raw = np.cov(obs_res.T)   if T_h  > N else np.eye(N) * 1e-4
    
    Q_raw = np.atleast_2d(Q_raw)
    R_raw = np.atleast_2d(R_raw)
    
    Q = _jitter(Q_raw)
    R = _jitter(R_raw)

    X_hat = np.zeros((T, d))
    X_hat[:T_h] = X_sub
    for t in range(T_h, T):
        X_hat[t] = A @ X_hat[t-1] + B @ U[t-1]

    lds = LDSParams(A=A, B=B, C=C, Q=Q, R=R, mu_0=X_hat[0].copy(), P_0=np.eye(d))

    return X_hat, lds


def n4sidd(
    Y:         np.ndarray,
    U:         np.ndarray,
    LatentDim: int,
    num_lags:  Optional[int] = None,
) -> tuple[np.ndarray, LDSParams]:
    """
    N4SID-style subspace identification with known inputs.

    1. Build block-Hankel matrices of Y and U.
    2. Oblique projection removes input contribution from output Hankel
       — avoids the biased C that standard SVD produces.
    3. SVD of projected matrix → observability matrix Γ = [C; CA; CA²; ...]
    4. Extract C (first block row) and A (shift structure of Γ).
    5. Joint regression for A, B, C simultaneously given recovered X.
    6. Estimate Q, R from residuals.

    Parameters
    ----------
    Y        : (T, N)
    U        : (T, M)
    LatentDim: int
    num_lags : int  — Hankel window.  Rule of thumb: ≥ 2π/ω_min.
                      None = auto (max(2d, 10), capped at T//3).

    Returns
    -------
    X_hat : (T, d)   estimated state sequence
    lds   : LDSParams
    """
    Y = _ensure_time_first(Y, 'Y')
    U = _ensure_time_first(U, 'U')

    T, N = Y.shape
    M    = U.shape[1]
    d    = LatentDim

    # Auto num_lags
    if num_lags is None:
        num_lags = max(2 * d, 10)

    # Guard 1: ensure T_h is large enough for a well-conditioned regression.
    # We need T_h - 1 >= d + M + 10 rows in the joint regression RHS.
    min_lags = int(np.ceil((d + 1) / max(N, 1)))
    max_lags = T // 3
    num_lags = int(np.clip(num_lags, min_lags, max_lags))

    T_h = T - num_lags + 1
    if T_h < d + M + 10:
        num_lags = max(min_lags, int(T - (d + M + 10)))
        num_lags = max(num_lags, min_lags)
        T_h      = T - num_lags + 1
        print(f"  [warn] num_lags reduced to {num_lags} to maintain column count "
              f"(T_h={T_h})")

    # Build Hankel matrices
    H_y = _build_hankel(Y, num_lags)   # (num_lags*N, T_h)
    H_u = _build_hankel(U, num_lags)   # (num_lags*M, T_h)

    # Oblique projection via QR (more stable than explicit inverse)
    Qu, _ = np.linalg.qr(H_u.T, mode='reduced')
    O_hat  = H_y - (H_y @ Qu) @ Qu.T   # (num_lags*N, T_h)

    # SVD → observability matrix
    Ug, Sg, _ = np.linalg.svd(O_hat, full_matrices=False)

    # Guard 2: cap d to the actual numerical rank of the SVD so we never
    # request more dimensions than the data can support.
    numerical_rank = int((Sg > Sg[0] * 1e-10).sum())
    actual_d = min(d, Ug.shape[1], numerical_rank)
    if actual_d < d:
        print(f"  [warn] Requested LatentDim={d} but SVD rank={actual_d}. "
              f"Truncating to {actual_d}.")
        d = actual_d

    Gamma = Ug[:, :d] * np.sqrt(np.maximum(Sg[:d], 0))  # (num_lags*N, d)

    # Extract A from shift structure of Gamma
    A_sub, _, _, _ = la.lstsq(Gamma[:-N, :], Gamma[N:, :])
    A_sub = A_sub[:d, :d]    # force square — lstsq may return (d, d) already

    # Recover state sequence
    X_sub = (np.linalg.pinv(Gamma) @ O_hat).T   # (T_h, d)

    # Joint regression: [x_{t+1}; y_t] = [A B; C 0] [x_t; u_t]
    n_reg = T_h - 1
    LHS   = np.hstack([X_sub[1:],   Y[:n_reg]])   # (n_reg, d+N)
    RHS   = np.hstack([X_sub[:-1],  U[:n_reg]])   # (n_reg, d+M)
    Theta, _, _, _ = la.lstsq(RHS, LHS)
    Theta = Theta.T                                # (d+N, d+M)

    A = Theta[:d, :d]    # (d, d)
    B = Theta[:d, d:]    # (d, M)
    C = Theta[d:, :d]    # (N, d)  — N rows guaranteed by Y.shape

    # Noise covariances from residuals
    X_pred    = (A @ X_sub[:-1].T + B @ U[:n_reg].T).T
    state_res = X_sub[1:] - X_pred
    obs_res   = Y[:T_h] - (C @ X_sub.T).T

    # Guard 3: np.cov returns a scalar when given a 1-row matrix (d=1 or N=1).
    # atleast_2d + shape check ensures Q and R are always 2-D square matrices.
    Q_raw = np.cov(state_res.T) if n_reg > d else np.eye(d) * 0.01
    R_raw = np.cov(obs_res.T)   if T_h  > N else np.eye(N) * 0.01
    Q_raw = np.atleast_2d(Q_raw)
    R_raw = np.atleast_2d(R_raw)
    if Q_raw.shape != (d, d):
        Q_raw = np.eye(d) * float(np.mean(np.diag(Q_raw)))
    if R_raw.shape != (N, N):
        R_raw = np.eye(N) * float(np.mean(np.diag(R_raw)))
    Q = _jitter(Q_raw)
    R = _jitter(R_raw)

    # Forward-simulate to fill tail (t >= T_h)
    X_hat = np.zeros((T, d))
    X_hat[:T_h] = X_sub
    for t in range(T_h, T):
        X_hat[t] = A @ X_hat[t-1] + B @ U[t-1]

    lds = LDSParams(
        A=A, B=B, C=C, Q=Q, R=R,
        mu_0=X_hat[0].copy(),
        P_0=np.eye(d),
    )

    # Final sanity check — raises with a clear message if anything is still off
    _check_dimensions(Y, U, lds, context="n4sid output")
    return X_hat, lds


# ──────────────────────────────────────────────────────────────────────────────
# 4. DIMENSION SELECTION — BIC ON SINGULAR VALUES
# ──────────────────────────────────────────────────────────────────────────────

def select_latent_dim(
    Y:         np.ndarray,
    U:         np.ndarray,
    num_lags:  int,
    max_dim:   int = 20,
    criterion: str = 'bic',
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Select LatentDim from singular values of the oblique-projected Hankel matrix.

    Returns best_d, ic_scores (per dimension), singular_vals.
    """
    Y = _ensure_time_first(Y, 'Y')
    U = _ensure_time_first(U, 'U')

    T, N = Y.shape
    M    = U.shape[1]
    T_h  = T - num_lags + 1

    H_y  = _build_hankel(Y, num_lags)
    H_u  = _build_hankel(U, num_lags)
    Qu, _ = np.linalg.qr(H_u.T, mode='reduced')
    O_hat = H_y - (H_y @ Qu) @ Qu.T

    _, S, _ = np.linalg.svd(O_hat, full_matrices=False)
    S        = S[:max_dim]
    total_var = (S**2).sum()

    scores = []
    for d in range(1, len(S) + 1):
        var_res  = max(1.0 - (S[:d]**2).sum() / total_var, 1e-12)
        n_params = 2*d**2 + d*M + N*d + N**2   # A, Q, B, C, R (rough)
        nll      = T_h * np.log(var_res)

        if criterion == 'aic':
            score = nll + 2 * n_params
        elif criterion == 'bic':
            score = nll + n_params * np.log(T_h)
        else:
            score = nll + 0.5 * n_params * np.log(T_h)

        scores.append(score)

    ic_scores = np.array(scores)
    best_d    = int(np.argmin(ic_scores)) + 1
    return best_d, ic_scores, S


# ──────────────────────────────────────────────────────────────────────────────
# 5. LAG SELECTION — CROSS-VALIDATED PREDICTION ERROR
# ──────────────────────────────────────────────────────────────────────────────

def select_num_lags(
    Y:          np.ndarray,
    U:          np.ndarray,
    LatentDim:  int,
    candidates: Optional[list] = None,
) -> int:
    """
    Select num_lags by held-out one-step-ahead prediction error.
    Uses first 80% as train, last 20% as validation.
    """
    Y = _ensure_time_first(Y, 'Y')
    U = _ensure_time_first(U, 'U')

    T     = Y.shape[0]
    split = int(T * 0.8)
    Ytr, Utr = Y[:split], U[:split]
    Yva, Uva = Y[split:], U[split:]

    if candidates is None:
        max_l    = min(split // 3, 60)
        min_l    = max(2 * LatentDim, 5)
        candidates = sorted(set(
            np.geomspace(min_l, max_l, 12).astype(int).tolist()
        ))

    best_lags, best_err = candidates[0], np.inf

    for lags in candidates:
        try:
            _, lds = n4sid(Ytr, Utr, LatentDim, num_lags=lags)
            x   = lds.mu_0.copy()
            err = 0.0
            for t in range(len(Yva) - 1):
                err += float(np.sum((Yva[t] - lds.C @ x) ** 2))
                x    = lds.A @ x + lds.B @ Uva[t]
            err /= len(Yva)
            if err < best_err:
                best_err  = err
                best_lags = lags
        except Exception:
            continue

    return best_lags


# ──────────────────────────────────────────────────────────────────────────────
# 6. KALMAN SMOOTHER WITH KNOWN INPUTS
# ──────────────────────────────────────────────────────────────────────────────

def _kalman_smoother(
    Y:   np.ndarray,   # (T, N)
    U:   np.ndarray,   # (T, M)
    lds: LDSParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    RTS Kalman smoother with known exogenous input.

    Returns
    -------
    mu_s    : (T, d)      smoothed state means
    P_s     : (T, d, d)   smoothed state covariances
    P_cross : (T-1, d, d) cross covariances  Cov[x_{t+1}, x_t | Y, U]
    mu_p    : (T, d)      predicted means (needed for M-step)
    P_p     : (T, d, d)   predicted covariances
    ll      : float       total log-likelihood
    """
    A, B, C, Q, R = lds.A, lds.B, lds.C, lds.Q, lds.R
    T, N_data = Y.shape
    d  = A.shape[0]
    N  = C.shape[0]   # authoritative N — use C, not Y, to avoid mismatch

    # Pre-flight: catch any shape inconsistency before entering the loop
    # where matmul errors are hard to diagnose
    if N != N_data:
        raise ValueError(
            f"_kalman_smoother: Y has {N_data} observation dims but "
            f"lds.C has {N} rows. Re-run n4sid — matrices may be stale."
        )
    if R.shape != (N, N):
        raise ValueError(
            f"_kalman_smoother: R shape {R.shape} inconsistent with "
            f"C rows N={N}. Re-run n4sid."
        )
    if Q.shape != (d, d):
        raise ValueError(
            f"_kalman_smoother: Q shape {Q.shape} inconsistent with "
            f"A dim d={d}."
        )
    if lds.mu_0.shape != (d,):
        raise ValueError(
            f"_kalman_smoother: mu_0 shape {lds.mu_0.shape} != ({d},)."
        )
    if lds.P_0.shape != (d, d):
        raise ValueError(
            f"_kalman_smoother: P_0 shape {lds.P_0.shape} != ({d},{d})."
        )

    mu_f = np.zeros((T, d))
    P_f  = np.zeros((T, d, d))
    mu_p = np.zeros((T, d))
    P_p  = np.zeros((T, d, d))
    ll   = 0.0

    mu_p[0] = lds.mu_0
    P_p[0]  = lds.P_0

    # Forward pass
    for t in range(T):
        innov  = Y[t] - C @ mu_p[t]
        S_inn  = C @ P_p[t] @ C.T + R
        S_inn  = _jitter(S_inn, eps=1e-8)

        # Log-likelihood
        sign, logdet = np.linalg.slogdet(S_inn)
        ll += -0.5 * (N * np.log(2 * np.pi) + logdet
                      + innov @ la.solve(S_inn, innov, assume_a='pos'))

        # Kalman gain — Joseph form for numerical stability
        K      = la.solve(S_inn.T, C @ P_p[t].T, assume_a='pos').T
        ImKC   = np.eye(d) - K @ C
        mu_f[t] = mu_p[t] + K @ innov
        P_f[t]  = _jitter(ImKC @ P_p[t] @ ImKC.T + K @ R @ K.T)

        if t < T - 1:
            mu_p[t+1] = A @ mu_f[t] + B @ U[t]    # known U
            P_p[t+1]  = _jitter(A @ P_f[t] @ A.T + Q)

    # Backward pass (RTS)
    mu_s    = mu_f.copy()
    P_s     = P_f.copy()
    P_cross = np.zeros((T - 1, d, d))

    for t in range(T - 2, -1, -1):
        G        = P_f[t] @ A.T @ la.solve(P_p[t+1].T, np.eye(d)).T
        mu_s[t]  = mu_f[t] + G @ (mu_s[t+1] - mu_p[t+1])
        P_s[t]   = _jitter(P_f[t] + G @ (P_s[t+1] - P_p[t+1]) @ G.T)
        P_cross[t] = P_s[t+1] @ G.T

    return mu_s, P_s, P_cross, mu_p, P_p, ll


# ──────────────────────────────────────────────────────────────────────────────
# 7. EM M-STEP
# ──────────────────────────────────────────────────────────────────────────────

def _em_mstep(
    Y:       np.ndarray,   # (T, N)
    U:       np.ndarray,   # (T, M)
    mu_s:    np.ndarray,   # (T, d)
    P_s:     np.ndarray,   # (T, d, d)
    P_cross: np.ndarray,   # (T-1, d, d)
) -> LDSParams:
    """
    Closed-form M-step: update A, B, C, Q, R, mu_0, P_0
    from the sufficient statistics produced by the smoother.
    """
    T, N = Y.shape
    d    = mu_s.shape[1]
    M    = U.shape[1]

    # Accumulate sufficient statistics over t = 0 .. T-2
    S00 = np.zeros((d, d))
    S11 = np.zeros((d, d))
    S10 = np.zeros((d, d))
    Su0 = np.zeros((M, d))
    Suu = np.zeros((M, M))
    Su1 = np.zeros((M, d))

    for t in range(T - 1):
        ex0  = mu_s[t]
        ex1  = mu_s[t+1]
        S00 += P_s[t]   + np.outer(ex0, ex0)
        S11 += P_s[t+1] + np.outer(ex1, ex1)
        S10 += P_cross[t] + np.outer(ex1, ex0)
        Su0 += np.outer(U[t], ex0)
        Suu += np.outer(U[t], U[t])
        Su1 += np.outer(U[t], ex1)

    # Update A and B jointly via block normal equations
    # [A B] @ [[S00, Su0.T], [Su0, Suu]] = [S10, Su1.T]
    LHS_AB = np.block([[S00,  Su0.T],
                       [Su0,  Suu  ]])               # (d+M, d+M)
    RHS_AB = np.hstack([S10.T, Su1.T])               # (d, d+M)
    try:
        AB = la.solve(LHS_AB.T, RHS_AB.T, assume_a='sym').T
    except la.LinAlgError:
        AB = np.linalg.lstsq(LHS_AB.T, RHS_AB.T, rcond=None)[0].T
    A_new = AB[:, :d]
    B_new = AB[:, d:]

    # Update Q — full expression including B cross-terms
# Update Q — full expression including B cross-terms
    Q_new = (S11
             - A_new @ S10.T   - B_new @ Su1        # <--- Fixed Su1
             - S10 @ A_new.T   - Su1.T @ B_new.T    # <--- Fixed Su1.T
             + A_new @ S00 @ A_new.T
             + A_new @ Su0.T @ B_new.T
             + B_new @ Su0 @ A_new.T
             + B_new @ Suu @ B_new.T) / (T - 1)
    Q_new = _jitter(Q_new)

    # Update C — use lstsq instead of inv to handle near-singular Sxx
    # C_new = Syx @ Sxx^{-1}  ⟺  Sxx @ C_new.T = Syx.T
    Syx = sum(np.outer(Y[t], mu_s[t])              for t in range(T))
    Sxx = sum(P_s[t] + np.outer(mu_s[t], mu_s[t]) for t in range(T))
    Sxx = _jitter(Sxx)   # ensure positive definite before solve
    C_new, _, _, _ = la.lstsq(Sxx, Syx.T)
    C_new = C_new.T      # (N, d)

    # Update R
    R_acc = np.zeros((N, N))
    for t in range(T):
        ex  = mu_s[t]
        exx = P_s[t] + np.outer(ex, ex)
        R_acc += (np.outer(Y[t], Y[t])
                  - C_new @ np.outer(ex,  Y[t])
                  - np.outer(Y[t], ex) @ C_new.T
                  + C_new @ exx @ C_new.T)
    R_new = _jitter(R_acc / T)

    # NaN/Inf guard — if any matrix contains non-finite values the EM has
    # diverged (usually due to ill-conditioned Sxx or Q).  Raise immediately
    # with a clear message rather than letting NaN propagate silently.
    for name, mat in [('A', A_new), ('B', B_new), ('C', C_new),
                      ('Q', Q_new), ('R', R_new)]:
        if not np.all(np.isfinite(mat)):
            raise ValueError(
                f"_em_mstep: {name} contains NaN/Inf after M-step update. "
                f"System may be unidentifiable at d={d} — try a smaller LatentDim."
            )

    return LDSParams(
        A=A_new, B=B_new, C=C_new, Q=Q_new, R=R_new,
        mu_0=mu_s[0],
        P_0=P_s[0],
    )


# ──────────────────────────────────────────────────────────────────────────────
# 8. FULL N4SID + EM REFINEMENT
# ──────────────────────────────────────────────────────────────────────────────

def identify_and_refine(
    Y:         np.ndarray,
    U:         np.ndarray,
    LatentDim: int,
    num_lags:  Optional[int] = None,
    max_iter:  int   = 30,
    tol:       float = 1e-4,
    verbose:   bool  = True,
    initial_lds: Optional[LDSParams] = None, # <--- THE UPGRADE
) -> tuple[np.ndarray, LDSParams, np.ndarray]:
    """
    N4SID initialisation (or Warm-Start) followed by EM with RTS Kalman smoother.
    """
    Y = _ensure_time_first(Y, 'Y')
    U = _ensure_time_first(U, 'U')

    # Phase 1: Initialization
    if initial_lds is not None:
        if verbose:
            print("  [1/2] Skipping N4SID. Using provided Warm-Start LDS...")
        lds = initial_lds
        actual_d = lds.A.shape[0]
    else:
        if verbose:
            print("  [1/2] N4SID initialisation...")
        _, lds = n4sid(Y, U, LatentDim, num_lags=num_lags)
        actual_d = lds.A.shape[0]

    if verbose and actual_d != LatentDim and initial_lds is None:
        print(f"  [info] LatentDim requested={LatentDim}, actual={actual_d} "
              f"(SVD rank limited)")
    if verbose:
        print(f"  [info] Shapes: A={lds.A.shape}, B={lds.B.shape}, "
              f"C={lds.C.shape}, Q={lds.Q.shape}, R={lds.R.shape}, "
              f"Y={Y.shape}, U={U.shape}")

    # Phase 2: EM refinement
    if verbose:
        print("  [2/2] EM refinement...")
    ll_history = []
    ll_prev    = -np.inf
    last_good_lds = lds   

    for iteration in range(max_iter):
        # E-step
        mu_s, P_s, P_cross, _, _, ll = _kalman_smoother(Y, U, lds)
        ll_history.append(ll)

        # M-step
        try:
            lds = _em_mstep(Y, U, mu_s, P_s, P_cross)
        except ValueError as e:
            if verbose:
                print(f"  [warn] EM diverged at iteration {iteration+1}: {e}")
                print(f"  [warn] Returning last good parameters.")
            lds = last_good_lds
            break

        _check_dimensions(Y, U, lds, context=f"EM iteration {iteration+1}")
        last_good_lds = lds

        delta = ll - ll_prev
        if verbose:
            print(f"    iter {iteration+1:3d}  ll={ll:.3f}  Δll={delta:+.3e}")

        if iteration > 0 and abs(delta) < tol:
            if verbose:
                print(f"  Converged at iteration {iteration+1}.")
            break
        ll_prev = ll
    else:
        if verbose:
            print(f"  Reached max_iter={max_iter}.")

    return mu_s, lds, np.array(ll_history)

# ──────────────────────────────────────────────────────────────────────────────
# 9. VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_vaf(
    Y_val: np.ndarray,
    U_val: np.ndarray,
    lds:   LDSParams,
) -> float:
    """
    True one-step-ahead VAF on held-out data using a forward Kalman Filter.
    This prevents accumulating open-loop drift from destroying the metric.
    """
    Y_val = _ensure_time_first(Y_val, 'Y_val')
    U_val = _ensure_time_first(U_val, 'U_val')

    T, N = Y_val.shape
    d = lds.A.shape[0]
    
    x_pred = lds.mu_0.copy()
    P_pred = lds.P_0.copy()
    
    Y_pred = np.zeros_like(Y_val)

    for t in range(T):
        # 1. Record the prediction for the current step BEFORE seeing Y_val
        Y_pred[t] = lds.C @ x_pred
        
        # 2. Compute the Kalman Gain using the actual measurement
        innov = Y_val[t] - Y_pred[t]
        S = lds.C @ P_pred @ lds.C.T + lds.R
        
        try:
            K = P_pred @ lds.C.T @ np.linalg.pinv(S)
        except np.linalg.LinAlgError:
            K = np.zeros((d, N)) # Fallback if perfectly singular
            
        # 3. Update the state (Correcting the drift)
        x_filt = x_pred + K @ innov
        P_filt = (np.eye(d) - K @ lds.C) @ P_pred
        
        # 4. Predict the NEXT step
        x_pred = lds.A @ x_filt + lds.B @ U_val[t]
        P_pred = lds.A @ P_filt @ lds.A.T + lds.Q

    # Calculate VAF
    residual = Y_val - Y_pred
    var_res = np.var(residual)
    var_y = np.var(Y_val)
    
    vaf = 100.0 * (1.0 - var_res / (var_y + 1e-12))
    return float(vaf)
# ──────────────────────────────────────────────────────────────────────────────
# 10. FULL PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    brain,
    n_samples:     int   = 2000,
    max_latent_dim: int  = 15,
    max_iter:      int   = 30,
    tol:           float = 1e-4,
    criterion:     str   = 'bic',
    verbose:       bool  = True,
    force_dim:     Optional[int] = None,  # <-- Add this
) -> PipelineResult:
    """
    End-to-end system identification pipeline.

    Steps
    -----
    1. Collect data with Schroeder multisine (persistent excitation)
    2. Select LatentDim via BIC on projected Hankel singular values
    3. Select num_lags via cross-validated prediction error
    4. N4SID + EM to fit LDS matrices
    5. Validate with one-step-ahead VAF on held-out data

    Parameters
    ----------
    brain          : Brain interface with .input_dim, .measure(), .next_state()
    n_samples      : total samples to collect (train + val)
    max_latent_dim : upper bound for BIC dimension search
    max_iter       : EM iterations
    tol            : EM convergence threshold
    criterion      : 'bic', 'aic', or 'mdl' for dimension selection
    verbose        : print progress

    Returns
    -------
    PipelineResult with all matrices, states, and diagnostics
    """
    # ── Step 1: Data collection ───────────────────────────────────────────────
    if verbose:
        print("=" * 60)
        print("Step 1: Data collection")
    Y_train, U_train, Y_val, U_val = collect_hybrid_data(brain, n_samples=n_samples,impulse_spacing=100)

    # ── Step 2: Dimension selection ───────────────────────────────────────────
    if verbose:
        print("\nStep 2: Selecting latent dimension")
    rough_lags = min(max(10, 2 * max_latent_dim), len(Y_train) // 3)
    best_d, ic_scores, singular_vals = select_latent_dim(
        Y_train, U_train,
        num_lags=rough_lags,
        max_dim=max_latent_dim,
        criterion=criterion,
    )

    if force_dim is not None:
        best_d = force_dim
        if verbose:
            print(f"  Forced LatentDim = {best_d} (User overridden)")
    else:
        # Keep your existing identifiability cap here
        T_train = len(Y_train)
        N_obs   = Y_train.shape[1]
        M_inp   = U_train.shape[1]
        while best_d > 1:
            n_params = 2*best_d**2 + best_d*M_inp + N_obs*best_d + N_obs**2
            if T_train >= 10 * n_params:
                break
            best_d -= 1

        if verbose:
            print(f"  Selected LatentDim = {best_d}  ({criterion.upper()}, "
                  f"identifiability-capped)")

    # ── Step 3: Lag selection ─────────────────────────────────────────────────
    if verbose:
        print("\nStep 3: Selecting num_lags")
    best_lags = select_num_lags(Y_train, U_train, LatentDim=best_d)
    if verbose:
        print(f"  Selected num_lags  = {best_lags}")

    # ── Step 4: Identification ────────────────────────────────────────────────
    if verbose:
        print("\nStep 4: N4SID + EM identification")
    X_smooth, lds, ll_history = identify_and_refine(
        Y_train, U_train,
        LatentDim=best_d,
        num_lags=best_lags,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )

    # ── Step 5: Validation ────────────────────────────────────────────────────
    if verbose:
        print("\nStep 5: Validation")
    vaf = compute_vaf(Y_val, U_val, lds)
    if verbose:
        print(f"  One-step-ahead VAF = {vaf:.1f}%")
        print("\n" + str(lds))
        print("=" * 60)

    return PipelineResult(
        lds=lds,
        X_smooth=X_smooth,
        Y_train=Y_train,
        U_train=U_train,
        Y_val=Y_val,
        U_val=U_val,
        vaf=vaf,
        latent_dim=best_d,
        num_lags=best_lags,
        ll_history=ll_history,
        singular_vals=singular_vals,
        ic_scores=ic_scores,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 11. DIAGNOSTICS
# ──────────────────────────────────────────────────────────────────────────────

def plot_diagnostics(result: PipelineResult):
    """Four-panel diagnostic plot for a PipelineResult."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"System ID diagnostics  —  d={result.latent_dim}, "
        f"lags={result.num_lags}, VAF={result.vaf:.1f}%",
        fontsize=12,
    )

    # Panel A: singular values with selected dimension marked
    ax = axes[0, 0]
    ax.plot(result.singular_vals, 'o-', markersize=5)
    ax.axvline(result.latent_dim - 1, color='red', linestyle='--',
               label=f'd={result.latent_dim}')
    ax.set_title("Singular values of projected Hankel")
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular value")
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)

    # Panel B: BIC / AIC scores
    ax = axes[0, 1]
    ax.plot(range(1, len(result.ic_scores) + 1), result.ic_scores, 'o-', markersize=5)
    ax.axvline(result.latent_dim, color='red', linestyle='--',
               label=f'selected d={result.latent_dim}')
    ax.set_title("Information criterion scores")
    ax.set_xlabel("Candidate LatentDim")
    ax.set_ylabel("Score (lower = better)")
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)

    # Panel C: EM log-likelihood convergence
    ax = axes[1, 0]
    ax.plot(result.ll_history, 'o-', markersize=5)
    ax.set_title("EM log-likelihood")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log-likelihood")
    ax.spines[['top', 'right']].set_visible(False)

    # Panel D: eigenvalues of A in complex plane
    ax = axes[1, 1]
    eigs = np.linalg.eigvals(result.lds.A)
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=0.8, alpha=0.4)
    ax.scatter(eigs.real, eigs.imag, zorder=5, s=60, color='steelblue')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_title("Eigenvalues of A (unit circle = stability boundary)")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# 12. ORTHOGONAL HIDDEN STATE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import scipy.linalg as la

def collect_targeted_data(
    brain,
    u_base_1d: np.ndarray,      # 1D array of your targeted signal (centered at 0)
    quiet_vector: np.ndarray,   # The perfectly scaled 1D array from SVD
    num_trials: int = 5,        # Number of passes for ensemble averaging
    n_settle: int = 200,
    val_fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collects data by projecting a 1D targeted signal along a specific spatial vector,
    and averages multiple trials to crush Gaussian measurement noise.
    """
    T = len(u_base_1d)
    M_dim = brain.input_dim
    dummy_y = np.array(brain.measure())
    Y_dim = len(dummy_y)

    # 1. Project the 1D signal along the 2D quiet vector to map to physical channels
    U_input = np.outer(u_base_1d, quiet_vector)
    
    # 2. Shift to baseline 0.5 and hard-cap to prevent physical clipping
    U_input = np.clip(U_input + 0.5, 0.0, 1.0)

    # 3. Settle the system at the mean baseline
    if n_settle > 0:
        print(f"  Settling ({n_settle} steps) at baseline 0.5...")
        for _ in range(n_settle):
            brain.next_state(np.zeros(M_dim) + 0.5)

    # 4. Collect Ensemble Averaged Data
    Y_all_trials = np.zeros((num_trials, T, Y_dim))
    
    print(f"  Injecting orthogonal signal ({num_trials} ensemble trials)...")
    for trial in range(num_trials):
        for k in range(T):
            Y_all_trials[trial, k, :] = np.array(brain.measure())
            brain.next_state(U_input[k, :])

    # 5. Average across trials to mathematically destroy zero-mean Gaussian noise
    Y_averaged = np.mean(Y_all_trials, axis=0)

    # 6. Train / Validation Split on the cleaned data
    split   = int(T * (1 - val_fraction))
    Y_train = Y_averaged[:split]
    U_train = U_input[:split]
    Y_val   = Y_averaged[split:]
    U_val   = U_input[split:]

    return Y_train, U_train, Y_val, U_val


def run_orthogonal_pipeline(
    brain,
    n_dominant: int = 2,
    n_hidden: int = 3,
    n_samples_per_trial: int = 500, # Length of a single pass
    num_trials: int = 5,            # Number of identical passes to average
    max_iter: int = 30,
    verbose: bool = True
) -> tuple[PipelineResult, PipelineResult]:
    """
    Executes the two-stage spatial nulling pipeline to extract buried states.

    Stage 1: Full-spectrum hybrid excitation to capture dominant dynamics.
    Stage 2: Orthogonal,scaled, averaged excitation to isolate hidden dynamics.
    """
    total_samples = n_samples_per_trial * num_trials
    
    # ── STAGE 1: Extract Dominant States ──────────────────────────────────────
    if verbose:
        print("\n" + "═" * 60)
        print(f"STAGE 1: EXTRACTING {n_dominant} DOMINANT STATES")
        print("═" * 60)

    res_dom = run_full_pipeline(
        brain,
        n_samples=total_samples,
        max_iter=max_iter,
        verbose=verbose,
        force_dim=n_dominant
    )

    # ── PIVOT: Calculate Spatial Null Space ───────────────────────────────────
    if verbose:
        print("\n" + "═" * 60)
        print("PIVOT: CALCULATING SPATIAL NULL SPACE")
        print("═" * 60)

    # SVD of the dominant B matrix
    B_dom = res_dom.lds.B
    U_svd, S_svd, Vh_svd = la.svd(B_dom, full_matrices=False)
    
    # The smallest right singular vector corresponds to the raw "quiet" direction
    v_quiet_raw = Vh_svd[-1, :]

    # THE FIX: Scale to L_infinity norm = 1.0 to maximize physical actuator stroke
    v_quiet_max = v_quiet_raw / np.max(np.abs(v_quiet_raw))

    if verbose:
        print(f"  Dominant B matrix singular values: {np.round(S_svd, 4)}")
        print(f"  Raw Orthogonal Vector (L2=1): {np.round(v_quiet_raw, 4)}")
        print(f"  Max-Scaled Orthogonal Vector (Linf=1): {np.round(v_quiet_max, 4)}")

    # Generate custom 1D targeted signal (1D wave bounded by [-0.45, 0.45])
    # Note: If you have implemented `generate_targeted_multisine`, replace this line!
    signal_1d = np.sign(np.random.randn(n_samples_per_trial)) * 0.45

    # ── STAGE 2: Extract Hidden States ────────────────────────────────────────
    if verbose:
        print("\n" + "═" * 60)
        print(f"STAGE 2: EXTRACTING {n_hidden} HIDDEN STATES")
        print("═" * 60)

    Ytr, Utr, Yva, Uva = collect_targeted_data(
        brain, 
        u_base_1d=signal_1d, 
        quiet_vector=v_quiet_max, 
        num_trials=num_trials,
        val_fraction=0.2
    )

    if verbose:
        print("\nStep 3: Selecting num_lags for hidden model")
    best_lags = select_num_lags(Ytr, Utr, LatentDim=n_hidden)
    if verbose:
        print(f"  Selected num_lags = {best_lags}")

    if verbose:
        print("\nStep 4: N4SID + Iterative refinement on nulled data")
    
    X_smooth, lds_hid, ll_history = identify_and_refine(
        Ytr, Utr,
        LatentDim=n_hidden,
        num_lags=best_lags,
        max_iter=max_iter,
        verbose=verbose
    )

    if verbose:
        print("\nStep 5: Validation")
        
    vaf = compute_vaf(Yva, Uva, lds_hid)

    if verbose:
        print(f"  Hidden States VAF = {vaf:.1f}%")
        print("\n" + str(lds_hid))
        print("═" * 60)

    # Package the result for the hidden states
    res_hid = PipelineResult(
        lds=lds_hid,
        X_smooth=X_smooth,
        Y_train=Ytr,
        U_train=Utr,
        Y_val=Yva,
        U_val=Uva,
        vaf=vaf,
        latent_dim=n_hidden,
        num_lags=best_lags,
        ll_history=ll_history,
        singular_vals=np.zeros(1), 
        ic_scores=np.zeros(1)
    )

    return res_dom, res_hid

import scipy.linalg as la

def estimate_full_system(
    res_dom: PipelineResult, 
    res_hid: PipelineResult, 
) -> tuple[LDSParams, float]:
    """
    Fuses the dominant and hidden models into a single unified realization.
    Uses Grey-Box optimization to solve the underdetermined input mapping (B matrix),
    and validates the full system using the forward Kalman filter.
    """
    dom = res_dom.lds
    hid = res_hid.lds
    
    # Extract the full-spectrum broadband data from Stage 1
    # This is crucial because it contains energy in all input directions
    Y_train = res_dom.Y_train
    U_train = res_dom.U_train
    Y_val   = res_dom.Y_val
    U_val   = res_dom.U_val

    # 1. Freeze the A matrix (The structural "bones" of the system)
    A_frozen = la.block_diag(dom.A, hid.A)
    
    # 2. Stack initial guesses for B and C
    B_init = np.vstack((dom.B, hid.B))
    C_init = np.hstack((dom.C, hid.C))
    
    Nx = A_frozen.shape[0]
    Nu = B_init.shape[1]
    Ny = C_init.shape[0]
    
    # Flatten B and C into a single 1D vector for the scipy optimizer
    theta_0 = np.concatenate((B_init.flatten(), C_init.flatten()))
    B_size  = Nx * Nu

    print("\n" + "═" * 60)
    print(f"STAGE 3: FULL SYSTEM ESTIMATION ({Nx} STATES)")
    print("═" * 60)
    print("  Running Grey-Box optimization to align input matrices...")

    # 3. Objective function: Open-loop simulation Mean Squared Error (MSE)
    def cost_fn(theta):
        B_opt = theta[:B_size].reshape((Nx, Nu))
        C_opt = theta[B_size:].reshape((Ny, Nx))
        
        # Fast vectorized open-loop simulation using scipy's dlsim
        sys = StateSpace(A_frozen, B_opt, C_opt, np.zeros((Ny, Nu)), dt=1.0)
        _, Y_hat, _ = dlsim(sys, U_train)
        
        return np.mean((Y_train - Y_hat)**2)

    # 4. Run the L-BFGS-B optimizer
    opt_res = minimize(cost_fn, theta_0, method='L-BFGS-B', options={'maxiter': 300, 'disp': False})
    
    if opt_res.success:
        print("  Optimization converged successfully.")
    else:
        print("  [warn] Optimization hit iteration limit (results may still be improved).")

    # 5. Extract the final optimized matrices
    B_final = opt_res.x[:B_size].reshape((Nx, Nu))
    C_final = opt_res.x[B_size:].reshape((Ny, Nx))
    
    # 6. Build the final noise covariances and initial states
    Q_final = la.block_diag(dom.Q, hid.Q)
    R_final = (dom.R + hid.R) / 2.0  # Average the sensor noise estimates
    mu_0_final = np.concatenate((dom.mu_0, hid.mu_0))
    P_0_final = la.block_diag(dom.P_0, hid.P_0)

    # Package into your standard LDSParams object
    lds_final = LDSParams(
        A=A_frozen, B=B_final, C=C_final, 
        Q=Q_final, R=R_final, 
        mu_0=mu_0_final, P_0=P_0_final
    )

    # 7. Validate using your Kalman-filter validation function
    print("  Evaluating full system using Kalman Filter...")
    vaf_final = compute_vaf(Y_val, U_val, lds_final)
    
    print(f"  Final Full System VAF: {vaf_final:.1f}%")
    print("═" * 60)

    return lds_final, vaf_final

def print_system_matrices(lds: LDSParams, precision: int = 4, suppress: bool = True):
    """
    Temporarily overrides numpy's print settings to display 
    the LDS matrices in a clean, readable format.
    """
    import numpy as np
    
    # Save current terminal print options so we don't permanently alter them
    original_opts = np.get_printoptions()
    
    # Set clean formatting: fixed precision, no scientific notation for near-zero values
    np.set_printoptions(precision=precision, suppress=suppress, linewidth=150)
    
    print("\n" + "═" * 60)
    print(f"SYSTEM MATRICES (States: {lds.A.shape[0]} | Inputs: {lds.B.shape[1]} | Sensors: {lds.C.shape[0]})")
    print("═" * 60)
    
    print("\nState Transition Matrix (A):")
    print(lds.A)
    
    print("\nInput Matrix (B):")
    print(lds.B)
    
    print("\nObservation Matrix (C):")
    print(lds.C)
    
    print("\nProcess Noise Covariance (Q):")
    print(lds.Q)
    
    print("\nObservation Noise Covariance (R):")
    print(lds.R)
    
    print("\nInitial State Mean (mu_0):")
    print(lds.mu_0)
    
    print("\n" + "═" * 60)
    
    # Restore the original terminal print options
    np.set_printoptions(**original_opts)

import numpy as np
import scipy.linalg as la

import time

def extract_full_orthogonal_model(
    brain,
    n_dominant: int = 2,
    n_hidden: int = 4,
    n_samples_per_trial: int = 400,
    num_trials: int = 5,
    max_iter_em: int = 30,
    verbose: bool = True
) -> tuple[LDSParams, float]:
    """
    Master Pipeline Function: Executes the complete Orthogonal System ID architecture.
    
    Stage 1: Extracts dominant states via broadband excitation.
    Stage 2: Extracts hidden states via L_inf scaled orthogonal ensemble excitation.
    Stage 3: Bridges the input matrices via Grey-Box numerical optimization.
    Stage 4: Discovers cross-talk via Warm-Start Global Expectation-Maximization.
    
    Returns
    -------
    lds_ultimate : LDSParams  (The final, optimized 6-state mathematical model)
    final_vaf    : float      (The validation VAF percentage)
    """
    start_time = time.time()
    total_states = n_dominant + n_hidden
    
    if verbose:
        print("\n" + "█" * 60)
        print(f"INITIATING MASTER ORTHOGONAL PIPELINE ({total_states} STATES)")
        print("█" * 60)

    # 1. Orthogonal Extraction (Handles Stage 1 and Stage 2 natively)
    res_dominant, res_hidden = run_orthogonal_pipeline(
        brain, 
        n_dominant=n_dominant, 
        n_hidden=n_hidden, 
        n_samples_per_trial=n_samples_per_trial,  
        num_trials=num_trials,             
        max_iter=max_iter_em,
        verbose=verbose
    )
    
    # 2. Grey-Box Bridge (Stage 3)
    # Fixes the B and C matrices against the broadband data without breaking the A matrix
    lds_greybox, vaf_greybox = estimate_full_system(res_dominant, res_hidden)
    
    # 3. Warm-Start Iterative Refinement (Stage 4)
    if verbose:
        print("\n" + "═" * 60)
        print("STAGE 4: GLOBAL ITERATIVE REFINEMENT (WARM-START)")
        print("═" * 60)
        print(f"  Pre-Refinement (Grey-Box) VAF: {vaf_greybox:.1f}%\n")
    
    # Feed the Grey-Box model back into the EM algorithm using the broadband data
    _, lds_ultimate, _ = identify_and_refine(
        res_dominant.Y_train, 
        res_dominant.U_train,
        LatentDim=total_states,
        num_lags=res_dominant.num_lags, 
        max_iter=max_iter_em,                     
        verbose=verbose,
        initial_lds=lds_greybox          
    )
    
    # 4. Final Validation Check
    final_vaf = compute_vaf(res_dominant.Y_val, res_dominant.U_val, lds_ultimate)
    
    if verbose:
        print("\n" + "█" * 60)
        print("MASTER PIPELINE COMPLETE")
        print("█" * 60)
        print(f"  Target Architecture  : {total_states} States (Dom: {n_dominant}, Hid: {n_hidden})")
        print(f"  Pre-EM Grey-Box VAF  : {vaf_greybox:.1f}%")
        print(f"  Ultimate System VAF  : {final_vaf:.1f}%")
        print(f"  Total Compute Time   : {time.time() - start_time:.1f} seconds")
        print("█" * 60 + "\n")
        
    return lds_ultimate, final_vaf