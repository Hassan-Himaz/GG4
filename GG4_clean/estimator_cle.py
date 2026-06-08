"""
SSI-seeded blind input estimator — single file, no Z* dependencies.
Pipeline:
    1. Stochastic N4SID  →  A, C, Q, R  (output-only)
    2. Augmented LGSSM   →  [x; u] state, AR(1) prior on u
    3. Dynamax EM        →  refine all parameters
    4. RTS smoother      →  extract x̂, û from smoothed means
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm

from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.parameters import ParameterProperties

jax.config.update("jax_enable_x64", True)


# ─────────────────────────────────────────────────────────────────────────────
# LDS parameter container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LDSParams:
    A:   np.ndarray
    B:   np.ndarray
    C:   np.ndarray
    Q:   np.ndarray
    R:   np.ndarray
    mu_0: np.ndarray
    P_0:  np.ndarray


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
# Stochastic N4SID  (output-only)
# ─────────────────────────────────────────────────────────────────────────────

def stochastic_n4sid(
    outputs: np.ndarray,
    latent_dim: int,
    horizon: int | None = None,
) -> LDSParams:
    """Stochastic subspace ID — identifies (A, C, Q, R) from outputs alone.

    No B, no D. Input contribution is absorbed into Q. Use as EM seed only.

    Parameters
    ----------
    outputs    : (T, output_dim)
    latent_dim : state order n
    horizon    : block-Hankel horizon i > latent_dim; default max(n+2, 2n)
    """
    T          = outputs.shape[0]
    output_dim = outputs.shape[1]
    n          = latent_dim
    i          = horizon if horizon is not None else max(n + 2, 2 * n)
    num_windows = T - 2 * i + 1

    if num_windows <= 0:
        raise ValueError(f"Trial too short for horizon {i}: need T > {2*i}, got T={T}")

    def hankel(data, num_blocks, start):
        dim = data.shape[1]
        H   = np.empty((num_blocks * dim, num_windows))
        for row in range(num_blocks):
            H[row * dim:(row + 1) * dim] = data[start + row:start + row + num_windows].T
        return H

    def orthogonal(target, onto):
        return target @ np.linalg.pinv(onto) @ onto

    Y_p    = hankel(outputs, i,     0)
    Y_f    = hankel(outputs, i,     i)
    proj_i = orthogonal(Y_f, Y_p)

    Y_p_sh  = hankel(outputs, i + 1, 0)
    Y_f_sh  = hankel(outputs, i - 1, i + 1)
    proj_i1 = orthogonal(Y_f_sh, Y_p_sh)

    left_sv, sing_vals, _ = np.linalg.svd(proj_i, full_matrices=False)
    observability   = left_sv[:, :n] * np.sqrt(sing_vals[:n])
    observability_m = observability[:-output_dim]

    states_i  = np.linalg.pinv(observability)   @ proj_i
    states_i1 = np.linalg.pinv(observability_m) @ proj_i1
    outputs_i = hankel(outputs, 1, i)

    regressors = states_i
    targets    = np.vstack([states_i1, outputs_i])
    theta      = targets @ np.linalg.pinv(regressors)
    A          = theta[:n, :]
    C          = theta[n:, :]

    residual = targets - theta @ regressors
    cov      = (residual @ residual.T) / num_windows
    Q        = cov[:n, :n]
    R        = cov[n:, n:]

    return LDSParams(
        A=A, B=np.zeros((n, 0)), C=C,
        Q=Q, R=R,
        mu_0=np.zeros(n), P_0=np.eye(n),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Top-level estimator
# ─────────────────────────────────────────────────────────────────────────────

def festimate_latent_and_input(
    observation: np.ndarray,
    LatentDim: int,
    InputDim: int,
    horizon: int | None = None,
    phi: float = 0.9,
    sigma_u_sq: float = 1.0,
    num_em_iters: int = 100,
    num_restarts: int = 10,
    psd_jitter: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate latent states and hidden inputs from observations.

    Parameters
    ----------
    observation  : (T, m)   observed emissions
    LatentDim    : n        state dimension
    InputDim     : p        hidden input dimension
    horizon      : i        N4SID block-Hankel horizon; auto if None
    phi          : float    AR(1) coefficient on input prior
    sigma_u_sq   : float    input prior variance
    num_em_iters : int      EM iterations per restart
    num_restarts : int      number of EM restarts (1 = SSID seed only)
    psd_jitter   : float    diagonal jitter for PSD safety

    Returns
    -------
    latent_states : (T, LatentDim)
    inputs        : (T, InputDim)
    """
    T, emission_dim = observation.shape
    n, m, p = LatentDim, InputDim, emission_dim

    # ── Auto horizon ─────────────────────────────────────────────────────────
    if horizon is None:
        horizon = min(2 * n, (T - 1) // 4)
        horizon = max(horizon, n + 2)

    # ── Stage 1: stochastic N4SID seed ───────────────────────────────────────
    seed = stochastic_n4sid(observation, latent_dim=LatentDim, horizon=horizon)
    seed.A = _stabilise_A(seed.A) # stabilise A matrix
    print('n4sid')

    A_seed = np.asarray(seed.A)
    C_seed = np.asarray(seed.C)
    Q_seed = 0.5 * (np.asarray(seed.Q) + np.asarray(seed.Q).T) + psd_jitter * np.eye(n)
    R_seed = 0.1 * np.diag(observation.var(axis=0))

    # ── Stage 2: build augmented LGSSM  [x; u] ───────────────────────────────
    # x_{t+1} = A x_t + B u_t + w,   u_{t+1} = phi * u_t + noise
    # Augmented: z_t = [x_t; u_t],  z_{t+1} = A_tilde z_t + w_tilde

    print('state augmentation')
    Phi     = phi * np.eye(m)
    Q_u     = sigma_u_sq * np.eye(m)
    B_init  = np.zeros((n, m))

    A_tilde = np.block([[A_seed,         B_init            ],
                        [np.zeros((m, n)), Phi              ]])
    C_tilde = np.hstack([C_seed, np.zeros((p, m))])
    Q_tilde = np.block([[Q_seed,           np.zeros((n, m))],
                        [np.zeros((m, n)), Q_u              ]])

    model = LinearGaussianSSM(
        state_dim=n + m,
        emission_dim=p,
        input_dim=0,
    )

    emissions = jnp.asarray(observation)

    # ── Stage 3: EM ───────────────────────────────────────────────────────────
    print('em')
    best_params, best_ll = None, float("-inf")
    best_ll_trace        = None

    bar = tqdm(range(num_restarts), desc="EM (SSID-init)")
    for restart in bar:
        key    = jr.PRNGKey(restart)
        params, props = model.initialize(
            key,
            initial_mean       = jnp.zeros(n + m),
            initial_covariance = jnp.eye(n + m),
            dynamics_weights   = jnp.asarray(A_tilde),
            dynamics_covariance= jnp.asarray(Q_tilde),
            emission_weights   = jnp.asarray(C_tilde),
            emission_covariance= jnp.asarray(R_seed),
        )

        # Freeze initial-state statistics
        props = props._replace(
            initial=props.initial._replace(
                mean=ParameterProperties(trainable=False),
                cov =ParameterProperties(
                    trainable=False,
                    constrainer=props.initial.cov.constrainer,
                ),
            )
        )

        params, ll_trace = model.fit_em(
            params, props,
            emissions=emissions,
            num_iters=num_em_iters,
        )

        ll_trace_np = np.asarray(ll_trace)
        if np.any(~np.isfinite(ll_trace_np)):
            tqdm.write(f"restart {restart}: diverged, skipping")
            continue

        final_ll = float(ll_trace[-1])
        bar.set_postfix(best_ll=f"{best_ll:.2f}", current_ll=f"{final_ll:.2f}")

        if final_ll > best_ll:
            best_ll       = final_ll
            best_params   = params
            best_ll_trace = ll_trace_np

    if best_params is None:
        raise RuntimeError("All EM restarts diverged. Check horizon and latent_dim.")

    # ── Stage 4: RTS smoother  →  extract x̂, û ───────────────────────────────
    posterior     = model.smoother(best_params, emissions=emissions)
    smoothed_mean = np.asarray(posterior.smoothed_means)   # (T, n+m)

    latent_states = smoothed_mean[:, :n]
    inputs        = smoothed_mean[:, n:]

    return latent_states, inputs