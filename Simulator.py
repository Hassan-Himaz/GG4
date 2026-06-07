"""
simulator.py

Self-contained simulation + plotting utilities for the Week 2 estimator tests.
Reconstructed to reproduce the behaviour used in Week2Competition.ipynb, so it
can be used as a drop-in replacement for the (missing) SimulationIllustrationFuncs.

Linear dynamical system convention (matches the estimator model in Fiana.py):

    x(t+1) = A x(t) + B u(t) + x_noise(t)
    y(t)   = C x(t) + y_noise(t)

with x(0) = B u(0) + x_noise(0) (i.e. the loop below starts from a zero state
before the first input/noise is applied).

Public API
----------
- simulate(A, B, C, inputs, x_noise, y_noise)
      -> states, observations

- simulate_from_latents(A, B, C, latents, x_noise, y_noise, maxu)
      -> states, observations, inputs, noiseless_states

- fit_transform_latents(true_states, est_states)
      -> est_states mapped into the true-state frame (least squares)

- plot_input_latent_observation(inputs, states, observations)
      -> Figure with three stacked panels

- plot_input_latent_comparison(inputs, states, results, ...)
- plot_input_latent_comparison_interactive(...)   # alias of the above
      -> Figure comparing each method's latents (and inputs) vs ground truth
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Core simulation
# ============================================================

def simulate(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    inputs: np.ndarray,
    x_noise: np.ndarray,
    y_noise: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Roll out a linear dynamical system.

    Parameters
    ----------
    A : (k, k) transition matrix
    B : (k, m) input matrix
    C : (n, k) observation matrix
    inputs  : (T, m) input sequence u(t)
    x_noise : (T, k) process noise added at each step
    y_noise : (T, n) observation noise added at each step

    Returns
    -------
    states       : (T, k) latent trajectory x(t)
    observations : (T, n) observed trajectory y(t)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    inputs = np.asarray(inputs, dtype=float)
    x_noise = np.asarray(x_noise, dtype=float)
    y_noise = np.asarray(y_noise, dtype=float)

    T = inputs.shape[0]
    k = A.shape[0]
    n = C.shape[0]

    states = np.zeros((T, k))
    observations = np.zeros((T, n))

    x = np.zeros(k)
    for t in range(T):
        if t == 0:
            x = B @ inputs[0] + x_noise[0]
        else:
            x = A @ states[t - 1] + B @ inputs[t] + x_noise[t]
        states[t] = x
        observations[t] = C @ x + y_noise[t]

    return states, observations


# ------------------------------------------------------------
# Min-energy trajectory planning for simulate_from_latents
# ------------------------------------------------------------

def _plan_min_energy_inputs(
    A: np.ndarray,
    B: np.ndarray,
    targets_by_t: Dict[int, Dict[int, float]],
    T: int,
    maxu: float,
    ridge: float = 1e-6,
) -> np.ndarray:
    """
    Find a min-energy input sequence (no noise) so that, at each constrained
    time step t, the specified latent components of x(t) hit their targets.

    Because B may not directly span a requested latent direction
    (the "null-space injection" cases, e.g. B = [[1],[1],[0]] cannot move
    latent index 2 in one step), we plan over the *window leading up to* each
    waypoint rather than a single step. For a waypoint at time t we solve

        min   sum_{s<=t} ||u(s)||^2
        s.t.  [ x(t) ]_rows = target_rows

    where x(t) is the linear (noiseless) response to u(0..t). This is the
    standard discrete-time min-energy reachability problem; the controllability
    structure of (A, B) is what lets an input applied earlier steer a latent
    component that B cannot touch instantaneously.

    Waypoints are processed in time order; inputs already committed for earlier
    waypoints are kept fixed, and each new waypoint plans only over the steps
    since the previous one (a receding-horizon scheme). This matches the
    incremental "drive toward this, then toward that" behaviour of the tests.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    k = A.shape[0]
    m = B.shape[1]

    inputs = np.zeros((T, m))

    # state reached so far using already-committed inputs (noiseless)
    def rollout(u_seq: np.ndarray, t_end: int) -> np.ndarray:
        x = np.zeros(k)
        for s in range(t_end + 1):
            x = A @ x + B @ u_seq[s] if s > 0 else B @ u_seq[0]
        return x

    sorted_times = sorted(targets_by_t.keys())
    prev_t = -1
    for t in sorted_times:
        tgt = targets_by_t[t]
        rows = sorted(tgt.keys())
        target_vals = np.array([tgt[r] for r in rows], dtype=float)

        # window of free steps we are allowed to set for this waypoint:
        # from just after the previous waypoint up to and including t.
        start = prev_t + 1
        window = list(range(start, t + 1))
        if not window:
            window = [t]

        # state at time start-1 from committed inputs (fixed contribution)
        if start - 1 >= 0:
            x_fixed_prev = rollout(inputs, start - 1)
        else:
            x_fixed_prev = np.zeros(k)

        # x(t) = A^{t-(start-1)} x_fixed_prev + sum_{s in window} A^{t-s} B u(s)
        # Build the linear map M: stacked u(window) -> x(t)
        cols = []
        for s in window:
            cols.append(np.linalg.matrix_power(A, t - s) @ B)   # (k, m)
        M = np.hstack(cols) if cols else np.zeros((k, 0))       # (k, m*len(window))

        # contribution of the fixed past propagated to time t
        steps = t - (start - 1)
        x_free0 = np.linalg.matrix_power(A, steps) @ x_fixed_prev

        Msub = M[rows, :]                       # (len(rows), m*len(window))
        desired = target_vals - x_free0[rows]

        # min-norm least squares: u = M^T (M M^T + ridge I)^-1 desired
        G = Msub @ Msub.T + ridge * np.eye(len(rows))
        u_stacked = Msub.T @ np.linalg.solve(G, desired)
        u_stacked = u_stacked.reshape(len(window), m)
        u_stacked = np.clip(u_stacked, -maxu, maxu)

        for i, s in enumerate(window):
            inputs[s] = u_stacked[i]

        prev_t = t

    return inputs


def simulate_from_latents(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    latents: Dict[Tuple[int, int], float],
    x_noise: np.ndarray,
    y_noise: np.ndarray,
    maxu: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Plan an input sequence that drives specified latent components toward target
    ("waypoint") values, then simulate the resulting system both with and
    without noise.

    `latents` maps (time_index, latent_index) -> target value.

    Returns
    -------
    states          : (T, k) noisy latent trajectory actually realised
    observations    : (T, n) noisy observations of `states`
    inputs          : (T, m) the input sequence that was applied
    noiseless_states: (T, k) latent trajectory with the same inputs but no noise
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    x_noise = np.asarray(x_noise, dtype=float)
    y_noise = np.asarray(y_noise, dtype=float)

    k = A.shape[0]
    n = C.shape[0]
    T = x_noise.shape[0]

    targets_by_t: Dict[int, Dict[int, float]] = {}
    for (t, idx), val in latents.items():
        targets_by_t.setdefault(int(t), {})[int(idx)] = float(val)

    inputs = _plan_min_energy_inputs(A, B, targets_by_t, T, maxu)

    noiseless_states, _ = simulate(
        A, B, C, inputs, np.zeros((T, k)), np.zeros((T, n))
    )
    states, observations = simulate(A, B, C, inputs, x_noise, y_noise)

    return states, observations, inputs, noiseless_states


# ============================================================
# Latent alignment
# ============================================================

def fit_transform_latents(
    true_states: np.ndarray,
    est_states: np.ndarray,
) -> np.ndarray:
    """
    Map estimated latents into the ground-truth frame by least squares.

    Latent states are only identifiable up to an invertible linear transform,
    so before comparing we solve  min_W ||est_states @ W - true_states||_F
    and return est_states @ W.
    """
    true_states = np.asarray(true_states, dtype=float)
    est_states = np.asarray(est_states, dtype=float)
    W, *_ = np.linalg.lstsq(est_states, true_states, rcond=None)
    return est_states @ W


# ============================================================
# Plotting
# ============================================================

def plot_input_latent_observation(
    inputs: np.ndarray,
    states: np.ndarray,
    observations: np.ndarray,
):
    """Three stacked panels: inputs u(t), latent states x(t), observations y(t)."""
    inputs = np.asarray(inputs, dtype=float)
    states = np.asarray(states, dtype=float)
    observations = np.asarray(observations, dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(inputs)
    axes[0].set_ylabel("input u(t)")
    axes[0].set_title("Inputs")

    axes[1].plot(states)
    axes[1].set_ylabel("latent x(t)")
    axes[1].set_title("Latent states")

    axes[2].plot(observations)
    axes[2].set_ylabel("obs y(t)")
    axes[2].set_xlabel("time")
    axes[2].set_title("Observations")

    fig.tight_layout()
    return fig


def plot_input_latent_comparison(
    inputs: np.ndarray,
    states: np.ndarray,
    results: Dict[str, Dict[str, np.ndarray]],
    align_inputs: bool = False,
    align_latents: bool = True,
    active: Optional[dict] = None,
):
    """
    Compare ground-truth latents (and inputs) against each method's estimate.

    `results` matches parallel_runner.run_functions_parallel's format:
        { name: {"latent_states": (T,k), "inputs": (T,m)}, ... }

    align_latents : map each method's latents into the true frame first
                    (latents identifiable only up to a linear transform).
    align_inputs  : least-squares sign/scale align each method's inputs.
    active        : dict; only names with truthy value are drawn. Empty/None -> all.
    """
    inputs = np.asarray(inputs, dtype=float)
    states = np.asarray(states, dtype=float)

    def is_active(name: str) -> bool:
        if not active:
            return True
        return bool(active.get(name, False))

    names = [nm for nm in results.keys() if is_active(nm)]

    cmap = plt.cm.tab10

    # Pre-align each method's latents into the true frame once, so every
    # per-dimension subplot below uses the same aligned estimate.
    aligned_latents = {}
    for nm in names:
        est = np.asarray(results[nm]["latent_states"], dtype=float)
        if align_latents and est.ndim == 2 and est.shape[0] == states.shape[0]:
            try:
                est = fit_transform_latents(states, est)
            except Exception:
                pass
        aligned_latents[nm] = est

    aligned_inputs = {}
    for nm in names:
        est_u = np.asarray(results[nm]["inputs"], dtype=float)
        if align_inputs and est_u.shape == inputs.shape:
            try:
                W, *_ = np.linalg.lstsq(est_u, inputs, rcond=None)
                est_u = est_u @ W
            except Exception:
                pass
        aligned_inputs[nm] = est_u

    k = states.shape[1]                 # number of latent dimensions
    m = inputs.shape[1] if inputs.ndim == 2 else 1   # number of input dimensions
    n_rows = k + m

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(11, 2.4 * n_rows), sharex=True, squeeze=False
    )
    axes = axes[:, 0]

    # ----- one subplot per latent dimension -----
    for d in range(k):
        ax = axes[d]
        ax.plot(states[:, d], color="black", lw=2.0, alpha=0.9,
                label="ground truth")
        for i, nm in enumerate(names):
            est = aligned_latents[nm]
            if est.ndim == 2 and est.shape[1] > d:
                ax.plot(est[:, d], color=cmap(i % 10), lw=1.2, alpha=0.85,
                        label=nm)
        ax.set_ylabel(f"latent x[{d}]")
        ax.set_title(f"Latent dimension {d}: ground truth (black) vs estimates",
                     fontsize=10)
        if d == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)

    # ----- one subplot per input dimension -----
    inputs_2d = inputs if inputs.ndim == 2 else inputs.reshape(-1, 1)
    for j in range(m):
        ax = axes[k + j]
        ax.plot(inputs_2d[:, j], color="black", lw=2.0, alpha=0.9,
                label="ground truth")
        for i, nm in enumerate(names):
            est_u = aligned_inputs[nm]
            est_u2 = est_u if est_u.ndim == 2 else est_u.reshape(-1, 1)
            if est_u2.shape[1] > j:
                ax.plot(est_u2[:, j], color=cmap(i % 10), lw=1.2, alpha=0.85,
                        label=nm)
        ax.set_ylabel(f"input u[{j}]")
        ax.set_title(f"Input dimension {j}: ground truth (black) vs estimates",
                     fontsize=10)

    axes[-1].set_xlabel("time")
    fig.tight_layout()
    return fig


# Backwards-compatible alias matching the notebook's original import name.
plot_input_latent_comparison_interactive = plot_input_latent_comparison
