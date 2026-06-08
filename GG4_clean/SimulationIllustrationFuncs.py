import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from numpy.linalg import svd, solve, inv, lstsq
from scipy.optimize import linear_sum_assignment

def simulate(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    inputs: np.ndarray,
    x_noise: np.ndarray,
    y_noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the LDS forward model.

    Parameters
    ----------
    A        : (n, n)   state transition
    B        : (n, p)   input matrix
    C        : (m, n)   observation matrix
    inputs   : (T, p)   input sequence
    x_noise  : (T, n)   process noise realisations
    y_noise  : (T, m)   observation noise realisations
    x0       : (n,)     initial state; defaults to zeros

    Returns
    -------
    states       : (T, n)
    observations : (T, m)
    """
    T, p = inputs.shape
    n    = A.shape[0]
    m    = C.shape[0]

    states       = np.zeros((T, n))
    observations = np.zeros((T, m))

    x = np.zeros(n) # gonna default initial states 0

    for t in range(T):
        observations[t] = C @ x + y_noise[t]
        x = A @ x + B @ inputs[t] + x_noise[t]
        states[t] = x

    return states, observations



def simulate_from_latents(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    latents: dict[tuple[int, int], float],
    x_noise: np.ndarray,
    y_noise: np.ndarray,
    x0: Optional[np.ndarray] = None,
    maxu: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate LDS with minimum-energy inputs that hit target latent waypoints.

    Solves:
        min   Σ_t ||u_t||²
        s.t.  x_{t+1} = A x_t + B u_t + w_t
              x_t[dim] = val   for (t, dim) in latents

    Strategy
    --------
    1. Run free (u=0) trajectory to get the noise-driven baseline x_free.
    2. For each waypoint (t, dim)=val, compute the deficit:
           δ_t[dim] = val - x_free_t[dim]
       This is the state deviation that inputs must produce.
    3. Build the controllability matrix mapping inputs → state deviations.
    4. Solve the minimum-norm input sequence via pseudoinverse.
    5. Clip inputs to maxu, re-simulate to get final trajectory.

    Parameters
    ----------
    A       : (n, n)
    B       : (n, p)
    C       : (m, n)
    latents : {(t, dim): value}  sparse waypoint constraints
    x_noise : (T, n)
    y_noise : (T, m)
    x0      : (n,) initial state; defaults to zeros
    maxu    : per-element input clipping threshold

    Returns
    -------
    states          : (T, n)
    observations    : (T, m)
    inputs          : (T, p)   recovered input sequence
    noiseless_states: (T, n)   trajectory without observation noise
    """
    T, n = x_noise.shape
    p    = B.shape[1]
    m    = C.shape[0]
    x0   = np.zeros(n) if x0 is None else x0.copy()

    # ── Step 1: free trajectory (u=0, with process noise) ───────────────────
    x_free = np.zeros((T, n))
    x = x0.copy()
    for t in range(T):
        x = A @ x + x_noise[t]
        x_free[t] = x

    # ── Step 2: build system of equations  Φ u = δ ──────────────────────────
    # State at time t due to inputs u_0..u_{t-1}:
    #   x_t^{input} = Σ_{k=0}^{t-1} A^{t-1-k} B u_k
    #
    # We have one scalar constraint per waypoint.
    # Stack into:  Φ (n_waypoints, T*p)  @  u_flat (T*p,)  =  delta (n_waypoints,)

    waypoint_list = sorted(latents.items())   # [(t, dim, val), ...]
    n_wp = len(waypoint_list)

    Phi   = np.zeros((n_wp, T * p))
    delta = np.zeros(n_wp)

    for row, ((t_wp, dim), val) in enumerate(waypoint_list):
        delta[row] = val - x_free[t_wp - 1][dim]   # deficit vs free trajectory

        # Contribution of u_k to x_{t_wp}[dim]:
        #   e_dim^T  A^{t_wp-1-k} B   for k = 0..t_wp-1
        e_dim = np.zeros(n)
        e_dim[dim] = 1.0

        for k in range(t_wp):
            power    = t_wp - 1 - k
            AB_power = np.linalg.matrix_power(A, power) @ B   # (n, p)
            contrib  = e_dim @ AB_power                        # (p,)
            Phi[row, k * p:(k + 1) * p] = contrib

    # ── Step 3: minimum-norm solution  u_flat = Φ^+ δ ───────────────────────
    u_flat, _, _, _ = lstsq(Phi, delta, rcond=None)            # (T*p,)

    # ── Step 4: reshape, clip, re-simulate ───────────────────────────────────
    inputs = u_flat.reshape(T, p)
    inputs = np.clip(inputs, -maxu, maxu)

    states          = np.zeros((T, n))
    observations    = np.zeros((T, m))
    noiseless_states = np.zeros((T, m)) # change dims

    x        = x0.copy()
    x_nless  = x0.copy()

    for t in range(T):
        observations[t]     = C @ x        + y_noise[t]
        noiseless_states[t] = C @ x_nless
        x       = A @ x       + B @ inputs[t] + x_noise[t]
        x_nless = A @ x_nless + B @ inputs[t]
        states[t] = x

    return states, observations, inputs, noiseless_states




def plot_input_latent_observation(
    inputs: np.ndarray,
    states: np.ndarray,
    observations: np.ndarray,
    figsize: tuple[int, int] = (12, 10),
    input_color: str = "tab:red",
    state_cmap: str = "tab10",
    obs_cmap: str = "Blues",
) -> None:
    """Plot inputs, each latent state on its own subplot, and observations.

    Parameters
    ----------
    inputs       : (T, p)  input sequence
    states       : (T, n)  latent state trajectory
    observations : (T, m)  observation matrix
    """
    T, p = inputs.shape
    _, n = states.shape
    _, m = observations.shape
    t    = np.arange(T)

    n_rows = 1 + n + 1   # inputs | n state panels | observations
    height_ratios = [1] + [1.2] * n + [2 if m > 8 else 1.2]

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.15},
    )

    colors = plt.get_cmap(state_cmap).colors

    # ── Inputs ────────────────────────────────────────────────────────────────
    ax_u = axes[0]
    for i in range(p):
        ax_u.stem(t, inputs[:, i], linefmt=input_color, markerfmt=" ",
                  basefmt="none", label=f"u{i}" if p > 1 else None)
    ax_u.set_ylabel("input", fontsize=9)
    ax_u.set_xlim(0, T - 1)
    ax_u.tick_params(labelbottom=False)
    ax_u.spines[["top", "right"]].set_visible(False)
    if p > 1:
        ax_u.legend(fontsize=8, loc="upper right", frameon=False)

    # ── One subplot per latent state ──────────────────────────────────────────
    for i in range(n):
        ax = axes[1 + i]
        ax.plot(t, states[:, i], lw=1.1, color=colors[i % len(colors)])
        ax.set_ylabel(f"$x_{i}$", fontsize=9, rotation=0, labelpad=14)
        ax.set_xlim(0, T - 1)
        ax.tick_params(labelbottom=False)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Observations ──────────────────────────────────────────────────────────
    ax_y = axes[-1]
    if m > 8:
        im = ax_y.imshow(
            observations.T,
            aspect="auto", interpolation="nearest",
            cmap=obs_cmap, extent=[0, T - 1, m - 0.5, -0.5],
        )
        ax_y.set_ylabel("channel", fontsize=9)
        fig.colorbar(im, ax=ax_y, fraction=0.015, pad=0.02)
    else:
        for i in range(m):
            ax_y.plot(t, observations[:, i], lw=0.9, alpha=0.8,
                      color=colors[i % len(colors)], label=f"y{i}")
        ax_y.set_ylabel("obs", fontsize=9)
        ax_y.legend(fontsize=8, loc="upper right", frameon=False, ncol=min(m, 4))

    ax_y.set_xlabel("timestep", fontsize=9)
    ax_y.set_xlim(0, T - 1)
    ax_y.spines[["top", "right"]].set_visible(False)

    # shared x-axis tick labels only on bottom panel
    ax_y.tick_params(labelbottom=True)

    

def fit_transform_latents(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align estimated latent states to reference via permutation and sign correction.

    Solves the optimal column permutation and sign pattern that maximises
    correlation between estimate and reference. Useful since SSI/EM recover
    states in an arbitrary basis.

    Parameters
    ----------
    reference : (T, n)  ground truth states
    estimate  : (T, n)  estimated states

    Returns
    -------
    aligned   : (T, n)  estimate aligned to reference
    perm      : (n,)    column permutation indices
    signs     : (n,)    sign vector applied after permutation
    """
    from scipy.optimize import linear_sum_assignment

    T, n  = reference.shape
    corr  = reference.T @ estimate / T             # (n, n)

    _, col_ind = linear_sum_assignment(-np.abs(corr))
    signs      = np.sign(corr[np.arange(n), col_ind])

    aligned = estimate[:, col_ind] * signs[np.newaxis, :]

    return aligned, col_ind, signs


def plot_input_latent_comparison_interactive(
    inputs: np.ndarray,
    states: np.ndarray,
    results: dict[str, dict],
    align_inputs: bool = False,
    align_latents: bool = True,
    active: dict[str, bool] = {},
    figsize: tuple[int, int] = (13, 11),
    state_cmap: str = "tab10",
    input_color: str = "tab:red",
) -> None:
    T, n = states.shape
    _, p = inputs.shape

    if not active:
        visible = {name: True for name in results}
    else:
        visible = {name: active.get(name, True) for name in results}
    visible_names = [name for name, show in visible.items() if show]

    result_colors = {
        name: plt.get_cmap(state_cmap)(i / max(len(results), 1))
        for i, name in enumerate(results)
    }

    n_rows        = n + 1
    height_ratios = [1.5] * n + [1.2]

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.12},
    )
    if n_rows == 1:
        axes = [axes]

    t = np.arange(T)

    # ── Pre-align states ──────────────────────────────────────────────────────
    aligned_states = {}
    for name in visible_names:
        est = results[name]["latent_states"]
        if align_latents:
            est, _, _ = fit_transform_latents(states, est)
        aligned_states[name] = est

    # ── Pre-align inputs ──────────────────────────────────────────────────────
    aligned_inputs = {}
    for name in visible_names:
        est = results[name]["inputs"]
        if align_inputs:
            est, _, _ = fit_transform_latents(inputs, est)
        aligned_inputs[name] = est

    # ── State panels ──────────────────────────────────────────────────────────
    for dim in range(n):
        ax = axes[dim]

        if align_latents:
            ax.plot(t, states[:, dim], color="black", lw=1.4,
                    label="ground truth" if dim == 0 else None, zorder=3)

        for name in visible_names:
            ax.plot(t, aligned_states[name][:, dim], color=result_colors[name],
                    lw=1.0, alpha=0.8, label=name if dim == 0 else None)

        ax.set_ylabel(f"$x_{dim}$", fontsize=9, rotation=0, labelpad=16)
        ax.set_xlim(0, T - 1)
        ax.tick_params(labelbottom=False)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Input panel ───────────────────────────────────────────────────────────
    ax_u = axes[-1]

    if align_inputs:
        ax_u.stem(t, inputs[:, 0], linefmt="black", markerfmt=" ",
                  basefmt="none", label="ground truth")

    for name in visible_names:
        ax_u.plot(t, aligned_inputs[name][:, 0], color=result_colors[name],
                  lw=1.0, alpha=0.8, label=name)

    ax_u.set_ylabel("input", fontsize=9)
    ax_u.set_xlabel("timestep", fontsize=9)
    ax_u.set_xlim(0, T - 1)
    ax_u.spines[["top", "right"]].set_visible(False)

    axes[0].legend(
        fontsize=8, loc="upper right", frameon=False,
        ncol=min(len(visible_names) + 1, 4),
    )