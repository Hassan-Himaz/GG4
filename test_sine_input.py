"""
test_sine_input.py  —  Input structure dependence test

This script tests whether the estimator behaves differently when the input
has a strong time structure, using sinusoidal inputs.

Main idea:
  - The estimator only receives the observation Y.
  - The true input u is known only because this is a synthetic simulation.
  - The true input is used after estimation to calculate aligned R^2.

Why this test matters:
  - In the random-input tests, input changes are spread across time and are
    usually easier to separate from autonomous dynamics.
  - A sinusoidal input is highly regular and predictable.
  - Part of its effect may be fitted by A_hat as if it were autonomous
    latent dynamics, instead of remaining clearly in the residual
    r[t] = x[t+1] - A_hat x[t].
  - Therefore latent recovery can stay very strong, while input recovery can
    become weaker.

This is not meant to show that the estimator interface is wrong.
It shows that residual-based input recovery depends on input time structure.

Usage:
  python test_sine_input.py

Output:
  figures/sine_input_recovery.png
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Estimator import estimate_latent_and_input

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)


def make_data(
    T=300,
    N=40,
    n=4,
    m=2,
    noise=0.1,
    input_strength=0.4,
    seed=7,
):
    """
    Generate synthetic data with sinusoidal inputs.

    Model:
        x[t+1] = A x[t] + B u[t] + process noise
        y[t]   = C x[t] + observation noise

    Returns:
        Y : observations, shape (T, N)
        x : true latent states, shape (T, n)
        u : true inputs, shape (T, m)
        A, B, C : true system matrices
    """
    rng = np.random.default_rng(seed)

    # A: rotating and decaying latent dynamics.
    # Spectral radius is approximately 0.95.
    theta = 0.3
    A = np.zeros((n, n))

    i = 0
    while i + 1 < n:
        A[i:i + 2, i:i + 2] = 0.95 * np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )
        i += 2

    if n % 2 == 1:
        A[-1, -1] = 0.95

    # Random input and observation matrices.
    B = rng.standard_normal((n, m)) * input_strength
    C = rng.standard_normal((N, n))

    # Two sinusoidal input channels.
    # They have the same frequency but are phase shifted.
    t_arr = np.arange(T)
    frequency = 0.05

    u = np.zeros((T, m))
    for j in range(m):
        phase = j * np.pi / 2
        u[:, j] = np.sin(2 * np.pi * frequency * t_arr + phase)

    # Simulate latent states and observations.
    x = np.zeros((T, n))
    x[0] = rng.standard_normal(n)

    for t in range(1, T):
        process_noise = 0.02 * rng.standard_normal(n)
        x[t] = A @ x[t - 1] + B @ u[t] + process_noise

    Y = x @ C.T + noise * rng.standard_normal((T, N))

    return Y, x, u, A, B, C


def aligned_r2(est, true):
    """
    Compute R^2 after linear alignment.

    This is needed because the estimated latent state and input are only
    identifiable up to a linear transformation.

    Args:
        est: estimated signal, shape (T, d_est)
        true: true signal, shape (T, d_true)

    Returns:
        mean_r2: average R^2 across true dimensions
        r2_per_dim: R^2 for each true dimension
        est_aligned: linearly aligned estimate
    """
    W, *_ = np.linalg.lstsq(est, true, rcond=None)
    est_aligned = est @ W

    r2_per_dim = []
    for j in range(true.shape[1]):
        ss_res = np.sum((true[:, j] - est_aligned[:, j]) ** 2)
        ss_tot = np.sum((true[:, j] - true[:, j].mean()) ** 2)

        if ss_tot > 1e-12:
            r2_per_dim.append(1 - ss_res / ss_tot)
        else:
            r2_per_dim.append(np.nan)

    r2_per_dim = np.array(r2_per_dim)
    mean_r2 = float(np.nanmean(r2_per_dim))

    return mean_r2, r2_per_dim, est_aligned


def print_singular_value_checks(u_true, B):
    """
    Print simple checks on the time structure of u and Bu.

    This does not force the interpretation that sine input is rank-1.
    It only helps diagnose whether one direction is much weaker than another
    in this specific simulation.
    """
    Bu = u_true @ B.T

    s_u = np.linalg.svd(u_true, compute_uv=False)
    s_bu = np.linalg.svd(Bu, compute_uv=False)

    print("\nInput structure check")
    print("---------------------")
    print("Singular values of true input u:")
    print(np.round(s_u[:4], 4))

    print("Singular values of latent drive Bu:")
    print(np.round(s_bu[:4], 4))

    if len(s_bu) >= 2 and s_bu[1] > 1e-12:
        print(f"Bu first/second singular value ratio: {s_bu[0] / s_bu[1]:.3f}")
    else:
        print("Bu second singular value is near zero.")


def print_frequency_check(u_true, inp_aligned, T):
    """
    Check whether the recovered input contains the main sine frequency.
    """
    print("\nFrequency-domain check")
    print("----------------------")

    for col in range(u_true.shape[1]):
        true_fft = np.abs(np.fft.rfft(u_true[:, col]))
        est_fft = np.abs(np.fft.rfft(inp_aligned[:, col]))

        true_peak = (np.argmax(true_fft[1:]) + 1) / T
        est_peak = (np.argmax(est_fft[1:]) + 1) / T

        print(
            f"input dim {col + 1}: "
            f"true peak = {true_peak:.4f}, "
            f"recovered peak = {est_peak:.4f}"
        )


def plot_sine_recovery(u_true, inp_aligned, input_r2, input_r2_per):
    """
    Save a sine-only figure showing both input dimensions.

    This figure is designed for the report. It avoids a random-vs-sine
    comparison and focuses only on the structured-input case.
    """
    T = u_true.shape[0]
    time = np.arange(T)
    m = u_true.shape[1]

    fig, axes = plt.subplots(m, 1, figsize=(8.2, 4.6), sharex=True)

    if m == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        ax.plot(time, u_true[:, j], label="true input", lw=1.2)
        ax.plot(time, inp_aligned[:, j], "--", label="recovered input", lw=1.0)

        ax.set_ylabel(f"input dim {j + 1}")
        ax.set_title(f"dim {j + 1}: aligned R² = {input_r2_per[j]:.2f}")
        ax.grid(alpha=0.3)

        if j == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time")

    fig.suptitle(
        f"Input recovery under sinusoidal drive "
        f"(mean aligned R² = {input_r2:.2f})",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(FIGDIR, "sine_input_recovery.png"),
        dpi=130,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    print("=" * 70)
    print("Sinusoidal input structure test")
    print("=" * 70)

    Y, x_true, u_true, A, B, C = make_data()

    latent_hat, input_hat = estimate_latent_and_input(
        observation=Y,
        LatentDim=x_true.shape[1],
        InputDim=u_true.shape[1],
    )

    latent_r2, latent_r2_per, latent_aligned = aligned_r2(latent_hat, x_true)
    input_r2, input_r2_per, input_aligned = aligned_r2(input_hat, u_true)

    print("\nRecovery scores")
    print("---------------")
    print(f"Latent recovery mean aligned R²: {latent_r2:.3f}")
    print(f"Latent recovery per dimension:  {np.round(latent_r2_per, 3)}")
    print(f"Input recovery mean aligned R²:  {input_r2:.3f}")
    print(f"Input recovery per dimension:   {np.round(input_r2_per, 3)}")

    print_singular_value_checks(u_true, B)
    print_frequency_check(u_true, input_aligned, T=u_true.shape[0])

    plot_sine_recovery(
        u_true=u_true,
        inp_aligned=input_aligned,
        input_r2=input_r2,
        input_r2_per=input_r2_per,
    )

    print(f"\nFigure saved to {FIGDIR}/sine_input_recovery.png")


if __name__ == "__main__":
    main()