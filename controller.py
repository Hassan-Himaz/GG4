# controller.py -- Final Week 3 controller
# Affine LDS + DARE LQR + empirical reference calibration
# Main idea:
#   identify: x[t+1] = A x[t] + B u[t] + d, y[t] = C x[t] + y_mean
#   control : u = u_base - K (x_hat - x_ref)
#   reference is from empirical calibration, not model inversion.

import numpy as np
from scipy.linalg import solve_discrete_are


def moving_average(x, window=9):
    x = np.asarray(x, float)
    if window <= 1:
        return x.copy()
    pad_l = window // 2
    pad_r = window - 1 - pad_l
    xp = np.pad(x, (pad_l, pad_r), mode="edge")
    return np.convolve(xp, np.ones(window) / window, mode="valid")


def collect_with_inputs(brain, U_seq):
    Y = []
    for u in U_seq:
        Y.append(np.array(brain.measure(), float))
        brain.next_state(np.clip(u, 0.0, 1.0))
    return np.array(Y), np.asarray(U_seq, float)


def make_mixed_inputs(n_steps, input_dim, seed=0):
    rng = np.random.default_rng(seed)
    U = np.zeros((n_steps, input_dim))
    s1 = n_steps // 5
    s2 = 2 * n_steps // 5
    s3 = 3 * n_steps // 5
    s4 = 4 * n_steps // 5
    s5 = n_steps

    U[:s1] = 0.0
    U[s1:s2] = 0.5

    for k in range(s2, s3):
        val = 1.0 if ((k - s2) // 30) % 2 == 0 else 0.0
        U[k] = val

    U[s3:s4] = rng.uniform(0.0, 1.0, size=(s4 - s3, input_dim))

    tt = np.arange(s5 - s4)
    for j in range(input_dim):
        phase = rng.uniform(0, 2 * np.pi)
        period = rng.choice([30, 40, 60, 80])
        U[s4:s5, j] = 0.5 + 0.5 * np.sin(2 * np.pi * tt / period + phase)

    return np.clip(U, 0.0, 1.0)


def collect_continuous(brain, n_steps, input_dim, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    Y, U = [], []
    for _ in range(n_steps):
        y = np.array(brain.measure(), float)
        u = rng.uniform(0.0, 1.0, input_dim)
        Y.append(y)
        U.append(u)
        brain.next_state(u)
    return np.array(Y), np.array(U)


def collect_continuous_mixed(brain, n_steps, input_dim, seed=0):
    """
    Identification data on a SINGLE continuous brain (no reset),
    but using a structured mixed-excitation sequence instead of pure
    white noise. The mixed sequence (steps, square waves, white noise,
    multi-period sines) excites both fast and slow modes, which is
    important when the true system has slow dynamics that white noise
    barely touches.

    'seed' only controls how the input SEQUENCE is generated; the brain
    itself is never re-seeded, so this is not seed-cheating.
    """
    U_seq = make_mixed_inputs(n_steps, input_dim, seed=seed)
    Y, U = [], []
    for u in U_seq:
        Y.append(np.array(brain.measure(), float))
        brain.next_state(np.clip(u, 0.0, 1.0))
        U.append(u)
    return np.array(Y), np.asarray(U, float)


def identify(Y, U, latent_dim, stabilize=False):
    Y = np.asarray(Y, float)
    U = np.asarray(U, float)
    T, p = Y.shape
    m = U.shape[1]
    n = latent_dim

    y_mean = Y.mean(axis=0, keepdims=True)
    Yc = Y - y_mean

    Usvd, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    k = min(n, Usvd.shape[1])
    Z = Usvd[:, :k] * S[:k]
    if k < n:
        Z = np.hstack([Z, np.zeros((T, n - k))])

    # affine dynamics: Z[t+1] = A Z[t] + B U[t] + d
    Phi = np.hstack([Z[:-1], U[:-1], np.ones((T - 1, 1))])
    Tgt = Z[1:]
    ABD = np.linalg.lstsq(Phi, Tgt, rcond=None)[0].T
    A = ABD[:, :n]
    B = ABD[:, n:n + m]
    d = ABD[:, -1]

    if stabilize:
        rho = np.max(np.abs(np.linalg.eigvals(A)))
        if rho > 1.0:
            A = A * (0.99 / rho)

    C = np.linalg.lstsq(Z, Yc, rcond=None)[0].T

    Z_pred = Z[:-1] @ A.T + U[:-1] @ B.T + d
    dyn_res = Z[1:] - Z_pred
    Q = (dyn_res.T @ dyn_res) / max(T - 1, 1)
    Q = 0.5 * (Q + Q.T) + 1e-6 * np.eye(n)

    obs_res = Yc - Z @ C.T
    R = np.diag(np.maximum(np.mean(obs_res ** 2, axis=0), 1e-6))

    return dict(A=A, B=B, d=d, C=C, Q=Q, R=R, y_mean=y_mean.ravel())


def design_lqr(A, B, q_weight=1.0, r_weight=0.3):
    n = A.shape[0]
    m = B.shape[1]
    Q = np.eye(n) * q_weight
    R = np.eye(m) * r_weight
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def y_to_x(params, y):
    C = params["C"]
    ym = params["y_mean"]
    return (np.asarray(y, float) - ym) @ np.linalg.pinv(C).T


def model_steady_norm(params, u):
    A, B, C = params["A"], params["B"], params["C"]
    d = params.get("d", np.zeros(A.shape[0]))
    ym = params["y_mean"]
    I = np.eye(A.shape[0])
    u = np.asarray(u, float)
    try:
        x_ss = np.linalg.solve(I - A, B @ u + d)
    except np.linalg.LinAlgError:
        x_ss = np.linalg.pinv(I - A) @ (B @ u + d)
    return float(np.linalg.norm(C @ x_ss + ym))


def calibrate_continuous_references(brain, params, input_dim, levels,
                                    hold=120, tail=50, settle=30,
                                    direction=None):
    """
    Continuous empirical calibration on the same brain.
    No reset. For each level a, apply u = a * direction.
    Default direction is [1,1].
    """
    if direction is None:
        direction = np.ones(input_dim)
    direction = np.asarray(direction, float)
    if np.max(np.abs(direction)) > 0:
        direction = direction / np.max(np.abs(direction))

    records = []
    for a in levels:
        u = np.clip(float(a) * direction, 0.0, 1.0)

        for _ in range(settle):
            brain.next_state(u)

        Ys = []
        for _ in range(hold):
            y = np.array(brain.measure(), float)
            Ys.append(y)
            brain.next_state(u)

        Ys = np.array(Ys)
        y_ss = Ys[-tail:].mean(axis=0)
        records.append(dict(
            a=float(a),
            u=u.copy(),
            y_ss=y_ss.copy(),
            x_ss=y_to_x(params, y_ss),
            norm_ss=float(np.linalg.norm(y_ss)),
        ))

    return sorted(records, key=lambda r: r["norm_ss"])


def choose_interpolated_reference(records, target_norm):
    norms = np.array([r["norm_ss"] for r in records], float)
    xs = np.array([r["x_ss"] for r in records], float)
    us = np.array([r["u"] for r in records], float)
    aas = np.array([r["a"] for r in records], float)

    target = float(np.clip(target_norm, norms.min(), norms.max()))

    x_ref = np.array([np.interp(target, norms, xs[:, i]) for i in range(xs.shape[1])])
    u_ref = np.array([np.interp(target, norms, us[:, j]) for j in range(us.shape[1])])
    a_ref = float(np.interp(target, norms, aas))
    return x_ref, np.clip(u_ref, 0.0, 1.0), target, a_ref


def choose_nearest_reference(records, target_norm):
    best = min(records, key=lambda r: abs(r["norm_ss"] - target_norm))
    return best["x_ss"], best["u"], best["norm_ss"], best["a"]


class Kalman:
    def __init__(self, params):
        self.A = params["A"]
        self.B = params["B"]
        self.C = params["C"]
        self.d = params.get("d", np.zeros(self.A.shape[0]))
        self.Q = params["Q"]
        self.R = params["R"]
        self.y_mean = params["y_mean"]
        n = self.A.shape[0]
        self.x = np.zeros(n)
        self.P = np.eye(n)
        self.first = True

    def step(self, y_raw, u_prev):
        y = np.asarray(y_raw, float) - self.y_mean
        if self.first:
            xp = self.x.copy()
            Pp = self.P.copy()
            self.first = False
        else:
            xp = self.A @ self.x + self.B @ np.asarray(u_prev, float).ravel() + self.d
            Pp = self.A @ self.P @ self.A.T + self.Q

        S = self.C @ Pp @ self.C.T + self.R
        try:
            Kgain = np.linalg.solve(S.T, (Pp @ self.C.T).T).T
        except np.linalg.LinAlgError:
            Kgain = Pp @ self.C.T @ np.linalg.pinv(S)

        self.x = xp + Kgain @ (y - self.C @ xp)
        self.P = (np.eye(self.A.shape[0]) - Kgain @ self.C) @ Pp
        return self.x.copy()


def run_empirical_lqr(brain, params, K, target_seq, records,
                      input_smoothing=0.2,
                      reference_mode="interp",
                      feedback_clip=0.25,
                      integral_gain=0.0,
                      integral_clip=15.0,
                      min_activation=0.0,
                      log_xhat=False):
    """
    Final main controller.

    reference_mode:
        "interp"  : smooth interpolated empirical reference
        "nearest" : nearest empirical reference

    feedback_clip:
        limits the LQR correction so that noisy x_hat does not cause
        violent input chattering. This makes plots and control effort cleaner.

    min_activation:
        a small lower floor applied to EVERY input channel whenever the
        reference asks for any positive drive (u_base > 0). This guarantees
        that we never drive the system through a single electrode only --
        both modes associated with the two input channels stay excited.
        Set 0.0 to disable.
    """
    kf = Kalman(params)
    m = params["B"].shape[1]
    u_prev = np.zeros(m)
    integ = 0.0

    y_log, u_log, ref_log, reach_log, a_log, xhat_log = [], [], [], [], [], []

    for target in target_seq:
        if reference_mode == "nearest":
            x_ref, u_base, reachable, a = choose_nearest_reference(records, target)
        else:
            x_ref, u_base, reachable, a = choose_interpolated_reference(records, target)

        y = np.array(brain.measure(), float)
        y_norm = np.linalg.norm(y)
        x_hat = kf.step(y, u_prev)

        fb = -K @ (x_hat - x_ref)
        fb = np.clip(fb, -feedback_clip, feedback_clip)

        integ = np.clip(integ + (target - y_norm), -integral_clip, integral_clip)
        integ_u = integral_gain * integ * np.ones(m)

        u_cmd = np.clip(u_base + fb + integ_u, 0.0, 1.0)
        # Keep both channels active: if any drive is requested, no channel
        # is allowed to fall below the floor. Prevents single-mode driving.
        if min_activation > 0.0 and np.any(u_base > 1e-6):
            u_cmd = np.maximum(u_cmd, min_activation)
        u = np.clip((1 - input_smoothing) * u_cmd + input_smoothing * u_prev, 0.0, 1.0)

        brain.next_state(u)
        u_prev = u

        y_log.append(y.copy())
        u_log.append(u.copy())
        ref_log.append(float(target))
        reach_log.append(float(reachable))
        a_log.append(float(a))
        if log_xhat:
            xhat_log.append(x_hat.copy())

    outputs = (np.array(y_log), np.array(u_log), np.array(ref_log),
               np.array(reach_log), np.array(a_log))
    if log_xhat:
        outputs = outputs + (np.array(xhat_log),)
    return outputs


def run_empirical_feedforward(brain, target_seq, records, reference_mode="interp"):
    y_log, u_log, reach_log, a_log = [], [], [], []
    for target in target_seq:
        if reference_mode == "nearest":
            _, u_base, reachable, a = choose_nearest_reference(records, target)
        else:
            _, u_base, reachable, a = choose_interpolated_reference(records, target)
        y = np.array(brain.measure(), float)
        brain.next_state(u_base)
        y_log.append(y.copy())
        u_log.append(u_base.copy())
        reach_log.append(float(reachable))
        a_log.append(float(a))
    return np.array(y_log), np.array(u_log), np.array(reach_log), np.array(a_log)


def evaluate_run(y_log, u_log, target_seq=None, target_norm=None, trim=0, tail=50):
    y_norm = np.linalg.norm(y_log, axis=1)
    u_use = u_log[trim:]
    y_use = y_norm[trim:]

    if target_seq is not None:
        target = np.asarray(target_seq, float)[trim:]
        tracking_error = float(np.mean(np.abs(y_use - target)))
        tail = min(tail, len(y_use))
        tail_error = float(np.mean(np.abs(y_use[-tail:] - target[-tail:])))
    elif target_norm is not None:
        tracking_error = float(np.mean(np.abs(y_use - target_norm)))
        tail = min(tail, len(y_use))
        tail_error = float(np.mean(np.abs(y_use[-tail:] - target_norm)))
    else:
        tracking_error = np.nan
        tail_error = np.nan

    return dict(
        steady_norm=float(np.mean(y_use[-min(tail, len(y_use)):])) if len(y_use) else np.nan,
        tracking_error=tracking_error,
        tail_tracking_error=tail_error,
        chatter=float(np.mean(np.abs(np.diff(u_use, axis=0)))) if len(u_use) > 1 else 0.0,
        saturation=float(np.mean((u_use <= 1e-6) | (u_use >= 1 - 1e-6))) if len(u_use) else 0.0,
    )


def step_responsiveness(y_norm, target_seq, trim=0, threshold=0.1):
    """
    Estimate step response delay and settling time for a two-level step target.
    Returns a list of events.
    """
    y = np.asarray(y_norm, float)
    target = np.asarray(target_seq, float)
    changes = np.where(np.abs(np.diff(target)) > 1e-9)[0] + 1
    events = []

    for c in changes:
        if c < trim:
            continue
        old = target[c - 1]
        new = target[c]
        amp = new - old
        if abs(amp) < 1e-9:
            continue

        direction = np.sign(amp)
        cross_level = old + 0.5 * amp
        tol = threshold * abs(amp)

        delay = np.nan
        settle = np.nan
        for t in range(c, len(y)):
            if direction * (y[t] - cross_level) >= 0:
                delay = t - c
                break

        for t in range(c, len(y)):
            # settled if the next 8 points are within tolerance
            end = min(t + 8, len(y))
            if np.all(np.abs(y[t:end] - new) <= tol):
                settle = t - c
                break

        events.append(dict(change_time=int(c), old=float(old), new=float(new),
                           delay_steps=float(delay), settling_steps=float(settle)))
    return events


def sine_lag_metrics(y_norm, target_seq, trim=0, max_lag=25):
    """
    Estimate lag, correlation, amplitude ratio for sine tracking.
    Positive lag means actual lags target.
    """
    y = np.asarray(y_norm, float)[trim:]
    target = np.asarray(target_seq, float)[trim:]
    y0 = y - np.mean(y)
    t0 = target - np.mean(target)

    best_lag = 0
    best_corr = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            yy = y0[-lag:]
            tt = t0[:len(yy)]
        elif lag > 0:
            tt = t0[lag:]
            yy = y0[:len(tt)]
        else:
            yy = y0
            tt = t0
        if len(yy) < 5 or np.std(yy) < 1e-9 or np.std(tt) < 1e-9:
            continue
        corr = np.corrcoef(tt, yy)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    amp_ratio = (np.std(y0) / (np.std(t0) + 1e-12))
    return dict(lag_steps=int(best_lag), corr=float(best_corr), amplitude_ratio=float(amp_ratio))


def channel_usage(u_log):
    """
    Quantify how much each input channel is actually used, to verify we are
    NOT driving the system through a single mode.

    Returns:
        per_channel_mean : mean drive per channel
        per_channel_frac : fraction of total drive carried by each channel
        balance          : min(frac)/max(frac) in [0,1]; 1.0 = perfectly
                           balanced, near 0 = effectively single-channel
    """
    u = np.asarray(u_log, float)
    means = u.mean(axis=0)
    total = means.sum() + 1e-12
    frac = means / total
    balance = float(frac.min() / (frac.max() + 1e-12))
    return dict(per_channel_mean=means, per_channel_frac=frac, balance=balance)


def run_open_loop_constant(brain, u_const, n_steps):
    """
    Apply a constant input on the continuous brain for n_steps and log y.
    Used for the uncontrolled / natural-level baseline. No reset.
    """
    u_const = np.asarray(u_const, float)
    y_log = []
    for _ in range(n_steps):
        y_log.append(np.array(brain.measure(), float))
        brain.next_state(np.clip(u_const, 0.0, 1.0))
    return np.array(y_log)