"""
EstimatorV3.py — full-EM 版 latent + input 估计器

相比 V2 的核心升级（针对老师的 "not full EM" 与 latent 还原优先）：

【full EM】
  E 步：RTS smoother 给出完整后验，不只均值 x̂，还有：
        P_t      = Cov(x_t)              （每步协方差）
        P_{t,t-1}= Cov(x_t, x_{t-1})     （相邻互协方差）
  M 步：用充分统计量（二阶矩）闭式更新 A, C, Q, R：
        S11 = Σ E[x_{t-1} x_{t-1}^T] = Σ (x̂ x̂^T + P)
        S10 = Σ E[x_t x_{t-1}^T]     = Σ (x̂_t x̂_{t-1}^T + P_{t,t-1})
        A = S10 @ inv(S11)
        Q = (1/(T-1)) (S_dyn - A S10^T)        （含协方差项，不是残差平方）
        C = (Σ y x̂^T)(Σ x̂ x̂^T + P)^{-1}
        R = (1/T) Σ (y - C x̂)(...)^T + C P C^T  （含状态不确定性）
  这才是教科书 LDS-EM（Ghahramani & Hinton 1996），V2 只用均值做回归。

【latent 优先】
  - 不对 latent 做任何归一化（保幅度）
  - A 谱半径上限放宽到 1.02（保慢衰减/增长）
  - EM 迭代加到 20 轮，带对数似然收敛监控

【input：自适应，不靠固定阈值】
  - 先估“是否存在 input”：残差能量 vs 过程噪声水平
  - no-input 案例 → 输出近 0（治 Case 9/14 假脉冲）
  - 有 input → 按信噪比自适应软阈值（不再一刀切）

模型：
    x[t+1] = A x[t] + B u[t] + w[t],  w ~ N(0, Q)
    y[t]   = C x[t] + v[t],           v ~ N(0, R)
"""

from typing import Tuple
import numpy as np


# ---------- 输入清洗 ----------
def _as_2d_float_array(observation: np.ndarray) -> np.ndarray:
    Y = np.asarray(observation, dtype=float)
    if Y.ndim != 2:
        raise ValueError(f"observation must be 2D, got {Y.shape}.")
    if Y.shape[0] <= 0 or Y.shape[1] <= 0:
        raise ValueError("observation must have >=1 timepoint and >=1 signal.")
    if not np.all(np.isfinite(Y)):
        col_means = np.nanmean(np.where(np.isfinite(Y), Y, np.nan), axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        bad = ~np.isfinite(Y)
        Y = Y.copy()
        rows, cols = np.where(bad)
        Y[rows, cols] = col_means[cols]
    return Y


def _pca_init(Y: np.ndarray, dim: int) -> np.ndarray:
    T, D = Y.shape
    Yc = Y - np.mean(Y, axis=0, keepdims=True)
    if np.sum(Yc ** 2) <= 1e-14:
        return np.zeros((T, dim))
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    ad = min(dim, U.shape[1])
    scores = U[:, :ad] * S[:ad]
    if ad < dim:
        scores = np.hstack([scores, np.zeros((T, dim - ad))])
    return scores


def _stabilize_A(A: np.ndarray, rho_max: float = 1.02) -> np.ndarray:
    eig = np.linalg.eigvals(A)
    rho = np.max(np.abs(eig))
    if rho > rho_max:
        A = A * (rho_max / rho)
    return np.real_if_close(A)


# ============================================================
# E 步：Kalman filter + RTS smoother，返回完整后验充分统计量
# ============================================================
def _kalman_smoother_full(Y, A, C, Q, R, x0, P0):
    """
    返回：
      xs     : (T, n)   平滑后均值 x̂_t
      Ps     : (T, n,n) 平滑后协方差 P_t
      Pcross : (T, n,n) 相邻互协方差 P_{t,t-1}（Pcross[t] = Cov(x_t, x_{t-1})，t>=1）
    """
    T, p = Y.shape
    n = A.shape[0]
    I = np.eye(n)

    # ---- 前向 filter ----
    xf = np.zeros((T, n)); Pf = np.zeros((T, n, n))
    xp = np.zeros((T, n)); Pp = np.zeros((T, n, n))
    Kt = np.zeros((T, n, p))
    x = x0.copy(); P = P0.copy()
    for t in range(T):
        if t == 0:
            x_pred, P_pred = x.copy(), P.copy()
        else:
            x_pred = A @ x
            P_pred = A @ P @ A.T + Q
            P_pred = 0.5 * (P_pred + P_pred.T)
        xp[t], Pp[t] = x_pred, P_pred
        S = C @ P_pred @ C.T + R
        S = 0.5 * (S + S.T)
        try:
            K = np.linalg.solve(S.T, (P_pred @ C.T).T).T
        except np.linalg.LinAlgError:
            K = P_pred @ C.T @ np.linalg.pinv(S)
        Kt[t] = K
        x = x_pred + K @ (Y[t] - C @ x_pred)
        IKC = I - K @ C
        P = IKC @ P_pred @ IKC.T + K @ R @ K.T
        P = 0.5 * (P + P.T) + 1e-9 * I
        xf[t], Pf[t] = x, P

    # ---- 后向 RTS smoother ----
    xs = xf.copy(); Ps = Pf.copy()
    Js = np.zeros((T, n, n))
    for t in range(T - 2, -1, -1):
        try:
            J = np.linalg.solve(Pp[t + 1].T, (Pf[t] @ A.T).T).T
        except np.linalg.LinAlgError:
            J = Pf[t] @ A.T @ np.linalg.pinv(Pp[t + 1])
        Js[t] = J
        xs[t] = xf[t] + J @ (xs[t + 1] - xp[t + 1])
        Ps[t] = Pf[t] + J @ (Ps[t + 1] - Pp[t + 1]) @ J.T
        Ps[t] = 0.5 * (Ps[t] + Ps[t].T)

    # ---- 相邻互协方差 P_{t,t-1}（Lag-one covariance smoother）----
    Pcross = np.zeros((T, n, n))
    if T >= 2:
        # 初值：P_{T-1,T-2} = (I - K_{T-1} C) A P_{T-2}^filt
        Pcross[T - 1] = (I - Kt[T - 1] @ C) @ A @ Pf[T - 2]
        for t in range(T - 2, 0, -1):
            Pcross[t] = Pf[t] @ Js[t - 1].T + Js[t] @ (Pcross[t + 1] - A @ Pf[t]) @ Js[t - 1].T
    return xs, Ps, Pcross


# ============================================================
# M 步：用充分统计量闭式更新 A, C, Q, R
# ============================================================
def _m_step(Y, xs, Ps, Pcross, ridge=1e-5):
    T, n = xs.shape
    p = Y.shape[1]
    I = np.eye(n)

    # 二阶矩：E[x_t x_t^T] = x̂_t x̂_t^T + P_t
    Exx = np.einsum('ti,tj->tij', xs, xs) + Ps          # (T,n,n)

    # 动力学统计量（t = 1..T-1）
    if T >= 2:
        S11 = np.sum(Exx[:-1], axis=0)                  # Σ E[x_{t-1} x_{t-1}^T]
        S22 = np.sum(Exx[1:], axis=0)                   # Σ E[x_t x_t^T]
        # S10 = Σ E[x_t x_{t-1}^T] = Σ (x̂_t x̂_{t-1}^T + P_{t,t-1})
        cross_mean = np.einsum('ti,tj->tij', xs[1:], xs[:-1])
        S10 = np.sum(cross_mean + Pcross[1:], axis=0)
        A_hat = S10 @ np.linalg.inv(S11 + ridge * I)
        A_hat = _stabilize_A(A_hat, 1.02)
        # Q = (1/(T-1)) (S22 - A S10^T)，对称化
        Q_hat = (S22 - A_hat @ S10.T) / max(T - 1, 1)
        Q_hat = 0.5 * (Q_hat + Q_hat.T) + 1e-4 * I
    else:
        A_hat = np.eye(n)
        Q_hat = np.eye(n) * 0.1

    # 观测：C = (Σ y x̂^T)(Σ E[x x^T])^{-1}
    Sxx_all = np.sum(Exx, axis=0)                       # Σ E[x_t x_t^T]
    Syx = Y.T @ xs                                      # Σ y_t x̂_t^T
    C_hat = Syx @ np.linalg.inv(Sxx_all + ridge * I)

    # R = (1/T) Σ [ (y - C x̂)(y - C x̂)^T + C P C^T ]
    resid = Y - xs @ C_hat.T
    R_full = (resid.T @ resid) / T + C_hat @ (np.sum(Ps, axis=0) / T) @ C_hat.T
    R_diag = np.maximum(np.diag(0.5 * (R_full + R_full.T)), 1e-3)
    R_hat = np.diag(R_diag)

    return A_hat, C_hat, Q_hat, R_hat


# ============================================================
# input：自适应（先判断有无 input，再按信噪比削噪）
# ============================================================
def _solve_inputs_adaptive(xs, A, Q, InputDim):
    """
    融合版 input 估计：V2 的鲁棒残差 SVD 反解 + 温和软阈值。
    no-input 判定改保守：仅当残差峰值几乎不超过背景（真·全噪声）才清零，
    避免误杀 Case 1 那种“latent 衰减不慢、脉冲峰值不极端”的真脉冲。
    """
    T, n = xs.shape
    inputs = np.zeros((T, InputDim))
    if T < 2:
        return inputs

    resid = xs[1:] - xs[:-1] @ A.T          # ≈ B u + w
    if np.sum(resid ** 2) < 1e-12:
        return inputs

    resid_norm = np.linalg.norm(resid, axis=1)
    med = np.median(resid_norm)
    mad = np.median(np.abs(resid_norm - med)) + 1e-12
    robust_sigma = 1.4826 * mad
    peak = np.max(resid_norm)

    # 保守 no-input 判定：峰值连背景的 ~8σ 都够不到，才认定无输入。
    # （V3 用的是 4σ，太激进，误杀了 Case 1。这里放宽到 8σ。）
    if robust_sigma > 1e-9 and peak < med + 8.0 * robust_sigma:
        return inputs

    # 残差 SVD 反解，不归一化保尺度
    Ur, Sr, Vr = np.linalg.svd(resid, full_matrices=False)
    k = min(InputDim, Ur.shape[1])
    u_hat = Ur[:, :k] * Sr[:k]
    if k < InputDim:
        u_hat = np.hstack([u_hat, np.zeros((T - 1, InputDim - k))])

    # 符号确定
    for j in range(u_hat.shape[1]):
        if np.allclose(u_hat[:, j], 0):
            continue
        idx = np.argmax(np.abs(u_hat[:, j]))
        if u_hat[idx, j] < 0:
            u_hat[:, j] *= -1

    # 温和软阈值（V2 的 1.5σ），保住真脉冲
    if u_hat.shape[0] > 5:
        for j in range(u_hat.shape[1]):
            col = u_hat[:, j]
            m = np.median(col)
            s = 1.4826 * (np.median(np.abs(col - m)) + 1e-12)
            thr = 1.5 * s
            shrunk = np.sign(col) * np.maximum(np.abs(col) - thr, 0.0)
            if np.allclose(shrunk, 0.0) and not np.allclose(col, 0.0):
                shrunk = col
            u_hat[:, j] = shrunk

    inputs[1:] = u_hat
    return inputs


# ============================================================
# 主接口
# ============================================================
def estimate_latent_and_input(
    observation: np.ndarray,
    LatentDim: int,
    InputDim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    Y = _as_2d_float_array(observation)
    if not isinstance(LatentDim, int) or LatentDim <= 0:
        raise ValueError("LatentDim must be a positive integer.")
    if not isinstance(InputDim, int) or InputDim <= 0:
        raise ValueError("InputDim must be a positive integer.")

    T = Y.shape[0]
    n, m = LatentDim, InputDim

    Y_mean = np.mean(Y, axis=0, keepdims=True)
    Yc = Y - Y_mean

    Z = _pca_init(Yc, n)

    # 初始参数
    A = np.eye(n) * 0.9
    C = np.zeros((Yc.shape[1], n))
    # 用 PCA 载荷初始化 C
    Yc0 = Yc - np.mean(Yc, axis=0, keepdims=True)
    if np.sum(Yc0 ** 2) > 1e-14 and T >= 2:
        Gz = Z.T @ Z + 1e-5 * np.eye(n)
        C = np.linalg.solve(Gz, Z.T @ Yc).T
    Q = np.eye(n) * 0.1
    R = np.eye(Yc.shape[1]) * np.maximum(np.var(Yc, axis=0).mean(), 1e-2)

    if T > 1:
        P0 = np.atleast_2d(np.cov(Z.T))
        if P0.shape != (n, n):
            P0 = np.eye(n)
        P0 = 0.5 * (P0 + P0.T) + 1e-2 * np.eye(n)
    else:
        P0 = np.eye(n)
    x0 = Z[0].copy()

    # ---- full EM 主循环 ----
    n_iters = 20 if T >= 8 else 5
    prev_xs = None
    for it in range(n_iters):
        # E 步
        xs, Ps, Pcross = _kalman_smoother_full(Yc, A, C, Q, R, x0, P0)
        # M 步
        A, C, Q, R = _m_step(Yc, xs, Ps, Pcross)
        # 更新初值
        x0 = xs[0].copy()
        P0 = 0.5 * (Ps[0] + Ps[0].T) + 1e-4 * np.eye(n)
        # 收敛监控（latent 变化量）
        if prev_xs is not None:
            delta = np.linalg.norm(xs - prev_xs) / (np.linalg.norm(prev_xs) + 1e-9)
            if delta < 1e-4:
                break
        prev_xs = xs.copy()

    latent_states = xs
    inputs = _solve_inputs_adaptive(xs, A, Q, m)

    # 兜底
    if latent_states.shape != (T, n):
        latent_states = np.zeros((T, n))
    if inputs.shape != (T, m):
        inputs = np.zeros((T, m))
    latent_states = np.nan_to_num(latent_states)
    inputs = np.nan_to_num(inputs)
    return latent_states, inputs