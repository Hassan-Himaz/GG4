from typing import Tuple
import numpy as np


# Estimator.py
# GG4 Week 2: Individual Estimation
#
# IMPORTANT:
# Do NOT change the signature of estimate_latent_and_input.
#
#     estimate_latent_and_input(observation, LatentDim, InputDim)
#
# expects:
#     latent_states.shape == (Timepoints, LatentDim)
#     inputs.shape        == (Timepoints, InputDim)


def _as_2d_float_array(observation: np.ndarray) -> np.ndarray:
    """把 observation 转成有限值的 2D float 数组，形状 (Timepoints, Neurons)。"""
    # 把输入强制转成 numpy 的浮点数组
    Y = np.asarray(observation, dtype=float)
    # Y不是2维的情况：
    if Y.ndim != 2:
        raise ValueError(
            f"observation must be 2D (Timepoints, Neurons), got shape {Y.shape}."
        )
    if Y.shape[0] <= 0:
        raise ValueError("observation must contain at least one timepoint.")
    if Y.shape[1] <= 0:
        raise ValueError("observation must contain at least one neuron/signal.")

    # 用列均值替换 NaN/Inf
    if not np.all(np.isfinite(Y)):
        col_means = np.nanmean(np.where(np.isfinite(Y), Y, np.nan), axis=0)
        # 极端情况:如果某一列全是 NaN,上一步算出的均值也会是 NaN。这里把它们替换成 0。
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        bad = ~np.isfinite(Y)
        Y = Y.copy()
        rows, cols = np.where(bad)
        # 把这些坏位置,替换成它所在列的均值
        Y[rows, cols] = col_means[cols]

    return Y


def _standardize_columns(Y: np.ndarray, eps: float = 1e-8):
    """每列标准化为均值 0、标准差 1，避免大尺度通道主导 PCA。"""
    mean = np.mean(Y, axis=0, keepdims=True)
    std = np.maximum(np.std(Y, axis=0, keepdims=True), eps)
    # 每列均值 0、std 1。
    return (Y - mean) / std, mean, std


def _pca_scores(Y: np.ndarray, dim: int):
    """用 SVD 做 PCA，返回前 dim 维得分 (Timepoints, dim)。不足时补零。"""
    if dim <= 0:
        raise ValueError("dim must be positive.")
    # time, neurons
    T, D = Y.shape
    Y_centered = Y - np.mean(Y, axis=0, keepdims=True)

    # 退化情况：数据完全是常数(中心化后全是 0)
    if np.sum(Y_centered ** 2) <= 1e-14:
        return np.zeros((T, dim)), np.zeros((0, D)), np.zeros(min(T, D))
    # SVD
    U, S, Vt = np.linalg.svd(Y_centered, full_matrices=False)
    explained_variance = (S ** 2) / np.sum(S ** 2)
    #  SVD 最多能给
    actual_dim = min(dim, U.shape[1])
    scores = U[:, :actual_dim] * S[:actual_dim]
    components = Vt[:actual_dim]

    # LatentDim 比 neuron 还多的情况
    if actual_dim < dim:
        scores = np.hstack([scores, np.zeros((T, dim - actual_dim))])

    return scores, components, explained_variance


def _fix_pca_signs(Z: np.ndarray) -> np.ndarray:
    """让每个分量的最大幅值元素为正，使 PCA/SVD 的符号确定、可复现。"""
    # PCA 的方向问题：一个方向朝上还是朝下,数学上都一样(整列乘 -1 不改变信息)
    Z = Z.copy()
    for j in range(Z.shape[1]):
        if np.allclose(Z[:, j], 0):
            continue
        idx = np.argmax(np.abs(Z[:, j]))
        # 负的就翻转一下
        if Z[idx, j] < 0:
            Z[:, j] *= -1
    return Z


def _normalize_columns(Z: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """每列除以自身标准差，归到约单位方差，让后续输出尺度稳定。"""
    std = np.maximum(np.std(Z, axis=0, keepdims=True), eps)
    return Z / std



# 稳定 A —— A_new = α A_old + (1-α) A_ref

def _stabilize_A(A: np.ndarray, rho_max: float = 0.98) -> np.ndarray:
    """
    如果 A 的谱半径 (最大特征值模) 超过 rho_max，就整体收缩 A。

    least squares 估出的 A 不保证稳定。若有特征值 > 1，latent
    dynamics 会发散，模型解释变差。这里做: spectral radius
    shrinkage，把 A 缩到 rho_max 以内，保证 latent 动力学稳定。
    """
    # 求 eigenvalues
    eigvals = np.linalg.eigvals(A)
    # 最大特征值magnitude
    rho = np.max(np.abs(eigvals))
    # 最大0.98
    if rho > rho_max:
        A = A * (rho_max / rho)
    return np.real_if_close(A)



# 从 (Z, Y) 估计 A, C, Q, R

def _identify_system(Z: np.ndarray, Y_std: np.ndarray, ridge: float = 1e-4):
    """
    从数据估计线性状态空间模型参数 (system identification)。

    模型 (高斯噪声):
        z[t+1] = A z[t] + w[t],  w ~ N(0, Q)
        y[t]   = C z[t] + v[t],  v ~ N(0, R)

    返回 A_hat (n,n), C_hat (p,n), Q_hat (n,n), R_hat (p,p)。
    """
    T, n = Z.shape
    _, p = Y_std.shape

    # A: 最小二乘 z[t+1] ≈ A z[t]
    if T >= 2:
        # Z[:-1] 是"去掉最后一行"(每个"当前时刻"),Z[1:] 是"去掉第一行"(每个"下一时刻")。两者配成"今天→明天"的对子,共 T-1 对。
        Z_prev, Z_next = Z[:-1], Z[1:]
        # 解 Z_prev · A^T ≈ Z_next
        lhs = Z_prev.T @ Z_prev + ridge * np.eye(n)
        rhs = Z_prev.T @ Z_next
        try:
            A_hat = np.linalg.solve(lhs, rhs).T
        except np.linalg.LinAlgError:
            A_hat = np.eye(n)
    else:
        A_hat = np.eye(n)
    # 无论怎样，稳定化
    A_hat = _stabilize_A(A_hat, rho_max=0.98)  # 改动(2)

    # Q: 动力学残差的协方差（A 解释不了的过程噪声）
    if T >= 2:
        # 残差 = 真实的下一刻 − A 预测的下一刻。
        residuals_dyn = Z_next - Z_prev @ A_hat.T
        # 残差的协方差矩阵
        Q_hat = (residuals_dyn.T @ residuals_dyn) / (T - 1)
        Q_hat = 0.5 * (Q_hat + Q_hat.T) + 1e-4 * np.eye(n)  # 对称 + 正定
    else:
        Q_hat = np.eye(n) * 0.1

    # C: 最小二乘 y[t] ≈ C z[t]
    lhs_c = Z.T @ Z + ridge * np.eye(n)
    rhs_c = Z.T @ Y_std
    try:
        C_hat = np.linalg.solve(lhs_c, rhs_c).T
    except np.linalg.LinAlgError:
        C_hat = np.zeros((p, n))

    # R: 观测残差的协方差，取对角（假设各通道独立，降复杂度）
    obs_residuals = Y_std - Z @ C_hat.T
    R_hat = (obs_residuals.T @ obs_residuals) / T
    R_diag = np.maximum(np.diag(0.5 * (R_hat + R_hat.T)), 1e-4)
    R_hat = np.diag(R_diag)

    return A_hat, C_hat, Q_hat, R_hat


#暂时不用
def _kalman_filter(Y_std, A, C, Q, R, x0, P0):
    """
    标准线性 Kalman Filter。先不用你了。

    关键修复: x0 = Z_init[0] 已经代表 t=0 的 latent state，所以
      - t == 0: 不做 A 预测，直接用 x0 / P0 进入 update
      - t >= 1: 才做 predict (x_pred = A x，P_pred = A P A^T + Q)
    这样 innovation = y[t] - C x_pred 在时间含义上才是对齐的！！！

    返回 filtered_states (T,n), innovations (T,p)。
    """
    # 初始化
    T, p = Y_std.shape
    n = A.shape[0]
    I_n = np.eye(n)

    filtered_states = np.zeros((T, n))
    innovations = np.zeros((T, p))

    x = x0.copy()
    P = P0.copy()

    for t in range(T):
        # ── 预测步 (t=0 跳过，避免 off-by-one) ──
        if t == 0:
            x_pred = x.copy()
            P_pred = P.copy()
        else:
            x_pred = A @ x
            P_pred = A @ P @ A.T + Q
            P_pred = 0.5 * (P_pred + P_pred.T)

        # innovation = 实际观测 − 根据预测应该看到的观测
        innovation = Y_std[t] - C @ x_pred        # 创新 = 实际 - 预测
        S = C @ P_pred @ C.T + R
        S = 0.5 * (S + S.T)

        # 观测越可靠(R 小),K 越大,越信观测;模型越可靠(Q 小),K 越小,越信预测。
        try:                                       # K = P_pred C^T inv(S)
            K = np.linalg.solve(S.T, (P_pred @ C.T).T).T
        except np.linalg.LinAlgError:
            K = P_pred @ C.T @ np.linalg.pinv(S)

        # 在预测的基础上,根据创新做修正。
        x = x_pred + K @ innovation

        # 协方差更新用 Joseph 形式??? ：P = (I-KC) P_pred
        IKC = I_n - K @ C
        P = IKC @ P_pred @ IKC.T + K @ R @ K.T
        P = 0.5 * (P + P.T) + 1e-8 * I_n

        filtered_states[t] = x
        innovations[t] = innovation

    return filtered_states, innovations


#  Kalman filter + RTS smoother
#  - filter 只用 "过去到当前" 的信息估每个 t 的状态
#  - smoother 再从最后一个时间点倒着走,把 "未来信息" 回传给前面

def _kalman_filter_with_covs(Y_std, A, C, Q, R, x0, P0):
    """
    标准 Kalman filter,但额外返回每步的 P_pred 和 P_filt,
    供后续 smoother 使用。其余逻辑与 _kalman_filter 完全相同
    """
    T, p = Y_std.shape
    n = A.shape[0]
    I_n = np.eye(n)

    x_filt = np.zeros((T, n))
    P_filt = np.zeros((T, n, n))
    x_pred_arr = np.zeros((T, n))
    P_pred_arr = np.zeros((T, n, n))
    innovations = np.zeros((T, p))

    x = x0.copy()
    P = P0.copy()

    for t in range(T):
        if t == 0:
            x_pred = x.copy()
            P_pred = P.copy()
        else:
            x_pred = A @ x
            P_pred = A @ P @ A.T + Q
            P_pred = 0.5 * (P_pred + P_pred.T)

        # filter 自己往前走的时候顺便把每一步的预测和滤波结果都录下来,给 smoother 回放用。牛的牛的
        x_pred_arr[t] = x_pred
        P_pred_arr[t] = P_pred

        innovation = Y_std[t] - C @ x_pred
        S = C @ P_pred @ C.T + R
        S = 0.5 * (S + S.T)
        try:
            K = np.linalg.solve(S.T, (P_pred @ C.T).T).T
        except np.linalg.LinAlgError:
            K = P_pred @ C.T @ np.linalg.pinv(S)
        x = x_pred + K @ innovation
        IKC = I_n - K @ C
        P = IKC @ P_pred @ IKC.T + K @ R @ K.T
        P = 0.5 * (P + P.T) + 1e-8 * I_n

        x_filt[t] = x
        P_filt[t] = P
        innovations[t] = innovation

    return x_filt, P_filt, x_pred_arr, P_pred_arr, innovations


def _rts_smoother(x_filt, P_filt, x_pred_arr, P_pred_arr, A):
    """
    RTS (Rauch-Tung-Striebel) 后向递推。

    对 t = T-2, T-3, ..., 0 倒着走,每一步把未来的信息回传:
        J_t = P_filt[t] A^T inv(P_pred[t+1])
        x_smooth[t] = x_filt[t] + J_t (x_smooth[t+1] - x_pred[t+1])
        P_smooth[t] = P_filt[t] + J_t (P_smooth[t+1] - P_pred[t+1]) J_t^T

    J_t 是"未来信息回传强度"。最后一个时间点没有未来,所以
    x_smooth[T-1] = x_filt[T-1] 作为初值。

    返回 x_smooth (T, n)。P_smooth 不返回
    """
    # x_smooth[T-1] = x_filt[T-1] 直接相等
    T, n = x_filt.shape
    x_smooth = x_filt.copy()
    P_smooth_next = P_filt[-1].copy()
    # 倒着走！
    for t in range(T - 2, -1, -1):
        # 取出"t+1 时刻的预测covariance"(从 filter 的录像里拿)。
        P_pred_next = P_pred_arr[t + 1]
        # 既然现在我知道了 t+1 时刻更准确的真相(x_smooth[t+1]),那这个'修正'按多大比例传回 t 时刻
        try:
            J = np.linalg.solve(P_pred_next.T, (P_filt[t] @ A.T).T).T
        except np.linalg.LinAlgError:
            J = P_filt[t] @ A.T @ np.linalg.pinv(P_pred_next)
        # x_pred_arr[t+1]:filter 在 t 时刻向前预测的 t+1 时刻状态
        x_smooth[t] = x_filt[t] + J @ (x_smooth[t + 1] - x_pred_arr[t + 1])
        P_smooth_t = P_filt[t] + J @ (P_smooth_next - P_pred_next) @ J.T
        P_smooth_next = 0.5 * (P_smooth_t + P_smooth_t.T)

    return x_smooth


def _estimate_inputs_from_dynamics(X: np.ndarray, A: np.ndarray, InputDim: int) -> np.ndarray:
    """
    从 latent 动力学残差估计输入信号。

    模型: x[t+1] = A x[t] + B u[t] + noise
    B 未知，所以取残差 (x[t+1] - A x[t]) 的主 PCA 方向作为
    最可能的低维 input 子空间。这比用 observation innovation 更贴
    板书: u 推动状态转移，而不是直接推动观测。

    残差只存在于 t=0..T-2，所以第一个时间点补零对齐。
    """
    if InputDim <= 0:
        raise ValueError("InputDim must be positive.")

    T, n = X.shape
    inputs = np.zeros((T, InputDim))
    if T < 2:
        return inputs
    # x[t+1]-A·x[t]
    residuals = X[1:] - X[:-1] @ A.T              # (T-1, n)
    if np.allclose(residuals, 0):
        return inputs
    
    # 对残差做标准化 + PCA
    residuals_std, _, _ = _standardize_columns(residuals)
    u_scores, _, _ = _pca_scores(residuals_std, InputDim)
    u_scores = _fix_pca_signs(u_scores)
    u_scores = _normalize_columns(u_scores)

    inputs[1:] = u_scores
    return inputs



# 主接口

def estimate_latent_and_input(
    observation: np.ndarray,
    LatentDim: int,
    InputDim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Call format:
        latent_states, inputs = estimate_latent_and_input(observation, LatentDim, InputDim)

    Test example:
        Y = np.random.randn(100, 20)
        latent_states, inputs = estimate_latent_and_input(Y, 3, 2)

    Expected output:
        latent_states.shape == (100, 3)
        inputs.shape == (100, 2)
        all values in both arrays are finite.

    Here observation must have shape (T, N), where T is the number of timepoints
    and N is the number of observed neurons/signals. The function should also be
    tested on edge cases such as T=1, T=2, constant observations, NaN/Inf values,
    and LatentDim greater than the number of observed signals.
    """
    Y = _as_2d_float_array(observation)

    if not isinstance(LatentDim, int) or LatentDim <= 0:
        raise ValueError("LatentDim must be a positive integer.")
    if not isinstance(InputDim, int) or InputDim <= 0:
        raise ValueError("InputDim must be a positive integer.")

    Timepoints = Y.shape[0]

    # ── Step 1: 标准化观测 ──
    Y_std, _, _ = _standardize_columns(Y)

    # ── Step 2: PCA 初始 latent 轨迹（仅作迭代的种子）──
    Z = _pca_scores(Y_std, LatentDim)[0]
    Z = _fix_pca_signs(Z)
    Z = _normalize_columns(Z)

    # 初始不确定性 P0：用 Z 的样本协方差，让 KF 快速收敛
    if Timepoints > 1:
        P0 = np.atleast_2d(np.cov(Z.T))
        if P0.shape != (LatentDim, LatentDim):
            P0 = np.eye(LatentDim)
        P0 = 0.5 * (P0 + P0.T) + 1e-4 * np.eye(LatentDim)
    else:
        P0 = np.eye(LatentDim)

    # Step 3: EM 迭代,用 filter + smoother estimate states
    # 重要！只在循环后做 _fix_pca_signs / _normalize_columns
    n_iters = 3
    innovations = np.zeros((Timepoints, Y_std.shape[1]))
    A_hat = np.eye(LatentDim)

    for _ in range(n_iters):
        # 给定当前 Z，估计参数
        A_hat, C_hat, Q_hat, R_hat = _identify_system(Z, Y_std)
        # M-step 的对偶: 给定参数，用 filter + smoother 重估 Z
        x_filt, P_filt, x_pred_arr, P_pred_arr, innovations = \
            _kalman_filter_with_covs(Y_std, A_hat, C_hat, Q_hat, R_hat, Z[0], P0)
        Z = _rts_smoother(x_filt, P_filt, x_pred_arr, P_pred_arr, A_hat)

    # ── Step 4: 后处理 latent（只此一次）──
    latent_states = _fix_pca_signs(Z)
    latent_states = _normalize_columns(latent_states)

    # ── Step 5: 从 latent dynamics residual 估计 input ──
    # 注意用规整后的 latent_states，与最终输出一致
    inputs = _estimate_inputs_from_dynamics(latent_states, A_hat, InputDim)

    # ── 接口检查 ──
    if latent_states.shape != (Timepoints, LatentDim):
        raise RuntimeError(
            f"latent_states has shape {latent_states.shape}, "
            f"expected {(Timepoints, LatentDim)}."
        )
    if inputs.shape != (Timepoints, InputDim):
        raise RuntimeError(
            f"inputs has shape {inputs.shape}, expected {(Timepoints, InputDim)}."
        )

    return latent_states, inputs