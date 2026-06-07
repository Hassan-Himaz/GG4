"""
============================================================
test_estimator.py  —  Estimator 验证脚本 (notebook 风格)
GG4 Week 2

两大部分:
  PART A. Sanity checks  —— 保证不报错、shape 对、边界情况稳健
  PART B. Recovery quality —— 在已知 ground-truth 的合成数据上,
          量化能多准地找回 latent states 和 input,并画图。

用法:
  python test_estimator.py          # 跑全部,存图到 ./figures/
  也可以一段段复制到 Jupyter notebook 里跑。
============================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")            # 无显示环境也能存图; 在 notebook 里可删掉这行
import matplotlib.pyplot as plt

from Estimator import estimate_latent_and_input

RNG = np.random.default_rng(0)
FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)


# ============================================================
# 辅助函数
# ============================================================
def make_synthetic(T=300, N=40, n=3, m=2, noise=0.1, seed=0,
                   rho=0.95, input_strength=0.4):
    """
    生成一个带 ground-truth 的线性状态空间数据集。

        x[t+1] = A x[t] + B u[t] + 过程噪声
        y[t]   = C x[t] + 观测噪声

    返回 Y(观测), x_true(真实 latent), u_true(真实 input),
    以及真实参数 A,B,C —— 用于后面对比恢复质量。
    """
    rng = np.random.default_rng(seed)

    # 造一个稳定、带旋转的 A (谱半径 ~ rho),让 latent 有真实动力学
    theta = 0.3
    blocks = []
    for k in range(n // 2):
        blocks.append(rho * np.array([[np.cos(theta), -np.sin(theta)],
                                      [np.sin(theta),  np.cos(theta)]]))
    A = np.zeros((n, n))
    i = 0
    for b in blocks:
        A[i:i+2, i:i+2] = b
        i += 2
    if n % 2 == 1:                      # 奇数维补一个标量衰减
        A[-1, -1] = rho

    B = rng.standard_normal((n, m)) * input_strength
    C = rng.standard_normal((N, n))

    x = np.zeros((T, n))
    x[0] = rng.standard_normal(n)
    u = rng.standard_normal((T, m))     # 随机 input
    for t in range(1, T):
        x[t] = A @ x[t-1] + B @ u[t] + 0.02 * rng.standard_normal(n)

    Y = x @ C.T + noise * rng.standard_normal((T, N))
    return Y, x, u, A, B, C


def aligned_corr(est, true):
    """
    估计的子空间和真实的之间,符号/顺序/旋转都可能不同。
    用线性回归把 est 对齐到 true 所在子空间,再算每个真实维度的 R^2。
    返回 (mean_R2, per_dim_R2, est_aligned)。
    """
    # 最小二乘: 找 W 使 est @ W ≈ true
    W, *_ = np.linalg.lstsq(est, true, rcond=None)
    est_aligned = est @ W
    r2 = []
    for j in range(true.shape[1]):
        ss_res = np.sum((true[:, j] - est_aligned[:, j]) ** 2)
        ss_tot = np.sum((true[:, j] - true[:, j].mean()) ** 2)
        r2.append(1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan)
    r2 = np.array(r2)
    return np.nanmean(r2), r2, est_aligned


# ============================================================
# PART A. SANITY CHECKS
# ============================================================
def part_a_sanity():
    print("=" * 60)
    print("PART A. SANITY CHECKS")
    print("=" * 60)
    results = []

    def check(name, fn):
        try:
            fn()
            print(f"  [PASS] {name}")
            results.append((name, True, ""))
        except Exception as e:
            print(f"  [FAIL] {name}  ->  {type(e).__name__}: {e}")
            results.append((name, False, str(e)))

    Y, *_ = make_synthetic()

    # A1. 正常输入: shape 正确
    def a1():
        lat, inp = estimate_latent_and_input(Y, 3, 2)
        assert lat.shape == (Y.shape[0], 3), lat.shape
        assert inp.shape == (Y.shape[0], 2), inp.shape
    check("A1  normal input -> correct shapes", a1)

    # A2. 输出全部有限 (无 NaN/Inf)
    def a2():
        lat, inp = estimate_latent_and_input(Y, 3, 2)
        assert np.all(np.isfinite(lat)) and np.all(np.isfinite(inp))
    check("A2  outputs are all finite", a2)

    # A3. 单时间点
    def a3():
        lat, inp = estimate_latent_and_input(Y[:1], 3, 2)
        assert lat.shape == (1, 3) and inp.shape == (1, 2)
    check("A3  single timepoint (T=1)", a3)

    # A4. 两个时间点 (T=2,刚好够算一次 transition)
    def a4():
        lat, inp = estimate_latent_and_input(Y[:2], 2, 1)
        assert lat.shape == (2, 2) and inp.shape == (2, 1)
    check("A4  two timepoints (T=2)", a4)

    # A5. 含 NaN 的观测 (应被列均值替换,不崩)
    def a5():
        Yn = Y.copy()
        Yn[5, 3] = np.nan
        Yn[10, 0] = np.inf
        lat, inp = estimate_latent_and_input(Yn, 3, 2)
        assert np.all(np.isfinite(lat)) and np.all(np.isfinite(inp))
    check("A5  NaN/Inf in observation handled", a5)

    # A6. 常数观测 (PCA 退化情况)
    def a6():
        lat, inp = estimate_latent_and_input(np.ones((50, 10)), 2, 1)
        assert lat.shape == (50, 2) and inp.shape == (50, 1)
        assert np.all(np.isfinite(lat)) and np.all(np.isfinite(inp))
    check("A6  constant observation (degenerate PCA)", a6)

    # A7. LatentDim 超过 neuron 数 (PCA 需补零)
    def a7():
        Ysmall = Y[:, :2]            # 只有 2 个 neuron
        lat, inp = estimate_latent_and_input(Ysmall, 5, 2)
        assert lat.shape == (Y.shape[0], 5)
    check("A7  LatentDim > #neurons (zero-padding)", a7)

    # A8. 非法参数应抛错
    def a8():
        for bad in [0, -1, 2.5]:
            try:
                estimate_latent_and_input(Y, bad, 2)
            except (ValueError, Exception):
                continue
            raise AssertionError(f"LatentDim={bad} should have raised")
    check("A8  invalid LatentDim raises", a8)

    # A9. 1D / 3D 输入应抛错
    def a9():
        for bad_shape in [Y.ravel(), Y[None, :, :]]:
            try:
                estimate_latent_and_input(bad_shape, 3, 2)
            except ValueError:
                continue
            raise AssertionError("non-2D input should raise ValueError")
    check("A9  non-2D observation raises ValueError", a9)

    # A10. 可复现性 (同输入两次结果一致)
    def a10():
        l1, i1 = estimate_latent_and_input(Y, 3, 2)
        l2, i2 = estimate_latent_and_input(Y, 3, 2)
        assert np.allclose(l1, l2) and np.allclose(i1, i2)
    check("A10 deterministic / reproducible", a10)

    n_pass = sum(r[1] for r in results)
    print(f"\n  PART A: {n_pass}/{len(results)} passed.\n")
    return results


# ============================================================
# PART B. RECOVERY QUALITY
# ============================================================
def part_b_recovery():
    print("=" * 60)
    print("PART B. RECOVERY QUALITY (on synthetic ground-truth)")
    print("=" * 60)

    # ---- B1. 基础恢复: latent + input 的 R^2 ----
    Y, x_true, u_true, A, B, C = make_synthetic(T=300, N=40, n=4, m=2,
                                                noise=0.1, seed=1)
    lat, inp = estimate_latent_and_input(Y, 4, 2)

    lat_r2, lat_per, lat_al = aligned_corr(lat, x_true)
    inp_r2, inp_per, inp_al = aligned_corr(inp, u_true)
    print(f"  Latent recovery  mean R^2 = {lat_r2:.3f}  (per-dim {np.round(lat_per,2)})")
    print(f"  Input  recovery  mean R^2 = {inp_r2:.3f}  (per-dim {np.round(inp_per,2)})")

    # ---- 图1: latent 真实 vs 恢复 (对齐后) ----
    n = x_true.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.0 * n), sharex=True)
    for j in range(n):
        axes[j].plot(x_true[:, j], label="true", lw=1.5)
        axes[j].plot(lat_al[:, j], "--", label="recovered", lw=1.2)
        axes[j].set_ylabel(f"latent {j+1}")
        axes[j].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time")
    fig.suptitle(f"Latent recovery (mean R^2={lat_r2:.2f})")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/b1_latent_recovery.png", dpi=110)
    plt.close(fig)

    # ---- 图2: input 真实 vs 恢复 ----
    m = u_true.shape[1]
    fig, axes = plt.subplots(m, 1, figsize=(10, 2.2 * m), sharex=True)
    if m == 1:
        axes = [axes]
    for j in range(m):
        axes[j].plot(u_true[:, j], label="true", lw=1.2, alpha=0.8)
        axes[j].plot(inp_al[:, j], "--", label="recovered", lw=1.0)
        axes[j].set_ylabel(f"input {j+1}")
        axes[j].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time")
    fig.suptitle(f"Input recovery (mean R^2={inp_r2:.2f})")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/b2_input_recovery.png", dpi=110)
    plt.close(fig)

    # ---- B2. 观测重构: Y_hat = C_est @ latent, 看能否重构观测 ----
    # 用恢复的 latent 线性回归回 Y, 计算重构 R^2
    W, *_ = np.linalg.lstsq(lat, Y, rcond=None)
    Y_hat = lat @ W
    ss_res = np.sum((Y - Y_hat) ** 2)
    ss_tot = np.sum((Y - Y.mean(0)) ** 2)
    recon_r2 = 1 - ss_res / ss_tot
    print(f"  Observation reconstruction R^2 = {recon_r2:.3f}")

    # ---- B3. 噪声鲁棒性: 不同观测噪声下的 latent R^2 ----
    noises = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]
    lat_scores, inp_scores = [], []
    for nz in noises:
        Yn, xn, un, *_ = make_synthetic(T=300, N=40, n=4, m=2, noise=nz, seed=2)
        la, ip = estimate_latent_and_input(Yn, 4, 2)
        lat_scores.append(aligned_corr(la, xn)[0])
        inp_scores.append(aligned_corr(ip, un)[0])
    print("  Noise sweep (latent R^2):",
          {nz: round(s, 2) for nz, s in zip(noises, lat_scores)})

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(noises, lat_scores, "o-", label="latent R^2")
    ax.plot(noises, inp_scores, "s--", label="input R^2")
    ax.set_xlabel("observation noise std")
    ax.set_ylabel("recovery R^2")
    ax.set_title("Robustness to observation noise")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/b3_noise_robustness.png", dpi=110)
    plt.close(fig)

    # ---- B4. baseline 对比: 纯 PCA vs 本估计器 ----
    # 纯 PCA: 直接对标准化观测做 PCA, 不用 KF/迭代
    from Estimator import _standardize_columns, _pca_scores, _fix_pca_signs, _normalize_columns
    Ys, _, _ = _standardize_columns(Y)
    pca_lat = _normalize_columns(_fix_pca_signs(_pca_scores(Ys, 4)[0]))
    pca_r2 = aligned_corr(pca_lat, x_true)[0]
    print(f"  Baseline (pure PCA) latent R^2 = {pca_r2:.3f}")
    print(f"  This estimator      latent R^2 = {lat_r2:.3f}")
    print(f"  -> improvement: {lat_r2 - pca_r2:+.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["pure PCA", "this estimator"], [pca_r2, lat_r2],
           color=["#bbb", "#4c72b0"])
    ax.set_ylabel("latent recovery R^2")
    ax.set_title("PCA baseline vs full estimator")
    ax.set_ylim(0, 1)
    for i, v in enumerate([pca_r2, lat_r2]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/b4_baseline_comparison.png", dpi=110)
    plt.close(fig)

    # ---- B5. KF 是否真的比 PCA 强? 多场景诊断 ----
    # 重要: 验证迭代+Kalman 相比纯 PCA 是否带来实质增益。
    from Estimator import _standardize_columns as _sc
    def pca_only(Yin, dim):
        Ys, _, _ = _sc(Yin)
        return _normalize_columns(_fix_pca_signs(_pca_scores(Ys, dim)[0]))

    print("  KF-vs-PCA across regimes (latent R^2):")
    configs = [
        ("low-noise  many-neuron", dict(N=40, noise=0.1)),
        ("high-noise many-neuron", dict(N=40, noise=1.5)),
        ("high-noise few-neuron ", dict(N=6,  noise=1.0)),
    ]
    for name, cfg in configs:
        Yc, xc, uc, *_ = make_synthetic(T=300, n=4, m=2, seed=3, **cfg)
        kf = aligned_corr(estimate_latent_and_input(Yc, 4, 2)[0], xc)[0]
        pc = aligned_corr(pca_only(Yc, 4), xc)[0]
        print(f"    {name}:  KF={kf:.3f}  PCA={pc:.3f}  diff={kf-pc:+.3f}")

    print(f"\n  Figures saved to ./{FIGDIR}/\n")
    return {
        "latent_r2": lat_r2, "input_r2": inp_r2,
        "recon_r2": recon_r2, "pca_r2": pca_r2,
        "noise_latent": lat_scores,
    }


if __name__ == "__main__":
    a = part_a_sanity()
    b = part_b_recovery()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Sanity: {sum(r[1] for r in a)}/{len(a)} passed")
    print(f"  Latent R^2 = {b['latent_r2']:.3f} | Input R^2 = {b['input_r2']:.3f}")
    print(f"  Recon R^2 = {b['recon_r2']:.3f} | PCA baseline = {b['pca_r2']:.3f}")


"""
============================================================
test_estimator_v2.py  —  Estimator 验证脚本 v2
GG4 Week 2 (smoother 版)

相比 v1 新增:
  PART C. A 矩阵诊断  —— 稳定性、normality、特征值结构
  PART D. Report-friendly 可视化:
    D1. 估计的 A 矩阵特征值在复平面上 vs 真实 A
    D2. 三方对比(PCA / filter / smoother)柱状图
    D3. Input recovery 按信噪比分层(解释为什么弱 input 难恢复)
    D4. EM 迭代收敛曲线(latent R^2 随迭代次数变化)
    D5. Latent 子空间相似度热图(估计 vs 真实)

用法:
  python test_estimator_v2.py
============================================================
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Estimator import (
    estimate_latent_and_input,
    _standardize_columns, _pca_scores, _fix_pca_signs, _normalize_columns,
    _stabilize_A, _identify_system,
    _kalman_filter, _kalman_filter_with_covs, _rts_smoother,
    _estimate_inputs_from_dynamics, _as_2d_float_array,
)

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)


# ============================================================
# 辅助
# ============================================================
def make_synthetic(T=300, N=40, n=3, m=2, noise=0.1, seed=0,
                   rho=0.95, input_strength=0.4, sparse_input=False):
    rng = np.random.default_rng(seed)
    theta = 0.3
    A = np.zeros((n, n))
    i = 0
    while i + 1 < n:
        A[i:i+2, i:i+2] = rho * np.array([[np.cos(theta), -np.sin(theta)],
                                          [np.sin(theta),  np.cos(theta)]])
        i += 2
    if n % 2 == 1:
        A[-1, -1] = rho
    B = rng.standard_normal((n, m)) * input_strength
    C = rng.standard_normal((N, n))
    if sparse_input:
        u = np.zeros((T, m))
        idx = rng.choice(T, size=T // 10, replace=False)
        u[idx] = rng.standard_normal((len(idx), m)) * 3.0
    else:
        u = rng.standard_normal((T, m))
    x = np.zeros((T, n)); x[0] = rng.standard_normal(n)
    for t in range(1, T):
        x[t] = A @ x[t-1] + B @ u[t] + 0.02 * rng.standard_normal(n)
    Y = x @ C.T + noise * rng.standard_normal((T, N))
    return Y, x, u, A, B, C


def aligned_r2(est, true):
    W, *_ = np.linalg.lstsq(est, true, rcond=None)
    est_al = est @ W
    r2_per = []
    for j in range(true.shape[1]):
        ss_res = np.sum((true[:, j] - est_al[:, j]) ** 2)
        ss_tot = np.sum((true[:, j] - true[:, j].mean()) ** 2)
        r2_per.append(1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan)
    r2_per = np.array(r2_per)
    return np.nanmean(r2_per), r2_per, est_al


def get_estimated_A(Y, LatentDim):
    """跑一遍管线,把估出来的 A 取出来给诊断用"""
    Y = _as_2d_float_array(Y)
    Y_std, _, _ = _standardize_columns(Y)
    Z = _pca_scores(Y_std, LatentDim)[0]
    Z = _fix_pca_signs(Z); Z = _normalize_columns(Z)
    P0 = np.atleast_2d(np.cov(Z.T))
    if P0.shape != (LatentDim, LatentDim):
        P0 = np.eye(LatentDim)
    P0 = 0.5 * (P0 + P0.T) + 1e-4 * np.eye(LatentDim)
    A_hat = np.eye(LatentDim)
    for _ in range(3):
        A_hat, C_hat, Q_hat, R_hat = _identify_system(Z, Y_std)
        x_filt, P_filt, x_pred_arr, P_pred_arr, _ = \
            _kalman_filter_with_covs(Y_std, A_hat, C_hat, Q_hat, R_hat, Z[0], P0)
        Z = _rts_smoother(x_filt, P_filt, x_pred_arr, P_pred_arr, A_hat)
    return A_hat


# ============================================================
# PART C. A 矩阵诊断
# ============================================================
def part_c_A_diagnostics():
    print("=" * 60)
    print("PART C. A 矩阵稳定性 / normality 诊断")
    print("=" * 60)

    Y, x_true, u_true, A_true, _, _ = make_synthetic(T=300, n=4, m=2,
                                                     N=40, noise=0.1, seed=1)
    A_hat = get_estimated_A(Y, LatentDim=4)

    # --- C1. 谱半径 (是否稳定) ---
    eig_hat = np.linalg.eigvals(A_hat)
    eig_true = np.linalg.eigvals(A_true)
    rho_hat = np.max(np.abs(eig_hat))
    rho_true = np.max(np.abs(eig_true))
    print(f"  C1 谱半径(stability):  estimated={rho_hat:.3f}  true={rho_true:.3f}")
    print(f"     {'稳定 ✓' if rho_hat < 1 else '发散 ✗'}  "
          f"(_stabilize_A 把上限钳在 0.98)")

    # --- C2. normality (A 是否近似正规矩阵)---
    # 正规矩阵的定义: A @ A.T == A.T @ A
    # 不正规 → 特征向量不正交 → 系统可能在过渡期"放大"扰动
    # 度量: ||A A^T - A^T A||_F / ||A||_F^2
    AAT = A_hat @ A_hat.T
    ATA = A_hat.T @ A_hat
    nonnormality = np.linalg.norm(AAT - ATA, 'fro') / (np.linalg.norm(A_hat, 'fro') ** 2 + 1e-12)
    print(f"  C2 non-normality 度量: {nonnormality:.3f}  "
          f"(0=完全正规, 越大越不正规)")
    if nonnormality < 0.05:
        print("     A 接近正规 → 良好,特征向量近似正交")
    elif nonnormality < 0.3:
        print("     A 中度不正规 → 可能存在 transient amplification")
    else:
        print("     A 高度不正规 → 系统可能放大短时扰动")

    # --- C3. 特征值是否成共轭对(说明有旋转成分)---
    eig_complex = [e for e in eig_hat if abs(e.imag) > 1e-6]
    print(f"  C3 复特征值数量: {len(eig_complex)} / {len(eig_hat)}  "
          f"(共轭对说明 A 有旋转动力学,如板书的旋转矩阵)")

    # --- C4. 估计的 A 和真实 A 的"动力学相似度"---
    # 用 frobenius norm,但需先按"特征值匹配"重排 (estimated A 可能在
    # 不同基下,直接对比 A 矩阵元意义不大). 这里用特征值集合距离.
    eig_hat_sorted = sorted(eig_hat, key=lambda z: (-abs(z), -z.real))
    eig_true_sorted = sorted(eig_true, key=lambda z: (-abs(z), -z.real))
    eig_dist = np.mean([abs(a - b) for a, b in zip(eig_hat_sorted, eig_true_sorted)])
    print(f"  C4 特征值集合距离(estimated vs true): {eig_dist:.3f}")

    # --- C5. 在复平面画出特征值 ---
    fig, ax = plt.subplots(figsize=(6, 6))
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.4, label='unit circle')
    ax.plot(0.98*np.cos(theta), 0.98*np.sin(theta), 'r:', alpha=0.4,
            label='stability cap (ρ=0.98)')
    ax.scatter(eig_true.real, eig_true.imag, marker='o', s=120,
               facecolors='none', edgecolors='C0', linewidth=2,
               label='true A eigenvalues')
    ax.scatter(eig_hat.real, eig_hat.imag, marker='x', s=120,
               c='C3', linewidth=2, label='estimated A eigenvalues')
    ax.axhline(0, color='gray', alpha=0.3, lw=0.5)
    ax.axvline(0, color='gray', alpha=0.3, lw=0.5)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("Eigenvalues of A in complex plane (estimated vs true)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/c5_A_eigenvalues.png", dpi=110)
    plt.close(fig)
    print(f"  → 图: {FIGDIR}/c5_A_eigenvalues.png\n")

    return {"rho_hat": rho_hat, "rho_true": rho_true,
            "nonnormality": nonnormality, "eig_dist": eig_dist}


# ============================================================
# PART D. Report-friendly 图
# ============================================================
def part_d_report_figures():
    print("=" * 60)
    print("PART D. Report-friendly figures")
    print("=" * 60)

    # --- D2. 三方对比柱状图 (放 report 最有说服力)---
    print("  D2 三方对比 (PCA vs filter vs smoother)...")
    scenarios = [
        ("low noise\nmany neurons",  dict(N=40, noise=0.1)),
        ("high noise\nmany neurons", dict(N=40, noise=1.5)),
        ("high noise\nfew neurons",  dict(N=6,  noise=1.0)),
        ("sparse input\nhigh noise", dict(N=6,  noise=1.0, sparse_input=True)),
    ]
    lat_results = {"PCA": [], "filter": [], "smoother": []}
    inp_results = {"PCA": [], "filter": [], "smoother": []}

    for _, cfg in scenarios:
        Y, x_true, u_true, *_ = make_synthetic(T=300, n=4, m=2, seed=1, **cfg)

        # PCA only
        Ys, _, _ = _standardize_columns(Y)
        pca_lat = _normalize_columns(_fix_pca_signs(_pca_scores(Ys, 4)[0]))
        A_pca = _identify_system(pca_lat, Ys)[0]
        pca_inp = _estimate_inputs_from_dynamics(pca_lat, A_pca, 2)
        lat_results["PCA"].append(aligned_r2(pca_lat, x_true)[0])
        inp_results["PCA"].append(aligned_r2(pca_inp, u_true)[0])

        # filter (手动跑,不用 smoother)
        Z = pca_lat.copy()
        P0 = np.atleast_2d(np.cov(Z.T))
        if P0.shape != (4, 4): P0 = np.eye(4)
        P0 = 0.5*(P0+P0.T) + 1e-4*np.eye(4)
        for _ in range(3):
            A_h, C_h, Q_h, R_h = _identify_system(Z, Ys)
            Z, _ = _kalman_filter(Ys, A_h, C_h, Q_h, R_h, Z[0], P0)
        flat = _normalize_columns(_fix_pca_signs(Z))
        finp = _estimate_inputs_from_dynamics(flat, A_h, 2)
        lat_results["filter"].append(aligned_r2(flat, x_true)[0])
        inp_results["filter"].append(aligned_r2(finp, u_true)[0])

        # smoother (= 正式版的 estimator)
        slat, sinp = estimate_latent_and_input(Y, 4, 2)
        lat_results["smoother"].append(aligned_r2(slat, x_true)[0])
        inp_results["smoother"].append(aligned_r2(sinp, u_true)[0])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x_pos = np.arange(len(scenarios))
    w = 0.27
    colors = {"PCA": "#bbb", "filter": "#4c72b0", "smoother": "#dd8452"}
    for i, k in enumerate(["PCA", "filter", "smoother"]):
        axes[0].bar(x_pos + (i-1)*w, lat_results[k], w, label=k, color=colors[k])
        axes[1].bar(x_pos + (i-1)*w, inp_results[k], w, label=k, color=colors[k])
    for ax, title in [(axes[0], "Latent recovery R²"),
                       (axes[1], "Input recovery R²")]:
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s[0] for s in scenarios], fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(loc="upper right")
    fig.suptitle("Method comparison: PCA vs EM+filter vs EM+smoother", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/d2_three_way_comparison.png", dpi=110)
    plt.close(fig)
    print(f"     → {FIGDIR}/d2_three_way_comparison.png")

    # --- D3. Input recovery 按信噪比分层 ---
    # 解释为什么有的 input 维度恢复好,有的差: 取决于 ||Bu||/||w||
    print("  D3 Input recovery vs input-to-noise ratio...")
    strengths = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    inp_r2_list = []
    for s in strengths:
        Y, x, u, *_ = make_synthetic(T=300, N=40, n=4, m=2,
                                     noise=0.1, seed=5, input_strength=s)
        _, inp = estimate_latent_and_input(Y, 4, 2)
        inp_r2_list.append(aligned_r2(inp, u)[0])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(strengths, inp_r2_list, "o-", color="#dd8452", lw=2, ms=8)
    ax.set_xscale("log")
    ax.set_xlabel("input strength  ||B|| (× observation noise std)")
    ax.set_ylabel("input recovery R²")
    ax.set_title("Input recovery degrades when input/noise ratio is low")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    for x, y in zip(strengths, inp_r2_list):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/d3_input_snr.png", dpi=110)
    plt.close(fig)
    print(f"     → {FIGDIR}/d3_input_snr.png")

    # --- D4. EM 迭代收敛曲线 ---
    # 证明 3 轮迭代够用 (老师可能问的)
    print("  D4 EM convergence curve...")
    Y, x_true, _, _, _, _ = make_synthetic(T=300, n=4, m=2, N=10,
                                           noise=0.8, seed=7)
    Y_std, _, _ = _standardize_columns(Y)
    Z = _normalize_columns(_fix_pca_signs(_pca_scores(Y_std, 4)[0]))
    P0 = np.atleast_2d(np.cov(Z.T))
    P0 = 0.5*(P0+P0.T) + 1e-4*np.eye(4)
    r2_per_iter = [aligned_r2(Z, x_true)[0]]
    for it in range(8):
        A_h, C_h, Q_h, R_h = _identify_system(Z, Y_std)
        x_filt, P_filt, x_pred_arr, P_pred_arr, _ = \
            _kalman_filter_with_covs(Y_std, A_h, C_h, Q_h, R_h, Z[0], P0)
        Z = _rts_smoother(x_filt, P_filt, x_pred_arr, P_pred_arr, A_h)
        r2_per_iter.append(aligned_r2(Z, x_true)[0])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(len(r2_per_iter)), r2_per_iter, "o-", lw=2, ms=8,
            color="#4c72b0")
    ax.axvline(3, color="red", linestyle="--", alpha=0.6,
               label="default n_iters=3")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("latent recovery R²")
    ax.set_title("EM convergence (high-noise few-neuron regime)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/d4_em_convergence.png", dpi=110)
    plt.close(fig)
    print(f"     → {FIGDIR}/d4_em_convergence.png  "
          f"(R^2 per iter: {[round(v,3) for v in r2_per_iter]})")

    # --- D5. Latent subspace 相似度热图 ---
    # 在 latent 空间里,估计的每一维 vs 真实的每一维做相关
    # 看是不是 "估计的某些维基本对应真实的某些维"
    print("  D5 Latent subspace similarity heatmap...")
    Y, x_true, _, _, _, _ = make_synthetic(T=300, n=4, m=2, N=40,
                                           noise=0.1, seed=11)
    lat, _ = estimate_latent_and_input(Y, 4, 2)
    # 标准化两边
    Le = (lat - lat.mean(0)) / (lat.std(0) + 1e-12)
    Lt = (x_true - x_true.mean(0)) / (x_true.std(0) + 1e-12)
    corr = (Le.T @ Lt) / Lt.shape[0]   # (n_est, n_true)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(np.abs(corr), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([f"true {i+1}" for i in range(4)])
    ax.set_yticklabels([f"est {i+1}" for i in range(4)])
    ax.set_title("|correlation| between\nestimated and true latent dims")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{abs(corr[i,j]):.2f}", ha="center", va="center",
                    color="white" if abs(corr[i,j]) < 0.5 else "black",
                    fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/d5_latent_similarity.png", dpi=110)
    plt.close(fig)
    print(f"     → {FIGDIR}/d5_latent_similarity.png")

    print()


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    print("\n注: PART A, B 直接用原 test_estimator.py 跑。本脚本只跑 C 和 D。\n")
    c = part_c_A_diagnostics()
    part_d_report_figures()
    print("=" * 60)
    print("DONE.  All figures in ./figures/")
    print("=" * 60)