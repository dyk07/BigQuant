import numpy as np

# 采样数据 (name, score, ic_ir, ic_mean, turnover, sharpe, stress)
data = [
    ("exp021",       0.51319, 1.4856,  0.09,   0.6312,  47.1289,  1.2541),
    ("exp022",       0.43132, 1.4462,  0.0842, 0.6638,  47.1855,  1.1587),
    ("exp020",       0.46923, 1.4572,  0.0872, 0.6441,  47.0019,  1.1825),
    ("exp020_cross", 0.43022, 1.4178,  0.0843, 0.6403,  44.9552,  1.1602),
    ("exp019",       0.42308, 1.4231,  0.0839, 0.6469,  45.3922,  1.1482),
    ("exp016",       0.45549, 1.4501,  0.0872, 0.6516,  43.4265,  1.2868),
    ("exp016b",      0.44396, 1.4204,  0.0851, 0.6352,  43.9437,  1.2281),
    ("exp012",       0.44396, 1.4176,  0.0862, 0.6391,  43.7804,  1.2439),
    ("exp00",        0.43462, 1.4008,  0.0843, 0.6307,  43.5292,  1.2294),
    ("ours_1017",    0.47253, 1.5232,  0.0886, 0.6584,  40.7586,  1.4054),
    ("ours_xsec",    0.22665, 1.0535,  0.0642, 0.6466,  31.0108,  0.8910),
    ("ours_amp",     0.32198, 0.9501,  0.0631, 0.4934,  26.3558,  0.7951),
    ("ours_2f",      0.06593, -1.5082, -0.089, 0.6156, -40.0046, -1.6092),
]

names  = [d[0] for d in data]
score  = np.array([d[1] for d in data])
ic_ir  = np.array([d[2] for d in data])
ic_mean= np.array([d[3] for d in data])
turn   = np.array([d[4] for d in data])
sharpe = np.array([d[5] for d in data])
stress = np.array([d[6] for d in data])

N = len(data)

metrics = {
    "ic_mean":  (ic_mean, 1),
    "ic_ir":    (ic_ir,   1),
    "sharpe":   (sharpe,  1),
    "stress":   (stress,  1),
    "turnover": (turn,   -1),
}

print("=" * 78)
print("方法 1: score 对各项的标准化单变量回归 (采样内敏感度)")
print("=" * 78)
for k, (vals, d) in metrics.items():
    v = vals if d == 1 else -vals
    v_std = (v - v.mean()) / v.std()
    s_std = (score - score.mean()) / score.std()
    beta = (v_std @ s_std) / (v_std @ v_std)
    r = np.corrcoef(v, score)[0, 1]
    print(f"  {k:<10} beta={beta:+.3f}   corr={r:+.3f}")

print()
print("=" * 78)
print("方法 5: 多元 OLS score ~ 5项 (标准化系数=独立边际贡献)")
print("=" * 78)
X = np.column_stack([ic_mean, ic_ir, sharpe, stress, -turn])
X_std = (X - X.mean(0)) / X.std(0)
y_std = (score - score.mean()) / score.std()
beta_multi, res, rank, sv = np.linalg.lstsq(X_std, y_std, rcond=None)
print("  多元标准化系数:")
for i, k in enumerate(["ic_mean", "ic_ir", "sharpe", "stress", "turnover"]):
    print(f"    {k:<10} beta={beta_multi[i]:+.3f}")

print()
print("=" * 78)
print("方法 6: 指标间相关性 (共线性 → 提升一项带动多项)")
print("=" * 78)
metric_keys = ["ic_mean", "ic_ir", "sharpe", "stress", "turnover"]
M = np.column_stack([ic_mean, ic_ir, sharpe, stress, turn])
corr = np.corrcoef(M.T)
print("        " + " ".join(f"{k:>9}" for k in metric_keys))
for i, k in enumerate(metric_keys):
    print(f"{k:<8}" + " ".join(f"{corr[i,j]:>9.3f}" for j in range(len(metric_keys))))

print()
print("=" * 78)
print("方法 7: 反推各项 rank 百分比 (利用 score 等式约束)")
print("  思路: score=0.2*sum(R_i)。已知 score, 反解 5 项 R_i 的分配。")
print("  假设各项 R_i 与原始值的排序单调一致, 用采样内 rank 作初始估计,")
print("  再按比例缩放使 0.2*sum(R_i)=score 成立。")
print("=" * 78)
OURS = "ours_1017"
oi = names.index(OURS)

def rank_pct_est(vals, direction):
    order = np.argsort(vals if direction == 1 else -vals)
    r = np.empty(N)
    r[order] = np.arange(1, N + 1)
    return r / N

print(f"\n对每个采样提交, 反解其 5 项 R_i (采样内 rank 按比例归一化到 score):")
print(f"{'name':<14}{'score':>8}  " + "  ".join(f"{k:>9}" for k in metric_keys))
for idx in range(N):
    rs = []
    for k, (vals, d) in metrics.items():
        r = rank_pct_est(vals, d)[idx]
        rs.append(r)
    # score = 0.2 * sum(R_i)  -> 理论 sum(R_i) = 5*score
    # 采样内 sum(r_i) 与 5*score 的比值 = 缩放因子 (全体 vs 采样 的 rank 尺度差)
    ratio = (5 * score[idx]) / sum(rs) if sum(rs) > 0 else 0
    r_scaled = [r * ratio for r in rs]
    print(f"{names[idx]:<14}{score[idx]:>8.4f}  " +
          "  ".join(f"{r:>9.3f}" for r in r_scaled))

print()
print("=" * 78)
print("结论: 每项指标 '可提升空间' = 1 - R_i (rank 百分比缺口)")
print("  用 ours_1017 反解的各项 R_i (缩放后) 作为估计")
print("=" * 78)
rs_ours = []
for k, (vals, d) in metrics.items():
    r = rank_pct_est(vals, d)[oi]
    rs_ours.append(r)
ratio_ours = (5 * score[oi]) / sum(rs_ours)
print(f"\n{OURS}: score={score[oi]:.4f}, 缩放因子(全体/采样)={ratio_ours:.3f}")
print(f"{'指标':<10}{'原始值':>10}{'采样rank':>10}{'估全体rank R':>14}{'缺口 1-R':>12}")
gaps = {}
for i, k in enumerate(metric_keys):
    r_scaled = rs_ours[i] * ratio_ours
    gaps[k] = 1 - r_scaled
    raw = [ic_mean[oi], ic_ir[oi], sharpe[oi], stress[oi], turn[oi]][i]
    print(f"{k:<10}{raw:>10.4f}{rs_ours[i]:>10.3f}{r_scaled:>14.3f}{1-r_scaled:>12.3f}")

print()
print("按'缺口 1-R' 从大到小排序 (缺口越大=越需要提升):")
for k in sorted(gaps, key=gaps.get, reverse=True):
    print(f"  {k:<10} 缺口={gaps[k]:.3f}")
