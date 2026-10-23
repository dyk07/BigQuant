"""后处理变体离线基准 (修正版): 真实 (day, inst, mod) 三维截面。

核心思想 (2026-10-23):
  EMA 是 L2 型平滑, 会把真实跳变也一起糊掉 -> IC 永久损失。
  换手指标是 mean|Δ pct_rank| (L1 型), 因此 L1 型收缩(软阈值)更匹配:
  小幅抖动收缩到 0, 大幅跳变保留, 只按 theta 削去边缘。
  另测用户直觉「排名一位抖动回退」: 用真实截面分位数赋值。

指标口径 (与平台一致):
  IC      : 逐 (day, mod) 截面 spearman(pred, y)
  IC_IR   : mean(IC) / std(IC)  (跨全部截面)
  Sharpe  : 逐截面 top20%-bottom20% 收益差的 mean/std
  turnover: 逐 (day, inst) 沿时间轴的 mean|Δ pct_rank|
"""
import numpy as np

NPZ = "submissions/_tools/_preds_val4m.npz"


def load():
    z = np.load(NPZ)
    inst = z["inst"].astype("int64")
    day = z["day"].astype("int64")
    mod = z["mod"].astype("int64")
    y = z["y"].astype("float64")
    pred = z["pred"].astype("float64")

    ud = np.unique(day); ui = np.unique(inst); um = np.unique(mod)
    di = np.searchsorted(ud, day)
    ii = np.searchsorted(ui, inst)
    mi = np.searchsorted(um, mod)

    A = np.full((len(ud), len(ui), len(um)), np.nan)
    Y = np.full((len(ud), len(ui), len(um)), np.nan)
    A[di, ii, mi] = pred
    Y[di, ii, mi] = y
    return A, Y, ud, ui, um


def _pct_rank_1d(v):
    out = np.full(len(v), np.nan)
    ok = ~np.isnan(v)
    n = ok.sum()
    if n < 2:
        return out
    x = v[ok]
    order = np.argsort(x, kind="mergesort")
    r = np.empty(n)
    r[order] = np.arange(n)
    out[ok] = r / (n - 1)
    return out


def _rank_mat(A):
    """逐截面 (day, mod) 做 pct 排名, 沿 inst 轴。A: (D, I, M) -> (D, M, I)。"""
    D, I, M = A.shape
    R = np.full((D, M, I), np.nan)
    for d in range(D):
        for m in range(M):
            R[d, m] = _pct_rank_1d(A[d, :, m])
    return R


def metrics(A, Y, want_rank=False):
    D, I, M = A.shape
    ics, sps = [], []
    for d in range(D):
        for m in range(M):
            a = A[d, :, m]; y = Y[d, :, m]
            ok = ~np.isnan(a) & ~np.isnan(y)
            n = ok.sum()
            if n < 20:
                continue
            av, yv = a[ok], y[ok]
            ra = _pct_rank_1d(av) * (n - 1)
            ry = _pct_rank_1d(yv) * (n - 1)
            if ra.std() < 1e-9 or ry.std() < 1e-9:
                continue
            ics.append(np.corrcoef(ra, ry)[0, 1])
            k = max(1, int(n * 0.2))
            o = np.argsort(av, kind="mergesort")
            sps.append(yv[o[-k:]].mean() - yv[o[:k]].mean())
    ics = np.asarray(ics); sps = np.asarray(sps)
    icm = ics.mean(); icir = icm / ics.std()
    sharpe = sps.mean() / sps.std()

    R = _rank_mat(A)
    dR = np.abs(np.diff(R, axis=1))
    to = np.nanmean(dR)
    if want_rank:
        return icm, icir, sharpe, to, dR
    return icm, icir, sharpe, to


def v_ema(A, a):
    out = np.full_like(A, np.nan)
    prev = A[:, :, 0].copy()
    out[:, :, 0] = prev
    for m in range(1, A.shape[2]):
        cur = A[:, :, m]
        nxt = a * cur + (1 - a) * prev
        nxt = np.where(np.isnan(cur), prev, nxt)
        out[:, :, m] = nxt
        prev = nxt
    return out


def _zscore(A):
    Z = np.full_like(A, np.nan)
    D, I, M = A.shape
    for d in range(D):
        for m in range(M):
            col = A[d, :, m]; ok = ~np.isnan(col)
            if ok.sum() < 2:
                continue
            v = col[ok]
            sd = v.std()
            Z[d, ok, m] = (v - v.mean()) / (sd if sd > 1e-12 else 1.0)
    return Z


def v_z_soft(A, theta):
    Z = _zscore(A)
    out = np.full_like(Z, np.nan)
    prev = Z[:, :, 0].copy()
    out[:, :, 0] = prev
    for m in range(1, Z.shape[2]):
        cur = Z[:, :, m]
        d = cur - prev
        ds = np.sign(d) * np.maximum(0.0, np.abs(d) - theta)
        nxt = prev + ds
        nxt = np.where(np.isnan(cur), prev, nxt)
        out[:, :, m] = nxt
        prev = nxt
    return out


def v_z_deadband(A, theta):
    Z = _zscore(A)
    out = np.full_like(Z, np.nan)
    prev = Z[:, :, 0].copy()
    out[:, :, 0] = prev
    for m in range(1, Z.shape[2]):
        cur = Z[:, :, m]
        freeze = np.abs(cur - prev) <= theta
        nxt = np.where(freeze | np.isnan(cur), prev, cur)
        out[:, :, m] = nxt
        prev = nxt
    return out


def v_rank_soft(A, theta_pct):
    """排名空间 L1 软阈值: 直接对 pct_rank 做软阈值。

    四个指标 (IC/IC_IR/Sharpe/turnover) 全部只依赖截面内排名的单调变换,
    因此在 pct_rank 空间操作与在原始分数空间操作**在指标上完全等价**,
    但阈值 theta 有了明确含义: 它直接就是"容忍的排名漂移量"。
    """
    D, I, M = A.shape
    out = np.full_like(A, np.nan)
    prev = np.full((D, I), np.nan)
    for d in range(D):
        col = A[d, :, 0]; ok = ~np.isnan(col)
        if ok.sum() >= 2:
            r = np.full(I, np.nan)
            r[ok] = _pct_rank_1d(col)[ok]
            prev[d, ok] = r[ok]
    out[:, :, 0] = A[:, :, 0]
    for m in range(1, M):
        for d in range(D):
            cur = A[d, :, m]; ok = ~np.isnan(cur)
            n = ok.sum()
            if n < 2:
                continue
            r = np.full(I, np.nan)
            r[ok] = _pct_rank_1d(cur)[ok]
            dd = r - prev[d]
            ds = np.sign(dd) * np.maximum(0.0, np.abs(dd) - theta_pct)
            tgt = prev[d] + ds
            vals = np.sort(cur[ok])
            sel = ok & ~np.isnan(tgt)
            pos = np.clip(np.round(tgt[sel] * (n - 1)).astype(int), 0, n - 1)
            out[d, sel, m] = vals[pos]
            prev[d, ok] = r[ok]
    return out


def v_qsoft(A, theta_max, gamma):
    """分位加权软阈值: 中位区激进收缩, 尾部保持锐利。

    洞察: 换手 = mean|Δ pct_rank| 对全部股票等权计数, 但组合仓位与 IC
    主要由尾部 (近 0/1) 决定; 中位区 (近 0.5) 权重≈0 却同等计入换手。
    因此最优策略是按分位分配容忍度:
        theta(u) = theta_max * (4u(1-u))^gamma     # u 为当期 pct_rank
    中位 u=0.5 处容忍度最大, 两端 u->0/1 处为 0 (尾部完全不受平滑)。
    """
    D, I, M = A.shape
    out = np.full_like(A, np.nan)
    prev = np.full((D, I), np.nan)
    for d in range(D):
        col = A[d, :, 0]; ok = ~np.isnan(col)
        if ok.sum() >= 2:
            r = np.full(I, np.nan)
            r[ok] = _pct_rank_1d(col)[ok]
            prev[d, ok] = r[ok]
    out[:, :, 0] = A[:, :, 0]
    for m in range(1, M):
        for d in range(D):
            cur = A[d, :, m]; ok = ~np.isnan(cur)
            n = ok.sum()
            if n < 2:
                continue
            r = np.full(I, np.nan)
            r[ok] = _pct_rank_1d(cur)[ok]
            th = theta_max * np.power(4.0 * r * (1.0 - r), gamma)
            dd = r - prev[d]
            ds = np.sign(dd) * np.maximum(0.0, np.abs(dd) - th)
            tgt = prev[d] + ds
            vals = np.sort(cur[ok])
            sel = ok & ~np.isnan(tgt)
            pos = np.clip(np.round(tgt[sel] * (n - 1)).astype(int), 0, n - 1)
            out[d, sel, m] = vals[pos]
            prev[d, ok] = r[ok]
    return out


def v_qsoft_edge(A, theta_max, gamma, edge):
    """分位加权软阈值 + 尾部硬保护区 (u<edge 或 u>1-edge 完全不平滑)。"""
    D, I, M = A.shape
    out = np.full_like(A, np.nan)
    prev = np.full((D, I), np.nan)
    for d in range(D):
        col = A[d, :, 0]; ok = ~np.isnan(col)
        if ok.sum() >= 2:
            r = np.full(I, np.nan)
            r[ok] = _pct_rank_1d(col)[ok]
            prev[d, ok] = r[ok]
    out[:, :, 0] = A[:, :, 0]
    for m in range(1, M):
        for d in range(D):
            cur = A[d, :, m]; ok = ~np.isnan(cur)
            n = ok.sum()
            if n < 2:
                continue
            r = np.full(I, np.nan)
            r[ok] = _pct_rank_1d(cur)[ok]
            th = theta_max * np.power(4.0 * r * (1.0 - r), gamma)
            th = np.where((r < edge) | (r > 1.0 - edge), 0.0, th)
            dd = r - prev[d]
            ds = np.sign(dd) * np.maximum(0.0, np.abs(dd) - th)
            tgt = prev[d] + ds
            vals = np.sort(cur[ok])
            sel = ok & ~np.isnan(tgt)
            pos = np.clip(np.round(tgt[sel] * (n - 1)).astype(int), 0, n - 1)
            out[d, sel, m] = vals[pos]
            prev[d, ok] = r[ok]
    return out


def v_rank_clip(A, k):
    """排名变动速率限制: |Δ pct_rank| 单步不超过 k (L-infinity 截断)。"""
    D, I, M = A.shape
    out = np.full_like(A, np.nan)
    prev = np.full((D, I), np.nan)
    for d in range(D):
        col = A[d, :, 0]; ok = ~np.isnan(col)
        if ok.sum() >= 2:
            r = np.full(I, np.nan)
            r[ok] = _pct_rank_1d(col)[ok]
            prev[d, ok] = r[ok]
    out[:, :, 0] = A[:, :, 0]
    for m in range(1, M):
        for d in range(D):
            cur = A[d, :, m]; ok = ~np.isnan(cur)
            n = ok.sum()
            if n < 2:
                continue
            r = np.full(I, np.nan)
            r[ok] = _pct_rank_1d(cur)[ok]
            dd = r - prev[d]
            tgt = prev[d] + np.clip(dd, -k, k)
            vals = np.sort(cur[ok])
            sel = ok & ~np.isnan(tgt)
            pos = np.clip(np.round(tgt[sel] * (n - 1)).astype(int), 0, n - 1)
            out[d, sel, m] = vals[pos]
            prev[d, ok] = r[ok]
    return out


def v_qsoft_ema(A, theta_max, gamma, a):
    """分位加权软阈值 + 中位区 EMA 残余平滑 (串联)。"""
    Z = v_qsoft(A, theta_max, gamma)
    return v_ema(Z, a)


def v_zsel_ema(A, theta, a):
    """小幅抖动走 EMA, 大幅跳变完全跟进。"""
    Z = _zscore(A)
    out = np.full_like(Z, np.nan)
    prev = Z[:, :, 0].copy()
    out[:, :, 0] = prev
    for m in range(1, Z.shape[2]):
        cur = Z[:, :, m]
        ema = a * cur + (1 - a) * prev
        jump = np.abs(cur - prev) > theta
        nxt = np.where(jump, cur, ema)
        nxt = np.where(np.isnan(cur), prev, nxt)
        out[:, :, m] = nxt
        prev = nxt
    return out


def v_rank_target(A, k):
    """排名变动 <= k 位 -> 沿用上期排名, 并映射回当期截面的真实分值。"""
    D, I, M = A.shape
    out = np.full_like(A, np.nan)
    prev_rk = np.full((D, I), np.nan)
    for d in range(D):
        col = A[d, :, 0]; ok = ~np.isnan(col)
        n = ok.sum()
        if n >= 2:
            r = np.full(I, np.nan)
            r[ok] = _pct_rank_1d(col)[ok] * (n - 1)
            prev_rk[d, ok] = r[ok]
    out[:, :, 0] = A[:, :, 0]
    for m in range(1, M):
        for d in range(D):
            cur = A[d, :, m]; ok = ~np.isnan(cur)
            n = ok.sum()
            if n < 2:
                continue
            rk = np.full(I, np.nan)
            rk[ok] = _pct_rank_1d(cur)[ok] * (n - 1)
            pr = prev_rk[d]
            tgt = np.where(ok & ~np.isnan(pr) & (np.abs(rk - pr) <= k), pr, rk)
            vals = np.sort(cur[ok])
            sel = ok & ~np.isnan(tgt)
            pos = np.clip(np.round(tgt[sel]).astype(int), 0, n - 1)
            out[d, sel, m] = vals[pos]
            prev_rk[d, ok] = rk[ok]
    return out


def main():
    A, Y, ud, ui, um = load()
    D, I, M = A.shape
    print(f"真实截面: {D} 天 x {I} 股 x {M} 截面 = {np.isfinite(A).sum()} 样本")

    _m = metrics(A, Y, want_rank=True)
    base, dR = _m[:4], _m[4]
    print(f"baseline IC={base[0]:+.4f} IC_IR={base[1]:.3f} "
          f"Sharpe={base[2]:.3f} TO={base[3]:.4f}")

    v = dR[~np.isnan(dR)]
    print("\n【用户假设检验】相邻截面 pct_rank 变动量 |dRank| 分布")
    print(f"  样本数={v.size}  均值={v.mean():.4f}  中位={np.median(v):.4f}")
    for q in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{q:<2} = {np.percentile(v, q):.4f}")
    one = 1.0 / (I - 1)
    print(f"  |dRank| <= 1 个位次 (={one:.4f}) 占比: "
          f"{np.mean(v <= one + 1e-12) * 100:.2f}%")

    print(f"\n{'变体':<26} {'IC':>8} {'IC_IR':>7} {'Sharpe':>7} {'TO':>7}  "
          f"{'dIC%':>7} {'dTO%':>7}")
    print("-" * 80)

    def show(name, Ar):
        m = metrics(Ar, Y)
        print(f"{name:<26} {m[0]:>+8.4f} {m[1]:>7.3f} {m[2]:>7.3f} {m[3]:>7.4f}  "
              f"{(m[0]/base[0]-1)*100:>+7.1f} {(m[3]/base[3]-1)*100:>+7.1f}")
        return m

    show("baseline (无后处理)", A)
    print()
    for a in [0.35, 0.45, 0.55, 0.65, 0.75]:
        show(f"EMA a={a}", v_ema(A, a))
    print()
    for th in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        show(f"z-soft theta={th}", v_z_soft(A, th))
    print()
    for th in [0.2, 0.3, 0.5, 0.7]:
        show(f"z-deadband theta={th}", v_z_deadband(A, th))
    print()
    for th, a in [(0.5, 0.7), (0.8, 0.6), (1.0, 0.5), (1.2, 0.4)]:
        show(f"zsel-ema th={th} a={a}", v_zsel_ema(A, th, a))
    print()
    for th in [1, 2, 3, 5, 10]:
        show(f"rank-target k={th}", v_rank_target(A, th))
    print()
    for th in [0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18]:
        show(f"rank-soft theta={th}", v_rank_soft(A, th))
    print()
    for k in [0.1, 0.15, 0.2, 0.3]:
        show(f"rank-clip k={k}", v_rank_clip(A, k))
    print()
    for th, g in [(0.10, 1.0), (0.15, 1.0), (0.20, 1.0), (0.25, 1.0),
                  (0.20, 2.0), (0.30, 2.0), (0.40, 3.0)]:
        show(f"qsoft th={th} g={g}", v_qsoft(A, th, g))
    print()
    for th, g, e in [(0.25, 1.0, 0.10), (0.30, 1.0, 0.15),
                     (0.40, 2.0, 0.10), (0.50, 2.0, 0.20)]:
        show(f"qsoft-edge {th}/{g}/{e}", v_qsoft_edge(A, th, g, e))
    print()
    for th, g, a in [(0.15, 1.0, 0.75), (0.20, 1.0, 0.6), (0.25, 1.0, 0.5)]:
        show(f"qsoft({th},{g})+EMA{a}", v_qsoft_ema(A, th, g, a))


if __name__ == "__main__":
    main()
