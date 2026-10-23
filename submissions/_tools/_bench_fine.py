"""后处理精细扫描: 以「公榜 TO=0.41 对应本地 TO=0.125」为预算约束。

本地->公榜换手映射 (由 baseline/v3/combo/EMA 四点拟合):
    pub_TO = 2.630 * local_TO + 0.082
因此公榜 0.41  <=>  本地 0.1247。

统一在 baseline 权重 (cnn_3f_1020_newlabel_to0005, 无有效换手正则) 上比较,
只改变后处理, 隔离变量。
"""
import numpy as np

NPZ = "submissions/_tools/_preds_val4m.npz"
TARGET_LOCAL_TO = 0.1247


def load():
    z = np.load(NPZ)
    inst = z["inst"].astype("int64"); day = z["day"].astype("int64")
    mod = z["mod"].astype("int64")
    y = z["y"].astype("float64"); pred = z["pred"].astype("float64")
    ud, ui, um = np.unique(day), np.unique(inst), np.unique(mod)
    A = np.full((len(ud), len(ui), len(um)), np.nan)
    Y = np.full((len(ud), len(ui), len(um)), np.nan)
    A[np.searchsorted(ud, day), np.searchsorted(ui, inst),
      np.searchsorted(um, mod)] = pred
    Y[np.searchsorted(ud, day), np.searchsorted(ui, inst),
      np.searchsorted(um, mod)] = y
    return A, Y


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


def metrics(A, Y):
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
    R = np.full((D, M, I), np.nan)
    for d in range(D):
        for m in range(M):
            R[d, m] = _pct_rank_1d(A[d, :, m])
    to = np.nanmean(np.abs(np.diff(R, axis=1)))
    return icm, icir, sharpe, to


def _zscore(A):
    Z = np.full_like(A, np.nan)
    D, I, M = A.shape
    for d in range(D):
        for m in range(M):
            col = A[d, :, m]; ok = ~np.isnan(col)
            if ok.sum() < 2:
                continue
            v = col[ok]; sd = v.std()
            Z[d, ok, m] = (v - v.mean()) / (sd if sd > 1e-12 else 1.0)
    return Z


# ---------------------------------------------------------------- 变体
def v_ema(A, a):
    out = np.full_like(A, np.nan)
    prev = A[:, :, 0].copy(); out[:, :, 0] = prev
    for m in range(1, A.shape[2]):
        cur = A[:, :, m]
        nxt = np.where(np.isnan(cur), prev, a * cur + (1 - a) * prev)
        out[:, :, m] = nxt; prev = nxt
    return out


def v_z_soft(A, theta):
    Z = _zscore(A)
    out = np.full_like(Z, np.nan)
    prev = Z[:, :, 0].copy(); out[:, :, 0] = prev
    for m in range(1, Z.shape[2]):
        cur = Z[:, :, m]
        d = cur - prev
        nxt = prev + np.sign(d) * np.maximum(0.0, np.abs(d) - theta)
        nxt = np.where(np.isnan(cur), prev, nxt)
        out[:, :, m] = nxt; prev = nxt
    return out


def v_conf_ema(A, klo, khi, zref):
    """置信度加权 EMA: |z| 越大越信任当期 (kappa 越大), 越小越平滑。

    仅用模型自身输出幅度作为置信度代理, 无需 label, 推理期可实现。
        kappa(z) = klo + (khi - klo) * min(1, |z| / zref)
    """
    Z = _zscore(A)
    out = np.full_like(Z, np.nan)
    prev = Z[:, :, 0].copy(); out[:, :, 0] = prev
    for m in range(1, Z.shape[2]):
        cur = Z[:, :, m]
        k = klo + (khi - klo) * np.minimum(1.0, np.abs(cur) / zref)
        nxt = k * cur + (1 - k) * prev
        nxt = np.where(np.isnan(cur), prev, nxt)
        out[:, :, m] = nxt; prev = nxt
    return out


def v_rank_soft(A, theta):
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
            tgt = prev[d] + np.sign(dd) * np.maximum(0.0, np.abs(dd) - theta)
            vals = np.sort(cur[ok])
            sel = ok & ~np.isnan(tgt)
            pos = np.clip(np.round(tgt[sel] * (n - 1)).astype(int), 0, n - 1)
            out[d, sel, m] = vals[pos]
            prev[d, ok] = r[ok]
    return out


def main():
    A, Y = load()
    base = metrics(A, Y)
    print(f"baseline: IC={base[0]:+.4f} IC_IR={base[1]:.3f} "
          f"Sharpe={base[2]:.3f} TO={base[3]:.4f}")
    print(f"目标: 本地 TO ≈ {TARGET_LOCAL_TO}  (公榜 0.41)\n")
    print(f"{'变体':<24} {'IC':>8} {'IC_IR':>7} {'Sharpe':>7} {'TO':>7}  "
          f"{'dIC%':>7} {'dIR%':>7} {'dSR%':>7}")
    print("-" * 84)

    rows = []

    def show(name, Ar):
        m = metrics(Ar, Y)
        rows.append((abs(m[3] - TARGET_LOCAL_TO), name, m))
        print(f"{name:<24} {m[0]:>+8.4f} {m[1]:>7.3f} {m[2]:>7.3f} {m[3]:>7.4f}  "
              f"{(m[0]/base[0]-1)*100:>+7.1f} {(m[1]/base[1]-1)*100:>+7.1f} "
              f"{(m[2]/base[2]-1)*100:>+7.1f}")

    show("baseline", A)
    print()
    for a in [0.45, 0.46, 0.47, 0.48, 0.50]:
        show(f"EMA a={a}", v_ema(A, a))
    print()
    for th in [0.40, 0.42, 0.44, 0.46, 0.48, 0.50]:
        show(f"z-soft th={th}", v_z_soft(A, th))
    print()
    for th in [0.075, 0.08, 0.085, 0.09, 0.095]:
        show(f"rank-soft th={th}", v_rank_soft(A, th))
    print()
    for klo, khi, zr in [(0.35, 0.95, 1.0), (0.30, 0.95, 1.5),
                         (0.40, 1.0, 1.0), (0.25, 0.9, 2.0)]:
        show(f"conf-ema {klo}/{khi}/{zr}", v_conf_ema(A, klo, khi, zr))

    print("\n各变体中 TO 最接近目标 (0.1247) 的:")
    for dist, name, m in sorted(rows)[:6]:
        print(f"  {name:<24} TO={m[3]:.4f} (差{dist:+.4f})  "
              f"IC={m[0]:+.4f} ({(m[0]/base[0]-1)*100:+.1f}%)  "
              f"IC_IR={m[1]:.3f} ({(m[1]/base[1]-1)*100:+.1f}%)  "
              f"Sharpe={m[2]:.3f} ({(m[2]/base[2]-1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
