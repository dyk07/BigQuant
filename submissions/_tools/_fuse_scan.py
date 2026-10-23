"""融合权重扫描 + 换手后处理联合实验。

原则 (防止过拟合):
  1. 权重扫描只在本地验证集上做, 结论以「趋势/平台」为准, 不取单点最优;
  2. 报告权重曲线的平坦区域, 而非 argmax;
  3. 同时报告 IC 与 IC_IR 两个维度的稳定性。

本地标签窗口只有 2020-01 约 3 周 (高分模型的训练期在 2023+),
因此这里的**绝对值不可直接对标公榜**, 只用于比较相对排序。
"""
import numpy as np
import pandas as pd

HORIZON = 30


def _pct(v):
    out = np.full(len(v), np.nan)
    ok = ~np.isnan(v)
    n = ok.sum()
    if n < 2:
        return out
    x = v[ok]
    o = np.argsort(x, kind="mergesort")
    r = np.empty(n)
    r[o] = np.arange(n)
    out[ok] = r / (n - 1)
    return out


def metrics(df, label_col="y"):
    ics, sps = [], []
    for dt, g in df.groupby("date", sort=False):
        a = g["factor"].to_numpy(); y = g[label_col].to_numpy()
        n = len(g)
        if n < 20:
            continue
        ra = _pct(a) * (n - 1); ry = _pct(y) * (n - 1)
        if ra.std() < 1e-9 or ry.std() < 1e-9:
            continue
        ics.append(np.corrcoef(ra, ry)[0, 1])
        k = max(1, int(n * 0.2))
        o = np.argsort(a, kind="mergesort")
        sps.append(y[o[-k:]].mean() - y[o[:k]].mean())
    ics = np.asarray(ics); sps = np.asarray(sps)
    d = df.assign(day=df["date"].dt.normalize())
    tots = []
    for (dy, ins), g in d.groupby(["day", "instrument"], sort=False):
        if len(g) < 2:
            continue
        g = g.sort_values("date")
        tots.append(np.abs(np.diff(_pct(g["factor"].to_numpy()))))
    return dict(ic=ics.mean(), ic_ir=ics.mean() / ics.std(),
                sharpe=sps.mean() / sps.std(),
                turnover=float(np.nanmean(np.concatenate(tots))) if tots else np.nan,
                n_sec=len(ics),
                ic_std=ics.std())


def load_merged():
    a = pd.read_parquet("_preds_highscore_merged.parquet")
    b = pd.read_parquet("_preds_highscore_benchmark.parquet")
    for x in (a, b):
        x["date"] = pd.to_datetime(x["date"])
        x["instrument"] = x["instrument"].astype(str)
    return a, b


def fuse(a, b, w, method="rank"):
    """按截面融合两模型分数; method='rank' 为 pct-rank 加权。"""
    left = a.rename(columns={"factor": "pa"})
    right = b.rename(columns={"factor": "pb"})
    m = left.merge(right, on=["date", "instrument"], how="outer")
    if method == "rank":
        ra = m.groupby("date", sort=False)["pa"].rank(pct=True)
        rb = m.groupby("date", sort=False)["pb"].rank(pct=True)
        f = w * ra + (1 - w) * rb
        f = f.mask(f.isna() & ra.notna(), ra).mask(f.isna() & rb.notna(), rb)
    else:  # z-score
        def z(c):
            g = m.groupby("date", sort=False)[c]
            return (m[c] - g.transform("mean")) / g.transform("std").replace(0, 1.0)
        za, zb = z("pa"), z("pb")
        f = w * za + (1 - w) * zb
        f = f.mask(f.isna() & za.notna(), za).mask(f.isna() & zb.notna(), zb)
    m["factor"] = f
    return m[["date", "instrument", "factor"]].dropna()


def main():
    import sys
    sys.path.insert(0, "submissions/_tools")
    from _eval_hs import build_labels

    lab = build_labels("2020-01-02", "2020-01-24")
    lab["instrument"] = lab["instrument"].astype(str)
    a, b = load_merged()

    print("=== 单模型 (本地 2020-01) ===")
    print(f"{'model':<14} {'IC':>8} {'IC_IR':>7} {'Sharpe':>7} {'TO':>7} {'n':>5}")
    singles = {}
    for name, p in [("merged(自带融合)", a), ("benchmark", b)]:
        m = p.merge(lab, on=["date", "instrument"], how="inner")
        r = metrics(m)
        singles[name] = r
        print(f"{name:<14} {r['ic']:>+8.4f} {r['ic_ir']:>7.3f} "
              f"{r['sharpe']:>7.3f} {r['turnover']:>7.4f} {r['n_sec']:>5}")

    # 拆出 merged 的两个成分是平台侧行为, 本地只有融合结果, 因此这里
    # 用 "merged 融合分" 与 "benchmark" 作为两条独立信号源做二次融合。
    print("\n=== 二次融合: merged(w) + benchmark(1-w), rank 空间 ===")
    print(f"{'w':>5} {'IC':>8} {'IC_IR':>7} {'Sharpe':>7} {'TO':>7}")
    rows = []
    for w in np.arange(0, 1.01, 0.1):
        f = fuse(a, b, w)
        m = f.merge(lab, on=["date", "instrument"], how="inner")
        r = metrics(m)
        rows.append((w, r))
        print(f"{w:>5.1f} {r['ic']:>+8.4f} {r['ic_ir']:>7.3f} "
              f"{r['sharpe']:>7.3f} {r['turnover']:>7.4f}")

    print("\n=== 二次融合: z-score 空间 ===")
    print(f"{'w':>5} {'IC':>8} {'IC_IR':>7} {'Sharpe':>7} {'TO':>7}")
    for w in np.arange(0, 1.01, 0.2):
        f = fuse(a, b, w, method="z")
        m = f.merge(lab, on=["date", "instrument"], how="inner")
        r = metrics(m)
        print(f"{w:>5.1f} {r['ic']:>+8.4f} {r['ic_ir']:>7.3f} "
              f"{r['sharpe']:>7.3f} {r['turnover']:>7.4f}")

    print("\n=== IC / IC_IR 的权重敏感度 (判断是否平坦) ===")
    ics = np.array([r["ic"] for _, r in rows])
    irs = np.array([r["ic_ir"] for _, r in rows])
    ws = np.array([w for w, _ in rows])
    print(f"  IC    : max={ics.max():.4f} @ w={ws[ics.argmax()]:.1f}, "
          f"范围 [{ics.min():.4f}, {ics.max():.4f}], 极差 {np.ptp(ics):.4f}")
    print(f"  IC_IR : max={irs.max():.3f} @ w={ws[irs.argmax()]:.1f}, "
          f"范围 [{irs.min():.3f}, {irs.max():.3f}], 极差 {np.ptp(irs):.3f}")
    plateau = ws[ics >= ics.max() - 0.001]
    print(f"  IC 在最优 0.001 内的 w 区间: [{plateau.min():.1f}, {plateau.max():.1f}]"
          f"  (宽度 {plateau.max()-plateau.min():.1f})")
    print(f"  单模型 merged(w=1.0) IC={ics[-1]:.4f}  vs  融合最优 {ics.max():.4f}"
          f"  -> 净增益 {ics.max()-ics[-1]:+.4f}")

    print("\n=== 信号相关性 (判断融合是否还有信息增益) ===")
    j = a.rename(columns={"factor": "pa"}).merge(
        b.rename(columns={"factor": "pb"}), on=["date", "instrument"], how="inner")
    ra = j.groupby("date", sort=False)["pa"].rank(pct=True)
    rb = j.groupby("date", sort=False)["pb"].rank(pct=True)
    print(f"  逐截面 rank 相关均值: {np.corrcoef(ra, rb)[0,1]:.4f}  "
          f"(重叠样本 {len(j)})")
    per = j.assign(ra=ra, rb=rb).groupby("date").apply(
        lambda g: np.corrcoef(g["ra"], g["rb"])[0, 1] if len(g) > 20 else np.nan)
    print(f"  截面相关分布: mean={per.mean():.3f}, "
          f"min={per.min():.3f}, max={per.max():.3f}")
    print("  -> 相关度接近 1 说明两模型高度同质, 融合无法带来独立信息")


if __name__ == "__main__":
    main()
