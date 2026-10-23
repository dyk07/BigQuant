"""跨时段验证「二次融合已饱和」这一结论是否稳健。

若在 3 个独立时段 w 的最优值都落在 1.0 (纯 merged) 或净增益≈0,
说明「不要再调融合权重」是结构性结论, 而非单期偶然。
"""
import glob
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "submissions/_tools")
from _eval_hs import build_labels, metrics  # noqa: E402

PERIODS = [
    ("2020-01", "2020-01-02", "2020-01-24"),
    ("2022-06", "2022-06-01", "2022-07-01"),
    ("2024-11", "2024-11-01", "2024-12-01"),
]
CACHE = {}


def load_pair(tag):
    if tag in CACHE:
        return CACHE[tag]
    stem = tag.replace("-", "")
    a = pd.read_parquet(f"_preds_highscore_merged_{stem}.parquet")
    b = pd.read_parquet(f"_preds_highscore_benchmark_{stem}.parquet")
    for x in (a, b):
        x["date"] = pd.to_datetime(x["date"])
        x["instrument"] = x["instrument"].astype(str)
    CACHE[tag] = (a, b)
    return a, b


def fuse(a, b, w):
    m = (a.rename(columns={"factor": "pa"})
         .merge(b.rename(columns={"factor": "pb"}),
                on=["date", "instrument"], how="outer"))
    ra = m.groupby("date", sort=False)["pa"].rank(pct=True)
    rb = m.groupby("date", sort=False)["pb"].rank(pct=True)
    f = w * ra + (1 - w) * rb
    f = f.mask(f.isna() & ra.notna(), ra).mask(f.isna() & rb.notna(), rb)
    m["factor"] = f
    return m[["date", "instrument", "factor"]].dropna()


def main():
    print("=" * 78)
    print("跨时段: 二次融合权重扫描 (merged 权重 w, benchmark 权重 1-w)")
    print("=" * 78)
    summary = []
    for tag, s, e in PERIODS:
        lab = build_labels(s, e)
        lab["instrument"] = lab["instrument"].astype(str)
        a, b = load_pair(tag)

        print(f"\n[{tag}]")
        print(f"{'w':>5} {'IC':>8} {'IC_IR':>7} {'Sharpe':>7} {'TO':>7}")
        ics, irs = [], []
        for w in np.arange(0, 1.01, 0.25):
            m = fuse(a, b, w).merge(lab, on=["date", "instrument"], how="inner")
            r = metrics(m)
            ics.append(r["ic"]); irs.append(r["ic_ir"])
            print(f"{w:>5.2f} {r['ic']:>+8.4f} {r['ic_ir']:>7.3f} "
                  f"{r['sharpe']:>7.3f} {r['turnover']:>7.4f}")
        ics = np.array(ics)
        best = ics.max()
        summary.append((tag, ics[-1], best, best - ics[-1]))

    print("\n" + "=" * 78)
    print("汇总: 纯 merged vs 融合最优")
    print("=" * 78)
    print(f"{'period':<10} {'IC(w=1.0)':>11} {'IC(best)':>10} {'净增益':>10}")
    for tag, pure, best, gain in summary:
        print(f"{tag:<10} {pure:>+11.4f} {best:>+10.4f} {gain:>+10.4f}")
    gains = [g for _, _, _, g in summary]
    print(f"\n净增益范围: [{min(gains):+.4f}, {max(gains):+.4f}]  "
          f"均值 {np.mean(gains):+.4f}")
    if all(abs(g) < 0.002 for g in gains):
        print("判定: 各时段净增益均 < 0.002 -> **融合已饱和, 不应再调权重**")
    else:
        print("判定: 存在时段性增益, 需要更多时段确认")


if __name__ == "__main__":
    main()
