"""跨时段稳健性检验: L1 软阈值 vs L2 EMA 是否在多个独立时段都成立。

如果 L1>L2 只在某一段成立, 那是噪声; 若三段一致, 则支持「算子与指标
同型」这一数学性质, 可以放心用于生产。
"""
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "submissions/_tools")
from _eval_hs import build_labels, metrics, soft_within_day  # noqa: E402


def ema_within_day(df, factor, alpha):
    d = pd.DataFrame({
        "_i": np.arange(len(df)),
        "inst": df["instrument"].to_numpy(),
        "day": pd.to_datetime(df["date"]).dt.normalize().to_numpy(),
        "t": pd.to_datetime(df["date"]).to_numpy(),
        "f": np.asarray(factor, dtype="float64"),
    })
    d = d.sort_values(["inst", "day", "t"], kind="mergesort")
    sm = d.groupby(["inst", "day"], sort=False, observed=True)["f"].transform(
        lambda x: x.ewm(alpha=alpha, adjust=False).mean())
    out = np.empty(len(df), dtype="float64")
    out[d["_i"].to_numpy()] = sm.to_numpy()
    return out


PERIODS = [
    ("2020-01", "_preds_highscore_merged.parquet", "2020-01-02", "2020-01-24"),
    ("2022-06", "_preds_highscore_merged_202206.parquet", "2022-06-01", "2022-07-01"),
    ("2024-11", "_preds_highscore_merged_202411.parquet", "2024-11-01", "2024-12-01"),
]


def run(tag, path, start, end):
    lab = build_labels(start, end)
    lab["instrument"] = lab["instrument"].astype(str)
    p = pd.read_parquet(path)
    p["date"] = pd.to_datetime(p["date"])
    p["instrument"] = p["instrument"].astype(str)
    base = p.merge(lab, on=["date", "instrument"], how="inner")
    if len(base) < 1000:
        print(f"[{tag}] 样本不足 ({len(base)}), 跳过")
        return None
    b = metrics(base)
    print(f"\n[{tag}] {len(base)} 行 / {base['date'].nunique()} 截面")
    print(f"{'变体':<16} {'IC':>8} {'IC_IR':>7} {'Sharpe':>7} {'TO':>7}  "
          f"{'dIC%':>7} {'dIR%':>7} {'dTO%':>7}")
    print("-" * 72)
    print(f"{'baseline':<16} {b['ic']:>+8.4f} {b['ic_ir']:>7.3f} "
          f"{b['sharpe']:>7.3f} {b['turnover']:>7.4f}      -       -       -")

    out = {"baseline": b}

    def show(name, f):
        d = base.copy(); d["factor"] = f
        r = metrics(d)
        out[name] = r
        print(f"{name:<16} {r['ic']:>+8.4f} {r['ic_ir']:>7.3f} "
              f"{r['sharpe']:>7.3f} {r['turnover']:>7.4f}  "
              f"{(r['ic']/b['ic']-1)*100:>+7.1f} "
              f"{(r['ic_ir']/b['ic_ir']-1)*100:>+7.1f} "
              f"{(r['turnover']/b['turnover']-1)*100:>+7.1f}")

    show("soft 0.25", soft_within_day(base, base["factor"].to_numpy(), 0.25))
    g = base.groupby("date", sort=False)["factor"]
    z = ((base["factor"] - g.transform("mean"))
         / g.transform("std").replace(0, 1.0)).to_numpy()
    show("ema 0.80", ema_within_day(base, z, 0.80))
    show("ema 0.70", ema_within_day(base, z, 0.70))
    return out


def main():
    res = {}
    for tag, path, s, e in PERIODS:
        try:
            r = run(tag, path, s, e)
        except FileNotFoundError:
            print(f"[{tag}] 缺少 {path}, 跳过")
            continue
        if r:
            res[tag] = r

    print("\n" + "=" * 72)
    print("跨时段汇总: 等换手附近的 IC_IR 对比 (L1 vs L2)")
    print("=" * 72)
    for tag, r in res.items():
        b = r["baseline"]
        print(f"\n[{tag}] baseline TO={b['turnover']:.4f} IC_IR={b['ic_ir']:.3f}")
        for k in ("soft 0.25", "ema 0.80", "ema 0.70"):
            if k not in r:
                continue
            x = r[k]
            print(f"  {k:<12} TO={x['turnover']:.4f} "
                  f"({(x['turnover']/b['turnover']-1)*100:+.1f}%)  "
                  f"IC_IR={x['ic_ir']:.3f} ({(x['ic_ir']/b['ic_ir']-1)*100:+.1f}%)  "
                  f"Sharpe={x['sharpe']:.3f} ({(x['sharpe']/b['sharpe']-1)*100:+.1f}%)")

    if len(res) >= 2:
        print("\n判定: 若 soft 在各时段 IC_IR 降幅都小于同换手的 ema, 则 L1>L2 稳健。")


if __name__ == "__main__":
    main()
