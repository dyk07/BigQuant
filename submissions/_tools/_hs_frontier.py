"""对 highscore_merged 做软阈值扫描, 并对比 EMA, 给出换手-IC 前沿。

merged 公榜: IC 0.1105 / IC_IR 1.8938 / Sharpe 50.30 / stress 1.6059 / TO 0.6315 / score 0.74
若软阈值能以较小 IC 代价把 TO 从 0.63 压到 ~0.45, 按平台当前换手弹性
(换手从 0.675 -> 0.40 曾带来 +0.048 score) 预计仍有明显正收益。
"""
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "submissions/_tools")
from _eval_hs import build_labels, metrics, soft_within_day  # noqa: E402


def zscore_by_section(df):
    g = df.groupby("date", sort=False)["factor"]
    return ((df["factor"] - g.transform("mean"))
            / g.transform("std").replace(0, 1.0))


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


def main():
    lab = build_labels("2020-01-02", "2020-01-24")
    lab["instrument"] = lab["instrument"].astype(str)
    p = pd.read_parquet("_preds_highscore_merged.parquet")
    p["date"] = pd.to_datetime(p["date"])
    p["instrument"] = p["instrument"].astype(str)
    base = p.merge(lab, on=["date", "instrument"], how="inner")
    print(f"样本 {len(base)} 行 / {base['date'].nunique()} 截面\n")

    b = metrics(base)
    print(f"{'变体':<18} {'IC':>8} {'IC_IR':>7} {'Sharpe':>7} {'TO':>7}  "
          f"{'dIC%':>7} {'dIR%':>7} {'dSR%':>7} {'dTO%':>7}")
    print("-" * 88)
    print(f"{'baseline':<18} {b['ic']:>+8.4f} {b['ic_ir']:>7.3f} "
          f"{b['sharpe']:>7.3f} {b['turnover']:>7.4f}      -       -       -       -")

    def show(name, f):
        d = base.copy(); d["factor"] = f
        r = metrics(d)
        print(f"{name:<18} {r['ic']:>+8.4f} {r['ic_ir']:>7.3f} "
              f"{r['sharpe']:>7.3f} {r['turnover']:>7.4f}  "
              f"{(r['ic']/b['ic']-1)*100:>+7.1f} {(r['ic_ir']/b['ic_ir']-1)*100:>+7.1f} "
              f"{(r['sharpe']/b['sharpe']-1)*100:>+7.1f} "
              f"{(r['turnover']/b['turnover']-1)*100:>+7.1f}")
        return r

    print("\n--- L1 软阈值 (推荐) ---")
    for th in [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        show(f"soft th={th}", soft_within_day(base, base["factor"].to_numpy(), th))

    print("\n--- EMA (对照) ---")
    z = zscore_by_section(base).to_numpy()
    for a in [0.5, 0.6, 0.7, 0.8]:
        show(f"ema a={a}", ema_within_day(base, z, a))

    print("\n--- 无后处理但先截面 z-score (确认 z 变换本身无影响) ---")
    show("zscore only", z)

    print("\n结论要点:")
    print("  * 若 soft 在 TO 降幅 20~30% 时 IC_IR 不降, 说明这是"
          "「免费降换手」, 直接可用于 merged。")
    print("  * 与 EMA 同 TO 处比较 IC_IR, 验证 L1>L2 是否在高分模型上同样成立。")


if __name__ == "__main__":
    main()
