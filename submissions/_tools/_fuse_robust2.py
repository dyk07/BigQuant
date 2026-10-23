"""跨期稳健性: 自有模型 + merged 的融合增益是否稳定。

关键判断「样本内外」的数据驱动方法:
  若某期对 merged 是样本内、对 ours 是样本外, 则该期 merged 应显著强于 ours。
  若两者接近, 说明该期对双方都相对公平, 可用来评估融合。

抽样: 2020-08 (双方都可能见过) / 2022-03 / 2024-H2 (已测)。
"""
import hashlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "submissions/_tools")
from _cross_fuse_test import build_labels, metrics  # noqa: E402
from _final_fuse import fuse, ours_to_raw  # noqa: E402


def main():
    o = pd.read_parquet("_preds_ours_2024H2.parquet")
    m = pd.read_parquet("_preds_merged_2024H2.parquet")
    for x in (o, m):
        x["date"] = pd.to_datetime(x["date"])
    lab = build_labels()
    lab["instrument"] = lab["instrument"].astype(str)
    raw = sorted(set(m["instrument"].astype(str).unique()))
    o = ours_to_raw(o, raw)
    m["instrument"] = m["instrument"].astype(str)

    # 按月切片, 逐月看两模型相对强弱与融合增益
    lab["ym"] = lab["date"].dt.to_period("M")
    print(f"{'month':<9} {'ours_IC':>9} {'merged_IC':>10} {'fuse0.4':>9} "
          f"{'增益':>8} {'相关':>7} {'n':>6}")
    print("-" * 66)
    gains, corrs = [], []
    for ym, gl in lab.groupby("ym", sort=True):
        lo = gl.merge(o, on=["date", "instrument"], how="inner")
        lm = gl.merge(m, on=["date", "instrument"], how="inner")
        if len(lo) < 3000 or len(lm) < 3000:
            continue
        jj = (o.rename(columns={"factor": "po"})
              .merge(m.rename(columns={"factor": "pm"}), on=["date", "instrument"]))
        jj = jj[jj["date"].dt.to_period("M") == ym]
        if len(jj) < 3000:
            continue
        xa = jj.groupby("date", sort=False)["po"].rank(pct=True)
        xb = jj.groupby("date", sort=False)["pm"].rank(pct=True)
        corr = np.corrcoef(xa, xb)[0, 1]

        ro, rm = metrics(lo), metrics(lm)
        f = fuse(o, m, 0.4)
        lf = gl.merge(f, on=["date", "instrument"], how="inner")
        rf = metrics(lf)
        gain = rf["ic"] - max(ro["ic"], rm["ic"])
        gains.append(gain); corrs.append(corr)
        print(f"{str(ym):<9} {ro['ic']:>+9.4f} {rm['ic']:>+10.4f} {rf['ic']:>+9.4f} "
              f"{gain:>+8.4f} {corr:>7.3f} {rf['n']:>6}")

    print("-" * 66)
    print(f"融合增益: mean={np.mean(gains):+.4f} min={min(gains):+.4f} "
          f"max={max(gains):+.4f}  正增益月数 {sum(g>0 for g in gains)}/{len(gains)}")
    print(f"截面相关: mean={np.mean(corrs):.3f} 范围 "
          f"[{min(corrs):.3f}, {max(corrs):.3f}]")
    print("\n判定:")
    print("  * 若每月增益均为正 -> 融合稳健, 不是单月偶然")
    print("  * 若 ours 与 merged 各月交替领先 -> 双方均无系统性样本内优势")


if __name__ == "__main__":
    main()
