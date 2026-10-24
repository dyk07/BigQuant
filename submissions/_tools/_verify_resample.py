"""验证: 用训练脚本的 resample_bars 从 bar1m 合成 bar30m/bar5m/bar15m。

平台注入的 datasources 值 = 表名字符串。前 10+ 次成功提交只声明了 bar1m,
若本次提交时 datasources 里没有 bar30m/bar5m, highscore 段的
validate_datasources 会直接 KeyError -> 运行错误。

因此需要确认: 只用 bar1m 能否逐位复现原生 bar30m/bar5m (含 adjust_factor)。
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PKG = os.path.join(ROOT, "submissions", "cnn_3f_1024_fuse_hs")
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)

from _hs_harness import install_fake_dai  # noqa: E402

COLS = ["open", "high", "low", "close", "volume", "amount", "deal_number",
        "adjust_factor"]


def load(tag, month):
    d0 = os.path.join(ROOT, "e2e_data", f"cpt_jyc_2026_e2e_{tag}")
    cand = [f for f in sorted(os.listdir(d0)) if month in f]
    d = pd.read_feather(os.path.join(d0, cand[0]))
    d["date"] = pd.to_datetime(d["date"])
    return d


def main():
    install_fake_dai()
    import Transformer_modelsave_train as T

    month = "202411"
    b1 = load("bar1m", month)
    print(f"bar1m {len(b1):,} 行")

    tj = T.tag_time(b1.copy())
    g = tj.groupby(["instrument_id", "day"])["adjust_factor"].nunique()
    print(f"adjust_factor 在 (instrument,day) 内唯一值: max={g.max()}, "
          f"非唯一组={int((g > 1).sum())}/{len(g)}")
    del tj, g

    for tag, k in (("bar30m", 30), ("bar15m", 15), ("bar5m", 5)):
        native = load(tag, month)
        tj = T.tag_time(b1.copy())
        synth = T.resample_bars(tj, k)
        del tj
        nat_mod = (native["date"].dt.hour * 60 + native["date"].dt.minute)
        grid = np.array(sorted(nat_mod.unique()), dtype=np.int64)
        syn_mod = (synth["date"].dt.hour * 60 + synth["date"].dt.minute)
        extra = sorted(set(syn_mod.unique().tolist()) - set(grid.tolist()))
        synth = synth[syn_mod.isin(grid)].reset_index(drop=True)

        cols = [c for c in COLS if c in native.columns and c in synth.columns]
        a = (native[["date", "instrument_id"] + cols]
             .sort_values(["instrument_id", "date"], kind="mergesort")
             .reset_index(drop=True))
        b = (synth[["date", "instrument_id"] + cols]
             .sort_values(["instrument_id", "date"], kind="mergesort")
             .reset_index(drop=True))
        m = a.merge(b, on=["date", "instrument_id"], how="outer",
                    suffixes=("_n", "_m"), indicator=True)
        only_n = int((m["_merge"] == "left_only").sum())
        only_m = int((m["_merge"] == "right_only").sum())
        both = m[m["_merge"] == "both"]
        print(f"\n=== {tag} 原生 {len(a):,} / 合成 {len(b):,} / 共同 {len(both):,}"
              f"  仅原生 {only_n}  仅合成 {only_m}")
        if extra:
            print(f"    !! 合成多出格点(已丢): {extra}")
        bad = 0
        for c in cols:
            x = both[f"{c}_n"].to_numpy("float64")
            y = both[f"{c}_m"].to_numpy("float64")
            ok = np.isfinite(x) & np.isfinite(y)
            diff = np.abs(x[ok] - y[ok])
            mx = diff.max() if diff.size else 0.0
            mism = int((np.isnan(x) != np.isnan(y)).sum())
            if mx > 1e-9 or mism:
                bad += 1
                print(f"    !! {c:<14} max|diff|={mx:.3e} NaN不匹配={mism}")
        print("    -> 完全逐位一致" if bad == 0 else f"    -> {bad} 列不一致")
        del native, synth, a, b, m, both


if __name__ == "__main__":
    main()
