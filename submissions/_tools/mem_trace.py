"""逐阶段 RSS 追踪: 找出 _build_month 里真正的内存峰值来源。"""
import os
import resource
import sys

import numpy as np
import pandas as pd

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
PKG = sys.argv[1] if len(sys.argv) > 1 else "cnn_2f"
PKG_DIR = os.path.join(ROOT, "submissions", PKG)
os.chdir(PKG_DIR)
sys.path.insert(0, PKG_DIR)

import config as CFG  # noqa: E402
import Transformer_modelsave_train as T  # noqa: E402

FEATHER = os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar1m", "202412.0.feather")


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def cur():
    """当前常驻内存 (VmRSS) —— 比 ru_maxrss 更适合看"现在占用"。"""
    with open("/proc/self/status") as fh:
        for ln in fh:
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) / 1024.0
    return -1


def mark(label):
    print(f"  {label:<34} cur={cur():8.0f}MB peak={rss():8.0f}MB")


if __name__ == "__main__":
    print(f"pkg={PKG} freqs={CFG.FREQS}")
    mark("start")
    df = pd.read_feather(FEATHER)
    mark(f"read feather ({len(df):,}x{df.shape[1]})")
    df["date"] = pd.to_datetime(df["date"])
    mark("to_datetime(date)")

    keep = [c for c in T.NEEDED_COLS if c in df.columns]
    df2 = df[keep]
    mark(f"pick_cols -> {len(keep)} cols")
    del df
    mark("del original df")

    df2 = T.restore_units(df2)
    mark("restore_units")
    df2 = T.tag_time(df2)
    mark("tag_time (filter >= 565)")

    for f in CFG.FREQS:
        km = T.FREQ_MINUTES[f]
        bars = T.resample_bars(df2, km)
        mark(f"resample {f} -> {len(bars):,} bars")
        feat = T.build_features(bars)
        mark(f"build_features {f} -> {feat.shape}")
        lab = T.build_labels(bars, max(1, T.HORIZON_MINUTES // km))
        mark(f"build_labels {f}")
        sel, y, keys = T.make_anchors(feat, lab, bars.reset_index(drop=True),
                                     max(1, CFG.LOOKBACK_MINUTES // km),
                                     T.ALIGNED_MINUTES, True)
        mark(f"make_anchors {f} -> N={len(sel):,}")
        del lab
        X = T.gather_windows(feat, sel, max(1, CFG.LOOKBACK_MINUTES // km))
        mark(f"gather_windows {f} -> {X.shape} ({X.nbytes/1e6:.0f}MB)")
        del X
        del bars, feat, sel, y, keys
    mark("end")
