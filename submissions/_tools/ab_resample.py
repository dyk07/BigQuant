"""A/B 正确性测试: 新 resample_bars 必须与旧(参考)实现**逐位一致**。

同时报告新旧实现的峰值内存差异。
"""
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


def resample_ref(df1m, kmins):
    """旧实现 (逐位参考)。"""
    from Transformer_modelsave_train import (_BOOK, _OHLC, AUCTION_MINUTE,
                                             MORNING_START)
    df = df1m.sort_values(["instrument_id", "date"], kind="mergesort")
    auction = df[df["minute_of_day"] == AUCTION_MINUTE].copy()
    cont = df[df["minute_of_day"] > AUCTION_MINUTE].copy()
    rel = cont["minute_of_day"].to_numpy() - MORNING_START
    cont["_end"] = MORNING_START + np.ceil(rel / kmins).astype("int64") * kmins
    agg = {"open": ("open", "first"), "high": ("high", "max"),
           "low": ("low", "min"), "close": ("close", "last"),
           "volume": ("volume", "sum"), "amount": ("amount", "sum"),
           "deal_number": ("deal_number", "sum")}
    for c in _BOOK:
        agg[c] = (c, "last")
    g = (cont.groupby(["instrument_id", "day", "_end"], sort=False)
             .agg(**agg).reset_index().rename(columns={"_end": "minute_of_day"}))
    cols = ["instrument_id", "minute_of_day", "day"] + list(agg.keys())
    a = auction[[c for c in cols if c in auction.columns]]
    out = pd.concat([a, g[[c for c in cols if c in g.columns]]], ignore_index=True)
    out["date"] = pd.to_datetime(out["day"]) + pd.to_timedelta(out["minute_of_day"], unit="m")
    return (out.dropna(subset=["day"])
               .sort_values(["instrument_id", "date"]).reset_index(drop=True))


if __name__ == "__main__":
    raw = pd.read_feather(FEATHER)
    raw = T._normalize(raw)
    raw = T.restore_units(raw)
    raw = T.tag_time(raw)
    print(f"[ab] bars1m={len(raw):,} peakRSS={rss():.0f}MB")

    for km in (1, 5, 15, 30):
        new = T.resample_bars(raw, km)
        ref = resample_ref(raw, km)
        assert list(new.columns) == list(ref.columns), (list(new.columns), list(ref.columns))
        assert len(new) == len(ref), f"km={km}: len {len(new)} vs {len(ref)}"
        bad = []
        for c in ref.columns:
            a, b = new[c].to_numpy(), ref[c].to_numpy()
            if a.dtype.kind == "f":
                ok = np.allclose(a, b, rtol=0, atol=0, equal_nan=True)
            else:
                ok = np.array_equal(a, b)
            if not ok:
                d = np.abs(a.astype("float64") - b.astype("float64"))
                bad.append((c, np.nanmax(d) if d.size else 0, np.sum(~(a == b) & ~(np.isnan(a) & np.isnan(b)))))
        status = "IDENTICAL" if not bad else f"DIFF {bad}"
        print(f"[ab] km={km:<2} rows={len(new):>8,} peakRSS={rss():.0f}MB  {status}")
        del new, ref
