"""诊断: 定位新 resample_bars 与旧实现在 km>1 时的分组差异。

只跑 km=5, 打印:
  * 基础统计 (标的数 / 天数 / 偏移范围)
  * 新旧分组键集合差集 (哪些 (instrument, day, bucket) 只在一侧出现)
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
PKG = sys.argv[1] if len(sys.argv) > 1 else "cnn_2f"
PKG_DIR = os.path.join(ROOT, "submissions", PKG)
os.chdir(PKG_DIR)
sys.path.insert(0, PKG_DIR)

import Transformer_modelsave_train as T  # noqa: E402
from Transformer_modelsave_train import (AUCTION_MINUTE, MORNING_START,  # noqa: E402
                                         _BOOK)

KM = int(sys.argv[2]) if len(sys.argv) > 2 else 5
FEATHER = os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar1m", "202412.0.feather")


def keys_ref(df1m, kmins):
    df = df1m.sort_values(["instrument_id", "date"], kind="mergesort")
    cont = df[df["minute_of_day"] > AUCTION_MINUTE]
    rel = cont["minute_of_day"].to_numpy() - MORNING_START
    end = MORNING_START + np.ceil(rel / kmins).astype("int64") * kmins
    k = pd.DataFrame({
        "i": cont["instrument_id"].to_numpy(),
        "d": cont["day"].to_numpy(),
        "b": end,
    })
    return k, df, cont


def keys_new(df1m, kmins):
    inst = df1m["instrument_id"].to_numpy()
    ti = df1m["minute_of_day"].to_numpy().astype("int64", copy=False)
    desc = (inst[1:] < inst[:-1]) | ((inst[1:] == inst[:-1]) & (ti[1:] < ti[:-1]))
    df = df1m if not bool(desc.any()) else df1m.sort_values(
        ["instrument_id", "date"], kind="mergesort")
    inst = df["instrument_id"].to_numpy()
    ti = df["minute_of_day"].to_numpy().astype("int64", copy=False)
    is_auc = ti == AUCTION_MINUTE
    rel = ti - MORNING_START
    bucket = np.empty(len(df), "int64")
    np.floor_divide(np.maximum(rel, 0), kmins, out=bucket)
    np.multiply(bucket, kmins, out=bucket)
    np.add(bucket, MORNING_START, out=bucket)
    bucket[is_auc] = -1
    k = pd.DataFrame({"i": inst, "d": df["day"].to_numpy(), "b": bucket})
    return k, df, is_auc


if __name__ == "__main__":
    raw = pd.read_feather(FEATHER)
    raw = T._normalize(raw)
    raw = T.restore_units(raw)
    raw = T.tag_time(raw)
    print(f"[dbg] rows={len(raw):,} inst={raw['instrument_id'].nunique()} "
          f"days={raw['day'].nunique()} min_mod={raw['minute_of_day'].min()} "
          f"max_mod={raw['minute_of_day'].max()}")

    kr, _, cont = keys_ref(raw, KM)
    kn, _, is_auc = keys_new(raw, KM)
    print(f"[dbg] km={KM} ref_cont_rows={len(kr):,} new_rows={len(kn):,} "
          f"auc_rows={int(is_auc.sum())}")
    # 忽略竞价行后比较
    kn2 = kn[~is_auc].reset_index(drop=True)
    print(f"[dbg] new_non_auc_rows={len(kn2):,}")

    ur = set(map(tuple, kr[["i", "d", "b"]].to_numpy().tolist())) if False else None
    # 用字符串化避免 datetime 不可哈希问题
    sr = pd.Series(list(map(tuple, zip(kr["i"].to_numpy(), kr["d"].astype("int64").to_numpy(),
                                       kr["b"].to_numpy()))))
    sn = pd.Series(list(map(tuple, zip(kn2["i"].to_numpy(), kn2["d"].astype("int64").to_numpy(),
                                       kn2["b"].to_numpy()))))
    gr = sr.drop_duplicates()
    gn = sn.drop_duplicates()
    print(f"[dbg] ref groups={len(gr):,} new groups={len(gn):,}")
    only_new = gn[~gn.isin(set(gr))]
    only_ref = gr[~gr.isin(set(gn))]
    print(f"[dbg] only_new={len(only_new):,} only_ref={len(only_ref):,}")
    for name, s in (("only_new", only_new), ("only_ref", only_ref)):
        if len(s):
            head = s.head(5).tolist()
            print(f"[dbg] {name} sample={head}")
            # 统计这些键的 bucket 值分布
            bs = np.array([t[2] for t in head])
            print(f"[dbg] {name} bucket_vals={sorted(set(bs))}")
    # 看 offset 分布
    off = raw["minute_of_day"].to_numpy().astype("int64") - MORNING_START
    for lo, hi, tag in ((0, 120, "morning"), (180, 300, "afternoon")):
        sel = (off >= lo) & (off <= hi)
        if sel.any():
            u = np.unique(off[sel])
            print(f"[dbg] {tag} off {u.min()}..{u.max()} n_uniq={len(u)} "
                  f"head={u[:6].tolist()} tail={u[-6:].tolist()}")
