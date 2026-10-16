"""按月内存趋势: 判断 `_accumulate` 的常驻内存是否随月份线性增长。

若"当前 RSS"在每个月后几乎不涨 (= 只留 114MB/月 的特征矩阵), 说明内存有界,
48 个月也安全; 若每月涨 GB 级, 说明有泄漏/分配器滞留, 必须修复。

用法: python -u full_mem_trend.py <pkg> [从末尾取N个月]
"""
import gc
import os
import resource
import sys
import time
import types

import numpy as np
import pandas as pd

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
PKG = sys.argv[1] if len(sys.argv) > 1 else "cnn_2f"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
TRIM = (len(sys.argv) > 3 and sys.argv[3] == "trim")
PKG_DIR = os.path.join(ROOT, "submissions", PKG)
os.chdir(PKG_DIR)
sys.path.insert(0, PKG_DIR)

DATA = os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar1m")
_ALL = [os.path.splitext(f)[0].split(".")[0] for f in sorted(os.listdir(DATA))
        if f.endswith(".feather")]


def _query(sql, filters=None):
    """忠实模拟平台: 直接返回当月的 DataFrame (不额外 concat 拷贝)。"""
    lo, hi = pd.Timestamp(filters["date"][0]), pd.Timestamp(filters["date"][1])
    parts = []
    for m in _ALL:
        p = os.path.join(DATA, f"{m}.0.feather")
        if not os.path.exists(p):
            continue
        ms = pd.Period(m, "M")
        if ms.end_time < lo or ms.start_time > hi:
            continue
        parts.append((m, p))
    if len(parts) == 1:                       # 单月: 直接返回, 无 concat 拷贝
        d = pd.read_feather(parts[0][1])
        d["date"] = pd.to_datetime(d["date"])
        return types.SimpleNamespace(df=lambda: d)
    out = []
    for _, p in parts:
        d = pd.read_feather(p)
        d["date"] = pd.to_datetime(d["date"])
        out.append(d[(d["date"] >= lo) & (d["date"] <= hi)])
        del d
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return types.SimpleNamespace(df=lambda: df)


fake_dai = types.ModuleType("dai")
fake_dai.query = _query
sys.modules["dai"] = fake_dai

import config as CFG  # noqa: E402
import Transformer_modelsave_train as T  # noqa: E402
from Transformer_modelsave_train import _BOOK, _OHLC  # noqa: E402


def vm():
    for ln in open("/proc/self/status"):
        if ln.startswith("VmRSS:"):
            return int(ln.split()[1]) / 1024.0
    return -1.0


def peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


if __name__ == "__main__":
    months = sorted(CFG.TRAIN_MONTHS)
    if N:                                    # N=0 -> 全部月份
        months = months[-N:]
    freqs = list(CFG.FREQS)
    seq_lens = {f: max(1, CFG.LOOKBACK_MINUTES // T.FREQ_MINUTES[f]) for f in freqs}
    print(f"[trend] pkg={PKG} freqs={freqs} months={months[0]}..{months[-1]} ({len(months)}个)")
    print(f"  start                     cur={vm():>7.0f}MB peak={peak():>7.0f}MB")

    acc = {"feats": {f: [] for f in freqs}, "anchors": {f: [] for f in freqs},
           "keys": [], "y": []}
    off = {f: 0 for f in freqs}
    for k, m in enumerate(months):
        first, last = T._month_span(m)
        rb = vm()
        t0 = time.time()
        raw = T._normalize(T._fetch_table("cpt_jyc_2026_e2e_bar1m", first, last))
        raw = T.restore_units(raw)
        raw = T.tag_time(raw)
        per = {f: T._make_freq(raw, f, int(seq_lens[f]), T.ALIGNED_MINUTES,
                               with_label=True) for f in freqs}
        d0 = np.datetime64(pd.Timestamp(first).date(), "D").astype("int64")
        d1 = np.datetime64(pd.Timestamp(last).date(), "D").astype("int64")
        for f in freqs:
            kk = per[f]["keys"]
            if len(kk):
                msk = (kk[:, 1] >= d0) & (kk[:, 1] <= d1)
                per[f] = T._subset_freq(per[f], np.where(msk)[0])
        idx = T._align_freqs(per, freqs)
        j0 = idx[freqs[0]]
        acc["keys"].append(per[freqs[0]]["keys"][j0])
        acc["y"].append(per[freqs[0]]["y"][j0])
        for f in freqs:
            acc["feats"][f].append(per[f]["feat"])
            acc["anchors"][f].append(per[f]["anchors"][idx[f]] + off[f])
            off[f] += len(per[f]["feat"])
        del raw, per
        gc.collect()
        after = vm()
        if TRIM:
            T._trim()
            print(f"  {m} raw_peak={peak():>7.0f}MB  after_gc={after:>7.0f}MB "
                  f"after_trim={vm():>7.0f}MB (Δ{after - rb:+.0f}, "
                  f"{time.time()-t0:.1f}s)", flush=True)
        else:
            print(f"  {m} raw_peak={peak():>7.0f}MB  after_month cur={after:>7.0f}MB "
                  f"(Δ{after - rb:+.0f}, {time.time()-t0:.1f}s)", flush=True)

    tot = sum(sum(a.nbytes for a in acc["feats"][f]) for f in freqs) / 1e6
    print(f"  feats 累计 = {tot:.1f}MB (N={sum(len(x) for x in acc['keys']):,})")
    print(f"  final peak={peak():.0f}MB  余量={51200 - peak():.0f}MB")
