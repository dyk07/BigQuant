"""全量数据构建内存测试: 度量私榜重训中**内存占主导**的阶段。

私榜会调用 `train_and_save(datasources)` 在隔离环境从零重训 (默认 48 个月)。
真正决定峰值的是 `_accumulate` 累积的 "频率特征矩阵 + 锚点" (而不是训练 step,
step 的窗口是按 batch 现场 gather 的)。

本脚本:
  1. 逐月流式走完 config.TRAIN_MONTHS 的全部月份 (默认 2020-01..2023-12);
  2. 在每个阶段打印 VmRSS(当前) 与 ru_maxrss(峰值);
  3. 输出最终特征矩阵总大小与相对 51200MB 的余量。

不训练、不评估 —— 只回答 "会不会超 51200MB"。
"""
import faulthandler
import gc
import os
import resource
import sys
import time
import types

import numpy as np
import pandas as pd

faulthandler.enable()

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
PKG = sys.argv[1] if len(sys.argv) > 1 else "cnn_2f"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0     # 0 = config.TRAIN_MONTHS
PKG_DIR = os.path.join(ROOT, "submissions", PKG)
os.chdir(PKG_DIR)
sys.path.insert(0, PKG_DIR)

DATA = os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar1m")

# ---- 伪造平台 dai: 只返回 filters 指定区间 (与平台按需取数一致) ----
def _query(sql, filters=None):
    """忠实模拟平台: 单月查询直接返回当月 DataFrame (不额外 concat 拷贝)。"""
    assert filters and "date" in filters, "平台查询必须带日期过滤"
    lo, hi = pd.Timestamp(filters["date"][0]), pd.Timestamp(filters["date"][1])
    parts = []
    for m in _ALL_MONTHS:
        p = os.path.join(DATA, f"{m}.0.feather")
        if not os.path.exists(p):
            continue
        ms = pd.Period(m, "M")
        if ms.end_time < lo or ms.start_time > hi:
            continue
        parts.append(p)
    if len(parts) == 1:                       # 单月: 直接返回, 无 concat 拷贝
        d = pd.read_feather(parts[0])
        d["date"] = pd.to_datetime(d["date"])
        return types.SimpleNamespace(df=lambda: d)
    out = []
    for p in parts:
        d = pd.read_feather(p)
        d["date"] = pd.to_datetime(d["date"])
        out.append(d[(d["date"] >= lo) & (d["date"] <= hi)])
        del d
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return types.SimpleNamespace(df=lambda: df)


_ALL_MONTHS = [os.path.splitext(f)[0].split(".")[0]
               for f in sorted(os.listdir(DATA)) if f.endswith(".feather")]
fake_dai = types.ModuleType("dai")
fake_dai.query = _query
sys.modules["dai"] = fake_dai

import config as CFG  # noqa: E402
import Transformer_modelsave_train as T  # noqa: E402


def vm():
    for ln in open("/proc/self/status"):
        if ln.startswith("VmRSS:"):
            return int(ln.split()[1]) / 1024.0
    return -1.0


def peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def mark(tag):
    print(f"  {tag:<44} cur={vm():>8.0f}MB peak={peak():>8.0f}MB", flush=True)


if __name__ == "__main__":
    months = sorted(CFG.TRAIN_MONTHS)
    if LIMIT:
        months = months[-LIMIT:]
    freqs = list(CFG.FREQS)
    seq_lens = {f: max(1, CFG.LOOKBACK_MINUTES // T.FREQ_MINUTES[f]) for f in freqs}

    # 复刻 _train_freqs 的真实持有状态: tr 全部保留, 再叠加 va (两份同时在内存)
    vm_ = CFG.TRAIN_CFG["val_months"]
    val_m = months[-vm_:] if len(months) > vm_ + 1 else months[-1:]
    tr_m = [m for m in months if m not in set(val_m)]
    print(f"[full] pkg={PKG} freqs={freqs} months={months[0]}..{months[-1]} "
          f"({len(months)}个月) train={len(tr_m)} val={len(val_m)}")
    mark("start")

    t0 = time.time()
    tr = T._accumulate("cpt_jyc_2026_e2e_bar1m", tr_m, freqs, seq_lens,
                       T.ALIGNED_MINUTES)
    print(f"  [accumulate tr] {time.time()-t0:.1f}s  N={len(tr['keys']):,}")
    mark("train _accumulate 完成 (仍持有)")

    stats = {}
    for f in freqs:
        mean, std = T._fit_scaler(tr["feats"][f])
        T._scale_feat(tr["feats"][f], mean, std)
        stats[f] = (mean, std)
    mark("train scaler + 就地标准化")

    t0 = time.time()
    va = T._accumulate("cpt_jyc_2026_e2e_bar1m", val_m, freqs, seq_lens,
                       T.ALIGNED_MINUTES, stats=stats)
    print(f"  [accumulate va] {time.time()-t0:.1f}s  N={len(va['keys']):,}")
    mark("val _accumulate 完成 (tr+va 同时持有)")

    total = 0
    for f in freqs:
        mb = (tr["feats"][f].nbytes + va["feats"][f].nbytes) / 1e6
        total += mb
        print(f"    feats[{f:<7}] tr={tr['feats'][f].shape} va={va['feats'][f].shape}"
              f" = {mb:>8.1f}MB")
    print(f"    feats 总计 = {total:.1f}MB")
    print(f"    anchors: tr={len(tr['keys']):,} va={len(va['keys']):,}")

    ytr = T._rank_targets(tr["y"], tr["keys"])
    mark(f"rank 目标 ytr={len(ytr):,}")

    pb = 51200 - peak()
    print(f"[full] 峰值={peak():.0f}MB  限制=51200MB  余量={pb:.0f}MB")
    print("OK" if pb > 0 else "!!! 超限 !!!")
