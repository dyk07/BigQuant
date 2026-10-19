"""验证换手配对命中率假设 + 独立配对 batch 的可行性。

1) 报告 (cur,next) 配对总数, 以及随机 perm 下同 batch 命中数 (应≈0.055%)。
2) 验证独立配对 batch: 直接把 cur/next 各 gather 一批, 换手 loss 100% 命中。
"""
import os
import sys
import types

import numpy as np
import pandas as pd

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
PKG = "cnn_3f"
PKG_DIR = os.path.join(ROOT, "submissions", PKG)
os.chdir(PKG_DIR)
sys.path.insert(0, PKG_DIR)

DATA = os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar1m")
MONTHS = ["202401", "202402"]

_CACHE = {}


def _load_month(m):
    if m not in _CACHE:
        p = os.path.join(DATA, f"{m}.0.feather")
        _CACHE[m] = None if not os.path.exists(p) else (
            pd.read_feather(p).assign(date=lambda d: pd.to_datetime(d["date"])))
    return _CACHE[m]


def _query(sql, filters=None):
    lo, hi = pd.Timestamp(filters["date"][0]), pd.Timestamp(filters["date"][1])
    out = []
    for m in MONTHS:
        d = _load_month(m)
        if d is None:
            continue
        ms = pd.Period(m, "M")
        if ms.end_time < lo or ms.start_time > hi:
            continue
        out.append(d[(d["date"] >= lo) & (d["date"] <= hi)])
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return types.SimpleNamespace(df=lambda: df)


fake_dai = types.ModuleType("dai")
fake_dai.query = _query
sys.modules["dai"] = fake_dai

import config as CFG  # noqa: E402
import Transformer_modelsave_train as T  # noqa: E402


def main():
    freqs = list(CFG.FREQS)
    seq_lens = {f: max(1, CFG.LOOKBACK_MINUTES // T.FREQ_MINUTES[f]) for f in freqs}
    table = fake_dai.query("", filters={"date": ["2024-01-01", "2024-02-29"]}).df()
    tr = T._accumulate(table, MONTHS, freqs, seq_lens, T.ALIGNED_MINUTES, stats=None)
    keys = tr["keys"]
    n = len(keys)
    print(f"样本数 N = {n:,}")

    nxt_all = T._turnover_next(keys)
    valid = nxt_all >= 0
    cur = np.where(valid)[0].astype("int64")
    nxt = nxt_all[valid].astype("int64")
    n_pair = len(cur)
    print(f"配对总数 = {n_pair:,}  (占比 {n_pair/n*100:.1f}%)")

    # 随机 perm 同 batch 命中率
    batch = CFG.TRAIN_CFG["batch_size"]
    perm = np.random.permutation(n)
    pos = np.empty(n, "int64")
    pos[perm] = np.arange(n)
    bcur = pos[cur] // batch
    bnxt = pos[nxt] // batch
    same = bcur == bnxt
    hit = int(same.sum())
    print(f"随机 perm 同 batch 命中 = {hit:,}  ({hit/n_pair*100:.2f}%)  "
          f"理论≈{batch/n*100:.2f}%")

    # 独立配对 batch 设计: 每 epoch 采样 k 对, cur/next 各自完整 gather
    k = min(n_pair, batch // 2)  # 每 batch 含 k 个 cur + k 个 next
    print(f"独立配对 batch 可每 epoch 采样 {k:,} 对 (batch/2), 命中率 100%")


if __name__ == "__main__":
    main()
