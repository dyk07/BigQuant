"""全量重训驱动: 在本地 feather 上忠实模拟平台 `train_and_save(datasources)`。

与私榜重训走的路径一致:
  * datasources["bar1m"] 传表名字符串 -> 走 dai.query 路径;
  * 伪造 dai 只读取 filters 指定日期区间的行 (按需取数);
  * train_months = config.TRAIN_MONTHS (2020-01 .. 2023-12), val = 末尾 2 个月。
"""
import faulthandler
import os
import resource
import sys
import time
import types

import numpy as np
import pandas as pd

faulthandler.enable()

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
PKG = sys.argv[1] if len(sys.argv) > 1 else "cnn_3f"
PKG_DIR = os.path.join(ROOT, "submissions", PKG)
os.chdir(PKG_DIR)
sys.path.insert(0, PKG_DIR)

DATA = os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar1m")

# ---- 伪造平台 dai: 只返回 filters 指定日期区间的行 ----
_ALL_MONTHS = [os.path.splitext(f)[0].split(".")[0]
               for f in sorted(os.listdir(DATA)) if f.endswith(".feather")]


def _query(sql, filters=None):
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
    if len(parts) == 1:
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


fake_dai = types.ModuleType("dai")
fake_dai.query = _query
sys.modules["dai"] = fake_dai

import config as CFG  # noqa: E402
import Transformer_modelsave_train as T  # noqa: E402


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


if __name__ == "__main__":
    months = sorted(CFG.TRAIN_MONTHS)
    print(f"[full_train] pkg={PKG} freqs={CFG.FREQS} "
          f"months={months[0]}..{months[-1]} ({len(months)}个月) "
          f"USE_XSEC={CFG.USE_XSEC} USE_TURNOVER={CFG.USE_TURNOVER} "
          f"LAMBDA_TURNOVER={CFG.LAMBDA_TURNOVER}", flush=True)
    t0 = time.time()
    path = T.train_and_save({"bar1m": "cpt_jyc_2026_e2e_bar1m"},
                            train_months=months)
    dt = time.time() - t0
    print(f"[full_train] done in {dt/60:.1f}min  peakRSS={peak_mb():.0f}MB  "
          f"-> {path}", flush=True)
