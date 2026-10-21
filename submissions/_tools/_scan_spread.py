"""top-bottom spread 损失 λ 扫描: 关 XSEC, 扫 LAMBDA_SPREAD 强度。

   python submissions/_tools/_scan_spread.py [PKG] [month_count] [lambdas]

PKG 默认 cnn_3f_1021_spread。
month_count 默认 6 (smoke), 传 0 用 config.TRAIN_MONTHS 全量。
lambdas: 逗号分隔的 λ_spread 列表 (默认 [0.0, 0.05, 0.1, 0.2, 0.5])。
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
PKG = sys.argv[1] if len(sys.argv) > 1 else "cnn_3f_1021_spread"
N_MONTHS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
LAMBDAS = ([float(x) for x in sys.argv[3].split(",")]
           if len(sys.argv) > 3 else [0.0, 0.05, 0.1, 0.2, 0.5])
PKG_DIR = os.path.join(ROOT, "submissions", PKG)
os.chdir(PKG_DIR)
sys.path.insert(0, PKG_DIR)

DATA = os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar1m")
_ALL_MONTHS = [os.path.splitext(f)[0].split(".")[0]
               for f in sorted(os.listdir(DATA)) if f.endswith(".feather")]


def _query(sql, filters=None):
    assert filters and "date" in filters
    lo, hi = pd.Timestamp(filters["date"][0]), pd.Timestamp(filters["date"][1])
    parts = []
    for m in _ALL_MONTHS:
        p = os.path.join(DATA, f"{m}.0.feather")
        if not os.path.exists(p):
            continue
        ms = pd.Period(m, "M")
        if ms.end_time < lo or ms.start_time > hi:
            continue
        d = pd.read_feather(p)
        d["date"] = pd.to_datetime(d["date"])
        parts.append(d[(d["date"] >= lo) & (d["date"] <= hi)])
        del d
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return types.SimpleNamespace(df=lambda: df)


fake_dai = types.ModuleType("dai")
fake_dai.query = _query
sys.modules["dai"] = fake_dai

import torch  # noqa: E402
import config as CFG  # noqa: E402
import Transformer_modelsave_train as T  # noqa: E402


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    if N_MONTHS > 0:
        months = sorted(CFG.TRAIN_MONTHS)[-N_MONTHS:]
    else:
        months = sorted(CFG.TRAIN_MONTHS)

    results = []
    print(f"[scan-spread] pkg={PKG} months={len(months)} "
          f"({months[0]}..{months[-1]}) lambdas={LAMBDAS} tau={CFG.SPREAD_TAU}",
          flush=True)

    for lam in LAMBDAS:
        out_path = os.path.join(PKG_DIR, f"_scan_spread_lam{lam}.json")
        t0 = time.time()
        print(f"\n===== lambda_spread={lam} =====", flush=True)
        T.train_and_save({"bar1m": "cpt_jyc_2026_e2e_bar1m"},
                         train_months=months, model_path=out_path,
                         lam_spread=lam)
        dt = (time.time() - t0) / 60.0
        print(f"  [scan-spread] lambda={lam} done {dt:.1f}min "
              f"peak={peak_mb():.0f}MB", flush=True)
        results.append({"lambda": lam, "min": round(dt, 1)})

    print("\n===== 汇总 =====", flush=True)
    for r in results:
        print(f"  lambda_spread={r['lambda']:<6} {r['min']:>6}min", flush=True)


if __name__ == "__main__":
    main()
