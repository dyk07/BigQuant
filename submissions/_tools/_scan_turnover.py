"""换手正则 λ 扫描: 关 XSEC, 独立配对 batch (100% 命中), 扫 LAMBDA_TURNOVER。

   python submissions/_tools/_scan_turnover.py [PKG] [month_count]

PKG 默认 cnn_3f_1019_newlabel_xsec_amp (已关 XSEC)。
month_count 默认 6 (smoke 规模快速扫趋势), 传 0 用 config.TRAIN_MONTHS 全量。
"""
import faulthandler
import json
import os
import resource
import sys
import time
import types

import numpy as np
import pandas as pd

faulthandler.enable()

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
PKG = sys.argv[1] if len(sys.argv) > 1 else "cnn_3f_1019_newlabel_xsec_amp"
N_MONTHS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
# 第 3 参数: 逗号分隔的 λ 列表 (默认 [0.0, 0.005, 0.02, 0.1])
LAMBDAS = ([float(x) for x in sys.argv[3].split(",")]
           if len(sys.argv) > 3 else [0.0, 0.005, 0.02, 0.1])
PKG_DIR = os.path.join(ROOT, "submissions", PKG)
os.chdir(PKG_DIR)
sys.path.insert(0, PKG_DIR)

DATA = os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar1m")
_ALL_MONTHS = [os.path.splitext(f)[0].split(".")[0]
               for f in sorted(os.listdir(DATA)) if f.endswith(".feather")]


def _query(sql, filters=None):
    """按 filters['date'] = [lo, hi] 只加载相交月份并裁剪 (先判断相交再读)。"""
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
        d = pd.read_feather(p)
        d["date"] = pd.to_datetime(d["date"])
        parts.append(d[(d["date"] >= lo) & (d["date"] <= hi)])
        del d
    if len(parts) == 1:
        return types.SimpleNamespace(df=lambda: parts[0])
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return types.SimpleNamespace(df=lambda: df)


fake_dai = types.ModuleType("dai")
fake_dai.query = _query
sys.modules["dai"] = fake_dai

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
    print(f"[scan] pkg={PKG} months={len(months)} ({months[0]}..{months[-1]}) "
          f"USE_XSEC={CFG.USE_XSEC} USE_TURNOVER={CFG.USE_TURNOVER} "
          f"lambdas={LAMBDAS}", flush=True)

    for lam in LAMBDAS:
        out_path = os.path.join(PKG_DIR, f"_scan_to_lam{lam}.json")
        t0 = time.time()
        print(f"\n===== lambda={lam} =====", flush=True)
        T.train_and_save({"bar1m": "cpt_jyc_2026_e2e_bar1m"},
                         train_months=months, model_path=out_path,
                         lam_turnover=lam)
        dt = (time.time() - t0) / 60.0
        print(f"  [scan] lambda={lam} done {dt:.1f}min peak={peak_mb():.0f}MB "
              f"-> {os.path.basename(out_path)}", flush=True)
        results.append({"lambda": lam, "min": round(dt, 1),
                        "weight": os.path.basename(out_path)})

    print("\n===== 汇总 =====", flush=True)
    for r in results:
        print(f"  lambda={r['lambda']:<6} {r['min']:>6}min  -> {r['weight']}",
              flush=True)


if __name__ == "__main__":
    main()
