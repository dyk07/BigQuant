"""训练路径冒烟测试: 跑通 train_and_save(短区间) 并报告峰值内存。

用途: 验证 `_accumulate` / `_batch_windows` / `_fit_scaler` / `train_and_save`
链路 (即**私榜重训**走的路径), 并确认峰值 RSS 远低于平台 51200MB 限制。

忠实模拟平台:
  * datasources["bar1m"] 传**表名字符串** (与平台一致), 因此走真正的
    `dai.query(sql, filters={"date": [lo, hi]}).df()` 路径;
  * 伪造的 dai 只读取 filters 指定的日期区间, 复现平台的**按需取数**行为。
"""
import faulthandler
import os
import resource
import sys
import time
import types

import numpy as np
import pandas as pd

# 打印所有线程栈 + 段错误回溯
faulthandler.enable()

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
PKG = sys.argv[1] if len(sys.argv) > 1 else "cnn_2f"
PKG_DIR = os.path.join(ROOT, "submissions", PKG)
os.chdir(PKG_DIR)
sys.path.insert(0, PKG_DIR)

DATA = os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar1m")
MONTHS = ["202401", "202402", "202403", "202404", "202405", "202406"]

# ---- 伪造平台 dai: 只返回 filters 指定日期区间的行 (与平台按需取数一致) ----
_CACHE = {}


def _load_month(m):
    if m not in _CACHE:
        p = os.path.join(DATA, f"{m}.0.feather")
        if not os.path.exists(p):
            _CACHE[m] = None
        else:
            d = pd.read_feather(p)
            d["date"] = pd.to_datetime(d["date"])
            _CACHE[m] = d
    return _CACHE[m]


def _query(sql, filters=None):
    """按 filters['date'] = [lo, hi] 只加载相交月份并裁剪。"""
    assert filters and "date" in filters, "平台查询必须带日期过滤"
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
        _CACHE.pop(m, None)          # 模拟平台: 取完即释放服务端侧缓存
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return types.SimpleNamespace(df=lambda: df)


fake_dai = types.ModuleType("dai")
fake_dai.query = _query
sys.modules["dai"] = fake_dai

import config as CFG  # noqa: E402
import Transformer_modelsave_train as T  # noqa: E402


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


if __name__ == "__main__":
    print(f"[smoke] pkg={PKG} freqs={CFG.FREQS} months={MONTHS}")
    print(f"[smoke] start peakRSS={rss_mb():.0f}MB")
    t0 = time.time()
    T.train_and_save({"bar1m": "cpt_jyc_2026_e2e_bar1m"}, train_months=MONTHS,
                     model_path="_tmp_smoke_model.json")
    dt = time.time() - t0
    print(f"[smoke] done in {dt:.1f}s  peakRSS={rss_mb():.0f}MB")
    print(f"[smoke] limit=51200MB -> headroom={51200 - rss_mb():.0f}MB")
    os.remove("_tmp_smoke_model.json")
