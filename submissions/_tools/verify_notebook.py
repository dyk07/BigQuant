"""验证单 notebook 提交物: 顺序执行所有代码单元格 + 调用 main() 出因子。"""
import json
import os
import sys
import types

import numpy as np
import pandas as pd

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
PKG = sys.argv[1] if len(sys.argv) > 1 else "cnn_2f"
PKG_DIR = os.path.join(ROOT, "submissions", PKG)
os.chdir(PKG_DIR)
sys.path.insert(0, PKG_DIR)

# ---- 伪造平台专有模块 dai (本地无) ----
fake_dai = types.ModuleType("dai")
_INST = pd.read_feather(
    os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar1m", "202412.0.feather"),
    columns=["date", "instrument_id"]).rename(
    columns={"instrument_id": "instrument"})


def _query(sql, filters=None):
    df = _INST.copy()
    return types.SimpleNamespace(df=lambda: df)


fake_dai.query = _query
sys.modules["dai"] = fake_dai

# ---- 伪造 structlog (平台专有) ----
fake_sl = types.ModuleType("structlog")


class _L:
    def info(self, *a, **k):
        pass

    def __getattr__(self, _):
        return lambda *a, **k: None


fake_sl.get_logger = lambda *a, **k: _L()
sys.modules["structlog"] = fake_sl

# ---- 从 notebook 顺序取出代码 ----
nb = json.load(open(os.path.join(PKG_DIR, "Transformer_modelsave_predict.ipynb")))
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"[notebook] code cells = {len(code)}")

ns = {"__name__": "__main__"}
for i, c in enumerate(code):
    src = "".join(c["source"])
    exec(compile(src, f"<cell{i}>", "exec"), ns)
    first = src.strip().splitlines()[0][:60]
    print(f"  cell{i}: OK   ({first})")

# ---- 用本地 feather 作为 datasources['bar1m'] 调 main ----
df1m = pd.read_feather(
    os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar1m", "202412.0.feather"))
datasources = {"bar1m": df1m}
out = ns["main"](datasources, "2024-12-01 00:00:00", "2024-12-31 23:59:59")
print(f"\n[main] 输出 {len(out):,} 行, 列={list(out.columns)}")
print(out.head(6).to_string(index=False))
print(f"  dates={out['date'].nunique()} instruments={out['instrument'].nunique()} "
      f"factor_mean={out['factor'].mean():.4f}")
print("OK")
