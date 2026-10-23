"""本地 harness: 让 highscore_merged / highscore_benchmark 在本地 feather 数据上跑通。

平台用 dai.query(TABLE) 取成分股截面表, 本地无此表 -> 用 bar30m 的
实际格点合成 (DECISION_HHMM 中的时刻), 与平台语义一致。

用途:
  1. 得到这两个高分模型在本地验证集上的 IC/IC_IR/Sharp/TO
  2. 为融合权重扫描、换手后处理提供可度量的实验台
"""
import importlib.util
import os
import sys
import types

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
DATA = {
    "bar30m": os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar30m"),
    "bar5m": os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar5m"),
    "bar1m": os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar1m"),
}
DECISION_HHMM = (930, 1000, 1030, 1100, 1130, 1330, 1400, 1430)

_CACHE = {}


def _load_table(tag, start, end):
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    key = (tag, str(lo), str(hi))
    if key in _CACHE:
        return _CACHE[key]
    parts = []
    d0 = DATA[tag]
    for f in sorted(os.listdir(d0)):
        if not f.endswith(".feather"):
            continue
        p = os.path.join(d0, f)
        d = pd.read_feather(p)
        d["date"] = pd.to_datetime(d["date"])
        if d["date"].max() < lo or d["date"].min() >= hi:
            del d
            continue
        parts.append(d[(d["date"] >= lo) & (d["date"] < hi)])
        del d
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    _CACHE[key] = out
    return out


def _query(sql, filters=None, compression=None):
    """最小 dai.query 替身: 解析 FROM 表名与 date/instrument 过滤。"""
    filters = filters or {}
    tag = "bar30m"
    for k in DATA:
        if k in sql:
            tag = k
            break
    dr = filters.get("date")
    if dr is None:
        raise RuntimeError("date filter required")
    lo, hi = pd.Timestamp(dr[0]), pd.Timestamp(dr[1]) + pd.Timedelta(days=1)

    if "cpt_jyc_2026_instruments" in sql:
        # 合成成分股截面表: bar30m 实际存在的决策时刻
        b = _load_table("bar30m", lo, hi)
        b = b[["date", "instrument_id"]].copy()
        hhmm = b["date"].dt.hour * 100 + b["date"].dt.minute
        b = b[hhmm.isin(DECISION_HHMM)]
        b = b.rename(columns={"instrument_id": "instrument"})
        b["instrument"] = b["instrument"].astype(str)
        return types.SimpleNamespace(
            df=lambda: b.drop_duplicates(["date", "instrument"]).reset_index(drop=True))

    d = _load_table(tag, lo, hi)
    pool = filters.get("instrument")
    if pool is not None and len(d):
        d = d[d["instrument_id"].astype(str).isin(set(map(str, pool)))]
    d = d.copy()
    d["instrument"] = d["instrument_id"].astype(str)
    cols = ["date", "instrument", "adjust_factor"]
    rest = [c for c in d.columns if c not in list(cols) + ["instrument_id"]]
    return types.SimpleNamespace(df=lambda: d[cols + rest].reset_index(drop=True))


def install_fake_dai():
    m = types.ModuleType("dai")
    m.query = _query
    sys.modules["dai"] = m
    return m


def _alias_dupes(d):
    """平台包内文件名带 (1)/(2) 后缀, 代码里引用的是无后缀名。

    不改动原包内容, 也不用磁盘链接 (Windows 侧会读坏): 直接在内存里
    把「无后缀名」注册成对应「带后缀名」模块的别名。
    """
    import re
    for f in sorted(os.listdir(d)):
        m = re.match(r"^(.*?)\((\d+)\)(\.[A-Za-z0-9]+)$", f)
        if not m:
            continue
        base, ext, dup = m.group(1) + m.group(3), m.group(3), f
        if os.path.exists(os.path.join(d, base)):
            continue
        if ext != ".py":
            # 非 .py (如 model.json) 通过 sys.modules 无法别名, 用 setattr 打补丁
            _FILE_ALIASES.setdefault(d, {})[base] = dup
    return _FILE_ALIASES.get(d, {})


_FILE_ALIASES = {}


def _patch_path_reads(d):
    """把 Path.read_text/read_bytes 的缺失文件名重定向到带 (n) 后缀的真实文件。"""
    aliases = _alias_dupes(d)
    if not aliases:
        return
    from pathlib import Path

    orig_read_text = Path.read_text
    orig_read_bytes = Path.read_bytes

    def _redirect(self):
        p = Path(self)
        al = _FILE_ALIASES.get(str(p.parent), {})
        if p.name in al:
            return p.parent / al[p.name]
        return p

    def read_text(self, *a, **k):
        return orig_read_text(_redirect(self), *a, **k)

    def read_bytes(self, *a, **k):
        return orig_read_bytes(_redirect(self), *a, **k)

    Path.read_text = read_text
    Path.read_bytes = read_bytes


def load_pkg(pkg, module_names=("model",), predict_name="predict.py"):
    """在隔离命名空间里加载某个 highscore 包 (含 model.py 与 predict.py)。"""
    d = os.path.join(ROOT, "submissions", pkg)
    _patch_path_reads(d)
    cwd = os.getcwd()
    os.chdir(d)
    sys.path.insert(0, d)
    for name in module_names:
        for cand in (f"{name}.py", f"{name}(1).py", f"{name}(2).py"):
            p = os.path.join(d, cand)
            if os.path.exists(p):
                spec = importlib.util.spec_from_file_location(
                    f"{pkg}_{name}", p)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
                break
    pp = None
    for cand in (predict_name, predict_name.replace("predict", "predict(1)")):
        p = os.path.join(d, cand)
        if os.path.exists(p):
            pp = p
            break
    spec = importlib.util.spec_from_file_location(f"{pkg}_predict", pp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    os.chdir(cwd)
    return mod


def main():
    install_fake_dai()
    pkg = sys.argv[1] if len(sys.argv) > 1 else "highscore_benchmark"
    start = sys.argv[2] if len(sys.argv) > 2 else "2020-01-02"
    end = sys.argv[3] if len(sys.argv) > 3 else "2020-01-23"
    pn = "predict.py" if pkg == "highscore_merged" else "predict(1).py"
    m = load_pkg(pkg, ("model",), pn)
    ds = {k: k for k in ("bar1m", "bar5m", "bar30m", "bar15m")}
    print(f"[{pkg}] running {start} ~ {end} ...")
    out = m.main(ds, start, end)
    print(f"rows={len(out)} cols={list(out.columns)}")
    print(out.head())
    print("dates:", out["date"].nunique(), "inst:", out["instrument"].nunique())
    OUT = f"_preds_{pkg}.parquet"
    out.to_parquet(OUT, index=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
