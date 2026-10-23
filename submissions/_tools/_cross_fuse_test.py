"""决定性实验: 「我们的模型」与「highscore_merged」的截面相关性 + 融合增益。

此前只验证过 merged+benchmark (相关 0.83, 融合饱和)。但我们的模型架构/特征/
训练方式都不同, 相关度可能远低, 融合收益可能显著。

样本期: 2024-07 ~ 2024-12 —— 完全在自有模型训练范围 (2020-2023) 之外。
"""
import faulthandler
import os
import sys
import types

import numpy as np
import pandas as pd

faulthandler.enable()

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
DATA = os.path.join(ROOT, "e2e_data")
MONTHS = ["202407", "202408", "202409", "202410", "202411", "202412"]
DECISION_HHMM = (930, 1000, 1030, 1100, 1130, 1330, 1400, 1430)
HORIZON = 30


# ---------------------------------------------------------------- dai 替身
_TABS = {}


def _load_tab(tag, lo, hi):
    key = (tag, str(lo), str(hi))
    if key in _TABS:
        return _TABS[key]
    d0 = os.path.join(DATA, f"cpt_jyc_2026_e2e_{tag}")
    parts = []
    for f in sorted(os.listdir(d0)):
        if not f.endswith(".feather"):
            continue
        d = pd.read_feather(os.path.join(d0, f))
        d["date"] = pd.to_datetime(d["date"])
        if d["date"].max() < lo or d["date"].min() >= hi:
            del d
            continue
        parts.append(d[(d["date"] >= lo) & (d["date"] < hi)])
        del d
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    _TABS[key] = out
    return out


def _query(sql, filters=None, compression=None):
    filters = filters or {}
    tag = "bar30m"
    for k in ("bar1m", "bar5m", "bar15m", "bar30m"):
        if k in sql:
            tag = k
            break
    dr = filters.get("date")
    lo, hi = pd.Timestamp(dr[0]), pd.Timestamp(dr[1]) + pd.Timedelta(days=1)
    if "cpt_jyc_2026_instruments" in sql:
        b = _load_tab("bar30m", lo, hi)[["date", "instrument_id"]].copy()
        hh = b["date"].dt.hour * 100 + b["date"].dt.minute
        b = b[hh.isin(DECISION_HHMM)].rename(columns={"instrument_id": "instrument"})
        b["instrument"] = b["instrument"].astype(str)
        return types.SimpleNamespace(
            df=lambda: b.drop_duplicates(["date", "instrument"]).reset_index(drop=True))
    d = _load_tab(tag, lo, hi)
    pool = filters.get("instrument")
    if pool is not None and len(d):
        d = d[d["instrument_id"].astype(str).isin(set(map(str, pool)))]
    d = d.copy()
    d["instrument"] = d["instrument_id"].astype(str)
    keep = ["date", "instrument", "adjust_factor"] + [
        c for c in d.columns if c not in ("date", "instrument", "adjust_factor",
                                          "instrument_id")]
    return types.SimpleNamespace(df=lambda: d[keep].reset_index(drop=True))


def install_dai():
    m = types.ModuleType("dai")
    m.query = _query
    sys.modules["dai"] = m


# ---------------------------------------------------------------- 标签
def build_labels():
    lo = pd.Timestamp("2024-07-01") - pd.Timedelta(days=2)
    hi = pd.Timestamp("2025-01-01")
    b = _load_tab("bar1m", lo, hi)
    b = b[["date", "instrument_id", "close", "volume", "amount"]].copy()
    b = b.rename(columns={"instrument_id": "instrument"})
    b["instrument"] = b["instrument"].astype(str)
    b["vwap"] = np.where(b["volume"] > 0, b["amount"] / b["volume"], b["close"])
    b = b.sort_values(["instrument", "date"], kind="mergesort").reset_index(drop=True)
    g = b.groupby("instrument", sort=False)
    fa = g["amount"].transform(lambda s: s.rolling(HORIZON, min_periods=1).sum().shift(-HORIZON))
    fv = g["volume"].transform(lambda s: s.rolling(HORIZON, min_periods=1).sum().shift(-HORIZON))
    fe = g["close"].transform(lambda s: s.shift(-HORIZON))
    fw = np.where(fv > 0, fa / fv, np.nan)
    b["y"] = fe / fw - 1.0
    hh = b["date"].dt.hour * 100 + b["date"].dt.minute
    lab = b[hh.isin(DECISION_HHMM)][["date", "instrument", "y"]]
    return lab.replace([np.inf, -np.inf], np.nan).dropna(subset=["y"])


# ---------------------------------------------------------------- 指标
def _pct(v):
    out = np.full(len(v), np.nan)
    ok = ~np.isnan(v)
    n = ok.sum()
    if n < 2:
        return out
    x = v[ok]
    o = np.argsort(x, kind="mergesort")
    r = np.empty(n)
    r[o] = np.arange(n)
    out[ok] = r / (n - 1)
    return out


def metrics(df):
    ics, sps = [], []
    for dt, g in df.groupby("date", sort=False):
        a = g["factor"].to_numpy(); y = g["y"].to_numpy()
        n = len(g)
        if n < 20:
            continue
        ra = _pct(a) * (n - 1); ry = _pct(y) * (n - 1)
        if ra.std() < 1e-9 or ry.std() < 1e-9:
            continue
        ics.append(np.corrcoef(ra, ry)[0, 1])
        k = max(1, int(n * 0.2))
        o = np.argsort(a, kind="mergesort")
        sps.append(y[o[-k:]].mean() - y[o[:k]].mean())
    ics = np.asarray(ics); sps = np.asarray(sps)
    d = df.assign(day=df["date"].dt.normalize())
    tots = []
    for (dy, ins), g in d.groupby(["day", "instrument"], sort=False):
        if len(g) < 2:
            continue
        g = g.sort_values("date")
        tots.append(np.abs(np.diff(_pct(g["factor"].to_numpy()))))
    return dict(ic=ics.mean(), ic_ir=ics.mean() / ics.std(),
                sharpe=sps.mean() / sps.std(),
                turnover=float(np.nanmean(np.concatenate(tots))) if tots else np.nan,
                n=len(ics))


# ---------------------------------------------------------------- 两个模型
def run_ours():
    """自有模型 (基线权重) 在 2024-07~12 上推理, 取原始分数 (无后处理)。"""
    pkg = os.path.join(ROOT, "submissions", "cnn_3f_1023_soft_post")
    cwd = os.getcwd()
    os.chdir(pkg)
    sys.path.insert(0, pkg)
    import torch
    import Transformer_modelsave_train as T
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = T.load_model(T.MODEL_PATH, map_location=device)
    freqs, seq_lens = ck["freqs"], ck["seq_lens"]
    stats = {f: (np.asarray(ck["mean"][f], np.float32),
                 np.asarray(ck["std"][f], np.float32)) for f in freqs}
    net = T.build_model(ck["model_cfg"]).to(device)
    net.load_state_dict(ck["state_dict"]); net.eval()
    va = T._accumulate("cpt_jyc_2026_e2e_bar1m", MONTHS, freqs, seq_lens,
                       ck["aligned_minutes"], stats=stats)
    pr = T._predict_all(net, va["feats"], va["anchors"], seq_lens, freqs,
                        va["keys"], device)
    keys = va["keys"]
    mod = keys[:, 2].copy()
    mod[mod == T.AUCTION_MINUTE] = T.OPENING_SECTION
    # keys 里的 instrument 是本地 int16 枚举原值(整数原样保留), 而 highscore
    # 用字符串代码。这里还原成字符串以便跨模型 join。
    df = pd.DataFrame({
        "date": pd.to_datetime(keys[:, 1], unit="D") + pd.to_timedelta(mod, unit="m"),
        "instrument": keys[:, 0].astype("int64").astype(str),
        "factor": pr.astype("float64"),
    })
    os.chdir(cwd)
    return df


def run_merged():
    """highscore_merged 在 2024-07~12 上推理。"""
    pkg = os.path.join(ROOT, "submissions", "highscore_merged")
    cwd = os.getcwd()
    os.chdir(pkg)
    sys.path.insert(0, pkg)
    import importlib.util
    import re
    from pathlib import Path
    # 文件名 (n) 后缀 -> 内存重定向
    al = {}
    for f in os.listdir(pkg):
        m = re.match(r"^(.*?)\((\d+)\)(\.[A-Za-z0-9]+)$", f)
        if m and not os.path.exists(os.path.join(pkg, m.group(1) + m.group(3))):
            al[m.group(1) + m.group(3)] = f
    o_rt, o_rb = Path.read_text, Path.read_bytes

    def _p(self):
        p = Path(self)
        if str(p.parent) == pkg and p.name in al:
            return p.parent / al[p.name]
        return p
    Path.read_text = lambda self, *a, **k: o_rt(_p(self), *a, **k)
    Path.read_bytes = lambda self, *a, **k: o_rb(_p(self), *a, **k)
    spec = importlib.util.spec_from_file_location("hs_pred", os.path.join(pkg, "predict.py"))
    # model.py 实际文件名是 model(1).py: 先按原名注册成模块再加载 predict
    mspec = importlib.util.spec_from_file_location(
        "model", os.path.join(pkg, "model(1).py"))
    mmod = importlib.util.module_from_spec(mspec)
    sys.modules["model"] = mmod
    mspec.loader.exec_module(mmod)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.main({k: k for k in ("bar1m", "bar5m", "bar15m", "bar30m")},
                   "2024-07-01", "2024-12-31")
    Path.read_text, Path.read_bytes = o_rt, o_rb
    out = out.copy()
    out["instrument"] = out["instrument"].astype(str)
    os.chdir(cwd)
    return out


def main():
    install_dai()
    lab = build_labels()
    print(f"labels: {len(lab)} 行 / {lab['date'].nunique()} 截面", flush=True)

    print("\n[1/2] 自有模型推理 2024-07~12 ...", flush=True)
    if os.path.exists("_preds_ours_2024H2.parquet"):
        o = pd.read_parquet("_preds_ours_2024H2.parquet")
        o["date"] = pd.to_datetime(o["date"])
        o["instrument"] = o["instrument"].astype(str)
        print(f"  (缓存) ours rows={len(o)} dates={o['date'].nunique()}", flush=True)
    else:
        o = run_ours()
        o.to_parquet("_preds_ours_2024H2.parquet", index=False)
        print(f"  ours rows={len(o)} dates={o['date'].nunique()}", flush=True)

    print("\n[2/2] highscore_merged 推理 2024-07~12 ...", flush=True)
    if os.path.exists("_preds_merged_2024H2.parquet"):
        m = pd.read_parquet("_preds_merged_2024H2.parquet")
        m["date"] = pd.to_datetime(m["date"])
        m["instrument"] = m["instrument"].astype(str)
        print(f"  (缓存) merged rows={len(m)} dates={m['date'].nunique()}", flush=True)
    else:
        m = run_merged()
        m.to_parquet("_preds_merged_2024H2.parquet", index=False)
        print(f"  merged rows={len(m)} dates={m['date'].nunique()}", flush=True)

    lo = lab.merge(o, on=["date", "instrument"], how="inner")
    lm = lab.merge(m, on=["date", "instrument"], how="inner")
    print(f"\n对齐样本: ours={len(lo)}, merged={len(lm)}")
    ro, rm = metrics(lo), metrics(lm)
    print(f"\n{'model':<20} {'IC':>8} {'IC_IR':>7} {'Sharpe':>7} {'TO':>7} {'n':>5}")
    print("-" * 60)
    for nm, r in (("ours(本地)", ro), ("highscore_merged", rm)):
        print(f"{nm:<20} {r['ic']:>+8.4f} {r['ic_ir']:>7.3f} "
              f"{r['sharpe']:>7.3f} {r['turnover']:>7.4f} {r['n']:>5}")

    # ---- 关键: 截面相关性 ----
    j = (o.rename(columns={"factor": "po"})
         .merge(m.rename(columns={"factor": "pm"}), on=["date", "instrument"]))
    ra = j.groupby("date", sort=False)["po"].rank(pct=True)
    rb = j.groupby("date", sort=False)["pm"].rank(pct=True)
    per = j.assign(a=ra, b=rb).groupby("date").apply(
        lambda g: np.corrcoef(g["a"], g["b"])[0, 1] if len(g) > 20 else np.nan)
    print(f"\n【核心】ours vs merged 逐截面 rank 相关: "
          f"整体 {np.corrcoef(ra, rb)[0,1]:.4f}, "
          f"各截面 mean={per.mean():.3f} min={per.min():.3f} max={per.max():.3f}")
    print("  参考: merged vs benchmark 为 0.83 (已饱和)。"
          "若此处显著更低, 融合有真实增益。")


if __name__ == "__main__":
    main()
