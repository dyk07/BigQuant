"""端到端方向验证: 用新权重在 2024H2 上算 factor vs label 截面 IC。

关键: label 已改为 end_price/vwap - 1 (与平台一致), 推理不再取负。
若 IC 为正 -> 方向正确; 若为负 -> 方向反了。
"""
import os
import sys
import types

import numpy as np
import pandas as pd
import torch

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
PKG = os.path.join(ROOT, "submissions", "cnn_3f")
os.chdir(PKG)
sys.path.insert(0, PKG)

DATA = os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar1m")
MONTHS = ["202407", "202408", "202409", "202410", "202411", "202412"]

_ALL_MONTHS = [os.path.splitext(f)[0].split(".")[0]
               for f in sorted(os.listdir(DATA)) if f.endswith(".feather")]


def _query(sql, filters=None):
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

import Transformer_modelsave_train as T  # noqa: E402

ck = T.load_model("transformer_model.json", map_location="cpu")
model = T.build_model(ck["model_cfg"]).eval()
model.load_state_dict(ck["state_dict"])
freqs = ck["freqs"]
seq_lens = ck["seq_lens"]
stats = {f: (np.asarray(ck["mean"][f], np.float32),
             np.asarray(ck["std"][f], np.float32)) for f in freqs}
minutes = ck["aligned_minutes"]
use_xsec = bool(ck.get("model_cfg", {}).get("use_xsec", False))
print(f"use_xsec={use_xsec} freqs={freqs}")


def spearman(a, b):
    if len(a) < 3:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


all_p, all_y, all_day, all_mod = [], [], [], []
for m in MONTHS:
    df = pd.read_feather(os.path.join(DATA, f"{m}.0.feather"))
    Xs, y, idx_df, keys = T.build_dataset(
        df, m + "01", m + "28", "eval", minutes, stats,
        freqs=freqs, seq_lens=seq_lens)
    if len(idx_df) == 0:
        continue
    Xt = [torch.from_numpy(np.ascontiguousarray(x)) for x in Xs]
    sec_ids = T._section_ids(keys) if use_xsec else None
    preds = []
    with torch.no_grad():
        for i in range(0, len(idx_df), 4096):
            xb = [t[i:i + 4096] for t in Xt]
            sidb = (None if sec_ids is None else
                    torch.from_numpy(sec_ids[i:i + 4096]))
            preds.append(model(xb, sec_ids=sidb).numpy())
    preds = np.concatenate(preds)
    all_p.append(preds); all_y.append(y)
    all_day.append(keys[:, 1]); all_mod.append(keys[:, 2])
    print(f"  {m}: n={len(preds):,} pred_mean={preds.mean():.4f}")

p = np.concatenate(all_p); y = np.concatenate(all_y)
day = np.concatenate(all_day); mod = np.concatenate(all_mod)
m = np.isfinite(y)
df = pd.DataFrame({"p": p[m], "y": y[m], "day": day[m], "mod": mod[m]})
ics = []
for _, g in df.groupby(["day", "mod"], sort=False):
    if len(g) < 10:
        continue
    ics.append(spearman(g["p"].to_numpy(), g["y"].to_numpy()))
ics = np.array([x for x in ics if np.isfinite(x)])
print(f"\n=== factor vs label IC (2024H2, {len(ics)} sections) ===")
print(f"  IC_mean = {ics.mean():+.4f}")
print(f"  IC_std  = {ics.std():.4f}")
print(f"  => {'方向正确 (IC>0)' if ics.mean() > 0 else '方向反了 (IC<0) !!!'}")
