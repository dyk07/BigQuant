"""
本地验证(不提交): 用 submissions 的 train 模块 (与平台同源) 走**完整推理链路**,
在本地 feather 上算 2024H2 的截面 IC, 与本地 test_mf 的结果对拍。

  python submissions/_tools/verify_submission.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
SUB = os.path.join(ROOT, "submissions", "cnn_2f")
sys.path.insert(0, SUB)

from Transformer_modelsave_train import (  # noqa: E402
    load_model, build_model, build_dataset,
)

DATA = os.path.join(ROOT, "e2e_data")
MONTHS = [f"2024{m:02d}" for m in range(7, 13)]
TABLE = "cpt_jyc_2026_e2e_bar1m"


def load_feather(month):
    return pd.read_feather(os.path.join(DATA, TABLE, f"{month}.0.feather"))


def spearman(a, b):
    if len(a) < 3:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def main():
    ck = load_model(os.path.join(SUB, "transformer_model.json"), map_location="cpu")
    model = build_model(ck["model_cfg"]).eval()
    model.load_state_dict(ck["state_dict"])
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[模型] arch={ck['arch']} freqs={ck['freqs']} seq_lens={ck['seq_lens']} "
          f"params={nparam:,}")

    stats = {f: (np.asarray(ck["mean"][f], np.float32),
                 np.asarray(ck["std"][f], np.float32)) for f in ck["freqs"]}

    all_p, all_y, all_day, all_mod = [], [], [], []
    for month in MONTHS:
        # mode="eval" -> 产出标签, 用于和本地对拍
        Xs, y, idx_df, keys = build_dataset(
            load_feather(month), month + "01", month + "28", "eval",
            ck["aligned_minutes"], stats, freqs=ck["freqs"], seq_lens=ck["seq_lens"])
        if len(idx_df) == 0:
            continue
        Xt = [torch.from_numpy(np.ascontiguousarray(x)) for x in Xs]
        preds = []
        with torch.no_grad():
            for i in range(0, len(idx_df), 4096):
                xb = [t[i:i + 4096] for t in Xt]
                preds.append(model(xb).numpy())
        preds = np.concatenate(preds)
        all_p.append(preds)
        all_y.append(y)
        all_day.append(keys[:, 1])
        all_mod.append(keys[:, 2])
        print(f"  {month}: n={len(preds):,} pred_std={preds.std():.4f}")

    preds = np.concatenate(all_p)
    y = np.concatenate(all_y)
    day = np.concatenate(all_day)
    mod = np.concatenate(all_mod)

    # 截面 IC (仅用有标签样本, 与本地 test_mf 同口径)
    m = np.isfinite(y)
    df = pd.DataFrame({"p": preds[m], "y": y[m], "day": day[m], "mod": mod[m]})
    ics = []
    for _, g in df.groupby(["day", "mod"], sort=False):
        if len(g) < 10:
            continue
        ics.append(spearman(g["p"].to_numpy(), g["y"].to_numpy()))
    ics = np.array([x for x in ics if np.isfinite(x)])
    ic_mean, ic_std = ics.mean(), ics.std()
    icir = ic_mean / ic_std
    se = ic_std / np.sqrt(len(ics))
    t = ic_mean / se
    print(f"\n[结果] 样本={m.sum():,}  截面数={len(ics)}")
    print(f"  IC     = {ic_mean:+.4f}")
    print(f"  ICIR   = {icir:+.4f}  (年化 {icir*np.sqrt(250):.2f})")
    print(f"  t-stat = {t:+.2f}   SE={se:.4f}")

    # 分日 IC
    dd = pd.DataFrame({"p": preds[m], "y": y[m], "day": day[m]})
    daily = []
    for _, g in dd.groupby("day"):
        if len(g) < 30:
            continue
        daily.append(spearman(g["p"].to_numpy(), g["y"].to_numpy()))
    daily = np.array([x for x in daily if np.isfinite(x)])
    print(f"  daily IC mean={daily.mean():.4f} std={daily.std():.4f} "
          f"pos_ratio={(daily > 0).mean():.2%} n_days={len(daily)}")
    print(f"\n[对拍] 本地 test_mf 记录: IC≈+0.0813  daily_mean≈0.1087  "
          f"pos_ratio≈87.2%")


if __name__ == "__main__":
    main()
