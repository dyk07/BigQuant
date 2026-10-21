"""批量诊断 spread λ 扫描的多空 Sharpe。

对每个 _scan_spread_lam*.json 权重, 在验证集 (202311~202312) 上评估:
  - val_IC / IC_IR (预测能力)
  - 多空 Sharpe (q=0.20)
  - 实际 spread / 完美 spread (头尾选股命中率)

结论: spread 损失是否在"IC 略降"的代价下换来"Sharpe 提升"。
"""
import faulthandler
import glob
import os
import sys
import types

import numpy as np
import pandas as pd

faulthandler.enable()

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
PKG = "cnn_3f_1021_spread"
PKG_DIR = os.path.join(ROOT, "submissions", PKG)
os.chdir(PKG_DIR)
sys.path.insert(0, PKG_DIR)

DATA = os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar1m")
VAL_MONTHS = ["202311", "202312"]

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


def eval_weight(weight_path, device):
    ckpt = T.load_model(weight_path, map_location=device)
    freqs = ckpt["freqs"]
    seq_lens = ckpt["seq_lens"]
    stats = {f: (np.asarray(ckpt["mean"][f], np.float32),
                 np.asarray(ckpt["std"][f], np.float32)) for f in freqs}
    model = T.build_model(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    va = T._accumulate("cpt_jyc_2026_e2e_bar1m", VAL_MONTHS, freqs, seq_lens,
                       ckpt["aligned_minutes"], stats=stats)
    keys = va["keys"]
    y = va["y"]
    preds = T._predict_all(model, va["feats"], va["anchors"], seq_lens,
                           freqs, keys, device)

    df = pd.DataFrame({"inst": keys[:, 0], "day": keys[:, 1], "mod": keys[:, 2],
                       "p": preds, "y": y})
    df = df[np.isfinite(df["y"])].copy()

    # IC / IC_IR
    ics = []
    for _, g in df.groupby(["day", "mod"], sort=False):
        if len(g) < 10:
            continue
        pr = g["p"].rank(); yr = g["y"].rank()
        ics.append(np.corrcoef(pr, yr)[0, 1])
    ics = np.array(ics)
    ic_mean = ics.mean()
    ic_ir = ic_mean / ics.std() if ics.std() > 0 else 0

    # 多空 Sharpe (q=0.20) + 命中率
    q = 0.20
    actual = []
    perfect = []
    for _, g in df.groupby(["day", "mod"], sort=False):
        if len(g) < 20:
            continue
        k = max(1, int(len(g) * q))
        gs = g.sort_values("p")
        actual.append(gs["y"].iloc[-k:].mean() - gs["y"].iloc[:k].mean())
        gy = g.sort_values("y")
        perfect.append(gy["y"].iloc[-k:].mean() - gy["y"].iloc[:k].mean())
    actual = np.array(actual)
    perfect = np.array(perfect)
    sharpe = actual.mean() / actual.std() if actual.std() > 0 else 0
    hit = actual.mean() / perfect.mean() if perfect.mean() != 0 else 0

    del model, va, df
    torch.cuda.empty_cache()
    return ic_mean, ic_ir, sharpe, hit


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = sorted(glob.glob(os.path.join(PKG_DIR, "_scan_spread_lam*.json")),
                     key=lambda p: float(os.path.basename(p)
                                         .split("lam")[1].split(".json")[0]))
    print(f"{'lambda':>8}  {'val_IC':>8}  {'IC_IR':>8}  {'Sharpe':>8}  {'hit%':>8}")
    for w in weights:
        lam = float(os.path.basename(w).split("lam")[1].split(".json")[0])
        ic_mean, ic_ir, sharpe, hit = eval_weight(w, device)
        print(f"{lam:>8.2f}  {ic_mean:>+8.4f}  {ic_ir:>8.3f}  "
              f"{sharpe:>8.3f}  {hit*100:>7.1f}%")


if __name__ == "__main__":
    main()
