"""导出真实预测到 npz, 供后处理变体快速离线基准。

用训练侧 _accumulate/_predict_all 走与 main() 相同的推理路径, 保存
(inst, day, mod, y, pred) 五列, 后续所有后处理变体都在纯 numpy 上跑,
避免反复加载模型。
"""
import faulthandler
import os
import sys
import types

import numpy as np
import pandas as pd

faulthandler.enable()

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
PKG = "cnn_3f_1023_ema_post"
PKG_DIR = os.path.join(ROOT, "submissions", PKG)
os.chdir(PKG_DIR)
sys.path.insert(0, PKG_DIR)

DATA = os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar1m")
MONTHS = ["202309", "202310", "202311", "202312"]
OUT = os.path.join(ROOT, "submissions", "_tools", "_preds_val4m.npz")
_ALL = [os.path.splitext(f)[0].split(".")[0]
        for f in sorted(os.listdir(DATA)) if f.endswith(".feather")]


def _query(sql, filters=None):
    lo, hi = pd.Timestamp(filters["date"][0]), pd.Timestamp(filters["date"][1])
    parts = []
    for m in _ALL:
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
import Transformer_modelsave_train as T  # noqa: E402


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = T.load_model(T.MODEL_PATH, map_location=device)
    freqs = ckpt["freqs"]; seq_lens = ckpt["seq_lens"]
    stats = {f: (np.asarray(ckpt["mean"][f], np.float32),
                 np.asarray(ckpt["std"][f], np.float32)) for f in freqs}
    model = T.build_model(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    va = T._accumulate("cpt_jyc_2026_e2e_bar1m", MONTHS, freqs, seq_lens,
                       ckpt["aligned_minutes"], stats=stats)
    preds = T._predict_all(model, va["feats"], va["anchors"], seq_lens,
                           freqs, va["keys"], device)
    keys = va["keys"]
    out_mod = keys[:, 2].copy()
    out_mod[out_mod == T.AUCTION_MINUTE] = T.OPENING_SECTION
    np.savez_compressed(
        OUT,
        inst=keys[:, 0].astype("int64"),
        day=keys[:, 1].astype("int64"),
        mod=out_mod.astype("int64"),
        y=va["y"].astype("float32"),
        pred=preds.astype("float32"),
    )
    print("saved", OUT, "rows=", len(preds), "days=", len(np.unique(keys[:, 1])))


if __name__ == "__main__":
    main()
