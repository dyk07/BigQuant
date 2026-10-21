"""4f (含 bar1m 微观分支) 冒烟: 验证数据管线 + 模型前向 + 参数量。

   python submissions/_tools/_smoke_4f.py [PKG] [month_count]
"""
import faulthandler
import os
import resource
import sys
import time
import types

import numpy as np
import pandas as pd
import torch

faulthandler.enable()

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
PKG = sys.argv[1] if len(sys.argv) > 1 else "cnn_4f_1020_1min_micro"
N_MONTHS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
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
    months = sorted(CFG.TRAIN_MONTHS)[-N_MONTHS:]
    freqs = list(CFG.FREQS)
    seq_lens = {f: max(1, CFG.LOOKBACK_MINUTES // T.FREQ_MINUTES[f]) for f in freqs}
    print(f"[smoke4f] pkg={PKG} freqs={freqs} seq_lens={seq_lens} months={months}",
          flush=True)

    # ---- 1) 数据管线: _accumulate 走通 bar1m ----
    t0 = time.time()
    tr = T._accumulate("cpt_jyc_2026_e2e_bar1m", months, freqs, seq_lens,
                       T.ALIGNED_MINUTES, stats=None)
    print(f"[data] accumulate {time.time()-t0:.1f}s N={len(tr['keys']):,}")
    for f in freqs:
        print(f"  feat[{f}] shape={tr['feats'][f].shape} anchors={len(tr['anchors'][f]):,}")
    stats = {}
    for f in freqs:
        mean, std = T._fit_scaler(tr["feats"][f])
        T._scale_feat(tr["feats"][f], mean, std)
        stats[f] = (mean.astype("float32"), std.astype("float32"))

    # ---- 2) 模型构建 + 参数量 ----
    model_cfg = dict(arch="multi", freqs=freqs, **CFG.MODEL_CFG,
                     use_xsec=bool(getattr(CFG, "USE_XSEC", False)),
                     xsec_layers=int(getattr(CFG, "XSEC_LAYERS", 1)),
                     xsec_heads=int(getattr(CFG, "XSEC_HEADS", 4)))
    model = T.build_model(model_cfg)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[model] params={nparam:,}  (≈{nparam*4/1e6:.1f}MB fp32)")

    # ---- 3) 前向冒烟: 取一小批 ----
    device = "cpu"
    model = model.to(device).eval()
    batch = 64
    idxb = np.arange(min(batch, len(tr["keys"])))
    xs = T._batch_windows(tr["feats"], tr["anchors"], idxb, seq_lens, freqs)
    print(f"[fwd] 各分支窗口形状:")
    for f, x in zip(freqs, xs):
        print(f"  {f}: {x.shape}")
    with torch.no_grad():
        out = model([torch.from_numpy(x) for x in xs], sec_ids=None)
    print(f"[fwd] output shape={out.shape}  mean={float(out.mean()):.4f} "
          f"std={float(out.std()):.4f}")
    print(f"[smoke4f] OK peak={peak_mb():.0f}MB")


if __name__ == "__main__":
    main()
