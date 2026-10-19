"""定位训练 step 的耗时构成: 数据 gather vs H2D 传输 vs GPU 前向/反向。

若数据侧占大头, AMP 的 GPU 加速会被掩盖, 应转向优化数据管线。
"""
import os
import sys
import time
import types

import numpy as np
import pandas as pd
import torch

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
PKG = "cnn_3f"
PKG_DIR = os.path.join(ROOT, "submissions", PKG)
os.chdir(PKG_DIR)
sys.path.insert(0, PKG_DIR)

DATA = os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar1m")
MONTHS = ["202401", "202402", "202403", "202404"]

_CACHE = {}


def _load_month(m):
    if m not in _CACHE:
        p = os.path.join(DATA, f"{m}.0.feather")
        _CACHE[m] = None if not os.path.exists(p) else (
            pd.read_feather(p).assign(
                date=lambda d: pd.to_datetime(d["date"])))
    return _CACHE[m]


def _query(sql, filters=None):
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
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return types.SimpleNamespace(df=lambda: df)


fake_dai = types.ModuleType("dai")
fake_dai.query = _query
sys.modules["dai"] = fake_dai

import config as CFG  # noqa: E402
import Transformer_modelsave_train as T  # noqa: E402


def main():
    months = sorted(MONTHS)
    freqs = list(CFG.FREQS)
    seq_lens = {f: max(1, CFG.LOOKBACK_MINUTES // T.FREQ_MINUTES[f]) for f in freqs}
    val_m = months[-2:]
    tr_m = [m for m in months if m not in set(val_m)]
    table = fake_dai.query("", filters={"date": ["2024-01-01", "2024-04-30"]}).df()

    tr = T._accumulate(table, tr_m, freqs, seq_lens, T.ALIGNED_MINUTES, stats=None)
    for f in freqs:
        mean, std = T._fit_scaler(tr["feats"][f])
        T._scale_feat(tr["feats"][f], mean, std)
    ytr = T._rank_targets(tr["y"], tr["keys"])

    model_cfg = dict(arch="multi", freqs=list(freqs), **CFG.MODEL_CFG,
                     use_xsec=bool(getattr(CFG, "USE_XSEC", False)),
                     xsec_layers=int(getattr(CFG, "XSEC_LAYERS", 1)),
                     xsec_heads=int(getattr(CFG, "XSEC_HEADS", 4)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T.build_model(model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    n = len(ytr)
    ytr_t = torch.from_numpy(ytr)
    batch = CFG.TRAIN_CFG["batch_size"]
    buf = {f: np.empty((batch, int(seq_lens[f]), T.N_FEATURES), "float32")
           for f in freqs}
    use_xsec = bool(getattr(CFG, "USE_XSEC", False))
    sec_ids_tr = T._section_ids(tr["keys"]) if use_xsec else None
    use_amp = bool(getattr(CFG, "USE_AMP", False)) and device.type == "cuda"
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    perm = torch.randperm(n)
    perm_np = perm.numpy()

    t_gather = t_h2d = t_fwd = t_bwd = t_opt = 0.0
    NSTEP = 200
    for k in range(NSTEP):
        s = (k * batch) % (n - batch)
        idxb = perm_np[s:s + batch]

        t0 = time.perf_counter()
        xs = T._batch_windows(tr["feats"], tr["anchors"], idxb, seq_lens, freqs, buf=buf)
        t_gather += time.perf_counter() - t0

        t0 = time.perf_counter()
        xb = [torch.from_numpy(x).to(device) for x in xs]
        yb = ytr_t[idxb].to(device)
        sidb = None if sec_ids_tr is None else torch.from_numpy(sec_ids_tr[idxb]).to(device)
        t_h2d += time.perf_counter() - t0

        t0 = time.perf_counter()
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            p = model(xb, sec_ids=sidb)
            loss = 1.0 - T._corr_batch(p, yb)
        torch.cuda.synchronize()
        t_fwd += time.perf_counter() - t0

        t0 = time.perf_counter()
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        torch.cuda.synchronize()
        t_bwd += time.perf_counter() - t0

    tot = t_gather + t_h2d + t_fwd + t_bwd
    print(f"amp={use_amp}")
    print(f"  gather(numpy窗口) = {t_gather/NSTEP*1000:.1f} ms/step  {t_gather/tot*100:.0f}%")
    print(f"  H2D(转GPU)        = {t_h2d/NSTEP*1000:.1f} ms/step  {t_h2d/tot*100:.0f}%")
    print(f"  fwd               = {t_fwd/NSTEP*1000:.1f} ms/step  {t_fwd/tot*100:.0f}%")
    print(f"  bwd+opt           = {t_bwd/NSTEP*1000:.1f} ms/step  {t_bwd/tot*100:.0f}%")
    print(f"  total             = {tot/NSTEP*1000:.1f} ms/step")


if __name__ == "__main__":
    main()
