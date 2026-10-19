"""AMP 吞吐基准: 对比 AMP on/off 的训练 step 吞吐与数据加载耗时。

用法:
  python _bench_amp.py          # AMP on (读 config.USE_AMP)
  python _bench_amp.py off      # 强制关 AMP

只计时:
  1) _accumulate 训练集 + 验证集 的纯数据耗时 (AMP 无关, 测一次)
  2) 固定 200 个训练 step 的纯 GPU/CPU 前向+反向耗时
以此外推全量 48 个月重训时长。
"""
import os
import sys
import time
import types

import numpy as np
import pandas as pd
import torch

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
PKG = sys.argv[1] if len(sys.argv) > 1 else "cnn_3f"
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


def bench(force_amp):
    if force_amp is not None:
        CFG.USE_AMP = force_amp
    months = sorted(MONTHS)
    freqs = list(CFG.FREQS)
    seq_lens = {f: max(1, CFG.LOOKBACK_MINUTES // T.FREQ_MINUTES[f]) for f in freqs}
    val_months = CFG.TRAIN_CFG["val_months"]
    val_m = months[-val_months:]
    tr_m = [m for m in months if m not in set(val_m)]

    table = fake_dai.query("", filters={"date": ["2024-01-01", "2024-04-30"]}).df()

    # ---- 数据耗时 ----
    t0 = time.time()
    tr = T._accumulate(table, tr_m, freqs, seq_lens, T.ALIGNED_MINUTES, stats=None)
    stats = {}
    for f in freqs:
        mean, std = T._fit_scaler(tr["feats"][f])
        T._scale_feat(tr["feats"][f], mean, std)
        stats[f] = (mean.astype("float32"), std.astype("float32"))
    va = T._accumulate(table, val_m, freqs, seq_lens, T.ALIGNED_MINUTES, stats=stats)
    ytr = T._rank_targets(tr["y"], tr["keys"])
    dt_data = time.time() - t0

    model_cfg = dict(arch="multi" if len(freqs) > 1 else "single",
                     freqs=list(freqs), **CFG.MODEL_CFG,
                     use_xsec=bool(getattr(CFG, "USE_XSEC", False)),
                     xsec_layers=int(getattr(CFG, "XSEC_LAYERS", 1)),
                     xsec_heads=int(getattr(CFG, "XSEC_HEADS", 4)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T.build_model(model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CFG.TRAIN_CFG["lr"],
                           weight_decay=CFG.TRAIN_CFG["weight_decay"])

    use_amp = bool(getattr(CFG, "USE_AMP", False)) and device.type == "cuda"
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
        str(getattr(CFG, "AMP_DTYPE", "float16")), torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    n = len(ytr)
    ytr_t = torch.from_numpy(ytr)
    batch = CFG.TRAIN_CFG["batch_size"]
    buf = {f: np.empty((batch, int(seq_lens[f]), T.N_FEATURES), "float32")
           for f in freqs}
    use_xsec = bool(getattr(CFG, "USE_XSEC", False))
    sec_ids_tr = T._section_ids(tr["keys"]) if use_xsec else None

    NSTEP = 200
    perm = torch.randperm(n)
    perm_np = perm.numpy()

    # warmup
    for _ in range(10):
        s = 0
        idxb = perm_np[s:s + batch]
        xs = T._batch_windows(tr["feats"], tr["anchors"], idxb, seq_lens, freqs, buf=buf)
        xb = [torch.from_numpy(x).to(device) for x in xs]
        yb = ytr_t[idxb].to(device)
        sidb = None if sec_ids_tr is None else torch.from_numpy(sec_ids_tr[idxb]).to(device)
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            p = model(xb, sec_ids=sidb)
            loss = 1.0 - T._corr_batch(p, yb)
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()

    torch.cuda.synchronize()
    t0 = time.time()
    for k in range(NSTEP):
        s = (k * batch) % (n - batch)
        idxb = perm_np[s:s + batch]
        xs = T._batch_windows(tr["feats"], tr["anchors"], idxb, seq_lens, freqs, buf=buf)
        xb = [torch.from_numpy(x).to(device) for x in xs]
        yb = ytr_t[idxb].to(device)
        sidb = None if sec_ids_tr is None else torch.from_numpy(sec_ids_tr[idxb]).to(device)
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            p = model(xb, sec_ids=sidb)
            loss = 1.0 - T._corr_batch(p, yb)
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt_step = time.time() - t0

    sps = NSTEP / dt_step
    steps_per_epoch = (n + batch - 1) // batch
    print(f"amp={use_amp} dtype={amp_dtype}")
    print(f"  data_accumulate = {dt_data:.1f}s  (train {len(tr_m)}月 + val {len(val_m)}月)")
    print(f"  train throughput = {sps:.1f} step/s  ({NSTEP} steps in {dt_step:.1f}s)")
    print(f"  steps/epoch = {steps_per_epoch}, sec/epoch = {steps_per_epoch/sps:.0f}s")


if __name__ == "__main__":
    force = None
    if len(sys.argv) > 2 and sys.argv[2] == "off":
        force = False
    bench(force)
