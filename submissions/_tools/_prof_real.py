"""Profile 真实训练 step (含 xsec 分组 loss + 换手正则), 外推全量时长。"""
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
            pd.read_feather(p).assign(date=lambda d: pd.to_datetime(d["date"])))
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
                     use_xsec=True, xsec_layers=1, xsec_heads=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T.build_model(model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    n = len(ytr)
    ytr_t = torch.from_numpy(ytr)
    batch = CFG.TRAIN_CFG["batch_size"]
    buf = {f: np.empty((batch, int(seq_lens[f]), T.N_FEATURES), "float32")
           for f in freqs}
    sec_ids_tr = T._section_ids(tr["keys"])
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    amp_dtype = torch.float16

    # 换手配对
    nxt_all = T._turnover_next(tr["keys"])
    valid = nxt_all >= 0
    cur_tr = np.where(valid)[0].astype("int64")
    nxt_tr = nxt_all[valid].astype("int64")

    perm = torch.randperm(n)
    perm_np = perm.numpy()

    # 每 epoch 预处理配对 (与真实循环一致)
    pos = np.empty(n, "int64")
    pos[perm_np] = np.arange(n)
    bcur = pos[cur_tr] // batch
    bnxt = pos[nxt_tr] // batch
    same = bcur == bnxt
    scur = pos[cur_tr[same]]
    snxt = pos[nxt_tr[same]]
    bs = (scur // batch) * batch
    pair_by_s = {}
    for s0, lc, ln in zip(bs, scur % batch, snxt % batch):
        pair_by_s.setdefault(int(s0), []).append((int(lc), int(ln)))

    t_gather = t_fwd = t_bwd = 0.0
    NSTEP = 100
    for k in range(NSTEP):
        s = (k * batch) % (n - batch)
        idxb = perm_np[s:s + batch]

        t0 = time.perf_counter()
        xs = T._batch_windows(tr["feats"], tr["anchors"], idxb, seq_lens, freqs, buf=buf)
        xb = [torch.from_numpy(x).to(device) for x in xs]
        yb = ytr_t[idxb].to(device)
        sidb = torch.from_numpy(sec_ids_tr[idxb]).to(device)
        t_gather += time.perf_counter() - t0

        t0 = time.perf_counter()
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
            p = model(xb, sec_ids=sidb)
            # 真实 xsec 分组 loss
            loss = 0.0
            cnt = 0
            for g in torch.unique(sidb):
                m = sidb == g
                if m.sum() < 2:
                    continue
                loss = loss + (1.0 - T._corr_batch(p[m], yb[m]))
                cnt += 1
            loss = loss / max(cnt, 1)
            if s in pair_by_s:
                lcs = torch.tensor([a for a, _ in pair_by_s[s]], device=device)
                lns = torch.tensor([b for _, b in pair_by_s[s]], device=device)
                if lcs.numel() >= 2:
                    loss = loss + 0.01 * (1.0 - T._corr_batch(p[lcs], p[lns]))
        torch.cuda.synchronize()
        t_fwd += time.perf_counter() - t0

        t0 = time.perf_counter()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        t_bwd += time.perf_counter() - t0

    tot = t_gather + t_fwd + t_bwd
    sps = NSTEP / tot
    steps_per_epoch = (7391450 + batch - 1) // batch  # 全量 46 月
    print(f"真实训练 step (xsec 分组 + 换手):")
    print(f"  gather+H2D = {t_gather/NSTEP*1000:.1f} ms/step  {t_gather/tot*100:.0f}%")
    print(f"  fwd(含分组loss) = {t_fwd/NSTEP*1000:.1f} ms/step  {t_fwd/tot*100:.0f}%")
    print(f"  bwd+opt = {t_bwd/NSTEP*1000:.1f} ms/step  {t_bwd/tot*100:.0f}%")
    print(f"  total = {tot/NSTEP*1000:.1f} ms/step  -> {sps:.2f} step/s")
    sec_per_epoch = steps_per_epoch / sps
    print(f"  全量 steps/epoch={steps_per_epoch},  {sec_per_epoch/60:.1f} min/epoch")
    print(f"  12 epoch 训练时间 ≈ {sec_per_epoch*12/60:.1f} min (仅训练 step, 不含 accumulate)")


if __name__ == "__main__":
    main()
