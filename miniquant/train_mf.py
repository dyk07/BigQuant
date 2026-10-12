"""
多频率融合训练（方案 A）

用法：
  # 四频率全融合（202301 训练 / 202302 验证，快速验证）
  python train_mf.py --freqs bar30m,bar15m,bar5m,bar1m \\
      --train-months 202301 --val-months 202302 --out checkpoints/mf4.pt

  # 双频率对照
  python train_mf.py --freqs bar5m,bar30m --out checkpoints/mf2.pt

  # 全量（2020-2023 训练 / 2024H1 验证）
  python train_mf.py --freqs bar30m,bar15m,bar5m,bar1m --out checkpoints/mf4.pt

流程：
  1) 逐月构建四频率对齐数据集并落盘（memmap 分块，避免 80GB 常驻内存）
  2) 用磁盘上的分块训练，每个 epoch 只读该读的块
"""
import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from pipeline import describe
from metrics import compute_ic, print_ic, make_targets
from losses import get_loss
import multifreq as mf


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--freqs", default="bar30m,bar15m,bar5m,bar1m",
                   help="逗号分隔，按 T 从大到小排列")
    p.add_argument("--train-months", default=None,
                   help="逗号分隔；默认用 config 的 train_months")
    p.add_argument("--val-months", default=None)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target", default="rank", choices=["rank", "raw"])
    p.add_argument("--loss", default="corr",
                   choices=["mse", "corr", "ic", "combined"])
    p.add_argument("--out", default="checkpoints/mf.pt")
    p.add_argument("--data-root", default="features_mf")
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=0,
                   help="memmap 建议 0（磁盘随机读，多进程未必更快）")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="只用前 N 个训练样本（快速 smoke）")
    p.add_argument("--rebuild", action="store_true",
                   help="强制重建磁盘数据集")
    p.add_argument("--lookback", type=int, default=None,
                   help="全局 lookback 分钟（覆盖 config 默认 120）")
    p.add_argument("--lookback-bar1m", type=int, default=None,
                   help="单独覆盖 bar1m 的 lookback（磁盘瓶颈主要在此）")
    p.add_argument("--lookback-bar5m", type=int, default=None,
                   help="单独覆盖 bar5m 的 lookback")
    p.add_argument("--keep-data", action="store_true",
                   help="训练结束后保留磁盘数据集（默认保留，便于复用）")
    p.add_argument("--model", default=None,
                   choices=["gru", "cnn"],
                   help="模型架构：gru（原多分支 GRU）或 cnn（CNN+Transformer）")
    p.add_argument("--cnn-hidden", type=int, default=None,
                   help="CNN 隐藏维（覆盖 cfg.cnn_hidden）")
    p.add_argument("--cnn-blocks", type=int, default=None,
                   help="CNN 残差块数（覆盖 cfg.cnn_blocks）")
    p.add_argument("--transformer-layers", type=int, default=None,
                   help="Transformer 层数（覆盖 cfg.transformer_layers）")
    p.add_argument("--dropout", type=float, default=None,
                   help="Dropout 率（覆盖 cfg.dropout）")
    return p.parse_args()


def build_cfg(args) -> Config:
    cfg = Config()
    cfg.freqs = [x.strip() for x in args.freqs.split(",") if x.strip()]
    cfg.loss = args.loss
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.patience = args.patience
    cfg.num_workers = args.num_workers
    if args.model:
        cfg.model_arch = args.model
    if args.lookback:
        cfg.lookback_minutes = args.lookback
    ovr = {}
    if args.lookback_bar1m:
        ovr["bar1m"] = args.lookback_bar1m
    if args.lookback_bar5m:
        ovr["bar5m"] = args.lookback_bar5m
    cfg.lookback_minutes_per_freq = ovr
    if args.device:
        cfg.device = args.device
    if args.cnn_hidden is not None:
        cfg.cnn_hidden = args.cnn_hidden
    if args.cnn_blocks is not None:
        cfg.cnn_blocks = args.cnn_blocks
    if args.transformer_layers is not None:
        cfg.transformer_layers = args.transformer_layers
    if args.dropout is not None:
        cfg.dropout = args.dropout
    return cfg


def disk_estimate_gb(cfg: Config, n_months: int,
                     n_per_month: int = 107817) -> float:
    """估算四频率对齐数据集在磁盘上的大小（每样本字节 × 月数）。"""
    per = 0
    for f in cfg.active_freqs():
        fc = cfg.for_freq(f)
        per += fc.seq_len_bars * 25 * 4          # float32 x 25 特征
    return per * n_per_month * n_months / 1e9


def main():
    args = parse_args()
    cfg = build_cfg(args)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    freqs = cfg.active_freqs()
    tr_months = ([x.strip() for x in args.train_months.split(",")]
                 if args.train_months else cfg.train_months)
    va_months = ([x.strip() for x in args.val_months.split(",")]
                 if args.val_months else cfg.val_months)

    # tag 必须包含 lookback 配置，否则不同窗口长度会命中同一缓存
    lb = "-".join(f"{f}{cfg.for_freq(f).seq_len_bars}" for f in freqs)
    tag_tr = (f"train_{'-'.join(freqs)}_{lb}_"
              f"{tr_months[0]}_{tr_months[-1]}")
    tag_va = (f"val_{'-'.join(freqs)}_{lb}_"
              f"{va_months[0]}_{va_months[-1]}")

    print("=" * 70)
    print(f"多频率融合 freqs={freqs}")
    for f in freqs:
        fc = cfg.for_freq(f)
        print(f"   {f:7s} lookback={fc.lookback_minutes:>3d}m  "
              f"T={fc.seq_len_bars:>3d}  horizon={fc.horizon_bars}bars")
    print(f"train months: {tr_months[0]}..{tr_months[-1]} ({len(tr_months)})")
    print(f"val   months: {va_months[0]}..{va_months[-1]} ({len(va_months)})")
    print(f"device={device} batch={cfg.batch_size} target={args.target} "
          f"loss={args.loss}")
    est_tr = disk_estimate_gb(cfg, len(tr_months))
    est_va = disk_estimate_gb(cfg, len(va_months))
    print(f"磁盘预估: train ~{est_tr:.0f} GB + val ~{est_va:.0f} GB "
          f"= {est_tr+est_va:.0f} GB")
    if est_tr + est_va > 45:
        print("  [警告] 可能超过可用磁盘(52GB)。建议加 "
              "--lookback-bar1m 30（可减到约 1/4）")
    print("=" * 70)

    root = args.data_root
    tr_dir = os.path.join(root, tag_tr)
    va_dir = os.path.join(root, tag_va)

    # ---- 建训练集 ----
    need_tr = args.rebuild or not os.path.exists(
        os.path.join(tr_dir, "meta.json"))
    if need_tr:
        print(f"[build] 训练集 -> {tr_dir}")
        t0 = time.time()
        tr_dir, clip, scaler = mf.build_multifreq_split(cfg, tr_months, tag_tr,
                                                        out_root=root)
        mf.save_scaler(tr_dir, scaler)
        print(f"[build] 训练集完成 {time.time()-t0:.0f}s")
    else:
        print(f"[build] 复用已有数据集 {tr_dir}")
        scaler = mf.load_scaler(tr_dir)
        import json
        with open(os.path.join(tr_dir, "meta.json")) as fh:
            clip = tuple(json.load(fh)["clip"])

    # ---- 建验证集（复用训练集 scaler 与 clip）----
    need_va = args.rebuild or not os.path.exists(
        os.path.join(va_dir, "meta.json"))
    if need_va:
        print(f"[build] 验证集 -> {va_dir}")
        t0 = time.time()
        mf.build_multifreq_split(cfg, va_months, tag_va, out_root=root,
                                 clip_bounds=clip)
        # 直接用训练集 scaler 覆盖
        mf.save_scaler(va_dir, scaler)
        print(f"[build] 验证集完成 {time.time()-t0:.0f}s")

    tr_ds = mf.MultiFreqDataset(tr_dir, mean={f: scaler[f][0] for f in freqs},
                                std={f: scaler[f][1] for f in freqs})
    va_ds = mf.MultiFreqDataset(va_dir, mean={f: scaler[f][0] for f in freqs},
                                std={f: scaler[f][1] for f in freqs})

    if args.max_train_samples:
        tr_ds = tr_ds.subset(args.max_train_samples)
    print(f"[data] train n={len(tr_ds):,}  val n={len(va_ds):,}  "
          f"T={tr_ds.seq_lens}")

    y_tr_raw = tr_ds.y.copy()
    y_va_raw = va_ds.y.copy()
    m_tr = [tr_ds.get_meta(i) for i in range(len(tr_ds))]
    m_va = [va_ds.get_meta(i) for i in range(len(va_ds))]
    describe(np.zeros((len(y_tr_raw), 1, 1)), y_tr_raw, m_tr, "train")
    describe(np.zeros((len(y_va_raw), 1, 1)), y_va_raw, m_va, "val")

    tr_ds.y = make_targets(y_tr_raw, m_tr, kind=args.target)
    va_ds.y = make_targets(y_va_raw, m_va, kind=args.target)
    print(f"[target] kind={args.target} train mean={tr_ds.y.mean():+.4f} "
          f"std={tr_ds.y.std():.4f}")

    tr_loader, tr_sampler = mf.make_mf_loader(tr_ds, cfg.batch_size,
                                              shuffle=True,
                                              num_workers=cfg.num_workers,
                                              seed=cfg.seed, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers,
                           collate_fn=mf.multifreq_collate)
    print(f"[loader] train batches/epoch={len(tr_loader)} "
          f"(按块分组，每 epoch 打乱块顺序)")

    if cfg.model_arch == "cnn":
        model = mf.CNNMultiFreqModel(cfg).to(device)
    else:
        model = mf.E2EMultiFreqModel(cfg).to(device)
    print(f"[model] {cfg.model_arch} params={mf.count_parameters(model):,} "
          f"branches={len(freqs)}")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=2)
    crit = get_loss(args.loss)
    print(f"[loss] {args.loss}")

    @torch.no_grad()
    def predict(loader):
        model.eval()
        out = []
        for xs, _ in loader:
            xs = [x.to(device) for x in xs]
            out.append(model(xs).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0,))

    @torch.no_grad()
    def probe_train(n=50000):
        """抽样训练集 IC，预警模型是否退化。"""
        model.eval()
        sub = tr_ds.subset(min(n, len(tr_ds)))
        loader = DataLoader(sub, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, collate_fn=mf.multifreq_collate)
        pred = np.concatenate([model([x.to(device) for x in xs]).cpu().numpy()
                               for xs, _ in loader])
        return compute_ic(pred, y_tr_raw[:len(pred)], m_tr[:len(pred)])

    best_ic, wait = -np.inf, 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        if tr_sampler is not None:
            tr_sampler.set_epoch(epoch)          # 每个 epoch 重新打乱块顺序
        model.train()
        tot, cnt = 0.0, 0
        for xs, yb in tr_loader:
            xs = [x.to(device) for x in xs]
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xs), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += loss.item() * len(yb)
            cnt += len(yb)
        tr_loss = tot / max(cnt, 1)

        tr_ic = probe_train()
        va_pred = predict(va_loader)
        stats = compute_ic(va_pred, y_va_raw, m_va)
        ic = stats[0]
        print(f"[epoch {epoch:02d}] loss={tr_loss:.6f} "
              f"train_IC={tr_ic[0]:+.4f}", end="  ")
        print_ic("val", stats)
        sched.step(ic)

        if ic > best_ic:
            best_ic = ic
            wait = 0
            torch.save({
                "model": model.state_dict(),
                "freqs": freqs,
                "seq_lens": tr_ds.seq_lens,
                "scaler": {f: scaler[f] for f in freqs},
                "clip": clip,
                "model_arch": cfg.model_arch,
                "target": args.target,
                "loss": args.loss,
                "val_ic": float(ic),
            }, args.out)
            print(f"  ↑ best IC -> {best_ic:.4f} ({args.out})")
        else:
            wait += 1
            if wait >= cfg.patience:
                print("[early stop]")
                break

    print(f"[done] best val IC = {best_ic:.4f}")
    print(f"[data] 磁盘数据集保留在 {root}/（如需删除请手动 rm -rf {root}）")


if __name__ == "__main__":
    main()
