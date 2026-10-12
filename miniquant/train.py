"""
训练主脚本。

用法：
  python train.py --smoke                 # 小规模跑通（202301 训练 / 202302 验证）
  python train.py --freq bar30m           # 全量训练
  python train.py --smoke --shuffle-label # 防泄漏对照：打乱标签，IC 应接近 0

流程：
  train/val 分别切分 -> 只用训练集 fit 标准化与标签截断 -> 按验证集 IC 早停
"""
import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import fit_scaler, apply_scaler, StockDataset
from pipeline import build_split, describe
from model import E2EModel, count_parameters
from model_cnn import CNNTransformerModel, count_parameters as count_parameters_cnn
from metrics import compute_ic, print_ic, make_targets
from losses import get_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device, clip):
    model.train()
    total, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total += loss.item() * len(yb)
        count += len(yb)
    return total / max(count, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for xb, _ in loader:
        preds.append(model(xb.to(device)).cpu().numpy())
    if not preds:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(preds, axis=0)


def _sub_loader(X, cfg, device):
    """为快速抽样评估构造一个只做前向的 loader（y 为占位）。"""
    ds = StockDataset(X, np.zeros(len(X), dtype=np.float32))
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                      num_workers=0, pin_memory=(device.type == "cuda"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="小规模跑通（用配置里的 smoke 月份）")
    p.add_argument("--freq", default=None,
                   choices=["bar1m", "bar5m", "bar15m", "bar30m"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None,
                   help="验证集 IC 无提升的容忍轮数")
    p.add_argument("--lookback-minutes", type=int, default=None,
                   help="输入窗口覆盖的历史分钟数")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="checkpoints/best.pt")
    p.add_argument("--target", default="rank", choices=["rank", "raw"],
                   help="rank=回归截面百分位排名（推荐，IC 目标）; raw=回归原始收益率")
    p.add_argument("--model", default=None,
                   choices=["gru", "cnn"],
                   help="模型架构：gru（原 GRU+MLP）或 cnn（CNN+Transformer）")
    p.add_argument("--loss", default="mse",
                   choices=["mse", "corr", "ic", "combined"],
                   help="corr/ic 直接优化排序，与评估指标一致")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--shuffle-label", action="store_true",
                   help="防泄漏对照：打乱训练/验证标签，IC 应≈0")
    p.add_argument("--max-train-months", type=int, default=None,
                   help="只用训练月份列表中的前 N 个月（快速调试）")
    p.add_argument("--train-months", default=None,
                   help="逗号分隔，覆盖训练月份，如 202301,202302")
    p.add_argument("--val-months", default=None,
                   help="逗号分隔，覆盖验证月份")
    return p.parse_args()


def build_cfg(args) -> Config:
    cfg = Config()
    cfg.smoke = bool(args.smoke)
    if args.freq:
        cfg.freq = args.freq
    if args.model:
        cfg.model_arch = args.model
    if args.epochs:
        cfg.epochs = args.epochs
    if args.patience:
        cfg.patience = args.patience
    if args.lookback_minutes:
        cfg.lookback_minutes = args.lookback_minutes
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.lr = args.lr
    if args.device:
        cfg.device = args.device
    if cfg.smoke:
        cfg.epochs = min(cfg.epochs, 4)
        cfg.patience = min(cfg.patience, 2)
    return cfg


def main():
    args = parse_args()
    cfg = build_cfg(args)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    tr_months, va_months, _ = cfg.resolved_months()
    if args.train_months:
        tr_months = [x.strip() for x in args.train_months.split(",") if x.strip()]
    if args.val_months:
        va_months = [x.strip() for x in args.val_months.split(",") if x.strip()]
    if args.max_train_months:
        tr_months = tr_months[:args.max_train_months]

    print("=" * 60)
    print(f"freq={cfg.freq} bar={cfg.bar_minutes}m horizon={cfg.horizon_bars}bars "
          f"seq_len={cfg.seq_len_bars}bars lookback={cfg.lookback_minutes}m "
          f"sample_bar_idx={cfg.aligned_bar_indices()}")
    print(f"device={device} smoke={cfg.smoke}")
    print(f"train months: {tr_months[0]}..{tr_months[-1]} ({len(tr_months)})")
    print(f"val   months: {va_months[0]}..{va_months[-1]} ({len(va_months)})")
    print("=" * 60)

    print("[data] building train ...")
    X_tr, y_tr, m_tr, clip = build_split(cfg, tr_months, verbose=True)
    describe(X_tr, y_tr, m_tr, "train")
    print("[data] building val ...")
    X_va, y_va, m_va, _ = build_split(cfg, va_months, clip_bounds=clip)
    describe(X_va, y_va, m_va, "val")
    if len(y_tr) == 0 or len(y_va) == 0:
        raise RuntimeError("empty train or val set; check months/freq/window")

    # 打乱标签的防泄漏对照
    if args.shuffle_label:
        print("[control] shuffling labels (IC should be ~0)")
        rng = np.random.default_rng(cfg.seed)
        y_tr = rng.permutation(y_tr)
        y_va = rng.permutation(y_va)

    # 标准化统计量只用训练集
    mean, std = fit_scaler(X_tr)
    X_tr = apply_scaler(X_tr, mean, std)
    X_va = apply_scaler(X_va, mean, std)

    # 训练目标：默认回归截面百分位排名（与 IC 目标一致，避免退化成常数）
    y_tr_raw = y_tr.copy()          # IC 永远用原始收益率标签评估
    y_va_raw = y_va.copy()
    y_tr = make_targets(y_tr, m_tr, kind=args.target)
    y_va_fit = make_targets(y_va, m_va, kind=args.target)
    print(f"[target] kind={args.target} "
          f"train mean={y_tr.mean():+.4f} std={y_tr.std():.4f}")

    train_loader = DataLoader(
        StockDataset(X_tr, y_tr), batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(
        StockDataset(X_va, y_va_fit), batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True)

    if cfg.model_arch == "cnn":
        model = CNNTransformerModel(cfg).to(device)
        n_params = count_parameters_cnn(model)
    else:
        model = E2EModel(cfg).to(device)
        n_params = count_parameters(model)
    print(f"[model] {cfg.model_arch} params={n_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)
    criterion = get_loss(args.loss)
    print(f"[loss] {args.loss}")

    best_ic, wait = -np.inf, 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(cfg.epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer,
                                  criterion, device, cfg.grad_clip)

        # 训练集 IC（抽样，快速预警模型是否退化成常数输出）
        n_probe = min(len(X_tr), 100000)
        tr_pred = predict(model, _sub_loader(X_tr[:n_probe], cfg, device),
                          device)
        tr_ic = compute_ic(tr_pred, y_tr_raw[:n_probe], m_tr[:n_probe])

        val_pred = predict(model, val_loader, device)
        stats = compute_ic(val_pred, y_va_raw, m_va)
        ic = stats[0]
        print(f"[epoch {epoch:02d}] loss={tr_loss:.6f} "
              f"train_IC={tr_ic[0]:+.4f}", end="  ")
        print_ic("val", stats)
        scheduler.step(ic)

        if ic > best_ic:
            best_ic = ic
            wait = 0
            torch.save({
                "model": model.state_dict(),
                "mean": mean, "std": std, "clip": clip,
                "freq": cfg.freq,
                "lookback_minutes": cfg.lookback_minutes,
                "horizon_minutes": cfg.horizon_minutes,
                "model_arch": cfg.model_arch,
                "target": args.target,
                "loss": args.loss,
                "val_ic": float(ic),
            }, args.out)
            print(f"  ↑ best IC updated: {best_ic:.4f} -> {args.out}")
        else:
            wait += 1
            if wait >= cfg.patience:
                print("[early stop]")
                break

    print(f"[done] best val IC = {best_ic:.4f}")


if __name__ == "__main__":
    main()
