"""
测试/推理脚本：加载 checkpoint，在测试月份上算 IC/ICIR。

用法：
  python test.py --ckpt checkpoints/best.pt --smoke
  python test.py --ckpt checkpoints/best.pt --freq bar30m
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import apply_scaler, StockDataset
from pipeline import build_split, describe
from model import E2EModel
from model_cnn import CNNTransformerModel
from metrics import compute_ic, print_ic, _spearman


@torch.no_grad()
def predict(model, X, device, batch_size=4096):
    model.eval()
    loader = DataLoader(StockDataset(X, np.zeros(len(X), dtype=np.float32)),
                        batch_size=batch_size, shuffle=False)
    preds = []
    for xb, _ in loader:
        preds.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.zeros((0,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--freq", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = Config()
    cfg.freq = args.freq or ckpt.get("freq", cfg.freq)
    cfg.lookback_minutes = ckpt.get("lookback_minutes", cfg.lookback_minutes)
    cfg.horizon_minutes = ckpt.get("horizon_minutes", cfg.horizon_minutes)
    cfg.model_arch = ckpt.get("model_arch", "gru")
    cfg.smoke = args.smoke

    _, _, te_months = cfg.resolved_months()
    print(f"[eval] freq={cfg.freq} months={te_months} device={device} "
          f"ckpt_val_ic={ckpt.get('val_ic')}")

    X, y, meta, _ = build_split(cfg, te_months, clip_bounds=ckpt.get("clip"))
    describe(X, y, meta, "test")
    if len(y) == 0:
        raise RuntimeError("empty test set")

    X = apply_scaler(X, ckpt["mean"], ckpt["std"])

    if cfg.model_arch == "cnn":
        model = CNNTransformerModel(cfg).to(device)
    else:
        model = E2EModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    preds = predict(model, X, device)
    print_ic("test", compute_ic(preds, y, meta))

    # 分日 IC 稳定性
    d = pd.DataFrame({"pred": preds, "label": y, "day": [m[1] for m in meta]})
    daily = []
    for _, g in d.groupby("day"):
        if len(g) < 30:
            continue
        daily.append(_spearman(g["pred"].to_numpy(), g["label"].to_numpy()))
    if daily:
        daily = np.array(daily)
        print(f"[test] daily IC mean={daily.mean():.4f} std={daily.std():.4f} "
              f"pos_ratio={(daily > 0).mean():.2%} n_days={len(daily)}")


if __name__ == "__main__":
    main()
