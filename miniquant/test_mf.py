"""
多频率融合测试集评估。

用法：
  python test_mf.py --ckpt checkpoints/mf4.pt --test-months 202407,202408
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from config import Config
from metrics import compute_ic, print_ic, _spearman
import multifreq as mf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/mf.pt")
    ap.add_argument("--test-months", default=None,
                    help="逗号分隔；默认用 config 的 test_months")
    ap.add_argument("--data-root", default="features_mf")
    ap.add_argument("--device", default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    freqs = ckpt["freqs"]
    scaler = ckpt["scaler"]
    clip = ckpt["clip"]
    model_arch = ckpt.get("model_arch", "gru")

    cfg = Config()
    cfg.freqs = freqs
    cfg.model_arch = model_arch
    # 从 checkpoint 的 seq_lens 反推 lookback，保证窗口长度一致
    cfg.lookback_minutes_per_freq = {}
    for f in freqs:
        T = ckpt["seq_lens"][f]
        cfg.lookback_minutes_per_freq[f] = T * cfg.for_freq(f).bar_minutes

    te_months = ([x.strip() for x in args.test_months.split(",")]
                 if args.test_months else cfg.test_months)
    lb = "-".join(f"{f}{cfg.for_freq(f).seq_len_bars}" for f in freqs)
    tag = f"test_{'-'.join(freqs)}_{lb}_{te_months[0]}_{te_months[-1]}"
    root = args.data_root
    out_dir = os.path.join(root, tag)

    print("=" * 70)
    print(f"[mf-eval] freqs={freqs} months={te_months}")
    print(f"  seq_lens={ckpt['seq_lens']} ckpt_val_ic="
          f"{ckpt.get('val_ic'):.4f} device={device}")
    print("=" * 70)

    if args.rebuild or not os.path.exists(os.path.join(out_dir, "meta.json")):
        t0 = __import__("time").time()
        mf.build_multifreq_split(cfg, te_months, tag, out_root=root,
                                 clip_bounds=clip)
        mf.save_scaler(out_dir, scaler)
        print(f"[build] 测试集完成 {__import__('time').time()-t0:.0f}s")

    ds = mf.MultiFreqDataset(out_dir,
                             mean={f: scaler[f][0] for f in freqs},
                             std={f: scaler[f][1] for f in freqs})
    if args.max_samples:
        ds = ds.subset(args.max_samples)
    print(f"[data] test n={len(ds):,}")

    loader, _ = mf.make_mf_loader(ds, 4096, shuffle=False, num_workers=0)
    model = mf.CNNMultiFreqModel(cfg).to(device) if model_arch == "cnn" else mf.E2EMultiFreqModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds = []
    with torch.no_grad():
        for xs, _ in loader:
            preds.append(model([x.to(device) for x in xs]).cpu().numpy())
    preds = np.concatenate(preds)
    y = ds.y
    meta = [ds.get_meta(i) for i in range(len(ds))]

    print_ic("test", compute_ic(preds, y, meta))

    d = pd.DataFrame({"pred": preds, "label": y, "day": [m[1] for m in meta]})
    daily = [(_spearman(g["pred"].to_numpy(), g["label"].to_numpy()))
             for _, g in d.groupby("day") if len(g) >= 30]
    if daily:
        daily = np.array(daily)
        print(f"[test] daily IC mean={daily.mean():.4f} std={daily.std():.4f} "
              f"pos_ratio={(daily > 0).mean():.2%} n_days={len(daily)}")


if __name__ == "__main__":
    main()
