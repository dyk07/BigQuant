"""
本地构建工具(不提交): 把 miniquant/checkpoints 下的 .pt 转成各提交目录的
transformer_model.json, 严格复用 submissions 里 train 模块的序列化格式,
保证权重能被平台侧 load_model 正确读回。

用法(在 miniquant 环境里):
  python submissions/_tools/convert_checkpoints.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # BigQuant/
SUBROOT = os.path.join(ROOT, "submissions")
CKPT_DIR = os.path.join(ROOT, "miniquant", "checkpoints")

# 借用提交目录里的 train 模块(序列化格式的唯一事实来源)
sys.path.insert(0, os.path.join(SUBROOT, "cnn_2f"))
from Transformer_modelsave_train import (  # noqa: E402
    save_model, ALL_FEATURES, ALIGNED_MINUTES, N_FEATURES, FREQ_MINUTES,
)
import config as CFG  # noqa: E402

JOBS = [
    # (ckpt, 目标目录, arch)
    ("cnn_best", "cnn_1f_bar30m", "single"),
    ("cnn_mf", "cnn_2f", "multi"),
    ("cnn_3f", "cnn_3f", "multi"),
]


def convert(ckpt_name, out_subdir, arch):
    ck = torch.load(os.path.join(CKPT_DIR, f"{ckpt_name}.pt"),
                    map_location="cpu", weights_only=False)
    sd = ck["model"]

    if arch == "single":
        freq = ck["freq"]
        seq_lens = {freq: ck["lookback_minutes"] // FREQ_MINUTES[freq]}
        mean = {freq: ck["mean"]}
        std = {freq: ck["std"]}
        freqs = [freq]
    else:
        freqs = list(ck["freqs"])
        seq_lens = {f: int(ck["seq_lens"][f]) for f in freqs}
        mean = {f: ck["scaler"][f][0] for f in freqs}
        std = {f: ck["scaler"][f][1] for f in freqs}

    model_cfg = dict(arch=arch, freqs=list(freqs), in_dim=N_FEATURES,
                     hidden=64, conv_blocks=3, kernel=3,
                     transformer_layers=2, heads=4, dropout=0.2, fused_dim=64)
    payload = {
        "format": "miniquant-transformer-v1",
        "arch": arch,
        "freqs": list(freqs),
        "seq_lens": {f: int(seq_lens[f]) for f in freqs},
        "model_cfg": model_cfg,
        "mean": {f: np.asarray(mean[f], "float32") for f in freqs},
        "std": {f: np.asarray(std[f], "float32") for f in freqs},
        "features": ALL_FEATURES,
        "aligned_minutes": ALIGNED_MINUTES,
        "train_months": [f"{y}{m:02d}" for y in range(2020, 2024) for m in range(1, 13)],
        "seed": int(CFG.SEED),
        "train_cfg": dict(CFG.TRAIN_CFG),
        "lookback_minutes": int(CFG.LOOKBACK_MINUTES),
        "horizon_minutes": int(CFG.HORIZON_MINUTES),
        "sort_key": 1,
        "state_dict": {k: v for k, v in sd.items()},
        "source_ckpt": ckpt_name,
        "val_ic": float(ck.get("val_ic", float("nan"))),
    }
    out = os.path.join(SUBROOT, out_subdir, "transformer_model.json")
    save_model(out, payload)
    print(f"[convert] {ckpt_name}.pt -> {out}\n"
          f"          ({os.path.getsize(out)/1e6:.2f} MB) arch={arch} "
          f"freqs={freqs} seq_lens={payload['seq_lens']} val_IC={payload['val_ic']:.4f}")


if __name__ == "__main__":
    for ck, sub, arch in JOBS:
        convert(ck, sub, arch)
    print("\n[done] 三个提交目录的 transformer_model.json 已生成")
