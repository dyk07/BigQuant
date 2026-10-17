# -*- coding: utf-8 -*-
"""exp001 模型定义与 JSON 文本权重存取（训练/推理共用单一事实来源）。

网络：参考官方 demo 的「窗口 -> Transformer encoder -> 池化 -> 回归头」骨架，
在 2M 参数量级上做了三点必要增强：
1. key_padding_mask：无效 bar（停牌/缺口/padding）不参与注意力，避免伪数据污染；
2. 截面槽位嵌入（slot embedding）：让头部知道样本 bar 是当日 7 个截面中的哪一个；
3. 头部用「末位置隐状态 + 槽位嵌入」而非均值池化：末位置恒为样本 bar 本身，语义明确。

权重一律存 JSON 文本（平台只接受文本文件），结构与官方 demo 一致：
{model_cfg, feature_spec, mean, std, data_alignment, state_dict:{k:{dtype,shape,data}}}。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

#: 权重文件里的对齐声明，推理侧据此校验数据口径
DATA_ALIGNMENT = "30m_bar_end_sample_window_strictly_before_next_bar_label"


class BaselineTransformer(nn.Module):
    """~2.2M 参数的截面收益预测模型（30m 窗口）。"""

    def __init__(self, n_in: int, d_model: int = 256, nhead: int = 8,
                 nlayers: int = 4, dim_ff: int = 512, dropout: float = 0.1,
                 seq_len: int = 16, n_slots: int = 7):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Linear(n_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.slot_emb = nn.Embedding(n_slots, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout,
            batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model),
            nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor, slot_id: torch.Tensor,
                pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, L, n_in)；slot_id: (B,) 0..6；pad_mask: (B,) True=无效 bar（不参与注意力）。"""
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        z = torch.cat([h[:, -1], self.slot_emb(slot_id)], dim=-1)
        return self.head(z).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def save_model(ckpt: dict, model_path: str | Path) -> Path:
    """把 checkpoint（state_dict + 配置 + 标准化统计）存成 JSON 文本。"""
    tensors = {}
    for k, v in ckpt["state_dict"].items():
        t = v.detach().cpu()
        tensors[k] = {"dtype": str(t.dtype).replace("torch.", ""),
                      "shape": list(t.shape), "data": t.reshape(-1).tolist()}
    payload = {k: v for k, v in ckpt.items() if k != "state_dict"}
    payload["state_dict"] = tensors
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return model_path


def load_model(model_path: str | Path, map_location: str = "cpu") -> dict:
    """读取 save_model 写出的 JSON，还原为 checkpoint dict（state_dict 为张量）。"""
    with open(model_path, encoding="utf-8") as f:
        payload = json.load(f)
    sd = {}
    for k, meta in payload["state_dict"].items():
        t = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
        sd[k] = t.reshape(meta["shape"]).to(map_location)
    ckpt = {k: v for k, v in payload.items() if k != "state_dict"}
    ckpt["state_dict"] = sd
    return ckpt


def build_model_from_cfg(model_cfg: dict) -> BaselineTransformer:
    """按权重文件里的 model_cfg 重建网络（推理侧入口）。"""
    return BaselineTransformer(**model_cfg)


def load_model_into(model_path: str | Path, model: BaselineTransformer,
                    map_location: str = "cpu") -> dict:
    """读取 JSON 权重并载入已按 model_cfg 重建的网络，返回附带配置的 checkpoint。"""
    ckpt = load_model(model_path, map_location=map_location)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return ckpt


def count_params_table(model: BaselineTransformer) -> dict[str, int]:
    """分部件参数量（写实验报告用）。"""
    def n(mod: nn.Module | nn.Parameter) -> int:
        if isinstance(mod, nn.Parameter):
            return mod.numel()
        return sum(p.numel() for p in mod.parameters())
    return {
        "proj(input 27->d_model)": n(model.proj),
        "pos_embedding": n(model.pos),
        "slot_embedding": n(model.slot_emb),
        "encoder_layers_total": n(model.encoder),
        "head": n(model.head),
        "total": model.num_parameters(),
    }


if __name__ == "__main__":
    cfg = dict(n_in=27, d_model=256, nhead=8, nlayers=4, dim_ff=512,
               dropout=0.1, seq_len=16, n_slots=7)
    model = BaselineTransformer(**cfg)
    model.eval()  # 关闭 dropout，保证两次前向可比
    print(json.dumps(count_params_table(model), ensure_ascii=False, indent=2))

    # 掩码有效性自检：改动无效 bar 的特征，末位置输出必须不变
    torch.manual_seed(0)
    x = torch.randn(4, 16, 27)
    slot = torch.tensor([0, 3, 6, 2])
    valid = torch.ones(4, 16, dtype=torch.bool)
    valid[:, :10] = False  # 前 10 根无效
    y1 = model(x, slot, ~valid)
    x2 = x.clone()
    x2[:, :10] = torch.randn_like(x2[:, :10]) * 100  # 无效位置乱填
    y2 = model(x2, slot, ~valid)
    assert torch.allclose(y1, y2, atol=1e-5), f"无效 bar 泄漏进输出: {(y1 - y2).abs().max()}"
    print("mask 自检通过：无效 bar 特征变化不影响输出")

