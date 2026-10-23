# -*- coding: utf-8 -*-
"""文件用处：定义 exp032 模型——exp022 30m 主干 + 5m 高频增量编码器 + 30m 查询驱动跨频注意力。"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


MAX_WEIGHT_JSON_BYTES = 50_000_000


def experiment_fingerprint() -> str:
    """对会影响实验语义的源码与固定配置计算稳定指纹，阻止复用陈旧门槛产物。"""
    experiment = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for relative in ("model.py", "alignment.py", "train.py", "verify.py", "config/final.json"):
        digest.update(relative.encode("utf-8"))
        digest.update((experiment / relative).read_bytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 30m 主干：与 exp022 residual_common 逐位一致
# ---------------------------------------------------------------------------
class MaskedTemporalConvBlock(nn.Module):
    """仅沿单只股票时间轴卷积的残差块；无效 token 始终在块末清零。"""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        valid = (~pad_mask).unsqueeze(-1)
        y = self.norm(h).masked_fill(~valid, 0.0).transpose(1, 2)
        y = self.conv1(y)
        y = self.dropout(F.gelu(y))
        y = self.conv2(y).transpose(1, 2)
        return (h + self.dropout(y)).masked_fill(~valid, 0.0)


class MaskedStridedConv(nn.Module):
    """mask-aware 的 3×1、stride=2 时序 CNN 下采样，不跨股票混合。"""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = ~pad_mask
        x = self.norm(h).masked_fill(~valid.unsqueeze(-1), 0.0).transpose(1, 2)
        out = self.dropout(F.gelu(self.conv(x))).transpose(1, 2)
        out_valid = F.max_pool1d(valid.to(torch.float32).unsqueeze(1), 3, 2, 1).squeeze(1).bool()
        out_pad = ~out_valid
        return out.masked_fill(out_pad.unsqueeze(-1), 0.0), out_pad


class MarketResidualDecomposition(nn.Module):
    """单个 C 层的公共分支；与 exp022 逐位一致，保证共享主干初始化不变。"""

    def __init__(self, d_model: int, gamma_init: float) -> None:
        super().__init__()
        self.residual_norm = nn.LayerNorm(d_model)
        self.market_norm = nn.LayerNorm(d_model)
        self.market_gate_norm = nn.LayerNorm(d_model)
        self.market_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, 1), nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def common(self, residual: torch.Tensor, market: torch.Tensor,
               valid: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        """计算标量 gate 与公共分支贡献；mean_gate 仅在有效股票内取均值。"""
        market_broadcast = market.unsqueeze(0).expand(residual.shape[0], -1, -1)
        gate_input = torch.cat((self.residual_norm(residual),
                                self.market_gate_norm(market_broadcast)), dim=-1)
        gate = self.gate_mlp(gate_input).masked_fill(~valid.unsqueeze(-1), 0.0)
        if mode == "mean_gate":
            count = valid.sum(dim=0).clamp_min(1).to(gate.dtype).reshape(1, -1, 1)
            mean_gate = gate.sum(dim=0, keepdim=True) / count
            gate = mean_gate.expand_as(gate).masked_fill(~valid.unsqueeze(-1), 0.0)
        common = self.market_proj(self.market_norm(market)).unsqueeze(0)
        contribution = self.gamma * gate * common
        if mode == "common_off":
            contribution = torch.zeros_like(contribution)
        return gate, contribution.masked_fill(~valid.unsqueeze(-1), 0.0)


class HierarchicalMarketResidualAttention(nn.Module):
    """exp022 的 16→8→4 主干，C1/C2 前做逐 section、逐时间市场—残差分解。"""

    def __init__(self, n_in: int = 27, d_model: int = 192, nhead: int = 8,
                 dim_ff: int = 224, dropout: float = 0.1, activation: str = "gelu",
                 norm_first: bool = True, seq_len: int = 16, n_slots: int = 8,
                 cross_time_chunk: int = 2, gamma_init: float = 0.1,
                 variant: str = "residual_common") -> None:
        super().__init__()
        if seq_len != 16:
            raise ValueError("本模型当前只定义了 16→8→4 的时间金字塔")
        if variant not in {"residual_only", "residual_common"}:
            raise ValueError("variant 必须为 residual_only 或 residual_common")
        self.model_config = {
            "n_in": n_in, "d_model": d_model, "nhead": nhead, "dim_ff": dim_ff,
            "dropout": dropout, "activation": activation, "norm_first": norm_first,
            "seq_len": seq_len, "n_slots": n_slots, "cross_time_chunk": cross_time_chunk,
            "gamma_init": gamma_init, "variant": variant,
        }
        self.seq_len, self.cross_time_chunk, self.variant = seq_len, cross_time_chunk, variant
        # 模块名称、创建顺序和初始化方式与 exp020/exp022 完全一致。
        self.input_proj = nn.Linear(n_in, d_model)
        self.pos16 = nn.Parameter(torch.zeros(1, 16, d_model))
        self.pos8 = nn.Parameter(torch.zeros(1, 8, d_model))
        self.pos4 = nn.Parameter(torch.zeros(1, 4, d_model))

        def attention_layer() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout,
                activation=activation, batch_first=True, norm_first=norm_first,
            )

        self.t1, self.t2, self.t3 = attention_layer(), attention_layer(), attention_layer()
        self.c1, self.c2 = attention_layer(), attention_layer()
        self.conv16 = MaskedTemporalConvBlock(d_model, dropout)
        self.down16_8 = MaskedStridedConv(d_model, dropout)
        self.conv8 = MaskedTemporalConvBlock(d_model, dropout)
        self.down8_4 = MaskedStridedConv(d_model, dropout)
        self.conv4 = MaskedTemporalConvBlock(d_model, dropout)
        self.slot_emb = nn.Embedding(n_slots, d_model)
        self.head = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, 1),
        )
        for position in (self.pos16, self.pos8, self.pos4):
            nn.init.normal_(position, mean=0.0, std=0.02)

        if variant == "residual_common":
            self.market_c1 = MarketResidualDecomposition(d_model, gamma_init)
            self.market_c2 = MarketResidualDecomposition(d_model, gamma_init)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _time(layer: nn.TransformerEncoderLayer, h: torch.Tensor,
              pad_mask: torch.Tensor) -> torch.Tensor:
        return layer(h, src_key_padding_mask=pad_mask).masked_fill(pad_mask.unsqueeze(-1), 0.0)

    def _cross(self, layer: nn.TransformerEncoderLayer, h: torch.Tensor,
               pad_mask: torch.Tensor) -> torch.Tensor:
        """沿股票维运行单个 section 的截面 Attention，并跳过全无效时间 slice。"""
        x, mask = h.transpose(0, 1), pad_mask.transpose(0, 1)
        pieces: list[torch.Tensor] = []
        step = max(1, int(self.cross_time_chunk))
        for lo in range(0, x.shape[0], step):
            xb, mb = x[lo:lo + step], mask[lo:lo + step]
            any_valid = (~mb).any(dim=1)
            if bool(any_valid.all()):
                pieces.append(layer(xb, src_key_padding_mask=mb).masked_fill(mb.unsqueeze(-1), 0.0))
            elif not bool(any_valid.any()):
                pieces.append(torch.zeros_like(xb))
            else:
                part = torch.zeros_like(xb)
                part[any_valid] = layer(xb[any_valid], src_key_padding_mask=mb[any_valid]).masked_fill(
                    mb[any_valid].unsqueeze(-1), 0.0)
                pieces.append(part)
        return torch.cat(pieces, dim=0).transpose(0, 1)

    @staticmethod
    def masked_market_residual(h: torch.Tensor, pad_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """对单个 section 逐时间求股票 masked mean，并将无效位置严格清零。"""
        valid = ~pad_mask
        count = valid.sum(dim=0).clamp_min(1).to(h.dtype).unsqueeze(-1)
        market = (h * valid.unsqueeze(-1)).sum(dim=0) / count
        market = market.masked_fill(~valid.any(dim=0).unsqueeze(-1), 0.0)
        residual = (h - market.unsqueeze(0)).masked_fill(~valid.unsqueeze(-1), 0.0)
        return market, residual

    def _decomposed_cross(self, layer: nn.TransformerEncoderLayer,
                          common_module: MarketResidualDecomposition | None,
                          h: torch.Tensor, pad_mask: torch.Tensor,
                          section_ids: torch.Tensor, position: torch.Tensor,
                          common_mode: str,
                          collect: bool) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """先分解纯隐表示，再给残差加位置编码并逐 section 做 Attention。"""
        if common_mode not in {"normal", "common_off", "mean_gate"}:
            raise ValueError("common_mode 必须为 normal、common_off 或 mean_gate")
        out = torch.zeros_like(h)
        traces: list[dict[str, torch.Tensor]] = []
        for section in torch.unique(section_ids, sorted=True):
            rows = section_ids == section
            hs, mask = h[rows], pad_mask[rows]
            market, residual = self.masked_market_residual(hs, mask)
            attention_input = self._add_position(residual, position, mask)
            attention = self._cross(layer, attention_input, mask)
            gate = None
            contribution = torch.zeros_like(attention)
            if common_module is not None:
                gate, contribution = common_module.common(residual, market, ~mask, common_mode)
            z = (attention + contribution).masked_fill(mask.unsqueeze(-1), 0.0)
            out[rows] = z
            if collect:
                traces.append({"section": section, "H": hs, "M": market, "R": residual,
                               "attention_input": attention_input,
                               "A": attention, "G": gate, "common": contribution, "Z": z,
                               "pad_mask": mask})
        return out, traces

    @staticmethod
    def _add_position(h: torch.Tensor, position: torch.Tensor,
                      pad_mask: torch.Tensor) -> torch.Tensor:
        return (h + position).masked_fill(pad_mask.unsqueeze(-1), 0.0)

    @staticmethod
    def _latest_and_mean(h: torch.Tensor, pad_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = ~pad_mask
        if not bool(valid.any(dim=1).all()):
            raise ValueError("每只股票至少需要一个有效的下采样时间 token")
        positions = torch.arange(h.shape[1], device=h.device).unsqueeze(0).expand_as(pad_mask)
        latest_index = positions.masked_fill(~valid, -1).max(dim=1).values
        latest = h[torch.arange(h.shape[0], device=h.device), latest_index]
        mean = (h * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        return latest, mean

    def forward_features(self, x: torch.Tensor, pad_mask: torch.Tensor,
                         section_ids: torch.Tensor | None = None, *,
                         common_mode: str = "normal", collect: bool = False):
        if x.ndim != 3 or x.shape[1:] != (16, self.model_config["n_in"]):
            raise ValueError(f"输入应为 (N,16,{self.model_config['n_in']})，实际 {tuple(x.shape)}")
        if pad_mask.shape != x.shape[:2] or pad_mask.dtype != torch.bool:
            raise ValueError("pad_mask 必须是 (N,16) 的 bool 张量")
        if section_ids is None:
            section_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        if section_ids.shape != (x.shape[0],):
            raise ValueError("section_ids 必须为每只股票一个截面标识")
        h = self._add_position(self.input_proj(x), self.pos16, pad_mask)
        h = self._time(self.t1, h, pad_mask)
        h = self.conv16(h, pad_mask)
        h, pad8 = self.down16_8(h, pad_mask)
        h, trace1 = self._decomposed_cross(
            self.c1, getattr(self, "market_c1", None), h, pad8, section_ids,
            self.pos8, common_mode, collect)
        h = self._time(self.t2, h, pad8)
        h = self.conv8(h, pad8)
        h, pad4 = self.down8_4(h, pad8)
        h, trace2 = self._decomposed_cross(
            self.c2, getattr(self, "market_c2", None), h, pad4, section_ids,
            self.pos4, common_mode, collect)
        h = self._time(self.t3, h, pad4)
        h = self.conv4(h, pad4)
        return (h, pad4, {"C1": trace1, "C2": trace2}) if collect else (h, pad4)


# ---------------------------------------------------------------------------
# 5m 高频增量编码器
# ---------------------------------------------------------------------------
class SegmentedCausalDepthwiseConv1d(nn.Module):
    """严格左因果的深度可分离时序卷积；segment 改变时历史被截断。"""

    def __init__(self, channels: int, kernel_size: int, *, bias: bool = True) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size 必须为正")
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=0,
                              groups=channels, bias=bias)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, segment: torch.Tensor) -> torch.Tensor:
        if x.shape[:2] != valid.shape or valid.shape != segment.shape:
            raise ValueError("x/valid/segment 形状不匹配")
        clean = x.masked_fill(~valid.unsqueeze(-1), 0.0).transpose(1, 2)
        padded = F.pad(clean, (self.kernel_size - 1, 0))
        windows = padded.unfold(-1, self.kernel_size, 1)
        shifted = F.pad(segment, (self.kernel_size - 1, 0), value=-1).unfold(1, self.kernel_size, 1)
        use = (shifted == segment.unsqueeze(-1)) & (shifted >= 0)
        weight = self.conv.weight.reshape(1, -1, 1, self.kernel_size)
        value = (windows * use.unsqueeze(1).to(windows.dtype) * weight).sum(-1)
        bias = self.conv.bias
        if bias is None:
            raise RuntimeError("分段深度卷积必须带 bias")
        value = value + bias[None, :, None]
        return value.transpose(1, 2)


class SdpaCrossAttention(nn.Module):
    """不物化完整权重的 SDPA cross attention，并提供小张量手算参考路径。"""

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model 必须能整除 heads")
        self.d_model, self.heads, self.head_dim = d_model, heads, d_model // heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = float(dropout)

    def _project(self, query: torch.Tensor, key_value: torch.Tensor):
        batch_shape = query.shape[:-2]
        b = int(np.prod(batch_shape))
        nq, nk = query.shape[-2], key_value.shape[-2]
        q = self.q_proj(query).reshape(b, nq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(b, nk, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(b, nk, self.heads, self.head_dim).transpose(1, 2)
        return batch_shape, b, nq, q, k, v

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, key_valid: torch.Tensor,
                query_valid: torch.Tensor | None = None) -> torch.Tensor:
        batch_shape, b, nq, q, k, v = self._project(query, key_value)
        valid = key_valid.reshape(b, key_valid.shape[-1])
        any_valid = valid.any(dim=-1)
        safe = valid.clone()
        safe[~any_valid, 0] = True
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=safe[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        ).transpose(1, 2).reshape(b, nq, self.d_model)
        out = self.out_proj(attended).reshape(*batch_shape, nq, self.d_model)
        out = out.masked_fill(~any_valid.reshape(*batch_shape, 1, 1), 0.0)
        if query_valid is not None:
            out = out.masked_fill(~query_valid.unsqueeze(-1), 0.0)
        return out

    def reference(self, query: torch.Tensor, key_value: torch.Tensor, key_valid: torch.Tensor,
                  query_valid: torch.Tensor | None = None) -> torch.Tensor:
        batch_shape, b, nq, q, k, v = self._project(query, key_value)
        valid = key_valid.reshape(b, key_valid.shape[-1])
        any_valid = valid.any(-1)
        safe = valid.clone()
        safe[~any_valid, 0] = True
        score = torch.matmul(q, k.transpose(-2, -1)) / float(self.head_dim) ** 0.5
        score = score.masked_fill(~safe[:, None, None, :], -torch.inf)
        attended = torch.matmul(torch.softmax(score, dim=-1), v)
        attended = attended.transpose(1, 2).reshape(b, nq, self.d_model)
        out = self.out_proj(attended).reshape(*batch_shape, nq, self.d_model)
        out = out.masked_fill(~any_valid.reshape(*batch_shape, 1, 1), 0.0)
        if query_valid is not None:
            out = out.masked_fill(~query_valid.unsqueeze(-1), 0.0)
        return out


class LatentSelfAttentionBlock(nn.Module):
    """逐 (section,time) 的单层 Latent self-attention + FFN。"""

    def __init__(self, d5: int, heads: int, ffn: int, dropout: float) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d5), nn.LayerNorm(d5)
        self.attn = SdpaCrossAttention(d5, heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(d5, ffn), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn, d5), nn.Dropout(dropout))

    def forward(self, latent: torch.Tensor, time_valid: torch.Tensor) -> torch.Tensor:
        all_valid = time_valid.unsqueeze(-1).expand(*time_valid.shape, latent.shape[-2])
        normed = self.norm1(latent).masked_fill(~all_valid.unsqueeze(-1), 0.0)
        h = latent + self.attn(normed, normed, all_valid)
        h = h + self.ffn(self.norm2(h))
        return h.masked_fill(~all_valid.unsqueeze(-1), 0.0)


class MaskedBinPool(nn.Module):
    """把 96 个 5m 格点按 16 个 bin（每 bin 6 格）做 masked mean + masked last 池化。"""

    def __init__(self, d5: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * d5, d5)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape[:-2]
        cells = x.reshape(*shape, 16, 6, x.shape[-1])
        mask = valid.reshape(*shape, 16, 6)
        weight = mask.unsqueeze(-1).to(x.dtype)
        denom = mask.sum(-1).clamp_min(1).to(x.dtype)
        mean = (cells * weight).sum(-2) / denom.unsqueeze(-1)
        position = torch.arange(6, device=x.device, dtype=x.dtype)
        latest_at = position.masked_fill(~mask, -1.0).max(-1).values
        safe = latest_at.clamp_min(0).to(torch.long)
        index = safe.unsqueeze(-1).unsqueeze(-1).expand(*shape, 16, 1, x.shape[-1])
        latest = cells.gather(-2, index).squeeze(-2)
        exists = mask.any(-1)
        value = self.proj(torch.cat((latest, mean), -1)).masked_fill(~exists.unsqueeze(-1), 0.0)
        return value, exists


class MaskedSegmentedTemporalConv(nn.Module):
    """96→16 的分段时序卷积（kernel=7、stride=6、右侧补 1），不跨日/午休混合。"""

    def __init__(self, channels: int, kernel_size: int = 7, stride: int = 6,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if kernel_size != stride + 1:
            raise ValueError("本实现固定 kernel=stride+1，保证输出恰好 16 个 bin")
        self.kernel_size, self.stride = int(kernel_size), int(stride)
        self.norm = nn.LayerNorm(channels)
        self.conv = nn.Conv1d(channels, channels, kernel_size, stride=stride,
                              padding=0, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor,
                segment: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != 96 or valid.shape != (x.shape[0], 96):
            raise ValueError("时序卷积输入必须是 (B,96,C)/(B,96)")
        clean = self.norm(x).masked_fill(~valid.unsqueeze(-1), 0.0).transpose(1, 2)
        padded = F.pad(clean, (0, self.kernel_size - self.stride))
        windows = padded.unfold(-1, self.kernel_size, self.stride)
        shifted = F.pad(segment, (0, self.kernel_size - self.stride), value=-1) \
            .unfold(1, self.kernel_size, self.stride)
        center = segment[:, ::self.stride]
        use = (shifted == center.unsqueeze(-1)) & (shifted >= 0)
        taps = windows * use.unsqueeze(1).to(windows.dtype)
        value = torch.einsum("bctk,ock->bot", taps, self.conv.weight)
        bias = self.conv.bias
        if bias is None:
            raise RuntimeError("分段时序卷积必须带 bias")
        value = F.gelu(value + bias[None, :, None])
        output_valid = valid.reshape(x.shape[0], 16, 6).any(-1)
        value = value.masked_fill(~output_valid.unsqueeze(1), 0.0)
        return self.dropout(value.transpose(1, 2)), output_valid


class FiveMinuteIncrementEncoder(nn.Module):
    """98 槽 → 96 格 → 16 个高频增量 token 的残差/公共编码器。"""

    def __init__(self, d5: int = 128, n_latents: int = 64, heads: int = 4,
                 latent_ffn: int = 192, local_kernel: int = 3, temporal_kernel: int = 7,
                 dropout: float = 0.1, d_model: int = 192,
                 gamma_cross_init: float = 0.1, gamma_common_init: float = 0.1) -> None:
        super().__init__()
        self.d5, self.n_latents = int(d5), int(n_latents)
        self.local_in = nn.Linear(27, d5)
        self.bin_pos = nn.Parameter(torch.zeros(1, 16, 1, d5))
        self.intra_pos = nn.Parameter(torch.zeros(1, 1, 6, d5))
        self.local_norm = nn.LayerNorm(d5)
        self.local_conv = SegmentedCausalDepthwiseConv1d(d5, local_kernel)
        self.latent_tokens = nn.Parameter(torch.empty(n_latents, d5))
        self.stock_latent_attention = SdpaCrossAttention(d5, heads, dropout)
        self.latent_self = LatentSelfAttentionBlock(d5, heads, latent_ffn, dropout)
        self.residual_norm = nn.LayerNorm(d5)
        self.bin_pool = MaskedBinPool(d5)
        self.latent_temporal = MaskedSegmentedTemporalConv(d5, temporal_kernel, 6, dropout)
        self.market_temporal = MaskedSegmentedTemporalConv(d5, temporal_kernel, 6, dropout)
        self.latent_stock_attention = SdpaCrossAttention(d5, heads, dropout)
        self.market_norm = nn.LayerNorm(d5)
        self.market_gate_norm = nn.LayerNorm(d5)
        self.market_proj = nn.Linear(d5, d5, bias=False)
        self.common_gate = nn.Sequential(
            nn.Linear(2 * d5, max(4, d5 // 4)), nn.GELU(),
            nn.Linear(max(4, d5 // 4), 1), nn.Sigmoid(),
        )
        self.gamma_cross = nn.Parameter(torch.tensor(float(gamma_cross_init)))
        self.gamma_common = nn.Parameter(torch.tensor(float(gamma_common_init)))
        self.to_model = nn.Linear(d5, d_model)
        for parameter in (self.bin_pos, self.intra_pos, self.latent_tokens):
            nn.init.normal_(parameter, mean=0.0, std=0.02)

    @staticmethod
    def _market_residual(h: torch.Tensor, valid: torch.Tensor):
        """h (S,R,T,d)、valid (S,R,T)；逐时间按真实截面求市场均值与残差。"""
        count = valid.sum(1).clamp_min(1).to(h.dtype).unsqueeze(-1)
        market = (h * valid.unsqueeze(-1)).sum(1) / count
        market = market.masked_fill(~valid.any(1).unsqueeze(-1), 0.0)
        residual = (h - market.unsqueeze(1)).masked_fill(~valid.unsqueeze(-1), 0.0)
        return market, residual

    @staticmethod
    def _group(h: torch.Tensor, valid: torch.Tensor, segment: torch.Tensor,
               section_ids: torch.Tensor):
        sections = torch.unique(section_ids, sorted=True)
        sizes = [int((section_ids == section).sum()) for section in sections]
        if not sizes or len(set(sizes)) != 1:
            raise ValueError("5m 快路要求每个 section 股票数相同；训练按 pair 两端分别前向")
        grouped_h = torch.stack([h[section_ids == section] for section in sections])
        grouped_valid = torch.stack([valid[section_ids == section] for section in sections])
        grouped_segment = torch.stack([segment[section_ids == section] for section in sections])
        return sections, grouped_h, grouped_valid, grouped_segment

    def forward(self, x5: torch.Tensor, pad5: torch.Tensor, slot: torch.Tensor,
                section_ids: torch.Tensor, gather: torch.Tensor,
                cell_segment: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        n = x5.shape[0]
        mapping = gather[slot]                                  # (n,16,6)
        flat = mapping.reshape(n, 96)
        safe = flat.clamp_min(0)
        cell_valid = (~pad5).gather(1, safe) & (flat >= 0)      # (n,96)
        cells = x5.gather(1, safe.unsqueeze(-1).expand(-1, -1, x5.shape[-1]))
        cells = cells * cell_valid.unsqueeze(-1).to(cells.dtype)
        segment = cell_segment[slot].reshape(n, 96)

        h = self.local_in(cells).reshape(n, 16, 6, self.d5)
        h = h + self.bin_pos.reshape(1, 16, 1, -1) + self.intra_pos.reshape(1, 1, 6, -1)
        h = h.reshape(n, 96, self.d5)
        h = h + self.local_conv(self.local_norm(h), cell_valid, segment)
        h = h.masked_fill(~cell_valid.unsqueeze(-1), 0.0)

        sections, gh, gv, gs = self._group(h, cell_valid, segment, section_ids)
        rows_per = gh.shape[1]
        market, residual = self._market_residual(gh, gv)

        residual_st = residual.permute(0, 2, 1, 3)              # (S,96,R,d5)
        valid_st = gv.permute(0, 2, 1)                          # (S,96,R)
        latent = self.stock_latent_attention(
            self.latent_tokens.reshape(1, 1, -1, self.d5).expand(residual_st.shape[0], 96, -1, -1),
            self.residual_norm(residual_st), valid_st)          # (S,96,L,d5)
        latent = self.latent_self(latent, valid_st.any(-1))

        section_count, latent_count = latent.shape[0], latent.shape[2]
        cell_any = gv.any(1)                                    # (S,96)
        seg_section = gs[:, 0]                                  # (S,96)
        latent_in = latent.permute(0, 2, 1, 3).reshape(section_count * latent_count, 96, self.d5)
        latent_valid = cell_any.unsqueeze(1).expand(section_count, latent_count, 96) \
            .reshape(section_count * latent_count, 96)
        latent_segment = seg_section.unsqueeze(1).expand(section_count, latent_count, 96) \
            .reshape(section_count * latent_count, 96)
        latent16, bin_valid = self.latent_temporal(latent_in, latent_valid, latent_segment)
        latent16 = latent16.reshape(section_count, latent_count, 16, self.d5).permute(0, 2, 1, 3)
        bin_valid = bin_valid.reshape(section_count, latent_count, 16).permute(0, 2, 1)

        market16, _ = self.market_temporal(market, cell_any, seg_section)   # (S,16,d5)

        r16, rv16 = self.bin_pool(residual.reshape(section_count * rows_per, 96, self.d5),
                                  gv.reshape(section_count * rows_per, 96))
        r16 = r16.reshape(section_count, rows_per, 16, self.d5).permute(0, 2, 1, 3)
        rv16 = rv16.reshape(section_count, rows_per, 16).permute(0, 2, 1)

        attention = self.latent_stock_attention(r16, latent16, bin_valid, rv16)
        market_broadcast = market16.unsqueeze(2).expand(-1, -1, rows_per, -1)
        common = self.market_proj(self.market_norm(market_broadcast))
        gate = self.common_gate(torch.cat((self.residual_norm(r16),
                                           self.market_gate_norm(market_broadcast)), -1))
        common_valid = rv16 & bin_valid.any(-1).unsqueeze(2)
        gate = gate.masked_fill(~common_valid.unsqueeze(-1), 0.0)
        attention = attention.masked_fill(~rv16.unsqueeze(-1), 0.0)
        contribution = (self.gamma_common * gate * common) \
            .masked_fill(~common_valid.unsqueeze(-1), 0.0)
        z = (r16 + self.gamma_cross * attention + contribution) \
            .masked_fill(~rv16.unsqueeze(-1), 0.0)

        z_flat = z.permute(0, 2, 1, 3).reshape(section_count * rows_per, 16, self.d5)
        valid_flat = rv16.permute(0, 2, 1).reshape(section_count * rows_per, 16)
        order = torch.cat([torch.nonzero(section_ids == section, as_tuple=False).squeeze(1)
                           for section in sections])
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(len(order), device=order.device)
        tokens = self.to_model(z_flat[inverse])
        pad16 = ~valid_flat[inverse]
        info = {"cell_valid": cell_valid, "cell_segment": segment, "R5": residual,
                "latent": latent, "latent16": latent16, "bin_valid": bin_valid,
                "R16": r16, "M16": market16, "A16": attention, "G16": gate,
                "common16": contribution, "tokens16": tokens}
        return tokens.masked_fill(pad16.unsqueeze(-1), 0.0), pad16, info


class QueryGuidedCrossFrequency(nn.Module):
    """4 个 30m 表征 token 作为 query，主动查询 16 个 5m 增量 token，门控残差回注。"""

    def __init__(self, d_model: int, heads: int, dropout: float, gamma_init: float) -> None:
        super().__init__()
        self.q_norm, self.kv_norm = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = SdpaCrossAttention(d_model, heads, dropout)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, 1), nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, query: torch.Tensor, query_pad: torch.Tensor,
                key_value: torch.Tensor, key_pad: torch.Tensor) -> tuple[torch.Tensor, dict]:
        query_valid = ~query_pad
        attention = self.attn(self.q_norm(query), self.kv_norm(key_value), ~key_pad, query_valid)
        gate = self.gate(torch.cat((self.q_norm(query), attention), -1)) \
            .masked_fill(~query_valid.unsqueeze(-1), 0.0)
        increment = self.gamma * gate * attention
        return query + increment, {"attention": attention, "gate": gate, "increment": increment}


class CrossFrequencyMarketResidual(HierarchicalMarketResidualAttention):
    """exp022 主干 + 5m 增量编码器 + 查询驱动跨频门控残差，最后接原预测 Head。"""

    def __init__(self, *, d5: int = 128, n_latents: int = 64, hf_heads: int = 4,
                 latent_ffn: int = 192, local_kernel: int = 3, temporal_kernel: int = 7,
                 freq_heads: int = 8, freq_dropout: float = 0.1,
                 gamma_cross_init: float = 0.1, gamma_common_init: float = 0.1,
                 gamma_freq_init: float = 0.1, **kwargs) -> None:
        kwargs.setdefault("variant", "residual_common")
        if kwargs["variant"] != "residual_common":
            raise ValueError("exp032 只定义 residual_common 主干")
        super().__init__(**kwargs)
        self.model_config = {
            **self.model_config, "d5": d5, "n_latents": n_latents, "hf_heads": hf_heads,
            "latent_ffn": latent_ffn, "local_kernel": local_kernel,
            "temporal_kernel": temporal_kernel, "freq_heads": freq_heads,
            "freq_dropout": freq_dropout, "gamma_cross_init": gamma_cross_init,
            "gamma_common_init": gamma_common_init, "gamma_freq_init": gamma_freq_init,
        }
        # 新模块必须晚于完整 exp022 主干创建，避免改变共享参数的随机初始化序列。
        self.five = FiveMinuteIncrementEncoder(
            d5=d5, n_latents=n_latents, heads=hf_heads, latent_ffn=latent_ffn,
            local_kernel=local_kernel, temporal_kernel=temporal_kernel,
            dropout=self.model_config["dropout"], d_model=self.model_config["d_model"],
            gamma_cross_init=gamma_cross_init, gamma_common_init=gamma_common_init)
        self.freq = QueryGuidedCrossFrequency(
            self.model_config["d_model"], freq_heads, freq_dropout, gamma_freq_init)

    def forward(self, x: torch.Tensor, slot: torch.Tensor, pad_mask: torch.Tensor,
                section_ids: torch.Tensor | None = None, *, x5: torch.Tensor | None = None,
                pad5: torch.Tensor | None = None, gather: torch.Tensor | None = None,
                cell_segment: torch.Tensor | None = None, common_mode: str = "normal",
                freq_off: bool = False, collect: bool = False):
        features = self.forward_features(x, pad_mask, section_ids,
                                         common_mode=common_mode, collect=collect)
        h, pad4 = features[:2]
        freq_trace = None
        if not freq_off:
            if x5 is None or pad5 is None or gather is None or cell_segment is None:
                raise ValueError("未关闭跨频分支时必须提供 x5/pad5/gather/cell_segment")
            if section_ids is None:
                section_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            tokens16, pad16, five_trace = self.five(
                x5, pad5, slot, section_ids, gather, cell_segment)
            h, freq_trace = self.freq(h, pad4, tokens16, pad16)
            freq_trace = {**freq_trace, "five": five_trace, "pad16": pad16}
        latest, mean = self._latest_and_mean(h, pad4)
        if slot.shape != (x.shape[0],):
            raise ValueError("slot 必须为每只股票一个整数")
        prediction = self.head(torch.cat((latest, mean, self.slot_emb(slot)), dim=-1)).squeeze(-1)
        if collect:
            traces = features[2]
            traces["freq"] = freq_trace
            return prediction, traces
        return prediction


def save_model(payload: dict, path: Path) -> Path:
    """把 checkpoint 张量无损编码进文本 JSON，并硬性限制文件小于 50 MB。"""
    def convert(value):
        if isinstance(value, torch.Tensor):
            cpu = value.detach().cpu().contiguous()
            return {"dtype": str(cpu.dtype).removeprefix("torch."), "shape": list(cpu.shape),
                    "encoding": "base64_raw", "data": base64.b64encode(cpu.numpy().tobytes()).decode("ascii")}
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value
    document = json.dumps(convert(payload), ensure_ascii=False, separators=(",", ":"))
    size = len(document.encode("utf-8"))
    if size >= MAX_WEIGHT_JSON_BYTES:
        raise RuntimeError(f"JSON 权重 {size} bytes，不满足 <{MAX_WEIGHT_JSON_BYTES} bytes 硬限制")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path


def load_model(path: str | Path, map_location: str = "cpu") -> dict:
    """读取 exp032 JSON 文本权重并还原 state_dict 张量。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    state = {}
    for name, meta in payload["state_dict"].items():
        dtype = getattr(torch, meta["dtype"])
        if meta.get("encoding") == "base64_raw":
            array = np.frombuffer(base64.b64decode(meta["data"]), dtype=meta["dtype"]).copy()
            state[name] = torch.from_numpy(array).to(dtype=dtype).reshape(meta["shape"]).to(map_location)
        else:
            state[name] = torch.tensor(meta["data"], dtype=dtype).reshape(meta["shape"]).to(map_location)
    return {**{key: value for key, value in payload.items() if key != "state_dict"}, "state_dict": state}


def build_model_from_cfg(model_cfg: dict) -> CrossFrequencyMarketResidual:
    """按 JSON 中保存的固定配置重建 exp032 模型。"""
    return CrossFrequencyMarketResidual(**model_cfg)
