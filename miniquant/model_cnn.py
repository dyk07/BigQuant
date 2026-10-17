"""
CNN 滑动窗口 + Transformer 模型

架构：
  Input [B, T, 25]
    → 1x1 Conv 投影到 hidden_dim（融合所有特征，替代拆分流）
    → 多个 Conv1d 残差块（kernel=3，捕获局部模式，可膨胀捕获多尺度）
    → 正弦位置编码
    → Transformer Encoder（self-attention 捕获长程依赖）
    → 注意力池化 + 平均池化 → [B, 2*hidden_dim]
    → MLP head → scalar

与原来 GRU 方案的关键差异：
  - CNN 替代 GRU：局部感受野更灵活，训练更快（全并行）
  - Transformer 替代简单池化：显式建模序列内任意两位置的关系
  - 早期融合 K 线 + 盘口：让 CNN 直接学习跨特征的局部模式
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """正弦位置编码（非学习式，泛化到任意长度）。"""

    def __init__(self, dim: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:, :x.size(1), :]


class ConvResBlock(nn.Module):
    """
    1D 卷积残差块（Pre-LN + GELU）。

    两层 Conv1d，每层前做 LayerNorm。膨胀系数递增以捕获多尺度模式。
    输入/输出形状均为 [B, dim, T]。
    """

    def __init__(self, dim: int, kernel_size: int = 3, dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=pad,
                               dilation=dilation)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=pad,
                               dilation=dilation)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, dim, T]
        residual = x
        # Pre-norm: LN over channel dim (transpose to [B, T, dim])
        x = x.transpose(1, 2)          # [B, T, dim]
        x = self.norm1(x)
        x = x.transpose(1, 2)          # [B, dim, T]
        x = F.gelu(x)
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + residual


class AttentionPooling(nn.Module):
    """注意力池化 + 平均池化拼接（与原有 TimeAttentionPooling 相同）。"""

    def __init__(self, dim: int):
        super().__init__()
        h = max(dim // 2, 8)
        self.score = nn.Sequential(
            nn.Linear(dim, h),
            nn.Tanh(),
            nn.Linear(h, 1),
        )
        self.out_dim = dim * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        w = torch.softmax(self.score(x), dim=1)  # [B, T, 1]
        att = (x * w).sum(dim=1)                   # [B, D]
        avg = x.mean(dim=1)                        # [B, D]
        return torch.cat([att, avg], dim=-1)        # [B, 2D]


class CrossSectionAttention(nn.Module):
    """截面自注意力：同截面内一只股票 attend 到其余股票。

    输入 [N, D]（一个截面的池化表示），输出 [N, D]（上下文增强后）。
    对 N 维置换等变，不引入 per-instrument embedding，避免记住股票身份、
    在测试集新股票上泛化变差。
    """

    def __init__(self, dim, heads=4, layers=1, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(dim, heads, dropout=dropout,
                                              batch_first=True),
                "norm1": nn.LayerNorm(dim),
                "ffn": nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                ),
                "norm2": nn.LayerNorm(dim),
            }))

    def forward(self, z):                    # [N, D]
        z = z.unsqueeze(0)                   # [1, N, D]（1 = 截面这一维）
        for block in self.blocks:
            a, _ = block["attn"](z, z, z, need_weights=False)
            z = block["norm1"](z + a)
            z = block["norm2"](z + block["ffn"](z))
        return z.squeeze(0)                  # [N, D]


class CNNTransformerEncoder(nn.Module):
    """
    CNN + Transformer 编码器（不含最终 head）。

    Input:  [B, T, in_dim]   (25 个标准化特征)
    Output: [B, out_dim]      (池化后的向量)
    """

    def __init__(self, in_dim: int = 25, hidden_dim: int = 64,
                 conv_blocks: int = 3, kernel_size: int = 3,
                 dilation_growth: int = 1,
                 transformer_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        # 1x1 卷积把 25 个特征投影到 hidden_dim
        self.stem = nn.Conv1d(in_dim, hidden_dim, 1)

        dilations = [dilation_growth ** i for i in range(conv_blocks)]
        self.conv_blocks = nn.ModuleList([
            ConvResBlock(hidden_dim, kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])

        self.pos_enc = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers)

        # 最后做 LayerNorm，给下游一个归一化的表示
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, in_dim]
        x = x.transpose(1, 2)               # [B, in_dim, T]
        x = self.stem(x)                    # [B, hidden_dim, T]
        for block in self.conv_blocks:
            x = block(x)                    # [B, hidden_dim, T]
        x = x.transpose(1, 2)               # [B, T, hidden_dim]
        x = self.pos_enc(x)
        x = self.transformer(x)             # [B, T, hidden_dim]
        x = self.out_norm(x)
        return x


class CNNTransformerModel(nn.Module):
    """
    完整的单频率 CNN + Transformer 模型。

    结构：Encoder → Pooling → Head → scalar

    用于替换原来的 E2EModel，前向接口完全兼容：
      forward(x: [B, T, 25]) -> [B]
    """

    def __init__(self, cfg):
        super().__init__()
        from features import KLINE_FEATURES, ORDERBOOK_FEATURES
        in_dim = len(KLINE_FEATURES) + len(ORDERBOOK_FEATURES)  # 25
        hidden = getattr(cfg, "cnn_hidden", 64)
        conv_blocks = getattr(cfg, "cnn_blocks", 3)
        kernel_size = getattr(cfg, "cnn_kernel", 3)
        transformer_layers = getattr(cfg, "transformer_layers", 2)
        n_heads = getattr(cfg, "transformer_heads", 4)
        dropout = getattr(cfg, "dropout", 0.1)

        self.encoder = CNNTransformerEncoder(
            in_dim=in_dim,
            hidden_dim=hidden,
            conv_blocks=conv_blocks,
            kernel_size=kernel_size,
            transformer_layers=transformer_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.pool = AttentionPooling(hidden)
        use_xsec = getattr(cfg, "use_xsec", False)
        if use_xsec:
            xsec_layers = getattr(cfg, "xsec_layers", 1)
            xsec_heads = getattr(cfg, "xsec_heads", 4)
            self.xsec = CrossSectionAttention(
                self.pool.out_dim,
                heads=xsec_heads,
                layers=xsec_layers,
                dropout=dropout,
            )
            head_in = self.pool.out_dim * 2   # 原始表示 ‖ 截面上下文表示
        else:
            self.xsec = None
            head_in = self.pool.out_dim
        head_dim = getattr(cfg, "fused_dim", 64)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 25]
        h = self.encoder(x)   # [B, T, hidden_dim]
        z = self.pool(h)      # [B, 2*hidden_dim]
        if self.xsec is not None:
            z = torch.cat([z, self.xsec(z)], dim=-1)  # 残差式拼接
        return self.head(z).squeeze(-1)  # [B]


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 快速验证
# ============================================================
if __name__ == "__main__":
    # Smoke test with random data
    B, T, F = 4, 32, 25
    x = torch.randn(B, T, F)

    class FakeCfg:
        cnn_hidden = 64
        cnn_blocks = 2
        cnn_kernel = 3
        transformer_layers = 2
        transformer_heads = 4
        dropout = 0.1
        fused_dim = 64

    model = CNNTransformerModel(FakeCfg())
    print(f"params = {count_parameters(model):,}")
    y = model(x)
    print(f"input {x.shape} -> output {y.shape}")  # [4]
    print(f"output range: [{y.min().item():.4f}, {y.max().item():.4f}]")