"""
双流模型：
- K 线流: 线性投影 + GRU（共享时间轴）
- 盘口流: 逐时间步 MLP（共享权重，等价于 1x1 卷积）
- 融合: concat + 时间注意力池化（再与平均池化拼接，提升稳定性）
- 输出: 标量分数（截面排序用，不需要激活函数）

注意：模型只做序列编码，不引入任何手工因子。盘口流与 K 线流各自被标准化后输入。
"""
import torch
import torch.nn as nn

from features import KLINE_FEATURES, ORDERBOOK_FEATURES


class KlineEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers,
                          batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):                    # [B, T, in_dim]
        x = self.proj(x)
        out, _ = self.gru(x)                 # [B, T, hidden]
        return self.norm(out)


class OrderBookEncoder(nn.Module):
    def __init__(self, in_dim, hidden=32, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):                    # [B, T, in_dim]
        return self.norm(self.mlp(x))        # [B, T, hidden]


class TimeAttentionPooling(nn.Module):
    """注意力池化 + 平均池化拼接（对时间轴做池化，输出与 T 无关）。"""

    def __init__(self, dim):
        super().__init__()
        h = max(dim // 2, 8)
        self.score = nn.Sequential(
            nn.Linear(dim, h),
            nn.Tanh(),
            nn.Linear(h, 1),
        )
        self.out_dim = dim * 2

    def forward(self, x):                    # [B, T, D]
        w = torch.softmax(self.score(x), dim=1)   # [B, T, 1]
        att = (x * w).sum(dim=1)                  # [B, D]
        avg = x.mean(dim=1)                       # [B, D]
        return torch.cat([att, avg], dim=-1)      # [B, 2D]


class E2EModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.kline_enc = KlineEncoder(
            len(KLINE_FEATURES),
            hidden=cfg.kline_hidden,
            layers=cfg.gru_layers,
            dropout=cfg.dropout,
        )
        self.ob_enc = OrderBookEncoder(
            len(ORDERBOOK_FEATURES),
            hidden=cfg.orderbook_hidden,
            dropout=cfg.dropout,
        )
        d = cfg.kline_hidden + cfg.orderbook_hidden
        self.fuse = nn.Sequential(
            nn.Linear(d, cfg.fused_dim),
            nn.ReLU(),
            nn.LayerNorm(cfg.fused_dim),
        )
        self.pool = TimeAttentionPooling(cfg.fused_dim)
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(self.pool.out_dim, cfg.fused_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fused_dim, 1),
        )

    def forward(self, x):                    # [B, T, F_total]
        n_k = len(KLINE_FEATURES)
        xk = x[..., :n_k]
        xo = x[..., n_k:]
        hk = self.kline_enc(xk)              # [B, T, d_k]
        ho = self.ob_enc(xo)                 # [B, T, d_o]
        h = torch.cat([hk, ho], dim=-1)      # [B, T, d_k + d_o]
        h = self.fuse(h)                     # [B, T, d_fused]
        h = self.pool(h)                     # [B, 2*d_fused]
        return self.head(h).squeeze(-1)      # [B]


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
