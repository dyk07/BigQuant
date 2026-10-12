"""
损失函数。

比赛的评估指标是截面 IC（秩相关），而 MSE 对"排序"是间接的：
当标签均值≈0 时，MSE 会被尺度项主导，模型容易退化成常数输出（IC=0）。
因此除了 MSE，这里提供与 IC 直接对齐的相关性损失。

- mse:  均方误差，回归原始收益率或 rank 目标
- corr: 1 - batch 内 Pearson 相关（对 rank 目标使用 -> soft-IC 代理）
- ic:   1 - batch 内 Spearman 相关（先做 batch 内 rank，再算相关）
"""
import torch


def mse_loss(pred, target, **_):
    return torch.nn.functional.mse_loss(pred, target)


def _pearson(pred, target, eps=1e-8):
    p = pred - pred.mean()
    t = target - target.mean()
    denom = torch.sqrt((p * p).sum() * (t * t).sum()) + eps
    return (p * t).sum() / denom


def _rank_torch(x):
    """可微性牺牲一点：batch 内平均秩（用 argsort 实现）。"""
    order = torch.argsort(x)
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(len(x), dtype=x.dtype, device=x.device)
    return ranks


def corr_loss(pred, target, **_):
    """1 - Pearson(pred, target)。配合 rank 目标即为 soft-IC 损失。"""
    if pred.numel() < 2:
        return pred.sum() * 0.0
    return 1.0 - _pearson(pred, target)


def ic_loss(pred, target, **_):
    """1 - Spearman(pred, target)（batch 内先取秩）。"""
    if pred.numel() < 2:
        return pred.sum() * 0.0
    return 1.0 - _pearson(_rank_torch(pred), _rank_torch(target))


def combined_loss(pred, target, alpha: float = 0.5, **_):
    """alpha*MSE + (1-alpha)*corr，兼顾数值拟合与排序。"""
    return (alpha * mse_loss(pred, target)
            + (1.0 - alpha) * corr_loss(pred, target))


LOSSES = {
    "mse": mse_loss,
    "corr": corr_loss,
    "ic": ic_loss,
}


def get_loss(name: str, **kwargs):
    if name == "combined":
        return lambda p, t: combined_loss(p, t, **kwargs)
    if name not in LOSSES:
        raise ValueError(f"unknown loss: {name}; choose from {list(LOSSES)}")
    return LOSSES[name]
