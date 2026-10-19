"""对拍: 向量化分组 corr vs 原串行分组 corr, 数值应一致。"""
import torch
import torch.nn.functional as F


def _corr_batch(p, y):
    p = p.float(); y = y.float()
    pc = p - p.mean(); yc = y - y.mean()
    denom = torch.sqrt((pc ** 2).sum() * (yc ** 2).sum() + 1e-8)
    return (pc * yc).sum() / denom


def grouped_corr_loop(p, y, sidb):
    """原实现: 逐 group 串行 corr。返回标量 loss 值 (各 group corr 均值)。"""
    total = 0.0
    cnt = 0
    for g in torch.unique(sidb):
        m = sidb == g
        if m.sum() < 2:
            continue
        total = total + _corr_batch(p[m], y[m])
        cnt += 1
    return total / cnt if cnt else _corr_batch(p, y)


def grouped_corr_vec(p, y, sidb):
    """向量化: scatter 归约, 一次算所有 group 的 corr。返回标量均值。"""
    p = p.float(); y = y.float()
    ids = torch.unique(sidb)
    gid = torch.searchsorted(ids, sidb)          # 连续 id 0..G-1
    G = ids.numel()
    cnt = torch.bincount(gid, minlength=G).float()  # [G]
    sp = torch.zeros(G, device=p.device, dtype=p.dtype)
    sy = torch.zeros(G, device=p.device, dtype=p.dtype)
    sp.scatter_add_(0, gid, p)
    sy.scatter_add_(0, gid, y)
    mp = sp / cnt
    my = sy / cnt
    pc = p - mp[gid]
    yc = y - my[gid]
    ssp = torch.zeros(G, device=p.device, dtype=p.dtype)
    ssy = torch.zeros(G, device=p.device, dtype=p.dtype)
    sxy = torch.zeros(G, device=p.device, dtype=p.dtype)
    ssp.scatter_add_(0, gid, pc * pc)
    ssy.scatter_add_(0, gid, yc * yc)
    sxy.scatter_add_(0, gid, pc * yc)
    denom = torch.sqrt(ssp * ssy + 1e-8)
    corrs = sxy / denom                        # [G]
    valid = cnt >= 2
    if valid.sum() == 0:
        return _corr_batch(p, y)
    return corrs[valid].mean()


torch.manual_seed(0)
for trial in range(5):
    B = 4096
    G = int(torch.randint(2, 80, (1,)).item())
    # 构造稀疏截面编号 (带空洞, 模拟 day*1440+mod)
    uniq = torch.randint(0, 1000000, (G,)).sort()[0]
    idx = torch.randint(0, G, (B,))
    sidb = uniq[idx]
    p = torch.randn(B)
    y = torch.randn(B)
    a = grouped_corr_loop(p, y, sidb)
    b = grouped_corr_vec(p, y, sidb)
    err = abs(a - b).item()
    # 梯度对拍
    p2 = p.clone().requires_grad_(True)
    y2 = y.clone().requires_grad_(True)
    grouped_corr_loop(p2, y2, sidb).backward()
    gp_a = p2.grad.clone()
    p3 = p.clone().requires_grad_(True)
    y3 = y.clone().requires_grad_(True)
    grouped_corr_vec(p3, y3, sidb).backward()
    gp_b = p3.grad.clone()
    gerr = (gp_a - gp_b).abs().max().item()
    print(f"trial{trial} G={G}  val_err={err:.2e}  grad_err={gerr:.2e}")
