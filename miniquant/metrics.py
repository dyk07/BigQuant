import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr as _spearmanr
except Exception:  # pragma: no cover
    _spearmanr = None


def _rank(x: np.ndarray) -> np.ndarray:
    """average rank（与 scipy 的 spearman 口径一致）。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype="float64")
    ranks[order] = np.arange(len(x), dtype="float64")
    # 处理并列：相同值取平均名次
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom <= 0:
        return np.nan
    return float((ra * rb).sum() / denom)


def rank_targets(labels, meta, scale: float = 1.0):
    """
    把标签转成"截面 rank 目标"（按 (day, bar_idx) 分组做百分位排名）。

    比赛按截面 IC 评估排序能力，而 MSE 直接回归收益率均值≈0 的信号时，
    模型容易退化成常数输出（IC=0）。改成回归截面百分位排名后，梯度只关心
    "谁应该排在谁前面"，与比赛的排序目标一致，且数值范围固定在 [-scale, scale]。

    只在同截面内排名，不使用任何未来信息。
    """
    labels = np.asarray(labels, dtype="float64")
    df = pd.DataFrame({
        "label": labels,
        "day": [m[1] for m in meta],
        "bar_idx": [m[2] for m in meta],
    })
    # pct=True -> (0, 1]，并列取平均名次
    pct = df.groupby(["day", "bar_idx"])["label"].rank(pct=True, method="average")
    out = (pct.to_numpy(dtype="float64") - 0.5) * 2.0 * scale
    return out.astype(np.float32)


def make_targets(labels, meta, kind: str = "rank", scale: float = 1.0):
    """kind: 'rank' -> 截面百分位目标；'raw' -> 原始收益率标签。"""
    if kind == "raw":
        return np.asarray(labels, dtype=np.float32)
    if kind == "rank":
        return rank_targets(labels, meta, scale=scale)
    raise ValueError(f"unknown target kind: {kind}")


def compute_ic(preds, labels, meta, min_stocks: int = 10, rank_inputs: bool = True):
    """
    meta: list of (instrument_id, day, bar_idx)
    按 (day, bar_idx) 分截面，算 Spearman IC，再对截面取平均。

    rank_inputs: 先做截面 rank 再算 Pearson，等价于 Spearman，但可避免
                 大量并列值（模型输出退化时）导致 scipy 返回 NaN。

    统一返回 5 元组 (IC_mean, IC_std, ICIR, n_sections, ICIR_ann)。
    """
    preds = np.asarray(preds).ravel()
    labels = np.asarray(labels).ravel()
    df = pd.DataFrame({
        "pred": preds,
        "label": labels,
        "day": [m[1] for m in meta],
        "bar_idx": [m[2] for m in meta],
    })
    ics = []
    for _, g in df.groupby(["day", "bar_idx"]):
        if len(g) < min_stocks:
            continue
        p = g["pred"].to_numpy(dtype="float64")
        l = g["label"].to_numpy(dtype="float64")
        if np.all(p == p[0]) or np.all(l == l[0]):
            continue
        if rank_inputs or _spearmanr is None:
            ic = _spearman(p, l)
        else:
            ic = float(_spearmanr(p, l)[0])
        if not np.isnan(ic):
            ics.append(ic)
    if not ics:
        return 0.0, 0.0, 0.0, 0, 0.0
    ics = np.array(ics, dtype="float64")
    ic_mean = float(ics.mean())
    ic_std = float(ics.std())
    icir = ic_mean / (ic_std + 1e-8)
    # 按截面数量年化（每个截面 = 一个采样时点）
    icir_ann = icir * np.sqrt(len(ics))
    return ic_mean, ic_std, icir, len(ics), icir_ann


def ic_significance(ic_stats):
    """均值 IC 的标准误与 t 值：|t|<2 说明与 0 无显著差异。"""
    ic_mean, ic_std, _, n, _ = ic_stats
    if n <= 1:
        return float("nan"), float("nan")
    se = ic_std / np.sqrt(n)
    t = ic_mean / (se + 1e-12)
    return float(se), float(t)


def print_ic(tag, ic_stats):
    ic_mean, ic_std, icir, n_sec, icir_ann = ic_stats
    se, t = ic_significance(ic_stats)
    print(f"[{tag}] IC={ic_mean:+.4f} IC_std={ic_std:.4f} "
          f"ICIR={icir:+.4f} ICIR_ann={icir_ann:+.2f} n={n_sec} "
          f"SE={se:.4f} t={t:+.2f}")
    return ic_mean
