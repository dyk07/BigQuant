"""
切窗 + Dataset

切窗规则（防泄漏）：
- 输入窗口 [t - seq_len, t) 与标签窗口 [t, t + horizon) 严格不重叠
- 标签窗口不得跨越交易日
- 采样点按**时钟分钟**对齐（minute_of_day），不依赖行号，
  因此缺 bar / 停牌不会导致采样时刻漂移
- 标准化统计量与标签截断边界只用训练集 fit
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from features import ALL_FEATURES


def make_samples(df: pd.DataFrame, seq_len: int, sample_minutes=None,
                 input_same_day_only: bool = True):
    """
    返回 (X, y, meta)

    X:    [N, seq_len, F] float32
    y:    [N] float32
    meta: list of (instrument_id, day, minute_of_day)，用于按截面计算 IC

    sample_minutes:      允许采样的**时钟分钟**集合（None 表示全部）
    input_same_day_only: 输入窗口是否必须落在同一交易日内。

    label 一定为 NaN 时会被剔除，而 label 只依赖**未来**窗口（且不跨交易日），
    所以输入窗口是否可以跨日回溯不影响标签的合法性，也不引入未来信息。
    对高频表（bar5m/bar1m）关闭该限制可以显著增加样本数。

    实现为向量化，避免在数百万行上做纯 Python 循环。
    """
    df = df.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    n = len(df)
    if n <= seq_len:
        return (np.zeros((0, seq_len, len(ALL_FEATURES)), dtype=np.float32),
                np.array([], dtype=np.float32), [])

    feat_arr = df[ALL_FEATURES].to_numpy(dtype=np.float32)
    label_arr = df["label"].to_numpy(dtype=np.float64)
    inst_arr = df["instrument_id"].to_numpy()
    minute_arr = df["minute_of_day"].to_numpy()
    day_arr = df["day"].to_numpy()

    i = np.arange(seq_len, n)
    keep = ~np.isnan(label_arr[i])
    if sample_minutes is not None:
        allowed = np.asarray(sorted(set(int(m) for m in sample_minutes)))
        keep &= np.isin(minute_arr[i], allowed)
    # 输入窗口必须全部来自同一只股票
    keep &= (inst_arr[i] == inst_arr[i - seq_len])
    if input_same_day_only:
        keep &= (day_arr[i] == day_arr[i - seq_len])

    sel = i[keep]
    if sel.size == 0:
        return (np.zeros((0, seq_len, len(ALL_FEATURES)), dtype=np.float32),
                np.array([], dtype=np.float32), [])

    offsets = np.arange(-seq_len, 0)
    win_idx = sel[:, None] + offsets[None, :]
    X = feat_arr[win_idx]                       # [N, seq_len, F]
    y = label_arr[sel].astype(np.float32)
    meta = [(int(inst_arr[j]), day_arr[j], int(minute_arr[j])) for j in sel]
    return X, y, meta


def fit_scaler(X: np.ndarray, chunk: int = 200_000):
    """
    只用训练集统计每个特征的均值和标准差，忽略 NaN。

    分块统计（Welford 风格的累积和），避免把整个训练集上转成 float64
    造成数 GB 的临时数组（大数据集下会被 OOM kill）。
    """
    n_feat = X.shape[-1]
    total = np.zeros(n_feat, dtype=np.float64)      # sum of valid values
    total_sq = np.zeros(n_feat, dtype=np.float64)   # sum of squares
    count = np.zeros(n_feat, dtype=np.float64)
    for start in range(0, len(X), chunk):
        block = X[start:start + chunk].reshape(-1, n_feat).astype(np.float64)
        mask = ~np.isnan(block)
        block = np.where(mask, block, 0.0)
        total += block.sum(axis=0)
        total_sq += (block * block).sum(axis=0)
        count += mask.sum(axis=0)

    count = np.maximum(count, 1.0)
    mean = total / count
    var = np.maximum(total_sq / count - mean * mean, 0.0)
    std = np.sqrt(var)
    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.where((np.isnan(std)) | (std < 1e-6), 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_scaler(X: np.ndarray, mean, std, inplace: bool = True):
    """
    标准化，并把 NaN（缺失值）填成 0（即训练集均值）。

    分块 + float32 原地运算，避免生成和输入同样大的额外副本。
    """
    out = X if inplace else X.copy()
    n_feat = out.shape[-1]
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, n_feat)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, n_feat)
    chunk = max(1, 2_000_000 // max(out.shape[1], 1))
    for start in range(0, len(out), chunk):
        block = out[start:start + chunk]
        # 用 nan_to_num 前先做 (x-mean)/std（NaN 会保留）
        np.subtract(block, mean, out=block)
        np.divide(block, std, out=block)
        np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def make_section_batches(meta, min_stocks=10, shuffle=True, seed=42):
    """按截面 (day, minute_of_day) 分组，每个 batch = 一个完整截面。

    返回 list[np.ndarray]，每个数组是该截面样本的**全局下标**。
    数组长度可不同（不同截面股票数不相等），由 batch_sampler + collate_fn
    直接按截面 stack，无需 padding。

    min_stocks 与 metrics.compute_ic 的阈值对齐：过小的截面在训练和评估中
    同时被丢弃，避免出现 pred=NaN 污染评估。
    """
    df = pd.DataFrame({
        "i": np.arange(len(meta), dtype=np.int64),
        "day": [m[1] for m in meta],
        "min": [m[2] for m in meta],
    })
    batches = [
        g["i"].to_numpy()
        for _, g in df.groupby(["day", "min"], sort=False)
        if len(g) >= min_stocks
    ]
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(batches)
    return batches


def collate_sections(batch):
    """单个截面内 [..., (X, y, idx)] -> (X, y, idx)。"""
    X = torch.stack([b[0] for b in batch], dim=0)          # [n, T, F]
    y = torch.stack([b[1] for b in batch], dim=0)          # [n]
    idx = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return X, y, idx


class StockDataset(Dataset):
    def __init__(self, X, y, return_idx=False):
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        self.return_idx = return_idx

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.return_idx:
            # 返回全局下标，供截面采样后的 predict 按原顺序还原
            return self.X[idx], self.y[idx], idx
        return self.X[idx], self.y[idx]
