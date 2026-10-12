"""
多频率融合（方案 A）

思路
----
把 bar30m / bar15m / bar5m / bar1m 四个频率作为**四条独立分支**同时喂给模型：
每个频率各自编码（K 线 GRU + 盘口 MLP + 时间池化），再拼接后统一出分。

前提（已实测验证）
------------------
四个频率在 (instrument_id, day, minute_of_day) 上的交集为 99.99%
（202301: 107,817 / 107,829），因此可以做**严格同一决策时刻**的
inner join —— 这是多频率融合正确性的基础。

为什么必须落盘
--------------
四频率的 X 需要 15.2 KB/样本（bar30m 0.4KB + bar15m 0.9KB + bar5m 2.6KB +
bar1m 12.0KB）。48 个月训练集约 517 万样本 -> **80 GB**，远超本机 15 GB 内存。
因此这里逐月构建、直接写成 `.npy` 分块，训练时用 memmap 惰性读取。

目录结构
--------
features_mf/<tag>/
  meta.json                 频率列表、每频率的 T、scaler、clip、分块清单
  c0000_X_bar1m.npy         第 0 个月块的 bar1m 特征 [n, T, 25] float32
  c0000_X_bar5m.npy
  c0000_y.npy               标签 [n] float32（已按训练集边界截断）
  c0000_keys.npy            [n, 3] int64: instrument_id, day_ordinal, minute_of_day
  c0001_...
"""
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import Config
from data_loader import load_months, NEEDED_COLS, filter_auction_and_tag_day
from features import (build_features, build_labels, compute_clip_bounds,
                      apply_clip, KLINE_FEATURES, ORDERBOOK_FEATURES)
from dataset import make_samples, fit_scaler
from model import KlineEncoder, OrderBookEncoder, TimeAttentionPooling
from model_cnn import CNNTransformerEncoder, AttentionPooling

N_FEATURES = len(KLINE_FEATURES) + len(ORDERBOOK_FEATURES)   # 25


# ==========================================================================
# 构建
# ==========================================================================

def _keys_of(meta):
    """meta 的 (inst, day, minute) -> 结构化数组，便于做集合运算。"""
    inst = np.array([m[0] for m in meta], dtype=np.int64)
    day = np.array([m[1].toordinal() for m in meta], dtype=np.int64)
    minute = np.array([m[2] for m in meta], dtype=np.int64)
    return np.stack([inst, day, minute], axis=1)


def build_multifreq_split(cfg: Config, months, tag: str,
                          out_root: str = "features_mf",
                          clip_bounds=None, verbose: bool = True):
    """
    逐月构建多频率对齐数据集并落盘。

    每个月块内：
      1) 每个频率各自构造 (X, y, meta)
      2) 按 (instrument_id, day, minute_of_day) 取四频率交集
      3) 只保留交集样本，写成 .npy 分块

    clip_bounds: 传入训练集算出的边界（验证/测试集复用）；None 表示用本数据集
                 分位数（只有训练集才该这么做）。

    返回 (out_dir, clip_bounds, scaler_dict)
    """
    freqs = cfg.active_freqs()
    os.makedirs(out_root, exist_ok=True)
    out_dir = os.path.join(out_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    # 每个频率的窗口长度（按该频率自己的 bar 长度换算）
    seq_lens = {}
    for f in freqs:
        seq_lens[f] = cfg.for_freq(f).seq_len_bars

    sums = {f: np.zeros(N_FEATURES, dtype=np.float64) for f in freqs}
    sumsq = {f: np.zeros(N_FEATURES, dtype=np.float64) for f in freqs}
    counts = {f: np.zeros(N_FEATURES, dtype=np.float64) for f in freqs}

    chunks = []
    all_labels = []
    first_pass_clip = clip_bounds is None
    raw_labels = [] if first_pass_clip else None

    total = 0
    for mi, month in enumerate(months):
        per = {}
        for f in freqs:
            fc = cfg.for_freq(f)
            df = load_months(cfg.data_dir, fc.table, [month],
                             columns=NEEDED_COLS, verbose=False)
            df = filter_auction_and_tag_day(df)
            df = build_features(df)
            df = build_labels(df, horizon_bars=fc.horizon_bars)
            X, y, meta = make_samples(df, fc.seq_len_bars, fc.aligned_minutes(),
                                      input_same_day_only=fc.input_same_day_only)
            per[f] = {"X": X, "y": y, "meta": meta, "keys": _keys_of(meta)}
            del df

        # ---- 求交集 ----
        key_sets = [set(map(tuple, per[f]["keys"])) for f in freqs]
        common = set.intersection(*key_sets)
        if not common:
            if verbose:
                print(f"  [warn] {month}: 交集为空，跳过")
            continue
        common = set(sorted(common))

        # ---- 逐频率裁到交集 ----
        sel = {}
        for f in freqs:
            k = per[f]["keys"]
            mask = np.fromiter((tuple(row) in common for row in k), dtype=bool,
                               count=len(k))
            sel[f] = np.where(mask)[0]
        n_ref = len(sel[freqs[0]])
        assert all(len(sel[f]) == n_ref for f in freqs), "交集大小不一致"

        # 用第一个频率的 meta 作为权威（四者一致）
        ref_meta = [per[freqs[0]]["meta"][i] for i in sel[freqs[0]]]
        keys_out = per[freqs[0]]["keys"][sel[freqs[0]]]
        y_out = per[freqs[0]]["y"][sel[freqs[0]]]

        if first_pass_clip:
            raw_labels.append(y_out.copy())
        else:
            y_out = apply_clip(y_out, clip_bounds)

        # ---- 写分块 + 累积 scaler 统计 ----
        names = {"y": f"c{mi:04d}_y.npy", "keys": f"c{mi:04d}_keys.npy"}
        np.save(os.path.join(out_dir, names["y"]), y_out.astype(np.float32))
        np.save(os.path.join(out_dir, names["keys"]), keys_out.astype(np.int64))
        for f in freqs:
            arr = np.ascontiguousarray(per[f]["X"][sel[f]])
            fn = f"c{mi:04d}_X_{f}.npy"
            np.save(os.path.join(out_dir, fn), arr)
            names[f"X_{f}"] = fn
            # 分块累积均值/方差（float64，避免 float32 抵消）
            flat = arr.reshape(-1, N_FEATURES).astype(np.float64)
            m = ~np.isnan(flat)
            z = np.where(m, flat, 0.0)
            sums[f] += z.sum(axis=0)
            sumsq[f] += (z * z).sum(axis=0)
            counts[f] += m.sum(axis=0)
            del arr, flat, z, m
        del per

        chunks.append({"month": month, "n": int(n_ref), "files": names})
        total += n_ref
        if verbose:
            print(f"  [{mi+1}/{len(months)}] {month}: n={n_ref:,} "
                  f"(累计 {total:,})")

    if total == 0:
        raise RuntimeError("多频率数据集为空；检查月份与对齐设置")

    if first_pass_clip:
        clip_bounds = compute_clip_bounds(np.concatenate(raw_labels))
        # 用训练集边界重写已落盘的 y
        for ch in chunks:
            p = os.path.join(out_dir, ch["files"]["y"])
            yv = np.load(p)
            np.save(p, apply_clip(yv, clip_bounds).astype(np.float32))

    scaler = {}
    for f in freqs:
        c = np.maximum(counts[f], 1.0)
        mean = sums[f] / c
        var = np.maximum(sumsq[f] / c - mean * mean, 0.0)
        std = np.sqrt(var)
        mean = np.where(np.isnan(mean), 0.0, mean)
        std = np.where((np.isnan(std) | (std < 1e-6)), 1.0, std)
        scaler[f] = (mean.astype(np.float32), std.astype(np.float32))

    meta = {
        "freqs": freqs,
        "seq_lens": {f: int(seq_lens[f]) for f in freqs},
        "n_samples": int(total),
        "months": list(months),
        "clip": [float(clip_bounds[0]), float(clip_bounds[1])],
        "chunks": chunks,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    if verbose:
        print(f"[multifreq] {out_dir}: n={total:,} freqs={freqs} "
              f"T={meta['seq_lens']}")
    return out_dir, clip_bounds, scaler


# ==========================================================================
# 读取
# ==========================================================================

class _ChunkCache:
    """惰性打开 memmap，并限制同时打开的文件数。"""

    def __init__(self, max_open: int = 64):
        self.max_open = max_open
        self._open = {}
        self._order = []

    def get(self, path):
        if path in self._open:
            return self._open[path]
        arr = np.load(path, mmap_mode="r")
        self._open[path] = arr
        self._order.append(path)
        while len(self._order) > self.max_open:
            old = self._order.pop(0)
            self._open.pop(old, None)
        return arr


class _BatchedMultiFreq:
    """内部批量样本，避免 DataLoader 再逐样本 stack 一次。"""

    def __init__(self, xs, y):
        self.xs = xs
        self.y = y


class MultiFreqDataset(Dataset):
    """
    多频率对齐数据集（memmap 惰性读取）。

    每个样本返回 (x_list, y)，其中 x_list 是各频率的张量列表，
    顺序与 meta['freqs'] 一致。
    """

    def __init__(self, root, mean=None, std=None, y=None,
                 n_samples=None, transform_y=False):
        with open(os.path.join(root, "meta.json")) as fh:
            self.meta = json.load(fh)
        self.root = root
        self.freqs = self.meta["freqs"]
        self.seq_lens = self.meta["seq_lens"]
        self.scaler = {}
        for f in self.freqs:
            if mean is not None and std is not None:
                self.scaler[f] = (mean[f], std[f])
            else:
                self.scaler[f] = None

        # 全局行索引 -> (chunk_idx, row)
        self.chunk_of = []
        self.row_of = []
        for ci, ch in enumerate(self.meta["chunks"]):
            self.chunk_of.extend([ci] * ch["n"])
            self.row_of.extend(range(ch["n"]))
        self.chunk_of = np.asarray(self.chunk_of, dtype=np.int64)
        self.row_of = np.asarray(self.row_of, dtype=np.int64)
        n = len(self.chunk_of)

        # 标签（小，全部读入）
        if y is None:
            buf = []
            for ch in self.meta["chunks"]:
                buf.append(np.load(os.path.join(root, ch["files"]["y"])))
            y = np.concatenate(buf).astype(np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        assert len(self.y) == n, f"y 长度 {len(self.y)} != {n}"

        # meta（instrument_id, day, minute）
        kb = []
        for ch in self.meta["chunks"]:
            kb.append(np.load(os.path.join(root, ch["files"]["keys"])))
        self.keys = np.concatenate(kb, axis=0)

        if n_samples is not None:
            self._slice(np.arange(min(n_samples, n)))
        else:
            self.idx_map = np.arange(n)

        self._cache = _ChunkCache()

    def _slice(self, idx):
        """子集视图（用于快速 smoke）。"""
        self.idx_map = np.asarray(idx, dtype=np.int64)
        self.y = self.y[self.idx_map]
        self.keys = self.keys[self.idx_map]

    def subset(self, n_or_idx):
        import copy
        c = copy.copy(self)
        c._cache = _ChunkCache()
        idx = (np.arange(min(n_or_idx, len(self))) if isinstance(n_or_idx, int)
               else np.asarray(n_or_idx))
        c.idx_map = self.idx_map[idx]
        c.y = self.y[idx]
        c.keys = self.keys[idx]
        return c

    def __len__(self):
        return len(self.idx_map)

    def get_meta(self, i):
        g = self.idx_map[i]
        k = self.keys[i]
        day = pd.Timestamp.fromordinal(int(k[1])).date()
        return int(k[0]), day, int(k[2])

    # ---- DataLoader worker 支持：memmap 句柄不可 pickle，需在子进程重建 ----
    def __getstate__(self):
        s = self.__dict__.copy()
        s["_cache"] = None
        return s

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache = _ChunkCache()

    def __getitem__(self, i):
        g = self.idx_map[i]
        ci, r = int(self.chunk_of[g]), int(self.row_of[g])
        ch = self.meta["chunks"][ci]
        xs = []
        for f in self.freqs:
            arr = self._cache.get(os.path.join(self.root, ch["files"][f"X_{f}"]))
            x = np.asarray(arr[r], dtype=np.float32)      # [T, 25]
            sc = self.scaler[f]
            if sc is not None:
                mean, std = sc
                x = (x - mean) / std
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            xs.append(torch.from_numpy(np.ascontiguousarray(x)))
        return xs, torch.tensor(self.y[i])

    def __getitems__(self, indices):
        """
        批量读取 DataLoader 请求的样本。

        训练批由 ChunkGroupedBatchSampler 生成，同一批通常属于同一个
        月块；这里按 chunk 做一次 memmap fancy-index 和一次标准化，避免
        __getitem__ 为每个样本重复进行 Python 调度、数组复制和 NaN 处理。
        """
        positions = np.asarray(indices, dtype=np.int64)
        global_idx = self.idx_map[positions]
        chunks = self.chunk_of[global_idx]
        rows = self.row_of[global_idx]
        unique_chunks = np.unique(chunks)
        xs = []

        for f in self.freqs:
            seq_len = int(self.seq_lens[f])
            batch = np.empty((len(positions), seq_len, N_FEATURES),
                             dtype=np.float32)
            for ci in unique_chunks:
                local = np.flatnonzero(chunks == ci)
                ch = self.meta["chunks"][int(ci)]
                path = os.path.join(self.root, ch["files"][f"X_{f}"])
                arr = self._cache.get(path)
                batch[local] = np.asarray(arr[rows[local]], dtype=np.float32)

            sc = self.scaler[f]
            if sc is not None:
                mean, std = sc
                np.subtract(batch, mean, out=batch)
                np.divide(batch, std, out=batch)
                np.nan_to_num(batch, copy=False, nan=0.0,
                              posinf=0.0, neginf=0.0)
            xs.append(torch.from_numpy(batch))

        y = torch.from_numpy(np.ascontiguousarray(self.y[positions]))
        return _BatchedMultiFreq(xs, y)


def multifreq_collate(batch):
    """把 list-of-dict 转成 (x_list, y)，x_list 中每个是 [B, T_f, F]。"""
    if isinstance(batch, _BatchedMultiFreq):
        return batch.xs, batch.y
    xs, ys = zip(*batch)
    n_freq = len(xs[0])
    stacked = [torch.stack([b[k] for b in xs], dim=0) for k in range(n_freq)]
    return stacked, torch.stack(ys, dim=0)


class ChunkGroupedBatchSampler(torch.utils.data.Sampler):
    """
    按"月块"分组生成 batch，每个 epoch 只打乱 batch 的顺序。

    为什么需要它（两个真实收益）：

    1) **磁盘局部性**：四频率数据量达到几十 GB，必须用 memmap 惰性读。
       若像普通训练那样全局随机 shuffle，每个 batch 都会跳遍所有月份块，
       memmap 随机读会把页面缓存反复冲爆，训练慢到不可用。
       按块分组后，同一时刻只需打开 1 个月块，读取几乎是顺序的。

    2) **更贴合评估口径**：每块 107k 样本 / batch 512 ≈ 211 个 batch，
       而一个截面约 998 只股票。因此每个 batch 内大约半数是**同一截面**
       的股票 —— 这使 `corr_loss`（batch 内相关性）非常接近真正的
       截面 IC，而不是随机混合截面的"伪相关"。

    代价：只在 batch 之间随机，同一 batch 内样本相邻（属于同一截面），
    这对 IC 训练反而是好事。
    """

    def __init__(self, dataset, batch_size, drop_last=True, seed=0):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        # 当前 dataset 视图中，每个位置属于哪个块
        blocks = dataset.chunk_of[dataset.idx_map]
        self._bounds = []
        start = 0
        for i in range(1, len(blocks) + 1):
            if i == len(blocks) or blocks[i] != blocks[start]:
                self._bounds.append((start, i))
                start = i

    def set_epoch(self, e):
        self.epoch = int(e)

    def __iter__(self):
        bs = self.batch_size
        batches = []
        for s, e in self._bounds:
            for b in range(s, e, bs):
                if b + bs <= e or not self.drop_last:
                    batches.append(list(range(b, min(b + bs, e))))
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bs = self.batch_size
        n = 0
        for s, e in self._bounds:
            length = e - s
            n += length // bs if self.drop_last else -(-length // bs)
        return n


def make_mf_loader(dataset, batch_size, shuffle=True, num_workers=0, seed=0,
                   drop_last=True):
    """
    统一的 loader 构造：默认使用按块分组的 batch sampler（磁盘友好）。
    """
    if shuffle:
        sampler = ChunkGroupedBatchSampler(dataset, batch_size,
                                           drop_last=drop_last, seed=seed)
        return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers,
                          collate_fn=multifreq_collate), sampler
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers,
                      collate_fn=multifreq_collate), None


# ==========================================================================
# 模型
# ==========================================================================

class E2EMultiFreqModel(nn.Module):
    """
    GRU 方案：每频率一条独立分支（K线GRU+盘口MLP），拼接→头→标量。
    """

    def __init__(self, cfg, seq_lens=None):
        super().__init__()
        freqs = cfg.active_freqs()
        self.freqs = freqs
        self.kline = nn.ModuleDict()
        self.ob = nn.ModuleDict()
        self.pool_k = nn.ModuleDict()
        self.pool_o = nn.ModuleDict()
        for f in freqs:
            self.kline[f] = KlineEncoder(
                len(KLINE_FEATURES), hidden=cfg.kline_hidden,
                layers=cfg.gru_layers, dropout=cfg.dropout)
            self.ob[f] = OrderBookEncoder(
                len(ORDERBOOK_FEATURES), hidden=cfg.orderbook_hidden,
                dropout=cfg.dropout)
            self.pool_k[f] = TimeAttentionPooling(cfg.kline_hidden)
            self.pool_o[f] = TimeAttentionPooling(cfg.orderbook_hidden)

        d = len(freqs) * 2 * (cfg.kline_hidden + cfg.orderbook_hidden)
        self.fuse = nn.Sequential(
            nn.Linear(d, cfg.fused_dim),
            nn.ReLU(),
            nn.LayerNorm(cfg.fused_dim),
        )
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fused_dim + d, cfg.fused_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fused_dim, 1),
        )

    def forward(self, xs):
        """xs: list of [B, T_f, F]"""
        parts = []
        for k, f in enumerate(self.freqs):
            x = xs[k]
            n_k = len(KLINE_FEATURES)
            hk = self.pool_k[f](self.kline[f](x[..., :n_k]))      # [B, 2dk]
            ho = self.pool_o[f](self.ob[f](x[..., n_k:]))         # [B, 2do]
            parts.append(torch.cat([hk, ho], dim=-1))
        h = torch.cat(parts, dim=-1)                              # [B, d]
        out = torch.cat([self.fuse(h), h], dim=-1)                # [B, d_f + d]
        return self.head(out).squeeze(-1)


class CNNMultiFreqModel(nn.Module):
    """
    CNN+Transformer 方案：每频率一条 CNN+Transformer 分支（统一编码
    K 线 + 盘口），池化后拼接，残差式 head 输出标量。

    与 E2EMultiFreqModel 的前向接口完全兼容：
        xs: list of [B, T_f, 25] -> [B]
    """

    def __init__(self, cfg, seq_lens=None):
        super().__init__()
        freqs = cfg.active_freqs()
        self.freqs = freqs
        hidden = getattr(cfg, "cnn_hidden", 64)
        conv_blocks = getattr(cfg, "cnn_blocks", 3)
        kernel_size = getattr(cfg, "cnn_kernel", 3)
        transformer_layers = getattr(cfg, "transformer_layers", 2)
        n_heads = getattr(cfg, "transformer_heads", 4)
        dropout = getattr(cfg, "dropout", 0.1)

        self.branch = nn.ModuleDict()
        self.pool = nn.ModuleDict()
        for f in freqs:
            self.branch[f] = CNNTransformerEncoder(
                in_dim=N_FEATURES,
                hidden_dim=hidden,
                conv_blocks=conv_blocks,
                kernel_size=kernel_size,
                transformer_layers=transformer_layers,
                n_heads=n_heads,
                dropout=dropout,
            )
            self.pool[f] = AttentionPooling(hidden)

        # 拼接维度 = n_freqs * 2 * hidden（attention ‖ avg）
        d = len(freqs) * 2 * hidden
        head_dim = getattr(cfg, "fused_dim", 64)
        self.fuse = nn.Sequential(
            nn.Linear(d, head_dim),
            nn.ReLU(),
            nn.LayerNorm(head_dim),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_dim + d, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    def forward(self, xs):
        """xs: list of [B, T_f, 25]"""
        parts = []
        for k, f in enumerate(self.freqs):
            x = xs[k]
            h = self.branch[f](x)            # [B, T, hidden]
            parts.append(self.pool[f](h))    # [B, 2*hidden]
        h = torch.cat(parts, dim=-1)         # [B, d]
        out = torch.cat([self.fuse(h), h], dim=-1)
        return self.head(out).squeeze(-1)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==========================================================================
# 辅助
# ==========================================================================

def load_scaler(root):
    """从 meta.json 同级读取 scaler（存在则返回 dict，否则 None）。"""
    p = os.path.join(root, "scaler.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    out = {}
    for k in z.files:
        if k.endswith("__mean"):
            f = k[:-6]
            out[f] = (z[k], z[f + "__std"])
    return out


def save_scaler(root, scaler):
    flat = {}
    for f, (mean, std) in scaler.items():
        flat[f + "__mean"] = mean
        flat[f + "__std"] = std
    np.savez(os.path.join(root, "scaler.npz"), **flat)
