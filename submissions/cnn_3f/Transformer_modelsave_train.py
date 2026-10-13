"""
# Transformer 端到端 —— 训练侧 (单一事实来源)

本文件同时承担两件事, 是训练与推理的**唯一事实来源**:

1. **本地阶段**: 在写死的训练区间上从零训练 CNN+Transformer, 把
   **权重 + 标准化统计 + 结构超参** 存到 `transformer_model.json` (纯文本 JSON,
   张量以 dtype/shape/list 形式序列化, 便于平台原样上传)。
2. **私榜重训阶段**: 平台在隔离环境调用 `train_and_save(datasources)` 从零重训,
   用于验证提交不含违规(未来数据等)。

推理脚本 `Transformer_modelsave_predict.py` 从本文件 import 共享定义, 不重复实现,
保证 train / infer 预处理**严格一致**, 杜绝 train-infer 漂移与数据泄漏。

## 关键设计

- **只依赖 1 分钟 K 线** (`cpt_jyc_2026_e2e_bar1m`)。比赛平台保证提供 1/5/15/30 分钟
  四种数据, 且实测 **bar1m 按 5/15/30 分钟聚合可逐位复现原生 bar5m/15m/30m**
  (100% 数值一致), 因此推理时只在内部做重采样, 提交物自包含、无需外部多频数据。
- **CNN 滑窗 + Transformer**: 1x1 Conv 投影 -> 多个膨胀卷积残差块 -> 正弦位置编码
  -> TransformerEncoder -> 注意力池化‖平均池化 -> MLP head。
- **多频融合**: 每个频率一条独立编码分支, 池化后拼接, 残差式 head 输出标量。
- **标准化统计随权重落盘**: 训练集 fit, 推理直接复用。
- 不做任何手工因子(滚动统计/量价组合/盘口不平衡), 只做单位还原、量纲统一、
  对数变换与缺失值处理, 符合"基础算子"边界。
"""
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================================
# 常量
# ==========================================================================
import config as CFG            # 超参数 + 随机种子 (单一事实来源)

MODEL_PATH = "transformer_model.json"
BATCH = CFG.TRAIN_CFG["batch_size"]
SEED = CFG.SEED

FREQ_MINUTES = {"bar1m": 1, "bar5m": 5, "bar15m": 15, "bar30m": 30}
MORNING_START, MORNING_END = 570, 690      # 09:30, 11:30 (分钟)
AFTERNOON_START, AFTERNOON_END = 780, 900  # 13:00, 15:00
AUCTION_MINUTE = 565                       # 09:25 集合竞价截面 (=9*60+25)
SAMPLE_STEP = CFG.SAMPLE_STEP_MINUTES      # 采样网格(分钟)
HORIZON_MINUTES = CFG.HORIZON_MINUTES      # 预测未来 30 分钟 VWAP

# 比赛要求的日内截面: 09:25 开盘 + 各整/半点 (与本地 config.aligned_minutes 一致;
# 15:00 无未来窗口, 训练/评估时标签为 NaN 会被自动剔除, 推理时仍输出)
ALIGNED_MINUTES = [AUCTION_MINUTE] + [
    m for m in range(MORNING_START + SAMPLE_STEP, MORNING_END + 1, SAMPLE_STEP)
] + [
    m for m in range(AFTERNOON_START + SAMPLE_STEP, AFTERNOON_END + 1, SAMPLE_STEP)
]

# 需要的原始列(e2e 表 schema)
_PRICE_FIELDS = ["open", "high", "low", "close", "amount",
                 "ask_price1", "ask_price2", "ask_price3",
                 "bid_price1", "bid_price2", "bid_price3"]
_OHLC = ["open", "high", "low", "close"]
_BOOK = [f"{s}_{k}{i}" for s in ("ask", "bid")
         for k in ("price", "volume", "num_orders") for i in (1, 2, 3)]

# 25 个特征(7 K线 + 18 盘口)
KLINE_FEATURES = ["open_rel", "high_rel", "low_rel", "close_ret",
                  "log_volume", "log_amount", "log_deal_number"]
ORDERBOOK_FEATURES = (
    [f"{s}_price{i}_rel" for s in ("ask", "bid") for i in (1, 2, 3)]
    + [f"log_{s}_volume{i}" for s in ("ask", "bid") for i in (1, 2, 3)]
    + [f"log_{s}_orders{i}" for s in ("ask", "bid") for i in (1, 2, 3)]
)
ALL_FEATURES = KLINE_FEATURES + ORDERBOOK_FEATURES
N_FEATURES = len(ALL_FEATURES)  # 25


# ==========================================================================
# 模型 (与本地训练代码结构逐位一致)
# ==========================================================================
class PositionalEncoding(nn.Module):
    """正弦位置编码(非学习式, 泛化到任意长度)。"""

    def __init__(self, dim: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ConvResBlock(nn.Module):
    """1D 卷积残差块 (Pre-LN + GELU)。输入/输出 [B, dim, T]。"""

    def __init__(self, dim, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(x)
        x = self.conv1(x)
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + residual


class AttentionPooling(nn.Module):
    """注意力池化 + 平均池化拼接: [B,T,D] -> [B,2D]。"""

    def __init__(self, dim):
        super().__init__()
        h = max(dim // 2, 8)
        self.score = nn.Sequential(nn.Linear(dim, h), nn.Tanh(), nn.Linear(h, 1))
        self.out_dim = dim * 2

    def forward(self, x):
        w = torch.softmax(self.score(x), dim=1)
        att = (x * w).sum(dim=1)
        avg = x.mean(dim=1)
        return torch.cat([att, avg], dim=-1)


class CNNTransformerEncoder(nn.Module):
    """CNN + Transformer 编码器: [B,T,in_dim] -> [B,T,hidden]。"""

    def __init__(self, in_dim=25, hidden_dim=64, conv_blocks=3, kernel_size=3,
                 dilation_growth=1, transformer_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.stem = nn.Conv1d(in_dim, hidden_dim, 1)
        dilations = [dilation_growth ** i for i in range(conv_blocks)]
        self.conv_blocks = nn.ModuleList(
            [ConvResBlock(hidden_dim, kernel_size, d, dropout) for d in dilations])
        self.pos_enc = PositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.stem(x)
        for block in self.conv_blocks:
            x = block(x)
        x = x.transpose(1, 2)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.out_norm(x)


class StockTransformer(nn.Module):
    """单频模型: Encoder -> Pool -> Head -> scalar。forward(x:[B,T,25])->[B]。"""

    def __init__(self, in_dim=N_FEATURES, hidden=64, conv_blocks=3, kernel=3,
                 transformer_layers=2, heads=4, dropout=0.1, fused_dim=64):
        super().__init__()
        self.encoder = CNNTransformerEncoder(
            in_dim, hidden, conv_blocks, kernel, 1, transformer_layers, heads, dropout)
        self.pool = AttentionPooling(hidden)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.pool.out_dim, fused_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(fused_dim, 1))

    def forward(self, x):
        return self.head(self.pool(self.encoder(x))).squeeze(-1)


class StockTransformerMF(nn.Module):
    """多频模型: 每频一条 CNN+Transformer 分支, 拼接后残差 head。
    forward(xs:list[[B,T_f,25]])->[B]。"""

    def __init__(self, freqs, in_dim=N_FEATURES, hidden=64, conv_blocks=3, kernel=3,
                 transformer_layers=2, heads=4, dropout=0.1, fused_dim=64):
        super().__init__()
        self.freqs = list(freqs)
        self.branch = nn.ModuleDict()
        self.pool = nn.ModuleDict()
        for f in self.freqs:
            self.branch[f] = CNNTransformerEncoder(
                in_dim, hidden, conv_blocks, kernel, 1, transformer_layers, heads, dropout)
            self.pool[f] = AttentionPooling(hidden)
        d = len(self.freqs) * 2 * hidden
        self.fuse = nn.Sequential(nn.Linear(d, fused_dim), nn.ReLU(),
                                  nn.LayerNorm(fused_dim))
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(fused_dim + d, fused_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(fused_dim, 1))

    def forward(self, xs):
        parts = [self.pool[f](self.branch[f](xs[k])) for k, f in enumerate(self.freqs)]
        h = torch.cat(parts, dim=-1)
        return self.head(torch.cat([self.fuse(h), h], dim=-1)).squeeze(-1)


def build_model(model_cfg):
    """按 model_cfg 构造模型(arch='single'|'multi')。"""
    kw = {k: v for k, v in model_cfg.items() if k not in ("arch", "freqs")}
    if model_cfg["arch"] == "multi":
        return StockTransformerMF(freqs=model_cfg["freqs"], **kw)
    return StockTransformer(**kw)


# ==========================================================================
# 数据: 读表 -> 还原 -> 重采样 -> 特征 -> 切窗
# ==========================================================================
def _to_dataframe(table):
    """
    把 datasources 里的表对象统一成 pandas DataFrame。

    平台注入口径未知, 这里兼容常见形式: DataFrame / .to_pandas() / .to_df() /
    .df() / 表名字符串(走 dai.query)。列名已在 _normalize 里做归一。
    """
    if isinstance(table, pd.DataFrame):
        return table
    for attr in ("to_pandas", "to_df"):
        if hasattr(table, attr):
            obj = getattr(table, attr)
            return obj() if callable(obj) else obj
    if hasattr(table, "df"):
        obj = table.df
        return obj() if callable(obj) else obj
    if isinstance(table, str):
        import dai
        return dai.query(f"SELECT * FROM {table}").df()
    raise TypeError(f"无法识别的数据表类型: {type(table)}")


def _normalize(df):
    """归一列名: 时间列 -> date, 标的列 -> instrument_id。"""
    df = df.copy()
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("time", "datetime", "date"):
            ren[c] = "date"
        elif lc in ("instrument", "symbol", "stock", "code", "sid"):
            ren[c] = "instrument_id"
    df = df.rename(columns=ren)
    df["date"] = pd.to_datetime(df["date"])
    return df


def restore_units(df):
    """单位还原: OHLC 的 -1 -> NaN 并 /100; amount/盘口价格 /100;
    盘口价格为 0 表示该侧无挂单 -> NaN(0 不是 0 元)。"""
    df = df.copy()
    for f in _OHLC:
        if f in df.columns:
            df[f] = df[f].astype("float64")
            df.loc[df[f] == -1, f] = np.nan
            df[f] = df[f] / 100.0
    for f in _PRICE_FIELDS:
        if f in df.columns and f not in _OHLC:
            df[f] = df[f].astype("float64") / 100.0
    for f in _BOOK:
        if f in df.columns and f.endswith(("1", "2", "3")) and "price" in f:
            df.loc[df[f] == 0.0, f] = np.nan
    return df


def tag_time(df):
    """打时间标签: minute_of_day; 剔除 09:25 之前的盘前指示价 bar(保留 09:25)。"""
    df = df.copy()
    mod = (df["date"].dt.hour * 60 + df["date"].dt.minute).astype("int32")
    keep = mod >= AUCTION_MINUTE
    df = df[keep].copy()
    df["minute_of_day"] = mod[keep].astype("int16")
    df["day"] = df["date"].dt.normalize()
    return df


def resample_bars(df1m, kmins):
    """
    把 1 分钟 bar 聚合成 kmins 分钟 bar。

    规则: 09:25 集合竞价 bar 原样保留; 连续竞价段(09:30 起)按
    `end = 09:30 + ceil((mod-09:30)/K)*K` 分桶, OHLC=首/高/低/末,
    volume/amount/deal_number=求和, 盘口字段=末值(快照)。
    已实测与原生 bar5m/bar15m/bar30m 数值逐位一致。

    关键: 必须按 (instrument_id, day, date) 排序后再取 first/last, 否则
    `open`/`close`/盘口快照 会因输入未按时间排序而取错。
    """
    df = df1m.sort_values(["instrument_id", "date"], kind="mergesort")
    auction = df[df["minute_of_day"] == AUCTION_MINUTE].copy()
    cont = df[df["minute_of_day"] > AUCTION_MINUTE].copy()

    rel = cont["minute_of_day"].to_numpy() - MORNING_START
    cont["_end"] = MORNING_START + np.ceil(rel / kmins).astype("int64") * kmins

    agg = {"open": ("open", "first"), "high": ("high", "max"),
           "low": ("low", "min"), "close": ("close", "last"),
           "volume": ("volume", "sum"), "amount": ("amount", "sum"),
           "deal_number": ("deal_number", "sum")}
    for c in _BOOK:
        agg[c] = (c, "last")

    # 必须把 day 带进分组键, 否则连续竞价段丢失交易日标签(跨日会串)
    g = (cont.groupby(["instrument_id", "day", "_end"], sort=False)
             .agg(**agg).reset_index()
             .rename(columns={"_end": "minute_of_day"}))

    cols = ["instrument_id", "minute_of_day", "day"] + list(agg.keys())
    a = auction[[c for c in cols if c in auction.columns]]
    out = pd.concat([a, g[[c for c in cols if c in g.columns]]], ignore_index=True)
    out["date"] = (pd.to_datetime(out["day"])
                   + pd.to_timedelta(out["minute_of_day"], unit="m"))
    return (out.dropna(subset=["day"])
               .sort_values(["instrument_id", "date"])
               .reset_index(drop=True))


def build_features(df):
    """构造 25 个特征(只做单位/量纲统一与对数变换)。"""
    close = df["close"]
    valid = close > 0
    out = pd.DataFrame(index=df.index)
    out["open_rel"] = np.where(valid, df["open"] / close - 1.0, np.nan)
    out["high_rel"] = np.where(valid, df["high"] / close - 1.0, np.nan)
    out["low_rel"] = np.where(valid, df["low"] / close - 1.0, np.nan)
    prev = df.groupby(["instrument_id", "day"], sort=False)["close"].shift(1)
    out["close_ret"] = np.where(prev > 0, close / prev - 1.0, np.nan)
    out["log_volume"] = np.log1p(df["volume"].clip(lower=0))
    out["log_amount"] = np.log1p(df["amount"].clip(lower=0))
    out["log_deal_number"] = np.log1p(df["deal_number"].clip(lower=0))
    for i in (1, 2, 3):
        for s in ("ask", "bid"):
            out[f"{s}_price{i}_rel"] = np.where(valid, df[f"{s}_price{i}"] / close - 1.0, np.nan)
            out[f"log_{s}_volume{i}"] = np.log1p(df[f"{s}_volume{i}"].clip(lower=0))
            out[f"log_{s}_orders{i}"] = np.log1p(df[f"{s}_num_orders{i}"].clip(lower=0))
    return out[ALL_FEATURES]


def build_labels(df, horizon_bars):
    """标签 = future_vwap / close[t+h] - 1, 窗口不跨交易日。"""
    grp = [df["instrument_id"], df["day"]]
    amt64 = df["amount"].astype("float64")
    vol64 = df["volume"].astype("float64")
    c_amt = amt64.groupby(grp, sort=False).cumsum()
    c_vol = vol64.groupby(grp, sort=False).cumsum()
    fwd_amt = c_amt.groupby(grp, sort=False).shift(-horizon_bars) - c_amt
    fwd_vol = c_vol.groupby(grp, sort=False).shift(-horizon_bars) - c_vol
    fwd_vwap = np.where(fwd_vol > 0, fwd_amt / fwd_vol, np.nan)
    end = df["close"].astype("float64").groupby(grp, sort=False).shift(-horizon_bars)
    close = df["close"].to_numpy(dtype="float64")
    end_a = end.to_numpy(dtype="float64")
    lab = np.where((close > 0) & (end_a > 0), fwd_vwap / np.where(end_a > 0, end_a, 1.0) - 1.0, np.nan)
    return lab.astype("float32")


def make_samples(feat, label, df, seq_len, minutes, with_label=True):
    """向量化滑窗: 返回 (X[N,T,F], y[N], keys[N,3]=inst,day_ord,mod)。

    采样点按**时钟分钟**对齐。输入窗口只要求**同一只股票**(允许回溯到前一日,
    与训练口径 input_same_day_only=False 一致; 标签窗口仍严格当日)。"""
    n = len(df)
    if n <= seq_len:
        return (np.zeros((0, seq_len, N_FEATURES), "float32"),
                np.zeros((0,), "float32"), np.zeros((0, 3), "int64"))
    inst = df["instrument_id"].to_numpy()
    mod = df["minute_of_day"].to_numpy()
    # day 归一为"自 epoch 起的天数"(int64), 便于存取与比较
    day = pd.to_datetime(df["day"]).to_numpy().astype("datetime64[D]").astype("int64")

    i = np.arange(seq_len, n)
    keep = np.isin(mod[i], np.asarray(sorted(set(minutes)), dtype=mod.dtype))
    keep &= (inst[i] == inst[i - seq_len])   # 输入窗口不跨股票(允许跨日回溯)
    if with_label:
        keep &= ~np.isnan(label[i])
    sel = i[keep]
    if sel.size == 0:
        return (np.zeros((0, seq_len, N_FEATURES), "float32"),
                np.zeros((0,), "float32"), np.zeros((0, 3), "int64"))
    win = sel[:, None] + np.arange(-seq_len, 0)[None, :]
    X = feat[win].astype("float32")
    y = label[sel].astype("float32") if with_label else np.zeros(len(sel), "float32")
    keys = np.stack([inst[sel], day[sel].astype("int64"), mod[sel]], axis=1)
    return X, y, keys


class _FreqData:
    """单频: 构造特征 + (可选)标签 + 滑窗。"""

    def __init__(self, df1m, freq, seq_len, minutes, with_label=True):
        self.freq = freq
        self.seq_len = seq_len
        km = FREQ_MINUTES[freq]
        bars = resample_bars(df1m, km)
        feat = build_features(bars).to_numpy(dtype="float32")
        lab = build_labels(bars, max(1, HORIZON_MINUTES // km)) if with_label \
            else np.full(len(bars), np.nan, "float32")
        self.X, self.y, self.keys = make_samples(feat, lab, bars.reset_index(drop=True),
                                                 seq_len, minutes, with_label)


def _subset_freq(fd, sel):
    """按行下标取 _FreqData 的子集(用于把采样时刻限制在测试区间)。"""
    o = _FreqData.__new__(_FreqData)
    o.freq, o.seq_len = fd.freq, fd.seq_len
    o.X = fd.X[sel]
    o.y = fd.y[sel]
    o.keys = fd.keys[sel]
    return o


def _fit_scaler(X, chunk=200_000):
    """训练集特征均值/标准差(分块累加, 忽略 NaN)。"""
    nf = X.shape[-1]
    s = np.zeros(nf, "float64"); sq = np.zeros(nf, "float64"); c = np.zeros(nf, "float64")
    for st in range(0, len(X), chunk):
        blk = X[st:st + chunk].reshape(-1, nf).astype("float64")
        m = ~np.isnan(blk)
        z = np.where(m, blk, 0.0)
        s += z.sum(0); sq += (z * z).sum(0); c += m.sum(0)
    c = np.maximum(c, 1.0)
    mean = s / c
    var = np.maximum(sq / c - mean * mean, 0.0)
    std = np.sqrt(var)
    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.where((np.isnan(std)) | (std < 1e-6), 1.0, std)
    return mean.astype("float32"), std.astype("float32")


def _apply_scaler(X, mean, std):
    """逐样本标准化, NaN -> 0(训练集均值)。"""
    out = X.astype("float32", copy=False)
    mean = np.asarray(mean, "float32").reshape(1, 1, -1)
    std = np.asarray(std, "float32").reshape(1, 1, -1)
    chunk = max(1, 2_000_000 // max(out.shape[1], 1))
    for st in range(0, len(out), chunk):
        blk = out[st:st + chunk]
        np.subtract(blk, mean, out=blk)
        np.divide(blk, std, out=blk)
        np.nan_to_num(blk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _rank_targets(y, keys):
    """截面 rank 目标: 同一 (day, mod) 内做百分位排名并映射到 [-1,1]。"""
    out = np.zeros_like(y, "float32")
    df = pd.DataFrame({"y": y, "day": keys[:, 1], "mod": keys[:, 2]})
    for _, g in df.groupby(["day", "mod"], sort=False):
        if len(g) < 2:
            continue
        r = g["y"].rank(method="average")
        out[g.index.to_numpy()] = ((r / len(g)) * 2.0 - 1.0).to_numpy("float32")
    return out


def clip_bounds(labels, lo=0.01, hi=0.99):
    x = labels[np.isfinite(labels)]
    if x.size == 0:
        return -np.inf, np.inf
    a, b = np.quantile(x, [lo, hi])
    return float(a), float(b)


# ==========================================================================
# 推理侧统一入口: build_dataset (训练脚本也用它, 保证口径一致)
# ==========================================================================
def build_dataset(table, start_date, end_date, mode, minutes, stats,
                  freqs=None, seq_lens=None):
    """
    从 1 分钟表构造模型输入。

    返回 (X_list, y, idx_df, keys):
      X_list : list[np.float32[N, T_f, 25]]  每个频率一项(单频即长度 1)
      y      : np.float32[N]  (infer 模式全 0)
      idx_df : DataFrame[date, instrument, minute_of_day]  预测截面键
      keys   : [N,3] (instrument_id, day_ordinal, minute_of_day)

    stats  : dict{freq: (mean, std)}; 推理直接复用训练统计量。
    """
    freqs = freqs or ["bar30m"]
    seq_lens = seq_lens or {}
    if minutes is None:
        minutes = ALIGNED_MINUTES

    df = _normalize(_to_dataframe(table))
    df = restore_units(df)
    df = tag_time(df)

    with_label = (mode != "infer")
    per = {}
    for f in freqs:
        T = int(seq_lens.get(f, max(1, 120 // FREQ_MINUTES[f])))
        per[f] = _FreqData(df, f, T, minutes, with_label=with_label)

    # 采样时刻限制在平台注入的测试区间[start_date, end_date]内; 表本身保留全量,
    # 以便首日的输入窗口能向前回看到历史 bar(平台会提供约 1 年向前数据)。
    d0 = np.datetime64(pd.Timestamp(start_date).date(), "D").astype("int64")
    d1 = np.datetime64(pd.Timestamp(end_date).date(), "D").astype("int64")
    for f in freqs:
        k = per[f].keys
        m = (k[:, 1] >= d0) & (k[:, 1] <= d1)
        per[f] = _subset_freq(per[f], np.where(m)[0])

    # 多频: 按 keys 取交集(严格同一决策时刻)
    if len(freqs) > 1:
        sets = [set(map(tuple, per[f].keys.tolist())) for f in freqs]
        common = set.intersection(*sets)
        common = set(sorted(common))
        idx = {}
        for f in freqs:
            mask = np.fromiter((tuple(r) in common for r in per[f].keys.tolist()),
                               dtype=bool, count=len(per[f].keys))
            idx[f] = np.where(mask)[0]
    else:
        idx = {freqs[0]: np.arange(len(per[freqs[0]].keys))}

    ref = freqs[0]
    keys = per[ref].keys[idx[ref]]
    y = per[ref].y[idx[ref]]

    X_list = []
    for f in freqs:
        X = per[f].X[idx[f]]
        mean, std = stats[f]
        X_list.append(_apply_scaler(X, mean, std))

    idx_df = pd.DataFrame({
        "date": (pd.to_datetime(keys[:, 1], unit="D")
                 + pd.to_timedelta(keys[:, 2], unit="m")),
        "instrument": keys[:, 0].astype("int64"),
        "minute_of_day": keys[:, 2].astype("int64"),
    })
    return X_list, y, idx_df, keys


# ==========================================================================
# 模型存取 (JSON: 张量按 dtype/shape/list 序列化)
# ==========================================================================
def _enc(obj):
    if isinstance(obj, dict):
        return {k: _enc(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_enc(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        a = obj.detach().cpu().numpy()
        return {"__t__": 1, "dtype": str(a.dtype), "shape": list(a.shape),
                "data": a.reshape(-1).tolist()}
    if isinstance(obj, np.ndarray):
        return {"__t__": 1, "dtype": str(obj.dtype), "shape": list(obj.shape),
                "data": obj.reshape(-1).tolist()}
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _dec(obj):
    if isinstance(obj, dict):
        if obj.get("__t__"):
            a = np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
            return a
        return {k: _dec(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dec(v) for v in obj]
    return obj


def save_model(path, payload):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_enc(payload), fh)


def load_model(path, map_location="cpu"):
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    ck = _dec(raw)
    if "state_dict" in ck:
        ck["state_dict"] = {k: torch.as_tensor(v, device=map_location)
                            for k, v in ck["state_dict"].items()}
    return ck


# ==========================================================================
# 训练
# ==========================================================================
def _load_range(table, months_or_start, end=None):
    """按月份列表或日期区间读 1 分钟表, 返回还原+打标后的 DataFrame。"""
    df = _normalize(_to_dataframe(table))
    df = restore_units(df)
    if end is None:
        months = months_or_start
        mask = df["date"].dt.strftime("%Y%m").isin([str(m) for m in months])
        df = df[mask]
    else:
        df = df[(df["date"] >= pd.Timestamp(months_or_start))
                & (df["date"] <= pd.Timestamp(end))]
    return tag_time(df)


def _train_freqs(df1m, freqs, seq_lens, minutes, seed=SEED, epochs=None, lr=None,
                 batch=None, patience=None, weight_decay=None,
                 val_months=None):
    """在给定 1 分钟数据上训练多频模型, 返回 (model, stats, model_cfg)。

    可复现性: 固定 Python / NumPy / PyTorch(含 CUDA) 随机种子, 并关闭 cudnn 的
    benchmark 自动调优, 使 CPU/GPU 结果尽量一致。
    """
    tc = CFG.TRAIN_CFG
    epochs = tc["epochs"] if epochs is None else epochs
    lr = tc["lr"] if lr is None else lr
    batch = tc["batch_size"] if batch is None else batch
    patience = tc["patience"] if patience is None else patience
    weight_decay = tc["weight_decay"] if weight_decay is None else weight_decay
    val_months = tc["val_months"] if val_months is None else val_months

    # ---- 固定随机种子 ----
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 切分 train/val: 用末尾 val_months 个月做验证(时间序列, 不打乱)
    months = sorted(df1m["date"].dt.strftime("%Y%m").unique())
    val_m = months[-val_months:] if len(months) > val_months + 1 else months[-1:]
    tr = df1m[~df1m["date"].dt.strftime("%Y%m").isin(val_m)].reset_index(drop=True)
    va = df1m[df1m["date"].dt.strftime("%Y%m").isin(val_m)].reset_index(drop=True)

    tr_per = {f: _FreqData(tr, f, seq_lens[f], minutes) for f in freqs}
    va_per = {f: _FreqData(va, f, seq_lens[f], minutes) for f in freqs}

    def align(per):
        if len(freqs) == 1:
            f = freqs[0]
            return {f: np.arange(len(per[f].keys))}
        sets = [set(map(tuple, per[f].keys.tolist())) for f in freqs]
        common = set(sorted(set.intersection(*sets)))
        out = {}
        for f in freqs:
            mask = np.fromiter((tuple(r) in common for r in per[f].keys.tolist()),
                               dtype=bool, count=len(per[f].keys))
            out[f] = np.where(mask)[0]
        return out

    tr_idx, va_idx = align(tr_per), align(va_per)

    stats = {}
    for f in freqs:
        stats[f] = _fit_scaler(tr_per[f].X[tr_idx[f]])
    Xtr = [tr_per[f].X[tr_idx[f]] for f in freqs]
    Xva = [va_per[f].X[va_idx[f]] for f in freqs]
    for i, f in enumerate(freqs):
        Xtr[i] = _apply_scaler(Xtr[i], *stats[f])
        Xva[i] = _apply_scaler(Xva[i], *stats[f])

    ytr = _rank_targets(tr_per[freqs[0]].y[tr_idx[freqs[0]]], tr_per[freqs[0]].keys[tr_idx[freqs[0]]])
    yva_raw = va_per[freqs[0]].y[va_idx[freqs[0]]]
    vkeys = va_per[freqs[0]].keys[va_idx[freqs[0]]]

    model_cfg = dict(arch="multi" if len(freqs) > 1 else "single",
                     freqs=list(freqs), **CFG.MODEL_CFG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=tc["lr_factor"], patience=tc["lr_patience"])

    Xtr_t = [torch.from_numpy(np.ascontiguousarray(x)) for x in Xtr]
    ytr_t = torch.from_numpy(ytr)
    n = len(ytr_t)
    best, best_state, bad = -9e9, None, 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0; nb = 0
        for s in range(0, n, batch):
            idxb = perm[s:s + batch]
            xb = [t[idxb].to(device) for t in Xtr_t]
            yb = ytr_t[idxb].to(device)
            opt.zero_grad()
            p = model(xb)
            pc = p - p.mean(); yc = yb - yb.mean()
            corr = (pc * yc).sum() / (torch.sqrt((pc ** 2).sum() * (yc ** 2).sum()) + 1e-8)
            loss = 1.0 - corr
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss); nb += 1

        ic = _eval_ic(model, Xva, yva_raw, vkeys, device)
        sched.step(-ic)
        print(f"  ep{ep+1:02d} loss={tot/max(nb,1):.4f} val_IC={ic:+.4f}")
        if ic > best:
            best, bad = ic, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  best val_IC={best:+.4f}")
    return model, stats, model_cfg


@torch.no_grad()
def _eval_ic(model, X, y, keys, device, batch=4096):
    model.eval()
    Xt = [torch.from_numpy(np.ascontiguousarray(x)) for x in X]
    preds = []
    for s in range(0, len(y), batch):
        xb = [t[s:s + batch].to(device) for t in Xt]
        preds.append(model(xb).cpu().numpy())
    preds = np.concatenate(preds) if preds else np.zeros(0)
    # 截面 IC(仅用有标签的样本)
    m = np.isfinite(y)
    if m.sum() < 10:
        return 0.0
    df = pd.DataFrame({"p": preds[m], "y": y[m], "day": keys[m, 1], "mod": keys[m, 2]})
    ics = []
    for _, g in df.groupby(["day", "mod"], sort=False):
        if len(g) < 10:
            continue
        pr = g["p"].rank(); yr = g["y"].rank()
        ics.append(np.corrcoef(pr, yr)[0, 1])
    return float(np.mean(ics)) if ics else 0.0


def train_and_save(datasources, train_months=None, freqs=None,
                   model_path=MODEL_PATH):
    """
    从零训练并保存 transformer_model.json(私榜重训亦调用本函数)。

    train_months: 训练月份列表; 默认 config.TRAIN_MONTHS (2020-01 .. 2023-12)。
    freqs:        参与建模的频率; 默认 config.FREQS。
    """
    if train_months is None:
        train_months = CFG.TRAIN_MONTHS
    if freqs is None:
        freqs = CFG.FREQS

    table = datasources["bar1m"]
    print(f"[train] 读取 {len(train_months)} 个月 bar1m ... seed={SEED}")
    df1m = _load_range(table, train_months)
    print(f"[train] 行数={len(df1m):,} 交易日={df1m['day'].nunique()}")

    # 各频率输入窗口 bar 数 = LOOKBACK_MINUTES / bar 长度
    seq_lens = {f: max(1, CFG.LOOKBACK_MINUTES // FREQ_MINUTES[f]) for f in freqs}
    model, stats, model_cfg = _train_freqs(
        df1m, list(freqs), seq_lens, ALIGNED_MINUTES, seed=SEED)

    payload = {
        "format": "miniquant-transformer-v1",
        "arch": model_cfg["arch"],
        "freqs": list(freqs),
        "seq_lens": {f: int(seq_lens[f]) for f in freqs},
        "model_cfg": model_cfg,
        "mean": {f: stats[f][0] for f in freqs},
        "std": {f: stats[f][1] for f in freqs},
        "features": ALL_FEATURES,
        "aligned_minutes": ALIGNED_MINUTES,
        "train_months": list(train_months),
        "seed": int(SEED),
        "train_cfg": dict(CFG.TRAIN_CFG),
        "lookback_minutes": int(CFG.LOOKBACK_MINUTES),
        "horizon_minutes": int(CFG.HORIZON_MINUTES),
        "sort_key": 1,           # resample/make_samples 已按时间排序
        "state_dict": {k: v for k, v in model.state_dict().items()},
    }
    save_model(model_path, payload)
    print(f"[train] 已保存 {model_path} ({os.path.getsize(model_path)/1e6:.2f} MB)")
    return model_path


if __name__ == "__main__":
    from bigmodule import M  # noqa: F401  (平台环境才有)
    datasources = {"bar1m": "cpt_jyc_2026_e2e_bar1m"}
    train_and_save(datasources)
