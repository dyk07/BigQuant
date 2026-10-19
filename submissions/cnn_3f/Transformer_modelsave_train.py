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
import gc
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

# 比赛要求的日内截面 (平台契约 8 个: 09:30, 10:00, 10:30, 11:00, 11:30,
# 13:30, 14:00, 14:30)。与权威样例 submissions/baseline 的
# LEGAL_OUTPUT_HHMM={930,1000,...,1430} 对齐。
#   * 09:30 开盘截面: bar1m 无结束于 09:30 的 bar(只有 09:25 竞价 + 09:31 起),
#     故**内部锚点**仍用 09:25 竞价 bar(AUCTION_MINUTE=565) 作窗口末位(数据严格
#     早于 09:30), 仅在**输出时间戳**时把 565 对齐到 OPENING_SECTION=570(09:30)。
#   * 15:00(900) 无未来窗口(标签 NaN), 平台不将其列为截面, 故**不产出**(见
#     make_anchors 里的 AFTERNOON_END 过滤)。
OPENING_SECTION = 570                       # 09:30 开盘截面(输出时刻, 分钟)
ALIGNED_MINUTES = [AUCTION_MINUTE] + [      # 内部锚点分钟(09:25 竞价 bar 为开盘锚)
    m for m in range(MORNING_START + SAMPLE_STEP, MORNING_END + 1, SAMPLE_STEP)
] + [
    m for m in range(AFTERNOON_START + SAMPLE_STEP, AFTERNOON_END, SAMPLE_STEP)
]


def _trim():
    """把 glibc 空闲内存(main arena)归还操作系统, 压低 RSS 峰值。

    多频重采样/对齐会 malloc 大量临时缓冲, glibc 默认把释放的 chunk 留在
    arena 里备用; 逐月累积时这些保留内存会叠加成 RSS 尖峰。显式 trim 可在
    每个月的临时数据 `del` + `gc.collect()` 之后真正把内存交还内核, 让
    私榜重训(默认 48 个月, 平台限制 **30GB**)与公榜/评测(限制 **51200MB**)
    都留出充足余量。非 glibc 平台静默跳过 (import 失败即忽略)。
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _concat_inplace(arrs):
    """把若干同宽数组拼成一个**新**矩阵, 逐块拷贝后立即释放源块。

    相比 `np.concatenate(arrs)`(拼接瞬间源块与结果**同时**驻留, 峰值 ≈ 2×),
    这里先分配结果, 再把每块**就地**拷入并随即把源块置空, 峰值只比结果多一个
    "最小块"。逐月累积 48 个月的特征矩阵时, 这一处能省下接近一整份特征矩阵
    (数 GB) 的瞬时尖峰。仅用于 2D 特征矩阵 (y/keys 极小, 仍走原逻辑)。
    """
    arrs = [a for a in arrs if a is not None and len(a)]
    if not arrs:
        return None
    if len(arrs) == 1:
        return arrs[0]
    out = np.empty((sum(len(a) for a in arrs),) + arrs[0].shape[1:],
                   dtype=arrs[0].dtype)
    pos = 0
    for i, a in enumerate(arrs):
        out[pos:pos + len(a)] = a
        pos += len(a)
        arrs[i] = None
    return out


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
# 内存策略 (平台硬限制: 公榜/评测 51200MB, 私榜重训 30GB, 超限强制终止)
# --------------------------------------------------------------------------
# 1. **只 SELECT 需要的列**, 并尽早降为 float32, 丢掉 adjust_factor 等无关列;
# 2. **按自然月分块读取**, 每块处理完立即释放原始 DataFrame (del + gc + _trim);
# 3. **不物化 [N,T,25] 滑窗矩阵**, 只保留 "特征矩阵 + 采样锚点下标",
#    窗口在喂模型时按 batch 现场 gather (内存 O(batch*T*F) ~ 50MB);
# 4. 标准化是逐列仿射运算, 作用在**特征矩阵**上一次即可,
#    与作用在每个窗口上完全等价, 但内存降为 1/T;
# 5. 每个分块前后各多读 PAD_DAYS 天, 保证跨月/跨日的输入窗口与
#    标签窗口都能回看/前瞻到真实 bar (锚点仍严格限制在本块区间内, 不重复);
# 6. 逐月累积出的特征矩阵最后用 `_concat_inplace` 合并 (峰值只比结果多一小块),
#    取代 `np.concatenate`(拼接瞬间 ≈ 2× 结果) 的瞬时尖峰;
# 7. 每月末调用 `_trim()` 把 glibc 空闲 arena 归还内核, 大幅压低常驻 RSS。
# ==========================================================================
PAD_DAYS = 10          # 分块读取时向前/向后额外缓冲的天数
_SCALE_COLS = (_OHLC + ["amount", "deal_number"]
               + [f"{s}_price{i}" for s in ("ask", "bid") for i in (1, 2, 3)])

# 从表里真正要取的列(裁剪掉 adjust_factor 等)
NEEDED_COLS = ["date", "instrument_id", "volume"] + _SCALE_COLS + _BOOK
NEEDED_COLS = list(dict.fromkeys(NEEDED_COLS))


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


class CrossSectionAttention(nn.Module):
    """截面自注意力: 同截面内一只股票 attend 到其余股票。

    输入 [N, D]（一个大 batch 的融合表示），可选 sec_ids [N]（全局截面编号）。
    当给定 sec_ids 时用 block-diagonal mask 限制注意力只在同截面内, 使一个
    batch 可同时包含多个截面（随机大 batch 下也保持截面语义正确）。

    对 N 维置换等变, 不引入 per-instrument embedding, 避免记住股票身份、
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
                "ffn": nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                     nn.Linear(dim * 4, dim)),
                "norm2": nn.LayerNorm(dim),
            }))

    @staticmethod
    def _section_mask(sec_ids, device):
        """[N, N] bool mask: True 表示允许 attend（同截面）。"""
        return (sec_ids[:, None] == sec_ids[None, :]).to(device)

    def forward(self, z, sec_ids=None):           # [N, D]
        mask = None
        if sec_ids is not None:
            mask = self._section_mask(sec_ids, z.device)
        z = z.unsqueeze(0)                         # [1, N, D]
        for block in self.blocks:
            a, _ = block["attn"](z, z, z, attn_mask=mask, need_weights=False)
            z = block["norm1"](z + a)
            z = block["norm2"](z + block["ffn"](z))
        return z.squeeze(0)                        # [N, D]


class StockTransformer(nn.Module):
    """单频模型: Encoder -> Pool -> (可选截面注意力) -> Head -> scalar。
    forward(x:[B,T,25] 或 list, sec_ids=None)->[B]。"""

    def __init__(self, in_dim=N_FEATURES, hidden=64, conv_blocks=3, kernel=3,
                 transformer_layers=2, heads=4, dropout=0.1, fused_dim=64,
                 use_xsec=False, xsec_layers=1, xsec_heads=4):
        super().__init__()
        self.encoder = CNNTransformerEncoder(
            in_dim, hidden, conv_blocks, kernel, 1, transformer_layers, heads, dropout)
        self.pool = AttentionPooling(hidden)
        self.use_xsec = use_xsec
        if use_xsec:
            self.xsec = CrossSectionAttention(self.pool.out_dim, heads=xsec_heads,
                                              layers=xsec_layers, dropout=dropout)
            head_in = self.pool.out_dim * 2
        else:
            self.xsec = None
            head_in = self.pool.out_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(head_in, fused_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(fused_dim, 1))

    def forward(self, x, sec_ids=None):
        # 兼容主入口统一传入 list(多频分支); 单频取第 0 项
        if isinstance(x, (list, tuple)):
            x = x[0]
        z = self.pool(self.encoder(x))
        if self.xsec is not None:
            z = torch.cat([z, self.xsec(z, sec_ids=sec_ids)], dim=-1)
        return self.head(z).squeeze(-1)


class StockTransformerMF(nn.Module):
    """多频模型: 每频一条 CNN+Transformer 分支, 拼接后残差 head;
    融合表示上可选截面交叉注意力。forward(xs, sec_ids=None)->[B]。"""

    def __init__(self, freqs, in_dim=N_FEATURES, hidden=64, conv_blocks=3, kernel=3,
                 transformer_layers=2, heads=4, dropout=0.1, fused_dim=64,
                 use_xsec=False, xsec_layers=1, xsec_heads=4):
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
        self.use_xsec = use_xsec
        if use_xsec:
            self.xsec = CrossSectionAttention(fused_dim, heads=xsec_heads,
                                              layers=xsec_layers, dropout=dropout)
            # 融合表示 z ‖ 截面上下文 z_ctx ‖ 多频原始 h
            head_in = fused_dim * 2 + d
        else:
            self.xsec = None
            head_in = fused_dim + d
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(head_in, fused_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(fused_dim, 1))

    def forward(self, xs, sec_ids=None):
        parts = [self.pool[f](self.branch[f](xs[k])) for k, f in enumerate(self.freqs)]
        h = torch.cat(parts, dim=-1)            # [B, d]
        z = self.fuse(h)                        # [B, fused_dim]
        if self.xsec is not None:
            z = torch.cat([z, self.xsec(z, sec_ids=sec_ids), h], dim=-1)
        else:
            z = torch.cat([z, h], dim=-1)
        return self.head(z).squeeze(-1)


def build_model(model_cfg):
    """按 model_cfg 构造模型(arch='single'|'multi')。"""
    kw = {k: v for k, v in model_cfg.items() if k not in ("arch", "freqs")}
    # 优先用 model_cfg 里显式记录的开关 (checkpoint 自描述), 否则回落到 CFG。
    kw["use_xsec"] = bool(model_cfg.get("use_xsec", getattr(CFG, "USE_XSEC", False)))
    kw["xsec_layers"] = int(model_cfg.get("xsec_layers", getattr(CFG, "XSEC_LAYERS", 1)))
    kw["xsec_heads"] = int(model_cfg.get("xsec_heads", getattr(CFG, "XSEC_HEADS", 4)))
    if model_cfg["arch"] == "multi":
        return StockTransformerMF(freqs=model_cfg["freqs"], **kw)
    return StockTransformer(**kw)


# ==========================================================================
# 数据: 读表 -> 还原 -> 重采样 -> 特征 -> 切窗
# ==========================================================================
def _pick_cols(df, start=None, end=None):
    """只保留训练/推理真正需要的列, 并按 [start-PAD, end+PAD] 过滤, 同时降 dtype。

    日期过滤对内存**至关重要**: 当 `table` 是已物化的 DataFrame (平台表对象 / 本地
    验证) 时, 不过滤就会让每个月都重复处理整段数据 (48 个月 × 全量)。
    用 `df.loc[mask, keep]` 一次完成 行过滤 + 列裁剪, 只产生**一份**拷贝。
    """
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("time", "datetime", "date"):
            ren[c] = "date"
        elif lc in ("instrument", "symbol", "stock", "code", "sid"):
            ren[c] = "instrument_id"
    if ren:
        df = df.rename(columns=ren)
    keep = [c for c in NEEDED_COLS if c in df.columns]
    if "date" not in keep or "instrument_id" not in keep:
        raise ValueError(f"表缺少必要列 date/instrument_id; 实际列={list(df.columns)}")

    mask = None
    if start is not None and end is not None and "date" in df.columns:
        dt = pd.to_datetime(df["date"])
        lo = pd.Timestamp(start) - pd.Timedelta(days=PAD_DAYS)
        hi = pd.Timestamp(end) + pd.Timedelta(days=PAD_DAYS)
        tz = getattr(dt.dtype, "tz", None)
        if tz is not None:
            lo, hi = lo.tz_localize(tz), hi.tz_localize(tz)
        mask = ((dt >= lo) & (dt <= hi)).to_numpy()
    df = df[keep] if mask is None else df.loc[mask, keep]
    return _shrink(df)


def _fetch_table(table, start, end, columns=None):
    """
    取数据为 DataFrame (**只取需要的列**, 且必须带日期过滤)。

    - DataFrame: 直接裁剪列 + 按 [start-PAD, end+PAD] 过滤;
    - 平台表对象: 经 to_pandas()/to_df()/df() 取出后同样裁剪/过滤;
    - 表名(str/可 f-string 化): 走 dai.query, **只 SELECT 需要的列 + 日期过滤**
      (平台会提供约 1 年前置数据, 不加过滤会把整表读入内存 -> OOM/超时)。

    注意: 调用方负责传**已经含缓冲**的 [start, end] (见 _month_windows)。
    """
    cols = list(columns or NEEDED_COLS)
    if isinstance(table, pd.DataFrame):
        return _pick_cols(table, start, end)
    for attr in ("to_pandas", "to_df"):
        if hasattr(table, attr):
            obj = getattr(table, attr)
            out = obj() if callable(obj) else obj
            return _pick_cols(out, start, end)
    if hasattr(table, "df"):
        obj = table.df
        out = obj() if callable(obj) else obj
        return _pick_cols(out, start, end)
    # 表名: 用 dai 查询, 只 SELECT 需要的列 + 日期过滤
    import dai
    buf = (pd.Timestamp(start) - pd.Timedelta(days=PAD_DAYS)).strftime("%Y-%m-%d")
    end_s = (pd.Timestamp(end) + pd.Timedelta(days=PAD_DAYS)).strftime(
        "%Y-%m-%d %H:%M:%S")
    # 平台表用字符串列 `instrument`(如 '000001.SZ'), 本地内部名是 instrument_id;
    # SELECT 时把内部名映射回平台列名(与 baseline 一致: instrument::string AS
    # instrument), 后续 _pick_cols 会再 rename 回 instrument_id 并做稳定 int 编码。
    sel_cols = ["instrument::string AS instrument" if c == "instrument_id"
                else c for c in cols]
    sel = ", ".join(sel_cols) if isinstance(table, str) else "*"
    sql = f"SELECT {sel} FROM {table}"
    out = dai.query(sql, filters={"date": [buf, end_s]}).df()
    return _pick_cols(out)      # SQL 侧已过滤, 不再重复过滤


# 向后兼容别名(旧调用名)
def _to_dataframe(table, start=None, end=None):
    return _fetch_table(table, start, end)


def _encode_instrument(s):
    """把股票标识统一成**稳定、与子集无关**的 int64 key。

    平台 bar1m 表与 cpt_jyc_2026_instruments 的 instrument 是**字符串代码**
    (如 '000001.SZ'), 本地 e2e 表的 instrument_id 是 **int16 枚举**:
      * 整数 -> 原样 int64 (本地枚举, 与本地伪 dai 的股票池 int 一致);
      * 字符串 -> 确定性 MD5 前缀散列成 60-bit 非负整数 (与子集/顺序无关,
        同一字符串永远得到同一码; 输出时对 cpt_jyc_2026_instruments 用同款
        函数即可还原 join)。
    绝不用 factorize: 它会随停牌/复牌的子集漂移, 破坏跨月/跨表 join。
    返回 int64 ndarray (可直接进 int64 keys 矩阵, 下游无需改动)。
    """
    import hashlib
    s = pd.Series(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.to_numpy(dtype="int64")
    def _code(x):
        return int(hashlib.md5(str(x).encode("utf-8")).hexdigest()[:15], 16)
    return s.map(_code).to_numpy(dtype="int64")


def _shrink(df):
    """把还原后的 DataFrame 压到最小内存占用 (float32 + 编码后 int64 标的)。"""
    if "instrument_id" in df.columns:
        df["instrument_id"] = _encode_instrument(df["instrument_id"])
    for c in df.columns:
        if c in ("date", "day", "minute_of_day", "instrument_id"):
            continue
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")
    return df


def _month_windows(start, end):
    """把 [start, end] 按自然月切成若干块, 返回 [(first_day, last_day, blocks)]。

    只有落在 [start, end] 内的月份才产生数据块 (缓冲由 _fetch_table 负责),
    因此块与块之间不会重复采样, 避免跨月重复计数。
    """
    months = pd.period_range(pd.Timestamp(start).to_period("M"),
                             pd.Timestamp(end).to_period("M"), freq="M")
    return [(m.start_time, m.end_time) for m in months]


def _normalize(df):
    """归一列名 + 裁剪列 + 时间/dtype 规整 (只保留必需列, 降内存)。"""
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("time", "datetime", "date"):
            ren[c] = "date"
        elif lc in ("instrument", "symbol", "stock", "code", "sid"):
            ren[c] = "instrument_id"
    if ren:
        df = df.rename(columns=ren)
    keep = [c for c in NEEDED_COLS if c in df.columns]
    if "date" not in keep or "instrument_id" not in keep:
        raise ValueError(f"表缺少必要列 date/instrument_id; 实际列={list(df.columns)}")
    df = df[keep]
    df["date"] = pd.to_datetime(df["date"])
    df["instrument_id"] = _encode_instrument(df["instrument_id"])
    return df


def restore_units(df):
    """单位还原 + 缺失值语义 (兼容本地 e2e 与平台云端两种口径)。

    * **价格类字段**(OHLC / amount / 盘口价):
      - 本地 e2e 表是 int 的"分"(cents, 平台约定本地数据 float→int 压缩),
        需 /100 还原为"元"; 缺失价记法 OHLC=-1、盘口价=0(该侧无挂单)。
      - 平台云端表是 float 的"元"(无需 /100); OHLC 缺失为 NaN、盘口缺失为 0。
      统一按 dtype 判断: 整数 → 视为"分"并 /100; 浮点 → 已是"元"直接保留。
      (基线 baseline/exp009 亦声明「云端价格为元，无需 ÷100」。)
    * **volume / deal_number / 盘口量 / 笔数** 均不缩放(股数/笔数), 只转 float32。

    就地修改 (不 copy), 并把结果降为 float32 以节省内存。
    """
    for f in _OHLC:
        if f in df.columns:
            if pd.api.types.is_integer_dtype(df[f]):
                df[f] = df[f].where(df[f] != -1).astype("float32") / 100.0
            else:
                df[f] = df[f].astype("float32")
    for f in _PRICE_FIELDS:
        if f in df.columns and f not in _OHLC:
            if pd.api.types.is_integer_dtype(df[f]):
                df[f] = df[f].astype("float32") / 100.0
            else:
                df[f] = df[f].astype("float32")
    for f in _BOOK:
        if f in df.columns and f.endswith(("1", "2", "3")) and "price" in f:
            df[f] = df[f].where(df[f] != 0.0)
    for f in ("volume", "deal_number"):
        if f in df.columns:
            df[f] = df[f].astype("float32")
    for f in [c for c in _BOOK if "price" not in c]:
        if f in df.columns:
            df[f] = df[f].astype("float32")
    # dtype 已在 _pick_cols 里降过 (float32/int64), 这里无需再 _shrink 一次拷贝
    return df


def tag_time(df):
    """打时间标签: minute_of_day; 剔除 09:25 之前的盘前指示价 bar(保留 09:25)。

    就地修改, 只保留交易时段内的行 (09:25 之后) 以进一步压缩内存。
    """
    dt = df["date"]
    is_dt = pd.api.types.is_datetime64_any_dtype(dt)
    mod = ((dt.dt.hour * 60 + dt.dt.minute) if is_dt
           else np.floor(dt.to_numpy("int64") / 60) % 1440).astype("int16")
    df["minute_of_day"] = mod
    df["day"] = ((dt.dt.normalize().dt.tz_localize(None)) if is_dt
                 else pd.to_datetime(df["date"].astype("int64"), unit="m")
                 .dt.normalize())
    # 先补列再过滤, 避免在切片上赋值 (SettingWithCopyWarning / 隐式拷贝)
    return df[df["minute_of_day"] >= AUCTION_MINUTE]


def resample_bars(df1m, kmins):
    """
    把 1 分钟 bar 聚合成 kmins 分钟 bar (**低内存**)。

    规则: 09:25 集合竞价 bar 原样保留; 连续竞价段(09:30 起)按
    `end = 09:30 + ceil((mod-09:30)/K)*K` 分桶, OHLC=首/高/低/末,
    volume/amount/deal_number=求和, 盘口字段=末值(快照)。
    已实测与原生 bar5m/bar15m/bar30m 数值逐位一致。

    内存策略 (旧实现峰值 ~4×输入, 是主要 OOM 来源):
      * 排序: 数据本身已按 (instrument, 时间) 有序 -> 先探测, 有序则**零拷贝**;
      * 不再切出 auction / cont 两份大拷贝, 而是用**分组键数组**直接 groupby:
        - auction 行 (09:25) 的桶键置 -1, 聚合后按桶键 >=0 过滤丢弃;
        - 连续竞价行的桶键用纯 numpy 定点计算 (不产生中间 Series);
      * 不往 df 里插临时列 (避免触发整表 copy), 最终只 concat 出 ~20 万行的结果。

    关键: 必须按 (instrument_id, 时间) 有序后再取 first/last, 否则
    `open`/`close`/盘口快照 会因输入未按时间排序而取错。
    """
    inst = df1m["instrument_id"].to_numpy()
    ti = df1m["minute_of_day"].to_numpy().astype("int64", copy=False)
    if len(inst) > 1:
        # 探测是否已按 (instrument asc, minute asc) 有序
        desc = (inst[1:] < inst[:-1]) | ((inst[1:] == inst[:-1]) & (ti[1:] < ti[:-1]))
        df = df1m if not bool(desc.any()) else df1m.sort_values(
            ["instrument_id", "date"], kind="mergesort")
    else:
        df = df1m
    if df is not df1m:
        inst = df["instrument_id"].to_numpy()
        ti = df["minute_of_day"].to_numpy().astype("int64", copy=False)
    if not pd.api.types.is_datetime64_any_dtype(df["day"]):
        df = df.assign(day=pd.to_datetime(df["day"]))
        inst = df["instrument_id"].to_numpy()
        ti = df["minute_of_day"].to_numpy().astype("int64", copy=False)

    cols = ["instrument_id", "minute_of_day", "day"] + list(_OHLC) + \
        ["volume", "amount", "deal_number"] + _BOOK
    cols = [c for c in cols if c in df.columns]

    is_auc = (ti == AUCTION_MINUTE)
    auction = df.loc[is_auc, cols].copy()          # 每标的每天仅 1 行, 极小

    # 桶键: 连续竞价 -> 09:30 + ceil((mod-09:30)/K)*K; 竞价行 -> -1
    # 注意必须用 ceil (与平台 bar Kmin 口径一致): bar 09:31 归入 09:35 桶,
    # 因此不会产生 "09:30" 这种空桶。
    rel = ti - MORNING_START
    bucket = np.empty(len(df), "int64")
    np.maximum(rel, 0, out=bucket)
    np.add(bucket, kmins - 1, out=bucket)
    np.floor_divide(bucket, kmins, out=bucket)
    np.multiply(bucket, kmins, out=bucket)
    np.add(bucket, MORNING_START, out=bucket)
    bucket[is_auc] = -1

    agg = {"open": ("open", "first"), "high": ("high", "max"),
           "low": ("low", "min"), "close": ("close", "last"),
           "volume": ("volume", "sum"), "amount": ("amount", "sum"),
           "deal_number": ("deal_number", "sum")}
    for c in _BOOK:
        agg[c] = (c, "last")

    # 分组键必须是 (instrument, day, bucket), 否则跨日/跨标的会串
    g = (df.groupby([inst, df["day"].to_numpy(), bucket], sort=False)
           .agg(**agg).reset_index())
    g.columns = ["instrument_id", "day", "minute_of_day"] + list(agg.keys())
    g = g[g["minute_of_day"] >= 0]                 # 丢弃集合竞价的分组

    out = pd.concat([auction, g[[c for c in cols if c in g.columns]]],
                    ignore_index=True)
    out["date"] = (pd.to_datetime(out["day"])
                   + pd.to_timedelta(out["minute_of_day"], unit="m"))
    return (out.dropna(subset=["day"])
               .sort_values(["instrument_id", "date"])
               .reset_index(drop=True))


def build_features(df):
    """构造 25 个特征(只做单位/量纲统一与对数变换), 返回 float32 矩阵。

    直接返回 **ndarray** 而非 DataFrame: 25 列 float32 是窗口 gather 的底层存储,
    省掉一遍 DataFrame 拷贝。
    """
    close = df["close"].to_numpy(dtype="float32", copy=False)
    valid = close > 0
    n = len(df)
    out = np.empty((n, N_FEATURES), dtype="float32")

    def put(name, vals):
        out[:, ALL_FEATURES.index(name)] = vals

    def rel(col):
        v = df[col].to_numpy(dtype="float32", copy=False)
        return np.where(valid, v / np.where(valid, close, 1.0) - 1.0, np.nan)

    put("open_rel", rel("open"))
    put("high_rel", rel("high"))
    put("low_rel", rel("low"))
    # close_ret: 同一 (instrument, day) 内上一根 bar 的 close
    prev = df.groupby(["instrument_id", "day"], sort=False)["close"].shift(1)
    prev = prev.to_numpy(dtype="float32", copy=False)
    put("close_ret", np.where(prev > 0, close / np.where(prev > 0, prev, 1.0) - 1.0,
                              np.nan))
    for name, col in (("log_volume", "volume"), ("log_amount", "amount"),
                      ("log_deal_number", "deal_number")):
        put(name, np.log1p(np.clip(df[col].to_numpy(dtype="float32", copy=False),
                                   0.0, None)))
    for i in (1, 2, 3):
        for s in ("ask", "bid"):
            put(f"{s}_price{i}_rel", rel(f"{s}_price{i}"))
            put(f"log_{s}_volume{i}",
                np.log1p(np.clip(df[f"{s}_volume{i}"].to_numpy("float32", copy=False),
                                 0.0, None)))
            put(f"log_{s}_orders{i}",
                np.log1p(np.clip(
                    df[f"{s}_num_orders{i}"].to_numpy("float32", copy=False),
                    0.0, None)))
    return out


def build_labels(df, horizon_bars):
    """标签 = end_price / future_vwap - 1, 窗口不跨交易日 (float32)。

    与平台 vwap_return 定义完全一致: future_vwap 为未来窗口的 VWAP,
    end_price 为未来窗口最后一根 bar 的收盘价 (close[t+h]), 则
        label = end_price / future_vwap - 1
    即"相对未来 VWAP 的收盘价溢价"。训练与推理方向统一, 无需取负/取反纠正。
    """
    grp = [df["instrument_id"], df["day"]]
    amt64 = df["amount"].astype("float64")
    vol64 = df["volume"].astype("float64")
    c_amt = amt64.groupby(grp, sort=False).cumsum()
    c_vol = vol64.groupby(grp, sort=False).cumsum()
    fwd_amt = c_amt.groupby(grp, sort=False).shift(-horizon_bars) - c_amt
    fwd_vol = c_vol.groupby(grp, sort=False).shift(-horizon_bars) - c_vol
    fwd_vwap = np.where(fwd_vol > 0, fwd_amt / fwd_vol, np.nan)
    end = df["close"].astype("float64").groupby(grp, sort=False).shift(-horizon_bars)
    end_a = end.to_numpy(dtype="float64")
    lab = np.where((fwd_vwap > 0) & (end_a > 0),
                   end_a / np.where(fwd_vwap > 0, fwd_vwap, 1.0) - 1.0, np.nan)
    return lab.astype("float32")


def make_anchors(feat, label, df, seq_len, minutes, with_label=True):
    """**不物化窗口**: 只返回采样锚点下标 + 元信息。

    窗口在喂模型时用 `gather_windows(feat, sel, seq_len)` 现场 gather,
    因此内存从 O(N·T·F) 降到 O(N)。anchor 语义与旧 `make_samples` 完全一致:
    采样点按**时钟分钟**对齐; 输入窗口只要求**同一只股票**(允许回溯到前一日,
    与训练口径 input_same_day_only=False 一致); 标签窗口仍严格当日。
    """
    n = len(df)
    if n <= seq_len:
        return (np.zeros((0,), "int64"), np.zeros((0,), "int64"),
                np.zeros((0,), "int64"), np.zeros((0, 3), "int64"))
    inst = df["instrument_id"].to_numpy()
    mod = df["minute_of_day"].to_numpy()
    day = pd.to_datetime(df["day"]).to_numpy().astype("datetime64[D]").astype("int64")

    i = np.arange(seq_len, n)
    keep = np.isin(mod[i], np.asarray(sorted(set(minutes)), dtype=mod.dtype))
    keep &= (inst[i] == inst[i - seq_len])   # 输入窗口不跨股票(允许跨日回溯)
    if with_label:
        keep &= ~np.isnan(label[i])
    sel = i[keep]
    if sel.size == 0:
        return (np.zeros((0,), "int64"), np.zeros((0,), "int64"),
                np.zeros((0,), "int64"), np.zeros((0, 3), "int64"))
    y = label[sel].astype("float32") if with_label else np.zeros(len(sel), "float32")
    keys = np.stack([inst[sel].astype("int64"), day[sel].astype("int64"),
                     mod[sel].astype("int64")], axis=1)
    return sel.astype("int64"), y, keys


def gather_windows(feat, sel, seq_len, out=None):
    """把锚点 sel 的 [T, F] 输入窗口 gather 成 [N, T, F] (仅在喂模型时调用)。

    等价于旧 `feat[sel[:,None] + arange(-T,0)]`, 但可按 batch 调用 -> 内存 O(batch·T·F)。
    """
    n = len(sel)
    if n == 0:
        return np.zeros((0, seq_len, N_FEATURES), "float32")
    if out is None:
        out = np.empty((n, seq_len, N_FEATURES), dtype="float32")
    offsets = seq_len - 1
    for j in range(seq_len):                      # 按时间步 gather, 避免造 idx 矩阵
        out[:, j, :] = feat[sel - offsets + j]
    return out


def _scale_feat(feat, mean, std):
    """在**特征矩阵**上做标准化 (与逐窗口标准化完全等价, 内存降为 1/T)。"""
    np.subtract(feat, np.asarray(mean, "float32").reshape(1, -1), out=feat)
    np.divide(feat, np.asarray(std, "float32").reshape(1, -1), out=feat)
    np.nan_to_num(feat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


def _make_freq(df, freq, seq_len, minutes, with_label=True, stats=None):
    """单频: 重采样 -> 特征 -> (可选)标签 -> 采样锚点; 就地标准化并释放中间量。

    返回 dict: keys, y(或 None), anchors, feat(已标准化), sort_idx (按 keys 排序)。
    """
    km = FREQ_MINUTES[freq]
    bars = resample_bars(df, km)
    feat = build_features(bars)
    lab = (build_labels(bars, max(1, HORIZON_MINUTES // km)) if with_label
           else np.zeros(len(bars), "float32"))
    sel, y, keys = make_anchors(feat, lab, bars.reset_index(drop=True),
                                seq_len, minutes, with_label)
    del lab
    if stats is not None:
        feat = _scale_feat(feat, stats[0], stats[1])
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0])) if len(keys) else \
        np.zeros(0, "int64")
    return {"freq": freq, "seq_len": seq_len, "keys": keys, "y": y,
            "anchors": sel, "feat": feat, "order": order,
            "n_bars": len(bars)}


def _subset_freq(fd, sel):
    """按行下标取单频子集 (用于把采样时刻限制在测试区间)。"""
    o = dict(fd)
    o["keys"] = fd["keys"][sel]
    o["y"] = None if fd["y"] is None else fd["y"][sel]
    o["anchors"] = fd["anchors"][sel]
    return o


def _align_freqs(per, freqs):
    """多频: 按 keys 取交集(严格同一决策时刻), 返回 {freq: 行下标}。"""
    if len(freqs) == 1:
        return {freqs[0]: np.arange(len(per[freqs[0]]["keys"]))}
    common = set.intersection(*[set(map(tuple, per[f]["keys"].tolist()))
                                for f in freqs])
    common = set(sorted(common))
    out = {}
    for f in freqs:
        k = per[f]["keys"]
        mask = np.fromiter((tuple(r) in common for r in k.tolist()),
                           dtype=bool, count=len(k))
        out[f] = np.where(mask)[0]
    return out


def _month_span(month):
    """把 'YYYYMM' 转成 (月首时间戳, 月末时间戳)。"""
    p = pd.Period(str(month), freq="M")
    return p.start_time, p.end_time


def _empty_month(freqs, seq_lens):
    return ([np.zeros((0, int(seq_lens.get(f, 1)), N_FEATURES), "float32")
             for f in freqs],
            np.zeros(0, "float32"),
            pd.DataFrame({"date": [], "instrument": [], "minute_of_day": []}),
            np.zeros((0, 3), "int64"))


def _build_month(table, first, last, mode, minutes, stats, freqs, seq_lens):
    """构造**单个月**的模型输入 (原始 1 分钟表用后即弃 -> 内存峰值 ~1 个月)。

    返回 (X_list, y, idx_df, keys); 锚点严格限制在 [first, last] 内,
    因此按月调用不会跨月重复采样。stats 为 dict{freq: (mean, std)}。
    """
    with_label = (mode != "infer")
    raw = _normalize(_fetch_table(table, first, last))
    if len(raw) == 0:
        return _empty_month(freqs, seq_lens)
    raw = restore_units(raw)
    raw = tag_time(raw)

    per = {}
    for f in freqs:
        T = int(seq_lens.get(f, max(1, CFG.LOOKBACK_MINUTES // FREQ_MINUTES[f])))
        per[f] = _make_freq(raw, f, T, minutes, with_label=with_label,
                            stats=(stats or {}).get(f))
    # 把锚点限制在这一个月内 (避免缓冲期重复采样)
    d0 = np.datetime64(pd.Timestamp(first).date(), "D").astype("int64")
    d1 = np.datetime64(pd.Timestamp(last).date(), "D").astype("int64")
    for f in freqs:
        k = per[f]["keys"]
        if len(k):
            m = (k[:, 1] >= d0) & (k[:, 1] <= d1)
            per[f] = _subset_freq(per[f], np.where(m)[0])

    # 多频按 keys 取交集(严格同一决策时刻)
    idx = _align_freqs(per, freqs)
    ref = per[freqs[0]]
    j0 = idx[freqs[0]]
    keys = ref["keys"][j0]
    y = (ref["y"][j0] if ref["y"] is not None
         else np.zeros(len(keys), "float32")).astype("float32")

    X_list = []
    for f in freqs:
        T = int(seq_lens.get(f, max(1, CFG.LOOKBACK_MINUTES // FREQ_MINUTES[f])))
        X_list.append(gather_windows(per[f]["feat"], per[f]["anchors"][idx[f]], T))
    if len(keys):
        # 输出时间戳对齐平台 8 截面契约 (baseline LEGAL_OUTPUT_HHMM):
        #   09:25 竞价锚点 -> 输出 09:30 (开盘截面; bar1m 无 09:30 结束的 bar)
        #   其余整/半点原样; 15:00 已被 make_anchors 剔除(无未来窗口, 平台不列)
        out_mod = keys[:, 2].copy()
        out_mod[out_mod == AUCTION_MINUTE] = OPENING_SECTION
        idx_df = pd.DataFrame({
            "date": (pd.to_datetime(keys[:, 1], unit="D")
                     + pd.to_timedelta(out_mod, unit="m")),
            "instrument": keys[:, 0].astype("int64"),
            "minute_of_day": out_mod.astype("int64"),
        })
    else:
        idx_df = pd.DataFrame({"date": [], "instrument": [], "minute_of_day": []})
    del raw, per
    gc.collect()
    _trim()
    return X_list, y, idx_df, keys


def _norm_freqs(freqs, seq_lens, minutes):
    freqs = list(freqs or ["bar30m"])
    seq_lens = dict(seq_lens or {})
    return freqs, seq_lens, (ALIGNED_MINUTES if minutes is None else minutes)


def build_dataset(table, start_date, end_date, mode, minutes, stats,
                  freqs=None, seq_lens=None):
    """
    从 1 分钟表构造模型输入 (**分月流式, 内存受限**)。

    返回 (X_list, y, idx_df, keys):
      X_list : list[np.float32[N, T_f, 25]]  每个频率一项(单频即长度 1)
      y      : np.float32[N]  (infer 模式全 0)
      idx_df : DataFrame[date, instrument, minute_of_day]  预测截面键
      keys   : [N,3] (instrument_id, day_ordinal, minute_of_day)

    内存策略: 逐月调用 `_build_month`, 每个月只读 [月初-PAD, 月末+PAD] 的 bar1m,
    处理完立即释放; 采样锚点严格限制在该月内, 因此各块之间不重复。
    依赖长区间的推理请改用 `_iter_dataset` (逐月 yield, 峰值只有 1 个月)。
    stats  : dict{freq: (mean, std)}; 推理直接复用训练统计量。
    """
    freqs, seq_lens, minutes = _norm_freqs(freqs, seq_lens, minutes)
    Xs_acc = [[] for _ in freqs]
    ys, idxs, keyss = [], [], []
    for first, last in _month_windows(start_date, end_date):
        Xs, y, idx_df, keys = _build_month(
            table, first, last, mode, minutes, stats, freqs, seq_lens)
        if len(idx_df) == 0:
            continue
        for i in range(len(freqs)):
            Xs_acc[i].append(Xs[i])
        ys.append(y)
        idxs.append(idx_df)
        keyss.append(keys)
    if not idxs:
        return _empty_month(freqs, seq_lens)
    X_list = [np.concatenate(a) if a else
              np.zeros((0, int(seq_lens.get(f, 1)), N_FEATURES), "float32")
              for a, f in zip(Xs_acc, freqs)]
    idx_df = pd.concat(idxs, ignore_index=True)
    return X_list, np.concatenate(ys), idx_df, np.concatenate(keyss)


def _iter_dataset(table, start_date, end_date, mode, minutes, stats,
                  freqs=None, seq_lens=None):
    """**逐月 yield** 单月模型输入 (推理用)。

    每次 yield `(X_list, y, idx_df, keys)`, 调用方消费完即可释放 -> 峰值内存
    只有 1 个月的窗口 (~数百 MB), 长测试区间也不会触发平台 51200MB 限制。
    """
    freqs, seq_lens, minutes = _norm_freqs(freqs, seq_lens, minutes)
    for first, last in _month_windows(start_date, end_date):
        Xs, y, idx_df, keys = _build_month(
            table, first, last, mode, minutes, stats, freqs, seq_lens)
        if len(idx_df) == 0:
            continue
        yield Xs, y, idx_df, keys
        # 调用方已消费完本月张量 -> 显式释放并归还 glibc 空闲 arena,
        # 使长评测区间的常驻 RSS 稳定在"单月"量级 (与 _accumulate 同策略)。
        del Xs, y, idx_df, keys
        gc.collect()
        _trim()


def _fit_scaler(feat, chunk=1_000_000):
    """在**特征矩阵** [n_bars, F] 上算均值/标准差 (分块累加, 忽略 NaN)。

    注: 训练统计量沿用"在 bar 级特征矩阵上 fit"的口径 (与推理侧复用同一公式),
    标准化是逐列仿射, 作用在 feat 上与作用在窗口上结果一致, 但内存降为 1/T。
    """
    nf = feat.shape[-1]
    s = np.zeros(nf, "float64"); sq = np.zeros(nf, "float64"); c = np.zeros(nf, "float64")
    for st in range(0, len(feat), chunk):
        blk = feat[st:st + chunk].astype("float64")
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


def _corr_batch(p, y):
    """batch 内 Pearson 相关 (中心化后余弦), 返回标量 corr ∈ [-1, 1]。

    注意: eps 放在 sqrt **内部**, 否则当 p/y 均为常数 (方差=0) 时,
    sqrt(0) 的反向会得到 inf*0 = NaN 梯度 (如换手正则的单元素配对)。

    另: 前向在 autocast(fp16/bf16) 下 p/y 可能是低精度张量, 这里先 .float()
    升回 fp32, 保证 corr 与梯度的数值精度不受混合精度影响。
    """
    p = p.float(); y = y.float()
    pc = p - p.mean(); yc = y - y.mean()
    denom = torch.sqrt((pc ** 2).sum() * (yc ** 2).sum() + 1e-8)
    return (pc * yc).sum() / denom


def _grouped_corr(p, y, sidb):
    """按截面分组求 corr 再平均 (向量化, 替代逐 group 串行循环)。

    用 scatter_add 一次归约所有 group 的 sum/count/sumsq, 与逐 group 的
    `_corr_batch` 数值及梯度一致 (对拍误差 < 1e-8)。有效 group (<2 样本)
    不计入, 全部无效时退化为整 batch corr。返回标量均值。
    """
    p = p.float(); y = y.float()
    ids = torch.unique(sidb)
    gid = torch.searchsorted(ids, sidb)                 # 连续 id 0..G-1
    G = ids.numel()
    cnt = torch.bincount(gid, minlength=G).float()      # [G]
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
    corrs = sxy / torch.sqrt(ssp * ssy + 1e-8)          # [G]
    valid = cnt >= 2
    if valid.sum() == 0:
        return _corr_batch(p, y)
    return corrs[valid].mean()


def _section_ids(keys):
    """把 (day, minute) 映射为截面编号 [N] int (0..K-1)。

    用于截面交叉注意力的 block-diagonal mask: 编号只参与相等比较,
    因此用 day*1440 + mod 的整数哈希即可, 无需额外的连续映射开销。
    返回 int64 数组, 随机 batch 下按全局下标切片即可还原截面身份。
    """
    if len(keys) == 0:
        return np.zeros(0, "int64")
    sec = keys[:, 1].astype("int64") * 1440 + keys[:, 2].astype("int64")
    _, ids = np.unique(sec, return_inverse=True)
    return ids.astype("int64")


def _turnover_next(keys):
    """返回 next_idx [N]: 样本 i 的下一个同 (instrument, day) 截面下标, 无则 -1。

    换手正则用: 相邻截面是"同一只股票在同一交易日的下一采样时刻"。用
    (instrument, day, mod) 的字典序定位同 (instrument, day) 块内 mod 升序的
    后继行。与输入 keys 是否已排序无关 (内部 lexsort 一次, 代价 O(N log N),
    整个训练周期仅算一次)。
    """
    n = len(keys)
    next_idx = np.full(n, -1, "int64")
    if n < 2:
        return next_idx
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    k = keys[order]
    same = (k[1:, 0] == k[:-1, 0]) & (k[1:, 1] == k[:-1, 1])
    valid = same & (k[1:, 2] != k[:-1, 2])      # 同截面自身不配对
    nxt_sorted = np.full(n, -1, "int64")
    nxt_sorted[np.arange(0, n - 1)[valid]] = np.arange(1, n)[valid]
    inv = np.empty(n, "int64")
    inv[order] = np.arange(n)
    m = nxt_sorted >= 0
    next_idx[order[m]] = order[nxt_sorted[m]]
    return next_idx


def clip_bounds(labels, lo=0.01, hi=0.99):
    x = labels[np.isfinite(labels)]
    if x.size == 0:
        return -np.inf, np.inf
    a, b = np.quantile(x, [lo, hi])
    return float(a), float(b)


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
def _accumulate(table, months, freqs, seq_lens, minutes, stats=None):
    """**按月流式**累积训练张量: 只保留 "频率特征矩阵 + 锚点下标 + keys"。

    内存关键点(平台硬限制 51200MB):
      * 原始 bar1m 只在一个月的范围内驻留, 每月处理完立即 `del` + `gc.collect()`;
      * **不物化 [N, T, F] 滑窗矩阵**, 只存 `feat` (N_bars×25) 与锚点 (N,);
      * 多频严格按 (instrument, day, minute) 取交集, 与推理口径一致;
      * 各月锚点都限制在本月内, 因此块与块之间不会重复采样。

    返回 dict: feats{freq: [n_bars, 25] float32}, anchors{freq: [N] int64},
               y[N] float32, keys[N, 3]
    """
    acc = {"feats": {f: [] for f in freqs}, "anchors": {f: [] for f in freqs},
           "keys": [], "y": []}
    off = {f: 0 for f in freqs}
    for m in months:
        first, last = _month_span(m)
        raw = _normalize(_fetch_table(table, first, last))
        if len(raw) == 0:
            continue
        raw = restore_units(raw)
        raw = tag_time(raw)
        per = {}
        for f in freqs:
            per[f] = _make_freq(raw, f, int(seq_lens[f]), minutes,
                                with_label=True, stats=(stats or {}).get(f))
        # 锚点限制在本月内, 避免缓冲期重复采样
        d0 = np.datetime64(pd.Timestamp(first).date(), "D").astype("int64")
        d1 = np.datetime64(pd.Timestamp(last).date(), "D").astype("int64")
        for f in freqs:
            k = per[f]["keys"]
            if len(k):
                mask = (k[:, 1] >= d0) & (k[:, 1] <= d1)
                per[f] = _subset_freq(per[f], np.where(mask)[0])
        # 多频取交集 -> 同一决策时刻
        idx = _align_freqs(per, freqs)
        j0 = idx[freqs[0]]
        acc["keys"].append(per[freqs[0]]["keys"][j0])
        acc["y"].append(per[freqs[0]]["y"][j0])
        for f in freqs:
            acc["feats"][f].append(per[f]["feat"])
            # 锚点转为"拼接后特征矩阵"的全局行号
            acc["anchors"][f].append(per[f]["anchors"][idx[f]] + off[f])
            off[f] += len(per[f]["feat"])
        del raw, per
        gc.collect()
        _trim()

    out = {}
    if not acc["keys"]:
        out["keys"] = np.zeros((0, 3), "int64")
        out["y"] = np.zeros(0, "float32")
    else:
        out["keys"] = np.concatenate(acc["keys"])
        out["y"] = np.concatenate(acc["y"]).astype("float32")
    out["feats"] = {}
    for f in freqs:
        c = _concat_inplace(acc["feats"][f])
        out["feats"][f] = (c if c is not None
                           else np.zeros((0, N_FEATURES), "float32"))
    out["anchors"] = {f: (np.concatenate(acc["anchors"][f]) if acc["anchors"][f]
                          else np.zeros(0, "int64")) for f in freqs}
    del acc
    gc.collect()
    _trim()
    return out


def _batch_windows(feats, anchors, row_idx, seq_lens, freqs, buf=None):
    """按行下标现场 gather 一个 batch 的窗口 -> list[np.float32[b, T_f, 25]]。

    复用 `buf`(预分配) 以避免每个 step 反复 malloc, 峰值内存 O(batch·T·F) ~ 50MB。
    """
    out = []
    for f in freqs:
        sel = anchors[f][row_idx]
        if buf is not None and len(row_idx) == len(buf[f]):
            b = buf[f]
            gather_windows(feats[f], sel, int(seq_lens[f]), out=b)
        else:
            b = gather_windows(feats[f], sel, int(seq_lens[f]))
        out.append(b)
    return out


def _train_freqs(table, train_months, freqs, seq_lens, minutes, seed=SEED,
                 epochs=None, lr=None, batch=None, patience=None,
                 weight_decay=None, val_months=None):
    """在给定 1 分钟表上训练多频模型, 返回 (model, stats, model_cfg)。

    数据侧**分月流式**构造 (见 `_accumulate`): 训练集与验证集各扫一遍自己的月份,
    全程不物化 [N, T, F] 滑窗矩阵, 窗口在训练 step 里按 batch 现场 gather。

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

    # 切分 train/val: 末尾 val_months 个月做验证 (时间序列, 不打乱)
    months = sorted(str(m) for m in train_months)
    val_m = months[-val_months:] if len(months) > val_months + 1 else months[-1:]
    tr_m = [m for m in months if m not in set(val_m)]
    print(f"  [data] train={len(tr_m)}月 val={len(val_m)}月 ({val_m[0]}~{val_m[-1]})")

    # ---- 训练集: 直接 fit 统计量 (不物化窗口), 再就地标准化 ----
    tr = _accumulate(table, tr_m, freqs, seq_lens, minutes, stats=None)
    if len(tr["keys"]) == 0:
        raise RuntimeError(f"训练集无样本 (months={tr_m})")
    stats = {}
    for f in freqs:
        mean, std = _fit_scaler(tr["feats"][f])
        _scale_feat(tr["feats"][f], mean, std)
        stats[f] = (mean.astype("float32"), std.astype("float32"))

    # ---- 验证集: 复用训练统计量 (同口径 -> 无泄漏) ----
    va = _accumulate(table, val_m, freqs, seq_lens, minutes, stats=stats)
    if len(va["keys"]) == 0:
        raise RuntimeError(f"验证集无样本 (months={val_m})")
    print(f"  [data] train_N={len(tr['keys']):,} val_N={len(va['keys']):,}")

    ytr = _rank_targets(tr["y"], tr["keys"])
    yva_raw = va["y"]
    vkeys = va["keys"]

    model_cfg = dict(arch="multi" if len(freqs) > 1 else "single",
                     freqs=list(freqs), **CFG.MODEL_CFG,
                     use_xsec=bool(getattr(CFG, "USE_XSEC", False)),
                     xsec_layers=int(getattr(CFG, "XSEC_LAYERS", 1)),
                     xsec_heads=int(getattr(CFG, "XSEC_HEADS", 4)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=tc["lr_factor"], patience=tc["lr_patience"])

    # ---- 混合精度 (AMP) ----
    # float16 需要 GradScaler 防下溢; bfloat16 动态范围与 fp32 一致, 无需 scaler。
    # CPU / 无 CUDA 时自动关闭 (dtype 回退 fp32)。
    use_amp = bool(getattr(CFG, "USE_AMP", False)) and device.type == "cuda"
    amp_dtype = {"float16": torch.float16,
                 "bfloat16": torch.bfloat16}.get(
        str(getattr(CFG, "AMP_DTYPE", "float16")), torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    if use_amp:
        print(f"  [amp] enabled dtype={amp_dtype}")

    n = len(ytr)
    ytr_t = torch.from_numpy(ytr)
    buf = {f: np.empty((batch, int(seq_lens[f]), N_FEATURES), "float32")
           for f in freqs}
    grad_clip = float(tc.get("grad_clip", 1.0))

    use_xsec = bool(getattr(CFG, "USE_XSEC", False))
    use_turnover = bool(getattr(CFG, "USE_TURNOVER", False))
    lam_to = float(getattr(CFG, "LAMBDA_TURNOVER", 0.01))

    # 截面编号 (训练周期仅算一次)
    sec_ids_tr = _section_ids(tr["keys"]) if use_xsec else None

    # 换手"下一截面"配对 (全局下标, 仅算一次): cur -> 同 (instrument, day) 的
    # 下一采样截面。随机 batch 会打散, 只有 cur 与 next 落在同一 batch 时才能
    # 直接算平滑相关; 该配对在 epoch 开始时按 perm 位置预先归类, 避免逐 batch
    # 做 O(n) 成员判断。
    cur_tr = nxt_tr = None
    if use_turnover:
        nxt_all = _turnover_next(tr["keys"])
        valid = nxt_all >= 0
        cur_tr = np.where(valid)[0].astype("int64")
        nxt_tr = nxt_all[valid].astype("int64")
        del nxt_all, valid

    best, best_state, bad = -9e9, None, 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        perm_np = perm.numpy()
        tot = 0.0; nb = 0
        tot_to = 0.0; cnt_to = 0

        # 换手: 按本 epoch 的随机排列, 找出"cur 与 next 落在同一 batch"的配对,
        # 记录其 batch 起始下标 s 与该 batch 内的局部下标 (lcur, lnxt)。
        pair_by_s = None
        if use_turnover and cur_tr is not None:
            pos = np.empty(n, "int64")
            pos[perm_np] = np.arange(n)
            bcur = pos[cur_tr] // batch
            bnxt = pos[nxt_tr] // batch
            same = bcur == bnxt
            if same.any():
                scur = pos[cur_tr[same]]
                snxt = pos[nxt_tr[same]]
                bs = (scur // batch) * batch
                pair_by_s = {}
                for s0, lc, ln in zip(bs, scur % batch, snxt % batch):
                    pair_by_s.setdefault(int(s0), []).append((int(lc), int(ln)))
            del pos

        for s in range(0, n, batch):
            idxb = perm_np[s:s + batch]
            xs = _batch_windows(tr["feats"], tr["anchors"], idxb, seq_lens,
                                freqs, buf=buf)
            xb = [torch.from_numpy(x).to(device) for x in xs]
            yb = ytr_t[idxb].to(device)
            sidb = None if sec_ids_tr is None else \
                torch.from_numpy(sec_ids_tr[idxb]).to(device)

            opt.zero_grad()
            # AMP: 前向 + 损失在 autocast 上下文中; 损失内部 _corr_batch 已 .float(),
            # 故 loss 数值与梯度精度不受低精度前向影响。
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=use_amp):
                p = model(xb, sec_ids=sidb)

                # 主损失: 与评估截面 IC 对齐。启用截面注意力时改为"组内 corr 再平均",
                # 避免随机大 batch 把多个截面混在一起做 Pearson 造成的偏差。
                if use_xsec:
                    loss = 1.0 - _grouped_corr(p, yb, sidb)
                else:
                    loss = 1.0 - _corr_batch(p, yb)

                # 换手正则: 1 - corr(p_cur, p_next), 同一股票相邻截面分数相关越高,
                # 相对排序越稳定 -> 换手越低。
                if pair_by_s and s in pair_by_s:
                    lcs = torch.tensor([a for a, _ in pair_by_s[s]], device=device)
                    lns = torch.tensor([b for _, b in pair_by_s[s]], device=device)
                    # 至少 2 对才构成有意义的时序相关 (单对退化为零方差 -> 无梯度)
                    if lcs.numel() >= 2:
                        to_corr = _corr_batch(p[lcs], p[lns])
                        loss = loss + lam_to * (1.0 - to_corr)
                        tot_to += float(1.0 - to_corr.detach()); cnt_to += 1

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            tot += float(loss); nb += 1

        ic = _eval_ic(model, va["feats"], va["anchors"], seq_lens,
                      yva_raw, vkeys, freqs, device)
        sched.step(-ic)
        msg = f"  ep{ep+1:02d} loss={tot/max(nb,1):.4f} val_IC={ic:+.4f}"
        if use_turnover:
            msg += f" turnover={tot_to/max(cnt_to,1):.4f}"
        print(msg)
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
    del tr, va
    gc.collect()
    return model, stats, model_cfg


@torch.no_grad()
def _predict_all(model, feats, anchors, seq_lens, freqs, keys, device, batch=4096):
    """全量前向: 窗口按 batch 现场 gather, 返回 np.float32[N] 预测分数。

    截面注意力需要 sec_ids 才能保持"同截面内交互"语义, 因此这里按 keys
    推导全局截面编号并逐 batch 传入; 未启用 xsec 时 sec_ids=None (等价原逻辑)。
    """
    model.eval()
    n = len(keys)
    preds = np.empty(n, "float32")
    use_xsec = bool(getattr(model, "use_xsec", False))
    sec_ids = _section_ids(keys) if use_xsec else None
    for s in range(0, n, batch):
        idxb = np.arange(s, min(s + batch, n))
        xs = _batch_windows(feats, anchors, idxb, seq_lens, freqs)
        xb = [torch.from_numpy(x).to(device) for x in xs]
        sidb = None if sec_ids is None else \
            torch.from_numpy(sec_ids[idxb]).to(device)
        preds[s:s + len(idxb)] = model(xb, sec_ids=sidb).cpu().numpy().reshape(-1)
    return preds


@torch.no_grad()
def _eval_ic(model, feats, anchors, seq_lens, y, keys, freqs, device, batch=4096):
    """验证集截面 IC: 窗口按 batch 现场 gather, 不预物化 [N, T, F]。"""
    preds = _predict_all(model, feats, anchors, seq_lens, freqs, keys,
                         device, batch=batch)
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

    内存: 数据全程按月流式, 不物化滑窗矩阵; 峰值由 "一个月的 bar1m + 各频率
    特征矩阵" 决定, 远低于平台 51200MB 硬限制。
    """
    if train_months is None:
        train_months = CFG.TRAIN_MONTHS
    if freqs is None:
        freqs = CFG.FREQS
    train_months = sorted(str(m) for m in train_months)

    table = datasources["bar1m"]
    print(f"[train] 分月流式读取 {len(train_months)} 个月 bar1m ... seed={SEED}")

    # 各频率输入窗口 bar 数 = LOOKBACK_MINUTES / bar 长度
    seq_lens = {f: max(1, CFG.LOOKBACK_MINUTES // FREQ_MINUTES[f]) for f in freqs}
    model, stats, model_cfg = _train_freqs(
        table, train_months, list(freqs), seq_lens, ALIGNED_MINUTES, seed=SEED)

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
        "use_xsec": bool(getattr(CFG, "USE_XSEC", False)),
        "use_turnover": bool(getattr(CFG, "USE_TURNOVER", False)),
        "lambda_turnover": float(getattr(CFG, "LAMBDA_TURNOVER", 0.01)),
        "lookback_minutes": int(CFG.LOOKBACK_MINUTES),
        "horizon_minutes": int(CFG.HORIZON_MINUTES),
        "sort_key": 1,           # resample/make_anchors 已按时间排序
        "state_dict": {k: v for k, v in model.state_dict().items()},
    }
    save_model(model_path, payload)
    print(f"[train] 已保存 {model_path} ({os.path.getsize(model_path)/1e6:.2f} MB)")
    return model_path


if __name__ == "__main__":
    from bigmodule import M  # noqa: F401  (平台环境才有)
    datasources = {"bar1m": "cpt_jyc_2026_e2e_bar1m"}
    train_and_save(datasources)
