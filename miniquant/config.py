"""
全局配置。

单位约定：
- 原始 feather 中 open/high/low/close/amount/盘口价格 的单位是"分"，读取后 /100 转为"元"。
- volume 单位为"股"，不做缩放。
- OHLC 中的 -1 表示缺失，读取后转 NaN。

时间约定（重要）：
- 本地表的时间戳是"K 线结束时刻"。
- 交易时段 09:30–11:30 与 13:00–15:00（午休不计入）。
  **保留** 09:25 集合竞价 bar（实测四个频率都有该 bar，约 40% 股票有真实
  成交，是平台"开盘截面"）；只剔除 bar1m 的 09:16–09:24 盘前指示价
  （volume=0、amount=0，close 与 09:25 相同，不是成交）。
- 采样按**时钟分钟**对齐（aligned_minutes），不依赖每只股票的行号：
  四个频率都在 09:25,10:00,10:30,11:00,11:30,13:30,14:00,14:30,15:00 采样，
  即每个交易日 **8 个日内截面**（15:00 无未来窗口，不参与训练/评估）。
- 标签严格按会议定义（end_price = 未来窗口最后一根 bar 的收盘价）：
  label = future_vwap / close[t+h] - 1，
  horizon_bars = 30 / bar_minutes（bar30m:1, bar15m:2, bar5m:6, bar1m:30）。
- 标签窗口严格限制在当日（不跨交易日）；输入窗口是否可跨日由
  input_same_day_only 控制。
"""
from dataclasses import dataclass, field
from typing import List

# 各频率的 bar 长度（分钟）
FREQ_MINUTES = {
    "bar1m": 1,
    "bar5m": 5,
    "bar15m": 15,
    "bar30m": 30,
}

# 交易时段（bar 结束时刻，含午休）
MORNING_START = 9 * 60 + 30      # 09:30 连续竞价开始
MORNING_END = 11 * 60 + 30       # 11:30 上午最后一根 bar 的结束时刻
AFTERNOON_START = 13 * 60        # 13:00 下午开始
AFTERNOON_END = 15 * 60          # 15:00 下午最后一根 bar 的结束时刻

# 09:25 集合竞价 bar 的结束时刻。实测该 bar 在四个频率中都存在且约 40% 的
# 股票有真实成交（volume>0）；它是平台"开盘截面"（该截面用集合竞价数据
# 预测），因此必须**保留**。需要剔除的是 bar1m 的 09:16–09:24 盘前指示价。
AUCTION_MINUTE = 9 * 60 + 25      # 09:25


@dataclass
class Config:
    # ===== 数据 =====
    data_dir: str = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant/e2e_data"
    freq: str = "bar30m"                    # 单频率模式: bar1m / bar5m / bar15m / bar30m
    # 多频率融合模式：非空时启用，所有分支必须在同一 (day, minute_of_day) 对齐。
    # 建议按 T 从大到小排列（决定模型分支顺序）。
    freqs: List[str] = field(default_factory=list)

    # 月份列表：文件名为 YYYYMM.0.feather，本地数据只到 202412
    train_months: List[str] = field(default_factory=lambda: [
        f"{y}{m:02d}" for y in range(2020, 2024) for m in range(1, 13)
    ])
    val_months: List[str] = field(default_factory=lambda: [
        f"2024{m:02d}" for m in range(1, 7)
    ])
    test_months: List[str] = field(default_factory=lambda: [
        f"2024{m:02d}" for m in range(7, 13)
    ])

    # 冒烟测试：只用少量月份小规模跑通
    smoke: bool = False
    smoke_train_months: List[str] = field(default_factory=lambda: ["202301"])
    smoke_val_months: List[str] = field(default_factory=lambda: ["202302"])

    # ===== 标签与窗口 =====
    horizon_minutes: int = 30              # 预测未来 30 分钟
    sample_step_minutes: int = 30          # 在整点/半点网格上采样
    lookback_minutes: int = 120            # 输入窗口覆盖的历史时长
    # 每频率单独覆盖 lookback（分钟），未指定的频率用 lookback_minutes。
    # 高频率分支（尤其 bar1m，T=120 时占 12KB/样本）是磁盘瓶颈，
    # 例如 {"bar1m": 60} 可把它减半，且 60 分钟对 1 分钟粒度通常已足够。
    lookback_minutes_per_freq: dict = field(default_factory=dict)
    # 输入窗口是否允许回溯到前一个交易日：
    # - False：窗口必须在当天内（对 bar5m/bar1m 这种高频表来说，
    #          一个 120 分钟的窗口会吃掉大半个交易日，样本极少）
    # - True ：允许跨日回溯（时间上连续的前 seq_len 根 bar），
    #          标签窗口仍然严格限制在当天，因此不会引入未来信息
    input_same_day_only: bool = False

    # ===== 模型 =====
    # 模型选择：gru（原双流 GRU+盘口 MLP + 注意力池化）
    #         cnn（CNN 滑窗 + Transformer Encoder，单流统一编码）
    model_arch: str = "gru"

    # --- GRU 方案参数（model_arch="gru" 时使用） ---
    kline_hidden: int = 64
    orderbook_hidden: int = 32
    fused_dim: int = 64
    gru_layers: int = 2
    dropout: float = 0.2

    # --- CNN+Transformer 方案参数（model_arch="cnn" 时使用） ---
    cnn_hidden: int = 64                     # 卷积/Transformer 隐藏维（单流统一）
    cnn_blocks: int = 3                      # 卷积残差块数（膨胀递进，1,2,4）
    cnn_kernel: int = 3                      # 卷积核大小
    transformer_layers: int = 2              # Transformer Encoder 层数
    transformer_heads: int = 4               # 注意力头数

    # ===== 训练 =====
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 30
    patience: int = 5                      # 早停 patience（看验证集 IC）
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 2

    # ===== 派生属性 =====
    @property
    def table(self) -> str:
        return f"cpt_jyc_2026_e2e_{self.freq}"

    @property
    def is_multifreq(self) -> bool:
        """是否启用多频率融合。"""
        return len(self.freqs) > 1

    def active_freqs(self):
        """参与建模的频率列表（多频模式用 freqs，否则用单 freq）。"""
        return list(self.freqs) if self.is_multifreq else [self.freq]

    def for_freq(self, freq: str) -> "Config":
        """返回一个只针对某单一频率的副本（用于逐分支构造数据）。"""
        import copy
        c = copy.copy(self)
        c.freq = freq
        c.freqs = []
        # 把该频率的 lookback 覆盖固化到 lookback_minutes，
        # 使 c.seq_len_bars 直接反映该分支的窗口长度
        c.lookback_minutes = self.lookback_for(freq)
        c.lookback_minutes_per_freq = {}
        return c

    @property
    def tables(self):
        return [(f, f"cpt_jyc_2026_e2e_{f}") for f in self.active_freqs()]

    @property
    def bar_minutes(self) -> int:
        return FREQ_MINUTES[self.freq]

    @property
    def bars_per_day(self) -> int:
        """
        每个交易日的 bar 数。

        = 09:25 集合竞价 bar + 连续竞价的 240 分钟（bar30m: 1+8=9，
        bar5m: 1+48=49，bar1m: 1+240=241），与本地表实测的每日 bar 数一致。
        """
        continuous = (MORNING_END - MORNING_START
                      + AFTERNOON_END - AFTERNOON_START) // self.bar_minutes
        return 1 + continuous

    @property
    def horizon_bars(self) -> int:
        """未来 horizon_minutes 对应多少根 bar。"""
        return max(1, self.horizon_minutes // self.bar_minutes)

    def lookback_for(self, freq=None) -> int:
        """某频率实际使用的 lookback（分钟），支持每频率覆盖。"""
        f = freq or self.freq
        return int(self.lookback_minutes_per_freq.get(
            f, self.lookback_minutes))

    @property
    def seq_len_bars(self) -> int:
        """输入窗口对应的 bar 数（由分钟换算，随频率自适应）。"""
        n = self.lookback_for(self.freq) // self.bar_minutes
        return max(1, min(n, self.bars_per_day - 1))

    @property
    def sample_step_bars(self) -> int:
        """采样间隔对应的 bar 数（在整点/半点网格上）。"""
        return max(1, self.sample_step_minutes // self.bar_minutes)

    def valid_minutes(self):
        """
        全部有效的 bar 结束时刻（自午夜起的分钟数）。

        顺序为：09:25 集合竞价 bar + 上午连续竞价 + 下午连续竞价。
        连续竞价分上下午两段，午休（11:30–13:00）不计入，因此不能简单按
        09:30 + k*bar_minutes 线性推算。这里显式生成，保证
        bar1m 得到 09:25 + 09:31..11:30 + 13:01..15:00，
        bar30m 得到 09:25 + 10:00..15:00，与本地表实测 bar 数完全一致。
        """
        step = self.bar_minutes
        out = [AUCTION_MINUTE]
        t = MORNING_START + step
        while t <= MORNING_END:
            out.append(t)
            t += step
        t = AFTERNOON_START + step
        while t <= AFTERNOON_END:
            out.append(t)
            t += step
        return out

    def aligned_minutes(self):
        """
        采样时刻（截面）集合。

        = 09:25 开盘截面 ∪ 分钟数能被 sample_step_minutes 整除的整点/半点。
        09:25（565 分钟）不在 30 分钟网格上，必须显式加入。

        只依赖**时钟**，不依赖每只股票当天的 bar 行号，因此
        - 缺 bar / 停牌 不会导致采样时刻漂移
        - 四个频率在同一时刻采样：
          09:25,10:00,10:30,11:00,11:30,13:30,14:00,14:30,15:00
        （15:00 无未来窗口，标签为 NaN，实际不参与训练/评估）
        即每个交易日 **8 个日内截面**，与平台要求一致。
        """
        step = self.sample_step_minutes
        out = [m for m in self.valid_minutes() if m == AUCTION_MINUTE]
        out += [m for m in self.valid_minutes()
                if m != AUCTION_MINUTE and m % step == 0]
        return sorted(out)

    def aligned_bar_indices(self):
        """向后兼容：采样时刻在 valid_minutes() 中的位置下标集合。"""
        vm = self.valid_minutes()
        target = set(self.aligned_minutes())
        return [i for i, m in enumerate(vm) if m in target]

    def resolved_months(self):
        """返回 (train_months, val_months, test_months)。smoke 模式下覆盖为小集合。"""
        if self.smoke:
            return self.smoke_train_months, self.smoke_val_months, self.test_months
        return self.train_months, self.val_months, self.test_months