"""
加载 e2e 本地表：
- 价格、amount、盘口价格的单位是"分"，需要 /100 转为"元"
- OHLC 的 -1 表示缺失，需要转 NaN
- 盘口**价格**的 0 表示该侧无挂单（真实微观结构：涨停时卖侧全空、
  跌停时买侧全空、开盘最初几分钟买盘尚未形成），不是"0 元"，
  必须转 NaN；若当作 0 元，价格相对量会得到 ≈ -1.0 的极端噪声
  （正常范围约 ±0.02），远大于任何真实信号
- volume 单位是"股"，不缩放
- 只有 instrument_id，没有字符串代码，也没有 pre_close
- 有 adjust_factor：同一交易日内为常数（只在除权日之间变化），
  由于本项目的窗口/标签都不跨交易日，因此不需要额外复权处理
"""
import os
import numpy as np
import pandas as pd

from config import AUCTION_MINUTE

# 需要除以 100 的字段（价格、金额）
PRICE_FIELDS = [
    "open", "high", "low", "close", "amount",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
]
OHLC_FIELDS = ["open", "high", "low", "close"]
# 盘口价格字段（0 表示该侧无挂单 -> 必须转 NaN）
BOOK_PRICE_FIELDS = [f"{side}_price{k}"
                     for side in ("ask", "bid") for k in range(1, 4)]
# 除 OHLC 外还需要 /100 的字段（OHLC 要额外做 -1 -> NaN）
SCALE_ONLY_FIELDS = [f for f in PRICE_FIELDS if f not in OHLC_FIELDS]
BOOK_FIELDS = [f"{side}_{kind}{k}"
               for side in ("ask", "bid")
               for kind in ("price", "volume", "num_orders")
               for k in range(1, 4)]

# 训练真正需要的列
NEEDED_COLS = [
    "date", "instrument_id",
    "open", "high", "low", "close",
    "volume", "amount", "deal_number",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
]


def restore_local_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    把本地整数化表示还原为浮点表示。

    - OHLC：-1 表示缺失 -> NaN，再 /100 转元
    - amount / 盘口价格：/100 转元
    - 盘口价格中的 0 -> NaN（该侧无挂单；涨停/跌停/开盘初段）
    """
    df = df.copy()
    # OHLC 的 -1 表示缺失，先转 NaN，再 /100 转元
    for field in OHLC_FIELDS:
        if field in df.columns:
            df[field] = df[field].astype("float64")
            df.loc[df[field] == -1, field] = np.nan
            df[field] = df[field] / 100.0
    # 金额与盘口价格：分 -> 元
    for field in SCALE_ONLY_FIELDS:
        if field in df.columns:
            df[field] = df[field].astype("float64") / 100.0
    # 盘口价格 0 -> NaN。0 不是"0 元"而是"该侧无挂单"：
    #   ask_price1==0  涨停封板（卖侧被买光），实测 1.9%
    #   bid_price1==0  跌停或开盘初段买盘未形成，实测 4.3%
    # 若保留 0，价格相对量 = 0/close - 1 ≈ -1.0，是正常范围(±0.02)的 50 倍，
    # 会成为主导性的输入噪声。转 NaN 后由标准化统一填 0（训练集均值）。
    # 注意：只处理**价格**，盘口 volume/num_orders 保留原值（0 是真实的"无挂单"）。
    for field in BOOK_PRICE_FIELDS:
        if field in df.columns:
            df.loc[df[field] == 0.0, field] = np.nan
    return df


def filter_auction_and_tag_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    剔除盘前指示价 bar，并打上交易日标签 day。

    剔除规则（按**时钟时间**，不看行号）：只剔除 bar1m 的 09:16–09:24。
    - 这段 bar 的 volume=0、amount=0，close 与 09:25 完全相同，是集合竞价
      过程中的**指示性价格**，不是成交，不能作为输入。
    - **保留 09:25**：实测四个频率都有该 bar，约 40% 股票有真实成交，
      它是平台"开盘截面"（该截面用集合竞价数据预测未来 30 分钟 VWAP）。
    - **保留 09:30–15:00 连续竞价**的全部 bar。

    时间戳是 K 线**结束**时刻，因此有效 bar 必须满足
    minute_of_day >= 09:25（bar1m 补上 09:30 缺的那根，即从 09:31 起）。

    同时生成 minute_of_day（自午夜起的分钟数），后续采样一律按它对齐，
    这样即使某只股票/某天缺了 bar，采样时刻也不会漂移。
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    minute_of_day = df["date"].dt.hour * 60 + df["date"].dt.minute
    # 保留 09:25 及以后的 bar；剔除 bar1m 的 09:16–09:24 盘前指示价
    keep = minute_of_day >= AUCTION_MINUTE
    df = df[keep].copy()

    df["minute_of_day"] = minute_of_day[keep].astype("int16")
    df["day"] = df["date"].dt.date
    # 当日 bar 序号（仅用于窗口连续性与调试，不用于采样对齐）
    df = df.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    df["bar_idx"] = df.groupby(["instrument_id", "day"]).cumcount()
    return df


def load_month(data_dir: str, table: str, yyyymm: str,
               columns=None) -> pd.DataFrame:
    """读取单个月份的 feather 文件并还原单位。"""
    path = os.path.join(data_dir, table, f"{yyyymm}.0.feather")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_feather(path, columns=columns)
    return restore_local_data(df)


def load_months(data_dir: str, table: str, months, columns=None,
                verbose: bool = False) -> pd.DataFrame:
    """按月份列表拼接数据，转 float32 节省内存。"""
    dfs = []
    for m in months:
        try:
            dfs.append(load_month(data_dir, table, m, columns=columns))
            if verbose:
                print(f"  loaded {m}: {len(dfs[-1])} rows")
        except FileNotFoundError:
            print(f"  [warn] missing month {m}, skipped")
    if not dfs:
        raise FileNotFoundError(f"no data loaded for months={list(months)}")
    df = pd.concat(dfs, ignore_index=True)
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")
    return df
