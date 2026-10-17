"""
基础数据工程（允许范围）：单位还原 / 量纲统一 / 对数变换 / 缺失值处理。

注意：
- 不做任何手工因子（不构造量价组合、盘口衍生指标、滚动统计量）
- 标准化在 dataset 里做（只用训练集统计量）
- 标签的极端值截断也用训练集分位数（见 compute_clip_bounds）

所有"相对"变换都除以同一个 bar 的 close，只是把量纲统一到"相对当前价"，
不引入任何未来信息。
"""
import numpy as np
import pandas as pd

# 最终的 K 线特征列
KLINE_FEATURES = [
    "open_rel", "high_rel", "low_rel", "close_ret",
    "log_volume", "log_amount", "log_deal_number",
]

# 最终的盘口特征列（前 3 档，共 3*6 = 18 个）
ORDERBOOK_FEATURES = [
    "ask_price1_rel", "ask_price2_rel", "ask_price3_rel",
    "bid_price1_rel", "bid_price2_rel", "bid_price3_rel",
    "log_ask_volume1", "log_ask_volume2", "log_ask_volume3",
    "log_bid_volume1", "log_bid_volume2", "log_bid_volume3",
    "log_ask_orders1", "log_ask_orders2", "log_ask_orders3",
    "log_bid_orders1", "log_bid_orders2", "log_bid_orders3",
]

ALL_FEATURES = KLINE_FEATURES + ORDERBOOK_FEATURES

EPS = 1e-8


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造模型输入特征（只做单位/量纲统一与对数变换）。"""
    df = df.copy()
    df = df.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    close = df["close"]
    valid = close > 0

    # ===== K 线特征 =====
    df["open_rel"] = np.where(valid, df["open"] / close - 1.0, np.nan)
    df["high_rel"] = np.where(valid, df["high"] / close - 1.0, np.nan)
    df["low_rel"] = np.where(valid, df["low"] / close - 1.0, np.nan)

    # 收益率：同一交易日内的 close pct_change（不跨交易日）
    prev_close = df.groupby(["instrument_id", "day"], sort=False)["close"].shift(1)
    df["close_ret"] = np.where(prev_close > 0, close / prev_close - 1.0, np.nan)

    df["log_volume"] = np.log1p(df["volume"].clip(lower=0))
    df["log_amount"] = np.log1p(df["amount"].clip(lower=0))
    df["log_deal_number"] = np.log1p(df["deal_number"].clip(lower=0))

    # ===== 盘口特征 =====
    for k in range(1, 4):
        for side in ("ask", "bid"):
            df[f"{side}_price{k}_rel"] = np.where(
                valid, df[f"{side}_price{k}"] / close - 1.0, np.nan)
            df[f"log_{side}_volume{k}"] = np.log1p(
                df[f"{side}_volume{k}"].clip(lower=0))
            df[f"log_{side}_orders{k}"] = np.log1p(
                df[f"{side}_num_orders{k}"].clip(lower=0))

    return df


def build_labels(df: pd.DataFrame, horizon_bars: int = 1) -> pd.DataFrame:
    """
    标签：未来 horizon_bars 根 bar 的 VWAP 收益（严格按平台定义）。

        future_vwap = sum(amount[t+1 .. t+h]) / sum(volume[t+1 .. t+h])
        end_price   = close[t+h]        # 未来窗口最后一根 bar 的收盘价
        label       = end_price / future_vwap - 1      # 平台口径 vwap_return

    平台答疑确认 vwap_return = **end_price / vwap - 1**（不是 vwap/end_price-1）。
    两者对小幅收益近似互为相反数，方向反了会导致因子 IC 整体翻负。
    end_price 用"未来窗口最后一根 bar 的 close"（= 最终一笔成交价）。

    约束：
    - 窗口不跨交易日（按 instrument_id + day 分组）
    - 窗口内必须**存在成交**（fwd_vol > 0），否则 fwd_vwap 无定义
      * 零成交的分钟 bar 对 VWAP 分子分母的贡献都是 0，不影响 VWAP，
        因此**不要求**每根 bar 都有成交（对 bar1m 尤其重要：许多小票的
        1 分钟 bar 会是 volume=0）
    - 尾部的 h 根 bar 没有完整未来窗口（shift 后为 NaN），label = NaN
    """
    h = int(horizon_bars)
    assert h >= 1, "horizon_bars must be >= 1"

    df = df.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    grp = [df["instrument_id"], df["day"]]

    # 组内累积量，用于 O(1) 求任意窗口和。
    # amount/volume 在 load_months 里被压成 float32，若直接 cumsum，
    # 两个大数相减会产生 float32 抵消误差（实测标签相对误差 ~5e-5）。
    # 这里先提成 float64 再累积，保证标签精度。
    amt64 = df["amount"].astype("float64")
    vol64 = df["volume"].astype("float64")
    c_amt = amt64.groupby([df["instrument_id"], df["day"]], sort=False).cumsum()
    c_vol = vol64.groupby([df["instrument_id"], df["day"]], sort=False).cumsum()

    # 组内偏移 h：cumsum[t+h] - cumsum[t] = sum(positions t+1 .. t+h)
    amt_next = c_amt.groupby(grp, sort=False).shift(-h)
    vol_next = c_vol.groupby(grp, sort=False).shift(-h)

    fwd_amt = amt_next - c_amt
    fwd_vol = vol_next - c_vol

    # 窗口内必须有成交；否则（含窗口不完整的情况）置 NaN
    fwd_vwap = np.where(fwd_vol > 0, fwd_amt / fwd_vol, np.nan)

    # end_price = 未来窗口最后一根 bar 的 close（组内 shift(-h)）
    end_price = df["close"].astype("float64").groupby(
        grp, sort=False).shift(-h)

    end_arr = end_price.to_numpy(dtype="float64")
    label = np.where((fwd_vwap > 0) & (end_arr > 0),
                     end_arr / np.where(fwd_vwap > 0, fwd_vwap, 1.0) - 1.0,
                     np.nan)

    df["future_vwap"] = fwd_vwap
    df["end_price"] = end_arr
    df["label"] = label
    return df


def compute_clip_bounds(labels: np.ndarray, lower: float = 0.01,
                        upper: float = 0.99):
    """用训练集标签分位数计算截断边界（避免用全量数据造成泄漏）。"""
    x = np.asarray(labels, dtype="float64")
    x = x[~np.isnan(x)]
    if x.size == 0:
        return -np.inf, np.inf
    lo, hi = np.quantile(x, [lower, upper])
    return float(lo), float(hi)


def apply_clip(labels: np.ndarray, bounds) -> np.ndarray:
    lo, hi = bounds
    return np.clip(labels, lo, hi)
