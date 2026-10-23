# -*- coding: utf-8 -*-
"""文件用处：平台上以 bar30m 与 bar5m 重建 exp034 双频输入并执行跨频查询注意力推理。

口径与训练缓存严格一致（exp034 = exp033 缓存 × exp032 exampleloss 损失）：
1. 30m 侧沿用 exp033 的 example 式 25 特征：复权 close = close * adjust_factor；相对价以**原始 close**
   为分母；close_ret 与同股上一**实际观测** bar 的复权 close 相比（跨日携带隔夜收益）；盘口 0 价与
   ask1 < bid1 的掩码；09:25 竞价空 bar 只把 15 个量类特征置 NaN；slot0 的 bar_valid 放宽为只看 OHLC。
   最后用权重里的 30m mean/std（各 27 维）标准化。
2. 5m 侧按 exp033 cache_5m 的 98 格（上一交易日 49 格 + 当日 49 格）构建同样的 25 特征，并用 cache_5m
   的训练集统计量标准化、clip ±50；第 25 通道写 bar_valid，第 26 通道恒 0。当日块按截面输入结束时刻
   截断，未写入格与未来格在缓存里存 0，标准化后即 -mean/std。
3. 5m 格点映射与本实验 alignment.py 的 (8,16,6) gather / cell_segment 逐位一致。
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

import model


ROOT = Path(__file__).resolve().parent
SECTION_TABLE = "cpt_jyc_2026_instruments"
DECISION_HHMM = (930, 1000, 1030, 1100, 1130, 1330, 1400, 1430)

# 原始字段顺序必须与 exp033 features.py 的 RAW_COLS 一致（位置索引即特征索引）。
RAW_FIELDS = (
    "open", "high", "low", "close", "volume", "amount", "deal_number",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
)
C_VOLUME, C_AMOUNT, C_DEAL = 4, 5, 6
N_FEATURES, N_CHANNEL, CH_BAR_VALID, CH_GAP_FLAG = 25, 27, 25, 26
# 09:25 竞价空 bar 需要置 NaN 的 15 个量类通道
QUANTITY_COLS = np.asarray([4, 5, 6, *range(13, 25)], dtype=np.int64)
# 盘口 18 列与 (价, 量, 笔数) 三元组
OB_COLS = list(range(7, 25))
OB_GROUPS = [(7 + level, 13 + level, 19 + level) for level in range(3)] + [
    (10 + level, 16 + level, 22 + level) for level in range(3)
]
COL_ASK1, COL_BID1 = 7, 10

WINDOW, MIN_VALID_BARS, N_DAY_SLOTS, LEAD_IN_DAYS = 16, 8, 9, 40
N_FINE, N_FINE_DAY = 98, 49
CLIP_Z = 50.0
GRID30 = np.asarray([565, 600, 630, 660, 690, 810, 840, 870, 900], dtype=np.int64)
INPUT30 = np.asarray([565, 600, 630, 660, 690, 810, 840, 870], dtype=np.int64)
GRID5 = np.asarray([565, *range(575, 691, 5), *range(785, 901, 5)], dtype=np.int64)

# exp033 cache_5m/stats_5m.npz 的前 25 通道训练集统计量（float64 -> float32 存储）。
# 权重 JSON 只携带 30m 的 27 维 mean/std；受平台五个文件的包契约限制，5m 统计量内嵌在此。
STATS_5M_MEAN = np.asarray([
    -1.0228791325062048e-05, 0.0025131588336080313, -0.0024907742626965046,
    8.036107828957029e-06, 11.730358123779297, 14.321381568908691, 5.214993000030518,
    0.0007045346428640187, 0.001939702546223998, 0.0031563700176775455,
    -0.0006759742391295731, -0.0018791864858940244, -0.0030583813786506653,
    8.788694381713867, 9.130496978759766, 9.138110160827637, 8.910387992858887,
    9.378867149353027, 9.418570518493652, 1.880773663520813, 2.0349669456481934,
    2.009232521057129, 2.0223891735076904, 2.2782208919525146, 2.2873470783233643,
], dtype=np.float64)
STATS_5M_STD = np.asarray([
    0.004427162930369377, 0.003119669621810317, 0.0034492728300392628,
    0.005138415843248367, 1.456773281097412, 1.379389762878418, 1.1471514701843262,
    0.001048273523338139, 0.0016601962270215154, 0.00244133360683918,
    0.0010221891570836306, 0.0016215358627960086, 0.002404178725555539,
    2.25759220123291, 2.3025965690612793, 2.2873685359954834, 2.3412699699401855,
    2.326787233352661, 2.287137031555176, 1.1234970092773438, 1.1155669689178467,
    1.094411015510559, 1.228183627128601, 1.2129415273666382, 1.1887296438217163,
], dtype=np.float64)


def validate_datasources(datasources: dict) -> None:
    """拒绝缺少任一频率的数据源，防止双频模型被静默降级为单频。"""
    if not isinstance(datasources, dict) or {"bar30m", "bar5m"} - set(datasources):
        raise KeyError("datasources 必须同时含 bar30m 与 bar5m")


def _query(sql: str, filters: dict) -> pd.DataFrame:
    import dai
    try:
        return dai.query(sql, filters=filters, compression=True).df()
    except TypeError:
        return dai.query(sql, filters=filters).df()


def _sections(start: str, end: str) -> pd.DataFrame:
    """读取成分股截面表，并映射到 0..7 的决策槽（09:30..14:30）。"""
    data = _query(f"SELECT date, instrument::string AS instrument FROM {SECTION_TABLE}",
                  {"date": [start, end]})
    data["date"] = pd.to_datetime(data["date"])
    data = data.dropna(subset=["date", "instrument"]).drop_duplicates(["date", "instrument"])
    data = data[(data.date >= pd.Timestamp(start)) & (data.date < pd.Timestamp(end))].copy()
    hhmm = data.date.dt.hour * 100 + data.date.dt.minute
    if data.empty or not hhmm.isin(DECISION_HHMM).all():
        raise RuntimeError("成分股表为空或含非法截面时刻")
    data["slot"] = np.searchsorted(np.asarray(DECISION_HHMM, dtype=np.int64), hhmm.to_numpy())
    return data


def _bars(table: str, start: str, end: str, pool: list[str]) -> pd.DataFrame:
    """读取行情并按 (instrument, date) 稳定排序；30m 与 5m 表结构一致，共用本函数。"""
    columns = ", ".join(["date", "instrument::string AS instrument", "adjust_factor", *RAW_FIELDS])
    data = _query(f"SELECT {columns} FROM {table}", {"date": [start, end], "instrument": pool})
    data["date"] = pd.to_datetime(data["date"])
    return data.dropna(subset=["date", "instrument"]).drop_duplicates(
        ["instrument", "date"]).sort_values(["instrument", "date"], kind="stable").reset_index(drop=True)


def _lattice(frame: pd.DataFrame, days: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把单只股票行情映射到「交易日下标 × 时间格点」，返回 (交易日下标, 格点下标, 分钟)。"""
    day_index = {pd.Timestamp(day): i for i, day in enumerate(days)}
    day = frame.date.dt.normalize().map(day_index).to_numpy()
    minute = (frame.date.dt.hour * 60 + frame.date.dt.minute).to_numpy(np.int64)
    slot = np.searchsorted(grid, minute).astype(np.int64)
    inside = (slot < len(grid)) & (grid[np.clip(slot, 0, len(grid) - 1)] == minute)
    if np.isnan(day).any() or not inside.all():
        raise RuntimeError("行情包含不在训练格点内的时间")
    return day.astype(np.int64), slot, minute


def _mask_order_book(raw: np.ndarray) -> None:
    """exp033 features.mask_order_book：盘口 0 价与倒挂掩码，必须在任何比值/对数变换之前做。"""
    for price_col, volume_col, order_col in OB_GROUPS:
        empty = raw[:, price_col] == 0
        if empty.any():
            raw[empty, price_col] = np.nan
            raw[empty, volume_col] = np.nan
            raw[empty, order_col] = np.nan
    ask1, bid1 = raw[:, COL_ASK1], raw[:, COL_BID1]
    crossed = np.isfinite(ask1) & np.isfinite(bid1) & (ask1 < bid1)
    if crossed.any():
        raw[np.ix_(np.nonzero(crossed)[0], OB_COLS)] = np.nan


def _compute_features(raw: np.ndarray, close_adj: np.ndarray, prev_close_adj: np.ndarray) -> np.ndarray:
    """exp033 features.compute_features：25 个 example 式特征，(n, 25) float64，缺失保持 NaN。"""
    feature = np.full((raw.shape[0], N_FEATURES), np.nan, dtype=np.float64)
    close = raw[:, 3]
    close_ok = np.isfinite(close) & (close > 0)
    denom = np.where(close_ok, close, 1.0)
    for index in (0, 1, 2):
        value = raw[:, index]
        feature[:, index] = np.where(close_ok & np.isfinite(value), value / denom - 1.0, np.nan)
    prev_ok = np.isfinite(prev_close_adj) & (prev_close_adj > 0)
    feature[:, 3] = np.where(close_ok & prev_ok,
                             close_adj / np.where(prev_ok, prev_close_adj, 1.0) - 1.0, np.nan)
    for index in (4, 5, 6):
        value = raw[:, index]
        feature[:, index] = np.where(np.isfinite(value), np.log1p(np.clip(value, 0.0, None)), np.nan)
    for index in range(7, 13):
        value = raw[:, index]
        feature[:, index] = np.where(close_ok & np.isfinite(value), value / denom - 1.0, np.nan)
    for index in range(13, 25):
        value = raw[:, index]
        feature[:, index] = np.where(np.isfinite(value), np.log1p(np.clip(value, 0.0, None)), np.nan)
    return feature


def _bar_valid(raw: np.ndarray, slot: np.ndarray) -> np.ndarray:
    """bar_valid := OHLC 有限 且 (volume > 0 或 slot == 0)；slot0 是 09:25 竞价 bar，只看价。"""
    finite_ohlc = np.isfinite(raw[:, :4]).all(axis=1)
    return finite_ohlc & ((raw[:, C_VOLUME] > 0) | (slot == 0))


def _previous_finite(values: np.ndarray) -> np.ndarray:
    """逐格给出同股上一条有限且为正的值（跳过缺失格），用于复权 close_ret。"""
    previous = np.full(values.shape, np.nan, dtype=np.float64)
    last = np.nan
    for index, value in enumerate(values):
        previous[index] = last
        if np.isfinite(value) and value > 0:
            last = value
    return previous


def _features30(bars: pd.DataFrame, mean: np.ndarray, std: np.ndarray,
                days: np.ndarray) -> dict[str, tuple[np.ndarray, int]]:
    """按 exp033 训练缓存口径重建 30m 27 通道，返回 {股票: (通道矩阵, 首个 lattice 下标)}。"""
    result: dict[str, tuple[np.ndarray, int]] = {}
    for code, frame in bars.groupby("instrument", sort=False):
        day, slot, _ = _lattice(frame, days, GRID30)
        lattice = day * N_DAY_SLOTS + slot
        first = int(lattice[0])
        local = lattice - first
        length = int(local[-1]) + 1
        dense = np.full((length, 26), np.nan, dtype=np.float64)
        dense[local, :N_FEATURES] = frame[list(RAW_FIELDS)].to_numpy(np.float64)
        dense[local, N_FEATURES] = frame["adjust_factor"].to_numpy(np.float64)
        _mask_order_book(dense)
        close = dense[:, 3]
        close_ok = np.isfinite(close) & (close > 0)
        close_adj = np.where(close_ok, close * dense[:, N_FEATURES], np.nan)
        feature = _compute_features(dense, close_adj, _previous_finite(close_adj))
        slot_all = (first + np.arange(length)) % N_DAY_SLOTS
        valid = _bar_valid(dense, slot_all)
        gap = np.zeros(length, dtype=np.float64)
        gap[local[1:]] = (np.diff(local) != 1).astype(np.float64)
        # 09:25 竞价空 bar：只把 15 个量类特征置 NaN（z-score 后为 0），价类保留。
        auction_empty = (slot_all == 0) & ~(dense[:, C_VOLUME] > 0)
        if auction_empty.any():
            feature[np.ix_(np.nonzero(auction_empty)[0], QUANTITY_COLS)] = np.nan
        z = (feature - mean[:N_FEATURES]) / std[:N_FEATURES]
        z[~np.isfinite(z)] = 0.0
        stacked = np.concatenate((z, valid[:, None].astype(np.float64), gap[:, None]), axis=1)
        result[str(code)] = (stacked.astype(np.float32), first)
    return result


def _features5(bars: pd.DataFrame, days: np.ndarray) -> dict[str, np.ndarray]:
    """按 exp033 build_data_5m 口径重建 5m 27 通道（未标准化），返回 {股票: (交易日, 49, 27)}。"""
    result: dict[str, np.ndarray] = {}
    last_close: dict[str, float] = {}
    for code, frame in bars.groupby("instrument", sort=False):
        day, _, minute = _lattice(frame, days, GRID5)
        values = frame[[*RAW_FIELDS, "adjust_factor"]].to_numpy(np.float64)
        raw, adjust = values[:, :N_FEATURES], values[:, N_FEATURES]
        close = raw[:, 3]
        close_ok = np.isfinite(close) & (close > 0)
        close_adj = np.where(close_ok, close * adjust, np.nan)
        # 与构建脚本一致：日内用「上一行」的复权收盘（不跳过无效行），跨日接该股上一交易日的最后有效收盘。
        previous = np.full(close_adj.shape, np.nan, dtype=np.float64)
        same_day = day[1:] == day[:-1]
        previous[1:][same_day] = close_adj[:-1][same_day]
        starts = np.flatnonzero(np.concatenate(([True], day[1:] != day[:-1])))
        previous[starts] = last_close.get(str(code), np.nan)
        _mask_order_book(raw)
        feature = _compute_features(raw, close_adj, previous)
        auction = minute == 565
        valid = _bar_valid(raw, np.where(auction, 0, 1))
        auction_empty = auction & ~(raw[:, C_VOLUME] > 0)
        if auction_empty.any():
            feature[np.ix_(np.nonzero(auction_empty)[0], QUANTITY_COLS)] = np.nan
        block = np.zeros((len(days), N_FINE_DAY, N_CHANNEL), dtype=np.float64)
        local = np.searchsorted(GRID5, minute)
        block[day, local, :N_FEATURES] = feature
        block[day, local, CH_BAR_VALID] = valid.astype(np.float64)
        for current in np.unique(day):
            closes = close_adj[day == current]
            closes = closes[np.isfinite(closes) & (closes > 0)]
            if closes.size:
                last_close[str(code)] = float(closes[-1])
        result[str(code)] = block.astype(np.float32)
    return result


def _cell_tables() -> tuple[np.ndarray, np.ndarray]:
    """按本实验 alignment.py 重建 8×16×6 的真实 5m 格点映射（每 30m token 最多六格）。"""
    bin_index = np.full((8, N_FINE), -1, dtype=np.int8)
    segment5 = np.asarray([
        day * 3 + (0 if minute == 565 else (1 if minute <= 690 else 2))
        for day in range(2) for minute in GRID5
    ], dtype=np.int8)
    for slot, input_end in enumerate(INPUT30):
        endpoint_index = int(np.flatnonzero(GRID30 == input_end)[0])
        previous_start = 6 - endpoint_index
        current_start = 15 - endpoint_index
        endpoint_to_bin = {(0, int(endpoint)): previous_start + index
                           for index, endpoint in enumerate(GRID30)}
        endpoint_to_bin.update({(1, int(GRID30[index])): current_start + index
                                for index in range(endpoint_index + 1)})
        for day in range(2):
            for local_index, minute in enumerate(GRID5):
                if day == 1 and minute > input_end:
                    continue
                endpoint = 565 if minute == 565 else next(
                    int(end) for end in GRID30[1:] if end - 25 <= minute <= end)
                target = endpoint_to_bin.get((day, endpoint), -1)
                if 0 <= target < 16:
                    bin_index[slot, day * len(GRID5) + local_index] = target
    gather = np.full((8, 16, 6), -1, dtype=np.int16)
    cell_segment = np.full((8, 16, 6), -1, dtype=np.int8)
    counts = np.zeros((8, 16), dtype=np.int16)
    for slot in range(8):
        for source in range(N_FINE):
            target = int(bin_index[slot, source])
            if target < 0:
                continue
            position = int(counts[slot, target])
            if position >= 6:
                raise RuntimeError("5m 格点映射超过每个 30m token 的六格契约")
            gather[slot, target, position] = source
            cell_segment[slot, target, position] = segment5[source]
            counts[slot, target] += 1
    return gather, cell_segment


def _samples(section: pd.DataFrame, features30: dict[str, tuple[np.ndarray, int]],
             features5: dict[str, np.ndarray], days: np.ndarray,
             mean5: np.ndarray, std5: np.ndarray, pad30: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """逐股票切出 (16,27) 的 30m 窗口与 (98,27) 的 5m 窗口，口径与训练样本一致。"""
    day_index = {pd.Timestamp(day): i for i, day in enumerate(days)}
    x30, x5, slots, codes = [], [], [], []
    day = pd.Timestamp(section.date.iloc[0]).normalize()
    current = day_index.get(day)
    if current is None:
        return (np.empty((0, WINDOW, N_CHANNEL), np.float32), np.empty((0, N_FINE, N_CHANNEL), np.float32),
                np.empty(0, np.int64), [])
    decision_slot = int(section.slot.iloc[0])
    future5 = N_FINE_DAY + np.flatnonzero(GRID5 > INPUT30[decision_slot])
    for row in section.itertuples(index=False):
        coarse = features30.get(row.instrument)
        fine = features5.get(row.instrument)
        if coarse is None or fine is None:
            continue
        flat30, first = coarse
        end30 = current * len(GRID30) + decision_slot - first
        if end30 < 0 or end30 >= flat30.shape[0]:
            continue
        # 09:30 以 09:25 竞价 bar 为窗口末位（允许无成交）；其余截面必须有当期有效 30m bar。
        if decision_slot > 0 and flat30[end30, CH_BAR_VALID] <= 0:
            continue
        padding = max(0, WINDOW - 1 - end30)
        window30 = flat30[end30 - (WINDOW - 1 - padding):end30 + 1]
        if padding:
            window30 = np.concatenate((np.tile(pad30, (padding, 1)), window30), axis=0)
        if int((window30[:, CH_BAR_VALID] > 0).sum()) < MIN_VALID_BARS:
            continue
        previous = fine[current - 1] if current >= 1 else np.zeros_like(fine[0])
        window5 = np.concatenate((previous, fine[current]), axis=0).copy()
        # 与缓存一致：先把未写入/未来格置 0，再标准化，这些格因此为 -mean/std 且 bar_valid=0。
        window5[future5] = 0.0
        z5 = (window5[:, :N_FEATURES] - mean5[:N_FEATURES]) / std5[:N_FEATURES]
        z5[~np.isfinite(z5)] = 0.0
        np.clip(z5, -CLIP_Z, CLIP_Z, out=z5)
        window5 = np.concatenate((z5, window5[:, N_FEATURES:]), axis=1).astype(np.float32)
        x30.append(window30.astype(np.float32))
        x5.append(window5)
        slots.append(decision_slot)
        codes.append(row.instrument)
    if not codes:
        return (np.empty((0, WINDOW, N_CHANNEL), np.float32), np.empty((0, N_FINE, N_CHANNEL), np.float32),
                np.empty(0, np.int64), [])
    return (np.asarray(x30, np.float32), np.asarray(x5, np.float32),
            np.asarray(slots, np.int64), codes)


def _fill_weak_sections(output: pd.DataFrame, sections: pd.DataFrame) -> pd.DataFrame:
    """覆盖率低于 60% 的截面用该截面中位因子补齐，避免平台硬校验直接判废。"""
    expected = sections.groupby("date").instrument.nunique()
    ratio = output.groupby("date").instrument.nunique().reindex(expected.index, fill_value=0) / expected
    weak = ratio.index[ratio < 0.6]
    if not len(weak):
        return output
    median = output.groupby("date").factor.median().reindex(weak).fillna(0.0)
    pad = sections.loc[sections.date.isin(weak), ["date", "instrument"]].copy()
    pad["factor"] = pad["date"].map(median).astype(float)
    print(f"[warn] {len(weak)} 个截面覆盖率低于 60%，按截面中位因子补齐 {len(pad):,} 行", flush=True)
    return pd.concat((output, pad), ignore_index=True).drop_duplicates(["date", "instrument"], keep="first")


def validate_output(output: pd.DataFrame, sections: pd.DataFrame,
                    start: pd.Timestamp, end: pd.Timestamp) -> None:
    """平台写盘前的五项输出自检（列名/时刻/覆盖率/有限性/唯一性/时间窗）。"""
    if set(output.columns) != {"date", "instrument", "factor"}:
        raise RuntimeError("输出列必须恰为 date/instrument/factor")
    legal = pd.MultiIndex.from_frame(sections[["date", "instrument"]])
    actual = pd.MultiIndex.from_frame(output[["date", "instrument"]])
    hhmm = output.date.dt.hour * 100 + output.date.dt.minute
    if (not np.isfinite(output.factor).all() or output.duplicated(["date", "instrument"]).any()
            or not actual.isin(legal).all() or not hhmm.isin(DECISION_HHMM).all()
            or not ((output.date >= start) & (output.date < end)).all()):
        raise RuntimeError("输出不满足平台日期、有限性或唯一性契约")
    expected = sections.groupby("date").instrument.nunique()
    coverage = output.groupby("date").instrument.nunique().reindex(expected.index, fill_value=0) / expected
    if (coverage < 0.6).any():
        raise RuntimeError(f"截面覆盖率不足 60%，最小值 {coverage.min():.3f}")


def main(datasources, start_date, end_date):
    """平台入口：同一截面联合使用 30m 与 5m 数据，返回 date/instrument/factor 三列。"""
    started = perf_counter()
    validate_datasources(datasources)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)
    sections = _sections(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    pool = sorted(sections.instrument.unique())
    history_start = (start.normalize() - pd.Timedelta(days=LEAD_IN_DAYS)).strftime("%Y-%m-%d")
    bars30 = _bars(datasources["bar30m"], history_start, end.strftime("%Y-%m-%d"), pool)
    bars5 = _bars(datasources["bar5m"], history_start, end.strftime("%Y-%m-%d"), pool)
    days = np.unique(np.concatenate((sections.date.dt.normalize().to_numpy("datetime64[ns]"),
                                     bars30.date.dt.normalize().to_numpy("datetime64[ns]"),
                                     bars5.date.dt.normalize().to_numpy("datetime64[ns]"))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = model.load_model(ROOT / "model.json", str(device))
    for key in ("model_cfg", "mean", "std", "state_dict"):
        if key not in payload:
            raise RuntimeError(f"exp034 权重缺少 {key}")
    if int(payload["model_cfg"].get("n_in", 0)) != N_CHANNEL:
        raise RuntimeError("exp034 权重 model_cfg.n_in 必须为 27")
    mean = np.asarray(payload["mean"], dtype=np.float64)
    std = np.asarray(payload["std"], dtype=np.float64)
    if mean.shape != (N_CHANNEL,) or std.shape != (N_CHANNEL,) \
            or not np.isfinite(std).all() or not (std > 0).all():
        raise RuntimeError("exp034 权重 mean/std 必须是 27 维有限正标准差")
    pad30 = np.concatenate((-mean[:N_FEATURES] / std[:N_FEATURES], np.zeros(2))).astype(np.float32)
    features30 = _features30(bars30, mean, std, days)
    features5 = _features5(bars5, days)
    model_instance = model.build_model_from_cfg(payload["model_cfg"]).to(device)
    model_instance.load_state_dict(payload["state_dict"])
    model_instance.eval()
    gather, cell_segment = _cell_tables()
    frames = []
    for date, section in sections.groupby("date", sort=True):
        x30, x5, slots, codes = _samples(section, features30, features5, days,
                                         STATS_5M_MEAN, STATS_5M_STD, pad30)
        universe = section.instrument.unique()
        if not len(codes):
            print(f"[warn] {pd.Timestamp(date).date()} 无可用样本，交由覆盖率补齐处理", flush=True)
            continue
        if len(codes) < 0.6 * len(universe):
            print(f"[warn] {pd.Timestamp(date).date()} 覆盖率 {len(codes)}/{len(universe)}", flush=True)
        tx30, tx5 = torch.from_numpy(x30).to(device), torch.from_numpy(x5).to(device)
        with torch.no_grad():
            factor = model_instance(
                tx30, torch.from_numpy(slots).to(device), ~(tx30[:, :, CH_BAR_VALID] > 0),
                torch.zeros(len(codes), dtype=torch.long, device=device), x5=tx5,
                pad5=~(tx5[:, :, CH_BAR_VALID] > 0),
                gather=torch.from_numpy(gather).to(device, dtype=torch.long),
                cell_segment=torch.from_numpy(cell_segment).to(device, dtype=torch.long),
            ).float().cpu().numpy()
        frames.append(pd.DataFrame({"date": date, "instrument": codes, "factor": factor}))
    if not frames:
        raise RuntimeError("没有可推理样本")
    output = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan) \
        .dropna(subset=["factor"]).drop_duplicates(["date", "instrument"]) \
        .sort_values(["date", "instrument"], kind="stable").reset_index(drop=True)
    output = _fill_weak_sections(output, sections)
    validate_output(output, sections, start, end)
    print(f"[{ROOT.name}] {len(output):,} 行，耗时 {perf_counter() - started:.1f}s", flush=True)
    return output[["date", "instrument", "factor"]]
