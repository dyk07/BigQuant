# -*- coding: utf-8 -*-
"""exp009 平台推理脚本（公榜提交主逻辑，与 Notebook 一起上传）。

流程 main(datasources, start_date, end_date)：
  1. 从权重文件 exp009_model.json 重建 exp009_model.BaselineTransformer（eval，CPU/GPU 自适应）；
  2. dai 查 30m 在线 bar 表（buf = start - 25 日历天起的 lead-in）与截面成分股表；
  3. 逐股重建与训练侧同口径的 27 通道特征（严格复刻 exp003 build_data.py 趟 A 的 27ch 子集：
     盘口掩码 -> dense lattice -> 复权对数收益 / 成交量 log1p 差分 -> 用权重内 train 期
     mean/std 做 z-score（非有限值置 0）-> 拼 bar_valid / gap_flag 两个 flag 通道）；
   4. 按「窗口 = [j-15, j] 共 16 根、末行为样本 bar、窗口内有效 bar >= 8」采样：
      10:00~14:30 共 7 个截面以 bar 结束时刻为样本 bar（slot_id = slot-1，0..6）；
      9:30 截面按 B1 近似以竞价 bar（结束于 09:25，slot 0）为窗口末位，末位允许
      无效（pad 掩码兜底），slot_id 借用 10:00 嵌入 0，数据严格早于截面时刻；
   5. 批量前向打分，输出平台契约三列 date / instrument / factor，并跑输出自检清单。

数据口径（与 docs/knowledge.md §3、src/data_io.py 对齐约定一致）：
- 云端价格为元（无需 ÷100），OHLC 缺失即 NaN，盘口缺失为 0，只用盘口前 3 档；
- 分组键用 instrument 字符串列（云端与 instrument_id 一一对应，本地无该列）。

平台提交契约（官方 wiki《新手入门》2026-09-30 版）：
- 输出仅三列 date, instrument, factor；date 为 30 分钟采样时点（精确到分钟）；
- 清理范式：±inf -> NaN -> dropna(factor) -> drop_duplicates([date, instrument])；
- 覆盖全部 8 个截面（9:30~14:30）；9:30 用 B1 近似的依据见
  SECTION_SLOT_BY_HHMM 注释与 docs/experiments/exp004_run7prime_datav2.md；
- 标准化统计量来自权重文件（训练期统计），推理侧不重算，无信息泄漏。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import exp009_model

__all__ = ["main"]

# ---------------------------------------------------------------------------
# 常量（与 exp009 search.json / exp003 build_data.py 严格一致，不得单方面改动）
# ---------------------------------------------------------------------------

#: 截面成分股表（官方 wiki 指定的股票池表）
SECTION_TABLE = "cpt_jyc_2026_instruments"

#: 特征 lead-in：向前多取的日历天（需 >= 15 个 lattice 槽位 + 节假日余量）
LEAD_IN_DAYS = 25

#: 窗口长度（30m bar 根数）与窗口内有效 bar 下限
WINDOW = 16
MIN_VALID_BARS = 8

#: 30m 日内 lattice 每天根数；样本槽位范围（slot 1..7 = 10:00..14:30）
N_SLOTS_PER_DAY = 9
FIRST_SAMPLE_SLOT = 1
LAST_SAMPLE_SLOT = 7

#: 推理批量
BATCH_SIZE = 8192

#: 日内 9 根 30m bar 的结束时刻（docs/knowledge.md §3.4），映射 slot 0..8
SLOT_BY_HHMM = {925: 0, 1000: 1, 1030: 2, 1100: 3, 1130: 4,
                1330: 5, 1400: 6, 1430: 7, 1500: 8}

#: 已知但不产出的截面时刻：无。9:30 截面已按 B1 近似覆盖（见下），不再跳过
SKIPPED_SECTION_HHMM: frozenset = frozenset()

#: 输出允许的时刻集合（平台 8 截面：9:30 + slot 1..7 的 bar 结束时刻）
LEGAL_OUTPUT_HHMM = frozenset({930, 1000, 1030, 1100, 1130, 1330, 1400, 1430})

#: 截面时刻 -> 采样槽位映射。9:30 截面没有结束于 09:30 的 30m bar，按方案 B1
#: 用竞价 bar（slot 0，结束于 09:25）作窗口末位：数据严格早于截面时刻、无泄漏；
#: 末位允许无效（pad 掩码让注意力只看窗口内有效 bar），slot_id 借用 10:00 嵌入。
#: 本地 2024H1 实测（compare_930.py，代理标签=复权 close(10:00)/close(09:25)-1，
#: 逐日 Spearman）：B1 IC 0.1188/ICIR 2.33，优于 B2（末位取最近有效 bar）
#: 的 0.0989/2.04，故采用 B1。
SECTION_SLOT_BY_HHMM = {**SLOT_BY_HHMM, 930: 0}

#: 27 通道语义（与权重文件 feature_spec.channel_order 逐字一致）
CHANNEL_ORDER_DESC = "price(10) -> volume_like(15) -> bar_valid -> gap_flag"

#: 价类通道：对「复权价」做对数收益差分（先除当行 adjust_factor 再 diff(log)）
PRICE_LOG_RETURN_FIELDS = (
    "open", "high", "low", "close",
    "ask_price1", "ask_price2", "ask_price3",
    "bid_price1", "bid_price2", "bid_price3",
)

#: 量类通道：log1p 后差分（volume 类量纲两端一致，不缩放）
VOLUME_LOG_DIFF_FIELDS = (
    "volume", "amount", "deal_number",
    "ask_volume1", "ask_volume2", "ask_volume3",
    "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
)

#: feat25 = 价类(0-9) | 量类(10-24)
FEATURE_COLUMNS = PRICE_LOG_RETURN_FIELDS + VOLUME_LOG_DIFF_FIELDS
assert len(FEATURE_COLUMNS) == 25

#: 盘口掩码分组：档价 == 0 -> 该档价 + 对应档量/笔数置 NaN（三档，买卖两侧）
OB_LEVELS = tuple(
    (f"{side}_price{k}", f"{side}_volume{k}", f"{side}_num_orders{k}")
    for side in ("ask", "bid") for k in (1, 2, 3)
)
#: 倒挂（ask1 < bid1 且两侧均有限）时整行置 NaN 的 18 个盘口列
OB_ALL_COLUMNS = tuple(c for lvl in OB_LEVELS for c in lvl)

#: OHLC（bar_valid = OHLC 全有限 且 volume > 0）
OHLC_FIELDS = ("open", "high", "low", "close")

#: 30m 在线表查询列
BAR_QUERY_COLUMNS = (
    "date", "instrument::string AS instrument", "adjust_factor",
    *OHLC_FIELDS, *FEATURE_COLUMNS,
)


def _log(msg: str) -> None:
    """带前缀的进度日志（平台从 stdout 采集）。"""
    print(f"[exp009] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 数据获取（dai 延迟导入：本地对账不经过本模块，见 local_check.py）
# ---------------------------------------------------------------------------

def _dai_query(sql: str, filters: dict) -> pd.DataFrame:
    """执行一次 dai 查询并返回 DataFrame。

    dai 只在平台评测环境可用，故延迟到调用时导入；compression=True 按平台要点
    开启（instrument 转 category 省内存，SQL 内已用 instrument::string 转回），
    若平台 API 变动导致不认该参数则降级用默认参数重试。
    """
    import dai  # noqa: PLC0415（平台环境专属依赖，本地不存在）
    try:
        return dai.query(sql, filters=filters, compression=True).df()
    except TypeError:
        _log("dai.query 不支持 compression=True，改用默认参数重试")
        return dai.query(sql, filters=filters).df()


def _load_pool(sd_q: str, ed_q: str) -> list[str]:
    """评测区间内的截面成分股代码（官方 wiki：股票池用 cpt_jyc_2026_instruments）。"""
    df = _dai_query(
        f"SELECT DISTINCT instrument::string AS instrument FROM {SECTION_TABLE}",
        filters={"date": [sd_q, ed_q]},
    )
    pool = sorted(df["instrument"].dropna().unique().tolist())
    if not pool:
        raise RuntimeError(f"截面成分股为空（{sd_q} ~ {ed_q}）")
    return pool


def _load_sections(sd_q: str, ed_q: str, pool: list[str]) -> pd.DataFrame:
    """截面表行（date=截面时刻, instrument），评测区间内。"""
    df = _dai_query(
        f"SELECT date, instrument::string AS instrument FROM {SECTION_TABLE}",
        filters={"date": [sd_q, ed_q], "instrument": pool},
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna(subset=["date", "instrument"]).drop_duplicates(["date", "instrument"])


def _load_bars(table: str, buf_q: str, ed_q: str, pool: list[str]) -> pd.DataFrame:
    """30m 在线 bar 表（含 lead-in），只取 27 通道所需原始列。"""
    cols = ", ".join(BAR_QUERY_COLUMNS)
    df = _dai_query(
        f"SELECT {cols} FROM {table}",
        filters={"date": [buf_q, ed_q], "instrument": pool},
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# 权重加载与校验
# ---------------------------------------------------------------------------

def _resolve_model_path() -> Path:
    """定位权重文件：优先与本脚本同目录，其次当前工作目录。"""
    here = Path(__file__).resolve().parent / "exp009_model.json"
    if here.exists():
        return here
    cwd = Path.cwd() / "exp009_model.json"
    if cwd.exists():
        return cwd
    raise FileNotFoundError("找不到 exp009_model.json（应与 exp009_predict.py 同目录上传）")


def _check_ckpt(ckpt: dict) -> None:
    """权重元信息校验：口径声明与 27 通道统计量必须与推理脚本一致。"""
    align = ckpt.get("data_alignment")
    if align is not None and align != exp009_model.DATA_ALIGNMENT:
        raise RuntimeError(f"权重 data_alignment={align!r} 与推理脚本口径不符")
    spec = ckpt.get("feature_spec")
    if isinstance(spec, dict):
        if spec.get("channel_order") not in (None, CHANNEL_ORDER_DESC):
            raise RuntimeError(f"权重 channel_order={spec.get('channel_order')!r} 与本脚本不符")
        for key, mine in (("price_log_return_fields", PRICE_LOG_RETURN_FIELDS),
                          ("volume_log_diff_fields", VOLUME_LOG_DIFF_FIELDS)):
            theirs = spec.get(key)
            if theirs is not None and list(theirs) != list(mine):
                raise RuntimeError(f"权重 feature_spec.{key} 与本脚本字段清单不符")
    mean, std = ckpt.get("mean"), ckpt.get("std")
    if mean is None or std is None or len(mean) != 27 or len(std) != 27:
        raise RuntimeError("权重缺少 27 维 mean/std 标准化统计量")


# ---------------------------------------------------------------------------
# 特征重建（严格复刻 exp003 build_data.py 趟 A 的 27ch 子集）
# ---------------------------------------------------------------------------

def _stock_features(g: pd.DataFrame, day_pos: pd.Series,
                    mean: np.ndarray, std: np.ndarray,
                    instrument: str) -> tuple[np.ndarray, int, int]:
    """重建单只股票的 27 通道 dense 特征。

    Args:
        g: 该股全部 30m bar 行（已按 date 升序、去重）。
        day_pos: 全局交易日 -> 序号（0..D-1）映射（由全部已加载 bar 的日期构成）。
        mean/std: 权重文件内的 27 维标准化统计量（只用前 25 通道）。
        instrument: 股票代码（仅用于报错信息）。

    Returns:
        (x, lat0, n_dense)：
        x — (n_dense, 27) float32，行号 = lattice 槽位 j（相对该股首根 bar）；
        lat0 — 该股首根 bar 的全局 lattice 位置（day_idx*9 + slot）；
        n_dense — dense 行数。
    """
    hhmm = g["date"].dt.hour * 100 + g["date"].dt.minute
    slot = hhmm.map(SLOT_BY_HHMM)
    if slot.isna().any():
        raise RuntimeError(f"{instrument} 存在不在 30m lattice（{sorted(SLOT_BY_HHMM)}）上的 bar 时间")
    day_idx = g["date"].dt.normalize().map(day_pos)
    if day_idx.isna().any():
        raise RuntimeError(f"{instrument} 存在不在 bar 日历中的日期")

    lat = day_idx.to_numpy(np.int64) * N_SLOTS_PER_DAY + slot.to_numpy(np.int64)
    if np.any(np.diff(lat) <= 0):
        raise RuntimeError(f"{instrument} bar 时间重复或乱序")
    lat0 = int(lat[0])
    j = lat - lat0
    n_dense = int(j[-1]) + 1

    # 取原始列并做盘口掩码（先掩后差分，exp003 build_data._mask_order_book 口径）：
    # 档价 == 0（云端缺失记法）-> 该档价 + 对应档量/笔数置 NaN；ask1/bid1 倒挂 -> 18 列全 NaN。
    col = {c: i for i, c in enumerate(FEATURE_COLUMNS + ("adjust_factor",))}
    vals = g[list(col)].to_numpy(np.float64)
    for p_col, v_col, n_col in OB_LEVELS:
        m = vals[:, col[p_col]] == 0.0  # NaN == 0 为 False，天然安全
        if m.any():
            vals[np.ix_(m, [col[p_col], col[v_col], col[n_col]])] = np.nan
    a1 = vals[:, col["ask_price1"]]
    b1 = vals[:, col["bid_price1"]]
    inv = np.isfinite(a1) & np.isfinite(b1) & (a1 < b1)
    if inv.any():
        vals[np.ix_(inv, [col[c] for c in OB_ALL_COLUMNS])] = np.nan

    # dense lattice：缺失槽位整行 NaN（停牌/缺槽对齐，docs/knowledge.md §3.7）
    dense = np.full((n_dense, len(col)), np.nan, np.float64)
    dense[j] = vals

    # flag 通道 25：bar_valid = OHLC 全有限 且 volume > 0
    ohlc = dense[:, [col[c] for c in OHLC_FIELDS]]
    bar_valid = np.isfinite(ohlc).all(axis=1) & (dense[:, col["volume"]] > 0)

    # flag 通道 26：gap_flag = 相邻两根存在 bar 的 lattice 槽位不连续（=停牌/缺槽），首行 0
    gap_flag = np.zeros(n_dense, np.float64)
    gap_flag[j[1:]] = np.diff(j) != 1

    with np.errstate(divide="ignore", invalid="ignore"):
        # 价类：复权（除以当行 adjust_factor）后对数收益差分；首行 NaN
        p = dense[:, [col[c] for c in PRICE_LOG_RETURN_FIELDS]]
        adj = dense[:, col["adjust_factor"]][:, None]
        p_adj = np.where(np.isfinite(p) & (p > 0), p / adj, np.nan)
        lr = np.full_like(p_adj, np.nan)
        lr[1:] = np.diff(np.log(p_adj), axis=0)
        # 量类：log1p 后差分；负值截 0，缺失 NaN；首行 NaN
        v = dense[:, [col[c] for c in VOLUME_LOG_DIFF_FIELDS]]
        v = np.where(np.isfinite(v), np.maximum(v, 0.0), np.nan)
        vl = np.log1p(v)
        vd = np.full_like(vl, np.nan)
        vd[1:] = np.diff(vl, axis=0)

    # z-score 只作用前 25 通道（统计量来自权重文件 = train 期统计），非有限值置 0，不截断
    feat = np.concatenate([lr, vd], axis=1)
    z = (feat - mean[None, :25]) / std[None, :25]
    z[~np.isfinite(z)] = 0.0

    x = np.empty((n_dense, 27), np.float32)
    x[:, :25] = z
    x[:, 25] = bar_valid
    x[:, 26] = gap_flag
    return x, lat0, n_dense


# ---------------------------------------------------------------------------
# 截面准备与采样
# ---------------------------------------------------------------------------

def _prepare_sections(sections: pd.DataFrame, day_pos: pd.Series,
                      sd_ts: pd.Timestamp, ed_next: pd.Timestamp
                      ) -> tuple[pd.DataFrame, int]:
    """给截面表补 slot/day_idx 列并过滤出可采样的截面。

    Returns:
        (sec, n_skipped)：sec 含 date/instrument/slot/day_idx 列；n_skipped 为
        无法映射到采样槽位的截面行数（正常应为 0）。
    """
    sec = sections[(sections["date"] >= sd_ts.normalize())
                   & (sections["date"] < ed_next)].copy()
    if (sec["date"].dt.second != 0).any():
        raise RuntimeError("截面时刻含非整分钟秒位，与 30 分钟采样点契约不符")
    hhmm = sec["date"].dt.hour * 100 + sec["date"].dt.minute
    known = hhmm.isin(SECTION_SLOT_BY_HHMM)
    unknown = hhmm[~known & ~hhmm.isin(SKIPPED_SECTION_HHMM)]
    if len(unknown):
        raise RuntimeError(f"出现未知截面时刻 {sorted(unknown.unique())}，"
                           f"既非 30m bar 结束时刻也不在已知跳过清单里")
    n_skipped = int((~known).sum())
    sec = sec[known].copy()
    sec["slot"] = hhmm[known].map(SECTION_SLOT_BY_HHMM).astype(np.int64)
    # slot 0 仅来自 9:30 截面的 B1 近似（样本 bar = 竞价 bar，允许无效），见常量注释
    sec = sec[sec["slot"].between(0, LAST_SAMPLE_SLOT)]
    day_idx = sec["date"].dt.normalize().map(day_pos)
    if day_idx.isna().any():
        missing = sorted(sec.loc[day_idx.isna(), "date"].dt.normalize().unique())
        raise RuntimeError(f"截面日期不在 30m bar 日历中（当日无任何 bar？）：{missing}")
    sec["day_idx"] = day_idx.astype(np.int64)
    return sec, n_skipped


def main(datasources, start_date, end_date):
    """平台推理入口（签名不得改动）：返回三列 date/instrument/factor。"""
    t0 = time.time()
    if not isinstance(datasources, dict) or "bar30m" not in datasources:
        raise KeyError(f"datasources 需包含逻辑名 bar30m，实际为 {datasources}")
    table = datasources["bar30m"]

    sd_ts = pd.to_datetime(start_date)
    ed_ts = pd.to_datetime(end_date)
    sd_q = sd_ts.strftime("%Y-%m-%d")
    ed_next = ed_ts.normalize() + pd.Timedelta(days=1)  # end 按自然日含当日
    ed_q = ed_next.strftime("%Y-%m-%d")
    buf_q = (sd_ts.normalize() - pd.Timedelta(days=LEAD_IN_DAYS)).strftime("%Y-%m-%d")
    _log(f"评测区间 {sd_q} ~ {ed_q}（特征 lead-in 自 {buf_q}），30m 表 {table}")

    # ---- 权重 ----
    model_path = _resolve_model_path()
    ckpt = exp009_model.load_model(model_path)
    _check_ckpt(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = exp009_model.build_model_from_cfg(ckpt["model_cfg"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    mean = np.asarray(ckpt["mean"], np.float64)
    std = np.asarray(ckpt["std"], np.float64)
    _log(f"权重加载完成（{model.num_parameters()} 参数，device={device}），"
         f"耗时 {time.time() - t0:.1f}s")

    # ---- 数据 ----
    pool = _load_pool(sd_q, ed_q)
    bars = _load_bars(table, buf_q, ed_q, pool)
    bars = bars[(bars["date"] >= sd_ts.normalize() - pd.Timedelta(days=LEAD_IN_DAYS))
                & (bars["date"] < ed_next)]
    if bars.empty:
        raise RuntimeError("30m bar 查询结果为空")
    if (bars["date"].dt.second != 0).any():
        raise RuntimeError("bar 时间含非整分钟秒位，与 30m lattice 口径不符")
    bars = bars.drop_duplicates(["instrument", "date"], keep="first")
    bars = bars.sort_values(["instrument", "date"], kind="stable")
    _log(f"bar 行数 {len(bars):,}（{bars['instrument'].nunique()} 只），"
         f"耗时 {time.time() - t0:.1f}s")

    sections = _load_sections(sd_q, ed_q, pool)
    days = np.unique(bars["date"].dt.normalize().to_numpy())
    day_pos = pd.Series(np.arange(len(days), dtype=np.int64), index=days)
    sections, n_skipped = _prepare_sections(sections, day_pos, sd_ts, ed_next)
    _log(f"截面行数 {len(sections):,}（按设计跳过无对应 bar 的截面 {n_skipped} 行）")

    groups = {k: v for k, v in bars.groupby("instrument", sort=False)}

    # ---- 采样 + 流式批量前向 ----
    x_buf: list[np.ndarray] = []
    meta_buf: list[tuple] = []
    score_parts: list[np.ndarray] = []
    meta_parts: list[tuple] = []

    def _flush() -> None:
        if not x_buf:
            return
        xb = torch.from_numpy(np.stack(x_buf)).to(device)
        st = torch.tensor([m[2] for m in meta_buf], dtype=torch.int64, device=device)
        pad = ~(xb[:, :, 25] > 0)  # True = 无效 bar，不参与注意力
        with torch.no_grad():
            s = model(xb, st, pad)
        score_parts.append(s.detach().float().cpu().numpy())
        meta_parts.extend((m[0], m[1]) for m in meta_buf)
        x_buf.clear()
        meta_buf.clear()

    for instrument, sec_g in sections.groupby("instrument", sort=False):
        g = groups.get(instrument)
        if g is None:  # 截面表有、bar 表无：无法构特征，按缺失处理
            continue
        x, lat0, n_dense = _stock_features(g, day_pos, mean, std, instrument)
        # 截面所在 lattice 槽位（相对该股首根 bar）
        d = (sec_g["day_idx"].to_numpy(np.int64) * N_SLOTS_PER_DAY
             + sec_g["slot"].to_numpy(np.int64)) - lat0
        cand = np.nonzero((d >= WINDOW - 1) & (d < n_dense))[0]
        g_dates = sec_g["date"].to_numpy()
        g_slots = sec_g["slot"].to_numpy()
        for pos in cand:
            di = int(d[pos])
            slot_pos = int(g_slots[pos])
            if slot_pos >= 1 and x[di, 25] <= 0:
                # slot 1..7：样本 bar 本身必须有效（恒为窗口末位）；
                # slot 0（9:30 截面 B1 近似）：竞价 bar 允许无效，pad 掩码兜底
                continue
            if int((x[di - WINDOW + 1: di + 1, 25] > 0).sum()) < MIN_VALID_BARS:
                continue
            x_buf.append(x[di - WINDOW + 1: di + 1])
            # slot_id：slot 1..7 -> 0..6；slot 0（9:30）借用 10:00 的嵌入 0
            meta_buf.append((g_dates[pos], instrument,
                             slot_pos - 1 if slot_pos >= 1 else 0))
            if len(x_buf) >= BATCH_SIZE:
                _flush()
    _flush()

    if not meta_parts:
        raise RuntimeError("无任何样本产出，请检查数据区间与股票池")
    _log(f"采样完成：{len(meta_parts):,} 个样本，耗时 {time.time() - t0:.1f}s")

    # ---- 输出清理（平台契约范式：±inf -> NaN -> dropna -> 去重 -> 仅三列）----
    out = pd.DataFrame(meta_parts, columns=["date", "instrument"])
    out["factor"] = np.concatenate(score_parts).astype(np.float64)
    out["date"] = pd.to_datetime(out["date"])
    out["factor"] = out["factor"].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["factor"])
    out = out.drop_duplicates(["date", "instrument"], keep="first")
    out = out.sort_values(["date", "instrument"], kind="stable").reset_index(drop=True)
    out = out[["date", "instrument", "factor"]]

    _run_checks(out, sections, sd_ts, ed_next)
    _log(f"输出 {len(out):,} 行 / {out['date'].nunique()} 个截面，"
         f"总耗时 {time.time() - t0:.1f}s")
    return out


def _run_checks(out: pd.DataFrame, sections: pd.DataFrame,
                sd_ts: pd.Timestamp, ed_next: pd.Timestamp) -> None:
    """输出自检清单（AGENTS.md 2.6，列名按官方 wiki 契约 date/instrument/factor）。"""
    # 1. 列名
    if set(out.columns) != {"date", "instrument", "factor"}:
        raise RuntimeError(f"输出列名不符契约：{list(out.columns)}")
    # 2. 合法 30 分钟采样点
    hhmm = out["date"].dt.hour * 100 + out["date"].dt.minute
    if not hhmm.isin(LEGAL_OUTPUT_HHMM).all():
        bad = sorted(out.loc[~hhmm.isin(LEGAL_OUTPUT_HHMM), "date"].unique())
        raise RuntimeError(f"输出含非法采样时刻：{bad[:5]}")
    # 3. 时间落在评测区间内（非未来、不越界）
    if not ((out["date"] >= sd_ts.normalize()) & (out["date"] < ed_next)).all():
        raise RuntimeError("输出时间越出评测区间")
    # 4. score 有限（无 NaN/inf）
    if not np.isfinite(out["factor"].to_numpy(np.float64)).all():
        raise RuntimeError("factor 含非有限值")
    # 5. (date, instrument) 不重复
    if out.duplicated(["date", "instrument"]).any():
        raise RuntimeError("输出存在重复 (date, instrument)")
    # 6. 每截面覆盖率 >= 60%（不足仅告警：平台硬校验会判无效，需人工核查）
    denom = sections.groupby("date")["instrument"].nunique()
    num = out.groupby("date")["instrument"].nunique().reindex(denom.index)
    cov = (num / denom).dropna()
    if len(cov) and cov.min() < 0.6:
        _log(f"警告：有截面覆盖率低于 0.6（最小 {cov.min():.3f} @ {cov.idxmin()}），"
             f"平台校验要求每截面缺失率 <= 40%，请先核查再提交")
    if len(cov):
        _log(f"截面覆盖率：min={cov.min():.3f}, mean={cov.mean():.3f}")

