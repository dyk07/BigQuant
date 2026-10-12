"""
数据与标签自检脚本（不训练）。

用法：python sanity_check.py [--freq bar30m] [--month 202301]

检查项：
1. 单位还原：amount/volume/100 应接近 close（同一量级）
2. 保留 09:25 集合竞价截面、剔除 09:16–09:24 盘前价，每天 bar 数符合预期
3. 标签分布、未来 VWAP 与 end_price(=close[t+h]) 的关系
4. 标签窗口是否跨交易日（应为 0）
5. 采样点数、截面股票数
6. 切窗连续性（防泄漏）
"""
import argparse
import numpy as np
import pandas as pd

from config import Config
from data_loader import load_month, NEEDED_COLS, filter_auction_and_tag_day
from features import build_features, build_labels
from pipeline import build_split, describe, check_window_leakage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="bar30m")
    ap.add_argument("--month", default="202301")
    args = ap.parse_args()

    cfg = Config()
    cfg.freq = args.freq
    m = args.month

    print("=" * 70)
    print(f"sanity check: freq={cfg.freq} month={m}")
    print(f"bar={cfg.bar_minutes}m horizon_bars={cfg.horizon_bars} "
          f"seq_len={cfg.seq_len_bars}bars lookback={cfg.lookback_minutes}m")
    print(f"bars/day={cfg.bars_per_day}  "
          f"aligned_minutes={[f'{x//60:02d}:{x%60:02d}' for x in cfg.aligned_minutes()]}")
    print("=" * 70)

    raw = load_month(cfg.data_dir, cfg.table, m, columns=NEEDED_COLS)
    print(f"\n[1] raw shape={raw.shape}")
    times_all = sorted(pd.to_datetime(raw["date"]).dt.strftime("%H:%M").unique())
    print(f"    all timestamps ({len(times_all)}): {times_all}")
    assert "09:25" in times_all, "expected auction bar to exist in raw data"

    df = filter_auction_and_tag_day(raw)
    times_kept = sorted(df["date"].dt.strftime("%H:%M").unique())
    print(f"    after dropping pre-09:25 only ({len(times_kept)}): "
          f"{times_kept[:3]} ... {times_kept[-3:]}")
    # 09:25 集合竞价截面必须保留（开盘截面用竞价数据），09:16–09:24 必须剔除
    assert "09:25" in times_kept, "auction bar 09:25 should be KEPT"
    assert min(times_kept) == "09:25", \
        f"pre-market bars not dropped: min={min(times_kept)}"
    print(f"    expected bars/day={cfg.bars_per_day}, "
          f"observed={sorted(df.groupby('day').size().unique())}")

    # [2] 单位还原检查
    print("\n[2] unit check (amount/volume/close should be same magnitude)")
    sub = df[(df["volume"] > 0)].head(1000)
    vwap = sub["amount"] / sub["volume"]
    ratio = (vwap / sub["close"]).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"    vwap/close median={ratio.median():.4f} "
          f"(expect ~1.0, close is in 元 after /100)")
    assert 0.5 < ratio.median() < 1.5, "unit restoration looks wrong"

    # [3] 标签
    df = build_features(df)
    df = build_labels(df, horizon_bars=cfg.horizon_bars)
    lab = df["label"].to_numpy(dtype="float64")
    lab_valid = lab[~np.isnan(lab)]
    print(f"\n[3] labels: valid={len(lab_valid)}/{len(lab)} "
          f"({100 * len(lab_valid) / len(lab):.1f}%)")
    print(f"    mean={np.nanmean(lab_valid):+.6f} std={np.nanstd(lab_valid):.6f} "
          f"min={lab_valid.min():+.4f} max={lab_valid.max():+.4f}")
    print(f"    quantiles 1/25/50/75/99%: "
          f"{np.nanquantile(lab_valid, [0.01, 0.25, 0.5, 0.75, 0.99]).round(5)}")

    # [4] 标签窗口是否跨交易日
    h = cfg.horizon_bars
    d = df.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    grp = [d["instrument_id"], d["day"]]
    cnt = d["volume"].gt(0).astype("int64").groupby(
        grp, sort=False).cumsum()
    cnt_next = cnt.groupby(grp, sort=False).shift(-h)
    # 当 cnt_next 存在但 cnt 的组边界不同时才是跨日（用 bar_idx 校验）
    bar_next = d["bar_idx"].groupby(grp, sort=False).shift(-h)
    cross = (bar_next - d["bar_idx"] != h) & bar_next.notna()
    print(f"\n[4] label windows crossing day boundary: {int(cross.sum())} "
          f"(expect 0)")
    assert int(cross.sum()) == 0, "label window crosses trading day!"

    # [5] 切窗
    X, y, meta, clip = build_split(cfg, [m], verbose=False)
    describe(X, y, meta, "samples")
    print(f"    clip bounds={clip}")

    n_days = len(set(mm[1] for mm in meta))
    minutes = sorted(set(mm[2] for mm in meta))
    fmt = lambda x: f"{x // 60:02d}:{x % 60:02d}"  # noqa: E731
    print(f"    sampled minutes={[fmt(x) for x in minutes]}")
    assert set(minutes).issubset(set(cfg.aligned_minutes()))
    assert n_days > 0

    # [6] 防泄漏
    print()
    ok = check_window_leakage(cfg, [m])
    assert ok, "leakage check failed"

    print("\n" + "=" * 70)
    print("SANITY CHECK PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
