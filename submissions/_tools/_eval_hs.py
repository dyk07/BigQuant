"""对 highscore 模型的本地预测做指标评估 + 换手后处理扫描。

本地无成分股截面表, 用 bar30m 合成; 标签用 end_price/vwap-1 (平台口径,
HORIZON=30min), 与主项目一致。
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(".")
DECISION_HHMM = (930, 1000, 1030, 1100, 1130, 1330, 1400, 1430)
HORIZON = 30


def _load_bars(tag, start, end):
    d0 = os.path.join(ROOT, "e2e_data", f"cpt_jyc_2026_e2e_{tag}")
    parts = []
    for f in sorted(os.listdir(d0)):
        if not f.endswith(".feather"):
            continue
        d = pd.read_feather(os.path.join(d0, f))
        d["date"] = pd.to_datetime(d["date"])
        if d["date"].max() < pd.Timestamp(start) or d["date"].min() >= pd.Timestamp(end):
            del d
            continue
        parts.append(d)
        del d
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_labels(start, end):
    """label = end_price/vwap - 1, 用未来 HORIZON 分钟 bar 的成交额加权 vwap。"""
    b = _load_bars("bar1m", pd.Timestamp(start) - pd.Timedelta(days=1), end)
    b = b[["date", "instrument_id", "close", "volume", "amount"]].copy()
    b = b.rename(columns={"instrument_id": "instrument"})
    b["vwap"] = np.where(b["volume"] > 0, b["amount"] / b["volume"], b["close"])
    b = b.sort_values(["instrument", "date"], kind="mergesort").reset_index(drop=True)
    # 未来 HORIZON 分钟: 每只股票按时间前移
    g = b.groupby("instrument", sort=False)
    fut_amt = g["amount"].transform(
        lambda s: s.rolling(HORIZON, min_periods=1).sum().shift(-HORIZON))
    fut_vol = g["volume"].transform(
        lambda s: s.rolling(HORIZON, min_periods=1).sum().shift(-HORIZON))
    fut_end = g["close"].transform(lambda s: s.shift(-HORIZON))
    fv = np.where(fut_vol > 0, fut_amt / fut_vol, np.nan)
    b["y"] = fut_end / fv - 1.0
    b["hhmm"] = b["date"].dt.hour * 100 + b["date"].dt.minute
    lab = b[b["hhmm"].isin(DECISION_HHMM)][["date", "instrument", "y"]]
    lab = lab.replace([np.inf, -np.inf], np.nan).dropna(subset=["y"])
    return lab


def _pct(v):
    out = np.full(len(v), np.nan)
    ok = ~np.isnan(v)
    n = ok.sum()
    if n < 2:
        return out
    x = v[ok]
    o = np.argsort(x, kind="mergesort")
    r = np.empty(n)
    r[o] = np.arange(n)
    out[ok] = r / (n - 1)
    return out


def metrics(df):
    ics, sps = [], []
    for dt, g in df.groupby("date", sort=False):
        a = g["factor"].to_numpy(); y = g["y"].to_numpy()
        n = len(g)
        if n < 20:
            continue
        ra = _pct(a) * (n - 1); ry = _pct(y) * (n - 1)
        if ra.std() < 1e-9 or ry.std() < 1e-9:
            continue
        ics.append(np.corrcoef(ra, ry)[0, 1])
        k = max(1, int(n * 0.2))
        o = np.argsort(a, kind="mergesort")
        sps.append(y[o[-k:]].mean() - y[o[:k]].mean())
    ics = np.asarray(ics); sps = np.asarray(sps)
    if len(ics) < 2:
        return None
    # 换手: 同股相邻决策截面 pct_rank 漂移
    dv = df["date"].dt.normalize()
    df2 = df.assign(day=dv)
    tots = []
    for (dy, ins), g in df2.groupby(["day", "instrument"], sort=False):
        if len(g) < 2:
            continue
        g = g.sort_values("date")
        tots.append(np.abs(np.diff(_pct(g["factor"].to_numpy()))))
    to = float(np.nanmean(np.concatenate(tots))) if tots else float("nan")
    return dict(ic=ics.mean(), ic_ir=ics.mean() / ics.std(),
                sharpe=sps.mean() / sps.std(), turnover=to,
                n_sec=len(ics))


def soft_within_day(df, factor, theta):
    """与生产代码一致的日内因果 L1 软阈值。"""
    if theta is None or theta <= 0:
        return factor
    d = pd.DataFrame({
        "_i": np.arange(len(df)),
        "inst": df["instrument"].to_numpy(),
        "day": pd.to_datetime(df["date"]).dt.normalize().to_numpy(),
        "t": pd.to_datetime(df["date"]).to_numpy(),
        "f": np.asarray(factor, dtype="float64"),
    })
    d["tod"] = (pd.to_datetime(d["t"]).dt.hour * 60
                + pd.to_datetime(d["t"]).dt.minute).astype("int64")
    g = d.groupby(["day", "tod"], sort=False, observed=True)["f"]
    mu = g.transform("mean"); sd = g.transform("std")
    d["z"] = ((d["f"] - mu) / sd.where(sd > 1e-12, 1.0)).fillna(0.0)
    piv = d.pivot_table(index=["day", "inst"], columns="tod", values="z",
                        aggfunc="first", sort=True)
    M = piv.to_numpy(dtype="float64", copy=True)
    if M.shape[1] > 1:
        # 状态机: prev 为 NaN 表示该行尚未开始 (股票当日还没出现)。
        prev = np.full(M.shape[0], np.nan)
        for j in range(M.shape[1]):
            cur = M[:, j]
            start = np.isnan(prev) & ~np.isnan(cur)
            dl = np.where(start, 0.0, cur - prev)
            nxt = prev + np.sign(dl) * np.maximum(0.0, np.abs(dl) - theta)
            nxt = np.where(start, cur, nxt)
            nxt = np.where(np.isnan(cur), prev, nxt)
            M[:, j] = nxt
            prev = nxt
    di, ii = piv.index.get_level_values(0), piv.index.get_level_values(1)
    ti = piv.columns.to_numpy()
    gidx, midx = np.meshgrid(np.arange(len(piv.index)), np.arange(len(ti)),
                             indexing="ij")
    flat = pd.DataFrame({"day": di.to_numpy()[gidx.ravel()],
                         "inst": ii.to_numpy()[gidx.ravel()],
                         "tod": ti[midx.ravel()].astype("int64"),
                         "sm": M.ravel()})
    o = d.merge(flat, on=["day", "inst", "tod"], how="left")
    out = np.empty(len(df), dtype="float64")
    out[o["_i"].to_numpy()] = o["sm"].to_numpy()
    return out


def main():
    start, end = "2020-01-02", "2020-01-24"
    lab = build_labels(start, end)
    print(f"labels: {len(lab)} rows, {lab['date'].nunique()} sections")

    files = sorted(glob.glob("_preds_highscore_*.parquet"))
    if not files:
        print("no preds; run _hs_harness.py first")
        return
    print(f"\n{'model':<26} {'IC':>8} {'IC_IR':>7} {'Sharpe':>7} {'TO':>7}")
    print("-" * 62)
    for f in files:
        p = pd.read_parquet(f)
        p["date"] = pd.to_datetime(p["date"])
        p["instrument"] = p["instrument"].astype(lab["instrument"].dtype)
        m = p.merge(lab, on=["date", "instrument"], how="inner")
        if not len(m):
            print(f"{os.path.basename(f):<26} no overlap")
            continue
        r = metrics(m)
        name = os.path.basename(f).replace("_preds_", "").replace(".parquet", "")
        print(f"{name:<26} {r['ic']:>+8.4f} {r['ic_ir']:>7.3f} "
              f"{r['sharpe']:>7.3f} {r['turnover']:>7.4f}  (n={r['n_sec']})")

        for th in [0.3, 0.44, 0.6]:
            m2 = m.copy()
            m2["factor"] = soft_within_day(m2, m2["factor"].to_numpy(), th)
            r2 = metrics(m2)
            print(f"{'  +soft(' + str(th) + ')':<26} {r2['ic']:>+8.4f} "
                  f"{r2['ic_ir']:>7.3f} {r2['sharpe']:>7.3f} "
                  f"{r2['turnover']:>7.4f}")


if __name__ == "__main__":
    main()
