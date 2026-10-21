"""多截面聚合标签 可行性数据探查。

验证核心假设:
1. 不同 horizon 的 VWAP 收益标签之间的截面相关性 (30m vs 60m vs 90m vs 120m)。
2. 各 horizon 标签自身的"截面 IC 稳定性"—— 衡量标签噪声 (相邻截面 IC 的方差)。
3. 更长 horizon 是否天然更平滑 (换手更低) —— 相邻截面排名变化。

关键结论决定: 用聚合标签 (未来多个截面的平均收益) 训练, 能否同时提升
IC 稳定性 (sharpe) 并降低换手, 同时不偏离平台评测口径 (未来 30min VWAP)。
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
sys.path.insert(0, os.path.join(ROOT, "submissions", "cnn_3f_1020_newlabel_to0005"))

import config as CFG  # noqa: E402
import Transformer_modelsave_train as T  # noqa: E402

DATA = os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar1m")

# 用 3 个月数据做探查 (2023 年尾, 与验证集口径一致)
MONTHS = ["202310", "202311", "202312"]


def load():
    parts = []
    for m in MONTHS:
        p = os.path.join(DATA, f"{m}.0.feather")
        d = pd.read_feather(p)
        d["date"] = pd.to_datetime(d["date"])
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df = T.tag_time(df)
    return df


def main():
    df = load()
    print(f"[data] rows={len(df):,} months={MONTHS}")

    # 用 bar30m 口径 (与标签 horizon 一致)
    bars = T.resample_bars(df, 30)
    print(f"[resample] bar30m rows={len(bars):,}")

    grp = [bars["instrument_id"], bars["day"]]
    amt = bars["amount"].astype("float64")
    vol = bars["volume"].astype("float64")
    c_amt = amt.groupby(grp, sort=False).cumsum()
    c_vol = vol.groupby(grp, sort=False).cumsum()
    close = bars["close"].astype("float64")

    # 各 horizon 标签: label_h = close[t+h] / vwap[t:t+h] - 1
    labels = {}
    for h in [1, 2, 3, 4]:  # 30/60/90/120 min
        fwd_amt = c_amt.groupby(grp, sort=False).shift(-h) - c_amt
        fwd_vol = c_vol.groupby(grp, sort=False).shift(-h) - c_vol
        fwd_vwap = np.where(fwd_vol > 0, fwd_amt / fwd_vol, np.nan)
        end = close.groupby(grp, sort=False).shift(-h)
        lab = np.where((fwd_vwap > 0) & (end > 0),
                       end / np.where(fwd_vwap > 0, fwd_vwap, 1.0) - 1.0, np.nan)
        labels[h] = lab

    # 有效样本 (所有 horizon 都有效)
    valid = np.isfinite(labels[1]) & np.isfinite(labels[2]) & \
            np.isfinite(labels[3]) & np.isfinite(labels[4])
    print(f"[valid] 全部 horizon 有效样本 = {valid.sum():,} / {len(bars):,}")

    # 截面键
    inst = bars["instrument_id"].to_numpy()
    day = pd.to_datetime(bars["day"]).to_numpy().astype("datetime64[D]").astype("int64")
    mod = bars["minute_of_day"].to_numpy()

    def sec_ic(y, x, mask):
        """截面 IC: 同 (day,mod) 内 corr"""
        df = pd.DataFrame({"x": x[mask], "y": y[mask], "day": day[mask],
                           "mod": mod[mask]})
        ics = []
        for _, g in df.dropna().groupby(["day", "mod"], sort=False):
            if len(g) < 10:
                continue
            xr = g["x"].rank(); yr = g["y"].rank()
            ics.append(np.corrcoef(xr, yr)[0, 1])
        return ics

    print("\n=== 标签间截面相关性 (有效样本上) ===")
    print(f"{'horizon':>8}" + "".join(f"{h*30:>10}m" for h in [1, 2, 3, 4]))
    for ha in [1, 2, 3, 4]:
        row = f"{ha*30:>6}m"
        for hb in [1, 2, 3, 4]:
            m = valid & np.isfinite(labels[ha]) & np.isfinite(labels[hb])
            c = np.corrcoef(labels[ha][m], labels[hb][m])[0, 1]
            row += f"{c:>10.3f}"
        print(row)

    print("\n=== 各 horizon 标签的截面 IC 稳定性 (用 30m 标签做基准) ===")
    print(f"{'label_h':>8}  IC_mean   IC_std   IC_IR(=mean/std)  N截面")
    for h in [1, 2, 3, 4]:
        m = np.isfinite(labels[1]) & np.isfinite(labels[h])
        ics = sec_ic(labels[1], labels[h], m)
        ics = np.array(ics)
        print(f"{h*30:>6}m   {ics.mean():+.4f}  {ics.std():.4f}  "
              f"{ics.mean()/ics.std():+.3f}  {len(ics)}")

    print("\n=== 相邻截面排名变化 (换手代理, 越低越平滑) ===")
    print(f"{'label_h':>8}  平均|rank漂移|")
    for h in [1, 2, 3, 4]:
        m = np.isfinite(labels[h])
        d = pd.DataFrame({"y": labels[h][m], "inst": inst[m], "day": day[m],
                          "mod": mod[m]})
        d["r"] = d.groupby(["day", "mod"], sort=False)["y"].rank(pct=True)
        diffs = []
        for _, g in d.groupby(["inst", "day"], sort=False):
            if len(g) < 2:
                continue
            g = g.sort_values("mod")
            diffs.extend(np.abs(np.diff(g["r"].to_numpy())).tolist())
        print(f"{h*30:>6}m   {np.mean(diffs):.4f}")


if __name__ == "__main__":
    main()
