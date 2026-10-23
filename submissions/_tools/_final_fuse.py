"""最终融合实验: 自有模型 + highscore_merged。

关键: 两个模型的 instrument 编码不同。
  ours   = md5(str(int_id))[:15] 的 hex -> int (即 _encode_instrument 的字符串分支)
  merged = 原始 int_id 的字符串
通过 md5 反查建立映射。

样本期 2024-07~12: 完全在自有模型训练范围 (2020-2023) 之外。
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "submissions/_tools")
from _cross_fuse_test import build_labels, metrics  # noqa: E402


def ours_to_raw(df, raw_ids):
    """把 ours 的 md5 编码还原成原始 int_id 字符串。"""
    m = {str(int(hashlib.md5(str(r).encode()).hexdigest()[:15], 16)): str(r)
         for r in raw_ids}
    d = df.copy()
    d["instrument"] = d["instrument"].astype(str).map(m)
    return d.dropna(subset=["instrument"])


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


def soft_within_day(df, factor, theta):
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
    gi, mi = np.meshgrid(np.arange(len(piv.index)), np.arange(len(ti)), indexing="ij")
    flat = pd.DataFrame({"day": di.to_numpy()[gi.ravel()],
                         "inst": ii.to_numpy()[gi.ravel()],
                         "tod": ti[mi.ravel()].astype("int64"),
                         "sm": M.ravel()})
    o = d.merge(flat, on=["day", "inst", "tod"], how="left")
    out = np.empty(len(df), dtype="float64")
    out[o["_i"].to_numpy()] = o["sm"].to_numpy()
    return out


def fuse(o, m, w):
    a = o.rename(columns={"factor": "po"})
    b = m.rename(columns={"factor": "pm"})
    j = a.merge(b, on=["date", "instrument"], how="inner")
    ra = j.groupby("date", sort=False)["po"].rank(pct=True)
    rb = j.groupby("date", sort=False)["pm"].rank(pct=True)
    j["factor"] = w * ra + (1 - w) * rb
    return j[["date", "instrument", "factor"]]


def main():
    lab = build_labels()
    lab["instrument"] = lab["instrument"].astype(str)
    o = pd.read_parquet("_preds_ours_2024H2.parquet")
    m = pd.read_parquet("_preds_merged_2024H2.parquet")
    for x in (o, m):
        x["date"] = pd.to_datetime(x["date"])
    raw = sorted(set(m["instrument"].astype(str).unique()))
    o = ours_to_raw(o, raw)
    print(f"编码映射后 ours 行数: {len(o)}, 未映射: "
          f"{pd.read_parquet('_preds_ours_2024H2.parquet')['instrument'].nunique() - o['instrument'].nunique()} 个 id 未匹配")
    m["instrument"] = m["instrument"].astype(str)

    lo = lab.merge(o, on=["date", "instrument"], how="inner")
    lm = lab.merge(m, on=["date", "instrument"], how="inner")
    print(f"对齐样本: ours={len(lo)}, merged={len(lm)}")
    if len(lo) < 1000:
        print("!! 对齐样本过少, 编码映射失败")
        return

    ro, rm = metrics(lo), metrics(lm)
    print(f"\n{'model':<20} {'IC':>8} {'IC_IR':>7} {'Sharpe':>7} {'TO':>7} {'n':>5}")
    print("-" * 62)
    for nm, r in (("ours(本地基线)", ro), ("highscore_merged", rm)):
        print(f"{nm:<20} {r['ic']:>+8.4f} {r['ic_ir']:>7.3f} "
              f"{r['sharpe']:>7.3f} {r['turnover']:>7.4f} {r['n']:>5}")

    j = fuse(o, m, 0.5)
    ra = j.groupby("date", sort=False)["factor"].rank(pct=True)
    # 相关性需在原始两列上算
    jj = (o.rename(columns={"factor": "po"})
          .merge(m.rename(columns={"factor": "pm"}), on=["date", "instrument"]))
    xa = jj.groupby("date", sort=False)["po"].rank(pct=True)
    xb = jj.groupby("date", sort=False)["pm"].rank(pct=True)
    per = jj.assign(a=xa, b=xb).groupby("date").apply(
        lambda g: np.corrcoef(g["a"], g["b"])[0, 1] if len(g) > 20 else np.nan)
    print(f"\n【核心1】ours vs merged 逐截面 rank 相关: "
          f"{np.corrcoef(xa, xb)[0,1]:.4f}, 各截面 mean={per.mean():.3f} "
          f"min={per.min():.3f} max={per.max():.3f}")
    print("  参考基准: merged vs benchmark = 0.83 (融合净增益 0)")

    print(f"\n【核心2】融合权重扫描 (w=ours 权重)")
    print(f"{'w':>5} {'IC':>8} {'IC_IR':>7} {'Sharpe':>7} {'TO':>7}")
    best_ic, best_w, rows = -9, None, []
    for w in np.arange(0, 1.01, 0.125):
        f = fuse(o, m, w).merge(lab, on=["date", "instrument"], how="inner")
        r = metrics(f)
        rows.append((w, r))
        print(f"{w:>5.3f} {r['ic']:>+8.4f} {r['ic_ir']:>7.3f} "
              f"{r['sharpe']:>7.3f} {r['turnover']:>7.4f}")
        if r["ic"] > best_ic:
            best_ic, best_w = r["ic"], w
    print(f"\nIC 最优: w={best_w:.3f}, IC={best_ic:+.4f}; "
          f"纯 merged(w=0) IC={rows[0][1]['ic']:+.4f}; "
          f"纯 ours(w=1) IC={rows[-1][1]['ic']:+.4f}")
    print(f"融合净增益 vs 纯 merged: {best_ic - rows[0][1]['ic']:+.4f}")

    print(f"\n【核心3】融合后再套软阈值 (选 w=0.25 附近, 降换手)")
    for w in [0.25, 0.375]:
        base_f = fuse(o, m, w)
        mb = base_f.merge(lab, on=["date", "instrument"], how="inner")
        rb = metrics(mb)
        print(f"  w={w:.3f} 融合: IC={rb['ic']:+.4f} IC_IR={rb['ic_ir']:.3f} "
              f"Sharpe={rb['sharpe']:.3f} TO={rb['turnover']:.4f}")
        for th in [0.2, 0.25, 0.3]:
            d = mb.copy()
            d["factor"] = soft_within_day(d, d["factor"].to_numpy(), th)
            r = metrics(d)
            print(f"    +soft({th}): IC={r['ic']:+.4f} IC_IR={r['ic_ir']:.3f} "
                  f"Sharpe={r['sharpe']:.3f} TO={r['turnover']:.4f}  "
                  f"(dIC {(r['ic']/rb['ic']-1)*100:+.1f}%, "
                  f"dIR {(r['ic_ir']/rb['ic_ir']-1)*100:+.1f}%, "
                  f"dTO {(r['turnover']/rb['turnover']-1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
