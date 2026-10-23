"""单元测试: _soft_within_day 必须与离线基准的矩阵实现逐位一致。

基准定义 (submissions/_tools/_bench_fine.py, v_z_soft):
    Z = 逐 (day, mod) 截面 z-score
    prev = Z[:, 0]
    for m in 1..M-1:
        d   = Z[:, m] - prev
        nxt = prev + sign(d) * max(0, |d| - theta)
        prev = nxt
"""
import importlib.util
import os
import sys
import types

import numpy as np
import pandas as pd

PKG = "cnn_3f_1023_soft_post"
D = os.path.abspath(f"submissions/{PKG}")


def _load():
    sys.path.insert(0, D)
    cwd = os.getcwd()
    os.chdir(D)
    sys.modules.setdefault("dai", types.ModuleType("dai"))
    sl = types.ModuleType("structlog")
    sl.get_logger = lambda *a, **k: types.SimpleNamespace(
        info=lambda *x, **y: None, warning=lambda *x, **y: None)
    sys.modules["structlog"] = sl
    spec = importlib.util.spec_from_file_location(
        "predmod", os.path.join(D, "Transformer_modelsave_predict.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    os.chdir(cwd)
    return m


def _reference(df, factor, theta):
    """基准矩阵实现 (与 _bench_fine.v_z_soft 同构)。"""
    d = df.copy()
    d["f"] = factor
    d["day"] = pd.to_datetime(d["date"]).dt.normalize()
    d["tod"] = (pd.to_datetime(d["date"]).dt.hour * 60
                + pd.to_datetime(d["date"]).dt.minute).astype("int64")
    g = d.groupby(["day", "tod"], sort=False)["f"]
    d["z"] = (d["f"] - g.transform("mean")) / g.transform("std").replace(0, 1.0)
    piv = d.pivot_table(index=["day", "instrument"], columns="tod",
                        values="z", aggfunc="first", sort=True)
    A = piv.to_numpy(dtype="float64", copy=True)
    prev = A[:, 0].copy()
    for j in range(1, A.shape[1]):
        cur = A[:, j]
        dd = cur - prev
        nxt = prev + np.sign(dd) * np.maximum(0.0, np.abs(dd) - theta)
        nxt = np.where(np.isnan(cur), prev, nxt)
        A[:, j] = nxt
        prev = nxt
    di, ii = piv.index.get_level_values(0), piv.index.get_level_values(1)
    ti = piv.columns.to_numpy()
    gidx, midx = np.meshgrid(np.arange(len(piv.index)), np.arange(len(ti)),
                             indexing="ij")
    flat = pd.DataFrame({
        "day": di.to_numpy()[gidx.ravel()],
        "instrument": ii.to_numpy()[gidx.ravel()],
        "tod": ti[midx.ravel()].astype("int64"),
        "sm": A.ravel(),
    })
    o = d.merge(flat, on=["day", "instrument", "tod"], how="left")
    return o.set_index("_i").loc[np.arange(len(df)), "sm"].to_numpy()


def _mk(n_days=2, n_inst=6, n_min=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for dy in range(n_days):
        for mi in range(n_min):
            for ins in range(n_inst):
                rows.append({
                    "date": (pd.Timestamp("2023-11-01") + pd.Timedelta(days=dy)
                             + pd.Timedelta(minutes=30 * mi)),
                    "instrument": ins,
                })
    df = pd.DataFrame(rows)
    df["_i"] = np.arange(len(df))
    f = rng.normal(size=len(df))
    return df, f


def main():
    m = _load()
    print(f"config: POST_MODE={m.POST_MODE} SOFT_THETA={m.SOFT_THETA}")

    ok = True

    # 1. 与基准逐位一致
    df, f = _mk()
    got = m._soft_within_day(df, f, 0.44)
    exp = _reference(df, f, 0.44)
    same = np.allclose(got, exp, atol=1e-10)
    ok &= same
    print(f"[1] 与基准矩阵实现逐位一致: {same}  "
          f"max|diff|={np.max(np.abs(got-exp)):.2e}")

    # 2. theta=0 恒等 (z 空间原样重建)
    z_id = _reference(df, f, 0.0)
    g0 = m._soft_within_day(df, f, 0.0)
    idn = np.allclose(g0, f, atol=1e-10)
    ok &= idn
    print(f"[2] theta=0 返回原值 (未做 z 变换): {idn}")

    # 3. 纯噪声下 theta 越大换手越低
    tos = []
    for th in [0.0, 0.3, 0.6, 1.2]:
        g = m._soft_within_day(df, f, th) if th > 0 else f
        dd = (df.assign(f=g).groupby(["instrument", "date"]).size())
        # 按 inst 日内序列算 |diff|
        tot = []
        for ins, sub in df.assign(g=g).groupby("instrument"):
            sub = sub.sort_values("date")
            tot.append(np.abs(np.diff(sub["g"].to_numpy())).sum())
        tos.append(np.mean(tot))
    mono = all(tos[i] >= tos[i + 1] - 1e-9 for i in range(len(tos) - 1))
    ok &= mono
    print(f"[3] 换手随 theta 单调不增: {mono}  值={[round(x,3) for x in tos]}")

    # 4. 因果性: 修改未来截面不得影响过去输出
    df2, f2 = _mk()
    g_a = m._soft_within_day(df2, f2, 0.44)
    # 找到最后一个截面, 大幅扰动它
    last_t = df2["date"].max()
    f3 = f2.copy()
    f3[df2["date"] == last_t] += 50.0
    g_b = m._soft_within_day(df2, f3, 0.44)
    past = df2["date"] < last_t
    causal = np.allclose(g_a[past.to_numpy()], g_b[past.to_numpy()], atol=1e-10)
    ok &= causal
    print(f"[4] 因果性 (未来不影响过去): {causal}")

    # 5. 跨日重置: 第 2 天结果不受第 1 天影响
    d1 = df2[df2["date"].dt.normalize() == df2["date"].min().normalize()]
    i1 = d1["_i"].to_numpy()
    g_full = m._soft_within_day(df2, f2, 0.44)
    g_day1 = m._soft_within_day(d1.reset_index(drop=True),
                               f2[i1], 0.44)
    reset = np.allclose(g_full[i1], g_day1, atol=1e-10)
    ok &= reset
    print(f"[5] 跨日重置 (第1天独立): {reset}")

    # 6. 乱序输入不改变结果 (行序无关)
    perm = np.random.default_rng(7).permutation(len(df2))
    dfp = df2.iloc[perm].reset_index(drop=True)
    g_p = m._soft_within_day(dfp, f2[perm], 0.44)
    unord = np.allclose(g_p, g_full[perm], atol=1e-10)
    ok &= unord
    print(f"[6] 乱序输入结果一致: {unord}")

    print("\n全部通过" if ok else "\n有失败项")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
