"""端到端验证: 用真实预测 + 生产代码里的 _soft_within_day/_ema_within_day。

缓存 npz 来自基线权重 cnn_3f_1020_newlabel_to0005 (与 soft_post 包权重相同),
因此这里等价于验证生产 predict 脚本的后处理链路。
"""
import importlib.util
import os
import sys
import types

import numpy as np
import pandas as pd

NPZ = "submissions/_tools/_preds_val4m.npz"
PKG = "cnn_3f_1023_soft_post"
D = os.path.abspath(f"submissions/{PKG}")


def load_mod():
    sys.path.insert(0, D)
    cwd = os.getcwd()
    os.chdir(D)
    sys.modules.setdefault("dai", types.ModuleType("dai"))
    sl = types.ModuleType("structlog")
    sl.get_logger = lambda *a, **k: types.SimpleNamespace(
        info=lambda *x, **y: None, warning=lambda *x, **y: None)
    sys.modules["structlog"] = sl
    spec = importlib.util.spec_from_file_location(
        "pm", os.path.join(D, "Transformer_modelsave_predict.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    os.chdir(cwd)
    return m


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


def metrics(sol):
    """sol: DataFrame[day, inst, mod, y, pred] -> (IC, IC_IR, Sharpe, TO)"""
    ics, sps = [], []
    for (dy, md), g in sol.groupby(["day", "mod"], sort=False):
        a = g["pred"].to_numpy(); y = g["y"].to_numpy()
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
    tots = []
    for (dy, ins), g in sol.groupby(["day", "inst"], sort=False):
        if len(g) < 2:
            continue
        g = g.sort_values("mod")
        tots.append(np.abs(np.diff(_pct(g["pred"].to_numpy()))))
    flat = np.concatenate(tots)
    return (ics.mean(), ics.mean() / ics.std(), sps.mean() / sps.std(),
            float(np.nanmean(flat)))


def main():
    m = load_mod()
    print(f"POST_MODE={m.POST_MODE} SOFT_THETA={m.SOFT_THETA} "
          f"EMA_ALPHA={m.EMA_ALPHA}")

    z = np.load(NPZ)
    base = pd.DataFrame({
        "day": z["day"].astype("int64"),
        "inst": z["inst"].astype("int64"),
        "mod": z["mod"].astype("int64"),
        "y": z["y"].astype("float64"),
        "pred": z["pred"].astype("float64"),
    })
    base["instrument"] = base["inst"]
    base["date"] = (pd.to_datetime(base["day"], unit="D")
                    + pd.to_timedelta(base["mod"], unit="m"))

    print(f"\n数据: {len(base)} 行, {base['day'].nunique()} 天, "
          f"{base['inst'].nunique()} 股\n")

    b = metrics(base)
    print(f"{'配置':<24} {'IC':>8} {'IC_IR':>7} {'Sharpe':>7} {'TO':>7}  "
          f"{'dIC%':>7} {'dTO%':>7}")
    print("-" * 74)
    print(f"{'baseline 无后处理':<24} {b[0]:>+8.4f} {b[1]:>7.3f} "
          f"{b[2]:>7.3f} {b[3]:>7.4f}       -       -")

    for th in [0.30, 0.38, 0.42, 0.44, 0.46, 0.50]:
        df = base.copy()
        df["pred"] = m._soft_within_day(df, df["pred"].to_numpy(), th)
        r = metrics(df)
        print(f"{'soft theta=' + str(th):<24} {r[0]:>+8.4f} {r[1]:>7.3f} "
              f"{r[2]:>7.3f} {r[3]:>7.4f}  "
              f"{(r[0]/b[0]-1)*100:>+7.1f} {(r[3]/b[3]-1)*100:>+7.1f}")

    for a in [0.45, 0.50]:
        df = base.copy()
        df["pred"] = m._ema_within_day(df, df["pred"].to_numpy(), a)
        r = metrics(df)
        print(f"{'EMA alpha=' + str(a):<24} {r[0]:>+8.4f} {r[1]:>7.3f} "
              f"{r[2]:>7.3f} {r[3]:>7.4f}  "
              f"{(r[0]/b[0]-1)*100:>+7.1f} {(r[3]/b[3]-1)*100:>+7.1f}")


if __name__ == "__main__":
    main()
