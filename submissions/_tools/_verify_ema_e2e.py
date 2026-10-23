"""端到端验证: 真实权重 + 真实数据, 对比 EMA 开关对 IC/换手的影响。

复用训练侧 _accumulate/_predict_all 得到与 main() 一致的预测, 再套用
predict 脚本里的 _ema_within_day, 验证实际效果。
"""
import faulthandler
import os
import sys
import types

import numpy as np
import pandas as pd

faulthandler.enable()

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
PKG = "cnn_3f_1023_ema_post"
PKG_DIR = os.path.join(ROOT, "submissions", PKG)
os.chdir(PKG_DIR)
sys.path.insert(0, PKG_DIR)

DATA = os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar1m")
MONTHS = ["202311", "202312"]
_ALL = [os.path.splitext(f)[0].split(".")[0]
        for f in sorted(os.listdir(DATA)) if f.endswith(".feather")]


def _query(sql, filters=None):
    lo, hi = pd.Timestamp(filters["date"][0]), pd.Timestamp(filters["date"][1])
    parts = []
    for m in _ALL:
        p = os.path.join(DATA, f"{m}.0.feather")
        if not os.path.exists(p):
            continue
        ms = pd.Period(m, "M")
        if ms.end_time < lo or ms.start_time > hi:
            continue
        d = pd.read_feather(p)
        d["date"] = pd.to_datetime(d["date"])
        parts.append(d[(d["date"] >= lo) & (d["date"] <= hi)])
        del d
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return types.SimpleNamespace(df=lambda: df)


fake_dai = types.ModuleType("dai")
fake_dai.query = _query
sys.modules["dai"] = fake_dai

import torch  # noqa: E402
import Transformer_modelsave_train as T  # noqa: E402

# 复用 predict 脚本里的 _ema_within_day (只取该函数, 避免 import 触发 dai 依赖)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "_ema_mod", os.path.join(PKG_DIR, "Transformer_modelsave_predict.py"))


def ema_within_day(df, factor, alpha):
    """与 predict 脚本中实现一致。"""
    if alpha is None or alpha >= 1.0 or len(df) == 0:
        return factor
    d = pd.DataFrame({
        "_i": np.arange(len(df)),
        "inst": df["instrument"].to_numpy(),
        "day": pd.to_datetime(df["date"]).dt.normalize().to_numpy(),
        "t": pd.to_datetime(df["date"]).to_numpy(),
        "f": np.asarray(factor, dtype="float64"),
    })
    d = d.sort_values(["inst", "day", "t"], kind="mergesort")
    sm = d.groupby(["inst", "day"], sort=False, observed=True)["f"].transform(
        lambda x: x.ewm(alpha=alpha, adjust=False).mean())
    out = np.empty(len(df), dtype="float64")
    out[d["_i"].to_numpy()] = sm.to_numpy()
    return out


def metrics(df, f):
    d = pd.DataFrame({"inst": df["inst"], "day": df["day"], "mod": df["mod"],
                      "f": np.asarray(f), "y": df["y"]})
    ics = []
    for _, g in d.groupby(["day", "mod"], sort=False):
        if len(g) >= 10:
            ics.append(np.corrcoef(g["f"].rank(), g["y"].rank())[0, 1])
    ics = np.array(ics)
    icm = ics.mean(); icir = icm / ics.std() if ics.std() > 0 else 0
    sp = []
    for _, g in d.groupby(["day", "mod"], sort=False):
        if len(g) < 20:
            continue
        k = max(1, int(len(g) * 0.20))
        gs = g.sort_values("f")
        sp.append(gs["y"].iloc[-k:].mean() - gs["y"].iloc[:k].mean())
    sp = np.array(sp)
    sh = sp.mean() / sp.std() if sp.std() > 0 else 0
    d["pr"] = d.groupby(["day", "mod"], sort=False)["f"].rank(pct=True)
    diffs = []
    for _, g in d.groupby(["inst", "day"], sort=False):
        if len(g) < 2:
            continue
        g = g.sort_values("mod")
        diffs.extend(np.abs(np.diff(g["pr"].to_numpy())).tolist())
    return icm, icir, sh, float(np.mean(diffs))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = T.load_model(T.MODEL_PATH, map_location=device)
    freqs = ckpt["freqs"]; seq_lens = ckpt["seq_lens"]
    stats = {f: (np.asarray(ckpt["mean"][f], np.float32),
                 np.asarray(ckpt["std"][f], np.float32)) for f in freqs}
    model = T.build_model(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    va = T._accumulate("cpt_jyc_2026_e2e_bar1m", MONTHS, freqs, seq_lens,
                       ckpt["aligned_minutes"], stats=stats)
    preds = T._predict_all(model, va["feats"], va["anchors"], seq_lens,
                           freqs, va["keys"], device)
    # 复现 predict 脚本的 DataFrame 结构 (date/instrument/minute_of_day)
    keys = va["keys"]
    out_mod = keys[:, 2].copy()
    out_mod[out_mod == T.AUCTION_MINUTE] = T.OPENING_SECTION
    idx_df = pd.DataFrame({
        "date": (pd.to_datetime(keys[:, 1], unit="D")
                 + pd.to_timedelta(out_mod, unit="m")),
        "instrument": keys[:, 0].astype("int64"),
        "minute_of_day": out_mod.astype("int64"),
    })
    df = pd.DataFrame({"inst": keys[:, 0], "day": keys[:, 1], "mod": keys[:, 2],
                       "y": va["y"]})

    print(f"{'配置':<18} {'IC_mean':>8} {'IC_IR':>8} {'Sharpe':>8} {'turnover':>9}")
    print("=" * 56)
    base = None
    for alpha in [None, 0.35, 0.4, 0.45, 0.5, 0.6]:
        f = ema_within_day(idx_df, preds, alpha)
        m = metrics(df, f)
        if base is None:
            base = m
        lab = "无后处理" if alpha is None else f"EMA α={alpha}"
        print(f"{lab:<18} {m[0]:>+8.4f} {m[1]:>8.3f} {m[2]:>8.3f} {m[3]:>9.4f}")


if __name__ == "__main__":
    main()
