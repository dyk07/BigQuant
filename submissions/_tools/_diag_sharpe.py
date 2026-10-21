"""多空组合 Sharpe 分解诊断。

加载最佳模型 (cnn_3f_1020_newlabel_to0005) 权重, 在验证集 (2023-11~12) 上
生成预测分数, 构建多空组合 (top/bottom quantile), 分解 Sharpe 低的根因:

1. IC 截面间波动 (IC_IR) —— 预测能力是否稳定;
2. 头部/尾部排序准确性 —— 多空两端选股是否正确;
3. 因子分布形状 —— 是否厚尾/离群值主导组合。

多空组合构造 (对齐平台 vwap_return 口径):
  每截面按 factor 排序, long top q%, short bottom q%, 组合收益 =
  mean(label[top]) - mean(label[bottom])。Sharpe = mean/std。
"""
import faulthandler
import os
import sys
import types

import numpy as np
import pandas as pd

faulthandler.enable()

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
PKG = "cnn_3f_1020_newlabel_to0005"
PKG_DIR = os.path.join(ROOT, "submissions", PKG)
os.chdir(PKG_DIR)
sys.path.insert(0, PKG_DIR)

DATA = os.path.join(ROOT, "e2e_data", "cpt_jyc_2026_e2e_bar1m")
VAL_MONTHS = ["202311", "202312"]

# 伪造 dai (与 smoke/full_train 一致)
_ALL_MONTHS = [os.path.splitext(f)[0].split(".")[0]
               for f in sorted(os.listdir(DATA)) if f.endswith(".feather")]


def _query(sql, filters=None):
    assert filters and "date" in filters
    lo, hi = pd.Timestamp(filters["date"][0]), pd.Timestamp(filters["date"][1])
    parts = []
    for m in _ALL_MONTHS:
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
import config as CFG  # noqa: E402
import Transformer_modelsave_train as T  # noqa: E402


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = T.load_model(T.MODEL_PATH, map_location=device)
    freqs = ckpt["freqs"]
    seq_lens = ckpt["seq_lens"]
    stats = {f: (np.asarray(ckpt["mean"][f], np.float32),
                 np.asarray(ckpt["std"][f], np.float32)) for f in freqs}
    model = T.build_model(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"[diag] 加载模型 {PKG}  freqs={freqs}  label_direction={ckpt.get('label_direction')}")

    # 构造验证集 (复用训练统计量, 无泄漏)
    va = T._accumulate("cpt_jyc_2026_e2e_bar1m", VAL_MONTHS, freqs, seq_lens,
                       ckpt["aligned_minutes"], stats=stats)
    keys = va["keys"]
    y = va["y"]
    n = len(y)
    print(f"[diag] 验证集样本 N={n:,}")

    # 生成预测
    preds = T._predict_all(model, va["feats"], va["anchors"], seq_lens,
                           freqs, keys, device)
    print(f"[diag] 预测完成 preds={preds.shape}")

    # =====================================================================
    # 组装截面数据
    # =====================================================================
    df = pd.DataFrame({
        "inst": keys[:, 0], "day": keys[:, 1], "mod": keys[:, 2],
        "p": preds, "y": y,
    })
    df = df[np.isfinite(df["y"])].copy()
    print(f"[diag] 有效样本 (finite label) = {len(df):,}")

    # =====================================================================
    # 1. 因子分布形状
    # =====================================================================
    print("\n=== 1. 因子分布形状 ===")
    p = df["p"].to_numpy()
    print(f"  mean={p.mean():.4f}  std={p.std():.4f}  "
          f"skew={pd.Series(p).skew():.3f}  kurt={pd.Series(p).kurt():.3f}")
    q = np.quantile(p, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(f"  分位 [1%,5%,25%,50%,75%,95%,99%] = {np.round(q, 4)}")
    # 离群值: |z|>5 的比例
    z = (p - p.mean()) / p.std()
    print(f"  |z|>3 比例={np.mean(np.abs(z)>3):.4f}  |z|>5 比例={np.mean(np.abs(z)>5):.4f}")

    # =====================================================================
    # 2. 截面 IC 及其稳定性
    # =====================================================================
    print("\n=== 2. 截面 IC 稳定性 ===")
    ics = []
    for _, g in df.groupby(["day", "mod"], sort=False):
        if len(g) < 10:
            continue
        pr = g["p"].rank(); yr = g["y"].rank()
        ics.append(np.corrcoef(pr, yr)[0, 1])
    ics = np.array(ics)
    print(f"  截面数={len(ics)}  IC_mean={ics.mean():+.4f}  "
          f"IC_std={ics.std():.4f}  IC_IR={ics.mean()/ics.std():+.3f}")
    print(f"  IC 分位 [5%,25%,50%,75%,95%] = {np.round(np.quantile(ics,[.05,.25,.5,.75,.95]),4)}")
    print(f"  负 IC 截面比例 = {np.mean(ics<0):.3f}")

    # =====================================================================
    # 3. 多空组合 Sharpe (top/bottom quantile)
    # =====================================================================
    print("\n=== 3. 多空组合 Sharpe (各 quantile) ===")
    for q in [0.05, 0.10, 0.20]:
        spreads = []
        for _, g in df.groupby(["day", "mod"], sort=False):
            if len(g) < 20:
                continue
            k = max(1, int(len(g) * q))
            gs = g.sort_values("p")
            bottom = gs["y"].iloc[:k].mean()
            top = gs["y"].iloc[-k:].mean()
            spreads.append(top - bottom)
        spreads = np.array(spreads)
        sharpe = spreads.mean() / spreads.std() if spreads.std() > 0 else 0
        print(f"  q={q:.2f}  N截面={len(spreads)}  "
              f"spread_mean={spreads.mean():+.5f}  spread_std={spreads.std():.5f}  "
              f"Sharpe={sharpe:+.3f}")

    # =====================================================================
    # 4. 头部/尾部排序准确性 —— 用 top/bottom 的实际收益 vs 完美排序收益
    # =====================================================================
    print("\n=== 4. 头尾排序准确性 (q=0.20) ===")
    q = 0.20
    # 完美排序: 按真实 y 排序的 top/bottom 收益 (理论上限)
    perfect_spreads = []
    actual_spreads = []
    for _, g in df.groupby(["day", "mod"], sort=False):
        if len(g) < 20:
            continue
        k = max(1, int(len(g) * q))
        # 实际 (按预测排序)
        gs = g.sort_values("p")
        actual_spreads.append(gs["y"].iloc[-k:].mean() - gs["y"].iloc[:k].mean())
        # 完美 (按真实 y 排序)
        gy = g.sort_values("y")
        perfect_spreads.append(gy["y"].iloc[-k:].mean() - gy["y"].iloc[:k].mean())
    actual = np.array(actual_spreads)
    perfect = np.array(perfect_spreads)
    print(f"  实际 top-bottom 收益 mean={actual.mean():+.5f}  "
          f"std={actual.std():.5f}  Sharpe={actual.mean()/actual.std():+.3f}")
    print(f"  完美 top-bottom 收益 mean={perfect.mean():+.5f}  "
          f"std={perfect.std():.5f}  Sharpe={perfect.mean()/perfect.std():+.3f}")
    # 选股命中率: 实际 top k 里有多少落在真实 top k
    print(f"  实际 spread / 完美 spread = {actual.mean()/perfect.mean():.3f}")

    # =====================================================================
    # 5. top/bottom 组内收益分布 (看是否两端噪声大)
    # =====================================================================
    print("\n=== 5. 各组收益 (q=0.20, 组内均值跨截面平均) ===")
    q = 0.20
    group_ret = {g_: [] for g_ in ["bottom", "mid", "top"]}
    for _, g in df.groupby(["day", "mod"], sort=False):
        if len(g) < 20:
            continue
        k = max(1, int(len(g) * q))
        gs = g.sort_values("p")
        group_ret["bottom"].append(gs["y"].iloc[:k].mean())
        group_ret["mid"].append(gs["y"].iloc[k:-k].mean())
        group_ret["top"].append(gs["y"].iloc[-k:].mean())
    for g_ in ["bottom", "mid", "top"]:
        v = np.array(group_ret[g_])
        print(f"  {g_:<6}  mean={v.mean():+.5f}  std={v.std():.5f}")

    # 完美排序下各组收益 (对比用)
    print("  --- 完美排序 (理论上限) ---")
    group_ret = {g_: [] for g_ in ["bottom", "mid", "top"]}
    for _, g in df.groupby(["day", "mod"], sort=False):
        if len(g) < 20:
            continue
        k = max(1, int(len(g) * q))
        gs = g.sort_values("y")
        group_ret["bottom"].append(gs["y"].iloc[:k].mean())
        group_ret["mid"].append(gs["y"].iloc[k:-k].mean())
        group_ret["top"].append(gs["y"].iloc[-k:].mean())
    for g_ in ["bottom", "mid", "top"]:
        v = np.array(group_ret[g_])
        print(f"  {g_:<6}  mean={v.mean():+.5f}  std={v.std():.5f}")


if __name__ == "__main__":
    main()
