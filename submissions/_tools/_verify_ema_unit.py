"""单元验证 _ema_within_day 的因果性、日内重置、正确性。"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
sys.path.insert(0, os.path.join(ROOT, "submissions", "cnn_3f_1023_ema_post"))

# 直接复制函数 (避免 import 时触发 dai/torch)
def _ema_within_day(df, factor, alpha):
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
    smoothed = d.groupby(["inst", "day"], sort=False, observed=True)["f"].transform(
        lambda x: x.ewm(alpha=alpha, adjust=False).mean())
    out = np.empty(len(df), dtype="float64")
    out[d["_i"].to_numpy()] = smoothed.to_numpy()
    return out


def test_basic():
    """单股票单日: 验证 EMA 数值正确。"""
    df = pd.DataFrame({
        "instrument": ["A"] * 4,
        "date": pd.to_datetime(["2023-11-01 10:00", "2023-11-01 10:30",
                                "2023-11-01 11:00", "2023-11-01 13:30"]),
    })
    f = np.array([1.0, 2.0, 3.0, 4.0])
    alpha = 0.5
    out = _ema_within_day(df, f, alpha)
    # 手算: 1.0; 0.5*2+0.5*1=1.5; 0.5*3+0.5*1.5=2.25; 0.5*4+0.5*2.25=3.125
    expected = np.array([1.0, 1.5, 2.25, 3.125])
    assert np.allclose(out, expected), f"got {out}, expected {expected}"
    print(f"✅ 基础 EMA 数值正确: {np.round(out, 4)}")


def test_intraday_reset():
    """跨日应重置: 第二日第一个值等于其原始预测。"""
    df = pd.DataFrame({
        "instrument": ["A"] * 4,
        "date": pd.to_datetime(["2023-11-01 10:00", "2023-11-01 14:30",
                                "2023-11-02 10:00", "2023-11-02 14:30"]),
    })
    f = np.array([1.0, 2.0, 100.0, 200.0])
    out = _ema_within_day(df, f, 0.5)
    # 第二日第一个 = 100.0 (重置, 不受第一日影响)
    assert abs(out[2] - 100.0) < 1e-9, f"跨日未重置: {out[2]}"
    # 第二日第二个 = 0.5*200+0.5*100 = 150
    assert abs(out[3] - 150.0) < 1e-9, f"跨日 EMA 错误: {out[3]}"
    print(f"✅ 日内重置正确: {np.round(out, 4)}")


def test_per_instrument():
    """不同股票互不干扰。"""
    df = pd.DataFrame({
        "instrument": ["A", "A", "B", "B"],
        "date": pd.to_datetime(["2023-11-01 10:00", "2023-11-01 10:30",
                                "2023-11-01 10:00", "2023-11-01 10:30"]),
    })
    f = np.array([1.0, 2.0, 100.0, 200.0])
    out = _ema_within_day(df, f, 0.5)
    assert np.allclose(out, [1.0, 1.5, 100.0, 150.0]), f"got {out}"
    print(f"✅ 分股票隔离正确: {np.round(out, 4)}")


def test_causality():
    """因果性: 修改未来的值不应影响过去的结果。"""
    df = pd.DataFrame({
        "instrument": ["A"] * 4,
        "date": pd.to_datetime(["2023-11-01 10:00", "2023-11-01 10:30",
                                "2023-11-01 11:00", "2023-11-01 13:30"]),
    })
    f1 = np.array([1.0, 2.0, 3.0, 4.0])
    f2 = np.array([1.0, 2.0, 3.0, 999.0])   # 只改最后一个
    o1 = _ema_within_day(df, f1, 0.5)
    o2 = _ema_within_day(df, f2, 0.5)
    assert np.allclose(o1[:3], o2[:3]), f"因果性被破坏: {o1[:3]} vs {o2[:3]}"
    print(f"✅ 因果性正确 (改未来不影响过去)")


def test_alpha_one():
    """alpha=1 应该等价不平滑。"""
    df = pd.DataFrame({
        "instrument": ["A"] * 3,
        "date": pd.to_datetime(["2023-11-01 10:00", "2023-11-01 10:30",
                                "2023-11-01 11:00"]),
    })
    f = np.array([1.0, 5.0, 3.0])
    out = _ema_within_day(df, f, 1.0)
    assert np.allclose(out, f), f"alpha=1 应不变: {out}"
    print(f"✅ alpha=1 等价不平滑")


def test_unsorted_input():
    """乱序输入应输出正确 (内部会排序)。"""
    df = pd.DataFrame({
        "instrument": ["A", "A", "A"],
        "date": pd.to_datetime(["2023-11-01 11:00", "2023-11-01 10:00",
                                "2023-11-01 10:30"]),
    })
    f = np.array([3.0, 1.0, 2.0])   # 乱序
    out = _ema_within_day(df, f, 0.5)
    # 按时间排序后: 10:00(1.0) -> 10:30(2.0) -> 11:00(3.0)
    # EMA: 1.0, 1.5, 2.25  => 映射回原顺序: [2.25, 1.0, 1.5]
    expected = np.array([2.25, 1.0, 1.5])
    assert np.allclose(out, expected), f"got {out}, expected {expected}"
    print(f"✅ 乱序输入正确处理: {np.round(out, 4)}")


if __name__ == "__main__":
    test_basic()
    test_intraday_reset()
    test_per_instrument()
    test_causality()
    test_alpha_one()
    test_unsorted_input()
    print("\nEMA POST-PROCESS VERIFY OK")
