"""端到端验证融合包: 在本地数据上真实执行 notebook 的全部 cell。

模拟平台行为:
  1. 按顺序 exec notebook 的每个 code cell (共享同一命名空间);
  2. 用本地 feather 数据伪装 dai.query (见 _hs_harness);
  3. 调用 main(datasources, start, end), 检查输出契约;
  4. 计算 IC / IC_IR / Sharpe / 换手。

指标口径与 _cross_fuse_test.py 完全一致 (复用其 build_labels / metrics),
避免自造的 label 定义引入系统性偏差。

用法: python _verify_fuse_e2e.py [包名] [start] [end] [--bar1m-only]

  --bar1m-only  只给 bar1m、不给 bar30m/bar5m, 用于验证高分模型段的重采样降级
                路径 (平台若未注入 bar30m/bar5m 时的兜底分支)。
"""
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _hs_harness import install_fake_dai  # noqa: E402
from _cross_fuse_test import build_labels, metrics  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PKG_NAME = "cnn_3f_1024_fuse_raw"
PKG = os.path.join(ROOT, "submissions", PKG_NAME)
NB = os.path.join(PKG, "Transformer_modelsave_predict.ipynb")


def install_stub_structlog():
    """structlog 由平台预装; 本地验证只需一个打印型替身, 不改环境。"""
    try:
        import structlog  # noqa: F401
        return
    except ImportError:
        pass
    import types

    class _L:
        def info(self, event="", **kw):
            print(f"    [info] {event} "
                  + " ".join(f"{k}={v}" for k, v in kw.items()))

        warning = error = info

        def debug(self, *a, **k):
            pass

    m = types.ModuleType("structlog")
    m.get_logger = lambda *a, **k: _L()
    sys.modules["structlog"] = m


def run_notebook():
    """按平台方式顺序执行 notebook 的所有 code cell, 返回最终命名空间。"""
    nb = json.load(open(NB, encoding="utf-8"))
    ns = {"__name__": "__main__", "__file__": os.path.join(PKG, "nb.py")}
    os.chdir(PKG)
    # Jupyter 会把 notebook 所在目录放进 sys.path, 训练脚本里的 `import config`
    # 依赖这一点; 本地用 exec 模拟时必须手动补上。
    if PKG not in sys.path:
        sys.path.insert(0, PKG)
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "RUN_LOCAL_DEMO" in src:
            print(f"  cell{i}: 跳过本地 demo 驱动")
            continue
        try:
            exec(compile(src, f"<cell{i}>", "exec"), ns)
        except Exception:
            print(f"!! cell{i} 执行失败")
            traceback.print_exc()
            raise
        print(f"  cell{i}: OK ({len(src)} 字符)")
    return ns


def show(tag, r):
    print(f"  {tag:<14} IC={r['ic']:+.4f} IC_IR={r['ic_ir']:>7.3f} "
          f"Sharpe={r['sharpe']:>7.3f} TO={r['turnover']:.4f}")


def main():
    global PKG, NB
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if args and not args[0][:2].isdigit():
        PKG = os.path.join(ROOT, "submissions", args.pop(0))
        NB = os.path.join(PKG, "Transformer_modelsave_predict.ipynb")
    start = args[0] if len(args) > 0 else "2024-11-01"
    end = args[1] if len(args) > 1 else "2024-11-29"
    bar1m_only = "--bar1m-only" in flags
    tag = PKG.split(os.sep)[-1] + ("_b1" if bar1m_only else "")
    install_fake_dai()
    install_stub_structlog()

    print(f"[verify] 包={PKG}")
    print("[verify] 执行 notebook 全部 cell ...")
    ns = run_notebook()

    print(f"[verify] 调用 main({start} ~ {end}), bar1m_only={bar1m_only} ...")
    # 平台注入的 datasources 值是**表名字符串**。--bar1m-only 用来验证
    # 「平台只给 bar1m」这一历史已证实的接口边界下, 高分模型段的降级路径。
    ds = {"bar1m": "bar1m"}
    if not bar1m_only:
        ds.update({"bar5m": "bar5m", "bar30m": "bar30m"})

    def cached(name, fn):
        """推理很慢; 缓存各路段预测, 便于反复调试指标口径。"""
        p = os.path.join(HERE, f"_preds_fuse_{tag}_{name}_{start}_{end}.parquet")
        if os.path.exists(p):
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            print(f"  [cache] {name} <- {os.path.basename(p)} ({len(df)} 行)")
            return df
        df = fn()
        df.to_parquet(p, index=False)
        return df

    out = cached("main", lambda: ns["main"](ds, start, end))
    print(f"  输出: {len(out)} 行, 列={list(out.columns)}, "
          f"日期={out['date'].nunique()}, 标的={out['instrument'].nunique()}")
    assert list(out.columns) == ["date", "instrument", "factor"], "输出列不符"
    assert out["factor"].notna().all(), "存在 NaN factor"
    assert np.isfinite(out["factor"].to_numpy()).all(), "存在 inf factor"
    assert not out.duplicated(["date", "instrument"]).any(), "存在重复键"
    print("  [OK] 输出契约通过")

    print("[verify] 指标口径 = _cross_fuse_test (bar1m 30min 前视 VWAP)")
    lab = build_labels()
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) + pd.Timedelta(days=1)
    lab = lab[(lab["date"] >= lo) & (lab["date"] < hi)]

    def ev(df):
        m = df.merge(lab, on=["date", "instrument"], how="inner")
        return metrics(m)

    print(f"\n  融合结果 (w={ns['FUSE_W']}, θ={ns['SOFT_THETA']}):")
    show("fuse+soft", ev(out))

    ours = cached("ours", lambda: ns["ours_main"](ds, start, end))
    hs = cached("hs", lambda: ns["hs_main"](ds, start, end))
    print()
    for nm, p in (("ours", ours), ("hs", hs)):
        show(nm, ev(p))
    show("fuse(no soft)", ev(ns["_fuse_predictions"](ours, hs, ns["FUSE_W"])))

    # 相关性决定融合上限: 同族模型间 0.83 已饱和, 0.52 量级才是强互补。
    j = (ours.rename(columns={"factor": "a"})
         .merge(hs.rename(columns={"factor": "b"}), on=["date", "instrument"]))
    ra = j.groupby("date", sort=False)["a"].rank(pct=True)
    rb = j.groupby("date", sort=False)["b"].rank(pct=True)
    print(f"\n  截面 rank 相关 ours↔hs: {np.corrcoef(ra, rb)[0,1]:.4f} "
          f"(参考: 同族 0.83 = 已饱和; ~0.52 = 强互补)")
    print("  [OK] 端到端验证完成")


if __name__ == "__main__":
    main()
