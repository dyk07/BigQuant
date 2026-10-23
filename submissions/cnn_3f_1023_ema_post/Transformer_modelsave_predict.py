"""
# Transformer 端到端 —— 加载本地模型 + 推理打分

本脚本只负责**推理**: 加载训练侧 `Transformer_modelsave_train.py` 产出的
`transformer_model.json`, 在平台注入的**测试集区间**上打分, **不训练**。

训练逻辑 (配置 / 模型结构 / 数据构建 / `train_and_save`) 全部沉淀在
`Transformer_modelsave_train.py` 里, 作为单一事实来源。流程分两阶段:

1. **阶段一 (参赛者本地运行一次)** —— 执行 `python Transformer_modelsave_train.py`,
   在写死的训练区间上从零训练, 把 **权重 + 标准化统计 + 结构超参** 存到
   `transformer_model.json`。
2. **阶段二 (平台公榜阶段调用 `main`)** —— 直接加载 `transformer_model.json`,
   在平台注入的测试区间上推理打分。

> 提交时请把 `Transformer_modelsave_train.py` 与训练好的 `transformer_model.json`
> 随本 notebook 一并上传(同一目录, 平台要求单层目录)。
> 公榜阶段平台只替换 `datasources / start_date / end_date` 并调用 `main`;
> 私榜阶段平台会用 `train_and_save` 在隔离环境从零重训, 用于合规性验证。

推理复用训练侧的 `build_dataset`, 且标准化统计 (mean/std) 随权重一起存盘、
推理时直接复用, 保证两阶段预处理严格一致, 杜绝数据泄漏与 train/infer 漂移。

**只依赖 1 分钟 K 线**: 模型需要的 5/15/30 分钟视角在 `build_dataset` 内部由
bar1m 重采样得到, 与原生多频数据数值逐位一致。
"""
import gc
import os

import numpy as np
import pandas as pd
import dai
import torch
import structlog

from Transformer_modelsave_train import (
    MODEL_PATH, BATCH, build_model, build_dataset, load_model, train_and_save,
    _iter_dataset, _encode_instrument, _section_ids,
)

logger = structlog.get_logger()

# ======================================================================
# 推理后处理: 日内因果 EMA 平滑 (降低换手率)
# ----------------------------------------------------------------------
# 关键发现 (2026-10-23): 训练时的换手正则与 IC 优化存在永久性的梯度冲突,
# 会削弱模型表示能力 (同换手下 IC_IR/Sharpe 损失 ~19%/21%)。改为在推理端做
# 时序平滑, 模型表示能力完整保留, 同换手下 IC_IR/Sharpe 损失仅 ~10%/10%。
#
# alpha 为新预测的权重: p'_t = alpha * p_t + (1 - alpha) * p'_{t-1}
#   alpha=1.0 -> 不平滑 (等于原基线)
#   alpha=0.45 -> 推荐值 (4 个月验证集: 换手 0.226->0.121, IC_IR 1.075->0.966)
# 因果性: 只使用当前及历史截面预测, 不含未来信息; 日内重置, 无跨日残留。
# ======================================================================
try:
    from config import EMA_ALPHA  # noqa: F401  (单一事实来源: config.py)
except ImportError:  # 平台单文件注入等极端场景下的兜底
    EMA_ALPHA = 0.45


def _ema_within_day(df, factor, alpha):
    """日内因果 EMA 平滑 (按 instrument + 交易日 分组, 日内重置)。

    df     : 含 date / instrument 的截面索引表 (与 factor 逐行对齐)
    factor : [N] 原始预测分数
    alpha  : 新样本权重 (越小越平滑)

    对每只股票在每个交易日内按时间顺序做因果 EMA:
        p'_t = alpha * p_t + (1 - alpha) * p'_{t-1},   p'_0 = p_0
    因果: 只用当前及更早截面; 日内重置: 每个交易日从 p_0 重新开始累积。
    仅使用已发生的数据, 不引入未来信息。
    """
    if alpha is None or alpha >= 1.0 or len(df) == 0:
        return factor
    d = pd.DataFrame({
        "_i": np.arange(len(df)),
        "inst": df["instrument"].to_numpy(),
        "day": pd.to_datetime(df["date"]).dt.normalize().to_numpy(),
        "t": pd.to_datetime(df["date"]).to_numpy(),
        "f": np.asarray(factor, dtype="float64"),
    })
    # 按 (inst, day) 分组, 组内按时间排序后做因果 EMA
    d = d.sort_values(["inst", "day", "t"], kind="mergesort")
    smoothed = d.groupby(["inst", "day"], sort=False, observed=True)["f"].transform(
        lambda x: x.ewm(alpha=alpha, adjust=False).mean())
    out = np.empty(len(df), dtype="float64")
    out[d["_i"].to_numpy()] = smoothed.to_numpy()
    return out


# ==== 加载权重做推理 (平台公榜阶段调用, 只替换 datasources/start_date/end_date) ====
def main(datasources, start_date, end_date):
    """加载已训练好的模型, 在样本外测试区间 (start_date~end_date) 上推理打分。

    本函数**不训练**: 权重来自随 notebook 上传的 MODEL_PATH 文件。
    start_date~end_date 为平台注入的【测试集区间】, 输出 ['date','instrument','factor']。"""
    table = datasources["bar1m"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"未找到模型文件 {MODEL_PATH}; 请先运行 train_and_save(...) 训练并保存, "
            f"再随 notebook 一起上传")

    # 模型存为文本类文件 (JSON), 用 load_model 读回 (张量按 dtype/shape 还原)
    ckpt = load_model(MODEL_PATH, map_location=device)
    freqs = ckpt["freqs"]
    seq_lens = ckpt["seq_lens"]
    stats = {f: (np.asarray(ckpt["mean"][f], np.float32),
                 np.asarray(ckpt["std"][f], np.float32)) for f in freqs}
    model = build_model(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    use_xsec = bool(ckpt.get("model_cfg", {}).get("use_xsec", False))
    # 方向标记: 权重 json 里的 label_direction 决定 factor 是否取负。
    #   * "end_over_vwap" (平台正确 label=end_price/vwap-1): 不取负;
    #   * "vwap_over_end" (旧方向权重, 排序相反): factor 取负以对齐平台口径。
    factor_sign = -1.0 if ckpt.get("label_direction") == "vwap_over_end" else 1.0
    logger.info("已加载模型", path=MODEL_PATH, freqs=freqs, device=str(device),
                use_xsec=use_xsec, factor_sign=factor_sign)

    # ---------- 推理 (逐月流式, 峰值内存 ~1 个月) ----------
    logger.info("逐月构建测试集并预测", start=str(start_date), end=str(end_date))
    parts = []
    n_rows = 0
    with torch.no_grad():
        for Xs, _, idx_df, keys in _iter_dataset(
                table, start_date, end_date, "infer",
                ckpt["aligned_minutes"], stats, freqs=freqs, seq_lens=seq_lens):
            if len(idx_df) == 0:
                continue
            Xt = [torch.from_numpy(np.ascontiguousarray(x)) for x in Xs]
            preds = []
            if use_xsec:
                sec_ids = _section_ids(keys)
            else:
                sec_ids = None
            for i in range(0, len(idx_df), BATCH):
                xb = [t[i:i + BATCH].to(device) for t in Xt]
                sidb = (None if sec_ids is None else
                        torch.from_numpy(sec_ids[i:i + BATCH]).to(device))
                preds.append(model(xb, sec_ids=sidb).cpu().numpy())
            out = idx_df.copy()
            # 训练与推理 label 方向已统一为 end_price/vwap - 1; 旧方向权重
            # (label_direction=vwap_over_end) 需乘 -1 对齐平台口径。
            raw_factor = ((np.concatenate(preds).astype(np.float64) * factor_sign)
                          if preds else np.zeros(len(out)))
            # ---- 推理后处理: 日内因果 EMA 平滑 (降换手, 不损模型表示能力) ----
            # 对同一 (instrument, 交易日) 的相邻截面做指数移动平均, 日内重置。
            # 因果性: 只用到当前及之前截面的预测, 不含未来信息;
            # 日内重置: 每个交易日从头累积, 跨月/跨月无需传递状态。
            if EMA_ALPHA is not None:
                raw_factor = _ema_within_day(out, raw_factor, EMA_ALPHA)
            out["factor"] = raw_factor
            parts.append(out)
            n_rows += len(out)
            del Xs, Xt, preds, idx_df
            gc.collect()
    if parts:
        pred_df = pd.concat(parts, ignore_index=True)
    else:
        pred_df = pd.DataFrame({"date": [], "instrument": [], "minute_of_day": [],
                                "factor": []})
    logger.info("推理完成", rows=n_rows)
    del parts
    gc.collect()

    # ---------- 对齐中证 1000 + 规范输出 ----------
    stk = dai.query("SELECT date, instrument::string AS instrument "
                    "FROM cpt_jyc_2026_instruments",
                    filters={"date": [start_date, end_date]}).df()
    # 逐月预测已按比赛的 8 个日内截面生成, 必须按完整时间精确对齐。
    # 平台 instrument 为字符串代码('000001.SZ'), 内部 keys 用的是 _encode_instrument
    # 的稳定 int 码; 对 cpt_jyc_2026_instruments 用同款函数还原成一致的码做 join,
    # 再输出原始字符串 instrument (与 baseline 一致)。
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    stk["date"] = pd.to_datetime(stk["date"])
    stk["_code"] = _encode_instrument(stk["instrument"])
    pred_df = pred_df.rename(columns={"instrument": "_code"})
    result = (pd.merge(stk, pred_df, on=["date", "_code"], how="inner")
                .replace([np.inf, -np.inf], np.nan).dropna(subset=["factor"])
                .drop_duplicates(["date", "instrument"])[["date", "instrument", "factor"]]
                .sort_values(["date", "instrument"])
                .reset_index(drop=True))
    logger.info("因子构建完成", rows=len(result), dates=result["date"].nunique(),
                instruments=result["instrument"].nunique())
    return result


if __name__ == "__main__":
    from bigmodule import M

    # 只用 1 分钟 K 线作为输入数据 (多频视角在内部重采样得到)
    datasources = {"bar1m": "cpt_jyc_2026_e2e_bar1m"}

    # 本地首次运行: 若权重文件不存在, 先训练并保存 (上传前请确保已生成 transformer_model.json)
    # 正式做法是在终端执行 `python Transformer_modelsave_train.py` 训练一次
    if not os.path.exists(MODEL_PATH):
        logger.info("未发现已保存模型, 开始训练", path=MODEL_PATH)
        train_and_save(datasources)

    # 本地用一小段区间模拟「平台注入的测试集区间」
    start_date, end_date = "2024-07-01 00:00:00", "2024-09-30 23:59:59"
    logger.info("计算因子 (仅加载权重推理, 不重训)", start=start_date, end=end_date)
    factor_data = main(datasources, start_date, end_date)
    print(factor_data.head())

    # 评估系统: 分数经风格剔除后等价于每日单因子, show=True 画绩效图 (IC / 分组 / 压力期)
    logger.info("开始评估分数")
    result = M.jyc_eval._latest(factor_data=factor_data, show=True)
