# 多频 XSEC + 换手正则 + 标签方向统一

- **日期**：2026-10-18
- **模型**：`submissions/cnn_3f` 的 `StockTransformerMF`（三频 bar30m/bar15m/bar5m，
  `use_xsec=True`，`use_turnover=True`）
- **原始日志**：[raw/full_cnn3f_xsec_to.log](./raw/full_cnn3f_xsec_to.log)
- **权重**：`submissions/cnn_3f/transformer_model.json`（16.49 MB，best val_IC=+0.0757）

## 本次改动（相对上一版提交包）

1. **标签方向统一为平台口径**
   - 旧：`label = future_vwap / end_price - 1`（本地曾因历史口径在推理侧 `-1` 取反）。
   - 新：`label = end_price / future_vwap - 1`，与平台 `vwap_return = end_price / vwap - 1`
     完全一致；推理侧撤销取负，直接输出 factor。
   - 涉及文件：三个提交包的 `build_labels` / `predict`、miniquant 的
     `features.py` / `config.py` / `pipeline.py` / `README.md` / `PROJECT_CONTEXT.md`、
     `discussion/diagnosis.md`。

2. **截面交叉注意力移植到多频提交包**（`Transformer_modelsave_train.py` 三包同步）
   - `CrossSectionAttention`：多头自注意力 + FFN + Pre-LN，置于 `pool` 之后
     （`[N, 2*hidden]` 池化向量），复杂度 `O(N²·d)`。
   - block-diagonal mask：`sec_ids[:,None] == sec_ids[None,:]` 传入 `MultiheadAttention`
     的 `attn_mask`，让注意力只在同一截面内进行。
   - 残差拼接：`[原始表示 ‖ 截面上下文]` 送入 head。
   - `_section_ids(keys)`：`day*1440 + minute_of_day` → 截面编号。

3. **换手率惩罚 loss**（`USE_TURNOVER=True`, `LAMBDA_TURNOVER=0.01`）
   - `_turnover_next(keys)`：`lexsort` 定位同 `(instrument, day)` 的下一采样截面。
   - 换手正则 `λ·(1 - corr(p_cur, p_next))`，batch 内配对，抑制相邻截面预测
     剧烈翻转（压低换手指标，与「换手越低排名越高」的评分方向一致）。

4. **`_corr_batch` NaN 梯度修复**
   - `eps` 移到 `sqrt` 内部，避免常数向量（方差=0）反向传播得到 `inf*0=NaN`。

## 实验结果（三频，全量 48 个月）

- 数据：2020–2023 共 48 个月（train 46 月，val 202311–202312 共 2 月），
  train_N=7,391,450，val_N=343,118。
- 训练耗时 510.9 min，峰值内存 26,708 MB（WSL 内存上限已调至 28GB）。

| epoch | val_IC | turnover(1-corr) |
| --- | --- | --- |
| ep01 | +0.0468 | 0.6583 |
| ep04 | +0.0585 | 0.5176 |
| ep06 | +0.0707 | 0.3946 |
| ep08 | **+0.0757** | 0.5692 |
| ep09 | +0.0743 | 0.4915 |
| ep12 | +0.0724 | 0.4044 |

- **best val_IC = +0.0757**（ep08）。
- 端到端方向验证（2024H2，976 个截面）：factor vs label 的 IC_mean=+0.0704，
  **方向正确（IC>0）**。

## 对比与结论

| 配置 | best val_IC | 备注 |
| --- | --- | --- |
| 三频基线（旧标签，无 XSEC/换手） | 0.0782 | 历史最佳 |
| 三频 + XSEC + 换手（新标签方向） | 0.0757 | 本次 |

1. **标签换向后方向正确**：端到端 factor vs label IC=+0.0704>0，推理无取反，
   与平台评估口径一致。
2. **XSEC + 换手组合净效果略降**：best val_IC 0.0757 略低于旧基线 0.0782
   （−0.0025）。换手正则（λ=0.01）以小幅 IC 代价换取换手下降
   （turnover 从 ep01 0.6583 收敛至 0.37–0.57 区间），与比赛「换手越低排名越高」
   的评分方向一致，最终评分需看五项指标加权而非单一 IC。
3. **口径不可直接比较**：本次标签方向与旧基线相反，val_IC 数值含口径差异，
   仅作趋势参考。

## 备注

- 新增 `submissions/_tools/full_train.py`（本地 feather 伪造 dai 的全量训练驱动）
  与 `_sign_check2.py`（端到端方向验证）。
- 修正 `build_notebook.py` 的 `CFG_KEYS`，纳入 `USE_XSEC/XSEC_LAYERS/XSEC_HEADS/
  USE_TURNOVER/LAMBDA_TURNOVER`，保证私榜从零重训时 notebook 内联配置与本地
  权重一致（否则会 fallback 到 `USE_XSEC=False`）。
