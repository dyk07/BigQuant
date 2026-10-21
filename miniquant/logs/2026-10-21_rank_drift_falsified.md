# 组内 rank 漂移换手正则（方案 B）—— 证伪

- **日期**：2026-10-21
- **模型**：`submissions/cnn_3f_1021_turnover_v3`（三频 StockTransformerMF，关 XSEC + 换手 λ=0.005）
- **实验设置**：6 个月 smoke（train=202401..202404，val=202405..202406），seed=42，AMP fp16
- **原始日志**：[raw/smoke_turnover_v3_rankdrift.log](./raw/smoke_turnover_v3_rankdrift.log)、
  [raw/smoke_turnover_v3_corr.log](./raw/smoke_turnover_v3_corr.log)

## 背景：换手正则"口径错位"的修复尝试

此前定位（[2026-10-21_turnover_mismatch_analysis.md](./2026-10-21_turnover_mismatch_analysis.md)）：
- 训练换手损失 `1-corr` 是**全局相关**（把所有配对混在一起算 Pearson）；
- 平台换手率是**逐股票组内的排名漂移**；
- 两者量的是不同东西，导致换手正则"失灵"。

本实验实现"口径正确"的组内 rank 漂移损失，验证是否真能降换手。

## 实现

- `_soft_rank(p)`：可导 soft rank（sigmoid 近似 CDF，scale-invariant）
- `_grouped_rank_drift(pc, pn, sec_c, sec_n)`：对每个截面内算 soft rank，
  惩罚同股票配对在相邻截面的排名漂移（`mean|rank_c - rank_n|`）
- 换手 step 改为**按截面采样**（保证每截面有足够股票算 rank）
- 单元验证：排名不变 loss=0、反转 loss=0.30、整体平移 loss=0（shift-invariant）、可导

## 实验结果（严格对照：同月份 202401~202406，仅 USE_RANK_DRIFT 开关不同）

| 换手 loss 形式 | best val_IC | val_turnover 区间 |
| --- | --- | --- |
| corr（全局，对照） | +0.0687 | 0.1605 ~ 0.2211 |
| rank drift（组内） | +0.0687 | 0.1612 ~ 0.2180 |

## 关键结论

1. **组内 rank 漂移正则没有带来换手改善**：val_turnover 末尾 0.214 vs 0.218，
   val_IC 完全相同（0.0687）。即使用"口径正确"的损失，换手率纹丝不动。
2. **这证实了换手率是"信号本质"而非"预测抖动"**：模型在 30min 跨度上的
   排名变化主要由信息更新驱动，不是任何正则形式能消除的噪声。
3. 结合之前的 λ 扫描（0.005 最优）、换手 loss 形式扫描（zsq/zabs 证伪），
   **换手率这个维度已被三次实验证明"到顶"**，继续投入无收益。

## 结论与后续

- **换手率维度正式关闭**。三个方向（λ 强度、loss 形式、口径对齐）全部证伪。
- 换手率在当前模型架构下是信号本质决定的，无法通过正则进一步降低。
- 后续应聚焦：
  1. spread 损失全量验证（若 Sharpe 提升，是当前唯一有效方向）；
  2. 上传公榜，观察 score 0.56720 的排名。

## 关联

- 换手口径错位分析：[2026-10-21_turnover_mismatch_analysis.md](./2026-10-21_turnover_mismatch_analysis.md)
- 换手 λ 扫描：[2026-10-20_turnover_lambda_scan.md](./2026-10-20_turnover_lambda_scan.md)
- 换手 loss 形式证伪：[2026-10-21_turnover_loss_form_falsified.md](./2026-10-21_turnover_loss_form_falsified.md)
- spread 损失扫描：[2026-10-21_spread_loss_scan.md](./2026-10-21_spread_loss_scan.md)
