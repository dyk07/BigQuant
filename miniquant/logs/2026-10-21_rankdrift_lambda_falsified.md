# rank-drift 换手正则 λ 扫描（大 λ）—— 换手维度最终证伪

- **日期**：2026-10-21
- **模型**：`submissions/cnn_3f_1021_turnover_v3`（三频 StockTransformerMF，关 XSEC，rank-drift 换手形式）
- **实验设置**：6 个月 smoke（train=202307..202310，val=202311..202312），seed=42，AMP fp16
- **原始日志**：[raw/scan_rankdrift_6mo.log](./raw/scan_rankdrift_6mo.log)

## 动机

rank-drift 换手损失的量纲（soft rank 差，~0~0.5）比 corr（1-corr，0~2）小约
4~8 倍，故 λ=0.005 的约束力几乎为零。本实验把 λ 加大到 1.0（×200），看是否
能用「牺牲 IC」换取「换手下降」。

## 实验结果

| λ (rank drift) | best val_IC | val_turnover 区间 |
| --- | --- | --- |
| 0.05 | +0.0956 | 0.17~0.23 |
| 0.1 | +0.0951 | 0.16~0.24 |
| 0.2 | +0.0954 | 0.17~0.24 |
| 0.5 | +0.0956 | 0.17~0.24 |
| 1.0 | +0.0948 | 0.16~0.26 |

（对照：corr 形式 λ=0.005 的 val_IC=0.0983，val_turnover=0.16~0.23）

## 关键结论

1. **λ 加大 200 倍（0.005→1.0），换手率纹丝不动**：val_turnover 始终 0.16~0.24，
   与 λ=0.005 时完全相同。val_IC 只从 0.0983 略降到 0.0948（-3.6%）。
2. **换手率是信号本质，不是正则强度或形式的问题**：模型在 30min 跨度上对个股
   排名的判断由信息更新驱动，任何换手正则都无法在不大幅牺牲 IC 的前提下降低它。
3. **换手维度四次实验全部证伪**，正式关闭：
   - λ 强度（corr 形式）：0.005 最优，再强无益
   - loss 形式（zsq/zabs 幅度约束）：证伪
   - 口径对齐（rank drift 组内）：证伪
   - 大 λ（rank drift ×200）：证伪

## 结论与建议

- **不要在换手率上继续投入**。这是四次实验的一致结论，属于信号本质限制。
- 当前最优仍是 `cnn_3f_1021_spread`（spread λ=0.5，本地 Sharpe 0.755），或
  `cnn_3f_1020_newlabel_to0005`（公榜 score=0.56720）。
- 后续方向应聚焦于提升 IC/IC_IR/Sharpe（因子质量），而非换手。

## 关联

- rank-drift 实现证伪：[2026-10-21_rank_drift_falsified.md](./2026-10-21_rank_drift_falsified.md)
- 换手口径错位分析：[2026-10-21_turnover_mismatch_analysis.md](./2026-10-21_turnover_mismatch_analysis.md)
- 换手 loss 形式证伪：[2026-10-21_turnover_loss_form_falsified.md](./2026-10-21_turnover_loss_form_falsified.md)
- spread 全量：[2026-10-21_spread_full_retrain.md](./2026-10-21_spread_full_retrain.md)
