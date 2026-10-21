# spread 损失全量验证 —— 温和正向，Sharpe +5.9%

- **日期**：2026-10-21
- **模型**：`submissions/cnn_3f_1021_spread`（三频 StockTransformerMF，关 XSEC + 换手 λ=0.005 + spread λ=0.5）
- **实验设置**：48 个月全量重训（train=202001..202312，val=202311..202312），seed=42，AMP fp16
- **原始日志**：[raw/full_spread_lam05.log](./raw/full_spread_lam05.log)

## 背景

诊断（Sharpe 分解）发现：soft-IC 损失对"头部/尾部选股"不敏感，实际多空 spread
只有完美排序的 ~9%，是 Sharpe 排名落后的根因。spread 损失用 softmax 加权构造
可导的"多头收益 - 空头收益"，引导模型优化两端。

## 全量结果对比

| 模型 | val_IC | IC_IR | 多空 Sharpe | 命中率 |
| --- | --- | --- | --- | --- |
| 基线（1020，无 spread） | **+0.1099** | 1.060 | 0.713 | 9.4% |
| spread（λ=0.5） | +0.1086 | **1.077** | **0.755** | **9.6%** |

## 训练曲线（spread λ=0.5）

| epoch | val_IC | val_turnover |
| --- | --- | --- |
| ep01 | +0.0991 | 0.2118 |
| ep02 | +0.1020 | 0.2014 |
| ep03 | +0.1016 | 0.1896 |
| ep04 | +0.1004 | 0.1777 |
| ep05 | +0.1048 | 0.1935 |
| ep06 | +0.1075 | 0.2142 |
| ep07 | +0.1022 | 0.2235 |
| ep08 | **+0.1086** | 0.2269 |
| ep09 | +0.1082 | 0.2177 |
| ep10 | +0.1058 | 0.2255 |
| ep11 | +0.1076 | 0.2380 |
| ep12 | +0.1083 | 0.2144 |

## 结论

1. **spread 损失在全量数据上温和有效**：Sharpe 0.713→0.755（+5.9%），IC_IR
   1.060→1.077（+1.6%），命中率 9.4%→9.6%。代价 val_IC 略降 0.0013（-1.2%）。
2. **与 smoke 方向一致**：smoke 里 λ=0.5 Sharpe 最优，全量验证确认了这个趋势。
3. **权衡判断**：
   - 收益：Sharpe（20%权重）+ IC_IR（20%）双提升；
   - 代价：ic_mean（20%）略降，且 ic_mean/stress 与 IC 强相关可能连带微降。
   - 净效果取决于公榜各指标竞争密度，但方向偏正向。

## 待办

- [ ] 决定是否上传公榜（需权衡 IC 略降 vs Sharpe 提升）
- [ ] 若上传，生成 notebook + 更新 leaderboard.md + ARCHIVE.md

## 关联

- spread 扫描（smoke）：[2026-10-21_spread_loss_scan.md](./2026-10-21_spread_loss_scan.md)
- Sharpe 分解诊断：[../files/2026-10-21_sharpe_diag.md](../files/2026-10-21_sharpe_diag.md)
- 换手正则证伪：[2026-10-21_rank_drift_falsified.md](./2026-10-21_rank_drift_falsified.md)
