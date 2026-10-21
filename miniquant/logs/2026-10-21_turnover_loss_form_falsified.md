# 换手 loss 形式扫描：幅度约束（zsq/zabs）—— 证伪

- **日期**：2026-10-21
- **模型**：`submissions/cnn_3f_1021_turnover_v2`（三频 StockTransformerMF，关 XSEC + 换手 λ=0.005）
- **实验设置**：6 个月 smoke（train=202307..202310，val=202311..202312），seed=42，AMP fp16，独立配对 batch（100% 命中）
- **原始日志**：[raw/scan_form_6mo.log](./raw/scan_form_6mo.log)

## 动机

评分标准中**换手率占 20% 权重**（换手越低越好）。当前换手正则 `λ·(1 - corr(p_cur, p_next))`
只约束**方向一致**，不约束**幅度一致**，因此"排名小幅漂移"不会被惩罚，换手率难以进一步下降。

假设：换手率本质是"信号变化幅度"，用**幅度约束**（对 cur/next 各自 z-score 后算幅度差）应能
更直接地压低换手。

## 实现

新增 `_turnover_loss(pc, pn, form, alpha)`，5 种 scale/shift-invariant 形式（预测分数 scale
未约束，直接幅度差会诱导模型把分数压到 0，故必须先 z-score）：

| form | 定义 |
| --- | --- |
| `corr` | `1 - corr(pc, pn)`（基线） |
| `zsq` | `mean((zc - zn)^2)` |
| `zabs` | `mean\|zc - zn\|` |
| `corr_zsq` | `(1-corr) + α·mean((zc-zn)^2)` |
| `corr_zabs` | `(1-corr) + α·mean\|zc-zn\|` |

数值验证：5 种形式均 scale/shift-invariant（scale×100 / shift±5 结果不变）。

## 实验结果（6 个月 smoke）

| form | best val_IC | val_turnover 区间 | 耗时 |
| --- | --- | --- | --- |
| **corr（基线）** | **+0.0983**（ep03） | 0.16~0.23 | 8.0min |
| zsq | +0.0961（ep06） | 0.17~0.27 | 10.3min |
| zabs | +0.0946（ep03） | 0.17~0.23 | 8.0min |
| corr_zsq | +0.0955（ep03） | 0.17~0.24 | 8.0min |
| corr_zabs | +0.0949（ep06） | 0.17~0.27 | 10.5min |

## 关键结论

1. **幅度约束方向证伪**：所有幅度形式（zsq/zabs/混合）的 val_IC 都**低于 corr 基线**
   （0.0946~0.0961 vs 0.0983，降 0.002~0.004），且 val_turnover **没有下降**（反而 zsq/zabs
   的末尾 epoch 换手更高）。
2. **换手率不是"方向 vs 幅度"问题**：更强地惩罚幅度反而损伤 IC 且不降换手，说明当前模型的
   换手率（val 0.16~0.23）主要由**信息更新**驱动（30min 跨度的信号本身在变），而非可由
   正则形式改变的"预测抖动"。
3. **corr 基线仍是最优**：`1 - corr` 在 IC 和换手之间已是最佳平衡，λ=0.005 + corr 形式不可
   再通过换形式改进。

## 结论与后续方向

- 换手 loss 形式扫描**证伪**。当前换手率的瓶颈是**信号本质**（30min 跨度的信息更新导致排名
  变化），不是正则形式或强度。
- 换手率的真正抓手可能在于：
  1. **更长的预测 horizon / 更粗的采样网格**（减少截面切换频率）；
  2. **多截面聚合**（对未来多个截面的平均 VWAP 收益做预测，天然平滑信号）；
  3. 接受当前换手水平，把 IC/IC_IR 作为主攻方向（IC 提升会稀释换手在总评分中的劣势）。
- **不建议再在换手 loss 上投入**：已证伪两个方向（λ 强度、loss 形式）。

## 关联

- λ 扫描：[2026-10-20_turnover_lambda_scan.md](./2026-10-20_turnover_lambda_scan.md)
- SWA 证伪：[2026-10-20_swa_falsified.md](./2026-10-20_swa_falsified.md)
