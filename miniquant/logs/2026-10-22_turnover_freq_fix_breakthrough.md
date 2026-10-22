# 换手正则"频率 bug"修复 + λ 重扫 —— 重大突破

- **日期**：2026-10-22
- **模型**：`submissions/cnn_3f_1021_turnover_v3`（三频 StockTransformerMF，关 XSEC）
- **实验设置**：6 个月 smoke（train=202307..202310，val=202311..202312），seed=42，AMP fp16
- **原始日志**：[raw/scan_to_fixed_corr.log](./raw/scan_to_fixed_corr.log)、
  [raw/scan_to_fixed_6mo.log](./raw/scan_to_fixed_6mo.log)

## 根因：换手正则"频率 bug"

此前四次"换手证伪"全部建立在同一个 bug 之上：
- 换手正则被设计成"每 epoch 只在末尾做 1 次独立 step"；
- 而主损失 soft-IC 每 epoch 更新 **158 次**（647190/4096）；
- 故换手正则的实际影响力 = λ/158，被主损失完全淹没。

证据：训练日志里换手 loss 项（`turnover=`）不降反升（ep01→ep12），
说明换手梯度被主损失反向推走，正则从未真正生效。

## 修复：换手 step 内联到 batch 循环（频率对齐）

把换手 step 从"每 epoch 1 次"改为"每个 batch 都做"（`TO_STEP_INTERVAL=1`），
换手梯度频率与主损失完全对齐。

## 实验结果

### rank drift 形式（频率对齐后）

| λ | best val_IC | val_turnover |
| --- | --- | --- |
| 0.0 | +0.0964 | 0.16~0.26 |
| 0.02 | +0.0980 | 0.13~0.26 |
| 0.5 | +0.0560 | 0.045~0.062 |

→ rank drift 无平滑 tradeoff：小 λ 无效，大 λ 毁 IC。

### corr 形式（频率对齐后）—— 最佳结果

| λ | best val_IC | val_turnover |
| --- | --- | --- |
| 0.05 | +0.0995 | 0.10~0.19 |
| **0.1** | **+0.1003** | **0.08~0.14** |
| 0.2 | +0.0938 | 0.06~0.10 |
| 0.5 | +0.0699 | 0.04~0.05 |

## 关键结论

1. **换手正则之前确实是写法 bug，不是"信号本质"**：频率对齐后换手率可大幅下降
   （val_turnover 从 0.16~0.23 降到 0.08~0.14）。
2. **corr 形式 λ=0.1 是最佳平衡点**：
   - val_IC = +0.1003（比旧基线 0.0983 还略高）
   - val_turnover = 0.08~0.14（比旧基线 0.16~0.23 显著下降）
   - **双赢：IC 升 + 换手降**
3. **rank drift 形式反而更差**（无平滑 tradeoff），corr 形式（全局相关）才是对的。

## 对比旧基线

| 配置 | val_IC | val_turnover |
| --- | --- | --- |
| 旧（corr, 频率 bug, λ=0.005） | 0.0983 | 0.16~0.23 |
| **新（corr, 频率对齐, λ=0.1）** | **+0.1003** | **0.08~0.14** |

## 下一步

- [ ] 用 λ=0.1 + corr 形式 + 频率对齐做 48 个月全量重训
- [ ] 验证全量后是否维持"IC 升 + 换手降"的双赢
- [ ] 若确认，上传公榜对比

## 关联

- 换手口径错位分析：[2026-10-21_turnover_mismatch_analysis.md](./2026-10-21_turnover_mismatch_analysis.md)
- 此前四次"证伪"（均基于频率 bug，结论作废）：
  [2026-10-21_rank_drift_falsified.md](./2026-10-21_rank_drift_falsified.md)、
  [2026-10-21_rankdrift_lambda_falsified.md](./2026-10-21_rankdrift_lambda_falsified.md)、
  [2026-10-21_turnover_loss_form_falsified.md](./2026-10-21_turnover_loss_form_falsified.md)
