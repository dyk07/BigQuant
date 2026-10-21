# 平台公榜成绩记录

> 用途：集中记录每一次 submission 在**平台公榜**上的五项指标原始值。
> 规则：**每进行一次新 submission，就在下方表格追加一行**（不要覆盖旧行）。
> 指标由平台 `M.jyc_eval._latest(factor_data=..., show=True)` 输出。

## 填写说明

- **提交日期**：平台提交时间（或本地模型/权重日期）。
- **提交包 / 归档目录**：对应 `submissions/` 下的目录名（如 `cnn_3f_1017_baseline_highscore`）。
- 五项指标均为**原始值**（排名/加权前的量纲）：
  - `ic_mean`：IC 均值（越高越好）
  - `ic_ir`：IC_IR（越高越好）
  - `sharpe_ratio`：多空组合夏普比率（越高越好）
  - `stress_stability`：压力场景稳定性（越高越好）
  - `turnover`：换手率（**越低越好**）
- **备注**：记录模型关键特征（频率、是否 XSEC/换手/AMP、hidden 等），便于回溯。

## 成绩表格

> 符号已统一为平台口径 `label = end_price/vwap - 1`（2026-10-19 修正）。
> 旧方向权重（`vwap_over_end`）曾反向打分，修正后负值全部转正。

| # | 提交日期 | 提交包 / 归档目录 | ic_mean | ic_ir | sharpe_ratio | stress_stability | turnover | 备注 |
|---|----------|-------------------|---------|-------|--------------|------------------|----------|------|
| 1 | 10-17 | `cnn_3f_1017_baseline_highscore` | 0.0886 | 1.5232 | 40.7392 | 1.6612 | 0.6584 | 干净三频基线（高分版） |
| 2 | 10-18 | `cnn_2f_1018_newlabel` | 0.0890 | 1.5082 | 40.0046 | 1.6092 | 0.6156 | 两频 newlabel（权重实为旧方向） |
| 3 | 10-18 | `cnn_3f_1018_newlabel_xsec` | 0.0642 | 1.0535 | 31.0108 | 0.8910 | 0.6466 | +XSEC（回退版） |
| 4 | 10-19 | `cnn_3f_1019_newlabel_xsec_amp` | 0.0631 | 0.9501 | 26.3558 | 0.7951 | 0.4934 | +XSEC+换手+AMP（回退版） |
| 5 | 10-20 | `cnn_3f_1020_newlabel_to0005` | 0.0930 | 1.6736 | 46.9558 | 1.4873 | 0.6750 | 关XSEC+λ=0.005，公榜 score=0.56720 |
| 6 | 10-20 | `cnn_3f_1020_swa` | — | — | — | — | — | 关XSEC+λ=0.005+SWA平台期权重平均（证伪，回退single best） |
| 7 | 10-21 | `cnn_3f_1021_spread` | — | — | — | — | — | 关XSEC+λ=0.005+spread λ=0.5，本地 Sharpe 0.755/IC_IR 1.077（待公榜） |

<!-- 新 submission 请在上表追加一行，格式：
| N | YYYY-MM-DD | submissions/ 目录名 | ic_mean | ic_ir | sharpe | stress | turnover | 备注 |
-->
