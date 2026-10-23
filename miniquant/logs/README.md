# 模型实验日志

本目录记录 miniquant 每一版模型的**特性、实验结果与经验**，便于回溯迭代过程。

## 目录结构

```
logs/
├── README.md                 # 本索引
├── raw/                      # 原始训练日志（*.log，git 忽略，仅作证据留档）
│   ├── full.log              # v1 GRU 双流
│   ├── full2.log             # v2 GRU 双流（全量样本）
│   ├── mf_smoke.log          # v3 四频率融合 smoke
│   ├── mf_compare.log        # v3 四频 vs 双频对照
│   ├── full_cnn.log          # v4 CNN+Transformer
│   ├── full_base_xsec_ab.log # xsec 实验 baseline（随机 batch 对照）
│   ├── full_secbatch_ab.log  # xsec 实验：截面 batch（无注意力）
│   ├── full_xsec_ab.log      # xsec 实验（截面 batch + 交叉注意力）
│   ├── full_grouped_ab.log   # xsec 实验（分组大 batch + 交叉注意力）
│   ├── full_cnn3f_xsec_to.log# 多频 XSEC + 换手正则全量重训（新标签方向）
│   ├── full_cnn3f_amp.log    # AMP 混合精度全量重训（96.6min）
│   └── full_cnn3f_hidden96.log# hidden 加宽 64->96 全量重训（负贡献）
└── <date>_<模型>.md          # 各版本模型记录（正式记录，纳入 git）
```

## 版本索引（按时间排序）

| 版本 | 日期 | 模型 | 记录 | 最佳验证 IC |
| --- | --- | --- | --- | --- |
| v1 | 2026-09-30 | GRU 双流（E2EModel） | [2026-09-30_gru_bar30m.md](./2026-09-30_gru_bar30m.md) | 0.0422 |
| v2 | 2026-10-06 | GRU 双流（全量样本） | [2026-10-06_gru_bar30m_full2.md](./2026-10-06_gru_bar30m_full2.md) | 0.0394 |
| v3 | 2026-10-06 | 多频率融合 GRU（multifreq） | [2026-10-06_multifreq.md](./2026-10-06_multifreq.md) | 0.0363（4 频） |
| v4 | 2026-10-09 | CNN+Transformer（单频） | [2026-10-09_cnn_bar30m.md](./2026-10-09_cnn_bar30m.md) | 0.0473 |
| v5 | 2026-10-17 | 截面交叉注意力 A/B（单频） | [2026-10-17_xsec_ab.md](./2026-10-17_xsec_ab.md) | 0.0477（随机）/ 0.0402（截面 batch）/ 0.0426（xsec）/ 0.0436（分组 xsec） |
| v6 | 2026-10-18 | 多频 XSEC + 换手正则 + 标签方向统一 | [2026-10-18_xsec_to_multifreq.md](./2026-10-18_xsec_to_multifreq.md) | 0.0757（XSEC+换手，新标签） |
| v7 | 2026-10-18 | AMP 混合精度 + 训练效率优化 | [2026-10-18_amp_efficiency.md](./2026-10-18_amp_efficiency.md) | 0.0767（96.6min，5.3×提速） |
| v8 | 2026-10-19 | hidden 加宽 A/B（64→96） | [2026-10-19_hidden_width.md](./2026-10-19_hidden_width.md) | 0.0713（负贡献，已回滚） |
| v9 | 2026-10-19 | 公榜回退诊断 + 提交历史归档 | [2026-10-19_regression_diagnosis.md](./2026-10-19_regression_diagnosis.md) | 高分 0.0886 对应干净基线 |
| v10 | 2026-10-23 | 推理端 L1 软阈值降换手（L1 优于 EMA 的 L2） | [2026-10-23_l1_soft_threshold_beats_ema.md](./2026-10-23_l1_soft_threshold_beats_ema.md) | 等换手下 IC_IR 损失减半 |
| v11 | 2026-10-24 | **自有模型 + 团队高分模型推理端融合** | [2026-10-24_fuse_highscore.md](./2026-10-24_fuse_highscore.md) | **0.1310**（+14.0%，相关仅 0.50） |
| — | 持续更新 | 平台公榜成绩记录 | [leaderboard.md](./leaderboard.md) | 每次 submission 追加一行 |

> 说明：各版本验证集口径不完全一致（采样截面数 / 样本数随配置演进有变化），
> IC 数值只作趋势参考，不宜直接跨版本做严格排序比较；细节见各记录。

## 关键经验速览

1. **`rank` 目标 + `corr` 损失是基础**：比赛按截面 IC 评估排序能力，MSE 易被
   尺度项主导、退化成常数输出导致 IC=0（见 v1 记录）。
2. **多频率融合暂未带来增益**：四频（IC≈0.036）并未优于单频 bar30m（IC≈0.042），
   且 bar1m 分支磁盘/内存开销巨大；需要更好的融合结构而非简单拼接。
3. **CNN+Transformer 单频表现最佳**（IC≈0.047），且训练更快（全并行），是
   后续迭代的主攻方向。
4. **截面交叉注意力（单频）**：注意力模块本身有效（+0.0024、ICIR 显著改善），
   单截面 batch 有害（−0.0075，梯度噪声）；分组大 batch 部分回收（+0.0010），
   但净效果仍低于随机 batch baseline。**单频弱信号下梯度稳定性 > 损失对齐**，
   截面注意力应转向多频（信号更强）验证（见 v5 记录）。
5. **标签方向已统一为平台口径**（`end_price/future_vwap - 1`），推理侧不再取反；
   多频 XSEC + 换手正则全量重训 best val_IC=0.0757（见 v6 记录）。
6. **AMP 混合精度带来 5.3× 训练提速**（510.9→96.6min），且 IC 不降（0.0767），
   是后续所有实验的默认配置（见 v7 记录）。
7. **「扩大参数量」方向证伪**：hidden 64→96 全量 best val_IC 0.0767→0.0713，
   负贡献且更快过拟合。瓶颈是弱信号 + 数据量，而非模型容量（见 v8 记录）。
