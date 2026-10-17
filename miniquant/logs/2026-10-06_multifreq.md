# v3 — 多频率融合（multifreq，方案 A）

- **日期**：2026-10-06
- **模型**：`train_mf.py` + `multifreq.py` 的多分支融合 GRU（每频率一个编码分支，concat 后输出）
- **原始日志**：[raw/mf_smoke.log](./raw/mf_smoke.log)（4 频 smoke）、
  [raw/mf_compare.log](./raw/mf_compare.log)（4 频 vs 双频对照）

## 特性

- 四频率：bar30m / bar15m / bar5m / bar1m，在相同 (day, minute_of_day) 对齐后融合。
- 数据落盘为 memmap 分块（`features_mf/`），按 epoch 只读该读的块，避免常驻内存爆炸。
- 参数：四频分支 322,505，双频（30m+5m）163,461。
- 高频分支（尤其 bar1m，T=120 时 12KB/样本）是磁盘瓶颈，可通过
  `--lookback-bar1m 30` 等每频率 lookback 覆盖缓解。

## 实验结果（smoke：202301 训练 / 202302 验证）

| 配置 | 最佳验证 IC | 备注 |
| --- | --- | --- |
| 四频，bar1m lookback=120m | 0.0299 | mf_smoke.log |
| 四频，bar1m lookback=30m | 0.0363 | mf_compare.log |
| 双频（30m+5m） | 0.0329 | mf_compare.log |

## 经验 / 教训

- **多频融合未跑赢单频 bar30m 基线**（v1/v2 IC≈0.039~0.042）：
  - 简单分支拼接没有带来增量，bar1m 分支反而占掉大量磁盘/内存（单月数据集即 2.2GB）。
  - bar1m 的 lookback 从 120m 缩到 30m 后 IC 反而提升（0.0299 → 0.0363），
    说明更长的 1 分钟历史里噪声多于信号。
- 后续若要继续多频方向，需要**跨频率注意力 / 门控融合**等更强结构，
  而非简单 concat；否则不如专注单频。

## 清理说明

- 本方案产生的 `features_mf/` 数据集未纳入 git、已在清理中删除；
  `_mf_smoke.sh` / `_mf_compare.sh` / `_monitor.sh` / `_disk_calc.sh` 等一次性脚本也已删除。
