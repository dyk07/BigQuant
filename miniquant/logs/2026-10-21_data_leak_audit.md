# 多频模型数据泄露排查报告

- **日期**：2026-10-21
- **排查对象**：基线 `cnn_3f_1020_newlabel_to0005` 及各最新版本（1021_spread、1021_turnover_v2 等均复用同一数据管道）
- **方法**：实证检查数据管道的 7 个关键时序点，逐一验证输入窗口与标签窗口的时序隔离

## 排查点与结论

| # | 排查点 | 结论 |
|---|--------|------|
| 1 | 输入窗口 gather 时序 | ✅ `gather_windows` 取 `feat[anchor-T+1 .. anchor]`，严格 ≤ 决策时刻 |
| 2 | 标签时序 | ✅ `label = close[t+h]/vwap[t:t+h]-1`，只用未来窗口，与输入无重叠 |
| 3 | 特征无未来信息 | ✅ `close_ret` 用 `shift(1)`（上一根 bar），手工对拍数值一致 |
| 4 | 多频对齐 | ✅ `_align_freqs` 按 `(inst,day,mod)` 取交集，各频同一决策时刻 |
| 5 | bar 时间戳语义 | ✅ bar 时间戳 = 桶结束时刻（ceil 口径），5m/15m/30m 对齐一致 |
| 6 | 跨日/跨月 label | ✅ `build_labels` 用 `groupby([inst,day]).shift(-h)`，组内 shift 不跨日 |
| 7 | 月初窗口跨月回溯 | ✅ 设计允许（`input_same_day_only=False`），且 `inst[i]==inst[i-seq_len]` 防跨股票 |

## 详细验证证据

### 1. 输入窗口与标签严格分离
- 输入窗口最后一行 = anchor 时刻的 bar（决策时刻及之前 T 根 bar）
- 标签 = anchor 之后的 h 根 bar（未来收益）
- 两者无任何时间重叠

### 2. 特征无未来信息
- `close_ret[10:30] = close[10:30]/close[10:00] - 1 = -0.006944`，手工计算一致
- `open_rel/high_rel/low_rel` 均为当前 bar 相对当前 close
- 盘口快照字段为"末值"（当前 bar 快照）

### 3. 多频对齐正确
- 决策时刻 10:00（mod=600）：bar30m 窗口末尾 = 600 结束，bar5m 窗口末尾 = 600 结束
- 高频分支（5min）不看到低频分支（30min）的未来

### 4. 跨日/跨月 label 无泄露
- 10-31 最后一根 bar（mod=900, 15:00）label = NaN（正确，无未来窗口）
- 10-31 mod=870 的 label 用 10-31 mod=900 计算，**未用 11-01 数据**
- `groupby([inst,day])` 组内 shift 确保 label 不跨交易日

### 5. 月初窗口的跨月回溯是"合法"的
- `_accumulate` 每月多读 PAD_DAYS=10 天缓冲，使月初窗口能回看上月末数据
- `make_anchors` 的 `inst[i]==inst[i-seq_len]` 确保窗口首尾同股票，防止跨股票污染
- 这是设计允许的（`input_same_day_only=False`），非泄露

## 结论

**多频模型的数据管道无数据泄露。** 5min 分支不会看到 30min 分支的未来信息，
标签严格用决策时刻之后的未来窗口，且不跨交易日。跨月缓冲是合法的跨日回溯。

## 补充说明

排查过程中确认了一个**设计正确性细节**：
- 开盘初期的样本（如每天 mod=565/600/630/660 部分）因 `seq_len` 要求足够历史 bar 而被**正确丢弃**，不是 bug。
- `_subset_freq` 不切 feat（只切 keys/y/anchors），anchors 行号仍指向完整 feat，跨月 `off` 偏移量计算正确。
