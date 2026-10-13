# 提交说明 (BigQuant 量化大赛 · 端到端赛道)

本目录是可直接上传平台的**完整模型包**。平台要求**单层目录**, 因此每个 `cnn_*`
子目录内只放文件、不嵌套子目录:

```
<提交目录>/
  Transformer_modelsave_train.py     # 训练脚本 (数据 + 模型 + 训练循环, 单一事实来源)
  Transformer_modelsave_predict.py   # 推理脚本 (notebook 入口, main(datasources, start_date, end_date))
  config.py                          # 超参数配置 + 随机种子
  requirements.txt                   # 依赖声明
  transformer_model.json             # 已训练模型权重 (+ 标准化统计 + 结构超参)
```

## 提交清单对照 (平台要求 ↔ 本包)

| 平台要求 | 对应文件 |
|---|---|
| 训练脚本 | `Transformer_modelsave_train.py` |
| 推理脚本 | `Transformer_modelsave_predict.py` |
| 依赖声明 | `requirements.txt` |
| 超参数配置 | `config.py` (`MODEL_CFG` / `TRAIN_CFG`) |
| 随机种子 | `config.py` (`SEED`), 训练脚本中固定 Python/NumPy/PyTorch 种子; 并写入 `transformer_model.json` 的 `seed` 字段 |
| 已训练模型权重 | `transformer_model.json` |

上传时把**该目录内的全部文件**一起提交 (平铺, 不要带子目录)。

## 目录一览

| 子目录 | 模型 | 频率 | 参数量 | val_IC | 2024H2 test_IC |
|---|---|---|---|---|---|
| `cnn_1f_bar30m` | CNN+Transformer 单频 | bar30m | 187K | +0.0473 | +0.0541 |
| `cnn_2f` | CNN+Transformer 双频 | bar30m+bar5m | 395K | +0.0746 | +0.0813 |
| `cnn_3f` | CNN+Transformer 三频 | bar30m+bar15m+bar5m | 590K | +0.0782 | +0.0842 |

> 建议优先提交 **`cnn_3f`** (IC 最高) 与 **`cnn_2f`** (性价比最高, 参数少 33%)。

## 超参数 (config.py)

- **模型**: hidden=64, conv_blocks=3 (dilation 1/2/4), kernel=3, transformer_layers=2,
  heads=4, dropout=0.2, fused_dim=64
- **训练**: epochs=12, lr=1e-3, weight_decay=1e-5, batch_size=4096, patience=4 (early stop),
  grad_clip=1.0, loss=corr (1−Pearson), target=rank (截面分位 → [−1,1]), val_months=2
- **窗口**: lookback=120 分钟, horizon=30 分钟, 采样网格=30 分钟 (09:25 + 整/半点)
- **随机种子**: 42

## 关键设计

- **只依赖 1 分钟 K 线** (`cpt_jyc_2026_e2e_bar1m`)。模型所需的 5/15/30 分钟视角在
  `build_dataset` 内部由 bar1m 重采样得到。已实测该重采样与原生 bar5m/15m/30m
  **数值逐位一致**, 因此提交物自包含, 不依赖外部多频表。
- **CNN 滑窗 + Transformer**: 1×1 Conv 投影 → 膨胀卷积残差块 → 正弦位置编码
  → TransformerEncoder (Pre-LN) → 注意力池化‖平均池化 → MLP head。
- **多频融合**: 每频一条独立编码分支, 池化后拼接, 残差式 head 输出标量。
- **无未来数据**: 输入窗口严格在决策时刻之前; 标签窗口严格在决策时刻之后且不跨交易日。
- **预处理一致性**: 标准化统计随权重落盘, 推理直接复用, 杜绝 train/infer 漂移。
- **合规**: 只用基础算子 (单位还原 / 量纲统一 / 对数变换 / 缺失值处理), 不做任何手工因子
  (滚动统计、买卖量不平衡、量价组合等一律不构造)。

## 复现方式

- **本地训练** (生成权重):
  ```bash
  python Transformer_modelsave_train.py     # 训练区间由 config.TRAIN_MONTHS 决定
  ```
- **公榜**: 平台只替换 `datasources / start_date / end_date` 并调用 `main`, 仅用提交的
  `transformer_model.json` 推理。
- **私榜**: 平台调用 `train_and_save(datasources)` 在隔离环境从零重训 (种子固定),
  用于合规性验证。

## 约束提醒

- 因子值越大越好 (模型输出已按此方向训练)。
- 不能缺失分钟, 每个截面缺失率需 ≤ 40%。
- 输出经 `cpt_jyc_2026_instruments` 对齐, 保证时间戳/截面合法。

## 本地构建工具 (不提交, 位于 `../_tools/`)

- `convert_checkpoints.py` — 把 `miniquant/checkpoints/*.pt` 转成各目录的
  `transformer_model.json` (含 seed / 超参元数据)。
- `verify_submission.py` — 用提交侧代码走完整推理链路, 在本地 feather 上复算 2024H2 IC 对拍。
