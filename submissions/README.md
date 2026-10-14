# 提交说明 (BigQuant 量化大赛 · 端到端赛道)

## 提交方式

平台要求 **提交压缩包内有且只有一个 notebook**。因此每个 `cnn_*` 文件夹即为一份
完整提交物: 打包时**只把该文件夹内的文件压到压缩包根目录** (平铺, 不要文件夹套壳),
压缩包内恰好 1 个 `.ipynb` 加若干附属文件。

```
<包名>/                          → 打包后 <包名>.zip
  Transformer_modelsave_predict.ipynb   # 唯一的 notebook (含 main, 自包含全部代码)
  Transformer_modelsave_train.py        # 训练脚本 (私榜重训导入 train_and_save)
  config.py                             # 超参数配置 + 随机种子
  transformer_model.json                # 已训练模型权重
  requirements.txt                      # 依赖声明
```

| 提交包 | 模型 | 频率 | 参数量 | val_IC | 2024H2 test_IC |
|---|---|---|---|---|---|
| `cnn_1f_bar30m` | CNN+Transformer 单频 | bar30m | 187K | +0.0473 | +0.0541 |
| `cnn_2f` | CNN+Transformer 双频 | bar30m+bar5m | 395K | +0.0746 | +0.0813 |
| `cnn_3f` | CNN+Transformer 三频 | bar30m+bar15m+bar5m | 590K | +0.0782 | +0.0842 |

> 建议优先提交 **`cnn_3f`** (IC 最高) 与 **`cnn_2f`** (性价比最高, 参数少 33%)。

## 平台要求 ↔ 对应内容

| 平台要求 | 位置 |
|---|---|
| **有且仅有一个 Notebook** | `Transformer_modelsave_predict.ipynb` |
| **可调用的 `main(datasources, start_date, end_date)`** | 该 notebook 中 (从 `datasources` 取表, 不硬编码物理表名) |
| **训练脚本** | 内联进 notebook (数据管线 + 模型定义 + `train_and_save`) + 附 `Transformer_modelsave_train.py` |
| **推理脚本** | 内联进 notebook (`main`) |
| **依赖声明** | `requirements.txt` |
| **超参数配置** | notebook 第 2 个代码单元格 (`CFG` = 原 `config.py` 全部内容) + 附 `config.py` |
| **随机种子** | `CFG.SEED = 42`; 训练脚本固定 Python/NumPy/PyTorch/CUDA 种子; 权重 json 亦含 `seed` |
| **已训练模型权重** | `transformer_model.json` (文本 JSON, 非 .pt 二进制) |

## 目录内文件说明

```
<包名>/
  Transformer_modelsave_train.py     # 训练脚本 (数据 + 模型 + 训练循环, 单一事实来源)
  Transformer_modelsave_predict.py   # 推理脚本 (main 的原型)
  config.py                          # 超参数配置 + 随机种子
  requirements.txt                   # 依赖声明
  transformer_model.json             # 已训练模型权重
  Transformer_modelsave_predict.ipynb  # 由上述源文件生成的单 notebook 提交物
```

## 超参数 (config.py)

- **模型**: hidden=64, conv_blocks=3 (dilation 1/2/4), kernel=3, transformer_layers=2,
  heads=4, dropout=0.2, fused_dim=64
- **训练**: epochs=12, lr=1e-3, weight_decay=1e-5, batch_size=4096, patience=4 (early stop),
  grad_clip=1.0, loss=corr (1−Pearson), target=rank (截面分位 → [−1,1]), val_months=2
- **窗口**: lookback=120 分钟, horizon=30 分钟, 采样网格=30 分钟 (09:25 + 整/半点)
- **随机种子**: 42

## 关键设计

- **只依赖 1 分钟 K 线** (`datasources["bar1m"]`)。模型所需的 5/15/30 分钟视角在
  `build_dataset` 内部由 bar1m 重采样得到。已实测该重采样与原生 bar5m/15m/30m
  **数值逐位一致**, 因此提交物自包含, 不依赖外部多频表。
- **CNN 滑窗 + Transformer**: 1×1 Conv 投影 → 膨胀卷积残差块 → 正弦位置编码
  → TransformerEncoder (Pre-LN) → 注意力池化‖平均池化 → MLP head。
- **多频融合**: 每频一条独立编码分支, 池化后拼接, 残差式 head 输出标量。
- **无未来数据**: 输入窗口严格在决策时刻之前; 标签窗口严格在决策时刻之后且不跨交易日。
- **预处理一致性**: 标准化统计随权重落盘, 推理直接复用, 杜绝 train/infer 漂移。
- **数据读取带日期过滤**: 只查询 `[start_date - 10 天, end_date]`, 避免整表读入内存。
- **合规**: 只用基础算子 (单位还原 / 量纲统一 / 对数变换 / 缺失值处理), 不做任何手工因子
  (滚动统计、买卖量不平衡、量价组合等一律不构造)。

## 复现方式

- **本地训练** (生成权重): 运行 notebook 内的 `train_and_save(datasources)`,
  训练区间由 `CFG.TRAIN_MONTHS` 决定。
- **公榜**: 平台只替换 `datasources / start_date / end_date` 并调用 `main`, 仅用提交的
  `transformer_model.json` 推理 (不重训)。
- **私榜**: 平台调用 `train_and_save(datasources)` 在隔离环境从零重训 (种子已固定),
  用于合规性验证。

## 约束提醒

- 因子值越大越好 (模型输出已按此方向训练)。
- 不能缺失分钟, 每个截面缺失率需 ≤ 40%。
- 输出经 `cpt_jyc_2026_instruments` 对齐, 保证时间戳/截面合法。
- 总运行时间 ≤ 3 小时; 评测环境无法访问外部网络。

## 本地构建/验证工具 (不提交, 位于 `_tools/`)

- `convert_checkpoints.py` — 把 `miniquant/checkpoints/*.pt` 转成各目录的
  `transformer_model.json` (含 seed / 超参元数据)。
- `build_notebook.py` — 把每个包的源文件打包成**单个 notebook**。
- `verify_notebook.py` — 顺序执行 notebook 全部单元格并调用 `main()`, 验证输出格式。
- `verify_submission.py` — 用提交侧代码走完整推理链路, 在本地 feather 上复算 2024H2 IC 对拍。
