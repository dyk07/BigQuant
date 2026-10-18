# miniquant

端到端神经网络量化选股原型：输入原始量价 + 盘口快照，预测**未来 30 分钟 VWAP 收益**，
输出每只股票的截面打分，用 **截面 Spearman IC / ICIR** 评估排序能力。

> 约束：不手工构造技术指标、量价组合、盘口衍生指标。
> 只允许：单位还原、缺失值处理、对数变换、标准化、序列切窗、时间对齐。

---

## 1. 快速开始（WSL + conda）

本项目在 **WSL Ubuntu** 的 `bigquant` conda 环境里运行（Python 3.9 + torch 2.8 + CUDA）。

```bash
# 统一入口
./run.sh check                 # 数据/标签自检（bar30m, 202301）
./run.sh check bar5m 202306    # 指定频率/月份
./run.sh smoke                 # 小规模训练跑通（202301 -> 202302）
./run.sh train                 # 全量训练（2020-2023 训练 / 2024H1 验证）
./run.sh test                  # 用 checkpoints/best.pt 在 2024H2 上评估
./run.sh leak                  # 打乱标签的防泄漏对照
./run.sh all                   # check -> smoke -> leak
```

训练默认 `--target rank --loss corr`（理由见第 6 节），可覆盖：

```bash
./run.sh train --loss mse                 # 换回 MSE
./run.sh train --target raw               # 回归原始收益率
./run.sh train --freq bar5m               # 换频率
./run.sh train --lookback-minutes 240     # 拉长输入窗口
./run.sh train --loss combined            # MSE + corr 混合
```

也可以直接调用（等价）：

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate bigquant

python sanity_check.py --freq bar30m --month 202301
python train.py --smoke                              # 小规模
python train.py --freq bar30m --epochs 30            # 全量
python test.py --ckpt checkpoints/best.pt
python leak_control.py --month 202301 --seeds 5      # 防泄漏
```

---

## 2. 文件结构

| 文件 | 作用 |
| --- | --- |
| `config.py` | 路径、频率、窗口、模型、训练超参；分钟 → bar 的换算 |
| `data_loader.py` | 读取 feather、`分→元`、`-1→NaN`、剔除集合竞价、打交易日标签 |
| `features.py` | 特征构造（量纲统一 + 对数变换）、标签构造、截断边界 |
| `dataset.py` | 向量化切窗 `make_samples`、标准化 `fit_scaler/apply_scaler`、`StockDataset` |
| `model.py` | `E2EModel`：双流（K 线 GRU + 盘口 MLP）+ 注意力池化 |
| `losses.py` | 损失函数：`mse` / `corr`（相关性）/ `ic`（soft-IC）/ `combined` |
| `metrics.py` | 截面 Spearman IC / ICIR / 显著性 t / `make_targets`（rank 目标） |
| `pipeline.py` | 训练/验证/测试共用的数据管线 + 防泄漏自检 |
| `train.py` | 训练主脚本（时间切分、MSE、按验证集 IC 早停） |
| `test.py` | 加载 checkpoint，在测试月份算 IC / 日频 IC 稳定性 |
| `sanity_check.py` | 数据与标签自检（不训练） |
| `leak_control.py` | 打乱标签的防泄漏对照实验 |
| `run.sh` | WSL 环境下的统一命令入口 |

---

## 3. 数据事实（已核实）

- 根目录：`/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant/e2e_data`
- 表：`cpt_jyc_2026_e2e_bar1m|bar5m|bar15m|bar30m`，文件 `YYYYMM.0.feather`（202001–202412）
- 单位：`open/high/low/close/amount/盘口价格` 原始为**分**（int32），读取后 `/100`
- `volume` 单位是**股**，不缩放
- `-1` 只出现在 OHLC，表示缺失 → 转 `NaN`
- 有 `adjust_factor`：同一交易日内为常数（除权日之间才变化）。本项目的窗口与标签
  都不跨交易日，因此无需额外复权处理
- 每行 28 列，含 `date`、`instrument_id`，约 1000 只股票
- `bar1m` 压缩包仍为 `cpt_jyc_2026_e2e_bar1m.tar.gz`（约 8.5 GB），尚未解压

### 时间戳（K 线**结束**时刻）

| 频率 | 全部时间戳 | 有效 bar/日 | 采样时刻（截面） |
| --- | --- | --- | --- |
| bar30m | 09:25, 10:00, 10:30 … 15:00 | 9 | ← 同右 |
| bar15m | 09:25, 09:45 … 15:00（每 15 分） | 17 | ← 同右 |
| bar5m  | 09:25, 09:35 … 15:00（每 5 分） | 49 | ← 同右 |
| bar1m  | **09:16–09:24（盘前）**, 09:25, 09:31 … 15:00 | 241 | ← 同右 |

- **保留 09:25 集合竞价 bar**：实测四个频率都有该 bar，约 40% 股票有真实
  成交（`volume>0`）。它是平台"开盘截面"（用集合竞价数据预测未来 30 分钟
  VWAP），**必须保留**。09:25 不在 30 分钟网格上，采样时显式加入。
- **只剔除 bar1m 的 09:16–09:24 盘前 bar**（`volume=0`、`amount=0`，close
  与 09:25 相同，是集合竞价过程中的指示性价格，不是成交）。
- `bar5m` / `bar1m` 连续竞价首根从 `09:35` / `09:31` 开始（`09:30` 缺一根带盘口的 bar）
- 采样按 `minute_of_day`（自午夜起分钟数）对齐，**不依赖每只股票的行号**：
  缺 bar / 停牌不会让采样时刻漂移，且四种频率在**同一时刻**采样
- 四个频率的采样截面完全一致：
  `09:25, 10:00, 10:30, 11:00, 11:30, 13:30, 14:00, 14:30, 15:00`
  即每个交易日 **8 个可用日内截面**（15:00 无未来窗口，标签为 NaN，自动剔除），
  与平台"每个交易日约 8 个日内截面"一致

---

## 4. 标签定义

未来 `horizon_minutes`（默认 30）的收益：

$$\text{future\_vwap} = \frac{\sum_{k=t+1}^{t+h} \text{amount}_k}{\sum_{k=t+1}^{t+h} \text{volume}_k},
\qquad \text{label} = \frac{\text{close}_{t+h}}{\text{future\_vwap}} - 1$$

其中 $\text{close}_{t+h}$ 是未来窗口最后一根 bar 的收盘价（= end_price），
$h = \text{horizon\_bars} = \text{horizon\_minutes} / \text{bar\_minutes}$：

| 频率 | h（bar 数） |
| --- | --- |
| bar30m | 1 |
| bar15m | 2 |
| bar5m | 6 |
| bar1m | 30 |

✅ **与平台一致**：`label = end_price / future_vwap - 1`，与平台
`vwap_return = end_price / vwap - 1` 定义完全相同，训练与推理方向统一，
无需取负/取反纠正。

约束：
- 窗口**不跨交易日**（按 `instrument_id + day` 分组）
- 仅要求窗口内**存在成交**（`fwd_vol > 0`），**不要求**每根 bar 都有成交：
  零成交分钟对 VWAP 的分子/分母贡献都是 0，不影响结果。该放宽使 bar1m
  标签有效率从 74.8% 提升到 **87.4%**（四种频率现均为 ~87.5%）
- 累积求和用 **float64**：`amount`/`volume` 加载时被压成 float32，
  若直接 `cumsum`，大数相减会产生 float32 抵消误差（实测标签相对误差 ~5e-5）
- 尾部不足 h 根的 bar 自动 `NaN`
- 极端值按训练集 **1% / 99%** 分位截断（避免用全量数据造成泄漏）

> **注意**：原代码把 `horizon=30` 当作 30 根 bar 传给 `build_labels`，对 bar30m 意味着
> 900 分钟。现已改为按分钟换算，保证"未来 30 分钟"与频率无关。

---

## 5. 切窗与防泄漏

- 采样点：按**时钟**对齐（`config.aligned_minutes()`），**四种频率完全一致**：

  | 频率 | `seq_len_bars` T | 采样时刻（每天 7 个） |
  | --- | --- | --- |
  | bar30m | 4 | 10:00, 10:30, 11:00, 11:30, 13:30, 14:00, 14:30 |
  | bar15m | 8 | 同上 |
  | bar5m | 24 | 同上 |
  | bar1m | 120 | 同上 |

  （15:00 因没有下一根 bar、标签为 NaN，故不采样）

- 输入窗口 `[t - seq_len, t)`，标签窗口 `[t, t + h)`，**严格不重叠**
- **标签窗口**必须与当前 bar 同一天（`build_labels` 按 `instrument_id + day` 分组）
- **输入窗口**是否允许回溯前一日由 `cfg.input_same_day_only` 控制：

  | 取值 | 含义 | 适用 |
  | --- | --- | --- |
  | `False`（默认） | 输入可跨日回溯（仍只是一连串**历史** bar） | 全部频率 |
  | `True` | 输入必须落在当天内 | 需要严格日内语义时 |

  单月样本数（202301，实测）：

  | 频率 | `input_same_day_only=True` | `False`（默认） |
  | --- | --- | --- |
  | bar30m | 47,925（每日 3 个采样点） | **107,829**（每日 7 个） |
  | bar15m | 47,925 | **107,829** |
  | bar5m | 47,901（每日仅 3 个） | **107,829** |
  | bar1m | — | **107,817** |

  跨日回溯只影响**输入**；标签仍然严格使用**未来**且在当天内，因此不引入未来信息。

- 采样间隔：`sample_step_minutes`（默认 30）
- 输入窗口覆盖历史时长：`lookback_minutes`（默认 120）
- 标准化 `mean/std` **只用训练集** fit；训练集/验证集/测试集按**时间**切分

### 防泄漏自检

| 手段 | 命令 | 判定 |
| --- | --- | --- |
| 窗口结构 + **标签重算比对** | `python sanity_check.py` | 四频率全 PASS |
| 标签窗口跨日计数 | `sanity_check.py` [4] | 0 |
| 打乱标签后 IC ≈ 0 | `python leak_control.py` | 跨种子 \|t\| < 2 |

其中"标签重算比对"是最强校验：对随机抽样的 200 个样本，用 `sum(amount)/sum(volume)`
当场重算标签并与 `build_labels` 的存值比对（`rtol=1e-6`），能同时抓出窗口错位、
跨日、午休跨越、精度损失等问题。

打乱标签实验是**唯一能证伪泄漏**的手段：若管线偷看了未来，随机标签也能"预测"出正 IC。

---

## 6. 模型与训练目标

```
输入 [B, T, 7 (K线)] ─→ Linear → GRU(2层) → LayerNorm ─┐
                                                    ├─→ Concat → Linear → LayerNorm
输入 [B, T, 18 (盘口)] ─→ MLP (逐时间步共享) → LayerNorm ─┘
                                                    ↓
                        注意力池化 ‖ 平均池化 → Dropout → MLP → 标量分数
```

约 6.9 万参数；输出线性标量（用于截面排序，不加激活）。

### 为什么用 rank 目标 + 相关性损失

比赛指标是**截面 IC（秩相关）**，而直接回归收益率的 MSE 有两个问题：

1. 标签均值≈0，MSE 被尺度项主导，梯度几乎不含排序信息
2. 模型极易退化成**常数输出**（IC 精确为 0）

实测（同一个 smoke 设定，202301 → 202302）：

| 配置 | train_IC | val IC |
| --- | --- | --- |
| `--target raw --loss mse`（原方案） | 0.0000 | -0.0001（退化成常数） |
| `--target rank --loss mse` | +0.047 | +0.008 |
| `--target rank --loss corr` | +0.070 | **+0.011** |

因此默认使用：

- **目标** `--target rank`：把标签转成**截面百分位排名**（`(pct-0.5)*2`），
  梯度只关心“谁排在谁前面”，与比赛目标一致
- **损失** `--loss corr`：`1 - Pearson(pred, rank_target)`，即 soft-IC

IC 评估始终使用**原始收益率标签**，与目标变换无关。

训练细节：Adam + `ReduceLROnPlateau`、梯度裁剪 1.0、按验证集 IC 早停。

---

## 7. 评估指标

按 `(day, bar_idx)` 分截面，每截面算 Spearman IC，再对所有截面取平均：

- **IC**：均值（排序能力）
- **IC_std**：截面间标准差（稳定性）
- **ICIR** = `IC / IC_std`
- **ICIR_ann** = `ICIR × sqrt(截面数)`（年化口径）
- **SE / t**：`SE = IC_std / sqrt(n)`，`t = IC / SE`；`|t| > 2` 才算显著

---

## 8. 实测结果（bar30m）

训练 2020–2023（48 个月，290 万样本），验证 2024H1，测试 2024H2
（`--target rank --loss corr`，早停于第 8 轮）：

| 数据集 | IC | IC_std | ICIR_ann | t | 截面数 |
| --- | --- | --- | --- | --- | --- |
| 验证 2024H1 | +0.0422 | 0.1119 | +7.06 | +7.06 | 351 |
| **测试 2024H2** | **+0.0461** | 0.1392 | +6.42 | +6.42 | 375 |

日频稳定性（2024H2，125 个交易日）：日 IC 均值 +0.0342，正 IC 比例 **62.4%**。

> 测试集 IC **高于**验证集，说明没有过拟合；t≈6.4 表示显著为正。

---

## 9. 复现检查清单

1. `./run.sh check` → `SANITY CHECK PASSED`
2. `./run.sh smoke` → 能跑通并产出 `checkpoints/smoke.pt`
3. `./run.sh leak` → 跨种子 `|t| < 2`（与 0 无显著差异）
4. `./run.sh train` → 记录最佳验证集 IC
5. `./run.sh test` → 记录测试集 IC 与日频 IC 正比例

---

## 10. 后续可扩展方向

- 多频率融合（bar5m + bar30m 双输入）
- 排序损失升级（pairwise ranking / listwise）
- 盘口增强（更多档位、逐笔数据）
- 解压 `bar1m` 做 1 分钟粒度实验
- 滚动重训 + 集成
