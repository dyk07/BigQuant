# 量化比赛项目上下文

## 1. 比赛目标
端到端神经网络，输入原始分钟级量价 + 盘口快照，预测未来 30 分钟 VWAP 收益。
模型输出每只股票一个分数，比赛按截面 IC / ICIR 评估排序能力。
禁止手工因子、技术指标、盘口衍生指标。

## 2. 数据（已核实）
- 根目录：/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant/e2e_data
- 表名：
  - cpt_jyc_2026_e2e_bar1m      ← 仍是 .tar.gz（约 8.5 GB），未解压
  - cpt_jyc_2026_e2e_bar5m
  - cpt_jyc_2026_e2e_bar15m
  - cpt_jyc_2026_e2e_bar30m
- 文件：YYYYMM.0.feather（202001 – 202412，共 60 个月）
- 本地表特点：
  - open/high/low/close/amount/盘口价格 为 int32，单位“分”，读取后 /100
  - volume 单位是“股”，不缩放
  - OHLC 缺失值为 -1，需先转 NaN，再 /100
  - 只有前 3 档盘口
  - 没有 pre_close、没有字符串 instrument
  - 有 instrument_id（约 1000 只股票，int16）
  - 有 adjust_factor：同一交易日内为常数（除权日之间才变化）；
    本项目的窗口/标签都不跨交易日，无需额外复权
- 时间戳：K 线 bar 结束时刻

### 时间点（已核实，均按**时钟**处理）
| 频率 | 全部时间戳 | 有效 bar/日 | 采样时刻（全部一致） |
| --- | --- | --- | --- |
| bar30m | 09:25, 10:00, 10:30, 11:00, 11:30, 13:30, 14:00, 14:30, 15:00 | 9 | ← 同右 |
| bar15m | 09:25, 09:45 … 15:00（每 15 分） | 17 | ← 同右 |
| bar5m  | 09:25, 09:35 … 15:00（每 5 分） | 49 | ← 同右 |
| bar1m  | **09:16–09:24**（盘前）, 09:25, 09:31 … 15:00 | 241 | ← 同右 |

- **09:25 集合竞价 bar 必须保留**：实测四个频率都有该 bar，约 40% 股票有
  真实成交（volume>0）。它是平台"开盘截面"（用集合竞价数据预测未来 30 分钟
  VWAP），因此保留。
- **只剔除 bar1m 的 09:16–09:24 盘前 bar**（volume=0、amount=0，close 与
  09:25 相同，是指示性价格），按**时钟**判断而非行号。
- bar5m/bar1m 连续竞价首根从 09:35/09:31 开始（09:30 缺一根带盘口的 bar）
- 采样一律按 `minute_of_day`（自午夜起的分钟数），不依赖每只股票的行号，
  因此缺 bar/停牌不会导致采样时刻漂移，且**四种频率在同一时刻采样**：
  09:25, 10:00, 10:30, 11:00, 11:30, 13:30, 14:00, 14:30, 15:00
  即每个交易日 **8 个可用日内截面**（尾盘 15:00 无未来窗口，标签为 NaN 故不采）

## 3. 标签
未来 30 分钟收益（end_price = 未来窗口最后一根 bar 的收盘价）：
- future_vwap = sum(amount[t+1..t+h]) / sum(volume[t+1..t+h])
- **本地训练 label = close[t+h] / future_vwap - 1**（分子是 end_price）
- h = horizon_minutes / bar_minutes → bar30m:1, bar15m:2, bar5m:6, bar1m:30
- ✅ **与平台 vwap_return = end_price / vwap - 1 定义完全一致**，
  训练与推理方向统一，无需取负/取反纠正。
- 窗口不跨交易日；仅要求窗口内**存在成交**（fwd_vol > 0），
  不要求每根 bar 都有成交（零成交分钟对 VWAP 分子分母贡献均为 0；
  该放宽使 bar1m 标签有效率从 74.8% 升到 87.4%）
- 累积求和用 **float64**（amount/volume 在加载时是 float32，
  直接 cumsum 会产生 float32 抵消误差，实测标签相对误差 ~5e-5）
- 极端值按**训练集** 1% / 99% 分位截断

## 4. 项目结构
miniquant/
├── config.py          # 路径/频率/窗口/超参；分钟→bar 换算；对齐采样点
├── data_loader.py     # 读 feather、分→元、-1→NaN、剔除集合竞价、打 day/bar_idx
├── features.py        # build_features / build_labels / compute_clip_bounds
├── dataset.py         # 向量化 make_samples / fit_scaler / apply_scaler / StockDataset
├── model.py           # E2EModel：双流 GRU+MLP + 注意力池化
├── metrics.py         # 截面 Spearman IC / ICIR / 显著性 t
├── pipeline.py        # 共用数据管线 + 防泄漏自检 + 打乱标签对照
├── train.py           # 训练（时间切分、MSE、按验证集 IC 早停）
├── test.py            # 加载 checkpoint 做测试集评估
├── sanity_check.py    # 数据/标签自检（不训练）
├── leak_control.py    # 打乱标签的防泄漏对照
├── run.sh             # WSL 统一命令入口
└── README.md          # 完整使用文档

## 5. 运行环境
- **WSL Ubuntu** + conda 环境 `bigquant`
- Python 3.9.25, torch 2.8.0+cu128, CUDA 可用（RTX 5060 Laptop 8GB）
- 统一入口：`./run.sh check|smoke|train|test|leak|all`

## 6. 待办完成情况
- [x] 剔除 09:25 集合竞价 **及 bar1m 的 09:16–09:24 盘前 bar**
- [x] 标记每天采样点（按 `minute_of_day` 时钟对齐，四频率同步）
- [x] 标签改为按未来 30 分钟对齐（按频率换算 bar 数）
- [x] 确认 15:00 样本处理（标签 NaN，自动剔除）
- [x] 确认各频率时间戳分布（含 bar1m 盘前 bar、bar5m/bar1m 缺 09:30）
- [x] 小规模跑通：202301 训练，202302 验证
- [x] 防泄漏检查：打乱标签后 IC 接近 0（跨种子 |t|<2 通过）
- [x] 向量化 make_samples + 分块 scaler（支持 677 万样本不 OOM）
- [x] 四频率 sanity check 全部 PASS（bar1m/5m/15m/30m）
- [x] rank 目标 + corr 损失（修复 MSE 退化成常数导致 IC=0）
- [ ] 全量训练（2020–2023 训练 / 2024H1 验证）
- [ ] 测试集评估（2024H2）
- [ ] 多频率融合、排序损失、盘口增强

## 7. 提交包内存设计（平台硬限制）
平台会议 / 答疑结论：**公榜（评测）容器内存上限 51200MB，私榜重训上限 30GB + 3h**，
超限**强制终止**。因此提交包必须全程低常驻、低尖峰。

`submissions/*/Transformer_modelsave_train.py` 采用的策略：
1. **只 SELECT 需要的 28 列**（`NEEDED_COLS`），并在 `_pick_cols` 里 `_shrink`
   到最小 dtype（OHLC/盘口价格 int32、volume/amount int64 保留、其余按需）；
2. **按自然月分块读取**（`_build_month`）：读一块 → 建特征/标签 → `del raw, per`
   → `gc.collect()` → `_trim()`；
3. **不物化 `[N,T,25]` 滑窗张量**：只保留 `(特征矩阵, 锚点下标)`，窗口在
   `_batch_windows` 中**按 batch 现场 gather**（峰值 O(batch·T·F)）；
4. 标准化为逐列仿射，作用于**特征矩阵**一次，与逐窗口等价但内存降为 `1/T`；
5. 累积出的月特征矩阵用 **`_concat_inplace`** 合并（先分配结果、逐块就地拷入
   并立即释放源块），取代 `np.concatenate`（拼接瞬间 ≈ 2× 结果）的瞬时尖峰；
6. 每月末调用 **`_trim()`**（`malloc_trim(0)`）把 glibc 空闲 arena 归还内核，
   实测常驻 RSS 由 ~4.2GB 降到 ~2.3–3.0GB。

本地实测（`_tools/verify_all.sh`，单月 `main` 评测路径）：
`Maximum resident set size` = 4242432 / 4305668 / 4351412 kbytes（1f/2f/3f），
远低于 51200MB 公榜限制。

## 7. 关键规则
- 标准化统计量只用训练集
- 按时间切分，不能随机切分
- 输入窗口与标签窗口严格不重叠；**标签窗口不跨交易日**
- 禁止手工构造技术指标、统计量、价量组合、盘口衍生指标
- 允许：单位还原、缺失值处理、对数变换、标准化、序列切窗、时间对齐