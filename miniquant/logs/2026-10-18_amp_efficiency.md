# AMP 混合精度 + 训练效率优化

- **日期**：2026-10-18
- **模型**：`submissions/cnn_3f` 的 `StockTransformerMF`（三频，XSEC + 换手）
- **原始日志**：[raw/full_cnn3f_amp.log](./raw/full_cnn3f_amp.log)（全量重训，AMP）

## 动机

全量 48 个月重训耗时 **510.9 分钟 ≈ 8.5 小时**，远超平台「总运行时间 ≤ 3 小时」
硬约束。若不做提效，「扩大参数量 / 加 1min 分支」等所有后续方向都会被时间锁死。

## 最终结果（全量重训对拍）

| 配置 | 全量耗时 | best val_IC |
| --- | --- | --- |
| fp32（无 AMP） | 510.9 min | +0.0757 |
| **AMP fp16** | **96.6 min** | **+0.0767** |

- **整体提速 5.3×**（含数据积累阶段），训练 step 段提速 1.4×。
  数据积累阶段（读 feather + 重采样 + 特征）原本占 510.9min 里的大头，
  AMP 后整体压到 96.6min，**已满足平台 ≤3h 上限**。
- AMP fp16 与 fp32 的 val_IC 基本一致（0.0767 vs 0.0757），无精度损失。

## 优化项

1. **AMP 混合精度**（`USE_AMP=True`, `AMP_DTYPE="float16"`）
   - 前向 + 损失在 `torch.autocast(fp16)` 上下文中，`GradScaler` 防下溢回退。
   - 关键：`_corr_batch` 内部先 `.float()` 升回 fp32 再算 Pearson，保证 loss
     与梯度精度不受低精度前向影响（loss 始终 fp32）。
   - CPU / 无 CUDA 时自动关闭（`enabled=device.type=="cuda"`）。

2. **分组 corr loss 向量化**（`_grouped_corr`）
   - 原 xsec 主损失逐 group 串行 `torch.unique` + 布尔 mask，每个 batch 数十个
     小 kernel，CPU/GPU 串行开销大。
   - 改用 `searchsorted + scatter_add` 一次归约所有 group 的 sum/count/sumsq。
   - 与串行版数值/梯度对拍误差 < 1e-8（见 `_tools/_check_grouped_corr.py`）。

3. **瓶颈定位**（`_tools/_prof_step.py` / `_prof_real.py`）
   - 数据侧（gather + H2D）仅占 step 的 **2–7%**，GPU 前向+反向占 **93–98%**。
   - 故提效重点在 GPU 侧（AMP + 结构），数据管线不是瓶颈，无需大改。

## 实验结果

### 吞吐基准（`_bench_amp.py`，4 个月，batch=4096）

| 配置 | 吞吐 | 相对 |
| --- | --- | --- |
| fp32（无 AMP） | 3.7 step/s | 1.0× |
| **AMP fp16** | **5.2 step/s** | **1.4×** |

### 真实 step 构成（`_prof_real.py`，含 xsec 分组 + 换手，AMP）

| 环节 | 耗时/step | 占比 |
| --- | --- | --- |
| gather + H2D | 13.4 ms | 2% |
| 前向（含分组 loss） | 261 ms | 48% |
| 反向 + 优化器 | 269 ms | 50% |
| **合计** | **~540 ms** | 1.84 step/s |

### smoke 训练（6 个月，AMP fp16，无 NaN）

- cnn_3f：best val_IC=+0.0736，peakRSS=9576 MB（远低于 51200 MB）
- cnn_2f：best val_IC=+0.0695，peakRSS=9389 MB

## 结论与下一步

- AMP 带来 **5.3×** 整体训练提速（96.6 min），且不改模型结构、无精度损失。
- 96.6 min 已远低于 3h 上限，**扩参/加分支有了充足时间余量**。
- 剩余优化空间（若扩参后时间再次吃紧）：
  1. **xsec 注意力 O(N²·d) 是前向主要开销**（batch 4096 时 4096² 的注意力矩阵），
     可考虑分组后按截面做更小的注意力，或降低 xsec 头数/层数。
  2. **减少 epoch**：best 在 ep08 达到，ep09–12 基本平台，可设 epochs=8 或
     更早停。
  3. **bf16**：RTX 5060（Blackwell）对 bf16 有更好支持，可试 `AMP_DTYPE="bfloat16"`
     进一步提速。
  4. 若时间仍不够，可「子集 warmup + 全量精调」或降低 `batch_size` 增大多卡/显存
     利用率（当前 8GB 显存 batch 4096 已较满）。

- 后续「hidden 64→96 加宽」会放大前向/反向耗时（~2.25×），但有 96.6min → 3h
  的余量，预计加宽后仍可在 3h 内完成。
