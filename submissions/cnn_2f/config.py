"""
提交包配置 —— 超参数 + 随机种子 (单一事实来源)。

本文件不依赖任何第三方库, 供训练脚本与推理脚本共同导入。
平台复现检查 / 私榜重训 均以本文件为准。
"""

# ======================================================================
# 随机种子 (Python / NumPy / PyTorch 全部固定, 保证可复现)
# ======================================================================
SEED = 42

# ======================================================================
# 数据与窗口
# ======================================================================
FREQS = ["bar30m", "bar5m"]      # 参与建模的频率 (由 bar1m 内部重采样得到)
LOOKBACK_MINUTES = 120           # 输入窗口回看时长 (分钟)
HORIZON_MINUTES = 30             # 预测未来 30 分钟 VWAP
SAMPLE_STEP_MINUTES = 30         # 日内采样网格 (09:25 + 整/半点)
INPUT_SAME_DAY_ONLY = False      # 输入窗口允许向前回看 (标签窗口仍严格当日)

# 训练区间 (数据只到 2024-12; 2024 年数据用于验证, 不参与最终训练)
TRAIN_MONTHS = [f"{y}{m:02d}" for y in range(2020, 2024) for m in range(1, 13)]

# ======================================================================
# 模型结构超参 (CNN 滑窗 + Transformer)
# ======================================================================
MODEL_CFG = dict(
    in_dim=25,            # 7 个 K 线特征 + 18 个盘口特征
    hidden=64,            # 卷积 / Transformer 隐藏维
    conv_blocks=3,        # 膨胀卷积残差块数 (dilation 1, 2, 4)
    kernel=3,             # 卷积核大小
    transformer_layers=2,  # TransformerEncoder 层数 (Pre-LN)
    heads=4,              # 多头注意力头数
    dropout=0.2,
    fused_dim=64,         # 融合 / head 隐含维
)

# ======================================================================
# 训练超参
# ======================================================================
TRAIN_CFG = dict(
    epochs=12,            # 最大 epoch
    lr=1e-3,              # Adam 初始学习率
    weight_decay=1e-5,
    batch_size=4096,
    patience=4,           # 早停 patience (监控验证集 IC)
    grad_clip=1.0,        # 梯度裁剪 max_norm
    val_months=2,         # 末尾 N 个月作为验证集
    lr_factor=0.5,        # ReduceLROnPlateau 衰减系数
    lr_patience=2,        # ReduceLROnPlateau patience
    loss="corr",          # 1 - Pearson (配合 rank 目标 = soft-IC)
    target="rank",        # 截面 rank 目标 -> [-1, 1]
)
