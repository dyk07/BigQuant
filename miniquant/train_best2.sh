#!/usr/bin/env bash
# 一键流程：全量训练(2020-2023) -> 样本外测试集评估(2024H2)
#
#   bash train_best2.sh            # 后台分离运行（推荐，终端关掉也不受影响）
#   bash train_best2.sh --fg       # 前台运行（直接看输出）
#
# 产出：
#   checkpoints/best2.pt   最佳模型（含 mean/std/clip 等元信息）
#   full2.log              完整日志
#
# 监控：
#   tail -f full2.log             # 实时日志
#   grep -E "epoch|test\]|done" full2.log
set -e

# ---- 后台模式：用 setsid 重新拉起自己，脱离终端 ----
if [ "${1:-}" != "--inner" ]; then
  LOG="$(cd "$(dirname "$0")" && pwd)/full2.log"
  if [ "${1:-}" = "--fg" ]; then
    exec bash "$0" --inner
  fi
  setsid nohup bash "$0" --inner > "$LOG" 2>&1 < /dev/null &
  echo "已在后台启动 (pid=$!)"
  echo "日志: $LOG"
  echo "监控: tail -f full2.log"
  exit 0
fi

# ---- 实际工作 ----
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bigquant
cd "$(dirname "$0")"

OUT=checkpoints/best2.pt

echo "===== $(date '+%F %T') 开始训练 ====="
echo "配置: bar30m | target=rank | loss=corr | epochs=30 | patience=5"
echo "训练 202001-202312 (48个月) -> 验证 202401-202406"

python -u train.py \
  --freq bar30m \
  --target rank \
  --loss corr \
  --epochs 30 \
  --patience 5 \
  --out "$OUT"

echo
echo "===== $(date '+%F %T') 开始测试集评估 (2024H2) ====="
python -u test.py --ckpt "$OUT" --freq bar30m

echo
echo "===== $(date '+%F %T') 全部完成 ====="
