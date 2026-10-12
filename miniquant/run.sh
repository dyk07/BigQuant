#!/usr/bin/env bash
# 统一入口：在 WSL 的 bigquant conda 环境里运行本项目。
#
#   ./run.sh check                 # 数据/标签自检（bar30m, 202301）
#   ./run.sh check bar5m 202306    # 指定频率/月份
#   ./run.sh smoke                 # 小规模训练跑通（202301 -> 202302）
#   ./run.sh train                 # 全年训练（2020-2023 训练 / 2024H1 验证）
#   ./run.sh test                  # 用 checkpoints/best.pt 在 2024H2 上评估
#   ./run.sh leak                  # 打乱标签的防泄漏对照
#   ./run.sh all                   # check -> smoke -> leak
#
# 训练默认：target=rank（截面百分位排名）+ loss=corr（相关性损失），
# 因为比赛按截面 IC 评估排序能力，MSE 容易被尺度项主导、退化成常数输出。
#
# 所有命令都可以直接加额外的 train/test 参数，例如：
#   ./run.sh train --freq bar5m --loss combined
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bigquant
cd "$(dirname "$0")"

CMD="${1:-check}"
shift || true

case "$CMD" in
  check)
    python sanity_check.py --freq "${1:-bar30m}" --month "${2:-202301}"
    ;;
  smoke)
    python train.py --smoke --freq bar30m --target rank --loss corr \
      --out checkpoints/smoke.pt "$@"
    ;;
  train)
    python train.py --freq bar30m --target rank --loss corr \
      --epochs 30 --patience 5 --out checkpoints/best.pt "$@"
    ;;
  test)
    python test.py --ckpt checkpoints/best.pt --freq bar30m "$@"
    ;;
  leak)
    python leak_control.py --freq bar30m --month 202301 --seeds 5 "$@"
    ;;
  all)
    python sanity_check.py --freq bar30m --month 202301
    python train.py --smoke --freq bar30m --target rank --loss corr \
      --out checkpoints/smoke.pt
    python leak_control.py --freq bar30m --month 202301 --seeds 5
    ;;
  *)
    echo "unknown command: $CMD"; exit 1
    ;;
esac
