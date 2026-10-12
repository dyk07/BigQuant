#!/usr/bin/env bash
# 执行任意命令（自动激活 bigquant 环境）
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bigquant
cd "$(dirname "$0")"
exec "$@"
