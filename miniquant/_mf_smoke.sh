#!/usr/bin/env bash
# 四频率融合 smoke：202301 训练 / 202302 验证
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bigquant
cd "$(dirname "$0")"

rm -rf features_mf
python -u train_mf.py \
  --freqs bar30m,bar15m,bar5m,bar1m \
  --train-months 202301 --val-months 202302 \
  --epochs 3 --patience 2 --batch-size 512 \
  --out checkpoints/mf4_smoke.pt
echo "EXIT=$?"
