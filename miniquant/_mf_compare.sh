#!/usr/bin/env bash
# 四频率融合 smoke（含每频率 lookback），并做 单频 vs 四频 对照
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bigquant
cd "$(dirname "$0")"

rm -rf features_mf

echo "################ 1) 四频率融合（bar1m 用 30 分钟窗口）################"
python -u train_mf.py \
  --freqs bar30m,bar15m,bar5m,bar1m \
  --lookback-bar1m 30 \
  --train-months 202301 --val-months 202302 \
  --epochs 4 --patience 2 --batch-size 512 \
  --out checkpoints/mf4_smoke.pt
echo "EXIT_4FREQ=$?"

echo
echo "################ 2) 双频率对照（5m+30m）################"
python -u train_mf.py \
  --freqs bar30m,bar5m \
  --train-months 202301 --val-months 202302 \
  --epochs 4 --patience 2 --batch-size 512 \
  --out checkpoints/mf2_smoke.pt
echo "EXIT_2FREQ=$?"

echo
echo "################ 3) 磁盘占用 ################"
du -sh features_mf
du -sh features_mf/*/
