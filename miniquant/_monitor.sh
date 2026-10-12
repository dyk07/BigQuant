#!/usr/bin/env bash
cd /mnt/c/Users/Kelvin/Documents/GitHub/BigQuant/miniquant
{
echo "--- ps ---"
ps -eo pid,etime,%cpu,%mem,cmd --cols 150 | grep -E 'train_mf|_hparam' | head -5
echo "--- features ---"
du -sh features_mf 2>/dev/null || echo "no features_mf"
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
echo "--- results ---"
cat _hparam_results.txt 2>/dev/null || echo "no results yet"
echo "--- ckpt ---"
ls checkpoints/_h_*.pt 2>/dev/null | head -5 || echo "no ckpt yet"
echo "--- log ---"
ls -lh /mnt/c/Users/Kelvin/Documents/GitHub/BigQuant/miniquant/_hparam_results.txt 2>/dev/null
} > _mon.txt 2>&1