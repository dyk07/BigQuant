#!/usr/bin/env bash
# 依次验证三个提交包, 并报告每个包的**峰值内存** (Maximum resident set size)。
set -u
ROOT=/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant
cd "$ROOT" || exit 1

for pkg in cnn_1f_bar30m cnn_2f cnn_3f; do
  echo "################ $pkg ################"
  /usr/bin/time -v python submissions/_tools/verify_notebook.py "$pkg" 2>&1 \
    | grep -E 'cell[0-9]+: OK|main\] 输出|dates=|factor_mean|OK$|Maximum resident|Elapsed|Error|Traceback|error' \
    | sed 's/^/  /'
  echo
done
