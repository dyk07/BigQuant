#!/usr/bin/env bash
# 验证每频率 lookback 能否把磁盘占用降到可承受范围
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bigquant
cd "$(dirname "$0")"

python - <<'PY'
from config import Config

F = 25
FREQS = ["bar30m", "bar15m", "bar5m", "bar1m"]

def per_sample_kb(cfg):
    tot = 0
    for f in FREQS:
        fc = cfg.for_freq(f)
        tot += fc.seq_len_bars * F * 4
    return tot / 1024

print("=" * 68)
print("每频率 lookback 对磁盘的影响（四频率融合，每样本字节）")
print("=" * 68)

configs = [
    ("基线 lookback=120", {}),
    ("bar1m=60",          {"bar1m": 60}),
    ("bar1m=30",          {"bar1m": 30}),
    ("bar1m=30, bar5m=60", {"bar1m": 30, "bar5m": 60}),
    ("bar1m=60, bar5m=120", {"bar1m": 60}),
]

# 实测：202301 单月 107,817 样本，磁盘 1.6GB
N_PER_MONTH = 107817
BASE_MEASURED_GB = 1.6

for label, ovr in configs:
    c = Config()
    c.freqs = FREQS
    c.lookback_minutes_per_freq = ovr
    kb = per_sample_kb(c)
    gb_48 = kb * 1024 * N_PER_MONTH * 48 / 1e9
    print(f"\n{label}")
    for f in FREQS:
        fc = c.for_freq(f)
        print(f"   {f:7s} T={fc.seq_len_bars:>3d}  "
              f"{fc.seq_len_bars*F*4/1024:>6.2f} KB/样本")
    print(f"   -> 合计 {kb:.2f} KB/样本  |  48个月磁盘 ~{gb_48:.0f} GB"
          f"{'  <-- 可行' if gb_48 < 45 else '  <-- 超 52GB 可用'}")

print()
print("=" * 68)
print("参考：202301+202302 实测磁盘 3.6 GB（基线 lookback=120）")
print("=" * 68)
PY
