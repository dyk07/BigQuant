"""
防泄漏对照实验：打乱标签后 IC 应接近 0（无显著正值）。

用法：python leak_control.py --freq bar30m --month 202301 --seeds 5
"""
import argparse

from config import Config
from pipeline import shuffled_label_ic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="bar30m")
    ap.add_argument("--month", default="202301")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    cfg = Config()
    cfg.freq = args.freq
    cfg.device = "cpu"  # 小实验，CPU 足够

    print("=" * 70)
    print(f"label-shuffle control: freq={cfg.freq} month={args.month} "
          f"seeds={args.seeds}")
    print("=" * 70)
    results = shuffled_label_ic(cfg, [args.month], n_seeds=args.seeds)

    if results:
        import numpy as np
        ics = np.array([r[0] for r in results])
        # 跨种子统计才是正确检验：滚动窗口重叠会让单次 SE 偏小
        n = len(ics)
        mean_ic = ics.mean()
        se = ics.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        t = mean_ic / (se + 1e-12)
        print(f"\n[summary] across-seed mean IC={mean_ic:+.4f} SE={se:.4f} "
              f"t={t:+.2f} n_seeds={n}")
        print(f"          per-seed ICs={ics.round(4).tolist()}")
        verdict = "PASS (与 0 无显著差异)" if abs(t) < 2 else "FAIL (存在泄漏信号)"
        print(f"          标签随机化后 IC 应与 0 无法区分 -> {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
