"""
数据管线：读表 -> 清洗 -> 特征 -> 标签 -> 切窗。

训练/验证/测试共用同一套流程，保证一致性、避免训练-推理偏移。
"""
import numpy as np
import pandas as pd

from config import Config
from data_loader import load_months, NEEDED_COLS, filter_auction_and_tag_day
from features import (build_features, build_labels, compute_clip_bounds,
                      apply_clip, ALL_FEATURES)
from dataset import make_samples


def build_split(cfg: Config, months, clip_bounds=None, verbose: bool = True,
                cache: bool = True):
    """
    读取指定月份并构造 (X, y, meta, clip_bounds)。

    clip_bounds: 训练集算出的 (lo, hi)；为 None 时用本次数据的分位数（只应在训练集上这样做）
    cache: 把结果缓存到 .cache/ 下（key 含频率/月份/窗口），避免重复计算数十秒
    """
    cache_file = None
    if cache:
        import hashlib
        import os
        key = "|".join([
            cfg.freq, ",".join(map(str, months)), str(cfg.seq_len_bars),
            str(cfg.horizon_bars), str(sorted(cfg.aligned_minutes())),
            str(cfg.input_same_day_only),
            # 截断边界必须进 key：验证/测试集要用训练集的边界，
            # 否则会命中"用自身分位数截断"的错误缓存
            "none" if clip_bounds is None else
            f"{clip_bounds[0]:.6g},{clip_bounds[1]:.6g}",
        ])
        digest = hashlib.md5(key.encode()).hexdigest()[:12]
        os.makedirs(".cache", exist_ok=True)
        cache_file = os.path.join(".cache", f"split_{cfg.freq}_{digest}.npz")
        if os.path.exists(cache_file):
            if verbose:
                print(f"  [cache] {cache_file}")
            data = np.load(cache_file, allow_pickle=False)
            X = data["X"]
            y = data["y"]
            # day 存为 datetime64[D]，读回时还原成 datetime.date，
            # 保证缓存路径与直算路径的 meta 类型完全一致
            meta = [(int(a), d.astype("datetime64[D]").astype(object), int(m))
                    for a, d, m in zip(data["inst"], data["day"],
                                       data["minute"])]
            cb = tuple(data["clip"]) if clip_bounds is None else clip_bounds
            return X, y, meta, cb

    df = load_months(cfg.data_dir, cfg.table, months,
                     columns=NEEDED_COLS, verbose=verbose)
    df = filter_auction_and_tag_day(df)
    df = build_features(df)
    df = build_labels(df, horizon_bars=cfg.horizon_bars)

    if clip_bounds is None:
        clip_bounds = compute_clip_bounds(df["label"].to_numpy())
    df["label"] = apply_clip(df["label"].to_numpy(), clip_bounds)

    X, y, meta = make_samples(df, cfg.seq_len_bars, cfg.aligned_minutes(),
                              input_same_day_only=cfg.input_same_day_only)

    if cache_file is not None:
        np.savez(cache_file,
                 X=X, y=y,
                 inst=np.array([m[0] for m in meta], dtype=np.int32),
                 day=np.array([np.datetime64(str(m[1])) for m in meta]),
                 minute=np.array([m[2] for m in meta], dtype=np.int16),
                 clip=np.asarray(clip_bounds, dtype=np.float64))
        if verbose:
            print(f"  [cache] wrote {cache_file}")
    return X, y, meta, clip_bounds


def describe(X, y, meta, tag=""):
    print(f"[{tag}] samples={len(y)} feat_dim={X.shape[-1] if X.size else 0} "
          f"seq_len={X.shape[1] if X.size else 0}")
    if len(y):
        print(f"       label mean={y.mean():+.6f} std={y.std():.6f} "
              f"min={y.min():+.4f} max={y.max():+.4f}")
        days = sorted(set(m[1] for m in meta))
        sections = {}
        for _, day, bar in meta:
            sections.setdefault((day, bar), 0)
            sections[(day, bar)] += 1
        sizes = np.array(list(sections.values()))
        print(f"       days={len(days)} ({days[0]} .. {days[-1]}) "
              f"sections={len(sections)} stocks/section: "
              f"min={sizes.min()} median={int(np.median(sizes))} max={sizes.max()}")


def check_window_leakage(cfg: Config, months, n_check: int = 200, seed: int = 0):
    """
    防泄漏自检（只校验 make_samples 真正会产出的样本）：
    1) 输入窗口的最后一根 bar 时间 < 当前 bar 时间（输入不含未来）
    2) 标签窗口（未来 h 根 bar）必须与当前 bar 同一天
    3) 标签从 t+1 开始，绝不包含 t 本身
    4) 同一截面内不重复出现同一只股票

    窗口长度用**时钟分钟**（minute_of_day）校验，因此缺 bar / 停牌也适用。
    输入窗口是否跨日由 cfg.input_same_day_only 决定；跨日只回看历史，
    不影响标签合法性。
    """
    df = load_months(cfg.data_dir, cfg.table, months, columns=NEEDED_COLS)
    df = filter_auction_and_tag_day(df)
    df = build_features(df)
    df = build_labels(df, horizon_bars=cfg.horizon_bars)

    seq = cfg.seq_len_bars
    h = cfg.horizon_bars
    same_day_input = cfg.input_same_day_only

    # 与 make_samples 完全一致的选择逻辑，确保校验对象就是训练用的样本
    X, y, meta = make_samples(df, seq, cfg.aligned_minutes(),
                              input_same_day_only=cfg.input_same_day_only)
    if len(meta) == 0:
        print("[leakage] no candidate samples")
        return False

    d = df.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    key_to_pos = {(int(r.instrument_id), r.day, int(r.minute_of_day)): i
                  for i, r in enumerate(d.itertuples(index=False))}
    # 全部用 numpy 位置数组，避免在 200 个样本上做 pandas 切片（很慢）
    date_arr = d["date"].to_numpy()
    inst_arr = d["instrument_id"].to_numpy()
    day_arr = d["day"].to_numpy()
    close_arr = d["close"].to_numpy(dtype="float64")
    label_arr = d["label"].to_numpy(dtype="float64")
    amt_arr = d["amount"].to_numpy(dtype="float64")
    vol_arr = d["volume"].to_numpy(dtype="float64")

    rng = np.random.default_rng(seed)
    n = len(meta)
    pick = rng.choice(n, size=min(n_check, n), replace=False)

    ok = True
    sec_seen = {}
    dup_sections = 0
    for j in pick:
        inst, day, mod = meta[j]
        pos = key_to_pos.get((inst, day, mod))
        if pos is None:
            print(f"[leakage] FAIL: sample not found at {meta[j]}")
            ok = False
            continue
        if pos - seq < 0 or pos + h >= len(d):
            print(f"[leakage] FAIL: window out of range at {meta[j]}")
            ok = False
            continue

        # ---- 1) 输入窗口 [pos-seq, pos) ----
        start_pos = pos - seq
        if inst_arr[start_pos] != inst:
            print(f"[leakage] FAIL: input covers another instrument at "
                  f"{meta[j]}")
            ok = False
            continue
        # 输入必须严格早于当前 bar（不含当前 bar，更不含未来）
        if not (date_arr[start_pos] < date_arr[pos]):
            print(f"[leakage] FAIL: input not before label time at {meta[j]}")
            ok = False
        if not (date_arr[pos - 1] < date_arr[pos]):
            print(f"[leakage] FAIL: input last bar >= label time at {meta[j]}")
            ok = False
        # 若要求同日，输入窗口必须全部落在同一天
        if same_day_input and day_arr[start_pos] != day:
            print(f"[leakage] FAIL: input window crosses day at {meta[j]}")
            ok = False

        # ---- 2) 标签窗口：未来恰好 h 根，且必须同一天 ----
        lo, hi = pos + 1, pos + h + 1
        if (inst_arr[lo:hi] != inst).any() or (day_arr[lo:hi] != day).any():
            print(f"[leakage] FAIL: label window crosses day at {meta[j]}")
            ok = False
        elif not (date_arr[pos] < date_arr[pos + 1]):
            print(f"[leakage] FAIL: label includes current bar at {meta[j]}")
            ok = False
        else:
            # ---- 3) 直接重算标签并与 build_labels 对比 ----
            # 自动兼容午休跨越、零成交分钟等情形，是最强的正确性校验。
            # 标签 = end_price / future_vwap - 1，end_price = 未来窗口最后
            # 一根 bar 的 close，即 close_arr[pos + h]（hi = pos + h + 1）。
            v = vol_arr[lo:hi].sum()
            fv = amt_arr[lo:hi].sum() / v if v > 0 else np.nan
            end_px = close_arr[hi - 1]
            recomputed = (end_px / fv - 1.0) if (end_px > 0 and np.isfinite(fv)) else np.nan
            stored = label_arr[pos]
            if np.isnan(recomputed) != np.isnan(stored) or (
                    not np.isnan(recomputed) and
                    not np.isclose(recomputed, stored, rtol=1e-6, atol=1e-12)):
                print(f"[leakage] FAIL: label mismatch at {meta[j]}: "
                      f"stored={stored!r} recomputed={recomputed!r}")
                ok = False

        # ---- 4) 截面唯一性 ----
        sec = (day, mod)
        if inst in sec_seen.setdefault(sec, set()):
            dup_sections += 1
        sec_seen[sec].add(inst)

    if dup_sections:
        print(f"[leakage] FAIL: {dup_sections} duplicate stocks within sections")
        ok = False

    print(f"[leakage] checked {len(pick)}/{n} windows -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def shuffled_label_ic(cfg: Config, months, n_seeds: int = 5, n_sub: int = 20000,
                      target: str = "rank", loss: str = "corr"):
    """
    防泄漏的强检验：反复打乱标签并短训模型。

    若管线存在泄漏，打乱标签后 IC 仍会显著为正；否则 IC 应在 0 附近波动。
    这里刻意使用与正式训练相同的 target/loss（rank + corr），因为相关性损失
    最擅长"发现"任何可被利用的排序信息——如果它都找不到信号，说明没有泄漏。

    注意：打乱标签必须发生在"构造截面 rank 目标"之前（这里 y 已打乱，
    rank 目标按打乱后的值在截面内排名，因此不再含真实排序信息）。
    """
    import torch
    from dataset import fit_scaler, apply_scaler
    from model import E2EModel
    from metrics import compute_ic, ic_significance, make_targets
    from losses import get_loss

    X, y, meta, _ = build_split(cfg, months, verbose=False)
    if len(y) == 0:
        print("[control] empty dataset")
        return []

    mean, std = fit_scaler(X)
    Xs = apply_scaler(X, mean, std)
    crit = get_loss(loss)
    results = []
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        y_shuf = rng.permutation(y)

        n_tr = int(len(y_shuf) * 0.8)
        # 打乱后再构造目标（rank 目标按打乱后的截面排名 -> 无真实信息）
        yt = make_targets(y_shuf[:n_tr], meta[:n_tr], kind=target)
        Xtr = torch.from_numpy(Xs[:n_tr])
        ytr = torch.from_numpy(yt)

        cut = min(n_tr + n_sub, len(y_shuf))
        Xte = torch.from_numpy(Xs[n_tr:cut])
        yte = y_shuf[n_tr:cut]           # 评估始终用原始（打乱后）收益率
        mte = meta[n_tr:cut]
        if Xte.shape[0] == 0:
            continue

        model = E2EModel(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        model.train()
        for _ in range(3):
            perm = torch.randperm(len(ytr))
            for b in range(0, len(perm), 4096):
                idx = perm[b:b + 4096]
                opt.zero_grad(set_to_none=True)
                loss_v = crit(model(Xtr[idx]), ytr[idx])
                loss_v.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(Xte).numpy()
        stats = compute_ic(preds, yte, mte)
        se, t = ic_significance(stats)
        results.append((stats[0], se, t))
        print(f"[control seed {s}] IC={stats[0]:+.4f} SE={se:.4f} t={t:+.2f} "
              f"n={stats[3]}")

    ics = np.array([r[0] for r in results])
    # 单个种子的 SE 偏小（滚动窗口高度重叠），跨种子统计才是正确的检验
    n = len(ics)
    mean_ic = float(ics.mean())
    se = float(ics.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    t = mean_ic / (se + 1e-12)
    verdict = "PASS (no leakage signal)" if abs(t) < 2 else "FAIL (leak signal)"
    print(f"[control] target={target} loss={loss} across-seed mean IC="
          f"{mean_ic:+.4f} SE={se:.4f} t={t:+.2f} n_seeds={n} -> {verdict}")
    print(f"[control] per-seed ICs={ics.round(4).tolist()}")
    return results

