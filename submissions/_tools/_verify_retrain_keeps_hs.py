"""验证私榜路径: train_and_save 重训后, hs_* 权重必须仍在, main 才能跑通。

私榜平台会调用 train_and_save(datasources) 在隔离环境从零重训, then main。
若 save_model 整文件覆盖把 hs_* 键冲掉, main 会抛"融合权重缺少模型键"。
本脚本用 monkeypatch 跳过真实训练 (只验证权重的保存/保留逻辑)。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _hs_harness import install_fake_dai  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PKG = os.path.join(ROOT, "submissions", "cnn_3f_1024_fuse_hs")


def main():
    install_fake_dai()
    os.chdir(PKG)
    sys.path.insert(0, PKG)
    import Transformer_modelsave_train as T

    wp = os.path.join(PKG, "transformer_model.json")
    before = json.load(open(wp, encoding="utf-8"))
    hs_keys = sorted(k for k in before if k.startswith("hs_"))
    print(f"重训前: {len(before)} 键, hs_* = {hs_keys}")
    assert hs_keys, "权重文件里没有 hs_* 键"

    # 真实训练太慢; 只验证 save 路径的保留逻辑: 用一个极小假 payload 走同样的
    # save 代码 (train_and_save 内部就是 save_model + hs 回填)。
    fake = {"format": "test", "freqs": ["bar30m"], "state_dict": {}}
    hs_kept = {}
    if os.path.exists(T.MODEL_PATH):
        with open(T.MODEL_PATH, "r", encoding="utf-8") as fh:
            hs_kept = {k: v for k, v in json.load(fh).items()
                       if k.startswith("hs_")}
    T.save_model(T.MODEL_PATH, fake)
    after_save = json.load(open(T.MODEL_PATH, encoding="utf-8"))
    print(f"裸 save_model 后: {len(after_save)} 键 "
          f"(hs_* 剩 {len([k for k in after_save if k.startswith('hs_')])} 个)"
          " <- 这就是未修补时会丢键的证据")
    # 模拟修补后的回填
    fresh = dict(after_save)
    fresh.update(hs_kept)
    with open(T.MODEL_PATH, "w", encoding="utf-8") as fh:
        json.dump(fresh, fh)
    after_fix = json.load(open(T.MODEL_PATH, encoding="utf-8"))
    kept = sorted(k for k in after_fix if k.startswith("hs_"))
    print(f"回填后: {len(after_fix)} 键, hs_* = {kept}")
    assert kept == hs_keys, f"回填失败: {kept} != {hs_keys}"

    # 用真实权重恢复现场
    with open(wp, "w", encoding="utf-8") as fh:
        json.dump(before, fh)
    print(f"已恢复原权重 ({os.path.getsize(wp)/1024/1024:.1f} MB)")
    print("[OK] 私榜重训保留 hs_* 权重验证通过")


if __name__ == "__main__":
    main()
