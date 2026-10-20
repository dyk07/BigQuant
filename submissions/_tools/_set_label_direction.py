"""给提交包权重 json 写入显式 label 方向字段, 供 predict 决定 factor 符号。

平台正确 label = end_price/vwap - 1 (factor 越大 = 预测收益越高)。

两类权重:
  * 旧方向 (vwap/end-1 训练): 排序方向相反, predict 需 factor 取负。
    - cnn_3f_1013_oldlabel, cnn_3f_1017_baseline_highscore
    - cnn_2f_1013_oldlabel, cnn_2f_1018_newlabel, cnn_2f
    - cnn_1f_1013_oldlabel, cnn_1f_1018_newlabel, cnn_1f_bar30m
  * 新方向 (end/vwap-1 训练): 方向正确, predict 不取负。
    - cnn_3f_1018_newlabel_xsec, cnn_3f_1019_newlabel_xsec_amp, cnn_3f

用法:
  python _set_label_direction.py
"""
import json
import os

ROOT = r"C:\Users\Kelvin\Documents\GitHub\BigQuant"

# 目录 -> 方向
OLD = "vwap_over_end"   # 旧方向, predict 需取负
NEW = "end_over_vwap"   # 新方向(平台正确), predict 不取负

PLAN = {
    # 旧方向权重
    "cnn_3f_1013_oldlabel": OLD,
    "cnn_3f_1017_baseline_highscore": OLD,
    "cnn_2f_1013_oldlabel": OLD,
    "cnn_2f_1018_newlabel": OLD,
    "cnn_2f": OLD,
    "cnn_1f_1013_oldlabel": OLD,
    "cnn_1f_1018_newlabel": OLD,
    "cnn_1f_bar30m": OLD,
    # 新方向权重
    "cnn_3f_1018_newlabel_xsec": NEW,
    "cnn_3f_1019_newlabel_xsec_amp": NEW,
    "cnn_3f": NEW,
}


def main():
    for sub, direction in PLAN.items():
        path = os.path.join(ROOT, "submissions", sub, "transformer_model.json")
        if not os.path.exists(path):
            print(f"[skip] {sub}: 无权重文件")
            continue
        with open(path, "r", encoding="utf-8") as fh:
            ck = json.load(fh)
        ck["label_direction"] = direction
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(ck, fh)
        print(f"[set] {sub} -> {direction}")


if __name__ == "__main__":
    main()
