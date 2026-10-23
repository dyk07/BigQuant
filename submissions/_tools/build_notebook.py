#!/usr/bin/env python
"""
把提交目录里的 .py 源码打包成**单个 notebook** (.ipynb), 保留原始文件夹 (不生成 zip)。

- 训练脚本 + 推理脚本 + 超参数配置 + 随机种子 全部内联进 notebook;
- 依赖声明 (requirements.txt) 与权重 (transformer_model.json) 作为同目录附属文件;
- 每个提交包内**有且只有一个 notebook** (平台要求)。
"""
import json
import os
import re

ROOT = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant"
SUB = os.path.join(ROOT, "submissions")

PACKAGES = [
    "cnn_1f_1013_oldlabel",
    "cnn_1f_1018_newlabel",
    "cnn_1f_bar30m",
    "cnn_2f",
    "cnn_2f_1013_oldlabel",
    "cnn_2f_1018_newlabel",
    "cnn_3f",
    "cnn_3f_1013_oldlabel",
    "cnn_3f_1017_baseline_highscore",
    "cnn_3f_1018_newlabel_xsec",
    "cnn_3f_1019_newlabel_xsec_amp",
    "cnn_3f_1020_newlabel_to0005",
    "cnn_3f_1020_swa",
    "cnn_3f_1021_spread",
    "cnn_3f_1021_turnover_v3",
    "cnn_3f_1022_turnover_spread",
    "cnn_3f_1023_ema_post",
]
NOTEBOOK_NAME = "Transformer_modelsave_predict.ipynb"

SEC_RE = re.compile(r'(?m)^# ={20,}\n# (.+?)\n# ={20,}\n')
CFG_KEYS = ["SEED", "FREQS", "LOOKBACK_MINUTES", "HORIZON_MINUTES",
            "SAMPLE_STEP_MINUTES", "INPUT_SAME_DAY_ONLY", "TRAIN_MONTHS",
            "MODEL_CFG", "TRAIN_CFG",
            "USE_XSEC", "XSEC_LAYERS", "XSEC_HEADS",
            "USE_TURNOVER", "LAMBDA_TURNOVER",
            "USE_AMP", "AMP_DTYPE",
            "USE_SWA", "SWA_START_EPOCH",
            "USE_SPREAD", "LAMBDA_SPREAD", "SPREAD_TAU",
            "USE_RANK_DRIFT", "TO_STEP_INTERVAL", "EMA_ALPHA"]

DRIVER = '''# ============================================================
# 本地 / 演示入口
# 平台公榜会直接调用 main(datasources, start_date, end_date);
# 私榜会调用 train_and_save(datasources) 在隔离环境从零重训。
# 因此正常提交无需运行本单元格 (平台自己调用函数)。
# ============================================================
RUN_LOCAL_DEMO = False          # 本地想看效果改成 True
if RUN_LOCAL_DEMO:
    from bigmodule import M
    datasources = {"bar1m": "cpt_jyc_2026_e2e_bar1m"}
    if not os.path.exists(MODEL_PATH):
        train_and_save(datasources)
    factor_data = main(datasources, "2024-07-01 00:00:00", "2024-09-30 23:59:59")
    print(factor_data.head())
    M.jyc_eval._latest(factor_data=factor_data, show=True)
'''

PLATFORM_IMPORTS = '''import gc
import os
import numpy as np
import pandas as pd
import torch
import dai
import structlog

logger = structlog.get_logger()
'''


def _lines(s):
    if not s.endswith("\n"):
        s += "\n"
    return s.splitlines(keepends=True)


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(src)}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(src)}


def build_config_cell(config_py):
    """把 config.py 内联成一个 SimpleNamespace 单元格 (不依赖外部模块)。"""
    ns = {}
    exec(open(config_py, encoding="utf-8").read(), ns)
    out = ["import types\n",
           "# ======================================================================\n",
           "# 超参数配置 + 随机种子 (单一事实来源)\n",
           "# ======================================================================\n",
           "CFG = types.SimpleNamespace(\n"]
    for k in CFG_KEYS:
        if k in ns:
            out.append(f"    {k}={ns[k]!r},\n")
    out.append(")\n")
    return code_cell("".join(out))


def split_train(src):
    """按 section 分隔符把训练脚本切成多段。"""
    src = re.split(r'\nif __name__ == "__main__":', src)[0]
    marks = [m.start() for m in SEC_RE.finditer(src)]
    titles = [m.group(1) for m in SEC_RE.finditer(src)]
    if not marks:
        return [("全部", src)]
    segs = []
    if marks[0] > 0:
        segs.append(("导入与常量", src[:marks[0]]))
    for i, st in enumerate(marks):
        en = marks[i + 1] if i + 1 < len(marks) else len(src)
        segs.append((titles[i], src[st:en]))
    return segs


def extract_main(predict_py):
    """抽取推理入口 main() 及其依赖的模块级辅助函数, 去掉跨文件 import。

    起点取脚本中**第一个模块级 def**, 这样定义在 main 之前的推理辅助函数
    (如 `_ema_within_day`) 会一并纳入, 避免 notebook 出现未定义符号。
    """
    src = open(predict_py, encoding="utf-8").read()
    src = re.sub(r'from Transformer_modelsave_train import \([\s\S]*?\)\n', '', src)
    m = re.search(r'^def ', src, flags=re.M)
    if m is None:
        m = re.search(r'\ndef main\(', src)
    body = src[m.start():]
    body = re.split(r'\nif __name__ == "__main__":', body)[0]
    head = PLATFORM_IMPORTS
    # 后处理系数(若脚本引用)从 CFG 读取, 保持配置单一事实来源
    if "EMA_ALPHA" in body:
        head += "\nEMA_ALPHA = CFG.EMA_ALPHA\n"
    return head + "\n" + body


def build_notebook(pkg_dir, title_note):
    train_py = os.path.join(pkg_dir, "Transformer_modelsave_train.py")
    predict_py = os.path.join(pkg_dir, "Transformer_modelsave_predict.py")
    config_py = os.path.join(pkg_dir, "config.py")

    segs = split_train(open(train_py, encoding="utf-8").read())

    cells = [md_cell(f"""# BigQuant 量化大赛 · 端到端赛道 · 单 notebook 提交

{title_note}

本 notebook **自包含**: 内含超参数配置 / 随机种子 / 模型定义 / 数据管线 / 训练函数
/ 推理入口 `main`。仅额外依赖同目录的:

- `transformer_model.json` —— 已训练权重 (含标准化统计与结构超参)
- `requirements.txt` —— 依赖声明

平台约定:
- **公榜**: 调用 `main(datasources, start_date, end_date)`, 加载提交权重推理;
- **私榜**: 调用 `train_and_save(datasources)` 在隔离环境从零重训 (种子已固定)。
"""),
             build_config_cell(config_py)]

    for title, text in segs:
        if title == "导入与常量":
            text = re.sub(r'^import config as CFG.*\n',
                          '# CFG 由上方"配置"单元格定义\n', text, flags=re.M)
        cells.append(code_cell(text))

    cells.append(code_cell(extract_main(predict_py)))
    cells.append(code_cell(DRIVER))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.9"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    for pkg in PACKAGES:
        pkg_dir = os.path.join(SUB, pkg)
        nb = build_notebook(pkg_dir, f"模型包: `{pkg}`")

        # 语法检查每个代码单元格
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                try:
                    compile("".join(c["source"]), f"<cell{i}>", "exec")
                except SyntaxError as e:
                    raise SyntaxError(f"{pkg} cell{i}: {e}")

        nb_path = os.path.join(pkg_dir, NOTEBOOK_NAME)
        with open(nb_path, "w", encoding="utf-8") as fh:
            json.dump(nb, fh, ensure_ascii=False, indent=1)
        print(f"[notebook] {pkg}/{NOTEBOOK_NAME}  cells={len(nb['cells'])}")

        # 检查包内**有且只有一个 notebook** (平台要求)
        n_nb = sum(fn.endswith(".ipynb") for fn in os.listdir(pkg_dir))
        assert n_nb == 1, f"{pkg}: notebook 数量 = {n_nb} (必须为 1)"
        # 列出包内全部文件 (不打包, 保留原始文件夹)
        files = sorted(os.listdir(pkg_dir))
        print(f"[pkg]      {pkg}/  ({n_nb} notebook) -> {files}")


if __name__ == "__main__":
    main()
