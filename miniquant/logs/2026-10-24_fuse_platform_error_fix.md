# 融合包平台运行错误：定位与修复

日期：2026-10-24
作废包：`submissions/cnn_3f_1024_fuse_hs`（平台运行错误）
修复包：`submissions/cnn_3f_1024_fuse_raw`（融合未平滑）+ `submissions/cnn_3f_1024_fuse_soft`（融合 + L1 软阈值 θ=0.20）

## 背景

上一轮交付的 `cnn_3f_1024_fuse_hs`（commit `7140c17`）本地端到端验证全部通过，
但提交平台后直接**运行错误**。本轮的定位结论是：本地验证器与平台在
**两处执行契约**上不一致，而这两处都只在平台侧触发。

## 缺陷 1：`from __future__` 出现在文件中段 → 拼接编译 SyntaxError

高分模型源码通篇使用 PEP 604/585 注解（`str | Path`、`list[str]`、`dict[str, int]`），
在 Python 3.9 下缺 `from __future__ import annotations` 会在 `def`/`class`
**定义处**就 `TypeError: unsupported operand type(s) for |`。上一轮因此把该语句
保留在 HS 段（cell 9）开头。

问题在于：**Jupyter 逐 cell 执行时合法，但平台若把各 cell 拼接后 `compile()`
整份代码，future 语句就落在第 1616 行**——Python 规定它必须位于文件最前。

本地复现（这是判定依据，不是猜测）：

```
>>> compile("\n".join(所有非 driver cell), "<concat>", "exec")
SyntaxError: from __future__ imports must occur at the beginning of the file (<concat>, line 1616)
```

### 修复：注解字符串化

不能删 future，也不能用 future，因此把**所有注解包成字符串字面量**——这正是
`from __future__ import annotations` 的运行期效果，两种执行模型下都安全。

实现要点（`_tools/_build_fuse_pkg2.py: _quote_annotations`）：

- 用 `ast` 遍历 `FunctionDef.returns` / `arg.annotation` / `AnnAssign.annotation`
  精确取 span，**不用正则**。正则版本在 `def load_model(path: str | Path, ...)`
  上会漏检（`|` 后接逗号）。
- `ast` 的 `col_offset` 是 **UTF-8 字节偏移**，而 HS 源码含大量中文注释，
  必须在 `line.encode("utf-8")` 上切片再 decode，否则插入位置整体错位。
- 生成的 notebook 里 283 处注解全部字符串化，`ast` 复核「注解总数 285、
  未字符串化 2（都是裸 `int`，无需引号）」。

## 缺陷 2：硬要求 bar30m + bar5m，超出已证实的接口边界

HS 段原 `validate_datasources` 是：

```python
if not isinstance(datasources, dict) or {"bar30m", "bar5m"} - set(datasources):
    raise KeyError("datasources 必须同时含 bar30m 与 bar5m")
```

而**此前 10+ 次成功提交**的 driver 全部只写：

```python
datasources = {"bar1m": "cpt_jyc_2026_e2e_bar1m"}
```

即自有模型从头到尾只依赖 `bar1m`。MG3 参考包确实遍历四频（说明平台**会**注入），
但历史只验证过 `bar1m` 这一条路径。把「平台一定给 bar30m/bar5m」当作前提，
是把一次未经验证的假设变成了硬性依赖——一旦不成立就是 `KeyError`，整份 0 分。

### 修复：bar1m 重采样降级（首选） + 原生直读（快路径）

`_hs_bars(datasources, tag, kmins, start, end, pool)`：

1. `datasources` 里有原生 `bar30m`/`bar5m` → 直接读（省一次重采样）；
2. 没有 → 读 `bar1m`，走训练脚本的 `tag_time` + `resample_bars` 合成；
3. 两者都没有 → 才 `KeyError`。

`validate_datasources` 相应放宽为「至少需要 `bar1m`，或同时有 `bar30m`+`bar5m`」。

**降级路径的正确性已实测**（`_tools/_verify_resample.py`）：

| 频率 | 行数 | 与原生表比对 |
|---|---|---|
| bar30m | 189,000 | 完全逐位一致 |
| bar15m | 357,000 | 完全逐位一致 |
| bar5m | 1,029,000 | 完全逐位一致 |

7 列全部 `max|diff| < 1e-9`，NaN 位置完全匹配，仅原生 0 行、仅合成 0 行。
`adjust_factor` 在 (instrument, 交易日) 内唯一（非唯一组 0/21000），
故重采样后按 (instrument, day) 单独回贴即可。

### 顺带修掉的第三个 bug（本轮新发现）

降级路径第一次跑时抛 `KeyError: 'adjust_factor'`——`resample_bars` 的聚合列
清单不含 `adjust_factor`，而 HS 的 `_features30` 要用它。已在 `_hs_bars` 内
先按 (instrument, day) 取一份再回贴。

## 缺陷 3（预防性）：一路失败不该拖垮整份

`main` 里把 `hs_main` 包进 try/except：高分模型一路任何异常都降级为
**自有模型单路输出**并打印原因，避免重演「一路异常 → 整份运行错误 → 0 分」。

## 验证

`_tools/_verify_fuse_e2e.py` 增加包名参数与 `--bar1m-only` 开关。四种组合全部通过：

| 包 | datasources | 行情来源 | IC | IC_IR | Sharpe | TO |
|---|---|---|---|---|---|---|
| `_fuse_raw` | 只给 bar1m | synth | +0.1310 | 0.994 | 0.839 | 0.3972 |
| `_fuse_raw` | 四频 | native | +0.1310 | 0.994 | 0.839 | 0.3972 |
| `_fuse_soft` | 只给 bar1m | synth | +0.1283 | 0.994 | 0.835 | 0.3555 |
| `_fuse_soft` | （同上，指标一致） | — | — | — | — | — |

`synth` 与 `native` 两条路径指标**完全相同**，直接证明降级无损。

对照参考：ours 单独 IC +0.1149 / TO 0.4040；hs 单独 IC +0.1184 / TO 0.3982；
截面 rank 相关 ours↔hs **0.5011**（同族 0.83 = 已饱和）。

其他校验：

- 逐 cell 编译 + 拼接整体编译，两者均通过（构建器内置，不通过不出包）；
- 输出契约：147,000 行 / 147 截面 / 1000 标的 / 无 NaN、无 inf、无重复键 / 三列；
- HS 段 41 个顶层符号、18 个顶层赋值与其余 cell **零冲突**；
- 私榜重训保留 `hs_*` 权重：重训前 23 键 → 裸 `save_model` 后 3 键（`hs_*` 剩 0）
  → 回填后恢复 `hs_exp033`/`hs_exp034`/`hs_fusion`。

## 两个提交版本

| 包名 | θ | 定位 |
|---|---|---|
| `cnn_3f_1024_fuse_raw` | 0.00 | 纯融合。IC 最高（+0.1310），换手与单路持平 |
| `cnn_3f_1024_fuse_soft` | 0.20 | 融合 + 日内因果 L1 软阈值。换手 0.3555（-10.5%），IC_IR 与 Sharpe 持平 |

用户上一轮明确提示「仅凭换手提升就能使总分上升，平衡非常重要」，因此两版同时提交，
用于实测平台的换手权重实际有多大——这是本轮唯一能通过平台回答的问题。

## 旧包处置

`cnn_3f_1024_fuse_hs` 已删除（避免与修复版混淆）。其内容在 commit `7140c17`
完整留存，文档在 [2026-10-24_fuse_highscore.md](./2026-10-24_fuse_highscore.md)。

## 复现

```bash
# 构建两个修复包（含逐 cell + 拼接双编译校验）
python submissions/_tools/_build_fuse_pkg2.py raw
python submissions/_tools/_build_fuse_pkg2.py soft

# 重采样逐位一致性
python submissions/_tools/_verify_resample.py

# 端到端（需 torch 环境；--bar1m-only 验证降级路径）
python submissions/_tools/_verify_fuse_e2e.py cnn_3f_1024_fuse_raw  2024-11-01 2024-11-29 --bar1m-only
python submissions/_tools/_verify_fuse_e2e.py cnn_3f_1024_fuse_soft 2024-11-01 2024-11-29 --bar1m-only

# 私榜路径
python submissions/_tools/_verify_retrain_keeps_hs.py cnn_3f_1024_fuse_raw
```
