# Fig. 1｜预测与审计流程

> 当前正文编号（2026-10-08）：**Fig. 1**。目录 `fig2` 与资产文件名沿用既有命名；下方日期记录中的旧图号为当时快照。

本目录是视觉叙事蓝图 v1.0 的当前方法流程图制作包。当前版本为 **v4**：沿用 v3 的 MemDiff 式表现
（两片底衬分区，上方推理路径 / 下方封存真值；立体模块块），并按全图集统一风格规则做了三处规范化
（见下节）。图身保留动作组合和80%覆盖率，方法公式及超参数由正文承接。

2026-10-04 测量含义同步：Retention 标签为 `Rank by audit score`，排序使用预测审计分数；真实 regret 仅在训练监督与评分冻结后的评价中使用。本次测量含义保持不变。

## 唯一正式版本（2026-10-05）

最终采用 **no red tag** 构图，文件统一使用 Fig2_AC_Twin_method_overview_v4 名称，不再输出备选版本。

- Fig2_AC_Twin_method_overview_v4.png：600 dpi，正文插图；
- Fig2_AC_Twin_method_overview_v4.svg：可编辑矢量稿；
- Fig2_AC_Twin_method_overview_v4.pdf：矢量排版稿。

以上三个格式是同一张正式图。删除单独的 	raining-only labels 红色挂签，保留下方真值分区和训练监督红色虚线。删除紧贴场景图的 Observed prefix 标签，其含义由图注承接。

旧版图片、red tag / no red tag 对照稿及历史草稿目录已移至 paper/_figure_archive/fig2_before_final_2026-10-05/，不得作为当前插图引用。完整 Word 已同步；由于六章结构重排，本目录方法总览在当前论文中编号为 **Fig. 1**，目录和文件名沿用原命名。

uild_fig2.py 只生成上述正式图；FIGURE_CONTRACT.md 和 FIGURE_LEGEND.md 保存图面约束及图注。

## v4 的风格规范化

- **删除图内加粗标题与页脚句**。原图内标题 `AC-Twin decision-risk auditing` 与原页脚句
  "The audit ranks the selected action of each decision group; it does not update the frozen
  predictor or execute a fallback policy." 都由 `FIGURE_LEGEND.md` 的图注承接，图内不再重复；
  这与 Fig. 1 / Fig. 3 的做法一致。
- **画布裁到实际图形**：由 180 × 100 mm 改为 180 × 88 mm。脚本保留 v3 的物理比例
  （每 x 单位 180 mm、每 y 单位 100 mm），只把 `AX_Y0..AX_Y1` 之外原本给标题与页脚的两条空白带裁掉，
  因此图标、缩略图与线宽的实际尺寸完全不变。
- **图内字号整体提升 8%**：单一缩放点 `FONT_SCALE`（作用于 `text()`），字号的相对层级不变。
  为容纳这一提升，按 DejaVu Sans 的真实字形宽度重排了几处几何：`training-only labels` 挂签、
  封存区标题与 `evaluates the audit` 的锁形图标间距，Retention 两行的图标/文字分配，
  真值链四项的宽度与间距，以及 `retained` / `not retained` 两个条标签的间距。

## 素材来源

配色采用作者指定的一组六色：`#4D93D6`、`#A4C8EA`、`#D57873`、`#F1C2AD`、`#A8A8A9`、`#D9D9D9`。语义分配固定为：主蓝 `#4D93D6` 与浅蓝 `#A4C8EA` 表示 AC-Twin 方法与推理区；灰 `#A8A8A9` 表示冻结预测器；暖红 `#D57873` 与浅桃 `#F1C2AD` 表示反事实分支以及训练/评价真值族；浅灰 `#D9D9D9` 表示框线与中性。明暗梯度由同源色相派生（`lighten` / `darken`），`build_fig2.py` 内的 `_PALETTE_*` 常量是唯一改动点。

`icon/` 目录是作者提供的 iconfont 素材。脚本用内置的 SVG 路径解析器把它们转成可改色的矢量（不贴位图），当前使用：

| 文件 | 用途 |
| --- | --- |
| `锁,密码,锁定.svg` | 真值区标题与 `training-only labels` 挂签 |
| `audit.png`（彩色插画） | `Audit risk score`；以位图嵌入，固定 8 mm |
| `authcode.svg`（盾形对勾） | `Rank by audit score` |
| `筛选.svg` | `Retain lowest-risk` |

`Observed prefix.png` 用作 Twin event 的**真实场景面板**（路网 + 轨迹）；重绘仅使用本目录素材，不读取实验数据；素材缺失时停止。

暂未使用：`Traffic Analysis.svg`（公路字形）、`车5.svg`（该图标为多色填充，单色化后细节丢失，只剩轮廓）。

## 运行

```powershell
python paper/figures/fig2/build_fig2.py
```

脚本只绘制既有方法流程，使用本目录素材，不读取实验数据或运行统计分析。

## 数字与符号的分工

图身只保留五类动作×两种时长及80%覆盖率；其余数字与公式由 `FIGURE_LEGEND.md`（正文图注）及§5承接：

- 十个候选计划：`code/ac_twin/stage4c_objective.py::ACTION_PLANS`（五种动作意图 × 两个持续时长）；
- 80% 保留覆盖率：`code/ac_twin/stage4c_protocol.py::COVERAGE_MAIN`；
- 特征维度、隐藏宽度、参数量、dropout 与两行输出公式：R2 计划 §3–4 与 `code/ac_twin/stage4cp_features.py`、`code/ac_twin/stage4cp_models.py`。
