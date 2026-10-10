# Fig. 4｜三范围决策确认

> 当前正文编号（2026-10-08）：**Fig. 4**。目录 `fig3` 与资产文件名沿用既有命名；下方日期记录中的旧图号为当时快照。

当前版本为 **v1（完成）**：三面板 risk–coverage 曲线 + 底部双指标效应条，180 × 118 mm，
配色与 Fig. 2 相同（主蓝 = `AC-Twin audit`，灰 = 该范围冻结主基线）。

## 当前交付

- `Fig3_three_scope_decision_risk.svg` / `.png`（600 dpi）/ `.pdf`：主版本；
- `A_merge_plot_data.csv`、`B_joint_ood_plot_data.csv`、`C_metadrive_plot_data.csv`：逐组描述性重绘的曲线数据（每个 2,000 点）；
- `fig3_data.xlsx`：数据表（三页曲线 + `Summary` 页的正式报告效果量与 95% CI）；
- `build_fig3.py`：确定性重绘脚本；
- `FIGURE_LEGEND.md`：正文图注草稿；
- `archive_draft_v1/`：旧的三面板预览，仅作追溯。

## 运行

```powershell
python paper/figures/fig3/build_fig3.py
```

脚本只读冻结逐组量与冻结确认报告，重绘曲线并导出 CSV；不重做 bootstrap、不重新裁判。
绘图前会逐位断言 80% coverage 处的曲线值等于正式报告的 `regret80_model` / `regret80_baseline`（atol 1e-12），
断言失败即中止，不会写出与裁判不一致的图。
若某个格式的文件正被阅读器占用，脚本会跳过该格式并给出提示，其余格式照常写出。

## 版面规则

- 画布 180 × 118 mm，按**双栏通栏 100% 比例**设计（正文引用时不缩放，图内字号即为最终字号）；
- 面板字母 `(a)`–`(c)` 为三个范围的 risk–coverage，`(d)` 为底部双指标效应条；图内不设总标题与页脚，二者由 caption 承担；
- 每个面板画出**全部三条标准基线**：该范围的冻结主基线为最深最粗的实线并标 `frozen primary`，另外两条按深浅递减、同为实线（不使用虚线，避免与曲线重叠时产生噪声）；三个面板各有**独立纵轴**与绝对刻度；面板名下方两行写明模型关系；
- 80% coverage 处以细虚线标出并标注；
- 底部两幅效应条给出 `AURC-regret` 与 `Regret@80` 的相对改善点估计与 95% CI；
- 正文调用：§6.1–§6.3 的范围结果分别引用 a/b/c；§6.3 末段引用 **Fig. 3d** 综合展示两项相对改善及区间，保留各自冻结分母和禁止跨范围排名/合并的边界（2026-10-03 补齐）；
- 三行的基线、任务分布与权重关系不同，横向不作排名或合并。

## 统一命名

图、表、正文使用同一套范围名：**`Merge ID`**（Merge 分布内）、**`JOINT_OOD`**（前瞻指定生成机失配）、**`MetaDrive`**（MetaDrive/ScenarioNet/Waymo，平台内重训）。全称只在 caption 中展开一次。

方法标签（2026-09-27 作者裁定，2026-09-28 表号机械前移）：图中的 **`AC-Twin audit`** 就是冻结报告里的方法键名 `DIRECT_ENSEMBLE`（曲线取逐组量 `ensemble_risk`，即五个 seed 集成后的直接监督风险头输出）。`Direct audit head` 只作**组件名**（Fig. 2 的架构块），裸 `AC-Twin` 只指整套分析—审计框架。本图与 Table 6、Fig. 4、Table S3 使用同一个结果层标签；更名是纯文字改动，未重估、未重算任何数字。

## 来源

曲线的逐组量：`experiments/stage4_reliability/stage4cr2_confirm/reports/confirm_group_quantities.npz`、
`.../stage4d_ood/reports/d1_group_quantities.npz`、`.../stage4d_d2f/confirm/confirm_group_quantities.npz`。
效果量与 CI：同目录的三份正式确认报告（Merge 与 JOINT_OOD 取 `decision.metrics`，MetaDrive 取 `metrics`）。
