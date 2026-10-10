# Fig. 5｜MetaDrive 任务与案例

> 当前正文编号（2026-10-08）：**Fig. 5**。目录 `fig5` 与资产文件名沿用既有命名；下方日期记录中的旧图号为当时快照。

> 状态：`AUTHOR_APPROVED_FULL_PNG / 2026-10-03`  
> **唯一正文图稿：[Fig5_metadrive_external_validation.png](Fig5_metadrive_external_validation.png)。** 作者已明确确认该文件；本次确认已回写总蓝图及制作包索引，替代原“平台-only 正文 + Fig. S-cases 补充”安排。文件身份见 [AUTHOR_APPROVAL.json](AUTHOR_APPROVAL.json)。

当前正文版本由脚本确定性生成，尺寸为 **180 × 160 mm**，保留作者选定的完整叙事结构：

- `(a)` MetaDrive 原生 Panda3D 视图；
- `(b)` 同一场景的地图、事实轨迹与红色虚线反事实轨迹；
- `(c)` 冻结的高 regret 正确检出案例（三子图）；
- `(d)` 冻结的最坏 false-negative 案例（三子图）。

## 当前交付

- **`Fig5_metadrive_external_validation.png`：唯一作者选定的正文图稿，4251×3779 px；排版、正文引用及后续核验均从本文件出发。**
- `Fig5_metadrive_external_validation.svg` / `.pdf`：同一完整图的配套格式，不构成独立正文版本，也不自动替代已确认 PNG；
- `Fig5_selected_platform_raw.png`：`(a)` 面板使用的 MetaDrive 原生截图素材，与 `(b)` 对应同一场景；
- `Fig5_render_receipt.json`：选择规则、冻结来源、案例记录及输出 hash；
- `build_fig5.py`：唯一当前渲染入口；
- `FIGURE_LEGEND.md`：正文图注草稿与中文边界备忘。

## 目录内文件身份

| 文件 | 当前身份 |
| --- | --- |
| `Fig5_metadrive_external_validation.png` | 唯一正文 Fig. 5，作者明确确认 |
| 同名 `.svg` / `.pdf` | 当前完整图的配套格式 |
| `Fig5_platform_panel.svg/png/pdf` | 旧平台-only 版本，仅历史追溯 |
| `archive_author_fixed/*` | 原作者图及六面板案例底稿，仅历史追溯；不再单列当前 Fig. S-cases |
| `Fig5_selected_platform_raw.png` | a 面板原生截图素材，不是另一份 Fig. 5 |
| `Fig5_render_receipt.json` / `Fig5_layout_manifest.json` | 来源、选择、布局回执，不定义新的正文版本 |

以上文件均保留，避免通过删除资产解决版本识别问题。

## 平台场景的选择规则

只在已经冻结的 20 个可视化候选中进行一次**非推断性、仅为图面可读性服务**的机械排序：

1. 可见实体数降序；
2. 事实/反事实 ego 分支最大间距降序；
3. 原始 legibility score 降序；
4. candidate hash 升序打破平局。

该规则选中：

- `scenario_id = 1cac180d139c96de`；
- `t* = 13`；
- 可见实体数 `5`；
- 最大事实/反事实分支间距约 `3.134 m`。

这替代了旧平台图中分支仅相差 `0.009983 m`、红线被绿线与 EGO 标记遮住的场景。它不改变训练、推理、Gate、确认裁判或任何论文数值。截图右上角 MetaDrive FPS 叠层允许保留。

## 复现入口（本轮未运行）

作者固定的是现有 PNG。本节命令仅用于复现，不应在后续整理时自动重生成或替换作者已确认的文件。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File code\run_stage4d_env.ps1 sim paper\figures\fig5\build_fig5.py
```

如需重新抓取对应的 MetaDrive 原生截图：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File code\run_stage4d_env.ps1 sim paper\figures\fig5\build_fig5.py --refresh-screenshot
```

必须使用冻结的 MetaDrive 仿真环境。脚本只读冻结产物；不会训练、采集、重推理模型、重算 Gate 或重新裁判确认集。

## 图面合同

- `(b)` 中事实 ego 轨迹为绿色，固定 `LEFT × 1` 反事实 ego 轨迹为红色虚线且置于顶层；
- 下方两行的案例 ID、动作、regret、风险百分位与 peak step 逐项复用 create-once 回执，不重新选择；
- `(c)` 的 best-of-205 与平台 `(a)/(b)` 的 best-rendered-of-20 都必须明确标注为定性展示，不代表总体成功率；
- selected action 与 oracle action 的差异只解释 decision regret，不表示实施了在线回退；
- MetaDrive 结果来自平台内训练与冻结的管线，不得写成 Merge 权重零样本迁移；
- 总体定量结论回指统计图表，不从本图个例推断效应量。

## 主要冻结来源

- 平台候选池：`experiments/stage4_reliability/stage4d_external/platform/viz_platform_pool.json`；
- pilot 轨迹：`experiments/stage4_reliability/stage4d_external/roles/pilot/`；
- 确认案例数据：`experiments/stage4_reliability/stage4d_d2f/roles/confirm/groups.npz`；
- 确认逐组量：`experiments/stage4_reliability/stage4d_d2f/confirm/confirm_group_quantities.npz`；
- create-once 选择回执：`paper/figures/stage4d_d2f/provenance/selection_receipt.json`；
- 地图/坐标变换：`code/ac_twin/stage4d_external_features.py` 与 `code/ac_twin/stage4d_paper_visualization.py`。
