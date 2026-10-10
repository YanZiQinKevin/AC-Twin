# Fig. 6｜稳健性与 OOD 资格

> 当前正文编号（2026-10-08）：**Fig. 6**。目录 `fig4` 与资产文件名沿用既有命名；下方日期记录中的旧图号为当时快照。

> 2026-10-04 叙事同步：图内采用 specified，图注解释 prospectively specified；未改任何效应量或区间。

当前版本为 **v1（完成）**：四面板 2×2，180 × 126 mm，按双栏通栏 100% 比例设计，
配色与 Fig. 1 / Fig. 2 / Fig. 3 相同。

## 评审修订记录（2026-09-28）

第一版经独立核对后按评审意见修订，共七处：

| # | 问题 | 处理 |
| --- | --- | --- |
| 1 | (d) 副标题宽 195.8 pt、居中后**右侧溢出 2.2 pt，收尾的 `]` 未渲染** | 副标题拆两行并左移面板块；整图四边墨迹边距现为 2.16 / 2.50 / 4.91 / 5.63 mm（原右缘仅 0.08 mm） |
| 2 | 图注把 **MetaDrive 的 10.1%** 挂到了 Merge 面板上 | 改为 Merge 的 **26.0%**（Regret@80 对冻结主基线）；范围串号警告写入图注备忘 |
| 3 | (b) 两个子坐标轴宽度不等（0.130 vs 0.123），每单位长度差 5.4%、刻度也不一致 | 两轴等宽 0.128，统一 `xticks = 0.0/0.2/0.4`，x 轴标签改为跨两轴共享 |
| 4 | 图题称 "survives … the prospectively specified mismatch test"，但 (d) 只画资格矩阵、不含任何 OOD 效应量 | 图题改为"不是单一基线/单 seed/单分层造成的"，(d) 副标题以 `effect in Fig. 3b` 交叉指向效应；蓝图 §4 Fig. 4 工作标题同步修订 |
| 5 | 合同把最小字号写成 6.2 pt（实测为 7.0） | 合同改为 7.0 pt，并登记为**密度例外**（蓝图 §9.2 要求新增图 7.5 pt） |
| 6 | 一色多义：(d) 的 `selected` 与 SMD 数据点同形同色；`no` 也用红 | `selected` 改**实心方块**；`no` 改**中性斜体**；红色只保留"参考线"一义 |
| 7 | (a) 横轴是"相对基线的提升"，条越长代表基线越弱，原排布易被误读 | 图内加 `longer bar = weaker baseline`；轴标签改为 `(95% CI for adjudicated rows)`（两条参考分数无区间）；(a) 标题与图例点名 `AC-Twin audit` |

另外顺手处理：`(c)` 六个分层/trim 统一同色（原两种蓝无解释）；`(c)` 副标题改
"not independent confirmation"；`(d)` 补样本量 `n = 5,000 / 2,000`；画布由 118 → 126 mm
以拉开 (a) 轴标签与 (c) 标题的间距（原净空隙仅 ≈0.7 mm）。

### 二次修订（2026-09-28，作者复核后）

| # | 决定 | 结果 |
| --- | --- | --- |
| 8 | 两条参考评分是否删除 | **保留**。查证 `TABLE_6.md` 的“支持性事实”块已列出二者同值的 ΔAURC/ΔRegret@80（+0.5848/+0.4131 与 +0.5240/+0.3604），且该块用途即支撑“击败全部评估基线”；删掉会让该主张只剩表格在扛。行标签由 `(reference)` 细化为 `(historical reference)` / `(curve reference)` |
| 9 | (d) 两行小字是否必要 | **部分移出**。图内只留 `3 of 6 SMDs shifted` 与 `domain AUC 0.983 [0.980, 0.986]` 两行各表一事；`n = 5,000 / 2,000` 与 `effect in Fig. 3b` 移入图注（样本量属域分类器、与六个特征的 SMD 不是同一个量；图内交叉引用非期刊惯例且图号可能再变） |

> 说明：合并成**一行**不安全 —— 合并串约 195.8 pt，而 (d) 板宽仅 188.8 pt，会溢出 2.4 mm，
> 正是首轮裁掉结尾 `]` 的同一处。两行是安全下限。

### 三次修订（2026-09-28，作者指出 (a) 图例遮挡）

| # | 问题 | 结果 |
| --- | --- | --- |
| 10 | (a) 的图例压住 MARGIN_ONLY 与 SEED_VOTE_ENTROPY 两行的数据条 | 删掉无图例符号的 `AC-Twin audit (this work)` 条目（**与面板标题重复**，标题已写 `AC-Twin audit: baseline ladder`），图例缩为两条并把 `handlelength` 收到 1.3，右端止于数据 x≈0.21，而最短的数据条从 x=0.293 起——**不再压任何一行** |
| 11 | 读图提示 `longer bar = weaker baseline` 紧贴 `CONSTANT_RANDOM` 行的菱形，像句子的一部分 | 移到坐标区**右上空白带**（`0.99, 1.005`），位于最长那条之上、图例之右，与数据零接触 |

> 图例现在只有两个符号条目；方法名由图内标题承担，满足蓝图 §12-3“结果图须点名 `AC-Twin audit`”。

## 当前交付

- `Fig4_credibility_closure.svg` / `.png`（600 dpi）/ `.pdf`：主版本；
- `fig4_panel_a_data.csv`：五条评分（三条标准基线 + 两条参考）在 Merge 确认集上相对 AC-Twin audit 的改善点估计与 95% CI；
- `fig4_panel_b_data.csv`：五个 audit-head seed 的两个指标点估计与 95% CI；
- `fig4_panel_c_data.csv`：四个边际分层与两个 trim 的 `I_AURC_REGRET`；
- `fig4_panel_d_data.csv`：三个 OOD profile 的资格/优先级/选中，以及六个场景前特征的绝对 SMD 与阈值判定；
- `fig4_data.xlsx`：作者整理的 D 面板数据表（与 CSV 同源）；
- `build_fig4.py`：确定性重绘脚本；
- `FIGURE_CONTRACT.md`：图面合同；
- `FIGURE_LEGEND.md`：正文图注草稿。

## 运行

```powershell
python paper/figures/fig4/build_fig4.py
```

脚本只读冻结确认表与冻结选择回执，不重跑 bootstrap、不重选基线、不重新裁判。
绘图前对三条标准基线的相对改善做逐位断言：脚本按 `1 - AURC_model / AURC_baseline`
重算，必须等于 `Table_C2_baselines.csv` 的正式 `i_*_point`（atol 1e-12）。
五 seed 面板另断言每个 seed 的 95% CI 下界 `> 0`（方向不翻转才画）。

## 面板与取数字段

| 面板 | 内容 | 字段 |
| --- | --- | --- |
| (a) | 基线阶梯：三条标准基线的 AURC 与 Regret@80 相对改善（含 95% CI）＋两条只能画点估计的参考评分 | `Table_C2_baselines.csv`；正式 CI 同 `stage4cr2_confirmatory_report.json#decision.metrics` |
| (b) | 五个 audit-head seed 的两指标配对效应，红虚线为正式集成的裁判值 | `Table_C3_seeds_strata_trim.csv` |
| (c) | 四个边际分层与 trim 1%/5% 的 `I_AURC_REGRET`；只说明方向稳定 | 同上 |
| (d) | 三个 OOD profile 的资格矩阵＋六个场景前特征的绝对 SMD（阈值 0.5）＋域分类 AUC | `stage4d_ood/pilot/ood_profile_selection.json`、`stage4d_ood/reports/stage4d_ood_confirmatory_report.json#decision.domain_shift` |

## 版面与表述规则

- (a) 三条标准基线的区间是**正式裁判值**；`COMBINED_TOP2_MAX` 与 `CONSTANT_RANDOM`
  只有点估计、没有裁判区间，图注必须写明，不得暗示它们有 CI；
- (a) 两个指标不指向同一条"最强"标准基线（AURC 最低为 `MARGIN_ONLY`，Regret@80 最低为
  `SEED_VOTE_ENTROPY`），因此任何一句"最强基线"都必须限定指标；
- (c) 只说明方向稳定，**不把任何分层当作独立 confirm**；完整数值见 Table S3；
- (d) 域分类器输出只证明失配存在，**从不进入方法打分**；`GEOMETRY_OOD` 与 `JOINT_OOD`
  均通过资格，`BEHAVIOR_OOD` 未通过，`JOINT_OOD` 按预定优先级选择；
- 不得写"对任意 OOD 鲁棒"；`interaction-high` 等分层区间若跨零须如实显示；
- 不把本图的开发/确认范围混写：本图全部来自 Merge 确认集与 D1 的资格记录；
- **(a) 的横轴是"相对基线的提升"，条越长 = 该基线越弱**，不得读成"最重要的胜利在最上面"；
- **跨范围最窄余量三者不同**：Merge **26.0%**、JOINT_OOD **18.6%**、MetaDrive **10.1%**。
  本图 (a)–(c) 只能用 26.0%；10.1% 自带 MetaDrive 范围限定，见 Table 6；
- **区间不可用于行间比较**：R2 确认的 bootstrap 对每次重采样用**同一个重采样索引向量**重算该
  draw 的全部指标，故 (a) 各行、(b) 各 seed 的 CI 彼此高度相关；"两区间不重叠 ⇒ 显著"是错的。

## 已登记的来源小瑕疵（只读，不修改）

`Table_C2_baselines.csv` 的两个区间列命名不一致：AURC 用 `i_aurc_ci95_lower/_upper`，而
Regret@80 用 `i_regret80_ci95_lower` + **`i_regret80_ci_upper`**（上界少了 `95`）。
脚本按实际列名显式映射（见 `CI_COLS`）并在代码中注明，**未改写冻结文件**。

## 来源

- (a) `experiments/stage4_reliability/stage4cr2_confirm/reports/Table_C2_baselines.csv`
- (b)(c) `experiments/stage4_reliability/stage4cr2_confirm/reports/Table_C3_seeds_strata_trim.csv`
- (d) `experiments/stage4_reliability/stage4d_ood/pilot/ood_profile_selection.json`、
  `experiments/stage4_reliability/stage4d_ood/reports/stage4d_ood_confirmatory_report.json#decision.domain_shift`
