# Graphical abstract｜从平均误差到可审计的反事实决策（绘制说明 v1）

> 依据：`paper/2026-09-26_AC-Twin视觉叙事蓝图_v1.0.md` 的 graphical abstract 条目与视觉规则；本图不占正文图号。
> 版式规则。本文件取代 2026-09-25 的旧版说明；旧版描述的"左场景 + 中两分支 + 右三张并列
> 测量卡片"设计及蓝/橙/紫配色**已作废**，不与蓝图 v1.0 兼容。

## 定位

**英文工作标题：** *AC-Twin overview: from response-conditioned factual evidence to auditable counterfactual decisions*

**唯一命题：** 同一个回溯反事实任务中，平均预测误差遗漏了两类决策所需信息——事实证据的条件价值，以及所选动作是否会产生高 regret；AC-Twin 分别测量并审计这两件事。

论文级 graphical abstract，横向、**非对称**，不做四个等宽方框。**图内无标题、无页脚**（由图注承接）。

## 画布与字号

| 项 | 取值 |
| --- | --- |
| 画布 | 180 × 84 mm（双栏通栏 100%） |
| 外边距 | 左右 3 mm |
| 坐标 | 直接以毫米为单位（x 0–180，y 0–84，等比） |
| 最小字号 | **7.5 pt**（投稿 GA，不适用正文 Fig. 1/2/3 的 6.6 pt 例外） |
| panel letter | 9.6 pt 加粗，左上角固定位置 |
| 面板标题 | 9.2 pt 加粗 |
| 正文标签 | 7.5–8.0 pt |

## 构图

### 主行（y 20–81 mm）

| 区 | 宽度 | 角色 | 必须出现 |
| --- | --- | --- | --- |
| **A Twin event** | 46.1 mm（28%） | 设定 | 干预前历史进入两个世界：事实 continuation（上）、替代 ego action（下）；3–4 辆车 + 简洁轨迹，**不画真实道路细节** |
| **B Evidence-value lens** | 55.9 mm（34%） | **发现问题** | 三张预测小卡片（history-only / correct suffix / shuffled suffix）；下方四个不同高度的小点表示 `V_evt` 随 response bin 变化 |
| **C Decision audit** | 62.5 mm（38%，hero） | **解决问题** | 十个动作分支汇入基础选择 → Direct risk head → 审计风险分 → 低风险保留/高风险标记；右端 risk–coverage 小曲线表示评价 |

- A 与 B 之间：细灰结构箭头（非因果）。
- B 与 C 之间：**不用实心因果箭头**，改为细虚线连接 + 竖排文字 **`Motivates a decision-level audit`**。
- C 用浅蓝底衬强化 hero 地位；B 用极浅中性底衬；A 不加底衬（留白起步，形成非对称）。

### 底部证据带（y 3–15 mm）

一条细带，三个范围名，**同色同一底衬、只放范围名、不放任何效应大小**：

`Merge in distribution` → `Preregistered generator mismatch` → `MetaDrive / Waymo · platform-internal retraining`

带上方左侧一行 7.5 pt 中性小字说明该带的类别（`Measured evidence scopes`）。三个范围**不得**使用三种不同主方法颜色（§9.1）。

## 视觉规则

- 配色**继承 Fig. 2 六色语义体系**，一色一义、跨图一致：
  `#4D93D6` 主蓝 = AC-Twin 方法；`#A4C8EA` 浅蓝 = 方法浅底；`#D57873` 暖红 = 反事实分支与真值族；`#F1C2AD` 浅桃 = 暖色浅底；`#A8A8A9` 灰 = 冻结预测器/中性；`#D9D9D9` 浅灰 = 框线与中性条。
- 事实 continuation 用中性深灰；替代 ego action 用暖红（与 Fig. 2 的“反事实分支”同义）。
- 每图只保留一个 legend；能在图内直接标注的不重复进 legend。
- 主结论面板不与辅助面板等权（C 获得底衬与最大宽度）。

## 数字与字符串禁区

| 禁止 | 原因 |
| --- | --- |
| 网络层数、315 维、参数量、损失函数、网络结构图 | 归 Fig. 2 |
| Stage / Gate / 终态字符串 / lock / hash | 只进补充材料与复现说明 |
| `23.91%` oracle 数字 | 蓝图 §157 明令 |
| 结果数字（四箱 `V_evt` 的数值、CI、效应大小） | 主图不标结果数字 |
| "external generalization" | 蓝图 §555；MetaDrive 固定表述 `platform-internal retraining` |
| "平均误差导致高 regret"的实心因果箭头 | 不画未证实的因果机制链 |

**唯一允许进入图内的冻结数据**：B 区四个小点的**高度**，取自 `experiments/stage2_confirmatory/merge/report.json#statistics.bins.*.V_evt`（`0.3636 / 0.0841 / -0.0661 / 0.2064`），必须与 Fig. 1 完全一致，且**不标数值**。

## 图注必须交代

1. C 区右端的 risk–coverage 曲线是**示意**，不是任何确认集的结果；
2. B 区四个点的高度是上述冻结 `V_evt` 值；
3. 底部三范围各自的模型关系（前两个共用冻结 Merge 权重，第三个为平台内重训），三行不作跨范围排名。

## 来源

- 概念、`V_evt`、响应分层：`plans/2026-09-06_AC-Twin反应式事实证据价值与路由研究计划_v1.0.md` §2、§25
- 十动作、成本与 regret：`plans/2026-09-16_AC-Twin_Stage4CR2决策风险扩样开发验证实施计划_v0.2.md` §3
- 三个确认范围：同上正式计划 §25
- 四箱 `V_evt`：`experiments/stage2_confirmatory/merge/report.json#statistics`

## 制作方式

确定性脚本 `build_graphical_abstract.py` 重绘，输出 SVG / 600 dpi PNG / PDF，保持可编辑矢量文本
（`svg.fonttype="none"`、`pdf.fonttype=42`）。不使用生成式图片作为最终图。
