# Graphical Abstract Caption Draft

**Graphical abstract | AC-Twin overview: from response-conditioned factual evidence to auditable counterfactual decisions.**
(a) A shared pre-intervention history (Observed prefix) opens two worlds: the factual
continuation and an alternative ego action. (b) Three prediction variants — history-only, the
correct factual suffix and a matched shuffled suffix — and, below them, the event-specific value
`V_evt` of the factual suffix in each frozen response bin (zero / low / mid / high). The four dot
heights are the frozen Stage-2 point estimates (`0.3636`, `0.0841`, `-0.0661`, `0.2064`) and carry
no interval here; the full per-bin intervals and the three preregistered contrasts are in Fig. 1.
(c) The ten candidate action branches of one decision group enter the frozen base selector; the
selected action is scored by the directly supervised audit head, and the lowest-risk decision
groups are retained while the high-risk groups are flagged. The risk–coverage glyph at the right is
a schematic of that evaluation and **is not a result from any confirmation set**; the confirmed
risk–coverage curves are in Fig. 3. The audit ranks the selected action of each decision group; it
does not update the frozen predictor and does not execute a fallback policy. The bottom band lists
the three measured evidence scopes: in-distribution Merge and the pre-registered generator mismatch
share the frozen Merge weights, whereas MetaDrive / Waymo uses a pipeline retrained and frozen
inside that platform. Panel (b) motivates the decision-level audit in (c); the three scopes differ
in task distribution, baseline and model relation, and their effect sizes are neither ranked nor
pooled against one another.

---

## 中文备忘（不入正文）

- 本图**无图内标题与页脚**，全部说明由本图注承接；与正文 Fig. 1 / Fig. 2 / Fig. 3 同一做法。
- 图身**不含任何结果数字**；唯一进入图内的冻结量是 (b) 四个点的**高度**。
- (c) 右端的 risk–coverage 曲线是**示意**，必须与 Fig. 3 的确认曲线明确区分，图注不可省略这一句。
- 底部三范围**只放范围名**，不放效应大小；三块用同一底色（§9.1：范围差异不用三种主方法颜色表达）。
- 禁用词：`external generalization`；MetaDrive 固定为 `platform-internal retraining`。
- 术语：结果层方法名统一 `AC-Twin audit`，`Direct audit head` 只作组件名（图内组件标签用
  `Direct risk head`，与 Fig. 2 的组件命名一致，正文首次出现时对齐）。
