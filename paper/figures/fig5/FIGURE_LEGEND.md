# Fig. 5 Legend

**Fig. 5 | MetaDrive scenes and qualitative decision-regret cases.** **a,b,** A Waymo-derived ScenarioNet scene and its ego-heading view: factual continuation in green, other vehicles in light blue, and LEFT × 1 counterfactual ego trajectory in dashed red. This pair was selected from 20 visualization candidates by a deterministic legibility rule, solely for illustration. **c,** A correctly detected high-regret example selected from 205 eligible confirmation cases by the specified visualization rule. **d,** The mechanically selected worst false-negative. The selected branch in c has greater proximity cost and less progress; d mainly reflects efficiency loss. Stars mark maximum selected–oracle ego-trajectory separation, not necessarily maximum proximity cost. These selected cases neither estimate success rate nor depict an executed fallback. The pipeline was trained and frozen within MetaDrive. Population results are in Fig. 4 and Table 6.

---

## 中文备忘（不入正文）

2026-10-04 成本解释：c 的所选分支同时有更高接近度成本及更低进度；d 的漏检主要由进度效率损失构成，而非较大的接近度项。星号是最大分支轨迹分离点，不是成本最大接近点。数值与来源见 [固定案例成本解释](FIXED_CASE_COST_EXPLANATION.md)；固定 PNG 不变。

- `(a)/(b)`：从冻结 K=20 候选池按预先写死的图面规则机械选择，场景 `1cac180d139c96de`、`t*=13`，最大事实/反事实分离约 `3.134 m`。这是定性可视化，不进入任何实验裁判。
- `(b)`：红色虚线必须在绿色事实轨迹之上清晰可见；图例固定为 `Counterfactual: LEFT × 1`。
- `(c)`：best-of-205 只能写作预先指定规则下的定性成功案例，不等于总体成功率，也不等于模型动作与 oracle 动作相同。
- `(d)`：必须与成功案例同时展示，避免只展示有利个例。
- MetaDrive/Waymo 是平台内训练与冻结后的跨平台验证；不得写“Merge 权重零样本迁移”。
- selected/oracle 的差异只解释 decision regret；不得写成已经执行在线回退或闭环控制改善。
- 正文定量结论必须回指统计图表，本图不单独承载总体效应量。
