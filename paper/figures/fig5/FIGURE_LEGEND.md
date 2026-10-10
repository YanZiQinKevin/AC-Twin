# Fig. 5 Legend

**Fig. 5 | MetaDrive scenes and qualitative decision-regret cases.** **a,b,** A Waymo-derived ScenarioNet scene and its ego-heading view: factual continuation in green, other vehicles in light blue, and LEFT × 1 counterfactual ego trajectory in dashed red. This pair was selected from 20 visualization candidates by a deterministic legibility rule, solely for illustration. **c,** A correctly detected high-regret example selected from 205 eligible confirmation cases by the specified visualization rule. **d,** The mechanically selected worst false-negative. The selected branch in c has greater proximity cost and less progress; d mainly reflects efficiency loss. Stars mark maximum selected–oracle ego-trajectory separation, not necessarily maximum proximity cost. These selected cases neither estimate success rate nor depict an executed fallback. The pipeline was trained and frozen within MetaDrive. Population results are in Fig. 4 and Table 6.

---

## Interpretation notes (outside the manuscript caption)

Cost clarification (2026-10-04): the selected branch in (c) has both higher proximity cost and lower progress; the missed high-regret case in (d) mainly reflects progress-efficiency loss rather than a large proximity term. Stars mark maximum branch-trajectory separation, not maximum proximity cost. See [fixed-case cost explanation](FIXED_CASE_COST_EXPLANATION.md); the fixed PNG is unchanged.

- (a)/(b): mechanically selected for readability from a frozen K=20 pool; scene `1cac180d139c96de`, `t*=13`, maximum factual/counterfactual separation approximately `3.134 m`. This is qualitative visualization, excluded from adjudication.
- (b): red dashed counterfactual trajectory must remain visible above the green factual trajectory; fixed legend `Counterfactual: LEFT × 1`.
- (c): best-of-205 is a qualitative case under a prespecified rule, not a population success rate or agreement between selected and oracle actions.
- (d): show the failure alongside the success to avoid favorable-case-only presentation.
- MetaDrive/Waymo evaluation follows platform-internal training and freezing, not zero-shot Merge-weight transfer.
- Selected/oracle differences illustrate decision regret, not executed online fallback or improved closed-loop control.
- Overall quantitative claims must refer to full statistical figures/tables; these cases do not independently establish population effects.
