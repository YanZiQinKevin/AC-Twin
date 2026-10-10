# Stage 4B 独立风险确认裁判

> 状态：`STAGE4B_RISK_CONFIRMED`
> 计划：`AC-TWIN-RFER-STAGE4B-RISK-CONFIRM-2026-09-15-V0.1`
> 性质：Merge 域内独立确认（confirmatory）；不承担跨场景外部有效性

- finalist：`COMBINED_RISK_HEAD/mlp`
- 主预测器：`END_TO_END_MIXTURE`
- 基线：`UNCERTAINTY_ONLY` + `GEOMETRY_ONLY`
- 确认 episodes：2000（selected_n=2000）

## 主确认条件（§8.2 AND）

- [x] `i_aurc_vs_uncertainty`
- [x] `i_c80_vs_uncertainty`
- [x] `i_aurc_vs_geometry`
- [x] `i_c80_vs_geometry`
- [x] `pr_auc_minus_prevalence`
- [x] `at_least_4_of_5_pairs_positive`
- [x] `no_strata_below_floor`
- [x] `gate_q_passed`

- I_AURC vs UNCERTAINTY = +0.0728（95% CI 下界 +0.0507）
- I_C80 vs UNCERTAINTY = +0.0307（95% CI 下界 +0.0126）
- I_AURC vs GEOMETRY = +0.2608（95% CI 下界 +0.2301）
- I_C80 vs GEOMETRY = +0.1202（95% CI 下界 +0.0948）
- PR-AUC - prevalence = +0.2897（95% CI 下界 +0.2485）

- 正向 pair 数 = 5/5
- 分层最小 I_AURC = +0.0182（下限 -0.10）

裁决：`STAGE4B_RISK_CONFIRMED`
