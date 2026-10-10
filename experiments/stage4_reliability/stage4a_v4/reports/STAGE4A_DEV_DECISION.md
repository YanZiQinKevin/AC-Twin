# Stage 4A Development-Screen 裁判

> 状态：`STAGE4A_RISK_SIGNAL_FEASIBLE`
> 计划：`AC-TWIN-RFER-STAGE4A-RISK-DEV-2026-09-11-V0.1`
> 性质：开发筛选（非确认性结果）；通过也需另立 Stage 2B 与独立风险确认计划

- finalist：`COMBINED_RISK_HEAD`（combined 实现 `mlp`）
- 最强标准基线：`UNCERTAINTY_ONLY`
- PRIMARY_PREDICTOR：`END_TO_END_MIXTURE`
- development-screen episodes：1000

## 继续条件（§9.2）

- [x] `i_aurc_ge_0.02_and_ci_lower_gt0`
- [x] `i_c80_ge_0.02_and_ci_lower_gt0`
- [x] `pr_auc_minus_prevalence_gt0`
- [x] `at_least_4_of_5_pairs_positive`
- [x] `no_stratified_reversal`
- [x] `qa_passed`

- I_AURC = +0.0866 (80% CI 下界 +0.0664)
- I_C80 = +0.0491 (80% CI 下界 +0.0244)
- PR-AUC - prevalence = +0.3518
- 正向 pair 数 = 5/5
- 分层重大反转 = False []

裁决：`STAGE4A_RISK_SIGNAL_FEASIBLE`
