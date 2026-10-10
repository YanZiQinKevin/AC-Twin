# Stage 4A Development-Screen Decision

> Status: `STAGE4A_RISK_SIGNAL_FEASIBLE`
> Plan: `AC-TWIN-RFER-STAGE4A-RISK-DEV-2026-09-11-V0.1`
> Scope: development screening, not confirmation; proceeding required separate Stage 2B and independent risk-confirmation plans.

- finalist: `COMBINED_RISK_HEAD`(combined implementation: `mlp`)
- Strongest standard baseline: `UNCERTAINTY_ONLY`
- PRIMARY_PREDICTOR: `END_TO_END_MIXTURE`
- development-screen episodes: 1000

## Continuation conditions (§9.2)

- [x] `i_aurc_ge_0.02_and_ci_lower_gt0`
- [x] `i_c80_ge_0.02_and_ci_lower_gt0`
- [x] `pr_auc_minus_prevalence_gt0`
- [x] `at_least_4_of_5_pairs_positive`
- [x] `no_stratified_reversal`
- [x] `qa_passed`

- I_AURC = +0.0866 (80% CI lower bound +0.0664)
- I_C80 = +0.0491 (80% CI lower bound +0.0244)
- PR-AUC - prevalence = +0.3518
- Positive pairs = 5/5
- Major stratified reversal = False []

Decision: `STAGE4A_RISK_SIGNAL_FEASIBLE`
