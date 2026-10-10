# Stage 4B Independent Risk Confirmation Decision

> Status: `STAGE4B_RISK_CONFIRMED`
> Plan: `AC-TWIN-RFER-STAGE4B-RISK-CONFIRM-2026-09-15-V0.1`
> Scope: independent within-Merge confirmation; no claim of cross-scene external validity

- finalist: `COMBINED_RISK_HEAD/mlp`
- Primary predictor: `END_TO_END_MIXTURE`
- Baselines: `UNCERTAINTY_ONLY` + `GEOMETRY_ONLY`
- Confirmation episodes: 2000 (selected_n=2000)

## Primary confirmation conditions (§8.2 AND)

- [x] `i_aurc_vs_uncertainty`
- [x] `i_c80_vs_uncertainty`
- [x] `i_aurc_vs_geometry`
- [x] `i_c80_vs_geometry`
- [x] `pr_auc_minus_prevalence`
- [x] `at_least_4_of_5_pairs_positive`
- [x] `no_strata_below_floor`
- [x] `gate_q_passed`

- I_AURC vs UNCERTAINTY = +0.0728(95% CI lower bound +0.0507)
- I_C80 vs UNCERTAINTY = +0.0307(95% CI lower bound +0.0126)
- I_AURC vs GEOMETRY = +0.2608(95% CI lower bound +0.2301)
- I_C80 vs GEOMETRY = +0.1202(95% CI lower bound +0.0948)
- PR-AUC - prevalence = +0.2897(95% CI lower bound +0.2485)

- Positive pairs = 5/5
- Minimum across strata I_AURC = +0.0182(floor -0.10)

Decision: `STAGE4B_RISK_CONFIRMED`
