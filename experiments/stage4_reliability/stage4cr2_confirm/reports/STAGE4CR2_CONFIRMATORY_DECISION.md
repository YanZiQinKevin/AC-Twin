# Stage 4C-R2 Independent Confirmation — Decision

- plan_id: AC-TWIN-RFER-STAGE4CR2-CONFIRM-2026-09-17-V0.1
- n_groups: 2000
- terminal status: **STAGE4CR2_DECISION_RISK_CONFIRMED**
- gate TASK-CONFIRM status: TASK_CONFIRM_QUALIFIED
- primary baseline (validation-frozen, Gate-only): SEED_VOTE_ENTROPY

## Primary endpoints (95% CI, two-sided percentile, 10,000 paired draws)

- I_AURC_REGRET = 0.4392150830944847 [0.30012840077251396, 0.50437280371748561]
- I_REGRET@80 = 0.26003258738455293 [0.16954813931198204, 0.32296069797164517]
- PR-AUC - prevalence = 0.314627150808102 [0.27440855743595216, 0.35550639898910447]
- AURC_REGRET (ensemble) = 0.010782781291834763; REGRET@80 (ensemble) = 0.014346448253686962; coverage = 0.80000000000000004; rejected fraction = 0.19999999999999996

## Paper boundary (plan section 7)

On the evaluated ten-action Merge task with retrospective factual suffixes available, directly supervised decision-risk auditing reduced regret among selectively retained groups.

Report confirmation action frequencies as observed. Passing the overall gate does not establish balanced evidence across ten actions, equal reliability for every action, or adequate validation of rare actions.

These results do not establish an advantage of complex DCR architectures, an independent causal contribution of response features, success attributable solely to more data, cross-scene or cross-platform generalization, online prediction readiness, lower actual collision rates, or improved closed-loop policies.

Trimming requires ground truth and is a preregistered robustness analysis, not a deployable filter. Individual CIs are marginal intervals; joint 95% coverage across all secondary analyses is not claimed. Confirmation outcomes do not overwrite the existing Stage 4B or R2 terminal states.
