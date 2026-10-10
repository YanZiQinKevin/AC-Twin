# Stage 4D D1 Model-Mismatch OOD Confirmation — Decision

- plan_id: AC-TWIN-RFER-STAGE4D-D0D1-OOD-2026-09-20-V1.0
- n_groups: 2000
- terminal status: **STAGE4D_OOD_ROBUSTNESS_CONFIRMED**
- gate TASK-OOD status: TASK_OOD_QUALIFIED
- primary baseline (validation-frozen, Gate-only): SEED_VOTE_ENTROPY

## Primary endpoints (95% CI, two-sided percentile, 10,000 paired draws)

- I_AURC_REGRET = 0.20765763192314823 [0.12759126569360502, 0.30501813924256599]
- I_REGRET@80 = 0.18548177054721987 [0.12738456252429023, 0.23662535510816202]
- PR-AUC - prevalence = 0.2156795832344866 [0.18054367549093595, 0.25308292097597934]
- AURC_REGRET (ensemble) = 0.024473184245042703; REGRET@80 (ensemble) = 0.02628875305549715; coverage = 0.80000000000000004; rejected fraction = 0.19999999999999996

## Paper boundary (plan §8.2)

The frozen decision-risk method remained supported under the preregistered Merge generator mismatch.

These results do not establish cross-platform external validity, real-road generalization, online safety gains, improved closed-loop policies, or robustness to arbitrary OOD shifts.

Individual CIs are marginal intervals; joint 95% coverage across all secondary analyses is not claimed. Descriptive secondary analyses belong in secondary tables, do not enter the primary AND gate, and cannot rescue the primary result. Adjudication outcomes do not overwrite Stage 4C-R2 or earlier terminal states.
