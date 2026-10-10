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

冻结 decision-risk 方法在预注册 Merge 生成机失配下仍获支持。

禁止由此声称：跨平台外部有效性、真实道路泛化、线上安全提升、闭环策略已改善或对任意 OOD 鲁棒。

单项 CI 是边际 CI，不宣称所有次分析联合 95% 覆盖。描述性次分析只入次表，不进主 AND Gate，不挽救主结果。无论裁判成败，不改写 Stage 4C-R2 或任何更早阶段已有终态。
