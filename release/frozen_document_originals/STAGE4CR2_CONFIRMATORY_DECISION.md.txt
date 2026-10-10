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

在所测Merge、回溯事实后缀可用的十动作任务中，直接监督决策风险审计改善了选择性保留组的regret。

确认集必须原样报告动作频率；即使总体Gate通过，也不得声称十动作获得均衡证据、每种动作同等可靠或稀有动作已被充分验证。

禁止由此声称：复杂DCR结构优势、响应特征的独立因果贡献、仅增加数据导致成功、跨场景/跨平台泛化、线上预测可用、实际碰撞率降低或闭环策略改善。

Trim 稳健性依赖真值，仅为预注册稳健性分析，不是可部署的过滤器。单项 CI 是边际 CI，不宣称所有次分析联合 95% 覆盖。无论确认成败，不改写 Stage 4B 或 R2 已有终态。
