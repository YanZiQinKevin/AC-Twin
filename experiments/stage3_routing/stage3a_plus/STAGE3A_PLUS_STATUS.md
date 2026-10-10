# AC-Twin Stage 3A Plus 实施状态（工程里程碑）

> 记录性质：**工程实施/QA 记录**，非科研日志，非确认性结果。
> 对齐计划：`AC-TWIN-RFER-STAGE3A-PLUS-2026-09-09-V0.1`（`APPROVED_FOR_AUTONOMOUS_IMPLEMENTATION`）
> 上位计划：`AC-TWIN-RFER-2026-09-07-V1`；直接前置：`AC-TWIN-RFER-STAGE3-2026-09-08-V0.2`、`AC-TWIN-RFER-STAGE3A-REPAIR-2026-09-09-V0.1`
> 形成日期：2026-09-09
> 边界：本文件只承载 Stage 3A Plus 的软件实现、测试、诊断、开发 round、冻结、功效、采集与资格裁判里程碑。科研结论在形成后统一写入科研日志，不在此处断言。

## 状态机位置

```text
APPROVED_FOR_AUTONOMOUS_IMPLEMENTATION
→ SOFTWARE_TESTED
→ FORENSIC_DIAGNOSTIC_COMPLETE
→ DEV_ROUND_0_COMPLETE
→ [DEV_ROUND_1_COMPLETE]
→ [DEV_ROUND_2_COMPLETE]
→ STAGE3A_PLUS_DEVELOPMENT_STABLE
→ FINAL_MODEL_FROZEN
→ COMPUTE_BUDGET_RECORDED_STAGE3A_PLUS
→ QUALIFICATION_POWER_FROZEN
→ MANUAL_COLLECTION_COMPLETE
→ FROZEN_INFERENCE_COMPLETE
→ AUTOMATIC_SINGLE_ADJUDICATION_STARTED
→ QUALIFICATION_ADJUDICATED
→ RESPONSE_DEV_QUALIFIED_STAGE3A_PLUS
   or INCONCLUSIVE_RESPONSE_HEAD_INSUFFICIENT_STAGE3A_PLUS
   or INVALID_RUN_ENGINEERING_OR_PROTOCOL_FAILURE
```

当前：**`RESPONSE_DEV_QUALIFIED_STAGE3A_PLUS`**（终态，2026-09-10）。一次性资格裁判通过：Round 0 双开发集通过（未触发重训）；校准家族 `CAL_PER_SEED_VECTOR`；联合功效 n=1000（min(A,B)=0.925）；资格集 1000 episode（十层各 100，零历史重叠）；正式 10,000 bootstrap（seed 20260929）核心+稳定性+Q+ 全过，十 seed `I_Brier` 全正。确认性结果见 `logs/2026-09-10_AC-Twin科研日志.md`。

## 数据角色（本计划冻结）

| 数据 | 角色 | 可拟合参数 | 可选模型 | 可作最终裁判 |
|---|---|---:|---:|---:|
| `merge_2000` split 0 | train | 是 | 否 | 否 |
| `merge_2000` split 1 | validation | 是，仅校准/早停/基线冻结 | 是 | 否 |
| `merge_2000` split 2 | historical development diagnostic A | 否 | 是 | 否 |
| `stage3a_replacement_dev` | historical development diagnostic B | 否 | 是 | 否 |
| `stage3a_plus_qualification` | single-use qualification | 否 | 否 | 是，一次 |
| Stage 2 holdout | sealed confirmatory evidence | 否 | 否 | 禁止读取 |
| 未来 Gate C 数据 | future sealed evidence | 否 | 否 | 禁止读取 |

## 冻结常量

- 生产 seed：`[101,202,303,404,505]`；压力 seed：`[606,707,808,909,1010]`（十 seed 全部进入稳定性报告，禁止删减）。
- 资格样本量候选：`[400,600,800,1000,1200,1600,2000]`；目标功效 0.90；功效 seed=`20260930`；资格采集 seed=`20261009`；正式裁判 10,000 次 bootstrap、seed=`20260929`。
- 资格集生成合同：`merge-v0`、action `0..4`、duration `{1,3}`、十个 action×duration 层等额、schedule 长度 = n×15；逐项同原 merge 合同。
- 资格集路径：`data/object_centric_stage/stage3a_plus_qualification/`；实验产物：`experiments/stage3_routing/stage3a_plus/`。

## 任务进度

- [x] **Task 1 治理与只读边界**：`assert_stage3a_plus_source_role`（qualification 角色按路径与已知 SHA256 拒绝 Stage 2 / old merge split 2 / replacement / 旧 stage3a* / Gate C）；`STAGE3A_PLUS_PLAN_ID`、`STAGE3A_PLUS_DEV_SEEDS`、`STAGE3A_PLUS_STRESS_SEEDS` 等冻结常量；本状态文件建立。`tests/test_stage3a_plus_governance.py`（7 项）。
- [x] **Task 2 统一诊断输出**：`stage3a_plus_statistics.decompose_seed_metrics`（十 seed 全指标 + 五核心指标 ensemble bootstrap CI + Brier 四维分解 + 15-bin reliability + 校准语义 TV/flip/entropy）、`classify_calibration_failure`（六态机械分类）、`direction_consistency`；`run_stage3a_plus.py` 诊断核心 `run_diagnostic` + `diagnose` 命令（四校准候选 validation 拟合、dev A/B 分解、工程复验）。诊断实跑（含 5 压力 seed 训练）由编排器统一启动。`tests/test_stage3a_plus_statistics.py`、`tests/test_stage3a_plus_driver.py`。
- [x] **Task 3 校准器与签名**：`stage3a_plus_calibration.py` 四冻结候选（SHARED_T / PER_SEED_T / PER_SEED_VECTOR / PER_SEED_T_SHRINK）、`fit_calibration_candidate`、确定性签名、validation-only 拟合守卫、原子 receipt 拒绝覆盖。`tests/test_stage3a_plus_calibration.py`（9 项）。
- [x] **Task 4 Round 0 校准筛选**：`calibration-screen` CLI；`calibration_j_cal`（冻结公式 mean+0.5·SD+2·max(0,ECE−0.05)）、确定性 tie-break（1e-8 内取更简单者，顺序 SHARED_T→PER_SEED_T→T_SHRINK→VECTOR）、`validation_calibration_screen`（十 seed per-seed Brier + 生产 ensemble ECE）、单次写入 receipt（O_EXCL + canonical_fingerprint 链）、`--resume` 只复验不重选、`calibrator_from_selection_receipt` 自检。测试 6 项。
- [x] **Task 5 Round 1/2 训练目标**：`multiclass_brier_loss`（valid 均值、无 class weight）；`response_masked_loss` 增 `lambda_brier=0.0`/`label_smoothing=0.0`（默认逐位复现旧行为）；`train_model` 增 `weight_decay=0.0`（默认复现旧 optimizer）。驱动侧 `train-round` CLI：冻结候选空间校验、J_train 选择唯一 finalist、仅 finalist 补训压力五 seed、Round 0 校准家族 refit、门控（上一轮未通过才允许进入；通过后拒绝继续搜索）、单次写入 + 幂等复验。测试：response/training 7 项（fork）+ 驱动 6 项。
- [x] **Task 6 Gate R+ 与 hierarchical bootstrap**：`evaluate_gate_r_plus_core`（六条件）、`evaluate_gate_r_plus_stability`（六条件：8/10 Brier 正、中位数、最差非劣界 δ=5%·baseline、分层 CI、非退化）、`hierarchical_seed_episode_bootstrap`（先抽 seed 再抽 episode）、`evaluate_gate_r_plus`（core∧stability∧Q+）；decompose per-seed 增补 `i_mae/i_pr_*/non_degenerate/finite` 供 stability 直接消费。统计层 24 项测试通过。
- [x] **Task 7 开发状态机与最终冻结**：`evaluate-dev`（十 seed Gate R+_dev A/B 报告，含分层 bootstrap 与 Q+ checks）、`freeze-final`（receipt 覆盖 config/calibration/十 checkpoint/baseline/Gate/代码 SHA256/两个 dev report SHA256；未通过即拒绝；单次写入）、状态机 `load_state`/`advance_state`（含 NOT_READY 终态与资格阶段全链）、`run-all`（自动前进、异常停、重启只复验历史阶段不覆盖）。`DEV_READINESS_CHECKLIST.md` 建立。驱动测试累计 27 项。
- [x] **Task 8 完整 Gate joint power**：`prepare_stage3a_plus_power.py` 实现完整 AND-gate Monte Carlo（每 draw 评估全部 12 个采样条件：5 核心 CI + ECE guard + per-seed 核心点正性 + 8/10 + 中位数 + 最差非劣 + 分层 CI + 非退化）；joint power = min(A,B)；均值型统计闭式 bootstrap SE、rank 统计精确点值 + 1000 次全池 bootstrap SE 按 √(E/n) 缩放；NaN draw 计为失败；冻结值（候选/2000 MC/1000 inner/0.90/seed）逐项拒绝偏差；不可行时硬停（rc 3，无产物）；可行时单次冻结 config + seed schedule，三方哈希互绑（config↔schedule↔power report）。`tests/test_stage3a_plus_power.py`（9 项）。
- [x] **Task 9 资格集采集器**：`collect_stage3a_plus_qualification.py` 复用 Stage 2 引擎（shim 其协议钩子到资格冻结链）；`validate_qualification_config` 逐字段等于 merge 合同 + 冻结 scene/actions/采集 seed + max_attempts=n×15=schedule 长度 + 等额层开关 + power report/schedule 哈希绑定 + 禁止写入 Stage 2/merge_2000/replacement/旧 stage3a*/Gate C 路径 + run-dir 一致性；`_finalize` QA 强制十个 action×duration 层**精确等额**、duration {1,3}、seed 唯一且取自冻结 schedule、零历史重叠、ledger 审计只许七类工程拒绝原因（禁止基于实现 R/模型/Gate 筛选）、Task 10 loader 复验；receipt 增补 purpose/plan/哈希链/qa/code_sha256；12 episode smoke 只产 `ENGINEERING_SMOKE_ONLY` receipt 且不发布正式数据，正式数据存在后拒绝 smoke；resume 从 smoke checkpoint 继续；30 秒心跳线程；main 门控 QUALIFICATION_POWER_FROZEN → MANUAL_COLLECTION_COMPLETE，config 文件漂移 → rc 3。`tests/test_stage3a_plus_qualification.py`（11 项，含注入 rollout 的端到端采集与幂等复跑）。
- [x] **Task 10 资格锁、冻结推理与一次裁判**：`lock-qualification`（单次锁绑定数据 SHA256 + 冻结 receipt + power 哈希链；replaced data → rc 3）；`qualification-infer`（十 seed 冻结推理，逐 checkpoint 签名复验， replaced checkpoint/data → rc 3；predictions/seed_<seed>.npz ×10 + inference_receipt.json；幂等重启只复验）；`gate-r-plus-qualification`（所有守卫——锁/数据/推理文件/Gate 常量——在消耗裁判前执行，失败 rc 3 且**不**烧锁；`qualification_used.lock` 在读取资格真值前原子获取；正式 10,000 次 bootstrap seed 20260929，分层 seed×episode 用 +7000；三终态 QUALIFIED rc 0 / INCONCLUSIVE rc 3 / INVALID rc 2；二次裁判 rc 3）；run-all 已接线 budget→power→collect→lock+infer→gate 全链，power 不可行/裁判已消耗均为硬停。`QUALIFICATION_READINESS_CHECKLIST.md` 建立。`tests/test_stage3a_plus_qualification_gate.py`（13 项）+ run-all 接线测试 3 项。
- [x] **Task 11 预算、全量验证与交接（预算与回归部分）**：`code/benchmark_stage3a_plus_budget.py` 实测四单元速率（训练 12ep×1epoch=1.185s；推理 0.00069 s/seed/ep；decompose 0.159 s/resample@E=400；采集 0.227 s/attempt），`COMPUTE_BUDGET.md` 给出 ±25% 区间：最长单任务 Round 2 无早停上界 33 h（+25% 41.3 h）< 72 h ✓；磁盘上界 1.0 GB，1.25× ≪ 可用 552 GB ✓；CUDA 可用 ✓。全量回归 **718 passed / 0 failed / 0 warnings**（`pytest tests -W error`，184.7 s）。研究日志在资格裁判产生确认性结果后另行写入。
- [x] **流水线实跑**：run-all 自主推进（smoke→diagnose→Round 0→…→冻结→预算→功效→采集→锁→推理→一次裁判）。

## 硬停止规则（计划 §10）

1. Round 0 通过后不得继续 Round 1；Round 1 通过后不得继续 Round 2。
2. Round 2 未通过不得采集资格集（`STAGE3A_PLUS_DEVELOPMENT_NOT_READY`）。
3. joint power 在 n=2000 不足不得采集（`STAGE3A_PLUS_QUALIFICATION_POWER_INFEASIBLE`）。
4. 任一历史 seed overlap、合同漂移、签名不符、episode 错位、非有限输出或资格锁异常立即停止。
5. qualification 科学失败后不得在同一数据上补样本、换 seed、改校准器、改 Gate 或二次裁判。
6. `RESPONSE_DEV_QUALIFIED_STAGE3A_PLUS` 形成前不得启动 Stage 3B 主矩阵或 Gate C 正式工作。
