# Stage 3A Plus 开发就绪检查单（DEV_READINESS_CHECKLIST）

> 对齐计划：`AC-TWIN-RFER-STAGE3A-PLUS-2026-09-09-V0.1` Task 7。
> 用途：在进入资格阶段（预算→功效→采集→锁→推理→裁判）之前，逐项确认开发阶段已真正"就绪"。
> 本文件为工程检查单，不含科研结论。所有项由 `run-all` 的状态机与 receipt 链自动满足；此表用于审计与人工复核。

## A. 软件与治理

| # | 检查项 | 证据 | 验证方式 |
|---|---|---|---|
| A1 | 全部 Stage 3A Plus 专项测试通过（`pytest -W error`，0 失败 0 警告） | 测试运行输出 | `run-all` 的 smoke 阶段（`SOFTWARE_TESTED`） |
| A2 | 资格数据源守卫生效：qualification 角色拒绝 Stage 2 封存集、已查看源（按 SHA256）、旧 `stage3a*`、Gate C 前缀 | `stage3_data.assert_stage3a_plus_source_role` + `test_stage3a_plus_governance.py` | 单元测试（7 项） |
| A3 | 十 seed 冻结：生产 `[101,202,303,404,505]`、压力 `[606,707,808,909,1010]`，两处相等且不相交 | `STAGE3A_PLUS_DEV_SEEDS` / `STAGE3A_PLUS_STRESS_SEEDS` | 单元测试 + dev_config 校验 |
| A4 | 旧目录只读：`stage3a/`、`stage3a_retrain_v1/`、`stage3a_replacement_dev/` 不删、不移、不覆盖 | 驱动只读访问 | 代码审查 + receipt 路径检查 |

## B. 诊断（FORENSIC_DIAGNOSTIC_COMPLETE）

| # | 检查项 | 证据 | 验证方式 |
|---|---|---|---|
| B1 | 诊断报告存在且 plan_id 匹配 | `diagnostics/diagnostics_round0.json` | 重启复验器 |
| B2 | 生产五 checkpoint 签名与冻结选择 receipt 一致 | 报告 `signatures_ok=true` | `cmd_diagnose` 工程复验 |
| B3 | 十 seed 推理全部有限、episode 顺序硬断言通过 | 报告 `finite_ok=true` | 同上 |
| B4 | 校准失败分类为六态之一且输入完整 | `calibration_failure_status` + `classification_input` | §3.2 机械分类 |

## C. 开发 rounds（DEV_ROUND_0/1/2_COMPLETE）

| # | 检查项 | 证据 | 验证方式 |
|---|---|---|---|
| C1 | Round 0 校准筛选 receipt 完整：四候选 J_cal、确定性 tie-break、SHA256 链、单次写入 | `development/round0/calibration_selection.json` | `verify_calibration_selection`（`--resume` 只复验） |
| C2 | 校准只在 validation 拟合；dev A/B 不进入拟合 | receipt `fit_split=validation`、签名绑定 | receipt 字段 + 代码审查 |
| C3 | Round 1/2 候选空间精确等于 §4.2/§4.3（λ_B∈{0.05,0.10,0.20}；wd×ls 六组合），额外值拒绝 | `round_candidate_space` 校验 | 单元测试 |
| C4 | Round 1/2 只在上一轮未通过时启动；通过后继续搜索被拒绝 | `check_train_round_allowed` 门控 | 单元测试 |
| C5 | 每个 round 的 Gate R+_dev 报告含 A、B 两套完整 checks（core 6 项 + stability 6 项 + Q+），无静默跳过 | `development/roundN/gate_r_plus_dev_report.json` | `evaluate_gate_r_plus` |
| C6 | 任一未通过即不得冻结；Round 2 失败强制 `STAGE3A_PLUS_DEVELOPMENT_NOT_READY` 并禁止 power/collection | 状态机终态 | 单元测试（run-all） |

## D. 最终冻结（FINAL_MODEL_FROZEN）

| # | 检查项 | 证据 | 验证方式 |
|---|---|---|---|
| D1 | freeze receipt 覆盖：config、calibration（含选中候选与签名）、十 checkpoint 路径+签名、baseline、Gate 定义常量、代码 SHA256、两个 dev report SHA256 | `freeze/final_freeze_receipt.json` | `build_final_freeze_receipt` 自检 |
| D2 | freeze receipt 单次写入、不得覆盖 | O_EXCL 写入 | 单元测试 |
| D3 | checkpoint 签名在冻结时逐一复验，漂移即失败 | `build_final_freeze_receipt` 内逐 seed 校验 | 单元测试 |

## E. 进入资格阶段的前置条件

- 状态机处于 `FINAL_MODEL_FROZEN`。
- A1–D3 全部通过。
- 任何一项不满足：禁止运行 `prepare_stage3a_plus_power.py` 与采集器（硬停止规则 2/3）。
