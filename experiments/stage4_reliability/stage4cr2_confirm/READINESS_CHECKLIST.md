# Stage 4C-R2 独立确认 READINESS_CHECKLIST

> 计划：`AC-TWIN-RFER-STAGE4CR2-CONFIRM-2026-09-17-V0.1`
> 创建：2026-09-20（Task 4 骨架）；2026-09-20 operational 阶段填写实际验证输出。
> 规则：readiness 未全过时，runner 不得进入 `lock-protocol`（runner 的 readiness
> 阶段强制本文件存在，并执行全量回归）。

## 1. 阻断测试（本阶段五个专项文件，必须全绿、零警告；不在任何豁免内）

- [x] `python -m pytest -q tests/test_stage4cr2_confirm_protocol.py` — 结果：79 passed（-W error 零警告）
- [x] `python -m pytest -q tests/test_stage4cr2_confirm_inference.py` — 结果：19 passed（-W error 零警告）
- [x] `python -m pytest -q tests/test_stage4cr2_confirm_collection.py` — 结果：26 passed（-W error 零警告）
- [x] `python -m pytest -q tests/test_stage4cr2_confirm_statistics.py` — 结果：43 passed（-W error 零警告）
- [x] `python -m pytest -q tests/test_stage4cr2_confirm_runner.py` — 结果：43 passed（-W error 零警告）

合计 210 passed（2026-09-20 终态复核更正：锁前 ANSI 修复 +2、Task 7-FIX +15 后实际总数；
早期填写值 193 为修复前计数，已按 `pytest --collect-only` 实测更正；各任务评审均独立复跑确认）。

## 2. 工程 smoke（ENGINEERING_SMOKE_ONLY，12 组；绝不触碰正式路径与锁）

- [x] smoke 采集完成（`smoke/smoke_collection_receipt.json`，accepted=12，
  partial=false）— 结果：12/12，attempts=21，26s
- [x] smoke 推理/打分完成（`smoke/smoke_inference_receipt.json`，
  status=ENGINEERING_SMOKE_ONLY）— 结果：120 branches，5 pair × 5 risk class × 5 Direct seed
  全覆盖；冻结 scaler `2f21c11a…`、final_selection `f135fdab…`、model_inventory `982ec5ea…`
  与 §2.1 钉值一致
- [x] smoke 重复复验（`smoke/smoke_recheck_receipt.json`）— 结果：12 组全数组 bitwise
  （np.array_equal，context_rows=120，无容差）
- [x] smoke 工程真值统计（`smoke/smoke_engineering_report.json`）— 结果：
  status=ENGINEERING_SMOKE_ONLY，gate_diagnostics 仅工程诊断，无科学裁判
- [x] smoke 隔离核对：正式四件（inputs/sealed archive/ledger/collection_receipt）与
  protocol_lock 均未创建 — 结果：确认未创建（spy 测试 + 实跑后目录核对）

## 3. 清单核对（code / source / seed inventory）

- [x] `PROTOCOL_CODE_FILES` 五项全部落码并登记（protocol / inference / collector /
  statistics / runner）——正式锁在此之后创建（计划 §5.3）
- [x] preflight 已验证 §2.1 十个钉住来源 hash 全部匹配；R2 已执行 v0.2 域（entropy
  2026092880，60,000 seeds）纳入审计 — `preflight` 输出 PREFLIGHT_PASSED；
  confirm schedule raw sha256 `4c0958fc5e80…` 与计划 §4.2 钉值一致
  （`protocol/source_inventory.json` 随锁生成并复核）
- [x] code inventory：本阶段五文件 + 复用冻结链 hash 与环境版本固定
  （随 `protocol/code_inventory.json` 锁时生成并复核）
- [x] confirm（entropy 2026093012 / 12,000）与 smoke（2026093100 / 64）候选域对全局
  历史+预留清单零重叠（420,804 seeds 全局清单；含 R2 已执行 60,000 域、Stage 2 18,000
  冻结域、Stage 2B legacy_test_compatibility 域）— preflight seed_audit 通过，
  `protocol/seed_audit.json` 随锁生成

## 4. 全量回归（首次长科研运行前一次）

- [x] `python -m pytest -q tests -rf` 全量回归 — 结果：1490 passed / 6 failed / 688s
  （2026-09-20，原始输出存 SDD workspace `full_regression_2026-09-20.log`）；
  readiness 阶段由 runner 再跑一次当时全量回归
- 已登记非阻断例外（计划 §10：按**具体 node + 已核验证据**登记；保留原始失败输出；
  不改冻结 tests；不屏蔽整个文件；不写"全量全绿"）：
  - `readiness_exemptions.json` 已登记 6 个 node（Stage 2 ×5 live-history 锁测试 +
    Stage 2B ×1 陈旧候选域测试），evidence 含失败原因（冻结 schedule 与增长后的
    live 历史池相交）与原始输出位置。
  - 新失败、真实使用/预留域重叠、本阶段阻断测试失败**不在豁免内**。

## 5. 锁前确认

- [x] 以上全部通过后，runner 进入 `lock-protocol`（create-once；二次调用拒绝）
