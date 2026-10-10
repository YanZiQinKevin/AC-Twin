# Stage 3A Plus 资格裁判就绪检查单（QUALIFICATION_READINESS_CHECKLIST）

> 计划：`AC-TWIN-RFER-STAGE3A-PLUS-2026-09-09-V0.1` Task 10。
> 用途：在 `gate-r-plus-qualification`（一次性资格裁判）触发前，逐项核对全部前置条件。任何一项不成立即硬停，不得消耗裁判。

## A. 数据与采集

- [ ] 资格集位于 `data/object_centric_stage/stage3a_plus_qualification/`，且仅此处（`assert_stage3a_plus_source_role` 通过）。
- [ ] 采集 receipt（`qualification/collection_receipt.json`）存在；`qa.strata_equal_counts == true`，十个 action×duration 层各 `n/10`；`cf_duration == {1,3}`；`cf_action ⊆ {0..4}`。
- [ ] 采集 receipt 哈希链完整：`data_sha256` == 数据文件 SHA256；`power_report_sha256` == `power/power_report.json` SHA256；`seed_schedule_sha256` == schedule 文档值。
- [ ] ledger 审计：全部 rejected 记录仅引用七类工程原因（prefix_terminated / object_prefix_mismatch / factual_future_short / counterfactual_future_short / non_finite / initial_fingerprint_mismatch / short_history）；无任何基于实现 R、模型输出或 Gate 指标的筛选。
- [ ] episode_seed 唯一、取自冻结 schedule、与全部历史数据（含 sealed Stage 2、merge_2000、replacement）零重叠。

## B. 冻结与锁

- [ ] `freeze/final_freeze_receipt.json` 完整性通过（`receipt_sha256 == canonical_fingerprint(payload)`），且两份 dev 报告均为 `both_passed == true`。
- [ ] 十个冻结 checkpoint 逐一 `load_predictor_state(expected_signature=...)` 通过（替换 checkpoint 会被拒绝，rc 3）。
- [ ] `qualification/qualification_lock.json` 存在且单次写入；绑定数据 SHA256 + 冻结 receipt 哈希 + power 哈希链。
- [ ] 冻结 receipt 的 Gate 常量与代码内冻结值逐项相等（core metrics、ECE guard、8/10、最差非劣 δ、非退化下限、十个 seed）；不一致即 rc 3 且不消耗裁判。

## C. 冻结推理

- [ ] `qualification/predictions/seed_<seed>.npz` ×10 存在；`inference_receipt.json` 绑定每个文件 SHA256 与对应 checkpoint 签名；episode 数 == 冻结 n。
- [ ] 推理输出全部有限；scaler 仅 transform（开发集拟合，资格集不参与任何拟合）。

## D. 一次性裁判

- [ ] 状态机位于 `FROZEN_INFERENCE_COMPLETE`；`qualification_used.lock` 与 `gate_r_plus_qualification_report.json` 均不存在。
- [ ] 正式 bootstrap 冻结：10,000 次、seed `20260929`（分层 seed×episode 用 `20260929+7000`）；Gate 无任何可调参数暴露。
- [ ] 裁判结果只可能为三终态之一：`RESPONSE_DEV_QUALIFIED_STAGE3A_PLUS`（rc 0）/ `INCONCLUSIVE_RESPONSE_HEAD_INSUFFICIENT_STAGE3A_PLUS`（rc 3，科学失败硬停）/ `INVALID_RUN_ENGINEERING_OR_PROTOCOL_FAILURE`（rc 2，工程/协议失败）。
- [ ] 裁判后：同一资格集上禁止补样本、换 seed、改校准器、改 Gate 或二次裁判（硬停规则 5）。
- [ ] `RESPONSE_DEV_QUALIFIED_STAGE3A_PLUS` 形成前，禁止启动 Stage 3B 主矩阵或 Gate C 任何正式工作（硬停规则 6）。
