# Stage 4A HANDOFF_STATE

- 当前状态：`ENGINEERING_ABORT_DURING_RISK_FIT / RECOVERY_TO_V3_REQUIRED`
- 冻结决定：见 `recovery_lock.json`（v2 recovery_from_v1）与 `../stage4a_v3/recovery_lock.json`（v3 recovery_from_v2）
- 最近验证：v3 recovery lock 复核 v2 推理产物完整一致（5000 episodes、receipt 一致、PRIMARY_PREDICTOR=END_TO_END_MIXTURE 冻结可复核、screen/selection 未触碰）
- 下一步：在 `stage4a_v3/` 从 `fit-risk` 起完整重训全部风险模型，随后 `select-candidate` → `adjudicate-dev-screen`
- 阻断项：MLP 序列化缺陷已修复（模块级 `_TorchEstimator`）；v3 复用 v2 推理与 PRIMARY_PREDICTOR receipt，不重新采集、不重跑 infer-dev
