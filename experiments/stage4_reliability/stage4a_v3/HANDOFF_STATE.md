# Stage 4A HANDOFF_STATE

- 当前状态：`ENGINEERING_ABORT_DURING_CANDIDATE_RECEIPT / RECOVERY_TO_V4_REQUIRED`
- 冻结决定：见 `protocol/recovery_lock.json`（v3 recovery_from_v2）；v4 见 `../stage4a_v4/protocol/recovery_lock.json`
- 最近验证：v3 risk fit 完整有效；`select-candidate` 因恢复版本推理 receipt 路径硬编码中断（未生成回执、未触碰 screen）
- 下一步：在 `stage4a_v4/` 只读复用 v2 推理 + v2 PRIMARY_PREDICTOR + v3 risk dataset/fit receipt/5 模型，从 `select-candidate` 起确定性重放 → `adjudicate-dev-screen`
- 阻断项：候选回执路径缺陷已修复（`_inference_paths`）；v4 锁定前先完成锁前恢复路径演练，再一次锁定
