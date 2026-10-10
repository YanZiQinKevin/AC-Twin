# Stage 4A 状态

> 当前状态：`ENGINEERING_ABORT_AFTER_COLLECTION / RECOVERY_REQUIRED`
> 计划：`AC-TWIN-RFER-STAGE4A-RISK-DEV-2026-09-11-V0.1`
> 更新：2026-09-11

- 说明：采集已完成且有效（5,000 episode、十层各 500、prefix/identity 误差 0.0），但首次 `infer-dev` 在 pair0 处因一处纯工程 bug 中止——`cmd_infer_dev` 中 e2e router 未移动到配置的 CUDA 设备（`run_stage4a.py` 单行 `.to(device)` 缺失）。**推理未开始，inference 目录为空，无任何推理/风险/筛选/screen 产物。** 该修复不改变任何科学内容，但 `run_stage4a.py` 的 hash 已写入 v1 protocol lock，故 v1 锁保持原样、不在原地改动。
- 恢复路径：在 `experiments/stage4_reliability/stage4a_v2/` 建立 write-once 恢复锁（`recovery_lock.json`），只读复用本目录的 NPZ / collection receipt / seed schedule，从 `infer-dev` 恢复，不重跑采集。
- v1 产物（保持不变）：`protocol/protocol_lock.json`、`protocol/candidate_seed_schedule.*`、`protocol/historical_seed_inventory.json`、`collection/collection_receipt.json`、`data/object_centric_stage/stage4_merge_risk_dev/object_twinworld_stage4_merge_risk_dev.npz`。
- 阻断项：v1 目录本身不再继续；后续在 v2 恢复链上推进。
