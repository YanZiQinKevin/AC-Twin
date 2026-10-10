# Stage 4A HANDOFF_STATE

- 当前状态：`ENGINEERING_ABORT_AFTER_COLLECTION / RECOVERY_REQUIRED`
- 冻结决定：v1 `protocol/protocol_lock.json` 原样保留（不原地改动）；采集数据有效。
- 已完成产物：protocol lock + 12,000 seed schedule + 历史 seed 清单；5,000 episode NPZ（十层各 500，角色划分 3,000/1,000/1,000，prefix/identity 误差 0.0）+ collection receipt。
- 最近验证：采集 NPZ 复核通过（5,000 唯一 seed、十层各 500、split 列与机械角色划分一致）。
- 中止原因：`infer-dev` 在 pair0 命中纯工程设备放置 bug（e2e router 未 `.to(device)`），推理未开始、无部分产物。修复为一行 `.to(torch.device(device))`，但因 `run_stage4a.py` hash 已入 v1 锁，转入 v2 恢复链。
- 下一步：在 `experiments/stage4_reliability/stage4a_v2/` 写 write-once `recovery_lock.json`（记录 v1 父锁 hash、原/修复 runner hash、变更原因与位置、seed schedule/collection receipt/NPZ hash、5,000 与十层复核、inference 目录为空、其余代码与 checkpoint/scaler/binding 无漂移），跑 Stage 4 专项测试与 CUDA 小规模 infer smoke，然后从 `infer-dev` 恢复（不重跑采集）。
- 阻断项：v1 锁保持只读；任何除 `run_stage4a.py` 外的未解释 hash 漂移或数据/receipt/schedule 不一致都将立即停止。
