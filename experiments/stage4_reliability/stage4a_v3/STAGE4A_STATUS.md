# Stage 4A 状态

> 当前状态：`ENGINEERING_ABORT_DURING_CANDIDATE_RECEIPT / RECOVERY_TO_V4_REQUIRED`
> 计划：`AC-TWIN-RFER-STAGE4A-RISK-DEV-2026-09-11-V0.1`
> 更新：2026-09-11T14:40:00

- 说明：v3 fit-risk 已在锁定代码下完整完成（`risk_dataset.npz`、5 个拟合模型、`fit_receipt.json` 均有效）。推进到 `select-candidate` 时暴露一处恢复兼容性工程缺陷：`cmd_select_candidate` 在写选择回执前硬编码读取 `run_dir/inference/predictions.npz`，而恢复版本的冻结推理在 v2，本地 v3 无此文件 → `FileNotFoundError`，`select-candidate` 无法在恢复版本上运行。

- 本次状态如实登记：
  - risk-validation 已被程序读取（用于在内存中机械计算候选分数）；
  - finalist、standard baseline、combined implementation 已按冻结规则在**内存中**机械计算；
  - 失败发生在写入 receipt 之前的 predictions hash 路径解析；
  - **未生成任何选择回执**，`candidate_selection.json` 不存在；未向用户展示结果，未据此改变候选、阈值或排序规则；
  - development-screen **完全未读取、未消耗**（`development_screen_used.lock` 不存在）。

- 因此 v4 是**相同冻结算法的确定性重放**，不是重新选择或调参。

- 中断性质：纯工程路径缺陷，仅触及 `run_stage4a.py` 的推理 receipt 路径解析，不涉及任何科学参数、特征、候选集、数据、seed 或 PRIMARY_PREDICTOR。

- 修复：`cmd_select_candidate` 的 predictions 哈希改走 `_inference_paths(run_dir, lock)`（恢复版本复用 v2 推理路径）。

- 阻断项：候选回执路径缺陷 → 已修复并迁移至 `stage4a_v4/`；v4 只读复用 v2 推理、v2 PRIMARY_PREDICTOR、v3 risk dataset / fit receipt / 5 模型，从 `select-candidate` 起重放。
