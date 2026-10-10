# Stage 4A 状态

> 当前状态：`ENGINEERING_ABORT_DURING_RISK_FIT / RECOVERY_TO_V3_REQUIRED`
> 计划：`AC-TWIN-RFER-STAGE4A-RISK-DEV-2026-09-11-V0.1`
> 更新：2026-09-11T14:05:00

- 说明：v2 冻结推理完成（5000 episodes × 5 pair）且已通过 v3 recovery lock 复核为不可变复用源；fit-risk 阶段在保存 MLP combined 模型时因 `_fit_mlp` 返回闭包内 `_TorchEstimator`（torch.save/pickle 无法序列化）而中断。
- 中断性质：纯工程序列化缺陷，不涉及任何科学参数、特征、候选集、数据、seed 或 PRIMARY_PREDICTOR。
- 修复：`_TorchEstimator` 移至 `code/ac_twin/stage4_risk.py` 模块级并显式持有训练完成的 PyTorch model。
- 已冻结且复用（read-only 引用自 v2）：`predictions.npz`、prediction receipt、`primary_predictor_receipt.json`（PRIMARY_PREDICTOR=END_TO_END_MIXTURE）。
- 阻断项：MLP 序列化缺陷 → 已修复并迁移至 `stage4a_v3/`，v3 从 fit-risk 起完整重训全部风险模型。
- 中断产物说明：v2 中已生成的 ridge/histgb 等文件属于未完成 fit-risk 的中断产物，保持原样、不用于裁判。
