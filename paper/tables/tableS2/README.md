# Table S2｜Data roles, model transfer, and information-access contracts

> 状态：`DRAFT_UPDATED / DATA_ROLE_COUNTS_VERIFIED / PENDING_FINAL_LAYOUT`  
> 作用：汇集完整范围/有效性表与复现合同；正文Table 2–3保留简明信息角色与数据规模。本补表供审稿人核对训练、迁移、推理权限与评价真值。

## A. Experimental scopes and model relations

| Study scope | Scientific question | Independent unit / n | Predictor and audit-head relation | Claim boundary |
| --- | --- | --- | --- | --- |
| Factual-evidence analysis | Does factual-evidence value vary with realized response? | episode / 1,200 | Frozen evidence models; response is used only for post-hoc stratification | Supports response-conditioned evidence-value heterogeneity, not a decision policy claim |
| Merge ID decision audit | Can AC-Twin identify high-regret selections in distribution? | ten-action group / 2,000 | Merge-trained prediction stack and independently trained Direct audit head, both frozen before confirmation | Independent in-distribution confirmation |
| `JOINT_OOD` | Does the frozen Merge audit survive the prospectively specified generator mismatch? | ten-action group / 2,000 | Same frozen Merge prediction and Direct weights; no OOD retraining | Supports only the measured generator-mismatch profile |
| MetaDrive/Waymo | Can the same audit framework be reconstructed on another platform? | ten-action group / 2,000 | Predictor, action-risk components and Direct audit head retrained and frozen inside MetaDrive | Platform external confirmation with platform-internal retraining; not zero-shot weight transfer |

## B. Information-access contract

| Information | Audit training | Confirmation inference | Standard baselines | Final evaluation |
| --- | --- | --- | --- | --- |
| Frozen predicted trajectories and action costs | yes | yes | where defined | yes |
| Frozen action-risk outputs and cross-seed uncertainty | yes | yes | method-specific legal subset | yes |
| Pre-intervention geometry, action identity/duration and intervention time | yes | yes | method-specific legal subset | yes |
| Paired counterfactual outcomes and `J_true` | label construction only | **no** | **no** | yes |
| Oracle action and true decision regret | label construction only | **no** | **no** | yes |
| Realized response and realized prediction error | historical task labels/analysis only | **no** | **no** | analysis/evaluation only |

训练来源说明：五个动作风险通道中，`COMBINED_RISK_HEAD`、`GEOMETRY_ONLY` 和 `RESPONSE_ONLY` 是开发 train 上以真实 effect 误差监督拟合的冻结模型；`UNCERTAINTY_ONLY` 与 `DISAGREEMENT_ONLY` 是对冻结预测的直接聚合，不是另外两个已拟合网络。响应预测组件只在允许的开发角色中使用真实响应监督/校准，待评组仅消费其预测输出。MetaDrive 使用平台内 train/validation 训练、选择及校准后冻结；确认组真值不进入任何上游训练或推理输出。训练输出不统一采用 OOF，不能将其视为独立证据。

## C. Manuscript boundary

- 正文方法段必须说明：paired counterfactual truth 与 true regret 从不进入确认推理。
- Merge ID 与 `JOINT_OOD` 共享冻结 Merge 权重；MetaDrive/Waymo 采用平台内重训并冻结的同构流程。
- 本表不罗列逐文件 SHA256、create-once 回执或完整状态机；这些保留在可复现包和冻结实验档案中。
- 本表不承载性能数字；定量确认结果见 Table 6，已发表适配方法的共享开发基准比较见 Table 7。

## D. Data-role resource map（2026-10-02 核对）

本表按最后用于论文证据的测量模型或 Direct 审计头列出角色，不是全部实验历史的累计数据量；“未另设”不是缺失样本，而是明确使用 validation 监控。完整 episode/group 才是计数单位。

| 证据范围 | 独立计数单位 | Train 总量 | 独立 monitor | Validation | 开发评测 / screen / 资格 pilot | Confirm |
| --- | --- | ---: | --- | --- | --- | ---: |
| Merge 事实证据价值 | episode | 1,400 | 未另设，使用 validation | 300（含 checkpoint 选择） | 300 个既有开发评测 episode | 1,200 |
| Merge 十动作 Direct audit | 完整十动作组 | 5,000 | train 内 500；参数拟合用其余 4,500 | 1,500 | 8,000 个一次性 development-screen 组 | 2,000 |
| `JOINT_OOD` | 完整十动作组 | 不新增，复用 Merge | 不新增，复用 Merge | 不新增，复用 Merge | 三种候选配置各 300 组资格 pilot | 2,000 |
| MetaDrive 十动作 Direct audit | 完整十动作组 | 5,000 | 未另设，使用 validation | 1,500（含 early stopping、配置选择） | 1,000 个最终路线 development-screen 组 | 2,000 |
| 已发表方法适配比较 | 完整十动作组 | 5,000，复用 Merge 开发 train | 适配方法不设独立 monitor；Direct 沿用其冻结划分 | 1,500，复用 Merge 开发 validation | 不新增 screen | 0（独立确认集未创建） |

- Merge train 的 5,000 组是 4,500 main + 500 monitor 的并集，不计为 5,500。与 validation 1,500、screen 8,000、confirm 2,000 分角色隔离。
- `JOINT_OOD` 不新增训练数据或权重。三份各 300 组 pilot 只用于分布/任务资格及机械配置选择，不按审计方法效应选择；其最终确认 2,000 组是另外的新资源。
- MetaDrive 的 5,000 train 与 1,500 validation 为既有平台内开发数据，D2E 在它们上修复及重训，D2F 只读复用最终冻结组件；不存在额外 500 monitor。最终 screen 1,000 与 confirm 2,000 使用新 ID。旧 D2 的 4,000 组失败 screen 保留且未复用；D2R/D2E 功效终止时未采新 screen/confirm。
- Stage 5 复用 R2 原有 5,000 train 与 1,500 validation。适配方法不设独立 early-stopping monitor，按拟合样本训练损失恢复 checkpoint；Conformal 从这 5,000 组 train 内划出 3,988 fit 和 1,012 calibration，两个子集互斥，不是额外数据。Direct 不重训，沿用 R2 的 4,500/500 划分与冻结权重。
- Stage 2 的原 300 个 test episode 已作为开发评测，不再承担独立确认；1,200 个新 episode 才是本次事实证据价值 confirm。响应资格、上游风险开发及早期探索另有其历史角色，不并入这里的确认样本量。
- 四份正文 confirm 均不参与开发；三个 decision-risk confirm 相互不重叠。same data reuse 指开发源的合法复用，不表示独立新增样本，也不授权已消耗 confirm 的再次使用。

### 逐角色计数来源

| 项目与数据字段 | 冻结入口 |
| --- | --- |
| Stage 1 Merge 1,400 / 300 / 300，分别为 train / validation / 既有开发 test | `experiments/stage1_evidence_value/merge/report.json`：`n_train,n_validation,n_test` |
| Stage 2 confirm 1,200 episode | `experiments/stage2_confirmatory/merge/report.json`；`data/object_centric_stage/stage2_merge_1200/collection_receipt.json` |
| Merge Direct 5,000 总 train；4,500 main / 500 monitor | `experiments/stage4_reliability/stage4c_r2/development/models/fit_receipt.json`：`n_train_groups,main_indices,monitor_indices` |
| Merge validation 1,500；screen 8,000 | 同目录 `validation_collection_receipt.json` / `screen_collection_receipt.json`：`accepted_groups` |
| Merge confirm 2,000 | `experiments/stage4_reliability/stage4cr2_confirm/collection/collection_receipt.json` 与正式确认报告 |
| OOD 三个 pilot 各 300 | `experiments/stage4_reliability/stage4d_ood/pilot/{behavior,geometry,joint}/collection_receipt.json`：`accepted_groups` |
| OOD confirm 2,000 | `experiments/stage4_reliability/stage4d_ood/collection/collection_receipt.json` 与正式确认报告 |
| MetaDrive train 5,000 / validation 1,500 | `experiments/stage4_reliability/stage4d_external/roles/{train,validation}/collection_receipt.json`：`accepted`；D2E `development/fit_risk_report.json` |
| MetaDrive 无独立 monitor，validation 早停 | `stage4d_external_training.py::fit_five_direct_models`；D2E `riskfit_EGO_TRAJECTORY_PLUS_RANK/development/models/fit_receipt.json` |
| MetaDrive 最终 screen 1,000 / confirm 2,000 | `experiments/stage4_reliability/stage4d_d2f/development/development_screen_report.json` / `confirm/confirm_report.json`：`n_groups` |
| Stage 5 共用源及 3,988 / 1,012 fit/calibration | 冻结 Stage 5 v0.3 计划的数据来源；`experiments/stage5_sota_comparison/development/models/fit_receipts.json`：conformal 条目 `notes.fit_groups,calibration_groups` |
| Stage 5 独立 confirm 未创建 | `experiments/stage5_sota_comparison/power/power_report.json` 与终态交接 |

本轮仅更新 Table S2 的 Markdown 工作稿；`tableS2_historical_source_data.xlsx` 仍是历史底表，不宣称其已同步以上字段或完成投稿排版。

## Sources

- Active plan §24.2 and §25.
- `experiments/stage4_reliability/stage4cr2_confirm/reports/stage4cr2_confirmatory_report.json`.
- `experiments/stage4_reliability/stage4d_ood/reports/stage4d_ood_confirmatory_report.json`.
- `experiments/stage4_reliability/stage4d_d2f/confirm/confirm_report.json` and `source_binding.json`.
- Historical data-role bottom sheet preserved as `paper/tables/tableS2/tableS2_historical_source_data.xlsx`.
