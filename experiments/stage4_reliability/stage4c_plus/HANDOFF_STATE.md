# Stage 4C Plus HANDOFF_STATE

> 计划：`AC-TWIN-RFER-STAGE4C-PLUS-DECISION-AWARE-RISK-2026-09-16-V0.1`
> 本文件是恢复锚点：记录当前状态、冻结决定、完成产物、最近验证、下一动作与 blocker。

## 当前状态

- 阶段：**`STAGE4CP_FEASIBILITY_NOT_SUPPORTED` — 科学硬停止（Phase 0 探索性 OOF 两个候选均未通过第 5.3 节 Go/No-Go）**。按计划"两个候选均失败则终态 / 此时停止，不采集任何新数据，不追加 Phase 0 候选或换 fold seed"规则，**不进入新 development 采集、有界训练、screen、功效或 confirm 链**。
- 授权：用户于 2026-09-16 以"执行该计划"指令明确批准；按计划自动连续推进至本停止点。
- 冻结来源链：执行前已重算 SHA256，全部 MATCH；冻结模型锁（15 个 checkpoint + scaler + Stage 3A Plus binding）全部解析成功。
- Phase 0 标记 `EXPLORATORY_OOF_NOT_CONFIRMATORY`；只读取旧 Stage 4C **500-group development-selection** 三类产物（数据/预测/动作风险），**未读取任何 Stage 4C 300-group screen 产物**（永久只读）。Phase 0 未做任何新采集或新冻结 predictor 推理。
- Phase 0 一次性产物已封：`phase0/fold_assignment.npy`、`phase0/oof_predictions.npz`、`phase0/phase0_report.json`（create-once）。两候选均失败，故**未写** `phase0_pass_receipt.json`。selection SHA256=`2447ae8a19192bce…`、predictions=`716bbd854a82af0a…`、risk_scores=`d9bd9c4aaaf25c54…`。

## 冻结决定（已锁定，不可更改）

- 统计单元 = 十动作决策组（非分支/车辆/时间点）；bootstrap 整组重采样。
- 动作集合、后果函数（`0.80*C_int + 0.20*C_eff`，交互尺度 5.0m/2.0m，regret 钳制 1e-10）与 regret 定义**与 Stage 4C bitwise 同构**，未为挽救 Stage 4C 失败而修改任务定义。
- 参考动作 `a0 = argmin_a J_hat`（冻结 ensemble），其余 9 动作为相对比较 token（34 维 = 20 连续 + 14 one-hot）+ 9 维全局特征。
- Phase 0 固定：`hidden_dim=64`、`lambda_cls=0.25`、init seeds `[6101,6202,6303]`（OOF 三 seed 取平均）、5 folds（entropy 2026092301，同 group 十动作永不跨 fold）、bootstrap 5,000 次（entropy 2026092302）、双侧 80% CI、scaler/类别权重仅在各折训练部分拟合。
- `high_regret_threshold = 0.01`（FIXED，与 Gate TASK  substantive regret 一致，不重拟合）。
- 参数量审计：DCR(64)=15,746，Direct(width=43)=15,740，相对差 0.038%（<±5% 容差，通过）。
- Phase 0 seed 契约：dev entropy 2026092401/size 30,000；confirm entropy 2026092601/size 25,000（仅生成未冻结，因未进入采集）。

## 完成产物

- `experiments/stage4_reliability/stage4c_plus/stage4cp_config.json`（通过 `validate_config`）。
- `code/ac_twin/stage4cp_protocol.py` — 协议/seed 契约/历史+预留 seed 清点/禁读守卫/dev+confirm 锁（未烧）。
- `code/ac_twin/stage4cp_features.py` — 34 维相对 token + 9 维全局特征 + train-only scaler。
- `code/ac_twin/stage4cp_models.py` — DCR/Direct 架构、参数量审计、训练与三分数导出。
- `code/ac_twin/stage4cp_statistics.py` — 风险曲线基元、fail-closed improvement、整组配对 bootstrap。
- `code/run_stage4cp.py` — 编排 runner；当前实现 preflight/phase0/verify/run-all（其余子命令留待后续任务）。
- 测试：`tests/test_stage4cp_protocol.py`、`test_stage4cp_features.py`、`test_stage4cp_models.py`、`test_stage4cp_phase0.py` 共 **32 项全部通过**（含 token 置换不变、无真值参数、fold 隔离、train-only scaler、参数量审计、Phase 0 数量向量与门逻辑）。

## 最近验证

- 32 项阻断测试全部通过（`-W error` 下）。
- `phase0` → **`STAGE4CP_FEASIBILITY_NOT_SUPPORTED`**（500/500 组特征有效，0 组被丢弃）：
  - **DCR**：`I_AURC_REGRET` point=`+0.1212`（80% CI 下界 `−0.0268`，需 >0 → 失败）；`I_REGRET@80` point=`+0.0424`（下界 `−0.0696` → 失败）；PR−prevalence 候选 `+0.0714` ≥ 基线 `+0.0606`（通过）；fold 正向 3/5（需 ≥4 → 失败）；1%/5% trim 均正（通过）。
  - **Direct**：`I_AURC_REGRET` point=`+0.0950`（下界 `−0.0935` → 失败）；`I_REGRET@80` point=`+0.0941`（下界 `−0.0289` → 失败）；PR−prevalence 候选 `+0.1237` ≥ 基线（通过）；fold 正向 5/5（通过）；trim 均正（通过）。
  - 共同模式：两候选点估计均为正，但 500 组上 80% CI 下界均跨 0，改进在统计上不可分辨；`retained_architectures=[]`。
  - 结论（Phase 0 单次探索性 OOF）：**在旧 500-group selection 上，inference-legal 的相对排序信息对降低 decision regret 的改进不具备统计支持，不足以证明继续采集新数据的合理性**。

## 下一动作（科学硬停止，非自动继续）

- **停止**。不得采集任何新 group，不得追加 Phase 0 候选，不得换 fold seed/init seed/hidden_dim 重试，不得扩大同分布样本原地续命。
- 待用户/研究负责人裁决：是否在 plans/ 中登记 Phase 0 失败结论并规划后续（如特征/监督目标重设计并另立新计划，或结束 Stage 4C Plus 主线）。本阶段 development/screen/confirm 链保持未启动。

## Blocker

- **科学硬停止**：Phase 0 两个候选均未通过第 5.3 节六条件 Go/No-Go（DCR 与 Direct 的 `I_AURC_REGRET`、`I_REGRET@80` 80% CI 下界均 ≤0）。这是计划定义的停止点，需人工裁决下一步方向。
