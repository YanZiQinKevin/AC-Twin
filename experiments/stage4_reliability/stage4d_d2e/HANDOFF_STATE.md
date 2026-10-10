# Stage 4D-D2E HANDOFF STATE（恢复/交接锚点）

> 计划编号：`AC-TWIN-RFER-STAGE4D-D2E-EGO-CONSISTENT-EXTERNAL-2026-09-24-V0.1`
> 计划文件：`plans/2026-09-24_AC-Twin_Stage4D-D2E自车合同修复与MetaDrive独立外部确认实施计划_v0.1.md`
> 批准：2026-09-24 用户"批准执行"（§7 全部六项决定），状态 `APPROVED_FOR_AUTONOMOUS_IMPLEMENTATION`。
> **终态（2026-09-24）：`D2E_SCREEN_UNDERPOWERED`。** §4.2 screen 功效预检在 n=2500、75% 收缩下联合功效 0.6759 < 0.80，系预注册科学硬停止，授权随终态结束；未创建任何 screen/confirm manifest/lock/采集。
> 上位计划：`AC-TWIN-RFER-2026-09-07-V1`；前置终态 D2 v0.2 `STAGE4D_EXTERNAL_DEV_NOT_READY`、D2R `D2R_SCREEN_UNDERPOWERED` 永久有效不改写。
> 执行只创建 `experiments/stage4_reliability/stage4d_d2e/` 下产物及新代码/测试；`stage4d_external/` 与 `stage4d_d2r/` 只读。

## 科学目标与唯一解释假设

修复跨平台十动作任务所需的**自车反事实预测支持**：旧管线 `compute_effect_eval_mask(exclude_ego=True)` 把自车从 effect 训练/评估掩码中排除，十动作目标 `J=0.8·C_int+0.2·C_eff` 的效率项依赖自车各动作最终进度，但预测自车进度跨度仅 ~0.16m（真值 ~7.19m），导致基础选择器 a0 无法正确排序。新 D2E 新增**可移植自车 effect 分支**（每动作输出 20 帧 (Δs,Δd)，`Ŷ_CF,ego=Y_F,ego+Δ̂_ego`，只替换五对象预测张量 slot 0；slot 1–4 仍用旧 E2E），用自车包含掩码 `M_ego` 训练。**验证该解释，不能预先宣布 MetaDrive 已复现。**

## 冻结决定（本计划）

- 新 salt：`AC-TWIN-D2E-V0.1|`（见 `config.json`）。环境入口唯一：`code\run_stage4d_env.ps1 sim`；禁止裸 python/pip/conda run。
- 两候选（有界）：`EGO_TRAJECTORY`（SmoothL1 自车轨迹监督）、`EGO_TRAJECTORY_PLUS_RANK`（+固定十动作进度差辅助损失，权重 0.25，不搜索）。共享自车头 `Linear(input_dim,128)→LN→ReLU→Linear(128,128)→LN→ReLU→Linear(128,40)`→reshape(20,2)。seed=预测 pair `[1101,1202,1303,1404,1505]`；AdamW(lr=1e-3,wd=1e-4)、batch=64 组、≤100 epoch、patience=12、按最低 validation loss 冻结 checkpoint。
- 选择规则：两候选均过 §3.3 资格时取 validation `a0` mean regret 较低者；差≤1e-12 取 `EGO_TRAJECTORY`；不得借 screen 再选；不得无限增添网络/特征/seed/损失。
- Direct 风险头保留 R2/D2 的 `315→43→43→2`（15,740 参数）、同风险目标、5 seed `[9101..9505]`、主风险分数，用**修复后训练预测+十动作 regret** 在旧 train 重训；三标准基线（MARGIN_ONLY/SEED_VOTE_ENTROPY/SEED_MARGIN_STD）共享完全相同的修复后 `J_hat`。
- **不变**：十动作集合、事实/反事实生成、t_star、5Hz/20-step、`J` 定义、0.8/0.2 权重、`C_eff` 组内归一化、regret 目标、coverage=80%、主风险分数、统计方向。近零进度跨度组数量须完整报告但不得事后改尺度/排除/重加权。
- 关键掩码纪律：反事实真值派生 mask 仅用于训练损失/标签，**不得**进入风险特征构造或任何 train/validation/screen/confirm 推理路径；推理路径统一用仅由合法输入计算的 `inference_mask`。

## 阶段链与硬停止（§3.3/§4）

`preflight → fit-ego → fit-risk → validate → power-screen → lock-screen → collect-screen → infer-screen → adjudicate-screen → power-confirm → lock-confirm → collect-confirm → infer-confirm → adjudicate-confirm`

- **§3.3 旧 validation 开发资格门**（全真才继续；失败→`D2E_INTERFACE_REPAIR_NOT_QUALIFIED`，不采新 screen）：自车 final-progress MAE 与十动作进度排序均优于旧 D2 原预测（配对点估计严格正）；`a0` mean regret ≤ ALWAYS_IDLE；对 RANDOM_ACTION 改善点估计严格正；非自车 effect ADE 不恶化 >5%；Direct 对 validation 冻结唯一最强基线 `I_AURC_REGRET>=0.05`、`I_REGRET@80>=0.03`，5 seed ≥4/5 两项正。标记 `ADAPTIVE_DEVELOPMENT_ONLY`。
- **§4.2 screen 功效**：旧 validation 逐组配对量估完整 screen AND 链条件性联合功效；n∈{1000,1500,2000,2500}，100%/75%/50% 收缩全报，75% 收缩≥0.80 选最小 n；2500 不足→`D2E_SCREEN_UNDERPOWERED` 不采。
- **screen 一次裁判**：Gate TASK 沿用 D2 §8.2 c1–c8（含 c5）、Gate RISK 沿用 §8.3 r1–r6；失败→`D2E_EXTERNAL_DEV_NOT_READY`，confirm 不采。
- **confirm 功效**（仅 screen 两门全过）：n∈{1000,...,3000}，75% 收缩≥0.80 选最小 n；3000 不足→`D2E_CONFIRM_UNDERPOWERED`。
- **confirm 一次裁判**（全新未尝试 ID、write-once lock、95% CI）：TASK 失败→`INCONCLUSIVE_D2E_EXTERNAL_TASK_INVALID`；任务有效而 RISK 失败→`D2E_EXTERNAL_RISK_NOT_CONFIRMED`；全过→`D2E_META_DRIVE_EXTERNAL_CONFIRMED`。
- 候选池：新 screen/confirm 全部用与所有已尝试 ID 零重叠的新场景；原 D2 confirm 池仅在全局 ID 审计证明**从未尝试且从未落盘**后才可作候选；不足→采集前停 `D2E_FRESH_POOL_INSUFFICIENT`。

## 当前状态

- **preflight 已通过（2026-09-24）：`D2E_PREFLIGHT_PASS`。** torch 2.9.1+cu130、GPU RTX 3070 Ti、转换 scenario DB 可读（70,544 文件）、MetaDrive/ScenarioNet editable 在 `D:/MetaDrive` 内、无 TensorFlow。`source_inventory.json`（24 个来源内容 SHA256 登记）与 `preflight_receipt.json` 均 create-once 已写。配置 `config.json` 已冻结（继承 D2 冻结常量+D2E 专项）。
- **授权登记已完成（2026-09-24）**：计划头部/§7 → `APPROVED_FOR_AUTONOMOUS_IMPLEMENTATION`；`AGENTS.md` 清单第 (12) 项 + 新增 D2E ACTIVE 专项条目；`00_READ_FIRST` §6、文档索引 D2E 行 → `ACTIVE`。
- **fit-ego 已完成（2026-09-24）：`D2E_EGO_FIT_COMPLETE`。** 2 候选×5 seed=10 个自车头已训练（GPU，80s，均早停收敛，5 seed 紧密聚集）。`input_dim=1002`（机械计算：history 400+factual_suffix 400+事实/反事实计划 one-hot 各 100+归一化 t_star/duration 各 1），参数量 150,568，rank_scale=5.769。checkpoint+scaler+receipt 已 create-once 落盘于 `development/ego_models/`。
- **关键机制已直接核验**：`trajectory_components`（stage4c_objective.py:111）的 progress 无条件读 slot 0，mask 只管 interaction；旧失败纯属 slot 0 值未训练（非 mask 从 `J` 剥离）。修复=只换 slot 0 值，`compute_predicted_costs`/`inference_mask`/risk 特征全不变。
- **修复探针（旧 validation，适应性开发）显示修复生效**：`a0` mean regret 从旧 D2 `0.11578`/运动学底 `0.11856` 降至 EGO_TRAJECTORY `0.05889`/PLUS_RANK `0.05711`（均 ≤ idle `0.10604` 且优于 random `0.11246`）；自车 final-progress MAE 从 3.70m 降至 2.07/2.09m；十动作进度排序 Spearman 从 −0.006 升至 +0.529/+0.524。真值 oracle `0.00267` 精确复现计划记载 ~0.0027，旧 D2 `0.11578` 复现 ~0.1158，管线正确性交叉确认。
- **fit-risk 已完成（2026-09-24）：`D2E_RISK_FIT_COMPLETE`。** 每候选在修复后 train 预测上重训 Direct（参数量 15,740 冻结不变），train_high_regret_frac≈0.46（修复未塌缩 regret 信号，风险方法仍有可预测对象）。inventory create-once 于 `riskfit_{candidate}/`。
- **§3.3 validate 门已通过（2026-09-24）：`D2E_INTERFACE_REPAIR_QUALIFIED`（首个真正科学 Go/No-Go = GO）。** 两候选全部 6 条件真；机械选定 **`EGO_TRAJECTORY_PLUS_RANK`**（validation a0 mean regret 0.05711 < EGO_TRAJECTORY 0.05889，差 0.00178 > 1e-12）。`final_selection.json` 已冻结。关键数字（旧 validation，适应性开发，非独立确认）：
  - 任务：a0 regret 旧 D2 0.11578 / 运动学底 0.11856 / **EGO_TRAJECTORY 0.05889** / **PLUS_RANK 0.05711** / oracle 0.00267；idle=0.10604、random=0.11246（两候选 a0≤idle 且优于 random）。
  - 自车质量：final-progress MAE 旧 3.70→2.07/2.09m；十动作进度排序 Spearman 旧 −0.006→+0.529/+0.524；非自车 ADE 五变体完全一致 0.06168306（slot-0 仅替换正确性证据）；近零进度跨度组 frac=0。
  - 风险（条件 5，对最强基线 SEED_MARGIN_STD）：EGO_TRAJECTORY I_AURC=0.2471/I_REGRET80=0.1623、PLUS_RANK 0.1937/0.1169，均 5/5 seed 正、PR-AUC−prev>0（0.220/0.193），远门槛。**decision-risk 方法在自车修复+平台内重训后仍有效。**
- **§4.2 power-screen 已完成并达预注册终态（2026-09-24）：`D2E_SCREEN_UNDERPOWERED`（第二个廉价 Go/No-Go = 停）。** 用修复后管线在旧 validation（n_ref=1500，适应性开发）逐组配对量估计完整 D2 screen AND 链联合功效：10,000 次 whole-group paired bootstrap（新冻结熵 `2026122405`，协方差直接取自 bootstrap  draws，不由 CI 宽度反推）+ 10,000 次 MVN Monte-Carlo，结构分数（oracle shares/反应分数/分层分数）与指标槽联合模拟。**结果：n∈{1000,1500,2000,2500} 下 75% 收缩联合功效 0.5352/0.6140/0.6501/0.6759，均 < 0.80；`selected_n=null`，终态 `D2E_SCREEN_UNDERPOWERED`。** 绑定条件为 **`c6_reactive_frac`**：validation 反应分数点估计 `0.272`，75% 收缩→`0.204` 仅勉强高于门槛 `0.20`，增大 n 几乎无助于跨越（点估计受限而非 CI 宽度），n=2500 边际功效仅 0.6795。**其余所有条件在 75% 收缩下功效充足**（c5 base-vs-random 与 base≤idle、r1 I_AURC=0.194、r2 I_REGRET80=0.117、r3 PR、r4 5/5 seed、c2/c3/c4/c7 均 ≈1.0）。**科学解读：自车修复本身合格（§3.3 通过，基础选择器与 decision-risk 在平台内重训后均有效），外部 screen 功效受限于反应率这一 MetaDrive 场景结构性属性——与 D2R 的反应率受限（同一 validation 反应分数 0.272、同法边际功效 0.6716）一致，是重复出现的平台结构边界，非自车修复或风险方法缺陷。** 终态禁止任何 in-place 阈值/覆盖率/seed/screen 规模/门槛/平台变更；未创建任何 screen/confirm manifest/lock/采集。
- **下一动作**：无（已达预注册科学终态 `D2E_SCREEN_UNDERPOWERED`）。授权随终态结束，不自动启动任何后续阶段。

## 已完成产物

- `config.json`、`source_inventory.json`、`preflight_receipt.json`（create-once）；本 HANDOFF。
- 代码：`code/ac_twin/stage4d_d2e_ego.py`（自车分支：特征组装/`M_ego` 掩码/共享头/SmoothL1+rank 辅损失/训练/推理/slot-0 替换）、`code/ac_twin/stage4d_d2e_pipeline.py`（修复管线+§3.3 指标）、`code/ac_twin/stage4d_d2e_power.py`（§4.2 功效：扩展功率向量=冻结 D2 32 槽逐位复用+结构分数 32–46、draw AND 链、MVN 功效、torch 侧量组装）、`code/drive_stage4d_d2e.py`（阶段入口 fit-ego/fit-risk/validate/power-screen）。
- 测试：`tests/test_stage4d_d2e_ego.py`（13 项）+ `tests/test_stage4d_d2e_pipeline.py`（4 项）+ `tests/test_stage4d_d2e_power.py`（6 项）= **23 项全过**：input_dim=1002、参数量、掩码自车包含且非全零、CF 真值不入推理特征、slot-0 仅替换、5 seed 索引配对、修复流入 progress 项机制回归守卫、Direct+基线共享修复后 J_hat、变体分派；功率向量前 32 槽与冻结 D2 向量逐位一致、结构分数正确（oracle 和=1、分层互补）、draw AND 链（点+80% CI 下界+base≤idle+5 seed+分层/反应）、simulate_power 选最小 n/不足→underpowered、协方差对称 PSD、estimate 端到端。
- 训练产物：`development/ego_models/`（10 checkpoint+`ego_scaler.json`+`ego_fit_receipt.json`+`ego_fit_report.json`，create-once）。
- 风险产物：`riskfit_EGO_TRAJECTORY/`、`riskfit_EGO_TRAJECTORY_PLUS_RANK/`（各 5 Direct seed checkpoint + fit_receipt，create-once）；`development/fit_risk_report.json`。
- 裁判产物：`development/validation_report.json`（§3.3 一次裁判，create-once）、`development/final_selection.json`（冻结选定 EGO_TRAJECTORY_PLUS_RANK）。
- 功效产物：`development/power_validation_quantities.npz`+`power_validation_quantities_receipt.json`（旧 validation 逐组配对量，create-once，`ADAPTIVE_DEVELOPMENT_ONLY`）、`power/screen_power_report.json`（§4.2 一次功效预检，create-once，终态 `D2E_SCREEN_UNDERPOWERED`）。
- 探针脚本（非冻结产物）：`code/stage4d_d2e_probe_ego.py`、`code/stage4d_d2e_probe_repair.py`。

## 最近验证（2026-09-24）

- preflight 全过（24 来源 hash 登记）。
- 自车探针：旧 validation 真值十动作自车进度跨度中位 `7.1877 m`（精确吻合计划 ~7.19m），确认自车目标提取正确。
- 修复探针：见上"当前状态"修复生效数字。
- §3.3 validate 门：`D2E_INTERFACE_REPAIR_QUALIFIED`（两候选 6 条件全真，机械选定 EGO_TRAJECTORY_PLUS_RANK）。
- §4.2 power-screen：终态 `D2E_SCREEN_UNDERPOWERED`（n=2500、75% 收缩联合功效 0.6759 < 0.80，绑定 c6_reactive_frac；其余条件功效充足）。D2E 测试套件 23 项全过。

## 阻塞项

- 无。

## 消耗登记

- 一次性资源（本计划）：均未创建（无 screen/confirm manifest/lock，无采集）。§4.2 power-screen 是开发性预检（读旧 validation，写 create-once 开发产物），**未消耗任何一次性 screen/confirm/fresh-pool 资源**；终态 `D2E_SCREEN_UNDERPOWERED` 在采集前停止，fresh-pool 从未触碰。旧 D2 screen 已消耗（不属于本计划，保持封存）。原 confirm 池的全局 ID 审计**不再需要进行**（功效不足，采集不会发生）。
