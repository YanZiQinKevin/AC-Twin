# AC-Twin Stage 3 实施与 QA 记录（工程里程碑）

> 记录性质：**工程实施/QA 记录**，非科研日志，非确认性结果。  
> 对齐计划：`AC-TWIN-RFER-STAGE3-2026-09-08-V0.2`（`APPROVED_FOR_IMPLEMENTATION`）  
> 更新日期：2026-09-09  
> 边界：本文件只承载 Stage 3 的软件实现、测试、工程 smoke 与预算准备里程碑。Stage 3A、Gate R_dev、Stage 3B、Gate C 的**科研证据**在形成后统一写入科研日志，不在此处断言。

## 状态机位置

`DRAFT_REVIEW → APPROVED_FOR_IMPLEMENTATION → SOFTWARE_COMPLETE → SMOKE_PASS → COMPUTE_BUDGET_APPROVED → [Stage 3A 长任务已执行] → OLD_GATE_R_DEV_INVALID_IMPLEMENTATION_BUG → REPAIR_SOFTWARE_TESTED → RETRAIN_SEARCH_COMPLETE → RETRAIN_FIVE_SEED_COMPLETE → RETRAIN_SELECTION_FROZEN → REPLACEMENT_DEV_LOCKED → REPLACEMENT_GATE_R_DEV_ADJUDICATED → INCONCLUSIVE_RESPONSE_HEAD_INSUFFICIENT_ID`

当前：**`STAGE3_DEV_NOT_READY_FOR_CONFIRMATION`**（2026-09-10，Stage 3B Gate T_dev 单次裁判未通过）。Stage 3B 软件与输入绑定已完成，screen（gate_hidden=32）与 fit-tier1（四条件×五 pair 共 20 run，容量审计 0 差）成功；但 Gate T_dev 在历史开发集 A（merge split2）/B（replacement dev）上**六项非 cross-fit 条件均未通过**：技术/审计不变量全过、oracle_gap>0（响应路由头间存在），但 Dual 相对 Direct/Uniform 的开发相对改善仅 ~0.3–2.8%（<5% 最小实质线）且 leave-one-pair-out 方向不稳。按合同**不进入 crossfit、不准备 Gate C Protocol Lock、不得扩展搜索空间**，回用户评审。详见 `stage3b/STAGE3B_STATUS.md` 与 `stage3b/gate_t_dev/gate_t_dev_report.json`。

## 批准范围（用户限定）

- 已批准：软件接线、测试、预算准备（COMPUTE_BUDGET.md）、smoke 级限时小 benchmark、PowerShell 命令准备、readiness checklist；**2026-09-09 用户确认预算（COMPUTE_BUDGET_APPROVED）**。
- 仍须用户**手动启动**（助手不自动运行）：正式单 seed 搜索、多 seed 训练、cross-fit、Gate R_dev/T_dev 裁判、Stage 3B 主矩阵、正式采集、全量推理、bootstrap、Protocol Lock、解封。

## 软件实现（11 个文件）

| 文件 | 职责 |
|---|---|
| `code/ac_twin/progress.py` | +30 秒时间心跳（fake-clock 验证） |
| `code/ac_twin/stage3_data.py` | 数据角色、mask、`R_pos`/四箱/censored onset、Stage 2 三重封存拒绝、duration `{1,3}` 断言 |
| `code/ac_twin/stage3_response.py` | 响应预测器、连续+分类简单基线、温度校准/PR-AUC/Brier/ECE |
| `code/ac_twin/stage3_routing.py` | z_base+z_dyn 重构、Direct/Dual 唯一乘法差异、十条件、容量审计、seed 配对、cross-fit |
| `code/ac_twin/stage3_statistics.py` | episode-level effect-ADE、H4、Gate R/T_dev、Gate C 分级状态机、联合功效 |
| `code/ac_twin/stage3_training.py` | 统一训练、断点续跑（早停状态一致）、run fingerprint、checkpoint 签名+完整性、cross-fit、§5.3 COMPUTE_BUDGET |
| `code/ac_twin/stage3_scaler.py` | train-only FeatureScaler（fit-once、落盘、签名校验 load，val/test 仅复用） |
| `code/ac_twin/stage3a_gate.py` | Gate R_dev numpy 评估管线（H4 全指标、episode bootstrap、七项 AND 裁判） |
| `code/ac_twin/stage3_protocol.py` | 配置校验、联合 lock、输出签名复算、二次解封拒绝、Gate Q |
| `code/run_stage3.py` | 状态机入口 + 12 ep/1 epoch smoke |
| `code/run_stage3a.py` | Stage 3A 开发驱动（validate-config/benchmark/budget/readiness/search/fit-five/gate-r-dev） |
| `code/collect_stage3_gatec.py` | 复用 Stage 2 引擎的两个正式 holdout 采集器 |

## 测试口径（统一）

- Stage 3 **新增**测试：**159 项**；
- 若计入原有的 4 项 progress 测试，Stage 3 相关合计：**163 项**；
- **全量回归：554 passed**（原有 391 项无破坏）。
- 以后不把相关合计表述为全部“新增”。

## 修复重训 v0.1 实施进度（2026-09-09，`AC-TWIN-RFER-STAGE3A-REPAIR-2026-09-09-V0.1`）

- Task 1 旧事故证据冻结：✅ `stage3a/INVALIDATION_RECORD_20260909.json`（三产物 SHA256 可复算），状态机改为 `OLD_GATE_R_DEV_INVALID_IMPLEMENTATION_BUG`；`tests/test_stage3a_invalidation.py` 5 项，先红后绿。
- Task 2 推理 episode 对齐：✅ `make_eval_batches`（无 shuffle）+ `_predict_split` 返回 `episode_indices` + `_assert_episode_order` 硬断言（eval 与 validation 双侧）；temperature 前对齐断言。新增测试 4 项（乱序/非连续、batch-size 不变、错位拒绝、gate 遇静默错位立即 RuntimeError）。
- Task 3 seed 初始化：✅ `seed_all` 移到模型构造前，补 CUDA deterministic（cudnn.deterministic=True、benchmark=False、`use_deterministic_algorithms` 严格模式）；同 config/seed 双目录签名一致、顺序反转不变（含环境 RNG 污染注入）。新增测试 4 项。
- Task 4 跨 lambda 选择与全局聚合：✅ `response_loss_sums`（float64 分子/分母）+ `selection_score = L_reg + 0.25·L_CE`（固定系数）；`global_validation_metrics` 对 batch size/尾批/排列 1e-7 不变；search/select-final 只按 `validation_selection_score` 排名（含反向构造测试）。新增/更新测试 6 项。
- Task 5 产物隔离与回执链：✅ 阶段子目录 `search/`、`fit_five/`、`selection/`、`gate_r_dev/`；无 `--resume` 时拒绝覆盖既有产物（exit 3），`--resume` 幂等复用并完整性复验；fit-five 对 finalist seed-101 与 search 签名交叉审计（不符即 `DETERMINISM_FAILURE`）；回执签名覆盖五 seed 选择分数与上游 fit-five 文件 SHA256；合同号升级为 `AC-TWIN-STAGE3A-REPAIR-CONTRACT-2026-09-09-V0.1`。新增测试 4 项。
- Task 7 双数据源 Gate R_dev：✅ `lock-replacement-dev` + `gate-r-dev-retrain`；拟合状态（scaler/类别权重/基线/温度）只来自 development source 的 train/validation，全部 Gate 指标只在 replacement 上计算；旧 split 2 作为 eval source 直接拒绝；单次锁在读 replacement 真值前原子获取，二次调用拒绝；输出保留原七项裁判、五 seed 指标、bootstrap 与完整签名。集成测试含端到端通过/拒绝、静默错位立即 RuntimeError、篡改回执/替换 finalist/锁错文件均拒绝。
- Task 6 replacement holdout 功效与采集合同：✅ `code/prepare_stage3a_replacement_power.py`（五指标 episode 重采样功效，候选 `[300..800]` 冻结、seed 20260909 冻结、最弱指标定样本量、800 不足即 `REPLACEMENT_DEV_POWER_INFEASIBLE` 硬停止、NaN 计失败；功效报告→配置→seed schedule 三层 SHA256/指纹链）；`code/collect_stage3a_replacement_dev.py`（复用 Stage 2 引擎，冻结 merge 生成合同逐项校验、purpose=`stage3a_replacement_development_test`、seed 全集扫描**包含** Stage 2 封存集、禁写 Stage2/merge_2000/Gate C/旧 stage3a 路径、receipt 增强含 data_sha256/QA/代码 SHA256、30 秒心跳线程）；`stage3_data.py` 新增 `scan_all_episode_seeds`/`assert_replacement_source_allowed` 与冻结常量；Gate 命令在烧毁单次锁**之前**拒绝 Gate C 路径（refusal 不消耗裁判）。新增测试 27 项，先红后绿。
- Task 8 软件验证与 readiness：✅ 全量回归 `pytest tests/ -W error` **608 passed / 0 failed / 0 warning**；12 ep/1 epoch 独立临时目录 smoke 全链通过（search→fit-five→select-final→lock→gate-r-dev-retrain，产物未进入正式 run 目录）；同 seed 双目录重现 QA（8 配置签名逐一致）与 eval 顺序 QA（顺序保持/逆序一致/batch 不变，含 episode 首尾 ID 与 SHA256）落盘于 `stage3a_retrain_v1/logs/`；`stage3a_retrain_v1/RETRAIN_READINESS_CHECKLIST.md` 全部勾选。
- 中途全量回归：**577 passed**（Task 1–5 后）→ Task 6/7 完成后 **608 passed**（`-W error`）。
- Task 9 重训：✅ 2026-09-09 执行（用户授权代为启动，RTX 3070 Ti，仅用旧 split 0/1）。search 8 配置（finalists `h64_r32_l0.1`、`h96_r32_l0.1`）→ fit-five 2×5 seed（五签名互异、seed101 与 search 交叉审计一致）→ select-final 唯一 final `h96_r32_l0.1`（五 seed 均值 0.14161 < 0.14421），最强基线 validation 冻结，`final_selection.json` 单次写入、二次运行 exit 3。
- Task 10 replacement Gate：✅ 2026-09-09 执行。功效（seed 20260909、2000 MC、最弱 `I_Brier`）：n=300→0.844 不足，**n=400→0.920 冻结**；采集 400/400（668 尝试、10 层均衡、seed 与 13,212 历史 seed 零重叠、合同逐项一致）；lock 绑定后 `gate-r-dev-retrain` 单次执行（2000 bootstrap）。
- **replacement Gate R_dev 裁判结果：`passed=false`**——五项改进指标 CI 下界全部 >0、ECE 校准通过（6/7），唯一失败项 `five_seed_consistent_nondegenerate`：seed 303 的 `I_Brier`=-0.0026（五 seed 为 [+0.0061, +0.0078, -0.0026, +0.0045, +0.0008]），触发全 seed×全指标方向一致性失败；非退化检查通过；全部工程/协议检查通过、无拒绝。裁判后签名终验全部 MATCH（dev/replacement 数据、lock、selection 回执、五 seed 签名）。
- **状态机终态：`REPLACEMENT_GATE_R_DEV_ADJUDICATED` → `INCONCLUSIVE_RESPONSE_HEAD_INSUFFICIENT_ID`**（科学条件失败、Gate Q 通过；非工程失败）。按完成判据：不进入 Stage 3B，主方法正式化停止；Stage 3B/Gate C 未启动；replacement holdout 已消耗（单次锁），不得再次裁判。科研记录见 `logs/2026-09-09_AC-Twin科研日志.md`。

> 口径演进：初版接线 122/126/517；预算+断点续跑+驱动后 136/140/531；7 项验收修复后 155/159/550；3 项修复后 156/160/551；本轮 development test 单次使用三重锁硬化（+3 集成测试）后 **159/163/554**。

## 工程 smoke

`python code/run_stage3.py smoke --episodes 12 --epochs 1` 实跑成功：数据加载 → 响应预测 1 epoch → Direct/Dual/Oracle 最小前向 → episode-level effect-ADE → 联合 lock 写入/校验 → 输出签名复算。状态仅 `ENGINEERING_SMOKE_ONLY`。smoke 中的 oracle gate 为随机值、批量仅 12 episode，任何 Direct/Dual 差值均无科研意义。

## 关键合同落实情况（均有测试）

- z_base+z_dyn 逐元素精确重构；Direct/Dual 共享 backbone/gate/decoder topology，w=1 时输出完全一致（唯一结构差异为显式乘法）；
- 五组 predictor-router 一一配对（禁 best-seed / 5×5）；Uniform/Direct/Dual 参数量相等（0% 差，无 dummy）；
- 2-fold cross-fit 无 episode 泄漏；Stage 2 holdout 按绝对路径/逻辑路径/已知 SHA256 三重拒绝；
- duration 集合断言 `{1,3}`，duration=2 标为 unseen；输出签名可从落盘 JSON 复算；已有正式 report 拒绝二次解封。

## 待办（预算确认后由用户手动启动）

1. Stage 3A 有界搜索（单 seed 筛选 → 入围 ≤2 配置 → 补五 seed）→ Gate R_dev；
2. Stage 3B 主矩阵 → Gate T_dev；cross-fit 敏感性；
3. 功效估计与 Protocol Lock 冻结；正式采集与一次解封。
