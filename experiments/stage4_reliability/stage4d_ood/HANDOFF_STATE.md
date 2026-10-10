# Stage 4D D0/D1 — HANDOFF_STATE

> **计划编号：** `AC-TWIN-RFER-STAGE4D-D0D1-OOD-2026-09-20-V1.0`
> **当前状态：** `D0_PILOTS_COLLECTED`
> **更新日期：** 2026-09-21
> **当前执行任务：** Task 6（唯一解封、裁判、表图与交接）— 已完成；确认集已永久消耗，本计划授权随终态终止

## 当前进度

Task 1 完成。建立了 stage4d 命名空间的协议/profile 原语，全部 focused tests 在
`-W error` 下通过（68 passed）。未修改任何 Stage 1~4C-R2 冻结文件；未采集任何科学数据；
未加载 Direct checkpoint；未读取任何 confirm 原始 truth 或 R2 train NPZ 的 CF/regret 字段。

Task 2 完成。建立了 `code/collect_stage4d_merge_ood.py`：只读复用 R2 的
`collect_group`/`validate_complete_group`，注入本阶段自己的 `branch_rollout`
（reset 后先 `assign_episode_entity_ids`  stamping，再 `apply_profile_to_env`，
十一分支扰动后快照用 `snapshots_identical` 两两校验，不一致=整组拒绝，工程原因
`branch_perturbation_inconsistent`）；双文件真值隔离发布（`inference_inputs.npz`
恰好 8 个合法字段 + sealed `confirmation_truth.npz`）、receipt 绑定两文件 hash 与组序、
原子 checkpoint/resume、rejection ledger、30 秒心跳、合法输入 loader（truth 移走仍可
加载，score 侧 truth-open 立即失败）。真实 12 组工程 smoke 已运行（smoke 域
2026100099/256，profile=BEHAVIOR_OOD——计划未定 smoke profile，按 Task 2 brief 选
BEHAVIOR_OOD 并在此说明）：12 accepted / 17 attempts，状态 `ENGINEERING_SMOKE_ONLY`，
`prefix_max_abs_diff=0.0`，5 次拒绝全部 `prefix_terminated`（工程原因），snapshot
一致性零拒绝；smoke 只写 smoke/ 子目录，正式路径未触碰；未产生任何科学 Gate / risk
分数。focused tests 97 项 `-W error` 全绿。

Task 3 完成。建立了 `code/ac_twin/stage4d_ood_statistics.py`（§5.2 逐字：六个干预前
量的 SMD vs R2-train 参考；固定 5 折 logistic 域分类器 + group-bootstrap 双侧 80% AUC
CI 下限、5000 resamples、entropy 2026104101/…02/…03；4 项 task-support 条件；4 项
integrity 条件；D0 禁区 fail-closed 读取守卫）与 `code/run_stage4d_ood.py`（§12 全
相位骨架；power 起后续相位显式 `not implemented yet`，绝不静默跳过；幂等 resume）。
真实顺序采集三个 pilot（behavior→geometry→joint，各 300 accepted）：behavior
300/558（11:29）、geometry 300/1043（18:18）、joint 300/1022（18:04），拒绝全部
`prefix_terminated`，候选预算（≤6000）远未耗尽，三份 receipt 的 schedule SHA 与
action_stream_root 全部核验。三份 profile report 与 create-once
`pilot/ood_profile_selection.json` 已落盘：**BEHAVIOR_OOD 不合格**（shift 失败：
0/6 量 |SMD|≥0.50，AUC 点 0.5263、CI 下限 0.5063<0.70；integrity 与 task support
全过）；**GEOMETRY_OOD 合格**（3/6 量偏移，AUC CI 下限 0.9805）；**JOINT_OOD 合格**
（3/6 量偏移，AUC CI 下限 0.9801）。按优先级 JOINT>BEHAVIOR>GEOMETRY 唯一选择
**JOINT_OOD**，status=`D0_PROFILE_SELECTED`。全程未加载 Direct checkpoint、未产生
risk score/AURC/coverage/improvement 量；域分类器仅证偏移、不入方法评分。

Task 4 完成。扩展 `stage4d_ood_statistics.py` 以 §6.1 逐字实现 `simulate_d1_power`
（唯一输入 = hash 钉死的 R2 development screen_group_quantities.npz + receipt；协方差
直接来自本轮逐组 paired bootstrap draws[frozen entropy 2026092702]，绝不从 CI 宽度反推；
MC=10000、entropy=2026104201 单生成器跨 4 n×4 保留；唯一阻断判决 = 50% 保留 AND 链
联合通过率 ≥0.80 的最小 n）。真实功效相位运行结果：n=1000 的 50% 保留联合通过率
0.7311<0.80，n=2000 为 0.9688≥0.80 → **冻结 selected n = 2000**（每个 n 的绑定条件均为
i_regret80）；status=`D1_POWER_FROZEN`，create-once `power/power_report.json` 落盘，未触发
UNDERPOWERED 硬停止。新建 `code/ac_twin/stage4d_confirm_protocol.py`：registry 先行的扩展
代码 inventory（collector/statistics/runner/本模块 + D1 链复用冻结依赖，共 41 文件；Task 1
base registry 18 文件单独绑定以满足 `verify_protocol_lock` 严格相等）、§2.1 末段展开为显式
path+hash（五个 Direct checkpoint 经 confirm 侧 `verify_r2_sources` 双核验、Full/History/E2E
predictor、五个 response predictor[绑定 receipt 显式 path+hash]、binding、scaler，共 45 文件）、
锁时 seed 审计**实时重导出**（重跑全域扫描 inventory=432,868 + 五域零重叠 + 相互不重叠）、
§7.1 条件 8 分层（t_star split=13 + R2-train-only 初始距离阈值=29.128996473819946，经
`derive_r2_train_distance_threshold` 只读 object_history 求值——与 R2 冻结 interaction 阈值
逐位一致，为一致性交叉核验）。write-once `protocol/{source_inventory(superset 核验替换),
code_inventory,seed_inventory,protocol_lock}.json` 落盘；`prot.verify_protocol_lock`（base）与
`verify_confirm_protocol_lock`（extended：扩展代码 drift、selected_n/profile 一致性、锁时
seed 审计 live 重跑）双绿；status=`D1_LOCKED`。runner 新增 power/lock-d1/locked-smoke-readiness
三相（§12 序），Task 5/6 相位仍显式 not implemented。未采集 D1、未裁决、未读 sealed truth。

Task 5 完成。按 controller ruling（Task 4 review Important #1）先落 §13 恢复链再采集：
`stage4d_confirm_protocol.py` 扩展 CONFIRM registry（+本阶段薄适配器
`ac_twin/stage4d_ood_inference.py` 为 stage4d_own、+冻结 R2-confirm 推理模块
`ac_twin/stage4cr2_confirm_inference.py` 为 reused_frozen 并经 stage4cr2 锁链
[钉死 recovery hash] 交叉绑定），新增完整 §13 恢复链原语（`recovery_lock_paths` 连续
编号 fail-closed / `resolve_locked_extended_code_hashes` / `create_recovery_lock` /
`verify_confirm_code_chain`），`verify_confirm_protocol_lock` 改经链解析；写一次锁本体
与 base registry 18 文件从未触碰。write-once `protocol/recovery_lock_r1.json` 落盘
（binds_previous=protocol_lock SHA256；changed=runner+confirm_protocol 新旧 hash；
added=适配器+cinf；reason=计划内 Task 5 phase 就位；未变科学绑定逐项 attestation；
声明扩展集==当前 registry），创建后 base+extended+live audit 全绿。新建薄适配器
`stage4d_ood_inference.py`：cinf 只读复用（`_bundle_from_arrays`/`_validate_predictions`/
`load_frozen_model_inventory`/`_shape_aligned_router_e2e`/`_context_aligned_combined_score`/
`_constant_random_group_scores`）+ r2stats `_context_aligned_direct_scores`，分数数学零
重实现（risk_s=sigmoid(logit_high)*0.01*expm1(softplus(raw_mu))、ensemble=五种子算术均值、
scaler 仅 transform）；引擎配置=stage4cr2 confirm config 只读并经锁链绑定（钉死 recovery
hash→protocol lock→config signature→cinf/confirm_protocol/confirm_statistics 链解析 hash
==磁盘）；D1 action 级 CONSTANT_RANDOM 根 [2026110994,4]+冻结 salt 2026093203（种子
2664752168），组级=纯 group-seed 函数；score 侧只读 `inference_inputs.npz` 8 合法字段，
`confirmation_used.lock` 存在即拒绝。runner 新增 collect-d1/inference-scoring/score-freeze
三相（采集前全锁链 live 审计+used-lock 缺席检查；打分/冻结走 cheap code-chain 核验），CLI
子命令 collect-d1/infer-d1/freeze-scores。真实 D1 采集：**2000/2000 accepted，7415
attempts，2h12m20s**（~27% 接受率，候选预算 30000 远未耗尽），receipt 双文件同序绑定、
schedule SHA `aafbba57…536a` 与锁定候选流一致。真实冻结推理（cuda/float32/batch=32）：
2000 组 × 10 动作 = 20000 分支，5 predictor pair + 5 Direct 种子 + 4 baseline 全部落盘，
n_valid=2000（零无效组、零 padding）。确定性复检：固定前 32 组两次独立重跑逐元素一致，
且等于正式产物前导切片（router/COMBINED/Direct 三处行数敏感前向零填充到正式上下文
context_rows=20000 / direct_context_rows=2000，逐位比较、无容差）。create-once
`inference/{predictions.npz,risk_scores.npz,inference_receipt.json,determinism_receipt.json,
score_freeze_receipt.json}` 落盘，freeze receipt 绑定全部 score 侧 hash + 锁/recovery 链
hash + 封存 truth 的 receipt 记录 hash。全程未读 truth（封存 hash 核验原样）、未计算任何
Gate/AURC/improvement 量（Task 6 专属）、未重训/重校准；status=`D1_SCORES_FROZEN`。

## 已完成产物

- `code/ac_twin/stage4d_protocol.py` — 配置校验、§2.1 来源链核验、seed 域/审计、
  代码 inventory、lock 校验原语、§7 状态机、§7.1 条件 8 阈值派生规则。
- `code/ac_twin/stage4d_ood_env.py` — 三个 OOD profile 的 §4 参数表、非 ego 类型绑定、
  几何初始化、分支一致性快照（duck-typed，不在模块顶层 import highway_env）。
- `tests/test_stage4d_protocol.py`、`tests/test_stage4d_ood.py` — 68 个测试。
- `experiments/stage4_reliability/stage4d_ood/config.json` — 冻结配置。
- `experiments/stage4_reliability/stage4d_ood/protocol/source_inventory.json` — §2.1
  十个来源全部实测 SHA256 = 匹配（含 R2 train NPZ 绑定
  `df240cac…53b8`），`inventory_sha256 = b53209d0…caa6`。
- `code/collect_stage4d_merge_ood.py` — D0/D1 十一分支 OOD collector（Task 2）。
- `tests/test_stage4d_collection.py` — 29 个 collector/真值隔离/心跳测试（Task 2）。
- smoke 产物（仅 smoke/ 子目录）：`data/object_centric_stage/stage4d_merge_ood/smoke/
  {inference_inputs.npz, confirmation_truth.npz, rejection_ledger.jsonl}` 与
  `experiments/stage4_reliability/stage4d_ood/smoke/smoke_collection_receipt.json`。
- `code/ac_twin/stage4d_ood_statistics.py`、`code/run_stage4d_ood.py`（Task 3）。
- `tests/test_stage4d_d0_statistics.py`（~40 测试）、`tests/test_stage4d_runner.py`
  （8 测试）（Task 3）。
- `experiments/stage4_reliability/stage4d_ood/readiness_exemptions.json` — 7 个已登记
  历史豁免节点（5 Stage 2 + 1 Stage 2B + 1 Stage 4C-R2 confirm）。
- `code/ac_twin/stage4d_confirm_protocol.py` — Task 4 新建：扩展代码 registry
  （CONFIRM_REGISTRY，Task 5 后 43 文件）、§2.1 来源展开、锁时真实 seed 审计、§7.1 条件 8 分层
  派生、write-once `create_protocol_lock` 与扩展校验 `verify_confirm_protocol_lock`；
  Task 5 新增 §13 恢复链原语。
- `tests/test_stage4d_d1_power.py`（16 测试）、`tests/test_stage4d_confirm_protocol.py`
  （17 测试）（Task 4）；`tests/test_stage4d_runner.py` 的 not-implemented 守卫最小化
  编辑（仅移除已实现的 power/lock-d1/locked-smoke-readiness 三相）。
- `experiments/stage4_reliability/stage4d_ood/power/power_report.json` — create-once，
  selected_n=2000，status=D1_POWER_FROZEN（Task 4）。
- `experiments/stage4_reliability/stage4d_ood/protocol/{source_inventory(45 文件 superset),
  code_inventory(41 文件扩展),seed_inventory(n=432,868),protocol_lock}.json` — 正式锁
  （Task 4）。
- pilot 产物：`data/object_centric_stage/stage4d_merge_ood/pilot_{behavior,geometry,
  joint}/{inference_inputs.npz, confirmation_truth.npz, rejection_ledger.jsonl}` 与
  `experiments/stage4_reliability/stage4d_ood/pilot/{behavior,geometry,joint}/
  {collection_receipt.json, profile_report.json}`，以及 create-once
  `experiments/stage4_reliability/stage4d_ood/pilot/ood_profile_selection.json`
  （selected_profile=JOINT_OOD，status=D0_PROFILE_SELECTED）。
- `code/ac_twin/stage4d_ood_inference.py` — Task 5 新建：D1 冻结推理/打分薄适配器
  （cinf/r2stats 只读复用；D1 action 根 [2026110994,4]；truth 隔离；确定性复检零填充；
  create-once 收据）。
- `tests/test_stage4d_inference.py` — 26 个测试（引擎链绑定、registry 覆盖、truth
  路径/key 拒绝与 spy、8 合法键、产物合同、风险公式逐字、D1 恒定随机根、create-once、
  used-lock 拒绝、确定性复检、Direct 上下文填充、冻结绑定、恢复链正负例、runner 接线）。
- `experiments/stage4_reliability/stage4d_ood/protocol/recovery_lock_r1.json` — §13
  恢复链首环（write-once；绑定 runner/confirm_protocol 修复与适配器/cinf 注册）。
- `experiments/stage4_reliability/stage4d_ood/protocol/recovery_lock_r2.json` — §13
  第二环（write-once；Task 6 裁判 phase 就位，changed=statistics+runner）。
- `experiments/stage4_reliability/stage4d_ood/protocol/recovery_lock_r3.json` — §13
  第三环（write-once；freeze 复验 recovery_locks 与数据绑定分离）。
- `experiments/stage4_reliability/stage4d_ood/protocol/recovery_lock_r4.json` — §13
  第四环（write-once；`_assert_confirmation_unused` 仅首次冻结调用）。
- `code/ac_twin/stage4d_ood_statistics.py` 的 Task 6 扩展 — 49 量向量、whole-group
  paired bootstrap、Gate TASK-OOD/OOD-RISK 裁判、§8.1 渲染器（byte-deterministic）。
- `tests/test_stage4d_adjudication.py` — 18 个 Gate/裁判/一次性纪律测试。
- `experiments/stage4_reliability/stage4d_ood/confirmation_used.lock` — 一次性烧毁。
- §8.1 裁判产物（create-once，`experiments/stage4_reliability/stage4d_ood/reports/`）：
  `stage4d_ood_confirmatory_report.json`、`STAGE4D_OOD_DECISION.md`、
  `Table_OOD_1_gate.csv`、`Table_OOD_2_baselines.csv`、`Table_OOD_3_seed_strata.csv`、
  `Fig_OOD_1_risk_coverage.png`、`Fig_OOD_2_domain_shift.png`、
  `d1_group_quantities.npz`、`d1_group_quantities_receipt.json`。
- D1 正式产物（create-once）：`collection/{inference_inputs.npz, confirmation_truth.npz
  (sealed), rejection_ledger.jsonl, collection_receipt.json}`（2000/2000，7415 attempts）与
  `inference/{predictions.npz(313MB), risk_scores.npz, inference_receipt.json,
  determinism_receipt.json(32 组 bitwise), score_freeze_receipt.json}`。
- 运行日志：`collect_d1.log`、`infer_d1.log`、`freeze_scores.log`、
  `full_regression_task5.log`（run 目录根，操作性产物、不入锁）。

## 冻结决定（本任务已固化）

- 三个 profile 的 §4.1 行为表（aggressive/cautious/ID 参考）与 §4.2 几何基值/抖动
  逐字落入 `stage4d_ood_env.py` 与 `config.json`，drift 即 fail-closed。
- 五个候选 seed 域的 (entropy, size, raw-bytes SHA256) 逐字固定于
  `_APPROVED_SCHEDULES`；分类器 bootstrap(2026104101/…02/…03)、功效(2026104201)、
  D1 bootstrap(2026110995) entropy 按 ruling 2 登记为 reserved。
- §7.1 条件 8 阈值派生规则固定为
  `median_of_pre_intervention_min_ego_distance_over_r2_train_groups`
  （Task 4 从 R2 train NPZ 干预前初始距离求值；规则不再更改）。
- Task 4 冻结：**selected n = 2000**（50% 保留 AND 链联合通过率 0.9688≥0.80 的最小 n；
  n=1000 为 0.7311<0.80）；D1 confirm 唯一 profile=JOINT_OOD、quota=2000、D1 bootstrap
  entropy=2026110995、分层阈值（t_star split=13、初始距离阈值 29.128996473819946）均
  写入 write-once `protocol/protocol_lock.json`。
- Task 5 固化：§13 恢复链首环 `protocol/recovery_lock_r1.json`（write-once；
  SHA256 `b64cfb4f…f5f6d`）绑定 Task 5 registry 扩展（runner 三相 + 适配器 + cinf 注册），
  写一次锁本体从未改动。D1 冻结打分产物全部 create-once：predictions.npz / risk_scores.npz /
  inference_receipt.json / determinism_receipt.json / score_freeze_receipt.json；freeze receipt
  绑定全部 hash（输入 `d94b29fc…90acf`、封存 truth `f8d76619…082845` 经 receipt 记录值、
  锁与 recovery 链）。primary baseline=SEED_VOTE_ENTROPY、5 Direct seeds=[8101,8202,8303,
  8404,8505]、ensemble=算术均值、D1 bootstrap entropy=2026110995 均已写入 freeze receipt。
- Task 6 固化：§13 恢复链 r2/r3/r4 三枚 write-once 锁（r2 SHA `1e1e5104…8bd8e9`，
  绑定 statistics+runner 裁判扩展；r3/r4 各绑定 inference 的两处纯工程修复——recovery
  链增长合法化与 burn 后分数复验解锁），链 r1→r4 全绿。`confirmation_used.lock` 已
  烧毁（原子、一次性）；裁判产物全部 create-once，report 最后写入；resume 不重跑
  burn/assemble/adjudicate。终态=`STAGE4D_OOD_ROBUSTNESS_CONFIRMED` 已固定，不得
  二次裁判、不得重算分数。

## 最新验证

- focused tests：`tests/test_stage4d_protocol.py` + `tests/test_stage4d_ood.py`
  → 68 passed，`-W error` 无警告。
- 全域 seed 零重叠审计（真实工作区，inventory=432,868 seeds）：五个候选域全部
  `zero_overlap=True`，且相互不重叠、与 R2/confirm 全域不重叠。扫描跳过 sealed
  `confirmation_truth`（其 accepted seeds 由已落盘候选 schedule 覆盖）。
- Task 2 focused tests：上述两文件 + `tests/test_stage4d_collection.py`
  → 97 passed，`-W error` 无警告。
- 真实 12 组工程 smoke（smoke 域 2026100099/256，BEHAVIOR_OOD）：12 accepted /
  17 attempts，`ENGINEERING_SMOKE_ONLY`，`prefix_max_abs_diff=0.0`，双文件真值隔离
  与 receipt 绑定核验通过，正式路径未触碰。运行约 20s（<30s 心跳间隔，故中途心跳
  未触发；30s 心跳机制由测试钉死）。
- 全量回归：见 task-2-report.md（含 Stage 2/2B 已登记豁免解读）。
- Task 3 focused tests：五个 stage4d 测试文件 → 151 passed，`-W error` 无警告。
- 真实 preflight：`PREFLIGHT_PASSED`（§2.1 来源链 + 全域 seed 零重叠审计）。
- 三个 pilot 真实采集与评估：receipt/report 数值见 task-3-report.md；全量回归结果
  与 7 项豁免解读同见 task-3-report.md。
- Task 4 focused tests：七个 stage4d 测试文件 → 184 passed，`-W error` 无警告。
- 真实 power 相位：`D1_POWER_FROZEN`，selected n=2000（见
  `power/power_report.json` 的 4 保留 × 4 n 联合通过率矩阵与绑定条件）。
- 真实 lock-d1 相位：`D1_LOCKED`；`prot.verify_protocol_lock`（base）与
  `verify_confirm_protocol_lock`（extended，含锁时 seed 审计 live 重跑）双绿。
- locked smoke + readiness：见 task-4-report.md（锁验证 + 工程 smoke 复验 +
  专项 suite `-W error` + 全量回归 7 项登记豁免解读）。
- Task 5 focused tests：八个 stage4d 测试文件 → 210 passed，`-W error` 无警告
  （含新增 `tests/test_stage4d_inference.py` 26 项；`tests/test_stage4d_collection.py`
  的 `_config` 做最小隔离修复——正式 `paths` 重定向到 tmp，因真实 D1 产物已存在于
  真实工作区，smoke 测试的“正式路径不存在”断言改为指向 tmp 重定向路径）。
- 真实 recovery-lock 创建后锁链验证：base `verify_protocol_lock` + extended
  `verify_confirm_protocol_lock`（含恢复链解析 + live §2.1 来源/seed 审计）全绿；
  `confirmation_used.lock` 缺席核验通过。
- 真实 D1 采集：`D1_COLLECTED`，2000/2000 accepted、7415 attempts、2h12m20s
  （30s 心跳、原子 checkpoint/resume、rejection ledger 全程）；receipt status=COLLECTED、
  双文件 group_seed SHA 同序绑定一致。
- 真实冻结推理：`INFERENCE_COMPLETE`，2000 组/20000 分支/n_valid=2000，cuda float32
  batch=32，constant-random 种子 2664752168（=[2026110994,4]+salt 2026093203）。
- 确定性复检 + 分数冻结：`determinism_receipt.json` bitwise_identical=True（前 32 组
  两次重跑逐元素一致且等于正式产物前导切片，上下文零填充 20000/2000）；
  `score_freeze_receipt.json` 落盘，status=`D1_SCORES_FROZEN`。

Task 6 完成。先按 §13 落三枚 write-once 恢复锁再裁判：`recovery_lock_r2.json`
（binds_previous=r1 SHA `b64cfb4f…f5f6d`，changed=statistics+runner，reason=计划内
Task 6 裁判 phase 就位）→ `recovery_lock_r3.json`（changed=inference：freeze 复验把
recovery_locks 从数据绑定中分离——冻结链 append-only 增长合法，数据/分数绑定仍逐位相等）
→ `recovery_lock_r4.json`（changed=inference：`_assert_confirmation_unused` 改为仅首次
冻结时调用，允许 burn 后的分数复验）；链 r1→r2→r3→r4 全部 resolve==当前（43 文件），
`verify_confirm_code_chain` 绿。唯一一次解封：`burn-confirmation-lock` 相位先经
`freeze_scores_d1` 全量复验，再原子烧毁 write-once `confirmation_used.lock`（此后
`assemble_d1_quantities`/`adjudicate_d1` 方可打开 sealed truth；burn 不读 truth 字节，
仅经 collection/freeze receipt 记录的 hash 绑定）。裁判：`adjudicate` 相位读取冻结
`risk_scores.npz` + 解封 truth + 锁分层阈值，计算 §7.1 八条件与 §7.2 五条件（10,000 次
整组 paired bootstrap、entropy `2026110995`、双侧 95% CI，每个 draw 的全部量从同一
resample 索引向量重算）。**Gate TASK-OOD（§7.1）八条件全过**、**Gate OOD-RISK（§7.2）
五条件全过**，终态=`STAGE4D_OOD_ROBUSTNESS_CONFIRMED`（n=2000，passed=True）。
主终点（95% CI）：`I_AURC_REGRET=0.2077`（[0.1276, 0.3050]）、`I_REGRET@80=0.1855`
（[0.1274, 0.2366]）、`PR-AUC−prevalence=0.2157`（[0.1805, 0.2531]）；5/5 Direct seed
双指标为正；域偏移条件二 n_shifted=3、分类器 AUC CI 下限 0.9803≥0.70。§8.1 产物全部
create-once 落盘：`stage4d_ood_confirmatory_report.json`、`STAGE4D_OOD_DECISION.md`、
`Table_OOD_1_gate.csv`、`Table_OOD_2_baselines.csv`、`Table_OOD_3_seed_strata.csv`、
`Fig_OOD_1_risk_coverage.png`、`Fig_OOD_2_domain_shift.png`、`d1_group_quantities.npz`、
`d1_group_quantities_receipt.json`。一次性纪律已实测：resume→`verified_terminal`（报告
逐字节一致），直接二次 `adjudicate_d1`→`FileExistsError`。论文边界（§8.2）：仅允许表述
"冻结 decision-risk 方法在预注册 Merge 生成机失配下仍获支持"；**禁止**声称跨平台外部
有效性、真实道路泛化、线上安全提升、闭环策略已改善或对任意 OOD 鲁棒。确认集已永久
消耗，不得复用或二次裁判；Stage 4B/R2/4C/4C Plus 既有终态不变。
- 最终复验：锁链（code/model/data hash + recovery 链）全绿；全部冻结产物 hash 与 receipt
  一致；封存 truth 自采集起原封未动；`confirmation_used.lock` 仍不存在。
- 全量回归（修复后完整重跑）：7 failed / 1727 passed（749.40s），7 个失败节点与
  `readiness_exemptions.json` 登记豁免逐一精确相等、零新增失败（capacity snapshot
  漂移类解读见 task-5-report.md §5）。
- Task 6 focused tests：九个 stage4d 测试文件（新增 `tests/test_stage4d_adjudication.py`
  18 项 + `tests/test_stage4d_inference.py` 27 项）→ 229 passed，`-W error` 无警告；
  `tests/test_stage4d_runner.py` 的 not-implemented 守卫改为"全相位已实现且可调用"。
- 真实恢复锁链：r1→r2→r3→r4 逐环 resolve==当前（43 文件），`verify_confirm_code_chain`
  绿；base `verify_protocol_lock` + extended `verify_confirm_protocol_lock` 双绿。
- 真实 burn：`confirmation_used.lock` 原子烧毁（唯一解封动作）；burn 前
  `freeze_scores_d1` 全量复验绿。
- 真实裁判：`STAGE4D_OOD_ROBUSTNESS_CONFIRMED`（n=2000，passed=True）；§7.1 八条件、
  §7.2 五条件全过；主终点 CI 见上。`stage4d_ood_confirmatory_report.json` 与
  `STAGE4D_OOD_DECISION.md` 落盘，表/图 hash 由 report 绑定。
- 一次性纪律实测：resume→`verified_terminal`（报告逐字节一致）；直接二次
  `adjudicate_d1`→`FileExistsError`；`reports-status-sync` 已同步 §8.1 产物核验。

## 终态（Task 6 已完成）

本计划已到达终态 `STAGE4D_OOD_ROBUSTNESS_CONFIRMED`，确认集已永久消耗，授权随终态终止。
无"下一步"：D0-X/D2、MetaDrive/ScenarioNet/SUMO 均未授权，须另立计划并重新审批。

Erratum（Task 6 裁判后复核登记）：本节原稿（"裁决"条目）曾写"5000 resamples、双侧
80% CI"，系早期草稿残留；实际执行为 **10,000 resamples、双侧 95% CI**（entropy
2026110995），与 `protocol_lock.json`、freeze receipt、`stage4d_ood_statistics.py`
冻结口径及最终裁判报告一致。文档性更正，不影响任何已裁判数值。

## 阻塞

无。

## 给后续任务的接口说明

- `build_profile_config(profile, group_seed)` → 完整确定性扰动 spec（行为表 + 几何抽样）。
- `assign_vehicle_behaviors(group_seed, entity_ids)` → `{entity_id: "aggressive"/"cautious"}`，
  恰 2+2，group-seed 稳定、与输入顺序无关。
- `apply_profile_to_env(env, profile_config, assignment=None)` → 在 reset 后应用扰动，
  返回分支一致性快照；`snapshots_identical(a, b)` 比较。
- `verify_protocol_lock(run_dir)` → base 校验原语（锁已创建；绿）。Task 4 另有
  `stage4d_confirm_protocol.verify_confirm_protocol_lock(run_dir)` → 扩展校验（扩展代码
  registry drift + selected_n/profile 一致性 + 锁时 seed 审计 live 重跑）。
- `verify_pinned_sources()` / `build_source_inventory()` / `build_code_inventory()` /
  `scan_historical_group_seeds()` / `audit_global_seed_overlap()` / `make_candidate_seed_schedule()`
  供后续复用。
- `PROTOCOL_CODE_FILES` 仍为 Task 1 的两个模块（Task 1 冻结文件未改）；扩展
  registry 在 `stage4d_confirm_protocol.CONFIRM_REGISTRY`（Task 5 后 43 文件），锁的
  `code_sha256`=Task 1 base registry（严格相等），`extended_code_sha256`=扩展 registry
  （Task 5 起经 recovery_lock_r1 链解析绑定）。
- `stage4d_ood_statistics.py` 接口（§9.2 签名逐字）：
  `compute_domain_shift(id_features, ood_features, bootstrap_seed) -> dict`、
  `evaluate_d0_profile(integrity, shift, task) -> dict`、
  `select_ood_profile(profile_reports) -> dict`、
  `simulate_d1_power(group_quantities_path, candidate_ns, retention_factors, seed) -> dict`；
  另有 `pre_intervention_features`、`load_r2_train_id_features`（hash 核验 + 只读
  object_history）、`evaluate_task_support`、`evaluate_pilot_integrity`、
  `assemble_pilot_task_quantities`、`evaluate_d1_power_draws`、`simulate_d1_joint_power`
  供 D1 复用；`assert_d0_read_legal(path)` 为 D0 禁区 fail-closed 守卫。
- `run_stage4d_ood.py`：`PIPELINE`（§12 相位序逐字，全部相位已实现并注册）、
  `run_phase(phase, config)`、CLI 子命令 preflight/collect-pilot/evaluate-pilots/
  select-profile/power/lock-d1/locked-smoke-readiness/collect-d1/infer-d1/freeze-scores/
  burn-lock/adjudicate/reports-sync/run-all；`TerminalState`
  （STAGE4D_OOD_TASK_INSUFFICIENT、STAGE4D_OOD_CONFIRMATION_UNDERPOWERED 与全部
  非 CONFIRMED 裁判终态为吸收态，resume 重抛；CONFIRMED resume 返回 verified_terminal）。
- `stage4d_ood_inference.py`（Task 5，registry 经 recovery_lock_r1 绑定）：
  `load_engine_config()`（stage4cr2 锁链绑定的引擎配置）、`constant_random_seed_d1(checked)`、
  `inference_artifact_paths(checked)`、`infer_and_score_d1(config)`（create-once，
  rerun=verify）、`run_determinism_recheck_d1(config)`（32 组 bitwise + 前导切片交叉核验）、
  `freeze_scores_d1(config)`（锁/链/收据全绑定 + used-lock 拒绝）；常量
  `DETERMINISM_GROUPS=32`、`D1_ROLE="d1_confirm"`、`STATUS_INFERENCE="INFERENCE_COMPLETE"`。
- `stage4d_confirm_protocol.py` 新增 §13 原语：`recovery_lock_paths(lock_path)`、
  `resolve_locked_extended_code_hashes(lock_path)`、`create_recovery_lock(config, *,
  changed_files, added_files, reason)`、`verify_confirm_code_chain(run_dir)`（cheap
  drift 检查，无 live 审计）；added_files 支持 role=stage4d_own / reused_frozen
  （后者经 stage4cr2 锁链交叉绑定）。
