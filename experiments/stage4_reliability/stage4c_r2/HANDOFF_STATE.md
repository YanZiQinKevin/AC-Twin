# Stage 4C-R2 HANDOFF_STATE

> 计划：`AC-TWIN-RFER-STAGE4C-R2-DECISION-RISK-SCALING-2026-09-16-V0.2`
> 本文件是恢复锚点：记录当前状态、冻结决定、完成产物、最近验证、下一动作与 blocker。

## 当前状态（终态）

- **科学终态：`STAGE4CR2_DEV_READY_FOR_CONFIRMATION`（2026-09-17，Task 5 完成，计划全部 5 个
  Task 执行完毕）。** 含义按第 1.3 节论文边界：多动作 decision-risk 的固定 Direct 模型
  **开发验证通过、值得确认**；这**不是**正式确认，本计划批准从未授权正式 confirm。
- 确认功效诊断（第 10 节，唯一一次）：候选 n={2000,3000,4000,5000} 在 75% 效应收缩下联合
  功效均为 **1.0000**（10,000 次 Monte Carlo 完整 AND 链模拟），50% 收缩下分别为
  0.9698/0.9982/0.9997/1.0000，100% 下全为 1.0000；**最小可行 n=2000**，binding 条件为
  `i_aurc_regret`（75% 边际功效 1.0）。协方差直接取自本次 10,000 次 whole-group paired
  bootstrap（entropy `2026092702`）的 draws，未从 80% CI 宽度反推 95% SE。
- **未创建任何 confirm 资源**：无 confirm lock、无 confirm schedule、无 confirm 采集；
  `confirmation_power_diagnostic.json` 中 `confirm_resources_created=false`。任何正式确认
  须另立计划并重新审批。
- Stage 4C Plus 终态 `STAGE4CP_FEASIBILITY_NOT_SUPPORTED` 与 Stage 4C 终态
  `STAGE4C_DEV_NOT_READY_FOR_CONFIRMATION` 永久保留，未被改写。

## 一次性资源消耗清单（不可复用、不可二次裁判）

- `development/development_screen_used.lock` 已烧毁（8,000-group screen 永久消耗）。
- `development/screen_group_quantities.npz` + receipt + `development_screen_report.json`
  均 create-once，二次裁判在 IO 层被拒绝。
- `development/confirmation_power_diagnostic.json` create-once（第二次写入抛
  `FileExistsError`）。
- validation 裁判与 `final_selection.json` create-once；二次 validation 裁判被拒绝。
- train 支持度纯工程 recovery：`train_support_report.json`（失败，保留）→
  `train_support_report_r1.json`（修复后通过），recovery 绑定已写入 screen 量 receipt。

## 关键数值（只登记实际结果）

- Screen（8,000 组，相对冻结最强基线 `SEED_VOTE_ENTROPY`）：`I_AURC_REGRET=+0.432452`
  （80% CI 下界 `+0.396432`）、`I_REGRET@80=+0.291028`（下界 `+0.265663`）、
  `PR-AUC−prevalence=+0.332658`（下界 `+0.319598`）；5/5 seed `I_AURC>0`、5/5 `I_R80>0`；
  四 strata 最小 `I_AURC=+0.361432`（interaction_high，下限 −0.10）；trim1/5 =
  `+0.433816`/`+0.459319`；|Spearman(ensemble, MARGIN_ONLY)| 低于 0.995 上限；
  Gate TASK-R2 与 Gate RISK-DECISION-R2_dev 全部条件通过。
- 功效诊断：`theta_hat` 即 screen 点估计；reference SE（n=8,000 尺度）`I_AURC 0.024092`、
  `I_R80 0.018553`、`PR 0.010145`。

## 冻结决定（不变，见前版与 protocol_lock.json）

- 统计单元=十动作决策组；10,000 次整组配对 bootstrap（entropy `2026092702`），双侧 80% CI。
- 唯一候选 `SAME_INFORMATION_DIRECT`（width=43，参数精确 15,740，`lambda_cls=0.25`），
  五 seed [8101,8202,8303,8404,8505]，ensemble=算术平均。
- 规模 5,000 train + 1,500 validation + 8,000 一次性 screen；seed schedule entropy
  `2026092880`、60,000 候选、固定角色片段；全局零重叠已审计。
- 功效诊断参数：candidate n={2000,3000,4000,5000}、shrink={100%,75%,50%}、75% 为唯一
  阻断裁判、目标联合功效 0.80、Monte Carlo 10,000。

## 完成产物（Task 5 新增）

- `code/ac_twin/stage4cr2_power.py` — 条件性确认功效诊断（只读 create-once 逐组量；
  协方差来自本次 bootstrap draws；完整 AND 链联合模拟；create-once 诊断写出；任何 outcome
  下不创建 confirm 资源）。
- `code/run_stage4cr2.py` — 统一入口：16 个子命令（preflight…power-diagnostic、verify、
  run-all）；hash 匹配跳过已完成阶段；一次性裁判只验证不重跑；每个科学硬停止返回专用
  非零退出码（3/4/5/6/7/8/9/10）；长任务 ≤30 s 心跳。
- `tests/test_stage4cr2_power.py`（12 项）、`tests/test_stage4cr2_driver.py`（16 项）。
- `experiments/stage4_reliability/stage4c_r2/development/confirmation_power_diagnostic.json`
  （create-once）。

## 最近验证

- `python -m pytest -q -W error`（stage4cr2 九个专项文件）：**197 passed，零 warning**。
- `python -m pytest -q -W error tests` 全量回归一次（交接前）：**1308 passed / 6 failed /
  零 warning**（564.6 s）。6 项失败全部是已登记的"冻结协议测试对活数据重扫"类：
  `tests/test_stage2_protocol.py` 5 项 lock 测试（2026-09-14 文档索引已登记，自 Stage 4A
  数据落地起失败，重叠种子 1984715017 非 R2 种子）+ `tests/test_stage2b_protocol.py` 1 项
  （stage2b 已被整体更换、从未使用的旧候选 entropy 重生成 schedule 与 R2 新接受种子的偶然
  重叠；stage2b 实际使用并持久化的 development_v04/pilot schedule 与 R2 零重叠已核实）。
  按既有纪律，冻结代码不得修改，是否修订这些测试由用户决定；与 Task 5 新增代码无关
  （本任务只新增文件）。
- `python code/run_stage4cr2.py verify --config …/stage4cr2_config.json`：exit 0，
  lock verified、final_selection=VALIDATION_QUALIFIED、
  screen=`STAGE4CR2_DEV_READY_FOR_CONFIRMATION`、power_diagnostic selected_n=2000。
- `python code/run_stage4cr2.py run-all --config …`：exit 0，15/15 阶段全部
  skipped/verified，无任何一次性阶段重跑（9.7 s）。

## 下一动作

- 无（本计划范围内）。正式 confirm 须另立计划并由用户重新审批；最小可行规模证据为
  n=2000（75% 收缩联合功效 1.0）。

## Blocker

- 无。

## Errata（2026-09-17 最终整体评审后登记；不影响任何 hash 绑定产物）

- Task 4 报告将 screen 采集墙钟时间误记为 5h56m48s;`screen_collection.log` 的实际累计值为
  06:10:06(16,623 attempts,与 receipt 一致)。以日志为准。
- Task 5 报告将新增测试误记为 "12+16 项";实际收集为 11(power)+ 17(driver)= 28 项，
  "28 passed" 总数正确。runner 子命令实际为 17 个（15 阶段 + verify + run-all)。
- `READINESS_CHECKLIST.md` 中 Task 2 与正式 lock 的勾选框仍显示"未开始",为陈旧状态;
  以本文件与 `development/protocol_lock.json` 为准。
