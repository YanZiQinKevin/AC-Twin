# Stage 4C HANDOFF_STATE

> 计划：`AC-TWIN-RFER-STAGE4C-MULTIACTION-REGRET-2026-09-15-V0.1`
> 本文件是恢复锚点：记录当前状态、冻结决定、完成产物、最近验证、下一动作与 blocker。

## 当前状态

- 阶段：**`STAGE4C_DEV_NOT_READY_FOR_CONFIRMATION` — 科学硬停止（development screen adjudication 判定 Gate RISK-DECISION_dev 失败）**。按计划"Gate TASK/Gate RISK-DECISION_dev 失败即停止"规则，**不进入 power/confirm 链**。
- 授权：用户于 2026-09-15 以"开始执行"明确批准；按计划自动连续推进至本停止点。
- 冻结来源链：执行前已重算 SHA256，全部 MATCH。
- 一次性锁已烧：`development_screen_used.lock`（2026-09-16T01:17:32）；screen 量已封 `screen_group_quantities.npz`(SHA256 81a6980c217d3ade61f13535)。不可重跑。

## 冻结决定（已锁定，不可更改）

- 统计单元 = 十动作决策组（非分支/车辆/时间点）；bootstrap 整组重采样。
- 动作集合 = 对称 5×2 = 10 计划，含 `(IDLE,1)` 与 `(IDLE,3)`；顺序即 tie-break 顺序。
- 后果函数 = `0.80*C_int + 0.20*C_eff`；交互尺度 5.0m/2.0m；regret 钳制容差 1e-10。
- `ALWAYS_IDLE` 资格基线取 `(IDLE,3)`（计划未指明两 IDLE 之一；此为已登记的 plan-interpretation 决定，写入 decision receipt）。
- 组级风险聚合 = `max(r(a1), r(a2))`（top-2）；`MARGIN_ONLY = -(J_hat(a2)-J_hat(a1))`。
- seed：dev entropy 2026091801/size 5000；confirm entropy 2026091901/size 15000。
- dev = 500 selection + 300 单次 screen；confirm 候选 {600,900,1200,1800,2400}。

## 完成产物

- `code/ac_twin/stage4c_objective.py` — 后果函数/regret（Task 1）。
- `code/ac_twin/stage4c_protocol.py` — 协议/seed/掩码/冻结清点/dev+confirm 锁（Task 1/5）。
- `code/collect_stage4c_merge_multiaction.py` — 同状态十分支组采集器（Task 2）。
- `code/ac_twin/stage4c_inference.py` — 组级冻结推理/动作风险/top-2 聚合/确定性复核（Task 3）。
- `code/ac_twin/stage4c_statistics.py` — 规则冻结/曲线/组bootstrap/Gate/组量装配（Task 4）。
- `code/ac_twin/stage4c_power.py` — 组级联合功效与选样（Task 4）。
- `code/run_stage4c.py` — 编排 runner,14 个子命令（Task 5）。
- 测试：objective/protocol/collector/inference/statistics/power/driver 共 49 项。
- `experiments/stage4_reliability/stage4c/stage4c_config.json`（通过 `validate_config`）。

## 工程从简决定（方案 B，不影响科学结果）

- development 的 selection(500）与 screen(300）写为两个独立 NPZ(`*_selection.npz` / `*_screen.npz`)，而非单一合并文件；角色天然隔离，无覆盖风险。
- `t_star` 两层用固定分裂 `t_star>=13`（数据无关、预注册）；density 两层用 selection 冻结中位阈值。

## 最近验证

- `preflight` → PREFLIGHT_OK；`lock-development` → dev_sha 804d639a5ade5430。
- `smoke` → ENGINEERING_SMOKE_OK（12/12 组，prefix_max_abs_diff=0.0；候选池修复为 60 个一次性 seed）。
- `collect-development` → selection 500/500、screen 300/300；所有拒绝均为 prefix_terminated，零完整性失败，prefix_max_abs_diff=0.0。
- `infer-development` → 两角色完成；确定性复核 bitwise_identical=True、max_abs_diff=0.0。
- `freeze-decision-rule` → 冻结：最强基线=MARGIN_ONLY；high_regret_threshold=0.0177638（80% 分位）；density_threshold=1.0（中位）；**reranking_lambda=0.0（退化 → 标记 RERANKING_SCORE_DEGENERATE）**。
- 测试回归 197 passed。
- **`adjudicate-development` → STAGE4C_DEV_NOT_READY_FOR_CONFIRMATION**：
  - Gate TASK：**TASK_QUALIFIED（5/5 全过）**——任务成立（oracle 可分辨，模型有信息量）。
  - Gate RISK-DECISION：**RISK_DECISION_FAILED（0/6）**——
    - I_AURC_REGRET point=−0.5077（CI_lower=−0.968，需 ≥0.05）；
    - I_REGRET@80 point=+0.0046（CI_lower=−0.199，需 ≥0.03）；
    - PR−prevalence point=+0.0136（CI_lower=−0.015，需 >0）；
    - 正方向对 0/5（需 ≥4）；strata 最小 I_AURC=−0.958（floor −0.10）；trim1/trim5 均负。
  - 结论（dev screen 单次裁决）：**冻结的 Stage 4B 风险头不能识别高 decision-regret 决策**；主要假设在 development 上未获支持。

## 下一动作（科学硬停止，非自动继续）

- **停止**。不得把 risk-penalty reranking（次要分析）提升为替代主判据；不得进入 power/confirm。
- 待用户/研究负责人裁决：是否在 plans/ 中登记 dev 失败结论并规划后续（如风险头重设计、或结束 Stage 4C 主线）。本阶段 confirm 链保持未启动。

## Blocker

- **科学硬停止**：development screen 上 Gate RISK-DECISION_dev 失败（0/6 条件）。这是计划定义的停止点，需人工裁决下一步方向。
