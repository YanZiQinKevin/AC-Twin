# Stage 3A 重训 Readiness Checklist（修复计划 v0.1 §5 Task 8）

> 计划：`AC-TWIN-RFER-STAGE3A-REPAIR-2026-09-09-V0.1`  
> 形成：2026-09-09  
> 性质：工程 readiness 记录，非科研证据。任何未勾选项阻止 Task 9/10 长任务。

## 测试与回归

- [x] 全部新增专项测试通过：`test_stage3a_invalidation`（5）、`test_stage3a_gate_inputs`、`test_stage3_response`、`test_stage3_training`、`test_stage3a_driver`、`test_stage3a_integration`（含 Task 5/7 硬化）、`test_stage3a_replacement_dev`（27），均先红后绿。
- [x] 全量回归 `pytest tests/ -W error`：**608 passed，0 failed，0 warning**（2026-09-09，含 Task 6 后）。
- [x] 旧事故证据：`stage3a/INVALIDATION_RECORD_20260909.json` 三个 SHA256 于 2026-09-09 复算一致（旧报告/旧 lock/旧管线日志均只读保留）。

## 工程 smoke（独立临时目录，12 episode / 1 epoch / CPU）

- [x] search → fit-five → select-final → lock-replacement-dev → gate-r-dev-retrain 全链通过；smoke 产物全部在临时目录，未进入本 run 目录；记录：`logs/retrain_smoke_20260909.json`（`ENGINEERING_SMOKE_ONLY`，合成数据上的 gate 判定无科研意义）。
- [x] 同 config/seed 双空目录重现 QA：8 配置 × 2 次独立 search 签名逐一致；记录：`logs/qa_determinism_20260909.json`（`all_identical=true`）。
- [x] eval 顺序 QA：请求顺序保持、逆序请求逐元素一致、batch_size=1 与整批一致；episode 首尾 ID 与 `r_hat` SHA256 记录：`logs/qa_eval_order_20260909.json`。

## 合同冻结确认（长任务前逐项核对）

- [x] 搜索空间未扩展：`hidden_dim {64,96}` × `response_feature_dim {32,64}` × `lambda_cls {0.10,0.25}`（`validate_stage3a_config` 机械校验）。
- [x] Stage 1 冻结阈值未改：`zero_tolerance=0.0001`、`t1=0.24843344965718578`、`t2=1.898992796585611`。
- [x] Stage 2 封存集三重拒绝在位（路径/逻辑路径/SHA256），并有测试。
- [x] replacement 候选样本量冻结为 `[300,400,500,600,700,800]`，功效 seed 冻结 20260909，最弱指标定样本量，800 不足硬停止（`REPLACEMENT_DEV_POWER_INFEASIBLE`）。
- [x] replacement 采集 seed 全集零重叠扫描**包含** Stage 2 封存集（`scan_all_episode_seeds`）；禁写 Stage 2 / merge_2000 / Gate C / 旧 stage3a 路径。
- [x] 旧 split 2 不能作为 replacement Gate 输入（dev-source SHA 相同即拒绝）；Gate C 路径在烧毁单次锁**之前**拒绝（refusal 不消耗裁判）。
- [x] `final_selection.json` 单次写入、幂等读取、拒绝覆盖；回执签名覆盖配置/基线/五 checkpoint 签名/五选择分数/上游文件 SHA256。
- [x] 本 run 目录当前只含本 checklist 与 `logs/` 治理记录；`search/`、`fit_five/`、`selection/`、`power/`、`replacement_dev/`、`gate_r_dev/` 均未创建。

## 待用户手动启动（助手不自动运行）

1. Task 9：`search` → `fit-five` → `select-final`（`$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'`；命令见修复计划 §5 Task 9）。
2. Task 10：`prepare_stage3a_replacement_power.py`（机械冻结 episode 数）→ `collect_stage3a_replacement_dev.py`（允许 `--resume`）→ `lock-replacement-dev` → `gate-r-dev-retrain` 一次。
3. 任一拒绝（exit 3）、`DETERMINISM_FAILURE`、`REPLACEMENT_DEV_POWER_INFEASIBLE` 或签名/seed/对齐失败：立即停止，回用户评审。
