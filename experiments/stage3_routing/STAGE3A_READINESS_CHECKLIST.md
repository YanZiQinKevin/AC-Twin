# AC-Twin Stage 3A 启动前 Readiness Checklist

> 计划编号：`AC-TWIN-RFER-STAGE3-2026-09-08-V0.2`（`APPROVED_FOR_IMPLEMENTATION`）
> 生成日期：2026-09-09
> 性质：启动前检查清单（非科研结论）。全部勾选并由用户**单独确认预算**后，才由用户手动启动 `search` / `fit-five` / `gate-r-dev`。
> 状态机：**`COMPUTE_BUDGET_APPROVED`**（2026-09-09 用户确认预算）。下一阶段 `RESPONSE_DEV_QUALIFIED` 须 Stage 3A 长任务 + Gate R_dev 裁判通过。

| # | 检查项 | 状态 | 证据 |
|---|---|---|---|
| 1 | v0.2 计划 §11 全部十项已批准，状态为 `APPROVED_FOR_IMPLEMENTATION` | ✅ | 计划头部 `批准记录`；`01_AC_TWIN_DOCUMENT_STATUS_INDEX.md` |
| 2 | 搜索空间为已登记范围，未扩展：`hidden_dim{64,96}` × `response_feature_dim{32,64}` × `λ_cls{0.10,0.25}`（≤8 配置） | ✅ | `stage3_training.SEARCH_*`；`stage3a_config.json`；`search_space_configs()`=8 |
| 3 | 单 seed 筛选 → 入围 ≤2 配置 → 补五 seed；选择只看 validation | ✅ | `MAX_FINALIST_CONFIGS=2`；`search`（`selection_basis=validation loss only`） |
| 4 | Stage 2 holdout 三重封存拒绝生效（绝对路径 / 逻辑路径 / 已知 SHA256），config 校验通过 | ✅ | `stage3_data.assert_not_sealed_stage2`；`validate_stage3a_config` |
| 5 | 训练监督只来自 train split 配对真值；模型输入无 CF future / R_true / 真值箱 | ✅ | `stage3_response` 模块 + 测试 |
| 6 | train-only scaler 已落盘，validation/test 仅复用、不得重拟合（fit-once + 签名校验 load） | ✅ | `stage3_scaler.FeatureScaler`；`prepare_scaler`/`scale_tensors`；`test_stage3_scaler.py` |
| 7 | run fingerprint 覆盖数据签名 / split / 完整配置 / scaler / 类别权重 / 合同标识；恢复后早停状态与不中断一致 | ✅ | `compute_run_fingerprint` + `STAGE3A_CONTRACT_ID`；checkpoint 持久化 `epochs_without_improvement`；`test_stage3a_gate_inputs.py` |
| 8 | 断点续跑（每 epoch 断点 + fingerprint 守卫）与失败退出（失败摘要 + 非零退出）已具备并有测试 | ✅ | `train_model(checkpoint_path/resume/run_fingerprint)`；`test_stage3_training.py` |
| 9 | 进度显示含 phase/seed/completed/total/%/elapsed/ETA 且 ≥30s 心跳 | ✅ | `progress.ProgressReporter`；`test_progress.py` |
| 10 | 全量回归 `pytest -W error` 通过，原有测试无破坏 | ✅ | 全量回归 **550 passed**（2026-09-09） |
| 11 | 12 episode/1 epoch 工程 smoke 通过，状态仅 `ENGINEERING_SMOKE_ONLY` | ✅ | `code/run_stage3.py smoke` |
| 12 | 正式设备完整 epoch benchmark 已实测（response + router，真实 VRAM），设备与预算一致 | ✅ | `benchmark.json`（cuda / RTX 3070 Ti / 2026-09-09；response 37.0s、router 27.0s、VRAM 0.146/0.149 GB） |
| 13 | `COMPUTE_BUDGET.md` §5.3 完整：3A(response)/cross-fit(router)/3B(router,tier1+2)/正式集合(采集+冻结推理+10,000×bootstrap)/失败重试断点/典型+最坏 | ✅ | `COMPUTE_BUDGET.md`（GPU 小时 + VRAM，设备一致） |
| 14 | Stage 3A 驱动接通完整 Gate R_dev：train-only 基线、validation 温度、五 seed 推理、H4 全指标、episode bootstrap、七项裁判、落盘报告 | ✅ | `gate-r-dev` 子命令 + `stage3a_gate.evaluate_h4_gate` |
| 15 | 两个 finalist 的五 seed checkpoint 均完成签名 + 完整性检查；fit-five 非 stub 覆盖 | ✅ | `save/verify_predictor_checkpoint` + `five_checkpoint_signatures`；集成测试 |
| 16 | 真实小数据 CLI 集成测试：search → fit-five → select-final → gate-r-dev 实际走通（非 monkeypatch trainer） | ✅ | `test_stage3a_integration.py`（40 episode 合成 npz） |
| 17 | 初始距离为**逐车** (E,N) 基线特征（非 episode 均值广播），不人为削弱基线 | ✅ | `make_initial_distance`→(E,N)；`LinearResponseBaseline`/`MultinomialRegimeClassifier` 逐车消费 |
| 18 | 五 seed 方向一致覆盖**全部**指标（I_MAE/Spearman/两个 PR-AUC/Brier），缺项即失败 | ✅ | `check_five_seed_consistent` + `CONSISTENCY_METRICS` |
| 19 | development test 单次使用**三重锁**：①`final_selection.json` 一旦生成不可覆盖（select-final 重跑→退出3）；②与配置无关的全局 `development_test_used.lock` 在读取 development test 前原子写入（任何配置下第二次运行→退出3）；③Gate 重验回执签名（覆盖最终配置+冻结基线+五 checkpoint 签名，篡改/换 finalist→退出3） | ✅ | `cmd_select_final`/`cmd_gate_r_dev`/`compute_selection_receipt_signature`；4 项集成测试 |
| 20 | **用户已单独确认预算**；长任务命令由用户手动启动 | ✅ | 2026-09-09 用户确认 → `COMPUTE_BUDGET_APPROVED` |

## 说明

- 第 20 项是进入 `COMPUTE_BUDGET_APPROVED` 的唯一闸门；确认前不得启动任何正式/长任务。
- benchmark 为**正式 CUDA 设备**（RTX 3070 Ti）完整 train epoch 实测，response 与 router 分别计时；显存为真实 VRAM。配置 `device=cuda` 与预算设备一致（不一致时 `budget` 直接报错）。
- Stage 3B 与 cross-fit 使用 **router** 实测耗时，**不**套用 response predictor 耗时；Stage 3B 按 §5.3 分两层（tier-1 主条件 20 runs / tier-2 诊断 30 runs）。
- 正式集合（采集 / 冻结推理 / 10,000× bootstrap）为**冻结前估算**，须 Gate R_dev + T_dev + C_dev + 功效全部通过后才冻结。
- development test 单次 Go/No-Go（三重锁）：`select-final`（validation 机械选定 + 基线冻结 + 不可变 receipt）→ `gate-r-dev`（先重验回执签名——覆盖最终配置/冻结基线/五 checkpoint 签名，再在读取 development test 前原子获取与配置无关的全局 `development_test_used.lock`；任一失败退出码 3，全程只有一份报告）。
