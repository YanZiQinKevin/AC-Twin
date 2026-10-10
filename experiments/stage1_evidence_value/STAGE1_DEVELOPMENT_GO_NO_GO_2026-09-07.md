# Stage 1 验尸实验 — 开发性 Go/No-Go 决策

> 文档状态：**COMPLETED_DEVELOPMENT_AUDIT**
> 日期：2026-09-07
> 协议修复版本：`AC-TWIN-RFER-STAGE1-CLOSURE-2026-09-07-V1`
> 上位研究计划：`AC-TWIN-RFER-2026-09-07-V1`
> Stage 1 实施计划：`AC-TWIN-RFER-STAGE1-IMPL-2026-09-07-V1`
> **性质：开发性结论，不是确认性结论。现有数据已被多次查看，本文件不构成 `GATE_A_CONFIRMED`。**

---

## 1. 环境与解释器

- 工作目录：AC-Twin 项目根（OneDrive 云盘）。
- 解释器：Python 3.13；PyTorch `2.14.0+cu132`；`torch.cuda.is_available()=True`；设备 `NVIDIA GeForce RTX 5070 Laptop GPU`。
- 完整测试基线：`python -m pytest tests -q -W error` → **260 passed**。
- 本目录不是 Git 仓库；封存产物靠 SHA256 清单保护。

## 2. 封存训练产物完整性与无重训证明

- 两场景各 3 条件 × 5 种子（`11,22,33,44,55`）= 各 15 个 run，三条件参数量一致（`145066`）。
- 修复前快照：`stage1_sealed_artifact_hashes_before_protocol_fix.json`。
- 本次协议修复**未重新训练**：两场景均用 `--reanalyze-only`，报告显示 `training_performed=false`、`checkpoint_modified=false`。
- 修复后逐项比对：所有 `.pt`、`*_training.json`、`*_run.json`、原有 `*_val_prediction.npy` / `*_test_prediction.npy` 的 SHA256 **全部不变**。
- 仅新增：每场景 5 个 `uniform_suffix_seed*_val_prediction_with_frozen_shuffled_suffix.npy` 与对应 `*_val_suffix_swap_metadata.json`，以及 `report.json`（重分析覆盖）与 `report_SUPERSEDED_protocol_incomplete_20260907.json`（旧报告备份）。

## 3. 三个协议缺口的修复证据

1. **Gate S/S1 split 泄漏与宽松阈值**：S1 非退化检查与自车正控制改为只用 validation；删除 `M1 ≤ 1.5×M0` 宽松阈值，固定为 validation 上 `mean_ego_error(M1) < mean_ego_error(M0)` 并报告点差与 episode-bootstrap 95% CI；Gate S 标记 `data_scope=train_validation_only`，代码路径不再读取 `indices["test"]` 或 `*_test_prediction.npy`。
2. **S3 预测证据改为固定 M1 suffix swap**：加载同一 `uniform_suffix_seed{seed}.pt`，固定 validation 的 `H/AF/ACF`，仅在「正确 suffix」与「冻结 validation manifest 指定的 donor suffix（特征+mask 同步交换）」之间切换；正确通道重推理与封存 validation 预测在冻结容差内一致（不一致即停止）；S3 不再读取 M2 权重或 M2 误差。
3. **S3 表征证据加入 episode-bootstrap CI**：ridge probe 仅用 split 0 表征/标签拟合，validation 上重建每车运动摘要，两个基线（scene mean、固定种子 shuffled label）差值各给 10,000 次 episode 聚簇 bootstrap CI，两 CI 上界均 <0 才通过。Gate A 补齐 leave-one-seed-out、leave-one-action-type-out、leave-one-episode-out + 固定 1%/5% 高误差删除，以及 manifest QA 作为显式必要条件。

## 4. Merge（主开发场景）结果

**Gate S = PASS**（S1/S2/S3 全过，仅 train/validation）：

- S1：三条件参数一致（145066）、损失有限、逐车输出非退化；自车正控制 `M1−M0 = -0.150`，95% CI `[-0.250, -0.056]`（<0，事实后缀对自车有用，符合既有正控制），独立 episode 300。
- S2：逐车响应幅度可预测。`Spearman(R_hat, R)=0.614`（CI 下界 0.597>0）；模型 MAE `0.210` 显著低于动作+horizon 基线 `0.416` 与加距离基线 `0.430`（差值 CI 上界均 <0）。
- S3：固定 M1 suffix swap 显示正确后缀显著优于打乱后缀（zero/positive_low/positive_mid 三箱 `L_wrong−L_correct` 的 95% CI 下界分别为 0.044/0.255/0.432，均 >0）；表征 probe 两基线差值 CI 上界均 <0。

**Gate A = PASS → `DEVELOPMENT_GO_TO_NEW_HOLDOUT`**（四个主统计条件 + 全部稳健性条件）：

- 低响应组（zero+positive_low）`V_evt = 0.281`，95% CI `[0.229, 0.332]`（>0）；
- `V_evt_low − V_evt_high = 0.068`，95% CI `[0.003, 0.131]`（>0，**勉强**，下界贴近 0）；
- `Spearman(R, eta_evt) = 0.287` `[0.259, 0.316]`（>0）；
- 协变量趋势 `log1p(R)` 系数 `0.283` `[0.239, 0.332]`（>0，控制 horizon/cf_action/cf_duration/初始距离/可见性后仍为正）；
- 稳健性：leave-one-seed-out 五点估计 `[0.078,0.064,0.060,0.071,0.066]` 全 >0；leave-one-action-type-out 全 >0（action 3 = 0.0023 最弱）；leave-one-episode-out 最小值 0.062、0 个反转；删除最高 1%/5% 高误差 episode 后点估计 0.062/0.059 均 >0；manifest QA 全过（self/same_seed/cross_split/duplicate_donor 全 0）。

**四箱原始误差与 V_evt（非单调，须如实陈述）：**

| 响应箱 | V_evt | rho_no | meanL_no | meanL_suffix | meanL_shuffled | agent-time | episode |
|---|---|---|---|---|---|---|---|
| zero | 0.339 | 0.371 | 0.116 | 0.073 | 0.111 | 20241 | 300 |
| positive_low | 0.032 | 0.129 | 0.432 | 0.377 | 0.389 | 1343 | 188 |
| positive_mid | -0.025 | -0.004 | 0.933 | 0.937 | 0.915 | 1249 | 169 |
| positive_high | 0.213 | 0.236 | 3.039 | 2.321 | 2.949 | 1167 | 111 |

`V_evt` 在四箱上**不是严格单调下降**（positive_high 高于 positive_low/mid）。因此结论只能表述为「**低响应组相对高响应组的事件特异价值更高、且随 R 存在总体衰减趋势（条件异质性）**」，**不得**表述为「每个响应箱严格单调衰减」。

**Gate B = NOT_SUPPORTED**：高响应区 `Delta = L_no − L_suffix` 点估计为正（0.627），95% CI 完全大于 0，独立 episode 111（≥30）。即高响应区事实后缀**没有**造成净负迁移。**禁止使用「双刃剑 / dual-edged evidence」表述。**

## 5. Roundabout（探索性场景）结果与边界

**Gate S = FAIL（S2 失败）→ `INCONCLUSIVE_MODEL_INSUFFICIENT`**；Gate A/B 保持 `BLOCKED`。

- S2 失败原因：模型逐车响应幅度 MAE（0.089）**不优于** 动作+horizon 基线（0.082）与加距离基线（0.085）；Spearman 虽 >0（0.152，CI 下界 0.112），但 MAE 条件不满足。
- 科学解读边界：只能表述为「**当前模型在响应稀疏的 roundabout 场景响应幅度预测能力不足**」，**不能**解释为 H2 不成立，也不能作为 Gate A 反例；不得据其选择任何增强。
- Roundabout 为探索性复现场景，**不得单独推翻或替代** merge 主判定；其 S2 失败记入跨场景风险登记。

## 6. 对照验收清单

- 修复前封存产物 SHA256 清单已建立；旧报告已存 superseded 版本；
- 未运行训练或 `--resume`；两场景 `--reanalyze-only` 成功，`training_performed=false`；
- S1 完全使用 validation 且删除 1.5× 阈值；S3 用同一 M1 checkpoint 做 suffix swap、不依赖 M2；S3 表征两基线差值均有 episode-bootstrap CI；
- Gate A 完成 leave-one-seed/action/episode-out 与固定 1%/5% 高误差删除；manifest QA 为显式必要条件；
- 完整测试 260 项全部通过 `-W error`；封存 `.pt`/日志/metadata/原有预测 hash 不变；
- Gate S/A/B 状态机未越级（Roundabout Gate S FAIL 时 Gate A/B 保持 BLOCKED）。

## 7. 唯一 Go/No-Go

**Merge：`DEVELOPMENT_GO_TO_NEW_HOLDOUT`。**
**Roundabout：`INCONCLUSIVE_MODEL_INSUFFICIENT`（探索性，不改变主判定）。**

唯一含义：授权**设计**一份全新的、未参与开发的确认性 holdout（Stage 2）。**不授权**实现 router、不授权把开发数据当作确认证据。

## 8. 允许主张 / 禁止主张 / 下一步

**允许（开发性、有条件）：**
- 在现有开发数据上观察到「事实后缀的事件特异价值随反事实响应强度存在条件异质性/总体衰减趋势」；
- 当前模型在 merge 场景具备最低逐车响应预测与事实后缀利用能力（Gate S 通过）。

**禁止：**
- 禁止声称 `GATE_A_CONFIRMED` 或「现象已被证实」；
- 禁止把 `R` 与证据价值的关系表述为因果作用；
- 禁止「双刃剑 / dual-edged evidence」表述（Gate B 不成立）；
- 禁止逐箱单调衰减的表述；
- 禁止在确认性 Gate A 通过前开发 router；
- 禁止在新 holdout 上调参、选阈值、挑案例或修模型。

**下一步：** 依据决策树分支 A，编写 `plans/2026-09-07_AC-Twin_Stage2新确认性Holdout实施计划_v1.0.md`（仅设计，不采集、不训练），完成后停止并等待用户审批。
