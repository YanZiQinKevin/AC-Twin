# Stage 1 验尸实验冻结协议

> 文档状态：**COMPLETED / 开发性收口完成**
> 实施计划编号：`AC-TWIN-RFER-STAGE1-IMPL-2026-09-07-V1`
> 收口计划编号：`AC-TWIN-RFER-STAGE1-CLOSURE-2026-09-07-V1`
> 上位计划：`AC-TWIN-RFER-2026-09-07-V1`
> 日期：2026-09-07
> 结果：见 `STAGE1_DEVELOPMENT_GO_NO_GO_2026-09-07.md`（merge=`DEVELOPMENT_GO_TO_NEW_HOLDOUT`；roundabout=`INCONCLUSIVE_MODEL_INSUFFICIENT`）。本文件只冻结协议与命令，实验结论以决策文档为准。
> 边界：本目录只承担 Stage 1 的逐车逐时 effect 预测、M0/M1/M2、公平打乱、指标、Gate S 与开发性 Gate A/B。不实现 router、response head、动态图、attention、SIR、参数溯因、多动作联合训练或新平台。

## 1. 唯一科学问题

在同架构、同参数量、同训练预算下，正确事实后缀相对错误 episode 事实后缀的事件特异价值 `V_evt`，是否随非自车车辆的反事实响应强度 `R` 增大而系统衰减。

Stage 1 只产出三类结论之一：`INCONCLUSIVE_MODEL_INSUFFICIENT`（Gate S 未过）、`DEVELOPMENT_GO_TO_NEW_HOLDOUT`（Gate S 过且开发集 Gate A 方向满足）、`DEVELOPMENT_NO_GO`（Gate S 过但 Gate A 不支持继续）。**本阶段禁止输出 `GATE_A_CONFIRMED`**；最终 Gate A 由后续全新 holdout 判定。

## 2. 三个公平条件

| 条件 | suffix 特征 | suffix mask |
|---|---|---|
| M0 `no_suffix` | 全零 | 全零 |
| M1 `uniform_suffix` | 同 episode 正确前 10 步 | 正确 presence 前 10 步 |
| M2 `shuffled_suffix` | 分层 derangement 的其他 episode 前 10 步 | donor presence 前 10 步 |

三者实例化同一 `PerAgentPairedEffectGRU` 类，hidden size、层数、action embedding、优化器、训练预算完全一致，参数名与参数量逐项相同。条件间唯一差异是 factual suffix 数据来源。模型输出 `B×T×N×2`，保留车辆维到解码端，不含图、attention、slot-specific embedding 或独立 head。

## 3. 数据与泄漏纪律

- 源数据：`data/object_centric_stage/merge_2000/object_twinworld_2000.npz`（主）与 `data/object_centric_stage/roundabout_2000/object_twinworld_2000.npz`（探索）。固定 split 1400/300/300；split 2 仅为 `development_test`。
- 八维特征顺序固定：`presence, is_ego, x, y, vx, vy, cos_heading, sin_heading`。
- 目标 `target_effect = cf_xy − factual_xy`（世界坐标）；`counterfactual_future` 只进入 target/R/mask，**绝不进入模型输入**。
- 有效 mask = history_agent_known AND factual_presence AND counterfactual_presence。
- 归一化与响应分箱阈值**只用 split 0**；anchor 为每 episode 历史末帧自车世界坐标。
- Gate A/B 主分析只用非自车（slot>0），自车仅作 Gate S 管线正控制；统计独立单位为 episode，训练种子先按 agent-time 取均值，不作独立样本。

## 4. Derangement 冻结

- base seed `20260907`；train 逐 epoch 用 `base_seed+epoch`；validation 固定 `20261907`；development_test 固定 `20262907`。
- 分层回退：`(cf_action, cf_duration, factual_actions[:,0], visible_count)` → `(cf_action, cf_duration, visible_count)` → `(cf_action, cf_duration)` → 仅同 split。
- 不变量：无自匹配、donor 不重复、donor seed 不同、不跨 split；每次回退计数写入 manifest QA；validation/test manifest 落盘并参与运行签名。

## 5. Gate 判定顺序

1. **Gate S**（只用 train/validation + 冻结 validation manifest）：S1 优化非退化 + 参数量一致 + 自车正控制；S2 逐车响应幅度可预测；S3 事实后缀通道被使用（预测证据 + 表示证据）。任一失败 → `INCONCLUSIVE_MODEL_INSUFFICIENT`，只允许一次预注册增强。
2. **Gate A**（仅 Gate S 过后）：低响应组 `V_evt` CI>0；`V_evt_low − V_evt_high` CI>0；`Spearman(R, eta_evt)` CI>0；协变量趋势 `log1p(R)` 系数 CI>0；方向一致且非单一种子/动作驱动；shuffled manifest QA 全过。
3. **Gate B**：高响应组绝对 `Delta = L_no − L_suffix < 0`，CI 完全低于 0，独立 episode ≥ 30；否则 `NOT_SUPPORTED`/`UNDERPOWERED`。
4. Gate S 未过时 Gate A/B 自动 `BLOCKED`；episode 聚簇 bootstrap（10,000 次，95% CI）。

## 6. 命令

短时 CPU smoke（已完成，结果见 `smoke/QA.md`）：

```powershell
python code/run_stage1_evidence_value.py --config config/stage1_evidence_value_smoke.json --device cpu
python code/run_stage1_evidence_value.py --config config/stage1_evidence_value_smoke.json --device cpu --resume
```

正式长训练（**由用户手动执行**，先确认 `torch.cuda.is_available()` 为 True）：

```powershell
python code/run_stage1_evidence_value.py --config config/stage1_evidence_value_merge.json --device cuda --resume
python code/run_stage1_evidence_value.py --config config/stage1_evidence_value_roundabout.json --device cuda --resume
```

中断后用同一命令加 `--resume` 继续；不要删除半成品重新开始，除非签名检查明确说明产物不兼容并由用户决定。merge 是主开发判断，roundabout 是探索性复现。

**当前状态：两场景正式训练均已完成并封存（各 15 run）。此后任何重算统计只能用只读推理，禁止再训练：**

```powershell
python code/run_stage1_evidence_value.py --config config/stage1_evidence_value_merge.json --device cuda --reanalyze-only
python code/run_stage1_evidence_value.py --config config/stage1_evidence_value_roundabout.json --device cuda --reanalyze-only
```

`--reanalyze-only` 加载封存 checkpoint 做只读前向推理并重算 Gate S/A/B，报告显示 `training_performed=false`、`checkpoint_modified=false`；不得使用 `--resume` 重训。
