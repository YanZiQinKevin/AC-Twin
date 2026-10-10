# Stage 4D-D2R HANDOFF STATE（恢复/交接锚点）

> 计划编号：`AC-TWIN-RFER-STAGE4D-D2R-RISK-GATED-IDLE-2026-09-24-V0.1`
> 计划文件：`plans/2026-09-24_AC-Twin_Stage4D-D2R同平台风险回退决策开发实施计划_v0.1.md`
> 批准：2026-09-24 用户"同意执行"（§9 全部四项决定），状态 `APPROVED_FOR_AUTONOMOUS_IMPLEMENTATION`。
> 上位计划：`AC-TWIN-RFER-2026-09-07-V1`；前置终态 D2 v0.2 `STAGE4D_EXTERNAL_DEV_NOT_READY` 永久有效不改写。
> 本文件按 AGENTS.md 风险分级原则在阶段边界更新。执行只创建 `experiments/stage4_reliability/stage4d_d2r/` 下产物及新代码/测试；D2 v0.2 全部文件只读。

## 当前状态

- **终态（2026-09-24，本计划已收口）：`D2R_SCREEN_UNDERPOWERED`。** §5 功效预检（validation 整组配对 bootstrap seed `2026122402` 估协方差 + 10000 MC + n∈{1500,2000,2500} + 收缩 {1.0,0.75,0.5}）完成：绑定条件为 **`i_idle`**（D2R 策略对 `ALWAYS_IDLE` 的改善），其 validation 点估计 `0.0541` 仅勉强高于 5% 实质线，75% 收缩（→`0.0406<0.05`）下 §7 完整 AND Gate 联合功效 n=1500/2000/2500 分别 `0.2266`/`0.2098`/`0.2002`，全远低于 0.80；且 **n 增大不改善**（短缺点在点估计而非 CI 宽度，正合计划"不得以样本变大能救替代"）。即使 100% 收缩联合功效也仅 ~0.57–0.59。次级弱链 `reactive_frac`（validation 0.272，75% 收缩边际功效 ~0.63–0.67）。**判据：75% 收缩联合功效≥0.80 选最小 n；2500 仍不足→`D2R_SCREEN_UNDERPOWERED`，不采集。** 产物 `power/screen_power_report.json`（create-once，含全部 marginal/joint 功效、theta_hat、se_reference、bindings）。
- **本终态的法律后果（计划 §7 终态优先级 + 长期授权）**：未创建/消耗任何新 screen、manifest、lock、smoke 或采集；本计划无 confirm。**授权随终态结束。** 任何失败状态不准原地换 threshold/coverage、换 seed、扩大 screen、放宽 Gate 或切换平台再判一次。`phase0/` 与 `power/` 产物永久保留不得改写；D2 v0.2 文件保持原样。
- **Task 1 + §7 Gate + §5 功效已建成（本计划全部已建产物）**：`code/ac_twin/stage4d_d2r_policy.py`（纯 NumPy 固定阈值回退规则+对照）、`code/ac_twin/stage4d_d2r_statistics.py`（整组 paired bootstrap + Phase 0 量向量 + **§7 screen 40 槽量向量 + 唯一裁判 Gate + 终态优先级 + 向量化 power AND-chain**）、`code/ac_twin/stage4d_d2r_power.py`（§5 功效：validation 协方差→MVN→选 n）、`code/stage4d_d2r_phase0.py`（Phase 0 只读编排）、`code/drive_stage4d_d2r_power.py`（§5 驱动）。测试：`tests/test_stage4d_d2r_policy.py`（**9/9**）+ `tests/test_stage4d_d2r_screen.py`（**8/8**，screen Gate 终态优先级/power 一致性/无效 vs 方法失败）。**Task 2（protocol manifest）、Task 3（collector）、Task 4（runner）因功效终态不再建设。**
- **Phase 0 结果（`D2R_PHASE0_FEASIBLE`，Go/No-Go 8 条件全真，标记 `ADAPTIVE_DEVELOPMENT_ONLY`）**：validation 全 1500/1500 valid、train 5000/5000 valid；tau_D=`0.113060`；最强标准回退=`MARGIN_ONLY`；H=300/L=1200（各≥200）；diff_H=`+0.0774`、diff_L=`+0.0072`。item5：mean_regret d2r=`0.1003`/idle=`0.1060`/base=`0.1158`/最强回退=`0.1144`；i_idle=`0.0541`（CI 下界 0.024）、i_base=`0.1337`、i_standard=`0.1230`；frac_changed=`0.198`、frac_non_idle=`0.726`、frac_above=`0.200`。item1 独立重建逐位复现 D2 train 数（base=0.1141/idle=0.1082/random=0.1139）——无技术错误。item3 为预期模型保真度形态（预测低估 0.2 效率项），非投影/mask/动作槽异常，未触发停止。产物 `phase0/{diagnostic.npz, report.json, final_policy.json}`（create-once）。
- **授权登记已完成（2026-09-24）；终态登记已完成（2026-09-24）**：计划头部/§0 状态 → `APPROVED_FOR_AUTONOMOUS_IMPLEMENTATION`；`AGENTS.md` 历史授权清单第 (11) 项 + Stage 4D-D2R 专项条目（已更新为 **now COMPLETE**，终态 `D2R_SCREEN_UNDERPOWERED`）；`00_READ_FIRST_AC_TWIN.md` §6、`01_AC_TWIN_DOCUMENT_STATUS_INDEX.md` D2R 行（待随终态更新为收口）。
- **§2.1 来源核验全部 MATCH（2026-09-24）**：9 项表内 SHA256 + 五 Direct checkpoint SHA256（9101/9202/9303/9404/9505）全部一致。任何后续漂移须先停。
- **复用契约已锁定（只读，不改 D2 代码/模型/种子/阈值/校准）**：Phase 0 重用冻结 `development/{train,validation}_predictions.npz` 与 `{train,validation}_scores.npz`，经只读函数链重建逐组量。**不重跑 torch 预测栈**（预测已冻结），score_models 可 CPU。§5 功效从 `phase0/diagnostic.npz`（冻结逐组数组）+ `final_policy.json`（冻结阈值）+ D2 `final_selection.json`（interaction_threshold）+ 原始 validation role（reactive）只读装配，不重跑 torch。
- **唯一候选规则（§3，固定）**：`a_D2R(g)=a0(g) if r_D(g)<=tau_D else IDLE(=(1,3),IDLE_PLAN_INDEX=3)`；`tau_D`=validation 上 `(r_D,scenario_id)` 升序第 `ceil(0.80*n)` 个分数；分数==阈值执行 a0；新数据只用冻结 tau_D 不重排队。主对照：ALWAYS_IDLE、ALWAYS_BASE、validation 冻结最强标准回退。次分析：确定性随机 20% 回退 + 逐 seed 阈值。

## 冻结决定（本计划）

- 新 salt：`AC-TWIN-D2R-V0.1|`；随机回退 salt：`AC-TWIN-D2R-RANDOM-2026-09-24|`。
- 唯一环境入口：`powershell.exe -NoProfile -ExecutionPolicy Bypass -File code\run_stage4d_env.ps1 sim code/run_stage4d_d2r.py run-all`；禁止裸 python/pip/conda run。
- Phase 0 bootstrap seed `2026122401`（2000 次）；功效 Monte Carlo seed `2026122402`（10000 次，候选 n∈{1500,2000,2500}，75% 收缩联合功效≥0.80 为唯一选 n 判据）；screen 裁判 bootstrap seed `2026122403`（10000 次，80% CI）。
- screen 规模 n∈{1500,2000,2500}；工程 smoke 独立候选池=排序后前 160 个 ID（目标 8 accepted）；新 screen 可用候选≥5×selected_n（扣除 160 后），原 confirm 池须仍≥8000 完整未触碰 ID。
- 旧 D2 screen 4,000 组逐组产物禁止加载；原 confirm 池保持未触碰。

## 已完成产物

- 授权登记（计划/`AGENTS.md`/`00_READ_FIRST`/文档索引）；终态登记（`AGENTS.md` D2R 条目 → now COMPLETE）。
- `phase0/{diagnostic.npz, report.json, final_policy.json}`（create-once，`D2R_PHASE0_FEASIBLE`）。
- `power/screen_power_report.json`（create-once，**终态 `D2R_SCREEN_UNDERPOWERED`**；含 n∈{1500,2000,2500} × 收缩{1.0,0.75,0.5} 全部 marginal/joint 功效、theta_hat、se_reference、binding=`i_idle`、bindings）。
- 代码：`stage4d_d2r_policy.py`、`stage4d_d2r_statistics.py`（Phase0+§7 Gate）、`stage4d_d2r_power.py`、`stage4d_d2r_phase0.py`、`drive_stage4d_d2r_power.py`、探针 `stage4d_d2r_probe_schemas.py`/`stage4d_d2r_probe_diagnostic.py`。
- 测试：`tests/test_stage4d_d2r_policy.py`（9/9）、`tests/test_stage4d_d2r_screen.py`（8/8）。

## 最近验证（2026-09-24）

- §2.1 九项来源 SHA256 全 MATCH；五 Direct checkpoint SHA256 全 MATCH。
- §5 功效复算自洽：i_idle SE=`0.0230`、点 `0.0541` → 80% CI 下界 `0.0246`（与 Phase 0 `0.024` 一致）；100% 收缩 P(i_idle≥0.05)=0.57（与报告 `0.5741` 一致）。
- 测试合计 17/17 通过（policy/Phase0 9 + screen/power 8）。

## 阻塞项

- 无（本计划已达终态收口，非阻塞）。

## 消耗登记

- 一次性资源（本计划）：**均未创建**（无 smoke/screen lock、无 screen_scores、无新 screen manifest、无裁判；功效与 Phase 0 均为只读开发预检，不消耗一次性采集资源）。D2 旧资源（pilot、development_screen）已消耗且不属于本计划。原 confirm 池保持完整未触碰。
