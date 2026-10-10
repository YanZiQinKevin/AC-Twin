# Stage 4D-D2F HANDOFF STATE — 终态

> 计划编号：`AC-TWIN-RFER-STAGE4D-D2F-POWER-CORRECTION-EXTERNAL-CONFIRM-2026-09-24-V0.1`
> 更新：2026-09-25（**终态**）
> 当前状态：**TERMINAL — `D2F_META_DRIVE_EXTERNAL_CONFIRMED`（2026-09-25）。D2F 授权随终态结束；confirm 集永久消耗不得复用或重判；本路线（MetaDrive/Waymo 十动作 decision-risk 外部效度）不再有 D2G 或同义延续。**

## 终态裁判（2026-09-25，唯一一次 confirm 裁判，create-once 报告）

- **终态 = `D2F_META_DRIVE_EXTERNAL_CONFIRMED`**：独立 confirm 集 n=2,000（旧 D2 confirm 池 15,528 中 salt 排序 primary 10,000 → 7,667 attempts，接受率 26.1% 与 screen 一致；拒绝 100% `no_eligible_t_star`，结局无关）；10,000 次 whole-group paired bootstrap、双侧 95% CI、熵 2026122409。
- **Gate TASK-X-CONFIRM c1–c8 全过**：high_regret 0.489（≥0.15）；7 个 oracle 动作 ≥5%；Spearman 0.726 CI95[0.710,0.741] 下限>0；base 0.0594 ≤ idle 0.1087 且胜 random 0.1156（base-vs-random 改善 0.486 [0.456,0.516]）；reactive 0.263（≥0.20）；strata 0.49–0.51（各 ≥0.10）；完整性/分母/投影/掩码全部有效。
- **Gate RISK-X-CONFIRM r1–r6 全过**：**I_AURC=0.1949 CI95[0.1281,0.2575]**（≥0.05 且下限>0）；**I_REGRET80=0.1532 CI95[0.1070,0.1975]**（≥0.03 且下限>0）；PR−prevalence=0.2117 CI95[0.1858,0.2383]；**5/5 seeds 双正**（≥4）；分母有效；无泄漏/无基线替换/无模型替换。
- **基线披露**：confirm 最强标准基线 = 冻结的 `SEED_MARGIN_STD`（未发生替换）；`model_beats_confirm_strongest_baseline = true`（模型 AURC 点估计与 CI95 上限均优于 confirm 最强基线，可按 D2 10.4 措辞写"优于 confirm 最强标准基线"）。
- 一次性顺序核验：split（真值物理隔离）→ 真值隔离评分（batch=32，与持久化口径一致）→ 32 组确定性逐位一致 → 冻结 → 烧 `confirmation_used.lock`（2026-09-25T15:58:08）→ 唯一裁判。全部 artifact 齐备（见下表）。

## 路线全景（D2 → D2R → D2E → D2F）

- D2 v0.2：screen c5 失败（base>idle），终态 `STAGE4D_EXTERNAL_DEV_NOT_READY`。
- D2R v0.1：功效不足（i_idle 点估计 0.0541 贴线），终态 `D2R_SCREEN_UNDERPOWERED`。
- D2E v0.1：ego 修复成功（`D2E_INTERFACE_REPAIR_QUALIFIED`）但 screen 功效不足（c6 reactive 0.272 贴 0.20 线），终态 `D2E_SCREEN_UNDERPOWERED`。
- **D2F v0.1（最终版本化尝试）：功效口径校正（效应/支撑分离 + 差异收缩 c5）→ screen n=1000 双门过（`D2F_DEV_READY_FOR_CONFIRM_POWER`）→ confirm 功效 n=2000（95% 链 75% 联合 0.895≥0.80）→ confirm 双门全过 → `D2F_META_DRIVE_EXTERNAL_CONFIRMED`。** D2E 诊断的 c6 结构边界在 confirm 实测 reactive=0.263 下不再是约束（screen 0.239、confirm 0.263 均真实高于 0.20 底线，旧功效缺陷是收缩口径错误而非平台边界）。

## 审批与授权轨迹

- 2026-09-24 用户"同意执行"批准 D2F v0.1（初版）；2026-09-25 用户修订计划对齐暂停点并以"确认执行！"再批准。登记：AGENTS.md 授权清单 (13) + D2F 详情条目；状态索引 D2F 行；计划头部状态行。
- **授权随终态 `D2F_META_DRIVE_EXTERNAL_CONFIRMED` 结束（2026-09-25）**。confirm 集（2,000 组 + 全部 confirm 池 attempts）永久消耗，不得复用或重判；D2/D2R/D2E 终态与产物保持只读不变。

## 关键过程记录

- **§3.4 parity 定位（锁前）**：batch=32（持久化口径）实时重推理 vs 持久化 npz 六输出逐位 0.0；端到端 vs D2E 冻结 npz 全 0.0、argmin 全同 → 按原合同 ≤1e-8 字面通过。旧 receipt "永不可能达 1e-8" 解释证伪（真实来源=Path B 误用 batch=4096，批次依赖 kernel 选择）。
- **82 项旧回归红灯逐节点分类（不整体豁免）**：71 highway_env 缺失；6 旧 seed schedule 冲突；3 sklearn 版本警告；2 陈旧快照断言；全部与 D2F 零交集、无真实失败。
- **内存收割事件**：confirm 采集 464/2000 处被 Claude Code 内存压力收割（非失败；checkpoint+ledger 核验完整零撕裂）；用户授权断点续采并无损恢复；**用户指示此后内存收割一律自动断点续采**（持久记忆 `d2f-auto-resume-on-memory-reap`）。
- **熵登记**：2026122406（功效）/ 2026122407（screen 裁判）/ 2026122408（confirm 功效）/ 2026122409（confirm 裁判），四个新鲜熵互不相同。
- **测试**：screen 锁前 36/36；confirm-power 前 57/57；**裁判后 §5.1 复跑 4 文件 57/57 通过（361s，2026-09-25，exit 0）+ 一次性 artifact 全部核验齐备**；从不跑旧全量回归。

## 已完成产物（create-once，永久保留）

| 产物 | 关键内容 |
|---|---|
| `source_binding.json` | 9 项计划 listed SHA 全 match；扩展 checkpoint/scaler/合同/代码 hash |
| `pool_capacity_audit.json` / `pool_capacity_audit_post_smoke.json` | 三分类；admissible screen {1000..2500}、confirm {1000..3000} |
| `power/power_correction_report.json` | `D2F_POWER_CORRECTION_PASS`；screen selected_n=1000 |
| `fresh_id_verification.json` | 13,626 候选零交集 |
| `smoke/` | ENGINEERING_SMOKE_ONLY；44 ID 烧毁；§3.3 全过 |
| `development/scorer_parity_receipt.json` | 组合 parity 全 0.0（create-once 保留，口径更正见上） |
| `screen_manifest.json` / `screen_protocol_lock.json` | n=1000；primary 5,000+reserve 8,582；熵 2026122407 |
| `roles/development_screen/` | screen 1000 组（已消耗） |
| `development/development_screen_report.json` | screen 裁判 `D2F_DEV_READY_FOR_CONFIRM_POWER`（I_AURC 0.2188 CI80[0.171,0.265] 等） |
| `confirm/power_report.json` | `D2F_CONFIRM_POWER_PASS`；**confirm selected_n=2000**（95% 链 75% 联合功效 0.605/0.797/0.895/0.943/0.970 @ n=1000..3000）；熵 2026122408 |
| `confirm_manifest.json` / `confirm_protocol_lock.json` | primary 10,000+reserve 5,528（与 screen 零交集）；95%/10,000/熵 2026122409 |
| `confirm_code_inventory.json` | confirm 期独立代码清单（21 注册文件；screen 清单冻结未重写） |
| `roles/confirm/` | **confirm 2000 组（永久消耗）**；接受率 26.1%；拒绝全为 no_eligible_t_star |
| `confirm/confirm_split/` | inference_inputs / sealed_truth 物理分离 |
| `confirm/confirm_scores.npz(+receipt)` / `confirm_determinism_receipt.json` | 真值隔离评分（batch=32）；32 组逐位一致 |
| `confirm/confirmation_used.lock` | 已烧 2026-09-25T15:58:08 |
| `confirm/confirm_group_quantities.npz(+receipt)` | 2000 组 per-group 量；QA 8/8 真 |
| `confirm/confirm_report.json` | **终态裁判报告 `D2F_META_DRIVE_EXTERNAL_CONFIRMED`**（数值见上） |

## 收尾记录（Task 5，2026-09-25 全部完成）

1. ~~confirm 链~~ ✅ 全部完成至终态 `D2F_META_DRIVE_EXTERNAL_CONFIRMED`。
2. 状态索引 D2F 行 → `COMPLETED_EXPERIMENT`（终态数值已登记）；AGENTS.md D2F 条目 → COMPLETE（随 D2/D2R/D2E 体例）；计划头部状态行 → `COMPLETED_EXPERIMENT`。
3. §5.1 裁判后复跑 D2F 专项 4 文件 **57/57 通过** + 一次性 artifact 核验齐备（split/scores/determinism/confirmation_used.lock/quantities/confirm_report）。
4. 研究日志 `logs/2026-09-25_AC-Twin科研日志.md` 已登记终态与路线收口。
