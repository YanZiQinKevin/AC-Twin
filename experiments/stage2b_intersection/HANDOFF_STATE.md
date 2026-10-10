# Stage 2B `intersection-v0` — HANDOFF / 恢复状态

> 计划编号：`AC-TWIN-RFER-STAGE2B-INTERSECTION-2026-09-14-V0.3`
> 最近更新：2026-09-15（pilot 终止后）
> 当前终态：**`INCONCLUSIVE_INTERSECTION_RESPONSE_SUPPORT`**

## 1. 终态裁定

P1 响应支持 pilot 两 profile 均完成采集且技术 QA 全部通过，但均未满足计划 §5.2 的
合格条件，`select-profile` 按机械规则于 2026-09-15 写入 create-once
`experiments/stage2b_intersection/profile_selection.json`：

| 条件（§5.2） | Profile A（10车） | Profile B（15车） |
|---|---:|---:|
| 600 accepted、六层各100 | ✅ 600/819 attempts（73.3%） | ✅ 600/855 attempts（70.2%） |
| prefix/identity QA | ✅（最大绝对差均为 0） | ✅ |
| ≥40% episode 含 R>1e-4 paired-valid 非自车 agent-time | ❌ **36.0%** | ✅ 43.0% |
| ≥300 独立 episode 贡献正响应 | ❌ **216** | ❌ **258** |
| paired-valid 非自车中位数 ≥3 | ✅ 5.0 | ✅ 6.0 |
| 三正响应临时三分位各 ≥100 episode | ✅ 207/206/154 | ✅ 250/240/173 |

**A 失败于条件 3+4，B 失败于条件 4 → 两 profile 均不合格 →
`INCONCLUSIVE_INTERSECTION_RESPONSE_SUPPORT`（§10 停止规则第 1 条）。**

development 采集、训练、Gate S、功效、confirm 与解封均未启动，也不得在本计划下启动。
该终态只表示固定参与者交叉口分布在本协议下的响应支持不足，**不能**解释为现象被否证，
也不影响 Stage 2 Merge 既有结论。

## 2. 保留产物（均不得删改）

- P0 smoke：`experiments/stage2b_intersection/smoke/smoke_receipt.json`（`P0_SMOKE_PASS`）
- P0 冻结 lookup：`experiments/stage2b_intersection/movement_lookup.json`
  （sha256=`18664d0acd2780449f3261a0fd56139945b60e2f5af4721ec6bf5b5a5af638eb`）
- pilot A/B 数据与回执：`data/object_centric_stage/stage2b_intersection_pilot_{a,b}/`
  （NPZ、rejection ledger、collection receipt）；manifest 与 schedule：
  `experiments/stage2b_intersection/pilot_{a,b}/`
- profile 选择回执：`experiments/stage2b_intersection/profile_selection.json`
  （`INCONCLUSIVE_INTERSECTION_RESPONSE_SUPPORT`，含两 profile 全部支持度指标）
- 软件与测试：`code/ac_twin/stage2b_protocol.py`、`stage2b_data.py`、
  `code/collect_stage2b_intersection.py`、`code/run_stage2b_intersection.py`；
  `tests/test_stage2b_*.py`（64 项，全部通过）
- development 配置未被使用（`initial_vehicle_count` 从未被注入）

## 3. 过程记录（2026-09-15）

- 首次 pilot A 启动时 manifest 在写盘前拦截种子冲突（pilot_a/development/confirm 首批
  集合种子与 62,255 个历史种子存在冲突）；按计划 §3.2 整体更换为
  pilot_a=2026091501、development=2026091504、confirm=2026091563（smoke=2026091450、
  pilot_b=2026091452 不变），并由阻断级测试
  `test_frozen_collection_seeds_disjoint_from_live_history_and_each_other` 钉住。
- 采集中断过一次（Ctrl+C，发生在首个 checkpoint 前），重跑 `--resume` 确定性重放，无损失。

## 4. 如需继续（不属于本计划）

任何挽救/扩展动作都必须另立版本并由用户审批，例如（仅为可能方向，非建议执行）：
响应支持阈值与 profile 集合的修订、允许动态入流的新配对数据协议、或不同的
动作/t_star/horizon 冻结值。**不得**在本计划内调阈值、换 profile 或复用 pilot 数据
进入 train/validation/confirm。

## 5. 既有登记项（不变）

- `tests/test_stage2_protocol.py` 5 项 lock 测试自 Stage 4A 数据落地后对活 `data/`
  重扫失败（stage4 dev 数据 seed 与 Stage 2 候选 schedule 偶然重叠）；与 Stage 2 已
  消耗的正式 lock 无关；冻结代码不得修改；是否修订测试由用户决定。
