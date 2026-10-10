# Stage 4A 开发就绪核对清单（DEV_READINESS_CHECKLIST）

> 计划：`AC-TWIN-RFER-STAGE4A-RISK-DEV-2026-09-11-V0.1`
> 性质：开发筛选（非确认性实验）。本清单记录进入正式采集前的工程就绪状态。

## 软件与回归

- [x] 五个 Stage 4A 测试文件存在并通过：`test_stage4_risk.py`、`test_stage4_statistics.py`、`test_stage4_protocol.py`、`test_stage4a_driver.py`、`test_stage4a_collector.py`（83 passed，`-q -W error`）
- [x] 采集前全量回归一次通过：`python -m pytest -q -W error`（895 passed）
- [x] 工程冒烟 `smoke` 通过，仅写 `smoke/`，状态 `ENGINEERING_SMOKE_ONLY`；白名单构造器拒绝任何 truth 侧关键字（`true_error`/`counterfactual_xy` 等为 TypeError）

## 冻结输入（只读复用，未修改 Stage 2/3 任何锁/结果/runner）

- [x] Stage 3R 五个 full-train seed 对 Full/History 专家 checkpoints（10 个 `.pt`）存在并纳入锁
- [x] `end_to_end_mixture_lc0_lr0` 五个 router checkpoints（5 个 `.pt`）存在并纳入锁
- [x] Stage 3R `scaler.json` 存在并 hash 入锁
- [x] Stage 3A Plus `input_binding_receipt.json` 存在并 hash 入锁
- [x] 上述 15 个 checkpoint + scaler + binding 全部 SHA256 写入 `protocol/protocol_lock.json`

## Seed 与数据角色

- [x] 固定 entropy `20260911` 生成 12,000 个不重复正 int32 候选 seed，范围 `[1, 2^31-2]`
- [x] 预检扫描全部历史**已实现 episode**（NPZ `episode_seed`/`episode_seeds`、receipt、manifest，共 15,012 个）→ 与候选调度**零重叠**
- [x] 扫描范围与既有两条历史扫描（`stage2_protocol.scan_historical_episode_seeds`、`stage3_data.scan_all_episode_seeds`）一致：只统计已实现的 episode seed，不把未被接受的候选 `seed_schedule.npy` 池当作污染（未实现候选不生成 episode、无污染风险）
- [x] 十层（cf_action∈{0..4} × cf_duration∈{1,3}）各 500；每层按 `SHA256(str(seed))` 排序机械划分 300/100/100 → 3,000/1,000/1,000

## 协议锁

- [x] `preflight` 通过：`STAGE4A_PREFLIGHT_PASS`，15 个 checkpoint、模型锁、零重叠全部复验
- [x] `lock-protocol` 一次性写入 `protocol/`：`protocol_lock.json`、`candidate_seed_schedule.npy/.json`、`historical_seed_inventory.json`（write-once，拒绝覆盖）

## 采集约束

- [x] 接受/拒绝只由干预前可知的层配额与仿真有效性决定；`_REASONS` 不含任何 response/effect/error/collision/risk 等 outcome 过滤（已有测试锁定）
- [x] 候选耗尽仍未满十层 → `COLLECTION_INCOMPLETE`（exit 4），不临时追加 seed，回用户评审
- [x] checkpoint/resume 不重复接受（pack/restore 一致性已有测试锁定）

## 仍未做（按 §11 顺序推进）

- [ ] `collect`（5,000 episode，进行中）
- [ ] `infer-dev` → `fit-risk` → `select-candidate` → `adjudicate-dev-screen`
