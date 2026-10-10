# AC-Twin Stage 2 确认性 Holdout — 机器可执行协议（冻结）

> **文档状态：PROTOCOL_LOCKED_READY_FOR_MANUAL_COLLECTION**
> 计划编号：`AC-TWIN-RFER-STAGE2-HOLDOUT-2026-09-07-V1`；上位研究计划：`AC-TWIN-RFER-2026-09-07-V1` v1.1
> 主场景：`merge-v0`；性质：实现 + 测试 + 冻结 + 用户手动采集 + 冻结模型推理 + 一次解封分析。
> **本文件只写协议与机械命令，不写任何结果。** 所有命令从项目根目录执行。

## 0. 冻结边界（不可逾越）

- 不训练、不微调、不选择 checkpoint、不在新数据上拟合 normalization/bins。
- 确认性分析只执行一次；二次解封、改 bootstrap seed、改 CI、换主对比均被机械禁止。
- Gate A 通过前不实现 router；Gate B 不支持则禁止“双刃剑”表述。
- 确认命题是**非单调响应状态异质性**，不是 U 形连续函数、不是因果机制。

## 1. 冻结输入（hash 全部登记在 protocol lock）

- 15 个 Stage 1 Merge checkpoint：M0/M1/M2 × seeds {11,22,33,44,55}（`experiments/stage1_evidence_value/merge/`）。
- `normalization.npz`（冻结 feature mean/std 与 target mean/std）、`response_bins.json`（`zero_tolerance=0.0001`、`t1=0.24843344965718578`、`t2=1.898992796585611`）。
- Stage 1 Gate S report（`experiments/stage1_evidence_value/merge/report.json`）。
- 冻结 Gate S* 参数 `frozen_gate_s_star_params.npz`：两个 MAE 基线系数（`beta_action_horizon` (3,)、`beta_action_horizon_distance` (4,)）与表示探针（`probe_coefficients` (65,6)、`probe_coefficients_shuffled` (65,6)、`probe_scene_mean` (6,)），全部只在 **Stage 1 train split** 上拟合并封存。
- 代码 hash 注册：runner、collector、`stage2_protocol`、`stage2_inference`、`stage2_statistics`、`stage1_model`、`stage1_data`、`stage1_statistics`、`stage1_plus_statistics`、`checkpointing`、`object_centric`、`progress`。
- 候选 seed schedule：18,000 个，派生自 `SeedSequence(20270999).spawn(3)[0]`，用 `choice(2147483646, size, replace=False)+1` 无放回抽取（**构造上保证内部唯一且落在 [1, 2147483646]**）。

## 2. 种子与随机流（冻结）

- `collection_seed = 20270999`（正式）；`SeedSequence(20270999).spawn(3)` 派生 `seed_schedule_rng` / `action_rng` / `stratum_rng`。
- episode seed 取自预生成 schedule 的第 `attempt_index` 项（不由运行中 RNG 抽取），保证 resume 等价。
- 历史 episode seed 扫描排除 Stage 2 自身产物（任何路径含 `stage2_` 的 npz），因此 smoke 采集不会污染正式 lock，unlock 时 Gate Q 重扫也不会误报重叠。
- 实测：`20270999` 的 18,000 schedule 与 12,000 个历史 seed **零重叠**；smoke（`collection_seed=20271999`）256 个候选亦零重叠。
- primary derangement seed = `20272907`；四个 alternate donor seed = `20272917, 20272927, 20272937, 20272947`。derangement 直接复用 `stage1_data.build_derangement` 的四级回退。
- primary bootstrap seed = `20270908`，10,000 次 episode 重采样，95% percentile CI。

## 3. 环境（实测记录）

- Python 3.13.5；NumPy 2.1.3；PyTorch 2.14.0+cu132；highway-env 1.10.2；gymnasium 1.2.3。
- GPU：NVIDIA GeForce RTX 5070 Laptop GPU（CUDA 可用）。推理显存需求 < 1 GB（模型仅 145,066 参数，batch 128）。
- Windows 11；PowerShell。所有 Python 入口已在 `import numpy/torch` 前设置 `MKL_THREADING_LAYER=SEQUENTIAL` 与 `KMP_DUPLICATE_LIB_OK=TRUE` 以规避 OMP Error #15。

## 4. 测试计数（实测）

- Stage 2 专项（5 个文件）：`tests/test_stage2_protocol.py`、`test_stage2_collector.py`、`test_stage2_inference.py`、`test_stage2_statistics.py`、`test_stage2_runner.py` → **83 通过**。
- 完整回归 `python -m pytest -q` → **395 通过，0 失败**。
- 关键复核：Task 1 在 development 数据上复算新五项联合功效，n=1200、10,000 次、75% 情景 joint = **0.918**（目标 0.91±0.02），empirical=0.994，50%=0.604；最弱项为 `C_high_low`。点估计复现 `C_presence=0.1976`、`C_high_low=0.181`、`V_zero=0.339`、`V_high=0.213`，与 Stage 1 Plus 一致。

## 5. 预计耗时（基于 smoke 实测外推）

- create-lock（含真实冻结探针/基线拟合）：约 10–20 s。
- 正式采集 1,200 episode：smoke 12 episode/18 attempt 用 6 s；按 ~67% 接受率与每个 attempt 双分支 rollout 外推，正式约 **15–30 min**（取决于碰撞/提前终止拒绝率）。支持断点 resume。
- 正式推理（15 主预测 + 5 M1 swap + 20 alternate donor + 2 探针摘要，CUDA）：smoke 12 episode CPU 用 2.9 s；正式 1,200 episode 用 `--device cuda` 预计 **1–3 min**。
- unlock-and-analyze（10,000 次 bootstrap × 各统计量）：预计 **1–5 min**。

## 6. 用户手动执行命令（依次，严格顺序）

### 6.1 只读核验正式 Protocol Lock

```powershell
python code/run_stage2_confirmatory.py verify-lock --config config/stage2_holdout_merge_confirm.json
```

完成判据：已有 `protocol_lock.json` 未被覆盖；输出打印历史 seed 数、candidate_schedule=18000、zero_overlap=True、checkpoints_ok=15 与 `PROTOCOL_LOCK_VERIFIED`。

### 6.2 正式采集（1,200 episode）

```powershell
python code/collect_stage2_holdout.py --config config/stage2_holdout_merge_confirm.json --resume
```

完成判据：`accepted=1200`；最终 NPZ 与 `collection_receipt.json` 原子发布；`rejection_ledger.jsonl` 存在且逐 attempt 记录；无 Gate 数值输出。**中断后重复同一命令即从 checkpoint 继续**（`--resume` 为幂等：无 checkpoint 时从头开始，有则从断点续跑；最终 NPZ 已存在则只读核验不覆盖）。

### 6.3 冻结模型推理（CUDA）

```powershell
python code/run_stage2_confirmatory.py infer --config config/stage2_holdout_merge_confirm.json --device cuda
```

完成判据：15 主预测 + 5 M1 swap + 20 alternate donor 预测 + 5 manifest + 2 探针摘要 + `prediction_receipt.json` 齐全；stdout 只显示预测/清单计数，**不含** V_evt、CI 或任何 Gate 字样。已有有效 receipt 时只验证不覆盖。

### 6.4 一次解封与确认性分析（执行前确认愿意正式解封）

```powershell
$stage2Lock = Get-Content 'experiments\stage2_confirmatory\merge\protocol_lock.json' -Raw -Encoding UTF8 | ConvertFrom-Json
python code/run_stage2_confirmatory.py unlock-and-analyze --config config/stage2_holdout_merge_confirm.json --expected-lock-sha256 $stage2Lock.lock_sha256
```

完成判据：仅生成一份 `report.json` 与 `STAGE2_CONFIRMATORY_DECISION.md`，均含 `output_signature`（可从落盘 JSON 删除签名字段后直接复算核对）；**再次运行同一命令被拒绝**（FileExistsError）。

## 7. 允许的机械终态（唯一裁判，见计划 §4）

`INVALID_TECHNICAL` / `INCONCLUSIVE_FROZEN_MODEL_OUT_OF_SAMPLE_INSUFFICIENT` /
`GATE_A_CONFIRMED_NONMONOTONIC` / `GATE_A_NOT_CONFIRMED_RESPONSE_SUPPORT` / `GATE_A_NOT_CONFIRMED`。
裁判优先级固定：Gate Q → Gate S* → 每箱 100 episode → 其余 Gate A 条件 → 全过。

## 8. 实施完成度（退出条件核对）

- [x] Task 1–6 专项与完整回归 0 失败（Stage 2 专项 83，全量 395）。
- [x] smoke 完成且只留在 `experiments/stage2_confirmatory/smoke/` 与 `data/object_centric_stage/stage2_merge_smoke/`；正式目录无 smoke 产物。
- [x] 正式 config、代码、统计、15 checkpoint、normalization、bins、seed schedule、冻结 Gate S* 参数全部登记 hash（见 `experiments/stage2_confirmatory/merge/protocol_lock.json`）。
- [x] 18,000 候选 seed 与历史全集零重叠（verify-lock 复核通过）。
- [x] 本 PROTOCOL.md 写明实际命令、测试计数、耗时估计、恢复方法与完成判据。
- [x] 正式 holdout 尚未被任何统计脚本读取；router 尚未实现。
