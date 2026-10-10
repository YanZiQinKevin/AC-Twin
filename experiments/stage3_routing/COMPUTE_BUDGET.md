# AC-Twin Stage 3 COMPUTE_BUDGET

> 状态：`ENGINEERING_ESTIMATE`。多 seed 长训练前由完整 epoch benchmark 生成；
> 须用户单独确认预算后，才由用户手动启动任何正式/长任务。
> 设备：`cuda`。时间单位 = GPU 小时；显存单位 = 峰值显存(VRAM)。

## 0. 估算来源与外推

- benchmark 来源：full-epoch response benchmark (full train epoch at widest config)
- benchmark 设备 / 日期：cuda / 2026-09-09
- response 实测 epoch_seconds（1400 episode）：36.970 s，峰值显存(VRAM)：0.146 GB
- router 实测 epoch_seconds：26.996 s，峰值显存(VRAM)：0.149 GB
- 外推方法：epoch_seconds scales linearly from the benchmark episode count to the full train split; per-seed time = full_epoch_seconds x assumed_epochs; Stage 3A uses the response benchmark, cross-fit and Stage 3B use the router benchmark; the widest config gives an upper bound for all configs.
- 外推至 train=1400 episode：response 单 seed（40 epoch）=1479 s，router 单 seed=1080 s
- 不确定范围：所有GPU 小时 ×[0.5, 2.0]

## 1. 每个配置、每个 seed 的预计时长与显存

| hidden_dim | resp_feat_dim | λ_cls | 单 seed 时长 | 峰值显存(VRAM) |
|---:|---:|---:|---:|---:|
| 64 | 32 | 0.1 | 0.41 h | 0.15 GB |
| 64 | 32 | 0.25 | 0.41 h | 0.15 GB |
| 64 | 64 | 0.1 | 0.41 h | 0.15 GB |
| 64 | 64 | 0.25 | 0.41 h | 0.15 GB |
| 96 | 32 | 0.1 | 0.41 h | 0.15 GB |
| 96 | 32 | 0.25 | 0.41 h | 0.15 GB |
| 96 | 64 | 0.1 | 0.41 h | 0.15 GB |
| 96 | 64 | 0.25 | 0.41 h | 0.15 GB |

## 2. Stage 3A 分项预算（response 计时）

### 2a. 单 seed 有界筛选

- 配置数 / 训练次数：8 / 8
- GPU 小时：3.29 h (range 1.64–6.57 h)
- 存储：4.00 GB

### 2b. 入围配置补五 seed

- 入围配置 / 训练次数：2 / 10
- GPU 小时：4.11 h (range 2.05–8.22 h)
- 存储：5.00 GB

**Stage 3A 合计：18 次训练，7.39 h (range 3.70–14.79 h)，存储 9.00 GB。**

## 3. cross-fit 额外成本（router 计时）

- 训练次数（Direct+Dual 各五组）：10
- GPU 小时：3.00 h (range 1.50–6.00 h)
- 存储：5.00 GB

## 4. Stage 3B 分项预算（router 计时，仅登记，未启动）

### 4a. tier-1 主条件（Uniform/Direct/Dual + Oracle reference）

- 条件：uniform_suffix, direct, dual_router, oracle_reference
- 训练次数（4 条件 × 5 seed-pair）：20
- GPU 小时：6.00 h (range 3.00–12.00 h)

### 4b. tier-2 诊断条件（开发门通过后补）

- 条件：no_suffix, shuffled_suffix, response_head_no_routing, random_gate, static_gate, whole_suffix_soft_gate
- 训练次数（6 条件 × 5 seed-pair）：30
- GPU 小时：9.00 h (range 4.50–18.00 h)

**Stage 3B 合计：50 次训练，15.00 h (range 7.50–30.00 h)，存储 25.00 GB。**

## 5. 正式集合（冻结前估算，不自动启动）

- 说明：PRE-LOCK ESTIMATE: formal collection, full frozen inference and the 10,000x bootstrap are frozen only after Gate R_dev + Gate T_dev + Gate C_dev + power all pass; the single unlock is not estimated.
- 采集：gatec_merge_id=1600 ep，gatec_merge_unseen_duration=800 ep，GPU 小时：0.03 h (range 0.02–0.07 h)
- 全量冻结推理：2400 ep，GPU 小时：0.01 h (range 0.01–0.03 h)
- 10,000× bootstrap（8 端点）：GPU 小时：0.01 h (range 0.01–0.02 h)
- 正式集合小计：0.06 h (range 0.03–0.12 h)

## 6. 失败、重试与断点续跑策略

- 每 epoch 写断点：True
- 断点续跑：add --resume to the same command; the run_fingerprint guards against resuming the wrong run
- 失败退出：a failed run writes a failure summary and exits non-zero; re-run with --resume
- 最大重试：3

## 7. 总计

- 总训练次数：78
- 训练 GPU 小时：25.39 h (range 12.70–50.78 h)
- 含正式集合的 GPU 小时：25.45 h (range 12.72–50.90 h)
- 典型总时长：25.45 h
- 最坏总时长：50.90 h
- 存储需求：39.00 GB
- 峰值显存(VRAM)（单任务峰值）：0.15 GB
