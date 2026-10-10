# Stage 3A Plus 算力预算（COMPUTE_BUDGET）

> 计划：`AC-TWIN-RFER-STAGE3A-PLUS-2026-09-09-V0.1` Task 11。
> 基准测量：2026-09-09，本机 CUDA（`code/benchmark_stage3a_plus_budget.py`，12 episode / 1 epoch 实测）。
> 区间约定：除注明外为中枢估计的 ±25%。

## 1. 实测单元速率

| 单元 | 测量 | 速率 |
|---|---|---|
| 训练（h96/r32，含 val 前向） | 12 ep × 1 epoch = 1.185 s（含初始化开销，偏保守） | 全量 epoch ≈ 83.9 s（train 1400 + val 300 episode） |
| 推理 | 50 ep × 1 seed = 0.034 s | 0.00069 s/seed/episode |
| Gate bootstrap（decompose） | E=400、50 resamples = 7.94 s | 0.1589 s/resample@E=400，随 resamples 与 E 近似线性 |
| 采集 attempt（merge-v0 双分支真实 rollout） | 3 attempts | 0.227 s/attempt |

经验锚点：同一训练配置在 `stage3a_retrain_v1` 实际早停于 9–36 epoch（冻结配置 h96_r32_l0.1 为 20 epoch）；无早停上界为 40 epoch。

## 2. 分阶段预算

| 阶段 | 构成 | 中枢估计 | ±25% 区间 | 无早停上界 |
|---|---|---|---|---|
| SOFTWARE_TESTED | 目标测试 + 全量回归 | 4 min | [3, 5] min | — |
| FORENSIC_DIAGNOSTIC | 5 压力 seed 训练（≈20 ep each）+ 校准候选 + A/B 分解 | 2.6 h | [2.0, 3.3] h | 4.9 h |
| DEV_ROUND_0 | 4 校准候选筛选 + evaluate-dev（A/B，2000 resamples + 分层） | 0.4 h | [0.3, 0.5] h | — |
| DEV_ROUND_1（仅 Round 0 未过） | 3 候选 × 5 seed + finalist 5 压力 seed + evaluate-dev | 9.5 h | [7.1, 11.9] h | 19.0 h |
| DEV_ROUND_2（仅 Round 1 未过） | 6 组合 × 5 seed + finalist 5 压力 seed + evaluate-dev | 16.5 h | [12.4, 20.7] h | 33.0 h |
| QUALIFICATION_POWER | 十 seed A/B 推理 + 2000 MC × 7 候选 × 2 集 + rank 池 bootstrap | 1.0 h | [0.75, 1.25] h | — |
| 采集（n ≤ 2000） | 上限 n×15 = 30,000 attempts | 1.9 h | [1.4, 2.4] h | 实际接受率高时 ≈ 15 min |
| 冻结推理 | 10 seed × n ≤ 2000 | < 1 min | — | — |
| 一次性裁判 | 10,000 resamples decompose（E ≤ 2000）+ 分层 bootstrap | 4.4 h | [3.3, 5.5] h | — |

**最长单任务**：DEV_ROUND_2 无早停上界 33.0 h（+25% = 41.3 h）≪ 72 h 硬停线 ✓。最差路径总墙钟（Round 0/1 均未过）≈ 35 h（中枢）/ ≈ 66 h（上界）。

## 3. 磁盘预算

| 项 | 上界 |
|---|---|
| checkpoint（≤ 60 次训练 × .pt/.ckpt/state） | 0.5 GB |
| 资格集数据（n=2000 npz + ledger） | 0.2 GB |
| predictions ×10 seed + 报告/诊断/功效产物 | 0.3 GB |
| **合计上界** | **1.0 GB** |

1.25× 上界 = 1.25 GB；当前 D: 盘可用 **552 GB** ✓（硬停条件 5 满足）。

## 4. 结论

- GPU/CUDA 可用 ✓；磁盘 552 GB ≥ 1.25 GB ✓；最长单任务 ETA（上界 41.3 h）< 72 h ✓。
- 三个硬停前置全部满足，允许进入 QUALIFICATION_POWER_FROZEN 及之后阶段。
