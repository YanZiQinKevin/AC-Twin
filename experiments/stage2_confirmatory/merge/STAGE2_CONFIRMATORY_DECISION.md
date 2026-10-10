# Stage 2 确认性 Holdout — 机械裁判决策文档

> 计划编号：`AC-TWIN-RFER-STAGE2-HOLDOUT-2026-09-07-V1`  
> claim_boundary：`confirmatory_holdout_single_unblinding; nonmonotonic_response_regime_heterogeneity_only; not_u_shape_not_causal_not_mechanism; no_dual_edged_wording_unless_gate_b_supported`

- lock_sha256：`8939478b1f57619d2c8cd37b48dcdd571c852accf9a968b89ebda6a58be4adab`
- Gate Q 通过：`True`
- output_signature：`fa95313832d0c4b3591fd8c65f16dac78802447d62dd773d979a587f6466aaf0`

## Gate Q 明细

- `lock_artifact_hashes` = True
- `expected_lock_sha256_match` = True
- `seed_schedule_valid_zero_overlap` = True
- `accepted_pairs_matches` = True
- `prefix_consistency` = True
- `identity_replay` = True
- `no_training` = True
- `inference_no_training` = True
- `predictions_hash_verified` = True
- `manifests_hash_verified` = True
- `data_output_hash_verified` = True

## 估计规模

- accepted episodes = 1200；有效非自车 agent-time = 96000
- 各正响应箱独立 episode 数 = {'positive_low': 788, 'positive_mid': 751, 'positive_high': 513}

## 四箱 V_evt

| 响应箱 | V_evt | agent-time | episodes |
|---|---:|---:|---:|
| zero | 0.363598 | 80128 | 1200 |
| positive_low | 0.084139 | 5238 | 788 |
| positive_mid | -0.066092 | 5391 | 751 |
| positive_high | 0.206375 | 5243 | 513 |

## 主对比（裁判 CI 仅落在三个光滑对比上）

| 量 | 点估计 [95% CI] | 0.05 筛选 |
|---|---:|:---:|
| `C_presence` | 0.2258 [0.1932, 0.2584] | True |
| `C_high_low` | 0.1222 [0.0312, 0.2180] | True |
| `C_high_mid` | 0.2725 [0.2234, 0.3263] | True |
| `V_zero` | 0.3636 [0.3378, 0.3885] | — |
| `V_high` | 0.2064 [0.1846, 0.2278] | — |
| `V_positive_all` | 0.1378 [0.1161, 0.1588] | — |

- `C_rebound`（无裁判 CI 的摘要点估计）= 0.1222

## 稳健性（均不得反转为负）

- leave_one_model_seed_out: {'c_presence': True, 'c_high_low': True, 'c_high_mid': True}
- leave_one_action_type_out: {'c_presence': True, 'c_high_low': True, 'c_high_mid': True}
- leave_one_episode_out: {'c_presence': True, 'c_high_low': True, 'c_high_mid': True}
- high_error_deletion: {'c_presence': True, 'c_high_low': True, 'c_high_mid': True}
- 合并 all_positive = True

## Alternate donor 敏感性（点估计）

- donor 20272917: c_presence=0.2268, c_high_low=0.1273, c_high_mid=0.2737
- donor 20272927: c_presence=0.2256, c_high_low=0.1237, c_high_mid=0.2749
- donor 20272937: c_presence=0.2258, c_high_low=0.1227, c_high_mid=0.2730
- donor 20272947: c_presence=0.2252, c_high_low=0.1238, c_high_mid=0.2760

## Gate S* / Gate B

- S1*=True S2*=True S3*=True -> pass=True
- Gate B = `GATE_B_NOT_SUPPORTED`

## 次分析（secondary_only，无 Gate A 方向要求）

- positive-only Spearman = 0.0607 [0.0233, 0.0972]
- positive-only trend log1p(R) = -0.0403 [-0.0845, 0.0043]
- 原 V_low - V_high = 0.1070 [0.0719, 0.1409]

## 唯一机械状态

**`GATE_A_CONFIRMED_NONMONOTONIC`**
