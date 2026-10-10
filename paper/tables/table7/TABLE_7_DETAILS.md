# Stage 5 已发表方法开发比较 — 完整内部审计包（非第二张论文 Comparison 表）

> 计划：`AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3`  
> **终态：`STAGE5_MERGE_COMPARISON_UNDERPOWERED`（2026-09-28，永久，不得重判）**  
> 范围：Stage 4C-R2 开发划分 validation（1,500 组）上的 `ADAPTIVE_DEVELOPMENT_ONLY` 诊断。Stage 5 comparison holdout **从未创建**（§6.2 功效终态禁止采集）。本包任何数字都不是确认比较结果。
>
> **论文表入口：** `TABLE_7.md`。下列 A–E 仅保留完整审计、复现和补充信息，不应拆成多张论文 Comparison 表。

## 主张边界（计划 §11）

- §11.1 的 SOTA 优势句**不可写**：主比较未达到 0.90 联合功效，未生成 holdout，未做裁判。
- 按 §11.3：保留 Stage 4 与 D2F 已确认结论；不写优于已发表方法；本开发比较按真实结果完整报告，不删除失败结果。
- 允许写法：完成了系统的文献定位与五项已发表方法的统一合法输入适配，并在开发数据上冻结了唯一 `STRONGEST_PUBLISHED_BASELINE`；正式比较因预注册功效不足而未进行。

## Table A：已发表方法适配矩阵（P1–P5，全部 `ELIGIBLE_ADAPTED_VERIFIED`）

| 方法 | 来源 | 家族 | 合法字段 | 训练目标 | 风险输出 | 冻结网格 | 适配偏差 | 中选 | 参数量 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CCTR_DRIVING_CALIBRATION_ADAPTED | Chengtai Cao et al. 2024 (AAAI 2024 (38(19):20949-20957)) | learned/driving calibrated confidence | J_hat_by_seed, global_features, selected_action, strata metadata | train-truth selected-action regret / consequence error under temperature-rescaled NLL | calibrated selected-action uncertainty sigma(x) (higher = riskier) | [{"depth": 3, "hidden_width": 64}, {"depth": 2, "hidden_width": 96}] | Original CCTR computes temperature from a CNN-over-BEV + GRU post-processing network and does NOT use ensembles; the adaptation uses five-seed frozen prediction disagreement + legal context features as the temperature inputs because AC-Twin's legal bundle carries seed-level consequence predictions, not BEV maps. | grid1 | 11522 |
| CONFIDNET_TCP_ADAPTED | Charles Corbiere et al. 2019 (NeurIPS 2019 (pp. 2902-2913)) | learned/driving calibrated confidence | J_hat, global_features, selected_action | probability the tempered J_hat softmax assigns to the train-time oracle action | 1 - predicted_TCP (higher = riskier) | [{"temperature": 0.5}, {"temperature": 1.0}, {"temperature": 2.0}] | Inputs are AC-Twin legal decision features instead of image encoder features; classifier confidence (TCP over actions) replaces class TCP. | grid1 | 9665 |
| CONFORMAL_REGRESSION_REJECT_ADAPTED | Ulf Johansson et al. 2024 (COPA 2024 (PMLR 230:277-294)) | conformal/risk-control rejection | tokens, token_mask, global_features, J_hat, selected_action, strata metadata | true selected-action decision regret (fit split only) | Mondrian interval width (higher = riskier) | [{"difficulty_bins": 4}, {"difficulty_bins": 8}] | Neural regressor on AC-Twin legal features replaces random forests on tabular data; difficulty rule frozen from legal strata/prediction features. | grid1 | 11201 |
| RWR_NN_KNN_REJECT_ADAPTED | Xiaocheng Li et al. 2024 (AISTATS 2024 (PMLR 238:1126-1134)) | joint selective/rejection | tokens, token_mask, global_features, J_hat, selected_action | true selected-action decision regret (train truth only) | kNN-averaged reference residual in representation space (higher = riskier) | [{"k": 16}, {"k": 32}, {"k": 64}] | Reference residuals are in-sample train residuals (validation is reserved for selection/power and cannot serve as the rejector reference); kernel weights reduce to uniform kNN average in the frozen representation. | grid0 | 11201 |
| SELECTIVENET_REGRET_ADAPTED | Yonatan Geifman et al. 2019 (ICML 2019 (PMLR 97:2151-2159)) | joint selective/rejection | tokens, token_mask, global_features, J_hat, selected_action | true selected-action decision regret (train truth only) | 1 - selection_probability (higher = riskier) | [{"hidden_width": 64}, {"hidden_width": 96}] | Regression of decision regret on legal driving features replaces image/tabular prediction; single-task auxiliary head mirrors the paper's regression variant. | grid1 | 23139 |

## Table B1：validation 主指标（开发范围；AC-Twin、五方法中选配置、三个内部基线）

| 方法 | AURC_REGRET | REGRET@70 | REGRET@80 | REGRET@90 | REGRET@95 | PR−prevalence | Spearman | per-seed AURC [min,max] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AC-Twin Direct (frozen R2 audit) | 0.010970 | 0.013534 | 0.014781 | 0.016172 | 0.017732 | 0.3314 | 0.4301 | [0.010475, 0.011855] |
| CCTR_DRIVING_CALIBRATION_ADAPTED (selected grid1) | 0.012762 | 0.015793 | 0.017978 | 0.018778 | 0.019828 | 0.1610 | 0.2830 | [0.013394, 0.015139] |
| CONFIDNET_TCP_ADAPTED (selected grid1) | 0.016524 | 0.017969 | 0.018650 | 0.021821 | 0.022765 | 0.0954 | 0.3227 | [0.015662, 0.017596] |
| CONFORMAL_REGRESSION_REJECT_ADAPTED (selected grid1) | 0.013329 | 0.016798 | 0.018141 | 0.020411 | 0.022083 | 0.1223 | 0.2928 | [0.013329, 0.013489] |
| RWR_NN_KNN_REJECT_ADAPTED (selected grid0) | 0.019536 | 0.021836 | 0.023220 | 0.023702 | 0.023331 | 0.0050 | 0.0862 | [0.018686, 0.023319] |
| SELECTIVENET_REGRET_ADAPTED (selected grid1) | 0.022915 | 0.021200 | 0.021438 | 0.021969 | 0.023117 | 0.0310 | 0.1524 | [0.021371, 0.024944] |
| MARGIN_ONLY (internal baseline) | 0.020824 | 0.021214 | 0.021217 | 0.021696 | 0.022712 | 0.0519 | 0.0748 | [0.020824, 0.020824] |
| SEED_MARGIN_STD (internal baseline) | 0.021838 | 0.022828 | 0.022366 | 0.022959 | 0.022868 | -0.0064 | 0.0033 | [0.021838, 0.021838] |
| SEED_VOTE_ENTROPY (internal baseline) | 0.018959 | 0.020078 | 0.020542 | 0.021644 | 0.022256 | 0.0542 | 0.1161 | [0.018959, 0.018959] |

## Table B2：validation 次要面板（开发范围）

| 方法 | 尾部严重度@20 | 尾部捕获@20 | oracle gap 闭合@80 | 保留80% regret p95 | 保留80% regret max | trim1% AURC | trim5% AURC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AC-Twin Direct (frozen R2 audit) | 0.055897 | 0.4444 | 0.3758 | 0.098665 | 0.389067 | 0.009436 | 0.005940 |
| CCTR_DRIVING_CALIBRATION_ADAPTED (selected grid1) | 0.043110 | 0.3305 | 0.2297 | 0.111214 | 0.389067 | 0.011462 | 0.006783 |
| CONFIDNET_TCP_ADAPTED (selected grid1) | 0.040423 | 0.3333 | 0.1990 | 0.126573 | 0.389067 | 0.013184 | 0.008029 |
| CONFORMAL_REGRESSION_REJECT_ADAPTED (selected grid1) | 0.042457 | 0.3647 | 0.2223 | 0.120990 | 0.389067 | 0.011327 | 0.007099 |
| RWR_NN_KNN_REJECT_ADAPTED (selected grid0) | 0.022140 | 0.1909 | -0.0099 | 0.164115 | 0.389067 | 0.016972 | 0.010576 |
| SELECTIVENET_REGRET_ADAPTED (selected grid1) | 0.029268 | 0.2365 | 0.0716 | 0.143901 | 0.389067 | 0.020513 | 0.012442 |
| MARGIN_ONLY (internal baseline) | 0.030155 | 0.2792 | 0.0817 | 0.143615 | 0.389067 | 0.017901 | 0.011167 |
| SEED_MARGIN_STD (internal baseline) | 0.025558 | 0.1966 | 0.0292 | 0.151046 | 0.389067 | 0.020053 | 0.013295 |
| SEED_VOTE_ENTROPY (internal baseline) | 0.032853 | 0.2764 | 0.1125 | 0.139407 | 0.364099 | 0.017121 | 0.009376 |

## Table B3：全部候选（含未中选；机械选择回执）

| 方法 | 配置 | AURC_REGRET | REGRET@80 | 参数量 | 单组推理 ms | 中选 |
| --- | --- | --- | --- | --- | --- | --- |
| CCTR_DRIVING_CALIBRATION_ADAPTED | grid0 | 0.012855 | 0.017250 | 9794 | 0.0133 | no |
| CCTR_DRIVING_CALIBRATION_ADAPTED | grid1 | 0.012762 | 0.017978 | 11522 | 0.0132 | **YES** |
| CONFIDNET_TCP_ADAPTED | grid0 | 0.016819 | 0.019853 | 9665 | 0.0116 | no |
| CONFIDNET_TCP_ADAPTED | grid1 | 0.016524 | 0.018650 | 9665 | 0.0097 | **YES** |
| CONFIDNET_TCP_ADAPTED | grid2 | 0.016891 | 0.019859 | 9665 | 0.0098 | no |
| CONFORMAL_REGRESSION_REJECT_ADAPTED | grid0 | 0.014089 | 0.018937 | 11201 | 0.0052 | no |
| CONFORMAL_REGRESSION_REJECT_ADAPTED | grid1 | 0.013329 | 0.018141 | 11201 | 0.0058 | **YES** |
| RWR_NN_KNN_REJECT_ADAPTED | grid0 | 0.019536 | 0.023220 | 11201 | 0.1450 | **YES** |
| RWR_NN_KNN_REJECT_ADAPTED | grid1 | 0.020953 | 0.023268 | 11201 | 0.1370 | no |
| RWR_NN_KNN_REJECT_ADAPTED | grid2 | 0.021159 | 0.023793 | 11201 | 0.1600 | no |
| SELECTIVENET_REGRET_ADAPTED | grid0 | 0.022952 | 0.021331 | 11331 | 0.0273 | no |
| SELECTIVENET_REGRET_ADAPTED | grid1 | 0.022915 | 0.021438 | 23139 | 0.0268 | **YES** |

## Table C：AC-Twin vs 冻结最强已发表基线（`CCTR_DRIVING_CALIBRATION_ADAPTED`，validation 配对 bootstrap，开发范围）

| 量 | 点估计 | 95% CI 下界 | 95% CI 上界 |
| --- | --- | --- | --- |
| I_AURC_REGRET | 0.1404 | -0.0097 | 0.2710 |
| I_REGRET@80 | 0.1778 | 0.0807 | 0.2789 |

实质效应（75% 收缩）：I_AURC 0.1404 → 0.1053 ≥ 0.05 ✓；I_REGRET80 0.1778 → 0.1334 ≥ 0.03 ✓ → **material**，进入功效网格。

逐 seed 与四分层点估计（开发范围，描述性）：

| 量 | 点估计 |
| --- | --- |
| seed0 I_AURC / I_REGRET@80 | 0.1340 / 0.1739 |
| seed1 I_AURC / I_REGRET@80 | 0.2804 / 0.2201 |
| seed2 I_AURC / I_REGRET@80 | 0.2776 / 0.1668 |
| seed3 I_AURC / I_REGRET@80 | 0.1990 / 0.2424 |
| seed4 I_AURC / I_REGRET@80 | 0.1423 / 0.1466 |
| tstar_low_i_aurc | 0.0899 |
| tstar_high_i_aurc | 0.1793 |
| interaction_low_i_aurc | 0.0455 |
| interaction_high_i_aurc | 0.2204 |

## Table D：联合功效与终态（计划 §6.2；候选 n 冻结于功效运行前）

| n | ρ=1.00 联合功效 | ρ=0.75 联合功效（裁判情景） | ρ=0.50 联合功效 | 绑定条件 | 绑定功效@0.75 |
| --- | --- | --- | --- | --- | --- |
| 2000 | 0.6111 | 0.3778 | 0.1571 | r1_i_aurc | 0.4000 |
| 3000 | 0.7889 | 0.5472 | 0.2486 | r1_i_aurc | 0.5605 |
| 4000 | 0.8937 | 0.6872 | 0.3473 | r1_i_aurc | 0.6930 |
| 6000 | 0.9696 | 0.8354 | 0.4890 | r1_i_aurc | 0.8371 |

75% 情景下最大冻结 n=6000 的联合功效 0.8354 < 0.90 → `selected_n=null`，终态 **`STAGE5_MERGE_COMPARISON_UNDERPOWERED`**。不创建 comparison holdout，不降低门槛、不换 seed、不换方法或主基线（计划 §6.2/§10）。

## Table E：成本与完整性（开发范围）

| 方法 | 参数量 | 近似前向 FLOPs/组（MAC×2） | 单组推理 ms | 原生校准 |
| --- | --- | --- | --- | --- |
| CCTR_DRIVING_CALIBRATION_ADAPTED | 11522 | 23044 | 0.0132 | OK |
| CONFIDNET_TCP_ADAPTED | 9665 | 19330 | 0.0097 | OK |
| CONFORMAL_REGRESSION_REJECT_ADAPTED | 11201 | 22402 | 0.0058 | OK |
| RWR_NN_KNN_REJECT_ADAPTED | 11201 | 22402 | 0.1450 | NOT_APPLICABLE |
| SELECTIVENET_REGRET_ADAPTED | 23139 | 46278 | 0.0268 | NOT_APPLICABLE |
| AC-Twin Direct (frozen R2 audit) | 15740 | 31480 | frozen (R2 receipt) | NOT_APPLICABLE |

原生校准细节：P1 ECE 0.2483；P4 经验覆盖 0.9527（目标 0.90）；P5 NLL 0.2255、95% 经验覆盖 0.8933；P2/P3 `NOT_APPLICABLE`。

训练成本：60 个（方法×配置×seed）拟合合计约 468 s 墙钟（CUDA；峰值显存 NOT_MEASURED）。

## 来源与重建

- 冻结源：`experiments/stage5_sota_comparison/` 下 `literature/`、`development/`、`power/` 的 create-once 产物；本表由 `build_table7_details.py` 重新导出，数字不改写、不重选、不重判。
- risk–coverage 曲线数据：`paper/figures/stage5_sota/figS_stage5_risk_coverage_data.csv`（同一冻结源，开发范围）。
