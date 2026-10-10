# Stage 5 Published-Method Development Comparison: Detailed Audit Tables

> Plan: `AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3`
> **Terminal state: `STAGE5_MERGE_COMPARISON_UNDERPOWERED` (2026-09-28; permanent, no re-adjudication)**
> Scope: `ADAPTIVE_DEVELOPMENT_ONLY` diagnostics on Stage 4C-R2 validation (1,500 groups). The Stage 5 comparison holdout was **never created**, because the §6.2 power terminal state forbids collection. No numbers here are confirmatory comparisons.
>
> Main manuscript table: `TABLE_7.md`. Tables A–E preserve audit, reproduction and supplementary detail, not additional manuscript comparison tables.

## Claim boundaries (plan §11)

- No §11.1 SOTA-superiority claim: primary comparison failed the 0.90 joint-power criterion; no holdout or adjudication.
- Under §11.3, Stage 4/D2F confirmations remain valid; superiority over published methods is not established. Report this development comparison fully, including failures.
- Supported statement: five published methods were adapted under common legal inputs; one `STRONGEST_PUBLISHED_BASELINE` was frozen on development data. Formal comparison was not performed because preregistered power was insufficient.

## Table A: Adaptation matrix (P1–P5, all `ELIGIBLE_ADAPTED_VERIFIED`)

| Method | Source | Family | Legal fields | Training objective | Risk output | Frozen grid | Adaptation deviation | Selected | Parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CCTR_DRIVING_CALIBRATION_ADAPTED | Chengtai Cao et al. 2024 (AAAI 2024 (38(19):20949-20957)) | learned/driving calibrated confidence | J_hat_by_seed, global_features, selected_action, strata metadata | train-truth selected-action regret / consequence error under temperature-rescaled NLL | calibrated selected-action uncertainty sigma(x) (higher = riskier) | [{"depth": 3, "hidden_width": 64}, {"depth": 2, "hidden_width": 96}] | Original CCTR computes temperature from a CNN-over-BEV + GRU post-processing network and does NOT use ensembles; the adaptation uses five-seed frozen prediction disagreement + legal context features as the temperature inputs because AC-Twin's legal bundle carries seed-level consequence predictions, not BEV maps. | grid1 | 11522 |
| CONFIDNET_TCP_ADAPTED | Charles Corbiere et al. 2019 (NeurIPS 2019 (pp. 2902-2913)) | learned/driving calibrated confidence | J_hat, global_features, selected_action | probability the tempered J_hat softmax assigns to the train-time oracle action | 1 - predicted_TCP (higher = riskier) | [{"temperature": 0.5}, {"temperature": 1.0}, {"temperature": 2.0}] | Inputs are AC-Twin legal decision features instead of image encoder features; classifier confidence (TCP over actions) replaces class TCP. | grid1 | 9665 |
| CONFORMAL_REGRESSION_REJECT_ADAPTED | Ulf Johansson et al. 2024 (COPA 2024 (PMLR 230:277-294)) | conformal/risk-control rejection | tokens, token_mask, global_features, J_hat, selected_action, strata metadata | true selected-action decision regret (fit split only) | Mondrian interval width (higher = riskier) | [{"difficulty_bins": 4}, {"difficulty_bins": 8}] | Neural regressor on AC-Twin legal features replaces random forests on tabular data; difficulty rule frozen from legal strata/prediction features. | grid1 | 11201 |
| RWR_NN_KNN_REJECT_ADAPTED | Xiaocheng Li et al. 2024 (AISTATS 2024 (PMLR 238:1126-1134)) | joint selective/rejection | tokens, token_mask, global_features, J_hat, selected_action | true selected-action decision regret (train truth only) | kNN-averaged reference residual in representation space (higher = riskier) | [{"k": 16}, {"k": 32}, {"k": 64}] | Reference residuals are in-sample train residuals (validation is reserved for selection/power and cannot serve as the rejector reference); kernel weights reduce to uniform kNN average in the frozen representation. | grid0 | 11201 |
| SELECTIVENET_REGRET_ADAPTED | Yonatan Geifman et al. 2019 (ICML 2019 (PMLR 97:2151-2159)) | joint selective/rejection | tokens, token_mask, global_features, J_hat, selected_action | true selected-action decision regret (train truth only) | 1 - selection_probability (higher = riskier) | [{"hidden_width": 64}, {"hidden_width": 96}] | Regression of decision regret on legal driving features replaces image/tabular prediction; single-task auxiliary head mirrors the paper's regression variant. | grid1 | 23139 |

## Table B1: Primary validation endpoints (development scope; AC-Twin, selected adaptations, three internal baselines)

| Method | AURC_REGRET | REGRET@70 | REGRET@80 | REGRET@90 | REGRET@95 | PR−prevalence | Spearman | per-seed AURC [min,max] |
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

## Table B2: Secondary validation endpoints (development scope)

| Method | Tail severity@20 | Tail capture@20 | Oracle-gap closure@80 | Retained-80% regret p95 | Retained-80% regret max | trim1% AURC | trim5% AURC |
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

## Table B3: All candidates, including unselected configurations (mechanical-selection receipt)

| Method | Configuration | AURC_REGRET | REGRET@80 | Parameters | Inference ms/group | Selected |
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

## Table C: AC-Twin vs frozen strongest published baseline (`CCTR_DRIVING_CALIBRATION_ADAPTED`; validation paired bootstrap; development scope)

| Quantity | Point estimate | 95% CI lower | 95% CI upper |
| --- | --- | --- | --- |
| I_AURC_REGRET | 0.1404 | -0.0097 | 0.2710 |
| I_REGRET@80 | 0.1778 | 0.0807 | 0.2789 |

Materiality at 75% effect shrinkage: I_AURC 0.1404 → 0.1053 ≥ 0.05 ✓; I_REGRET80 0.1778 → 0.1334 ≥ 0.03 ✓ → **material**, proceeding to the power grid.

Per-seed and four-stratum point estimates (descriptive development evidence):

| Quantity | Point estimate |
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

## Table D: Joint power and terminal state (plan §6.2; candidate n frozen before power analysis)

| n | ρ=1.00 joint power | ρ=0.75 joint power (adjudication scenario) | ρ=0.50 joint power | Binding condition | Binding power@0.75 |
| --- | --- | --- | --- | --- | --- |
| 2000 | 0.6111 | 0.3778 | 0.1571 | r1_i_aurc | 0.4000 |
| 3000 | 0.7889 | 0.5472 | 0.2486 | r1_i_aurc | 0.5605 |
| 4000 | 0.8937 | 0.6872 | 0.3473 | r1_i_aurc | 0.6930 |
| 6000 | 0.9696 | 0.8354 | 0.4890 | r1_i_aurc | 0.8371 |

At the maximum frozen n=6000 under 75% shrinkage, joint power 0.8354 < 0.90 → `selected_n=null`; terminal state **`STAGE5_MERGE_COMPARISON_UNDERPOWERED`**. No comparison holdout, gate relaxation, seed substitution, method replacement or primary-baseline change (plan §6.2/§10).

## Table E: Cost and integrity (development scope)

| Method | Parameters | Approximate forward FLOPs/group (MAC×2) | Inference ms/group | Native calibration |
| --- | --- | --- | --- | --- |
| CCTR_DRIVING_CALIBRATION_ADAPTED | 11522 | 23044 | 0.0132 | OK |
| CONFIDNET_TCP_ADAPTED | 9665 | 19330 | 0.0097 | OK |
| CONFORMAL_REGRESSION_REJECT_ADAPTED | 11201 | 22402 | 0.0058 | OK |
| RWR_NN_KNN_REJECT_ADAPTED | 11201 | 22402 | 0.1450 | NOT_APPLICABLE |
| SELECTIVENET_REGRET_ADAPTED | 23139 | 46278 | 0.0268 | NOT_APPLICABLE |
| AC-Twin Direct (frozen R2 audit) | 15740 | 31480 | frozen (R2 receipt) | NOT_APPLICABLE |

Native calibration: P1 ECE 0.2483; P4 empirical coverage 0.9527 (target 0.90); P5 NLL 0.2255, 95% empirical coverage 0.8933; P2/P3 `NOT_APPLICABLE`.

Training cost: 60 method×configuration×seed fits, approximately 468 s wall time (CUDA; peak GPU memory NOT_MEASURED).

## Sources and reconstruction

- Frozen sources: create-once `literature/`, `development/` and `power/` products in `experiments/stage5_sota_comparison/`. `build_table7_details.py` re-exports them without numerical rewriting, reselection or re-adjudication.
- Risk–coverage data: `paper/figures/stage5_sota/figS_stage5_risk_coverage_data.csv`, from the same frozen development sources.
