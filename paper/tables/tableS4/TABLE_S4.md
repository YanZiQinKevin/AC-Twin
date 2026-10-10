# Table S4 Task validity and action support

Status: EXTRACTED_FROM_FROZEN_REPORTS / INCLUDED_IN_ENGLISH_V3 / 2026-10-09

Each scope contains 2,000 ten-action groups. Existing 95% intervals are copied, not recomputed. Full-precision fields and original report keys are retained in `tableS4_extracted_fields.json`.

## Task quantities

| Quantity | Merge | JOINT_OOD | MetaDrive |
| --- | --- | --- | --- |
| Base mean regret | 0.022478 [0.020140, 0.024894] | 0.034056 [0.031141, 0.037011] | 0.059399 [0.055849, 0.063071] |
| Idle mean regret | 0.084158 [0.080768, 0.087522] | 0.088735 [0.085326, 0.092119] | 0.108734 [0.104668, 0.112828] |
| Random mean regret | 0.082715 [0.080467, 0.085014] | 0.085603 [0.083288, 0.087959] | 0.115627 [0.113136, 0.118138] |
| Base vs random improvement | 0.728242 [0.698486, 0.756996] | 0.602166 [0.563973, 0.638510] | 0.486287 [0.455818, 0.516145] |
| Spearman branch | 0.412103 [0.390368, 0.432904] | 0.428196 [0.405783, 0.450395] | 0.725942 [0.709549, 0.740732] |
| High regret frac | 0.229500 [0.211500, 0.247500] | 0.313000 [0.292500, 0.333500] | 0.489000 [0.467988, 0.511000] |
| Reactive group fraction | Not required/reported | Not required/reported | 0.263000 |
| Oracle actions with at least 5% share | 7 | 6 | 7 |
| All task criteria | Pass | Pass | Pass |

## Oracle action shares

| Action | Merge | JOINT_OOD | MetaDrive |
| --- | --- | --- | --- |
| LANE_LEFT × 1 | 0.107000 | 0.098500 | 0.122500 |
| LANE_LEFT × 3 | 0.071500 | 0.056000 | 0.155000 |
| IDLE × 1 | 0.076000 | 0.072000 | 0.031500 |
| IDLE × 3 | 0.054500 | 0.043500 | 0.073500 |
| LANE_RIGHT × 1 | 0.047000 | 0.044000 | 0.099000 |
| LANE_RIGHT × 3 | 0.029500 | 0.027500 | 0.118500 |
| FASTER × 1 | 0.214500 | 0.224500 | 0.097000 |
| FASTER × 3 | 0.328000 | 0.351500 | 0.285500 |
| SLOWER × 1 | 0.057500 | 0.068000 | 0.011500 |
| SLOWER × 3 | 0.014500 | 0.014500 | 0.006000 |

## Marginal stratum counts

| Stratum | Merge | JOINT_OOD | MetaDrive |
| --- | --- | --- | --- |
| tstar_low | 1053 | 1114 | 1007 |
| tstar_high | 947 | 886 | 993 |
| interaction_low | 996 | 1725 | 1020 |
| interaction_high | 1004 | 275 | 980 |

## Merge task criteria

| Registered condition | Outcome |
| --- | --- |
| c1_all_complete_gate_q | Pass |
| c2_high_regret_frac_ge_15pct | Pass |
| c3_at_least_3_oracle_actions_ge_5pct | Pass |
| c4_branch_spearman_ci_lower_positive | Pass |
| c5_base_vs_random_point_and_ci_positive | Pass |
| c6_base_not_worse_than_idle | Pass |
| c7_strata_ge_200_groups | Pass |

## JOINT_OOD task criteria

| Registered condition | Outcome |
| --- | --- |
| c1_complete_gate_q | Pass |
| c2_domain_shift | Pass |
| c3_high_regret_frac | Pass |
| c4_oracle_diversity | Pass |
| c5_branch_spearman | Pass |
| c6_base_beats_random | Pass |
| c7_base_not_worse_than_idle | Pass |
| c8_strata_min_10pct | Pass |

## MetaDrive task criteria

| Registered condition | Outcome |
| --- | --- |
| c1_completeness_and_qa | Pass |
| c2_high_regret_frac_ge_15pct | Pass |
| c3_at_least_3_oracle_actions_ge_5pct | Pass |
| c4_branch_spearman_ci_lower_positive | Pass |
| c5_base_beats_random_and_not_worse_than_idle | Pass |
| c6_reactive_response_ge_20pct | Pass |
| c7_strata_each_ge_10pct | Pass |
| c8_denominators_projection_mask_valid | Pass |

The action order is LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER, with durations 1 and 3 for each. Rare oracle actions have limited support. Reactive fraction is registered and reported for MetaDrive; the Merge/OOD reports do not require or report that field. This extraction makes no new adjudication and does not compare selective-coverage gains with task-reference regrets.
