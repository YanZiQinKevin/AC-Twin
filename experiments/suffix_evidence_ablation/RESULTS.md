# Factual-Evidence Source Ablation Results

Run time: 2026-09-04T12:58:45.717392+08:00

## Confirmatory conclusion

Decision: **STRONG_PASS**. Complete same-episode factual suffixes reduced effect-FDE@20 by 32.77% relative to action-matched shuffled suffixes.

The episode-level paired mean difference (shuffled - full) was 1.1262 m, 95% CI [0.8623, 1.3884], n=300.

## All conditions

| Condition | effect-ADE@20 (m) | effect-FDE@20 (m) | FDE SD across seeds |
|---|---:|---:|---:|
| history_only | 1.8685 | 3.2428 | 0.1875 |
| shuffled_full | 1.9959 | 3.4365 | 0.1096 |
| last_frame | 1.8298 | 3.1560 | 0.1768 |
| prefix_1 | 1.8340 | 3.1852 | 0.0550 |
| prefix_5 | 1.4827 | 2.6309 | 0.0626 |
| prefix_10 | 1.3307 | 2.2975 | 0.1074 |
| full_20 | 1.3560 | 2.3103 | 0.1764 |
| residual_innovation_20 | 2.3655 | 4.1562 | 0.2731 |

## Interpretation

- An advantage of `full_20` over `shuffled_full` supports same-episode factual evidence rather than input length alone.
- `prefix_1/5/10/20` describes the information-length curve; these analyses are exploratory.
- `residual_innovation_20` is a first-order dynamics-residual proxy, not true exogenous noise or completed structural causal abduction.
- Episodes are independent units for all inference; five training seeds do not increase n.

See `report.json` for contrast CIs, shuffle-matching quality and residual-model details.
