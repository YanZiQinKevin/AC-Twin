# pilot_2000 Results

Run time: 2026-09-04T12:15:56.206178+08:00

## Conclusion

Decision: **STRONG_PASS**. Factual suffixes reduced effect-FDE@20 by 20.50% relative to the history model.

The episode-level paired mean difference (history - suffix) was 0.6404 m, 95% CI [0.4023, 0.8802], n=300 independent test episodes.

## Summary results

| Model | effect-ADE@20 (m) | effect-FDE@20 (m) |
|---|---:|---:|
| zero_effect | 2.7309 | 3.9935 |
| action_pair_mean | 2.4462 | 3.7674 |
| history_only | 1.8253 | 3.1238 |
| factual_suffix | 1.4798 | 2.4833 |

## Action strata

| Counterfactual action | Duration (steps) | n | history FDE | suffix FDE | Improvement |
|---|---:|---:|---:|---:|---:|
| LANE_LEFT | 1 | 30 | 1.8379 | 1.9355 | -5.31% |
| LANE_LEFT | 3 | 30 | 3.2459 | 2.1083 | 35.05% |
| IDLE | 1 | 30 | 2.8997 | 2.0556 | 29.11% |
| IDLE | 3 | 30 | 3.0824 | 2.7533 | 10.68% |
| LANE_RIGHT | 1 | 30 | 1.8173 | 1.7903 | 1.48% |
| LANE_RIGHT | 3 | 30 | 3.3863 | 2.3265 | 31.30% |
| FASTER | 1 | 30 | 2.6834 | 2.8954 | -7.90% |
| FASTER | 3 | 30 | 4.4731 | 3.7190 | 16.86% |
| SLOWER | 1 | 30 | 3.8245 | 2.3286 | 39.11% |
| SLOWER | 3 | 30 | 3.9873 | 2.9210 | 26.74% |

Factual suffixes reduced endpoint error in 8/10 action-by-duration subgroups. Effects were not consistent across all intervention types; this heterogeneity is a boundary requiring explanation in subsequent work.

## Statistical specification

- Independent unit: episode; neither time points nor training seeds increase n.
- Primary endpoint: prespecified effect-FDE@20.
- Inference: compute test-episode errors separately for five training seeds, average errors, then perform paired bootstrap resampling.
- Secondary endpoints and strata are exploratory descriptions; no multiple-significance claims are made.

## Interpretation boundaries

- A passing result establishes only additional predictive information from factual suffixes in this retrospective Highway-env task.
- It does not establish real-road causal effects or online planning capabilities.
- Subsequent work should examine effect magnitude, action type and a second platform, and implement structural innovation inversion.

See `report.json` for detailed values, the five training seeds, strata and data-integrity information.
