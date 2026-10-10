# replication_v2 Combined Results

Overall decision: **PASS**.

| Dataset | shuffled-full FDE (m) | prefix-10 FDE (m) | Improvement | Paired difference and 95% CI (m) | Decision |
|---|---:|---:|---:|---:|---|
| merge_original | 3.4365 | 2.2975 | 33.14% | 1.1390 [0.8936, 1.3887] | PASS |
| merge_independent_seed | 3.3350 | 2.2575 | 32.31% | 1.0775 [0.8515, 1.3045] | PASS |
| roundabout_independent_scene | 5.8847 | 3.5767 | 39.22% | 2.3080 [1.9088, 2.7253] | PASS |

## Action-stratum audit

Across two newly collected replication datasets, 20 action-by-duration strata were checked, with 0 mean-effect reversals. See `stratum_results.csv` for complete results.

## Conclusion

The advantage of same-episode 10-step factual suffixes over matched shuffled suffixes replicated under independent collection seeds and a different road topology. At that historical stage, this supported proceeding to stable entity IDs and object-centric dynamic-interaction models.

## Boundaries

The evidence supports event-specific information in factual suffixes, not identified structural exogenous noise. Cross-simulator validation had not yet been completed at this stage.
