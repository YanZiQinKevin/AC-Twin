# replication_v2 quality assurance

## Data collection

| Check | merge independent seed | roundabout independent scene |
|---|---:|---:|
| Accepted episode pairs | 2,000 | 2,000 |
| Unique episode seeds | 2,000 | 2,000 |
| Train / validation / test | 1,400 / 300 / 300 | 1,400 / 300 / 300 |
| Identity replay checks | 20 | 20 |
| Maximum identity replay difference | 0.0 | 0.0 |
| Maximum factual/counterfactual prefix difference | 0.0 | 0.0 |
| Action×duration strata | 10, approximately 200 each | 10, approximately 200 each |

The merge collection accepted 2,000 of 3,135 attempts. The roundabout collection accepted 2,000 of 3,176 attempts; 606 attempts terminated in the prefix and 570 lacked a complete future horizon. No quality threshold was relaxed.

## Model artifacts

- Three configured conditions per dataset: `history_only`, `shuffled_full`, and `prefix_10`.
- Five training seeds per condition.
- Each dataset contains 15 model checkpoints and 15 training logs.
- Each saved prediction tensor has shape `[5, 300, 20, 2]`.
- Each test split contains 300 unique episode indices.
- The failed residual proxy was not retrained because it is outside the frozen replication comparison.

## Negative-control integrity

| Dataset | Self matches | Cross-split matches | CF-action match | Duration match | First factual-action match |
|---|---:|---:|---:|---:|---:|
| merge independent seed | 0 | 0 | 100% | 100% | 99.50% |
| roundabout independent scene | 0 | 0 | 100% | 100% | 99.85% |

## Statistical audit

- The independent inferential unit is the test episode, not a trajectory time point or training seed.
- Both new datasets use 300 paired test episodes and 10,000 paired-bootstrap resamples.
- Both new datasets exceed the predeclared 5% practical-improvement threshold and have 95% confidence-interval lower bounds above zero.
- Across the 20 new action×duration diagnostic strata, the mean paired effect reverses in 0 strata. Individual strata contain 30 test episodes and are not treated as separately powered confirmatory tests.
- Overall replication verdict: `PASS`.

## Integrity hashes (SHA-256)

- `synthesis_report.json`: `4D55F89DC629CB941DC2D343520CA466A47E66BE22843D183B74DC75C23B17C9`
- `stratum_results.csv`: `B4B74A032B91CA8DC622909390DBA269FE8A718AA5EA0D472F2F126544D2508B`
- merge dataset: `64B6FF17DC20A2B467BEFBD2F14EEF80C1D325A55D3F06B56476D69D59F56F50`
- roundabout dataset: `95CC8823EA9F91749B9E72B05AEE1590C0395E36C6C09BD9B50DF71A99E1F114`

## Remaining boundaries

- Both scenes come from Highway-env; simulator-level external validation remains open.
- Stable identities for neighbouring vehicles are not yet represented.
- The present evidence establishes episode-specific predictive information, not identified structural exogenous noise.
