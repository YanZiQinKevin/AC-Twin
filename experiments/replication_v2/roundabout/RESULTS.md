# Factual-Evidence Source Ablation Results

Run time: 2026-09-04T15:12:40.787762+08:00

## Confirmatory conclusion

Decision: **STRONG_PASS**. `prefix_10` reduced effect-FDE@20 by 39.22% relative to `shuffled_full`.

The episode-level paired mean difference (shuffled_full - prefix_10) was 2.3080 m, 95% CI [1.9155, 2.7293], n=300.

## All conditions

| Condition | effect-ADE@20 (m) | effect-FDE@20 (m) | FDE SD across seeds |
|---|---:|---:|---:|
| history_only | 2.5492 | 4.5127 | 0.3796 |
| shuffled_full | 3.3614 | 5.8847 | 0.2512 |
| prefix_10 | 2.0657 | 3.5767 | 0.2065 |

## Interpretation

- An advantage of `prefix_10` over `shuffled_full` supports same-episode factual information rather than input length alone.
- Episodes are independent units for all inference; five training seeds do not increase n.

See `report.json` for contrast CIs, shuffle-matching quality and residual-model details.
