# Factual-Evidence Source Ablation Results

Run time: 2026-09-04T14:54:16.101769+08:00

## Confirmatory conclusion

Decision: **STRONG_PASS**. `prefix_10` reduced effect-FDE@20 by 32.31% relative to `shuffled_full`.

The episode-level paired mean difference (shuffled_full - prefix_10) was 1.0775 m, 95% CI [0.8509, 1.3078], n=300.

## All conditions

| Condition | effect-ADE@20 (m) | effect-FDE@20 (m) | FDE SD across seeds |
|---|---:|---:|---:|
| history_only | 1.8306 | 3.2210 | 0.1746 |
| shuffled_full | 1.9250 | 3.3350 | 0.1906 |
| prefix_10 | 1.2861 | 2.2575 | 0.1002 |

## Interpretation

- An advantage of `prefix_10` over `shuffled_full` supports same-episode factual information rather than input length alone.
- Episodes are independent units for all inference; five training seeds do not increase n.

See `report.json` for contrast CIs, shuffle-matching quality and residual-model details.
