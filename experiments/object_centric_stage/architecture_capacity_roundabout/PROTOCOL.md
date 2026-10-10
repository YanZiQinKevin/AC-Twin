# Roundabout Stable-ID Cross-Scene Replication Protocol

Status: formal data collection, training, statistical analysis and QA completed; replication decision: `PASS_STABLE_ID_CROSS_SCENE_REPLICATION`.

## Research question

Does a set encoder with stable within-episode vehicle IDs outperform a distance-sorted GRU in the independent `roundabout-v0` scene? This was the `GO_STABLE_ID_ONLY` cross-scene gate after the Merge capacity-matched experiment.

## Data protocol

- Configuration: `config/object_centric_roundabout_collect.json`
- Scene: `roundabout-v0`
- Valid episode pairs: 2,000
- Split: 1,400 training, 300 validation, 300 test
- History: 10 steps; factual suffix: first 10 steps; prediction: 20 steps
- Actions: lane left, idle, lane right, faster, slower
- Durations: 1 step, 3 steps
- Independent unit: episode
- Formal data: `data/object_centric_stage/roundabout_2000/object_twinworld_2000.npz`
- SHA-256：`7E50BF4A17F55D6666549D7743B1D2E5924F5DFA10411F94BD62431146DE8672`

Formal data QA:

- 2,000 unique episode seeds;
- 200 samples in each of 10 action-by-duration strata;
- Split: 1,400 training, 300 validation, 300 test;
- Maximum common-prefix difference across factual/counterfactual branches: 0;
- Maximum difference across 20 deterministic replays: 0;
- Maximum ego history, factual-future and counterfactual-future coordinate differences between distance-sorted and stable-ID observations: 0;
- All required arrays exist and are finite; collection checkpoint files were cleaned up normally.

Data must include distance-sorted and stable-ID observations, both branch futures, complete action plans, episode seeds, intervention times and fixed splits. Do not relax termination or common-prefix consistency thresholds to fill quotas.

## Model protocol

Configuration: `config/architecture_capacity_roundabout.json`

| Condition | Encoder | Hidden dimension | Parameters | Suffix source |
|---|---|---:|---:|---|
| `row_65k` | Distance-sorted GRU | 54 | 64,734 | Same episode |
| `set_65k` | Stable-ID set encoder | 48 | 64,266 | Same episode |
| `set_shuffled_65k` | Stable-ID set encoder | 48 | 64,266 | Shuffled within test split |
| `row_83k` | Distance-sorted GRU | 62 | 83,246 | Same episode |
| `set_83k` | Stable-ID set encoder | 55 | 83,642 | Same episode |
| `set_shuffled_83k` | Stable-ID set encoder | 55 | 83,642 | Shuffled within test split |

- Training seeds: 11, 22, 33, 44, 55
- Maximum epochs: 200
- Early-stopping patience: 25
- batch size：128
- Learning rate: 0.001
- weight decay：0.0001
- Primary endpoint: `effect-FDE@20`
- Statistics: test-episode paired bootstrap, 10,000 resamples, 95% CIs

## Prespecified contrasts

1. `row_65k - set_65k`
2. `row_83k - set_83k`
3. `set_shuffled_65k - set_65k`, evidence-source negative control
4. `set_shuffled_83k - set_83k`, evidence-source negative control

Do not run static/dynamic graphs or reselect hidden dimensions based on Roundabout test results.

## Cross-scene decision

Replication requires all of the following:

1. Positive mean paired `row - set` differences at both budgets;
2. Pooled episode-level paired 95% CI lower bound above 0;
3. No clear systematic directional reversal across action, duration or effect-magnitude strata;
4. Clearly worse performance with shuffled factual suffixes than same-episode suffixes.

If the gate fails, stable IDs remain an identity-consistency design choice but cannot be presented as a cross-scene performance innovation.

## Long-running tasks and recovery

Formal collection and training were manually launched by the user. Collection prints progress every 20 samples and checkpoints every 100; training prints progress every 5 epochs and saves recoverable records after each condition-by-seed run.

Resume collection with the same configuration and `--resume`. Training also uses `--resume`; completed seeds are skipped only when configurations, data SHA-256 and run metadata match exactly.
