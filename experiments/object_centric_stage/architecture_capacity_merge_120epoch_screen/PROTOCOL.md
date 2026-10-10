# Capacity-Matched Architecture Comparison Protocol: Merge

Frozen on: 2026-09-04
Experiment: `architecture_capacity_merge`

## Research question

At matched parameter budgets, do stable entity identity, static message passing and distance-gated dynamic message passing each reduce ego counterfactual-effect error?

## Data and independent unit

- Data: `data/object_centric_stage/merge_2000/object_twinworld_2000.npz`
- Data SHA256: `EB6122AC62AF20219FC0B27D346740A935587FF32D82E4000127A50A41466E2B`
- 2,000 independent episodes; train/validation/test: 1,400/300/300.
- Independent unit: test episode; training seeds do not increase sample size.

## Frozen configuration

- Configuration: `config/architecture_capacity_merge.json`
- 120-epoch configuration SHA256: `502BC2B24096BBF1DBA11FBCC19734AE3C33BA4EA6E46AF52E1D4F7BC5CA8742`
- Two parameter budgets: approximately 65k and 83k.
- Each includes a distance-sorted GRU, stable-ID set encoder, static fully connected graph, 60 m distance-gated dynamic graph and shuffled-suffix dynamic-graph negative control.
- Factual-evidence window: 10 steps; prediction horizon: 20 steps; training seeds: `[11,22,33,44,55]`.
- AdamW, learning rate 0.001, weight decay 0.0001, batch size 128.
- Maximum 120 epochs; early-stopping patience 15.

## Prespecified trigger for extending training

The initial 60-epoch screen is preserved in `experiments/object_centric_stage/architecture_capacity_merge_60epoch_screen/`. Every core condition had best epochs in the final 6 epochs for at least 3/5 seeds, mostly 5/5. The prespecified rule therefore uniformly extended all conditions at both budgets to 120 epochs, without selectively retraining based on test performance.

## Parameter matching

| Parameter budget | Distance-sorted GRU | Stable-ID set encoder | Static graph | Dynamic graph |
|---|---:|---:|---:|---:|
| ~65k | 64,734 | 64,266 | 64,302 | 64,302 |
| ~83k | 83,246 | 83,642 | 83,274 | 83,274 |

Within-budget deviations must not exceed 1.5%. Static/dynamic graphs have identical parameter counts and message functions; only adjacency differs.

## Endpoints, statistics and decisions

- Primary endpoint: `effect-FDE@20`; secondary endpoint: `effect-ADE@20`.
- Contrasts: row-minus-set, set-minus-static, static-minus-dynamic, shuffled-dynamic-minus-same-episode-dynamic.
- Average episode errors across 5 training seeds, then use 10,000 paired bootstrap resamples and 95% CIs.
- Independent dynamic-structure gains require consistent direction at both budgets and an episode-level pooled primary 95% CI excluding 0.
- Shuffled suffixes are evidence negative controls only, not architecture-selection criteria.
- Without intervention-edge ground truth, use dynamic interaction graph, not discovered true causal graph.

## Implementation repairs

- 32 model, mask, parameter-count, configuration and reproducibility tests passed.
- Fixed incompatible units between standardized positions and the 60 m threshold, which had collapsed dynamic adjacency to a fully connected graph.
- Fixed model creation before seed setting, which prevented reproducible initialization of the first condition.
- CuBLAS deterministic workspace configured; internal smoke predictions reproduced byte for byte.

## Environment

- Python 3.12.7；PyTorch 2.9.1+cu130；NumPy 1.26.4。
- NVIDIA GeForce RTX 3070 Ti, 8192 MiB; driver 616.56; CUDA available.

