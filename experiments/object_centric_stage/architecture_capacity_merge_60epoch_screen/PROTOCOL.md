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
- Configuration SHA256: `28C5AE3DF0E58DE9769B024A995BE110626B815D8316DCE486B315F4E08760BE`
- Two parameter budgets: approximately 65k and 83k.
- Each includes a distance-sorted GRU, stable-ID set encoder, static fully connected graph, 60 m distance-gated dynamic graph and shuffled-suffix dynamic-graph negative control.
- Factual-evidence window: 10 steps; prediction horizon: 20 steps.
- Training seeds for every condition: `[11,22,33,44,55]`.
- AdamW; learning rate 0.001; weight decay 0.0001; batch size 128.
- Maximum 60 epochs; early-stopping patience 10.

## Parameter matching

| Parameter budget | Distance-sorted GRU | Stable-ID set encoder | Static graph | Dynamic graph |
|---|---:|---:|---:|---:|
| ~65k | 64,734 | 64,266 | 64,302 | 64,302 |
| ~83k | 83,246 | 83,642 | 83,274 | 83,274 |

Deviation from 65,000 or 83,000 must not exceed 1.5% within each budget. Static/dynamic graphs have identical parameters and message functions; only adjacency differs.

## Endpoints and statistics

- Primary: `effect-FDE@20`.
- Secondary: `effect-ADE@20`.
- Contrasts: row-minus-set, set-minus-static, static-minus-dynamic, shuffled-dynamic-minus-same-episode-dynamic.
- Average test-episode errors across 5 seeds, then use 10,000 paired bootstrap resamples and 95% CIs.

## Prespecified decisions

- Independent dynamic-structure gains require consistent direction at both budgets and a pooled episode-level primary 95% CI excluding 0.
- If any core condition has best epochs in the last 6 epochs for at least 3/5 seeds, uniformly rerun all core conditions at that budget with 120 epochs and patience 15.
- Shuffled suffixes are evidence negative controls only, not architecture-selection criteria.
- This is a distance-gated dynamic interaction graph; without intervention-edge ground truth, it is not a discovered true causal graph.

## Pre-run implementation repairs

- Added model, mask, parameter-count, configuration-validity and reproducible-initialization tests; all 32 passed.
- Fixed direct use of a 60 m threshold after position standardization; standardized node inputs are retained, while training position SDs restore metric distances for adjacency.
- Fixed model creation before seed setting, which prevented reproducible initialization of the first condition.
- Set `CUBLAS_WORKSPACE_CONFIG=:4096:8`; all internal prediction arrays reproduced byte for byte under the same smoke configuration.

## Environment

- Python 3.12.7
- PyTorch 2.9.1+cu130
- NumPy 1.26.4
- CUDA available: Yes
- GPU：NVIDIA GeForce RTX 3070 Ti，8192 MiB
- NVIDIA driver: 616.56

