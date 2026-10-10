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

## Final frozen configuration

- Configuration: `config/architecture_capacity_merge.json`
- Configuration SHA256: `784503C0CDFDCFDCE81F91670369E7D171139F18F5B288226DEFF906F89DC765`
- Two parameter budgets: approximately 65k and 83k.
- Each budget includes a distance-sorted GRU, stable-ID set encoder, static fully connected graph, 60 m distance-gated dynamic graph and shuffled-suffix dynamic-graph negative control.
- Factual-evidence window: 10 steps; prediction horizon: 20 steps; training seeds: `[11,22,33,44,55]`.
- AdamW, learning rate 0.001, weight decay 0.0001, batch size 128.
- Maximum 200 epochs; early-stopping patience 25.

## Basis for uniformly extending training

- The 60-epoch screen is preserved in `architecture_capacity_merge_60epoch_screen/`: every core condition triggered the prespecified late-best-epoch rule.
- The 120-epoch screen is preserved in `architecture_capacity_merge_120epoch_screen/`: several conditions still had best epochs near the cap for at least 3/5 seeds; the static graph's median validation-loss improvement over the last 10 epochs was approximately 5%.
- All conditions were therefore extended uniformly to 200 epochs based only on validation curves, not test-set winners. Models, data, learning rates and evaluation definitions were unchanged.

## Parameter matching

| Parameter budget | Distance-sorted GRU | Stable-ID set encoder | Static graph | Dynamic graph |
|---|---:|---:|---:|---:|
| ~65k | 64,734 | 64,266 | 64,302 | 64,302 |
| ~83k | 83,246 | 83,642 | 83,274 | 83,274 |

Within-budget deviations are at most 1.5%. Static and dynamic graphs have identical parameter counts and message functions; only adjacency rules differ.

## Endpoints, statistics and decisions

- Primary endpoint: `effect-FDE@20`; secondary endpoint: `effect-ADE@20`.
- Contrasts: row-minus-set, set-minus-static, static-minus-dynamic, shuffled-dynamic-minus-same-episode-dynamic.
- Average test-episode errors over 5 training seeds before 10,000 paired bootstrap resamples.
- Independent dynamic-structure gains require consistent direction across both budgets and a pooled 95% CI excluding 0.
- Shuffled suffixes are factual-evidence negative controls only; without intervention-edge ground truth, the dynamic graph is not called a true causal graph.

## Implementation quality and environment

- 32 tests passed.
- Fixed use of metric distance thresholds on standardized coordinates and model initialization before seed setting.
- CuBLAS deterministic workspace configured; internal smoke-prediction arrays reproduced byte for byte.
- Python 3.12.7；PyTorch 2.9.1+cu130；NumPy 1.26.4。
- NVIDIA GeForce RTX 3070 Ti, 8192 MiB; driver 616.56.

