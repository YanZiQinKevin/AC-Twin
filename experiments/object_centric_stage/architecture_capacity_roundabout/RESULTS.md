# Roundabout Stable-ID Cross-Scene Replication Results

Overall decision: **PASS_STABLE_ID_CROSS_SCENE_REPLICATION**.

On frozen `roundabout-v0` object-centric data, two approximately matched parameter budgets, 5 training seeds and 300 independent test episodes, the stable-ID set encoder outperformed the distance-sorted GRU at both budgets.

| Parameter budget | Distance-sorted GRU | Stable-ID set encoder | Relative improvement | Paired GRU minus set difference and 95% CI |
|---|---:|---:|---:|---:|
| ~65k | 2.7884 m | **2.4637 m** | 11.65% | 0.3247 [0.1344, 0.5172] m |
| ~83k | 2.7241 m | **2.4793 m** | 8.99% | 0.2448 [0.0671, 0.4291] m |

After averaging episode errors across budgets and applying paired bootstrap, the pooled stable-ID set improvement was 0.2847 m, 95% CI [0.1145, 0.4605] m.

## Factual-suffix negative control

| Parameter budget | Shuffled-suffix set encoder | Same-episode-suffix set encoder | Paired shuffled minus same-episode difference and 95% CI |
|---|---:|---:|---:|
| ~65k | 3.0633 m | **2.4637 m** | 0.5997 [0.3978, 0.8042] m |
| ~83k | 3.5795 m | **2.4793 m** | 1.1002 [0.8737, 1.3312] m |

The pooled shuffled-suffix disadvantage was 0.8499 m, 95% CI [0.6495, 1.0498] m. The model's gain therefore still depends on same-event factual evidence, rather than object-encoder capacity alone.

## Robustness and heterogeneity

- Positive paired differences for 5/5 seeds at approximately 65k and 4/5 at approximately 83k; seed 33 showed a small reversal (-0.0480 m), but the budget's episode-level 95% CI remained above 0.
- Mean differences did not reverse for any of the five actions. Left/right lane changes and deceleration had 95% CIs above 0; idle and acceleration were positive with intervals crossing 0.
- The 3-step intervention improvement was 0.3825 m, 95% CI [0.1252, 0.6410]; the 1-step improvement was 0.1870 m, with its interval crossing 0.
- Large-effect improvement was 0.6113 m, 95% CI [0.2177, 1.0221]; zero- and medium-effect estimates were positive with intervals crossing 0.
- The 0-to-1 m small-effect group contained only 7 test episodes and supports no independent conclusion.

There was no systematic stratum-level directional reversal, but performance gains were heterogeneous and concentrated in lane changes, deceleration, longer interventions and large effects.

## Cross-scene conclusion at this historical stage

The pooled Merge improvement was 0.1278 m, 95% CI [0.0500, 0.2095]; independent Roundabout improvement was 0.2847 m, 95% CI [0.1145, 0.4605]. Both met the prespecified positive-direction and nonzero-CI criteria.

This supported adopting stable within-episode vehicle IDs plus object-set encoding as the AC-Twin representation module. It does not establish true exogenous-variable identification, structural innovation replay or multi-vehicle causal propagation.

## Training boundaries

Some set-encoder runs approached the 200-epoch cap. At approximately 65k, two seeds reached 200 epochs, with best epochs 191 and 199; at approximately 83k, one reached 200, with best epoch 188. Results use the frozen 200-epoch budget and best-validation checkpoints, without extra training to pursue favorable outcomes. Because the winning set models approached the cap, this does not show an extra training advantage for the GRU, but should still be disclosed.
