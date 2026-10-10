# Merge Capacity-Matched Architecture Comparison Results

Completed on: 2026-09-05

## Conclusion

Merge decision: **stable-ID object representation passed; the current distance-gated dynamic graph did not pass.**

Under a uniform final 200-epoch budget, the stable-ID set encoder outperformed the capacity-matched distance-sorted GRU at approximately 65k and 83k parameters. Conversely, the 60 m distance-gated dynamic graph underperformed the parameter-identical static fully connected message-passing model at both budgets. Dynamic adjacency construction is not supported as an independent method contribution.

## Main results

| Parameter budget | Model | Parameters | effect-ADE@20 | effect-FDE@20 | FDE SD across seeds |
|---|---|---:|---:|---:|---:|
| ~65k | Distance-sorted GRU | 64,734 | 0.8725 | 1.5126 | 0.1087 |
| ~65k | Stable-ID set encoder | 64,266 | 0.8053 | **1.3707** | 0.1367 |
| ~65k | Static fully connected graph | 64,302 | 0.8035 | 1.4054 | 0.0805 |
| ~65k | 60 m dynamic graph | 64,302 | 0.8376 | 1.4578 | 0.1093 |
| ~65k | 60 m dynamic graph + shuffled suffix | 64,302 | 1.2761 | 2.2716 | 0.1998 |
| ~83k | Distance-sorted GRU | 83,246 | 0.7941 | 1.3699 | 0.1197 |
| ~83k | Stable-ID set encoder | 83,642 | 0.7420 | **1.2563** | 0.0238 |
| ~83k | Static fully connected graph | 83,274 | 0.7678 | 1.3232 | 0.0861 |
| ~83k | 60 m dynamic graph | 83,274 | 0.7911 | 1.3854 | 0.0754 |
| ~83k | 60 m dynamic graph + shuffled suffix | 83,274 | 1.1919 | 2.1450 | 0.3196 |

## Prespecified paired contrasts

Positive differences favor the candidate model over its reference.

| Parameter budget | Contrast | Paired difference | 95% CI | Decision |
|---|---|---:|---:|---|
| ~65k | Distance-sorted minus stable-ID set | 0.1419 m | [0.0524, 0.2345] | Stable-ID set better |
| ~83k | Distance-sorted minus stable-ID set | 0.1137 m | [0.0315, 0.1960] | Stable-ID set better |
| ~65k | Stable-ID set minus static graph | -0.0346 m | [-0.1065, 0.0327] | No evidence of static-graph gain |
| ~83k | Stable-ID set minus static graph | -0.0669 m | [-0.1360, -0.0008] | Static graph worse |
| ~65k | Static minus dynamic graph | -0.0524 m | [-0.0931, -0.0100] | Dynamic graph worse |
| ~83k | Static minus dynamic graph | -0.0622 m | [-0.1023, -0.0231] | Dynamic graph worse |
| ~65k | Shuffled minus same-episode dynamic graph | 0.8138 m | [0.6412, 0.9949] | Factual-evidence negative control passed |
| ~83k | Shuffled minus same-episode dynamic graph | 0.7596 m | [0.5909, 0.9381] | Factual-evidence negative control passed |

Exploratory pooled results after averaging episode errors across budgets:

- Distance-sorted minus stable-ID set: 0.1278 m, 95% CI [0.0500, 0.2095];
- Static minus dynamic graph: -0.0573 m, 95% CI [-0.0913, -0.0231].

## Training-budget sensitivity

| Budget | 65k static minus dynamic graph | 83k static minus dynamic graph |
|---|---:|---:|
| 60 epoch | 0.0340 [-0.0025, 0.0698] | 0.0566 [0.0281, 0.0850] |
| 120 epoch | -0.0176 [-0.0528, 0.0180] | 0.0520 [0.0043, 0.1005] |
| 200 epoch | -0.0524 [-0.0931, -0.0100] | -0.0622 [-0.1023, -0.0231] |

Positive values favor the dynamic graph; negative values favor the static graph. Direction reverses with training budget, indicating different convergence rates. Complete 60- and 120-epoch screens are preserved; selecting only early budgets favorable to the dynamic graph is not justified.

## Research recommendations at this historical stage

1. Exclude the current distance-gated dynamic graph from primary contributions and stop categorical claims that it outperforms set encoding based on early runs.
2. Retain stable within-episode vehicle IDs and object-centric set encoding, which outperform distance-sorted inputs at both final budgets.
3. Replicate the distance-sorted versus stable-ID set contrast in `roundabout-v0`; retain the dynamic graph as a diagnostic control, without extending to more complex graph modules on this basis.
4. Until replication, the stable-ID gain is limited to Merge evidence.
5. Factual suffixes again outperform shuffled suffixes; this result is unaffected by the architecture-selection failure.

## Boundaries

- Several static/dynamic graph seeds had best validation epochs near the 200-epoch cap, so theoretical optimality is not established. However, the absence of a stable dynamic-graph gain across budgets is sufficient to reject the innovation claim for this implementation.
- This experiment evaluates only the current 60 m hard distance gate; it does not reject every dynamic-graph method.
- Only ego counterfactual effects are evaluated; surrounding-vehicle response propagation and structural ground truth are not evaluated.

