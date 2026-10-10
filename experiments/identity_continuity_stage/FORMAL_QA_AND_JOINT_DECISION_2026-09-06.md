# Formal Identity-Continuity Experiment: Joint QA, Statistics and Decision

Date: 2026-09-06
Final engineering status: `QA_PASS`
Scientific decision: `INCONCLUSIVE`

## 1. Meaning of the decision

The fair comparison did not provide sufficient evidence for a stable independent performance gain from vehicle-ID continuity. This is neither an experimental failure nor evidence that stable IDs are useless. Across two scenes, two capacities and five training seeds, the advantage depended on scene, capacity and seed, failing the frozen cross-scene decision gate.

Do not record `PASS_IDENTITY_CONTINUITY`, `SCENE_LIMITED_IDENTITY_EVIDENCE` or `DISTANCE_ONLY_DIFFERENCE_NOT_IDENTITY`. At Roundabout 83k, the primary `set_pool - stable` CI lower bound exceeded 0, so the four scene-by-capacity combinations did not all show no reliable difference between set pooling and stable identity. The frozen preference rule for `PREFER_SET_POOL` therefore also failed. The remaining valid status is `INCONCLUSIVE`.

## 2. Formal QA

All nine checks were repeated on formal outputs and passed:

| QA item | Result | Main evidence |
|---|---|---|
| `fairness` | PASS | Shared object features, samples, train/validation/test indices, normalization, actions, targets and training settings; identical parameter counts for the three temporal-identity conditions at each capacity |
| `object_multiset` | PASS | Exhaustive framewise history/future checks on source and frozen data in both scenes; sorting and permutation neither added nor removed or modified object rows |
| `ego_slot` | PASS | Slot 0 matched the stable input elementwise in all four conditions |
| `permutation` | PASS | Deterministic reproducible shuffling changed non-ego cross-frame slots: 269,654 rows in Merge and 270,129 in Roundabout; distance sorting used nondecreasing original metric distance |
| `data_consistency` | PASS | Both formal preflights passed; source/frozen scenes, dimensions, features, capacity, history, horizon and seed separation matched |
| `capacity` | PASS | Deviations from 65k/83k targets were 0.773%–1.185%, below the frozen 1.5% cap |
| `finite_values` | PASS | Across scenes, 80 weights, 80 logs, 80 per-seed predictions, 80 run metadata records, two aggregate predictions and report statistics contained no NaN/Inf |
| `resume` | PASS | Repeating the original command in each scene reported `resumed=complete` for 40/40 runs; exit code 0 |
| `holdout_collection` | PASS | Both frozen sets passed integrity rechecks, with 1,000 episodes each; hashes, configuration bindings and seed separation matched frozen QA records |

The complete test command treated warnings as errors: `181 passed in 39.75s`.

### Data and experiment signatures

| Scene | Source data SHA-256 | Frozen test SHA-256 | Formal preflight signature |
|---|---|---|---|
| Merge | `eb6122ac62af20219fc0b27d346740a935587ff32d82e4000127a50a41466e2b` | `c533be651eeefffdc38c0d0929bedaa9e24da498757062ab86f7292da9c542d5` | `248b751f14fa73f45457a616ed711d2cd8c5de33e7f245b1dd3579914df2bef4` |
| Roundabout | `7e50bf4a17f55d6666549d7743b1d2e5924f5dfa10411f94bd62431146de8672` | `6bfc294488a399b3fe9d6ab708693446386666799e7a916b9614d38dd82920d5` | `72c2da78484a16badb32e4a73cbe3d98e0a8710ae88eef02f707582dd11de6ce` |

Each scene used 1,400 source-training episodes, 300 validation episodes and all 1,000 independent frozen test episodes. Each contained 4 conditions × 2 capacities × 5 seeds = 40 runs.

## 3. Mean errors by condition

Values average five training seeds and 1,000 frozen episodes. Lower is better.

| Scene | Capacity | Condition | ADE | FDE@20 |
|---|---:|---|---:|---:|
| Merge | 65k | stable_temporal | 0.8214 | 1.5132 |
| Merge | 65k | distance_temporal | 0.8373 | 1.5393 |
| Merge | 65k | permuted_temporal | 0.8339 | 1.5253 |
| Merge | 65k | set_pool | 0.8543 | 1.5521 |
| Merge | 83k | stable_temporal | 0.7831 | 1.4687 |
| Merge | 83k | distance_temporal | 0.7827 | 1.4709 |
| Merge | 83k | permuted_temporal | 0.7678 | 1.4226 |
| Merge | 83k | set_pool | 0.8157 | 1.4811 |
| Roundabout | 65k | stable_temporal | 1.4836 | 2.6407 |
| Roundabout | 65k | distance_temporal | 1.4478 | 2.6138 |
| Roundabout | 65k | permuted_temporal | 1.5094 | 2.6921 |
| Roundabout | 65k | set_pool | 1.5724 | 2.7770 |
| Roundabout | 83k | stable_temporal | 1.3443 | 2.4343 |
| Roundabout | 83k | distance_temporal | 1.4234 | 2.5418 |
| Roundabout | 83k | permuted_temporal | 1.5113 | 2.6983 |
| Roundabout | 83k | set_pool | 1.4588 | 2.5983 |

## 4. Two-level paired statistics for the primary endpoint

Differences are control FDE minus stable FDE; positive values favor stable identity. Each comparison uses five aligned seeds × the same 1,000 frozen episodes, 10,000 two-level paired bootstrap resamples and 95% CIs. Reliable positivity additionally requires at least four positive seed means and positive four-seed means after omitting any seed.

| Scene | Capacity | Contrast | Mean difference | 95% CI | Positive seeds | Positive after omitting any seed | Reliably positive |
|---|---:|---|---:|---|---:|---|---|
| Merge | 65k | permuted − stable | +0.0121 | [-0.0826, 0.1244] | 2/5 | No | No |
| Merge | 65k | distance − stable | +0.0261 | [-0.0487, 0.1055] | 3/5 | No | No |
| Merge | 65k | set_pool − stable | +0.0389 | [-0.0680, 0.1440] | 3/5 | No | No |
| Merge | 83k | permuted − stable | -0.0460 | [-0.0816, -0.0122] | 0/5 | No | No; opposite direction |
| Merge | 83k | distance − stable | +0.0023 | [-0.0389, 0.0447] | 2/5 | No | No |
| Merge | 83k | set_pool − stable | +0.0124 | [-0.0762, 0.1124] | 2/5 | No | No |
| Roundabout | 65k | permuted − stable | +0.0514 | [-0.1118, 0.2093] | 3/5 | No | No |
| Roundabout | 65k | distance − stable | -0.0269 | [-0.1601, 0.1140] | 1/5 | No | No |
| Roundabout | 65k | set_pool − stable | +0.1363 | [-0.0082, 0.2881] | 4/5 | Yes | No; CI crosses 0 |
| Roundabout | 83k | permuted − stable | +0.2640 | [0.1198, 0.3945] | 5/5 | Yes | Yes |
| Roundabout | 83k | distance − stable | +0.1075 | [-0.0030, 0.2137] | 4/5 | Yes | No; CI slightly crosses 0 |
| Roundabout | 83k | set_pool − stable | +0.1640 | [0.0220, 0.3334] | 5/5 | Yes | Yes |

Two-level statistics were recomputed from `[training seed, frozen episode]` error arrays and matched both original reports exactly.

## 5. Interpretation and boundaries

The clearest positive signal occurred at Roundabout 83k: stable identity outperformed ID permutation and set pooling, but its FDE CI lower bound against distance sorting was -0.0030, below the frozen gate. No Roundabout 65k comparison was reliably positive. Merge had no reliably positive comparison, and at 83k ID permutation had significantly lower FDE than stable identity. Results depend on scene, capacity and training seed.

Secondary ADE broadly agreed: all three Roundabout 83k intervals were positive. However, FDE@20 was the frozen primary endpoint; ADE cannot replace FDE to change the decision.

This study addresses independent representation-level performance only. It does not establish contributions to causal identifiability, explicit abduction, multi-vehicle response, safety gains or cross-scene generalization. Frozen sets contain only events with complete 20-step futures in both branches and do not represent early termination or other safety-critical events.

## 6. Historical recommendations

Without altering this frozen decision, diagnose why gains were clearest only at Roundabout 83k: missing slots/vehicle entry and exit, capacity-dependent optimization stability, and possible regularization from permutation in Merge 83k. Use development/validation data or a new preregistered study; do not tune retrospectively and reuse this frozen set to claim confirmation.

