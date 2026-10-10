# Stable Entity-ID Audit Results

Overall decision: **PASS**.

| Scene | Requested events | Completed audits | Initial identity failures | Common-prefix state failures | Unregistered vehicles | Maximum prefix difference | Completed events with distance-sort row swaps |
|---|---:|---:|---:|---:|---:|---:|---:|
| merge-v0 | 100 | 100 | 0 | 0 | 0 | 0.0 | 94/100 |
| roundabout-v0 | 100 | 78 | 0 | 0 | 0 | 0.0 | 78/78 |

In Roundabout, 22 events terminated normally during the 12-step common prefix and did not enter post-intervention auditing; none had identity failures. No short-trajectory padding or relaxed termination rules were used.

## Conclusion

1. Under identical seeds, factual/counterfactual branches establish consistent within-episode entity IDs using ego first and deterministic creation order for other vehicles.
2. With identical action prefixes, per-ID object states match exactly across branches; maximum difference is 0.
3. Distance-sorted observations swap surrounding-vehicle rows in most events; a fixed row index cannot be interpreted directly as a fixed causal node.
4. Subsequent object models require explicit entity IDs, presence masks and object-centric tensors rather than flattened distance-sorted rows.

## Capacity-matched architecture comparison (2026-09-05)

Overall historical decision: **GO_STABLE_ID_ONLY**.

Compare four architectures by `effect-FDE@20` on 300 independent Merge test episodes with shared splits, 5 training seeds and two approximately matched budgets:

| Parameter budget | Distance-sorted GRU | Stable-ID set encoder | Static graph | Dynamic graph |
|---|---:|---:|---:|---:|
| ~65k | 1.5126 m | **1.3707 m** | 1.4054 m | 1.4578 m |
| ~83k | 1.3699 m | **1.2563 m** | 1.3232 m | 1.3854 m |

- Stable-ID set encoding beat the distance-sorted GRU at both budgets; pooled difference 0.1278 m, paired 95% CI [0.0500, 0.2095] m.
- Dynamic graphs underperformed static graphs after 200 epochs at both budgets; pooled disadvantage 0.0573 m, paired 95% CI [0.0231, 0.0913] m.
- Dynamic graphs showed advantages at 60 epochs and some 120-epoch budgets, reversing at 200, indicating convergence effects in early results.
- Shuffled factual suffixes significantly worsened dynamic-graph performance, supporting suffix information but not the graph construction itself.

At that stage, AC-Twin adopted stable-ID set encoding as its object representation. The current 60 m distance-threshold graph was retained as a failed boundary/control rather than an algorithmic innovation; independent Roundabout replication was the next gate.

## Scope boundaries

- IDs are stable within episodes only, not across events.
- Both scenes create 5 vehicle objects at reset. Dynamic vehicle creation would trigger explicit unregistered-entity failure until birth-event rules are added.
- Capacity-matched model gains were replicated in Merge and independent Roundabout.
- The current 60 m dynamic graph did not beat the static graph; independent graph-network gains are unsupported.

## Roundabout cross-scene replication (2026-09-05)

Overall decision: **PASS_STABLE_ID_CROSS_SCENE_REPLICATION**.

| Parameter budget | Distance-sorted GRU | Stable-ID set encoder | Relative improvement | Paired difference and 95% CI |
|---|---:|---:|---:|---:|
| ~65k | 2.7884 m | **2.4637 m** | 11.65% | 0.3247 [0.1344, 0.5172] m |
| ~83k | 2.7241 m | **2.4793 m** | 8.99% | 0.2448 [0.0671, 0.4291] m |

Pooled improvement was 0.2847 m, 95% CI [0.1145, 0.4605] m. Both scenes had consistent direction and CIs excluding 0, supporting adoption of the stable-ID object-set encoder at this historical stage. Gains were clearer for lane changes, deceleration, 3-step interventions and large effects. Stratum means did not reverse, but some smaller-stratum intervals crossed 0.

The pooled shuffled-suffix disadvantage was 0.8499 m, 95% CI [0.6495, 1.0498] m, supporting event-specific factual evidence. Complete results: `experiments/object_centric_stage/architecture_capacity_roundabout/`.
