# Stage 4D-D2F: Six Case Panels

These independent panels show two frozen cases and may be assembled manually. Each case presents the factual scene and future, the selected-action counterfactual future, and the oracle-action counterfactual future. Matplotlib renders ScenarioNet/Waymo maps and frozen trajectories; these are neither native MetaDrive screenshots nor AI-generated images.

## Successful risk detection (suggested top row)

This best-of-K example was chosen by a frozen rule from **205 eligible confirmation candidates**. True regret is **0.4484**. Direct's score is at the **83.5th percentile**, entering the top 20% risk group; the frozen baseline is at the **69.8th percentile**, outside that group. Success means **warning about a high-regret decision**, not choosing the correct action.

| Panel | Description |
| --- | --- |
| [success_01_factual.png](success_01_factual.png) | Road, vehicles and factual future at intervention; common reference for the other panels. Gray is the factual ego future. |
| [success_02_selected.png](success_02_selected.png) | Counterfactual future for selected **FASTER × 3 steps**. Ego trajectory is red. The yellow star marks maximum separation from the oracle ego trajectory. |
| [success_03_oracle.png](success_03_oracle.png) | Counterfactual future from the same start under oracle **LEFT × 1 step**, evaluated by the frozen objective. Ego trajectory is green; comparison shows selected/oracle consequences. |

## Missed high-regret case (suggested bottom row)

The frozen rule selected this worst false negative from **70 eligible missed cases**. True regret **0.2004** meets the high-regret criterion, but Direct's score is only at the **1.6th percentile**, without a high-risk warning; the baseline is at the **45.7th percentile**. Failure means **missing a high-regret decision**, not simulation failure or a collision.

| Panel | Description |
| --- | --- |
| [failure_01_factual.png](failure_01_factual.png) | Road, vehicles and factual future of the second scene at intervention; shared reference for its two counterfactual branches. |
| [failure_02_selected.png](failure_02_selected.png) | Selected **FASTER × 3 steps** counterfactual future, with red ego trajectory. The decision has high regret despite a low Direct risk score. |
| [failure_03_oracle.png](failure_03_oracle.png) | Oracle **RIGHT × 3 steps** future from the same start, with green ego trajectory, illustrating the missed consequences. |

## Assembly and interpretation boundaries

- Coordinates are ego-heading aligned: forward distance horizontally and leftward distance vertically, in metres. Preserve identical axes within each triplet. Gray denotes history; light blue denotes other-vehicle futures. `EGO` and `V1`–`V4` label vehicles at intervention.
- Yellow stars mark only maximum selected/oracle ego separation, at step 20/20 in both cases, not collision or accident locations.
- Regret is defined by the frozen objective, not collision probability or physical harm. Risk percentiles are empirical confirmation-set ranks, not calibrated probabilities.
- The success is a nonrepresentative **best-of-205** illustration; the failure is also a prespecified boundary case. Both were selected after formal adjudication for qualitative explanation only, not model selection, statistical tests or population-effect estimation. Cite full-set tables/CIs for overall conclusions; do not infer success rates from these panels.

See [selection_receipt.json](../provenance/selection_receipt.json) and [panels_manifest.json](../provenance/panels_manifest.json) for data and selection provenance.
