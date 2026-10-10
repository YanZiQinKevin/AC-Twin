# Cost Interpretation of the Fixed Fig. 5 Cases

> 2026-10-04: descriptive decomposition of the two cases in the author-fixed PNG using frozen trajectories and the fixed cost function. No case reselection, redrawing, rescoring or adjudication; no new population-level inference.

| Panel / group | Branch | Fixed action | Longitudinal progress (m) | C_int | C_eff | J_true |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| c / 550 | Selected | FASTER × 3 | 2.0131 | 0.657523 | 0.999994 | 0.726017 |
| c / 550 | oracle | LEFT × 1 | 3.9419 | 0.346960 | 0 | 0.277568 |
| d / 1821 | Selected | FASTER × 3 | 4.8850 | 0.000487 | 1.000000 (rounded) | 0.200389 |
| d / 1821 | oracle | RIGHT × 3 | 22.1717 | 0.000007 | 0 | 0.000005 |

In (c), higher proximity cost and lower progress jointly produce regret 0.448449. Selected-branch maximum proximity occurs at predicted frame 16, vehicle slot 4, with longitudinal/lateral separation approximately 4.5755/0.0671 m. The oracle maximum occurs at frame 18, slot 4, approximately 3.9271/2.4497 m. The anisotropic cost is not determined by Euclidean nearest distance alone.

In (d), regret 0.200384 mainly reflects efficiency cost: selected progress 4.8850 m versus oracle progress 22.1717 m, with small proximity terms in both branches. This false negative misses high cost under the defined decision objective; it is not directly a missed hazard or collision.

Stars still mark maximum ego-branch trajectory separation (frame 20 in both cases), not maximum proximity cost. No markers were added to the approved figure.

Sources: fixed group indices in `Fig5_render_receipt.json#frozen_cases`; frozen trajectories in `experiments/stage4_reliability/stage4d_d2f/roles/confirm/groups.npz`; action indices/J_true in `confirm/confirm_group_quantities.npz`; fixed cost in `code/ac_twin/stage4c_objective.py`. Pure decomposition for groups 550 and 1821 reproduced ten-action J_true with maximum error 0.0, without bootstrap, model inference or new adjudication. The inspection script remains in the original research workspace.
