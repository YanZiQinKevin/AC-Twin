# Table 7. Comparison with adapted published methods on the shared development benchmark

> **Scope:** `ADAPTIVE_DEVELOPMENT_ONLY`; Stage 5 terminal state `STAGE5_MERGE_COMPARISON_UNDERPOWERED`. The comparison holdout was never created. The table supports a transparent development benchmark, not a confirmatory SOTA-superiority claim.

| Method | Venue | AURC-regret ↓ | Regret@80 ↓ | PR-AUC − prevalence ↑ | Spearman ↑ | Params/seed | Approx. forward FLOPs/group (MAC×2) ↓ | Inference (ms/group) ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConfidNet (adapted) [`corbiere2019confidnet`] | NeurIPS 2019 (volume 32) | 0.0165 | 0.0186 | 0.0954 | 0.3227 | 9665 | 19330 | 1.866700 |
| SelectiveNet (adapted) [`geifman2019selectivenet`] | ICML 2019 (PMLR 97:2151-2159) | 0.0229 | 0.0214 | 0.0310 | 0.1524 | 23139 | 46278 | 3.027450 |
| CCTR (adapted) [`cao2024cctr`] | AAAI 2024 (38(19):20949-20957) | 0.0128 | 0.0180 | 0.1610 | 0.2830 | 11522 | 23044 | 1.577550 |
| Conformal regression + reject (adapted) [`johansson2024conformal`] | COPA 2024 (PMLR 230:277-294) | 0.0133 | 0.0181 | 0.1223 | 0.2928 | 11201 | 22402 | 0.043200 |
| RwR-kNNRej (adapted) [`li2024rwr`] | AISTATS 2024 (PMLR 238:1126-1134) | 0.0195 | 0.0232 | 0.0050 | 0.0862 | 11201 | 22402 | 4.485150 |
|  |  |  |  |  |  |  |  |  |
| MARGIN_ONLY | — | 0.0208 | 0.0212 | 0.0519 | 0.0748 | — | — | — |
| SEED_MARGIN_STD | — | 0.0218 | 0.0224 | -0.0064 | 0.0033 | — | — | — |
| SEED_VOTE_ENTROPY | — | 0.0190 | 0.0205 | 0.0542 | 0.1161 | — | — | — |
|  |  |  |  |  |  |  |  |  |
| **AC-Twin audit** | This work | **0.0110** | **0.0148** | **0.3314** | **0.4301** | 15740 | 31480 | 2.357200 |

All learned rows use five model seeds. Parameters are reported per seed. Approximate forward FLOPs are the per-seed simplified parameterized-forward estimate MAC×2 per group; they exclude the shared upstream predictor and method-specific non-parametric search or calibration operations. Inference time is the median single-group (batch size 1) five-seed ensemble risk-head time after model/input preload on an NVIDIA GeForce RTX 3070 Ti; host materialisation is included and the shared upstream predictor is excluded. Internal baselines are closed-form scores and are therefore not assigned trainable parameters or risk-head latency. Full grids, secondary metrics, calibration diagnostics, pairwise intervals and power results remain in `TABLE_7_DETAILS.md` as audit material, not additional manuscript comparison tables.

Reference keys map to `paper/references/REFERENCE_LEDGER.md`; manuscript numbers will be assigned only after the bibliography order is fixed.
