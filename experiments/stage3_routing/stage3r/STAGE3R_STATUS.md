# Stage 3R Status

> Plan: `AC-TWIN-RFER-STAGE3R-2026-09-10-V0.1`  
> State: `STAGE3R_DEV_NOT_READY_FOR_CONFIRMATION`（终态）

- Pre-long-run regression: `798 passed in 195.17s`.
- Stage 3R focused suite: `35 passed`; real-data OOF engineering smoke passed.
- OOF experts: `60/60`, 1400 train episodes with exactly-once OOF coverage, 300 validation episodes; both archive hashes independently verified.
- Phase 0: `STAGE3R_SOFT_HEADROOM_CONFIRMED`; point improvement `23.91%`, 80% CI `[22.68%, 25.19%]`, `5/5` pairs positive, `BEST_CONSTANT_W=0.60`.
- Router candidates: 30/30 complete; five response-feature archives use two-fold OOF training features and frozen production validation features.
- Phase 1: `STAGE3R_VALUE_WEIGHT_PREDICTABLE`; Spearman 80% CI lower `0.0828`, PR-AUC-minus-prevalence 80% CI lower `0.00246`, `5/5` pairs positive; validation effect-ADE improves `0.459%` over `BEST_CONSTANT_W`.
- Finalist: frozen once at `lambda_CEW=1.0`, `lambda_rank=0.0`; input dim `70`, Direct hidden width `63`.
- Power: `STAGE3R_RSCREEN_POWER_FEASIBLE`; frozen `n=400`.
- R-screen: 400/400 accepted from 630 attempts, historical seed overlap `0`; frozen inference completed for 7 conditions × 5 seed pairs.
- Single adjudication: consumed once. The frozen finalist improved over `BEST_CONSTANT_W` by `0.554%` (80% CI `[0.300%, 0.817%]`) but not over the strongest baseline `end_to_end`: `-0.072%` (80% CI `[-0.286%, 0.151%]`), with only `2/5` seed pairs positive.
- Formal report: `rscreen/rscreen_report.json`, SHA256 `0b226a33161ae78803bb7b891c64d0c5cd636f5288fd8fc867cb716dc23de612`.
- A pre-report NumPy stratum-indexing dimension error was corrected without changing data, models, seeds, thresholds or frozen predictions; amendment SHA256 `b8633c60c8a108152c85ee9b1e6c25d23eaa82906fdcd8da1d27523f90bd8f5c` is recorded in the report.
- Registered consequence: the learned routing method mainline is closed; no Stage 3R Plus and no Gate C. Stage 2 and Stage 3A Plus evidence remain valid within their existing boundaries.
- Final verification: Stage 3R focused suite `48 passed in 3.44s`; full regression `811 passed in 186.21s`.
