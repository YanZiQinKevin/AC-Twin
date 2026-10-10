# Stage 3R R-screen Readiness Checklist

> Purpose: `stage3r_rscreen_development_test`  
> This screen is a one-shot development resource. It is not Gate C.

- [x] Phase 0 status is `STAGE3R_SOFT_HEADROOM_CONFIRMED`.
- [x] Phase 1 status is `STAGE3R_VALUE_WEIGHT_PREDICTABLE`.
- [x] `finalist_receipt.json` exists and is write-once.
- [x] `power_report.json` is feasible and freezes `n=400` at MDES 5% and joint power at least 0.80.
- [x] `model_lock.json`, analysis code hashes and all five finalist checkpoint hashes verify.
- [x] R-screen config, model lock, power report, finalist receipt and seed schedule hashes agree.
- [x] Seed schedule is unique, int64, in `[1, 2_000_000_000)`, and has zero overlap with all historical data under `data/`.
- [x] Output lives only under `data/object_centric_stage/stage3r_rscreen` and `experiments/stage3_routing/stage3r/rscreen`.
- [x] Stage 2, Stage 3A Plus, old Stage 3B/Plus and Gate C paths are not writable targets.
- [x] Full regression passes after collector/inference implementation and before collection (`810 passed`).
- [x] Collection acceptance rules do not inspect response, CEW, expert loss or model prediction.
- [x] Inference emits no contrasts; adjudication runs once and refuses overwrite.

> Completed 2026-09-11. The one-shot screen and adjudication are consumed; final status is `STAGE3R_DEV_NOT_READY_FOR_CONFIRMATION`.

The executable readiness validator, not manual checkbox marks, is authoritative.
