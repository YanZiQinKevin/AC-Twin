# Stage 3R Handoff State

- Current state: `STAGE3R_DEV_NOT_READY_FOR_CONFIRMATION` (terminal).
- Frozen design: CEW convex teacher; Full/History experts; five-fold OOF; five 1:1 seed pairs; four router candidates; Phase 3 failure closes routing.
- Completed artifacts: approved design v0.2; implementation plan v0.1; 60 expert checkpoints; OOF teacher; Phase 0/1 reports; finalist receipt; power report; frozen 400-episode R-screen; 7-condition × 5-pair predictions; one adjudication report.
- Latest scientific result: strongest baseline=`end_to_end`; finalist relative improvement=`-0.0717%`, 80% CI=`[-0.2861%, 0.1514%]`; positive seed pairs=`2/5`; terminal state=`STAGE3R_DEV_NOT_READY_FOR_CONFIRMATION`.
- Frozen development result: `BEST_CONSTANT_W=0.60`.
- Frozen finalist: `lambda_CEW=1.0`, `lambda_rank=0.0`, `BEST_CONSTANT_W=0.60`, input dim `70`, Direct hidden width `63`.
- Provenance: predictions SHA256 `03e72b168a2fc0ab75733ead50cf80c8a3bf8382c876c115f865248272563242`; report SHA256 `0b226a33161ae78803bb7b891c64d0c5cd636f5288fd8fc867cb716dc23de612`; adjudication amendment SHA256 `b8633c60c8a108152c85ee9b1e6c25d23eaa82906fdcd8da1d27523f90bd8f5c`.
- Next action: no further learned-router experiment under this mainline; proceed only by revising the paper scope around Stage 2 phenomenon and Stage 3A Plus response predictability, or by separately approved non-routing work.
- Blocker/stop: registered Phase 3 stop rule triggered. Do not create Stage 3R Plus, change seeds, tune on R-screen, or rerun adjudication.
- One-shot resources: R-screen and its adjudication consumed; Gate C untouched.
- Latest verification: Stage 3R focused suite `48 passed in 3.44s`; full regression `811 passed in 186.21s`; frozen prediction, report and amendment hashes reverified.
