# Stage 1 Diagnostic Experiment: Development Go/No-Go Decision

> Document status: **COMPLETED_DEVELOPMENT_AUDIT**
> Date: 2026-09-07
> Protocol repair: `AC-TWIN-RFER-STAGE1-CLOSURE-2026-09-07-V1`
> Parent research plan: `AC-TWIN-RFER-2026-09-07-V1`
> Stage 1 implementation plan: `AC-TWIN-RFER-STAGE1-IMPL-2026-09-07-V1`
> **Development evidence, not confirmation. These data had been inspected repeatedly; this document does not establish `GATE_A_CONFIRMED`.**

---

## 1. Environment and interpreter

- Working directory: the original AC-Twin research root on the synchronized cloud drive.
- Python 3.13; PyTorch `2.14.0+cu132`; `torch.cuda.is_available()=True`; device `NVIDIA GeForce RTX 5070 Laptop GPU`.
- Full-test baseline: `python -m pytest tests -q -W error` → **260 passed**.
- The original research directory was not a Git repository; sealed artifacts were protected by SHA256 inventories.

## 2. Sealed training artifacts and no-retraining evidence

- Per scene: 3 conditions × 5 seeds (`11,22,33,44,55`) = 15 runs, with identical parameter counts (`145066`).
- Pre-repair snapshot: `stage1_sealed_artifact_hashes_before_protocol_fix.json`.
- **No retraining** during protocol repair: both scenes used `--reanalyze-only`, reporting `training_performed=false`, `checkpoint_modified=false`.
- SHA256 checks after repair: every `.pt`, `*_training.json`, `*_run.json` and existing `*_val_prediction.npy` / `*_test_prediction.npy` was **unchanged**.
- Only additions: 5 `uniform_suffix_seed*_val_prediction_with_frozen_shuffled_suffix.npy` files and associated `*_val_suffix_swap_metadata.json` per scene, reanalysis `report.json`, and original-report backup `report_SUPERSEDED_protocol_incomplete_20260907.json`.

## 3. Evidence for repairing three protocol gaps

1. **Gate S/S1 split leakage and permissive threshold**: nondegeneracy and ego controls now use validation only. Replace `M1 ≤ 1.5×M0` with validation `mean_ego_error(M1) < mean_ego_error(M0)`, reporting the point difference and episode-bootstrap 95% CI. Gate S records `data_scope=train_validation_only`; its code no longer reads `indices["test"]` or `*_test_prediction.npy`.
2. **S3 prediction evidence from fixed-M1 suffix swap**: load the same `uniform_suffix_seed{seed}.pt`, hold validation `H/AF/ACF` fixed, and switch only correct versus frozen-manifest donor suffixes, swapping features and masks together. Correct-channel re-inference must match sealed validation predictions within tolerance or stop. S3 no longer uses M2 weights/errors.
3. **S3 representation evidence with episode-bootstrap CIs**: fit ridge probes on split 0 representations/labels only; reconstruct per-vehicle motion summaries on validation. Compare against scene-mean and fixed-seed shuffled-label baselines using 10,000 episode-cluster resamples; both difference-CI upper bounds must be <0. Gate A additionally requires leave-one-seed/action/episode-out checks, fixed 1%/5% high-error deletion and manifest QA.

## 4. Merge: primary development scene

**Gate S = PASS** (S1/S2/S3 passed using train/validation only):

- S1: identical parameters (145066), finite losses and nondegenerate per-vehicle outputs; ego positive control `M1−M0 = -0.150`, 95% CI `[-0.250, -0.056]` (<0, consistent with useful ego suffix evidence), 300 independent episodes.
- S2: predictable per-vehicle response magnitude. `Spearman(R_hat, R)=0.614` (CI lower bound 0.597>0); model MAE `0.210` was below action+horizon `0.416` and distance-augmented baseline `0.430` (both difference-CI upper bounds <0).
- S3: fixed-M1 suffix swaps favored correct suffixes in zero/positive_low/positive_mid bins; 95% CI lower bounds for `L_wrong−L_correct` were 0.044/0.255/0.432, all >0. Both representation-probe baseline difference-CI upper bounds were <0.

**Gate A = PASS → `DEVELOPMENT_GO_TO_NEW_HOLDOUT`** (four primary numerical conditions and all robustness conditions):

- Low-response (zero+positive_low) `V_evt = 0.281`, 95% CI `[0.229, 0.332]` (>0);
- `V_evt_low − V_evt_high = 0.068`, 95% CI `[0.003, 0.131]` (>0, **marginal**, with lower bound near 0);
- `Spearman(R, eta_evt) = 0.287` `[0.259, 0.316]`（>0）；
- Covariate-adjusted `log1p(R)` coefficient `0.283` `[0.239, 0.332]` (>0 after horizon/cf_action/cf_duration/initial-distance/visibility adjustment);
- Robustness: leave-one-seed-out estimates `[0.078,0.064,0.060,0.071,0.066]` all >0; leave-one-action-type-out all >0 (action 3 weakest at 0.0023); leave-one-episode-out minimum 0.062, 0 reversals; top 1%/5% high-error deletion estimates 0.062/0.059, both >0; manifest QA passed (self/same_seed/cross_split/duplicate_donor all 0).

**Four-bin raw errors and V_evt (nonmonotonic; report as observed):**

| Response bin | V_evt | rho_no | meanL_no | meanL_suffix | meanL_shuffled | agent-time | episode |
|---|---|---|---|---|---|---|---|
| zero | 0.339 | 0.371 | 0.116 | 0.073 | 0.111 | 20241 | 300 |
| positive_low | 0.032 | 0.129 | 0.432 | 0.377 | 0.389 | 1343 | 188 |
| positive_mid | -0.025 | -0.004 | 0.933 | 0.937 | 0.915 | 1249 | 169 |
| positive_high | 0.213 | 0.236 | 3.039 | 2.321 | 2.949 | 1167 | 111 |

`V_evt` is **not strictly decreasing across bins**: positive_high exceeds positive_low/mid. The supported development statement is higher event-specific value in low versus high responses with an overall attenuation trend (conditional heterogeneity), **not** strict attenuation in every bin.

**Gate B = NOT_SUPPORTED**: high-response `Delta = L_no − L_suffix` was positive (0.627), with its 95% CI wholly above 0 and 111 independent episodes (≥30). Factual suffixes did **not** cause net negative transfer in this group. **Dual-edged evidence wording is prohibited.**

## 5. Roundabout: exploratory results and boundaries

**Gate S = FAIL (S2 failed) → `INCONCLUSIVE_MODEL_INSUFFICIENT`**; Gates A/B remain `BLOCKED`.

- S2 failure: per-vehicle response MAE (0.089) did **not** beat action+horizon (0.082) or distance-augmented baseline (0.085). Spearman was positive (0.152, CI lower bound 0.112), but the MAE condition failed.
- Interpretation is limited to **insufficient response-magnitude prediction by the current model in response-sparse Roundabout**. It does not refute H2 or provide a Gate A counterexample; do not select an enhancement based on it.
- Roundabout is exploratory replication and **cannot independently overturn or replace** the primary Merge decision. Its S2 failure is a cross-scene risk.

## 6. Acceptance checks

- Pre-repair sealed SHA256 inventory created; original report archived as superseded;
- No training or `--resume`; successful `--reanalyze-only` in both scenes with `training_performed=false`;
- S1 uses validation only without the 1.5× threshold; S3 swaps suffixes under one M1 checkpoint without M2, with episode-bootstrap CIs for both probe baselines;
- Gate A includes seed/action/episode omission and fixed 1%/5% high-error deletion; manifest QA is mandatory;
- All 260 tests passed with `-W error`; sealed `.pt`, log, metadata and original-prediction hashes unchanged;
- Gate order preserved: Roundabout Gate S FAIL leaves Gates A/B BLOCKED.

## 7. Unique Go/No-Go

**Merge：`DEVELOPMENT_GO_TO_NEW_HOLDOUT`。**
**Roundabout: `INCONCLUSIVE_MODEL_INSUFFICIENT` (exploratory; primary decision unchanged).**

This permits **designing** a fresh Stage 2 confirmatory holdout never used for development. It does **not** authorize router implementation or treating development data as confirmation.

## 8. Permitted claims, unsupported claims and historical next step

**Permitted, conditional development claims:**
- Existing development data show conditional heterogeneity/overall attenuation of event-specific suffix value with counterfactual response magnitude;
- The current Merge model meets minimum per-vehicle response-prediction and suffix-use requirements (Gate S passed).

**Unsupported/prohibited claims:**
- Do not claim `GATE_A_CONFIRMED` or a confirmed phenomenon;
- Do not interpret the response–evidence-value association as causal;
- Do not use dual-edged evidence wording (Gate B unsupported);
- Do not claim strict binwise monotonic attenuation;
- Do not develop a router before confirmatory Gate A passes;
- Do not tune parameters/thresholds, select cases or repair models on the new holdout.

**Historical next step:** draft the Stage 2 fresh-holdout implementation plan under decision-tree branch A, without collection/training, then stop for user approval.
