# Table S2｜Data roles, model transfer, and information-access contracts

> Status: `DRAFT_UPDATED / DATA_ROLE_COUNTS_VERIFIED / PENDING_FINAL_LAYOUT`
> Purpose: complete scope/validity and reproduction contracts. Main Tables 2–3 summarize information roles and sample sizes; this supplement documents training, transfer, inference access and evaluation truth.

## A. Experimental scopes and model relations

| Study scope | Scientific question | Independent unit / n | Predictor and audit-head relation | Claim boundary |
| --- | --- | --- | --- | --- |
| Factual-evidence analysis | Does factual-evidence value vary with realized response? | episode / 1,200 | Frozen evidence models; response is used only for post-hoc stratification | Supports response-conditioned evidence-value heterogeneity, not a decision policy claim |
| Merge ID decision audit | Can AC-Twin identify high-regret selections in distribution? | ten-action group / 2,000 | Merge-trained prediction stack and independently trained Direct audit head, both frozen before confirmation | Independent in-distribution confirmation |
| `JOINT_OOD` | Does the frozen Merge audit survive the prospectively specified generator mismatch? | ten-action group / 2,000 | Same frozen Merge prediction and Direct weights; no OOD retraining | Supports only the measured generator-mismatch profile |
| MetaDrive/Waymo | Can the same audit framework be reconstructed on another platform? | ten-action group / 2,000 | Predictor, action-risk components and Direct audit head retrained and frozen inside MetaDrive | Platform external confirmation with platform-internal retraining; not zero-shot weight transfer |

## B. Information-access contract

| Information | Audit training | Confirmation inference | Standard baselines | Final evaluation |
| --- | --- | --- | --- | --- |
| Frozen predicted trajectories and action costs | yes | yes | where defined | yes |
| Frozen action-risk outputs and cross-seed uncertainty | yes | yes | method-specific legal subset | yes |
| Pre-intervention geometry, action identity/duration and intervention time | yes | yes | method-specific legal subset | yes |
| Paired counterfactual outcomes and `J_true` | label construction only | **no** | **no** | yes |
| Oracle action and true decision regret | label construction only | **no** | **no** | yes |
| Realized response and realized prediction error | historical task labels/analysis only | **no** | **no** | analysis/evaluation only |

Training provenance: among five action-risk channels, `COMBINED_RISK_HEAD`, `GEOMETRY_ONLY` and `RESPONSE_ONLY` are frozen models supervised by true effect errors on development train. `UNCERTAINTY_ONLY` and `DISAGREEMENT_ONLY` directly aggregate frozen predictions, not two additional fitted networks. Response components use true-response supervision/calibration only in permitted development roles; evaluated groups consume predictions only. MetaDrive uses platform-internal train/validation for fitting, selection/calibration, then freezing. Confirmation truth never enters upstream fitting or inference outputs. Training outputs are not uniformly OOF and are not independent evidence.

## C. Manuscript boundary

- Paired counterfactual truth and true regret never enter confirmation inference.
- Merge ID and `JOINT_OOD` share frozen Merge weights; MetaDrive/Waymo uses a corresponding platform-internally retrained/frozen pipeline.
- Per-file SHA256, create-once receipts and state machines belong in the reproduction package and frozen experiment archives.
- Performance results are in Table 6; shared-development adapted-method comparisons are in Table 7.

## D. Data-role resource map (verified 2026-10-02)

Roles refer to the measurement model or Direct audit head used in the final paper evidence, not cumulative historical data use. No separate monitor means validation was explicitly used, not missing samples. Counts are complete episodes/groups.

| Evidence scope | Independent unit | Total train | Independent monitor | Validation | Development evaluation / screen / qualification pilot | Confirm |
| --- | --- | ---: | --- | --- | --- | ---: |
| Merge factual-evidence value | episode | 1,400 | No separate set; validation used | 300 (includes checkpoint selection) | 300 existing development-evaluation episodes | 1,200 |
| Merge ten-action Direct audit | Complete ten-action group | 5,000 | 500 within train; remaining 4,500 for fitting | 1,500 | 8,000 one-use development-screen groups | 2,000 |
| `JOINT_OOD` | Complete ten-action group | No new data; reuse Merge | No new data; reuse Merge | No new data; reuse Merge | 300 qualification-pilot groups for each of three profiles | 2,000 |
| MetaDrive ten-action Direct audit | Complete ten-action group | 5,000 | No separate set; validation used | 1,500 (early stopping and configuration selection) | 1,000 final-route development-screen groups | 2,000 |
| Adapted published-method comparison | Complete ten-action group | 5,000, reused Merge development train | No separate monitor for adaptations; Direct retains its frozen split | 1,500, reused Merge validation | No new screen | 0 (independent confirmation set not created) |

- Merge's 5,000 training groups comprise 4,500 main + 500 monitor, not 5,500, and are separated from validation 1,500, screen 8,000 and confirm 2,000.
- `JOINT_OOD` adds no training data or weights. Three 300-group pilots qualify distributions/tasks and mechanically select the profile, not audit-method effects. Its 2,000-group confirmation is fresh.
- MetaDrive's 5,000 train and 1,500 validation groups are existing platform-internal development data, repaired/retrained in D2E and reused read-only in D2F. No extra 500-group monitor exists. Final screen 1,000 and confirm 2,000 use fresh IDs. The old failed D2 screen of 4,000 groups is preserved and not reused; D2R/D2E collected no new screen/confirm after power stops.
- Stage 5 reuses R2 train 5,000 and validation 1,500. Adaptations have no independent early-stopping monitor; checkpoints follow fitting-sample training loss. Conformal divides train into disjoint 3,988 fit and 1,012 calibration groups, not extra data. Direct is not retrained, retaining its 4,500/500 split and frozen weights.
- Stage 2's original 300 test episodes became development evaluation, not independent confirmation; 1,200 fresh episodes form its confirmation set. Response qualification, upstream risk development and early exploration retain separate historical roles and do not increase confirmation n.
- None of the four main-text confirmation sets participates in development; the three decision-risk sets do not overlap. Same-data reuse means legitimate reuse of development sources, not fresh independent observations or authorization to reuse consumed confirmation.

### Sources for role counts

| Item and data fields | Frozen source |
| --- | --- |
| Stage 1 Merge 1,400 / 300 / 300: train / validation / historical development test | `experiments/stage1_evidence_value/merge/report.json`: `n_train,n_validation,n_test` |
| Stage 2 confirm 1,200 episode | `experiments/stage2_confirmatory/merge/report.json`；`data/object_centric_stage/stage2_merge_1200/collection_receipt.json` |
| Merge Direct total train 5,000; 4,500 main / 500 monitor | `experiments/stage4_reliability/stage4c_r2/development/models/fit_receipt.json`: `n_train_groups,main_indices,monitor_indices` |
| Merge validation 1,500; screen 8,000 | Same development directory, `validation_collection_receipt.json` / `screen_collection_receipt.json`: `accepted_groups` |
| Merge confirm 2,000 | `experiments/stage4_reliability/stage4cr2_confirm/collection/collection_receipt.json` and formal confirmation report |
| OOD three pilots, 300 each | `experiments/stage4_reliability/stage4d_ood/pilot/{behavior,geometry,joint}/collection_receipt.json`: `accepted_groups` |
| OOD confirm 2,000 | `experiments/stage4_reliability/stage4d_ood/collection/collection_receipt.json` and formal confirmation report |
| MetaDrive train 5,000 / validation 1,500 | `experiments/stage4_reliability/stage4d_external/roles/{train,validation}/collection_receipt.json`：`accepted`；D2E `development/fit_risk_report.json` |
| MetaDrive no independent monitor; validation early stopping | `stage4d_external_training.py::fit_five_direct_models`; D2E `riskfit_EGO_TRAJECTORY_PLUS_RANK/development/models/fit_receipt.json` |
| MetaDrive final screen 1,000 / confirm 2,000 | `experiments/stage4_reliability/stage4d_d2f/development/development_screen_report.json` / `confirm/confirm_report.json`: `n_groups` |
| Stage 5 shared sources and 3,988 / 1,012 fit/calibration | Frozen Stage 5 v0.3 plan; `experiments/stage5_sota_comparison/development/models/fit_receipts.json`: conformal `notes.fit_groups,calibration_groups` |
| Stage 5 independent confirm never created | `experiments/stage5_sota_comparison/power/power_report.json` and preserved terminal records |

The Markdown is the current data-role document. `tableS2_historical_source_data.xlsx` remains a historical workbook and is not claimed to incorporate these updated fields or final submission layout.

## Sources

- Active plan §24.2 and §25.
- `experiments/stage4_reliability/stage4cr2_confirm/reports/stage4cr2_confirmatory_report.json`.
- `experiments/stage4_reliability/stage4d_ood/reports/stage4d_ood_confirmatory_report.json`.
- `experiments/stage4_reliability/stage4d_d2f/confirm/confirm_report.json` and `source_binding.json`.
- Historical data-role bottom sheet preserved as `paper/tables/tableS2/tableS2_historical_source_data.xlsx`.
