# Stage 1 Diagnostic Experiment: Frozen Protocol

> Document status: **COMPLETED / DEVELOPMENT CLOSURE COMPLETE**
> Implementation plan: `AC-TWIN-RFER-STAGE1-IMPL-2026-09-07-V1`
> Closure plan: `AC-TWIN-RFER-STAGE1-CLOSURE-2026-09-07-V1`
> Parent plan: `AC-TWIN-RFER-2026-09-07-V1`
> Date: 2026-09-07
> Results: see `STAGE1_DEVELOPMENT_GO_NO_GO_2026-09-07.md` (merge=`DEVELOPMENT_GO_TO_NEW_HOLDOUT`; roundabout=`INCONCLUSIVE_MODEL_INSUFFICIENT`). This file freezes the protocol and commands; decisions are authoritative in the result document.
> Scope: per-vehicle/per-time effect prediction, M0/M1/M2, fair shuffling, endpoints, Gate S and development Gates A/B. No router, response head, dynamic graph, attention, SIR, parameter abduction, joint multi-action training or new platform is implemented here.

## 1. Scientific question

With identical architectures, parameter counts and training budgets, does event-specific value `V_evt` of correct versus wrong-episode factual suffixes systematically decline with non-ego counterfactual response magnitude `R`?

Stage 1 permits only `INCONCLUSIVE_MODEL_INSUFFICIENT` (Gate S failed), `DEVELOPMENT_GO_TO_NEW_HOLDOUT` (Gate S passed and development Gate A direction supported), or `DEVELOPMENT_NO_GO` (Gate S passed but Gate A did not support proceeding). **`GATE_A_CONFIRMED` is prohibited here**; final Gate A requires a subsequent fresh holdout.

## 2. Three fair conditions

| Condition | Suffix features | Suffix mask |
|---|---|---|
| M0 `no_suffix` | All zero | All zero |
| M1 `uniform_suffix` | Correct first 10 steps from the same episode | Correct presence for first 10 steps |
| M2 `shuffled_suffix` | First 10 steps from another episode via stratified derangement | Donor presence for first 10 steps |

All instantiate the same `PerAgentPairedEffectGRU`, with identical hidden size, layers, action embedding, optimizer, training budget, parameter names and counts. Only the factual-suffix source differs. Outputs are `B×T×N×2`, preserving the vehicle dimension through decoding; there is no graph, attention, slot-specific embedding or independent head.

## 3. Data and leakage controls

- Source data: `data/object_centric_stage/merge_2000/object_twinworld_2000.npz` (primary) and `data/object_centric_stage/roundabout_2000/object_twinworld_2000.npz` (exploratory). Fixed split 1400/300/300; split 2 is `development_test` only.
- Fixed eight-feature order: `presence, is_ego, x, y, vx, vy, cos_heading, sin_heading`.
- Target: `target_effect = cf_xy − factual_xy` in world coordinates. `counterfactual_future` enters target/R/mask only, **never model inputs**.
- Valid mask = history_agent_known AND factual_presence AND counterfactual_presence.
- Fit normalization and response-bin thresholds **using split 0 only**; anchor at each episode's final historical ego world position.
- Gates A/B use non-ego slots (slot>0) only. Ego is a Gate S pipeline positive control. Episodes are independent units; average training seeds per agent-time unit, not as independent samples.

## 4. Frozen derangement

- Base seed `20260907`; training uses `base_seed+epoch`; fixed validation seed `20261907`; fixed development_test seed `20262907`.
- Stratified fallback: `(cf_action, cf_duration, factual_actions[:,0], visible_count)` → `(cf_action, cf_duration, visible_count)` → `(cf_action, cf_duration)` → same split only.
- Invariants: no self-match, duplicate donor, same donor seed or cross-split pairing. Record fallback counts in manifest QA; persisted validation/test manifests enter run signatures.

## 5. Gate order

1. **Gate S** (train/validation and frozen validation manifest only): S1 nondegenerate optimization, matched parameters and ego positive control; S2 predictable per-vehicle response magnitude; S3 suffix use supported by prediction and representation evidence. Any failure yields `INCONCLUSIVE_MODEL_INSUFFICIENT`; only one preregistered enhancement is permitted.
2. **Gate A** (only after Gate S): low-response `V_evt` CI>0; `V_evt_low − V_evt_high` CI>0; `Spearman(R, eta_evt)` CI>0; covariate-adjusted `log1p(R)` coefficient CI>0; consistent direction not driven by one seed/action; all shuffled-manifest QA passed.
3. **Gate B**: high-response absolute `Delta = L_no − L_suffix < 0`, CI wholly below 0, at least 30 independent episodes; otherwise `NOT_SUPPORTED`/`UNDERPOWERED`.
4. If Gate S fails, Gates A/B are `BLOCKED`; episode-cluster bootstrap uses 10,000 resamples and 95% CIs.

## 6. Commands

Short CPU smoke (completed; engineering records retained in the original research workspace):

```powershell
python code/run_stage1_evidence_value.py --config config/stage1_evidence_value_smoke.json --device cpu
python code/run_stage1_evidence_value.py --config config/stage1_evidence_value_smoke.json --device cpu --resume
```

Formal long training (**historically launched manually by the user**, after verifying `torch.cuda.is_available()` is True):

```powershell
python code/run_stage1_evidence_value.py --config config/stage1_evidence_value_merge.json --device cuda --resume
python code/run_stage1_evidence_value.py --config config/stage1_evidence_value_roundabout.json --device cuda --resume
```

Resume interruptions with the same command plus `--resume`. Do not delete partial outputs and restart unless signature checks show incompatibility and the user decides. Merge is primary development evidence; Roundabout is exploratory replication.

**Current status: formal training in both scenes is completed and sealed (15 runs each). Any subsequent statistical recomputation must use read-only inference; retraining is prohibited:**

```powershell
python code/run_stage1_evidence_value.py --config config/stage1_evidence_value_merge.json --device cuda --reanalyze-only
python code/run_stage1_evidence_value.py --config config/stage1_evidence_value_roundabout.json --device cuda --reanalyze-only
```

`--reanalyze-only` loads sealed checkpoints for read-only forward inference and recomputes Gates S/A/B, reporting `training_performed=false` and `checkpoint_modified=false`; do not retrain via `--resume`.
