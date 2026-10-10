# AC-Twin Stage 2 Confirmatory Holdout: Frozen Executable Protocol

> **Status at protocol freezing: PROTOCOL_LOCKED_READY_FOR_MANUAL_COLLECTION**
> Plan ID: `AC-TWIN-RFER-STAGE2-HOLDOUT-2026-09-07-V1`; parent research plan: `AC-TWIN-RFER-2026-09-07-V1` v1.1
> Primary scene: `merge-v0`; scope: implementation, tests, freezing, manual collection, frozen-model inference and single unblinding.
> **Protocol and historical commands only; no results here.** Run commands from the project root. The completed holdout must not be unblinded again.

## 0. Immutable boundaries

- No training, fine-tuning, checkpoint selection or fitting normalization/bins on new data.
- Confirmatory analysis runs once; repeated unblinding or changes to bootstrap seed, CIs or primary contrasts are mechanically prohibited.
- No router implementation before Gate A; no dual-edged wording unless Gate B is supported.
- The claim is **nonmonotonic response-regime heterogeneity**, not a continuous U-shaped function or causal mechanism.

## 1. Frozen inputs (all hashes registered in protocol lock)

- 15 Stage 1 Merge checkpoints: M0/M1/M2 × seeds {11,22,33,44,55} (`experiments/stage1_evidence_value/merge/`).
- `normalization.npz` (frozen feature/target means and SDs), `response_bins.json` (`zero_tolerance=0.0001`, `t1=0.24843344965718578`, `t2=1.898992796585611`).
- Stage 1 Gate S report（`experiments/stage1_evidence_value/merge/report.json`）。
- Frozen Gate S* parameters in `frozen_gate_s_star_params.npz`: two MAE-baseline coefficients (`beta_action_horizon` (3,), `beta_action_horizon_distance` (4,)) and representation probes (`probe_coefficients` (65,6), `probe_coefficients_shuffled` (65,6), `probe_scene_mean` (6,)), fitted on **Stage 1 train split only**, then sealed.
- Registered code hashes: runner, collector, `stage2_protocol`, `stage2_inference`, `stage2_statistics`, `stage1_model`, `stage1_data`, `stage1_statistics`, `stage1_plus_statistics`, `checkpointing`, `object_centric`, `progress`.
- Candidate seed schedule: 18,000 seeds from `SeedSequence(20270999).spawn(3)[0]`, sampled without replacement via `choice(2147483646, size, replace=False)+1`, guaranteeing internal uniqueness and range [1, 2147483646].

## 2. Frozen seeds and random streams

- Formal `collection_seed = 20270999`; `SeedSequence(20270999).spawn(3)` derives `seed_schedule_rng` / `action_rng` / `stratum_rng`.
- Episode seed is the precomputed schedule's `attempt_index` entry, not sampled from a running RNG; this preserves resume equivalence.
- Historical-seed scans exclude Stage 2 products (NPZ paths containing `stage2_`), so smoke collection does not contaminate the formal lock or create false Gate Q overlap reports at unblinding.
- Observed: the 18,000-seed `20270999` schedule had **zero overlap** with 12,000 historical seeds; the 256 smoke candidates (`collection_seed=20271999`) also had zero overlap.
- Primary derangement seed = `20272907`; alternate donor seeds = `20272917, 20272927, 20272937, 20272947`. Reuse the four fallback levels in `stage1_data.build_derangement`.
- Primary bootstrap seed = `20270908`, 10,000 episode resamples, 95% percentile CIs.

## 3. Observed environment

- Python 3.13.5；NumPy 2.1.3；PyTorch 2.14.0+cu132；highway-env 1.10.2；gymnasium 1.2.3。
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU (CUDA available). Inference requires < 1 GB GPU memory (145,066 model parameters; batch 128).
- Windows 11; PowerShell. Python entry points set `MKL_THREADING_LAYER=SEQUENTIAL` and `KMP_DUPLICATE_LIB_OK=TRUE` before importing numpy/torch to avoid OMP Error #15.

## 4. Observed test counts

- Stage 2 suite (5 files): `tests/test_stage2_protocol.py`, `test_stage2_collector.py`, `test_stage2_inference.py`, `test_stage2_statistics.py`, `test_stage2_runner.py` → **83 passed**.
- Full regression `python -m pytest -q` → **395 passed, 0 failed**.
- Key check: Task 1 recomputed joint power for five new conditions on development data, n=1200, 10,000 replicates; 75% scenario joint power = **0.918** (target 0.91±0.02), empirical=0.994, 50%=0.604; weakest condition `C_high_low`. Point estimates `C_presence=0.1976`, `C_high_low=0.181`, `V_zero=0.339`, `V_high=0.213` reproduced Stage 1 Plus.

## 5. Expected durations extrapolated from smoke

- create-lock, including frozen probe/baseline fitting: approximately 10–20 s.
- Formal collection of 1,200 episodes: smoke collected 12 episodes from 18 attempts in 6 s; extrapolating ~67% acceptance and paired rollouts gives **15–30 min**, depending on collision/early-termination rejection. Checkpoint resume is supported.
- Formal inference (15 primary predictions, 5 M1 swaps, 20 alternate donors, 2 probe summaries; CUDA): 12-episode CPU smoke took 2.9 s; 1,200 episodes with `--device cuda` expected **1–3 min**.
- unlock-and-analyze (10,000 bootstrap resamples across statistics): **1–5 min**.

## 6. Historical manual commands (strict sequence; not authorization to rerun)

### 6.1 Read-only verification of the formal protocol lock

```powershell
python code/run_stage2_confirmatory.py verify-lock --config config/stage2_holdout_merge_confirm.json
```

Completion: existing `protocol_lock.json` unchanged; output includes historical seed count, candidate_schedule=18000, zero_overlap=True, checkpoints_ok=15 and `PROTOCOL_LOCK_VERIFIED`.

### 6.2 Formal collection (1,200 episodes)

```powershell
python code/collect_stage2_holdout.py --config config/stage2_holdout_merge_confirm.json --resume
```

Completion: `accepted=1200`; final NPZ and `collection_receipt.json` published atomically; per-attempt `rejection_ledger.jsonl`; no gate values printed. Repeating the same command after interruption resumes the checkpoint (`--resume` is idempotent: starts fresh without a checkpoint, resumes when present, verifies an existing final NPZ read-only without overwriting).

### 6.3 Frozen-model inference (CUDA)

```powershell
python code/run_stage2_confirmatory.py infer --config config/stage2_holdout_merge_confirm.json --device cuda
```

Completion: 15 primary predictions, 5 M1 swaps, 20 alternate-donor predictions, 5 manifests, 2 probe summaries and `prediction_receipt.json`. Standard output contains prediction/manifest counts only, **no** V_evt, CIs or gate labels. Valid existing receipts are verified without overwriting.

### 6.4 Single unblinding and confirmatory analysis (historically required explicit unblinding authorization)

```powershell
$stage2Lock = Get-Content 'experiments\stage2_confirmatory\merge\protocol_lock.json' -Raw -Encoding UTF8 | ConvertFrom-Json
python code/run_stage2_confirmatory.py unlock-and-analyze --config config/stage2_holdout_merge_confirm.json --expected-lock-sha256 $stage2Lock.lock_sha256
```

Completion: one `report.json` and `STAGE2_CONFIRMATORY_DECISION.md`, each with `output_signature` (recomputable after removing that field from saved JSON). **Repeating the command is refused** (FileExistsError).

## 7. Permitted mechanical terminal states (single adjudication; plan §4)

`INVALID_TECHNICAL` / `INCONCLUSIVE_FROZEN_MODEL_OUT_OF_SAMPLE_INSUFFICIENT` /
`GATE_A_CONFIRMED_NONMONOTONIC` / `GATE_A_NOT_CONFIRMED_RESPONSE_SUPPORT` / `GATE_A_NOT_CONFIRMED`。
Fixed priority: Gate Q → Gate S* → 100 episodes per bin → remaining Gate A conditions → all passed.

## 8. Implementation completion checks

- [x] Tasks 1–6 and full regression had 0 failures (Stage 2 suite 83; full suite 395).
- [x] Smoke completed only in `experiments/stage2_confirmatory/smoke/` and `data/object_centric_stage/stage2_merge_smoke/`; no smoke products in the formal directory.
- [x] Formal configuration, code, statistics, 15 checkpoints, normalization, bins, seed schedule and frozen Gate S* parameters registered by hash in `experiments/stage2_confirmatory/merge/protocol_lock.json`.
- [x] 18,000 candidate seeds had zero overlap with historical seeds (verified by verify-lock).
- [x] This PROTOCOL.md records actual commands, test counts, durations, recovery and completion criteria.
- [x] At protocol freezing, no statistical script had read the formal holdout and no router had been implemented.
