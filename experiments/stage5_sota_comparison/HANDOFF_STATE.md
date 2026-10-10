# Stage 5 SOTA comparison — HANDOFF STATE

> Plan: `AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3` (user-approved 2026-09-27, "批准以上事项并执行。")
> Executor ledger: `.superpowers/sdd/2026-09-27_Stage5_v0.3/progress.md` (resume anchor; read it first after any context break)
> Status: **TERMINAL — `STAGE5_MERGE_COMPARISON_UNDERPOWERED` (2026-09-28).** Experiment branch closed; **Task 6 paper closure COMPLETE (2026-09-28).** No comparison holdout exists or may be created. All in-scope tasks of the approved plan are done; the Stage 5 authorization has ended.

## Terminal state (permanent, never re-adjudicated)

- Materiality gate PASSED on validation (10k paired bootstrap, 95%, entropy 2026092721): I_AURC = **0.1404**, I_REGRET80 = **0.1778** (0.75× = 0.105 ≥ 0.05; 0.133 ≥ 0.03).
- Joint SOTA-M AND-chain power at 75% effect retention (`power/power_report.json`, entropy 2026092722, 10k whole-group draws per n, structural identities verified per draw): **0.3778 / 0.5472 / 0.6872 / 0.8354 at n = 2000/3000/4000/6000** — all < 0.90 target; n=6000 is the frozen maximum → `selected_n = null`, terminal `STAGE5_MERGE_COMPARISON_UNDERPOWERED`.
- Binding condition: `r1_i_aurc` (I_AURC ≥ 0.05 AND 95% CI lower > 0; marginal 0.837 at n=6000). r2/r3/r4/r5 at n=6000: 0.999/0.999/0.987/1.0. At 1.00 retention n=6000 joint = 0.9696. The shortfall is the preregistered 75%-retention judge on the primary endpoint's CI condition — not a support-proportion artifact.
- Per plan §6.2/§10/§12.9: no in-place threshold/seed/gate/method/baseline change, no n-grid widening, no Stage 5 Plus, no holdout manifest/lock/collection. Task 5 was never started and must not be started.

## Frozen decisions (all create-once artifacts exist under `experiments/stage5_sota_comparison/`)

- **`STRONGEST_PUBLISHED_BASELINE = CCTR_DRIVING_CALIBRATION_ADAPTED`** (grid1 = {hidden_width 96, depth 2}, validation AURC 0.012762, 11,522 params) — `development/final_method_selection.json`. Never swappable.
- All 12 candidates valid; full validation metrics in `development/validation_comparison_report.json` (+ native calibration: P1/P4/P5 OK, P2/P3 NOT_APPLICABLE; P5 empirical coverage 0.893 vs 0.95).
- Experimental candidates: P1 CONFIDNET_TCP, P2 SELECTIVENET_REGRET, P3 RWR_NN_KNN_REJECT, P4 CONFORMAL_REGRESSION_REJECT, P5 CCTR_DRIVING_CALIBRATION (all `ELIGIBLE_ADAPTED_VERIFIED`). OOSR supplemental only; CONFORMAL_RISK_CONTROL registered alternate, not activated.
- **Holdout seed entropy = `2026093073`** (40,000-seed schedule, zero-overlap audited; replaces `2026092730` which failed pre-registration with 4 chance overlaps — nothing was ever persisted under it). **The schedule was NEVER consumed: no holdout was collected.**
- Entropies: bootstrap 2026092721 (consumed by the validation report), MC 2026092722 (consumed by the power report), adjudication 2026092723 (NEVER consumed).
- AC-Twin validation scoring reproduces the frozen R2 gate quantities bitwise (dAURC=dR80=0.0).
- Power calibre: corrected D2F typing (effect/anchor/support partition; support proportions never shrunk; oracle/stratum shares sum to 1 per draw; joint AND-chain at 95% z; SE at current n; no marginal multiplication).

## Completed artifacts

- Task 1: `literature/method_registry.json|md`, `compatibility_decision.json`, `phase_l/sanity/`, `paper/tables/table1/`.
- Task 2: `code/ac_twin/stage5_protocol.py`, `stage5_features.py`, `config/stage5_sota_comparison_v0.3.json` (+ 56 tests, slow seed audit).
- Task 3: `code/ac_twin/stage5_methods.py`, `stage5_training.py` (+ 22 tests).
- Task 4: `code/ac_twin/stage5_statistics.py`, `stage5_selection.py`, `stage5_power.py` (+ 21 tests); `development/method_search_space.json`, `development/final_method_selection.json`, `development/validation_comparison_report.json`, `development/validation_risks_selected.npz`, `development/models/fit_receipts.json` (60 fits), `power/power_report.json`; logs `development_freeze_run.log`, `power_run.log`.
- Task 5: **does not exist (forbidden by the terminal state).**
- Task 6: `paper/tables/table_stage5_sota/` (`TABLE_STAGE5_SOTA.md`, `table_stage5_sota_data.xlsx`, `table_stage5_sota_source.json`, `build_table_stage5_sota.py`, `README.md`); `paper/figures/stage5_sota/` (`figS_stage5_risk_coverage_data.csv`, `FIGURE_LEGEND.md`); `paper/FIGURE_TABLE_PACKAGE_INDEX.md` + blueprint (Table S7 / Fig. S-Stage5 reserved); `00_READ_FIRST_AC_TWIN.md` (2026-09-28 override + Stage 5 authorization bullet), `01_AC_TWIN_DOCUMENT_STATUS_INDEX.md` (header + v0.3 row → COMPLETED_EXPERIMENT + 6 new rows), `logs/2026-09-28_AC-Twin科研日志.md` (entropy swap 2026092730→2026093073, frozen STRONGEST_PUBLISHED_BASELINE, underpowered terminal).

## Forbidden (blocking)

- Old confirm per-group products (`stage4cr2_confirm/`, `stage4d_ood/`, whole `stage4d_d2f/`, `stage4b/`) and consumed R2 screen products: triple rejection enforced in `stage5_protocol.assert_allowed_stage5_data`.
- No comparison holdout may be created; the 2026093073 schedule and 2026092723 entropy stay unconsumed.

## Latest verification

- `python -m pytest tests/test_stage5_protocol.py tests/test_stage5_features.py tests/test_stage5_methods.py tests/test_stage5_training.py tests/test_stage5_statistics.py tests/test_stage5_selection.py tests/test_stage5_power.py -q` → 99 passed, 1 deselected (2026-09-28).

## Paper-facing boundary (§11)

- The SOTA-advantage claim (§11.1) is NOT writable. Per §11.3: keep Stage 4 + D2F confirmed conclusions; do not claim superiority over published methods; report the development comparison honestly with its development-only label and the underpowered terminal state.

## Next action

None. The approved plan's scope is fully executed (Tasks 1–4 complete, Task 5 forbidden and never started, Task 6 paper closure complete). Any new work requires a new, separately approved plan.

## Blockers

None.
