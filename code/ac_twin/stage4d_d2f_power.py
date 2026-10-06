"""Stage 4D-D2F Phase P: corrected development-screen power (NO new collection, lock or manifest).

Plan ``AC-TWIN-RFER-STAGE4D-D2F-POWER-CORRECTION-EXTERNAL-CONFIRM-2026-09-24-V0.1`` section 2.
The D2E power pre-check applied the 75% "effect retention" mechanically to ALL 33 simulated
quantities (``stage4d_d2e_power.simulate_power``: ``mean = shrink * theta_hat``), which turned
scenario/task SUPPORT proportions into model effects: the MetaDrive reactive-group share 0.272
became 0.204 next to the 0.20 task floor, and the ten oracle action shares stopped summing to 1
(0.75). That was a pre-collection power-simulation calibre defect; it never adjudicated the
model. D2F corrects the calibre WITHOUT touching the frozen D2/D2R/D2E artifacts and WITHOUT
claiming the method is confirmed.

Corrected typing (plan 2.1; every one of the 33 power quantities is typed exactly):

- MODEL/DECISION EFFECTS (shrink the effect CENTRE by rho in {1, .75, .5}): I_AURC_REGRET,
  I_REGRET@80, PR-AUC-prevalence, branch-level Spearman, base-vs-RANDOM improvement, the five
  per-seed improvements (both metrics), and the decision DIFFERENCE
  ``D_idle = mean_regret(IDLE) - mean_regret(a0)``. With E_ref the old-validation effect and
  E_b one n-group paired-bootstrap effect, the simulated effect is
  ``E_b^(rho) = (E_b - E_ref) + rho * E_ref``: the same group-level resampling covariance,
  sign and pairing are preserved. ``D_idle`` is formed from the SAME resample's two regrets
  and then transformed by that formula; the c5 "not worse than IDLE" check tests ONLY
  ``D_idle^(rho) >= -tie_tolerance`` — the raw base/idle absolute means are report anchors and
  are NEVER separately multiplied by rho and then compared.
- SCENARIO/TASK SUPPORT (never shrunk with "method-effect retention"): the regret>0.01
  fraction, the ten oracle action shares, the reactive-group fraction and the two pairs of
  complementary stratum shares keep their resampled values; per draw the oracle shares and each
  complementary stratum pair sum EXACTLY to 1 and every share stays in [0, 1] (asserted
  fail-closed).

Simulation (plan 2.1): the whole scenario group is the ONLY resampling unit; all candidate /
baseline / task quantities follow the SAME index vector; for each admissible n, 10,000 joint
resamples recompute the COMPLETE gate vector; CI lower bounds use the resampling SE AT THE
CURRENT n at the same 80% confidence (never reverse-derived from an old CI width); the
point-estimate materiality lines and the 4/5-seed joint condition enter EVERY simulated draw;
marginal powers are never multiplied. Fresh frozen entropy ``2026122406`` (distinct from the
D2E power seed, registered in the report). The candidate n set is the plan set INTERSECTED
with the pre-frozen pool-capacity admissible set (``pool_capacity_audit.json``) and can never
be changed after seeing power results. The smallest admissible n whose 75% joint power reaches
0.80 is selected; otherwise the preregistered terminal state is ``D2F_SCREEN_UNDERPOWERED``
and NO new ID manifest / lock / collection is created.

A separate NON-SELECTIVE scenario-support stress table (plan 2.1) shifts ONLY the centre of
the reactive share to {0.24, 0.22, 0.20} and of the high-regret share to {0.30, 0.20, 0.15}
(every other support quantity keeps its coupled resampled value) and reports c6/c2 marginal
and joint pass sensitivity, clearly labelled ``scenario-support shift``; it never participates
in the n selection and never changes the c2/c6 thresholds.

This module imports the frozen D2E power module READ-ONLY (layout constants, the extended
47-slot quantity vector, the validation quantities loader/validator) — the module file is
never modified — and the frozen D2 screen statistics (gate thresholds) bitwise. All writes
land under ``experiments/stage4_reliability/stage4d_d2f/``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ac_twin import stage4d_d2e_power as POW
from ac_twin import stage4d_d2f_protocol as PROT
from ac_twin import stage4d_external_statistics as D2STAT
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

# --- frozen plan contract (section 2.1) -------------------------------------------------------------
POWER_SEED = PROT.POWER_SEED                       # 2026122406 (fresh, registered)
POWER_RESAMPLES = 10000
CONFIDENCE = POW.CONFIDENCE                        # 0.80 (the screen never uses 95%)
SCREEN_Z = POW.SCREEN_Z                            # two-sided 80% z
SHRINK_SCENARIOS = (1.0, 0.75, 0.5)
SELECTION_SCENARIO = 0.75
POWER_TARGET = 0.80

STATUS_UNDERPOWERED = "D2F_SCREEN_UNDERPOWERED"    # preregistered terminal
STATUS_POWER_PASS = "D2F_POWER_CORRECTION_PASS"    # descriptive, NON-terminal

POWER_DIR = PROT.D2F_DIR / "power"
REPORT_NAME = "power_correction_report.json"

# --- the plan-2.1 two-class typing of the 33 power quantities (exact partition) -----------------------
#: model/decision effects: the CENTRE shrinks by rho (resampling covariance/sign preserved).
EFFECT_SLOTS = (POW.P_I_AURC, POW.P_I_REGRET80, POW.P_PR, POW.P_SPEARMAN,
                POW.P_BASE_VS_RANDOM) + tuple(POW.P_SEED_I_AURC) + tuple(POW.P_SEED_I_R80)
#: report anchors ONLY (absolute regrets; never gate inputs, never separately shrunk).
ANCHOR_SLOTS = (POW.P_BASE_MEAN, POW.P_IDLE_MEAN)
#: scenario/task support: never shrunk with "method-effect retention".
SUPPORT_SLOTS = (POW.P_HIGH_REGRET,) + tuple(POW.P_ORACLE) + (POW.P_REACTIVE,) \
    + tuple(POW.P_STRATA)
N_POWER_QUANTITIES = POW.N_POWER_QUANTITIES        # 33

#: non-selective scenario-support stress centres (plan 2.1; never used for the n selection).
STRESS_REACTIVE_CENTERS = (0.24, 0.22, 0.20)
STRESS_HIGH_REGRET_CENTERS = (0.30, 0.20, 0.15)

_BASELINE_NPZ_PREFIX = "baseline_risk__"


def classify_power_slots() -> dict[str, tuple[int, ...]]:
    """The exact plan-2.1 partition of the 33 power quantities (fail closed if not a partition)."""
    effect = tuple(int(s) for s in EFFECT_SLOTS)
    anchor = tuple(int(s) for s in ANCHOR_SLOTS)
    support = tuple(int(s) for s in SUPPORT_SLOTS)
    union = sorted(effect + anchor + support)
    if union != list(range(N_POWER_QUANTITIES)):
        raise AssertionError(
            f"power-slot typing is not an exact partition of the {N_POWER_QUANTITIES} "
            f"quantities: {union}")
    return {"effect": effect, "anchor": anchor, "support": support}


# --- identity reference + structural identities --------------------------------------------------------


def reference_vector(quantities: Mapping[str, Any]) -> np.ndarray:
    """The 33-slot identity evaluation on the old-validation groups (E_ref / support anchors)."""
    identity = np.arange(int(np.asarray(quantities["regret"]).size), dtype=np.int64)
    vector = POW.compute_d2e_power_vector(identity, quantities)
    subset = np.asarray(vector, dtype=np.float64)[list(POW.POWER_SLOTS)]
    if not np.isfinite(subset).all():
        raise ValueError("non-finite identity power quantity on validation (fail closed)")
    return subset


def assert_structural_identities(draws: np.ndarray) -> None:
    """Fail closed unless every draw's support proportions satisfy the structural identities.

    Oracle action shares sum to 1; each complementary stratum pair (tstar low/high,
    interaction low/high) sums to 1; every support proportion lies in [0, 1]. These identities
    must hold on the RAW resampled draws — the corrected transform never touches support slots.
    """
    draws = np.asarray(draws, dtype=np.float64)
    oracle = draws[:, POW.X_ORACLE:POW.X_ORACLE + 10]
    tstar = draws[:, POW.X_STRATA:POW.X_STRATA + 2]
    interaction = draws[:, POW.X_STRATA + 2:POW.X_STRATA + 4]
    atol = 1e-9
    if not np.allclose(oracle.sum(axis=1), 1.0, rtol=0.0, atol=atol):
        raise ValueError("oracle action shares do not sum to 1 on every draw")
    if not np.allclose(tstar.sum(axis=1), 1.0, rtol=0.0, atol=atol):
        raise ValueError("t_star stratum shares do not sum to 1 on every draw")
    if not np.allclose(interaction.sum(axis=1), 1.0, rtol=0.0, atol=atol):
        raise ValueError("interaction stratum shares do not sum to 1 on every draw")
    support = draws[:, [POW.X_ORACLE + a for a in range(10)]
                      + [POW.X_REACTIVE] + [POW.X_STRATA + k for k in range(4)]]
    if float(support.min()) < -atol or float(support.max()) > 1.0 + atol:
        raise ValueError("a support proportion left [0, 1] on a draw")


# --- the corrected transform + the D2 screen AND-chain ----------------------------------------------------


def transform_draws(draws33: np.ndarray, reference33: np.ndarray, rho: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Apply the plan-2.1 transform: E^(rho) = E_b + (rho-1) * E_ref on EFFECT slots and D_idle.

    Support and anchor slots keep their resampled values. Returns the transformed 33-slot
    draws and the transformed decision difference ``D_idle^(rho)`` per draw.
    """
    draws33 = np.asarray(draws33, dtype=np.float64)
    reference33 = np.asarray(reference33, dtype=np.float64)
    if draws33.ndim != 2 or draws33.shape[1] != N_POWER_QUANTITIES \
            or reference33.shape != (N_POWER_QUANTITIES,):
        raise ValueError("draws/reference must align with the 33-slot power subset")
    slots = classify_power_slots()
    transformed = draws33.copy()
    for slot in slots["effect"]:
        transformed[:, slot] = draws33[:, slot] + (float(rho) - 1.0) * reference33[slot]
    d_idle = draws33[:, POW.P_IDLE_MEAN] - draws33[:, POW.P_BASE_MEAN]
    d_idle_ref = reference33[POW.P_IDLE_MEAN] - reference33[POW.P_BASE_MEAN]
    d_idle_t = d_idle + (float(rho) - 1.0) * d_idle_ref
    return transformed, d_idle_t


def evaluate_corrected_draws(transformed33: np.ndarray, d_idle_t: np.ndarray,
                             standard_errors: np.ndarray, *,
                             z: float = SCREEN_Z) -> dict[str, Any]:
    """The D2 screen statistical AND-chain over corrected joint draws (mirrors the frozen gate).

    CI conditions use the lower bound computed from the resampling SE AT THE CURRENT n at the
    given two-sided z (the screen chain uses the frozen 80% ``SCREEN_Z``; the confirm chain
    calls :func:`evaluate_confirm_draws` with the 95% z); fraction / oracle / strata /
    seed-count conditions use point values; c5 tests the transformed base-vs-random improvement
    (point + CI) and ONLY the transformed difference ``D_idle^(rho) >= -tie_tolerance``.
    Structural conditions (c1/c8/r5/r6) are conditioned on, exactly as in the D2E pre-check.
    """
    t = np.asarray(transformed33, dtype=np.float64)
    se = np.asarray(standard_errors, dtype=np.float64)
    d_idle_t = np.asarray(d_idle_t, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] != N_POWER_QUANTITIES \
            or se.shape != (N_POWER_QUANTITIES,) or d_idle_t.shape != (t.shape[0],):
        raise ValueError("corrected draws / SE / D_idle must align with the power subset")

    def ci_lower_at_z(slot: int) -> np.ndarray:
        return t[:, slot] - float(z) * se[slot]

    seeds_both = ((t[:, list(POW.P_SEED_I_AURC)] > 0.0)
                  & (t[:, list(POW.P_SEED_I_R80)] > 0.0)).sum(axis=1)
    oracle_count = (t[:, list(POW.P_ORACLE)]
                    >= D2STAT.GATE_TASK_MIN_ORACLE_FRAC).sum(axis=1)
    strata_min = t[:, list(POW.P_STRATA)].min(axis=1)
    conditions = {
        "c2_high_regret_frac": t[:, POW.P_HIGH_REGRET] >= D2STAT.GATE_TASK_MIN_HIGH_REGRET_FRAC,
        "c3_oracle_coverage": oracle_count >= D2STAT.GATE_TASK_MIN_DISTINCT_ORACLE,
        "c4_spearman_ci_lower_positive": ci_lower_at_z(POW.P_SPEARMAN) > 0.0,
        "c5_base_vs_random_and_not_worse_than_idle": (
            (t[:, POW.P_BASE_VS_RANDOM] > 0.0) & (ci_lower_at_z(POW.P_BASE_VS_RANDOM) > 0.0)
            & (d_idle_t >= -D2STAT.BASELINE_TIE_TOLERANCE)),
        "c6_reactive_frac": t[:, POW.P_REACTIVE] >= D2STAT.GATE_TASK_MIN_REACTIVE_FRAC,
        "c7_strata_floor": strata_min >= D2STAT.GATE_TASK_MIN_STRATUM_FRAC,
        "r1_i_aurc_regret": (t[:, POW.P_I_AURC] >= D2STAT.I_AURC_MIN)
                            & (ci_lower_at_z(POW.P_I_AURC) > 0.0),
        "r2_i_regret80": (t[:, POW.P_I_REGRET80] >= D2STAT.I_REGRET80_MIN)
                         & (ci_lower_at_z(POW.P_I_REGRET80) > 0.0),
        "r3_pr_auc_minus_prevalence": (t[:, POW.P_PR] > 0.0) & (ci_lower_at_z(POW.P_PR) > 0.0),
        "r4_seed_consistency_both_positive": seeds_both >= D2STAT.MIN_POSITIVE_SEEDS,
    }
    passed = np.logical_and.reduce(list(conditions.values()))
    return {"passed": passed, "conditions": conditions}


#: two-sided 95% z for the confirm AND-chain (D2 10.3/10.4: every CI condition at 95%)
CONFIRM_Z = 1.959963984540054
CONFIRM_CONFIDENCE = 0.95
CONFIRM_POWER_SEED = PROT.D2F_CONFIRM_POWER_SEED     # 2026122408 (fresh, registered)


def evaluate_confirm_draws(transformed33: np.ndarray, d_idle_t: np.ndarray,
                           standard_errors: np.ndarray) -> dict[str, Any]:
    """The D2 confirm AND-chain (10.3/10.4): the screen conditions with CI conditions at 95%."""
    return evaluate_corrected_draws(transformed33, d_idle_t, standard_errors, z=CONFIRM_Z)


def _joint_with_replacement(conditions: Mapping[str, np.ndarray], replaced: str,
                            replacement: np.ndarray) -> np.ndarray:
    """The AND-chain with ONE condition replaced (scenario-support stress sensitivity)."""
    parts = [np.asarray(replacement, dtype=bool) if key == replaced
             else np.asarray(value, dtype=bool) for key, value in conditions.items()]
    return np.logical_and.reduce(parts)


# --- resampling driver + scenario-support stress table -----------------------------------------------------


def resample_draws(quantities: Mapping[str, Any], n: int, *,
                   n_resamples: int = POWER_RESAMPLES, seed: int = POWER_SEED,
                   progress: Callable[[str], None] | None = None) -> np.ndarray:
    """10,000 joint whole-group resamples of size n; per-n stream from SeedSequence([seed, n]).

    Every quantity of a draw is recomputed from the SAME with-replacement group index vector
    (the whole scenario group is the only resampling unit). The per-n seeds are derived
    deterministically from the registered frozen entropy and recorded in the report.
    """
    groups = int(np.asarray(quantities["regret"]).size)
    if groups < 2:
        raise ValueError("corrected power needs at least 2 groups")
    generator = np.random.default_rng(np.random.SeedSequence([int(seed), int(n)]))
    draws = np.empty((int(n_resamples), POW.N_EXTENDED), dtype=np.float64)
    for draw in range(int(n_resamples)):
        draws[draw] = POW.compute_d2e_power_vector(
            generator.integers(0, groups, size=int(n)), quantities)
        if progress and (draw + 1) % 1000 == 0:
            progress(f"n={n}: joint resample {draw + 1}/{n_resamples}")
    return draws


def _stress_table(transformed33: np.ndarray, d_idle_t: np.ndarray, se33: np.ndarray,
                  reference33: np.ndarray, *,
                  z: float = SCREEN_Z) -> dict[str, Any]:
    """Non-selective scenario-support stress sensitivity at the 75% model-effect scenario.

    Only the named support slot's CENTRE is shifted (additive shift of the coupled resampled
    values, clipped to [0, 1]); every other support quantity keeps its resampled coupling.
    Labelled ``scenario-support shift``; never participates in the n selection.
    """
    base_eval = evaluate_corrected_draws(transformed33, d_idle_t, se33, z=z)
    conditions = base_eval["conditions"]

    def shifted(slot: int, target: float) -> np.ndarray:
        return np.clip(transformed33[:, slot]
                       + (float(target) - float(reference33[slot])), 0.0, 1.0)

    table: dict[str, Any] = {"label": "scenario-support shift (NON-SELECTIVE; never used for "
                                      "the n selection, never changes the c2/c6 thresholds)",
                             "model_effect_retention": SELECTION_SCENARIO,
                             "reactive_frac": {"reference": float(reference33[POW.P_REACTIVE]),
                                               "gate_floor": D2STAT.GATE_TASK_MIN_REACTIVE_FRAC,
                                               "scenarios": {}},
                             "high_regret_frac": {
                                 "reference": float(reference33[POW.P_HIGH_REGRET]),
                                 "gate_floor": D2STAT.GATE_TASK_MIN_HIGH_REGRET_FRAC,
                                 "scenarios": {}}}
    for target in STRESS_REACTIVE_CENTERS:
        c6 = shifted(POW.P_REACTIVE, target) >= D2STAT.GATE_TASK_MIN_REACTIVE_FRAC
        table["reactive_frac"]["scenarios"][f"{target:.2f}"] = {
            "c6_marginal_power": float(c6.mean()),
            "joint_power": float(_joint_with_replacement(
                conditions, "c6_reactive_frac", c6).mean())}
    for target in STRESS_HIGH_REGRET_CENTERS:
        c2 = shifted(POW.P_HIGH_REGRET, target) >= D2STAT.GATE_TASK_MIN_HIGH_REGRET_FRAC
        table["high_regret_frac"]["scenarios"][f"{target:.2f}"] = {
            "c2_marginal_power": float(c2.mean()),
            "joint_power": float(_joint_with_replacement(
                conditions, "c2_high_regret_frac", c2).mean())}
    return table


def simulate_corrected_power(quantities: Mapping[str, Any], *,
                             candidate_ns: Sequence[int],
                             n_resamples: int = POWER_RESAMPLES,
                             seed: int = POWER_SEED,
                             shrink_scenarios: Sequence[float] = SHRINK_SCENARIOS,
                             selection_scenario: float = SELECTION_SCENARIO,
                             target: float = POWER_TARGET,
                             z: float = SCREEN_Z,
                             progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Joint D2 AND-chain power at each admissible n under the corrected calibre.

    The 75% model-effect scenario is the only blocking judge; the smallest admissible n
    reaching the target is selected. The candidate set must already be the pre-frozen
    capacity-admissible intersection (it is never widened after seeing power results). The
    CI-level z is the frozen screen 80% z for the screen chain and the 95% z for the confirm
    chain (D2 10.3/10.4).
    """
    if not candidate_ns:
        raise ValueError("no admissible candidate n (capacity audit must pass first)")
    reference33 = reference_vector(quantities)
    per_n: dict[int, Any] = {}
    selected_n: int | None = None
    for n in (int(v) for v in candidate_ns):
        if progress:
            progress(f"resampling the joint gate vector at n={n} "
                     f"({n_resamples} whole-group draws, seed stream [{seed}, {n}])")
        draws47 = resample_draws(quantities, n, n_resamples=n_resamples, seed=seed,
                                 progress=progress)
        assert_structural_identities(draws47)
        draws33 = draws47[:, list(POW.POWER_SLOTS)]
        se33 = draws33.std(axis=0, ddof=1)
        scenarios: dict[str, Any] = {}
        for rho in (float(s) for s in shrink_scenarios):
            transformed, d_idle_t = transform_draws(draws33, reference33, rho)
            evaluation = evaluate_corrected_draws(transformed, d_idle_t, se33, z=z)
            scenarios[f"{rho:.2f}"] = {
                "joint_power": float(evaluation["passed"].mean()),
                "marginal_power": {k: float(v.mean())
                                   for k, v in evaluation["conditions"].items()},
            }
        main = scenarios[f"{float(selection_scenario):.2f}"]
        binding = min(main["marginal_power"], key=main["marginal_power"].get)
        transformed75, d_idle75 = transform_draws(draws33, reference33, selection_scenario)
        per_n[n] = {
            "scenarios": scenarios,
            "joint_power_at_selection": main["joint_power"],
            "binding_condition": binding,
            "binding_power": main["marginal_power"][binding],
            "standard_errors": {name: float(se33[i])
                                for i, name in enumerate(POW.POWER_QUANTITY_NAMES)},
            "stress_table": _stress_table(transformed75, d_idle75, se33, reference33, z=z),
            "structural_identities_verified": True,
        }
        if selected_n is None and main["joint_power"] >= float(target):
            selected_n = n
    return {
        "selected_n": selected_n,
        "underpowered": selected_n is None,
        "per_n": per_n,
        "reference": {name: float(reference33[i])
                      for i, name in enumerate(POW.POWER_QUANTITY_NAMES)},
        "candidate_ns": [int(n) for n in candidate_ns],
        "n_reference": int(np.asarray(quantities["regret"]).size),
        "selection_scenario": float(selection_scenario),
        "target": float(target),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "ci_z": float(z),
    }


# --- read-only validation-quantities loading + old-report parity --------------------------------------------


def load_d2e_validation_quantities() -> dict[str, Any]:
    """Load the frozen D2E repaired-pipeline validation quantities (READ-ONLY power input)."""
    npz_path = PROT.D2E_DIR / "development" / POW.QUANTITIES_NPZ_NAME
    receipt_path = PROT.D2E_DIR / "development" / POW.QUANTITIES_RECEIPT_NAME
    if not npz_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(
            f"the D2E frozen power quantities are missing: {npz_path} / {receipt_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    baselines = {key[len(_BASELINE_NPZ_PREFIX):]: np.asarray(value, dtype=np.float64)
                 for key, value in data.items() if key.startswith(_BASELINE_NPZ_PREFIX)}
    quantities: dict[str, Any] = {
        key: np.asarray(value) for key, value in data.items()
        if not key.startswith(_BASELINE_NPZ_PREFIX)}
    quantities["baseline_risks"] = baselines
    quantities["strongest_baseline"] = str(receipt["strongest_standard_baseline"])
    quantities["n_groups"] = int(np.asarray(data["n_groups"]).item())
    quantities["qa"] = {key: True for key in D2STAT.QA_KEYS}
    quantities["projection_nondegenerate"] = True
    quantities["eval_mask_nondegenerate"] = True
    POW._validate_power_quantities(quantities)  # frozen fail-closed validator (read-only reuse)
    return quantities


def assert_old_report_parity(quantities: Mapping[str, Any]) -> dict[str, Any]:
    """Cross-check the identity vector against the frozen D2E report theta (fail closed).

    The corrected power must start from EXACTLY the old validation quantities: the identity
    evaluation of all 33 slots must reproduce the old report's ``theta_hat`` bitwise (1e-12);
    any drift means the input or the vector assembly diverged from the frozen old run.
    Returns the old report's quotable summary (its terminal state is never re-adjudicated).
    """
    report_path = PROT.D2E_DIR / "power" / POW.REPORT_NAME
    old = json.loads(report_path.read_text(encoding="utf-8"))
    reference33 = reference_vector(quantities)
    old_theta = old["theta_hat"]
    diffs = {name: abs(float(reference33[i]) - float(old_theta[name]))
             for i, name in enumerate(POW.POWER_QUANTITY_NAMES)}
    worst = max(diffs.values())
    if worst > 1e-12:
        raise ValueError(
            f"the identity power vector diverges from the frozen D2E report theta "
            f"(max |diff|={worst:.3e}); the corrected power refuses to start from a "
            f"different input")
    return {
        "old_report_path": str(report_path),
        "old_report_sha256": sha256_file(report_path),
        "old_status": str(old["status"]),
        "old_selected_n": old["selected_n"],
        "old_joint_power_075_at_n2500": float(
            old["per_n"]["2500"]["joint_power_at_selection"]),
        "old_binding_condition_at_n2500": str(old["per_n"]["2500"]["binding_condition"]),
        "old_bootstrap_seed": int(old["bootstrap"]["seed"]),
        "identity_parity_max_abs_diff": float(worst),
        "note": ("the OLD report is quoted as a preserved terminal state "
                 "(D2E_SCREEN_UNDERPOWERED); it is never overwritten or re-adjudicated, and "
                 "its overall 0.75 x theta shrink is a pre-collection power-simulation "
                 "calibre defect, not evidence about the model"),
    }


# --- orchestration -----------------------------------------------------------------------------------------


def run_power_correction(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Execute Phase P corrected power and write the create-once report. No collection.

    Prerequisites (fail closed): the create-once source binding passed and the pool capacity
    audit passed with a non-empty admissible screen n set. ``D2F_SCREEN_UNDERPOWERED`` is a
    preregistered terminal state that forbids any manifest / lock / collection.
    """
    binding_path = PROT.D2F_DIR / PROT.SOURCE_BINDING_NAME
    audit_path = PROT.D2F_DIR / PROT.POOL_AUDIT_NAME
    if not binding_path.is_file():
        raise RuntimeError("run the source binding before the corrected power")
    if not audit_path.is_file():
        raise RuntimeError("run the dual-pool capacity audit before the corrected power")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != PROT.STATUS_AUDIT_PASS:
        raise RuntimeError(
            f"the pool capacity audit ended as {audit.get('status')!r}; the corrected power "
            f"must not run")
    admissible = [int(n) for n in audit["capacity"]["screen_admissible_ns"]]
    if not admissible:
        raise RuntimeError("the capacity-admissible screen n set is empty")

    POWER_DIR.mkdir(parents=True, exist_ok=True)
    report_path = POWER_DIR / REPORT_NAME
    if report_path.exists():
        if progress:
            progress("power_correction_report.json already exists (create-once); reloading")
        return json.loads(report_path.read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    if progress:
        progress("loading the frozen D2E validation quantities (read-only) and checking "
                 "identity parity against the old report")
    quantities = load_d2e_validation_quantities()
    old_summary = assert_old_report_parity(quantities)
    if progress:
        progress(f"parity OK (max |diff|={old_summary['identity_parity_max_abs_diff']:.3e}); "
                 f"simulating the corrected AND-chain at admissible n={admissible}")
    result = simulate_corrected_power(quantities, candidate_ns=admissible, progress=progress)
    status = STATUS_UNDERPOWERED if result["underpowered"] else STATUS_POWER_PASS

    payload = {
        "plan_id": PROT.PLAN_ID,
        "scope": "ADAPTIVE_DEVELOPMENT_ONLY",
        "status": status,
        "terminal": bool(result["underpowered"]),
        "selected_n": result["selected_n"],
        "underpowered": bool(result["underpowered"]),
        "target": result["target"],
        "selection_scenario": result["selection_scenario"],
        "shrink_scenarios": [float(s) for s in SHRINK_SCENARIOS],
        "n_reference": result["n_reference"],
        "admissible_ns": result["candidate_ns"],
        "per_n": {str(n): entry for n, entry in result["per_n"].items()},
        "reference": result["reference"],
        "power_seed": int(POWER_SEED),
        "n_resamples": int(result["n_resamples"]),
        "confidence": float(CONFIDENCE),
        "seed_derivation": ("per-n whole-group resample streams derive deterministically from "
                            "SeedSequence([2026122406, n]); distinct from the D2E power seed "
                            "2026122405 and from every consumed stream"),
        "typing": {
            "effect_slots": [POW.POWER_QUANTITY_NAMES[i]
                             for i in classify_power_slots()["effect"]] + ["d_idle (derived)"],
            "support_slots": [POW.POWER_QUANTITY_NAMES[i]
                              for i in classify_power_slots()["support"]],
            "anchor_slots": [POW.POWER_QUANTITY_NAMES[i]
                             for i in classify_power_slots()["anchor"]],
            "rule": ("model/decision effects shrink as E_b + (rho-1)*E_ref; D_idle is formed "
                     "from the same resample's two regrets and shrunk as a DIFFERENCE; "
                     "scenario/task support keeps its coupled resampled value with oracle / "
                     "stratum shares summing to 1 on every draw; c5 tests ONLY "
                     "D_idle^(rho) >= -tie_tolerance"),
        },
        "old_algorithm": old_summary,
        "stress_table_note": ("the per-n scenario-support stress tables (reactive share "
                              "0.24/0.22/0.20; high-regret share 0.30/0.20/0.15) are "
                              "NON-SELECTIVE sensitivity; they never participate in the n "
                              "selection and never change the c2/c6 thresholds"),
        "semantics": ("a collection-feasibility decision under the adaptively-used old "
                      "validation; never a promise that the new screen passes. A selected n "
                      "permits Phase S; D2F_SCREEN_UNDERPOWERED forbids any new ID manifest / "
                      "lock / collection."),
        "bindings": {
            "source_binding_sha256": sha256_file(binding_path),
            "pool_capacity_audit_sha256": sha256_file(audit_path),
            "power_validation_quantities_sha256": sha256_file(
                PROT.D2E_DIR / "development" / POW.QUANTITIES_NPZ_NAME),
            "d2f_power_code_sha256": sha256_file(Path(__file__).resolve()),
            "d2f_protocol_code_sha256": sha256_file(
                PROT.ROOT / "code/ac_twin/stage4d_d2f_protocol.py"),
            "d2e_power_code_sha256": sha256_file(
                PROT.ROOT / "code/ac_twin/stage4d_d2e_power.py"),
            "statistics_code_sha256": sha256_file(
                PROT.ROOT / "code/ac_twin/stage4d_external_statistics.py"),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(report_path, payload)
    if progress:
        progress(f"Phase-P corrected power COMPLETE status={status} "
                 f"selected_n={result['selected_n']} "
                 f"total_elapsed={time.perf_counter()-t0:.1f}s")
    payload["report_path"] = str(report_path)
    return payload


# --- Phase C: confirm power from the frozen D2F screen quantities (plan 4) ----------------------------

CONFIRM_DIR = PROT.D2F_DIR / PROT.CONFIRM_DIR_NAME
CONFIRM_REPORT_NAME = "power_report.json"
_SCREEN_REPORT_REL = Path("development") / "development_screen_report.json"
_SCREEN_QUANTITIES_REL = Path("development") / "screen_group_quantities.npz"
_SCREEN_QUANTITIES_RECEIPT_REL = Path("development") / "screen_group_quantities_receipt.json"


def load_d2f_screen_quantities() -> dict[str, Any]:
    """Load the frozen D2F screen group quantities (READ-ONLY confirm-power input, plan 4)."""
    npz_path = PROT.D2F_DIR / _SCREEN_QUANTITIES_REL
    receipt_path = PROT.D2F_DIR / _SCREEN_QUANTITIES_RECEIPT_REL
    if not npz_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(
            f"the frozen D2F screen quantities are missing: {npz_path} / {receipt_path}; the "
            "single screen adjudication must complete first (plan 4)")
    with np.load(npz_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    baselines = {key[len(_BASELINE_NPZ_PREFIX):]: np.asarray(value, dtype=np.float64)
                 for key, value in data.items() if key.startswith(_BASELINE_NPZ_PREFIX)}
    quantities: dict[str, Any] = {
        key: np.asarray(value) for key, value in data.items()
        if not key.startswith(_BASELINE_NPZ_PREFIX)}
    quantities["baseline_risks"] = baselines
    quantities["strongest_baseline"] = str(receipt["strongest_standard_baseline"])
    quantities["n_groups"] = int(np.asarray(data["n_groups"]).item())
    quantities["qa"] = {key: bool(value) for key, value in receipt["qa"].items()}
    quantities["projection_nondegenerate"] = bool(receipt["projection_nondegenerate"])
    quantities["eval_mask_nondegenerate"] = bool(receipt["eval_mask_nondegenerate"])
    POW._validate_power_quantities(quantities)  # frozen fail-closed validator (read-only reuse)
    return quantities


def run_confirm_power(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Execute the Phase-C confirm power and write the create-once report (plan 4). No collection.

    Prerequisites (fail closed): the single screen adjudication passed BOTH gates
    (``D2F_DEV_READY_FOR_CONFIRM_POWER``) and the pre-frozen capacity audit lists a non-empty
    admissible confirm n set (``{1000,1500,2000,2500,3000}`` intersected with capacity BEFORE
    any power ran). The ONLY input is the frozen ``screen_group_quantities.npz`` (D2 9.1: the
    per-group table, never a reverse-derived SE). The 95% CI complete confirm AND-chain is
    simulated jointly; the smallest admissible n whose 75% model-effect joint power reaches
    0.80 is selected, otherwise the preregistered terminal ``D2F_CONFIRM_UNDERPOWERED`` is
    written and NO confirm lock / manifest / collection is created. This is conditional power
    GIVEN the screen passed — never an unconditional success probability.
    """
    PROT.assert_confirm_phase_open()
    audit_path = PROT.D2F_DIR / PROT.POOL_AUDIT_NAME
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != PROT.STATUS_AUDIT_PASS:
        raise RuntimeError(f"the pool capacity audit ended as {audit.get('status')!r}; the "
                           "confirm power must not run")
    admissible = [int(n) for n in audit["capacity"]["confirm_admissible_ns"]]
    if not admissible:
        raise RuntimeError("the pre-frozen capacity-admissible confirm n set is empty")

    CONFIRM_DIR.mkdir(parents=True, exist_ok=True)
    report_path = CONFIRM_DIR / CONFIRM_REPORT_NAME
    if report_path.exists():
        if progress:
            progress("confirm/power_report.json already exists (create-once); reloading")
        return json.loads(report_path.read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    if progress:
        progress("loading the frozen D2F screen group quantities (read-only, D2 9.1 table)")
    quantities = load_d2f_screen_quantities()
    if progress:
        progress(f"simulating the 95% CI confirm AND-chain at admissible n={admissible} "
                 f"(fresh entropy {CONFIRM_POWER_SEED})")
    result = simulate_corrected_power(quantities, candidate_ns=admissible,
                                      seed=CONFIRM_POWER_SEED, z=CONFIRM_Z, progress=progress)
    status = PROT.STATUS_CONFIRM_UNDERPOWERED if result["underpowered"] \
        else PROT.STATUS_CONFIRM_POWER_PASS

    payload = {
        "plan_id": PROT.PLAN_ID,
        "scope": "CONDITIONAL_POWER_GIVEN_SCREEN_PASSED",
        "status": status,
        "terminal": bool(result["underpowered"]),
        "selected_n": result["selected_n"],
        "underpowered": bool(result["underpowered"]),
        "target": result["target"],
        "selection_scenario": result["selection_scenario"],
        "shrink_scenarios": [float(s) for s in SHRINK_SCENARIOS],
        "n_reference": result["n_reference"],
        "admissible_ns": result["candidate_ns"],
        "per_n": {str(n): entry for n, entry in result["per_n"].items()},
        "reference": result["reference"],
        "power_seed": int(CONFIRM_POWER_SEED),
        "n_resamples": int(result["n_resamples"]),
        "confidence": float(CONFIRM_CONFIDENCE),
        "ci_z": float(CONFIRM_Z),
        "seed_derivation": ("per-n whole-group resample streams derive deterministically from "
                            "SeedSequence([2026122408, n]); distinct from the D2F screen power "
                            "seed 2026122406 and the screen adjudication entropy 2026122407"),
        "typing": {
            "effect_slots": [POW.POWER_QUANTITY_NAMES[i]
                             for i in classify_power_slots()["effect"]] + ["d_idle (derived)"],
            "support_slots": [POW.POWER_QUANTITY_NAMES[i]
                              for i in classify_power_slots()["support"]],
            "anchor_slots": [POW.POWER_QUANTITY_NAMES[i]
                             for i in classify_power_slots()["anchor"]],
            "rule": ("the Phase-P effect/support separation is unchanged: model/decision "
                     "effects shrink as E_b + (rho-1)*E_ref with the screen point estimates as "
                     "the reference; scenario/task support keeps its coupled resampled value; "
                     "CI conditions use the two-sided 95% z (D2 10.3/10.4)"),
        },
        "semantics": ("conditional power GIVEN the D2F screen passed both gates; never an "
                      "unconditional success probability. A selected n permits the confirm "
                      "lock; D2F_CONFIRM_UNDERPOWERED forbids any confirm manifest / lock / "
                      "collection (plan 4 / plan 7)."),
        "bindings": {
            "screen_report_sha256": sha256_file(PROT.D2F_DIR / _SCREEN_REPORT_REL),
            "screen_group_quantities_sha256": sha256_file(
                PROT.D2F_DIR / _SCREEN_QUANTITIES_REL),
            "screen_group_quantities_receipt_sha256": sha256_file(
                PROT.D2F_DIR / _SCREEN_QUANTITIES_RECEIPT_REL),
            "pool_capacity_audit_sha256": sha256_file(audit_path),
            "screen_lock_sha256": sha256_file(PROT.D2F_DIR / PROT.SCREEN_LOCK_NAME),
            "d2f_power_code_sha256": sha256_file(Path(__file__).resolve()),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(report_path, payload)
    if progress:
        progress(f"Phase-C confirm power COMPLETE status={status} "
                 f"selected_n={result['selected_n']} "
                 f"total_elapsed={time.perf_counter()-t0:.1f}s")
    payload["report_path"] = str(report_path)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    last = {"at": 0.0}

    def progress(line: str) -> None:
        now = time.perf_counter()
        if now - last["at"] >= 5.0:
            last["at"] = now
            print(f"[stage4d-d2f-power] {line}", flush=True)

    report = run_power_correction(progress=progress)
    print(f"[stage4d-d2f-power] status={report['status']} selected_n={report['selected_n']} "
          f"underpowered={report['underpowered']}", flush=True)
    return 5 if report["underpowered"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANCHOR_SLOTS",
    "CONFIDENCE",
    "EFFECT_SLOTS",
    "N_POWER_QUANTITIES",
    "POWER_DIR",
    "POWER_RESAMPLES",
    "POWER_SEED",
    "POWER_TARGET",
    "REPORT_NAME",
    "SCREEN_Z",
    "SELECTION_SCENARIO",
    "SHRINK_SCENARIOS",
    "STATUS_POWER_PASS",
    "STATUS_UNDERPOWERED",
    "STRESS_HIGH_REGRET_CENTERS",
    "STRESS_REACTIVE_CENTERS",
    "SUPPORT_SLOTS",
    "assert_old_report_parity",
    "assert_structural_identities",
    "classify_power_slots",
    "evaluate_corrected_draws",
    "load_d2e_validation_quantities",
    "reference_vector",
    "resample_draws",
    "run_power_correction",
    "simulate_corrected_power",
    "transform_draws",
]
