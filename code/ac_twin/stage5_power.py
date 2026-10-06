"""Stage 5 joint power for the SOTA-M gate (plan v0.3 section 6.2, Task 4).

Plan ``AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3``. Uses the corrected
D2F calibre: model/decision EFFECT centres shrink by rho in {1, .75, .5}
(``E^(rho) = E_b + (rho-1) * E_ref``, same resampling covariance/sign/pairing);
scenario/task SUPPORT proportions are NEVER shrunk (they keep their coupled
resampled values; oracle and stratum shares sum exactly to 1 on every draw,
asserted fail-closed). The whole decision group is the only resampling unit; the
same index vector recomputes every quantity of a draw (paired: same group, same
selected action, same model-seed index). The full randomizable SOTA-M AND-chain
(r1-r5) is simulated jointly — marginal powers are never multiplied, and CI lower
bounds use the resampling SE at the CURRENT n at the frozen two-sided 95% z (the
single holdout adjudication runs at 95%; integrity conditions are preconditions,
not probabilities).

Materiality early-stop (plan 6.2): ``0.75*I_AURC < 0.05`` or
``0.75*I_REGRET80 < 0.03`` on validation -> ``STAGE5_MERGE_SOTA_NOT_MATERIAL``; no
n grid, no holdout. If material but no candidate n reaches 0.90 joint power at
75% retention -> ``STAGE5_MERGE_COMPARISON_UNDERPOWERED``.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ac_twin import stage4cp_statistics as s4cs
from ac_twin import stage4cr2_protocol as s4cr2

# --- frozen thresholds (SOTA-M gate inherits the Stage 4C-R2 task contract) -------------
I_AURC_MIN = s4cr2.VALIDATION_I_AURC_MIN              # 0.05
I_REGRET80_MIN = s4cr2.VALIDATION_I_REGRET80_MIN      # 0.03
MIN_POSITIVE_SEEDS = s4cr2.VALIDATION_MIN_POSITIVE_SEEDS  # 4
STRATA_I_AURC_FLOOR = s4cr2.GATE_STRATA_I_AURC_FLOOR  # -0.10
STRATUM_MIN_SHARE = s4cr2.STRATUM_IMBALANCE_FRAC      # 0.20
MIN_HIGH_REGRET_FRAC = s4cr2.GATE_TASK_MIN_HIGH_REGRET_FRAC  # 0.15
MIN_DISTINCT_ORACLE = s4cr2.GATE_TASK_MIN_DISTINCT_ORACLE    # 3
MIN_ORACLE_FRAC = s4cr2.GATE_TASK_MIN_ORACLE_FRAC     # 0.05
TIE_TOLERANCE = s4cr2.BASELINE_TIE_TOLERANCE          # 1e-12
HIGH_REGRET_THRESHOLD = s4cr2.HIGH_REGRET_THRESHOLD   # 0.01
COVERAGE_MAIN = 0.80
N_SEEDS = 5
Z95 = 1.959963984540054  # the single adjudication runs at two-sided 95%

MATERIALITY_SHRINK = 0.75
SHRINK_SCENARIOS = (1.0, 0.75, 0.5)
SELECTION_SCENARIO = 0.75
POWER_TARGET = 0.90

STATUS_NOT_MATERIAL = "STAGE5_MERGE_SOTA_NOT_MATERIAL"
STATUS_UNDERPOWERED = "STAGE5_MERGE_COMPARISON_UNDERPOWERED"
STATUS_POWER_PASS = "STAGE5_MERGE_POWER_PASS"  # descriptive, NON-terminal

# --- quantity-vector layout (37 slots) ---------------------------------------------------------
Q_I_AURC = 0
Q_I_REGRET80 = 1
Q_BASE_VS_RANDOM = 2
Q_TSTAR_LOW = 3
Q_TSTAR_HIGH = 4
Q_INTERACT_LOW = 5
Q_INTERACT_HIGH = 6
Q_AURC_SOTA = 7
Q_R80_SOTA = 8
Q_BASE_MEAN = 9
Q_IDLE_MEAN = 10
Q_RANDOM_MEAN = 11
Q_HIGH_REGRET_FRAC = 12
Q_TSTAR_LOW_SHARE = 13
Q_TSTAR_HIGH_SHARE = 14
Q_INTERACT_LOW_SHARE = 15
Q_INTERACT_HIGH_SHARE = 16
Q_ORACLE = 17            # 17..26: ten oracle-action shares
Q_SEED_I_AURC = 27       # 27..31
Q_SEED_I_R80 = 32        # 32..36
N_QUANTITIES = 37

STRATA_EFFECT_SLOTS = (Q_TSTAR_LOW, Q_TSTAR_HIGH, Q_INTERACT_LOW, Q_INTERACT_HIGH)
STRATA_SHARE_SLOTS = (Q_TSTAR_LOW_SHARE, Q_TSTAR_HIGH_SHARE,
                      Q_INTERACT_LOW_SHARE, Q_INTERACT_HIGH_SHARE)
ORACLE_SLOTS = tuple(range(Q_ORACLE, Q_ORACLE + 10))
SEED_I_AURC_SLOTS = tuple(range(Q_SEED_I_AURC, Q_SEED_I_AURC + N_SEEDS))
SEED_I_R80_SLOTS = tuple(range(Q_SEED_I_R80, Q_SEED_I_R80 + N_SEEDS))

QUANTITY_NAMES = (
    "i_aurc", "i_regret80", "base_vs_random",
    "tstar_low_i_aurc", "tstar_high_i_aurc",
    "interaction_low_i_aurc", "interaction_high_i_aurc",
    "aurc_sota", "regret80_sota", "base_mean_regret", "idle_mean_regret",
    "random_mean_regret", "high_regret_frac",
    "tstar_low_share", "tstar_high_share",
    "interaction_low_share", "interaction_high_share",
) + tuple(f"oracle_share_{a}" for a in range(10)) \
    + tuple(f"seed{i}_i_aurc" for i in range(N_SEEDS)) \
    + tuple(f"seed{i}_i_regret80" for i in range(N_SEEDS))

REQUIRED_KEYS = (
    "regret", "y_high", "idle_regret", "random_regret", "a_star",
    "tstar_layer", "interaction_layer",
    "actwin_ensemble", "actwin_by_seed", "sota_ensemble", "sota_by_seed",
)


def classify_power_slots() -> dict[str, tuple[int, ...]]:
    """The exact effect/anchor/support partition of the 37 slots (fail closed)."""
    effect = (Q_I_AURC, Q_I_REGRET80, Q_BASE_VS_RANDOM) + STRATA_EFFECT_SLOTS \
        + SEED_I_AURC_SLOTS + SEED_I_R80_SLOTS
    anchor = (Q_AURC_SOTA, Q_R80_SOTA, Q_BASE_MEAN, Q_IDLE_MEAN, Q_RANDOM_MEAN)
    support = (Q_HIGH_REGRET_FRAC,) + STRATA_SHARE_SLOTS + ORACLE_SLOTS
    union = sorted(effect + anchor + support)
    if union != list(range(N_QUANTITIES)):
        raise AssertionError(f"power-slot typing is not an exact partition: {union}")
    return {"effect": effect, "anchor": anchor, "support": support}


# --- the joint quantity vector ------------------------------------------------------------------


def compute_quantity_vector(index: np.ndarray, quantities: Mapping[str, Any]) -> np.ndarray:
    """Evaluate all 37 quantities on ONE whole-group resample (fail-closed NaN semantics)."""
    index = np.asarray(index, dtype=np.int64)
    reg = np.asarray(quantities["regret"], dtype=np.float64)[index]
    ens = np.asarray(quantities["actwin_ensemble"], dtype=np.float64)[index]
    sota = np.asarray(quantities["sota_ensemble"], dtype=np.float64)[index]
    ens_seed = np.asarray(quantities["actwin_by_seed"], dtype=np.float64)[:, index]
    sota_seed = np.asarray(quantities["sota_by_seed"], dtype=np.float64)[:, index]
    idle = np.asarray(quantities["idle_regret"], dtype=np.float64)[index]
    rand = np.asarray(quantities["random_regret"], dtype=np.float64)[index]
    a_star = np.asarray(quantities["a_star"], dtype=np.int64)[index]

    a_sota = s4cs.aurc_regret(s4cs.order_by_risk(sota), reg)
    a_ens = s4cs.aurc_regret(s4cs.order_by_risk(ens), reg)
    r80_sota = s4cs.regret_at_coverage(s4cs.order_by_risk(sota), reg, COVERAGE_MAIN)
    r80_ens = s4cs.regret_at_coverage(s4cs.order_by_risk(ens), reg, COVERAGE_MAIN)

    q = np.full(N_QUANTITIES, np.nan, dtype=np.float64)
    q[Q_I_AURC] = s4cs.improvement(a_sota, a_ens)
    q[Q_I_REGRET80] = s4cs.improvement(r80_sota, r80_ens)
    rand_mean = float(rand.mean())
    base_mean = float(reg.mean())
    q[Q_BASE_VS_RANDOM] = ((rand_mean - base_mean) / rand_mean
                           if rand_mean > 0 else float("nan"))
    for slot, layer_key, offset in ((Q_TSTAR_LOW, "tstar_layer", 0),
                                    (Q_INTERACT_LOW, "interaction_layer", 0)):
        layer = np.asarray(quantities[layer_key], dtype=np.int64)[index]
        for level in (0, 1):
            mask = layer == level
            if mask.sum() >= 2:
                q[slot + level] = s4cs.improvement(
                    s4cs.aurc_regret(s4cs.order_by_risk(sota[mask]), reg[mask]),
                    s4cs.aurc_regret(s4cs.order_by_risk(ens[mask]), reg[mask]))
    q[Q_AURC_SOTA] = a_sota
    q[Q_R80_SOTA] = r80_sota
    q[Q_BASE_MEAN] = base_mean
    q[Q_IDLE_MEAN] = float(idle.mean())
    q[Q_RANDOM_MEAN] = rand_mean
    q[Q_HIGH_REGRET_FRAC] = float((reg > HIGH_REGRET_THRESHOLD).mean())
    tstar = np.asarray(quantities["tstar_layer"], dtype=np.int64)[index]
    inter = np.asarray(quantities["interaction_layer"], dtype=np.int64)[index]
    q[Q_TSTAR_LOW_SHARE] = float((tstar == 0).mean())
    q[Q_TSTAR_HIGH_SHARE] = float((tstar == 1).mean())
    q[Q_INTERACT_LOW_SHARE] = float((inter == 0).mean())
    q[Q_INTERACT_HIGH_SHARE] = float((inter == 1).mean())
    for a in range(10):
        q[Q_ORACLE + a] = float((a_star == a).mean())
    for i in range(N_SEEDS):
        a_s = s4cs.aurc_regret(s4cs.order_by_risk(sota_seed[i]), reg)
        r_s = s4cs.regret_at_coverage(s4cs.order_by_risk(sota_seed[i]), reg, COVERAGE_MAIN)
        q[Q_SEED_I_AURC + i] = s4cs.improvement(
            a_s, s4cs.aurc_regret(s4cs.order_by_risk(ens_seed[i]), reg))
        q[Q_SEED_I_R80 + i] = s4cs.improvement(
            r_s, s4cs.regret_at_coverage(s4cs.order_by_risk(ens_seed[i]), reg,
                                         COVERAGE_MAIN))
    return q


def reference_vector(quantities: Mapping[str, Any]) -> np.ndarray:
    """The 37-slot identity evaluation (E_ref / support anchors); fail closed."""
    for key in REQUIRED_KEYS:
        if key not in quantities:
            raise ValueError(f"power quantities missing {key!r}")
    identity = np.arange(int(np.asarray(quantities["regret"]).size), dtype=np.int64)
    vector = compute_quantity_vector(identity, quantities)
    if not np.isfinite(vector).all():
        raise ValueError("non-finite identity power quantity on validation (fail closed)")
    return vector


def resample_draws(quantities: Mapping[str, Any], n: int, *,
                   n_resamples: int, seed: int,
                   progress: Callable[[str], None] | None = None) -> np.ndarray:
    """Joint whole-group resamples of size n; per-n stream from SeedSequence([seed, n])."""
    groups = int(np.asarray(quantities["regret"]).size)
    if groups < 2:
        raise ValueError("power needs at least 2 groups")
    generator = np.random.default_rng(np.random.SeedSequence([int(seed), int(n)]))
    draws = np.empty((int(n_resamples), N_QUANTITIES), dtype=np.float64)
    for draw in range(int(n_resamples)):
        draws[draw] = compute_quantity_vector(
            generator.integers(0, groups, size=int(n)), quantities)
        if progress and (draw + 1) % 1000 == 0:
            progress(f"n={n}: joint resample {draw + 1}/{n_resamples}")
    return draws


def assert_structural_identities(draws: np.ndarray) -> None:
    """Fail closed unless oracle/stratum shares sum to 1 and stay in [0, 1] per draw."""
    draws = np.asarray(draws, dtype=np.float64)
    atol = 1e-9
    oracle = draws[:, list(ORACLE_SLOTS)]
    tstar = draws[:, [Q_TSTAR_LOW_SHARE, Q_TSTAR_HIGH_SHARE]]
    inter = draws[:, [Q_INTERACT_LOW_SHARE, Q_INTERACT_HIGH_SHARE]]
    if not np.allclose(oracle.sum(axis=1), 1.0, rtol=0.0, atol=atol):
        raise ValueError("oracle action shares do not sum to 1 on every draw")
    if not np.allclose(tstar.sum(axis=1), 1.0, rtol=0.0, atol=atol):
        raise ValueError("t_star stratum shares do not sum to 1 on every draw")
    if not np.allclose(inter.sum(axis=1), 1.0, rtol=0.0, atol=atol):
        raise ValueError("interaction stratum shares do not sum to 1 on every draw")
    support = draws[:, [Q_HIGH_REGRET_FRAC] + list(STRATA_SHARE_SLOTS) + list(ORACLE_SLOTS)]
    if float(support.min()) < -atol or float(support.max()) > 1.0 + atol:
        raise ValueError("a support proportion left [0, 1] on a draw")


def transform_draws(draws: np.ndarray, reference: np.ndarray, rho: float):
    """E^(rho) = E_b + (rho-1)*E_ref on EFFECT slots + D_idle; support/anchor untouched."""
    draws = np.asarray(draws, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[1] != N_QUANTITIES \
            or reference.shape != (N_QUANTITIES,):
        raise ValueError("draws/reference must align with the 37-slot power vector")
    transformed = draws.copy()
    for slot in classify_power_slots()["effect"]:
        transformed[:, slot] = draws[:, slot] + (float(rho) - 1.0) * reference[slot]
    d_idle = draws[:, Q_IDLE_MEAN] - draws[:, Q_BASE_MEAN]
    d_idle_ref = reference[Q_IDLE_MEAN] - reference[Q_BASE_MEAN]
    d_idle_t = d_idle + (float(rho) - 1.0) * d_idle_ref
    return transformed, d_idle_t


def evaluate_gate_draws(transformed: np.ndarray, d_idle_t: np.ndarray,
                        standard_errors: np.ndarray, *, z: float = Z95) -> dict[str, Any]:
    """The randomizable SOTA-M AND-chain (r1-r5) over corrected joint draws.

    NaN comparisons evaluate False (fail closed): a non-positive denominator, a
    degenerate stratum, or a missing resample member fails its condition.
    """
    t = np.asarray(transformed, dtype=np.float64)
    se = np.asarray(standard_errors, dtype=np.float64)
    d_idle_t = np.asarray(d_idle_t, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] != N_QUANTITIES or se.shape != (N_QUANTITIES,) \
            or d_idle_t.shape != (t.shape[0],):
        raise ValueError("transformed draws / SE / D_idle must align with the power vector")

    def ci_lower(slot: int) -> np.ndarray:
        return t[:, slot] - float(z) * se[slot]

    seeds_both = ((t[:, list(SEED_I_AURC_SLOTS)] > 0.0)
                  & (t[:, list(SEED_I_R80_SLOTS)] > 0.0)).sum(axis=1)
    oracle_count = (t[:, list(ORACLE_SLOTS)] >= MIN_ORACLE_FRAC).sum(axis=1)
    conditions = {
        "r1_i_aurc": (t[:, Q_I_AURC] >= I_AURC_MIN) & (ci_lower(Q_I_AURC) > 0.0),
        "r2_i_regret80": (t[:, Q_I_REGRET80] >= I_REGRET80_MIN)
                         & (ci_lower(Q_I_REGRET80) > 0.0),
        "r3_seed_consistency": seeds_both >= MIN_POSITIVE_SEEDS,
        "r4_strata_protection": (
            (t[:, list(STRATA_EFFECT_SLOTS)].min(axis=1) >= STRATA_I_AURC_FLOOR)
            & (t[:, list(STRATA_SHARE_SLOTS)].min(axis=1) >= STRATUM_MIN_SHARE)),
        "r5_task_contract": (
            (t[:, Q_BASE_VS_RANDOM] > 0.0) & (ci_lower(Q_BASE_VS_RANDOM) > 0.0)
            & (d_idle_t >= -TIE_TOLERANCE)
            & (t[:, Q_HIGH_REGRET_FRAC] >= MIN_HIGH_REGRET_FRAC)
            & (oracle_count >= MIN_DISTINCT_ORACLE)),
    }
    passed = np.logical_and.reduce(list(conditions.values()))
    return {"passed": passed, "conditions": conditions}


# --- materiality early stop (plan 6.2) --------------------------------------------------------------


def check_materiality(i_aurc: float, i_regret80: float, *,
                      shrink: float = MATERIALITY_SHRINK) -> bool:
    """True iff both 75%-shrunk point estimates stay above the materiality lines."""
    i_aurc, i_regret80 = float(i_aurc), float(i_regret80)
    if not (np.isfinite(i_aurc) and np.isfinite(i_regret80)):
        return False
    return bool(float(shrink) * i_aurc >= I_AURC_MIN
                and float(shrink) * i_regret80 >= I_REGRET80_MIN)


# --- joint simulation ---------------------------------------------------------------------------------


def simulate_power(quantities: Mapping[str, Any], *,
                   candidate_ns: Sequence[int],
                   n_resamples: int,
                   seed: int,
                   shrink_scenarios: Sequence[float] = SHRINK_SCENARIOS,
                   selection_scenario: float = SELECTION_SCENARIO,
                   target: float = POWER_TARGET,
                   z: float = Z95,
                   progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Joint SOTA-M power at each candidate n under the corrected calibre.

    The 75% model-effect scenario is the only blocking judge; the smallest
    candidate n reaching ``target`` is selected, else ``underpowered`` (the
    candidate set is frozen before any power run and never widened after).
    """
    if not candidate_ns:
        raise ValueError("no candidate n (the frozen set must be non-empty)")
    reference = reference_vector(quantities)
    per_n: dict[int, Any] = {}
    selected_n: int | None = None
    for n in (int(v) for v in candidate_ns):
        if progress:
            progress(f"resampling the joint SOTA-M vector at n={n} "
                     f"({n_resamples} whole-group draws, seed stream [{seed}, {n}])")
        draws = resample_draws(quantities, n, n_resamples=n_resamples, seed=seed,
                               progress=progress)
        assert_structural_identities(draws)
        se = draws.std(axis=0, ddof=1)
        scenarios: dict[str, Any] = {}
        for rho in (float(s) for s in shrink_scenarios):
            transformed, d_idle_t = transform_draws(draws, reference, rho)
            evaluation = evaluate_gate_draws(transformed, d_idle_t, se, z=z)
            scenarios[f"{rho:.2f}"] = {
                "joint_power": float(evaluation["passed"].mean()),
                "marginal_power": {k: float(v.mean())
                                   for k, v in evaluation["conditions"].items()},
            }
        main = scenarios[f"{float(selection_scenario):.2f}"]
        binding = min(main["marginal_power"], key=main["marginal_power"].get)
        per_n[n] = {
            "scenarios": scenarios,
            "joint_power_at_selection": main["joint_power"],
            "binding_condition": binding,
            "binding_power": main["marginal_power"][binding],
            "standard_errors": {name: float(se[i])
                                for i, name in enumerate(QUANTITY_NAMES)},
            "structural_identities_verified": True,
        }
        if selected_n is None and main["joint_power"] >= float(target):
            selected_n = n
    return {
        "selected_n": selected_n,
        "underpowered": selected_n is None,
        "per_n": per_n,
        "reference": {name: float(reference[i])
                      for i, name in enumerate(QUANTITY_NAMES)},
        "candidate_ns": [int(n) for n in candidate_ns],
        "n_reference": int(np.asarray(quantities["regret"]).size),
        "selection_scenario": float(selection_scenario),
        "target": float(target),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "ci_z": float(z),
    }


__all__ = [
    "COVERAGE_MAIN", "I_AURC_MIN", "I_REGRET80_MIN", "MATERIALITY_SHRINK",
    "MIN_POSITIVE_SEEDS", "N_QUANTITIES", "POWER_TARGET", "QUANTITY_NAMES",
    "Q_I_AURC", "Q_I_REGRET80", "Q_BASE_VS_RANDOM", "Q_AURC_SOTA", "Q_R80_SOTA",
    "Q_BASE_MEAN", "Q_IDLE_MEAN", "Q_RANDOM_MEAN", "Q_HIGH_REGRET_FRAC",
    "Q_ORACLE", "Q_SEED_I_AURC", "Q_SEED_I_R80", "Q_TSTAR_LOW_SHARE",
    "SELECTION_SCENARIO", "SHRINK_SCENARIOS", "STATUS_NOT_MATERIAL",
    "STATUS_POWER_PASS", "STATUS_UNDERPOWERED", "Z95",
    "assert_structural_identities", "check_materiality", "classify_power_slots",
    "compute_quantity_vector", "evaluate_gate_draws", "reference_vector",
    "resample_draws", "simulate_power", "transform_draws",
]


# --- Phase P power orchestration (plan 6.2) ---------------------------------------------------
#
# Long-running entry point: reads the frozen validation artifacts, applies the
# materiality early-stop, then simulates the full randomizable SOTA-M AND-chain
# jointly at each frozen candidate n. Terminal plain-text progress per the
# project long-task rule. Create-once: power_report.json is never rewritten.

ROOT = Path(__file__).resolve().parents[2]
POWER_DIR_NAME = "power"
POWER_REPORT_NAME = "power_report.json"


def run_power(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Phase P (plan 6.2): materiality early-stop, then joint power / selected_n.

    Fail closed unless the development freeze completed first. If the validation
    75%-shrunk point estimates fall below the materiality lines the terminal
    state ``STAGE5_MERGE_SOTA_NOT_MATERIAL`` is written WITHOUT running the n
    grid and no comparison holdout may be created. If material but no candidate
    n reaches 0.90 joint power at 75% retention the terminal state is
    ``STAGE5_MERGE_COMPARISON_UNDERPOWERED``.
    """
    import json

    from ac_twin import stage5_protocol as s5prot
    from ac_twin import stage5_selection as s5sel
    from ac_twin.stage1_data import sha256_file
    from ac_twin.stage3r_protocol import write_once_json

    checked = s5prot.validate_config(
        json.loads((s5sel.CONFIG_PATH).read_text(encoding="utf-8")))
    dev_dir = ROOT / checked["stage_root"] / s5sel.DEV_DIR_NAME
    final_path = dev_dir / s5sel.FINAL_SELECTION_NAME
    report_path = dev_dir / s5sel.VALIDATION_REPORT_NAME
    risks_path = dev_dir / s5sel.VALIDATION_RISKS_NAME
    for required in (final_path, report_path, risks_path):
        if not required.is_file():
            raise RuntimeError(f"the development freeze must complete first: {required}")
    power_dir = ROOT / checked["stage_root"] / POWER_DIR_NAME
    power_dir.mkdir(parents=True, exist_ok=True)
    power_path = power_dir / POWER_REPORT_NAME
    if power_path.exists():
        if progress:
            progress("power_report.json already exists (create-once); reloading")
        return json.loads(power_path.read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    final = json.loads(final_path.read_text(encoding="utf-8"))
    strongest = str(final["strongest_published_baseline"])
    with np.load(risks_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    quantities: dict[str, Any] = {
        "regret": data["regret"], "y_high": data["y_high"],
        "idle_regret": data["idle_regret"], "random_regret": data["random_regret"],
        "a_star": data["a_star"], "tstar_layer": data["tstar_layer"],
        "interaction_layer": data["interaction_layer"],
        "actwin_ensemble": data["actwin_ensemble"],
        "actwin_by_seed": data["actwin_by_seed"],
        "sota_ensemble": data[f"method_ensemble__{strongest}"],
        "sota_by_seed": data[f"method_by_seed__{strongest}"],
    }
    reference = reference_vector(quantities)
    point_i_aurc = float(reference[Q_I_AURC])
    point_i_r80 = float(reference[Q_I_REGRET80])
    material = check_materiality(point_i_aurc, point_i_r80,
                                 shrink=float(checked["materiality_shrink"]))
    if progress:
        progress(f"reference: i_aurc={point_i_aurc:.4f} i_regret80={point_i_r80:.4f} "
                 f"material={material}")

    payload: dict[str, Any] = {
        "plan_id": checked["plan_id"],
        "strongest_published_baseline": strongest,
        "reference": {name: float(reference[i]) for i, name in enumerate(QUANTITY_NAMES)},
        "materiality": {
            "shrink": float(checked["materiality_shrink"]),
            "shrunk_i_aurc": float(checked["materiality_shrink"]) * point_i_aurc,
            "shrunk_i_regret80": float(checked["materiality_shrink"]) * point_i_r80,
            "i_aurc_min": float(checked["materiality_i_aurc_min"]),
            "i_regret80_min": float(checked["materiality_i_regret80_min"]),
            "material": bool(material),
        },
        "bindings": {
            "final_method_selection_sha256": sha256_file(final_path),
            "validation_comparison_report_sha256": sha256_file(report_path),
            "validation_risks_sha256": sha256_file(risks_path),
            "stage5_power_code_sha256": sha256_file(Path(__file__).resolve()),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if not material:
        payload.update({
            "status": STATUS_NOT_MATERIAL,
            "terminal": True,
            "selected_n": None,
            "semantics": ("the validation 75%-shrunk point estimate of at least one "
                          "co-primary endpoint fell below its materiality line; per plan "
                          "6.2 the n grid was NOT run and no comparison holdout may be "
                          "created"),
        })
        write_once_json(power_path, payload)
        if progress:
            progress(f"terminal {STATUS_NOT_MATERIAL}; no n grid, no holdout "
                     f"(elapsed={time.perf_counter()-t0:.0f}s)")
        return payload

    result = simulate_power(
        quantities, candidate_ns=[int(n) for n in checked["power_candidate_n"]],
        n_resamples=int(checked["power_monte_carlo"]),
        seed=int(checked["power_mc_entropy"]), progress=progress)
    status = STATUS_UNDERPOWERED if result["underpowered"] else STATUS_POWER_PASS
    payload.update({
        "status": status,
        "terminal": bool(result["underpowered"]),
        "selected_n": result["selected_n"],
        "underpowered": bool(result["underpowered"]),
        "target": result["target"],
        "selection_scenario": result["selection_scenario"],
        "shrink_scenarios": [float(s) for s in SHRINK_SCENARIOS],
        "candidate_ns": result["candidate_ns"],
        "per_n": {str(n): entry for n, entry in result["per_n"].items()},
        "n_reference": result["n_reference"],
        "n_resamples": result["n_resamples"],
        "ci_z": result["ci_z"],
        "power_monte_carlo_entropy": int(checked["power_mc_entropy"]),
        "power_bootstrap_entropy": int(checked["power_bootstrap_entropy"]),
        "seed_derivation": ("per-n whole-group resample streams derive deterministically "
                            "from SeedSequence([power_monte_carlo_entropy, n]); the "
                            "validation comparison bootstrap used the distinct frozen "
                            "power_bootstrap_entropy; both differ from the adjudication "
                            "entropy and the holdout schedule entropy"),
        "typing": {
            "effect_slots": [QUANTITY_NAMES[i]
                             for i in classify_power_slots()["effect"]] + ["d_idle (derived)"],
            "support_slots": [QUANTITY_NAMES[i]
                              for i in classify_power_slots()["support"]],
            "anchor_slots": [QUANTITY_NAMES[i]
                             for i in classify_power_slots()["anchor"]],
            "rule": ("model/decision effects shrink as E_b + (rho-1)*E_ref; D_idle is "
                     "formed from the same resample's two regrets and shrunk as a "
                     "DIFFERENCE; scenario/task support keeps its coupled resampled "
                     "value (oracle/stratum shares sum to 1 on every draw, asserted); "
                     "CI conditions use the resampling SE at the current n at the "
                     "two-sided 95% z; marginal powers are never multiplied"),
        },
        "semantics": ("a collection-feasibility decision under the adaptively-used "
                      "validation; never a promise that the holdout adjudication "
                      "passes. A selected n permits Task 5 collection; "
                      "STAGE5_MERGE_COMPARISON_UNDERPOWERED forbids any holdout "
                      "manifest/lock/collection"),
    })
    write_once_json(power_path, payload)
    if progress:
        progress(f"Phase-P power COMPLETE status={status} selected_n={result['selected_n']} "
                 f"elapsed={time.perf_counter()-t0:.0f}s")
    payload["report_path"] = str(power_path)
    return payload


def main() -> int:
    last = {"at": 0.0}

    def progress(line: str) -> None:
        now = time.perf_counter()
        if now - last["at"] >= 5.0:
            last["at"] = now
            print(f"[stage5-power] {line}", flush=True)

    report = run_power(progress=progress)
    print(f"[stage5-power] summary: status={report['status']} "
          f"selected_n={report.get('selected_n')}", flush=True)
    return 0 if report["status"] == STATUS_POWER_PASS else 5


if __name__ == "__main__":
    raise SystemExit(main())
