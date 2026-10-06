"""Stage 4D-D2E plan-4.2 development-screen power pre-check (cheap Go/No-Go, NO collection).

Plan ``AC-TWIN-RFER-STAGE4D-D2E-EGO-CONSISTENT-EXTERNAL-2026-09-24-V0.1`` section 4 step 2.
Legal ONLY after the section-3.3 validation gate passed (``D2E_INTERFACE_REPAIR_QUALIFIED``) and
BEFORE any new collection / manifest / lock. It estimates the JOINT power of the COMPLETE D2
screen AND-chain (Gate TASK-X-DEV c1-c8 + Gate RISK-X-DEV r1-r6) by whole-group paired resampling
of the OLD validation role under the REPAIRED pipeline (the selected ego candidate + the retrained
Direct head). It is ``ADAPTIVE_DEVELOPMENT_ONLY``: validation was already used for D2/D2R model
early stopping / baseline selection, so this is a conservative collection-feasibility screen,
never a promise that the new screen passes.

Frozen contract (plan 4.2 / config.json): candidate new-screen sizes ``n in {1000,1500,2000,
2500}``; two-sided 80% CI (``SCREEN_Z``); 10,000 paired-bootstrap resamples for the covariance;
10,000 Monte-Carlo draws; the covariance is taken DIRECTLY from the validation paired-bootstrap
draws (NEVER reverse-derived from an aggregate CI width); effect shrink {100%, 75%, 50%} all
reported with signs never flipped; the 75% scenario is the ONLY blocking selection judge; the
point-estimate materiality lines (c5 base-vs-random, r1 I_AURC>=0.05, r2 I_REGRET80>=0.03) enter
every simulated draw (never rescued by "larger n"); the 5-seed consistency and the stratification
support are simulated jointly with the metric slots (never a product of marginal powers). The
smallest n whose 75%-shrink joint power reaches 0.80 is selected. If n=2500 still falls short, the
terminal state is ``D2E_SCREEN_UNDERPOWERED`` and NO collection / manifest / lock is created.

How the full AND-chain is simulated: the frozen D2 screen quantity vector
(``stage4d_external_statistics.compute_screen_quantity_vector``, reused bitwise, slots 0..31)
carries the metric slots; the structural fractions the D2 gate reads off the per-group arrays
(oracle action shares c3, reactive fraction c6, the four stratum fractions c7) are APPENDED as
slots 32..46 so a single paired bootstrap / Monte-Carlo draw couples EVERY statistical condition
through the SAME group multiplicities. The structural conditions that are NOT statistical (Gate Q
isolation, n/n completeness, zero ID overlap, no outcome rejection, denominator / projection /
mask validity) are conditioned on; if they fail the real adjudication is
``STAGE4D_EXTERNAL_INVALID_ANALYSIS``, not a power shortfall.

The metric definitions and gate thresholds are REUSED bitwise from ``stage4d_external_statistics``
(never redefined); only the extended-vector assembly, the draw-AND-chain and the MVN power driver
are new. The torch-side quantity assembly (repair the validation e2e + score the retrained Direct)
lives here too but is separated from the torch-free power math so the latter is testable on
synthetic data. Nothing here modifies a frozen D2/D2R artifact; all writes land under
``experiments/stage4_reliability/stage4d_d2e/``.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ac_twin import stage4d_d2e_ego as ego
from ac_twin import stage4d_d2e_pipeline as PIPE
from ac_twin import stage4d_external_inference as INF
from ac_twin import stage4d_external_statistics as D2STAT
from ac_twin import stage4d_external_training as T
from ac_twin.checkpointing import atomic_save_npz
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

ROOT = Path(__file__).resolve().parents[2]
D2E_DIR = ego.D2E_DIR
DEV_DIR = D2E_DIR / "development"
POWER_DIR = D2E_DIR / "power"
D2_VALIDATION_ROLE_DIR = INF.ROLES_DIR / "validation"

# --- frozen plan-4.2 contract (config.json) ------------------------------------------------------
#: Fresh frozen paired-bootstrap entropy for the D2E power pre-check. Plan 4.2 fixes 10,000
#: resamples + 80% CI but names no seed; this constant is distinct from every consumed stream
#: (R2 screen 2026092702, R2 confirm 2026093202, D1 confirm 2026110995, D2 screen 2026122101,
#: D2R phase0/power/screen 2026122401/02/03). FLAGGED for review.
POWER_BOOTSTRAP_SEED = 2026122405
POWER_BOOTSTRAP_RESAMPLES = 10000
POWER_MONTE_CARLO = 10000
CANDIDATE_NS = (1000, 1500, 2000, 2500)
SHRINK_SCENARIOS = (1.0, 0.75, 0.5)
SELECTION_SCENARIO = 0.75
POWER_TARGET = 0.80
CONFIDENCE = D2STAT.CONFIDENCE_SCREEN            # 0.80 (the screen never uses 95%)
#: z for the two-sided 80% CI lower bound (standard-normal ppf(0.90)); mirrors the screen.
SCREEN_Z = 1.2815515655446004

STATUS_UNDERPOWERED = "D2E_SCREEN_UNDERPOWERED"
STATUS_POWER_PASS = "D2E_SCREEN_POWER_PASS"      # descriptive, NON-terminal (the plan continues)

QUANTITIES_NPZ_NAME = "power_validation_quantities.npz"
QUANTITIES_RECEIPT_NAME = "power_validation_quantities_receipt.json"
REPORT_NAME = "screen_power_report.json"

_BASELINE_NPZ_PREFIX = "baseline_risk__"
N_MODEL_SEEDS = len(T.MODEL_SEEDS)                # 5 frozen Direct seeds (matches the Q_SEED slots)

# --- extended power quantity-vector layout ---------------------------------------------------------
# Slots 0..31 are the frozen D2 screen vector (``compute_screen_quantity_vector``); slots 32..46
# append the structural fractions the D2 gate reads off the per-group arrays so they are jointly
# simulated with the metric slots.
X_ORACLE = D2STAT.N_QUANTITIES                    # 32..41 (10 oracle action shares, c3)
X_REACTIVE = X_ORACLE + T.N_ACTIONS               # 42 (reactive fraction, c6)
X_STRATA = X_REACTIVE + 1                         # 43..46 (tstar_low/high, interaction_low/high, c7)
N_EXTENDED = X_STRATA + 4                         # 47

#: gate-relevant statistical subset of the 47-vector, in order. The absolute-curve / reporting
#: slots (Q_AURC_BASELINE, Q_R80_BASELINE, Q_RANDOM_MEAN_REGRET, Q_BASELINE_CURVE, Q_*_MODEL) only
#: derive conditions already modelled; the structural condition c1 / c8 / r5 / r6 are NOT
#: statistical and the power estimate is conditional on a structurally valid screen.
POWER_SLOTS = (
    D2STAT.Q_I_AURC, D2STAT.Q_I_REGRET80, D2STAT.Q_PR, D2STAT.Q_SPEARMAN_BRANCH,
    D2STAT.Q_BASE_VS_RANDOM, D2STAT.Q_BASE_MEAN_REGRET, D2STAT.Q_IDLE_MEAN_REGRET,
    D2STAT.Q_HIGH_REGRET_FRAC,
) + tuple(range(D2STAT.Q_SEED_I_AURC, D2STAT.Q_SEED_I_AURC + N_MODEL_SEEDS)) \
    + tuple(range(D2STAT.Q_SEED_I_R80, D2STAT.Q_SEED_I_R80 + N_MODEL_SEEDS)) \
    + tuple(range(X_ORACLE, X_ORACLE + T.N_ACTIONS)) \
    + (X_REACTIVE,) \
    + tuple(range(X_STRATA, X_STRATA + 4))
N_POWER_QUANTITIES = len(POWER_SLOTS)             # 33

# Local indices inside the power subset (lists: numpy fancy-indexing requires them).
P_I_AURC = 0
P_I_REGRET80 = 1
P_PR = 2
P_SPEARMAN = 3
P_BASE_VS_RANDOM = 4
P_BASE_MEAN = 5
P_IDLE_MEAN = 6
P_HIGH_REGRET = 7
P_SEED_I_AURC = list(range(8, 8 + N_MODEL_SEEDS))  # 8..12
P_SEED_I_R80 = list(range(13, 13 + N_MODEL_SEEDS))  # 13..17
P_ORACLE = list(range(18, 18 + T.N_ACTIONS))       # 18..27
P_REACTIVE = 28
P_STRATA = list(range(29, 33))                     # 29..32

#: name every power slot explicitly (robust to the private ``_QUANTITY_NAMES`` table layout).
POWER_QUANTITY_NAMES = (
    "i_aurc_regret", "i_regret80", "pr_auc_minus_prevalence", "spearman_branch",
    "base_vs_random_improvement", "base_mean_regret", "idle_mean_regret", "high_regret_frac",
) + tuple(f"seed{s}_i_aurc" for s in T.MODEL_SEEDS) \
    + tuple(f"seed{s}_i_regret80" for s in T.MODEL_SEEDS) \
    + tuple(f"oracle_share_a{a}" for a in range(T.N_ACTIONS)) \
    + ("reactive_frac",) \
    + ("tstar_low_frac", "tstar_high_frac", "interaction_low_frac", "interaction_high_frac")


# --- the extended quantity vector (frozen 32 D2 slots + structural fractions) ---------------------


def compute_d2e_power_vector(index: np.ndarray, quantities: Mapping[str, Any]) -> np.ndarray:
    """Evaluate the 32 frozen D2 screen quantities + the 15 structural fractions on ONE resample.

    The first 32 slots are computed by the frozen ``compute_screen_quantity_vector`` (bitwise
    reuse; improvement slots are NaN on a degenerate baseline denominator, which the power
    pre-check turns into a fail-closed error). Slots 32..46 are recomputed from the SAME
    with-replacement group index vector so every condition stays coupled through the same group
    multiplicities.
    """
    index = np.asarray(index, dtype=np.int64)
    q = np.full(N_EXTENDED, np.nan, dtype=np.float64)
    q[:D2STAT.N_QUANTITIES] = D2STAT.compute_screen_quantity_vector(index, quantities)

    a_star = np.asarray(quantities["a_star"], dtype=np.int64)[index]
    for a in range(T.N_ACTIONS):
        q[X_ORACLE + a] = float((a_star == a).mean())
    q[X_REACTIVE] = float(np.asarray(quantities["reactive"], dtype=bool)[index].mean())
    tstar = np.asarray(quantities["tstar_layer"], dtype=np.int64)[index]
    inter = np.asarray(quantities["interaction_layer"], dtype=np.int64)[index]
    q[X_STRATA + 0] = float((tstar == 0).mean())
    q[X_STRATA + 1] = float((tstar == 1).mean())
    q[X_STRATA + 2] = float((inter == 0).mean())
    q[X_STRATA + 3] = float((inter == 1).mean())
    return q


# --- the plan-4.2 power AND-chain (vectorised over joint Monte-Carlo draws) -------------------------


def evaluate_screen_draws(draws: np.ndarray, standard_errors: np.ndarray) -> dict[str, Any]:
    """Vectorised D2 screen statistical AND-chain over joint Monte-Carlo draws of the power subset.

    ``draws`` is ``(replicates, 33)``; ``standard_errors`` is ``sqrt(diag(cov_n))`` for the same
    subset. CI conditions use the screen-side 80% bound (``estimate - SCREEN_Z*SE > 0``) computed
    from the draw covariance; fraction / oracle / strata / seed-count / base-vs-idle conditions
    use point values, exactly mirroring
    :func:`stage4d_external_statistics.evaluate_screen_gates`. The point-estimate materiality
    lines (c5 base>random, r1>=0.05, r2>=0.03) enter every draw (never rescued by "larger n").
    The structural conditions (c1 completeness+QA, c8 denominator/projection/mask, r5, r6) are
    NOT statistical and are conditioned on.
    """
    draws = np.asarray(draws, dtype=np.float64)
    se = np.asarray(standard_errors, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[1] != N_POWER_QUANTITIES:
        raise ValueError(f"draws must align as (replicates, {N_POWER_QUANTITIES})")
    if se.shape != (N_POWER_QUANTITIES,):
        raise ValueError("standard_errors must align with the power quantity subset")

    def ci80_lower(slot: int) -> np.ndarray:
        return draws[:, slot] - SCREEN_Z * se[slot]

    seeds_both = ((draws[:, P_SEED_I_AURC] > 0.0)
                  & (draws[:, P_SEED_I_R80] > 0.0)).sum(axis=1)
    oracle_count = (draws[:, P_ORACLE] >= D2STAT.GATE_TASK_MIN_ORACLE_FRAC).sum(axis=1)
    strata_min = draws[:, P_STRATA].min(axis=1)
    conditions = {
        # Gate TASK-X-DEV statistical conditions (c1/c8 structural -> conditioned on).
        "c2_high_regret_frac": draws[:, P_HIGH_REGRET] >= D2STAT.GATE_TASK_MIN_HIGH_REGRET_FRAC,
        "c3_oracle_coverage": oracle_count >= D2STAT.GATE_TASK_MIN_DISTINCT_ORACLE,
        "c4_spearman_ci_lower_positive": ci80_lower(P_SPEARMAN) > 0.0,
        "c5_base_vs_random_and_not_worse_than_idle": (
            (draws[:, P_BASE_VS_RANDOM] > 0.0) & (ci80_lower(P_BASE_VS_RANDOM) > 0.0)
            & (draws[:, P_BASE_MEAN] <= draws[:, P_IDLE_MEAN] + T.BASELINE_TIE_TOLERANCE)),
        "c6_reactive_frac": draws[:, P_REACTIVE] >= D2STAT.GATE_TASK_MIN_REACTIVE_FRAC,
        "c7_strata_floor": strata_min >= D2STAT.GATE_TASK_MIN_STRATUM_FRAC,
        # Gate RISK-X-DEV statistical conditions (r5/r6 structural -> conditioned on).
        "r1_i_aurc_regret": (draws[:, P_I_AURC] >= D2STAT.I_AURC_MIN)
                            & (ci80_lower(P_I_AURC) > 0.0),
        "r2_i_regret80": (draws[:, P_I_REGRET80] >= D2STAT.I_REGRET80_MIN)
                         & (ci80_lower(P_I_REGRET80) > 0.0),
        "r3_pr_auc_minus_prevalence": (draws[:, P_PR] > 0.0) & (ci80_lower(P_PR) > 0.0),
        "r4_seed_consistency_both_positive": seeds_both >= D2STAT.MIN_POSITIVE_SEEDS,
    }
    passed = np.logical_and.reduce(list(conditions.values()))
    return {"passed": passed, "conditions": conditions}


# --- covariance + Monte-Carlo power (adapts the frozen cr2/d2r parametric-MVN method) ---------------


def symmetrised_psd_covariance(draws: np.ndarray) -> np.ndarray:
    """Symmetrised, eigenvalue-clipped covariance of bootstrap draw columns (fail-closed)."""
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[0] < 2:
        raise ValueError("draws must align as (n_resamples, n_quantities)")
    if not np.isfinite(draws).all():
        raise ValueError("non-finite bootstrap draw in a power quantity (fail-closed)")
    covariance = np.cov(draws, rowvar=False)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    clipped = np.clip(eigenvalues, 1e-12, None)
    covariance = (eigenvectors * clipped) @ eigenvectors.T
    return 0.5 * (covariance + covariance.T)


def _monte_carlo_seed(bootstrap_seed: int, data_sha256: str) -> int:
    """Deterministic MC seed derived from the frozen power entropy + data hash (no new constant)."""
    material = int(data_sha256[:8], 16) % (2**31)
    return int(np.random.SeedSequence([int(bootstrap_seed), material]).generate_state(1)[0])


def simulate_power(theta_hat: np.ndarray, covariance: np.ndarray, *,
                   candidate_ns: Sequence[int], n_reference: int, monte_carlo: int,
                   seed: int, shrink_scenarios: Sequence[float],
                   selection_scenario: float, target: float) -> dict[str, Any]:
    """Smallest n whose SELECTION-scenario (75%) joint power reaches the target.

    Covariance scales by ``n_reference / n``; effects shrink to ``scenario * theta`` with signs
    preserved (never flipped). The gate is the FULL D2 screen statistical AND-chain on each joint
    draw (never a product of marginal powers). Only the 75% scenario selects; 100%/50% are
    sensitivity.
    """
    theta_hat = np.asarray(theta_hat, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if theta_hat.shape != (N_POWER_QUANTITIES,) \
            or covariance.shape != (N_POWER_QUANTITIES, N_POWER_QUANTITIES):
        raise ValueError("theta/covariance must match the power quantity subset")
    generator = np.random.default_rng(int(seed))
    max_n = int(max(candidate_ns))

    per_n: dict[int, Any] = {}
    selected_n: int | None = None
    for n in (int(v) for v in candidate_ns):
        cov_n = covariance * (float(n_reference) / n)
        se_n = np.sqrt(np.diag(cov_n))
        scenarios: dict[str, Any] = {}
        for shrink in (float(s) for s in shrink_scenarios):
            mean = float(shrink) * theta_hat
            draws = generator.multivariate_normal(mean, cov_n, size=int(monte_carlo))
            evaluation = evaluate_screen_draws(draws, se_n)
            scenarios[f"{shrink:.2f}"] = {
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
        }
        if selected_n is None and main["joint_power"] >= float(target):
            selected_n = n

    return {
        "selected_n": selected_n,
        "underpowered": selected_n is None,
        "per_n": per_n,
        "n_reference": int(n_reference),
        "candidate_ns": [int(n) for n in candidate_ns],
        "selection_scenario": float(selection_scenario),
        "target": float(target),
        "max_candidate_n": max_n,
        "sensitivity_at_max_n": {
            scenario: per_n[max_n]["scenarios"][scenario]["joint_power"]
            for scenario in (f"{float(s):.2f}" for s in shrink_scenarios)
        },
    }


def _paired_bootstrap(quantities: Mapping[str, Any], *, seed: int, n_resamples: int,
                      confidence: float,
                      progress: Callable[[str], None] | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Whole-group paired bootstrap of the extended power vector; returns (point, draws).

    Every quantity of a draw is recomputed from the SAME with-replacement group index vector; the
    identity evaluation is the point estimate. The CI is never derived here — the covariance of
    the draws feeds the MVN power directly (plan 4.2 forbids reverse-deriving SE from a CI width).
    """
    groups = int(np.asarray(quantities["regret"]).size)
    if groups < 2:
        raise ValueError("paired bootstrap needs at least 2 groups")
    generator = np.random.default_rng(int(seed))
    identity = np.arange(groups, dtype=np.int64)
    point = compute_d2e_power_vector(identity, quantities)
    draws = np.empty((int(n_resamples), N_EXTENDED), dtype=np.float64)
    for draw in range(int(n_resamples)):
        draws[draw] = compute_d2e_power_vector(
            generator.integers(0, groups, size=groups), quantities)
        if progress and (draw + 1) % 1000 == 0:
            progress(f"paired bootstrap {draw + 1}/{n_resamples}")
    return point, draws


def estimate_screen_power(quantities: Mapping[str, Any], *,
                          bootstrap_seed: int = POWER_BOOTSTRAP_SEED,
                          bootstrap_resamples: int = POWER_BOOTSTRAP_RESAMPLES,
                          monte_carlo: int = POWER_MONTE_CARLO,
                          candidate_ns: Sequence[int] = CANDIDATE_NS,
                          shrink_scenarios: Sequence[float] = SHRINK_SCENARIOS,
                          selection_scenario: float = SELECTION_SCENARIO,
                          target: float = POWER_TARGET,
                          data_sha256: str = "",
                          progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Joint D2 screen AND-chain power at each candidate n from the validation paired bootstrap."""
    n_reference = int(np.asarray(quantities["regret"]).size)
    if progress:
        progress(f"running the {bootstrap_resamples}-draw whole-group paired bootstrap "
                 f"(entropy {bootstrap_seed}) on validation (n_reference={n_reference}) for the "
                 f"covariance")
    point, draws = _paired_bootstrap(quantities, seed=int(bootstrap_seed),
                                     n_resamples=int(bootstrap_resamples),
                                     confidence=CONFIDENCE, progress=progress)
    subset = list(POWER_SLOTS)
    theta_hat = np.asarray(point, dtype=np.float64)[subset]
    sub_draws = np.asarray(draws, dtype=np.float64)[:, subset]
    if not np.isfinite(theta_hat).all() or not np.isfinite(sub_draws).all():
        raise ValueError(
            "non-finite identity/draw in the power subset (a fail-closed baseline denominator "
            "went degenerate on validation); the power pre-check cannot run")
    covariance = symmetrised_psd_covariance(sub_draws)
    if progress:
        progress("covariance ready; simulating the D2 screen AND-chain over joint Monte-Carlo "
                 "draws at each candidate n")
    result = simulate_power(
        theta_hat, covariance, candidate_ns=candidate_ns, n_reference=n_reference,
        monte_carlo=monte_carlo, seed=_monte_carlo_seed(bootstrap_seed, data_sha256),
        shrink_scenarios=shrink_scenarios, selection_scenario=selection_scenario, target=target)
    theta_named = {name: float(theta_hat[i]) for i, name in enumerate(POWER_QUANTITY_NAMES)}
    se_named = {name: float(np.sqrt(covariance[i, i]))
                for i, name in enumerate(POWER_QUANTITY_NAMES)}
    return {
        **result,
        "theta_hat": theta_named,
        "se_reference": se_named,
        "n_reference": n_reference,
        "monte_carlo": int(monte_carlo),
        "bootstrap": {
            "seed": int(bootstrap_seed),
            "n_resamples": int(bootstrap_resamples),
            "confidence": float(CONFIDENCE),
            "design": ("covariance taken DIRECTLY from the validation whole-group paired "
                       "bootstrap draws (frozen entropy); an 80% SE is never reverse-derived "
                       "from the CI width"),
        },
    }


# --- read-only torch-side assembly of the validation screen quantities ----------------------------


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _quantities_to_npz_payload(quantities: Mapping[str, Any]) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "group_seed": np.asarray(quantities["group_seed"], dtype=np.int64),
        "regret": np.asarray(quantities["regret"], dtype=np.float64),
        "y_high": np.asarray(quantities["y_high"], dtype=np.int64),
        "random_regret": np.asarray(quantities["random_regret"], dtype=np.float64),
        "idle_regret": np.asarray(quantities["idle_regret"], dtype=np.float64),
        "ensemble_risk": np.asarray(quantities["ensemble_risk"], dtype=np.float64),
        "risk_by_seed": np.asarray(quantities["risk_by_seed"], dtype=np.float64),
        "J_hat": np.asarray(quantities["J_hat"], dtype=np.float64),
        "J_true": np.asarray(quantities["J_true"], dtype=np.float64),
        "a_star": np.asarray(quantities["a_star"], dtype=np.int64),
        "tstar_layer": np.asarray(quantities["tstar_layer"], dtype=np.int64),
        "interaction_layer": np.asarray(quantities["interaction_layer"], dtype=np.int64),
        "t_star": np.asarray(quantities["t_star"], dtype=np.int64),
        "interaction_min_distance": np.asarray(
            quantities["interaction_min_distance"], dtype=np.float64),
        "reactive": np.asarray(quantities["reactive"], dtype=bool),
        "n_groups": np.int64(int(quantities["n_groups"])),
    }
    for name, risk in quantities["baseline_risks"].items():
        payload[_BASELINE_NPZ_PREFIX + name] = np.asarray(risk, dtype=np.float64).ravel()
    return payload


def _quantities_from_npz(data: Mapping[str, np.ndarray], receipt: Mapping[str, Any]
                         ) -> dict[str, Any]:
    baselines = {key[len(_BASELINE_NPZ_PREFIX):]: np.asarray(value, dtype=np.float64)
                 for key, value in data.items() if key.startswith(_BASELINE_NPZ_PREFIX)}
    quantities: dict[str, Any] = {
        key: np.asarray(value) for key, value in data.items()
        if not key.startswith(_BASELINE_NPZ_PREFIX)}
    quantities["baseline_risks"] = baselines
    quantities["strongest_baseline"] = str(receipt["strongest_standard_baseline"])
    quantities["n_groups"] = int(np.asarray(data["n_groups"]).item())
    # structural-validity flags: the power estimate is conditional on a structurally valid screen.
    quantities["qa"] = {key: True for key in D2STAT.QA_KEYS}
    quantities["projection_nondegenerate"] = True
    quantities["eval_mask_nondegenerate"] = True
    return quantities


def _validate_power_quantities(quantities: Mapping[str, Any]) -> int:
    """Fail closed on missing/misaligned/non-finite per-group power quantities."""
    regret = np.asarray(quantities["regret"], dtype=np.float64).ravel()
    groups = int(regret.size)
    if groups < 2:
        raise ValueError("power quantities need at least 2 groups")
    for key in ("group_seed", "y_high", "random_regret", "idle_regret", "ensemble_risk",
                "tstar_layer", "interaction_layer", "a_star", "t_star",
                "interaction_min_distance", "reactive"):
        if np.asarray(quantities[key]).ravel().shape != (groups,):
            raise ValueError(f"power quantity {key!r} must align as ({groups},)")
    for key in ("regret", "random_regret", "idle_regret", "ensemble_risk",
                "interaction_min_distance"):
        if not np.isfinite(np.asarray(quantities[key], dtype=np.float64)).all():
            raise ValueError(f"power quantity {key!r} must be finite")
    by_seed = np.asarray(quantities["risk_by_seed"], dtype=np.float64)
    if by_seed.shape != (N_MODEL_SEEDS, groups) or not np.isfinite(by_seed).all():
        raise ValueError(f"risk_by_seed must be finite and align as ({N_MODEL_SEEDS},{groups})")
    for key in ("J_hat", "J_true"):
        arr = np.asarray(quantities[key], dtype=np.float64)
        if arr.shape != (groups, T.N_ACTIONS) or not np.isfinite(arr).all():
            raise ValueError(f"power quantity {key!r} must be finite and align as ({groups},10)")
    baselines = quantities["baseline_risks"]
    strongest = str(quantities["strongest_baseline"])
    if strongest not in D2STAT.STANDARD_BASELINES:
        raise ValueError(f"strongest_baseline {strongest!r} is not a frozen eligible candidate")
    for name in D2STAT.STANDARD_BASELINES:
        risk = np.asarray(baselines[name], dtype=np.float64).ravel()
        if risk.shape != (groups,) or not np.isfinite(risk).all():
            raise ValueError(f"baseline risk {name!r} must be finite and group-aligned")
    return groups


def assemble_validation_quantities(device: str, batch_size: int,
                                   progress: Callable[[str], None] | None = None
                                   ) -> dict[str, Any]:
    """Assemble the plan-4.2 validation screen quantities under the REPAIRED pipeline (read-only).

    Uses the frozen ``final_selection.json`` selected candidate: repair the OLD validation e2e
    slot 0 with the trained ego branch, rebuild the consequence ``J_hat``/labels with the frozen
    D2 functions, score the retrained Direct head, and derive the oracle/reactive/strata arrays.
    Persisted create-once (``power_validation_quantities.npz`` + receipt) so the power estimate is
    reproducible and the Monte-Carlo seed binds to a real artifact. A rerun reloads the NPZ.
    """
    npz_path = DEV_DIR / QUANTITIES_NPZ_NAME
    receipt_path = DEV_DIR / QUANTITIES_RECEIPT_NAME
    if npz_path.exists() and receipt_path.exists():
        if progress:
            progress("power_validation_quantities.npz exists (create-once); reloading")
        with np.load(npz_path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        return _quantities_from_npz(data, _load_json(receipt_path))
    if npz_path.exists() != receipt_path.exists():
        raise RuntimeError(
            "power_validation_quantities npz/receipt mismatch (partial write); refusing to "
            "rebuild over an inconsistent state")

    def note(msg: str) -> None:
        if progress:
            progress(msg)

    final_path = DEV_DIR / "final_selection.json"
    report_path = DEV_DIR / "validation_report.json"
    if not final_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(
            "plan-4.2 power is legal only after the section-3.3 gate passed; "
            "final_selection.json / validation_report.json missing (run validate first)")
    final = _load_json(final_path)
    selected = str(final["selected_candidate"])
    recorded_strongest = str(
        _load_json(report_path)["candidate_gate"][selected]["strongest_baseline"])

    note(f"loading validation bundle + frozen base predictions; selected candidate={selected}")
    val_preds, val_risks = PIPE.load_persisted_base("validation")
    val_bundle = PIPE.load_bundle("validation", val_preds["group_seed"])
    val_arrays = INF.load_role_arrays(D2_VALIDATION_ROLE_DIR, "validation")
    scaler = PIPE.load_ego_scaler()
    models = PIPE.load_ego_models(selected)

    note("repairing the validation e2e slot 0 + rebuilding the consequence/labels")
    e2e = PIPE.candidate_e2e(val_bundle, val_preds["e2e"], scaler, models,
                             device=device, batch_size=batch_size)
    inputs = PIPE.build_direct_role_inputs(val_bundle, e2e, val_risks)

    note("scoring the retrained Direct head (5 seeds) on validation")
    inventory = PIPE.load_risk_inventory(selected)
    risk = T.score_models(
        inventory, {"tokens": inputs["tokens"], "global_features": inputs["global_features"],
                    "group_seed": inputs["group_seed"]}, device=device)

    # the strongest standard baseline is validation-frozen; re-derive deterministically and
    # cross-check against the section-3.3 adjudication (fail closed on any drift).
    derived_strongest = T.select_strongest_standard_baseline(
        {"regret": inputs["regret"], "y_high": inputs["y_high"],
         "baseline_risks": inputs["baselines"]}).strongest
    if derived_strongest != recorded_strongest:
        raise ValueError(
            f"strongest-baseline drift: validate recorded {recorded_strongest!r} but the power "
            f"assembly derived {derived_strongest!r}; the frozen baseline must never move")

    j_true = np.asarray(inputs["j_true"], dtype=np.float64)
    a_star = np.argmin(j_true, axis=-1)
    reactive = D2STAT.compute_reactive_response(
        val_arrays.object_factual_future, val_arrays.object_counterfactual_futures)
    t_star = np.asarray(val_arrays.t_star, dtype=np.int64).ravel()
    interaction_threshold = float(inventory.interaction_threshold)
    distance = np.asarray(inputs["interaction_min_distance"], dtype=np.float64).ravel()

    quantities: dict[str, Any] = {
        "group_seed": np.asarray(inputs["group_seed"], dtype=np.int64).ravel(),
        "regret": np.asarray(inputs["regret"], dtype=np.float64).ravel(),
        "y_high": np.asarray(inputs["y_high"], dtype=np.int64).ravel(),
        "random_regret": np.asarray(inputs["random_regret"], dtype=np.float64).ravel(),
        "idle_regret": np.asarray(inputs["idle_regret"], dtype=np.float64).ravel(),
        "ensemble_risk": np.asarray(risk.risk_ensemble, dtype=np.float64).ravel(),
        "risk_by_seed": np.asarray(risk.risk_by_seed, dtype=np.float64),
        "J_hat": np.asarray(inputs["j_hat"], dtype=np.float64),
        "J_true": j_true,
        "a_star": a_star.astype(np.int64),
        "tstar_layer": (t_star >= D2STAT.TSTAR_SPLIT).astype(np.int64),
        "interaction_layer": (distance > interaction_threshold).astype(np.int64),
        "t_star": t_star,
        "interaction_min_distance": distance,
        "reactive": np.asarray(reactive, dtype=bool),
        "baseline_risks": {k: np.asarray(v, dtype=np.float64).ravel()
                           for k, v in inputs["baselines"].items()},
        "strongest_baseline": recorded_strongest,
        "n_groups": int(np.asarray(inputs["group_seed"]).size),
        # structural-validity flags: conditioned on (the power estimate assumes a structurally
        # valid screen; a real denominator/projection/mask failure is INVALID_ANALYSIS, not a
        # power shortfall).
        "qa": {key: True for key in D2STAT.QA_KEYS},
        "projection_nondegenerate": True,
        "eval_mask_nondegenerate": True,
    }
    _validate_power_quantities(quantities)

    atomic_save_npz(npz_path, **_quantities_to_npz_payload(quantities))
    receipt = {
        "plan_id": ego.PLAN_ID,
        "scope": "ADAPTIVE_DEVELOPMENT_ONLY",
        "artifact": "power_validation_quantities",
        "create_once": True,
        "selected_candidate": selected,
        "strongest_standard_baseline": recorded_strongest,
        "interaction_threshold": interaction_threshold,
        "n_groups": int(quantities["n_groups"]),
        "quantities_npz_sha256": sha256_file(npz_path),
        "note": ("per-group OLD-validation screen quantities under the repaired pipeline "
                 "(selected ego candidate + retrained Direct), for the plan-4.2 power pre-check "
                 "ONLY; never a confirmation input. Structural-validity flags are conditioned on "
                 "(set True); the power estimate assumes a structurally valid screen."),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(receipt_path, receipt)
    return quantities


# --- the orchestration ----------------------------------------------------------------------------


def run_power(device: str = "cuda", batch_size: int = 4096,
              progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Execute the plan-4.2 power pre-check and write the create-once report. No collection.

    Create-once: a rerun with the report present reloads it. A selected n continues the plan to
    manifest/lock/collection; ``D2E_SCREEN_UNDERPOWERED`` is a preregistered terminal state that
    forbids any collection.
    """
    POWER_DIR.mkdir(parents=True, exist_ok=True)
    report_path = POWER_DIR / REPORT_NAME
    if report_path.exists():
        if progress:
            progress("screen_power_report.json already exists (create-once); reloading")
        return _load_json(report_path)

    t0 = time.perf_counter()
    quantities = assemble_validation_quantities(device, batch_size, progress=progress)
    if progress:
        progress(f"validation screen quantities assembled: {quantities['n_groups']} groups "
                 f"(repaired pipeline; elapsed={time.perf_counter()-t0:.1f}s)")
    npz_path = DEV_DIR / QUANTITIES_NPZ_NAME
    data_sha = sha256_file(npz_path)
    result = estimate_screen_power(quantities, data_sha256=data_sha, progress=progress)
    status = STATUS_UNDERPOWERED if result["underpowered"] else STATUS_POWER_PASS

    payload = {
        "plan_id": ego.PLAN_ID,
        "scope": "ADAPTIVE_DEVELOPMENT_ONLY",
        "status": status,
        "terminal": bool(result["underpowered"]),
        "selected_n": result["selected_n"],
        "underpowered": bool(result["underpowered"]),
        "target": result["target"],
        "selection_scenario": result["selection_scenario"],
        "monte_carlo": result["monte_carlo"],
        "n_reference": result["n_reference"],
        "candidate_ns": result["candidate_ns"],
        "shrink_scenarios": [float(s) for s in SHRINK_SCENARIOS],
        "per_n": {str(n): entry for n, entry in result["per_n"].items()},
        "sensitivity_at_max_n": result["sensitivity_at_max_n"],
        "theta_hat": result["theta_hat"],
        "se_reference": result["se_reference"],
        "bootstrap": result["bootstrap"],
        "conditioned_on": ("a structurally valid screen: Gate Q isolation, n/n completeness, "
                           "zero new/old ID overlap, no outcome rejection, and the denominator / "
                           "projection / mask validity are NOT statistical; their failure is "
                           "STAGE4D_EXTERNAL_INVALID_ANALYSIS, not a power shortfall"),
        "semantics": ("conservative development power pre-check from the adaptively-used "
                      "validation role under the repaired pipeline; never a promise that the "
                      "screen passes. A selected n permits the plan to continue to "
                      "manifest/lock/collection; D2E_SCREEN_UNDERPOWERED is a terminal state "
                      "that forbids collection."),
        "bindings": {
            "power_validation_quantities_sha256": data_sha,
            "final_selection_sha256": sha256_file(DEV_DIR / "final_selection.json"),
            "validation_report_sha256": sha256_file(DEV_DIR / "validation_report.json"),
            "power_code_sha256": sha256_file(Path(__file__).resolve()),
            "statistics_code_sha256": sha256_file(
                ROOT / "code/ac_twin/stage4d_external_statistics.py"),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(report_path, payload)
    if progress:
        progress(f"plan-4.2 power COMPLETE status={status} selected_n={result['selected_n']} "
                 f"total_elapsed={time.perf_counter()-t0:.1f}s")
    payload["report_path"] = str(report_path)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    last = {"at": 0.0}

    def progress(line: str) -> None:
        now = time.perf_counter()
        if now - last["at"] >= 5.0:
            last["at"] = now
            print(f"[stage4d-d2e-power] {line}", flush=True)

    report = run_power(progress=progress)
    print(f"[stage4d-d2e-power] status={report['status']} selected_n={report['selected_n']} "
          f"underpowered={report['underpowered']}", flush=True)
    return 5 if report["underpowered"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_NS",
    "CONFIDENCE",
    "N_EXTENDED",
    "N_POWER_QUANTITIES",
    "POWER_BOOTSTRAP_RESAMPLES",
    "POWER_BOOTSTRAP_SEED",
    "POWER_DIR",
    "POWER_MONTE_CARLO",
    "POWER_QUANTITY_NAMES",
    "POWER_SLOTS",
    "POWER_TARGET",
    "P_BASE_MEAN",
    "P_BASE_VS_RANDOM",
    "P_HIGH_REGRET",
    "P_IDLE_MEAN",
    "P_I_AURC",
    "P_I_REGRET80",
    "P_ORACLE",
    "P_PR",
    "P_REACTIVE",
    "P_SEED_I_AURC",
    "P_SEED_I_R80",
    "P_SPEARMAN",
    "P_STRATA",
    "REPORT_NAME",
    "SCREEN_Z",
    "SELECTION_SCENARIO",
    "SHRINK_SCENARIOS",
    "STATUS_POWER_PASS",
    "STATUS_UNDERPOWERED",
    "X_ORACLE",
    "X_REACTIVE",
    "X_STRATA",
    "assemble_validation_quantities",
    "compute_d2e_power_vector",
    "estimate_screen_power",
    "evaluate_screen_draws",
    "run_power",
    "simulate_power",
    "symmetrised_psd_covariance",
]
