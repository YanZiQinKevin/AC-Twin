"""Stage 4D-D2R plan-5 development-screen power pre-check (cheap Go/No-Go, NO collection).

Plan ``AC-TWIN-RFER-STAGE4D-D2R-RISK-GATED-IDLE-2026-09-24-V0.1`` section 5. Legal ONLY after
Phase 0 passed (``D2R_PHASE0_FEASIBLE``) and BEFORE any new collection / manifest / lock. It
estimates the JOINT power of the COMPLETE plan-7 AND gate by whole-group paired resampling of the
OLD validation role (n_reference = 1500). It is ``ADAPTIVE_DEVELOPMENT_ONLY``: validation was
already used for D2 early stopping / baseline selection, so this is a conservative development
pre-check, never a promise that the screen passes.

Frozen contract (plan 5): candidate new-screen sizes ``n in {1500, 2000, 2500}``; two-sided 80%
CI (``SCREEN_Z``); 10,000 Monte-Carlo draws; covariance taken DIRECTLY from the validation paired
bootstrap draws (never from a CI width); effect shrink {100%, 75%, 50%} all reported with signs
never flipped; the 75% scenario is the ONLY blocking selection judge; the 5% point-improvement
line enters every simulated draw (never rescued by "larger n"). The smallest n whose 75%-shrink
joint power reaches 0.80 is selected. If n=2500 still falls short, the terminal state is
``D2R_SCREEN_UNDERPOWERED`` and NO collection / manifest / lock / screen is created.

The power estimate is CONDITIONAL on a structurally valid screen: plan-7 condition 1 (Gate Q
isolation, n/n completeness, zero ID overlap, no outcome rejection) and the denominator /
projection / mask validity are NOT statistical; if they fail the adjudication is
``D2R_INVALID_ANALYSIS``, not a power shortfall.

Read-only inputs (all frozen, never re-fit): the Phase 0 ``diagnostic.npz`` per-group arrays, the
Phase 0 ``final_policy.json`` frozen thresholds (``tau_D``, per-seed, per-fallback, strongest
fallback), the D2 ``final_selection.json`` ``interaction_threshold`` (a frozen structural constant),
and the raw validation role arrays (for the truth-side reactive response). No torch stack is
re-run; the reactive response reuses the frozen D2 pilot computation bitwise.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ac_twin import stage4d_d2r_policy as P
from ac_twin import stage4d_d2r_statistics as STAT
from ac_twin import stage4d_external_inference as INF
from ac_twin import stage4d_external_statistics as D2STAT
from ac_twin.stage3r_protocol import write_once_json

ROOT = Path(__file__).resolve().parents[2]
D2_DEV_DIR = ROOT / "experiments/stage4_reliability/stage4d_external/development"
D2_VALIDATION_ROLE_DIR = ROOT / "experiments/stage4_reliability/stage4d_external/roles/validation"
D2R_DIR = ROOT / "experiments/stage4_reliability/stage4d_d2r"
PHASE0_DIR = D2R_DIR / "phase0"
POWER_DIR = D2R_DIR / "power"

# --- frozen plan-5 contract ----------------------------------------------------------------------
POWER_BOOTSTRAP_SEED = 2026122402
POWER_BOOTSTRAP_RESAMPLES = 10000
POWER_MONTE_CARLO = 10000
CANDIDATE_NS = (1500, 2000, 2500)
SHRINK_SCENARIOS = (1.0, 0.75, 0.5)
SELECTION_SCENARIO = 0.75
POWER_TARGET = 0.80

STATUS_UNDERPOWERED = "D2R_SCREEN_UNDERPOWERED"
STATUS_POWER_PASS = "D2R_POWER_PASS"          # descriptive, NON-terminal (the plan continues)

REPORT_NAME = "screen_power_report.json"


# --- read-only assembly of the validation plan-7 quantities ----------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validation_reactive(group_seed: np.ndarray) -> np.ndarray:
    """Truth-side reactive response for the validation role, aligned to ``group_seed``.

    Reuses the frozen D2 pilot reactive computation bitwise; legal on validation (truth side is
    adaptive-development). Aligned by group seed so the valid-subset ordering of the Phase 0
    diagnostic cannot drift.
    """
    arrays = INF.load_role_arrays(D2_VALIDATION_ROLE_DIR, "validation")
    reactive_all = D2STAT.compute_reactive_response(
        arrays.object_factual_future, arrays.object_counterfactual_futures)
    by_seed = {int(s): bool(r) for s, r in zip(arrays.group_seed, reactive_all)}
    out = np.empty(group_seed.size, dtype=bool)
    for i, s in enumerate(group_seed):
        key = int(s)
        if key not in by_seed:
            raise ValueError(f"validation group_seed {key} missing from the raw role arrays")
        out[i] = by_seed[key]
    return out


def build_validation_screen_quantities() -> dict[str, Any]:
    """Assemble the plan-7 screen quantities mapping for the OLD validation role (read-only).

    Reuses the frozen Phase 0 per-group arrays + frozen thresholds; computes the per-seed and
    per-fallback EXECUTED regrets with those frozen thresholds (never re-fit) and the reactive
    response from the raw role arrays. Returns the mapping consumed by
    :func:`stage4d_d2r_statistics.compute_screen_quantity_vector`.
    """
    diag_path = PHASE0_DIR / "diagnostic.npz"
    policy_path = PHASE0_DIR / "final_policy.json"
    if not diag_path.is_file() or not policy_path.is_file():
        raise FileNotFoundError(
            "plan-5 power requires the Phase 0 diagnostic.npz + final_policy.json (run Phase 0)")
    policy = _load_json(policy_path)
    if policy.get("phase0_status") != STAT.STATUS_PHASE0_FEASIBLE:
        raise ValueError(
            f"plan-5 power is legal only after Phase 0 passed ({STAT.STATUS_PHASE0_FEASIBLE}); "
            f"phase0_status={policy.get('phase0_status')!r}")
    interaction_threshold = float(
        _load_json(D2_DEV_DIR / "final_selection.json")["interaction_threshold"])

    with np.load(diag_path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}

    j_true = np.asarray(d["j_true"], dtype=np.float64)
    a_star = np.asarray(d["a_star"], dtype=np.int64)
    a0 = np.asarray(d["a0"], dtype=np.int64)
    group_seed = np.asarray(d["group_seed"], dtype=np.int64)
    risk_by_seed = np.asarray(d["risk_direct_by_seed"], dtype=np.float64)
    baseline_risks = {name: np.asarray(d[f"baseline__{name}"], dtype=np.float64)
                      for name in P.FALLBACK_BASELINES}

    # per-seed EXECUTED regrets under each frozen per-seed threshold (condition 6). The
    # ``risk_by_seed`` row order is the frozen ``model_seeds`` order recorded in final_policy.json.
    per_seed_tau = policy["per_seed_tau"]
    model_seeds = [int(s) for s in policy["model_seeds"]]
    if risk_by_seed.shape[0] != len(model_seeds):
        raise ValueError("risk_by_seed does not align with the frozen model_seeds")
    regret_by_seed = np.empty((STAT.N_MODEL_SEEDS, group_seed.size), dtype=np.float64)
    for i, seed in enumerate(model_seeds):
        tau_i = float(per_seed_tau[str(seed)])
        chosen_i = P.apply_fallback(a0, risk_by_seed[i], tau_i)
        regret_by_seed[i] = P.action_regret(j_true, chosen_i, a_star)

    # per-fallback EXECUTED regrets under each frozen fallback threshold (condition 5 reporting).
    per_baseline_tau = policy["per_baseline_tau"]
    regret_by_fallback = np.empty((STAT.N_FALLBACKS, group_seed.size), dtype=np.float64)
    for bi, name in enumerate(P.FALLBACK_BASELINES):
        chosen_fb = P.apply_fallback(a0, baseline_risks[name], float(per_baseline_tau[name]))
        regret_by_fallback[bi] = P.action_regret(j_true, chosen_fb, a_star)

    t_star = np.asarray(d["t_star"], dtype=np.int64)
    distance = np.asarray(d["interaction_min_distance"], dtype=np.float64)

    return {
        "regret_d2r": np.asarray(d["regret_d2r"], dtype=np.float64),
        "regret_idle": np.asarray(d["regret_idle"], dtype=np.float64),
        "regret_base": np.asarray(d["regret_a0"], dtype=np.float64),
        "regret_strongest_fallback": np.asarray(d["regret_strongest_fallback"], dtype=np.float64),
        "regret_by_seed": regret_by_seed,
        "regret_by_fallback": regret_by_fallback,
        "chosen_d2r": np.asarray(d["chosen_d2r"], dtype=np.int64),
        "a0": a0,
        "risk_d2r": np.asarray(d["risk_direct_ensemble"], dtype=np.float64),
        "tau_d2r": float(policy["tau_d2r"]),
        "j_hat": np.asarray(d["j_hat"], dtype=np.float64),
        "j_true": j_true,
        "a_star": a_star,
        "reactive": _validation_reactive(group_seed),
        "tstar_layer": (t_star >= STAT.TSTAR_SPLIT).astype(np.int64),
        "interaction_layer": (distance > interaction_threshold).astype(np.int64),
        "strongest_fallback": str(policy["strongest_standard_fallback"]),
        "group_seed": group_seed,
        "n_groups": int(group_seed.size),
    }


# --- covariance from the validation paired bootstrap draws ------------------------------------------


def _monte_carlo_seed(bootstrap_seed: int, data_sha256: str) -> int:
    """Deterministic MC seed derived from the frozen power entropy + data hash (no new constant)."""
    material = int(data_sha256[:8], 16) % (2**31)
    return int(np.random.SeedSequence([int(bootstrap_seed), material]).generate_state(1)[0])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def simulate_power(theta_hat: np.ndarray, covariance: np.ndarray, *,
                   candidate_ns: Sequence[int], n_reference: int, monte_carlo: int,
                   seed: int, shrink_scenarios: Sequence[float],
                   selection_scenario: float, target: float) -> dict[str, Any]:
    """Smallest n whose SELECTION-scenario (75%) joint power reaches the target.

    Covariance scales by ``n_reference / n``; effects shrink to ``scenario * theta`` with signs
    preserved (never flipped). The gate is the FULL plan-7 statistical AND-chain on each joint
    draw (never a product of marginal powers). Only the 75% scenario selects; 100%/50% are
    sensitivity.
    """
    theta_hat = np.asarray(theta_hat, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if theta_hat.shape != (STAT.N_POWER_QUANTITIES,) \
            or covariance.shape != (STAT.N_POWER_QUANTITIES, STAT.N_POWER_QUANTITIES):
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
            evaluation = STAT.evaluate_screen_draws(draws, se_n)
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
        "selection_scenario": float(selection_scenario),
        "target": float(target),
        "max_candidate_n": max_n,
        "sensitivity_at_max_n": {
            scenario: per_n[max_n]["scenarios"][scenario]["joint_power"]
            for scenario in (f"{float(s):.2f}" for s in shrink_scenarios)
        },
    }


def estimate_screen_power(quantities: Mapping[str, Any], *,
                          bootstrap_seed: int = POWER_BOOTSTRAP_SEED,
                          bootstrap_resamples: int = POWER_BOOTSTRAP_RESAMPLES,
                          monte_carlo: int = POWER_MONTE_CARLO,
                          candidate_ns: Sequence[int] = CANDIDATE_NS,
                          shrink_scenarios: Sequence[float] = SHRINK_SCENARIOS,
                          selection_scenario: float = SELECTION_SCENARIO,
                          target: float = POWER_TARGET,
                          progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Joint plan-7 power at each candidate n from the validation paired bootstrap (plan 5)."""
    n_reference = int(np.asarray(quantities["regret_d2r"]).size)
    if progress:
        progress(f"recomputing the {bootstrap_resamples}-draw whole-group paired bootstrap "
                 f"(entropy {bootstrap_seed}) on validation (n_reference={n_reference}) for the "
                 f"covariance")
    bootstrap = STAT.paired_bootstrap(
        lambda idx: STAT.compute_screen_quantity_vector(idx, quantities), n_reference,
        seed=int(bootstrap_seed), n_resamples=int(bootstrap_resamples),
        confidence=STAT.CONFIDENCE)
    subset = list(STAT.POWER_SLOTS)
    theta_hat = np.asarray(bootstrap.point, dtype=np.float64)[subset]
    draws = np.asarray(bootstrap.draws, dtype=np.float64)[:, subset]
    if not np.isfinite(theta_hat).all() or not np.isfinite(draws).all():
        raise ValueError(
            "non-finite identity/draw in the power subset (a fail-closed denominator went "
            "degenerate on validation); the power pre-check cannot run")
    covariance = symmetrised_psd_covariance(draws)
    if progress:
        progress("covariance ready; simulating the plan-7 AND-chain over joint Monte-Carlo draws")
    data_sha = _sha256_file(PHASE0_DIR / "diagnostic.npz")
    result = simulate_power(
        theta_hat, covariance, candidate_ns=candidate_ns, n_reference=n_reference,
        monte_carlo=monte_carlo, seed=_monte_carlo_seed(bootstrap_seed, data_sha),
        shrink_scenarios=shrink_scenarios, selection_scenario=selection_scenario, target=target)
    theta_named = {name: float(theta_hat[i]) for i, name in enumerate(STAT.POWER_QUANTITY_NAMES)}
    se_named = {name: float(np.sqrt(covariance[i, i]))
                for i, name in enumerate(STAT.POWER_QUANTITY_NAMES)}
    return {
        **result,
        "theta_hat": theta_named,
        "se_reference": se_named,
        "n_reference": n_reference,
        "monte_carlo": int(monte_carlo),
        "bootstrap": {
            "seed": int(bootstrap_seed),
            "n_resamples": int(bootstrap_resamples),
            "confidence": float(STAT.CONFIDENCE),
            "design": ("covariance taken DIRECTLY from the validation whole-group paired "
                       "bootstrap draws (frozen entropy); an 80% SE is never derived from the "
                       "CI width"),
        },
    }


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


def run_power(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Execute the plan-5 power pre-check and write the create-once report. No collection.

    Create-once: a rerun with the report present reloads it. A selected n continues the plan;
    ``D2R_SCREEN_UNDERPOWERED`` is a preregistered terminal state that forbids any collection.
    """
    POWER_DIR.mkdir(parents=True, exist_ok=True)
    report_path = POWER_DIR / REPORT_NAME
    if report_path.exists():
        if progress:
            progress("screen_power_report.json already exists (create-once); reloading")
        return _load_json(report_path)

    t0 = time.perf_counter()
    quantities = build_validation_screen_quantities()
    if progress:
        progress(f"validation plan-7 quantities assembled: {quantities['n_groups']} groups "
                 f"(reactive/strata/per-seed/per-fallback derived; elapsed={time.perf_counter()-t0:.1f}s)")
    result = estimate_screen_power(quantities, progress=progress)
    status = STATUS_UNDERPOWERED if result["underpowered"] else STATUS_POWER_PASS

    payload = {
        "plan_id": P.PLAN_ID,
        "scope": "ADAPTIVE_DEVELOPMENT_ONLY",
        "status": status,
        "terminal": bool(result["underpowered"]),
        "selected_n": result["selected_n"],
        "underpowered": bool(result["underpowered"]),
        "target": result["target"],
        "selection_scenario": result["selection_scenario"],
        "monte_carlo": result["monte_carlo"],
        "n_reference": result["n_reference"],
        "candidate_ns": [int(n) for n in result["candidate_ns"]] if "candidate_ns" in result
                        else [int(n) for n in CANDIDATE_NS],
        "shrink_scenarios": [float(s) for s in SHRINK_SCENARIOS],
        "per_n": {str(n): entry for n, entry in result["per_n"].items()},
        "sensitivity_at_max_n": result["sensitivity_at_max_n"],
        "theta_hat": result["theta_hat"],
        "se_reference": result["se_reference"],
        "bootstrap": result["bootstrap"],
        "conditioned_on": ("a structurally valid screen: plan-7 condition 1 (Gate Q isolation, "
                           "n/n completeness, zero ID overlap, no outcome rejection) and the "
                           "denominator/projection/mask validity are NOT statistical; their "
                           "failure is D2R_INVALID_ANALYSIS, not a power shortfall"),
        "semantics": ("conservative development power pre-check from the adaptively-used "
                      "validation role; never a promise that the screen passes. A selected n "
                      "permits the plan to continue to manifest/lock/collection; "
                      "D2R_SCREEN_UNDERPOWERED is a terminal state that forbids collection."),
        "bindings": {
            "phase0_diagnostic_sha256": _sha256_file(PHASE0_DIR / "diagnostic.npz"),
            "phase0_final_policy_sha256": _sha256_file(PHASE0_DIR / "final_policy.json"),
            "power_code_sha256": _sha256_file(Path(__file__).resolve()),
            "statistics_code_sha256": _sha256_file(
                ROOT / "code/ac_twin/stage4d_d2r_statistics.py"),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(report_path, payload)
    if progress:
        progress(f"plan-5 power COMPLETE status={status} selected_n={result['selected_n']} "
                 f"total_elapsed={time.perf_counter()-t0:.1f}s")
    payload["report_path"] = str(report_path)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    last = {"at": 0.0}

    def progress(line: str) -> None:
        now = time.perf_counter()
        if now - last["at"] >= 5.0:
            last["at"] = now
            print(f"[stage4d-d2r-power] {line}", flush=True)

    report = run_power(progress=progress)
    print(f"[stage4d-d2r-power] status={report['status']} selected_n={report['selected_n']} "
          f"underpowered={report['underpowered']}", flush=True)
    return 5 if report["underpowered"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_NS",
    "POWER_BOOTSTRAP_RESAMPLES",
    "POWER_BOOTSTRAP_SEED",
    "POWER_DIR",
    "POWER_MONTE_CARLO",
    "POWER_TARGET",
    "REPORT_NAME",
    "SELECTION_SCENARIO",
    "SHRINK_SCENARIOS",
    "STATUS_POWER_PASS",
    "STATUS_UNDERPOWERED",
    "build_validation_screen_quantities",
    "estimate_screen_power",
    "run_power",
    "simulate_power",
    "symmetrised_psd_covariance",
]
