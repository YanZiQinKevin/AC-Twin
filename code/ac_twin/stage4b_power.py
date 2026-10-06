"""Stage 4B joint power analysis and mechanical confirmation-size selection.

Plan ``AC-TWIN-RFER-STAGE4B-RISK-CONFIRM-2026-09-15-V0.1`` section 5. The ONLY legal
per-episode power input is the Stage 4A development-screen subset (``roles == 2``,
exactly 1,000 episodes) of the frozen ``risk_dataset.npz``, re-scored with the frozen
``COMBINED_RISK_HEAD/MLP`` and ``GEOMETRY_ONLY/Ridge`` models and the direct
``UNCERTAINTY_ONLY`` aggregation. risk-train / risk-validation / Stage 2B / any
``roles != 2`` row never participates.

Gate-0 (plan section 5.1): the recomputed shared point estimates must match the frozen
``stage4a_dev_report.json`` to ``<=1e-8`` BEFORE any Monte-Carlo draw; otherwise the
power run stops.

Joint statistics (17-vector, episode-level paired bootstrap, plan section 5.3.1):
    0  I_AURC  vs UNCERTAINTY_ONLY
    1  I_C80   vs UNCERTAINTY_ONLY
    2  I_AURC  vs GEOMETRY_ONLY
    3  I_C80   vs GEOMETRY_ONLY
    4  PR-AUC - prevalence (combined finalist)
    5..9   per-pair I_AURC vs UNCERTAINTY_ONLY (5 model-seed pairs)
    10..14 per-action I_AURC vs UNCERTAINTY_ONLY (cf_action 0..4)
    15..16 per-duration I_AURC vs UNCERTAINTY_ONLY (cf_duration 1, 3)
The SAME bootstrap index drives all 17 quantities and repeated-episode multiplicity is
preserved (integer resampling with replacement).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ac_twin import stage4_protocol as s4p
from ac_twin import stage4_risk as s4r
from ac_twin import stage4_statistics as s4s
from ac_twin import stage4b_protocol as s4bp
from ac_twin.stage1_data import sha256_file

# Quantity-vector indices (frozen layout above).
Q_I_AURC_UNC = 0
Q_I_C80_UNC = 1
Q_I_AURC_GEO = 2
Q_I_C80_GEO = 3
Q_PR = 4
Q_PAIR_START = 5            # 5..9
Q_ACTION_START = 10         # 10..14
Q_DURATION_START = 15       # 15..16
N_PAIRS = 5
N_QUANTITIES = 17
PAIR_INDEX = list(range(Q_PAIR_START, Q_PAIR_START + N_PAIRS))
ACTION_INDEX = {a: Q_ACTION_START + i for i, a in enumerate(s4bp.CF_ACTIONS)}
DURATION_INDEX = {d: Q_DURATION_START + i for i, d in enumerate(s4bp.CF_DURATIONS)}

Z_95 = 1.96  # plan section 5.3.4: estimate - 1.96*SE_n lower bound
_SELF_CHECK_TOL = 1e-8


# --- screen loading + frozen re-scoring ----------------------------------------


def load_screen_inputs(
    dataset_path: str | Path,
    finalist_model_path: str | Path,
    geometry_model_path: str | Path,
) -> dict[str, Any]:
    """Load the frozen risk_dataset, select ``roles == 2`` and re-score.

    Returns per-episode combined/geometry/uncertainty scores plus the labels and
    stratification axes needed for the joint statistics. Read-only; fits nothing.
    """
    dataset_path = Path(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    roles = np.asarray(data["roles"], dtype=np.int64)
    if int((roles == s4p.ROLE_DEVELOPMENT_SCREEN).sum()) != 1000:
        raise ValueError("development-screen subset must contain exactly 1,000 episodes")
    screen = np.flatnonzero(roles == s4p.ROLE_DEVELOPMENT_SCREEN)

    features = {
        key[len("feature__"):]: np.asarray(value, dtype=np.float64)
        for key, value in data.items() if key.startswith("feature__")
    }
    sub = {name: value[screen] for name, value in features.items()}

    finalist = s4r.FittedRiskModel.load(finalist_model_path)
    geometry = s4r.FittedRiskModel.load(geometry_model_path)
    comb = finalist.predict(s4r.design_matrix(sub, s4r.CANDIDATE_FEATURES[s4r.CANDIDATE_COMBINED]))
    geo = geometry.predict(s4r.design_matrix(sub, s4r.CANDIDATE_FEATURES[s4r.BASELINE_GEOMETRY]))
    unc = s4r.direct_risk_score(s4r.CANDIDATE_UNCERTAINTY, sub)

    risk_target = np.asarray(data["risk_target"], dtype=np.float64)[screen]
    primary_by_pair = np.asarray(data["primary_error_by_pair"], dtype=np.float64)[:, screen]
    if primary_by_pair.shape != (N_PAIRS, screen.size):
        raise ValueError("primary_error_by_pair must align as (5, n_screen)")
    return {
        "comb_score": comb,
        "geo_score": geo,
        "unc_score": unc,
        "risk_target": risk_target,
        "primary_error_by_pair": primary_by_pair,
        "cf_action": np.asarray(data["cf_action"], dtype=np.int64)[screen],
        "cf_duration": np.asarray(data["cf_duration"], dtype=np.int64)[screen],
        "episode_seed": np.asarray(data["episode_seed"], dtype=np.int64)[screen],
        "top_error_threshold": float(np.asarray(data["top_error_threshold"], dtype=np.float64)),
        "dataset_sha256": sha256_file(dataset_path),
        "finalist_model_sha256": sha256_file(finalist_model_path),
        "geometry_model_sha256": sha256_file(geometry_model_path),
    }


# --- joint statistic vector ------------------------------------------------------


def _aurc_from_order(order: np.ndarray, error: np.ndarray) -> float:
    sorted_error = error[order]
    cumulative = np.cumsum(sorted_error)
    return float((cumulative / np.arange(1, error.size + 1)).mean())


def _ade80_from_order(order: np.ndarray, error: np.ndarray) -> float:
    k = s4s.coverage_index(0.80, error.size)
    return float(error[order][:k].mean())


def _improvement(standard: float, candidate: float) -> float:
    return float((standard - candidate) / standard) if standard > 0 else float("nan")


def compute_quantity_vector(
    index: np.ndarray,
    comb_score: np.ndarray,
    geo_score: np.ndarray,
    unc_score: np.ndarray,
    risk_target: np.ndarray,
    primary_error_by_pair: np.ndarray,
    cf_action: np.ndarray,
    cf_duration: np.ndarray,
    top_error_threshold: float,
) -> np.ndarray:
    """Evaluate all 17 joint quantities on one resampled episode index set.

    ``index`` is an integer resample (with replacement) into the screen episodes, so
    repeated-episode multiplicity is preserved. Score sort orders are reused across
    the main target and the five per-pair errors to keep the bootstrap tractable.
    """
    comb = comb_score[index]
    geo = geo_score[index]
    unc = unc_score[index]
    err = risk_target[index]
    order_comb = np.argsort(comb, kind="stable")
    order_unc = np.argsort(unc, kind="stable")
    order_geo = np.argsort(geo, kind="stable")

    a_unc = _aurc_from_order(order_unc, err)
    a_comb = _aurc_from_order(order_comb, err)
    a_geo = _aurc_from_order(order_geo, err)
    c_unc = _ade80_from_order(order_unc, err)
    c_comb = _ade80_from_order(order_comb, err)
    c_geo = _ade80_from_order(order_geo, err)

    q = np.full(N_QUANTITIES, np.nan, dtype=np.float64)
    q[Q_I_AURC_UNC] = _improvement(a_unc, a_comb)
    q[Q_I_C80_UNC] = _improvement(c_unc, c_comb)
    q[Q_I_AURC_GEO] = _improvement(a_geo, a_comb)
    q[Q_I_C80_GEO] = _improvement(c_geo, c_comb)
    q[Q_PR] = s4s.pr_auc_minus_prevalence(comb, err, top_error_threshold)[
        "pr_auc_minus_prevalence"
    ]

    for p in range(N_PAIRS):
        err_p = primary_error_by_pair[p][index]
        a_unc_p = _aurc_from_order(order_unc, err_p)
        a_comb_p = _aurc_from_order(order_comb, err_p)
        q[Q_PAIR_START + p] = _improvement(a_unc_p, a_comb_p)

    ca = cf_action[index]
    cd = cf_duration[index]
    for action, slot in ACTION_INDEX.items():
        mask = ca == action
        if mask.sum() < 2:
            q[slot] = float("nan")
            continue
        e = err[mask]
        a_s = _aurc_from_order(np.argsort(unc[mask], kind="stable"), e)
        a_c = _aurc_from_order(np.argsort(comb[mask], kind="stable"), e)
        q[slot] = _improvement(a_s, a_c)
    for duration, slot in DURATION_INDEX.items():
        mask = cd == duration
        if mask.sum() < 2:
            q[slot] = float("nan")
            continue
        e = err[mask]
        a_s = _aurc_from_order(np.argsort(unc[mask], kind="stable"), e)
        a_c = _aurc_from_order(np.argsort(comb[mask], kind="stable"), e)
        q[slot] = _improvement(a_s, a_c)
    return q


# --- Gate-0 self-check ------------------------------------------------------------


def self_check_vs_report(quantities: np.ndarray, dev_report: Mapping[str, Any]) -> dict[str, Any]:
    """Verify recomputed shared point estimates match the frozen report to <=1e-8.

    ``quantities`` is the 17-vector evaluated on the FULL screen (identity index).
    Raises before any Monte-Carlo draw if a shared field drifts past tolerance.
    """
    boot = dev_report["improvement_bootstrap"]
    pairs = np.asarray(dev_report["per_pair_i_aurc"], dtype=np.float64)
    strata = dev_report["stratified"]["strata"]
    comparisons = {
        "i_aurc": (quantities[Q_I_AURC_UNC], float(boot["i_aurc"])),
        "i_c80": (quantities[Q_I_C80_UNC], float(boot["i_c80"])),
        "pr_auc_minus_prevalence": (
            quantities[Q_PR],
            float(dev_report["finalist_report"]["pr_auc_minus_prevalence"]),
        ),
    }
    for p in range(N_PAIRS):
        comparisons[f"pair_{p}_i_aurc"] = (quantities[Q_PAIR_START + p], float(pairs[p]))
    for action, slot in ACTION_INDEX.items():
        comparisons[f"action_{action}_i_aurc"] = (
            quantities[slot], float(strata[f"action_{action}"]["i_aurc"]))
    for duration, slot in DURATION_INDEX.items():
        comparisons[f"duration_{duration}_i_aurc"] = (
            quantities[slot], float(strata[f"duration_{duration}"]["i_aurc"]))

    deltas = {name: abs(float(got) - float(exp)) for name, (got, exp) in comparisons.items()}
    worst = max(deltas.values())
    if worst > _SELF_CHECK_TOL:
        worst_name = max(deltas, key=deltas.get)
        raise ValueError(
            f"Stage 4B power Gate-0 self-check failed: {worst_name} drifted "
            f"{worst:.3e} (> {_SELF_CHECK_TOL}) vs the frozen Stage 4A report"
        )
    return {"max_abs_delta": float(worst), "n_fields": len(comparisons), "passed": True}


# --- joint bootstrap --------------------------------------------------------------


def joint_bootstrap(
    inputs: Mapping[str, Any],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """10,000 episode-level paired bootstrap -> theta_hat (full-screen) + Sigma.

    Sigma is the covariance of the bootstrap quantity distribution, symmetrised with
    negative eigenvalues below ``1e-12`` clipped to ``1e-12`` (plan section 5.3.2).
    No unfavourable quantity or covariance is dropped.
    """
    generator = np.random.default_rng(int(seed))
    episodes = inputs["risk_target"].size
    draws = np.empty((int(n_resamples), N_QUANTITIES), dtype=np.float64)
    identity = np.arange(episodes, dtype=np.int64)
    theta_hat = compute_quantity_vector(
        identity,
        inputs["comb_score"], inputs["geo_score"], inputs["unc_score"],
        inputs["risk_target"], inputs["primary_error_by_pair"],
        inputs["cf_action"], inputs["cf_duration"], inputs["top_error_threshold"],
    )
    for draw in range(int(n_resamples)):
        index = generator.integers(0, episodes, size=episodes)
        draws[draw] = compute_quantity_vector(
            index,
            inputs["comb_score"], inputs["geo_score"], inputs["unc_score"],
            inputs["risk_target"], inputs["primary_error_by_pair"],
            inputs["cf_action"], inputs["cf_duration"], inputs["top_error_threshold"],
        )
    if not np.isfinite(draws).all():
        raise ValueError("non-finite bootstrap quantity encountered")
    covariance = np.cov(draws, rowvar=False)
    covariance = 0.5 * (covariance + covariance.T)  # symmetrise
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    clipped = np.clip(eigenvalues, 1e-12, None)
    covariance = (eigenvectors * clipped) @ eigenvectors.T
    covariance = 0.5 * (covariance + covariance.T)
    return {
        "theta_hat": theta_hat,
        "covariance": covariance,
        "bootstrap_mean": draws.mean(axis=0),
        "n_resamples": int(n_resamples),
        "n_episodes": int(episodes),
        "min_eigenvalue": float(eigenvalues.min()),
    }


# --- Gate RISK_confirm evaluation on a joint draw ---------------------------------


def evaluate_draws(
    draws: np.ndarray, standard_errors: np.ndarray
) -> dict[str, np.ndarray]:
    """Vectorised AND-chain (plan section 8.2) over joint Monte-Carlo draws.

    ``draws`` is ``(replicates, 17)``; ``standard_errors`` is the per-quantity
    ``sqrt(diag(Sigma * 1000/n))``. CI-based conditions use ``estimate - 1.96*SE_n``.
    Returns a boolean pass vector plus each condition's pass vector.
    """
    draws = np.asarray(draws, dtype=np.float64)
    se = np.asarray(standard_errors, dtype=np.float64)

    def ci_lower(slot):
        return draws[:, slot] - Z_95 * se[slot]

    conditions = {
        "i_aurc_vs_uncertainty": (draws[:, Q_I_AURC_UNC] >= s4bp.GATE_I_AURC_MIN)
        & (ci_lower(Q_I_AURC_UNC) > 0.0),
        "i_c80_vs_uncertainty": (draws[:, Q_I_C80_UNC] >= s4bp.GATE_I_C80_MIN)
        & (ci_lower(Q_I_C80_UNC) > 0.0),
        "i_aurc_vs_geometry": (draws[:, Q_I_AURC_GEO] > 0.0)
        & (ci_lower(Q_I_AURC_GEO) > 0.0),
        "i_c80_vs_geometry": (draws[:, Q_I_C80_GEO] > 0.0)
        & (ci_lower(Q_I_C80_GEO) > 0.0),
        "pr_auc_minus_prevalence": (draws[:, Q_PR] > 0.0) & (ci_lower(Q_PR) > 0.0),
        "at_least_4_of_5_pairs_positive": (draws[:, PAIR_INDEX] > 0.0).sum(axis=1)
        >= s4bp.GATE_MIN_POSITIVE_PAIRS,
        "no_strata_below_floor": (
            draws[:, list(ACTION_INDEX.values()) + list(DURATION_INDEX.values())]
            >= s4bp.GATE_STRATA_I_AURC_FLOOR
        ).all(axis=1),
    }
    passed = np.logical_and.reduce(list(conditions.values()))
    return {"passed": passed, "conditions": conditions}


# --- mechanical sample-size selection --------------------------------------------


def select_confirmation_size(
    theta_hat: np.ndarray,
    covariance: np.ndarray,
    *,
    candidate_ns: tuple[int, ...],
    monte_carlo: int,
    seed: int,
    n_reference: int,
    shrink_scenarios: tuple[float, ...],
    shrink_main: float,
    target: float,
) -> dict[str, Any]:
    """Pick the smallest n whose 75%-shrink joint power is >= target.

    For each candidate n the covariance scales by ``n_reference / n``; positive
    development effects shrink to ``scenario * theta_hat`` (negatives keep their sign
    under the same positive scaling — they are never flipped favourable). The main
    blocking scenario is 75%; 100% and 50% are sensitivity only.
    """
    theta_hat = np.asarray(theta_hat, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    generator = np.random.default_rng(int(seed))
    max_n = int(max(candidate_ns))

    per_n: dict[int, Any] = {}
    selected_n: int | None = None
    for n in candidate_ns:
        n = int(n)
        cov_n = covariance * (float(n_reference) / n)
        se_n = np.sqrt(np.diag(cov_n))
        scenarios: dict[str, Any] = {}
        for shrink in shrink_scenarios:
            mean = float(shrink) * theta_hat
            draws = generator.multivariate_normal(mean, cov_n, size=int(monte_carlo))
            evaluation = evaluate_draws(draws, se_n)
            scenarios[f"{shrink:.2f}"] = {
                "joint_power": float(evaluation["passed"].mean()),
                "marginal_power": {
                    name: float(vec.mean()) for name, vec in evaluation["conditions"].items()
                },
            }
        main = scenarios[f"{shrink_main:.2f}"]
        binding = min(main["marginal_power"], key=main["marginal_power"].get)
        per_n[n] = {
            "scenarios": scenarios,
            "joint_power_main": main["joint_power"],
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
        "shrink_main": float(shrink_main),
        "target": float(target),
        "max_candidate_n": max_n,
        "sensitivity_at_max_n": {
            scenario: per_n[max_n]["scenarios"][scenario]["joint_power"]
            for scenario in (f"{s:.2f}" for s in shrink_scenarios)
        },
    }


# --- top-level orchestration -------------------------------------------------------


def run_power(
    config: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Run the full create-once power analysis and return the report payload.

    Ordering (plan section 5): load + re-score -> Gate-0 self-check vs the frozen
    report (raise before any MC) -> joint bootstrap covariance -> mechanical
    selection. The caller persists the report create-once.
    """
    checked = s4bp.validate_stage4b_config(config)
    frozen = s4bp.STAGE4A_FROZEN
    inputs = load_screen_inputs(
        project_root / frozen["stage4a_risk_dataset"][0],
        project_root / frozen["finalist_risk_model"][0],
        project_root / frozen["geometry_risk_model"][0],
    )
    identity = np.arange(inputs["risk_target"].size, dtype=np.int64)
    point = compute_quantity_vector(
        identity,
        inputs["comb_score"], inputs["geo_score"], inputs["unc_score"],
        inputs["risk_target"], inputs["primary_error_by_pair"],
        inputs["cf_action"], inputs["cf_duration"], inputs["top_error_threshold"],
    )
    dev_report = json.loads(
        (project_root / "experiments/stage4_reliability/stage4a_v4/reports/stage4a_dev_report.json")
        .read_text(encoding="utf-8")
    )
    gate0 = self_check_vs_report(point, dev_report)

    bootstrap = joint_bootstrap(
        inputs,
        n_resamples=int(checked["power_bootstrap_resamples"]),
        seed=int(checked["bootstrap_entropy_confirm"]) - 1,  # power stream; confirm stream reserved
    )
    selection = select_confirmation_size(
        bootstrap["theta_hat"], bootstrap["covariance"],
        candidate_ns=tuple(int(n) for n in checked["candidate_ns"]),
        monte_carlo=int(checked["power_monte_carlo"]),
        seed=int(checked["power_entropy"]),
        n_reference=int(inputs["risk_target"].size),
        shrink_scenarios=tuple(float(s) for s in checked.get(
            "power_shrink_scenarios", s4bp.POWER_SHRINK_SCENARIOS)),
        shrink_main=float(checked["power_shrink_main"]),
        target=float(checked["power_target"]),
    )

    report = {
        "plan_id": s4bp.PLAN_ID,
        "purpose": s4bp.PURPOSE,
        "screen_role_integer": s4p.ROLE_DEVELOPMENT_SCREEN,
        "n_screen_episodes": int(inputs["risk_target"].size),
        "risk_dataset_sha256": inputs["dataset_sha256"],
        "finalist_model_sha256": inputs["finalist_model_sha256"],
        "geometry_model_sha256": inputs["geometry_model_sha256"],
        "stage4a_dev_report_sha256": sha256_file(
            project_root / "experiments/stage4_reliability/stage4a_v4/reports/stage4a_dev_report.json"
        ),
        "power_algorithm_sha256": sha256_file(Path(__file__)),
        "gate0_self_check": gate0,
        "theta_hat": bootstrap["theta_hat"].tolist(),
        "covariance_min_eigenvalue": bootstrap["min_eigenvalue"],
        "power_bootstrap_resamples": int(checked["power_bootstrap_resamples"]),
        "power_monte_carlo": int(checked["power_monte_carlo"]),
        "power_entropy": int(checked["power_entropy"]),
        "candidate_ns": [int(n) for n in checked["candidate_ns"]],
        "selection": selection,
        "selected_n": selection["selected_n"],
        "status": (
            s4bp.STATUS_UNDERPOWERED if selection["underpowered"] else "POWERED"
        ),
    }
    return report


def report_sha256(report: Mapping[str, Any]) -> str:
    """Stable content fingerprint of a power report dict."""
    return hashlib.sha256(
        json.dumps(report, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


__all__ = [
    "ACTION_INDEX",
    "DURATION_INDEX",
    "N_QUANTITIES",
    "PAIR_INDEX",
    "Q_ACTION_START",
    "Q_DURATION_START",
    "Q_I_AURC_GEO",
    "Q_I_AURC_UNC",
    "Q_I_C80_GEO",
    "Q_I_C80_UNC",
    "Q_PAIR_START",
    "Q_PR",
    "Z_95",
    "compute_quantity_vector",
    "evaluate_draws",
    "joint_bootstrap",
    "load_screen_inputs",
    "report_sha256",
    "run_power",
    "select_confirmation_size",
    "self_check_vs_report",
]
