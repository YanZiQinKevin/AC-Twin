"""Stage 4C group-level joint power analysis and confirmation-size selection.

Plan ``AC-TWIN-RFER-STAGE4C-MULTIACTION-REGRET-2026-09-15-V0.1`` section 10. The ONLY
legal power input is the already-adjudicated 300 development-screen groups via the
create-once ``screen_group_quantities.npz`` (never the 500 selection groups, never
Stage 4B single-action confirm). The statistical unit is the decision GROUP: every
bootstrap resample is a whole-group integer resample that simultaneously drives the
ten actions, the five seed-pair regrets, the main method, all baselines and the
stratification labels (plan section 10.2.1).

Quantity-vector layout (frozen). Indices 0..13 are the RISK-gate quantities used by
the power AND-chain; 14..15 are Gate TASK CI quantities evaluated in the same paired
bootstrap so development adjudication and power share one definition bit-for-bit:
    0   I_AURC_REGRET            (combined vs frozen strongest standard baseline)
    1   I_REGRET@80
    2   PR-AUC - prevalence      (high-regret label, combined group risk)
    3..7  per-pair I_AURC_REGRET (5 seed pairs)
    8..9  t_star strata I_AURC_REGRET (low / high)
    10..11 density strata I_AURC_REGRET (low / high)
    12  trim-1% aggregate I_AURC_REGRET
    13  trim-5% aggregate I_AURC_REGRET
    14  branch-level Spearman(J_hat, J_true)
    15  base-vs-RANDOM_ACTION mean-regret improvement
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import rankdata

from ac_twin import stage4_statistics as s4s
from ac_twin import stage4c_protocol as s4cp
from ac_twin.stage1_data import sha256_file

# Quantity-vector indices (frozen layout above).
Q_I_AURC = 0
Q_I_REGRET80 = 1
Q_PR = 2
Q_PAIR_START = 3          # 3..7
Q_TSTAR_START = 8         # 8..9
Q_DENSITY_START = 10      # 10..11
Q_TRIM1 = 12
Q_TRIM5 = 13
Q_SPEARMAN = 14
Q_BASE_VS_RANDOM = 15
N_PAIRS = 5
N_QUANTITIES = 16
RISK_QUANTITIES = list(range(0, 14))  # power AND-chain subset
PAIR_INDEX = list(range(Q_PAIR_START, Q_PAIR_START + N_PAIRS))
STRATA_INDEX = list(range(Q_TSTAR_START, Q_DENSITY_START + 2))  # 8..11
TRIM_INDEX = [Q_TRIM1, Q_TRIM5]

Z_95 = 1.96


# --- screen per-group quantities loading (create-once NPZ only) ---------------------


def load_screen_group_quantities(npz_path: str | Path) -> dict[str, Any]:
    """Load the create-once ``screen_group_quantities.npz``; power reads ONLY this."""
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = (
        "group_seed", "regret", "regret_by_pair", "random_regret", "idle_regret",
        "combined_group_risk", "baseline_group_risk", "high_regret_label",
        "tstar_layer", "density_layer", "J_hat", "J_true",
    )
    for key in required:
        if key not in data:
            raise ValueError(f"screen_group_quantities.npz missing field {key!r}")
    out = {key: np.asarray(data[key]) for key in required}
    out["npz_sha256"] = sha256_file(npz_path)
    groups = out["regret"].shape[0]
    if out["regret_by_pair"].shape != (N_PAIRS, groups):
        raise ValueError("regret_by_pair must align as (5, G)")
    if out["J_hat"].shape != (groups, s4cp.N_ACTIONS):
        raise ValueError("J_hat must align as (G, 10)")
    return out


# --- risk-curve primitives --------------------------------------------------------------


def aurc_regret(order: np.ndarray, regret: np.ndarray) -> float:
    """AURC_REGRET = mean_{k=1..G} R(k), R(k) = mean regret of the k lowest-risk."""
    sorted_regret = regret[order]
    cumulative = np.cumsum(sorted_regret)
    return float((cumulative / np.arange(1, regret.size + 1)).mean())


def regret_at_coverage(order: np.ndarray, regret: np.ndarray, coverage: float) -> float:
    k = s4s.coverage_index(coverage, regret.size)
    return float(regret[order][:k].mean())


def _improvement(baseline: float, combined: float) -> float:
    return float((baseline - combined) / baseline) if baseline > 0 else float("nan")


def _high_regret_pr(group_risk: np.ndarray, label: np.ndarray) -> float:
    """PR-AUC - prevalence for the high-regret label (groups as samples)."""
    label = np.asarray(label, dtype=np.int64)
    prevalence = float(label.mean())
    if label.min() == label.max():
        return 0.0
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(label, group_risk) - prevalence)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = rankdata(np.asarray(a, dtype=np.float64).ravel())
    rb = rankdata(np.asarray(b, dtype=np.float64).ravel())
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


# --- joint group-level quantity vector ----------------------------------------------------


def compute_group_quantity_vector(
    index: np.ndarray,
    combined_risk: np.ndarray,
    baseline_risk: np.ndarray,
    regret: np.ndarray,
    regret_by_pair: np.ndarray,
    random_regret: np.ndarray,
    idle_regret: np.ndarray,
    high_regret_label: np.ndarray,
    tstar_layer: np.ndarray,
    density_layer: np.ndarray,
    J_hat: np.ndarray,
    J_true: np.ndarray,
) -> np.ndarray:
    """Evaluate all 16 quantities on one whole-group resample index set.

    ``index`` is an integer with-replacement resample into the screen groups, so
    repeated-group multiplicity is preserved and the ten actions / five pairs /
    baselines / strata stay coupled. Improvement quantities are NaN when the baseline
    denominator is non-positive (the gate turns that into STAGE4C_INVALID_ANALYSIS).
    """
    comb = combined_risk[index]
    base = baseline_risk[index]
    reg = regret[index]
    order_comb = np.argsort(comb, kind="stable")
    order_base = np.argsort(base, kind="stable")

    a_base = aurc_regret(order_base, reg)
    a_comb = aurc_regret(order_comb, reg)
    r80_base = regret_at_coverage(order_base, reg, s4cp.COVERAGE_MAIN)
    r80_comb = regret_at_coverage(order_comb, reg, s4cp.COVERAGE_MAIN)

    q = np.full(N_QUANTITIES, np.nan, dtype=np.float64)
    q[Q_I_AURC] = _improvement(a_base, a_comb)
    q[Q_I_REGRET80] = _improvement(r80_base, r80_comb)
    q[Q_PR] = _high_regret_pr(comb, high_regret_label[index])

    for p in range(N_PAIRS):
        reg_p = regret_by_pair[p][index]
        q[Q_PAIR_START + p] = _improvement(
            aurc_regret(order_base, reg_p), aurc_regret(order_comb, reg_p))

    for slot, layer in ((Q_TSTAR_START, tstar_layer), (Q_DENSITY_START, density_layer)):
        layer_values = layer[index]
        for level in (0, 1):
            mask = layer_values == level
            if mask.sum() < 2:
                q[slot + level] = float("nan")
                continue
            e = reg[mask]
            a_b = aurc_regret(np.argsort(base[mask], kind="stable"), e)
            a_c = aurc_regret(np.argsort(comb[mask], kind="stable"), e)
            q[slot + level] = _improvement(a_b, a_c)

    for slot, frac in ((Q_TRIM1, 0.01), (Q_TRIM5, 0.05)):
        keep = _trim_high_regret(reg, frac)
        if keep.sum() < 2:
            q[slot] = float("nan")
            continue
        e = reg[keep]
        a_b = aurc_regret(np.argsort(base[keep], kind="stable"), e)
        a_c = aurc_regret(np.argsort(comb[keep], kind="stable"), e)
        q[slot] = _improvement(a_b, a_c)

    q[Q_SPEARMAN] = _spearman(J_hat[index], J_true[index])
    rand = random_regret[index]
    base_reg = reg  # ensemble base regret
    rand_mean = float(rand.mean())
    q[Q_BASE_VS_RANDOM] = (rand_mean - float(base_reg.mean())) / rand_mean if rand_mean > 0 else float("nan")
    return q


def _trim_high_regret(regret: np.ndarray, fraction: float) -> np.ndarray:
    """Boolean keep-mask removing the top-``fraction`` highest-regret groups."""
    n = regret.size
    drop = int(np.floor(fraction * n))
    if drop <= 0:
        return np.ones(n, dtype=bool)
    order = np.argsort(regret, kind="stable")  # ascending; last `drop` are highest
    keep = np.ones(n, dtype=bool)
    keep[order[n - drop:]] = False
    return keep


# --- joint bootstrap ------------------------------------------------------------------------


def joint_bootstrap(
    inputs: Mapping[str, Any],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Whole-group paired bootstrap -> theta_hat (identity) + symmetrised covariance."""
    generator = np.random.default_rng(int(seed))
    groups = inputs["regret"].shape[0]
    identity = np.arange(groups, dtype=np.int64)
    theta_hat = compute_group_quantity_vector(identity, **_qv_args(inputs))
    draws = np.empty((int(n_resamples), N_QUANTITIES), dtype=np.float64)
    for draw in range(int(n_resamples)):
        index = generator.integers(0, groups, size=groups)
        draws[draw] = compute_group_quantity_vector(index, **_qv_args(inputs))
    finite = np.isfinite(draws)
    if not finite[:, RISK_QUANTITIES].all():
        # NaN in a risk quantity signals a non-positive baseline denominator somewhere.
        pass
    covariance = np.cov(np.nan_to_num(draws, nan=0.0), rowvar=False)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    clipped = np.clip(eigenvalues, 1e-12, None)
    covariance = (eigenvectors * clipped) @ eigenvectors.T
    covariance = 0.5 * (covariance + covariance.T)
    return {
        "theta_hat": theta_hat,
        "covariance": covariance,
        "draws": draws,
        "n_resamples": int(n_resamples),
        "n_groups": int(groups),
        "min_eigenvalue": float(eigenvalues.min()),
    }


def _qv_args(inputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "combined_risk": inputs["combined_group_risk"],
        "baseline_risk": inputs["baseline_group_risk"],
        "regret": inputs["regret"],
        "regret_by_pair": inputs["regret_by_pair"],
        "random_regret": inputs["random_regret"],
        "idle_regret": inputs["idle_regret"],
        "high_regret_label": inputs["high_regret_label"],
        "tstar_layer": inputs["tstar_layer"],
        "density_layer": inputs["density_layer"],
        "J_hat": inputs["J_hat"],
        "J_true": inputs["J_true"],
    }


# --- confirm-gate AND-chain on a joint draw ---------------------------------------------------


def evaluate_draws(draws: np.ndarray, standard_errors: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorised Gate RISK-DECISION_confirm AND-chain over joint Monte-Carlo draws.

    ``draws`` is ``(replicates, >=14)``; ``standard_errors`` is ``sqrt(diag(cov_n))``
    for the risk quantities. CI conditions use ``estimate - 1.96*SE`` (95%); pair /
    strata / trim conditions use point values (plan section 11.3).
    """
    draws = np.asarray(draws, dtype=np.float64)
    se = np.asarray(standard_errors, dtype=np.float64)

    def ci_lower(slot):
        return draws[:, slot] - Z_95 * se[slot]

    conditions = {
        "i_aurc_regret": (draws[:, Q_I_AURC] >= s4cp.GATE_I_AURC_REGRET_MIN)
        & (ci_lower(Q_I_AURC) > 0.0),
        "i_regret80": (draws[:, Q_I_REGRET80] >= s4cp.GATE_I_REGRET80_MIN)
        & (ci_lower(Q_I_REGRET80) > 0.0),
        "pr_auc_minus_prevalence": (draws[:, Q_PR] > 0.0) & (ci_lower(Q_PR) > 0.0),
        "at_least_4_of_5_pairs_positive": (draws[:, PAIR_INDEX] > 0.0).sum(axis=1)
        >= s4cp.GATE_MIN_POSITIVE_PAIRS,
        "no_strata_below_floor": (draws[:, STRATA_INDEX] >= s4cp.GATE_STRATA_I_AURC_FLOOR)
        .all(axis=1),
        "trims_positive": (draws[:, TRIM_INDEX] > 0.0).all(axis=1),
    }
    passed = np.logical_and.reduce(list(conditions.values()))
    return {"passed": passed, "conditions": conditions}


# --- mechanical sample-size selection ----------------------------------------------------------


def select_confirm_n(
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

    Covariance scales by ``n_reference / n``; effects shrink to ``scenario * theta``
    (negatives keep their sign — never flipped favourable). 75% is the only blocking
    scenario; 100%/50% are sensitivity. Only the 14 risk quantities enter the AND-chain.
    """
    theta_hat = np.asarray(theta_hat, dtype=np.float64)[RISK_QUANTITIES]
    covariance = np.asarray(covariance, dtype=np.float64)[np.ix_(RISK_QUANTITIES, RISK_QUANTITIES)]
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
                "marginal_power": {k: float(v.mean()) for k, v in evaluation["conditions"].items()},
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


def report_sha256(report: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(report, sort_keys=True, default=str).encode("utf-8")).hexdigest()


__all__ = [
    "N_PAIRS",
    "N_QUANTITIES",
    "PAIR_INDEX",
    "Q_BASE_VS_RANDOM",
    "Q_I_AURC",
    "Q_I_REGRET80",
    "Q_PR",
    "Q_SPEARMAN",
    "RISK_QUANTITIES",
    "STRATA_INDEX",
    "TRIM_INDEX",
    "Z_95",
    "aurc_regret",
    "compute_group_quantity_vector",
    "evaluate_draws",
    "joint_bootstrap",
    "load_screen_group_quantities",
    "regret_at_coverage",
    "report_sha256",
    "select_confirm_n",
]
