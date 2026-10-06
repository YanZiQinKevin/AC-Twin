"""Stage 5 statistics layer (plan v0.3 section 7.2, Task 4).

Plan ``AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3``. Primary metrics are
thin wrappers over the frozen Stage 4 formulas (``stage4cp_statistics``) so parity
is structural, not re-derived. The secondary panel (coverage profile, tail
severity/capture, oracle gap closure, retained-tail stats) is frozen HERE for
Stage 5 — these quantities did not exist in earlier stages. Native calibration is
reported only where the adaptation yields a semantically checkable probability or
interval (P1 TCP-vs-oracle-hit, P4 conformal coverage, P5 Gaussian NLL/coverage);
every other method is ``NOT_APPLICABLE``. Improvements fail closed (NaN) on
non-positive denominators.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ac_twin import stage4cp_statistics as s4cs

COVERAGE_MAIN = 0.80
COVERAGE_POINTS = tuple(np.linspace(0.05, 1.0, 20))
TAIL_FRACTION = 0.20
TRIM_FRACTIONS = (0.01, 0.05)
TOST_ALPHA = 0.05  # two one-sided tests -> two-sided 90% CI inside the bounds


def improvement_fail_closed(baseline: float, model: float) -> float:
    """Relative improvement (baseline - model)/baseline; NaN when baseline <= 0."""
    baseline = float(baseline)
    return float((baseline - model) / baseline) if baseline > 0 else float("nan")


# --- primary + secondary group metrics -----------------------------------------------------


def evaluate_group_metrics(risk_ensemble: np.ndarray, risk_by_seed: np.ndarray,
                           truth: dict) -> dict[str, Any]:
    """Full per-method metric block on one role (validation or holdout)."""
    ens = np.asarray(risk_ensemble, dtype=np.float64).ravel()
    by_seed = np.asarray(risk_by_seed, dtype=np.float64)
    regret = np.asarray(truth["regret"], dtype=np.float64).ravel()
    y_high = np.asarray(truth["high_regret"], dtype=np.int64).ravel()
    order = s4cs.order_by_risk(ens)

    per_seed_aurc = np.empty(by_seed.shape[0], dtype=np.float64)
    per_seed_r80 = np.empty_like(per_seed_aurc)
    for i in range(by_seed.shape[0]):
        o = s4cs.order_by_risk(by_seed[i])
        per_seed_aurc[i] = s4cs.aurc_regret(o, regret)
        per_seed_r80[i] = s4cs.regret_at_coverage(o, regret, COVERAGE_MAIN)

    trims: dict[str, float] = {}
    for frac in TRIM_FRACTIONS:
        keep = s4cs.trim_high_regret(regret, frac)
        if int(keep.sum()) >= 2:
            trims[f"trim{int(frac * 100)}_aurc"] = s4cs.aurc_regret(
                s4cs.order_by_risk(ens[keep]), regret[keep])
        else:
            trims[f"trim{int(frac * 100)}_aurc"] = float("nan")

    coverages, profile = coverage_profile(ens, regret)
    p95, mx = retained_tail_stats(ens, regret, COVERAGE_MAIN)
    return {
        "aurc_regret": s4cs.aurc_regret(order, regret),
        "regret_at_70": s4cs.regret_at_coverage(order, regret, 0.70),
        "regret_at_80": s4cs.regret_at_coverage(order, regret, 0.80),
        "regret_at_90": s4cs.regret_at_coverage(order, regret, 0.90),
        "regret_at_95": s4cs.regret_at_coverage(order, regret, 0.95),
        "pr_auc_minus_prevalence": s4cs.high_regret_pr(ens, y_high),
        "spearman_risk_regret": s4cs.spearman(ens, regret),
        "per_seed_aurc": per_seed_aurc,
        "per_seed_regret80": per_seed_r80,
        "coverage_profile_coverages": coverages,
        "coverage_profile_values": profile,
        "tail_severity_at_20": tail_severity(ens, regret, TAIL_FRACTION),
        "tail_capture_at_20": tail_capture(ens, y_high, TAIL_FRACTION),
        "oracle_gap_closure_at_80": oracle_gap_closure(ens, regret, COVERAGE_MAIN),
        "retained80_regret_p95": p95,
        "retained80_regret_max": mx,
        **trims,
    }


def coverage_profile(risk: np.ndarray, regret: np.ndarray,
                     points: tuple = COVERAGE_POINTS):
    """The 5%-100% risk-coverage curve R(c) (plan 7.2 primary endpoint companion)."""
    risk = np.asarray(risk, dtype=np.float64).ravel()
    regret = np.asarray(regret, dtype=np.float64).ravel()
    order = s4cs.order_by_risk(risk)
    coverages = np.asarray(points, dtype=np.float64)
    values = np.array([s4cs.regret_at_coverage(order, regret, float(c))
                       for c in coverages], dtype=np.float64)
    return coverages, values


def tail_severity(risk: np.ndarray, regret: np.ndarray,
                  fraction: float = TAIL_FRACTION) -> float:
    """Mean regret among the top-``fraction`` highest-risk groups (the rejected tail)."""
    risk = np.asarray(risk, dtype=np.float64).ravel()
    regret = np.asarray(regret, dtype=np.float64).ravel()
    order = s4cs.order_by_risk(risk)
    k = max(1, int(round(fraction * risk.size)))
    return float(regret[order][-k:].mean())


def tail_capture(risk: np.ndarray, y_high: np.ndarray,
                 fraction: float = TAIL_FRACTION) -> float:
    """Fraction of all high-regret groups inside the top-``fraction`` risk tail."""
    risk = np.asarray(risk, dtype=np.float64).ravel()
    y_high = np.asarray(y_high, dtype=np.int64).ravel()
    total = int(y_high.sum())
    if total == 0:
        return 0.0
    order = s4cs.order_by_risk(risk)
    k = max(1, int(round(fraction * risk.size)))
    return float(y_high[order][-k:].sum() / total)


def oracle_gap_closure(risk: np.ndarray, regret: np.ndarray,
                       coverage: float = COVERAGE_MAIN) -> float:
    """Share of the achievable regret reduction captured by this ranking at ``coverage``.

    ``(mean_all - R(c)) / (mean_all - R_oracle(c))`` where ``R_oracle(c)`` is the
    regret of the best achievable ranking (groups sorted by true regret). Fail-closed
    NaN when the achievable reduction is zero (denominator <= 0).
    """
    risk = np.asarray(risk, dtype=np.float64).ravel()
    regret = np.asarray(regret, dtype=np.float64).ravel()
    mean_all = float(regret.mean())
    r_c = s4cs.regret_at_coverage(s4cs.order_by_risk(risk), regret, coverage)
    r_oracle = s4cs.regret_at_coverage(
        np.argsort(regret, kind="stable"), regret, coverage)
    denom = mean_all - r_oracle
    return float((mean_all - r_c) / denom) if denom > 0 else float("nan")


def retained_tail_stats(risk: np.ndarray, regret: np.ndarray,
                        coverage: float = COVERAGE_MAIN) -> tuple[float, float]:
    """(p95, max) of the regret on the retained lowest-risk ``coverage`` fraction."""
    risk = np.asarray(risk, dtype=np.float64).ravel()
    regret = np.asarray(regret, dtype=np.float64).ravel()
    order = s4cs.order_by_risk(risk)
    k = s4cs.coverage_index(coverage, risk.size)
    retained = regret[order][:k]
    return float(np.quantile(retained, 0.95)), float(retained.max())


# --- native calibration (integrity report; NOT_APPLICABLE by default) --------------------------


def native_calibration(method_id: str, raw_outputs: dict | None,
                       truth: dict) -> dict[str, Any]:
    """Native probability/interval calibration where the adaptation supports one.

    P1: predicted TCP vs empirical P(selected action == oracle action) — 10-bin ECE
    + Brier. P4: empirical coverage of [mu - half_width, mu + half_width]. P5:
    Gaussian NLL + empirical 95% interval coverage. Everything else NOT_APPLICABLE.
    """
    regret = np.asarray(truth["regret"], dtype=np.float64).ravel()
    if method_id == "CONFIDNET_TCP_ADAPTED":
        if raw_outputs is None or "tcp" not in raw_outputs:
            return {"status": "NOT_APPLICABLE"}
        tcp = np.asarray(raw_outputs["tcp"], dtype=np.float64).ravel()
        hit = (np.asarray(truth["selected_action"], dtype=np.int64).ravel()
               == np.asarray(truth["oracle_action"], dtype=np.int64).ravel()).astype(np.float64)
        bins = np.linspace(0.0, 1.0, 11)
        idx = np.clip(np.digitize(tcp, bins[1:-1]), 0, 9)
        ece = 0.0
        for b in range(10):
            mask = idx == b
            if mask.any():
                ece += float(mask.mean()) * abs(float(tcp[mask].mean())
                                                - float(hit[mask].mean()))
        return {"status": "OK", "target": "P(selected action == oracle action)",
                "ece": float(ece),
                "brier": float(((tcp - hit) ** 2).mean()),
                "mean_tcp": float(tcp.mean()), "empirical_hit_rate": float(hit.mean())}
    if method_id == "CONFORMAL_REGRESSION_REJECT_ADAPTED":
        if raw_outputs is None or "mu" not in raw_outputs or "half_width" not in raw_outputs:
            return {"status": "NOT_APPLICABLE"}
        mu = np.asarray(raw_outputs["mu"], dtype=np.float64).ravel()
        half = np.asarray(raw_outputs["half_width"], dtype=np.float64).ravel()
        coverage = float((np.abs(regret - mu) <= half).mean())
        return {"status": "OK", "target_coverage": 0.90,
                "empirical_coverage": coverage,
                "coverage_error": coverage - 0.90,
                "mean_interval_width": float((2.0 * half).mean())}
    if method_id == "CCTR_DRIVING_CALIBRATION_ADAPTED":
        if raw_outputs is None or "mu" not in raw_outputs or "sigma" not in raw_outputs:
            return {"status": "NOT_APPLICABLE"}
        mu = np.asarray(raw_outputs["mu"], dtype=np.float64).ravel()
        sigma = np.asarray(raw_outputs["sigma"], dtype=np.float64).ravel()
        import math
        nll = float((np.log(sigma) + 0.5 * math.log(2 * math.pi)
                     + (regret - mu) ** 2 / (2 * sigma ** 2)).mean())
        cov95 = float((np.abs(regret - mu) <= 1.959963984540054 * sigma).mean())
        return {"status": "OK", "nll": nll, "target_coverage_95": 0.95,
                "empirical_coverage_95": cov95, "coverage_error_95": cov95 - 0.95,
                "mean_sigma": float(sigma.mean())}
    return {"status": "NOT_APPLICABLE"}


# --- TOST equivalence -------------------------------------------------------------------------


def tost_equivalent(ci_lower_90: float, ci_upper_90: float,
                    bounds: tuple[float, float]) -> bool:
    """TOST: the two-sided 90% CI must lie strictly inside the equivalence bounds.

    Fail-closed: any non-finite bound or CI edge returns False.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    ci_lo, ci_hi = float(ci_lower_90), float(ci_upper_90)
    if not all(np.isfinite(v) for v in (lo, hi, ci_lo, ci_hi)):
        return False
    return bool(ci_lo > lo and ci_hi < hi)


__all__ = [
    "COVERAGE_MAIN", "COVERAGE_POINTS", "TAIL_FRACTION", "TRIM_FRACTIONS", "TOST_ALPHA",
    "improvement_fail_closed", "evaluate_group_metrics", "coverage_profile",
    "tail_severity", "tail_capture", "oracle_gap_closure", "retained_tail_stats",
    "native_calibration", "tost_equivalent",
]
