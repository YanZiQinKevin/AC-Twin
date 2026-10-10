"""M1 supervised-baseline ablation: paired group statistics, fixed contrasts (plan 6/7).

The statistical unit is the complete ten-action GROUP. All methods are scored on the
same 1,500 validation groups; every bootstrap draw applies ONE whole-group resample
index to every arm, every heuristic and the labels, and recomputes every metric and
ranking. The main family is the five AURC-regret contrasts C1-C5 (family alpha 0.05,
per-contrast two-sided 0.01 -> 99% percentile CI, quantiles 0.005/0.995); the five
matching Regret@80 contrasts form a separate pre-specified secondary family with the
same 99% intervals. Uncorrected 95% intervals may accompany them, clearly labelled.
Relative improvement whose reference denominator is <= 1e-12 is NOT_IDENTIFIABLE: the
absolute difference is still reported, the relative interval is not.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from ac_twin import m1_ablation_protocol as m1p
from ac_twin import stage4cp_statistics as s4cs

METRIC_AURC = "aurc_regret"
METRIC_R80 = "regret_at_80"


def method_metrics(order_scores: np.ndarray, regret: np.ndarray) -> dict[str, float]:
    """AURC-regret and Regret@80 for one score vector on one group index set."""
    order = s4cs.order_by_risk(order_scores)
    return {
        METRIC_AURC: s4cs.aurc_regret(order, regret),
        METRIC_R80: s4cs.regret_at_coverage(order, regret, m1p.COVERAGE_MAIN),
    }


def improvement_and_diff(reference: float, candidate: float) -> tuple[float, float]:
    """(relative improvement 1 - cand/ref, absolute difference ref - cand).

    The relative form is NaN (NOT_IDENTIFIABLE) when the denominator is at/below the
    frozen 1e-12 floor; the absolute difference is always defined.
    """
    diff = float(reference - candidate)
    if reference <= m1p.DENOMINATOR_FLOOR:
        return float("nan"), diff
    return float(1.0 - candidate / reference), diff


def contrast_names() -> list[str]:
    names: list[str] = []
    for cid, method, reference in m1p.CONTRASTS:
        for metric in (METRIC_AURC, METRIC_R80):
            names.append(f"{cid}_{metric}_improvement")
            names.append(f"{cid}_{metric}_absdiff")
    return names


def quantity_vector(index: np.ndarray,
                    scores: Mapping[str, np.ndarray],
                    regret: np.ndarray) -> np.ndarray:
    """All absolute metrics + all contrast quantities on one whole-group index.

    Rankings are recomputed inside every call; a stable sort keeps the fixed source
    row order among ties (and the fixed resampled order inside bootstrap draws).
    """
    index = np.asarray(index, dtype=np.int64)
    regret = np.asarray(regret, dtype=np.float64).ravel()[index]
    absolute: dict[str, dict[str, float]] = {}
    values: list[float] = []
    for method in m1p.ALL_METHODS:
        metrics = method_metrics(np.asarray(scores[method], dtype=np.float64)[index],
                                 regret)
        absolute[method] = metrics
        values.extend([metrics[METRIC_AURC], metrics[METRIC_R80]])
    for cid, method, reference in m1p.CONTRASTS:
        for metric in (METRIC_AURC, METRIC_R80):
            improvement, diff = improvement_and_diff(
                absolute[reference][metric], absolute[method][metric])
            values.extend([improvement, diff])
    return np.asarray(values, dtype=np.float64)


def absolute_metric_names() -> list[str]:
    return [f"{m}_{metric}" for m in m1p.ALL_METHODS
            for metric in (METRIC_AURC, METRIC_R80)]


def all_quantity_names() -> list[str]:
    return absolute_metric_names() + contrast_names()


def paired_bootstrap(scores: Mapping[str, np.ndarray],
                     regret: np.ndarray,
                     *,
                     n_draws: int = m1p.BOOTSTRAP_DRAWS,
                     entropy: int = m1p.BOOTSTRAP_ENTROPY,
                     progress: Callable[[str], None] | None = None,
                     ) -> dict[str, Any]:
    """10,000-draw whole-group paired bootstrap of the full quantity vector.

    One shared resample index per draw couples every arm, heuristic and label; the
    same draws back both the improvement and the absolute-difference form of each
    contrast (two expressions of one comparison, never two independent selections).
    """
    regret = np.asarray(regret, dtype=np.float64).ravel()
    groups = regret.size
    identity = np.arange(groups, dtype=np.int64)
    point = quantity_vector(identity, scores, regret)
    generator = np.random.default_rng(int(entropy))
    draws = np.empty((int(n_draws), point.size), dtype=np.float64)
    for draw in range(int(n_draws)):
        index = generator.integers(0, groups, size=groups)
        draws[draw] = quantity_vector(index, scores, regret)
        if progress is not None and ((draw + 1) % 500 == 0 or draw + 1 == int(n_draws)):
            progress(f"bootstrap draws {draw + 1}/{int(n_draws)}")
    return {
        "names": all_quantity_names(),
        "point": point,
        "draws": draws,
        "n_draws": int(n_draws),
        "entropy": int(entropy),
    }


def _interval(draws: np.ndarray, quantiles: tuple[float, float]) -> tuple[float, float]:
    return (float(np.quantile(draws, quantiles[0])),
            float(np.quantile(draws, quantiles[1])))


def contrast_table(bootstrap: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per contrast x metric: point, 99% (family) CI, uncorrected 95% CI.

    A relative-improvement row with ANY invalid (NOT_IDENTIFIABLE) draw reports the
    invalid-draw count and NO relative interval; its absolute-difference interval is
    still reported (plan section 7).
    """
    names = list(bootstrap["names"])
    point = np.asarray(bootstrap["point"], dtype=np.float64)
    draws = np.asarray(bootstrap["draws"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for cid, method, reference in m1p.CONTRASTS:
        for metric in (METRIC_AURC, METRIC_R80):
            for form in ("improvement", "absdiff"):
                key = f"{cid}_{metric}_{form}"
                column = names.index(key)
                values = draws[:, column]
                invalid = int((~np.isfinite(values)).sum())
                row: dict[str, Any] = {
                    "contrast": cid, "metric": metric, "form": form,
                    "method": method, "reference": reference,
                    "point": float(point[column]),
                    "invalid_draws": invalid,
                }
                if invalid > 0 and form == "improvement":
                    row["ci99_lower"] = row["ci99_upper"] = None
                    row["ci95_lower"] = row["ci95_upper"] = None
                    row["status"] = "NOT_IDENTIFIABLE"
                else:
                    valid = values[np.isfinite(values)]
                    lo99, hi99 = _interval(valid, m1p.MAIN_CI_QUANTILES)
                    lo95, hi95 = _interval(valid, m1p.UNCORRECTED_CI_QUANTILES)
                    row.update(ci99_lower=lo99, ci99_upper=hi99,
                               ci95_lower=lo95, ci95_upper=hi95,
                               status="OK")
                rows.append(row)
    return rows


def absolute_metrics_table(bootstrap: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per method: AURC/Regret@80 point estimates (paired, same draws)."""
    names = list(bootstrap["names"])
    point = np.asarray(bootstrap["point"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for method in m1p.ALL_METHODS:
        row: dict[str, Any] = {"method": method}
        for metric in (METRIC_AURC, METRIC_R80):
            row[metric] = float(point[names.index(f"{method}_{metric}")])
        rows.append(row)
    return rows


def per_seed_contrasts(scores_by_seed: Mapping[str, np.ndarray],
                       heuristics: Mapping[str, np.ndarray],
                       regret: np.ndarray) -> list[dict[str, Any]]:
    """Per-seed C1-C5 point estimates (five matched seeds; plan section 7 item 6).

    C1/C2 compare the same seed of FULL vs BASELINE3/NO_RISK; C3-C5 compare that
    seed's BASELINE3 with the fixed heuristic. Descriptive only: no intervals, no
    upgrade to the main family.
    """
    regret = np.asarray(regret, dtype=np.float64).ravel()
    heuristic_metrics = {
        name: method_metrics(np.asarray(values, dtype=np.float64), regret)
        for name, values in heuristics.items()}
    rows: list[dict[str, Any]] = []
    n_seeds = np.asarray(scores_by_seed["FULL_MATCHED"]).shape[0]
    for i in range(n_seeds):
        arm_metrics = {
            arm: method_metrics(np.asarray(scores_by_seed[arm], dtype=np.float64)[i],
                                regret)
            for arm in m1p.ARMS}
        pairs = (
            ("C1", arm_metrics["FULL_MATCHED"], arm_metrics["BASELINE3_MATCHED"]),
            ("C2", arm_metrics["FULL_MATCHED"], arm_metrics["NO_RISK_MATCHED"]),
            ("C3", arm_metrics["BASELINE3_MATCHED"], heuristic_metrics["SEED_VOTE_ENTROPY"]),
            ("C4", arm_metrics["BASELINE3_MATCHED"], heuristic_metrics["MARGIN_ONLY"]),
            ("C5", arm_metrics["BASELINE3_MATCHED"], heuristic_metrics["SEED_MARGIN_STD"]),
        )
        for cid, cand, ref in pairs:
            for metric in (METRIC_AURC, METRIC_R80):
                improvement, diff = improvement_and_diff(ref[metric], cand[metric])
                rows.append({
                    "contrast": cid, "metric": metric, "seed_index": i,
                    "improvement": improvement, "absdiff": diff,
                    "direction": ("positive" if improvement > 0 else
                                  "negative" if improvement < 0 else "zero"),
                })
    return rows


def high_regret_pr_by_method(scores: Mapping[str, np.ndarray],
                             high_label: np.ndarray) -> dict[str, float]:
    """Descriptive PR-AUC - prevalence per method (never a main endpoint)."""
    return {method: s4cs.high_regret_pr(np.asarray(values, dtype=np.float64),
                                        np.asarray(high_label, dtype=np.int64))
            for method, values in scores.items()}


def risk_coverage_curve(scores: np.ndarray, regret: np.ndarray,
                        *, points: int = 101) -> dict[str, np.ndarray]:
    """Equally spaced risk-coverage curve for the report figure."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    regret = np.asarray(regret, dtype=np.float64).ravel()
    order = s4cs.order_by_risk(scores)
    coverages = np.linspace(1.0 / regret.size, 1.0, int(points))
    values = np.asarray([
        s4cs.regret_at_coverage(order, regret, float(c)) for c in coverages])
    return {"coverage": coverages, "risk": values}


def continuity_check(absolute_rows: list[dict[str, Any]],
                     gate_report: Mapping[str, Any]) -> dict[str, Any]:
    """OLD frozen Direct on validation vs the frozen R2 gate report (plan P3).

    The absolute AURC/Regret@80 of FULL_FROZEN_REFERENCE must reproduce the frozen
    ``validation_gate_report.json`` values within the frozen 1e-8 tolerance; exceeding
    it is an engineering localisation stop, never a silent pass.
    """
    metrics = gate_report["metrics"]
    row = next(r for r in absolute_rows if r["method"] == "FULL_FROZEN_REFERENCE")
    expected = {
        METRIC_AURC: float(metrics["aurc_regret_ensemble"]),
        METRIC_R80: float(metrics["regret80_ensemble"]),
    }
    result = {"tolerance": m1p.CONTINUITY_TOLERANCE, "checks": {}, "passed": True}
    for metric, want in expected.items():
        delta = abs(row[metric] - want)
        result["checks"][metric] = {
            "expected": want, "actual": row[metric], "abs_delta": float(delta),
            "within_tolerance": bool(delta <= m1p.CONTINUITY_TOLERANCE)}
        result["passed"] = bool(result["passed"] and delta <= m1p.CONTINUITY_TOLERANCE)
    # the fixed heuristics are deterministic given the frozen predictions: same check.
    baseline_receipt = gate_report["baseline_receipt"]
    for method in m1p.HEURISTIC_METHODS:
        row_m = next(r for r in absolute_rows if r["method"] == method)
        for metric, key in ((METRIC_AURC, "aurc_regret"), (METRIC_R80, "regret_at_80")):
            want = float(baseline_receipt[key][method])
            delta = abs(row_m[metric] - want)
            result["checks"][f"{method}_{metric}"] = {
                "expected": want, "actual": row_m[metric], "abs_delta": float(delta),
                "within_tolerance": bool(delta <= m1p.CONTINUITY_TOLERANCE)}
            result["passed"] = bool(result["passed"] and delta <= m1p.CONTINUITY_TOLERANCE)
    return result


__all__ = [
    "METRIC_AURC", "METRIC_R80", "absolute_metric_names", "absolute_metrics_table",
    "all_quantity_names", "continuity_check", "contrast_names", "contrast_table",
    "high_regret_pr_by_method", "improvement_and_diff", "method_metrics",
    "paired_bootstrap", "per_seed_contrasts", "quantity_vector",
    "risk_coverage_curve",
]
