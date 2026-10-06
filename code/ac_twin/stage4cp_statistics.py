"""Stage 4C Plus group-level risk-curve statistics primitives (plan sections 5/7/8/11).

The statistical unit is the decision GROUP. A risk curve sorts groups by a group-level
risk score ascending; ``R(k)`` is the mean regret of the ``k`` lowest-risk groups and
``AURC_REGRET`` its mean over ``k``. Improvement is relative to a frozen baseline and is
FAIL-CLOSED: a non-positive baseline denominator yields NaN (which the gates turn into
``STAGE4CP_INVALID_ANALYSIS``), never an absolute-difference substitute (plan 8.3/11.3).

These primitives are shared by the Phase 0 OOF screen, validation selection, the
development screen and the confirm adjudication; the full 16-quantity joint vector and
gate evaluation are layered on in the power/screen stage.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
from scipy.stats import rankdata

from ac_twin import stage4_statistics as s4s

coverage_index = s4s.coverage_index


# --- risk-curve primitives -------------------------------------------------------------


def order_by_risk(group_risk: np.ndarray) -> np.ndarray:
    """Ascending stable sort: lowest-risk group first (auto-executed earliest)."""
    return np.argsort(np.asarray(group_risk, dtype=np.float64), kind="stable")


def aurc_regret(order: np.ndarray, regret: np.ndarray) -> float:
    """AURC_REGRET = mean_{k=1..G} R(k); R(k) = mean regret of the k lowest-risk."""
    regret = np.asarray(regret, dtype=np.float64)
    sorted_regret = regret[np.asarray(order)]
    cumulative = np.cumsum(sorted_regret)
    return float((cumulative / np.arange(1, regret.size + 1)).mean())


def regret_at_coverage(order: np.ndarray, regret: np.ndarray, coverage: float) -> float:
    """R(round(coverage*G)): mean regret of the lowest-risk ``coverage`` fraction."""
    regret = np.asarray(regret, dtype=np.float64)
    k = coverage_index(coverage, regret.size)
    return float(regret[np.asarray(order)][:k].mean())


def improvement(baseline: float, combined: float) -> float:
    """Relative improvement ``(baseline - combined)/baseline``; NaN when denom <= 0."""
    return float((baseline - combined) / baseline) if baseline > 0 else float("nan")


def high_regret_pr(group_risk: np.ndarray, label: np.ndarray) -> float:
    """PR-AUC - prevalence for the high-regret label (groups as samples)."""
    label = np.asarray(label, dtype=np.int64)
    prevalence = float(label.mean())
    if label.min() == label.max():
        return 0.0
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(label, np.asarray(group_risk, dtype=np.float64)) - prevalence)


def trim_high_regret(regret: np.ndarray, fraction: float) -> np.ndarray:
    """Boolean keep-mask removing the top-``fraction`` highest-regret groups."""
    regret = np.asarray(regret, dtype=np.float64)
    n = regret.size
    drop = int(np.floor(fraction * n))
    if drop <= 0:
        return np.ones(n, dtype=bool)
    order = np.argsort(regret, kind="stable")  # ascending; last `drop` are highest
    keep = np.ones(n, dtype=bool)
    keep[order[n - drop:]] = False
    return keep


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; 0 when either side is degenerate."""
    ra = rankdata(np.asarray(a, dtype=np.float64).ravel())
    rb = rankdata(np.asarray(b, dtype=np.float64).ravel())
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


# --- generic whole-group paired bootstrap --------------------------------------------------


def paired_group_bootstrap(
    quantity_fn: Callable[[np.ndarray], np.ndarray],
    groups: int,
    *,
    n_resamples: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    """Whole-group paired bootstrap of a quantity vector.

    ``quantity_fn(index)`` recomputes every quantity on the same whole-group resample
    ``index`` (with replacement), so all quantities stay coupled. Returns the identity
    point estimate and two-sided ``confidence`` CI from the resample quantiles.
    """
    generator = np.random.default_rng(int(seed))
    identity = np.arange(int(groups), dtype=np.int64)
    point = np.asarray(quantity_fn(identity), dtype=np.float64)
    n_quantities = point.size
    draws = np.empty((int(n_resamples), n_quantities), dtype=np.float64)
    for draw in range(int(n_resamples)):
        index = generator.integers(0, groups, size=int(groups))
        draws[draw] = np.asarray(quantity_fn(index), dtype=np.float64)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "point": point,
        "ci_lower": np.quantile(draws, alpha, axis=0),
        "ci_upper": np.quantile(draws, 1.0 - alpha, axis=0),
        "draws": draws,
        "n_resamples": int(n_resamples),
        "bootstrap_seed": int(seed),
        "confidence_level": float(confidence),
    }


__all__ = [
    "aurc_regret",
    "coverage_index",
    "high_regret_pr",
    "improvement",
    "order_by_risk",
    "paired_group_bootstrap",
    "regret_at_coverage",
    "spearman",
    "trim_high_regret",
]
