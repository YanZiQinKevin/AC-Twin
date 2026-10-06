"""Stage 4D-D2R fixed risk-gated idle-fallback policy (pure NumPy).

Plan ``AC-TWIN-RFER-STAGE4D-D2R-RISK-GATED-IDLE-2026-09-24-V0.1`` section 3. This module
implements the SINGLE fixed candidate rule and its frozen controls, nothing else. It never
touches a model, a threshold search, or an outcome label beyond what a frozen rule needs.

The rule under test (plan 3): with ``a0(g) = argmin_a J_hat(g, a)`` the frozen base selector,
``IDLE_PLAN_INDEX = 3`` (the ``(1,3)`` idle plan) and ``r_D(g)`` the frozen five-seed Direct
arithmetic-mean risk score (the regret risk of the ORIGINAL ``a0``, never an idle-risk
estimate), freeze ``tau_D`` on the OLD validation role as the ``ceil(0.80 * n)``-th score in
ascending ``(score, scenario_id)`` order. On any new scenario the fixed rule is::

    a_D2R(g) = a0(g)            if r_D(g) <= tau_D
               IDLE_PLAN_INDEX  otherwise

A score exactly equal to the threshold executes ``a0`` (``<=``). For a group whose base
selector already picked idle (``a0 == IDLE``) the fallback changes nothing. The screen only
uses the frozen scalar ``tau_D``; it never re-ranks new data by scenario id.

Every quantity here is a pure function of per-group arrays already aligned on the group axis
(``j_hat``/``j_true`` ``(G,10)``, ``a0``/``a_star``/risk ``(G,)``). Truth-side arrays
(``j_true``, ``a_star``, ``regret``) are legal for train/validation diagnostics and — after the
burned lock — the one-shot screen adjudication; they are never used to pick the threshold.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from ac_twin import stage4c_objective as s4co
from ac_twin import stage4cp_protocol as s4cp
from ac_twin import stage4d_external_training as T

PLAN_ID = "AC-TWIN-RFER-STAGE4D-D2R-RISK-GATED-IDLE-2026-09-24-V0.1"

# --- frozen contract (plan 3) ---------------------------------------------------------------
IDLE_PLAN_INDEX = T.IDLE_PLAN_INDEX            # 3 -> (1,3)
N_ACTIONS = T.N_ACTIONS                        # 10
COVERAGE = 0.80                                # validation 80% quantile -> 20% fallback
BASELINE_TIE_TOLERANCE = T.BASELINE_TIE_TOLERANCE  # 1e-12
#: the three simple risk scores eligible as the same-mechanism standard fallback (frozen order).
FALLBACK_BASELINES = (s4cp.BASELINE_MARGIN, s4cp.BASELINE_SEED_VOTE_ENTROPY,
                      s4cp.BASELINE_SEED_MARGIN_STD)
#: deterministic salt for the secondary random-20%-fallback control (plan 3 次分析).
RANDOM_FALLBACK_SALT = "AC-TWIN-D2R-RANDOM-2026-09-24|"


# --- threshold freezing ----------------------------------------------------------------------


def freeze_threshold(scores: np.ndarray, scenario_ids, coverage: float = COVERAGE) -> float:
    """Freeze ``tau`` = the ``ceil(coverage * n)``-th score in ascending ``(score, id)`` order.

    The sort key is ``(score, scenario_id lexicographic)`` so the frozen scalar is deterministic
    even when validation scores tie around the quantile; the RETURNED value is only the score at
    that position. On new data the policy compares ``risk <= tau`` directly and never re-ranks.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    ids = [str(s) for s in scenario_ids]
    n = int(scores.size)
    if n == 0:
        raise ValueError("freeze_threshold needs at least one score")
    if len(ids) != n:
        raise ValueError("scores and scenario_ids must align")
    if not np.isfinite(scores).all():
        raise ValueError("freeze_threshold requires finite scores")
    order = sorted(range(n), key=lambda i: (scores[i], ids[i]))
    k = int(np.ceil(float(coverage) * n))
    k = max(1, min(k, n))                       # 1-indexed position clamped to [1, n]
    return float(scores[order[k - 1]])


def apply_fallback(a0: np.ndarray, risk: np.ndarray, tau: float) -> np.ndarray:
    """The fixed rule: ``a0`` where ``risk <= tau``, else ``IDLE_PLAN_INDEX`` (score==tau -> a0)."""
    a0 = np.asarray(a0, dtype=np.int64).ravel()
    risk = np.asarray(risk, dtype=np.float64).ravel()
    if a0.shape != risk.shape:
        raise ValueError("a0 and risk must align on the group axis")
    return np.where(risk <= float(tau), a0, IDLE_PLAN_INDEX).astype(np.int64)


def fallback_fractions(a0: np.ndarray, risk: np.ndarray, tau: float) -> dict[str, float]:
    """The plan-3 coverage reporting: above-threshold / actually-changed / final-non-idle."""
    a0 = np.asarray(a0, dtype=np.int64).ravel()
    risk = np.asarray(risk, dtype=np.float64).ravel()
    chosen = apply_fallback(a0, risk, tau)
    above = risk > float(tau)
    changed = above & (a0 != IDLE_PLAN_INDEX)
    non_idle = chosen != IDLE_PLAN_INDEX
    return {
        "frac_above_threshold": float(above.mean()),
        "frac_changed_action": float(changed.mean()),
        "frac_final_non_idle": float(non_idle.mean()),
    }


# --- executed-action regret -------------------------------------------------------------------


def action_regret(j_true: np.ndarray, chosen: np.ndarray, a_star: np.ndarray) -> np.ndarray:
    """Per-group regret of EXECUTING ``chosen``: ``clamp_regret(J_true[chosen] - J_true[a_star])``.

    The final policy regret is always recomputed from the truly executed action; the old Direct
    AURC on the original ``a0`` is a different quantity and is never substituted for it.
    """
    j_true = np.asarray(j_true, dtype=np.float64)
    chosen = np.asarray(chosen, dtype=np.int64).ravel()
    a_star = np.asarray(a_star, dtype=np.int64).ravel()
    groups = j_true.shape[0]
    if j_true.ndim != 2 or j_true.shape[1] != N_ACTIONS:
        raise ValueError("j_true must align as (G, 10)")
    if chosen.shape != (groups,) or a_star.shape != (groups,):
        raise ValueError("chosen and a_star must align on the group axis")
    rows = np.arange(groups)
    return s4co.clamp_regret(j_true[rows, chosen] - j_true[rows, a_star])


def relative_improvement(mean_baseline: float, mean_policy: float) -> float:
    """``(mean_baseline - mean_policy) / mean_baseline``; NaN when the denominator is non-positive.

    Fail-closed like the frozen D2 improvement: a non-positive or non-finite denominator yields
    NaN (never an epsilon, never an absolute difference), so a degenerate baseline can never be
    reported as a real improvement.
    """
    mean_baseline = float(mean_baseline)
    mean_policy = float(mean_policy)
    if not np.isfinite(mean_baseline) or not np.isfinite(mean_policy) or mean_baseline <= 0.0:
        return float("nan")
    return (mean_baseline - mean_policy) / mean_baseline


# --- controls ----------------------------------------------------------------------------------


def random_fallback_scores(scenario_ids) -> np.ndarray:
    """Deterministic pseudo-random score in [0,1) per scenario id (secondary control only).

    Salt ``AC-TWIN-D2R-RANDOM-2026-09-24|``; never used to re-select the main rule.
    """
    ids = [str(s) for s in scenario_ids]
    out = np.empty(len(ids), dtype=np.float64)
    for i, sid in enumerate(ids):
        digest = hashlib.sha256((RANDOM_FALLBACK_SALT + sid).encode("utf-8")).digest()
        out[i] = int.from_bytes(digest[:8], "little") / float(2**64)
    return out


def evaluate_policy(j_true: np.ndarray, a_star: np.ndarray, a0: np.ndarray,
                    risk: np.ndarray, tau: float) -> dict[str, Any]:
    """One frozen rule on one role: chosen action, executed regret and coverage fractions."""
    chosen = apply_fallback(a0, risk, tau)
    regret = action_regret(j_true, chosen, a_star)
    fractions = fallback_fractions(a0, risk, tau)
    return {
        "chosen_action": chosen,
        "regret": regret,
        "mean_regret": float(regret.mean()),
        **fractions,
    }


def select_strongest_standard_fallback(j_true: np.ndarray, a_star: np.ndarray,
                                       a0: np.ndarray,
                                       baseline_risks: Mapping[str, np.ndarray],
                                       scenario_ids) -> dict[str, Any]:
    """Freeze each simple score's own threshold on validation, pick the lowest-mean-regret one.

    Every candidate uses the SAME 80% validation threshold + SAME idle-fallback formula as the
    Direct rule; only its own validation distribution sets its threshold. Ties within 1e-12 break
    in the frozen order (MARGIN_ONLY, SEED_VOTE_ENTROPY, SEED_MARGIN_STD). This is the unique
    ``STRONGEST_STANDARD_FALLBACK``; the screen reports all three but never substitutes.
    """
    per_name: dict[str, dict[str, Any]] = {}
    for name in FALLBACK_BASELINES:
        risk = np.asarray(baseline_risks[name], dtype=np.float64).ravel()
        tau = freeze_threshold(risk, scenario_ids)
        per_name[name] = {"tau": tau, **evaluate_policy(j_true, a_star, a0, risk, tau)}
    strongest = FALLBACK_BASELINES[0]
    for name in FALLBACK_BASELINES[1:]:
        if per_name[name]["mean_regret"] < per_name[strongest]["mean_regret"] \
                - BASELINE_TIE_TOLERANCE:
            strongest = name
    return {
        "strongest": strongest,
        "per_baseline": per_name,
        "tie_break_order": list(FALLBACK_BASELINES),
        "tie_tolerance": BASELINE_TIE_TOLERANCE,
    }


__all__ = [
    "BASELINE_TIE_TOLERANCE",
    "COVERAGE",
    "FALLBACK_BASELINES",
    "IDLE_PLAN_INDEX",
    "N_ACTIONS",
    "PLAN_ID",
    "RANDOM_FALLBACK_SALT",
    "action_regret",
    "apply_fallback",
    "evaluate_policy",
    "fallback_fractions",
    "freeze_threshold",
    "random_fallback_scores",
    "relative_improvement",
    "select_strongest_standard_fallback",
]
