"""Stage 4D-D2R whole-group paired bootstrap + Phase 0 Go/No-Go gate.

Plan ``AC-TWIN-RFER-STAGE4D-D2R-RISK-GATED-IDLE-2026-09-24-V0.1`` sections 4 and 7. The
statistical unit is the 10-action decision GROUP. One paired bootstrap resamples whole groups
with replacement; EVERY quantity of a draw is recomputed from the SAME resample index vector and
the two-sided CI is read off the resample quantiles. The frozen fallback thresholds are point
constants fit on the FULL validation role (plan 3 解释限制); the bootstrap never re-fits them —
it only propagates group resampling through the fixed policy's executed-regret evaluation.

This module currently implements the Phase 0 development Go/No-Go gate (plan 4). The plan-7
one-shot screen adjudication gate is added when (and only if) Phase 0 + power qualify; it uses a
DIFFERENT frozen seed (``2026122403``), 10,000 resamples and the plan-7 AND conditions.

Phase 0 is ``ADAPTIVE_DEVELOPMENT_ONLY``: the old validation role was already used for D2 model
early stopping / baseline selection, so every Phase 0 number is a development cost screen, never
an unbiased confirmation CI (plan 2.2 / 4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from ac_twin import stage4cp_statistics as s4cs
from ac_twin import stage4d_d2r_policy as P
from ac_twin import stage4d_external_training as T

# --- frozen Phase 0 bootstrap contract (plan 4) ----------------------------------------------
PHASE0_BOOTSTRAP_SEED = 2026122401
PHASE0_RESAMPLES = 2000
CONFIDENCE = 0.80
#: screen adjudication contract (plan 7) — used only by the later screen gate, frozen here so the
#: two streams can never collide with each other or with any consumed D2 stream.
SCREEN_BOOTSTRAP_SEED = 2026122403
SCREEN_RESAMPLES = 10000

# --- Phase 0 Go/No-Go thresholds (plan 4) ------------------------------------------------------
MIN_ZONE_GROUPS = 200
MIN_RELATIVE_IMPROVEMENT = 0.05
MIN_FRAC_CHANGED = 0.10
MIN_FRAC_NON_IDLE = 0.20

# Phase 0 quantity-vector layout.
Q_MEAN_D2R = 0
Q_MEAN_IDLE = 1
Q_MEAN_BASE = 2
Q_MEAN_STRONGEST = 3
Q_I_IDLE = 4
Q_I_BASE = 5
Q_I_STANDARD = 6
Q_FRAC_CHANGED = 7
Q_FRAC_NON_IDLE = 8
Q_FRAC_ABOVE = 9
N_PHASE0_QUANTITIES = 10

_PHASE0_QUANTITY_NAMES = (
    "mean_regret_d2r", "mean_regret_idle", "mean_regret_base", "mean_regret_strongest_fallback",
    "i_idle", "i_base", "i_standard", "frac_changed_action", "frac_final_non_idle",
    "frac_above_threshold",
)

STATUS_POLICY_SIGNAL_NOT_FEASIBLE = "D2R_POLICY_SIGNAL_NOT_FEASIBLE"
STATUS_PHASE0_FEASIBLE = "D2R_PHASE0_FEASIBLE"


@dataclass(frozen=True)
class PairedBootstrap:
    """One whole-group paired bootstrap: identity point + CI + raw draws."""
    point: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    draws: np.ndarray
    n_resamples: int
    seed: int
    confidence: float


def paired_bootstrap(quantity_fn: Callable[[np.ndarray], np.ndarray], groups: int, *,
                     seed: int, n_resamples: int, confidence: float = CONFIDENCE
                     ) -> PairedBootstrap:
    """Whole-group paired bootstrap with shared indices and two-sided CI.

    ``quantity_fn(index)`` maps one with-replacement group index vector to a fixed-length
    quantity vector; the identity evaluation is the point estimate; the CI is the resample
    quantile band. Every quantity of a draw shares the same index vector (paired by construction).
    """
    groups = int(groups)
    if groups < 2:
        raise ValueError("paired_bootstrap needs at least 2 groups")
    generator = np.random.default_rng(int(seed))
    identity = np.arange(groups, dtype=np.int64)
    point = np.asarray(quantity_fn(identity), dtype=np.float64)
    draws = np.empty((int(n_resamples), point.size), dtype=np.float64)
    for draw in range(int(n_resamples)):
        draws[draw] = quantity_fn(generator.integers(0, groups, size=groups))
    alpha = (1.0 - float(confidence)) / 2.0
    return PairedBootstrap(
        point=point,
        ci_lower=np.quantile(draws, alpha, axis=0),
        ci_upper=np.quantile(draws, 1.0 - alpha, axis=0),
        draws=draws, n_resamples=int(n_resamples), seed=int(seed),
        confidence=float(confidence))


# --- Phase 0 quantity vector --------------------------------------------------------------------


def _validate_phase0(quantities: Mapping[str, Any]) -> int:
    """Fail closed on missing/misaligned/non-finite Phase 0 arrays."""
    required = ("regret_d2r", "regret_idle", "regret_base", "regret_strongest_fallback",
                "chosen_d2r", "a0", "risk_d2r", "tau_d2r",
                "n_high_zone", "n_low_zone", "diff_high_zone", "diff_low_zone")
    for key in required:
        if key not in quantities:
            raise ValueError(f"phase0 quantities missing {key!r}")
    regret = np.asarray(quantities["regret_d2r"], dtype=np.float64).ravel()
    groups = int(regret.size)
    if groups < 2:
        raise ValueError("phase0 needs at least 2 groups")
    for key in ("regret_d2r", "regret_idle", "regret_base", "regret_strongest_fallback"):
        arr = np.asarray(quantities[key], dtype=np.float64).ravel()
        if arr.shape != (groups,) or not np.isfinite(arr).all():
            raise ValueError(f"phase0 array {key!r} must be finite and align as ({groups},)")
    for key in ("chosen_d2r", "a0"):
        if np.asarray(quantities[key]).ravel().shape != (groups,):
            raise ValueError(f"phase0 array {key!r} must align as ({groups},)")
    if np.asarray(quantities["risk_d2r"], dtype=np.float64).ravel().shape != (groups,):
        raise ValueError("phase0 array 'risk_d2r' must align on the group axis")
    return groups


def compute_phase0_quantity_vector(index: np.ndarray, quantities: Mapping[str, Any]
                                   ) -> np.ndarray:
    """All Phase 0 policy quantities on ONE whole-group resample ``index``.

    The per-group policy regrets are precomputed with the FROZEN thresholds; a draw only
    re-averages them (and the improvements) over the resampled groups, so every quantity of the
    draw stays coupled through the same group multiplicities.
    """
    index = np.asarray(index, dtype=np.int64)
    reg_d2r = np.asarray(quantities["regret_d2r"], dtype=np.float64)[index]
    reg_idle = np.asarray(quantities["regret_idle"], dtype=np.float64)[index]
    reg_base = np.asarray(quantities["regret_base"], dtype=np.float64)[index]
    reg_strongest = np.asarray(quantities["regret_strongest_fallback"], dtype=np.float64)[index]

    mean_d2r = float(reg_d2r.mean())
    mean_idle = float(reg_idle.mean())
    mean_base = float(reg_base.mean())
    mean_strongest = float(reg_strongest.mean())

    q = np.full(N_PHASE0_QUANTITIES, np.nan, dtype=np.float64)
    q[Q_MEAN_D2R] = mean_d2r
    q[Q_MEAN_IDLE] = mean_idle
    q[Q_MEAN_BASE] = mean_base
    q[Q_MEAN_STRONGEST] = mean_strongest
    q[Q_I_IDLE] = P.relative_improvement(mean_idle, mean_d2r)
    q[Q_I_BASE] = P.relative_improvement(mean_base, mean_d2r)
    q[Q_I_STANDARD] = P.relative_improvement(mean_strongest, mean_d2r)
    # coverage fractions are properties of the frozen rule; resampled to reflect group mix.
    chosen = np.asarray(quantities["chosen_d2r"], dtype=np.int64)[index]
    a0 = np.asarray(quantities["a0"], dtype=np.int64)[index]
    risk = np.asarray(quantities["risk_d2r"], dtype=np.float64)[index]
    tau = float(quantities["tau_d2r"])
    above = risk > tau
    q[Q_FRAC_CHANGED] = float((above & (a0 != P.IDLE_PLAN_INDEX)).mean())
    q[Q_FRAC_NON_IDLE] = float((chosen != P.IDLE_PLAN_INDEX).mean())
    q[Q_FRAC_ABOVE] = float(above.mean())
    return q


def evaluate_phase0_gate(quantities: Mapping[str, Any], bootstrap: PairedBootstrap
                         ) -> dict[str, Any]:
    """Apply the plan-4 Go/No-Go conditions exactly once (fail-closed, development cost screen).

    ALL must hold to continue: both zones >= 200 independent scenarios; both zone paired
    difference point estimates > 0; D2R relative improvements over ALWAYS_IDLE / ALWAYS_BASE /
    the frozen strongest standard fallback all >= 0.05 (point estimates); the actually-changed
    fraction >= 0.10 and the final-non-idle fraction >= 0.20. Any failure is the terminal
    ``D2R_POLICY_SIGNAL_NOT_FEASIBLE`` (no new collection, no threshold retune, no fallback
    search). NaN improvements fail by construction.
    """
    groups = _validate_phase0(quantities)
    point = np.asarray(bootstrap.point, dtype=np.float64)
    if point.shape != (N_PHASE0_QUANTITIES,):
        raise ValueError("phase0 bootstrap point does not match the quantity vector")

    n_high = int(quantities["n_high_zone"])
    n_low = int(quantities["n_low_zone"])
    diff_high = float(quantities["diff_high_zone"])   # E[regret(a0)-regret(IDLE) | H]
    diff_low = float(quantities["diff_low_zone"])     # E[regret(IDLE)-regret(a0) | L]
    frac_changed = float(point[Q_FRAC_CHANGED])
    frac_non_idle = float(point[Q_FRAC_NON_IDLE])

    conditions = {
        "zones_each_ge_200": bool(n_high >= MIN_ZONE_GROUPS and n_low >= MIN_ZONE_GROUPS),
        "high_zone_base_worse_than_idle_point_positive": bool(
            np.isfinite(diff_high) and diff_high > 0.0),
        "low_zone_idle_worse_than_base_point_positive": bool(
            np.isfinite(diff_low) and diff_low > 0.0),
        "i_idle_point_ge_0.05": bool(point[Q_I_IDLE] >= MIN_RELATIVE_IMPROVEMENT),
        "i_base_point_ge_0.05": bool(point[Q_I_BASE] >= MIN_RELATIVE_IMPROVEMENT),
        "i_standard_point_ge_0.05": bool(point[Q_I_STANDARD] >= MIN_RELATIVE_IMPROVEMENT),
        "frac_changed_ge_0.10": bool(frac_changed >= MIN_FRAC_CHANGED),
        "frac_non_idle_ge_0.20": bool(frac_non_idle >= MIN_FRAC_NON_IDLE),
    }
    passed = bool(all(conditions.values()))
    return {
        "plan_id": P.PLAN_ID,
        "scope": "ADAPTIVE_DEVELOPMENT_ONLY",
        "n_groups": int(groups),
        "conditions": conditions,
        "passed": passed,
        "status": STATUS_PHASE0_FEASIBLE if passed else STATUS_POLICY_SIGNAL_NOT_FEASIBLE,
        "values": {
            "n_high_zone": n_high,
            "n_low_zone": n_low,
            "diff_high_zone": diff_high,
            "diff_low_zone": diff_low,
            "frac_changed_action": frac_changed,
            "frac_final_non_idle": frac_non_idle,
            "frac_above_threshold": float(point[Q_FRAC_ABOVE]),
        },
        "point": {name: float(point[i]) for i, name in enumerate(_PHASE0_QUANTITY_NAMES)},
        "ci80_lower": {name: float(bootstrap.ci_lower[i])
                       for i, name in enumerate(_PHASE0_QUANTITY_NAMES)},
        "ci80_upper": {name: float(bootstrap.ci_upper[i])
                       for i, name in enumerate(_PHASE0_QUANTITY_NAMES)},
        "bootstrap": {
            "seed": int(bootstrap.seed),
            "n_resamples": int(bootstrap.n_resamples),
            "confidence": float(bootstrap.confidence),
            "note": ("development cost screen on the adaptively-used validation role; NOT an "
                     "unbiased confirmation CI (plan 2.2 / 4)"),
        },
    }


# ===========================================================================
# Plan 7: the unique development-screen adjudication gate.
#
# Built once and reused verbatim for (a) the plan-5 power pre-check on the OLD
# validation role and (b) the one-shot NEW screen adjudication — the SAME
# quantity vector + SAME AND conditions can never drift between the two. The
# screen bootstrap uses the frozen seed ``2026122403`` / 10,000 resamples /
# two-sided 80% CI. The policy thresholds (``tau_D``, per-seed, per-fallback)
# are point constants fit on FULL validation; the bootstrap NEVER re-fits them,
# it only re-averages the frozen-rule executed regrets over resampled groups.
#
# The statistical unit is the 10-action decision GROUP. ``R_X(g) = J_true(g,a_X)
# - min_a J_true(g,a)`` is the EXECUTED-action regret; ``I_B = (mean R_B - mean
# R_D2R) / mean R_B`` for B in {ALWAYS_IDLE, ALWAYS_BASE, STRONGEST_STANDARD_FALLBACK}.
# The three denominators are FAIL-CLOSED: a non-positive / non-finite one is
# ``D2R_INVALID_ANALYSIS`` — never an epsilon, never an absolute difference.
#
# This module is torch-free: it consumes per-group arrays already aligned on
# the group axis (assembled read-only by the driver), so the one-shot ordering
# is testable on synthetic data.
# ===========================================================================

#: z for the two-sided 80% CI lower bound (standard-normal ppf(0.90)); the screen NEVER uses 95%.
SCREEN_Z = 1.2815515655446004
N_MODEL_SEEDS = len(T.MODEL_SEEDS)               # 5
N_FALLBACKS = len(P.FALLBACK_BASELINES)          # 3

# --- plan-7 gate thresholds (section 7) ---------------------------------------------------------
MIN_IMPROVEMENT = 0.05                            # I_IDLE / I_BASE / I_STANDARD point line
MIN_POSITIVE_SEEDS = 4                            # >=4/5 seeds with I_IDLE and I_STANDARD positive
HIGH_REGRET_THRESHOLD = T.HIGH_REGRET_THRESHOLD   # 0.01
MIN_HIGH_REGRET_FRAC = 0.15
MIN_DISTINCT_ORACLE = 3
MIN_ORACLE_FRAC = 0.05
MIN_REACTIVE_FRAC = 0.20
MIN_STRATUM_FRAC = 0.10
TSTAR_SPLIT = 13                                  # frozen [t_star_low, t_star_high) split

# --- plan-7 terminal states (section 7 终态优先级) ----------------------------------------------
STATUS_INVALID_ANALYSIS = "D2R_INVALID_ANALYSIS"
STATUS_DEV_NOT_READY = "D2R_DEV_NOT_READY"
STATUS_DEV_READY = "D2R_DEV_READY_FOR_INDEPENDENT_CONFIRMATION"

# --- frozen screen quantity-vector layout (N_SCREEN_QUANTITIES = 40) -----------------------------
S_MEAN_D2R = 0
S_MEAN_IDLE = 1
S_MEAN_BASE = 2
S_MEAN_STRONGEST = 3
S_I_IDLE = 4
S_I_BASE = 5
S_I_STANDARD = 6
S_FRAC_CHANGED = 7
S_FRAC_NON_IDLE = 8
S_FRAC_ABOVE = 9
S_HIGH_REGRET_FRAC = 10
S_SPEARMAN_BRANCH = 11
S_REACTIVE_FRAC = 12
S_TSTAR_LOW = 13
S_TSTAR_HIGH = 14
S_DIST_LOW = 15
S_DIST_HIGH = 16
S_ORACLE = 17                                     # 17..26 (10 oracle action shares)
S_SEED_I_IDLE = 27                                # 27..31 (5 per-seed I_IDLE)
S_SEED_I_STANDARD = 32                            # 32..36 (5 per-seed I_STANDARD)
S_FB_IMPROVEMENT = 37                             # 37..39 (3 per-fallback improvement, reporting)
N_SCREEN_QUANTITIES = 40

SEED_I_IDLE_INDEX = tuple(range(S_SEED_I_IDLE, S_SEED_I_IDLE + N_MODEL_SEEDS))
SEED_I_STANDARD_INDEX = tuple(range(S_SEED_I_STANDARD, S_SEED_I_STANDARD + N_MODEL_SEEDS))
ORACLE_INDEX = tuple(range(S_ORACLE, S_ORACLE + P.N_ACTIONS))
STRATA_INDEX = (S_TSTAR_LOW, S_TSTAR_HIGH, S_DIST_LOW, S_DIST_HIGH)
FALLBACK_INDEX = tuple(range(S_FB_IMPROVEMENT, S_FB_IMPROVEMENT + N_FALLBACKS))

_SCREEN_QUANTITY_NAMES = (
    "mean_regret_d2r", "mean_regret_idle", "mean_regret_base", "mean_regret_strongest_fallback",
    "i_idle", "i_base", "i_standard",
    "frac_changed_action", "frac_final_non_idle", "frac_above_threshold",
    "high_regret_frac", "spearman_branch", "reactive_frac",
    "tstar_low_frac", "tstar_high_frac", "dist_low_frac", "dist_high_frac",
) + tuple(f"oracle_share_a{a}" for a in range(P.N_ACTIONS)) \
    + tuple(f"seed{s}_i_idle" for s in T.MODEL_SEEDS) \
    + tuple(f"seed{s}_i_standard" for s in T.MODEL_SEEDS) \
    + tuple(f"{name.lower()}_improvement" for name in P.FALLBACK_BASELINES)

#: per-group arrays the screen quantities mapping must carry (aligned on the group axis). The
#: per-seed / per-fallback executed regrets are precomputed ONCE with the frozen thresholds so a
#: bootstrap draw only re-averages them (never re-fits a threshold, never re-runs a model).
REQUIRED_SCREEN_ARRAY_KEYS = (
    "regret_d2r", "regret_idle", "regret_base", "regret_strongest_fallback",
    "regret_by_seed", "regret_by_fallback",
    "chosen_d2r", "a0", "risk_d2r", "j_hat", "j_true", "a_star", "reactive",
    "tstar_layer", "interaction_layer",
)


def _validate_screen(quantities: Mapping[str, Any]) -> int:
    """Fail closed on any missing/misaligned/non-finite per-group screen quantity."""
    for key in REQUIRED_SCREEN_ARRAY_KEYS:
        if key not in quantities:
            raise ValueError(f"screen quantities missing array {key!r}")
    reg = np.asarray(quantities["regret_d2r"], dtype=np.float64).ravel()
    groups = int(reg.size)
    if groups < 2:
        raise ValueError("screen quantities need at least 2 groups")
    for key in ("regret_d2r", "regret_idle", "regret_base", "regret_strongest_fallback"):
        arr = np.asarray(quantities[key], dtype=np.float64).ravel()
        if arr.shape != (groups,) or not np.isfinite(arr).all():
            raise ValueError(f"screen array {key!r} must be finite and align as ({groups},)")
    for key in ("chosen_d2r", "a0", "a_star", "tstar_layer", "interaction_layer"):
        if np.asarray(quantities[key]).ravel().shape != (groups,):
            raise ValueError(f"screen array {key!r} must align as ({groups},)")
    if np.asarray(quantities["risk_d2r"], dtype=np.float64).ravel().shape != (groups,):
        raise ValueError("screen array 'risk_d2r' must align on the group axis")
    if np.asarray(quantities["reactive"]).ravel().shape != (groups,):
        raise ValueError("screen array 'reactive' must align on the group axis")
    by_seed = np.asarray(quantities["regret_by_seed"], dtype=np.float64)
    if by_seed.shape != (N_MODEL_SEEDS, groups) or not np.isfinite(by_seed).all():
        raise ValueError(f"regret_by_seed must be finite and align as ({N_MODEL_SEEDS},{groups})")
    by_fb = np.asarray(quantities["regret_by_fallback"], dtype=np.float64)
    if by_fb.shape != (N_FALLBACKS, groups) or not np.isfinite(by_fb).all():
        raise ValueError(f"regret_by_fallback must be finite and align as ({N_FALLBACKS},{groups})")
    for key in ("j_hat", "j_true"):
        arr = np.asarray(quantities[key], dtype=np.float64)
        if arr.shape != (groups, P.N_ACTIONS) or not np.isfinite(arr).all():
            raise ValueError(f"screen array {key!r} must be finite and align as ({groups},10)")
    if "tau_d2r" not in quantities or not np.isfinite(float(quantities["tau_d2r"])):
        raise ValueError("screen quantities missing a finite scalar 'tau_d2r'")
    return groups


def compute_screen_quantity_vector(index: np.ndarray, quantities: Mapping[str, Any]
                                   ) -> np.ndarray:
    """Evaluate all 40 screen quantities on ONE whole-group resample ``index``.

    Every quantity of the draw is recomputed from the SAME with-replacement group index vector,
    so they stay coupled through the same group multiplicities. The frozen thresholds are point
    constants (already applied upstream to build ``regret_*``); a draw only re-averages. Improvement
    quantities are NaN when a denominator mean is non-positive (the gate turns that into
    ``D2R_INVALID_ANALYSIS``).
    """
    index = np.asarray(index, dtype=np.int64)
    reg_d2r = np.asarray(quantities["regret_d2r"], dtype=np.float64)[index]
    reg_idle = np.asarray(quantities["regret_idle"], dtype=np.float64)[index]
    reg_base = np.asarray(quantities["regret_base"], dtype=np.float64)[index]
    reg_strong = np.asarray(quantities["regret_strongest_fallback"], dtype=np.float64)[index]
    by_seed = np.asarray(quantities["regret_by_seed"], dtype=np.float64)[:, index]
    by_fb = np.asarray(quantities["regret_by_fallback"], dtype=np.float64)[:, index]

    mean_d2r = float(reg_d2r.mean())
    mean_idle = float(reg_idle.mean())
    mean_base = float(reg_base.mean())
    mean_strong = float(reg_strong.mean())

    q = np.full(N_SCREEN_QUANTITIES, np.nan, dtype=np.float64)
    q[S_MEAN_D2R] = mean_d2r
    q[S_MEAN_IDLE] = mean_idle
    q[S_MEAN_BASE] = mean_base
    q[S_MEAN_STRONGEST] = mean_strong
    q[S_I_IDLE] = P.relative_improvement(mean_idle, mean_d2r)
    q[S_I_BASE] = P.relative_improvement(mean_base, mean_d2r)
    q[S_I_STANDARD] = P.relative_improvement(mean_strong, mean_d2r)

    chosen = np.asarray(quantities["chosen_d2r"], dtype=np.int64)[index]
    a0 = np.asarray(quantities["a0"], dtype=np.int64)[index]
    risk = np.asarray(quantities["risk_d2r"], dtype=np.float64)[index]
    tau = float(quantities["tau_d2r"])
    above = risk > tau
    q[S_FRAC_CHANGED] = float((above & (a0 != P.IDLE_PLAN_INDEX)).mean())
    q[S_FRAC_NON_IDLE] = float((chosen != P.IDLE_PLAN_INDEX).mean())
    q[S_FRAC_ABOVE] = float(above.mean())

    q[S_HIGH_REGRET_FRAC] = float((reg_base > HIGH_REGRET_THRESHOLD).mean())
    q[S_SPEARMAN_BRANCH] = s4cs.spearman(
        np.asarray(quantities["j_hat"], dtype=np.float64)[index],
        np.asarray(quantities["j_true"], dtype=np.float64)[index])
    q[S_REACTIVE_FRAC] = float(np.asarray(quantities["reactive"], dtype=bool)[index].mean())
    tstar = np.asarray(quantities["tstar_layer"], dtype=np.int64)[index]
    inter = np.asarray(quantities["interaction_layer"], dtype=np.int64)[index]
    q[S_TSTAR_LOW] = float((tstar == 0).mean())
    q[S_TSTAR_HIGH] = float((tstar == 1).mean())
    q[S_DIST_LOW] = float((inter == 0).mean())
    q[S_DIST_HIGH] = float((inter == 1).mean())

    a_star = np.asarray(quantities["a_star"], dtype=np.int64)[index]
    for a in range(P.N_ACTIONS):
        q[S_ORACLE + a] = float((a_star == a).mean())

    for i in range(N_MODEL_SEEDS):
        mean_seed = float(by_seed[i].mean())
        q[S_SEED_I_IDLE + i] = P.relative_improvement(mean_idle, mean_seed)
        q[S_SEED_I_STANDARD + i] = P.relative_improvement(mean_strong, mean_seed)

    for bi in range(N_FALLBACKS):
        q[S_FB_IMPROVEMENT + bi] = P.relative_improvement(float(by_fb[bi].mean()), mean_d2r)
    return q


# --- the single adjudication gate (plan 7 AND conditions + terminal priority) ----------------------


def _screen_named(vector: np.ndarray) -> dict[str, float]:
    return {name: float(vector[i]) for i, name in enumerate(_SCREEN_QUANTITY_NAMES)}


def evaluate_screen_gates(quantities: Mapping[str, Any], bootstrap: PairedBootstrap,
                          *, n_expected: int) -> dict[str, Any]:
    """Adjudicate the plan-7 main AND-gate exactly once (fail-closed).

    Condition 1 (data/action/map/lock/truth isolation Gate Q, n/n completeness, zero new/old ID
    overlap, no outcome-based rejection) is structural and arrives via the ``qa`` mapping +
    ``n_expected``; conditions 2-7 are the statistical AND-chain read off the paired bootstrap.
    Denominators are fail-closed (mean R_B finite and strictly >0 for all three B). Terminal
    priority: a structural/QA failure or a degenerate denominator / projection / mask is
    ``D2R_INVALID_ANALYSIS`` (an invalid test is never a method failure); a valid screen with any
    main condition failed is ``D2R_DEV_NOT_READY``; all pass is
    ``D2R_DEV_READY_FOR_INDEPENDENT_CONFIRMATION`` (which only permits a SEPARATE, separately
    approved confirm plan — this plan never collects confirm).
    """
    groups = _validate_screen(quantities)
    point = np.asarray(bootstrap.point, dtype=np.float64)
    ci_lower = np.asarray(bootstrap.ci_lower, dtype=np.float64)
    ci_upper = np.asarray(bootstrap.ci_upper, dtype=np.float64)
    if point.shape != (N_SCREEN_QUANTITIES,) or ci_lower.shape != (N_SCREEN_QUANTITIES,) \
            or ci_upper.shape != (N_SCREEN_QUANTITIES,):
        raise ValueError("screen bootstrap does not match the frozen quantity vector")

    qa = dict(quantities.get("qa", {}))
    qa_passed = bool(qa) and all(bool(v) for v in qa.values()) and groups == int(n_expected)
    denominators_valid = bool(
        np.isfinite([point[S_MEAN_IDLE], point[S_MEAN_BASE], point[S_MEAN_STRONGEST]]).all()
        and point[S_MEAN_IDLE] > 0.0 and point[S_MEAN_BASE] > 0.0
        and point[S_MEAN_STRONGEST] > 0.0)
    projection_ok = bool(quantities.get("projection_nondegenerate", False))
    mask_ok = bool(quantities.get("eval_mask_nondegenerate", False))

    oracle_shares = [float(point[S_ORACLE + a]) for a in range(P.N_ACTIONS)]
    n_oracle = int(sum(s >= MIN_ORACLE_FRAC for s in oracle_shares))
    strata_min = float(min(point[i] for i in STRATA_INDEX))
    seeds_both = int(sum(
        point[SEED_I_IDLE_INDEX[i]] > 0.0 and point[SEED_I_STANDARD_INDEX[i]] > 0.0
        for i in range(N_MODEL_SEEDS)))

    conditions = {
        "c1_integrity_qa_and_completeness": qa_passed,
        "c2_coverage": bool(point[S_FRAC_CHANGED] >= MIN_FRAC_CHANGED
                            and point[S_FRAC_NON_IDLE] >= MIN_FRAC_NON_IDLE),
        "c3_i_idle": bool(point[S_I_IDLE] >= MIN_IMPROVEMENT and ci_lower[S_I_IDLE] > 0.0),
        "c4_i_base": bool(point[S_I_BASE] >= MIN_IMPROVEMENT and ci_lower[S_I_BASE] > 0.0),
        "c5_i_standard": bool(point[S_I_STANDARD] >= MIN_IMPROVEMENT
                              and ci_lower[S_I_STANDARD] > 0.0),
        "c6_seeds_positive": bool(seeds_both >= MIN_POSITIVE_SEEDS),
        "c7_base_selector_task_support": bool(
            point[S_HIGH_REGRET_FRAC] >= MIN_HIGH_REGRET_FRAC
            and n_oracle >= MIN_DISTINCT_ORACLE
            and ci_lower[S_SPEARMAN_BRANCH] > 0.0
            and point[S_REACTIVE_FRAC] >= MIN_REACTIVE_FRAC
            and strata_min >= MIN_STRATUM_FRAC
            and denominators_valid and projection_ok and mask_ok),
    }
    invalid = (not qa_passed) or (not denominators_valid) \
        or (not projection_ok) or (not mask_ok)
    if invalid:
        status = STATUS_INVALID_ANALYSIS
    elif not all(conditions.values()):
        status = STATUS_DEV_NOT_READY
    else:
        status = STATUS_DEV_READY

    # all three simple fallbacks reported with point + 80% CI (plan 7 condition 5 note); the
    # validation-frozen strongest fallback is never substituted even if the screen ranks differ.
    fallback_curves = {
        name: {
            "improvement_over_d2r_denominator": float(point[S_FB_IMPROVEMENT + bi]),
            "improvement_ci80": [float(ci_lower[S_FB_IMPROVEMENT + bi]),
                                 float(ci_upper[S_FB_IMPROVEMENT + bi])],
        }
        for bi, name in enumerate(P.FALLBACK_BASELINES)
    }

    return {
        "plan_id": P.PLAN_ID,
        "n_groups": int(groups),
        "n_expected": int(n_expected),
        "status": status,
        "passed": bool(status == STATUS_DEV_READY),
        "conditions": conditions,
        "denominators_valid": denominators_valid,
        "projection_nondegenerate": projection_ok,
        "eval_mask_nondegenerate": mask_ok,
        "qa": qa,
        "values": {
            "oracle_action_shares": oracle_shares,
            "oracle_actions_covered_ge_5pct": n_oracle,
            "strata_min_frac": strata_min,
            "seeds_i_idle_and_i_standard_positive": seeds_both,
            "strongest_standard_fallback": str(quantities.get("strongest_fallback", "")),
            "fallback_curves": fallback_curves,
        },
        "point": _screen_named(point),
        "ci80_lower": _screen_named(ci_lower),
        "ci80_upper": _screen_named(ci_upper),
        "bootstrap": {
            "seed": int(bootstrap.seed),
            "n_resamples": int(bootstrap.n_resamples),
            "confidence": float(bootstrap.confidence),
            "design": ("whole-group paired bootstrap; every quantity of a draw recomputed from "
                       "the SAME resample index vector; two-sided 80% CI from the resample "
                       "quantiles; frozen thresholds never re-fit"),
        },
    }


# --- the plan-5 power AND-chain (vectorised over joint Monte-Carlo draws) --------------------------

#: gate-relevant statistical subset of the 40-vector, in order. The absolute means / frac_above /
#: per-fallback reporting slots are excluded (they only derive the improvements already modelled);
#: the structural condition 1 / denominator / projection / mask are NOT statistical and the power
#: estimate is conditional on a structurally valid screen.
POWER_SLOTS = (
    S_I_IDLE, S_I_BASE, S_I_STANDARD,
    S_FRAC_CHANGED, S_FRAC_NON_IDLE,
    S_HIGH_REGRET_FRAC, S_SPEARMAN_BRANCH, S_REACTIVE_FRAC,
    S_TSTAR_LOW, S_TSTAR_HIGH, S_DIST_LOW, S_DIST_HIGH,
) + ORACLE_INDEX + SEED_I_IDLE_INDEX + SEED_I_STANDARD_INDEX
N_POWER_QUANTITIES = len(POWER_SLOTS)             # 32

# Local indices inside the power subset (lists: numpy fancy-indexing requires them).
P_I_IDLE = 0
P_I_BASE = 1
P_I_STANDARD = 2
P_FRAC_CHANGED = 3
P_FRAC_NON_IDLE = 4
P_HIGH_REGRET = 5
P_SPEARMAN = 6
P_REACTIVE = 7
P_STRATA = [8, 9, 10, 11]
P_ORACLE = list(range(12, 12 + P.N_ACTIONS))       # 12..21
P_SEED_I_IDLE = list(range(22, 22 + N_MODEL_SEEDS))      # 22..26
P_SEED_I_STANDARD = list(range(27, 27 + N_MODEL_SEEDS))  # 27..31

POWER_QUANTITY_NAMES = tuple(_SCREEN_QUANTITY_NAMES[i] for i in POWER_SLOTS)


def evaluate_screen_draws(draws: np.ndarray, standard_errors: np.ndarray) -> dict[str, Any]:
    """Vectorised plan-7 statistical AND-chain over joint Monte-Carlo draws of the power subset.

    ``draws`` is ``(replicates, 32)``; ``standard_errors`` is ``sqrt(diag(cov_n))`` for the same
    subset. CI conditions use the screen-side 80% bound (``estimate - SCREEN_Z*SE > 0``) computed
    from the draw covariance; fraction / oracle / strata / seed-count conditions use point values,
    exactly mirroring :func:`evaluate_screen_gates`. The 5% point-improvement line enters every
    draw (plan 5: the 5% 实质线 must be simulated, never rescued by "larger n").
    """
    draws = np.asarray(draws, dtype=np.float64)
    se = np.asarray(standard_errors, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[1] != N_POWER_QUANTITIES:
        raise ValueError(f"draws must align as (replicates, {N_POWER_QUANTITIES})")
    if se.shape != (N_POWER_QUANTITIES,):
        raise ValueError("standard_errors must align with the power quantity subset")

    def ci80_lower(slot: int) -> np.ndarray:
        return draws[:, slot] - SCREEN_Z * se[slot]

    seeds_both = ((draws[:, P_SEED_I_IDLE] > 0.0)
                  & (draws[:, P_SEED_I_STANDARD] > 0.0)).sum(axis=1)
    oracle_count = (draws[:, P_ORACLE] >= MIN_ORACLE_FRAC).sum(axis=1)
    strata_min = draws[:, P_STRATA].min(axis=1)
    conditions = {
        "frac_changed": draws[:, P_FRAC_CHANGED] >= MIN_FRAC_CHANGED,
        "frac_non_idle": draws[:, P_FRAC_NON_IDLE] >= MIN_FRAC_NON_IDLE,
        "i_idle": (draws[:, P_I_IDLE] >= MIN_IMPROVEMENT) & (ci80_lower(P_I_IDLE) > 0.0),
        "i_base": (draws[:, P_I_BASE] >= MIN_IMPROVEMENT) & (ci80_lower(P_I_BASE) > 0.0),
        "i_standard": (draws[:, P_I_STANDARD] >= MIN_IMPROVEMENT)
                      & (ci80_lower(P_I_STANDARD) > 0.0),
        "seeds_i_idle_and_i_standard_positive": seeds_both >= MIN_POSITIVE_SEEDS,
        "high_regret_frac": draws[:, P_HIGH_REGRET] >= MIN_HIGH_REGRET_FRAC,
        "oracle_coverage": oracle_count >= MIN_DISTINCT_ORACLE,
        "spearman_ci_lower_positive": ci80_lower(P_SPEARMAN) > 0.0,
        "reactive_frac": draws[:, P_REACTIVE] >= MIN_REACTIVE_FRAC,
        "strata_floor": strata_min >= MIN_STRATUM_FRAC,
    }
    passed = np.logical_and.reduce(list(conditions.values()))
    return {"passed": passed, "conditions": conditions}


__all__ = [
    "CONFIDENCE",
    "FALLBACK_INDEX",
    "HIGH_REGRET_THRESHOLD",
    "MIN_DISTINCT_ORACLE",
    "MIN_FRAC_CHANGED",
    "MIN_FRAC_NON_IDLE",
    "MIN_HIGH_REGRET_FRAC",
    "MIN_IMPROVEMENT",
    "MIN_ORACLE_FRAC",
    "MIN_POSITIVE_SEEDS",
    "MIN_REACTIVE_FRAC",
    "MIN_RELATIVE_IMPROVEMENT",
    "MIN_STRATUM_FRAC",
    "MIN_ZONE_GROUPS",
    "N_FALLBACKS",
    "N_MODEL_SEEDS",
    "N_PHASE0_QUANTITIES",
    "N_POWER_QUANTITIES",
    "N_SCREEN_QUANTITIES",
    "ORACLE_INDEX",
    "PHASE0_BOOTSTRAP_SEED",
    "PHASE0_RESAMPLES",
    "POWER_QUANTITY_NAMES",
    "POWER_SLOTS",
    "P_FRAC_CHANGED",
    "P_FRAC_NON_IDLE",
    "P_HIGH_REGRET",
    "P_I_BASE",
    "P_I_IDLE",
    "P_I_STANDARD",
    "P_ORACLE",
    "P_REACTIVE",
    "P_SEED_I_IDLE",
    "P_SEED_I_STANDARD",
    "P_SPEARMAN",
    "P_STRATA",
    "PairedBootstrap",
    "REQUIRED_SCREEN_ARRAY_KEYS",
    "SCREEN_BOOTSTRAP_SEED",
    "SCREEN_RESAMPLES",
    "SCREEN_Z",
    "SEED_I_IDLE_INDEX",
    "SEED_I_STANDARD_INDEX",
    "STATUS_DEV_NOT_READY",
    "STATUS_DEV_READY",
    "STATUS_INVALID_ANALYSIS",
    "STATUS_PHASE0_FEASIBLE",
    "STATUS_POLICY_SIGNAL_NOT_FEASIBLE",
    "STRATA_INDEX",
    "S_DIST_HIGH",
    "S_DIST_LOW",
    "S_FB_IMPROVEMENT",
    "S_FRAC_ABOVE",
    "S_FRAC_CHANGED",
    "S_FRAC_NON_IDLE",
    "S_HIGH_REGRET_FRAC",
    "S_I_BASE",
    "S_I_IDLE",
    "S_I_STANDARD",
    "S_MEAN_BASE",
    "S_MEAN_D2R",
    "S_MEAN_IDLE",
    "S_MEAN_STRONGEST",
    "S_ORACLE",
    "S_REACTIVE_FRAC",
    "S_SEED_I_IDLE",
    "S_SEED_I_STANDARD",
    "S_SPEARMAN_BRANCH",
    "S_TSTAR_HIGH",
    "S_TSTAR_LOW",
    "TSTAR_SPLIT",
    "compute_phase0_quantity_vector",
    "compute_screen_quantity_vector",
    "evaluate_phase0_gate",
    "evaluate_screen_draws",
    "evaluate_screen_gates",
    "paired_bootstrap",
]
