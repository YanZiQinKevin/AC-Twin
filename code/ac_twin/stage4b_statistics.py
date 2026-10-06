"""Stage 4B confirmatory statistics and the Gate RISK_confirm (plan section 7 / 8).

The statistical unit is always the episode. The confirm adjudication runs ONE
10,000-fold episode-level paired bootstrap (frozen entropy ``2026091703``); the SAME
bootstrap draw resamples the candidate, every baseline, the five per-pair errors and
the stratification labels together. The 17-quantity vector is reused verbatim from
``stage4b_power.compute_quantity_vector`` so the development-screen power model and
the confirmatory adjudication share one definition bit-for-bit.

Intervals are two-sided 95% (plan section 7.3). The main Gate is a pre-registered AND
intersection-union裁判 (plan section 8.2); no single significant metric may substitute
for整体通过. SE / covariance never come from the Stage 4A 80% CI widths.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ac_twin import stage4b_power as s4bpw
from ac_twin import stage4b_protocol as s4bp


def confirm_quantity_point(inputs: Mapping[str, Any]) -> np.ndarray:
    """17-quantity point vector on the FULL confirm set (identity index)."""
    episodes = inputs["risk_target"].size
    return s4bpw.compute_quantity_vector(
        np.arange(episodes, dtype=np.int64),
        inputs["comb_score"], inputs["geo_score"], inputs["unc_score"],
        inputs["risk_target"], inputs["primary_error_by_pair"],
        inputs["cf_action"], inputs["cf_duration"], inputs["top_error_threshold"],
    )


def confirm_paired_bootstrap(
    inputs: Mapping[str, Any],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """One paired bootstrap over the confirm episodes -> point + 95% CI lower.

    Every quantity is recomputed on the same resampled episodes each draw (paired),
    preserving repeated-episode multiplicity. Returns the point vector, the per-draw
    matrix and the two-sided 95% lower quantile for each quantity.
    """
    generator = np.random.default_rng(int(seed))
    episodes = inputs["risk_target"].size
    draws = np.empty((int(n_resamples), s4bpw.N_QUANTITIES), dtype=np.float64)
    for draw in range(int(n_resamples)):
        index = generator.integers(0, episodes, size=episodes)
        draws[draw] = s4bpw.compute_quantity_vector(
            index,
            inputs["comb_score"], inputs["geo_score"], inputs["unc_score"],
            inputs["risk_target"], inputs["primary_error_by_pair"],
            inputs["cf_action"], inputs["cf_duration"], inputs["top_error_threshold"],
        )
    if not np.isfinite(draws).all():
        raise ValueError("non-finite confirm bootstrap quantity encountered")
    point = confirm_quantity_point(inputs)
    ci_lower = np.quantile(draws, 0.025, axis=0)
    ci_upper = np.quantile(draws, 0.975, axis=0)
    return {
        "point": point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "draws": draws,
        "n_resamples": int(n_resamples),
        "bootstrap_seed": int(seed),
        "confidence_level": s4bp.CONFIDENCE_LEVEL,
    }


def evaluate_gate_risk_confirm(
    point: np.ndarray,
    ci_lower: np.ndarray,
    *,
    qa_passed: bool,
) -> dict[str, Any]:
    """Apply the eight pre-registered AND conditions (plan section 8.2).

    ``point`` / ``ci_lower`` are the 17-quantity point vector and its 95% lower
    confidence bound. ``qa_passed`` is Gate Q (section 8.1). Returns the per-condition
    verdict and the mechanical terminal state.
    """
    point = np.asarray(point, dtype=np.float64)
    ci_lower = np.asarray(ci_lower, dtype=np.float64)
    pair_positive = int((point[s4bpw.PAIR_INDEX] > 0.0).sum())
    strata_slots = list(s4bpw.ACTION_INDEX.values()) + list(s4bpw.DURATION_INDEX.values())
    strata_min = float(point[strata_slots].min())

    conditions = {
        "i_aurc_vs_uncertainty": bool(
            point[s4bpw.Q_I_AURC_UNC] >= s4bp.GATE_I_AURC_MIN
            and ci_lower[s4bpw.Q_I_AURC_UNC] > 0.0
        ),
        "i_c80_vs_uncertainty": bool(
            point[s4bpw.Q_I_C80_UNC] >= s4bp.GATE_I_C80_MIN
            and ci_lower[s4bpw.Q_I_C80_UNC] > 0.0
        ),
        "i_aurc_vs_geometry": bool(
            point[s4bpw.Q_I_AURC_GEO] > 0.0 and ci_lower[s4bpw.Q_I_AURC_GEO] > 0.0
        ),
        "i_c80_vs_geometry": bool(
            point[s4bpw.Q_I_C80_GEO] > 0.0 and ci_lower[s4bpw.Q_I_C80_GEO] > 0.0
        ),
        "pr_auc_minus_prevalence": bool(
            point[s4bpw.Q_PR] > 0.0 and ci_lower[s4bpw.Q_PR] > 0.0
        ),
        "at_least_4_of_5_pairs_positive": bool(pair_positive >= s4bp.GATE_MIN_POSITIVE_PAIRS),
        "no_strata_below_floor": bool(strata_min >= s4bp.GATE_STRATA_I_AURC_FLOOR),
        "gate_q_passed": bool(qa_passed),
    }
    passed = all(conditions.values())
    if not qa_passed:
        status = s4bp.STATUS_INVALID
    elif passed:
        status = s4bp.STATUS_CONFIRMED
    else:
        status = s4bp.STATUS_NOT_CONFIRMED
    return {
        "conditions": conditions,
        "positive_pairs": pair_positive,
        "strata_min_i_aurc": strata_min,
        "passed": bool(passed),
        "gate_q_passed": bool(qa_passed),
        "status": status,
    }


__all__ = [
    "confirm_paired_bootstrap",
    "confirm_quantity_point",
    "evaluate_gate_risk_confirm",
]
