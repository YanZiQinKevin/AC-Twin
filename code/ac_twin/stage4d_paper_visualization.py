"""Pure, non-adjudicating selection and coordinate helpers for D2F paper images.

No model fitting or Gate logic belongs here. The confirm table is read only after
the original one-shot adjudication has completed; these helpers never write to it.
"""

from __future__ import annotations

import numpy as np


def _percentiles(values: np.ndarray) -> np.ndarray:
    """Empirical upper-rank percentiles; tied scores have the same rank."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("risk scores must be finite one-dimensional arrays")
    return np.searchsorted(np.sort(values), values, side="right") / len(values)


def select_cases(
    regret: np.ndarray,
    direct_risk: np.ndarray,
    baseline_risk: np.ndarray,
    a_hat: np.ndarray,
    a_star: np.ndarray,
    ids: list[str],
) -> dict:
    """Apply frozen D2 §14.2 success/failure rules to an adjudicated confirm table."""
    regret = np.asarray(regret, dtype=np.float64)
    direct_risk = np.asarray(direct_risk, dtype=np.float64)
    baseline_risk = np.asarray(baseline_risk, dtype=np.float64)
    a_hat = np.asarray(a_hat)
    a_star = np.asarray(a_star)
    n = len(regret)
    if n == 0 or any(len(x) != n for x in (direct_risk, baseline_risk, a_hat, a_star, ids)):
        raise ValueError("case fields must be aligned and nonempty")
    if not (np.isfinite(regret).all() and np.isfinite(direct_risk).all()
            and np.isfinite(baseline_risk).all()):
        raise ValueError("case fields must be finite")
    d_pct = _percentiles(direct_risk)
    b_pct = _percentiles(baseline_risk)
    d_top = direct_risk >= np.quantile(direct_risk, .8)
    b_top = baseline_risk >= np.quantile(baseline_risk, .8)
    success = np.flatnonzero((regret > .01) & d_top & ~b_top & (a_hat != a_star)).tolist()
    success.sort(key=lambda i: (-regret[i], -(d_pct[i] - b_pct[i]), ids[i]))
    failure = np.flatnonzero((regret >= np.quantile(regret, .95)) & ~d_top).tolist()
    failure.sort(key=lambda i: (d_pct[i], -regret[i], ids[i]))
    return {
        "success_indices": success,
        "failure_indices": failure,
        "success_index": success[0] if success else None,
        "failure_index": failure[0] if failure else None,
        "direct_percentile": d_pct,
        "baseline_percentile": b_pct,
        "direct_top20_threshold": float(np.quantile(direct_risk, .8)),
        "baseline_top20_threshold": float(np.quantile(baseline_risk, .8)),
        "regret_top5_threshold": float(np.quantile(regret, .95)),
    }


def platform_legibility(history: np.ndarray) -> dict:
    """Geometry-only, fixed-view legibility metric for the frozen pilot pool.

The view is a 60 x 30 m rectangle in the ego-centred Map-Frenet frame. The
measure reads only the pre-intervention history's final frame (presence,s,d).
"""
    arr = np.asarray(history)
    if arr.shape != (10, 5, 8):
        raise ValueError(f"expected history (10,5,8), got {arr.shape}")
    present = arr[-1, :, 0] > .5
    sd = arr[-1, :, 2:4].astype(np.float64)
    inside = present & (sd[:, 0] >= -30) & (sd[:, 0] <= 30) & (sd[:, 1] >= -15) & (sd[:, 1] <= 15)
    visible = int(inside.sum())
    if visible > 1:
        points = sd[inside]
        dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        np.fill_diagonal(dist, np.inf)
        separation = float(np.min(dist))
        occlusions = int(np.sum(np.triu(dist < 3.0, 1)))
    else:
        separation, occlusions = 0.0, 0
    margins = np.minimum.reduce([
        sd[inside, 0] + 30, 30 - sd[inside, 0],
        sd[inside, 1] + 15, 15 - sd[inside, 1],
    ]) if visible else np.array([0.0])
    min_margin = float(np.min(margins))
    # Presence/viewport dominates; otherwise prefer separation, room for labels,
    # and fewer projected overlaps. No outcome or model-derived quantity enters.
    score = visible * 100 + min(separation, 12) * 2 + min(min_margin, 10) - occlusions * 10
    return {
        "score": float(score), "visible_entities": visible,
        "min_separation_m": separation, "min_margin_m": min_margin,
        "occluded_pairs_lt3m": occlusions,
    }


def frenet_to_world(reference, s: np.ndarray, d: np.ndarray, *, origin_s: float) -> np.ndarray:
    """Invert the frozen polyline's local (s,d) coordinates for honest map overlay."""
    s, d = np.broadcast_arrays(np.asarray(s, dtype=np.float64), np.asarray(d, dtype=np.float64))
    absolute = s + float(origin_s)
    flat = absolute.ravel()
    nodes = reference.s_nodes
    idx = np.clip(np.searchsorted(nodes, flat, side="right") - 1, 0, len(reference.seg_len) - 1)
    distance = flat - nodes[idx]
    tangent = reference.tangent[idx]
    base = reference.points[idx] + distance[:, None] * tangent
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    xy = base + d.ravel()[:, None] * normal
    return xy.reshape(*s.shape, 2)


def world_to_ego_plot(xy: np.ndarray, reference, *, origin_s: float) -> np.ndarray:
    """Rigidly rotate/translate world XY so ego heading at t* points right."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[-1] != 2:
        raise ValueError("world positions must end with two coordinates")
    origin = frenet_to_world(reference, np.asarray(0.0), np.asarray(0.0), origin_s=origin_s)
    heading = float(reference.heading_at(origin_s))
    tangent = np.array([np.cos(heading), np.sin(heading)])
    normal = np.array([-np.sin(heading), np.cos(heading)])
    relative = xy - origin
    return np.stack([relative @ tangent, relative @ normal], axis=-1)


def peak_step(selected_xy: np.ndarray, oracle_xy: np.ndarray) -> int:
    """Earliest frozen horizon index with maximal ego branch separation."""
    selected_xy = np.asarray(selected_xy, dtype=np.float64)
    oracle_xy = np.asarray(oracle_xy, dtype=np.float64)
    if selected_xy.shape != oracle_xy.shape or selected_xy.ndim != 2 or selected_xy.shape[1] != 2:
        raise ValueError("branch trajectories must align as (horizon,2)")
    distances = np.linalg.norm(selected_xy - oracle_xy, axis=1)
    if not np.isfinite(distances).all():
        raise ValueError("branch trajectories must be finite")
    return int(np.argmax(distances))
