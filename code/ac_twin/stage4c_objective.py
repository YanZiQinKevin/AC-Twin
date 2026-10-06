"""Stage 4C frozen trajectory-consequence function and decision regret.

Plan ``AC-TWIN-RFER-STAGE4C-MULTIACTION-REGRET-2026-09-15-V0.1`` section 5. This
module is pure, frozen and label-symmetric: the SAME consequence function is applied
to the ensemble-predicted counterfactual trajectories (``Y_hat_CF``) and to the true
counterfactual trajectories (``Y_CF``) to produce ``J_hat`` and ``J_true``. Nothing
here reads a risk score, an oracle identity, a response or any counterfactual
presence mask beyond the evaluation mask passed in by the caller.

The consequence function is a transparent proxy — NOT a collision probability and
NOT a full driving utility (plan sections 2.3 / 5.2). The constants below are frozen
before any development data is touched and are never tuned on data.
"""
from __future__ import annotations

import numpy as np

# --- frozen consequence constants (plan section 5) -------------------------------

#: ego occupies object slot 0 by the Stage 4 object-centric contract.
EGO_SLOT = 0

#: longitudinal / lateral centre-distance scales (metres) for the interaction proxy.
INTERACTION_LONGITUDINAL_SCALE = 5.0
INTERACTION_LATERAL_SCALE = 2.0

#: primary frozen consequence weights (plan section 5.3).
WEIGHT_INTERACTION = 0.80
WEIGHT_EFFICIENCY = 0.20

#: pre-registered secondary sensitivity weights (never a substitute primary judge).
SECONDARY_WEIGHTS = ((0.70, 0.30), (0.90, 0.10))

#: progress-range floor below which every action's efficiency cost is 0.
PROGRESS_EPS = 1e-6

#: numerical tolerance for the regret non-negativity clamp (plan section 5.4).
REGRET_CLAMP_TOL = 1e-10

#: regret above this marks a "meaningful" non-zero decision regret (Gate TASK).
REGRET_MEANINGFUL = 0.01

#: the ten frozen candidate plans, in canonical tie-break order (plan section 4.1).
ACTION_PLANS = (
    (0, 1), (0, 3),  # LANE_LEFT
    (1, 1), (1, 3),  # IDLE
    (2, 1), (2, 3),  # LANE_RIGHT
    (3, 1), (3, 3),  # FASTER
    (4, 1), (4, 3),  # SLOWER
)

#: canonical "always idle" qualification baseline = the longest IDLE plan (1, 3).
#: (Plan interpretation: with two IDLE candidates the pure-default floor is the
#: maximal-duration idle; recorded in the decision-rule receipt.)
IDLE_PLAN_INDEX = ACTION_PLANS.index((1, 3))


def _check_trajectory(trajectory_xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(trajectory_xy, dtype=np.float64)
    if xy.ndim < 4 or xy.shape[-1] != 2:
        raise ValueError(
            f"trajectory_xy must have shape (..., T, N, 2); got {xy.shape}")
    return xy


def trajectory_components(
    trajectory_xy: np.ndarray,
    history_last_xy: np.ndarray,
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-candidate interaction severity and ego progress.

    Parameters
    ----------
    trajectory_xy
        Candidate trajectories, shape ``(..., T, N, 2)`` where the trailing
        dimension is ``(x, y)`` and object slot 0 is the ego.
    history_last_xy
        Ego position at the last history frame, shape ``(..., 2)``. Broadcastable
        against the leading dims of ``trajectory_xy``.
    mask
        Boolean legal-cell mask, shape ``(..., T, N)`` (non-ego, evaluation-legal).
        Broadcastable against ``trajectory_xy[..., :, :, :]``. Cells outside the
        mask never contribute to the interaction severity.

    Returns
    -------
    dict with
        ``interaction``: ``(...)`` ``C_int`` in ``[0, 1]`` — max masked Gaussian
            centre-proximity (plan section 5.2). 0 when no legal cell exists.
        ``progress``: ``(...)`` raw ego longitudinal progress ``P`` (metres),
            ``x_ego_final - x_ego_history_last`` (plan section 5.3).
    """
    xy = _check_trajectory(trajectory_xy)
    hist_last = np.asarray(history_last_xy, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if hist_last.shape[-1] != 2:
        raise ValueError(f"history_last_xy must end in 2; got {hist_last.shape}")
    if mask.shape != xy.shape[:-1]:
        raise ValueError(
            f"mask shape {mask.shape} must equal trajectory shape {xy.shape[:-1]}")

    ego_xy = xy[..., EGO_SLOT, :]  # (..., T, 2)
    dx = (ego_xy[..., None, 0] - xy[..., 0]) / INTERACTION_LONGITUDINAL_SCALE
    dy = (ego_xy[..., None, 1] - xy[..., 1]) / INTERACTION_LATERAL_SCALE
    proximity = np.exp(-0.5 * dx * dx - 0.5 * dy * dy)  # (..., T, N) in (0, 1]
    proximity = np.where(mask, proximity, -np.inf)
    interaction = proximity.max(axis=(-2, -1))  # (...,)
    interaction = np.where(np.isneginf(interaction), 0.0, interaction)

    final_ego_x = xy[..., -1, EGO_SLOT, 0]  # (...,)
    progress = final_ego_x - hist_last[..., 0]
    return {
        "interaction": interaction.astype(np.float64),
        "progress": progress.astype(np.float64),
    }


def efficiency_cost(progress: np.ndarray) -> np.ndarray:
    """Group-normalised efficiency cost ``C_eff`` from raw progress.

    ``progress`` has shape ``(..., A)`` where the last axis is the within-group
    action axis. ``C_eff(a) = (max_b P_b - P_a) / (max_b P_b - min_b P_b + 1e-6)``;
    when the within-group progress range is ``< 1e-6`` every action's cost is 0
    (plan section 5.3).
    """
    p = np.asarray(progress, dtype=np.float64)
    if p.ndim < 1:
        raise ValueError("progress must have an action axis")
    pmax = p.max(axis=-1, keepdims=True)
    pmin = p.min(axis=-1, keepdims=True)
    span = pmax - pmin
    c_eff = (pmax - p) / (span + PROGRESS_EPS)
    c_eff = np.where(span < PROGRESS_EPS, 0.0, c_eff)
    return c_eff.astype(np.float64)


def total_cost(
    interaction: np.ndarray,
    efficiency: np.ndarray,
    *,
    weight_interaction: float = WEIGHT_INTERACTION,
    weight_efficiency: float = WEIGHT_EFFICIENCY,
) -> np.ndarray:
    """Frozen total consequence ``J = w_int * C_int + w_eff * C_eff``."""
    return (
        float(weight_interaction) * np.asarray(interaction, dtype=np.float64)
        + float(weight_efficiency) * np.asarray(efficiency, dtype=np.float64)
    )


def _argmin_plan_order(cost: np.ndarray) -> np.ndarray:
    """argmin over the last (action) axis; exact ties break to the lowest plan index.

    ``np.argmin`` returns the first occurrence of the minimum, which is exactly the
    canonical ``ACTION_PLANS`` order — the unique frozen tie-break (plan section 4.1).
    """
    return np.argmin(np.asarray(cost, dtype=np.float64), axis=-1)


def clamp_regret(regret: np.ndarray) -> np.ndarray:
    """Enforce regret non-negativity within numerical tolerance (plan section 5.4).

    Values in ``[-1e-10, 0)`` are clamped to 0; anything below ``-1e-10`` indicates
    an implementation error and raises.
    """
    r = np.asarray(regret, dtype=np.float64)
    if (r < -REGRET_CLAMP_TOL).any():
        raise ValueError(
            f"negative decision regret below tolerance: min {r.min():.3e} "
            f"< -{REGRET_CLAMP_TOL}")
    return np.where((r < 0.0), 0.0, r)


def decision_outputs(
    predicted_xy: np.ndarray,
    true_xy: np.ndarray,
    history_last_xy: np.ndarray,
    mask: np.ndarray,
    *,
    weight_interaction: float = WEIGHT_INTERACTION,
    weight_efficiency: float = WEIGHT_EFFICIENCY,
) -> dict[str, np.ndarray]:
    """Group-level decision consequences, base/oracle actions and regret.

    Parameters
    ----------
    predicted_xy, true_xy
        ``(G, A, T, N, 2)`` ensemble-predicted and true counterfactual trajectories
        for the ``A`` candidate plans of each of the ``G`` decision groups.
    history_last_xy
        ``(G, 2)`` ego position at the last history frame (shared across actions).
    mask
        ``(G, T, N)`` evaluation mask; identical across the ``A`` actions of a group.

    Returns
    -------
    dict with ``J_hat``, ``J_true`` ``(G, A)``; ``a_hat``, ``a_star`` ``(G,)``;
    ``regret``, ``always_idle_regret``, ``random_action_regret`` ``(G,)``; and the
    intermediate components for diagnostics. The same frozen weights are applied to
    predicted and true branches.
    """
    pred = _check_trajectory(predicted_xy)
    true = _check_trajectory(true_xy)
    if pred.shape != true.shape:
        raise ValueError(
            f"predicted/true shapes must match; got {pred.shape} vs {true.shape}")
    if pred.ndim != 5:
        raise ValueError(f"predicted_xy must be (G, A, T, N, 2); got {pred.shape}")
    groups, n_actions = pred.shape[0], pred.shape[1]
    if n_actions != len(ACTION_PLANS):
        raise ValueError(
            f"action axis must have {len(ACTION_PLANS)} plans; got {n_actions}")
    hist_last = np.asarray(history_last_xy, dtype=np.float64)
    if hist_last.shape != (groups, 2):
        raise ValueError(f"history_last_xy must be (G, 2); got {hist_last.shape}")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (groups,) + pred.shape[2:4]:
        raise ValueError(
            f"mask must be (G, T, N) = {(groups,) + pred.shape[2:4]}; got {mask.shape}")

    # Broadcast the shared history anchor and mask across the action axis.
    hist_b = np.broadcast_to(hist_last[:, None, :], pred.shape[:2] + (2,))
    mask_b = np.broadcast_to(mask[:, None, :, :], pred.shape[:-1])

    comp_hat = trajectory_components(pred, hist_b, mask_b)
    comp_true = trajectory_components(true, hist_b, mask_b)

    j_hat = total_cost(comp_hat["interaction"], efficiency_cost(comp_hat["progress"]),
                       weight_interaction=weight_interaction,
                       weight_efficiency=weight_efficiency)
    j_true = total_cost(comp_true["interaction"], efficiency_cost(comp_true["progress"]),
                        weight_interaction=weight_interaction,
                        weight_efficiency=weight_efficiency)

    a_hat = _argmin_plan_order(j_hat)  # (G,)
    a_star = _argmin_plan_order(j_true)  # (G,)
    rows = np.arange(groups)
    j_true_hat = j_true[rows, a_hat]
    j_true_star = j_true[rows, a_star]
    regret = clamp_regret(j_true_hat - j_true_star)

    # Decision-qualification baselines (plan section 5.4): NOT risk-ranking baselines.
    always_idle_regret = clamp_regret(j_true[:, IDLE_PLAN_INDEX] - j_true_star)
    random_action_regret = clamp_regret(j_true.mean(axis=-1) - j_true_star)

    return {
        "J_hat": j_hat,
        "J_true": j_true,
        "a_hat": a_hat.astype(np.int64),
        "a_star": a_star.astype(np.int64),
        "regret": regret,
        "always_idle_regret": always_idle_regret,
        "random_action_regret": random_action_regret,
        "interaction_hat": comp_hat["interaction"],
        "interaction_true": comp_true["interaction"],
        "progress_hat": comp_hat["progress"],
        "progress_true": comp_true["progress"],
        "efficiency_hat": efficiency_cost(comp_hat["progress"]),
        "efficiency_true": efficiency_cost(comp_true["progress"]),
    }


__all__ = [
    "ACTION_PLANS",
    "EGO_SLOT",
    "IDLE_PLAN_INDEX",
    "INTERACTION_LATERAL_SCALE",
    "INTERACTION_LONGITUDINAL_SCALE",
    "PROGRESS_EPS",
    "REGRET_CLAMP_TOL",
    "REGRET_MEANINGFUL",
    "SECONDARY_WEIGHTS",
    "WEIGHT_EFFICIENCY",
    "WEIGHT_INTERACTION",
    "clamp_regret",
    "decision_outputs",
    "efficiency_cost",
    "total_cost",
    "trajectory_components",
]
