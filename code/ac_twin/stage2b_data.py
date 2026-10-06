"""Stage 2B intersection data contract, movement-signature lookup and the
pilot response-support summaries.

Plan: AC-TWIN-RFER-STAGE2B-INTERSECTION-2026-09-14-V0.3.

Contents:
- NPZ contract validation for every Stage 2B role;
- the mechanical train/validation split assignment for the development role;
- the P0-frozen movement-signature / conflict lookup built from the
  intersection-v0 road network geometry (plan section 4), with a stable hash so
  the interaction-type rules cannot change after unblinding;
- pilot response-support metrics (plan section 5.1): the only quantities the
  pilot is allowed to compute. Nothing here reads model errors or V_evt.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from ac_twin.stage1_data import compute_valid_mask
from ac_twin import stage1_statistics as _s1
from ac_twin.checkpointing import canonical_fingerprint
from ac_twin import stage2b_protocol as sp

# --- split assignment (development role, plan section 6.1) -------------------------


def development_split_label(stratum_ordinal: int, *,
                            train_quota: int = sp.DEV_TRAIN_PER_STRATUM,
                            stratum_quota: int = sp.DEV_PER_STRATUM) -> int:
    """Mechanical pre-collection split: within each stratum, the first
    ``train_quota`` accepted episodes are train, the rest validation. The
    frozen protocol quotas are the defaults; the collector passes the
    (validation-enforced identical) config values so unit tests can exercise
    the rule at small scale."""
    if stratum_ordinal < 0 or stratum_ordinal >= int(stratum_quota):
        raise ValueError("stratum ordinal outside the development stratum quota")
    return (sp.SPLIT_LABELS["train"] if stratum_ordinal < int(train_quota)
            else sp.SPLIT_LABELS["validation"])


# --- NPZ contract --------------------------------------------------------------------


def validate_stage2b_source(data: Mapping[str, np.ndarray],
                            *, expected_split_labels: set[int],
                            expected_episodes: int) -> dict[str, Any]:
    """Validate a Stage 2B NPZ against the role contract; raise before use.

    Builds on the Stage 1 core contract (shapes, finiteness, unique seeds) and
    adds the Stage 2B capacity / split / diagnostic fields. Returns a small
    read-only summary.
    """
    required = (
        "object_history", "object_factual_future", "object_counterfactual_future",
        "factual_actions", "counterfactual_actions", "cf_action", "cf_duration",
        "episode_seed", "t_star", "split", "initial_identity_count",
        "max_road_vehicles", "object_capacity", "object_features_json",
        "vehicle_metadata_json",
    )
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(f"Stage 2B data missing required fields: {missing}")

    history = np.asarray(data["object_history"])
    episodes = history.shape[0]
    if episodes != expected_episodes:
        raise ValueError(f"expected {expected_episodes} episodes, found {episodes}")
    for name in ("object_history", "object_factual_future", "object_counterfactual_future"):
        if not np.isfinite(np.asarray(data[name])).all():
            raise ValueError(f"{name} contains NaN/Inf values")
    if len(np.unique(np.asarray(data["episode_seed"]))) != episodes:
        raise ValueError("episode_seed must be unique per episode")
    capacity = int(np.asarray(data["object_capacity"]))
    if capacity != sp.OBJECT_CAPACITY:
        raise ValueError(f"object_capacity must be {sp.OBJECT_CAPACITY}")
    if history.shape[2] != capacity:
        raise ValueError("object tensor width must equal object_capacity")

    split = np.asarray(data["split"])
    if not set(np.unique(split).tolist()) <= set(expected_split_labels):
        raise ValueError(
            f"split labels must be a subset of {sorted(expected_split_labels)}")
    identity_count = np.asarray(data["initial_identity_count"])
    road_max = np.asarray(data["max_road_vehicles"])
    if int(identity_count.max()) > capacity or int(road_max.max()) > capacity:
        raise ValueError("recorded vehicle counts exceed the frozen capacity")
    if (identity_count < 1).any():
        raise ValueError("every episode must register at least the ego vehicle")

    cf_action = np.asarray(data["cf_action"])
    if not set(np.unique(cf_action).tolist()) <= set(sp.ACTION_IDS):
        raise ValueError("cf_action outside the frozen three-action set")
    if not set(np.unique(np.asarray(data["cf_duration"])).tolist()) <= set(sp.INTERVENTION_DURATIONS):
        raise ValueError("cf_duration outside the frozen {1,3} set")
    factual_first = np.asarray(data["factual_actions"])[:, 0]
    if (cf_action == factual_first).any():
        raise ValueError("counterfactual action must differ from the factual first action")

    return {"episodes": int(episodes), "capacity": capacity,
            "split_counts": {int(s): int((split == s).sum())
                             for s in np.unique(split)}}


# --- movement signatures + P0 conflict lookup (plan section 4) ------------------------

# Fixed geometric constants of the intersection-v0 four-arm network
# (lane_width=4, right_turn_radius=9 -> outer_distance=11, access_length=100).
INTERIOR_BOUND_M = 15.0
SAMPLES_PER_LANE = 50

# Turn type per (incoming_arm, outgoing_arm): j=(i-1)%4 right, j=(i+2)%4
# straight, j=(i+1)%4 left (from intersection_env._make_road lane topology).
def movement_route_lanes(incoming: int, outgoing: int) -> list[tuple[str, str, int]]:
    """The three lane tuples of the canonical route for one movement."""
    i, j = int(incoming), int(outgoing)
    if i == j or not (0 <= i < 4) or not (0 <= j < 4):
        raise ValueError(f"invalid movement ({i}, {j})")
    return [("o" + str(i), "ir" + str(i), 0),
            ("ir" + str(i), "il" + str(j), 0),
            ("il" + str(j), "o" + str(j), 0)]


MOVEMENTS: tuple[tuple[int, int], ...] = tuple(
    (i, j) for i in range(4) for j in range(4) if i != j)


def _arm_of_node(node: Any, *, incoming: bool) -> int | None:
    """Arm index of a route node. Incoming accepts ``o{i}`` (approach lane) or
    ``ir{i}`` (inner lane: initial routes recorded after reset's simulation
    steps may already have dropped the approach segment as the vehicle
    advances). Outgoing must be an exit node ``o{j}``."""
    text = str(node)
    try:
        if incoming and text.startswith("ir"):
            return int(text[2:])
        if text.startswith("o"):
            return int(text[1:])
    except ValueError:
        return None
    return None


def movement_signature_from_route(route: Any) -> tuple[int, int] | None:
    """(incoming_arm, outgoing_arm) from a recorded route; None if unparseable.

    ``route`` is the metadata form: a list of [from_node, to_node, lane_index]
    string triples, e.g. [["o0","ir0","0"], ["ir0","il2","0"], ...]. The first
    segment may already be the inner lane (``ir{i}->il{j}``) because vehicles
    advance during reset's built-in simulation steps; the incoming arm is then
    read from the ``ir{i}`` node.
    """
    if not route:
        return None
    try:
        incoming = _arm_of_node(route[0][0], incoming=True)
        outgoing = _arm_of_node(route[-1][1], incoming=False)
    except (TypeError, IndexError):
        return None
    if incoming is None or outgoing is None:
        return None
    if incoming == outgoing or not (0 <= incoming < 4) or not (0 <= outgoing < 4):
        return None
    return (incoming, outgoing)


def _segments_intersect(p1: np.ndarray, p2: np.ndarray,
                        q1: np.ndarray, q2: np.ndarray) -> np.ndarray | None:
    """Proper 2D segment intersection point; None for parallel/disjoint/touching."""
    r = p2 - p1
    s = q2 - q1
    denom = float(r[0] * s[1] - r[1] * s[0])
    if abs(denom) < 1e-12:
        return None
    qp = q1 - p1
    t = float((qp[0] * s[1] - qp[1] * s[0]) / denom)
    u = float((qp[0] * r[1] - qp[1] * r[0]) / denom)
    # Strictly interior to both segments: shared endpoints (lane junctions)
    # are connectivity, not lateral crossings.
    eps = 1e-6
    if eps < t < 1.0 - eps and eps < u < 1.0 - eps:
        return p1 + t * r
    return None


def _polyline_crossing_points(p: np.ndarray, q: np.ndarray) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    for i in range(len(p) - 1):
        for j in range(len(q) - 1):
            hit = _segments_intersect(p[i], p[i + 1], q[j], q[j + 1])
            if hit is not None:
                points.append(hit)
    return points


def build_movement_lookup(polylines: Mapping[tuple[str, str, int], np.ndarray]
                          ) -> dict[str, Any]:
    """Build the P0 movement/conflict lookup from sampled lane polylines.

    ``polylines`` maps each lane tuple (from, to, index) to an (S,2) array of
    sampled centerline points. Two movements have a *crossing conflict* when
    their canonical routes share no directed route edge and their polylines
    properly intersect at a point inside the intersection interior
    (``INTERIOR_BOUND_M``). The lookup is hashed; once frozen at P0 the type
    rules cannot change after unblinding.
    """
    movements: dict[str, Any] = {}
    for i, j in MOVEMENTS:
        lanes = movement_route_lanes(i, j)
        points = []
        for lane in lanes:
            key = tuple(map(str, lane[:2])) + (int(lane[2]),)
            if key not in polylines:
                raise ValueError(f"road network is missing lane {key}")
            points.append(np.asarray(polylines[key], dtype=np.float64))
        movements[f"{i}->{j}"] = {"incoming": i, "outgoing": j, "lanes": lanes,
                                  "polyline": np.concatenate(points, axis=0)}

    crossing: dict[str, dict[str, bool]] = {}
    shared_edge: dict[str, dict[str, bool]] = {}
    for name_a, move_a in movements.items():
        crossing[name_a] = {}
        shared_edge[name_a] = {}
        lanes_a = {tuple(lane) for lane in move_a["lanes"]}
        for name_b, move_b in movements.items():
            if name_a == name_b:
                crossing[name_a][name_b] = False
                shared_edge[name_a][name_b] = True
                continue
            lanes_b = {tuple(lane) for lane in move_b["lanes"]}
            shared = bool(lanes_a & lanes_b)
            shared_edge[name_a][name_b] = shared
            hit = False
            if not shared:
                for point in _polyline_crossing_points(move_a["polyline"],
                                                       move_b["polyline"]):
                    if max(abs(float(point[0])), abs(float(point[1]))) <= INTERIOR_BOUND_M:
                        hit = True
                        break
            crossing[name_a][name_b] = hit

    serializable = {
        "constants": {"interior_bound_m": INTERIOR_BOUND_M,
                      "samples_per_lane": SAMPLES_PER_LANE},
        "movements": {name: {"incoming": m["incoming"], "outgoing": m["outgoing"],
                             "lanes": [list(map(str, lane)) for lane in m["lanes"]],
                             # Rounded for a machine-stable hash; geometry is
                             # deterministic for this frozen network.
                             "polyline": np.round(m["polyline"], 6).tolist()}
                      for name, m in movements.items()},
        "crossing": crossing,
        "shared_edge": shared_edge,
    }
    serializable["lookup_sha256"] = canonical_fingerprint(serializable)
    return serializable


def extract_network_polylines(env: Any,
                              samples: int = SAMPLES_PER_LANE) -> dict[tuple[str, str, int], np.ndarray]:
    """Sample every lane centerline of a reset intersection-v0 env (real adapter)."""
    base = getattr(env, "unwrapped", env)
    if base.road is None:
        raise RuntimeError("environment must be reset before sampling the network")
    out: dict[tuple[str, str, int], np.ndarray] = {}
    graph = base.road.network.graph
    for from_node, to_map in graph.items():
        for to_node, lanes in to_map.items():
            for index, lane in enumerate(lanes):
                key = (str(from_node), str(to_node), int(index))
                s_values = np.linspace(0.0, float(lane.length), int(samples))
                out[key] = np.asarray(
                    [lane.position(float(s), 0.0) for s in s_values], dtype=np.float64)
    return out


# --- pilot response-support metrics (plan section 5.1; the only pilot numbers) -------


def paired_valid_non_ego_mask(history: np.ndarray, factual: np.ndarray,
                              counterfactual: np.ndarray) -> np.ndarray:
    """E×T×N paired-valid non-ego mask (stable-ID matched in both branches)."""
    valid = compute_valid_mask(history, factual, counterfactual).copy()
    valid[:, :, 0] = False  # ego never enters estimation
    return valid


def pilot_support_metrics(data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """The complete, closed list of quantities a pilot may compute.

    Realized R is used only to judge scenario response support at the whole-
    profile level: support rate, positive episode count, median paired-valid
    non-ego vehicles per episode, and the temporary positive-tercile episode
    occupancy. No model outputs, no V_evt, no value shapes.
    """
    history = np.asarray(data["object_history"])
    factual = np.asarray(data["object_factual_future"])
    counterfactual = np.asarray(data["object_counterfactual_future"])
    valid = paired_valid_non_ego_mask(history, factual, counterfactual)
    r_pos, _ = _s1.response_magnitudes(counterfactual, factual)

    positive = valid & (r_pos > sp.ZERO_TOLERANCE)
    has_positive = positive.any(axis=(1, 2))
    episodes = history.shape[0]
    support_rate = float(has_positive.mean()) if episodes else 0.0
    paired_per_episode = valid.any(axis=1).sum(axis=1)
    median_paired = float(np.median(paired_per_episode)) if episodes else 0.0

    r_positive_values = r_pos[positive]
    if r_positive_values.size == 0:
        t1 = t2 = float("nan")
        bin_counts = {"positive_low": 0, "positive_mid": 0, "positive_high": 0}
    else:
        t1, t2 = np.quantile(r_positive_values, [1.0 / 3.0, 2.0 / 3.0])
        low = valid & (r_pos > sp.ZERO_TOLERANCE) & (r_pos <= t1)
        mid = valid & (r_pos > t1) & (r_pos <= t2)
        high = valid & (r_pos > t2)
        bin_counts = {
            "positive_low": int(low.any(axis=(1, 2)).sum()),
            "positive_mid": int(mid.any(axis=(1, 2)).sum()),
            "positive_high": int(high.any(axis=(1, 2)).sum()),
        }
    return {
        "accepted_pairs": int(episodes),
        "support_rate": support_rate,
        "positive_episodes": int(has_positive.sum()),
        "median_paired_valid_non_ego": median_paired,
        "temporary_terciles": {"t1": float(t1), "t2": float(t2)},
        "bin_episode_counts": bin_counts,
    }


# --- smoke semantic QA (plan section 4) ------------------------------------------------


def _paired_valid_entities(history: np.ndarray, factual: np.ndarray,
                           counterfactual: np.ndarray) -> np.ndarray:
    """E×N mask: known in history and present at >=1 frame in both futures."""
    known = (history[..., 0] > 0).any(axis=1)
    f_present = (factual[..., 0] > 0).any(axis=1)
    c_present = (counterfactual[..., 0] > 0).any(axis=1)
    return known & f_present & c_present


def smoke_semantic_qa(data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """P0 mechanism-field semantics QA on the smoke NPZ (plan section 4).

    Checks, over all accepted smoke episodes:
    - ``vehicle_metadata_json.route`` is non-empty for every initial
      paired-valid vehicle;
    - at least 3 distinct movement signatures appear in total;
    - present vehicles' lane_index / target_lane_index are not all -1;
    - cmd_accel / cmd_steer are not all-NaN on present cells.
    Any failure is a P0 failure (no pilot may start).
    """
    history = np.asarray(data["object_history"])
    factual = np.asarray(data["object_factual_future"])
    counterfactual = np.asarray(data["object_counterfactual_future"])
    episodes = history.shape[0]
    paired = _paired_valid_entities(history, factual, counterfactual)

    signatures: set[tuple[int, int]] = set()
    missing_route = 0
    for e in range(episodes):
        metadata = json.loads(str(np.asarray(data["vehicle_metadata_json"])[e]))
        by_id = {int(record["entity_id"]): record for record in metadata}
        for entity_id in np.nonzero(paired[e])[0]:
            record = by_id.get(int(entity_id))
            signature = (movement_signature_from_route(record.get("route"))
                         if record is not None else None)
            if signature is None:
                missing_route += 1
            else:
                signatures.add(signature)

    present_history = history[..., 0] > 0
    present_factual = factual[..., 0] > 0
    present = np.concatenate([present_history, present_factual], axis=1)
    detail: dict[str, Any] = {"paired_valid_entities": int(paired.sum())}
    detail["route_nonempty_for_paired_valid"] = bool(missing_route == 0)
    detail["distinct_movement_signatures"] = int(len(signatures))
    detail["movement_signatures"] = sorted(list(signatures))
    detail["min_distinct_signatures"] = bool(len(signatures) >= 3)

    lane = np.asarray(data["factual_lane_index"])
    target_lane = np.asarray(data["factual_target_lane_index"])
    detail["lane_index_present_not_all_minus1"] = bool((lane[present] != -1).any())
    detail["target_lane_index_present_not_all_minus1"] = bool(
        (target_lane[present] != -1).any())

    accel_mask = np.asarray(data["factual_cmd_accel_mask"])
    steer_mask = np.asarray(data["factual_cmd_steer_mask"])
    detail["cmd_accel_present_not_all_nan"] = bool((accel_mask & present).any())
    detail["cmd_steer_present_not_all_nan"] = bool((steer_mask & present).any())

    detail["pass"] = all(bool(v) for k, v in detail.items()
                         if k.endswith("_not_all_minus1") or k.endswith("_not_all_nan")
                         or k in ("route_nonempty_for_paired_valid",
                                  "min_distinct_signatures"))
    return detail
