"""Stage 4D-D2 pure geometry / eligibility / feature layer (no MetaDrive import).

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` sections 4.2
(eligibility), 4.3 (roles), 5.1-5.2 (action plans), 5.4 (entity correspondence),
5.5 (common Map-Frenet frame and the 8-dim object schema). Everything here is a
pure function of the converted ScenarioNet scenario dict (a plain pickle of
numpy arrays) plus frozen constants, so it is unit-testable without the
simulator and identical for every branch of a group.

Frame mapping (frozen): the scenario's native frames run at ~10 Hz. The 5 Hz
policy layer samples every 2nd native frame: 5 Hz index k <-> native frame 2k.
History = 5 Hz steps [t_star-9 .. t_star] (native 2k-18 .. 2k, 10 frames,
anchor included); future = 5 Hz steps [t_star+1 .. t_star+20] (native
2k+2 .. 2k+40, 20 frames). This mirrors the R2/Merge collector contract
(history ends AT t_star; the 20 future frames start one policy step after).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ac_twin.stage4c_objective import ACTION_PLANS, IDLE_PLAN_INDEX
from ac_twin.stage4_protocol import CF_ACTIONS, CF_DURATIONS

# --- frozen time/grid contract ----------------------------------------------------
NATIVE_PER_POLICY = 2  # 5 Hz policy layer over the ~10 Hz native scenario frames
HISTORY_LEN = 10
FUTURE_LEN = 20
N_ACTIONS = len(ACTION_PLANS)  # 10
OBJECT_CAPACITY = 5
N_OBJECT_FEATURES = 8
T_STAR_CANDIDATES = (10, 11, 12, 13, 14, 15)

# --- frozen eligibility constants (plan 4.2; pre-t_star information only) ---------
DRIVABLE_LANE_TYPES = frozenset({"LANE_SURFACE_STREET", "LANE_FREEWAY"})
STATIC_STD_THRESHOLD_M = 3.0  # platform STATIC_THRESHOLD, computed pre-t_star ONLY
EGO_LANE_MAX_LATERAL_M = 3.0
EGO_LANE_MAX_HEADING_DIFF_RAD = math.radians(75.0)
SAME_DIRECTION_MIN_CHORD_DOT = 0.0
MIN_DYNAMIC_NON_EGO = 4
REF_FORWARD_TARGET_M = 150.0
REF_BACKWARD_TARGET_M = 60.0
LANE_GRAPH_MAX_HOPS = 8

# --- frozen factual-plan sampling (plan 5.2; R2 weights, never retuned) -----------
# Intent index semantics come from the frozen ACTION_PLANS comments: 0=LANE_LEFT,
# 1=IDLE, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER. Weights are indexed by that intent.
ACTION_WEIGHTS = np.array([0.095, 0.45, 0.095, 0.18, 0.18], dtype=np.float64)
assert tuple(CF_ACTIONS) == (0, 1, 2, 3, 4)
assert tuple(CF_DURATIONS) == (1, 3)
assert tuple(a for a, _d in ACTION_PLANS) == (0, 0, 1, 1, 2, 2, 3, 3, 4, 4)
assert IDLE_PLAN_INDEX == 3 and ACTION_PLANS[IDLE_PLAN_INDEX] == (1, 3)

FEATURE_NAMES = (
    "presence", "is_ego", "s", "d", "v_s", "v_d", "cos_heading_rel", "sin_heading_rel",
)

#: eligibility rejection reason codes (plan 4.4: fixed set, no free text)
REASON_NO_SDC = "no_sdc_track"
REASON_SDC_PREFIX_INVALID = "sdc_prefix_invalid"
REASON_NO_FUTURE_CAPACITY = "no_future_capacity"
REASON_NO_DRIVABLE_LANE = "no_drivable_lane"
REASON_NO_EGO_LANE = "no_ego_lane_at_t_star"
REASON_NO_LEFT_LANE = "no_left_lane_at_t_star"
REASON_NO_RIGHT_LANE = "no_right_lane_at_t_star"
REASON_FRAME_BUILD_FAILED = "frenet_frame_build_failed"
REASON_FEW_DYNAMIC_NON_EGO = "fewer_than_4_dynamic_non_ego"


# ==================================================================================
# Polyline geometry
# ==================================================================================


def _wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


class PolylineFrame:
    """Arc-length frame over a 2D polyline with linear end extrapolation.

    ``project(xy)`` returns ``(s, d)`` with ``d > 0`` on the LEFT of the reference
    direction. Points beyond either end are projected onto the linear extension of
    the end tangents; ``inside`` marks projections on the real polyline.
    """

    def __init__(self, points: np.ndarray):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
            raise ValueError(f"polyline must be (N>=2, 2); got {pts.shape}")
        seg = np.diff(pts, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        keep = np.concatenate([[True], seg_len > 1e-9])
        pts = pts[keep]
        if pts.shape[0] < 2:
            raise ValueError("degenerate polyline (all points coincide)")
        seg = np.diff(pts, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.points = pts
        self.seg = seg
        self.seg_len = seg_len
        self.s_nodes = np.concatenate([[0.0], np.cumsum(seg_len)])
        self.tangent = seg / seg_len[:, None]
        self.length = float(self.s_nodes[-1])

    def project(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised projection. Returns (s, d, inside) for (..., 2) input."""
        pts = np.asarray(xy, dtype=np.float64)
        flat = pts.reshape(-1, 2)
        p0 = self.points[:-1]  # (M, 2)
        seg = self.seg  # (M, 2)
        seg_len2 = (seg * seg).sum(axis=1)  # (M,)
        rel = flat[:, None, :] - p0[None, :, :]  # (N, M, 2)
        u = (rel * seg[None, :, :]).sum(axis=2) / seg_len2[None, :]  # (N, M)
        u_clip = np.clip(u, 0.0, 1.0)
        proj = p0[None, :, :] + u_clip[..., None] * seg[None, :, :]  # (N, M, 2)
        dist2 = ((flat[:, None, :] - proj) ** 2).sum(axis=2)  # (N, M)
        best = np.argmin(dist2, axis=1)  # (N,)
        n = flat.shape[0]
        u_best = u[np.arange(n), best]
        proj_best = proj[np.arange(n), best]
        tan_best = self.tangent[best]
        s = self.s_nodes[best] + u_best * self.seg_len[best]
        # signed distance: positive on the left of the tangent direction
        diff = flat - proj_best
        cross = tan_best[:, 0] * diff[:, 1] - tan_best[:, 1] * diff[:, 0]
        d = np.where(cross >= 0.0, 1.0, -1.0) * np.sqrt(dist2[np.arange(n), best])
        inside = (u_best >= 0.0) & (u_best <= 1.0)
        # end extrapolation: before node 0 / after last node along the end tangents
        before = s < 0.0
        after = s > self.length
        if before.any():
            s0 = np.where(before, s, 0.0)
            proj0 = self.points[0] + s0[..., None] * self.tangent[0]
            diff0 = flat - proj0
            cross0 = self.tangent[0, 0] * diff0[:, 1] - self.tangent[0, 1] * diff0[:, 0]
            d0 = np.where(cross0 >= 0.0, 1.0, -1.0) * np.hypot(diff0[:, 0], diff0[:, 1])
            s = np.where(before, s0, s)
            d = np.where(before, d0, d)
            inside = inside & ~before
        if after.any():
            s1 = np.where(after, s - self.length, 0.0)
            proj1 = self.points[-1] + s1[..., None] * self.tangent[-1]
            diff1 = flat - proj1
            cross1 = self.tangent[-1, 0] * diff1[:, 1] - self.tangent[-1, 1] * diff1[:, 0]
            d1 = np.where(cross1 >= 0.0, 1.0, -1.0) * np.hypot(diff1[:, 0], diff1[:, 1])
            s = np.where(after, self.length + s1, s)
            d = np.where(after, d1, d)
            inside = inside & ~after
        shape = pts.shape[:-1]
        return s.reshape(shape), d.reshape(shape), inside.reshape(shape)

    def closest_distance(self, xy: np.ndarray) -> np.ndarray:
        """Euclidean distance to the closest point on the polyline with the
        projection parameter CLAMPED to [0, 1] (endpoints included, NO linear end
        extrapolation). Lane matching uses this so a far lane cannot win via an
        extrapolated near-zero lateral offset."""
        pts = np.asarray(xy, dtype=np.float64)
        flat = pts.reshape(-1, 2)
        p0 = self.points[:-1]
        seg = self.seg
        seg_len2 = (seg * seg).sum(axis=1)
        rel = flat[:, None, :] - p0[None, :, :]
        u = np.clip((rel * seg[None, :, :]).sum(axis=2) / seg_len2[None, :], 0.0, 1.0)
        proj = p0[None, :, :] + u[..., None] * seg[None, :, :]
        dist = np.sqrt(((flat[:, None, :] - proj) ** 2).sum(axis=2)).min(axis=1)
        return dist.reshape(pts.shape[:-1])

    def heading_at(self, s: np.ndarray | float) -> np.ndarray:
        """Reference tangent heading at arc position s (clamped to end tangents)."""
        s_arr = np.asarray(s, dtype=np.float64).reshape(-1)
        idx = np.searchsorted(self.s_nodes, s_arr, side="right") - 1
        idx = np.clip(idx, 0, len(self.tangent) - 1)
        tan = self.tangent[idx]
        return np.arctan2(tan[:, 1], tan[:, 0]).reshape(np.asarray(s).shape if np.ndim(s) else ())


# ==================================================================================
# Lane graph over ScenarioNet map_features
# ==================================================================================


@dataclass(frozen=True)
class LaneInfo:
    lane_id: str
    polyline: np.ndarray  # (N, 2)
    left_neighbors: tuple  # raw neighbor dicts
    right_neighbors: tuple
    entry_lanes: tuple
    exit_lanes: tuple


def build_lane_graph(map_features: Mapping[str, Any]) -> dict[str, LaneInfo]:
    """Drivable-lane graph from converted map_features (plan 5.5)."""
    graph: dict[str, LaneInfo] = {}
    for fid, feature in map_features.items():
        if not isinstance(feature, Mapping):
            continue
        if feature.get("type") not in DRIVABLE_LANE_TYPES:
            continue
        polyline = np.asarray(feature.get("polyline", []), dtype=np.float64)
        if polyline.ndim != 2 or polyline.shape[0] < 2:
            continue
        graph[str(fid)] = LaneInfo(
            lane_id=str(fid),
            polyline=polyline[:, :2],
            left_neighbors=tuple(feature.get("left_neighbor", []) or ()),
            right_neighbors=tuple(feature.get("right_neighbor", []) or ()),
            entry_lanes=tuple(feature.get("entry_lanes", []) or ()),
            exit_lanes=tuple(feature.get("exit_lanes", []) or ()),
        )
    return graph


def _chord_direction(polyline: np.ndarray) -> np.ndarray:
    """Robust chord direction: 10th-percentile point -> 90th-percentile point."""
    n = polyline.shape[0]
    a = polyline[max(0, int(0.1 * (n - 1)))]
    b = polyline[min(n - 1, int(0.9 * (n - 1)))]
    chord = b - a
    norm = float(np.hypot(chord[0], chord[1]))
    if norm < 1e-9:
        return np.array([0.0, 0.0])
    return chord / norm


def same_direction(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    """Two lanes are same-direction iff their chord directions align (frozen: dot > 0)."""
    return float(np.dot(_chord_direction(poly_a), _chord_direction(poly_b))) > \
        SAME_DIRECTION_MIN_CHORD_DOT


def _neighbor_station_overlap(neighbor: Mapping[str, Any], station_index: int) -> bool:
    """The ego projection index must fall inside the neighbor relation's self range."""
    try:
        lo = int(neighbor.get("self_start_index", 0))
        hi = int(neighbor.get("self_end_index", 2**31 - 1))
    except (TypeError, ValueError):
        return False
    return lo <= station_index <= hi


def adjacent_lane(
    graph: Mapping[str, LaneInfo], lane_id: str, side: str, station_index: int
) -> str | None:
    """First drivable, same-direction neighbor on ``side`` whose self-index range
    covers the ego station (plan 4.2 item 3). Deterministic list order."""
    info = graph.get(lane_id)
    if info is None:
        return None
    neighbors = info.left_neighbors if side == "left" else info.right_neighbors
    for neighbor in neighbors:
        if not isinstance(neighbor, Mapping):
            continue
        nid = str(neighbor.get("feature_id", ""))
        if nid not in graph:
            continue
        if not _neighbor_station_overlap(neighbor, station_index):
            continue
        if not same_direction(info.polyline, graph[nid].polyline):
            continue
        return nid
    return None


def find_ego_lane(
    graph: Mapping[str, LaneInfo], position: np.ndarray, heading: float
) -> tuple[str, int] | None:
    """Nearest drivable lane to the ego at t_star, with heading alignment.

    Matching uses the CLAMPED closest distance (no end extrapolation) so a lane
    the ego has not reached / has left cannot win via an extrapolated projection.
    Returns (lane_id, station_index) or None. Tie-break: smallest distance, then
    smallest lane_id (deterministic).
    """
    best: tuple[float, str, int] | None = None
    for lane_id, info in graph.items():
        frame = PolylineFrame(info.polyline)
        dist = float(frame.closest_distance(np.asarray(position, dtype=np.float64)))
        if dist > EGO_LANE_MAX_LATERAL_M:
            continue
        s, _d, _in = frame.project(np.asarray(position, dtype=np.float64))
        s_c = float(np.clip(float(s), 0.0, frame.length))
        ref_heading = float(frame.heading_at(np.array([s_c]))[0])
        if abs(float(_wrap_to_pi(heading - ref_heading))) > EGO_LANE_MAX_HEADING_DIFF_RAD:
            continue
        station = int(np.searchsorted(frame.s_nodes, s_c, side="right") - 1)
        station = int(np.clip(station, 0, len(info.polyline) - 2))
        key = (dist, lane_id)
        if best is None or key < (best[0], best[1]):
            best = (dist, lane_id, station)
    if best is None:
        return None
    return best[1], best[2]


def _polyline_length(polyline: np.ndarray) -> float:
    seg = np.diff(np.asarray(polyline, dtype=np.float64), axis=0)
    return float(np.hypot(seg[:, 0], seg[:, 1]).sum())


def _follow_lanes(
    graph: Mapping[str, LaneInfo], lane_id: str, direction: str, target_m: float
) -> list[str]:
    """Greedy exit/entry chain with best heading continuation (frozen rule).

    Accumulates arc length and stops once ``target_m`` is reached (or the chain
    ends / hits the hop cap), so the reference stays bounded without any fragile
    endpoint-projection truncation.
    """
    chain: list[str] = []
    current = lane_id
    accumulated = 0.0
    for _ in range(LANE_GRAPH_MAX_HOPS):
        info = graph.get(current)
        if info is None:
            break
        links = info.exit_lanes if direction == "exit" else info.entry_lanes
        candidates = [str(x) for x in links if str(x) in graph]
        if not candidates:
            break
        ref_dir = _chord_direction(info.polyline)
        best = max(
            candidates,
            key=lambda cid: (float(np.dot(ref_dir, _chord_direction(graph[cid].polyline))),
                             tuple(-ord(c) for c in cid)),
        )
        chain.append(best)
        accumulated += _polyline_length(graph[best].polyline)
        current = best
        if accumulated >= target_m:
            break
    return chain


def build_reference_polyline(
    graph: Mapping[str, LaneInfo], lane_id: str
) -> np.ndarray:
    """Reference = ego lane centreline extended backward via entry lanes and
    forward via exit lanes (best heading continuation, frozen greedy rule), each
    bounded to the frozen target lengths. The ego lane geometry is always present
    verbatim in the middle of the concatenation, so the ego's t_star projection
    lands on its own lane.
    """
    forward = _follow_lanes(graph, lane_id, "exit", REF_FORWARD_TARGET_M)
    backward = _follow_lanes(graph, lane_id, "entry", REF_BACKWARD_TARGET_M)
    parts: list[np.ndarray] = []
    for cid in reversed(backward):
        parts.append(graph[cid].polyline)
    parts.append(graph[lane_id].polyline)
    for cid in forward:
        parts.append(graph[cid].polyline)
    return np.concatenate(parts, axis=0)


class MapFrenetFrame:
    """Frozen common Map-Frenet frame anchored at the ego t_star projection.

    s: arc length along the reference polyline, origin at the ego projection.
    d: signed lateral offset (positive = left of reference direction).
    """

    def __init__(self, reference: PolylineFrame, ego_xy: np.ndarray):
        s_ego, _, inside = reference.project(np.asarray(ego_xy, dtype=np.float64))
        self.reference = reference
        self.origin_s = float(s_ego)
        self.origin_inside = bool(inside)

    def project(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s, d, inside = self.reference.project(xy)
        return s - self.origin_s, d, inside

    def transform(
        self,
        xy: np.ndarray,
        heading: np.ndarray,
        velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(s, d, v_s, v_d, cos(h_rel), sin(h_rel)) for arrays of states."""
        s, d, _inside = self.project(xy)
        ref_heading = self.reference.heading_at(np.asarray(s) + self.origin_s)
        h_rel = _wrap_to_pi(np.asarray(heading, dtype=np.float64) - ref_heading)
        tangent = np.stack([np.cos(ref_heading), np.sin(ref_heading)], axis=-1)
        normal = np.stack([-np.sin(ref_heading), np.cos(ref_heading)], axis=-1)
        v = np.asarray(velocity, dtype=np.float64)
        v_s = (v * tangent).sum(axis=-1)
        v_d = (v * normal).sum(axis=-1)
        return s, d, v_s, v_d, np.cos(h_rel), np.sin(h_rel)


# ==================================================================================
# Track helpers (pre-t_star information only)
# ==================================================================================


def _track_state(track: Mapping[str, Any]) -> Mapping[str, Any]:
    return track.get("state", {})


def track_valid_at(track: Mapping[str, Any], frame: int) -> bool:
    valid = _track_state(track).get("valid")
    if valid is None or frame < 0 or frame >= len(valid):
        return False
    return bool(valid[frame])


def track_is_dynamic_pre_t_star(track: Mapping[str, Any], t_star_native: int) -> bool:
    """Dynamic iff the platform's static statistic (max xy std of valid positions)
    exceeds the frozen threshold, computed on frames <= t_star ONLY (plan 4.2.6:
    no future reads)."""
    state = _track_state(track)
    position = state.get("position")
    valid = state.get("valid")
    if position is None or valid is None:
        return False
    upto = min(int(t_star_native) + 1, len(valid))
    mask = np.asarray(valid[:upto], dtype=bool)
    pts = np.asarray(position, dtype=np.float64)[:upto][mask][:, :2]
    if pts.shape[0] < 2:
        return False
    return float(np.max(np.std(pts, axis=0))) > STATIC_STD_THRESHOLD_M


def track_position_at(track: Mapping[str, Any], frame: int) -> np.ndarray:
    return np.asarray(_track_state(track)["position"][frame], dtype=np.float64)[:2]


def track_heading_at(track: Mapping[str, Any], frame: int) -> float:
    return float(_track_state(track)["heading"][frame])


# ==================================================================================
# Plan 4.2 eligibility + t_star selection + entity selection
# ==================================================================================


def sdc_prefix_valid(sdc_track: Mapping[str, Any], t_star_native: int) -> bool:
    """SDC valid at every native frame of the prefix [0, t_star] (replay source)."""
    state = _track_state(sdc_track)
    valid = state.get("valid")
    if valid is None or len(valid) <= t_star_native:
        return False
    return bool(np.all(np.asarray(valid[: t_star_native + 1], dtype=bool)))


def future_capacity_ok(scenario_length: int, t_star_5hz: int) -> bool:
    """5 Hz future frames t_star+1..t_star+20 -> native 2k+2..2k+40 must exist."""
    return int(scenario_length) >= 2 * int(t_star_5hz) + 2 * FUTURE_LEN + 1


def screen_t_star(
    scenario: Mapping[str, Any], t_star_5hz: int
) -> tuple[bool, str, dict[str, Any]]:
    """Full plan-4.2 eligibility for ONE candidate t_star. Pure, pre-outcome."""
    tracks = scenario.get("tracks", {})
    metadata = scenario.get("metadata", {})
    sdc_id = str(metadata.get("sdc_id", ""))
    if not sdc_id or sdc_id not in tracks:
        return False, REASON_NO_SDC, {}
    sdc = tracks[sdc_id]
    t_native = 2 * int(t_star_5hz)
    length = int(scenario.get("length", 0))
    if not sdc_prefix_valid(sdc, t_native):
        return False, REASON_SDC_PREFIX_INVALID, {}
    if not future_capacity_ok(length, t_star_5hz):
        return False, REASON_NO_FUTURE_CAPACITY, {}
    graph = build_lane_graph(scenario.get("map_features", {}))
    if not graph:
        return False, REASON_NO_DRIVABLE_LANE, {}
    ego_pos = track_position_at(sdc, t_native)
    ego_heading = track_heading_at(sdc, t_native)
    ego_lane = find_ego_lane(graph, ego_pos, ego_heading)
    if ego_lane is None:
        return False, REASON_NO_EGO_LANE, {}
    lane_id, station = ego_lane
    left = adjacent_lane(graph, lane_id, "left", station)
    if left is None:
        return False, REASON_NO_LEFT_LANE, {}
    right = adjacent_lane(graph, lane_id, "right", station)
    if right is None:
        return False, REASON_NO_RIGHT_LANE, {}
    dynamic_ids = [
        tid
        for tid, track in tracks.items()
        if str(tid) != sdc_id
        and isinstance(track, Mapping)
        and track.get("type") == "VEHICLE"
        and track_valid_at(track, t_native)
        and track_is_dynamic_pre_t_star(track, t_native)
    ]
    if len(dynamic_ids) < MIN_DYNAMIC_NON_EGO:
        return False, REASON_FEW_DYNAMIC_NON_EGO, {}
    info = {
        "sdc_id": sdc_id,
        "ego_lane_id": lane_id,
        "ego_station_index": station,
        "left_lane_id": left,
        "right_lane_id": right,
        "dynamic_non_ego_ids": sorted(str(t) for t in dynamic_ids),
    }
    return True, "eligible", info


def select_t_star(
    scenario: Mapping[str, Any],
    scenario_id: str,
    role_salt: str,
    candidates: Sequence[int] = T_STAR_CANDIDATES,
) -> tuple[int | None, str, dict[str, Any], list[dict[str, Any]]]:
    """Plan 4.2: first eligible candidate in the frozen per-scenario hash order."""
    from ac_twin.stage4d_external_split import t_star_candidate_order

    attempts: list[dict[str, Any]] = []
    for t in t_star_candidate_order(scenario_id, role_salt, candidates):
        ok, reason, info = screen_t_star(scenario, t)
        attempts.append({"t_star": int(t), "ok": bool(ok), "reason": reason})
        if ok:
            return int(t), "eligible", info, attempts
    return None, "no_eligible_t_star", {}, attempts


def select_eval_entities(
    scenario: Mapping[str, Any], t_star_5hz: int, dynamic_non_ego_ids: Sequence[str]
) -> list[str]:
    """Plan 5.4: ego + 4 nearest dynamic non-ego at t_star, (distance, track_id)
    tie-break. Returns [sdc_id, id1..id4] (slot 0 = ego)."""
    metadata = scenario.get("metadata", {})
    sdc_id = str(metadata.get("sdc_id", ""))
    tracks = scenario["tracks"]
    t_native = 2 * int(t_star_5hz)
    ego_pos = track_position_at(tracks[sdc_id], t_native)
    ranked = sorted(
        (
            (
                float(np.hypot(*(track_position_at(tracks[tid], t_native) - ego_pos))),
                str(tid),
            )
            for tid in dynamic_non_ego_ids
        ),
        key=lambda pair: (pair[0], pair[1]),
    )
    selected = [tid for _dist, tid in ranked[: OBJECT_CAPACITY - 1]]
    if len(selected) != OBJECT_CAPACITY - 1:
        raise ValueError("select_eval_entities requires exactly >=4 candidates")
    return [sdc_id] + selected


def build_group_frame(
    scenario: Mapping[str, Any], t_star_5hz: int, ego_lane_id: str
) -> MapFrenetFrame:
    """The group's common Map-Frenet frame (plan 5.5): reference from the ego lane
    bundle, origin at the ego t_star projection. Built from pre-t_star data only."""
    graph = build_lane_graph(scenario.get("map_features", {}))
    if ego_lane_id not in graph:
        raise ValueError(f"ego lane {ego_lane_id} missing from lane graph")
    polyline = build_reference_polyline(graph, ego_lane_id)
    reference = PolylineFrame(polyline)
    sdc_id = str(scenario.get("metadata", {}).get("sdc_id", ""))
    ego_xy = track_position_at(scenario["tracks"][sdc_id], 2 * int(t_star_5hz))
    return MapFrenetFrame(reference, ego_xy)


# ==================================================================================
# Factual plan sampling (plan 5.2) and the 8-dim feature assembly
# ==================================================================================


def action_rng_root(scenario_id: str, role_salt: str) -> int:
    """Per-group independent action RNG root: first 8 bytes of
    SHA256(role_salt + scenario_id + '|action_plan') as uint64 (frozen)."""
    digest = hashlib.sha256(
        (role_salt + scenario_id + "|action_plan").encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def sample_factual_plan(
    scenario_id: str, role_salt: str, plan_len: int = FUTURE_LEN
) -> np.ndarray:
    """20-step factual meta-action plan, R2 frozen weights (plan 5.2)."""
    seed_seq = np.random.SeedSequence(action_rng_root(scenario_id, role_salt))
    rng = np.random.Generator(np.random.PCG64(seed_seq))
    return rng.choice(len(ACTION_WEIGHTS), size=int(plan_len), p=ACTION_WEIGHTS).astype(
        np.int64
    )


def states_to_features(
    frame: MapFrenetFrame,
    xy: np.ndarray,
    heading: np.ndarray,
    velocity: np.ndarray,
    presence: np.ndarray,
) -> np.ndarray:
    """Assemble the 8-dim object features for one policy step.

    ``xy`` (N,2), ``heading`` (N,), ``velocity`` (N,2), ``presence`` (N,) bool with
    N = OBJECT_CAPACITY and slot 0 = ego. Returns (N, 8) float64.
    """
    n = xy.shape[0]
    s, d, v_s, v_d, cos_h, sin_h = frame.transform(xy, heading, velocity)
    out = np.zeros((n, N_OBJECT_FEATURES), dtype=np.float64)
    is_ego = np.zeros(n, dtype=np.float64)
    is_ego[0] = 1.0
    out[:, 0] = np.asarray(presence, dtype=np.float64)
    out[:, 1] = is_ego
    out[:, 2] = np.where(presence, s, 0.0)
    out[:, 3] = np.where(presence, d, 0.0)
    out[:, 4] = np.where(presence, v_s, 0.0)
    out[:, 5] = np.where(presence, v_d, 0.0)
    out[:, 6] = np.where(presence, cos_h, 0.0)
    out[:, 7] = np.where(presence, sin_h, 0.0)
    return out


__all__ = [
    "ACTION_WEIGHTS",
    "DRIVABLE_LANE_TYPES",
    "EGO_LANE_MAX_HEADING_DIFF_RAD",
    "EGO_LANE_MAX_LATERAL_M",
    "FEATURE_NAMES",
    "FUTURE_LEN",
    "HISTORY_LEN",
    "LANE_GRAPH_MAX_HOPS",
    "MIN_DYNAMIC_NON_EGO",
    "MapFrenetFrame",
    "NATIVE_PER_POLICY",
    "N_ACTIONS",
    "N_OBJECT_FEATURES",
    "OBJECT_CAPACITY",
    "PolylineFrame",
    "REASON_FEW_DYNAMIC_NON_EGO",
    "REASON_FRAME_BUILD_FAILED",
    "REASON_NO_DRIVABLE_LANE",
    "REASON_NO_EGO_LANE",
    "REASON_NO_FUTURE_CAPACITY",
    "REASON_NO_LEFT_LANE",
    "REASON_NO_RIGHT_LANE",
    "REASON_NO_SDC",
    "REASON_SDC_PREFIX_INVALID",
    "STATIC_STD_THRESHOLD_M",
    "T_STAR_CANDIDATES",
    "action_rng_root",
    "adjacent_lane",
    "build_group_frame",
    "build_lane_graph",
    "build_reference_polyline",
    "find_ego_lane",
    "future_capacity_ok",
    "same_direction",
    "sample_factual_plan",
    "screen_t_star",
    "select_eval_entities",
    "select_t_star",
    "sdc_prefix_valid",
    "states_to_features",
    "track_is_dynamic_pre_t_star",
    "track_position_at",
    "track_heading_at",
    "track_valid_at",
]
