"""Blocking tests for the Stage 4D-D2 pure geometry/eligibility/feature layer.

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` sections 4.2,
5.2, 5.4, 5.5 and 12.7. These pin the frozen, pre-outcome screening and framing
rules using synthetic scenarios -- no MetaDrive, no converted data, no rollout.
"""
from __future__ import annotations

import numpy as np
import pytest

from ac_twin import stage4d_external_features as F

ROLE_SALT = "AC-TWIN-D2-V0.1|"
LENGTH = 199  # native frames (>= 2*15 + 41 = 71 worst case)


def _lane_polyline(y: float, x0: float = -200.0, x1: float = 400.0, n: int = 61) -> np.ndarray:
    xs = np.linspace(x0, x1, n)
    return np.stack([xs, np.full_like(xs, y), np.zeros_like(xs)], axis=1)


def _lane_feature(fid: int, y: float, *, left=(), right=(), entry=(), exit_=()) -> dict:
    return {
        "type": "LANE_SURFACE_STREET",
        "polyline": _lane_polyline(y),
        "left_neighbor": list(left),
        "right_neighbor": list(right),
        "entry_lanes": list(entry),
        "exit_lanes": list(exit_),
    }


def _neighbor(fid: int) -> dict:
    return {"feature_id": fid, "self_start_index": 0, "self_end_index": 60}


def _three_lane_map() -> dict:
    # lanes 1 (y=0, ego), 2 (y=3.5, left), 0 (y=-3.5, right). Waymo: left neighbor
    # is at higher y when heading is +x.
    return {
        "1": _lane_feature(1, 0.0, left=[_neighbor(2)], right=[_neighbor(0)]),
        "2": _lane_feature(2, 3.5, right=[_neighbor(1)]),
        "0": _lane_feature(0, -3.5, left=[_neighbor(1)]),
    }


def _moving_track(seed_x: float, speed: float = 8.0, lane_y: float = 0.0) -> dict:
    xs = seed_x + speed * 0.1 * np.arange(LENGTH)
    pos = np.stack([xs, np.full(LENGTH, lane_y), np.zeros(LENGTH)], axis=1)
    vel = np.stack([np.full(LENGTH, speed), np.zeros(LENGTH)], axis=1)
    return {
        "type": "VEHICLE",
        "metadata": {},
        "state": {
            "position": pos,
            "heading": np.zeros(LENGTH),
            "velocity": vel,
            "valid": np.ones(LENGTH, dtype=bool),
            "length": np.full(LENGTH, 4.5),
            "width": np.full(LENGTH, 2.0),
            "height": np.full(LENGTH, 1.5),
        },
    }


def _scenario(sdc_id: str = "100", *, n_dynamic: int = 5) -> dict:
    tracks = {sdc_id: _moving_track(0.0, speed=8.0)}
    for i in range(n_dynamic):
        tracks[str(200 + i)] = _moving_track(20.0 + 15.0 * i, speed=7.0 + i, lane_y=0.0)
    return {
        "id": "scen_test",
        "version": "v1.2",
        "length": LENGTH,
        "tracks": tracks,
        "metadata": {"sdc_id": sdc_id, "scenario_id": "scen_test", "ts": list(range(LENGTH))},
        "map_features": _three_lane_map(),
        "dynamic_map_states": {},
    }


# --- PolylineFrame -----------------------------------------------------------------


def test_polyline_project_straight_line():
    frame = F.PolylineFrame(np.array([[0.0, 0.0], [10.0, 0.0]]))
    s, d, inside = frame.project(np.array([[5.0, 2.0], [5.0, -3.0], [10.0, 0.5]]))
    assert np.allclose(s, [5.0, 5.0, 10.0])
    assert np.allclose(d, [2.0, -3.0, 0.5])  # left = +d
    assert inside.all()


def test_polyline_project_extrapolates_beyond_ends():
    frame = F.PolylineFrame(np.array([[0.0, 0.0], [10.0, 0.0]]))
    s, d, inside = frame.project(np.array([[-4.0, 1.0], [14.0, 1.0]]))
    assert np.allclose(s, [-4.0, 14.0])
    assert not inside.any()


def test_polyline_rejects_degenerate():
    with pytest.raises(ValueError):
        F.PolylineFrame(np.array([[1.0, 1.0], [1.0, 1.0]]))


# --- lane graph / neighbors ---------------------------------------------------------


def test_adjacent_lane_left_and_right():
    graph = F.build_lane_graph(_three_lane_map())
    assert F.adjacent_lane(graph, "1", "left", 30) == "2"
    assert F.adjacent_lane(graph, "1", "right", 30) == "0"
    assert F.adjacent_lane(graph, "2", "left", 30) is None


def test_adjacent_lane_respects_station_overlap():
    m = _three_lane_map()
    m["1"]["left_neighbor"] = [{"feature_id": 2, "self_start_index": 40, "self_end_index": 60}]
    graph = F.build_lane_graph(m)
    assert F.adjacent_lane(graph, "1", "left", 30) is None  # station 30 outside [40,60]
    assert F.adjacent_lane(graph, "1", "left", 50) == "2"


def test_adjacent_lane_rejects_opposite_direction():
    m = _three_lane_map()
    m["2"]["polyline"] = m["2"]["polyline"][::-1].copy()  # reversed chord direction
    graph = F.build_lane_graph(m)
    assert F.adjacent_lane(graph, "1", "left", 30) is None


def test_find_ego_lane_aligns_heading():
    graph = F.build_lane_graph(_three_lane_map())
    got = F.find_ego_lane(graph, np.array([50.0, 0.2]), heading=0.05)
    assert got is not None and got[0] == "1"
    # anti-parallel heading is rejected even on the geometric lane
    assert F.find_ego_lane(graph, np.array([50.0, 0.2]), heading=np.pi) is None


# --- eligibility (plan 4.2) ----------------------------------------------------------


def test_screen_t_star_eligible_nominal():
    ok, reason, info = F.screen_t_star(_scenario(), 12)
    assert ok, reason
    assert info["ego_lane_id"] == "1"
    assert info["left_lane_id"] == "2" and info["right_lane_id"] == "0"
    assert len(info["dynamic_non_ego_ids"]) >= F.MIN_DYNAMIC_NON_EGO


def test_screen_t_star_rejects_missing_left_lane():
    sc = _scenario()
    sc["map_features"]["1"]["left_neighbor"] = []
    ok, reason, _ = F.screen_t_star(sc, 12)
    assert not ok and reason == F.REASON_NO_LEFT_LANE


def test_screen_t_star_rejects_few_dynamic():
    sc = _scenario(n_dynamic=2)
    ok, reason, _ = F.screen_t_star(sc, 12)
    assert not ok and reason == F.REASON_FEW_DYNAMIC_NON_EGO


def test_screen_t_star_rejects_short_scenario():
    sc = _scenario()
    sc["length"] = 40  # cannot fit native frame 2*12+40
    ok, reason, _ = F.screen_t_star(sc, 12)
    assert not ok and reason == F.REASON_NO_FUTURE_CAPACITY


def test_screen_t_star_rejects_invalid_sdc_prefix():
    sc = _scenario()
    sc["tracks"]["100"]["state"]["valid"][5] = False
    ok, reason, _ = F.screen_t_star(sc, 12)
    assert not ok and reason == F.REASON_SDC_PREFIX_INVALID


def test_static_vehicle_not_counted_as_dynamic():
    sc = _scenario()
    # overwrite one non-ego with a static (parked) track
    st = sc["tracks"]["200"]["state"]
    st["position"][:] = st["position"][0]
    st["velocity"][:] = 0.0
    assert not F.track_is_dynamic_pre_t_star(sc["tracks"]["200"], 24)


def test_select_t_star_picks_first_eligible_in_hash_order():
    sc = _scenario()
    t, reason, info, attempts = F.select_t_star(sc, "scen_test", ROLE_SALT)
    assert t in F.T_STAR_CANDIDATES and reason == "eligible"
    # attempts are exactly the hash order, all eligible -> single attempt
    assert len(attempts) == 1 and attempts[0]["t_star"] == t and attempts[0]["ok"]


def test_select_t_star_none_when_all_ineligible():
    sc = _scenario(n_dynamic=0)
    t, reason, _info, attempts = F.select_t_star(sc, "scen_test", ROLE_SALT)
    assert t is None and reason == "no_eligible_t_star"
    assert len(attempts) == len(F.T_STAR_CANDIDATES)
    assert all(not a["ok"] for a in attempts)


# --- entity selection (plan 5.4) ------------------------------------------------------


def test_select_eval_entities_ego_plus_four_nearest():
    sc = _scenario()
    _ok, _r, info = F.screen_t_star(sc, 12)
    ids = F.select_eval_entities(sc, 12, info["dynamic_non_ego_ids"])
    assert ids[0] == "100"  # ego in slot 0
    assert len(ids) == F.OBJECT_CAPACITY
    assert len(set(ids)) == F.OBJECT_CAPACITY


def test_select_eval_entities_deterministic_tie_break():
    sc = _scenario()
    # place two non-ego at identical distance from ego -> track_id breaks the tie
    for tid, x in (("300", 30.0), ("299", 30.0)):
        sc["tracks"][tid] = _moving_track(x, speed=8.0)
    _ok, _r, info = F.screen_t_star(sc, 12)
    ids1 = F.select_eval_entities(sc, 12, info["dynamic_non_ego_ids"])
    ids2 = F.select_eval_entities(sc, 12, info["dynamic_non_ego_ids"])
    assert ids1 == ids2


# --- common frame + features (plan 5.5) ----------------------------------------------


def test_build_group_frame_origin_at_ego_projection():
    sc = _scenario()
    frame = F.build_group_frame(sc, 12, "1")
    sdc = sc["tracks"]["100"]
    ego_xy = F.track_position_at(sdc, 24)
    s, d, _ = frame.project(ego_xy)
    assert abs(float(s)) < 1e-6  # origin == ego projection
    assert abs(float(d)) < 1e-6  # ego is on its lane centreline (y=0)


def test_states_to_features_schema_and_ego_flag():
    sc = _scenario()
    frame = F.build_group_frame(sc, 12, "1")
    n = F.OBJECT_CAPACITY
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 3.5], [30.0, -3.5], [40.0, 0.0]])
    heading = np.zeros(n)
    vel = np.stack([np.full(n, 8.0), np.zeros(n)], axis=1)
    presence = np.array([True, True, True, True, False])
    feats = F.states_to_features(frame, xy, heading, vel, presence)
    assert feats.shape == (n, F.N_OBJECT_FEATURES)
    assert feats[0, 1] == 1.0 and feats[1:, 1].sum() == 0.0  # is_ego only slot 0
    assert feats[4, 0] == 0.0 and np.allclose(feats[4, 2:], 0.0)  # absent -> zeroed
    # ego at origin, straight reference, +x motion -> v_s=8, v_d=0, heading_rel=0
    assert np.allclose(feats[0, 4], 8.0) and np.allclose(feats[0, 5], 0.0)
    assert np.allclose(feats[0, 6], 1.0) and np.allclose(feats[0, 7], 0.0)
    assert tuple(F.FEATURE_NAMES) == (
        "presence", "is_ego", "s", "d", "v_s", "v_d", "cos_heading_rel", "sin_heading_rel",
    )


# --- factual plan sampling (plan 5.2) ---------------------------------------------------


def test_sample_factual_plan_deterministic_and_weighted_support():
    p1 = F.sample_factual_plan("scenA", ROLE_SALT)
    p2 = F.sample_factual_plan("scenA", ROLE_SALT)
    assert np.array_equal(p1, p2)  # deterministic per scenario
    assert p1.shape == (F.FUTURE_LEN,) and p1.dtype == np.int64
    assert set(np.unique(p1)).issubset(set(range(5)))
    p3 = F.sample_factual_plan("scenB", ROLE_SALT)
    assert not np.array_equal(p1, p3)  # independent per scenario id


def test_sample_factual_plan_respects_idle_dominance_over_many():
    plans = np.stack([F.sample_factual_plan(f"s{i}", ROLE_SALT) for i in range(400)])
    freq = np.bincount(plans.ravel(), minlength=5) / plans.size
    # IDLE weight .45 dominates; every frozen weight class appears
    assert freq[1] == freq.max()
    assert (freq > 0.01).all()
