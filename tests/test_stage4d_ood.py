"""Blocking tests for the Stage 4D D0/D1 OOD profile environment (plan §4, §11 item 2).

Plan ``AC-TWIN-RFER-STAGE4D-D0D1-OOD-2026-09-20-V1.0``. These tests prove the three
OOD profiles carry exactly the plan-§4 parameters, the four non-ego vehicles are always
assigned 2 aggressive + 2 cautious by a group-seed-stable permutation, the geometry /
behavior perturbation is identical across the eleven branches of one group seed, and the
ego vehicle is never modified. highway_env is NOT imported here (pygame fails under
``-W error``); the env application is exercised against a duck-typed fake env.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4d_ood_env as ood  # noqa: E402


# --- duck-typed fake env mirroring the merge-v0 layout ---------------------------------


class _FakeLane:
    """A straight lane along +x at a fixed lateral offset (merge-v0 lanes are straight)."""

    def __init__(self, y: float):
        self.y = float(y)

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return np.array([float(longitudinal), self.y + float(lateral)], dtype=float)

    def heading_at(self, longitudinal: float) -> float:
        return 0.0


class _FakeNetwork:
    def __init__(self):
        self._lanes = {
            ("a", "b", 0): _FakeLane(0.0),
            ("a", "b", 1): _FakeLane(4.0),
            ("j", "k", 0): _FakeLane(14.5),
            ("k", "b", 0): _FakeLane(10.0),
        }

    def get_lane(self, lane_index):
        return self._lanes[tuple(lane_index)]


class _FakeVehicle:
    """Minimal vehicle with the kinematic + behavior attributes the OOD hooks touch."""

    def __init__(self, cls_name, lane_index, x, speed, target_speed=np.nan):
        self._cls_name = cls_name
        self.lane_index = tuple(lane_index)
        self.position = np.array([float(x), 0.0], dtype=float)
        self.speed = float(speed)
        self.heading = 0.0
        self.target_speed = float(target_speed)
        # IDM/MOBIL defaults (the "ID 参考" column of plan §4.1).
        self.DISTANCE_WANTED = 10.0
        self.TIME_WANTED = 1.5
        self.COMFORT_ACC_MAX = 3.0
        self.COMFORT_ACC_MIN = -5.0
        self.POLITENESS = 0.0
        self.LANE_CHANGE_MIN_ACC_GAIN = 0.20
        self.LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0
        self.ACC_MAX = 6.0
        self.DELTA = 4.0
        self.LANE_CHANGE_DELAY = 1.0
        self._ac_twin_entity_id = -1


class _FakeRoad:
    def __init__(self, vehicles):
        self.vehicles = list(vehicles)
        self.network = _FakeNetwork()


class _FakeEnv:
    def __init__(self):
        # ego on the highway right lane; 3 highway IDM vehicles; 1 merging IDM vehicle.
        self.ego = _FakeVehicle("MDPVehicle", ("a", "b", 1), 30.0, 30.0)
        self.highway = [
            _FakeVehicle("IDMVehicle", ("a", "b", 0), 90.0, 29.0),
            _FakeVehicle("IDMVehicle", ("a", "b", 1), 70.0, 31.0),
            _FakeVehicle("IDMVehicle", ("a", "b", 0), 5.0, 31.5),
        ]
        self.merging = _FakeVehicle("IDMVehicle", ("j", "k", 0), 110.0, 20.0, 30.0)
        self.vehicle = self.ego
        self.road = _FakeRoad([self.ego, *self.highway, self.merging])
        # deterministic entity ids in creation order (ego=0, then highway, then merging).
        for eid, v in enumerate(self.road.vehicles):
            v._ac_twin_entity_id = eid

    @property
    def unwrapped(self):
        return self


# --- frozen profile / parameter tables (plan §4) ----------------------------------------


def test_profiles_frozen_and_priority():
    assert tuple(ood.PROFILES) == ("BEHAVIOR_OOD", "GEOMETRY_OOD", "JOINT_OOD")
    # plan §5.3 selection priority: JOINT > BEHAVIOR > GEOMETRY.
    assert tuple(ood.PROFILE_PRIORITY) == ("JOINT_OOD", "BEHAVIOR_OOD", "GEOMETRY_OOD")


def test_behavior_param_tables_match_plan():
    aggressive = {
        "DISTANCE_WANTED": 5.0, "TIME_WANTED": 0.8, "COMFORT_ACC_MAX": 4.0,
        "COMFORT_ACC_MIN": -6.0, "POLITENESS": 0.0, "LANE_CHANGE_MIN_ACC_GAIN": 0.10,
        "LANE_CHANGE_MAX_BRAKING_IMPOSED": 3.0,
    }
    cautious = {
        "DISTANCE_WANTED": 15.0, "TIME_WANTED": 2.2, "COMFORT_ACC_MAX": 2.0,
        "COMFORT_ACC_MIN": -3.0, "POLITENESS": 0.5, "LANE_CHANGE_MIN_ACC_GAIN": 0.30,
        "LANE_CHANGE_MAX_BRAKING_IMPOSED": 1.5,
    }
    id_reference = {
        "DISTANCE_WANTED": 10.0, "TIME_WANTED": 1.5, "COMFORT_ACC_MAX": 3.0,
        "COMFORT_ACC_MIN": -5.0, "POLITENESS": 0.0, "LANE_CHANGE_MIN_ACC_GAIN": 0.20,
        "LANE_CHANGE_MAX_BRAKING_IMPOSED": 2.0,
    }
    assert dict(ood.AGGRESSIVE_PARAMS) == aggressive
    assert dict(ood.CAUTIOUS_PARAMS) == cautious
    assert dict(ood.ID_REFERENCE_PARAMS) == id_reference
    assert tuple(ood.BEHAVIOR_PARAM_NAMES) == tuple(aggressive.keys())
    # ACC_MAX / DELTA / LANE_CHANGE_DELAY stay at the in-distribution values.
    assert dict(ood.UNCHANGED_BEHAVIOR_PARAMS) == {
        "ACC_MAX": 6.0, "DELTA": 4.0, "LANE_CHANGE_DELAY": 1.0}
    assert (ood.N_BEHAVIOR_VEHICLES, ood.N_AGGRESSIVE, ood.N_CAUTIOUS) == (4, 2, 2)


def test_geometry_constants_match_plan():
    assert tuple(ood.HIGHWAY_POSITIONS_BASE) == (78.0, 58.0, 12.0)
    assert ood.HIGHWAY_POSITIONS_JITTER == 8.0
    assert tuple(ood.HIGHWAY_SPEEDS_BASE) == (27.0, 33.0, 29.0)
    assert ood.HIGHWAY_SPEEDS_JITTER == 2.0
    assert ood.MERGING_X_BASE == 125.0
    assert ood.MERGING_X_JITTER == 5.0
    assert ood.MERGING_SPEED_BASE == 17.0
    assert ood.MERGING_SPEED_JITTER == 1.0
    assert ood.MERGING_TARGET_SPEED == 30.0


# --- behavior assignment: 2+2, group-seed-stable, order-independent ---------------------


def test_assign_vehicle_behaviors_is_two_plus_two():
    ids = ("0", "1", "2", "3")
    for seed in (1, 2, 123, 2026101000, 987654321):
        assignment = ood.assign_vehicle_behaviors(seed, ids)
        assert set(assignment) == set(ids)
        classes = list(assignment.values())
        assert classes.count("aggressive") == 2
        assert classes.count("cautious") == 2


def test_assign_vehicle_behaviors_stable_and_order_independent():
    ids = ("3", "1", "2", "0")  # deliberately unsorted
    a = ood.assign_vehicle_behaviors(2026101000, ids)
    b = ood.assign_vehicle_behaviors(2026101000, tuple(reversed(ids)))
    c = ood.assign_vehicle_behaviors(2026101000, ("0", "1", "2", "3"))
    assert a == b == c  # same group seed + same id set => identical assignment
    # int entity ids are accepted and normalised to their string form.
    d = ood.assign_vehicle_behaviors(2026101000, (0, 1, 2, 3))
    assert d == c


def test_assign_vehicle_behaviors_rejects_wrong_count():
    with pytest.raises(ValueError):
        ood.assign_vehicle_behaviors(1, ("0", "1", "2"))  # not four non-ego vehicles
    with pytest.raises(ValueError):
        ood.assign_vehicle_behaviors(1, ("0", "1", "2", "3", "4"))
    with pytest.raises(ValueError):
        ood.assign_vehicle_behaviors(1, ("0", "1", "2", "2"))  # duplicate id


# --- build_profile_config: per-profile enablement + deterministic geometry ---------------


def test_build_profile_config_enablement():
    behavior = ood.build_profile_config("BEHAVIOR_OOD", 7)
    geometry = ood.build_profile_config("GEOMETRY_OOD", 7)
    joint = ood.build_profile_config("JOINT_OOD", 7)
    assert behavior["behavior"]["enabled"] is True
    assert behavior["geometry"]["enabled"] is False
    assert geometry["behavior"]["enabled"] is False
    assert geometry["geometry"]["enabled"] is True
    assert joint["behavior"]["enabled"] is True
    assert joint["geometry"]["enabled"] is True
    for cfg in (behavior, geometry, joint):
        assert cfg["group_seed"] == 7
    with pytest.raises(ValueError):
        ood.build_profile_config("SPEED_OOD", 7)  # no fourth profile


def test_build_profile_config_behavior_tables_frozen():
    cfg = ood.build_profile_config("BEHAVIOR_OOD", 11)
    assert cfg["behavior"]["aggressive"] == dict(ood.AGGRESSIVE_PARAMS)
    assert cfg["behavior"]["cautious"] == dict(ood.CAUTIOUS_PARAMS)
    assert cfg["behavior"]["id_reference"] == dict(ood.ID_REFERENCE_PARAMS)
    assert cfg["behavior"]["n_aggressive"] == 2 and cfg["behavior"]["n_cautious"] == 2


def test_geometry_draws_deterministic_and_within_bounds():
    a = ood.build_profile_config("GEOMETRY_OOD", 2026102001)
    b = ood.build_profile_config("GEOMETRY_OOD", 2026102001)
    assert a["geometry"] == b["geometry"]  # identical across the eleven branches
    geo = a["geometry"]
    base_p = np.asarray(ood.HIGHWAY_POSITIONS_BASE)
    base_s = np.asarray(ood.HIGHWAY_SPEEDS_BASE)
    assert np.all(np.abs(np.asarray(geo["highway_positions"]) - base_p)
                  <= ood.HIGHWAY_POSITIONS_JITTER + 1e-9)
    assert np.all(np.abs(np.asarray(geo["highway_speeds"]) - base_s)
                  <= ood.HIGHWAY_SPEEDS_JITTER + 1e-9)
    assert abs(geo["merging_x"] - ood.MERGING_X_BASE) <= ood.MERGING_X_JITTER + 1e-9
    assert abs(geo["merging_speed"] - ood.MERGING_SPEED_BASE) <= ood.MERGING_SPEED_JITTER + 1e-9
    assert geo["merging_target_speed"] == ood.MERGING_TARGET_SPEED
    # a different group seed draws a different (but still in-bounds) geometry.
    other = ood.build_profile_config("GEOMETRY_OOD", 2026102002)["geometry"]
    assert other != geo


def test_validate_profile_config_detects_param_drift():
    cfg = ood.build_profile_config("JOINT_OOD", 3)
    ood.validate_profile_config(cfg)  # clean
    tampered = ood.build_profile_config("JOINT_OOD", 3)
    tampered["behavior"]["aggressive"]["DISTANCE_WANTED"] = 99.0
    with pytest.raises(ValueError):
        ood.validate_profile_config(tampered)
    tampered_geo = ood.build_profile_config("GEOMETRY_OOD", 3)
    tampered_geo["geometry"]["highway_positions"][0] = 1000.0  # outside base±jitter
    with pytest.raises(ValueError):
        ood.validate_profile_config(tampered_geo)


# --- env application: behavior / geometry, ego untouched, branch consistency -------------


def test_split_merge_vehicles_layout():
    env = _FakeEnv()
    ego, highway, merging = ood.split_merge_vehicles(env)
    assert ego is env.ego
    assert len(highway) == 3 and merging is env.merging
    assert all(v is not ego for v in (*highway, merging))


def test_apply_behavior_sets_params_and_leaves_ego():
    env = _FakeEnv()
    before = ood.snapshot_vehicle_state(env)
    ego_id = str(env.ego._ac_twin_entity_id)
    cfg = ood.build_profile_config("BEHAVIOR_OOD", 42)
    ood.apply_profile_to_env(env, cfg)
    after = ood.snapshot_vehicle_state(env)
    # ego never modified (position/speed/behavior identical).
    assert after[ego_id] == before[ego_id]
    # the four non-ego vehicles got exactly 2 aggressive + 2 cautious parameter sets.
    classes = []
    for v in (*env.highway, env.merging):
        if v.DISTANCE_WANTED == ood.AGGRESSIVE_PARAMS["DISTANCE_WANTED"]:
            classes.append("aggressive")
            assert v.TIME_WANTED == 0.8 and v.POLITENESS == 0.0
        else:
            classes.append("cautious")
            assert v.DISTANCE_WANTED == 15.0 and v.TIME_WANTED == 2.2
        # unchanged parameters are never perturbed.
        assert v.ACC_MAX == 6.0 and v.DELTA == 4.0 and v.LANE_CHANGE_DELAY == 1.0
    assert classes.count("aggressive") == 2 and classes.count("cautious") == 2


def test_apply_geometry_rewrites_state_and_leaves_ego():
    env = _FakeEnv()
    ego_id = str(env.ego._ac_twin_entity_id)
    before = ood.snapshot_vehicle_state(env)
    cfg = ood.build_profile_config("GEOMETRY_OOD", 2026102001)
    geo = cfg["geometry"]
    ood.apply_profile_to_env(env, cfg)
    after = ood.snapshot_vehicle_state(env)
    assert after[ego_id] == before[ego_id]  # ego geometry untouched
    # highway vehicles moved to the deterministic positions / speeds.
    for v, px, sp in zip(env.highway, geo["highway_positions"], geo["highway_speeds"]):
        assert abs(float(v.position[0]) - px) < 1e-6
        assert abs(v.speed - sp) < 1e-9
    # merging vehicle moved and its target speed frozen to 30 m/s.
    assert abs(float(env.merging.position[0]) - geo["merging_x"]) < 1e-6
    assert abs(env.merging.speed - geo["merging_speed"]) < 1e-9
    assert env.merging.target_speed == ood.MERGING_TARGET_SPEED
    # IDM/MOBIL params are NOT perturbed in a geometry-only profile.
    for v in (*env.highway, env.merging):
        assert v.DISTANCE_WANTED == ood.ID_REFERENCE_PARAMS["DISTANCE_WANTED"]


def test_branch_consistency_identical_across_eleven_branches():
    group_seed = 2026103002
    cfg = ood.build_profile_config("JOINT_OOD", group_seed)
    snapshots = []
    for _branch in range(11):  # one factual + ten action branches
        env = _FakeEnv()  # a fresh env reset with the same group seed
        ood.apply_profile_to_env(env, cfg)
        snapshots.append(ood.snapshot_vehicle_state(env))
    reference = snapshots[0]
    for snap in snapshots[1:]:
        assert ood.snapshots_identical(reference, snap)


def test_branch_consistency_detects_divergence():
    cfg = ood.build_profile_config("JOINT_OOD", 5)
    env_a = _FakeEnv()
    ood.apply_profile_to_env(env_a, cfg)
    snap_a = ood.snapshot_vehicle_state(env_a)
    env_b = _FakeEnv()
    ood.apply_profile_to_env(env_b, cfg)
    snap_b = ood.snapshot_vehicle_state(env_b)
    assert ood.snapshots_identical(snap_a, snap_b)
    # a single divergent non-ego geometry must be detected.
    env_b.highway[0].position = np.array([999.0, 0.0])
    snap_b_drift = ood.snapshot_vehicle_state(env_b)
    assert not ood.snapshots_identical(snap_a, snap_b_drift)
    # a divergent behavior class must be detected too.
    env_c = _FakeEnv()
    ood.apply_profile_to_env(env_c, cfg)
    env_c.merging.DISTANCE_WANTED = (
        ood.CAUTIOUS_PARAMS["DISTANCE_WANTED"]
        if env_c.merging.DISTANCE_WANTED == ood.AGGRESSIVE_PARAMS["DISTANCE_WANTED"]
        else ood.AGGRESSIVE_PARAMS["DISTANCE_WANTED"])
    assert not ood.snapshots_identical(snap_a, ood.snapshot_vehicle_state(env_c))


def test_apply_rejects_unknown_profile_and_bad_assignment():
    env = _FakeEnv()
    cfg = ood.build_profile_config("BEHAVIOR_OOD", 9)
    bad_assignment = {"0": "aggressive", "1": "aggressive", "2": "aggressive", "3": "cautious"}
    with pytest.raises(ValueError):
        ood.apply_profile_to_env(env, cfg, assignment=bad_assignment)  # 3+1 not 2+2
    with pytest.raises(ValueError):
        ood.apply_profile_to_env(env, cfg, assignment={"0": "aggressive"})  # incomplete
