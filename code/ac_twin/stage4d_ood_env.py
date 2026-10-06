"""Stage 4D D0/D1 OOD profile environment (plan §4).

Plan ``AC-TWIN-RFER-STAGE4D-D0D1-OOD-2026-09-20-V1.0``. This module freezes the three
pre-registered Merge generator-mismatch profiles and the deterministic perturbation
hooks applied to ``merge-v0`` AFTER ``env.reset(seed=group_seed)`` and BEFORE the
eleven-branch rollout:

* ``BEHAVIOR_OOD`` — keep the in-distribution (ID) initial geometry; perturb the four
  non-ego IDM/MOBIL parameter sets to 2 aggressive + 2 cautious, assigned by a
  group-seed-stable permutation (plan §4.1).
* ``GEOMETRY_OOD`` — keep the ID IDM/MOBIL parameters; deterministically rewrite the
  initial highway/merging positions and speeds (plan §4.2).
* ``JOINT_OOD`` — apply both (plan §4.3).

Every draw is a pure function of the group seed via a dedicated ``SeedSequence`` stream,
so all eleven branches of one group see bit-identical types and geometry. The ego
vehicle is never modified. highway_env is NOT imported at module top (pygame fails under
pytest ``-W error``); the env hooks operate on the duck-typed vehicle/road interface that
``merge-v0`` exposes, so unit tests use a fake env.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# --- the three pre-registered profiles and the §5.3 selection priority -------------------
PROFILES = ("BEHAVIOR_OOD", "GEOMETRY_OOD", "JOINT_OOD")
PROFILE_PRIORITY = ("JOINT_OOD", "BEHAVIOR_OOD", "GEOMETRY_OOD")

# --- §4.1 behavior parameter tables (IDM/MOBIL instance attributes) ----------------------
BEHAVIOR_PARAM_NAMES = (
    "DISTANCE_WANTED",
    "TIME_WANTED",
    "COMFORT_ACC_MAX",
    "COMFORT_ACC_MIN",
    "POLITENESS",
    "LANE_CHANGE_MIN_ACC_GAIN",
    "LANE_CHANGE_MAX_BRAKING_IMPOSED",
)
AGGRESSIVE_PARAMS = {
    "DISTANCE_WANTED": 5.0,
    "TIME_WANTED": 0.8,
    "COMFORT_ACC_MAX": 4.0,
    "COMFORT_ACC_MIN": -6.0,
    "POLITENESS": 0.0,
    "LANE_CHANGE_MIN_ACC_GAIN": 0.10,
    "LANE_CHANGE_MAX_BRAKING_IMPOSED": 3.0,
}
CAUTIOUS_PARAMS = {
    "DISTANCE_WANTED": 15.0,
    "TIME_WANTED": 2.2,
    "COMFORT_ACC_MAX": 2.0,
    "COMFORT_ACC_MIN": -3.0,
    "POLITENESS": 0.5,
    "LANE_CHANGE_MIN_ACC_GAIN": 0.30,
    "LANE_CHANGE_MAX_BRAKING_IMPOSED": 1.5,
}
#: in-distribution reference (highway-env 1.10.2 IDMVehicle defaults; DISTANCE_WANTED is
#: 5.0 + Vehicle.LENGTH = 5.0 + 5.0 = 10.0). Documentation/audit only; never applied.
ID_REFERENCE_PARAMS = {
    "DISTANCE_WANTED": 10.0,
    "TIME_WANTED": 1.5,
    "COMFORT_ACC_MAX": 3.0,
    "COMFORT_ACC_MIN": -5.0,
    "POLITENESS": 0.0,
    "LANE_CHANGE_MIN_ACC_GAIN": 0.20,
    "LANE_CHANGE_MAX_BRAKING_IMPOSED": 2.0,
}
#: parameters left at their in-distribution values for every profile (plan §4.1).
UNCHANGED_BEHAVIOR_PARAMS = {"ACC_MAX": 6.0, "DELTA": 4.0, "LANE_CHANGE_DELAY": 1.0}

N_BEHAVIOR_VEHICLES = 4   # the four non-ego vehicles of merge-v0
N_AGGRESSIVE = 2
N_CAUTIOUS = 2

# --- §4.2 geometry bases and uniform jitter half-widths ----------------------------------
HIGHWAY_POSITIONS_BASE = (78.0, 58.0, 12.0)     # m, + Uniform[-8, 8]
HIGHWAY_POSITIONS_JITTER = 8.0
HIGHWAY_SPEEDS_BASE = (27.0, 33.0, 29.0)        # m/s, + Uniform[-2, 2]
HIGHWAY_SPEEDS_JITTER = 2.0
MERGING_X_BASE = 125.0                          # m, + Uniform[-5, 5]
MERGING_X_JITTER = 5.0
MERGING_SPEED_BASE = 17.0                       # m/s, + Uniform[-1, 1]
MERGING_SPEED_JITTER = 1.0
MERGING_TARGET_SPEED = 30.0                     # m/s (fixed)

#: dedicated deterministic streams for the group-seed-derived draws. These are internal
#: derivations (``SeedSequence([salt, group_seed])``); they never consume the global
#: candidate-seed budget and never alias the env's own RNG stream.
BEHAVIOR_ASSIGN_STREAM_ENTROPY = 2026105101
GEOMETRY_STREAM_ENTROPY = 2026105102

#: lane-prefix tokens identifying the merging ramp in merge-v0 (j->k->b->c).
_MERGING_LANE_PREFIXES = ("j", "k")


def _geometry_rng(group_seed: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([GEOMETRY_STREAM_ENTROPY, int(group_seed)]))


def _assignment_rng(group_seed: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([BEHAVIOR_ASSIGN_STREAM_ENTROPY, int(group_seed)]))


# --- behavior assignment (§4.1) ------------------------------------------------------------


def assign_vehicle_behaviors(group_seed: int, entity_ids: tuple[str, ...]) -> dict[str, str]:
    """Assign the four non-ego vehicles 2 aggressive + 2 cautious, stably per group seed.

    ``entity_ids`` are the four non-ego entity ids (strings or ints; ints are normalised
    to their string form). The permutation is drawn from a dedicated
    ``SeedSequence([salt, group_seed])`` stream over the SORTED id set, so the assignment
    is identical across the eleven branches of one group and independent of the caller's
    id ordering. Exactly four distinct ids are required.
    """
    ids = tuple(str(e) for e in entity_ids)
    if len(ids) != N_BEHAVIOR_VEHICLES:
        raise ValueError(
            f"exactly {N_BEHAVIOR_VEHICLES} non-ego entity ids are required; "
            f"got {len(ids)}")
    if len(set(ids)) != N_BEHAVIOR_VEHICLES:
        raise ValueError(f"non-ego entity ids must be distinct: {ids}")
    ordered = sorted(ids)
    permutation = _assignment_rng(group_seed).permutation(N_BEHAVIOR_VEHICLES)
    aggressive = {ordered[i] for i in permutation[:N_AGGRESSIVE]}
    return {eid: ("aggressive" if eid in aggressive else "cautious") for eid in ordered}


# --- profile config (§4; deterministic per group seed) -------------------------------------


def _draw_geometry(group_seed: int) -> dict[str, Any]:
    """The deterministic §4.2 geometry draws for one group seed (fixed draw order)."""
    rng = _geometry_rng(group_seed)
    highway_positions = (
        np.asarray(HIGHWAY_POSITIONS_BASE, dtype=float)
        + rng.uniform(-HIGHWAY_POSITIONS_JITTER, HIGHWAY_POSITIONS_JITTER, size=3))
    highway_speeds = (
        np.asarray(HIGHWAY_SPEEDS_BASE, dtype=float)
        + rng.uniform(-HIGHWAY_SPEEDS_JITTER, HIGHWAY_SPEEDS_JITTER, size=3))
    merging_x = MERGING_X_BASE + rng.uniform(-MERGING_X_JITTER, MERGING_X_JITTER)
    merging_speed = (
        MERGING_SPEED_BASE + rng.uniform(-MERGING_SPEED_JITTER, MERGING_SPEED_JITTER))
    return {
        "enabled": True,
        "highway_positions": [float(v) for v in highway_positions],
        "highway_speeds": [float(v) for v in highway_speeds],
        "merging_x": float(merging_x),
        "merging_speed": float(merging_speed),
        "merging_target_speed": float(MERGING_TARGET_SPEED),
    }


def build_profile_config(profile: str, group_seed: int) -> dict:
    """The complete deterministic perturbation spec for one profile + group seed.

    The behavior parameter tables and (when enabled) the geometry draws are pure
    functions of ``group_seed``; the per-vehicle behavior *class* assignment is computed
    separately by :func:`assign_vehicle_behaviors` once the four non-ego entity ids are
    known after reset. Geometry is drawn only for ``GEOMETRY_OOD``/``JOINT_OOD``; behavior
    only for ``BEHAVIOR_OOD``/``JOINT_OOD``.
    """
    profile = str(profile)
    if profile not in PROFILES:
        raise ValueError(
            f"unknown OOD profile {profile!r}; the only pre-registered profiles are "
            f"{list(PROFILES)}")
    behavior_enabled = profile in ("BEHAVIOR_OOD", "JOINT_OOD")
    geometry_enabled = profile in ("GEOMETRY_OOD", "JOINT_OOD")
    return {
        "profile": profile,
        "group_seed": int(group_seed),
        "behavior": {
            "enabled": behavior_enabled,
            "n_vehicles": N_BEHAVIOR_VEHICLES,
            "n_aggressive": N_AGGRESSIVE,
            "n_cautious": N_CAUTIOUS,
            "aggressive": dict(AGGRESSIVE_PARAMS),
            "cautious": dict(CAUTIOUS_PARAMS),
            "id_reference": dict(ID_REFERENCE_PARAMS),
            "unchanged": dict(UNCHANGED_BEHAVIOR_PARAMS),
        },
        "geometry": (_draw_geometry(group_seed) if geometry_enabled
                     else {"enabled": False}),
    }


def validate_profile_config(profile_config: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if a built profile config drifts from the plan-§4 tables."""
    profile = profile_config.get("profile")
    if profile not in PROFILES:
        raise ValueError(f"unknown OOD profile in config: {profile!r}")
    behavior = profile_config.get("behavior", {})
    geometry = profile_config.get("geometry", {})
    expected_behavior = profile in ("BEHAVIOR_OOD", "JOINT_OOD")
    expected_geometry = profile in ("GEOMETRY_OOD", "JOINT_OOD")
    if bool(behavior.get("enabled")) != expected_behavior:
        raise ValueError(f"behavior enablement drifted for profile {profile}")
    if bool(geometry.get("enabled")) != expected_geometry:
        raise ValueError(f"geometry enablement drifted for profile {profile}")
    for name, expected in (("aggressive", AGGRESSIVE_PARAMS),
                           ("cautious", CAUTIOUS_PARAMS),
                           ("id_reference", ID_REFERENCE_PARAMS),
                           ("unchanged", UNCHANGED_BEHAVIOR_PARAMS)):
        if {k: float(v) for k, v in behavior.get(name, {}).items()} != expected:
            raise ValueError(f"behavior.{name} drifted from the plan-§4.1 table")
    if (int(behavior.get("n_aggressive", -1)), int(behavior.get("n_cautious", -1))
            ) != (N_AGGRESSIVE, N_CAUTIOUS):
        raise ValueError("behavior vehicle split must be 2 aggressive + 2 cautious")
    if expected_geometry:
        base_p = np.asarray(HIGHWAY_POSITIONS_BASE)
        base_s = np.asarray(HIGHWAY_SPEEDS_BASE)
        positions = np.asarray(geometry.get("highway_positions", []), dtype=float)
        speeds = np.asarray(geometry.get("highway_speeds", []), dtype=float)
        if positions.shape != (3,) or speeds.shape != (3,):
            raise ValueError("geometry must carry three highway positions and speeds")
        if not np.all(np.abs(positions - base_p) <= HIGHWAY_POSITIONS_JITTER + 1e-9):
            raise ValueError("highway positions outside base ± jitter (plan §4.2)")
        if not np.all(np.abs(speeds - base_s) <= HIGHWAY_SPEEDS_JITTER + 1e-9):
            raise ValueError("highway speeds outside base ± jitter (plan §4.2)")
        if abs(float(geometry.get("merging_x", np.nan)) - MERGING_X_BASE) \
                > MERGING_X_JITTER + 1e-9:
            raise ValueError("merging x outside base ± jitter (plan §4.2)")
        if abs(float(geometry.get("merging_speed", np.nan)) - MERGING_SPEED_BASE) \
                > MERGING_SPEED_JITTER + 1e-9:
            raise ValueError("merging speed outside base ± jitter (plan §4.2)")
        if float(geometry.get("merging_target_speed", np.nan)) != MERGING_TARGET_SPEED:
            raise ValueError("merging target speed must stay 30 m/s (plan §4.2)")


# --- merge-v0 vehicle layout ---------------------------------------------------------------


def split_merge_vehicles(env: Any) -> tuple[Any, tuple[Any, ...], Any]:
    """Split a freshly-reset merge-v0 env into (ego, highway vehicles, merging vehicle).

    At reset, merge-v0 places the ego on the highway right lane, three IDM highway
    vehicles on lanes ``("a", "b", *)`` and one IDM merging vehicle on the ramp
    (``("j", "k", 0)``). The merging vehicle is identified by its ramp lane prefix; the
    highway vehicles keep their creation order (which fixes which §4.2 base position each
    receives). The layout must be exactly 1 ego + 3 highway + 1 merging.
    """
    base = getattr(env, "unwrapped", env)
    if base.road is None or base.vehicle is None:
        raise RuntimeError("environment must be reset before splitting vehicles")
    ego = base.vehicle
    highway: list[Any] = []
    merging: Any = None
    for vehicle in base.road.vehicles:
        if vehicle is ego:
            continue
        lane_index = getattr(vehicle, "lane_index", None)
        if lane_index is not None and str(lane_index[0]) in _MERGING_LANE_PREFIXES:
            if merging is not None:
                raise RuntimeError("merge-v0 must have exactly one merging vehicle")
            merging = vehicle
        else:
            highway.append(vehicle)
    if merging is None:
        raise RuntimeError("no merging vehicle found on the ramp lane")
    if len(highway) != 3:
        raise RuntimeError(
            f"merge-v0 must place exactly three highway vehicles; got {len(highway)}")
    return ego, tuple(highway), merging


# --- env application (reset → perturb → rollout) --------------------------------------------


def _set_on_lane(base: Any, vehicle: Any, longitudinal: float, speed: float) -> None:
    """Deterministically rewrite a vehicle's longitudinal position / speed on its lane."""
    lane = base.road.network.get_lane(vehicle.lane_index)
    longitudinal = float(longitudinal)
    vehicle.position = lane.position(longitudinal, 0.0)
    vehicle.heading = float(lane.heading_at(longitudinal))
    vehicle.speed = float(speed)


def apply_profile_to_env(
    env: Any,
    profile_config: Mapping[str, Any],
    assignment: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Apply one group seed's deterministic OOD perturbation; return the state snapshot.

    Called AFTER ``env.reset(seed=group_seed)`` and BEFORE the branch rollout. The ego is
    never touched. When ``assignment`` is omitted it is derived from the four non-ego
    entity ids via :func:`assign_vehicle_behaviors`; when supplied it must be a valid
    2+2 assignment over exactly those four ids. Behavior is applied by shadowing the
    IDM/MOBIL instance attributes; geometry by rewriting position/speed/target_speed.
    """
    validate_profile_config(profile_config)
    base = getattr(env, "unwrapped", env)
    ego, highway, merging = split_merge_vehicles(env)
    non_ego = (*highway, merging)
    behavior = profile_config["behavior"]
    geometry = profile_config["geometry"]

    if behavior["enabled"]:
        entity_ids = tuple(str(v._ac_twin_entity_id) for v in non_ego)
        if assignment is None:
            assignment = assign_vehicle_behaviors(
                int(profile_config["group_seed"]), entity_ids)
        else:
            assignment = {str(k): str(v) for k, v in assignment.items()}
            expected_ids = set(entity_ids)
            if set(assignment) != expected_ids:
                raise ValueError(
                    "behavior assignment must cover exactly the four non-ego entity ids")
            classes = list(assignment.values())
            if classes.count("aggressive") != N_AGGRESSIVE \
                    or classes.count("cautious") != N_CAUTIOUS:
                raise ValueError("behavior assignment must be 2 aggressive + 2 cautious")
        for vehicle in non_ego:
            if vehicle is ego:  # defensive; ego is never in non_ego
                continue
            klass = assignment[str(vehicle._ac_twin_entity_id)]
            params = behavior["aggressive"] if klass == "aggressive" else behavior["cautious"]
            for name in BEHAVIOR_PARAM_NAMES:
                setattr(vehicle, name, float(params[name]))

    if geometry["enabled"]:
        for vehicle, position, speed in zip(
                highway, geometry["highway_positions"], geometry["highway_speeds"]):
            _set_on_lane(base, vehicle, position, speed)
        _set_on_lane(base, merging, geometry["merging_x"], geometry["merging_speed"])
        merging.target_speed = float(geometry["merging_target_speed"])

    return snapshot_vehicle_state(env)


# --- branch-consistency snapshot -------------------------------------------------------------


def snapshot_vehicle_state(env: Any) -> dict[str, dict[str, Any]]:
    """A hash-comparable snapshot of every vehicle's identity + kinematic + behavior state.

    Keyed by string entity id. Used to prove the eleven branches of one group seed see
    bit-identical types and geometry, and that the ego is never modified.
    """
    base = getattr(env, "unwrapped", env)
    snapshot: dict[str, dict[str, Any]] = {}
    for vehicle in base.road.vehicles:
        eid = str(getattr(vehicle, "_ac_twin_entity_id", id(vehicle)))
        snapshot[eid] = {
            "class": type(vehicle).__name__,
            "is_ego": vehicle is base.vehicle,
            "lane_index": tuple(str(x) for x in getattr(vehicle, "lane_index", ())),
            "position": tuple(float(x) for x in np.asarray(vehicle.position, dtype=float)),
            "speed": float(vehicle.speed),
            "heading": float(getattr(vehicle, "heading", 0.0)),
            "target_speed": float(getattr(vehicle, "target_speed", float("nan"))),
            "behavior": {
                name: float(getattr(vehicle, name))
                for name in BEHAVIOR_PARAM_NAMES if hasattr(vehicle, name)
            },
        }
    return snapshot


def snapshots_identical(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    *,
    atol: float = 1e-9,
) -> bool:
    """True iff two vehicle-state snapshots are identical (within ``atol`` for floats)."""
    if set(left) != set(right):
        return False
    for eid in left:
        a, b = left[eid], right[eid]
        for key in ("class", "is_ego", "lane_index"):
            if a[key] != b[key]:
                return False
        for key in ("speed", "heading", "target_speed"):
            if not _float_equal(a[key], b[key], atol):
                return False
        if not _float_equal(a["position"], b["position"], atol):
            return False
        if set(a["behavior"]) != set(b["behavior"]):
            return False
        for name in a["behavior"]:
            if not _float_equal(a["behavior"][name], b["behavior"][name], atol):
                return False
    return True


def _float_equal(a: Any, b: Any, atol: float) -> bool:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    # NaN == NaN for our purposes (a missing target_speed must compare equal).
    both_nan = np.isnan(a_arr) & np.isnan(b_arr)
    return bool(np.all(both_nan | (np.abs(a_arr - b_arr) <= atol)))


__all__ = [
    "AGGRESSIVE_PARAMS",
    "BEHAVIOR_ASSIGN_STREAM_ENTROPY",
    "BEHAVIOR_PARAM_NAMES",
    "CAUTIOUS_PARAMS",
    "GEOMETRY_STREAM_ENTROPY",
    "HIGHWAY_POSITIONS_BASE",
    "HIGHWAY_POSITIONS_JITTER",
    "HIGHWAY_SPEEDS_BASE",
    "HIGHWAY_SPEEDS_JITTER",
    "ID_REFERENCE_PARAMS",
    "MERGING_SPEED_BASE",
    "MERGING_SPEED_JITTER",
    "MERGING_TARGET_SPEED",
    "MERGING_X_BASE",
    "MERGING_X_JITTER",
    "N_AGGRESSIVE",
    "N_BEHAVIOR_VEHICLES",
    "N_CAUTIOUS",
    "PROFILES",
    "PROFILE_PRIORITY",
    "UNCHANGED_BEHAVIOR_PARAMS",
    "apply_profile_to_env",
    "assign_vehicle_behaviors",
    "build_profile_config",
    "snapshot_vehicle_state",
    "snapshots_identical",
    "split_merge_vehicles",
    "validate_profile_config",
]
