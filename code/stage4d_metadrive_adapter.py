"""Stage 4D-D2 MetaDrive adapter: 11-branch decision-group rollouts in Waymo scenarios.

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` sections 5 and 7.
This module is the ONLY place that touches MetaDrive. It mirrors the frozen R2/Merge
collector contract (``collect_stage4cr2_merge_multiaction``) inside the MetaDrive +
ScenarioNet/Waymo ``training_20s`` platform:

* one scenario -> one decision group; ego prefix (native frames 0 .. 2*t_star) is an
  EXACT kinematic replay of the recorded SDC track (vehicle held static so there is
  zero integration drift; the fork state at t_star is the real observed state);
* from t_star the ego is driven by a persistent meta-action controller with
  highway-env semantics (the R2 semantics): the target lane index and target speed
  are persistent state, modified per 5 Hz policy step by the sampled/overridden
  intents (LANE_LEFT/RIGHT change the committed lane one step per active intent;
  FASTER/SLOWER shift the persistent target speed by +/- DELTA; IDLE holds both);
* non-ego traffic is reactive (``reactive_traffic=True`` -> TrajectoryIDMPolicy);
  vehicles that reach their IDM destination are HELD with a full brake command
  instead of being cleaned, so branch presence is symmetric;
* ``filter_overlapping_car=False`` so spawning does not depend on the diverging ego.

The 8-dim object schema and the common Map-Frenet frame come from
``stage4d_external_features`` (pure numpy). Frame mapping: 5 Hz policy step k <->
native frame 2k; history = policy steps [t_star-9 .. t_star] (native 2k-18 .. 2k),
future = [t_star+1 .. t_star+20] (native 2k+2 .. 2k+40). History features are read
from the recorded track (exact, branch-independent); futures are read from the sim.

Controller gains / speed deltas live in ``ActionContract`` and are frozen into
``action_contract.json`` AFTER the 12-group engineering smoke (tracking correctness
only); they are never tuned on any outcome.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ac_twin import stage4d_external_features as F
from ac_twin.stage4c_objective import ACTION_PLANS
from ac_twin.stage4c_protocol import build_inference_mask, materialize_action_plan

# Intent indices (frozen by ACTION_PLANS comments): 0=LANE_LEFT 1=IDLE 2=LANE_RIGHT
# 3=FASTER 4=SLOWER.
INTENT_LANE_LEFT, INTENT_IDLE, INTENT_LANE_RIGHT, INTENT_FASTER, INTENT_SLOWER = range(5)

NATIVE_DT = 0.1  # MetaDrive default: decision_repeat(5) * physics_world_step(0.02)


@dataclass(frozen=True)
class ActionContract:
    """Controller constants. Defaults are engineering starting points; the frozen
    values are written to action_contract.json after the engineering smoke and then
    bound by hash. All are tracking-correctness parameters, never outcome-tuned.

    Longitudinal control mirrors highway-env / MetaDrive ``IDMPolicy`` (the R2
    reference): the meta-action sets a *persistent* target speed (``v_star`` state,
    initialised to the ego's t_star speed) and an IDM term accelerates toward it while
    braking for the front object, so the ego does not self-collide under FASTER.
    """
    heading_pid: tuple = (1.7, 0.01, 3.5)     # IDMPolicy steering heading PID
    lateral_pid: tuple = (0.3, 0.002, 0.05)   # IDMPolicy steering lateral PID
    delta_speed_kmh: float = 10.8             # 3 m/s per FASTER/SLOWER step (plan 5.1)
    max_target_speed_kmh: float = 60.0        # frozen cap standing in for map speed limit
    min_target_speed_kmh: float = 0.0
    steering_limit: float = 1.0
    throttle_limit: float = 1.0
    # IDM longitudinal (IDMPolicy constants, highway-env derived)
    idm_acc_factor: float = 1.0               # ACC_FACTOR
    idm_deacc_factor: float = -5.0            # DEACC_FACTOR
    idm_delta: float = 10.0                   # velocity-term exponent DELTA
    idm_distance_wanted: float = 10.0         # DISTANCE_WANTED jam gap (m)
    idm_time_wanted: float = 1.5              # TIME_WANTED headway (s)
    idm_max_long_dist: float = 30.0           # front-object search range (m)


DEFAULT_CONTRACT = ActionContract()


def group_seed_from_index(scenario_index: int) -> int:
    """The env seed IS the scenario index (ScenarioEnv selects scenario by seed).
    Using the index seeds the IDM np_random deterministically per group."""
    return int(scenario_index)


# ==================================================================================
# Lane adjacency + driving corridors for meta-control
# ==================================================================================


def build_lane_adjacency(
    graph: Mapping[str, F.LaneInfo],
    ego_lane_id: str,
    left_lane_id: str,
    right_lane_id: str,
) -> dict[str, tuple[str | None, str | None]]:
    """Static committed-lane transition table: lane_id -> (left_id, right_id).

    The ego lane's neighbours are the station-aware eligibility results (plan 4.2);
    every other lane uses the first same-direction drivable entry in its neighbour
    list (deterministic list order, station-agnostic). Missing neighbour -> None
    (the lane-change intent then holds the current lane, mirroring highway-env's
    clamp at the road edge).
    """
    adjacency: dict[str, tuple[str | None, str | None]] = {}
    for lane_id, info in graph.items():
        adjacency[lane_id] = (
            _first_same_direction_neighbor(graph, info, "left"),
            _first_same_direction_neighbor(graph, info, "right"),
        )
    # pin the ego lane to the eligibility-computed, station-aware neighbours
    adjacency[ego_lane_id] = (left_lane_id, right_lane_id)
    return adjacency


def _first_same_direction_neighbor(
    graph: Mapping[str, F.LaneInfo], info: F.LaneInfo, side: str
) -> str | None:
    neighbors = info.left_neighbors if side == "left" else info.right_neighbors
    for neighbor in neighbors:
        if not isinstance(neighbor, Mapping):
            continue
        nid = str(neighbor.get("feature_id", ""))
        if nid in graph and F.same_direction(info.polyline, graph[nid].polyline):
            return nid
    return None


def build_target_lanes(
    graph: Mapping[str, F.LaneInfo], lane_ids: Sequence[str]
):
    """Build a PointLane driving corridor for each candidate committed lane.

    Returns {lane_id: PointLane}. Imported lazily so this module stays importable
    without MetaDrive for the pure-numpy tests.
    """
    from metadrive.component.lane.point_lane import PointLane

    lanes = {}
    for lane_id in dict.fromkeys(lane_ids):  # dedup, keep order
        info = graph.get(lane_id)
        if info is None:
            continue
        corridor = F.build_reference_polyline(graph, lane_id)
        width = float(np.median(info.polyline[:, 2])) if info.polyline.shape[1] > 2 else 3.5
        lanes[lane_id] = PointLane(
            corridor, width=max(width, 1.0), need_lane_localization=False
        )
    return lanes


# ==================================================================================
# Ego policy: exact prefix replay + persistent meta-action control
# ==================================================================================


class ActwinEgoPolicy:
    """Class placeholder; the real policy is built by ``make_ego_policy_class`` so the
    MetaDrive import stays lazy. See there."""

    pass  # pragma: no cover - documentation shim


def make_ego_policy_class():
    """Build the ego policy class (MetaDrive imported lazily)."""
    from metadrive.policy.base_policy import BasePolicy
    from metadrive.scenario.parse_object_state import parse_object_state
    from metadrive.component.vehicle.PID_controller import PIDController
    from metadrive.utils.math import wrap_to_pi

    class _ActwinEgoPolicy(BasePolicy):
        """Prefix kinematic replay (static, zero-drift) then PID meta-control."""

        DEBUG_MARK_COLOR = (255, 180, 0, 255)

        def __init__(self, control_object, random_seed=None, config=None):
            super().__init__(control_object=control_object, random_seed=random_seed,
                             config=config)
            self._ctx: dict[str, Any] | None = None

        # -- context injection (called by the runner after reset) -------------------
        def set_group_context(self, ctx: Mapping[str, Any]) -> None:
            self._ctx = dict(ctx)
            self._committed_lane = ctx["ego_lane_id"]
            self._target_speed_kmh: float | None = None
            self._heading_pid = PIDController(*ctx["contract"].heading_pid)
            self._lateral_pid = PIDController(*ctx["contract"].lateral_pid)
            self._was_static = False

        @property
        def _t_star_native(self) -> int:
            return 2 * int(self._ctx["t_star"])

        def _replay_prefix(self, frame: int):
            sdc = self._ctx["sdc_track"]
            state = parse_object_state(sdc, frame)
            v = self.control_object
            v.set_position(state["position"])
            v.set_velocity(state["velocity"], in_local_frame=False)
            v.set_heading_theta(state["heading"])
            v.set_angular_velocity(state["angular_velocity"])
            if not self._was_static:
                v.set_static(True)
                self._was_static = True
            return [0.0, 0.0]

        def _idm_acceleration(self, v, target_lane, contract: ActionContract) -> float:
            """IDM longitudinal control toward the persistent target speed, braking for
            the front object on the committed lane (mirrors IDMPolicy.acceleration)."""
            from metadrive.policy.idm_policy import FrontBackObjects  # lazy
            from metadrive.utils.math import not_zero
            try:
                all_objects = v.lidar.get_surrounding_objects(v)
                surround = FrontBackObjects.get_find_front_back_objs_single_lane(
                    all_objects, target_lane, v.position,
                    max_distance=contract.idm_max_long_dist)
                front_obj = surround.front_object()
                front_dist = surround.front_min_distance()
            except Exception:
                front_obj, front_dist = None, contract.idm_max_long_dist

            target = not_zero(self._target_speed_kmh, 0.1)
            acc = contract.idm_acc_factor * (
                1 - np.power(max(v.speed_km_h, 0) / target, contract.idm_delta))
            if front_obj is not None:
                d = not_zero(front_dist, 0.1)
                d0 = contract.idm_distance_wanted
                tau = contract.idm_time_wanted
                ab = -contract.idm_acc_factor * contract.idm_deacc_factor
                dv = float(np.dot(v.velocity_km_h - front_obj.velocity_km_h, v.heading))
                d_star = d0 + v.speed_km_h * tau + v.speed_km_h * dv / (2 * np.sqrt(ab))
                acc -= contract.idm_acc_factor * (d_star / d) ** 2
            return float(acc)

        def _meta_control(self):
            ctx = self._ctx
            v = self.control_object
            if self._was_static:
                v.set_static(False)
                self._was_static = False
            n = int(self.episode_step)
            j = (n - self._t_star_native + 1) // 2  # future policy step, 1..20
            j = min(max(j, 1), len(ctx["plan"]))
            intent = int(ctx["plan"][j - 1])

            # persistent target speed state v_star (highway-env Target semantics):
            # FASTER/SLOWER shift it by +-3 m/s per active step; IDLE/LANE hold it.
            if self._target_speed_kmh is None:
                self._target_speed_kmh = float(v.speed_km_h)
            contract: ActionContract = ctx["contract"]
            if intent == INTENT_FASTER:
                self._target_speed_kmh = min(contract.max_target_speed_kmh,
                                             self._target_speed_kmh + contract.delta_speed_kmh)
            elif intent == INTENT_SLOWER:
                self._target_speed_kmh = max(contract.min_target_speed_kmh,
                                             self._target_speed_kmh - contract.delta_speed_kmh)

            # persistent committed lane (per-step transition, clamped at road edge)
            if intent in (INTENT_LANE_LEFT, INTENT_LANE_RIGHT):
                side = 0 if intent == INTENT_LANE_LEFT else 1
                nxt = ctx["adjacency"].get(self._committed_lane, (None, None))[side]
                if nxt is not None and nxt in ctx["target_lanes"]:
                    self._committed_lane = nxt

            target_lane = ctx["target_lanes"].get(self._committed_lane)
            if target_lane is None:
                # defensive: unknown lane -> straight hold, cruise
                steering = 0.0
            else:
                long, lat = target_lane.local_coordinates(v.position)
                lane_heading = target_lane.heading_theta_at(long + 1)
                steering = self._heading_pid.get_result(
                    -wrap_to_pi(lane_heading - v.heading_theta))
                steering += self._lateral_pid.get_result(-lat)
            throttle = self._idm_acceleration(v, target_lane, contract)
            steering = float(np.clip(steering, -contract.steering_limit,
                                     contract.steering_limit))
            throttle = float(np.clip(throttle, -contract.throttle_limit,
                                     contract.throttle_limit))
            self.action_info["intent"] = intent
            self.action_info["committed_lane"] = self._committed_lane
            self.action_info["target_speed_kmh"] = self._target_speed_kmh
            return [steering, throttle]

        def act(self, agent_id=None, *args, **kwargs):
            if self._ctx is None:
                return [0.0, 0.0]
            if int(self.episode_step) <= self._t_star_native:
                return self._replay_prefix(int(self.episode_step))
            return self._meta_control()

    return _ActwinEgoPolicy


def make_traffic_manager_class():
    """ScenarioTrafficManager that HOLDS IDM vehicles at destination (no clean)."""
    from metadrive.manager.scenario_traffic_manager import ScenarioTrafficManager
    from metadrive.policy.idm_policy import TrajectoryIDMPolicy

    class _ActwinTrafficManager(ScenarioTrafficManager):
        def before_step(self, *args, **kwargs):
            self._obj_to_clean_this_frame = []
            for v in self.spawned_objects.values():
                if self.engine.has_policy(v.id, TrajectoryIDMPolicy):
                    p = self.engine.get_policy(v.name)
                    if p.arrive_destination:
                        # hold in place (full brake) instead of cleaning, so branch
                        # presence stays symmetric across the group's 11 branches.
                        v.before_step([0.0, -1.0])
                    else:
                        do_speed_control = (
                            self.episode_step % self.IDM_ACT_BATCH_SIZE == p.policy_index
                        )
                        v.before_step(p.act(do_speed_control))

    return _ActwinTrafficManager


def make_env_class():
    """ScenarioEnv wiring the Actwin ego policy + traffic manager."""
    from metadrive.envs.scenario_env import ScenarioEnv

    ego_policy = make_ego_policy_class()
    traffic_manager = make_traffic_manager_class()

    class _ActwinScenarioEnv(ScenarioEnv):
        def setup_engine(self):
            super().setup_engine()
            self.engine.update_manager("traffic_manager", traffic_manager())

    _ActwinScenarioEnv.ACTWIN_EGO_POLICY = ego_policy
    return _ActwinScenarioEnv


def build_env(db_dir: str, num_scenarios: int):
    """Create the frozen-config scenario env (single instance per process)."""
    env_cls = make_env_class()
    config = dict(
        data_directory=str(db_dir),
        start_scenario_index=0,
        num_scenarios=int(num_scenarios),
        sequential_seed=False,
        agent_policy=env_cls.ACTWIN_EGO_POLICY,
        reactive_traffic=True,
        filter_overlapping_car=False,
        no_static_vehicles=False,
        no_light=False,
        store_map=True,
        store_data=True,
        use_render=False,
        manual_control=False,
        horizon=None,
        decision_repeat=5,
        physics_world_step_size=NATIVE_DT / 5.0,
        show_policy_mark=False,
        # never terminate early: the runner steps a fixed number of frames
        out_of_route_done=False,
        crash_vehicle_done=False,
        crash_object_done=False,
        crash_human_done=False,
        relax_out_of_road_done=True,
        max_lateral_dist=1e6,
        truncate_as_terminate=False,
    )
    return env_cls(config)


# ==================================================================================
# Group context + 11-branch runner
# ==================================================================================


def scenario_id_to_index(db_dir: str) -> dict[str, int]:
    """Map scenario_id -> ScenarioEnv seed index, using the frozen dataset_summary
    ordering (``list(summary_dict.keys())``)."""
    import pickle
    from pathlib import Path

    with open(Path(db_dir) / "dataset_summary.pkl", "rb") as fh:
        summary = pickle.load(fh)
    lookup = list(summary.keys())  # seed index = position in this list
    out = {}
    for idx, file_name in enumerate(lookup):
        sid = str(summary[file_name].get("scenario_id",
                                         file_name.replace("sd_waymo_v1.2_", "").replace(".pkl", "")))
        out[sid] = idx
    return out


@dataclass
class GroupContext:
    scenario_id: str
    t_star: int
    ego_lane_id: str
    left_lane_id: str
    right_lane_id: str
    eval_track_ids: list[str]          # [sdc, e1..e4]
    factual_plan: np.ndarray           # (20,) intents
    frame: Any                         # MapFrenetFrame
    graph: Mapping[str, F.LaneInfo]
    adjacency: Mapping[str, tuple]
    target_lanes: Mapping[str, Any]
    contract: ActionContract = field(default=ActionContract)


def prepare_group_context(
    scenario: Mapping[str, Any],
    scenario_id: str,
    role_salt: str,
    contract: ActionContract = DEFAULT_CONTRACT,
) -> GroupContext:
    """Eligibility -> t_star -> frame -> entities -> target lanes (pre-outcome).

    Raises ValueError with a reason code if the scenario is ineligible; the
    collector screens eligibility BEFORE calling this, so a raise here is an
    engineering invariant violation, not a rejection path.
    """
    t_star, reason, info, _attempts = F.select_t_star(scenario, scenario_id, role_salt)
    if t_star is None:
        raise ValueError(f"scenario {scenario_id} ineligible: {reason}")
    graph = F.build_lane_graph(scenario.get("map_features", {}))
    frame = F.build_group_frame(scenario, t_star, info["ego_lane_id"])
    eval_ids = F.select_eval_entities(scenario, t_star, info["dynamic_non_ego_ids"])
    factual_plan = F.sample_factual_plan(scenario_id, role_salt)
    adjacency = build_lane_adjacency(
        graph, info["ego_lane_id"], info["left_lane_id"], info["right_lane_id"])
    candidate_lanes = sorted(adjacency.keys())
    target_lanes = build_target_lanes(graph, candidate_lanes)
    return GroupContext(
        scenario_id=scenario_id,
        t_star=t_star,
        ego_lane_id=info["ego_lane_id"],
        left_lane_id=info["left_lane_id"],
        right_lane_id=info["right_lane_id"],
        eval_track_ids=eval_ids,
        factual_plan=factual_plan,
        frame=frame,
        graph=graph,
        adjacency=adjacency,
        target_lanes=target_lanes,
        contract=contract,
    )


def history_features_from_track(
    scenario: Mapping[str, Any], ctx: GroupContext
) -> np.ndarray:
    """(HISTORY_LEN, OBJECT_CAPACITY, 8) history from the RECORDED tracks (exact).

    History frames are 5 Hz steps [t_star-9 .. t_star] -> native 2k-18 .. 2k (step 2).
    This is branch-independent ground truth and is never read from the sim.
    """
    tracks = scenario["tracks"]
    t_star = ctx.t_star
    frames = [2 * (t_star - 9) + 2 * i for i in range(F.HISTORY_LEN)]
    out = np.zeros((F.HISTORY_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES), dtype=np.float64)
    for fi, native in enumerate(frames):
        xy = np.zeros((F.OBJECT_CAPACITY, 2))
        heading = np.zeros(F.OBJECT_CAPACITY)
        vel = np.zeros((F.OBJECT_CAPACITY, 2))
        presence = np.zeros(F.OBJECT_CAPACITY, dtype=bool)
        for slot, tid in enumerate(ctx.eval_track_ids):
            track = tracks.get(tid)
            if track is None or not F.track_valid_at(track, native):
                continue
            presence[slot] = True
            xy[slot] = F.track_position_at(track, native)
            heading[slot] = F.track_heading_at(track, native)
            vel[slot] = np.asarray(track["state"]["velocity"][native], dtype=np.float64)[:2]
        out[fi] = F.states_to_features(ctx.frame, xy, heading, vel, presence)
    return out


def _read_entity_sim_state(env, traffic_manager, track_id, ego_obj_id):
    """(xy, heading, velocity, present) for one entity from the live sim."""
    if track_id == "__ego__":
        obj = env.agent
    else:
        obj_id = traffic_manager.scenario_id_to_obj_id.get(track_id)
        if obj_id is None:
            return None
        try:
            found = env.engine.get_object(obj_id)  # engine returns {id: obj}
            obj = found.get(obj_id) if isinstance(found, dict) else found
        except Exception:  # noqa: BLE001 - object may be cleaned this frame
            return None
        if obj is None:
            return None
    pos = np.asarray(obj.position, dtype=np.float64)[:2]
    heading = float(obj.heading_theta)
    vel = np.asarray(obj.velocity, dtype=np.float64)[:2]
    return pos, heading, vel, True


def run_single_branch(
    env,
    scenario_index: int,
    ctx: GroupContext,
    plan: np.ndarray,
) -> dict[str, Any]:
    """Roll out ONE branch (factual or one counterfactual) and record the future.

    Returns future features (FUTURE_LEN, OBJECT_CAPACITY, 8) plus the raw world-frame
    ego trajectory and per-step diagnostics. Steps native frames 0 .. 2*t_star+40.
    """
    from metadrive.constants import DEFAULT_AGENT

    t_star_native = 2 * ctx.t_star
    last_native = t_star_native + 2 * F.FUTURE_LEN

    env.reset(seed=int(scenario_index))
    ego_obj_id = env.engine.agent_manager.agent_to_object(DEFAULT_AGENT)
    policy = env.engine.get_policy(ego_obj_id)
    traffic_manager = env.engine.traffic_manager
    policy.set_group_context(dict(
        t_star=ctx.t_star,
        sdc_track=env.engine.data_manager.current_scenario["tracks"][
            str(env.engine.data_manager.current_scenario["metadata"]["sdc_id"])],
        ego_lane_id=ctx.ego_lane_id,
        adjacency=ctx.adjacency,
        target_lanes=ctx.target_lanes,
        plan=np.asarray(plan, dtype=np.int64),
        contract=ctx.contract,
    ))

    future = np.zeros((F.FUTURE_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES),
                      dtype=np.float64)
    diag_intent = np.zeros(F.FUTURE_LEN, dtype=np.int64)
    n_steps = 0
    for step in range(1, last_native + 1):
        env.step([0.0, 0.0])
        n_steps = step
        if step > t_star_native and (step - t_star_native) % 2 == 0:
            fj = (step - t_star_native) // 2  # 1..20
            xy = np.zeros((F.OBJECT_CAPACITY, 2))
            heading = np.zeros(F.OBJECT_CAPACITY)
            vel = np.zeros((F.OBJECT_CAPACITY, 2))
            presence = np.zeros(F.OBJECT_CAPACITY, dtype=bool)
            for slot, tid in enumerate(ctx.eval_track_ids):
                key = "__ego__" if slot == 0 else tid
                st = _read_entity_sim_state(env, traffic_manager, key, ego_obj_id)
                if st is None:
                    continue
                pos, h, v, _ = st
                presence[slot] = True
                xy[slot] = pos
                heading[slot] = h
                vel[slot] = v
            future[fj - 1] = F.states_to_features(ctx.frame, xy, heading, vel, presence)
            diag_intent[fj - 1] = int(policy.action_info.get("intent", -1))
    return {
        "future": future,
        "diag_intent": diag_intent,
        "n_native_steps": n_steps,
        "completed": n_steps >= last_native,
    }


def run_group(
    env,
    scenario: Mapping[str, Any],
    scenario_index: int,
    ctx: GroupContext,
) -> dict[str, Any]:
    """Run the full 11-branch decision group and assemble the R2-contract tensors.

    Branches share the same scenario reset seed (``scenario_index``), so the prefix
    and the IDM randomness are identical across branches; only the meta-action plan
    differs. Determinism across reruns is pinned by the pilot's double-replay gate.
    """
    history = history_features_from_track(scenario, ctx)
    factual = run_single_branch(env, scenario_index, ctx, ctx.factual_plan)
    cf_futures = np.zeros(
        (F.N_ACTIONS, F.FUTURE_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES),
        dtype=np.float64)
    cf_completed = np.ones(F.N_ACTIONS, dtype=bool)
    for ai, (action, duration) in enumerate(ACTION_PLANS):
        plan = materialize_action_plan(ctx.factual_plan, action, duration)
        res = run_single_branch(env, scenario_index, ctx, plan)
        cf_futures[ai] = res["future"]
        cf_completed[ai] = res["completed"]

    inference_mask = build_inference_mask(
        history[None], factual["future"][None]
    )[0]  # (FUTURE_LEN, OBJECT_CAPACITY)

    return {
        "object_history": history.astype(np.float32),
        "object_factual_future": factual["future"].astype(np.float32),
        "object_counterfactual_futures": cf_futures.astype(np.float32),
        "factual_actions": np.asarray(ctx.factual_plan, dtype=np.int64),
        "cf_actions": np.asarray([a for a, _ in ACTION_PLANS], dtype=np.int64),
        "cf_durations": np.asarray([d for _, d in ACTION_PLANS], dtype=np.int64),
        "inference_mask": np.asarray(inference_mask, dtype=bool),
        "t_star": np.asarray(ctx.t_star, dtype=np.int64),
        "eval_track_ids": list(ctx.eval_track_ids),
        "ego_lane_id": ctx.ego_lane_id,
        "branches_completed": bool(factual["completed"] and cf_completed.all()),
        "factual_diag_intent": factual["diag_intent"],
    }


__all__ = [
    "ACTION_PLANS",
    "ActionContract",
    "DEFAULT_CONTRACT",
    "GroupContext",
    "INTENT_FASTER",
    "INTENT_IDLE",
    "INTENT_LANE_LEFT",
    "INTENT_LANE_RIGHT",
    "INTENT_SLOWER",
    "NATIVE_DT",
    "build_env",
    "build_lane_adjacency",
    "build_target_lanes",
    "group_seed_from_index",
    "history_features_from_track",
    "make_env_class",
    "prepare_group_context",
    "run_group",
    "run_single_branch",
    "scenario_id_to_index",
]
