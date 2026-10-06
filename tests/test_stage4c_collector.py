"""Stage 4C collector tests: shared-state fork, whole-group rejection, quota, resume."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import collect_stage4c_merge_multiaction as coll  # noqa: E402
from ac_twin import stage4c_protocol as s4cp  # noqa: E402


def _config(tmp_path):
    base = {
        "plan_id": s4cp.PLAN_ID, "scene": "merge", "env_id": "merge-v0",
        "experiment_name": "stage4c_test",
        "action_plans": [list(p) for p in s4cp.ACTION_PLANS],
        "cf_actions": list(s4cp.CF_ACTIONS),
        "intervention_durations": list(s4cp.CF_DURATIONS),
        "history_len": 10, "horizon": 20, "t_star_low": 10, "t_star_high": 15,
        "simulation_frequency": 15, "policy_frequency": 5, "vehicles_count": 25,
        "observed_vehicles_count": 5, "object_capacity": 5, "duration": 100,
        "dev_selection_groups": 500, "dev_screen_groups": 300,
        "candidate_ns": [600, 900, 1200, 1800, 2400],
        "dev_seed_entropy": 2026091801, "dev_schedule_size": 5000,
        "confirm_seed_entropy": 2026091901, "confirm_schedule_size": 15000,
        "bootstrap_entropy_screen": 2026091802, "bootstrap_entropy_confirm": 2026092002,
        "power_monte_carlo_entropy": 2026092001,
        "confidence_screen": 0.80, "confidence_confirm": 0.95,
        "power_target": 0.80, "power_shrink_main": 0.75,
        "top_error_threshold": 0.41601440122246675, "primary_constant_w": 0.60,
        "response_predictor_seeds": [101, 202, 303, 404, 505],
        "expert_router_seeds": [1101, 1202, 1303, 1404, 1505],
        "lambda_candidates": [0.0, 0.25, 0.5, 1.0],
        "gate_i_aurc_regret_min": 0.05, "gate_i_regret80_min": 0.03,
        "gate_strata_i_aurc_floor": -0.10, "gate_task_min_nonzero_regret_frac": 0.20,
        "gate_min_positive_pairs": 4,
        "collection_seed": 7,
        "run_dir": str(tmp_path / "run"),
        "data_output": str(tmp_path / "data" / s4cp.PURPOSE_DEV / "dev.npz"),
        "rejection_ledger": str(tmp_path / "data" / s4cp.PURPOSE_DEV / "ledger.jsonl"),
        "checkpoint_every_accepted": 2,
    }
    return base


def _fake_probe(config):
    """Action mapping / probabilities / capacity without building a real env."""
    capacity = int(config["object_capacity"])
    action_ids = np.arange(5, dtype=np.int64)
    action_probs = np.full(5, 0.2, dtype=np.float64)
    return action_ids, action_probs, capacity


def _fingerprints(seed, capacity):
    rng = np.random.default_rng(int(seed))
    return [{
        "entity_id": i, "class": "car", "route": (),
        "position_xy": [float(rng.normal()), float(rng.normal())],
        "heading": 0.0, "speed": 1.0,
    } for i in range(capacity)]


def make_fake_rollout(config, *, fail_cf_index=None, cf_present_mismatch_index=None):
    """Deterministic fake branch rollout; shared prefix across branches of a group."""
    capacity = int(config["object_capacity"])
    horizon = int(config["horizon"])
    calls = []

    def fake(env_id, env_cfg, seed, prefix_actions, future_actions, cap):
        calls.append((int(seed), tuple(int(a) for a in prefix_actions),
                      tuple(int(a) for a in future_actions)))
        prefix_actions = np.asarray(prefix_actions)
        future_actions = np.asarray(future_actions)
        t_star = len(prefix_actions)
        # prefix depends ONLY on (seed, prefix_actions) -> identical across branches.
        prng = np.random.default_rng((int(seed) * 100003 + int(prefix_actions.sum())) % (2**31))
        prefix = prng.normal(size=(t_star + 1, capacity, 8)).astype(np.float32)
        # future depends on the plan so the ten cf branches diverge.
        frng = np.random.default_rng(
            (int(seed) * 7919 + int(prefix_actions.sum()) * 131
             + int(future_actions.sum()) * 17 + len(future_actions)) % (2**31))
        future = prng.normal(size=(horizon, capacity, 8)).astype(np.float32)
        for frames in (prefix, future):
            frames[..., 0] = 0.0
            frames[:, :4, 0] = 1.0   # slots 0..3 present, slot 4 absent
        diag = {k: np.zeros((len(prefix), capacity), dtype=np.float32)
                for k in ("lane_index", "target_lane_index", "commanded_accel",
                          "commanded_steering", "crashed")}
        fdiag = {k: np.zeros((horizon, capacity), dtype=np.float32)
                 for k in ("lane_index", "target_lane_index", "commanded_accel",
                           "commanded_steering", "crashed")}
        return {
            "prefix_frames": prefix, "future_frames": future,
            "terminated": False, "truncated": False,
            "termination_reason": "", "first_contact_step": -1,
            "diag_prefix": diag, "diag_future": fdiag,
            "fingerprints": _fingerprints(seed, capacity),
            "static_metadata": [],
        }
    fake.calls = calls
    return fake


def test_factual_runs_once_and_ten_branches_share_state():
    config = _config(Path("/tmp/x"))
    fake = make_fake_rollout(config)
    t_star = 12
    prefix = np.full(t_star, 4, dtype=np.int64)
    factual_plan = np.full(20, 1, dtype=np.int64)
    group = coll.collect_group(12345, t_star, prefix, factual_plan, config, fake)
    # 11 calls: 1 factual + 10 counterfactual.
    assert len(fake.calls) == 11
    # first call uses the factual plan; the other ten use the materialised plans.
    assert fake.calls[0][2] == tuple(int(a) for a in factual_plan)
    expected_cf = [tuple(int(x) for x in s4cp.materialize_action_plan(factual_plan, a, d))
                   for a, d in s4cp.ACTION_PLANS]
    assert [c[2] for c in fake.calls[1:]] == expected_cf
    # all eleven share identical prefix frames (same initial state).
    ref = group["factual"]["prefix_frames"]
    for b in group["cf"]:
        assert np.array_equal(b["prefix_frames"], ref)
    verdict = coll.validate_complete_group(group, config)
    assert verdict["accepted"], verdict
    assert verdict["cf_futures"].shape == (10, 20, 5, 8)


def test_any_branch_failure_rejects_whole_group():
    config = _config(Path("/tmp/x"))
    t_star = 12
    prefix = np.full(t_star, 4, dtype=np.int64)
    factual_plan = np.full(20, 1, dtype=np.int64)

    base_fake = make_fake_rollout(config)

    def terminating(env_id, env_cfg, seed, prefix_actions, future_actions, cap):
        out = base_fake(env_id, env_cfg, seed, prefix_actions, future_actions, cap)
        # kill the (FASTER,3) branch = index 7 in ACTION_PLANS.
        plan = tuple(int(x) for x in future_actions)
        target = tuple(int(x) for x in s4cp.materialize_action_plan(factual_plan, 3, 3))
        if plan == target:
            out["terminated"] = True
        return out

    group = coll.collect_group(999, t_star, prefix, factual_plan, config, terminating)
    verdict = coll.validate_complete_group(group, config)
    assert not verdict["accepted"]
    assert verdict["reason"] == "prefix_terminated"
    assert verdict["failed_branch"] == "cf_7"  # (FASTER,3) is plan index 7


def test_present_set_mismatch_rejects_group():
    config = _config(Path("/tmp/x"))
    t_star = 12
    prefix = np.full(t_star, 4, dtype=np.int64)
    factual_plan = np.full(20, 1, dtype=np.int64)
    base_fake = make_fake_rollout(config)

    def mismatch(env_id, env_cfg, seed, prefix_actions, future_actions, cap):
        out = base_fake(env_id, env_cfg, seed, prefix_actions, future_actions, cap)
        target = tuple(int(x) for x in s4cp.materialize_action_plan(factual_plan, 0, 1))
        if tuple(int(x) for x in future_actions) == target:
            out["future_frames"][:, 3, 0] = 0.0  # drop a present entity mid-horizon
        return out

    group = coll.collect_group(555, t_star, prefix, factual_plan, config, mismatch)
    verdict = coll.validate_complete_group(group, config)
    assert not verdict["accepted"]
    assert verdict["reason"] == "present_set_mismatch"


def test_role_quota_counts_groups_and_resume_no_duplicate(tmp_path):
    config = _config(tmp_path)
    schedule = np.arange(1000, 1020, dtype=np.int64)
    fake = make_fake_rollout(config)
    # first pass: stop after 2 accepted groups.
    receipt1 = coll.collect_role("development_selection", 4, schedule, config,
                                 branch_rollout=fake, probe_fn=_fake_probe,
                                 stop_after_accepted=2)
    assert receipt1["accepted_groups"] == 2 and receipt1["partial"]
    # resume to the full quota of 4.
    fake2 = make_fake_rollout(config)
    receipt2 = coll.collect_role("development_selection", 4, schedule, config,
                                 branch_rollout=fake2, probe_fn=_fake_probe)
    assert receipt2["accepted_groups"] == 4 and not receipt2["partial"]
    data = np.load(_role_dev_npz(config), allow_pickle=True)
    seeds = np.asarray(data["episode_seed"])
    assert len(np.unique(seeds)) == seeds.size == 4  # no duplicated group on resume
    assert data["object_counterfactual_futures"].shape == (4, 10, 20, 5, 8)


def _role_dev_npz(config):
    return Path(coll._role_path(config["data_output"], "development_selection"))
