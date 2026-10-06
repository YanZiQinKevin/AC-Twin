"""Stage 4C protocol tests: action plans, seed schedules, guards, inference mask."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4c_protocol as s4cp  # noqa: E402
from ac_twin.stage2_protocol import validate_seed_schedule  # noqa: E402


# --- action-plan semantics -------------------------------------------------------------


def test_idle1_and_idle3_differ_when_factual_steps_non_idle():
    # factual steps 1-2 are non-IDLE so (IDLE,1) and (IDLE,3) must diverge.
    factual = np.array([0, 2, 3, 4, 4, 4, 4, 4, 4, 4,
                        4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=np.int64)
    plan_i1 = s4cp.materialize_action_plan(factual, action=1, duration=1)
    plan_i3 = s4cp.materialize_action_plan(factual, action=1, duration=3)
    assert not np.array_equal(plan_i1, plan_i3)
    assert plan_i1[0] == 1 and plan_i1[1] == factual[1] and plan_i1[2] == factual[2]
    assert plan_i3[0] == 1 and plan_i3[1] == 1 and plan_i3[2] == 1
    assert plan_i3[3] == factual[3]


def test_action_plans_table_matches_materialization():
    for action, duration in s4cp.ACTION_PLANS:
        factual = np.full(20, 4, dtype=np.int64)  # SLOWER everywhere
        plan = s4cp.materialize_action_plan(factual, action, duration)
        assert (plan[:duration] == action).all()
        assert (plan[duration:] == 4).all()


# --- group seed schedules ---------------------------------------------------------------


def test_dev_and_confirm_schedules_match_frozen_contract():
    import numpy.random as npr
    dev = s4cp.make_group_seed_schedule(s4cp.DEV_SEED_ENTROPY, s4cp.DEV_SCHEDULE_SIZE)
    assert dev.size == 5000 and len(np.unique(dev)) == 5000
    expected = npr.default_rng(npr.SeedSequence(2026091801).spawn(3)[0]).choice(
        2147483646, size=5000, replace=False) + 1
    assert np.array_equal(dev, expected.astype(np.int64))

    confirm = s4cp.make_group_seed_schedule(
        s4cp.CONFIRM_SEED_ENTROPY, s4cp.CONFIRM_SCHEDULE_SIZE)
    assert confirm.size == 15000 and len(np.unique(confirm)) == 15000
    # dev and confirm schedules must be disjoint.
    assert np.intersect1d(dev, confirm).size == 0


def test_group_seed_schedule_entropy_and_size_frozen():
    with pytest.raises(ValueError):
        s4cp.make_group_seed_schedule(entropy=123, size=5000)
    with pytest.raises(ValueError):
        s4cp.make_group_seed_schedule(entropy=s4cp.DEV_SEED_ENTROPY, size=100)


def test_zero_overlap_rejects_overlapping_schedule():
    dev = s4cp.make_group_seed_schedule(s4cp.DEV_SEED_ENTROPY, s4cp.DEV_SCHEDULE_SIZE)
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_schedule(dev, np.array([dev[0]], dtype=np.int64))
    validate_seed_schedule(dev, np.array([0], dtype=np.int64))


# --- forbidden-data guard ------------------------------------------------------------------


def test_forbidden_stage4b_confirm_rejected():
    with pytest.raises(ValueError):
        s4cp.assert_allowed_stage4c_data(
            "data/object_centric_stage/stage4_merge_risk_confirm/x.npz")


def test_stage4c_role_allowed():
    s4cp.assert_allowed_stage4c_data(
        "data/object_centric_stage/stage4c_merge_multiaction_dev/x.npz")
    s4cp.assert_allowed_stage4c_data(
        "data/object_centric_stage/stage4c_merge_multiaction_confirm/x.npz")


# --- inference mask ---------------------------------------------------------------------------


def _history_factual():
    # G=1, history_len=2, T=2, N=3. slot0 ego, slot1 always present, slot2 sparse.
    history = np.zeros((1, 2, 3, 8), dtype=float)
    factual = np.zeros((1, 2, 3, 8), dtype=float)
    history[..., 0] = 1.0  # all present in history
    factual[..., 0] = 1.0
    history[0, :, 2, 0] = 0.0  # slot2 never known in history
    factual[0, 1, 1, 0] = 0.0  # slot1 absent at factual t=1
    return history, factual


def test_inference_mask_excludes_ego_and_uses_history_and_factual_only():
    history, factual = _history_factual()
    mask = s4cp.build_inference_mask(history, factual)
    assert mask.shape == (1, 2, 3)
    assert not mask[..., 0].any()             # ego slot always excluded
    assert not mask[..., 2].any()             # not history-known -> excluded
    assert mask[0, 0, 1] and not mask[0, 1, 1]  # factual presence gates


def test_inference_mask_rejects_counterfactual_argument():
    history, factual = _history_factual()
    counterfactual = np.ones_like(factual)
    with pytest.raises(TypeError):
        s4cp.build_inference_mask(history, factual, counterfactual)  # noqa: F841


# --- config validation ---------------------------------------------------------------------------


def _base_config():
    return {
        "plan_id": s4cp.PLAN_ID,
        "scene": "merge",
        "env_id": "merge-v0",
        "action_plans": [list(p) for p in s4cp.ACTION_PLANS],
        "cf_actions": list(s4cp.CF_ACTIONS),
        "intervention_durations": list(s4cp.CF_DURATIONS),
        "history_len": 10,
        "horizon": 20,
        "t_star_low": 10,
        "t_star_high": 15,
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "vehicles_count": 25,
        "observed_vehicles_count": 5,
        "duration": 100,
        "object_capacity": 5,
        "dev_selection_groups": 500,
        "dev_screen_groups": 300,
        "candidate_ns": [600, 900, 1200, 1800, 2400],
        "dev_seed_entropy": 2026091801,
        "dev_schedule_size": 5000,
        "confirm_seed_entropy": 2026091901,
        "confirm_schedule_size": 15000,
        "bootstrap_entropy_screen": 2026091802,
        "bootstrap_entropy_confirm": 2026092002,
        "power_monte_carlo_entropy": 2026092001,
        "confidence_screen": 0.80,
        "confidence_confirm": 0.95,
        "power_target": 0.80,
        "power_shrink_main": 0.75,
        "top_error_threshold": 0.41601440122246675,
        "primary_constant_w": 0.60,
        "response_predictor_seeds": [101, 202, 303, 404, 505],
        "expert_router_seeds": [1101, 1202, 1303, 1404, 1505],
        "lambda_candidates": [0.0, 0.25, 0.5, 1.0],
        "gate_i_aurc_regret_min": 0.05,
        "gate_i_regret80_min": 0.03,
        "gate_strata_i_aurc_floor": -0.10,
        "gate_task_min_nonzero_regret_frac": 0.20,
        "gate_min_positive_pairs": 4,
    }


def test_config_validates_frozen_constants():
    checked = s4cp.validate_config(_base_config())
    assert checked["plan_id"] == s4cp.PLAN_ID
    assert checked["horizon"] == 20


@pytest.mark.parametrize("key,value", [
    ("horizon", 21),
    ("dev_screen_groups", 301),
    ("gate_i_aurc_regret_min", 0.04),
    ("primary_constant_w", 0.61),
    ("top_error_threshold", 0.5),
])
def test_config_rejects_drift(key, value):
    config = _base_config()
    config[key] = value
    with pytest.raises(ValueError):
        s4cp.validate_config(config)
