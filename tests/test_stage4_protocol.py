from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4_protocol as s4p  # noqa: E402
from ac_twin.stage2_protocol import validate_seed_schedule  # noqa: E402


# --- seed schedule ----------------------------------------------------------------


def test_candidate_seed_schedule_is_12000_unique_positive_int32():
    schedule = s4p.make_candidate_seed_schedule()
    assert schedule.shape == (s4p.SEED_SCHEDULE_SIZE,)
    assert schedule.shape == (12000,)
    assert len(np.unique(schedule)) == schedule.size
    assert schedule.min() >= 1
    assert schedule.max() <= 2**31 - 2


def test_candidate_seed_schedule_is_deterministic():
    first = s4p.make_candidate_seed_schedule()
    second = s4p.make_candidate_seed_schedule()
    assert np.array_equal(first, second)


def test_candidate_seed_schedule_entropy_and_size_are_frozen():
    with pytest.raises(ValueError, match="entropy is frozen"):
        s4p.make_candidate_seed_schedule(entropy=999)
    with pytest.raises(ValueError, match="size is frozen"):
        s4p.make_candidate_seed_schedule(size=100)


def test_validate_seed_schedule_rejects_overlap():
    schedule = s4p.make_candidate_seed_schedule()
    historical = np.array([int(schedule[0]), 7, 8], dtype=np.int64)
    with pytest.raises(ValueError, match="overlaps"):
        validate_seed_schedule(schedule, historical)


def test_validate_seed_schedule_rejects_duplicates():
    schedule = np.array([5, 5, 6, 7], dtype=np.int64)
    with pytest.raises(ValueError, match="duplicate"):
        validate_seed_schedule(schedule, np.array([100], dtype=np.int64))


# --- historical seed scan ----------------------------------------------------------


def test_scan_historical_seeds_unions_npz_and_receipt(tmp_path: Path):
    root = tmp_path / "historical"
    root.mkdir()
    # An NPZ carrying episode_seeds.
    np.savez(root / "a.npz", episode_seeds=np.array([11, 22]))
    # An NPZ carrying episode_seed (singular).
    np.savez(root / "b.npz", episode_seed=np.array([33]))
    # A receipt/manifest JSON with embedded episode-seed lists.
    (root / "receipt.json").write_text(
        json.dumps({"episode_seeds": [55, 66], "other": 1}), encoding="utf-8"
    )
    found = s4p.scan_historical_seeds([root])
    assert set(found.tolist()) == {11, 22, 33, 55, 66}


def test_scan_historical_seeds_ignores_unrealized_candidate_schedules(tmp_path: Path):
    # A raw *candidate* seed_schedule.npy is NOT a realized episode seed: an
    # unaccepted candidate never generated an episode, so it must not be counted
    # (matches stage2_protocol.scan_historical_episode_seeds / stage3_data).
    root = tmp_path / "historical"
    root.mkdir()
    np.save(root / "c_seed_schedule.npy", np.array([44, 45, 46]))  # candidates only
    np.savez(root / "real.npz", episode_seed=np.array([11]))
    found = s4p.scan_historical_seeds([root])
    assert 44 not in set(found.tolist())
    assert 11 in set(found.tolist())


def test_scan_historical_seeds_excludes_stage4a_own_dir(tmp_path: Path):
    root = tmp_path / "historical"
    stage4 = root / "stage4_merge_risk_dev"
    stage4.mkdir(parents=True)
    np.savez(stage4 / "partial.npz", episode_seed=np.array([999]))
    np.savez(root / "real.npz", episode_seed=np.array([11]))
    found = s4p.scan_historical_seeds([root])
    assert 999 not in set(found.tolist())
    assert 11 in set(found.tolist())


def test_scan_historical_seeds_requires_some_seed(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no historical episode seeds"):
        s4p.scan_historical_seeds([empty])


# --- forbidden-data guards ---------------------------------------------------------


@pytest.mark.parametrize(
    "token",
    [
        "stage2_merge_1200",
        "stage2_confirmatory",
        "stage3a_plus_qualification",
        "stage3a_replacement_dev",
        "stage3b",
        "stage3r_rscreen",
        "gatec",
        "merge_2000",
        "roundabout",
    ],
)
def test_forbidden_stage2_stage3_paths_rejected(token: str):
    bad = f"data/object_centric_stage/{token}/object_twinworld_{token}.npz"
    # Rejected either by the inner sealed-Stage-2 guard or the Stage 4A token guard;
    # the invariant is that the path is refused, regardless of which guard fires.
    with pytest.raises(ValueError):
        s4p.assert_allowed_stage4_data(bad)


def test_non_role_path_rejected():
    with pytest.raises(ValueError, match="must live under"):
        s4p.assert_allowed_stage4_data("data/object_centric_stage/other_role/x.npz")


def test_stage4a_role_path_allowed():
    ok = "data/object_centric_stage/stage4_merge_risk_dev/object_twinworld_stage4_merge_risk_dev.npz"
    s4p.assert_allowed_stage4_data(ok)  # must not raise


# --- per-stratum role allocation ----------------------------------------------------


def _full_strata_arrays(per_stratum: int = 500):
    """Build 5000 episodes: 500 per (action,duration) stratum with unique seeds."""
    cf_action, cf_duration, episode_seed = [], [], []
    seed = 1
    for action in s4p.CF_ACTIONS:
        for duration in s4p.CF_DURATIONS:
            for _ in range(per_stratum):
                cf_action.append(action)
                cf_duration.append(duration)
                episode_seed.append(seed)
                seed += 1
    return (
        np.asarray(cf_action, dtype=np.int64),
        np.asarray(cf_duration, dtype=np.int64),
        np.asarray(episode_seed, dtype=np.int64),
    )


def test_assign_stratum_roles_exact_3000_1000_1000():
    cf_action, cf_duration, episode_seed = _full_strata_arrays()
    roles = s4p.assign_stratum_roles(cf_action, cf_duration, episode_seed)
    assert int((roles == s4p.ROLE_RISK_TRAIN).sum()) == 3000
    assert int((roles == s4p.ROLE_RISK_VALIDATION).sum()) == 1000
    assert int((roles == s4p.ROLE_DEVELOPMENT_SCREEN).sum()) == 1000


def test_assign_stratum_roles_is_hash_only_and_per_stratum():
    cf_action, cf_duration, episode_seed = _full_strata_arrays()
    roles = s4p.assign_stratum_roles(cf_action, cf_duration, episode_seed)
    # Each stratum independently yields 300/100/100.
    for action in s4p.CF_ACTIONS:
        for duration in s4p.CF_DURATIONS:
            mask = (cf_action == action) & (cf_duration == duration)
            cell = roles[mask]
            assert int((cell == 0).sum()) == 300
            assert int((cell == 1).sum()) == 100
            assert int((cell == 2).sum()) == 100
    # Deterministic: identical call reproduces identical roles.
    roles2 = s4p.assign_stratum_roles(cf_action, cf_duration, episode_seed)
    assert np.array_equal(roles, roles2)


def test_assign_stratum_roles_requires_exactly_500_per_stratum():
    cf_action, cf_duration, episode_seed = _full_strata_arrays(per_stratum=499)
    with pytest.raises(ValueError, match="exactly"):
        s4p.assign_stratum_roles(cf_action, cf_duration, episode_seed)


def test_role_indices_partition():
    roles = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    idx = s4p.role_indices(roles)
    assert idx["risk_train"].tolist() == [0, 3]
    assert idx["risk_validation"].tolist() == [1, 4]
    assert idx["development_screen"].tolist() == [2]


# --- state machine ------------------------------------------------------------------


def test_state_machine_is_a_linear_dag_ending_at_adjudication():
    order = [
        "DRAFT", "SOFTWARE_READY", "PROTOCOL_LOCKED", "DEV_COLLECTED",
        "DEV_INFERENCE_COMPLETE", "DEV_CANDIDATE_SELECTED", "DEV_SCREEN_ADJUDICATED",
    ]
    for current, nxt in zip(order, order[1:]):
        assert nxt in s4p.STATE_TRANSITIONS[current]
    assert s4p.STATE_TRANSITIONS["DEV_SCREEN_ADJUDICATED"] == set()


# --- config validation + write-once lock -------------------------------------------


def _valid_config(tmp_path: Path) -> dict:
    role_dir = tmp_path / "data" / "object_centric_stage" / "stage4_merge_risk_dev"
    role_dir.mkdir(parents=True)
    run_dir = tmp_path / "experiments" / "stage4_reliability" / "stage4a"
    run_dir.mkdir(parents=True)
    return {
        "plan_id": s4p.PLAN_ID,
        "purpose": s4p.PURPOSE,
        "analysis_status": s4p.PURPOSE,
        "scene": "merge",
        "env_id": "merge-v0",
        "n_pairs": 5000,
        "strata_per_cell": 500,
        "cf_actions": [0, 1, 2, 3, 4],
        "intervention_durations": [1, 3],
        "split_per_stratum": {"risk_train": 300, "risk_validation": 100,
                              "development_screen": 100},
        "seed_entropy": 20260911,
        "seed_schedule_size": 12000,
        "max_attempts": 12000,
        "response_predictor_seeds": list(s4p.RESPONSE_PREDICTOR_SEEDS),
        "expert_router_seeds": list(s4p.EXPERT_ROUTER_SEEDS),
        "primary_constant_w": 0.60,
        "confidence_level": 0.80,
        "bootstrap_entropy_screen": 2026091101,
        "data_output": str(role_dir / "object_twinworld_stage4_merge_risk_dev.npz"),
        "run_dir": str(run_dir),
        "output_dir": str(role_dir),
        "collection_receipt": str(role_dir / "collection_receipt.json"),
        "rejection_ledger": str(role_dir / "rejection_ledger.jsonl"),
        "stage3r_run_dir": str(tmp_path / "stage3r"),
        "stage3b_binding": str(tmp_path / "binding.json"),
        "scaler_path": str(tmp_path / "stage3r" / "scaler.json"),
        "historical_data_root": str(tmp_path / "data"),
    }


def test_validate_config_rejects_frozen_drift(tmp_path: Path):
    config = _valid_config(tmp_path)
    config["seed_entropy"] = 123  # drift from the frozen entropy
    with pytest.raises(ValueError, match="seed_entropy must be"):
        s4p.validate_stage4a_config(config)


def test_validate_config_rejects_wrong_split(tmp_path: Path):
    config = _valid_config(tmp_path)
    config["split_per_stratum"] = {"risk_train": 400, "risk_validation": 50,
                                   "development_screen": 50}
    with pytest.raises(ValueError, match="300/100/100"):
        s4p.validate_stage4a_config(config)


def test_write_protocol_lock_refuses_overwrite(tmp_path: Path):
    config = _valid_config(tmp_path)
    config_path = tmp_path / "stage4a_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    # Pre-create the lock so write_protocol_lock short-circuits BEFORE any scan.
    protocol_dir = Path(config["run_dir"]) / "protocol"
    protocol_dir.mkdir(parents=True)
    (protocol_dir / "protocol_lock.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        s4p.write_protocol_lock(config_path)
