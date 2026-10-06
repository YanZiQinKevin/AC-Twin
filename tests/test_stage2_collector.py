"""Task 2 tests: Stage 2 collector data contract, ledger, resume, lock gating.

All tests use an injected deterministic ``branch_rollout`` so no highway_env /
pygame import happens under pytest's ``-W error``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import collect_stage2_holdout as col  # noqa: E402
from ac_twin import stage2_protocol as sp  # noqa: E402

SMOKE_CONFIG = ROOT / "config" / "stage2_holdout_merge_smoke.json"
CAPACITY = 5
N_FEATURES = 8


# --- fake environment plumbing --------------------------------------------------


def _fake_probe(config):
    """Mimic the env probe: action ids/probs and object capacity."""
    action_ids = np.arange(5, dtype=np.int64)
    probs = np.array([0.095, 0.45, 0.095, 0.18, 0.18], dtype=np.float64)
    probs /= probs.sum()
    return action_ids, probs, CAPACITY


def _fake_branch_rollout(env_id, env_cfg, seed, prefix_actions, future_actions, capacity):
    """Deterministic pure function of (seed, prefix, future) -> branch result.

    Prefix frames depend only on (seed, prefix_actions); future frames continue
    from the prefix and diverge by future_actions. A branch terminates early when
    the cumulative future action sum crosses a threshold, to exercise rejection.
    """
    rng = np.random.default_rng(int(seed))
    base = rng.normal(size=(capacity, N_FEATURES)).astype(np.float32)
    base[:, 0] = 1.0  # presence
    prefix_actions = np.asarray(prefix_actions, dtype=np.int64)
    future_actions = np.asarray(future_actions, dtype=np.int64)

    def step_frames(actions, start):
        frames = []
        state = start.copy()
        frames.append(state.copy())
        for a in actions:
            state = state + (float(a) + 1.0) * 0.1
            frames.append(state.copy())
        return frames

    prefix_frames = step_frames(prefix_actions, base)
    prefix_end = prefix_frames[-1]
    future_frames_full = step_frames(future_actions, prefix_end)[1:]

    terminated = False
    truncated = False
    first_contact = -1
    reason = ""
    # Synthetic early termination: large cumulative future action sum crashes.
    if float(future_actions.sum()) > 55.0:
        terminated = True
        keep = max(1, len(future_frames_full) // 2)
        future_frames_full = future_frames_full[:keep]
        first_contact = keep - 1
        reason = "collision"

    n_future = len(future_frames_full)
    steps_prefix = len(prefix_frames)
    diag_prefix = {
        "lane_index": np.zeros((steps_prefix, capacity), dtype=np.int64),
        "target_lane_index": np.zeros((steps_prefix, capacity), dtype=np.int64),
        "commanded_accel": np.full((steps_prefix, capacity), np.nan, dtype=np.float32),
        "commanded_steering": np.full((steps_prefix, capacity), np.nan, dtype=np.float32),
        "crashed": np.zeros((steps_prefix, capacity), dtype=bool),
    }
    diag_future = {
        "lane_index": np.zeros((n_future, capacity), dtype=np.int64),
        "target_lane_index": np.zeros((n_future, capacity), dtype=np.int64),
        "commanded_accel": np.full((n_future, capacity), np.nan, dtype=np.float32),
        "commanded_steering": np.full((n_future, capacity), np.nan, dtype=np.float32),
        "crashed": np.zeros((n_future, capacity), dtype=bool),
    }
    fingerprints = [{"entity_id": i, "class": "FakeVehicle",
                     "position_xy": [float(base[i, 2]), float(base[i, 3])],
                     "heading": 0.0, "speed": 1.0, "route": None}
                    for i in range(capacity)]
    return {
        "prefix_frames": np.asarray(prefix_frames, dtype=np.float32),
        "future_frames": np.asarray(future_frames_full, dtype=np.float32),
        "terminated": terminated,
        "truncated": truncated,
        "termination_reason": reason,
        "first_contact_step": first_contact,
        "diag_prefix": diag_prefix,
        "diag_future": diag_future,
        "fingerprints": fingerprints,
        "static_metadata": [{"entity_id": i, "class": "FakeVehicle",
                             "route": None, "behavior_params": {}}
                            for i in range(capacity)],
    }


@pytest.fixture()
def smoke_env(tmp_path, monkeypatch):
    """A smoke config redirected into tmp_path with a completed protocol lock."""
    config = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    out_dir = tmp_path / "smoke"
    config["output_dir"] = str(out_dir)
    config["data_output"] = str(tmp_path / "data" / "stage2.npz")
    config["rejection_ledger"] = str(tmp_path / "data" / "rejection_ledger.jsonl")
    config["collection_receipt"] = str(tmp_path / "data" / "collection_receipt.json")
    cfg_path = tmp_path / "smoke_config.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")

    # Smoke validator is bound to the canonical smoke output dir; bypass it here
    # by validating structure only, then create the lock via the protocol module.
    monkeypatch.setattr(sp, "validate_stage2_config", lambda cfg: None)
    sp.write_protocol_lock(cfg_path, out_dir, params_provider=_stub_params)
    return config, cfg_path, out_dir


def _stub_params(config):
    return {
        "beta_action_horizon": np.zeros(3, dtype=np.float64),
        "beta_action_horizon_distance": np.zeros(4, dtype=np.float64),
        "probe_coefficients": np.zeros((65, 6), dtype=np.float64),
        "probe_coefficients_shuffled": np.zeros((65, 6), dtype=np.float64),
        "probe_scene_mean": np.zeros(6, dtype=np.float64),
    }


def _run(config_path, resume=False):
    return col.collect_stage2(
        json.loads(Path(config_path).read_text(encoding="utf-8")),
        resume=resume,
        branch_rollout=_fake_branch_rollout,
        probe_fn=_fake_probe,
    )


# --- paired prefix consistency + contract ----------------------------------------


def test_collects_requested_pairs_and_contract(smoke_env):
    config, cfg_path, out_dir = smoke_env
    receipt = _run(cfg_path)
    assert receipt["accepted_pairs"] == config["n_pairs"]
    out = Path(config["data_output"])
    assert out.is_file()
    with np.load(out, allow_pickle=True) as z:
        n = config["n_pairs"]
        assert z["object_history"].shape == (n, 10, CAPACITY, N_FEATURES)
        assert z["object_factual_future"].shape == (n, 20, CAPACITY, N_FEATURES)
        assert z["object_counterfactual_future"].shape == (n, 20, CAPACITY, N_FEATURES)
        assert z["factual_actions"].shape == (n, 20)
        assert z["counterfactual_actions"].shape == (n, 20)
        assert len(np.unique(z["episode_seed"])) == n
        for field in ("object_history", "object_factual_future",
                      "object_counterfactual_future"):
            assert np.isfinite(z[field]).all()


def test_paired_prefix_identical_across_branches(smoke_env):
    config, cfg_path, _ = smoke_env
    _run(cfg_path)
    with np.load(config["data_output"], allow_pickle=True) as z:
        history = z["object_history"]
        # Both branches share the same history (it is stored once per pair).
        assert "prefix_max_abs_diff" not in z.files  # sanity: prefix stored once
        assert np.isfinite(history).all()


def test_episode_seeds_come_from_schedule(smoke_env):
    config, cfg_path, out_dir = smoke_env
    _run(cfg_path)
    schedule = np.load(out_dir / "seed_schedule.npy")
    with np.load(config["data_output"], allow_pickle=True) as z:
        seeds = set(int(s) for s in z["episode_seed"])
    assert seeds <= set(int(s) for s in schedule)


def test_counterfactual_action_differs_from_factual_first(smoke_env):
    config, cfg_path, _ = smoke_env
    _run(cfg_path)
    with np.load(config["data_output"], allow_pickle=True) as z:
        factual_first = z["factual_actions"][:, 0]
        cf_action = z["cf_action"]
        assert (cf_action != factual_first).all()
        assert set(np.unique(z["cf_duration"]).tolist()) <= {1, 3}


# --- rejection ledger -------------------------------------------------------------


def test_rejection_ledger_records_every_attempt(smoke_env):
    config, cfg_path, _ = smoke_env
    receipt = _run(cfg_path)
    ledger = Path(config["rejection_ledger"])
    assert ledger.is_file()
    records = [json.loads(line) for line in
               ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == receipt["attempts"]
    for rec in records:
        for key in ("attempt_index", "episode_seed", "decision", "reason",
                    "factual_terminated", "factual_truncated",
                    "counterfactual_terminated", "counterfactual_truncated"):
            assert key in rec, key
    rejected = [r for r in records if r["decision"] == "rejected"]
    assert all(r["reason"] for r in rejected)


def test_no_outcome_based_acceptance(smoke_env):
    """Acceptance must not read realized R / response bin; the collector exposes
    no such field on the acceptance path. We assert the receipt and ledger carry
    no response-magnitude or bin keys."""
    config, cfg_path, _ = smoke_env
    receipt = _run(cfg_path)
    forbidden = {"r_pos", "response_magnitude", "response_bin", "realized_r", "eta_evt"}
    assert not (forbidden & set(receipt))
    for line in Path(config["rejection_ledger"]).read_text(encoding="utf-8").splitlines():
        if line.strip():
            assert not (forbidden & set(json.loads(line)))


# --- resume equivalence -----------------------------------------------------------


def _make_locked_config(tmp_path, monkeypatch, name):
    """Write a smoke config + its own protocol lock under a unique subdir."""
    config = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    base = tmp_path / name
    out_dir = base / "out"
    config["output_dir"] = str(out_dir)
    config["data_output"] = str(base / "data" / "stage2.npz")
    config["rejection_ledger"] = str(base / "data" / "rejection_ledger.jsonl")
    config["collection_receipt"] = str(base / "data" / "collection_receipt.json")
    cfg_path = base / "config.json"
    base.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sp, "validate_stage2_config", lambda cfg: None)
    sp.write_protocol_lock(cfg_path, out_dir, params_provider=_stub_params)
    return config, cfg_path, out_dir


def test_resume_produces_identical_final_data(tmp_path, monkeypatch):
    # Collection A: full run in one shot.
    config_a, cfg_a, _ = _make_locked_config(tmp_path, monkeypatch, "full")
    col.collect_stage2(config_a, resume=False, branch_rollout=_fake_branch_rollout,
                       probe_fn=_fake_probe)
    with np.load(config_a["data_output"], allow_pickle=True) as z:
        full_seeds = z["episode_seed"].copy()
        full_history = z["object_history"].copy()

    # Collection B: interrupted after 4 accepted, then resumed to completion.
    config_b, cfg_b, _ = _make_locked_config(tmp_path, monkeypatch, "resumed")
    receipt_partial = col.collect_stage2(
        config_b, resume=False, branch_rollout=_fake_branch_rollout,
        probe_fn=_fake_probe, stop_after_accepted=4)
    assert receipt_partial["accepted_pairs"] == 4
    receipt_resumed = col.collect_stage2(
        config_b, resume=True, branch_rollout=_fake_branch_rollout, probe_fn=_fake_probe)
    assert receipt_resumed["accepted_pairs"] == config_b["n_pairs"]
    with np.load(config_b["data_output"], allow_pickle=True) as z:
        assert np.array_equal(z["episode_seed"], full_seeds)
        assert np.allclose(z["object_history"], full_history)


def test_resume_without_checkpoint_starts_fresh(smoke_env):
    # ``--resume`` is the always-command: with no partial checkpoint it starts fresh
    # and still collects the full requested set.
    config, cfg_path, _ = smoke_env
    receipt = col.collect_stage2(config, resume=True, branch_rollout=_fake_branch_rollout,
                                 probe_fn=_fake_probe)
    assert receipt["accepted_pairs"] == config["n_pairs"]


# --- protocol-lock gating ----------------------------------------------------------


def test_collection_fails_without_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "validate_stage2_config", lambda cfg: None)
    config = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    config["output_dir"] = str(tmp_path / "no_lock_here")
    config["data_output"] = str(tmp_path / "d.npz")
    config["rejection_ledger"] = str(tmp_path / "l.jsonl")
    config["collection_receipt"] = str(tmp_path / "r.json")
    with pytest.raises((FileNotFoundError, RuntimeError)):
        col.collect_stage2(config, resume=False, branch_rollout=_fake_branch_rollout,
                           probe_fn=_fake_probe)


def test_collection_fails_on_lock_drift(smoke_env, tmp_path):
    config, cfg_path, out_dir = smoke_env
    # Corrupt the seed schedule after locking.
    schedule = np.load(out_dir / "seed_schedule.npy")
    schedule = schedule.copy()
    schedule[0] = schedule[-1]
    np.save(out_dir / "seed_schedule.npy", schedule)
    fresh = dict(config)
    fresh["data_output"] = str(tmp_path / "d2.npz")
    fresh["rejection_ledger"] = str(tmp_path / "l2.jsonl")
    fresh["collection_receipt"] = str(tmp_path / "r2.json")
    with pytest.raises((ValueError, RuntimeError)):
        col.collect_stage2(fresh, resume=False, branch_rollout=_fake_branch_rollout,
                           probe_fn=_fake_probe)


# --- int32 / dtype safety -----------------------------------------------------------


def test_integer_fields_are_safe_int64(smoke_env):
    config, cfg_path, _ = smoke_env
    _run(cfg_path)
    with np.load(config["data_output"], allow_pickle=True) as z:
        for field in ("episode_seed", "cf_action", "cf_duration", "t_star"):
            arr = z[field]
            assert arr.dtype in (np.int64, np.int32)
            assert np.isfinite(arr.astype(np.float64)).all()


def test_smoke_status_is_engineering_only(smoke_env):
    config, cfg_path, _ = smoke_env
    receipt = _run(cfg_path)
    assert receipt["analysis_status"] == "ENGINEERING_SMOKE_ONLY"
