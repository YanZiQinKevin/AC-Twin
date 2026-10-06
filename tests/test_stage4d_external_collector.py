"""Blocking tests for the Stage 4D-D2 role collector's pure (non-simulator) parts.

Plan section 12.7: candidate ordering (primary-then-reserve in frozen rank order),
the append-only rejection ledger's burn-once semantics, and the npz stacking contract.
These never touch MetaDrive or the converted DB.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import collect_stage4d_external as C
from ac_twin import stage4d_external_features as F
from ac_twin import stage4d_external_split as split

pytestmark = pytest.mark.skipif(
    not C.PLATFORM_DIR.is_dir() or not (C.PLATFORM_DIR / split.POOLS_FILE).is_file(),
    reason="scenario split pools not frozen in this workspace",
)


def test_candidates_primary_then_reserve_in_rank_order():
    for role in ("smoke", "pilot"):
        cands = C.load_role_candidates(role)
        rows = [r for r in split.load_split_pools(C.PLATFORM_DIR) if r["role"] == role]
        rows.sort(key=lambda r: r["global_rank"])
        expected = [r["scenario_id"] for r in rows if r["pool"] == "primary"] + \
                   [r["scenario_id"] for r in rows if r["pool"] == "reserve"]
        assert cands == expected
        # smoke primary quota is 12: the first 12 candidates are the primary list
        if role == "smoke":
            assert len([r for r in rows if r["pool"] == "primary"]) == 12


def test_roles_are_disjoint_across_pools():
    seen = {}
    for role in split.ROLE_ORDER:
        for sid in C.load_role_candidates(role):
            assert sid not in seen, f"{sid} in both {seen.get(sid)} and {role}"
            seen[sid] = role


def test_ledger_burn_once_and_fsync(tmp_path):
    led = C._Ledger(tmp_path / "ledger.jsonl")
    led.append({"scenario_id": "a", "outcome": "rejected", "reason": "no_left_lane_at_t_star"})
    led.append({"scenario_id": "b", "outcome": "accepted", "t_star": 12})
    led.append({"scenario_id": "c", "outcome": "rejected", "reason": "branch_incomplete"})
    assert led.burned_ids == {"a", "b", "c"}
    # re-opening the same file preserves all rows (resume anchor)
    led2 = C._Ledger(tmp_path / "ledger.jsonl")
    assert led2.rows == led.rows
    assert led2.burned_ids == {"a", "b", "c"}


def _fake_group(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    hist = rng.normal(size=(F.HISTORY_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES))
    fact = rng.normal(size=(F.FUTURE_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES))
    cf = rng.normal(size=(F.N_ACTIONS, F.FUTURE_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES))
    for arr in (hist, fact, cf):
        arr[..., 0] = 1.0  # presence channel must be a valid {0,1}-like flag
    return {
        "object_history": hist,
        "object_factual_future": fact,
        "object_counterfactual_futures": cf,
        "inference_mask": np.ones((F.FUTURE_LEN, F.OBJECT_CAPACITY), dtype=bool),
        "t_star": 12,
        "factual_actions": rng.integers(0, 5, size=F.FUTURE_LEN),
        "scenario_id": f"sid{seed}",
        "eval_track_ids": ["e", "a", "b", "c", "d"],
        "ego_lane_id": "1",
    }


def test_stack_arrays_contract_shapes():
    groups = [_fake_group(i) for i in range(3)]
    arrays = C._stack_arrays(groups)
    assert arrays["object_history"].shape == (3, F.HISTORY_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES)
    assert arrays["object_factual_future"].shape == (3, F.FUTURE_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES)
    assert arrays["object_counterfactual_futures"].shape == (
        3, F.N_ACTIONS, F.FUTURE_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES)
    assert arrays["factual_actions"].shape == (3, F.FUTURE_LEN)
    assert arrays["inference_mask"].shape == (3, F.FUTURE_LEN, F.OBJECT_CAPACITY)
    assert arrays["inference_mask"].dtype == bool
    assert arrays["t_star"].shape == (3,)
    # canonical frozen action table
    assert arrays["cf_actions"].tolist() == [a for a, _ in C.A.ACTION_PLANS]
    assert arrays["cf_durations"].tolist() == [d for _, d in C.A.ACTION_PLANS]
    assert int(arrays["n_actions"]) == 10
    feats = json.loads(str(arrays["object_features_json"]))
    assert feats == list(F.FEATURE_NAMES)


def test_valid_simulation_rejects_structural_failures():
    good = _fake_group(0)
    good["branches_completed"] = True
    ok, _ = C._valid_simulation(good)
    assert ok
    bad = dict(good, branches_completed=False)
    ok, reason = C._valid_simulation(bad)
    assert not ok and reason == "branch_incomplete"
    nan = _fake_group(1)
    nan["branches_completed"] = True
    nan["object_factual_future"] = nan["object_factual_future"].copy()
    nan["object_factual_future"][0, 0, 0] = np.nan
    ok, reason = C._valid_simulation(nan)
    assert not ok and reason == "non_finite"


def test_checkpoint_resume_reads_each_compressed_array_once(tmp_path, monkeypatch):
    """Regression: archive[key][i] in a loop repeatedly inflated the whole NPZ member."""
    groups = [_fake_group(i) for i in range(3)]
    arrays = C._stack_arrays(groups)
    meta = [C._group_meta(g) for g in groups]
    data = {
        **{key: arrays[key] for key in C._ARRAY_FIELDS},
        "__config_fingerprint__": np.asarray("fp"),
        "__meta_json__": np.asarray(json.dumps(meta)),
    }
    counts = {key: 0 for key in data}

    class CountingArchive:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __getitem__(self, key):
            counts[key] += 1
            return data[key]

    monkeypatch.setattr(C.np, "load", lambda *_args, **_kwargs: CountingArchive())
    restored = C._load_checkpoint_groups(tmp_path / "checkpoint.npz", "fp")

    assert len(restored) == 3
    assert [g["scenario_id"] for g in restored] == ["sid0", "sid1", "sid2"]
    for key in C._ARRAY_FIELDS:
        assert counts[key] == 1, f"compressed member {key} was inflated more than once"


def test_bounded_worker_pauses_then_resumes_bitwise(tmp_path, monkeypatch):
    """A short-lived worker must checkpoint, close MetaDrive, and resume identically."""
    candidates = ["s0", "s1", "s2"]
    monkeypatch.setattr(C, "load_role_candidates", lambda _role: list(candidates))
    monkeypatch.setattr(
        C, "scenario_index_and_file",
        lambda _db: {sid: (i, f"{sid}.pkl") for i, sid in enumerate(candidates)},
    )
    monkeypatch.setattr(C, "_load_centralized", lambda path: {"sid": path.stem})
    monkeypatch.setattr(
        C.F, "select_t_star",
        lambda scenario, sid, salt: (12, "accepted", {}, []),
    )
    monkeypatch.setattr(
        C.A, "prepare_group_context",
        lambda scenario, sid, salt, contract: SimpleNamespace(
            eval_track_ids=["e", "a", "b", "c", "d"], ego_lane_id="lane0"),
    )

    class FakeMapManager:
        def clear_stored_maps(self):
            return None

    class FakeEnv:
        def __init__(self):
            self.engine = SimpleNamespace(map_manager=FakeMapManager())
            self.closed = False

        def close(self):
            self.closed = True

    envs = []

    def build_env(*_args, **_kwargs):
        env = FakeEnv()
        envs.append(env)
        return env

    monkeypatch.setattr(C.A, "build_env", build_env)

    def run_group(_env, scenario, _index, _ctx):
        seed = int(scenario["sid"][1:])
        group = _fake_group(seed)
        group["branches_completed"] = True
        return group

    monkeypatch.setattr(C.A, "run_group", run_group)
    monkeypatch.setattr(C, "DB_DIR", tmp_path / "db")

    out_dir = tmp_path / "role"
    paused = C.collect_role(
        "train", 3, out_dir, max_attempts_per_process=2)
    assert paused["status"] == "COLLECTION_PAUSED"
    assert paused["accepted"] == 2
    assert envs[-1].closed

    completed = C.collect_role(
        "train", 3, out_dir, max_attempts_per_process=2)
    assert completed["status"] == "COLLECTION_COMPLETE"
    assert completed["accepted"] == 3
    assert envs[-1].closed
    with np.load(out_dir / "groups.npz", allow_pickle=False) as archive:
        actual = {key: archive[key] for key in C._ARRAY_FIELDS}
    expected = C._stack_arrays([_fake_group(i) for i in range(3)])
    for key in C._ARRAY_FIELDS:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_supervisor_progress_line_is_human_observable():
    line = C._supervisor_progress_line(
        role="train", accepted=3200, quota=5000, burned=12000,
        child_pid=4321, elapsed_seconds=600.0, start_accepted=3000)
    assert "child_pid=4321" in line
    assert "3200/5000" in line
    assert "64.0%" in line
    assert "elapsed=00:10:00" in line
    assert "ETA=" in line
