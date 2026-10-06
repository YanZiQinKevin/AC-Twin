"""Task 1 blocking tests: Stage 2B collector — paired-branch identity, exact
stratum quotas, capacity/identity rejection rules, ledger, resume equivalence,
manifest gating, development split assignment and smoke semantic QA.

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

import collect_stage2b_intersection as col  # noqa: E402
from ac_twin import stage2b_protocol as sp  # noqa: E402
from ac_twin import stage2b_data as sd  # noqa: E402

SMOKE_CONFIG = ROOT / "config" / "stage2b_intersection_smoke.json"
CAPACITY = 20
N_FEATURES = 8
N_VEHICLES = 6  # present vehicles (entity ids 0..5); the rest is padding

# Routes that yield >=3 distinct movement signatures across the episode.
_FAKE_ROUTES = [
    [["o0", "ir0", "0"], ["ir0", "il1", "0"], ["il1", "o1", "0"]],   # (0,1)
    [["o1", "ir1", "0"], ["ir1", "il3", "0"], ["il3", "o3", "0"]],   # (1,3)
    [["o2", "ir2", "0"], ["ir2", "il0", "0"], ["il0", "o0", "0"]],   # (2,0)
    [["o3", "ir3", "0"], ["ir3", "il1", "0"], ["il1", "o1", "0"]],   # (3,1)
    [["o0", "ir0", "0"], ["ir0", "il2", "0"], ["il2", "o2", "0"]],   # (0,2)
    [["o1", "ir1", "0"], ["ir1", "il2", "0"], ["il2", "o2", "0"]],   # (1,2)
]


# --- fake environment plumbing --------------------------------------------------


def _fake_probe(config):
    action_ids = np.asarray([0, 1, 2], dtype=np.int64)
    probs = np.full(3, 1.0 / 3.0, dtype=np.float64)
    return action_ids, probs, CAPACITY


def _fake_branch_rollout(env_id, env_cfg, seed, prefix_actions, future_actions, capacity):
    """Deterministic pure function of (seed, prefix, future) -> branch result."""
    assert capacity == CAPACITY
    rng = np.random.default_rng(int(seed))
    base = np.zeros((capacity, N_FEATURES), dtype=np.float32)
    base[:N_VEHICLES] = rng.normal(size=(N_VEHICLES, N_FEATURES)).astype(np.float32)
    base[:N_VEHICLES, 0] = 1.0  # presence
    base[0, 1] = 1.0  # ego slot
    prefix_actions = np.asarray(prefix_actions, dtype=np.int64)
    future_actions = np.asarray(future_actions, dtype=np.int64)

    def step_frames(actions, start):
        frames = [start.copy()]
        state = start.copy()
        for a in actions:
            state = state + (float(a) + 1.0) * 0.1
            state[N_VEHICLES:] = 0.0  # padding stays zero
            frames.append(state.copy())
        return frames

    prefix_frames = step_frames(prefix_actions, base)
    future_frames = step_frames(future_actions, prefix_frames[-1])[1:]

    terminated = False
    first_contact = -1
    reason = ""
    if float(future_actions.sum()) > 55.0:  # synthetic early termination
        terminated = True
        keep = max(1, len(future_frames) // 2)
        future_frames = future_frames[:keep]
        first_contact = keep - 1
        reason = "collision"

    def diag(n_steps):
        lane = np.full((n_steps, capacity), -1, dtype=np.int64)
        lane[:, :N_VEHICLES] = 0
        cmd = np.full((n_steps, capacity), np.nan, dtype=np.float32)
        cmd[:, :N_VEHICLES] = 0.5
        return {"lane_index": lane, "target_lane_index": lane.copy(),
                "commanded_accel": cmd, "commanded_steering": cmd.copy(),
                "crashed": np.zeros((n_steps, capacity), dtype=bool)}

    fingerprints = [{"entity_id": i, "class": "FakeVehicle",
                     "position_xy": [float(base[i, 2]), float(base[i, 3])],
                     "heading": 0.0, "speed": 1.0, "route": _FAKE_ROUTES[i]}
                    for i in range(N_VEHICLES)]
    return {
        "prefix_frames": np.asarray(prefix_frames, dtype=np.float32),
        "future_frames": np.asarray(future_frames, dtype=np.float32),
        "terminated": terminated,
        "truncated": False,
        "termination_reason": reason,
        "first_contact_step": first_contact,
        "diag_prefix": diag(len(prefix_frames)),
        "diag_future": diag(len(future_frames)),
        "fingerprints": fingerprints,
        "static_metadata": [{"entity_id": i, "class": "FakeVehicle",
                             "route": _FAKE_ROUTES[i], "behavior_params": {}}
                            for i in range(N_VEHICLES)],
        "initial_identity_count": N_VEHICLES,
        "max_road_vehicles": N_VEHICLES,
    }


@pytest.fixture()
def smoke_env(tmp_path, monkeypatch):
    """A smoke config redirected into tmp_path with a collection manifest."""
    monkeypatch.setattr(sp, "scan_historical_episode_seeds",
                        lambda *a, **k: np.array([1, 2, 3], dtype=np.int64))
    config = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    out_dir = tmp_path / "smoke"
    config["output_dir"] = str(out_dir)
    config["data_output"] = str(tmp_path / "data" / "stage2b.npz")
    config["rejection_ledger"] = str(tmp_path / "data" / "rejection_ledger.jsonl")
    config["collection_receipt"] = str(tmp_path / "data" / "collection_receipt.json")
    config["movement_lookup_json"] = str(tmp_path / "movement_lookup.json")
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    sp.prepare_collection(config, out_dir, config_path=cfg_path,
                          stage2b_root=tmp_path / "no_roles")
    return config, cfg_path, out_dir


def _run(config, resume=False, stop_after_accepted=None):
    return col.collect_stage2b(
        config, resume=resume, branch_rollout=_fake_branch_rollout,
        probe_fn=_fake_probe, stop_after_accepted=stop_after_accepted)


# --- contract + paired validity --------------------------------------------------


def test_collects_requested_pairs_and_contract(smoke_env):
    config, _, _ = smoke_env
    receipt = _run(config)
    assert receipt["accepted_pairs"] == config["n_pairs"]
    with np.load(config["data_output"], allow_pickle=True) as z:
        n = config["n_pairs"]
        assert z["object_history"].shape == (n, 10, CAPACITY, N_FEATURES)
        assert z["object_factual_future"].shape == (n, 20, CAPACITY, N_FEATURES)
        assert z["object_counterfactual_future"].shape == (n, 20, CAPACITY, N_FEATURES)
        assert len(np.unique(z["episode_seed"])) == n
        for field in ("object_history", "object_factual_future",
                      "object_counterfactual_future"):
            assert np.isfinite(z[field]).all()
        assert set(np.unique(z["split"]).tolist()) == {sp.SPLIT_LABELS["smoke"]}
        assert int(z["initial_identity_count"].max()) <= CAPACITY
        assert int(z["max_road_vehicles"].max()) <= CAPACITY
        assert int(z["object_capacity"]) == 20


def test_exact_per_stratum_quotas(smoke_env):
    config, _, _ = smoke_env
    _run(config)
    with np.load(config["data_output"], allow_pickle=True) as z:
        keys = list(zip(z["cf_action"].tolist(), z["cf_duration"].tolist()))
    from collections import Counter
    counts = Counter(keys)
    assert len(counts) == 6
    assert all(c == config["per_stratum"] for c in counts.values())


def test_counterfactual_action_differs_from_factual_first(smoke_env):
    config, _, _ = smoke_env
    _run(config)
    with np.load(config["data_output"], allow_pickle=True) as z:
        assert (z["cf_action"] != z["factual_actions"][:, 0]).all()
        assert set(np.unique(z["cf_duration"]).tolist()) <= {1, 3}


def test_identity_replay_and_prefix_consistency(smoke_env):
    config, _, _ = smoke_env
    receipt = _run(config)
    assert receipt["prefix_max_abs_diff"] <= 1e-6
    assert receipt["identity_max_abs_diff"] <= 1e-6


# --- ledger ----------------------------------------------------------------------


def test_rejection_ledger_records_every_attempt(smoke_env):
    config, _, _ = smoke_env
    receipt = _run(config)
    records = [json.loads(line) for line in
               Path(config["rejection_ledger"]).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    assert len(records) == receipt["attempts"]
    for rec in records:
        for key in ("attempt_index", "episode_seed", "decision", "reason"):
            assert key in rec, key
    rejected = [r for r in records if r["decision"] == "rejected"]
    assert all(r["reason"] for r in rejected)


def test_no_outcome_based_acceptance(smoke_env):
    """Acceptance must not read realized R / response bins / future error."""
    config, _, _ = smoke_env
    receipt = _run(config)
    forbidden = {"r_pos", "response_magnitude", "response_bin", "realized_r",
                 "v_evt", "l_suffix", "l_shuffled", "l_no"}
    assert not (forbidden & set(receipt))
    for line in Path(config["rejection_ledger"]).read_text(encoding="utf-8").splitlines():
        if line.strip():
            assert not (forbidden & set(json.loads(line)))


# --- capacity / identity rejection rules -------------------------------------------


def _raising_rollout(exc):
    def rollout(*args, **kwargs):
        raise exc
    return rollout


def test_capacity_exceeded_rejection(smoke_env):
    config, _, _ = smoke_env
    with pytest.raises(RuntimeError, match="collected 0|expected"):
        col.collect_stage2b(config, branch_rollout=_raising_rollout(
            col.CapacityExceededError("boom")), probe_fn=_fake_probe)
    records = [json.loads(line) for line in
               Path(config["rejection_ledger"]).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    assert records and all(r["reason"] == "capacity_exceeded" for r in records)


def test_unregistered_identity_rejection(smoke_env):
    config, _, _ = smoke_env
    with pytest.raises(RuntimeError):
        col.collect_stage2b(config, branch_rollout=_raising_rollout(
            col.UnregisteredIdentityError("boom")), probe_fn=_fake_probe)
    records = [json.loads(line) for line in
               Path(config["rejection_ledger"]).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    assert records and all(r["reason"] == "unregistered_identity" for r in records)


def test_identity_inconsistency_rejection(smoke_env):
    config, _, _ = smoke_env
    with pytest.raises(RuntimeError):
        col.collect_stage2b(config, branch_rollout=_raising_rollout(
            col.IdentityInconsistencyError("boom")), probe_fn=_fake_probe)
    records = [json.loads(line) for line in
               Path(config["rejection_ledger"]).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    assert records and all(r["reason"] == "identity_inconsistency" for r in records)


def test_early_termination_rejected(smoke_env):
    """The synthetic rollout terminates when the future action sum is large;
    such attempts must be rejected, never truncated into the dataset."""
    config, _, _ = smoke_env
    _run(config)
    with np.load(config["data_output"], allow_pickle=True) as z:
        assert (z["factual_actions"].sum(axis=1) <= 55.0).all()


# --- resume equivalence -------------------------------------------------------------


def _make_manifest_config(tmp_path, monkeypatch, name):
    monkeypatch.setattr(sp, "scan_historical_episode_seeds",
                        lambda *a, **k: np.array([1, 2, 3], dtype=np.int64))
    config = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    base = tmp_path / name
    out_dir = base / "out"
    config["output_dir"] = str(out_dir)
    config["data_output"] = str(base / "data" / "stage2b.npz")
    config["rejection_ledger"] = str(base / "data" / "l.jsonl")
    config["collection_receipt"] = str(base / "data" / "r.json")
    config["movement_lookup_json"] = str(base / "movement_lookup.json")
    base.mkdir(parents=True, exist_ok=True)
    cfg_path = base / "config.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    sp.prepare_collection(config, out_dir, config_path=cfg_path,
                          stage2b_root=base / "no_roles")
    return config


def test_resume_produces_identical_final_data(tmp_path, monkeypatch):
    config_a = _make_manifest_config(tmp_path, monkeypatch, "full")
    _run(config_a)
    with np.load(config_a["data_output"], allow_pickle=True) as z:
        full_seeds = z["episode_seed"].copy()
        full_history = z["object_history"].copy()

    config_b = _make_manifest_config(tmp_path, monkeypatch, "resumed")
    partial = _run(config_b, stop_after_accepted=4)
    assert partial["accepted_pairs"] == 4 and partial["partial"] is True
    resumed = _run(config_b, resume=True)
    assert resumed["accepted_pairs"] == config_b["n_pairs"]
    with np.load(config_b["data_output"], allow_pickle=True) as z:
        assert np.array_equal(z["episode_seed"], full_seeds)
        assert np.allclose(z["object_history"], full_history)


# --- manifest gating -----------------------------------------------------------------


def test_collection_fails_without_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "scan_historical_episode_seeds",
                        lambda *a, **k: np.array([1, 2, 3], dtype=np.int64))
    config = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    config["output_dir"] = str(tmp_path / "no_manifest_here")
    config["data_output"] = str(tmp_path / "d.npz")
    config["rejection_ledger"] = str(tmp_path / "l.jsonl")
    config["collection_receipt"] = str(tmp_path / "r.json")
    config["movement_lookup_json"] = str(tmp_path / "movement_lookup.json")
    with pytest.raises((FileNotFoundError, RuntimeError)):
        _run(config)


# --- development split assignment -----------------------------------------------------


def test_development_split_labels(tmp_path, monkeypatch):
    """A small development-shaped run assigns train/validation mechanically."""
    monkeypatch.setattr(sp, "scan_historical_episode_seeds",
                        lambda *a, **k: np.array([1, 2, 3], dtype=np.int64))
    monkeypatch.setattr(sp, "validate_stage2b_config", lambda cfg: None)
    config = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    config["role"] = "development"
    config["analysis_status"] = "DEVELOPMENT"
    config["per_stratum"] = 4
    config["n_pairs"] = 24
    config["train_per_stratum"] = 3
    config["validation_per_stratum"] = 1
    config["initial_vehicle_count"] = 10
    out_dir = tmp_path / "dev"
    config["output_dir"] = str(out_dir)
    config["data_output"] = str(tmp_path / "data" / "dev.npz")
    config["rejection_ledger"] = str(tmp_path / "data" / "dev.jsonl")
    config["collection_receipt"] = str(tmp_path / "data" / "dev_receipt.json")
    config["movement_lookup_json"] = str(tmp_path / "movement_lookup.json")
    cfg_path = tmp_path / "dev_config.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    sp.prepare_collection(config, out_dir, config_path=cfg_path,
                          stage2b_root=tmp_path / "no_roles")
    _run(config)
    with np.load(config["data_output"], allow_pickle=True) as z:
        split = z["split"]
        assert set(np.unique(split).tolist()) == {0, 1}
        for action in (0, 1, 2):
            for duration in (1, 3):
                mask = ((z["cf_action"] == action) & (z["cf_duration"] == duration))
                labels = split[mask]
                assert len(labels) == 4
                assert (labels[:3] == 0).all() and labels[3] == 1


# --- smoke semantic QA on collector output ---------------------------------------------


def test_smoke_semantic_qa_passes_on_valid_output(smoke_env):
    config, _, _ = smoke_env
    _run(config)
    with np.load(config["data_output"], allow_pickle=True) as z:
        data = {key: z[key] for key in z.files}
    qa = sd.smoke_semantic_qa(data)
    assert qa["pass"] is True
    assert qa["distinct_movement_signatures"] >= 3
    assert qa["route_nonempty_for_paired_valid"] is True


def test_smoke_semantic_qa_fails_on_missing_routes(smoke_env):
    config, _, _ = smoke_env
    _run(config)
    with np.load(config["data_output"], allow_pickle=True) as z:
        data = {key: z[key] for key in z.files}
    broken = []
    for raw in data["vehicle_metadata_json"]:
        records = json.loads(str(raw))
        for record in records:
            record["route"] = None
        broken.append(json.dumps(records))
    data["vehicle_metadata_json"] = np.asarray(broken, dtype=object)
    qa = sd.smoke_semantic_qa(data)
    assert qa["route_nonempty_for_paired_valid"] is False
    assert qa["pass"] is False


def test_smoke_status_is_engineering_only(smoke_env):
    config, _, _ = smoke_env
    receipt = _run(config)
    assert receipt["analysis_status"] == "ENGINEERING_SMOKE_ONLY"
    assert receipt["role"] == "smoke"
