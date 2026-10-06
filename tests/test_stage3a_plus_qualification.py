"""Stage 3A Plus Task 9: qualification-set collector.

Contract validation, schedule/hash binding, forbidden locations, historical
seed scanning, per-stratum equality, ledger audit (no R-/model-based
selection), resumable collection with an injected rollout, and the 12-episode
engineering smoke whose only legal state is ``ENGINEERING_SMOKE_ONLY``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import collect_stage3a_plus_qualification as cq  # noqa: E402
import prepare_stage3a_plus_power as pp  # noqa: E402
from ac_twin import stage3_data as s3d  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402

CAPACITY = 5
N_FEATURES = 8


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _frozen_setup(tmp_path, n=20, tag=""):
    """A real frozen qualification config + schedule in tmp (Task 8 output)."""
    base = tmp_path / tag if tag else tmp_path
    run_dir = base / "run"
    run_dir.mkdir(parents=True)
    report = {
        "plan_id": s3d.STAGE3A_PLUS_PLAN_ID,
        "artifact_type": "power_report",
        "status": "POWER_FEASIBLE",
        "selected_episode_count": n,
        "candidates": list(s3d.STAGE3A_PLUS_CANDIDATE_EPISODE_COUNTS),
        "target_power": s3d.STAGE3A_PLUS_TARGET_POWER,
        "monte_carlo_replicates": s3d.STAGE3A_PLUS_MONTE_CARLO_REPLICATES,
        "inner_bootstrap_resamples": s3d.STAGE3A_PLUS_INNER_BOOTSTRAP,
        "seed": s3d.STAGE3A_PLUS_POWER_SEED,
    }
    power_path = run_dir / "power" / "power_report.json"
    power_path.parent.mkdir(parents=True)
    power_path.write_text(json.dumps(report), encoding="utf-8")
    sha = sha256_file(power_path)
    paths = pp.qualification_paths(run_dir, data_root=base / "data")
    out = pp.freeze_qualification_artifacts(
        run_dir, report, np.arange(5000), s3d.STAGE3A_PLUS_COLLECTION_SEED,
        power_report_sha256=sha, paths=paths)
    config = json.loads(out["config_path"].read_text(encoding="utf-8"))
    return run_dir, paths, config, out


def _fake_probe(config):
    action_ids = np.arange(5, dtype=np.int64)
    probs = np.array([0.095, 0.45, 0.095, 0.18, 0.18], dtype=np.float64)
    return action_ids, probs / probs.sum(), CAPACITY


def _fake_branch_rollout(env_id, env_cfg, seed, prefix_actions, future_actions, capacity):
    """Deterministic pure function of (seed, prefix, future); F=8 frames with
    presence=1, non-ego, and position features so downstream QA/loading works."""
    rng = np.random.default_rng(int(seed))
    base = rng.normal(size=(capacity, N_FEATURES)).astype(np.float32)
    base[:, 0] = 1.0   # presence
    base[:, 1] = 0.0   # non-ego

    def step_frames(actions, start):
        frames = [start.copy()]
        state = start.copy()
        for a in np.asarray(actions, dtype=np.int64):
            state = state + (float(a) + 1.0) * 0.1
            frames.append(state.copy())
        return frames

    prefix_frames = step_frames(prefix_actions, base)
    future_frames = step_frames(future_actions, prefix_frames[-1])[1:]
    n_p, n_f = len(prefix_frames), len(future_frames)
    diag_p = {
        "lane_index": np.zeros((n_p, capacity), dtype=np.int64),
        "target_lane_index": np.zeros((n_p, capacity), dtype=np.int64),
        "commanded_accel": np.full((n_p, capacity), np.nan, dtype=np.float32),
        "commanded_steering": np.full((n_p, capacity), np.nan, dtype=np.float32),
        "crashed": np.zeros((n_p, capacity), dtype=bool),
    }
    diag_f = {
        "lane_index": np.zeros((n_f, capacity), dtype=np.int64),
        "target_lane_index": np.zeros((n_f, capacity), dtype=np.int64),
        "commanded_accel": np.full((n_f, capacity), np.nan, dtype=np.float32),
        "commanded_steering": np.full((n_f, capacity), np.nan, dtype=np.float32),
        "crashed": np.zeros((n_f, capacity), dtype=bool),
    }
    fp = [{"entity_id": i, "class": "FakeVehicle",
           "position_xy": [float(base[i, 2]), float(base[i, 3])],
           "heading": 0.0, "speed": 1.0, "route": None} for i in range(capacity)]
    return {
        "prefix_frames": np.asarray(prefix_frames, dtype=np.float32),
        "future_frames": np.asarray(future_frames, dtype=np.float32),
        "terminated": False, "truncated": False, "termination_reason": "",
        "first_contact_step": -1,
        "diag_prefix": diag_p, "diag_future": diag_f,
        "fingerprints": fp,
        "static_metadata": [{"entity_id": i, "class": "FakeVehicle",
                             "route": None, "behavior_params": {}}
                            for i in range(capacity)],
    }


def _write_fake_npz(path, seeds, actions, durations):
    E = len(seeds)
    rng = np.random.default_rng(0)
    hist = rng.normal(size=(E, 10, CAPACITY, N_FEATURES)).astype(np.float32)
    fact = rng.normal(size=(E, 20, CAPACITY, N_FEATURES)).astype(np.float32)
    cf = fact + 0.2
    for arr in (hist, fact, cf):
        arr[..., 0] = 1.0   # presence
        arr[..., 1] = 0.0   # non-ego
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        object_history=hist, object_factual_future=fact, object_counterfactual_future=cf,
        factual_actions=np.zeros((E, 20), dtype=np.int64),
        counterfactual_actions=np.zeros((E, 20), dtype=np.int64),
        cf_action=np.asarray(actions, dtype=np.int64),
        cf_duration=np.asarray(durations, dtype=np.int64),
        episode_seed=np.asarray(seeds, dtype=np.int64),
        t_star=np.full(E, 12, dtype=np.int64),
    )


# --------------------------------------------------------------------------- #
# Step 1: generation contract field-by-field
# --------------------------------------------------------------------------- #
def test_validate_accepts_the_frozen_config(tmp_path):
    run_dir, paths, config, _ = _frozen_setup(tmp_path)
    cq.validate_qualification_config(config, run_dir=run_dir)  # no raise


def test_validate_rejects_any_generation_contract_drift(tmp_path):
    run_dir, paths, config, _ = _frozen_setup(tmp_path)
    contract = s3d.MERGE_GENERATION_CONTRACT
    drifts = [
        ("horizon", 21), ("history_len", 9), ("t_star_low", 9), ("t_star_high", 16),
        ("vehicles_count", 24), ("observed_vehicles_count", 4), ("duration", 99),
        ("simulation_frequency", 10), ("policy_frequency", 4), ("determinism_checks", 10),
        ("intervention_durations", [1, 2]),
        ("action_weights", {"idle_maintain": 1.0}),
        ("scene", "merge"), ("env_id", "intersection-v0"),
        ("actions", [0, 1, 2, 3]),
        ("episodes_per_stratum", 3),
        ("max_attempts", int(config["n_pairs"]) * 14),
        ("stratified_equal_counts", False),
        ("no_posthoc_selection", False),
        ("collection_seed", 123),
    ]
    for key, bad in drifts:
        drifted = dict(config, **{key: bad})
        with pytest.raises(ValueError, match=key.split("_")[0]):
            cq.validate_qualification_config(drifted, run_dir=run_dir)
    # train/validation split keys are forbidden: every episode is an eval episode
    with pytest.raises(ValueError):
        cq.validate_qualification_config(dict(config, split={"train": 0.6}),
                                         run_dir=run_dir)
    with pytest.raises(ValueError):
        cq.validate_qualification_config(dict(config, plan_id="WRONG"),
                                         run_dir=run_dir)


# --------------------------------------------------------------------------- #
# Step 2/3: historical scan + forbidden locations
# --------------------------------------------------------------------------- #
def test_historical_scan_covers_all_data_except_own_dir(tmp_path):
    run_dir, paths, config, _ = _frozen_setup(tmp_path)
    data_root = tmp_path / "data" / "object_centric_stage"
    for sub, seeds in (("merge_2000", [11, 12]), ("stage2_merge_1200", [21]),
                       ("stage3a_replacement_dev", [31, 32, 33]),
                       ("stage3a_plus_qualification", [41])):
        d = data_root / sub
        d.mkdir(parents=True)
        np.savez_compressed(d / "x.npz", episode_seed=np.asarray(seeds, dtype=np.int64))
    historical = cq._historical_seeds_for_collection(config)
    assert set(historical.tolist()) == {11, 12, 21, 31, 32, 33}  # 41 excluded (own dir)


def test_validate_forbids_stage2_old_stage3_replacement_and_gatec_paths(tmp_path):
    run_dir, paths, config, _ = _frozen_setup(tmp_path)
    bad_outputs = [
        "data/object_centric_stage/stage2_merge_1200/q.npz",
        "data/object_centric_stage/merge_2000/q.npz",
        "data/object_centric_stage/stage3a_replacement_dev/q.npz",
        "experiments/stage3_routing/stage3a/q.npz",
        "experiments/stage3_routing/stage3a_retrain_v1/q.npz",
        "experiments/stage3_routing/gatec_holdout/q.npz",
        "data/object_centric_stage/somewhere_else/q.npz",  # no qualification segment
    ]
    for bad in bad_outputs:
        with pytest.raises(ValueError):
            cq.validate_qualification_config(dict(config, data_output=bad),
                                             run_dir=run_dir)
    # run-dir consistency
    with pytest.raises(ValueError):
        cq.validate_qualification_config(
            dict(config, collection_receipt=str(tmp_path / "elsewhere.json")),
            run_dir=run_dir)


def test_validate_binds_power_report_and_schedule(tmp_path):
    run_dir, paths, config, _ = _frozen_setup(tmp_path)
    # Tamper the power report AFTER freezing -> sha mismatch.
    power_path = run_dir / "power" / "power_report.json"
    power_path.write_text(json.dumps({"status": "POWER_FEASIBLE",
                                      "selected_episode_count": 20, "tampered": True}),
                          encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        cq.validate_qualification_config(config, run_dir=run_dir)


# --------------------------------------------------------------------------- #
# Step 4: strata equality + ledger audit (no R-based selection)
# --------------------------------------------------------------------------- #
def _finalize_inputs(tmp_path, *, equal=True, ledger_reasons=(), tag=""):
    run_dir, paths, config, out = _frozen_setup(tmp_path, n=20, tag=tag)
    schedule = out["schedule"]
    seeds = [int(s) for s in schedule[:20]]
    if equal:
        pairs = [(a, d) for a in range(5) for d in (1, 3)] * 2
    else:
        pairs = [(a, d) for a in range(5) for d in (1, 3)] * 2
        pairs[0] = (1, 1)  # break equality: (0,1) missing, (1,1) doubled
    actions = [p[0] for p in pairs]
    durations = [p[1] for p in pairs]
    _write_fake_npz(Path(config["data_output"]), seeds, actions, durations)
    ledger = Path(config["rejection_ledger"])
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "w", encoding="utf-8") as fh:
        for s in seeds:
            fh.write(json.dumps({"decision": "accepted", "episode_seed": int(s)}) + "\n")
        for reason in ledger_reasons:
            fh.write(json.dumps({"decision": "rejected", "reason": reason}) + "\n")
    receipt_path = Path(config["collection_receipt"])
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(
        {"partial": False, "accepted_pairs": 20, "attempts": 20}), encoding="utf-8")
    return config, run_dir, schedule


def test_finalize_requires_exactly_equal_strata(tmp_path):
    config, run_dir, schedule = _finalize_inputs(tmp_path, equal=False, tag="bad")
    with pytest.raises(RuntimeError, match="strata"):
        cq._finalize(config, run_dir, schedule, np.arange(5000), "fp")
    config, run_dir, schedule = _finalize_inputs(tmp_path, equal=True, tag="good")
    receipt = cq._finalize(config, run_dir, schedule, np.arange(5000), "fp")
    assert receipt["qa"]["strata_equal_counts"] is True
    assert receipt["qa"]["episodes_per_stratum"] == 2
    assert receipt["purpose"] == s3d.STAGE3A_PLUS_QUALIFICATION_PURPOSE
    assert receipt["plan_id"] == s3d.STAGE3A_PLUS_PLAN_ID
    assert receipt["data_sha256"]


def test_finalize_rejects_foreign_rejection_reasons(tmp_path):
    # Rejection may only cite engineering completeness rules — never realized
    # response, model predictions or gate metrics.
    config, run_dir, schedule = _finalize_inputs(
        tmp_path, ledger_reasons=["low_response_value"], tag="bad")
    with pytest.raises(RuntimeError, match="rejection"):
        cq._finalize(config, run_dir, schedule, np.arange(5000), "fp")
    config, run_dir, schedule = _finalize_inputs(
        tmp_path, ledger_reasons=["prefix_terminated", "non_finite"], tag="good")
    cq._finalize(config, run_dir, schedule, np.arange(5000), "fp")  # allowed


def test_finalize_rejects_seed_overlap_and_off_schedule(tmp_path):
    config, run_dir, schedule = _finalize_inputs(tmp_path, equal=True)
    with pytest.raises(RuntimeError, match="overlap"):
        cq._finalize(config, run_dir, schedule, np.arange(5000), "fp",
                     historical_override=np.asarray([int(schedule[0])]))


# --------------------------------------------------------------------------- #
# Step 5/6: end-to-end collection with injected rollout + smoke
# --------------------------------------------------------------------------- #
def test_collect_end_to_end_publishes_receipt_with_hash_chain(tmp_path):
    run_dir, paths, config, out = _frozen_setup(tmp_path, n=20)
    receipt = cq.collect_qualification(
        config, run_dir=run_dir, branch_rollout=_fake_branch_rollout,
        probe_fn=_fake_probe)
    assert receipt.get("partial") is not True
    assert receipt["purpose"] == s3d.STAGE3A_PLUS_QUALIFICATION_PURPOSE
    assert receipt["power_report_sha256"] == config["power_report_sha256"]
    assert receipt["seed_schedule_sha256"] == config["seed_schedule_sha256"]
    assert receipt["qa"]["strata_equal_counts"] is True
    data_output = Path(config["data_output"])
    assert data_output.is_file()
    with np.load(data_output, allow_pickle=True) as z:
        assert len(z["episode_seed"]) == 20
        assert set(np.unique(z["cf_duration"]).tolist()) == {1, 3}
    # A second full collection does not overwrite: the engine verifies the
    # existing dataset and returns without rewriting.
    before = data_output.read_bytes()
    receipt2 = cq.collect_qualification(
        config, run_dir=run_dir, branch_rollout=_fake_branch_rollout,
        probe_fn=_fake_probe)
    assert data_output.read_bytes() == before


def test_smoke_twelve_episodes_engineering_only(tmp_path):
    run_dir, paths, config, out = _frozen_setup(tmp_path, n=20)
    receipt = cq.collect_qualification(
        config, run_dir=run_dir, branch_rollout=_fake_branch_rollout,
        probe_fn=_fake_probe, smoke=True)
    assert receipt["state"] == cq.SMOKE_STATE == "ENGINEERING_SMOKE_ONLY"
    assert receipt["accepted_pairs"] == cq.SMOKE_EPISODE_COUNT == 12
    # Smoke never publishes the final dataset.
    assert not Path(config["data_output"]).exists()
    smoke_receipt = run_dir / "qualification" / "smoke_receipt.json"
    assert smoke_receipt.is_file()
    doc = json.loads(smoke_receipt.read_text(encoding="utf-8"))
    assert doc["state"] == "ENGINEERING_SMOKE_ONLY"
    # Smoke after a real collection is refused.
    cq.collect_qualification(config, run_dir=run_dir,
                             branch_rollout=_fake_branch_rollout, probe_fn=_fake_probe)
    with pytest.raises((ValueError, RuntimeError)):
        cq.collect_qualification(config, run_dir=run_dir,
                                 branch_rollout=_fake_branch_rollout,
                                 probe_fn=_fake_probe, smoke=True)


def test_resume_continues_from_smoke_checkpoint(tmp_path):
    run_dir, paths, config, out = _frozen_setup(tmp_path, n=20)
    cq.collect_qualification(config, run_dir=run_dir,
                             branch_rollout=_fake_branch_rollout,
                             probe_fn=_fake_probe, smoke=True)
    receipt = cq.collect_qualification(
        config, run_dir=run_dir, resume=True, branch_rollout=_fake_branch_rollout,
        probe_fn=_fake_probe)
    assert receipt.get("partial") is not True
    with np.load(config["data_output"], allow_pickle=True) as z:
        assert len(z["episode_seed"]) == 20
