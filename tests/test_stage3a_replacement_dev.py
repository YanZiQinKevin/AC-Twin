"""Task 6 tests: replacement development holdout — power sizing, collection
contract, seed disjointness and forbidden gate inputs (repair plan v0.1 §3 / §5).

Power-statistic tests are pure numpy (synthetic per-episode effect pools);
collector tests inject a deterministic ``branch_rollout`` so no highway_env /
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

import collect_stage3a_replacement_dev as coll  # noqa: E402
import prepare_stage3a_replacement_power as power  # noqa: E402
import run_stage3a as drv  # noqa: E402
from ac_twin import stage3_data as s3d  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage2_protocol import make_seed_schedule, validate_seed_schedule  # noqa: E402

from test_stage3a_integration import (  # noqa: E402
    _fabricate_collection_receipt,
    _lock_replacement,
    _run_pipeline_to_selection,
    _write_replacement_npz,
)

CAPACITY = 5
N_FEATURES = 8


# --- synthetic power statistics -------------------------------------------------


def _standardized_pool(size: int, mean: float, sd: float, seed: int) -> np.ndarray:
    """A pool with EXACTLY the requested mean/sd (no pool-construction noise)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size)
    x = (x - x.mean()) / x.std()
    return mean + sd * x


def _mean_statistic(pool: np.ndarray):
    pool = np.asarray(pool, dtype=np.float64)

    def stat(indices: np.ndarray) -> float:
        return float(np.mean(pool[np.asarray(indices)]))

    return stat


def _five_statistics(weak_mean: float, *, pool: int = 300) -> dict:
    """Four comfortably strong metrics + one tunable weak metric (i_brier)."""
    return {
        "i_mae": _mean_statistic(_standardized_pool(pool, 0.50, 0.50, seed=11)),
        "spearman": _mean_statistic(_standardized_pool(pool, 0.60, 0.40, seed=12)),
        "i_pr_presence": _mean_statistic(_standardized_pool(pool, 0.55, 0.45, seed=13)),
        "i_pr_high_vs_lowmid": _mean_statistic(_standardized_pool(pool, 0.45, 0.55, seed=14)),
        "i_brier": _mean_statistic(_standardized_pool(pool, weak_mean, 1.00, seed=15)),
    }


# --- power: candidate freezing, weakest-metric binding, hard stop ----------------


def test_power_rejects_non_frozen_candidates() -> None:
    stats = _five_statistics(0.17)
    with pytest.raises(ValueError, match="candidate"):
        power.estimate_replacement_dev_power(
            stats, pool_size=300, candidates=[100, 200], monte_carlo_replicates=50)


def test_power_rejects_non_frozen_seed() -> None:
    stats = _five_statistics(0.17)
    with pytest.raises(ValueError, match="seed"):
        power.estimate_replacement_dev_power(
            stats, pool_size=300, monte_carlo_replicates=50, seed=123)


def test_power_requires_exactly_the_five_statistics() -> None:
    stats = _five_statistics(0.17)
    del stats["i_brier"]
    with pytest.raises(ValueError, match="i_brier"):
        power.estimate_replacement_dev_power(stats, pool_size=300, monte_carlo_replicates=50)


def test_weakest_metric_binds_sample_size() -> None:
    """Weak i_brier (~0.17 sd 1) crosses 0.90 only at n=400; the strong four pass
    at 300. The AND chain must therefore freeze 400, never 300."""
    report = power.estimate_replacement_dev_power(
        _five_statistics(0.17), pool_size=300, monte_carlo_replicates=2000)
    assert report["status"] == "POWER_FEASIBLE"
    assert report["selected_episode_count"] == 400
    per = report["per_candidate"]
    assert per[300]["weakest"]["condition"] == "i_brier"
    assert per[300]["weakest"]["power"] < s3d.REPLACEMENT_DEV_TARGET_POWER
    assert per[400]["weakest"]["power"] >= s3d.REPLACEMENT_DEV_TARGET_POWER
    # every marginal is reported, and selection is the FIRST all-pass candidate
    for n in s3d.REPLACEMENT_DEV_CANDIDATE_EPISODE_COUNTS:
        assert set(per[n]["marginal_power"]) == set(power.POWER_STATISTIC_NAMES)


def test_infeasible_when_800_insufficient() -> None:
    """A near-zero effect never reaches 0.90 within 800 -> hard stop."""
    report = power.estimate_replacement_dev_power(
        _five_statistics(0.02), pool_size=300, monte_carlo_replicates=2000)
    assert report["status"] == s3d.REPLACEMENT_DEV_INFEASIBLE
    assert report["selected_episode_count"] is None


def test_power_nan_draws_count_as_failures() -> None:
    """A degenerate resample (NaN statistic) must lower power, never be dropped."""
    stats = _five_statistics(0.17)
    stats["i_brier"] = lambda indices: float("nan")
    report = power.estimate_replacement_dev_power(
        stats, pool_size=300, monte_carlo_replicates=200)
    assert report["per_candidate"][800]["marginal_power"]["i_brier"] == 0.0
    assert report["status"] == s3d.REPLACEMENT_DEV_INFEASIBLE


# --- power pool statistic construction (sign conventions match the gate) --------


def test_build_power_statistics_positive_when_model_dominates() -> None:
    episodes, steps, agents = 40, 20, 5
    rng = np.random.default_rng(7)
    r_true = np.abs(rng.normal(1.0, 0.5, (episodes, steps, agents)))
    valid = np.ones((episodes, steps, agents), dtype=bool)
    valid[..., 4] = False
    labels = rng.integers(0, 4, (episodes, steps, agents))

    model_r = r_true * 0.9  # lower error than the constant baseline
    baseline_r = np.full_like(r_true, float(r_true[valid].mean()))

    model_probs = np.full((episodes, steps, agents, 4), 0.05)
    for c in range(4):
        model_probs[..., c][labels == c] = 0.85  # near-perfect classifier
    baseline_probs = np.full((episodes, steps, agents, 4), 0.25)

    stats = power.build_power_statistics(
        model_r_hat=model_r, model_probs=model_probs,
        r_true=r_true, regime_labels=labels, valid_mask=valid,
        baseline_r_hat=baseline_r,
        baseline_probs_presence=baseline_probs,
        baseline_probs_high=baseline_probs,
        baseline_probs_brier=baseline_probs,
    )
    assert set(stats) == set(power.POWER_STATISTIC_NAMES)
    full = np.arange(episodes)
    for name, fn in stats.items():
        assert fn(full) > 0.0, f"{name} must be positive when the model dominates"


# --- artifact freezing -----------------------------------------------------------


def _write_power_report(run_dir: Path, selected: int | None, status: str) -> Path:
    power_dir = run_dir / "power"
    power_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "plan_id": power.REPAIR_PLAN_ID,
        "purpose": s3d.REPLACEMENT_DEV_PURPOSE,
        "status": status,
        "selected_episode_count": selected,
        "seed": s3d.REPLACEMENT_DEV_POWER_SEED,
    }
    path = power_dir / "power_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def _frozen_setup(tmp_path: Path, *, n_pairs: int = 12, historical=None):
    """Fabricate a feasible power report and freeze config + seed schedule."""
    run_dir = tmp_path / "run"
    report_path = _write_power_report(run_dir, n_pairs, "POWER_FEASIBLE")
    hist = np.asarray([] if historical is None else historical, dtype=np.int64)
    artifacts = power.freeze_replacement_artifacts(
        run_dir,
        selected_episode_count=n_pairs,
        power_report_sha256=sha256_file(report_path),
        historical_seeds=hist,
        paths=power.replacement_dev_paths(run_dir, data_root=tmp_path / "data"),
    )
    config = json.loads(Path(artifacts["config_path"]).read_text(encoding="utf-8"))
    return run_dir, config, artifacts


def test_freeze_refuses_when_infeasible(tmp_path) -> None:
    run_dir = tmp_path / "run"
    report_path = _write_power_report(run_dir, None, s3d.REPLACEMENT_DEV_INFEASIBLE)
    with pytest.raises(RuntimeError, match="INFEASIBLE"):
        power.freeze_replacement_artifacts(
            run_dir,
            selected_episode_count=None,
            power_report_sha256=sha256_file(report_path),
            historical_seeds=np.asarray([], dtype=np.int64),
            paths=power.replacement_dev_paths(run_dir, data_root=tmp_path / "data"),
        )
    # hard stop: no config, no schedule, nothing to collect with
    assert not (run_dir / "replacement_dev" / "replacement_dev_config.json").exists()
    assert not (run_dir / "replacement_dev" / "replacement_dev_seed_schedule.json").exists()


def test_freeze_chain_binds_config_schedule_and_power_report(tmp_path) -> None:
    run_dir, config, artifacts = _frozen_setup(tmp_path)
    schedule_json = json.loads(
        Path(artifacts["seed_schedule_path"]).read_text(encoding="utf-8"))
    assert config["seed_schedule_sha256"] == schedule_json["schedule_sha256"]
    assert config["power_report_sha256"] == schedule_json["power_report_sha256"]
    assert schedule_json["config_file_sha256"] == sha256_file(artifacts["config_path"])
    assert schedule_json["collection_seed"] == config["collection_seed"]
    assert config["n_pairs"] == 12
    assert config["analysis_status"] == s3d.REPLACEMENT_DEV_PURPOSE
    # the frozen artifacts pass the collector's own validation
    coll.validate_replacement_dev_config(config, run_dir=run_dir)


def test_freeze_refuses_overwrite(tmp_path) -> None:
    run_dir, config, artifacts = _frozen_setup(tmp_path)
    with pytest.raises(FileExistsError):
        power.freeze_replacement_artifacts(
            run_dir,
            selected_episode_count=12,
            power_report_sha256=config["power_report_sha256"],
            historical_seeds=np.asarray([], dtype=np.int64),
            paths=power.replacement_dev_paths(run_dir, data_root=tmp_path / "data"),
        )


def test_freeze_rejects_schedule_overlapping_history(tmp_path) -> None:
    run_dir = tmp_path / "run"
    report_path = _write_power_report(run_dir, 12, "POWER_FEASIBLE")
    schedule = make_seed_schedule(s3d.REPLACEMENT_DEV_COLLECTION_SEED, size=12 * 15)
    with pytest.raises(ValueError, match="overlap"):
        power.freeze_replacement_artifacts(
            run_dir,
            selected_episode_count=12,
            power_report_sha256=sha256_file(report_path),
            historical_seeds=schedule[:5],  # five schedule seeds "already used"
            paths=power.replacement_dev_paths(run_dir, data_root=tmp_path / "data"),
        )


# --- historical seed scanning (Stage 2 holdout INCLUDED for the replacement) -----


def _write_seed_npz(path: Path, seeds) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, episode_seed=np.asarray(seeds, dtype=np.int64))


def test_scan_all_episode_seeds_includes_stage2_holdout(tmp_path) -> None:
    """Unlike the Stage 2 scan (which excludes its own products), the replacement
    seed universe must include the sealed Stage 2 holdout and old merge split 2."""
    _write_seed_npz(tmp_path / "object_centric_stage/merge_2000/x.npz", [101, 202])
    _write_seed_npz(tmp_path / "object_centric_stage/stage2_merge_1200/y.npz", [303])
    _write_seed_npz(tmp_path / "pilot_2000/z.npz", [404])
    found = s3d.scan_all_episode_seeds(tmp_path)
    assert found.tolist() == [101, 202, 303, 404]


def test_collector_schedule_validation_against_history(tmp_path) -> None:
    """A schedule seed present in ANY historical manifest (incl. stage2_) fails."""
    _write_seed_npz(tmp_path / "stage2_merge_1200/y.npz", [555])
    historical = s3d.scan_all_episode_seeds(tmp_path)
    schedule = np.asarray([555, 777], dtype=np.int64)
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_schedule(schedule, historical)


# --- collection config contract ---------------------------------------------------


def test_config_requires_exact_purpose(tmp_path) -> None:
    run_dir, config, _ = _frozen_setup(tmp_path)
    config["analysis_status"] = "CONFIRMATORY_HOLDOUT"
    with pytest.raises(ValueError, match="purpose"):
        coll.validate_replacement_dev_config(config, run_dir=run_dir)


def test_config_generation_contract_is_exact(tmp_path) -> None:
    """Same scenario/actions/intensities/durations/onset rules as the old merge;
    any deviation is refused."""
    run_dir, config, _ = _frozen_setup(tmp_path)
    deviations = {
        "env_id": "roundabout-v0",
        "intervention_durations": [1, 2, 3],
        "t_star_low": 9,
        "t_star_high": 16,
        "horizon": 19,
        "history_len": 9,
        "vehicles_count": 24,
        "observed_vehicles_count": 6,
        "duration": 99,
        "simulation_frequency": 14,
        "policy_frequency": 4,
        "determinism_checks": 10,
    }
    for key, bad in deviations.items():
        mutated = dict(config)
        mutated[key] = bad
        with pytest.raises(ValueError, match=key):
            coll.validate_replacement_dev_config(mutated, run_dir=run_dir)
    mutated = dict(config)
    mutated["action_weights"] = {"idle_maintain": 0.5, "faster": 0.2, "slower": 0.2, "other": 0.1}
    with pytest.raises(ValueError, match="action_weights"):
        coll.validate_replacement_dev_config(mutated, run_dir=run_dir)


def test_config_forbids_split_key(tmp_path) -> None:
    run_dir, config, _ = _frozen_setup(tmp_path)
    config["split"] = {"train": 0.7}
    with pytest.raises(ValueError, match="split"):
        coll.validate_replacement_dev_config(config, run_dir=run_dir)


def test_config_rejects_n_pairs_not_matching_power_report(tmp_path) -> None:
    run_dir, config, _ = _frozen_setup(tmp_path)
    config["n_pairs"] = 13
    with pytest.raises(ValueError, match="n_pairs"):
        coll.validate_replacement_dev_config(config, run_dir=run_dir)


def test_config_rejects_tampered_power_report(tmp_path) -> None:
    run_dir, config, _ = _frozen_setup(tmp_path)
    report_path = run_dir / "power" / "power_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["selected_episode_count"] = 999  # tamper AFTER freezing
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="power_report"):
        coll.validate_replacement_dev_config(config, run_dir=run_dir)


def test_config_rejects_forbidden_output_paths(tmp_path) -> None:
    run_dir, config, _ = _frozen_setup(tmp_path)
    forbidden = [
        "data/object_centric_stage/stage2_merge_1200/object_twinworld_stage2_merge_1200.npz",
        "data/object_centric_stage/merge_2000/object_twinworld_2000.npz",
        "data/object_centric_stage/gatec_merge_id/x.npz",
        "experiments/stage3_routing/stage3a/legacy.npz",
    ]
    for bad in forbidden:
        mutated = dict(config)
        mutated["data_output"] = bad
        with pytest.raises(ValueError):
            coll.validate_replacement_dev_config(mutated, run_dir=run_dir)


# --- forbidden replacement gate inputs --------------------------------------------


def test_replacement_source_helper_rejects_stage2_path_and_hash() -> None:
    with pytest.raises(ValueError, match="sealed"):
        s3d.assert_replacement_source_allowed(s3d.STAGE2_HOLDOUT_LOGICAL_PATH)
    with pytest.raises(ValueError, match="sealed"):
        s3d.assert_replacement_source_allowed(
            None, file_sha256=s3d.STAGE2_HOLDOUT_FILE_SHA256)


def test_replacement_source_helper_rejects_old_development_source() -> None:
    with pytest.raises(ValueError, match="merge_2000"):
        s3d.assert_replacement_source_allowed(
            "data/object_centric_stage/merge_2000/object_twinworld_2000.npz")


def test_replacement_source_helper_rejects_gatec_paths(tmp_path) -> None:
    for bad in ("data/object_centric_stage/gatec_merge_id/x.npz",
                "experiments/stage3_routing/gatec/y.npz"):
        with pytest.raises(ValueError, match="Gate C"):
            s3d.assert_replacement_source_allowed(bad)
    # a genuine replacement path is allowed
    s3d.assert_replacement_source_allowed(
        tmp_path / "stage3a_replacement_dev" / "replacement.npz")


def test_gate_loader_rejects_forbidden_paths(tmp_path) -> None:
    forbidden = tmp_path / "gatec_merge_id" / "replacement.npz"
    forbidden.parent.mkdir(parents=True)
    _write_replacement_npz(forbidden)
    with pytest.raises(ValueError, match="Gate C"):
        drv.load_replacement_tensors(forbidden)


def test_gate_refuses_gatec_path_before_burning_single_use_lock(tmp_path) -> None:
    """A Gate C path must be refused BEFORE the one-shot replacement lock is
    consumed — a refusal must never waste the single adjudication."""
    config, npz, run_dir, receipt = _run_pipeline_to_selection(tmp_path)
    forbidden = tmp_path / "gatec_merge_id" / "replacement.npz"
    forbidden.parent.mkdir(parents=True)
    _write_replacement_npz(forbidden)
    _fabricate_collection_receipt(run_dir, forbidden)
    assert _lock_replacement(run_dir, forbidden) == 0
    rc = drv.main([
        "gate-r-dev-retrain", "--config", str(config),
        "--development-source", str(npz), "--replacement-test-source", str(forbidden),
        "--run-dir", str(run_dir), "--n-resamples", "50",
    ])
    assert rc == 3
    assert not (run_dir / "replacement_dev_used.lock").exists()


# --- collector end-to-end (injected deterministic rollout) -------------------------


def _fake_probe(config):
    action_ids = np.arange(5, dtype=np.int64)
    probs = np.array([0.095, 0.45, 0.095, 0.18, 0.18], dtype=np.float64)
    probs /= probs.sum()
    return action_ids, probs, CAPACITY


def _fake_branch_rollout(env_id, env_cfg, seed, prefix_actions, future_actions, capacity):
    """Deterministic pure function of (seed, prefix, future) -> branch result."""
    rng = np.random.default_rng(int(seed))
    base = rng.normal(size=(capacity, N_FEATURES)).astype(np.float32)
    base[:, 0] = 1.0  # presence
    prefix_actions = np.asarray(prefix_actions, dtype=np.int64)
    future_actions = np.asarray(future_actions, dtype=np.int64)

    def step_frames(actions, start):
        frames = [start.copy()]
        state = start.copy()
        for a in actions:
            state = state + (float(a) + 1.0) * 0.1
            frames.append(state.copy())
        return frames

    prefix_frames = step_frames(prefix_actions, base)
    future_frames = step_frames(future_actions, prefix_frames[-1])[1:]

    terminated = False
    first_contact = -1
    reason = ""
    if float(future_actions.sum()) > 55.0:  # synthetic rejection path
        terminated = True
        keep = max(1, len(future_frames) // 2)
        future_frames = future_frames[:keep]
        first_contact = keep - 1
        reason = "collision"

    n_future, n_prefix = len(future_frames), len(prefix_frames)

    def diag(n):
        return {
            "lane_index": np.zeros((n, capacity), dtype=np.int64),
            "target_lane_index": np.zeros((n, capacity), dtype=np.int64),
            "commanded_accel": np.full((n, capacity), np.nan, dtype=np.float32),
            "commanded_steering": np.full((n, capacity), np.nan, dtype=np.float32),
            "crashed": np.zeros((n, capacity), dtype=bool),
        }

    return {
        "prefix_frames": np.asarray(prefix_frames, dtype=np.float32),
        "future_frames": np.asarray(future_frames, dtype=np.float32),
        "terminated": terminated,
        "truncated": False,
        "termination_reason": reason,
        "first_contact_step": first_contact,
        "diag_prefix": diag(n_prefix),
        "diag_future": diag(n_future),
        "fingerprints": [{"entity_id": i, "class": "FakeVehicle",
                          "position_xy": [float(base[i, 2]), float(base[i, 3])],
                          "heading": 0.0, "speed": 1.0, "route": None}
                         for i in range(capacity)],
        "static_metadata": [{"entity_id": i, "class": "FakeVehicle",
                             "route": None, "behavior_params": {}}
                            for i in range(capacity)],
    }


def test_collect_end_to_end_contract_and_receipt(tmp_path) -> None:
    run_dir, config, artifacts = _frozen_setup(tmp_path)
    receipt = coll.collect_replacement_dev(
        config, run_dir=run_dir,
        branch_rollout=_fake_branch_rollout, probe_fn=_fake_probe)

    data_output = Path(config["data_output"])
    assert data_output.is_file()
    schedule = make_seed_schedule(config["collection_seed"], size=config["max_attempts"])
    historical_here = np.asarray([], dtype=np.int64)
    with np.load(data_output, allow_pickle=True) as z:
        assert z["object_history"].shape[0] == 12
        assert set(np.unique(z["cf_duration"]).tolist()) == {1, 3}
        seeds = z["episode_seed"]
        assert len(np.unique(seeds)) == 12
        assert set(seeds.tolist()) <= set(schedule.tolist())
        assert not set(seeds.tolist()) & set(historical_here.tolist())
        for field in ("object_history", "object_factual_future", "object_counterfactual_future"):
            assert np.isfinite(z[field]).all()

    # the published receipt binds purpose, data, schedule, power report and code
    receipt_path = run_dir / "replacement_dev" / "replacement_dev_collection_receipt.json"
    published = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert published["purpose"] == s3d.REPLACEMENT_DEV_PURPOSE
    assert published["data_sha256"] == sha256_file(data_output)
    assert published["seed_schedule_sha256"] == config["seed_schedule_sha256"]
    assert published["power_report_sha256"] == config["power_report_sha256"]
    assert published["accepted_pairs"] == 12
    assert published["qa"]["episode_count"] == 12
    assert published["qa"]["duration_set"] == [1, 3]
    assert published["qa"]["historical_seed_overlap"] == 0
    assert "collect_stage3a_replacement_dev.py" in published["code_sha256"]
    assert "collect_stage2_holdout.py" in published["code_sha256"]

    # the collected file satisfies the Task 7 gate loader (all-eval, durations {1,3})
    loaded = drv.load_replacement_tensors(data_output)
    assert loaded["roles"]["eval"].tolist() == list(range(12))
    assert loaded["data_signature"] == sha256_file(data_output)
    # (the Task 7 lock command's selection-receipt precondition is covered by the
    # integration tests; here the receipt's data_sha256 binding is what matters)
    assert published["data_sha256"] == loaded["data_signature"]


def test_collect_refuses_config_drift_after_freeze(tmp_path) -> None:
    run_dir, config, _ = _frozen_setup(tmp_path)
    config["n_pairs"] = 11  # any post-freeze edit breaks the fingerprint chain
    with pytest.raises(ValueError, match="n_pairs"):
        coll.collect_replacement_dev(
            config, run_dir=run_dir,
            branch_rollout=_fake_branch_rollout, probe_fn=_fake_probe)
    assert not Path(config["data_output"]).exists()


def test_collect_resume_produces_identical_data(tmp_path) -> None:
    run_a, config_a, _ = _frozen_setup(tmp_path / "a")
    coll.collect_replacement_dev(
        config_a, run_dir=run_a,
        branch_rollout=_fake_branch_rollout, probe_fn=_fake_probe)

    run_b, config_b, _ = _frozen_setup(tmp_path / "b")
    partial = coll.collect_replacement_dev(
        config_b, run_dir=run_b, stop_after_accepted=4,
        branch_rollout=_fake_branch_rollout, probe_fn=_fake_probe)
    assert partial["partial"] is True
    assert not Path(config_b["data_output"]).exists()  # nothing published early
    coll.collect_replacement_dev(
        config_b, run_dir=run_b, resume=True,
        branch_rollout=_fake_branch_rollout, probe_fn=_fake_probe)

    with np.load(config_a["data_output"], allow_pickle=True) as za, \
            np.load(config_b["data_output"], allow_pickle=True) as zb:
        assert np.array_equal(za["episode_seed"], zb["episode_seed"])
        assert np.allclose(za["object_history"], zb["object_history"])
        assert np.allclose(za["object_counterfactual_future"], zb["object_counterfactual_future"])
