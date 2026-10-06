"""Task 1 tests: Stage 2 protocol, seed schedule, historical scan, lock, power."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage2_protocol as sp  # noqa: E402

CONFIRM_CONFIG = ROOT / "config" / "stage2_holdout_merge_confirm.json"
SMOKE_CONFIG = ROOT / "config" / "stage2_holdout_merge_smoke.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _confirm() -> dict:
    return _load(CONFIRM_CONFIG)


def _smoke() -> dict:
    return _load(SMOKE_CONFIG)


# --- config validation ---------------------------------------------------------


def test_confirm_config_validates():
    sp.validate_stage2_config(_confirm())


def test_smoke_config_validates():
    sp.validate_stage2_config(_smoke())


def test_formal_rejects_wrong_n_pairs():
    cfg = _confirm()
    cfg["n_pairs"] = 900
    with pytest.raises(ValueError):
        sp.validate_stage2_config(cfg)


def test_formal_rejects_retrain_fields():
    for key in ("learning_rate", "max_epochs", "early_stopping_patience",
                "weight_decay", "gradient_clip_norm", "training_seeds_to_fit"):
        cfg = _confirm()
        cfg["model"][key] = 0.1
        with pytest.raises(ValueError, match="retrain|train|forbid"):
            sp.validate_stage2_config(cfg)


def test_formal_rejects_wrong_analysis_status():
    cfg = _confirm()
    cfg["analysis_status"] = "ENGINEERING_SMOKE_ONLY"
    with pytest.raises(ValueError):
        sp.validate_stage2_config(cfg)


def test_formal_rejects_nonfrozen_bins():
    cfg = _confirm()
    cfg["response_bins"]["t1"] = 0.5
    with pytest.raises(ValueError):
        sp.validate_stage2_config(cfg)


def test_formal_rejects_frozen_bins_removed():
    cfg = _confirm()
    del cfg["response_bins"]
    with pytest.raises(ValueError):
        sp.validate_stage2_config(cfg)


def test_formal_rejects_smoke_output_dir():
    cfg = _confirm()
    cfg["output_dir"] = "experiments/stage2_confirmatory/smoke"
    with pytest.raises(ValueError):
        sp.validate_stage2_config(cfg)


def test_smoke_rejects_formal_n_pairs():
    cfg = _smoke()
    cfg["n_pairs"] = 1200
    with pytest.raises(ValueError):
        sp.validate_stage2_config(cfg)


def test_smoke_rejects_formal_output_dir():
    cfg = _smoke()
    cfg["output_dir"] = "experiments/stage2_confirmatory/merge"
    with pytest.raises(ValueError):
        sp.validate_stage2_config(cfg)


def test_smoke_rejects_confirm_status():
    cfg = _smoke()
    cfg["analysis_status"] = "CONFIRMATORY_HOLDOUT"
    with pytest.raises(ValueError):
        sp.validate_stage2_config(cfg)


def test_formal_rejects_train_validation_split():
    cfg = _confirm()
    cfg["split"] = {"train": 0.7, "val": 0.15}
    with pytest.raises(ValueError):
        sp.validate_stage2_config(cfg)


# --- seed schedule -------------------------------------------------------------


def test_seed_schedule_shape_range_unique():
    schedule = sp.make_seed_schedule(20270999, size=18000)
    assert schedule.shape == (18000,)
    assert schedule.dtype == np.int64
    assert int(schedule.min()) >= 1
    assert int(schedule.max()) <= 2147483646
    assert len(np.unique(schedule)) == 18000


def test_seed_schedule_deterministic():
    a = sp.make_seed_schedule(20270999, size=18000)
    b = sp.make_seed_schedule(20270999, size=18000)
    assert np.array_equal(a, b)


def test_seed_schedule_changes_with_seed():
    a = sp.make_seed_schedule(20270999, size=18000)
    b = sp.make_seed_schedule(20271000, size=18000)
    assert not np.array_equal(a, b)


def test_frozen_collection_seed_has_zero_historical_overlap():
    """The frozen collection_seed must yield a schedule disjoint from history."""
    historical = sp.scan_historical_episode_seeds(ROOT / "data")
    schedule = sp.make_seed_schedule(20270999, size=18000)
    sp.validate_seed_schedule(schedule, historical)  # must not raise


# --- historical seed scan -------------------------------------------------------


def test_scan_historical_episode_seeds_finds_known_files():
    historical = sp.scan_historical_episode_seeds(ROOT / "data")
    assert historical.dtype == np.int64
    assert len(historical) == len(np.unique(historical))
    # merge_2000 alone contributes 2000 unique seeds; union across files is larger.
    assert len(historical) >= 2000
    assert int(historical.min()) >= 1


def test_scan_historical_episode_seeds_handles_missing_dir(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        sp.scan_historical_episode_seeds(tmp_path / "does_not_exist")


def test_validate_seed_schedule_rejects_overlap():
    historical = np.array([100, 200, 300], dtype=np.int64)
    schedule = np.array([100, 400, 500], dtype=np.int64)
    with pytest.raises(ValueError, match="overlap|historical"):
        sp.validate_seed_schedule(schedule, historical)


def test_validate_seed_schedule_rejects_duplicates():
    historical = np.array([999999999], dtype=np.int64)
    schedule = np.array([5, 5, 6], dtype=np.int64)
    with pytest.raises(ValueError, match="duplicate|unique"):
        sp.validate_seed_schedule(schedule, historical)


def test_validate_seed_schedule_rejects_out_of_range():
    historical = np.array([999999999], dtype=np.int64)
    with pytest.raises(ValueError, match="range"):
        sp.validate_seed_schedule(np.array([0, 5, 6], dtype=np.int64), historical)
    with pytest.raises(ValueError, match="range"):
        sp.validate_seed_schedule(np.array([5, 6, 2147483647], dtype=np.int64), historical)


# --- frozen Gate S* baselines ---------------------------------------------------


def test_fit_frozen_baselines_shapes_and_finiteness():
    params = sp.fit_frozen_baselines(_confirm())
    assert set(params) >= {"beta_action_horizon", "beta_action_horizon_distance"}
    beta2 = np.asarray(params["beta_action_horizon"], dtype=np.float64)
    beta3 = np.asarray(params["beta_action_horizon_distance"], dtype=np.float64)
    assert beta2.shape == (3,)
    assert beta3.shape == (4,)
    assert np.isfinite(beta2).all() and np.isfinite(beta3).all()


def test_fit_frozen_baselines_deterministic():
    a = sp.fit_frozen_baselines(_confirm())
    b = sp.fit_frozen_baselines(_confirm())
    assert np.allclose(a["beta_action_horizon"], b["beta_action_horizon"])
    assert np.allclose(a["beta_action_horizon_distance"],
                       b["beta_action_horizon_distance"])


# --- protocol lock ---------------------------------------------------------------


def _stub_params_provider(config):
    return {
        "beta_action_horizon": np.zeros(3, dtype=np.float64),
        "beta_action_horizon_distance": np.zeros(4, dtype=np.float64),
        "probe_coefficients": np.zeros((65, 6), dtype=np.float64),
        "probe_coefficients_shuffled": np.zeros((65, 6), dtype=np.float64),
        "probe_scene_mean": np.zeros(6, dtype=np.float64),
    }


def test_write_protocol_lock_creates_lock_schedule_and_params(tmp_path):
    out_dir = tmp_path / "merge"
    lock = sp.write_protocol_lock(
        CONFIRM_CONFIG, out_dir, params_provider=_stub_params_provider)
    assert (out_dir / "protocol_lock.json").is_file()
    assert (out_dir / "seed_schedule.npy").is_file()
    assert (out_dir / "frozen_gate_s_star_params.npz").is_file()
    assert "lock_sha256" in lock
    schedule = np.load(out_dir / "seed_schedule.npy")
    assert schedule.shape == (18000,)


def test_lock_records_all_required_hashes(tmp_path):
    out_dir = tmp_path / "merge"
    sp.write_protocol_lock(CONFIRM_CONFIG, out_dir, params_provider=_stub_params_provider)
    lock = _load(out_dir / "protocol_lock.json")
    code = lock["code_sha256"]
    for name in ("run_stage2_confirmatory.py", "collect_stage2_holdout.py",
                 "ac_twin/stage2_protocol.py", "ac_twin/stage2_inference.py",
                 "ac_twin/stage2_statistics.py", "ac_twin/stage1_model.py",
                 "ac_twin/stage1_data.py", "ac_twin/stage1_statistics.py",
                 "ac_twin/stage1_plus_statistics.py", "ac_twin/checkpointing.py",
                 "ac_twin/object_centric.py", "ac_twin/progress.py"):
        assert name in code, name
    assert len(lock["checkpoint_sha256"]) == 15
    for key in ("config_sha256", "normalization_npz_sha256", "response_bins_sha256",
                "stage1_gate_s_report_sha256", "frozen_params_sha256",
                "seed_schedule_sha256", "stage1_source_data_sha256"):
        assert key in lock, key
    assert lock["donor_seeds"]["primary"] == 20272907
    assert list(lock["donor_seeds"]["alternates"]) == [20272917, 20272927, 20272937, 20272947]
    assert "python_version" in lock["environment"]
    assert "numpy_version" in lock["environment"]
    assert "torch_version" in lock["environment"]
    assert "highway_env_version" in lock["environment"]


def test_lock_refuses_to_overwrite(tmp_path):
    out_dir = tmp_path / "merge"
    sp.write_protocol_lock(CONFIRM_CONFIG, out_dir, params_provider=_stub_params_provider)
    with pytest.raises((FileExistsError, RuntimeError)):
        sp.write_protocol_lock(CONFIRM_CONFIG, out_dir, params_provider=_stub_params_provider)


def test_verify_lock_passes_and_detects_drift(tmp_path):
    out_dir = tmp_path / "merge"
    sp.write_protocol_lock(CONFIRM_CONFIG, out_dir, params_provider=_stub_params_provider)
    # Read-only verification passes.
    sp.verify_protocol_lock(out_dir)
    # Drift in the seed schedule must be detected.
    schedule = np.load(out_dir / "seed_schedule.npy")
    schedule[0] = schedule[1] if schedule[1] != schedule[0] else schedule[0] + 1
    np.save(out_dir / "seed_schedule.npy", schedule)
    with pytest.raises((ValueError, RuntimeError)):
        sp.verify_protocol_lock(out_dir)


# --- confirmatory power (Task 1) -------------------------------------------------


def test_five_statistics_order_is_frozen():
    assert sp.STAGE2_POWER_STATISTICS == (
        "c_presence", "v_zero", "c_high_low", "c_high_mid", "v_high")


def test_power_statistics_built_from_development():
    stats = sp.build_stage2_power_statistics(_confirm())
    assert set(stats) == set(sp.STAGE2_POWER_STATISTICS)
    pool_sizes = {int(getattr(fn, "pool_size", 0)) for fn in stats.values()}
    assert pool_sizes == {300}


def test_joint_power_n1200_75pct_near_091():
    """New five-condition joint power at n=1200 (75% scenario) ~ 0.91 +/- 0.02."""
    stats = sp.build_stage2_power_statistics(_confirm())
    out = sp.estimate_stage2_joint_power(
        statistics=stats, n=1200, monte_carlo_replicates=10000,
        seed=20261070, confidence_level=0.95, point_screen=0.05)
    power_75 = out["scenarios"]["75pct"]["joint"]
    assert 0.89 <= power_75 <= 0.93, power_75


def test_joint_power_is_and_not_product():
    stats = sp.build_stage2_power_statistics(_confirm())
    out = sp.estimate_stage2_joint_power(
        statistics=stats, n=1200, monte_carlo_replicates=2000,
        seed=20261070, confidence_level=0.95, point_screen=0.05)
    joint = out["scenarios"]["empirical"]["joint"]
    singles = [out["scenarios"]["empirical"]["single"][name]
               for name in sp.STAGE2_POWER_STATISTICS]
    product = float(np.prod(singles))
    # Joint of correlated statistics is generally not the product of marginals.
    assert 0.0 <= joint <= 1.0
    if all(0.0 < s < 1.0 for s in singles):
        assert joint != pytest.approx(product, abs=1e-3)
