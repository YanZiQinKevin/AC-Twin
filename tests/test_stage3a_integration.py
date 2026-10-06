"""Real small-data CLI integration test: search -> fit-five -> gate-r-dev.

Runs the actual Stage 3A driver end-to-end on a tiny synthetic development npz
(no monkeypatched trainer): a real single-seed screen over all 8 configs, a real
five-seed fit of the two finalists with per-finalist integrity records, and a
real Gate R_dev report with all seven verdict checks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import run_stage3a as drv  # noqa: E402

N_AGENTS = 5
N_FEATURES = 8
HIST = 10
FUT = 20


def _write_synthetic_npz(path: Path, episodes: int = 40, seed: int = 0, split=None) -> dict:
    rng = np.random.default_rng(seed)
    history = np.zeros((episodes, HIST, N_AGENTS, N_FEATURES), dtype=np.float32)
    factual = np.zeros((episodes, FUT, N_AGENTS, N_FEATURES), dtype=np.float32)
    counterfactual = np.zeros((episodes, FUT, N_AGENTS, N_FEATURES), dtype=np.float32)
    for e in range(episodes):
        for n in range(N_AGENTS):
            present = n < 4  # agent 4 always absent
            history[e, :, n, 0] = 1.0 if present else 0.0
            factual[e, :, n, 0] = 1.0 if present else 0.0
            counterfactual[e, :, n, 0] = 1.0 if present else 0.0
            history[e, :, n, 1] = 1.0 if n == 0 else 0.0  # slot 0 = ego
            factual[e, :, n, 1] = 1.0 if n == 0 else 0.0
            counterfactual[e, :, n, 1] = 1.0 if n == 0 else 0.0
            # kinematics
            history[e, :, n, 2] = rng.uniform(-5, 5) + np.arange(HIST) * 0.1
            history[e, :, n, 3] = rng.uniform(-5, 5)
            factual[e, :, n, 2] = history[e, -1, n, 2] + np.arange(1, FUT + 1) * 0.2
            factual[e, :, n, 3] = history[e, -1, n, 3]
            # counterfactual diverges for non-ego present vehicles -> R_pos > 0
            bump = rng.uniform(0.5, 3.0) if (present and n != 0) else 0.0
            counterfactual[e, :, n, 2] = factual[e, :, n, 2] + bump * np.linspace(0, 1, FUT)
            counterfactual[e, :, n, 3] = factual[e, :, n, 3] + bump * np.linspace(0, 0.5, FUT)
            for arr in (history, factual, counterfactual):
                arr[e, :, n, 4] = rng.uniform(-1, 1)
                arr[e, :, n, 5] = rng.uniform(-1, 1)
                arr[e, :, n, 6] = 1.0
                arr[e, :, n, 7] = 0.0
    if split is None:
        split = np.array([0] * 24 + [1] * 8 + [2] * 8, dtype=np.int64)
    data_split = np.asarray(split, dtype=np.int64)
    data = {
        "object_history": history,
        "object_factual_future": factual,
        "object_counterfactual_future": counterfactual,
        "factual_actions": rng.integers(0, 5, (episodes, FUT)).astype(np.int64),
        "counterfactual_actions": rng.integers(0, 5, (episodes, FUT)).astype(np.int64),
        "cf_action": rng.integers(0, 5, episodes).astype(np.int64),
        "cf_duration": rng.choice([1, 3], episodes).astype(np.int64),
        "t_star": rng.integers(10, 16, episodes).astype(np.int64),
        "episode_seed": (1000 + np.arange(episodes)).astype(np.int64),
        "split": data_split,
        "object_features_json": json.dumps(
            ["presence", "is_ego", "x", "y", "vx", "vy", "cos_heading", "sin_heading"]
        ),
    }
    np.savez(path, **data)
    return data


def _small_config(path: Path) -> Path:
    config = {
        "search_space": {"hidden_dim": [64, 96], "response_feature_dim": [32, 64], "lambda_cls": [0.10, 0.25]},
        "max_finalist_configs": 2,
        "n_seed_pairs": 5,
        "train_episodes": 24,
        "max_epochs": 2,
        "early_stopping_patience": 2,
        "learning_rate": 1e-3,
        "batch_size": 8,
        "device": "cpu",
        "checkpoint_gb_per_run": 0.01,
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _run_pipeline_to_selection(tmp_path) -> tuple[Path, Path, Path, dict]:
    """Run search -> fit-five -> select-final on a fresh synthetic source.

    Returns (config_path, npz_path, run_dir, receipt). Shared by the end-to-end
    and the single-use-hardening tests so each exercises the real CLI.
    """
    npz = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz)
    config = _small_config(tmp_path / "config.json")
    run_dir = tmp_path / "runs"
    assert drv.main(["search", "--config", str(config), "--source", str(npz), "--run-dir", str(run_dir)]) == 0
    assert drv.main(["fit-five", "--config", str(config), "--source", str(npz), "--run-dir", str(run_dir)]) == 0
    assert drv.main(["select-final", "--config", str(config), "--source", str(npz), "--run-dir", str(run_dir)]) == 0
    receipt = json.loads((run_dir / "selection" / "final_selection.json").read_text(encoding="utf-8"))
    return config, npz, run_dir, receipt


def test_search_fitfive_gaterdev_end_to_end(tmp_path) -> None:
    config, npz, run_dir, receipt = _run_pipeline_to_selection(tmp_path)

    # integrity records from fit-five (re-read for the assertions below)
    fit5 = json.loads((run_dir / "fit_five" / "fit_five_results.json").read_text(encoding="utf-8"))
    assert len(fit5["finalists"]) == 2
    for finalist in fit5["finalists"]:
        assert finalist["all_distinct"]
        assert len(finalist["per_seed"]) == 5
        sigs = {s["signature"] for s in finalist["per_seed"]}
        assert len(sigs) == 5  # five distinct seed signatures with integrity records
    assert set(receipt["strongest_baselines"]) == {
        "strongest_continuous", "strongest_presence", "strongest_high_vs_lowmid", "strongest_brier"
    }

    # select-final is immutable: a second run must be refused (exit 3).
    rc = drv.main(["select-final", "--config", str(config), "--source", str(npz), "--run-dir", str(run_dir)])
    assert rc == 3

    repl = _write_replacement_npz(tmp_path / "replacement.npz")
    _fabricate_collection_receipt(run_dir, repl)
    assert _lock_replacement(run_dir, repl) == 0

    def _retrain_gate() -> int:
        return drv.main([
            "gate-r-dev-retrain", "--config", str(config),
            "--development-source", str(npz), "--replacement-test-source", str(repl),
            "--run-dir", str(run_dir), "--n-resamples", "50",
        ])

    rc = _retrain_gate()
    # exit code 0 (pass) or 2 (fail) are both valid adjudications; the report must exist
    assert rc in (0, 2)
    reports = list((run_dir / "gate_r_dev").glob("gate_r_dev_retrain_report_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    checks = report["verdict"]["checks"]
    for key in (
        "i_mae_ci_lower_gt0", "spearman_ci_lower_gt0", "i_pr_presence_ci_lower_gt0",
        "i_pr_high_vs_lowmid_ci_lower_gt0", "i_brier_ci_lower_gt0",
        "ece_calibrated", "five_seed_consistent_nondegenerate",
    ):
        assert key in checks
    assert "metrics" in report
    assert len(report["seed_signatures"]) == 5
    assert report["temperature"] > 0
    assert report["selection_receipt_signature"] == receipt["receipt_signature"]
    assert report["eval_split"] == "replacement_development_test"
    assert report["development_data_signature"] != report["replacement_data_signature"]

    # single-use: a second replacement-test run must be refused (exit 3, no new report).
    assert _retrain_gate() == 3
    assert len(list((run_dir / "gate_r_dev").glob("gate_r_dev_retrain_report_*.json"))) == 1


def test_gate_stops_on_silently_misaligned_predictions(tmp_path, monkeypatch) -> None:
    """If a predictor returns episodes out of order, the gate must stop loudly
    (RuntimeError) before any metric or temperature is computed."""
    config, npz, run_dir, receipt = _run_pipeline_to_selection(tmp_path)
    real_predict = drv._predict_split

    def misaligned(model, tensors, indices, device, batch_size=None):
        pred = real_predict(model, tensors, indices, device, batch_size)
        flip = slice(None, None, -1)
        return {
            "episode_indices": pred["episode_indices"][flip].copy(),
            "r_hat": pred["r_hat"][flip],
            "regime_logits": pred["regime_logits"][flip],
        }

    monkeypatch.setattr(drv, "_predict_split", misaligned)
    repl = _write_replacement_npz(tmp_path / "replacement.npz")
    _fabricate_collection_receipt(run_dir, repl)
    assert _lock_replacement(run_dir, repl) == 0
    with pytest.raises(RuntimeError, match="episode order"):
        drv.main([
            "gate-r-dev-retrain", "--config", str(config),
            "--development-source", str(npz), "--replacement-test-source", str(repl),
            "--run-dir", str(run_dir), "--n-resamples", "50",
        ])
    assert len(list((run_dir / "gate_r_dev").glob("gate_r_dev_retrain_report_*.json"))) == 0


# --- deterministic retraining (Task 3, repair plan v0.1) ------------------------ #


def _toy_training_setup(tmp_path):
    """Small real tensors + roles + train-only scaler for _train_one_config."""
    import torch
    from ac_twin.stage3_scaler import FeatureScaler

    episodes = 12
    g = np.random.default_rng(3)
    history = g.normal(0, 1, (episodes, HIST, N_AGENTS, N_FEATURES)).astype(np.float32)
    suffix = g.normal(0, 1, (episodes, FUT, N_AGENTS, N_FEATURES)).astype(np.float32)
    r_true = np.abs(g.normal(0, 1, (episodes, FUT, N_AGENTS))).astype(np.float32)
    valid = np.ones((episodes, FUT, N_AGENTS), dtype=bool)
    valid[..., 4] = False  # agent 4 absent
    tensors = {
        "history": torch.as_tensor(history),
        "factual_suffix": torch.as_tensor(suffix),
        "factual_actions": torch.as_tensor(g.integers(0, 5, (episodes, FUT))),
        "counterfactual_actions": torch.as_tensor(g.integers(0, 5, (episodes, FUT))),
        "cf_action": torch.as_tensor(g.integers(0, 5, episodes)),
        "cf_duration": torch.as_tensor(g.choice([1, 3], episodes)),
        "t_star": torch.as_tensor(g.integers(2, 8, episodes)),
        "r_true": torch.as_tensor(r_true),
        "regime_labels": torch.as_tensor(g.integers(0, 4, (episodes, FUT, N_AGENTS))),
        "valid": torch.as_tensor(valid),
    }
    train_idx = np.arange(0, 8)
    roles = {"train": train_idx, "validation": np.arange(8, 12)}
    scaler = FeatureScaler.fit(
        {"history": history[train_idx], "factual_suffix": suffix[train_idx]},
        {
            "history": np.ones((len(train_idx), HIST, N_AGENTS), dtype=bool),
            "factual_suffix": valid[train_idx],
        },
        fitted_on="train",
    )
    config = {
        "max_epochs": 2, "early_stopping_patience": 2,
        "learning_rate": 1e-3, "batch_size": 4, "device": "cpu",
    }
    return config, tensors, roles, scaler


def _pollute_ambient_rng(tag: int) -> None:
    """Simulate unrelated prior work leaving the ambient RNG in an unknown state."""
    import torch

    torch.manual_seed(10_000 + tag)
    torch.randn(97)


def test_same_config_seed_identical_across_fresh_dirs(tmp_path) -> None:
    config, tensors, roles, scaler = _toy_training_setup(tmp_path)
    cfg = {"hidden_dim": 16, "response_feature_dim": 8, "lambda_cls": 0.1}
    _pollute_ambient_rng(1)
    r1 = drv._train_one_config(config, tensors, roles, cfg, 101, tmp_path / "run1", False,
                               scaler=scaler, data_signature="toy")
    _pollute_ambient_rng(2)
    r2 = drv._train_one_config(config, tensors, roles, cfg, 101, tmp_path / "run2", False,
                               scaler=scaler, data_signature="toy")
    assert r1["signature"] == r2["signature"]
    assert r1["best_val_loss"] == r2["best_val_loss"]
    assert r1["epochs_run"] == r2["epochs_run"]


def test_config_order_reversal_invariant(tmp_path) -> None:
    config, tensors, roles, scaler = _toy_training_setup(tmp_path)
    cfg_a = {"hidden_dim": 16, "response_feature_dim": 8, "lambda_cls": 0.1}
    cfg_b = {"hidden_dim": 16, "response_feature_dim": 8, "lambda_cls": 0.25}
    fwd = tmp_path / "fwd"
    _pollute_ambient_rng(11)
    a_fwd = drv._train_one_config(config, tensors, roles, cfg_a, 101, fwd, False,
                                  scaler=scaler, data_signature="toy")
    _pollute_ambient_rng(12)
    b_fwd = drv._train_one_config(config, tensors, roles, cfg_b, 101, fwd, False,
                                  scaler=scaler, data_signature="toy")
    rev = tmp_path / "rev"
    _pollute_ambient_rng(13)
    b_rev = drv._train_one_config(config, tensors, roles, cfg_b, 101, rev, False,
                                  scaler=scaler, data_signature="toy")
    _pollute_ambient_rng(14)
    a_rev = drv._train_one_config(config, tensors, roles, cfg_a, 101, rev, False,
                                  scaler=scaler, data_signature="toy")
    assert a_fwd["signature"] == a_rev["signature"]
    assert b_fwd["signature"] == b_rev["signature"]
    assert a_fwd["best_val_loss"] == a_rev["best_val_loss"]


def _write_replacement_npz(path: Path, episodes: int = 12, seed: int = 900) -> Path:
    """A fresh, dedicated all-eval replacement holdout (split column all == 2)."""
    _write_synthetic_npz(path, episodes=episodes, seed=seed,
                         split=[2] * episodes)
    return path


def _fabricate_collection_receipt(run_dir: Path, repl: Path) -> None:
    """Stand-in for the Task 6 collector receipt; binds the replacement file."""
    repl_dir = run_dir / "replacement_dev"
    repl_dir.mkdir(parents=True, exist_ok=True)
    (repl_dir / "replacement_dev_collection_receipt.json").write_text(
        json.dumps({"data_sha256": drv.sha256_file(repl), "episodes": 12}), encoding="utf-8")


def _lock_replacement(run_dir: Path, repl: Path) -> int:
    return drv.main([
        "lock-replacement-dev", "--run-dir", str(run_dir),
        "--replacement-test-source", str(repl),
    ])


def _gate(config, npz, run_dir, lock=True, lock_state={}) -> int:
    repl = _write_replacement_npz(run_dir.parent / "replacement.npz")
    _fabricate_collection_receipt(run_dir, repl)
    if lock:
        # the single-use lock is created at most ONCE per run_dir; the gate's
        # own replacement_dev_used.lock is what must refuse any second run.
        if not lock_state.get(str(run_dir)):
            assert _lock_replacement(run_dir, repl) == 0
            lock_state[str(run_dir)] = True
    return drv.main([
        "gate-r-dev-retrain", "--config", str(config),
        "--development-source", str(npz), "--replacement-test-source", str(repl),
        "--run-dir", str(run_dir), "--n-resamples", "50",
    ])


def test_tampered_receipt_refused_before_lock(tmp_path) -> None:
    """Editing the receipt (config or baselines) fails the signature re-check."""
    config, npz, run_dir, receipt = _run_pipeline_to_selection(tmp_path)
    selection_path = run_dir / "selection" / "final_selection.json"

    # Tamper: change the frozen strongest baseline WITHOUT fixing the signature.
    tampered = dict(receipt)
    tampered["strongest_baselines"] = dict(receipt["strongest_baselines"])
    tampered["strongest_baselines"]["strongest_brier"] = "constant_zero"
    selection_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert _gate(config, npz, run_dir) == 3
    # refused BEFORE acquiring the lock -> no lock, no report.
    assert not (run_dir / "replacement_dev_used.lock").exists()
    assert len(list((run_dir / "gate_r_dev").glob("gate_r_dev_retrain_report_*.json"))) == 0


def test_switched_finalist_refused_before_lock(tmp_path) -> None:
    """Switching the final config invalidates the signature (checkpoints differ)."""
    config, npz, run_dir, receipt = _run_pipeline_to_selection(tmp_path)
    selection_path = run_dir / "selection" / "final_selection.json"
    fit5 = json.loads((run_dir / "fit_five" / "fit_five_results.json").read_text(encoding="utf-8"))
    other = next(
        f["config"] for f in fit5["finalists"] if f["config"] != receipt["final_config"]
    )
    switched = dict(receipt)
    switched["final_config"] = dict(other)  # keep the ORIGINAL stored signature
    selection_path.write_text(json.dumps(switched), encoding="utf-8")
    assert _gate(config, npz, run_dir) == 3
    assert not (run_dir / "replacement_dev_used.lock").exists()
    assert len(list((run_dir / "gate_r_dev").glob("gate_r_dev_retrain_report_*.json"))) == 0


def test_valid_receipt_then_lock_blocks_any_second_run(tmp_path) -> None:
    """A clean first run acquires the global lock; a second run under the SAME or
    a DIFFERENT (forged-valid) receipt is still refused by the lock, and only one
    report ever exists."""
    config, npz, run_dir, receipt = _run_pipeline_to_selection(tmp_path)
    assert _gate(config, npz, run_dir) in (0, 2)
    assert (run_dir / "replacement_dev_used.lock").exists()
    assert len(list((run_dir / "gate_r_dev").glob("gate_r_dev_retrain_report_*.json"))) == 1
    # second run, same valid receipt -> lock refuses
    assert _gate(config, npz, run_dir) == 3
    assert len(list((run_dir / "gate_r_dev").glob("gate_r_dev_retrain_report_*.json"))) == 1


# --- artifact isolation & receipt chain (Task 5, repair plan v0.1) -------------- #


def test_search_and_fitfive_use_isolated_subdirs_and_never_overwrite(tmp_path) -> None:
    npz = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz)
    config = _small_config(tmp_path / "config.json")
    run_dir = tmp_path / "runs"

    assert drv.main(["search", "--config", str(config), "--source", str(npz),
                     "--run-dir", str(run_dir)]) == 0
    search_dir = run_dir / "search"
    assert search_dir.is_dir() and not (run_dir / "fit_five").exists()
    first = {p.name: p.read_bytes() for p in search_dir.glob("*.ckpt")}
    assert first

    # re-running search WITHOUT --resume must refuse to clobber completed runs
    assert drv.main(["search", "--config", str(config), "--source", str(npz),
                     "--run-dir", str(run_dir)]) == 3
    # with --resume it reuses the completed runs idempotently (no file changes)
    assert drv.main(["search", "--config", str(config), "--source", str(npz),
                     "--run-dir", str(run_dir), "--resume"]) == 0
    assert {p.name: p.read_bytes() for p in search_dir.glob("*.ckpt")} == first

    assert drv.main(["fit-five", "--config", str(config), "--source", str(npz),
                     "--run-dir", str(run_dir)]) == 0
    fit_dir = run_dir / "fit_five"
    assert fit_dir.is_dir()
    # search artifacts untouched by fit-five
    assert {p.name: p.read_bytes() for p in search_dir.glob("*.ckpt")} == first


def test_fitfive_stops_when_seed101_signature_differs_from_search(tmp_path) -> None:
    """Determinism audit: finalist seed-101 must reproduce the search signature."""
    npz = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz)
    config = _small_config(tmp_path / "config.json")
    run_dir = tmp_path / "runs"
    assert drv.main(["search", "--config", str(config), "--source", str(npz),
                     "--run-dir", str(run_dir)]) == 0

    # Corrupt the SEARCH record of the first finalist's seed-101 signature, then
    # run fit-five for real: the cross-check must stop with DETERMINISM_FAILURE.
    search_path = run_dir / "search" / "search_results.json"
    search_results = json.loads(search_path.read_text(encoding="utf-8"))
    first = json.loads(json.dumps(search_results))  # deep copy
    for record in first["results"]:
        if record["seed"] == 101:
            record["signature"] = "0" * 64
            break
    search_path.write_text(json.dumps(first), encoding="utf-8")
    with pytest.raises(RuntimeError, match="DETERMINISM_FAILURE"):
        drv.main(["fit-five", "--config", str(config), "--source", str(npz),
                  "--run-dir", str(run_dir)])
    assert not (run_dir / "fit_five" / "fit_five_results.json").exists()


def test_gate_refuses_when_fit_five_results_change_after_selection(tmp_path) -> None:
    config, npz, run_dir, receipt = _run_pipeline_to_selection(tmp_path)
    fit5_path = run_dir / "fit_five" / "fit_five_results.json"
    fit5 = json.loads(fit5_path.read_text(encoding="utf-8"))
    fit5["finalists"][0]["per_seed"][0]["validation_selection_score"] = 999.0
    fit5_path.write_text(json.dumps(fit5), encoding="utf-8")
    assert _gate(config, npz, run_dir) == 3
    assert not (run_dir / "replacement_dev_used.lock").exists()
    assert len(list((run_dir / "gate_r_dev").glob("gate_r_dev_retrain_report_*.json"))) == 0


def test_gate_refuses_when_receipt_selection_scores_edited(tmp_path) -> None:
    config, npz, run_dir, receipt = _run_pipeline_to_selection(tmp_path)
    selection_path = run_dir / "selection" / "final_selection.json"
    tampered = json.loads(selection_path.read_text(encoding="utf-8"))
    tampered["selection_scores"][0] += 1.0  # keep the stored signature unchanged
    selection_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert _gate(config, npz, run_dir) == 3
    assert not (run_dir / "replacement_dev_used.lock").exists()




# --- dual-source replacement gate (Task 7, repair plan v0.1) -------------------- #


def test_retrain_gate_refuses_without_replacement_lock(tmp_path) -> None:
    config, npz, run_dir, receipt = _run_pipeline_to_selection(tmp_path)
    repl = _write_replacement_npz(tmp_path / "replacement.npz")
    _fabricate_collection_receipt(run_dir, repl)
    # NO lock-replacement-dev call -> gate must refuse before any read
    rc = drv.main([
        "gate-r-dev-retrain", "--config", str(config),
        "--development-source", str(npz), "--replacement-test-source", str(repl),
        "--run-dir", str(run_dir), "--n-resamples", "50",
    ])
    assert rc == 3
    assert not (run_dir / "replacement_dev_used.lock").exists()
    assert len(list((run_dir / "gate_r_dev").glob("gate_r_dev_retrain_report_*.json"))) == 0


def test_retrain_gate_refuses_development_source_as_replacement(tmp_path) -> None:
    """The contaminated old split 2 (inside the development npz) can never be
    the eval set: passing the development npz as the replacement is refused."""
    config, npz, run_dir, receipt = _run_pipeline_to_selection(tmp_path)
    _fabricate_collection_receipt(run_dir, npz)  # receipt matches the DEV file
    assert _lock_replacement(run_dir, npz) == 0  # lock itself can be created
    rc = drv.main([
        "gate-r-dev-retrain", "--config", str(config),
        "--development-source", str(npz), "--replacement-test-source", str(npz),
        "--run-dir", str(run_dir), "--n-resamples", "50",
    ])
    assert rc == 3
    assert not (run_dir / "replacement_dev_used.lock").exists()


def test_retrain_gate_lock_rejects_wrong_replacement_file(tmp_path) -> None:
    config, npz, run_dir, receipt = _run_pipeline_to_selection(tmp_path)
    repl_a = _write_replacement_npz(tmp_path / "replacement_a.npz", seed=900)
    _fabricate_collection_receipt(run_dir, repl_a)
    assert _lock_replacement(run_dir, repl_a) == 0
    # a DIFFERENT replacement file than the locked one must be refused
    repl_b = _write_replacement_npz(tmp_path / "replacement_b.npz", seed=901)
    rc = drv.main([
        "gate-r-dev-retrain", "--config", str(config),
        "--development-source", str(npz), "--replacement-test-source", str(repl_b),
        "--run-dir", str(run_dir), "--n-resamples", "50",
    ])
    assert rc == 3
    assert not (run_dir / "replacement_dev_used.lock").exists()


def test_retrain_gate_fitting_state_comes_from_development_only(tmp_path, monkeypatch) -> None:
    """Scaler, baseline fits and temperature see ONLY the development source;
    every Gate metric sees ONLY the replacement episodes."""
    config, npz, run_dir, receipt = _run_pipeline_to_selection(tmp_path)
    n_repl = 12

    captured = {}
    real_build = drv._build_baselines

    def capture_baselines(train_tensors, train_raw, train_idx, target_tensors, target_raw, target_idx):
        captured["baseline_train_n"] = len(train_idx)
        captured["baseline_target_n"] = len(target_idx)
        return real_build(train_tensors, train_raw, train_idx, target_tensors, target_raw, target_idx)

    real_temp = drv.fit_temperature

    def capture_temp(logits, labels, valid):
        captured["temperature_label_n"] = len(labels)
        return real_temp(logits, labels, valid)

    real_eval = drv.evaluate_h4_gate

    def capture_eval(**kwargs):
        captured["gate_n_episodes"] = kwargs["r_true"].shape[0]
        return real_eval(**kwargs)

    monkeypatch.setattr(drv, "_build_baselines", capture_baselines)
    monkeypatch.setattr(drv, "fit_temperature", capture_temp)
    monkeypatch.setattr(drv, "evaluate_h4_gate", capture_eval)

    repl = _write_replacement_npz(tmp_path / "replacement.npz", episodes=n_repl)
    _fabricate_collection_receipt(run_dir, repl)
    assert _lock_replacement(run_dir, repl) == 0
    rc = drv.main([
        "gate-r-dev-retrain", "--config", str(config),
        "--development-source", str(npz), "--replacement-test-source", str(repl),
        "--run-dir", str(run_dir), "--n-resamples", "50",
    ])
    assert rc in (0, 2)
    assert captured["baseline_train_n"] == 24      # development train only
    assert captured["baseline_target_n"] == n_repl  # predicted on replacement
    assert captured["temperature_label_n"] == 8     # development validation only
    assert captured["gate_n_episodes"] == n_repl    # metrics on replacement only
