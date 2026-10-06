"""Pre-lock rehearsal for the Stage 4A v4 recovery chain (plan point 8/9).

This builds a FULLY SYNTHETIC v1->v2->v3->v4 recovery chain under an independent
temporary directory and drives the REAL, unmodified command path:

    write_recovery_lock_v4 -> verify_recovery_lock -> cmd_select_candidate
        -> candidate_selection.json -> cmd_adjudicate_dev_screen -> final report

It never touches the real risk-validation or development-screen data: every artifact
(predictions, risk dataset, fit receipt, the five fitted models, the three parent
locks, the frozen checkpoints/scaler/binding) is fabricated small and self-consistent
so the genuine production functions can be exercised end-to-end before the real v4
lock is frozen. It also proves the write-once selection/screen behaviour and that any
post-lock tampering with a reused artifact is rejected.
"""
from __future__ import annotations

import hashlib
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
from ac_twin import stage4_recovery as s4rec  # noqa: E402
from ac_twin import stage4_risk as s4r  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
import run_stage4a as driver  # noqa: E402

# The rehearsal fixes the runner "before" hash to a sentinel that differs from the
# current runner; the patch receipt and the fabricated v3 lock agree on it.
RUNNER_BEFORE = "0" * 64


def _write_json(path: Path, payload) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return sha256_file(path)


def _make_frozen_stage3r(base: Path) -> dict:
    """Fabricate dummy frozen checkpoints / scaler / binding and their hashes."""
    checkpoints: dict[str, str] = {}
    for pair_index, seed in enumerate(s4p.EXPERT_ROUTER_SEEDS):
        for condition in ("full", "history"):
            p = s4p.expert_checkpoint_path(base, pair_index, condition, seed)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f"expert-{pair_index}-{condition}-{seed}".encode())
            checkpoints[f"pair{pair_index}_fulltrain_{condition}_seed{seed}"] = sha256_file(p)
        e2e = s4p.e2e_checkpoint_path(base, pair_index, seed)
        e2e.parent.mkdir(parents=True, exist_ok=True)
        e2e.write_bytes(f"e2e-{pair_index}-{seed}".encode())
        checkpoints[f"end_to_end_mixture_pair{pair_index}_s{seed}"] = sha256_file(e2e)
    scaler = base / "scaler.json"
    scaler.write_text(json.dumps({"scaler": "synthetic"}), encoding="utf-8")
    binding = base.parent / "stage3b" / "input_binding_receipt.json"
    binding.parent.mkdir(parents=True, exist_ok=True)
    binding.write_text(json.dumps({"binding": "synthetic"}), encoding="utf-8")
    return {
        "checkpoint_sha256": checkpoints,
        "scaler_path": str(scaler),
        "scaler_sha256": sha256_file(scaler),
        "stage3b_binding_path": str(binding),
        "stage3b_binding_sha256": sha256_file(binding),
    }


def _make_dataset(path: Path) -> np.ndarray:
    """Synthetic 5000-episode risk dataset; DISAGREEMENT (D_expert_mean) is the signal."""
    rng = np.random.default_rng(20260911)
    n = s4p.N_PAIRS
    # Ten strata of 500 with the mechanical 300/100/100 split.
    cf_action = np.repeat(np.array(s4p.CF_ACTIONS), 1000).astype(np.int64)
    cf_duration = np.tile(np.repeat(np.array(s4p.CF_DURATIONS), 500), 5).astype(np.int64)
    episode_seed = (20260911 + np.arange(n)).astype(np.int64)
    roles = np.concatenate([
        np.zeros(300, dtype=np.int64),
        np.ones(100, dtype=np.int64),
        np.full(100, 2, dtype=np.int64),
    ])
    roles = np.tile(roles, 10)  # 500 per stratum block -> 3000/1000/1000 overall

    risk_target = np.abs(rng.normal(loc=0.5, scale=0.2, size=n)) + 0.05
    features: dict[str, np.ndarray] = {}
    # The predictive signal: D_expert tracks the true error.
    features["D_expert_mean"] = risk_target + 0.02 * rng.normal(size=n)
    features["D_expert_max"] = features["D_expert_mean"] * 1.5
    # Noise-only uncertainty / geometry / response so DISAGREEMENT beats the baseline.
    for name in ("U_seed_mean", "U_seed_max"):
        features[name] = np.abs(rng.normal(size=n))
    for name in s4r.RESPONSE_FEATURES:
        features[name] = np.abs(rng.normal(size=n))
    for name in ("geo_dist_mean", "geo_dist_min", "geo_dist_max",
                 "geo_relspeed_mean", "geo_relspeed_max", "geo_visible_count"):
        features[name] = np.abs(rng.normal(size=n))
    for a in range(5):
        features[f"action_{a}"] = (cf_action == a).astype(np.float64)
    features["duration"] = cf_duration.astype(np.float64)
    features["horizon"] = np.full(n, 20.0)

    primary_error_by_pair = np.stack(
        [risk_target + 0.01 * p * rng.normal(size=n) for p in range(5)]
    ).astype(np.float64)
    constant_error_by_pair = primary_error_by_pair + 0.001
    e2e_error_by_pair = primary_error_by_pair - 0.001

    payload = {
        "risk_target": risk_target.astype(np.float64),
        "primary_error_by_pair": primary_error_by_pair,
        "constant_error_by_pair": constant_error_by_pair,
        "e2e_error_by_pair": e2e_error_by_pair,
        "roles": roles,
        "cf_action": cf_action,
        "cf_duration": cf_duration,
        "episode_seed": episode_seed,
        "top_error_threshold": np.float64(np.quantile(risk_target, 0.80)),
    }
    for name, value in features.items():
        payload[f"feature__{name}"] = np.asarray(value, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return risk_target


def _fit_five_models(models_dir: Path, dataset_path: Path) -> dict[str, str]:
    """Fit the five production models on synthetic risk-train features; save to v3."""
    with np.load(dataset_path, allow_pickle=False) as z:
        features = {k[len("feature__"):]: z[k] for k in z.files if k.startswith("feature__")}
        target = z["risk_target"]
        roles = z["roles"]
    train = roles == 0
    train_features = {k: v[train] for k, v in features.items()}
    train_target = target[train]
    config = {
        "ridge_alpha": 1.0, "winsor_lower_q": 0.01, "winsor_upper_q": 0.99,
        "histgb": {"learning_rate": 0.05, "max_iter": 10, "max_leaf_nodes": 8,
                   "min_samples_leaf": 5, "l2_regularization": 1.0, "early_stopping": False},
        "mlp": {"hidden_dim": 8, "max_epochs": 3, "patience": 2,
                "batch_size": 64, "learning_rate": 1e-3},
    }
    specs = [
        (s4r.CANDIDATE_COMBINED, "ridge"),
        (s4r.CANDIDATE_COMBINED, "histgb"),
        (s4r.CANDIDATE_COMBINED, "mlp"),
        (s4r.BASELINE_GEOMETRY, "ridge"),
        (s4r.CANDIDATE_RESPONSE, "ridge"),
    ]
    paths: dict[str, str] = {}
    for name, impl in specs:
        model = s4r.fit_risk_candidate(name, train_features, train_target, config,
                                       implementation=impl)
        out = models_dir / f"{name.lower()}_{impl}.pt"
        model.save(out)
        paths[f"{name}:{impl}"] = str(out)
    return paths


def _base_config(tmp: Path, v4_run: Path, stage3r: Path, data_output: Path) -> dict:
    return {
        "plan_id": s4p.PLAN_ID, "purpose": s4p.PURPOSE, "analysis_status": s4p.PURPOSE,
        "experiment_name": "stage4a_merge_risk_dev", "scene": "merge", "env_id": "merge-v0",
        "run_dir": str(v4_run), "output_dir": str(v4_run / "collection"),
        "data_output": str(data_output),
        "rejection_ledger": str(tmp / "rejection_ledger.jsonl"),
        "collection_receipt": str(tmp / "v1" / "collection" / "collection_receipt.json"),
        "historical_data_root": str(tmp),
        "n_pairs": 5000, "strata_per_cell": 500,
        "cf_actions": [0, 1, 2, 3, 4], "intervention_durations": [1, 3],
        "split_per_stratum": {"risk_train": 300, "risk_validation": 100, "development_screen": 100},
        "seed_entropy": 20260911, "collection_seed": 20260911, "seed_schedule_size": 12000,
        "max_attempts": 12000, "seed_low": 1, "seed_high_inclusive": 2147483646,
        "horizon": 20, "history_len": 10, "t_star_low": 10, "t_star_high": 15,
        "vehicles_count": 25, "observed_vehicles_count": 5, "duration": 100,
        "simulation_frequency": 15, "policy_frequency": 5, "determinism_checks": 20,
        "checkpoint_every_accepted": 50,
        "action_weights": {"idle_maintain": 0.45, "faster": 0.18, "slower": 0.18, "other": 0.095},
        "stage3r_run_dir": str(stage3r),
        "stage3b_binding": str(stage3r.parent / "stage3b" / "input_binding_receipt.json"),
        "scaler_path": str(stage3r / "scaler.json"),
        "response_predictor_seeds": [101, 202, 303, 404, 505],
        "expert_router_seeds": [1101, 1202, 1303, 1404, 1505],
        "primary_constant_w": 0.60, "batch_size": 32, "device": "cpu",
        "winsor_lower_q": 0.01, "winsor_upper_q": 0.99, "top_error_fraction": 0.20,
        "coverage_levels": [0.80, 0.90],
        "learning_curve_sizes": [500, 1000, 2000, 3000],
        "learning_curve_per_stratum": [50, 100, 200, 300],
        "ridge_alpha": 1.0,
        "histgb": {"learning_rate": 0.05, "max_iter": 10, "max_leaf_nodes": 8,
                   "min_samples_leaf": 5, "l2_regularization": 1.0, "early_stopping": False},
        "mlp": {"hidden_dim": 8, "dropout": 0.0, "learning_rate": 1e-3,
                "batch_size": 64, "max_epochs": 3, "patience": 2},
        "bootstrap_resamples_screen": 200, "bootstrap_entropy_screen": 2026091101,
        "confidence_level": 0.80, "gate_i_aurc_min": 0.02, "gate_i_c80_min": 0.02,
        "gate_min_positive_pairs": 4, "reversal_aurc_tolerance": 0.10,
    }


def _build_chain(tmp_path: Path):
    """Fabricate the whole synthetic v1->v3 chain + v4 run dir; return (config, paths)."""
    tmp = tmp_path
    stage3r = tmp / "stage3r"
    stage3r.mkdir(parents=True)
    model_lock = _make_frozen_stage3r(stage3r)

    data_output = tmp / "data" / "object_twinworld_stage4_merge_risk_dev.npz"
    data_output.parent.mkdir(parents=True)
    np.savez(data_output, episode_seed=np.arange(5))  # dummy; only its hash is bound
    data_npz_sha256 = sha256_file(data_output)

    seed_schedule = tmp / "v1" / "protocol" / "candidate_seed_schedule.npy"
    seed_schedule.parent.mkdir(parents=True)
    np.save(seed_schedule, (20260911 + np.arange(12000)).astype(np.int64))
    seed_schedule_sha256 = hashlib.sha256(
        np.load(seed_schedule).astype(np.int64).tobytes()).hexdigest()

    # v2 frozen inference + PRIMARY_PREDICTOR receipt.
    v2 = tmp / "v2"
    pred_npz = v2 / "inference" / "predictions.npz"
    pred_npz.parent.mkdir(parents=True)
    np.savez(pred_npz, dummy=np.zeros(3))
    predictions_sha256 = sha256_file(pred_npz)
    pred_receipt = v2 / "inference" / "prediction_receipt.json"
    pred_receipt_sha = _write_json(pred_receipt, {
        "predictions_sha256": predictions_sha256, "episode_count": s4p.N_PAIRS})
    pp_receipt = v2 / "risk" / "primary_predictor_receipt.json"
    pp_sha = _write_json(pp_receipt, {
        "primary_predictor": "END_TO_END_MIXTURE", "constant_w": 0.60,
        "predictions_sha256": predictions_sha256})

    # v3 risk fit: dataset + five models + fit receipt.
    v3 = tmp / "v3"
    v3_dataset = v3 / "risk" / "risk_dataset.npz"
    _make_dataset(v3_dataset)
    v3_dataset_sha = sha256_file(v3_dataset)
    fitted = _fit_five_models(v3 / "risk" / "models", v3_dataset)
    v3_fit_receipt = v3 / "risk" / "fit_receipt.json"
    v3_fit_sha = _write_json(v3_fit_receipt, {
        "plan_id": s4p.PLAN_ID, "status": "RISK_FIT_COMPLETE",
        "primary_predictor": "END_TO_END_MIXTURE",
        "risk_dataset_sha256": v3_dataset_sha,
        "predictions_sha256": predictions_sha256,
        "top_error_threshold": float(np.load(v3_dataset)["top_error_threshold"]),
        "fitted_models": fitted,
    })

    # Parent locks (self-consistent chain). code_sha256 uses REAL current code except
    # the runner, whose "before" value is the sentinel the patch receipt agrees on.
    current_code = {rel: sha256_file(CODE / rel) for rel in s4p.CODE_FILES}
    parent_code = dict(current_code)
    parent_code["run_stage4a.py"] = RUNNER_BEFORE

    v1_lock_path = tmp / "v1" / "protocol" / "protocol_lock.json"
    v1_lock_sha = _write_json(v1_lock_path, {
        "plan_id": s4p.PLAN_ID, "code_sha256": parent_code,
        "checkpoint_sha256": model_lock["checkpoint_sha256"],
        "scaler_sha256": model_lock["scaler_sha256"],
        "stage3b_binding_sha256": model_lock["stage3b_binding_sha256"],
        "seed_schedule_sha256": seed_schedule_sha256,
    })
    v2_lock_path = v2 / "protocol" / "recovery_lock.json"
    v2_lock_sha = _write_json(v2_lock_path, {
        "plan_id": s4p.PLAN_ID, "lock_type": "recovery_from_v1",
        "parent_run_dir": str(tmp / "v1"), "parent_protocol_lock_sha256": v1_lock_sha,
        "code_sha256": parent_code,
    })
    v3_lock_path = v3 / "protocol" / "recovery_lock.json"
    _write_json(v3_lock_path, {
        "plan_id": s4p.PLAN_ID, "lock_type": "recovery_from_v2",
        "chain": {
            "v1_protocol_lock_path": str(v1_lock_path),
            "v1_protocol_lock_sha256": v1_lock_sha,
            "v2_recovery_lock_path": str(v2_lock_path),
            "v2_recovery_lock_sha256": v2_lock_sha,
        },
        "code_sha256": parent_code,
        "reused_inference": {
            "predictions_npz_path": str(pred_npz),
            "predictions_npz_sha256": predictions_sha256,
            "prediction_receipt_path": str(pred_receipt),
            "prediction_receipt_sha256": pred_receipt_sha,
        },
        "primary_predictor": {
            "receipt_path": str(pp_receipt), "receipt_sha256": pp_sha,
            "primary_predictor": "END_TO_END_MIXTURE", "constant_w": 0.60,
        },
        "data_npz_sha256": data_npz_sha256,
        "seed_schedule_path": str(seed_schedule),
        "seed_schedule_sha256": seed_schedule_sha256,
    })

    # v4 run dir: byte-identical reuse copies + the runner-fix patch receipt.
    v4 = tmp / "v4"
    (v4 / "risk").mkdir(parents=True)
    (v4 / "risk" / "risk_dataset.npz").write_bytes(v3_dataset.read_bytes())
    (v4 / "risk" / "fit_receipt.json").write_bytes(v3_fit_receipt.read_bytes())
    patch = v4 / "protocol" / "runner_fix_patch.diff"
    patch.parent.mkdir(parents=True)
    patch.write_text("--- a\n+++ b\n-fix\n+fix\n", encoding="utf-8")
    _write_json(v4 / "protocol" / "runner_fix_receipt.json", {
        "runner_scientific_fix": {
            "before_sha256": RUNNER_BEFORE, "after_sha256": current_code["run_stage4a.py"],
            "patch_file": "runner_fix_patch.diff",
            "code_location": "run_stage4a.py::cmd_select_candidate",
        }
    })

    config = _base_config(tmp, v4, stage3r, data_output)
    config_path = v4 / "stage4a_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config, config_path, {
        "v3": v3, "v4": v4, "v3_dataset": v3_dataset,
        "v3_fit_receipt": v3_fit_receipt, "pred_npz": pred_npz,
        "models": fitted,
    }


def test_v4_recovery_rehearsal_full_command_path(tmp_path: Path, monkeypatch):
    """Drive lock -> verify -> select-candidate -> adjudicate on synthetic data."""
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "4")
    config, config_path, paths = _build_chain(tmp_path)
    v4 = paths["v4"]

    # 1. Build + write the v4 lock, then re-verify it through the recovery dispatcher.
    lock = s4rec.write_recovery_lock_v4(
        config_path,
        v3_run_dir=str(paths["v3"]),
        patch_receipt_path=v4 / "protocol" / "runner_fix_receipt.json",
        reason=config["run_dir"] and "rehearsal",
        code_location="run_stage4a.py::cmd_select_candidate",
        project_root=ROOT,
    )
    assert lock["lock_type"] == "recovery_from_v3"
    assert s4rec.is_recovery_run(v4)
    verified = s4rec.verify_recovery_lock(v4, project_root=ROOT)
    assert verified["reuse_verification"]["candidate_receipt_not_generated"] is True
    assert verified["reuse_verification"]["development_screen_untouched"] is True

    # 2. The real select-candidate resolves the reused v2 inference + v3 risk artifacts.
    selection = driver.cmd_select_candidate(config)
    assert selection["status"] == "DEV_CANDIDATE_SELECTED"
    assert selection["primary_predictor"] == "END_TO_END_MIXTURE"
    # predictions hash recorded must be the REUSED v2 predictions, not a local file.
    assert selection["predictions_sha256"] == sha256_file(paths["pred_npz"])
    selection_path = v4 / "risk" / "candidate_selection.json"
    assert selection_path.is_file()

    # 3. Write-once: a second select-candidate is refused.
    with pytest.raises(FileExistsError):
        driver.cmd_select_candidate(config)

    # 4. The real adjudicate reads the same reused artifacts and writes the report.
    report = driver.cmd_adjudicate_dev_screen(config)
    assert report["n_screen_episodes"] == 1000
    assert (v4 / "reports" / "stage4a_dev_report.json").is_file()
    assert (v4 / "reports" / "STAGE4A_DEV_DECISION.md").is_file()
    assert (v4 / "development_screen_used.lock").is_file()  # single-use lock burned
    assert report["verdict"]["conditions"]["qa_passed"] is True

    # 5. Single-use: a second adjudication is refused.
    with pytest.raises(FileExistsError):
        driver.cmd_adjudicate_dev_screen(config)


def test_v4_recovery_rehearsal_rejects_tamper(tmp_path: Path, monkeypatch):
    """Any change to a reused artifact's hash after locking must be rejected."""
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "4")
    config, config_path, paths = _build_chain(tmp_path)
    v4 = paths["v4"]
    s4rec.write_recovery_lock_v4(
        config_path,
        v3_run_dir=str(paths["v3"]),
        patch_receipt_path=v4 / "protocol" / "runner_fix_receipt.json",
        reason="rehearsal",
        code_location="run_stage4a.py::cmd_select_candidate",
        project_root=ROOT,
    )
    assert s4rec.verify_recovery_lock(v4, project_root=ROOT)

    # Tamper with one reused v3 fitted model -> verification must fail.
    some_model = next(iter(paths["models"].values()))
    with open(some_model, "ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="fitted model changed"):
        s4rec.verify_recovery_lock(v4, project_root=ROOT)


def test_v4_recovery_rehearsal_rejects_missing_reuse_copy(tmp_path: Path, monkeypatch):
    """The builder refuses to lock when the v4 byte-identical reuse copy is absent."""
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "4")
    config, config_path, paths = _build_chain(tmp_path)
    (paths["v4"] / "risk" / "risk_dataset.npz").unlink()  # remove the reuse copy
    with pytest.raises(FileNotFoundError, match="reuse copy missing"):
        s4rec.build_recovery_lock_v4(
            config,
            v3_run_dir=str(paths["v3"]),
            patch_receipt_path=paths["v4"] / "protocol" / "runner_fix_receipt.json",
            reason="rehearsal",
            code_location="run_stage4a.py::cmd_select_candidate",
            project_root=ROOT,
        )
