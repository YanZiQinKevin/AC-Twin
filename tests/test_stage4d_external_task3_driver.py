"""Focused integration test for the Stage 4D-D2 Task 3 DRIVER orchestration (plan 12.4).

The base-stack and Direct/gate internals are unit-tested elsewhere; this test proves the
distinct critical invariant that is NOT covered there — the driver's own wiring:
``fit-base`` sequences the create-once stack artifacts, ``assemble_role`` aligns Direct
features with truth labels and binds sha256 receipts, and the plan-7.5 train-support gate
fails CLOSED on insufficient data (run-all exits with the terminal code, never reaching the
screen). Synthetic only; heavy training constants are monkeypatched small. Not a science run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ac_twin import stage4c_objective as s4co
from ac_twin import stage4d_external_inference as INF
from ac_twin import stage4d_external_training as T
from ac_twin.stage4c_protocol import build_inference_mask

import drive_stage4d_external_task3 as DRV

ACTION_PLANS = [tuple(p) for p in s4co.ACTION_PLANS]
N_ACTIONS = len(ACTION_PLANS)


def _synthetic_role(dir_path: Path, groups: int, seed: int) -> Path:
    """Minimal schema-valid D2 role (groups.npz + groups_meta.json), cf-jittered per action."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    H, T_, N, F = INF.HISTORY_LEN, INF.FUTURE_LEN, INF.OBJECT_CAPACITY, INF.N_FEATURES
    history = np.zeros((groups, H, N, F), dtype=np.float32)
    factual = np.zeros((groups, T_, N, F), dtype=np.float32)
    cf = np.zeros((groups, N_ACTIONS, T_, N, F), dtype=np.float32)
    for g in range(groups):
        base = rng.normal(0.0, 5.0, size=(N, 2))
        vel = rng.normal(0.0, 1.0, size=(N, 2))
        history[g, :, :, 0] = 1.0
        history[g, :, 0, 1] = 1.0
        factual[g, :, :, 0] = 1.0
        factual[g, :, 0, 1] = 1.0
        for t in range(H):
            history[g, t, :, 2:4] = base + vel * (t - H)
            history[g, t, :, 4:6] = vel
        for t in range(T_):
            factual[g, t, :, 2:4] = base + vel * t
            factual[g, t, :, 4:6] = vel
        for a in range(N_ACTIONS):
            cf[g, a, :, :, 0] = 1.0
            cf[g, a, :, 0, 1] = 1.0
            jitter = rng.normal(0.0, 0.5, size=(T_, N, 2))
            for t in range(T_):
                cf[g, a, t, :, 2:4] = base + vel * t + jitter[t] * (1 + a * 0.1)
                cf[g, a, t, :, 4:6] = vel
    inference_mask = build_inference_mask(history, factual)
    t_star = rng.integers(10, 16, size=(groups,)).astype(np.int64)
    np.savez(
        dir_path / "groups.npz",
        object_history=history, object_factual_future=factual,
        object_counterfactual_futures=cf,
        factual_actions=rng.integers(0, 5, size=(groups, T_)).astype(np.int64),
        inference_mask=inference_mask, t_star=t_star,
        cf_actions=np.asarray([a for a, _ in ACTION_PLANS], dtype=np.int64),
        cf_durations=np.asarray([d for _, d in ACTION_PLANS], dtype=np.int64),
        n_actions=np.asarray(N_ACTIONS, dtype=np.int64),
        object_features_json=np.asarray(json.dumps(["presence", "is_ego", "s", "d",
                                                    "v_s", "v_d", "cos", "sin"])),
        action_plans_json=np.asarray(json.dumps([list(p) for p in ACTION_PLANS])),
    )
    meta = [{"scenario_id": f"sd_drv_{seed}_{g}", "t_star": int(t_star[g]),
             "eval_track_ids": [1, 2, 3, 4], "ego_lane_id": "lane0"} for g in range(groups)]
    (dir_path / "groups_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return dir_path


@pytest.fixture()
def driver_env(tmp_path: Path, monkeypatch):
    """Redirect every driver path constant under tmp_path and shrink the training constants."""
    roles = tmp_path / "roles"
    run_dir = tmp_path / "run"
    dev = run_dir / "development"
    base = dev / "base_models"
    monkeypatch.setattr(DRV, "ROLES_DIR", roles)
    monkeypatch.setattr(DRV, "RUN_DIR", run_dir)
    monkeypatch.setattr(DRV, "DEV_DIR", dev)
    monkeypatch.setattr(DRV, "BASE_MODEL_DIR", base)
    groups = 6
    monkeypatch.setattr(DRV, "ROLE_QUOTA", {"train": groups, "validation": groups})
    monkeypatch.setattr(INF, "BASE_MAX_EPOCHS", 2)
    monkeypatch.setattr(INF, "BASE_EARLY_STOP_PATIENCE", 2)
    monkeypatch.setattr(INF, "BASE_BATCH_SIZE", 16)
    monkeypatch.setattr(INF, "RESPONSE_PREDICTOR_SEEDS", (101,))
    monkeypatch.setattr(INF, "PREDICTOR_PAIR_SEEDS", (1101,))
    monkeypatch.setattr(INF, "N_SEEDS", 1)
    _synthetic_role(roles / "train", groups, seed=11)
    _synthetic_role(roles / "validation", groups, seed=12)
    return {"roles": roles, "run_dir": run_dir, "dev": dev, "base": base, "groups": groups}


def test_fit_base_freezes_stack_and_create_once(driver_env):
    base, dev = driver_env["base"], driver_env["dev"]
    stack = DRV.cmd_fit_base(device="cpu", batch_size=16)
    # create-once artifacts written.
    assert (base / "predictor_stack.json").is_file()
    assert (base / "scaler.json").is_file()
    assert (base / "calibration.json").is_file()
    assert (dev / "train_predictions.npz").is_file()
    assert stack.risk_paths, "risk heads fitted and recorded"
    # create-once: a second call reloads the manifest instead of retraining.
    again = DRV.cmd_fit_base(device="cpu", batch_size=16)
    assert again == stack


def test_assemble_role_alignment_and_support_gate_fail_closed(driver_env):
    dev, groups = driver_env["dev"], driver_env["groups"]
    stack = DRV.cmd_fit_base(device="cpu", batch_size=16)

    # assemble_role aligns Direct features with truth labels and binds receipts.
    assembled = DRV.assemble_role(stack, "train", device="cpu", batch_size=16)
    feats = assembled["features"]
    G = assembled["groups"]
    assert G == groups and assembled["n_actions"] == N_ACTIONS
    assert feats["tokens"].shape[0] == G and feats["global_features"].shape[0] == G
    assert assembled["regret"].shape == (G,)
    assert assembled["y_reg"].shape == (G,) and assembled["y_high"].shape == (G,)
    assert assembled["oracle_action"].shape == (G,)
    assert assembled["group_seed"].shape == (G,)
    # y_high is the binarized regret at the frozen threshold (supervision consistency).
    assert np.array_equal(assembled["y_high"],
                          (assembled["regret"] > T.HIGH_REGRET_THRESHOLD).astype(np.int64))
    # receipt bindings recorded.
    for key in ("data_sha256", "predictions_sha256", "scores_sha256"):
        assert isinstance(assembled[key], str) and assembled[key]
    assert (dev / "train_scores.npz").is_file()

    # plan-7.5 gate fails CLOSED on 6 groups (needs >=750 high / >=1250 nonzero).
    report = DRV.cmd_check_train_support(stack, device="cpu", batch_size=16)
    assert (dev / "train_support_report.json").is_file()
    assert report["status"] == T.STATUS_TRAIN_SUPPORT_INSUFFICIENT
    assert report["passed"] is False
    assert report["values"]["groups"] == groups
    assert report["values"]["high_regret_groups"] <= groups
    assert "valid_feature_groups" in report and "inference_code_sha256" in report


def test_run_all_terminates_at_support_gate_without_screen(driver_env):
    # run-all stops at the plan-7.5 terminal state on insufficient support; never trains Direct.
    assert DRV.run_all(device="cpu", batch_size=16) == 3
    assert not (driver_env["dev"] / "final_selection.json").exists()
    assert not (driver_env["dev"] / "models").exists(), "no Direct models on support failure"
