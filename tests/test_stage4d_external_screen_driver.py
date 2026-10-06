"""Focused test for the Stage 4D-D2 development-screen DRIVER seams (plan 8.1).

Proves the driver's two distinct critical invariants on synthetic data, without the torch stack:
(1) the scoring bundle is built from ``inference_inputs.npz`` alone and still builds with the
sealed truth physically deleted (the scoring process has no truth path); (2) the >=32-group
determinism recheck fails closed when two repeated scorings differ. The heavy frozen scoring
(``score_screen``) is monkeypatched; bundle construction runs for real. Not a science run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ac_twin import stage4c_objective as s4co
from ac_twin import stage4d_external_statistics as STATS

import drive_stage4d_external_screen as DRV
import stage4d_split_screen_truth as SPLIT

ACTION_PLANS = [tuple(p) for p in s4co.ACTION_PLANS]
N_ACTIONS = len(ACTION_PLANS)
G = 8


def _write_role(role_dir: Path, groups: int, seed: int) -> None:
    role_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    H, T_, N, F = 10, 20, 5, 8
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
            for t in range(T_):
                cf[g, a, t, :, 2:4] = base + vel * t
                cf[g, a, t, :, 4:6] = vel
    np.savez(
        role_dir / "groups.npz",
        object_history=history, object_factual_future=factual,
        object_counterfactual_futures=cf,
        factual_actions=rng.integers(0, 5, size=(groups, T_)).astype(np.int64),
        inference_mask=np.ones((groups, T_, N), dtype=bool),
        t_star=rng.integers(10, 16, size=(groups,)).astype(np.int64),
        cf_actions=np.asarray([a for a, _ in ACTION_PLANS], dtype=np.int64),
        cf_durations=np.asarray([d for _, d in ACTION_PLANS], dtype=np.int64),
        n_actions=np.asarray(N_ACTIONS, dtype=np.int64),
        object_features_json=np.asarray(json.dumps(
            ["presence", "is_ego", "s", "d", "v_s", "v_d", "cos", "sin"])),
        action_plans_json=np.asarray(json.dumps([list(p) for p in ACTION_PLANS])),
    )
    meta = [{"scenario_id": f"sd_drv_{seed}_{g}", "t_star": 12,
             "eval_track_ids": [1, 2, 3, 4], "ego_lane_id": "lane0"} for g in range(groups)]
    (role_dir / "groups_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (role_dir / "collection_receipt.json").write_text(json.dumps(
        {"role": "development_screen", "accepted": groups, "status": "complete"}),
        encoding="utf-8")


def test_scoring_bundle_builds_with_truth_deleted(tmp_path):
    role_dir = tmp_path / "roles" / "development_screen"
    split_dir = tmp_path / "split"
    _write_role(role_dir, G, seed=41)
    out = SPLIT.split_screen_truth(role_dir, split_dir,
                                   role_salt="AC-TWIN-D2-V0.1|", expect_groups=G)
    # physically remove the sealed truth: the scoring bundle must still build from inputs alone.
    out["sealed_truth"].unlink()
    bundle = DRV.build_scoring_bundle(out["inference_inputs"])
    assert bundle.groups == G and bundle.n_actions == N_ACTIONS
    assert bundle.branches == G * N_ACTIONS
    assert bundle.history.shape == (G * N_ACTIONS, 10, 5, 8)
    assert bundle.factual_suffix.shape == (G * N_ACTIONS, 10, 5, 8)
    assert bundle.inference_mask.shape == (G, 20, 5)
    assert bundle.group_seed.shape == (G,)
    # subset construction for the determinism recheck.
    subset = DRV.build_scoring_bundle(out["inference_inputs"], group_index=np.arange(4))
    assert subset.groups == 4 and subset.branches == 4 * N_ACTIONS


def _fake_scores(groups: int, offset: float) -> dict:
    rng = np.random.default_rng(7)
    gseed = np.arange(1, groups + 1, dtype=np.int64)
    ensemble = rng.uniform(0.0, 1.0, groups) + offset
    return {
        "group_seed": gseed,
        "t_star": np.full(groups, 12, dtype=np.int64),
        "interaction_min_distance": rng.uniform(0.0, 20.0, groups),
        "j_hat": rng.uniform(0.0, 1.0, (groups, N_ACTIONS)) + offset,
        "ensemble_risk": ensemble,
        "risk_by_seed": np.stack([ensemble] * 5),
        "mu_only_ensemble": ensemble * 0.5,
        "class_only_ensemble": ensemble * 0.5,
        "valid_mask": np.ones(groups, dtype=bool),
        "baselines": {name: rng.uniform(0.0, 1.0, groups)
                      for name in STATS.ALL_BASELINE_NAMES},
    }


def test_determinism_recheck_detects_mismatch(tmp_path, monkeypatch):
    role_dir = tmp_path / "roles" / "development_screen"
    split_dir = tmp_path / "split"
    _write_role(role_dir, G, seed=42)
    out = SPLIT.split_screen_truth(role_dir, split_dir,
                                   role_salt="AC-TWIN-D2-V0.1|", expect_groups=G)

    calls = {"n": 0}

    def flaky_score(stack, inventory, inputs_path, *, device, batch_size, group_index=None):
        calls["n"] += 1
        groups = G if group_index is None else int(len(group_index))
        return _fake_scores(groups, offset=float(calls["n"]))  # differs between the two passes

    monkeypatch.setattr(DRV, "score_screen", flaky_score)
    with pytest.raises(RuntimeError):
        DRV.determinism_recheck(None, None, out["inference_inputs"],
                                device="cpu", batch_size=4, n_groups=4)

    # a reproducible scorer passes the recheck.
    monkeypatch.setattr(DRV, "score_screen",
                        lambda *a, **k: _fake_scores(
                            G if k.get("group_index") is None else int(len(k["group_index"])), 0.0))
    receipt = DRV.determinism_recheck(None, None, out["inference_inputs"],
                                      device="cpu", batch_size=4, n_groups=4)
    assert receipt["bitwise_identical"] is True and receipt["n_groups"] == 4
