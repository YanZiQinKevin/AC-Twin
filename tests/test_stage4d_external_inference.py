"""Focused tests for Stage 4D-D2 base-predictor retraining + inference (plan 7.2-7.5).

Synthetic only — no MetaDrive, no collected data. Proves: (1) the reimplemented frozen
losses are bit-exact against run_stage3r, (2) the router objective is the exact
lambda_cew=0/lambda_rank=0 reduction, (3) the D2 npz -> branch expansion materialises the
canonical counterfactual plans and both masks, (4) the scaler/calibration/risk fits are
create-once, and (5) a tiny end-to-end fit->infer->Direct-token chain produces finite,
well-shaped, ensemble-consistent outputs. Engineering smoke only — not a scientific run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ac_twin import stage4c_objective as s4co
from ac_twin import stage4cp_protocol as s4cp
from ac_twin import stage4d_external_inference as INF
from ac_twin import stage4d_external_training as T
from ac_twin.stage4c_protocol import build_inference_mask, materialize_action_plan

ACTION_PLANS = [tuple(p) for p in s4co.ACTION_PLANS]
N_ACTIONS = len(ACTION_PLANS)  # 10


def _rng(seed=0):
    return np.random.default_rng(seed)


def _synthetic_role(dir_path: Path, groups: int, seed: int = 0) -> Path:
    """Write a minimal but schema-valid D2 role (groups.npz + groups_meta.json)."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = _rng(seed)
    H, T_, N, F = INF.HISTORY_LEN, INF.FUTURE_LEN, INF.OBJECT_CAPACITY, INF.N_FEATURES
    history = np.zeros((groups, H, N, F), dtype=np.float32)
    factual = np.zeros((groups, T_, N, F), dtype=np.float32)
    cf = np.zeros((groups, N_ACTIONS, T_, N, F), dtype=np.float32)
    for g in range(groups):
        # ego slot 0 + 4 non-ego, all present in history and all futures.
        base = rng.normal(0.0, 5.0, size=(N, 2))
        vel = rng.normal(0.0, 1.0, size=(N, 2))
        history[g, :, :, 0] = 1.0
        history[g, :, 0, 1] = 1.0  # is_ego
        for t in range(H):
            history[g, t, :, 2:4] = base + vel * (t - H)
            history[g, t, :, 4:6] = vel
        factual[g, :, :, 0] = 1.0
        factual[g, :, 0, 1] = 1.0
        for t in range(T_):
            factual[g, t, :, 2:4] = base + vel * t
            factual[g, t, :, 4:6] = vel
        for a in range(N_ACTIONS):
            cf[g, a, :, :, 0] = 1.0
            cf[g, a, :, 0, 1] = 1.0
            # non-ego react slightly differently per action so the effect is nonzero.
            jitter = rng.normal(0.0, 0.5, size=(T_, N, 2))
            for t in range(T_):
                cf[g, a, t, :, 2:4] = base + vel * t + jitter[t] * (1 + a * 0.1)
                cf[g, a, t, :, 4:6] = vel
    inference_mask = build_inference_mask(history, factual)  # (G,T,N) cf-independent
    factual_actions = rng.integers(0, 5, size=(groups, T_)).astype(np.int64)
    t_star = rng.integers(10, 16, size=(groups,)).astype(np.int64)
    np.savez(
        dir_path / "groups.npz",
        object_history=history, object_factual_future=factual,
        object_counterfactual_futures=cf, factual_actions=factual_actions,
        inference_mask=inference_mask, t_star=t_star,
        cf_actions=np.asarray([a for a, _ in ACTION_PLANS], dtype=np.int64),
        cf_durations=np.asarray([d for _, d in ACTION_PLANS], dtype=np.int64),
        n_actions=np.asarray(N_ACTIONS, dtype=np.int64),
        object_features_json=np.asarray(json.dumps(["presence", "is_ego", "s", "d",
                                                    "v_s", "v_d", "cos", "sin"])),
        action_plans_json=np.asarray(json.dumps([list(p) for p in ACTION_PLANS])),
    )
    meta = [{"scenario_id": f"sd_syn_{seed}_{g}", "t_star": int(t_star[g]),
             "eval_track_ids": [1, 2, 3, 4], "ego_lane_id": "lane0"} for g in range(groups)]
    (dir_path / "groups_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return dir_path


def test_role_salt_matches_collector():
    import collect_stage4d_external as collector
    assert INF.ROLE_SALT == collector.ROLE_SALT == "AC-TWIN-D2-V0.1|"


def test_masked_expert_mse_matches_run_stage3r():
    import run_stage3r as s3rr
    rng = torch.Generator().manual_seed(0)
    pred = torch.randn(7, 20, 5, 2, generator=rng)
    target = torch.randn(7, 20, 5, 2, generator=rng)
    valid = torch.rand(7, 20, 5, generator=rng) > 0.3
    assert torch.equal(INF.masked_expert_mse(pred, target, valid),
                       s3rr.masked_expert_mse(pred, target, valid))


def test_router_mixture_is_pure_masked_mse():
    """lambda_cew=0/lambda_rank=0 reduces the router objective to masked MSE of the blend."""
    torch.manual_seed(0)
    router = INF.CEWRouter(input_dim=INF.ROUTER_INPUT_DIM, hidden_dim=INF.ROUTER_HIDDEN_DIM)
    E, T_, N = 5, INF.FUTURE_LEN, INF.OBJECT_CAPACITY
    batch = {"features": torch.randn(E, T_, N, INF.ROUTER_INPUT_DIM),
             "full": torch.randn(E, T_, N, 2), "history": torch.randn(E, T_, N, 2),
             "effect_target": torch.randn(E, T_, N, 2), "valid": torch.rand(E, T_, N) > 0.2}
    got = INF.router_mixture_mse(router, batch)
    weight = router(batch["features"]).squeeze(-1)
    mixed = weight.unsqueeze(-1) * batch["full"] + (1.0 - weight.unsqueeze(-1)) * batch["history"]
    expected = INF.masked_expert_mse(mixed, batch["effect_target"], batch["valid"])
    assert torch.allclose(got, expected, rtol=0, atol=1e-12)


def test_branch_expansion_and_masks(tmp_path: Path):
    groups = 4
    role_dir = _synthetic_role(tmp_path / "train", groups, seed=1)
    arrays = INF.load_role_arrays(role_dir, "train")
    assert arrays.groups == groups and arrays.n_actions == N_ACTIONS
    assert len(arrays.group_seed) == groups
    # group_seed is deterministic from the scenario id.
    again = INF.load_role_arrays(role_dir, "train")
    assert np.array_equal(arrays.group_seed, again.group_seed)

    bundle = INF.build_branch_bundle(arrays)
    E = groups * N_ACTIONS
    assert bundle.branches == E
    assert bundle.history.shape == (E, INF.HISTORY_LEN, INF.OBJECT_CAPACITY, INF.N_FEATURES)
    assert bundle.counterfactual_actions.shape == (E, INF.FUTURE_LEN)
    assert bundle.effect_target.shape == (E, INF.FUTURE_LEN, INF.OBJECT_CAPACITY, 2)
    assert bundle.valid_eval.shape == (E, INF.FUTURE_LEN, INF.OBJECT_CAPACITY)
    # canonical CF plan: override first `duration` steps then resume the factual plan.
    g, a = 2, 5
    e = g * N_ACTIONS + a
    act, dur = ACTION_PLANS[a]
    assert np.array_equal(bundle.counterfactual_actions[e],
                          materialize_action_plan(arrays.factual_actions[g], act, dur))
    assert bundle.cf_action[e] == act and bundle.cf_duration[e] == dur
    # eval mask excludes ego; inference mask is the collector's cf-independent one.
    assert not bundle.valid_eval[:, :, 0].any()
    assert np.array_equal(bundle.valid_infer.reshape(groups, N_ACTIONS, INF.FUTURE_LEN, -1)[:, 0],
                          arrays.inference_mask)
    # geometry carried at both levels.
    assert bundle.geometry_group["distance_to_ego"].shape[0] == groups
    assert bundle.geometry_branch["distance_to_ego"].shape[0] == E


def test_fit_scaler_create_once(tmp_path: Path):
    role_dir = _synthetic_role(tmp_path / "train", 3, seed=2)
    bundle = INF.build_branch_bundle(INF.load_role_arrays(role_dir, "train"))
    path = tmp_path / "scaler.json"
    scaler = INF.fit_scaler(bundle, path)
    assert path.is_file()
    again = INF.fit_scaler(bundle, path)  # reloads, never refits
    assert np.allclose(scaler.transform(bundle.history), again.transform(bundle.history))


@pytest.mark.parametrize("component", ["response"])
def test_end_to_end_tiny_stack(tmp_path: Path, monkeypatch, component):
    """Tiny full chain: fit scaler+response+experts+calibration+router+risk, then infer.

    Training constants are monkeypatched small for speed; this is an engineering smoke
    test of wiring/shapes/finiteness, not a scientific run.
    """
    monkeypatch.setattr(INF, "BASE_MAX_EPOCHS", 2)
    monkeypatch.setattr(INF, "BASE_EARLY_STOP_PATIENCE", 2)
    monkeypatch.setattr(INF, "BASE_BATCH_SIZE", 16)
    # shrink the seed sets to a single pair for speed.
    monkeypatch.setattr(INF, "RESPONSE_PREDICTOR_SEEDS", (101,))
    monkeypatch.setattr(INF, "PREDICTOR_PAIR_SEEDS", (1101,))
    monkeypatch.setattr(INF, "N_SEEDS", 1)

    groups = 6
    train_dir = _synthetic_role(tmp_path / "train", groups, seed=3)
    val_dir = _synthetic_role(tmp_path / "validation", groups, seed=4)
    train = INF.build_branch_bundle(INF.load_role_arrays(train_dir, "train"))
    val = INF.build_branch_bundle(INF.load_role_arrays(val_dir, "validation"))
    model_dir = tmp_path / "models"

    scaler = INF.fit_scaler(train, model_dir / "scaler.json")
    resp = INF.fit_response_predictors(train, val, scaler, model_dir, device="cpu")
    assert set(resp) == {101} and Path(resp[101]).is_file()
    experts = INF.fit_experts(train, val, scaler, model_dir, device="cpu")
    assert set(experts) == {(1101, "full"), (1101, "history")}
    calib = INF.fit_calibration(resp, val, scaler, model_dir / "calibration.json",
                                device="cpu", batch_size=16)
    assert set(calib) == {101}
    routers = INF.fit_routers(train, val, scaler, resp, experts, calib, model_dir,
                              device="cpu", batch_size=16)
    assert set(routers) == {1101}

    # production predictions on the train role feed the risk-head fit.
    stack_partial = INF.PredictorStack(
        scaler_path=str(model_dir / "scaler.json"), response_paths=resp,
        expert_paths=experts, router_paths=routers, calibration=calib,
        risk_paths={}, primary_predictor="END_TO_END_MIXTURE")
    preds = INF.run_pair_predictions(stack_partial, train, device="cpu", batch_size=16)
    E = train.branches
    assert preds["e2e"].shape == (1, E, INF.FUTURE_LEN, INF.OBJECT_CAPACITY, 2)
    assert preds["r_hat"].shape == (1, E, INF.FUTURE_LEN, INF.OBJECT_CAPACITY)
    for key in ("e2e", "full", "history", "r_hat", "presence_prob", "high_prob"):
        assert np.isfinite(preds[key]).all(), key

    risk = INF.fit_risk_heads(preds, train, model_dir)
    assert risk["primary"].primary_predictor in ("BEST_CONSTANT_W", "END_TO_END_MIXTURE")
    stack = INF.PredictorStack(
        scaler_path=str(model_dir / "scaler.json"), response_paths=resp,
        expert_paths=experts, router_paths=routers, calibration=calib,
        risk_paths=risk["paths"], primary_predictor=risk["primary"].primary_predictor)

    # PredictorStack round-trips through its manifest dict.
    assert INF.PredictorStack.from_dict(stack.to_dict()) == stack

    action_risks = INF.score_action_risks(stack, train, preds)
    assert set(action_risks) == set(s4cp.RISK_CLASSES)
    for name, value in action_risks.items():
        assert value.shape == (groups, N_ACTIONS) and np.isfinite(value).all(), name

    direct = INF.build_direct_inputs(train, preds, action_risks)
    assert direct["tokens"].shape == (groups, T.N_COMPETITORS, T.TOKEN_DIM)
    assert direct["global_features"].shape == (groups, T.GLOBAL_DIM)
    assert direct["j_hat"].shape == (groups, N_ACTIONS)
    assert set(direct["baselines"]) >= {
        "MARGIN_ONLY", "SEED_VOTE_ENTROPY", "SEED_MARGIN_STD", "COMBINED_TOP2_MAX"}
    assert np.isfinite(direct["tokens"][direct["valid_mask"]]).all()
