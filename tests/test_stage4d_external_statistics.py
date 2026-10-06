"""Synthetic-data tests for the Stage 4D-D2 development-screen statistics + gates (plan 8).

Proves each distinct critical invariant once: the paired bootstrap runs at the frozen 80% level
with coupled resamples and reproducible draws; Gate TASK-X-DEV / RISK-X-DEV pass on a qualifying
synthetic screen; the fail-closed cases (degenerate baseline denominator -> INVALID_ANALYSIS, a
non-positive 80% CI lower bound fails its condition, the >=4/5 seed-both-positive rule) are each
enforced; and the one-shot sequence (split -> freeze -> lock -> adjudicate) refuses a second
adjudication. The torch scoring is NOT exercised here (that is the driver's seam); the score
arrays are injected synthetically. Not a science run.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from ac_twin import stage4d_external_statistics as STATS
from ac_twin import stage4d_external_training as T
from ac_twin import stage4c_objective as s4co

import stage4d_split_screen_truth as SPLIT

ACTION_PLANS = [tuple(p) for p in s4co.ACTION_PLANS]
N_ACTIONS = len(ACTION_PLANS)
G = 300  # synthetic screen size (DEV_SCREEN_GROUPS monkeypatched to this)


@pytest.fixture(autouse=True)
def _screen_groups(monkeypatch):
    monkeypatch.setattr(STATS, "DEV_SCREEN_GROUPS", G)


def _passing_quantities(seed: int = 0) -> dict:
    """A synthetic screen quantity mapping that passes both gates (strong, robust signals)."""
    rng = np.random.default_rng(seed)
    high = rng.uniform(0.0, 1.0, G) < 0.35
    regret = np.where(high, rng.uniform(0.05, 0.5, G), rng.uniform(0.0, 0.005, G))
    ensemble = regret + rng.normal(0.0, 1e-4, G)          # near-perfect ranking
    baselines = {
        "MARGIN_ONLY": rng.uniform(0.0, 1.0, G),          # near-random (weakest) = frozen primary
        "SEED_VOTE_ENTROPY": rng.uniform(0.0, 1.0, G),
        "SEED_MARGIN_STD": rng.uniform(0.0, 1.0, G),
        "COMBINED_TOP2_MAX": rng.uniform(0.0, 1.0, G),
    }
    risk_by_seed = np.stack([regret + rng.normal(0.0, 1e-4, G) for _ in range(5)])
    j_true = rng.uniform(0.0, 1.0, (G, N_ACTIONS))
    j_hat = j_true + rng.normal(0.0, 1e-3, (G, N_ACTIONS))  # ~identical -> spearman ~ 1
    return {
        "group_seed": np.arange(1, G + 1, dtype=np.int64),
        "regret": regret,
        "y_high": (regret > STATS.HIGH_REGRET_THRESHOLD).astype(np.int64),
        "random_regret": regret + rng.uniform(0.05, 0.10, G),
        "idle_regret": regret + rng.uniform(0.01, 0.03, G),
        "ensemble_risk": ensemble,
        "risk_by_seed": risk_by_seed,
        "J_hat": j_hat,
        "J_true": j_true,
        "a_star": np.argmin(j_true, axis=1).astype(np.int64),
        "tstar_layer": (np.arange(G) % 2).astype(np.int64),       # balanced strata
        "interaction_layer": (np.arange(G) % 2).astype(np.int64),
        "t_star": np.where(np.arange(G) % 2 == 0, 10, 14).astype(np.int64),
        "interaction_min_distance": np.where(np.arange(G) % 2 == 0, 5.0, 15.0),
        "reactive": (np.arange(G) % 3 == 0),                       # 33% reactive
        "baseline_risks": baselines,
        "strongest_baseline": "MARGIN_ONLY",
        "interaction_threshold": 10.0,
        "n_groups": G,
        "qa": {key: True for key in STATS.QA_KEYS},
        "projection_nondegenerate": True,
        "eval_mask_nondegenerate": True,
    }


def _bootstrap(quantities, n_resamples=200, seed=1234):
    return STATS.paired_group_bootstrap(quantities, seed=seed, n_resamples=n_resamples)


# --- bootstrap sanity ---------------------------------------------------------------------------------


def test_bootstrap_level_direction_and_reproducibility():
    quantities = _passing_quantities()
    result = _bootstrap(quantities, n_resamples=200, seed=1234)
    assert result.confidence == 0.80
    assert result.point.shape == (STATS.N_QUANTITIES,)
    assert result.ci_lower.shape == result.ci_upper.shape == (STATS.N_QUANTITIES,)
    assert result.draws.shape == (200, STATS.N_QUANTITIES)
    finite = np.isfinite(result.ci_lower) & np.isfinite(result.ci_upper)
    assert (result.ci_lower[finite] <= result.ci_upper[finite]).all()
    # strong, well-separated signals have a strictly positive 80% CI lower bound (direction).
    for slot in (STATS.Q_I_AURC, STATS.Q_I_REGRET80, STATS.Q_PR, STATS.Q_SPEARMAN_BRANCH):
        assert result.ci_lower[slot] > 0.0
    # coupled resamples: identical seed reproduces identical draws; a different seed does not.
    again = _bootstrap(quantities, n_resamples=200, seed=1234)
    assert np.array_equal(result.draws, again.draws)
    other = _bootstrap(quantities, n_resamples=200, seed=999)
    assert not np.array_equal(result.draws, other.draws)


# --- gate pass ----------------------------------------------------------------------------------------


def test_gates_pass_on_qualifying_screen():
    quantities = _passing_quantities()
    decision = STATS.evaluate_screen_gates(quantities, _bootstrap(quantities))
    assert decision.status == STATS.STATUS_DEV_READY
    assert decision.passed is True
    assert all(decision.task["conditions"].values())
    assert all(decision.risk["conditions"].values())
    assert decision.risk["seeds_both_directions_positive"] == 5
    # three standard baselines reported with point + 80% CI (plan 8.3 note).
    for name in STATS.STANDARD_BASELINES:
        curve = decision.baselines["curves"][name]
        assert np.isfinite(curve["aurc_regret"])
        assert len(curve["aurc_regret_ci80"]) == 2


# --- fail-closed cases ---------------------------------------------------------------------------------


def test_degenerate_baseline_denominator_is_invalid_analysis():
    quantities = _passing_quantities()
    quantities["regret"] = np.zeros(G)                      # all regret 0 -> baseline AURC = 0
    quantities["random_regret"] = np.zeros(G)
    quantities["idle_regret"] = np.zeros(G)
    quantities["y_high"] = np.zeros(G, dtype=np.int64)
    decision = STATS.evaluate_screen_gates(quantities, _bootstrap(quantities))
    assert decision.status == STATS.STATUS_INVALID_ANALYSIS
    assert decision.passed is False
    assert decision.task["conditions"]["c8_denominators_projection_mask_valid"] is False
    assert decision.risk["conditions"]["r5_baseline_denominators_valid"] is False


def _passing_bootstrap_result(quantities):
    result = _bootstrap(quantities, n_resamples=200, seed=7)
    return result


def test_positive_point_with_nonpositive_ci_lower_fails_closed():
    quantities = _passing_quantities()
    base = _passing_bootstrap_result(quantities)
    ci_lower = base.ci_lower.copy()
    ci_lower[STATS.Q_I_AURC] = -0.001                        # point positive, CI lower <= 0
    assert base.point[STATS.Q_I_AURC] >= STATS.I_AURC_MIN
    injected = dataclasses.replace(base, ci_lower=ci_lower)
    decision = STATS.evaluate_screen_gates(quantities, injected)
    assert decision.risk["conditions"]["r1_i_aurc_regret"] is False
    assert decision.status == STATS.STATUS_DEV_NOT_READY
    assert decision.passed is False


def test_seed_both_positive_rule_enforced():
    quantities = _passing_quantities()
    base = _passing_bootstrap_result(quantities)
    point = base.point.copy()
    # only 3 of 5 seeds keep both primary improvements positive.
    point[STATS.SEED_I_AURC_INDEX[3]] = -0.01
    point[STATS.SEED_I_AURC_INDEX[4]] = -0.01
    injected = dataclasses.replace(base, point=point)
    decision = STATS.evaluate_screen_gates(quantities, injected)
    assert decision.risk["seeds_both_directions_positive"] == 3
    assert decision.risk["conditions"]["r4_seed_consistency_both_positive"] is False
    assert decision.status == STATS.STATUS_DEV_NOT_READY


def test_integrity_failure_is_invalid_not_method_failure():
    quantities = _passing_quantities()
    quantities["qa"] = {key: True for key in STATS.QA_KEYS}
    quantities["qa"]["hashes_match"] = False
    decision = STATS.evaluate_screen_gates(quantities, _bootstrap(quantities))
    assert decision.status == STATS.STATUS_INVALID_ANALYSIS


# --- split + one-shot sequencing -----------------------------------------------------------------------


def _write_synthetic_role(role_dir: Path, groups: int, seed: int) -> None:
    """Minimal schema-valid D2 screen role: groups.npz + groups_meta.json + collection_receipt."""
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
            jitter = rng.normal(0.0, 0.5, size=(T_, N, 2))
            for t in range(T_):
                cf[g, a, t, :, 2:4] = base + vel * t + jitter[t] * (1 + a * 0.1)
                cf[g, a, t, :, 4:6] = vel
    inference_mask = np.ones((groups, T_, N), dtype=bool)
    t_star = rng.integers(10, 16, size=(groups,)).astype(np.int64)
    np.savez(
        role_dir / "groups.npz",
        object_history=history, object_factual_future=factual,
        object_counterfactual_futures=cf,
        factual_actions=rng.integers(0, 5, size=(groups, T_)).astype(np.int64),
        inference_mask=inference_mask, t_star=t_star,
        cf_actions=np.asarray([a for a, _ in ACTION_PLANS], dtype=np.int64),
        cf_durations=np.asarray([d for _, d in ACTION_PLANS], dtype=np.int64),
        n_actions=np.asarray(N_ACTIONS, dtype=np.int64),
        object_features_json=np.asarray(json.dumps(
            ["presence", "is_ego", "s", "d", "v_s", "v_d", "cos", "sin"])),
        action_plans_json=np.asarray(json.dumps([list(p) for p in ACTION_PLANS])),
    )
    meta = [{"scenario_id": f"sd_scr_{seed}_{g}", "t_star": int(t_star[g]),
             "eval_track_ids": [1, 2, 3, 4], "ego_lane_id": "lane0"} for g in range(groups)]
    (role_dir / "groups_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (role_dir / "collection_receipt.json").write_text(json.dumps({
        "role": "development_screen", "quota": groups, "accepted": groups, "status": "complete",
    }), encoding="utf-8")


def _screen_config(tmp_path: Path, groups: int) -> dict:
    run_dir = tmp_path / "run"
    dev = run_dir / "development"
    (dev / "models").mkdir(parents=True, exist_ok=True)
    ckpt = {str(s): {"path": f"m{s}.pt", "sha256": f"h{s}"} for s in T.MODEL_SEEDS}
    (dev / "final_selection.json").write_text(json.dumps({
        "plan_id": T.PLAN_ID, "status": T.STATUS_VALIDATION_QUALIFIED,
        "strongest_standard_baseline": "MARGIN_ONLY", "interaction_threshold": 10.0,
        "interaction_threshold_source": "train preintervention_min_distance median (train-only)",
        "primary_score_definition": T.PRIMARY_SCORE_DEFINITION,
        "checkpoints": ckpt, "model_seeds": [int(s) for s in T.MODEL_SEEDS],
    }), encoding="utf-8")
    (dev / "models" / T.FIT_RECEIPT_NAME).write_text(json.dumps({
        "plan_id": T.PLAN_ID, "arch": T.ARCH_DIRECT, "seeds": [int(s) for s in T.MODEL_SEEDS],
        "param_count": T.EXPECTED_TRAINABLE_PARAMETERS, "run_dir": str(run_dir),
        "checkpoint_paths": {str(s): f"m{s}.pt" for s in T.MODEL_SEEDS},
        "checkpoint_sha256": {str(s): f"h{s}" for s in T.MODEL_SEEDS},
        "scaler": {"token_cont_mean": [0.0], "token_cont_std": [1.0],
                   "global_mean": [0.0], "global_std": [1.0]},
        "pos_weight": 1.0, "interaction_threshold": 10.0,
        "train_losses": {}, "validation_losses": {}, "epochs_run": {},
        "n_train_groups": 0, "n_validation_groups": 0, "written_at": "2026-09-23",
    }), encoding="utf-8")
    return {"run_dir": str(run_dir), "screen_role": "development_screen",
            "role_salt": "AC-TWIN-D2-V0.1|", "screen_groups": groups,
            "determinism_check_groups": 32}


def test_split_physical_separation_and_idempotence(tmp_path):
    config = _screen_config(tmp_path, G)
    paths = STATS.screen_artifact_paths(config)
    _write_synthetic_role(paths["role_dir"], G, seed=21)
    out = SPLIT.split_screen_truth(paths["role_dir"], paths["split_dir"],
                                   role_salt=config["role_salt"], expect_groups=G)
    with np.load(out["inference_inputs"]) as z:
        inputs = set(z.files)
    with np.load(out["sealed_truth"]) as z:
        truth = set(z.files)
    # the legal inputs carry NO counterfactual future; the truth seals it + the eval mask.
    assert "object_counterfactual_futures" not in inputs
    assert "object_counterfactual_futures" in truth
    assert "evaluation_mask" in truth and "history_last_xy" in truth
    assert "object_history" in inputs and "object_factual_future" in inputs
    # group-key alignment: identical derived group_seed in both files.
    with np.load(out["inference_inputs"]) as z:
        gi = z["group_seed"]
    with np.load(out["sealed_truth"]) as z:
        gt = z["group_seed"]
    assert np.array_equal(gi, gt) and np.unique(gi).size == G
    # idempotent + drift-verifying re-run.
    again = SPLIT.split_screen_truth(paths["role_dir"], paths["split_dir"],
                                     role_salt=config["role_salt"], expect_groups=G)
    assert again["inference_inputs"] == out["inference_inputs"]


def _synthetic_scores(quantities: dict) -> dict:
    by_seed = np.asarray(quantities["risk_by_seed"], dtype=np.float64)
    ensemble = by_seed.mean(axis=0)  # exactly the arithmetic mean (create-once check)
    return {
        "group_seed": quantities["group_seed"],
        "t_star": quantities["t_star"],
        "interaction_min_distance": quantities["interaction_min_distance"],
        "j_hat": quantities["J_hat"],
        "ensemble_risk": ensemble,
        "risk_by_seed": by_seed,
        "mu_only_ensemble": ensemble * 0.5,
        "class_only_ensemble": ensemble * 0.5,
        "valid_mask": np.ones(len(quantities["group_seed"]), dtype=bool),
        "baselines": quantities["baseline_risks"],
    }


def test_one_shot_sequence_refuses_second_adjudication(tmp_path):
    config = _screen_config(tmp_path, G)
    paths = STATS.screen_artifact_paths(config)
    _write_synthetic_role(paths["role_dir"], G, seed=33)
    SPLIT.split_screen_truth(paths["role_dir"], paths["split_dir"],
                             role_salt=config["role_salt"], expect_groups=G)

    quantities = _passing_quantities(seed=5)
    # align the synthetic scores to the split's derived group_seed / scenario order.
    with np.load(paths["inference_inputs"]) as z:
        quantities["group_seed"] = z["group_seed"]
    scores = _synthetic_scores(quantities)
    determinism = {"role": "development_screen", "n_groups": 32,
                   "group_indices": list(range(32)), "bitwise_identical": True,
                   "max_abs_diff": 0.0, "device": "cpu", "batch_size": 32}

    STATS.freeze_screen_scores(config, scores=scores, determinism=determinism)
    assert paths["scores_npz"].is_file() and paths["scores_receipt"].is_file()
    STATS.burn_screen_lock(config)
    assert paths["lock"].is_file()

    decision = STATS.adjudicate_screen(config)
    assert paths["report"].is_file() and paths["quantities_npz"].is_file()
    assert decision.status in (STATS.STATUS_DEV_READY, STATS.STATUS_DEV_NOT_READY,
                               STATS.STATUS_INVALID_ANALYSIS)

    # a second adjudication is refused once the formal report exists.
    with pytest.raises(FileExistsError):
        STATS.adjudicate_screen(config)
    with pytest.raises(FileExistsError):
        STATS.assert_screen_adjudication_open(config)
