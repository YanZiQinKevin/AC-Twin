"""Focused tests for Stage 4D-D2 Direct training / gates / freeze (plan 7.4-7.6).

Synthetic only — no MetaDrive, no collected data. Proves the Direct contract (15,740
params, 315-dim input, five seeds, arithmetic-mean ensemble), the train-support gate
(plan 7.5), the validation gate + baseline tie-break (plan 7.6) and the create-once
freeze, independent of the base-predictor stack.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ac_twin import stage4cp_features as s4cpf
from ac_twin import stage4c_objective as s4co
from ac_twin import stage4d_external_training as T

ROLE_SALT = "AC-TWIN-D2-V0.1|"


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _synthetic_features(groups: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Finite (G,9,34) tokens + (G,9) globals with genuine variance on every column."""
    rng = _rng(seed)
    tokens = np.zeros((groups, T.N_COMPETITORS, T.TOKEN_DIM), dtype=np.float64)
    cont = rng.normal(0.0, 1.0, size=(groups, T.N_COMPETITORS, s4cpf.N_TOKEN_CONT))
    tokens[:, :, s4cpf.TOKEN_CONT_IDX] = cont
    # valid one-hot blocks: exactly one of each action/duration group is set.
    cat = np.zeros((groups, T.N_COMPETITORS, T.TOKEN_DIM - s4cpf.N_TOKEN_CONT))
    for g in range(groups):
        for k in range(T.N_COMPETITORS):
            cat[g, k, rng.integers(0, 5)] = 1.0            # selected_action_0..4
            cat[g, k, 5 + rng.integers(0, 2)] = 1.0        # selected_duration_1/3
            cat[g, k, 7 + rng.integers(0, 5)] = 1.0        # competitor_action_0..4
            cat[g, k, 12 + rng.integers(0, 2)] = 1.0       # competitor_duration_1/3
    tokens[:, :, s4cpf.TOKEN_CAT_IDX] = cat
    global_features = rng.normal(0.0, 1.0, size=(groups, T.GLOBAL_DIM))
    # visible_vehicle_count (index 7) is the structural constant 4 by contract.
    global_features[:, 7] = 4.0
    return tokens, global_features


def _labels_from_features(tokens: np.ndarray, seed: int = 1):
    """A learnable signal: regret grows with the first token's first continuous feature."""
    rng = _rng(seed)
    latent = tokens[:, 0, 0] + 0.1 * rng.normal(0.0, 1.0, size=tokens.shape[0])
    regret = np.clip(0.02 * (latent - latent.min() + 0.5), 0.0, None)
    y_reg = np.log1p(regret / T.HIGH_REGRET_THRESHOLD)
    y_high = (regret > T.HIGH_REGRET_THRESHOLD).astype(np.float64)
    return regret, y_reg, y_high


def _train_bundle(groups: int, seed: int = 0) -> dict[str, np.ndarray]:
    tokens, global_features = _synthetic_features(groups, seed)
    regret, y_reg, y_high = _labels_from_features(tokens, seed + 1)
    return {
        "tokens": tokens,
        "global_features": global_features,
        "y_reg": y_reg,
        "y_high": y_high,
        "regret": regret,
        "group_seed": np.arange(1, groups + 1, dtype=np.int64),
        "interaction_min_distance": np.linspace(5.0, 40.0, groups),
    }


def _validation_bundle(groups: int, seed: int = 100) -> dict[str, np.ndarray]:
    tokens, global_features = _synthetic_features(groups, seed)
    regret, y_reg, y_high = _labels_from_features(tokens, seed + 1)
    return {
        "tokens": tokens,
        "global_features": global_features,
        "y_reg": y_reg,
        "y_high": y_high,
        "regret": regret,
        "group_seed": np.arange(100001, 100001 + groups, dtype=np.int64),
    }


def test_direct_parameter_count_is_frozen() -> None:
    assert T.count_direct_parameters() == T.EXPECTED_TRAINABLE_PARAMETERS == 15740
    assert T.DIRECT_FLAT_DIM == 315


def test_group_seed_deterministic_and_in_range() -> None:
    a = T.group_seed_from_scenario_id("sd_waymo_v1.2_abc", ROLE_SALT)
    b = T.group_seed_from_scenario_id("sd_waymo_v1.2_abc", ROLE_SALT)
    c = T.group_seed_from_scenario_id("sd_waymo_v1.2_xyz", ROLE_SALT)
    assert a == b and a != c
    assert 1 <= a <= 2**31 - 2


def test_compute_true_labels_oracle_and_regret() -> None:
    groups, n_actions, T_, N = 6, T.N_ACTIONS, 20, 5
    rng = _rng(7)
    true_xy = rng.normal(0.0, 1.0, size=(groups, n_actions, T_, N, 2))
    history_last_xy = np.zeros((groups, 2))
    mask = np.ones((groups, T_, N), dtype=bool)
    # j_hat picks action 0 as argmin for every group.
    j_hat = np.tile(np.arange(n_actions, dtype=np.float64), (groups, 1))
    out = T.compute_true_labels(true_xy, history_last_xy, mask, j_hat)
    assert out["j_true"].shape == (groups, n_actions)
    assert out["a_hat"].shape == (groups,)
    assert (out["a_hat"] == 0).all()
    assert out["regret"].shape == (groups,)
    assert np.isfinite(out["y_reg"]).all()
    assert set(np.unique(out["y_high"])).issubset({0, 1})
    # regret is clamped non-negative and idle/random regrets are finite.
    assert (out["regret"] >= 0.0).all()
    assert np.isfinite(out["idle_mean_regret"])
    assert np.isfinite(out["random_mean_regret"])


def test_train_support_gate_pass_and_fail() -> None:
    bundle = _train_bundle(T.DEV_TRAIN_GROUPS, seed=0)
    # make enough high/nonzero regret groups by construction.
    report = T.evaluate_train_support_gate(bundle)
    assert report["status"] in (
        T.STATUS_TRAIN_SUPPORT_OK, T.STATUS_TRAIN_SUPPORT_INSUFFICIENT)
    assert set(report["conditions"]) == {
        "all_complete_10_action", "high_regret_groups_ge_750",
        "nonzero_regret_groups_ge_1250", "output_targets_finite",
        "continuous_features_finite_with_variance"}

    # failing: too few groups -> completeness fails.
    small = _train_bundle(100, seed=2)
    bad = T.evaluate_train_support_gate(small)
    assert not bad["passed"]
    assert bad["status"] == T.STATUS_TRAIN_SUPPORT_INSUFFICIENT
    assert not bad["conditions"]["all_complete_10_action"]

    # failing: degenerate (zero) regret -> support conditions fail.
    degen = _train_bundle(T.DEV_TRAIN_GROUPS, seed=3)
    degen["regret"] = np.zeros(T.DEV_TRAIN_GROUPS)
    degen["y_reg"] = np.zeros(T.DEV_TRAIN_GROUPS)
    degen["y_high"] = np.zeros(T.DEV_TRAIN_GROUPS)
    bad2 = T.evaluate_train_support_gate(degen)
    assert not bad2["passed"]
    assert not bad2["conditions"]["high_regret_groups_ge_750"]
    assert not bad2["conditions"]["nonzero_regret_groups_ge_1250"]


def test_fit_five_direct_models_create_once_and_ensemble(tmp_path: Path) -> None:
    train = _train_bundle(300, seed=0)
    validation = _validation_bundle(80, seed=100)
    inventory = T.fit_five_direct_models(train, validation, tmp_path, device="cpu")
    assert inventory.seeds == T.MODEL_SEEDS == (9101, 9202, 9303, 9404, 9505)
    assert inventory.param_count == 15740
    assert len(inventory.checkpoint_paths) == 5
    for seed, path in inventory.checkpoint_paths.items():
        assert Path(path).is_file()
    # create-once: a second call verifies and does not retrain.
    again = T.fit_five_direct_models(train, validation, tmp_path, device="cpu")
    assert again.verification == "existing_inventory_verified_no_overwrite"
    assert again.checkpoint_sha256 == inventory.checkpoint_sha256

    # scoring: ensemble == arithmetic mean of the five per-seed risk scores.
    scores = T.score_models(inventory, validation, device="cpu")
    assert scores.risk_ensemble.shape == (80,)
    assert scores.risk_by_seed.shape == (5, 80)
    assert np.allclose(scores.risk_ensemble, scores.risk_by_seed.mean(axis=0),
                       rtol=0.0, atol=1e-15)
    assert np.isfinite(scores.risk_ensemble).all()
    # a learnable signal was planted: the ensemble should rank training- regret better
    # than chance (spearman > 0) on the validation bundle it early-stopped on.
    from ac_twin import stage4cp_statistics as s4cs
    rho = s4cs.spearman(scores.risk_ensemble, validation["regret"])
    assert rho > 0.0


def test_select_strongest_baseline_tie_break() -> None:
    groups = 200
    rng = _rng(5)
    regret = np.clip(rng.normal(0.02, 0.02, size=groups), 0.0, None)
    y_high = (regret > T.HIGH_REGRET_THRESHOLD).astype(np.int64)
    # MARGIN_ONLY correlates with regret (best AURC); the others are noise.
    risks = {
        T.BASELINE_MARGIN: regret + 0.001 * rng.normal(size=groups),
        T.BASELINE_SEED_VOTE_ENTROPY: rng.normal(size=groups),
        T.BASELINE_SEED_MARGIN_STD: rng.normal(size=groups),
    }
    receipt = T.select_strongest_standard_baseline(
        {"regret": regret, "y_high": y_high, "baseline_risks": risks})
    assert receipt.strongest == T.BASELINE_MARGIN
    assert receipt.tie_break_order == T.STRONGEST_BASELINE_CANDIDATES
    for name in T.STRONGEST_BASELINE_CANDIDATES:
        assert np.isfinite(receipt.aurc_regret[name])
        assert np.isfinite(receipt.regret_at_80[name])
        assert np.isfinite(receipt.pr_auc_minus_prevalence[name])


def test_validation_gate_and_freeze_create_once(tmp_path: Path) -> None:
    # Direct ensemble strongly tracks regret; baseline does not -> gate should pass.
    groups = 300
    rng = _rng(11)
    regret = np.clip(rng.normal(0.03, 0.03, size=groups), 0.0, None)
    y_high = (regret > T.HIGH_REGRET_THRESHOLD).astype(np.int64)
    group_seed = np.arange(1, groups + 1, dtype=np.int64)
    by_seed = np.stack([regret + 0.001 * rng.normal(size=groups)
                        for _ in T.MODEL_SEEDS])  # (5,G) strongly correlated
    scores = T.RiskScores(
        group_seed=group_seed,
        risk_ensemble=by_seed.mean(axis=0),
        risk_by_seed=by_seed,
        mu_only_ensemble=by_seed.mean(axis=0),
        class_only_ensemble=by_seed.mean(axis=0),
        seeds=T.MODEL_SEEDS)
    baseline_risks = {
        T.BASELINE_MARGIN: rng.normal(size=groups),          # uncorrelated -> worse AURC
        T.BASELINE_SEED_VOTE_ENTROPY: rng.normal(size=groups),
        T.BASELINE_SEED_MARGIN_STD: rng.normal(size=groups),
    }
    receipt = T.select_strongest_standard_baseline(
        {"regret": regret, "y_high": y_high, "baseline_risks": baseline_risks})
    truth = {"regret": regret, "y_high": y_high, "baseline_risks": baseline_risks,
             "baseline_receipt": receipt, "group_seed": group_seed,
             "interaction_threshold": 20.0}
    report = T.evaluate_validation_gate(scores, truth)
    assert report.passed, report.conditions
    assert report.status == T.STATUS_VALIDATION_QUALIFIED
    assert report.metrics["i_aurc_regret"] >= T.VALIDATION_I_AURC_MIN
    assert report.metrics["i_regret80"] >= T.VALIDATION_I_REGRET80_MIN
    assert report.metrics["seeds_i_aurc_positive"] >= T.VALIDATION_MIN_POSITIVE_SEEDS
    assert report.metrics["seeds_i_regret80_positive"] >= T.VALIDATION_MIN_POSITIVE_SEEDS

    # freeze: writes the five create-once artifacts; a second freeze is refused.
    inventory = T.fit_five_direct_models(
        _train_bundle(200, seed=0), _validation_bundle(60, seed=100),
        tmp_path / "run", device="cpu")
    dev_dir = tmp_path / "run" / "development"
    final_path = T.freeze_final_selection(report, inventory, dev_dir)
    assert final_path.is_file()
    for name in T.FREEZE_ARTIFACTS:
        assert (dev_dir / name).is_file()
    with pytest.raises(FileExistsError):
        T.freeze_final_selection(report, inventory, dev_dir)


def test_freeze_refuses_failed_gate(tmp_path: Path) -> None:
    groups = 120
    rng = _rng(13)
    regret = np.clip(rng.normal(0.02, 0.02, size=groups), 0.0, None)
    y_high = (regret > T.HIGH_REGRET_THRESHOLD).astype(np.int64)
    group_seed = np.arange(1, groups + 1, dtype=np.int64)
    # ensemble UNCORRELATED with regret -> improvements ~ 0 -> gate fails.
    by_seed = np.stack([rng.normal(size=groups) for _ in T.MODEL_SEEDS])
    scores = T.RiskScores(group_seed=group_seed, risk_ensemble=by_seed.mean(axis=0),
                          risk_by_seed=by_seed, mu_only_ensemble=by_seed.mean(axis=0),
                          class_only_ensemble=by_seed.mean(axis=0), seeds=T.MODEL_SEEDS)
    # baseline strongly tracks regret -> baseline better -> negative improvement.
    baseline_risks = {
        T.BASELINE_MARGIN: regret + 0.001 * rng.normal(size=groups),
        T.BASELINE_SEED_VOTE_ENTROPY: rng.normal(size=groups),
        T.BASELINE_SEED_MARGIN_STD: rng.normal(size=groups),
    }
    receipt = T.select_strongest_standard_baseline(
        {"regret": regret, "y_high": y_high, "baseline_risks": baseline_risks})
    truth = {"regret": regret, "y_high": y_high, "baseline_risks": baseline_risks,
             "baseline_receipt": receipt, "group_seed": group_seed,
             "interaction_threshold": 20.0}
    report = T.evaluate_validation_gate(scores, truth)
    assert not report.passed
    assert report.status == T.STATUS_VALIDATION_NOT_QUALIFIED
    inventory = T.fit_five_direct_models(
        _train_bundle(120, seed=0), _validation_bundle(40, seed=100),
        tmp_path / "run", device="cpu")
    with pytest.raises(ValueError):
        T.freeze_final_selection(report, inventory, tmp_path / "run" / "development")
