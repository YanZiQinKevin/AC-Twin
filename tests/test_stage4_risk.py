from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4_risk as s4r  # noqa: E402


def _synthetic(episodes: int = 6, agents: int = 4, horizon: int = 5, seeds: int = 3):
    rng = np.random.default_rng(7)
    valid = np.ones((episodes, horizon, agents), dtype=bool)
    primary = rng.normal(size=(seeds, episodes, horizon, agents, 2)).astype(np.float64)
    full = primary + 0.05 * rng.normal(size=(seeds, episodes, horizon, agents, 2))
    history = primary + 0.05 * rng.normal(size=(seeds, episodes, horizon, agents, 2))
    response = {
        "r_hat": np.abs(rng.normal(size=(seeds, episodes, horizon, agents))),
        "presence_prob": rng.uniform(size=(seeds, episodes, horizon, agents)),
        "high_prob": rng.uniform(size=(seeds, episodes, horizon, agents)),
    }
    geometry = {
        "distance_to_ego": np.abs(rng.normal(size=(episodes, agents))),
        "relative_speed": np.abs(rng.normal(size=(episodes, agents))),
        "visible_count": np.full(episodes, float(agents - 1)),
    }
    action = rng.integers(0, 5, size=episodes)
    duration = rng.choice([1, 3], size=episodes).astype(float)
    return primary, full, history, response, geometry, action, duration, valid


def test_build_effect_target_is_counterfactual_minus_factual():
    factual = np.zeros((2, 4, 3, 2), dtype=np.float32)
    counterfactual = np.ones((2, 4, 3, 2), dtype=np.float32)
    valid = np.ones((2, 4, 3), dtype=bool)
    effect, out_valid = s4r.build_effect_target(factual, counterfactual, valid)
    assert np.allclose(effect, 1.0)
    assert out_valid.shape == (2, 4, 3)


def test_episode_effect_ade_hand_computed():
    # One episode, one agent, two valid timesteps; pred==true -> 0; off by (3,4) -> 5.
    true = np.zeros((1, 2, 1, 2), dtype=np.float64)
    pred = np.zeros((1, 2, 1, 2), dtype=np.float64)
    valid = np.ones((1, 2, 1), dtype=bool)
    assert s4r.episode_effect_ade(pred, true, valid)[0, 0] == pytest.approx(0.0)
    pred[0, :, 0, :] = np.array([[3.0, 4.0], [3.0, 4.0]])
    assert s4r.episode_effect_ade(pred, true, valid)[0, 0] == pytest.approx(5.0)


def test_episode_effect_ade_per_seed_shape():
    episodes, horizon, agents, seeds = 3, 4, 2, 5
    true = np.zeros((episodes, horizon, agents, 2))
    pred = np.zeros((seeds, episodes, horizon, agents, 2))
    valid = np.ones((episodes, horizon, agents), dtype=bool)
    out = s4r.episode_effect_ade(pred, true, valid)
    assert out.shape == (seeds, episodes)


def test_episode_effect_ade_requires_a_valid_cell():
    true = np.zeros((1, 2, 1, 2))
    pred = np.zeros((1, 2, 1, 2))
    valid = np.zeros((1, 2, 1), dtype=bool)
    with pytest.raises(ValueError, match="at least one valid cell"):
        s4r.episode_effect_ade(pred, true, valid)


def test_risk_target_is_mean_over_seed_pairs():
    errors = np.array([[1.0, 3.0], [3.0, 5.0]])  # S=2, E=2
    target = s4r.risk_target_from_pairs(errors)
    assert np.allclose(target, [2.0, 4.0])


def test_legal_features_reject_truth_side_kwarg():
    primary, full, history, response, geometry, action, duration, valid = _synthetic()
    with pytest.raises(TypeError):
        s4r.build_legal_risk_features(
            primary_predictions=primary,
            full_predictions=full,
            history_predictions=history,
            response_predictions=response,
            pre_intervention_geometry=geometry,
            action=action,
            duration=duration,
            valid_mask=valid,
            true_error=np.zeros(primary.shape[1]),  # illegal leak
        )


def test_legal_features_reject_counterfactual_kwarg():
    primary, full, history, response, geometry, action, duration, valid = _synthetic()
    with pytest.raises(TypeError):
        s4r.build_legal_risk_features(
            primary_predictions=primary,
            full_predictions=full,
            history_predictions=history,
            response_predictions=response,
            pre_intervention_geometry=geometry,
            action=action,
            duration=duration,
            valid_mask=valid,
            counterfactual_xy=np.zeros((1, 1, 1, 2)),
        )


def test_legal_features_shape_and_finiteness():
    primary, full, history, response, geometry, action, duration, valid = _synthetic()
    episodes = primary.shape[1]
    features = s4r.build_legal_risk_features(
        primary_predictions=primary,
        full_predictions=full,
        history_predictions=history,
        response_predictions=response,
        pre_intervention_geometry=geometry,
        action=action,
        duration=duration,
        valid_mask=valid,
    )
    assert set(s4r.ALL_FEATURES) <= set(features)
    for name in s4r.ALL_FEATURES:
        value = features[name]
        assert value.shape == (episodes,)
        assert np.isfinite(value).all()


def test_uncertainty_is_zero_when_seeds_agree():
    primary, full, history, response, geometry, action, duration, valid = _synthetic()
    # Force all seed-pairs to the SAME prediction -> cross-seed dispersion == 0.
    primary[:] = primary[0:1]
    features = s4r.build_legal_risk_features(
        primary_predictions=primary,
        full_predictions=full,
        history_predictions=history,
        response_predictions=response,
        pre_intervention_geometry=geometry,
        action=action,
        duration=duration,
        valid_mask=valid,
    )
    assert np.allclose(features["U_seed_mean"], 0.0)
    assert np.allclose(features["U_seed_max"], 0.0)


def test_disagreement_is_zero_when_experts_agree():
    primary, full, history, response, geometry, action, duration, valid = _synthetic()
    history[:] = full  # Full == History -> D_expert == 0
    features = s4r.build_legal_risk_features(
        primary_predictions=primary,
        full_predictions=full,
        history_predictions=history,
        response_predictions=response,
        pre_intervention_geometry=geometry,
        action=action,
        duration=duration,
        valid_mask=valid,
    )
    assert np.allclose(features["D_expert_mean"], 0.0)
    assert np.allclose(features["D_expert_max"], 0.0)


def test_action_one_hot_and_duration_metadata():
    primary, full, history, response, geometry, action, duration, valid = _synthetic()
    action = np.array([0, 1, 2, 3, 4, 0])
    features = s4r.build_legal_risk_features(
        primary_predictions=primary,
        full_predictions=full,
        history_predictions=history,
        response_predictions=response,
        pre_intervention_geometry=geometry,
        action=action,
        duration=duration,
        valid_mask=valid,
    )
    for episode, code in enumerate(action):
        assert features[f"action_{code}"][episode] == 1.0
        assert sum(features[f"action_{k}"][episode] for k in range(5)) == 1.0
    assert np.allclose(features["horizon"], float(valid.shape[1]))


def test_direct_risk_score_uses_preregistered_mean():
    features = {"U_seed_mean": np.array([0.5, 1.5]), "D_expert_mean": np.array([2.0, 0.1])}
    assert np.allclose(s4r.direct_risk_score(s4r.CANDIDATE_UNCERTAINTY, features), [0.5, 1.5])
    assert np.allclose(s4r.direct_risk_score(s4r.CANDIDATE_DISAGREEMENT, features), [2.0, 0.1])


def test_fit_risk_candidate_refuses_direct_candidate():
    primary, full, history, response, geometry, action, duration, valid = _synthetic()
    features = s4r.build_legal_risk_features(
        primary_predictions=primary, full_predictions=full, history_predictions=history,
        response_predictions=response, pre_intervention_geometry=geometry,
        action=action, duration=duration, valid_mask=valid,
    )
    target = np.linspace(0.0, 1.0, primary.shape[1])
    with pytest.raises(ValueError, match="direct"):
        s4r.fit_risk_candidate(s4r.CANDIDATE_UNCERTAINTY, features, target, {})


def test_fit_risk_candidate_ridge_roundtrip():
    primary, full, history, response, geometry, action, duration, valid = _synthetic(
        episodes=30
    )
    features = s4r.build_legal_risk_features(
        primary_predictions=primary, full_predictions=full, history_predictions=history,
        response_predictions=response, pre_intervention_geometry=geometry,
        action=action, duration=duration, valid_mask=valid,
    )
    target = np.abs(np.random.default_rng(3).normal(size=30)) + 0.1
    config = {"ridge_alpha": 1.0, "winsor_lower_q": 0.01, "winsor_upper_q": 0.99}
    model = s4r.fit_risk_candidate(
        s4r.CANDIDATE_COMBINED, features, target, config, implementation="ridge"
    )
    matrix = s4r.design_matrix(features, s4r.CANDIDATE_FEATURES[s4r.CANDIDATE_COMBINED])
    score = model.predict(matrix)
    assert score.shape == (30,)
    assert np.isfinite(score).all()


def test_winsorization_bounds_fit_on_train_only():
    # A huge outlier in TRAIN must be clipped by the train-fitted winsor bounds,
    # but the frozen bounds are stored, not refit, at predict time.
    rng = np.random.default_rng(11)
    episodes, agents, horizon, seeds = 40, 3, 4, 2
    valid = np.ones((episodes, horizon, agents), dtype=bool)
    primary = rng.normal(size=(seeds, episodes, horizon, agents, 2))
    response = {
        "r_hat": np.abs(rng.normal(size=(seeds, episodes, horizon, agents))),
        "presence_prob": rng.uniform(size=(seeds, episodes, horizon, agents)),
        "high_prob": rng.uniform(size=(seeds, episodes, horizon, agents)),
    }
    geometry = {
        "distance_to_ego": np.abs(rng.normal(size=(episodes, agents))),
        "relative_speed": np.abs(rng.normal(size=(episodes, agents))),
        "visible_count": np.full(episodes, float(agents - 1)),
    }
    action = rng.integers(0, 5, size=episodes)
    duration = rng.choice([1, 3], size=episodes).astype(float)
    features = s4r.build_legal_risk_features(
        primary_predictions=primary, full_predictions=primary, history_predictions=primary,
        response_predictions=response, pre_intervention_geometry=geometry,
        action=action, duration=duration, valid_mask=valid,
    )
    target = np.abs(rng.normal(size=episodes)) + 0.1
    target[0] = 1e6  # extreme outlier
    config = {"ridge_alpha": 1.0, "winsor_lower_q": 0.01, "winsor_upper_q": 0.99}
    model = s4r.fit_risk_candidate(
        s4r.CANDIDATE_COMBINED, features, target, config, implementation="ridge"
    )
    low, high = model.winsor_bounds
    assert high < 1e6  # the outlier is clipped by the frozen upper bound


def test_combined_requires_explicit_implementation():
    primary, full, history, response, geometry, action, duration, valid = _synthetic(
        episodes=20
    )
    features = s4r.build_legal_risk_features(
        primary_predictions=primary, full_predictions=full, history_predictions=history,
        response_predictions=response, pre_intervention_geometry=geometry,
        action=action, duration=duration, valid_mask=valid,
    )
    target = np.abs(np.random.default_rng(2).normal(size=20)) + 0.1
    with pytest.raises(ValueError, match="explicit implementation"):
        s4r.fit_risk_candidate(s4r.CANDIDATE_COMBINED, features, target, {})


# --- save/load round-trips (MLP serialization regression + all five model paths) ----
#
# fit-risk originally crashed saving the MLP: `_fit_mlp` returned a closure-local
# `_TorchEstimator` that torch.save/pickle could not serialize. The fix hoists
# `_TorchEstimator` to module level holding the trained model. These tests lock the
# round-trip for every formal model path used by cmd_fit_risk.


def _fit_for_roundtrip(name, implementation, episodes=40):
    primary, full, history, response, geometry, action, duration, valid = _synthetic(
        episodes=episodes
    )
    features = s4r.build_legal_risk_features(
        primary_predictions=primary, full_predictions=full, history_predictions=history,
        response_predictions=response, pre_intervention_geometry=geometry,
        action=action, duration=duration, valid_mask=valid,
    )
    target = np.abs(np.random.default_rng(4).normal(size=episodes)) + 0.1
    config = {
        "ridge_alpha": 1.0,
        "winsor_lower_q": 0.01, "winsor_upper_q": 0.99,
        "histgb": {"learning_rate": 0.05, "max_iter": 20, "max_leaf_nodes": 15,
                   "min_samples_leaf": 5, "l2_regularization": 1.0, "early_stopping": False},
        "mlp": {"hidden_dim": 32, "max_epochs": 5, "patience": 2,
                "batch_size": 64, "learning_rate": 1e-3},
    }
    model = s4r.fit_risk_candidate(name, features, target, config, implementation=implementation)
    matrix = s4r.design_matrix(features, s4r.CANDIDATE_FEATURES[name])
    return model, matrix


@pytest.mark.parametrize(
    "name,impl",
    [
        (s4r.CANDIDATE_COMBINED, "ridge"),
        (s4r.CANDIDATE_COMBINED, "histgb"),
        (s4r.CANDIDATE_COMBINED, "mlp"),
        (s4r.BASELINE_GEOMETRY, "ridge"),
        (s4r.CANDIDATE_RESPONSE, "ridge"),
    ],
)
def test_model_save_load_roundtrip_all_paths(name, impl, tmp_path: Path, monkeypatch):
    # joblib/loky cannot detect physical cores on this Windows host and emits a
    # UserWarning that the project pytest config (filterwarnings=error) would turn
    # into an error for the HistGB path. The production runner already sets the
    # OpenMP env vars; here we pin the CPU count so loky skips detection. This is
    # an environment quirk only — the real fit-risk run fit HistGB successfully.
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "4")
    model, matrix = _fit_for_roundtrip(name, impl)
    before = model.predict(matrix)
    path = tmp_path / f"{name.lower()}_{impl}.pt"
    model.save(path)
    assert path.is_file()
    # The temporary serialization file must be cleaned up after a successful save.
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    loaded = s4r.FittedRiskModel.load(path)
    after = loaded.predict(matrix)
    assert loaded.implementation == impl
    # Predictions are identical before and after the save/load round-trip.
    assert np.allclose(before, after, atol=0.0, rtol=0.0)


def test_mlp_estimator_is_module_level_and_picklable():
    # The MLP estimator wrapper must be a module-level class (picklable), not a closure.
    import pickle

    import torch
    from torch import nn

    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
    estimator = s4r._TorchEstimator(model)
    restored = pickle.loads(pickle.dumps(estimator))
    x = np.zeros((2, 3))
    assert np.allclose(estimator.predict(x), restored.predict(x))
