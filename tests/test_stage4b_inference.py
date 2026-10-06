"""Stage 4B inference tests: whitelist, frozen-model load/determinism, recheck logic."""
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
from ac_twin import stage4b_inference as s4bi  # noqa: E402
from ac_twin import stage4b_protocol as s4bp  # noqa: E402


def _feature_inputs(episodes=8, agents=5, horizon=20):
    rng = np.random.default_rng(0)
    valid = np.ones((episodes, horizon, agents), dtype=bool)
    primary = rng.normal(size=(5, episodes, horizon, agents, 2)).astype(np.float32)
    full = rng.normal(size=(5, episodes, horizon, agents, 2)).astype(np.float32)
    history = rng.normal(size=(5, episodes, horizon, agents, 2)).astype(np.float32)
    response = {
        "r_hat": np.abs(rng.normal(size=(5, episodes, horizon, agents))).astype(np.float32),
        "presence_prob": rng.uniform(size=(5, episodes, horizon, agents)).astype(np.float32),
        "high_prob": rng.uniform(size=(5, episodes, horizon, agents)).astype(np.float32),
    }
    geometry = {
        "distance_to_ego": np.abs(rng.normal(size=(episodes, agents))),
        "relative_speed": np.abs(rng.normal(size=(episodes, agents))),
        "visible_count": np.full(episodes, float(agents - 1)),
    }
    action = rng.integers(0, 5, size=episodes)
    duration = rng.choice([1, 3], size=episodes)
    return dict(primary_predictions=primary, full_predictions=full,
                history_predictions=history, response_predictions=response,
                pre_intervention_geometry=geometry, action=action, duration=duration,
                valid_mask=valid)


def test_feature_constructor_rejects_truth_side_kwargs():
    inputs = _feature_inputs()
    for leak in ("true_error", "true_response", "counterfactual_xy", "cea", "cew", "oracle"):
        with pytest.raises(TypeError):
            s4r.build_legal_risk_features(**inputs, **{leak: np.ones(8)})


def test_feature_constructor_emits_all_29_features():
    features = s4r.build_legal_risk_features(**_feature_inputs())
    assert len(features) == len(s4r.ALL_FEATURES) == 29
    assert all(np.isfinite(v).all() for v in features.values())


def test_frozen_mlp_loads_and_deterministic_in_environment():
    frozen = s4bp.STAGE4A_FROZEN
    model = s4r.FittedRiskModel.load(ROOT / frozen["finalist_risk_model"][0])
    features = s4r.build_legal_risk_features(**_feature_inputs())
    matrix = s4r.design_matrix(features, s4r.CANDIDATE_FEATURES[s4r.CANDIDATE_COMBINED])
    first = model.predict(matrix)
    second = model.predict(matrix)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_frozen_geometry_ridge_loads():
    frozen = s4bp.STAGE4A_FROZEN
    model = s4r.FittedRiskModel.load(ROOT / frozen["geometry_risk_model"][0])
    features = s4r.build_legal_risk_features(**_feature_inputs())
    score = model.predict(s4r.design_matrix(features, s4r.CANDIDATE_FEATURES[s4r.BASELINE_GEOMETRY]))
    assert np.isfinite(score).all()


# --- determinism recheck (bitwise vs 1e-7 tolerance) ---------------------------------------


def _fake_bundle(episodes=40):
    rng = np.random.default_rng(0)
    return {"episode_seed": rng.integers(1, 10**6, size=episodes),
            "scaled": {}}


def test_determinism_recheck_bitwise(monkeypatch):
    output = {"e2e": np.ones((32, 20, 5, 2), dtype=np.float32)}
    monkeypatch.setattr(s4bi, "run_pair_inference", lambda *a, **k: dict(output))
    result = s4bi.determinism_recheck(
        {"device": "cpu", "batch_size": 32}, _fake_bundle(), binding={}, stage3r_dir=Path("."))
    assert result["bitwise_identical"] and result["max_abs_diff"] == 0.0
    assert result["n_episodes"] == 32 and result["pair_index"] == 0


def test_determinism_recheck_within_tolerance(monkeypatch):
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        return {"e2e": np.full((32, 20, 5, 2), 1e-8 * calls["n"], dtype=np.float32)}

    monkeypatch.setattr(s4bi, "run_pair_inference", fake)
    result = s4bi.determinism_recheck(
        {"device": "cuda", "batch_size": 32}, _fake_bundle(), binding={}, stage3r_dir=Path("."))
    assert not result["bitwise_identical"]
    assert result["max_abs_diff"] <= s4bi.DETERMINISM_TOL
    assert result["device"] == "cuda" and result["dtype"] == "float32"


def test_determinism_recheck_raises_beyond_tolerance(monkeypatch):
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        return {"e2e": np.full((32, 20, 5, 2), float(calls["n"]), dtype=np.float32)}

    monkeypatch.setattr(s4bi, "run_pair_inference", fake)
    with pytest.raises(RuntimeError, match="determinism recheck"):
        s4bi.determinism_recheck(
            {"device": "cuda", "batch_size": 32}, _fake_bundle(), binding={}, stage3r_dir=Path("."))
