from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin.stage3_data import FROZEN_T1  # noqa: E402
from ac_twin.stage3_response import Stage3ResponsePredictor, softmax  # noqa: E402
from ac_twin.stage3b_bridge import (  # noqa: E402
    build_effect_target,
    calibrated_regime_probs,
    make_oracle_gate,
    predictor_features_ordered,
    router_masked_mse,
    router_masked_mse_sums,
)


def test_effect_target_is_counterfactual_minus_factual_xy():
    rng = np.random.default_rng(0)
    factual = rng.normal(size=(3, 20, 4, 5)).astype(np.float32)
    counterfactual = rng.normal(size=(3, 20, 4, 5)).astype(np.float32)
    target = build_effect_target(factual, counterfactual)
    assert target.shape == (3, 20, 4, 2)
    expected = counterfactual[..., 2:4] - factual[..., 2:4]
    np.testing.assert_allclose(target, expected, rtol=1e-6)
    # Only the xy position channels contribute; other feature channels must not.
    changed = counterfactual.copy()
    changed[..., 0] += 100.0
    np.testing.assert_allclose(
        build_effect_target(factual, changed), expected, rtol=1e-6
    )


def test_oracle_gate_is_frozen_monotone_bounded_map():
    r = np.array([0.0, FROZEN_T1, 10 * FROZEN_T1], dtype=np.float32).reshape(1, 1, 3)
    gate = make_oracle_gate(r)
    assert gate.shape == (1, 1, 3, 1)
    assert gate.dtype == np.float32
    # R=0 -> 0, R=t1 -> 0.5, monotone non-decreasing, bounded in [0, 1).
    assert gate[0, 0, 0, 0] == pytest.approx(0.0, abs=1e-7)
    assert gate[0, 0, 1, 0] == pytest.approx(0.5, abs=1e-6)
    assert np.all(np.diff(gate[0, 0, :, 0]) >= 0.0)
    assert np.all((gate >= 0.0) & (gate < 1.0))


def test_oracle_gate_rejects_negative_response():
    with pytest.raises(ValueError, match="non-negative"):
        make_oracle_gate(np.array([[[-1.0]]], dtype=np.float32))


def test_router_masked_mse_excludes_padding():
    effect = torch.zeros(1, 2, 2, 2)
    target = torch.zeros(1, 2, 2, 2)
    # One valid agent-time with squared error 4 per dim (8 total), one padded
    # position with huge error that must be ignored.
    target[0, 0, 0, :] = 2.0
    target[0, 1, 1, :] = 1e6
    valid = torch.tensor([[[[True, False], [False, False]]]], dtype=torch.bool).reshape(1, 2, 2)
    loss = router_masked_mse(effect, target, valid)
    # denom = 1 valid agent-time * 2 dims = 2; mass = 8 -> 4.0
    assert float(loss) == pytest.approx(4.0, rel=1e-6)


def test_router_masked_mse_sums_invariant_to_batching_and_permutation():
    rng = np.random.default_rng(1)
    effect = torch.as_tensor(rng.normal(size=(5, 4, 3, 2)), dtype=torch.float32)
    target = torch.as_tensor(rng.normal(size=(5, 4, 3, 2)), dtype=torch.float32)
    valid = torch.as_tensor(rng.random(size=(5, 4, 3)) > 0.3, dtype=torch.bool)

    def pooled(e, t, v):
        s = router_masked_mse_sums(e, t, v)
        return float(s["reg_sum"]) / max(float(s["valid_count"]), 1.0)

    whole = pooled(effect, target, valid)
    # Split into two batches along episodes and recombine via sums.
    s1 = router_masked_mse_sums(effect[:2], target[:2], valid[:2])
    s2 = router_masked_mse_sums(effect[2:], target[2:], valid[2:])
    split = (float(s1["reg_sum"]) + float(s2["reg_sum"])) / (
        float(s1["valid_count"]) + float(s2["valid_count"])
    )
    perm = torch.as_tensor(np.random.default_rng(2).permutation(5))
    permuted = pooled(effect[perm], target[perm], valid[perm])
    assert whole == pytest.approx(split, rel=1e-9)
    assert whole == pytest.approx(permuted, rel=1e-9)


def test_calibrated_regime_probs_match_manual_per_seed_vector():
    rng = np.random.default_rng(3)
    n_seeds, n = 5, 40
    logits = rng.normal(size=(n_seeds, n, 4)).astype(np.float64)
    per_seed = [
        {"a": rng.normal(size=4).tolist(), "b": rng.normal(size=4).tolist()}
        for _ in range(n_seeds)
    ]
    probs = calibrated_regime_probs(per_seed, logits)
    assert probs.shape == (n_seeds, n, 4)
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, rtol=1e-6)
    for i in range(n_seeds):
        a = np.asarray(per_seed[i]["a"])
        b = np.asarray(per_seed[i]["b"])
        scale = np.log1p(np.exp(-np.abs(a))) + np.maximum(a, 0.0)  # softplus
        expected = softmax(logits[i] * scale + b, axis=-1)
        np.testing.assert_allclose(probs[i], expected, rtol=1e-6)


def test_calibrated_regime_probs_requires_matching_seed_count():
    per_seed = [{"a": [0.0] * 4, "b": [0.0] * 4} for _ in range(5)]
    with pytest.raises(ValueError):
        calibrated_regime_probs(per_seed, np.zeros((3, 10, 4)))


def _tiny_tensors(n_episodes: int) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(7)
    return {
        "history": torch.as_tensor(np.abs(rng.normal(size=(n_episodes, 10, 3, 8))), dtype=torch.float32),
        "factual_suffix": torch.as_tensor(np.abs(rng.normal(size=(n_episodes, 10, 3, 8))), dtype=torch.float32),
        "factual_actions": torch.as_tensor(rng.integers(0, 5, size=(n_episodes, 20)), dtype=torch.long),
        "counterfactual_actions": torch.as_tensor(rng.integers(0, 5, size=(n_episodes, 20)), dtype=torch.long),
        "cf_action": torch.as_tensor(rng.integers(0, 5, size=n_episodes), dtype=torch.long),
        "cf_duration": torch.as_tensor(rng.integers(0, 4, size=n_episodes), dtype=torch.long),
        "t_star": torch.as_tensor(rng.integers(10, 16, size=n_episodes), dtype=torch.long),
    }


def test_predictor_features_ordered_preserves_requested_episode_order():
    # Frozen inference runs under the same deterministicAlgorithms the pipeline
    # always sets via seed_all; batch-invariance is only guaranteed there.
    from ac_twin.stage3_training import seed_all

    seed_all(1101)
    tensors = _tiny_tensors(9)
    model = Stage3ResponsePredictor(hidden_dim=8, response_feature_dim=6)
    requested = np.array([7, 0, 5, 2])  # non-contiguous, non-sorted
    out = predictor_features_ordered(model, tensors, requested, "cpu", batch_size=2)
    np.testing.assert_array_equal(out["episode_indices"], requested)
    assert out["response_features"].shape == (4, 20, 3, 6)
    assert out["regime_logits"].shape == (4, 20, 3, 4)
    # Feature row i must correspond to the requested episode, not position i.
    single = predictor_features_ordered(model, tensors, np.array([7]), "cpu", batch_size=1)
    np.testing.assert_allclose(
        out["response_features"][0], single["response_features"][0], rtol=1e-5
    )
