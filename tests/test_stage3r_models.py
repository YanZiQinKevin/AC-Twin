from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin.stage3r_models import (  # noqa: E402
    CEWRouter,
    SameInformationDirect,
    Stage3RExpert,
    build_router_features,
    make_oof_folds,
    match_direct_hidden_width,
    parameter_count,
    stage3r_forward,
)


def _inputs(batch: int = 2, horizon: int = 4, agents: int = 3):
    generator = torch.Generator().manual_seed(77)
    history = torch.randn(batch, 5, agents, 8, generator=generator)
    suffix_a = torch.randn(batch, 5, agents, 8, generator=generator)
    suffix_b = torch.randn(batch, 5, agents, 8, generator=generator)
    history[..., 0] = 1.0
    suffix_a[..., 0] = 1.0
    suffix_b[..., 0] = 1.0
    factual_actions = torch.randint(0, 5, (batch, horizon), generator=generator)
    counterfactual_actions = torch.randint(0, 5, (batch, horizon), generator=generator)
    return history, suffix_a, suffix_b, factual_actions, counterfactual_actions


def _expert(condition: str) -> Stage3RExpert:
    torch.manual_seed(19)
    return Stage3RExpert(
        condition=condition,
        feature_dim=8,
        object_embedding_dim=12,
        hidden_dim=12,
        action_embedding_dim=4,
    ).eval()


def test_history_expert_is_invariant_to_factual_suffix():
    history, suffix_a, suffix_b, factual_actions, counterfactual_actions = _inputs()
    model = _expert("history")
    y1 = model(history, suffix_a, factual_actions, counterfactual_actions)
    y2 = model(history, suffix_b, factual_actions, counterfactual_actions)
    assert torch.equal(y1, y2)


def test_full_expert_can_respond_to_factual_suffix():
    history, suffix_a, suffix_b, factual_actions, counterfactual_actions = _inputs()
    model = _expert("full")
    y1 = model(history, suffix_a, factual_actions, counterfactual_actions)
    y2 = model(history, suffix_b, factual_actions, counterfactual_actions)
    assert not torch.equal(y1, y2)


def test_experts_have_equal_parameter_count_and_matched_initialization():
    full = _expert("full")
    history = _expert("history")
    assert parameter_count(full) == parameter_count(history)
    for left, right in zip(full.parameters(), history.parameters(), strict=True):
        torch.testing.assert_close(left, right)


def test_oof_folds_are_deterministic_disjoint_and_complete():
    episode_seed = np.arange(100, 135, dtype=np.int64)
    split = np.array([0] * 25 + [1] * 5 + [2] * 5, dtype=np.int8)

    first = make_oof_folds(episode_seed, split, k=5, entropy=20260910)
    second = make_oof_folds(episode_seed, split, k=5, entropy=20260910)

    held_out = np.concatenate([fold.holdout_indices for fold in first])
    assert sorted(held_out.tolist()) == list(range(25))
    assert len(first) == 5
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.train_indices, right.train_indices)
        assert np.array_equal(left.holdout_indices, right.holdout_indices)
        assert not set(left.train_indices) & set(left.holdout_indices)
        assert set(split[left.train_indices]) == {0}


def test_oof_folds_reject_duplicate_episode_seeds():
    episode_seed = np.array([10, 10, 11, 12, 13], dtype=np.int64)
    split = np.zeros(5, dtype=np.int8)
    try:
        make_oof_folds(episode_seed, split, k=5, entropy=20260910)
    except ValueError as exc:
        assert "episode_seed must be unique" in str(exc)
    else:
        raise AssertionError("duplicate episode seeds were accepted")


def _mixture_inputs():
    generator = torch.Generator().manual_seed(91)
    full = torch.randn(2, 4, 3, 2, generator=generator)
    history = torch.randn(2, 4, 3, 2, generator=generator)
    features = torch.randn(2, 4, 3, 48, generator=generator)
    return full, history, features


def test_router_output_is_exact_convex_combination():
    full, history, features = _mixture_inputs()
    router = CEWRouter(input_dim=48, hidden_dim=64)
    output, trace = stage3r_forward(
        "cew_distilled_router",
        full,
        history,
        features,
        constant_w=0.6,
        router=router,
        return_trace=True,
    )
    assert torch.all((trace["weight"] >= 0) & (trace["weight"] <= 1))
    expected = trace["weight"] * full + (1.0 - trace["weight"]) * history
    torch.testing.assert_close(output, expected)


def test_direct_is_best_constant_mix_plus_free_residual():
    full, history, features = _mixture_inputs()
    hidden = match_direct_hidden_width(input_dim=48, router_hidden_dim=64)
    direct = SameInformationDirect(input_dim=48, hidden_dim=hidden)
    output, trace = stage3r_forward(
        "same_information_direct",
        full,
        history,
        features,
        constant_w=0.6,
        direct=direct,
        return_trace=True,
    )
    base = 0.6 * full + 0.4 * history
    torch.testing.assert_close(output, base + trace["direct_residual"])


def test_router_and_direct_trainable_capacity_match_within_one_percent():
    router = CEWRouter(input_dim=48, hidden_dim=64)
    hidden = match_direct_hidden_width(input_dim=48, router_hidden_dim=64)
    direct = SameInformationDirect(input_dim=48, hidden_dim=hidden)
    router_count = parameter_count(router, trainable_only=True)
    direct_count = parameter_count(direct, trainable_only=True)
    assert abs(router_count - direct_count) / router_count <= 0.01


def test_non_oracle_forward_rejects_truth_inputs():
    full, history, features = _mixture_inputs()
    router = CEWRouter(input_dim=48, hidden_dim=64)
    try:
        stage3r_forward(
            "cew_distilled_router",
            full,
            history,
            features,
            constant_w=0.6,
            router=router,
            target=torch.zeros_like(full),
        )
    except ValueError as exc:
        assert "truth inputs forbidden" in str(exc)
    else:
        raise AssertionError("non-oracle condition accepted target truth")


def test_router_features_align_cells_and_expose_response_slice():
    history, suffix, _, factual_actions, counterfactual_actions = _inputs()
    full, history_prediction, _ = _mixture_inputs()
    response_features = torch.randn(2, 4, 3, 6)
    regime_probs = torch.softmax(torch.randn(2, 4, 3, 4), dim=-1)
    features, response_only, metadata = build_router_features(
        history=history,
        factual_suffix=suffix,
        factual_actions=factual_actions,
        counterfactual_actions=counterfactual_actions,
        response_features=response_features,
        regime_probs=regime_probs,
        full_prediction=full,
        history_prediction=history_prediction,
    )
    assert features.shape[:3] == full.shape[:3]
    assert response_only.shape == features.shape
    assert metadata["input_dim"] == features.shape[-1]
    assert torch.equal(
        response_only[..., metadata["response_slice_start"] : metadata["response_slice_end"]],
        features[..., metadata["response_slice_start"] : metadata["response_slice_end"]],
    )
    outside = response_only.clone()
    outside[..., metadata["response_slice_start"] : metadata["response_slice_end"]] = 0
    assert torch.count_nonzero(outside) == 0
