from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin.stage3r_statistics import (  # noqa: E402
    evaluate_phase0,
    evaluate_phase1,
    estimate_joint_power,
    find_best_constant_weight,
    masked_episode_ade,
    paired_relative_bootstrap,
    pairwise_rank_loss,
    weighted_huber,
)
from ac_twin.stage3r_teacher import optimal_convex_weight  # noqa: E402


def _predictions():
    history = torch.zeros(4, 2, 1, 2)
    full = torch.zeros_like(history)
    full[..., 0] = 2.0
    target = torch.zeros_like(history)
    target[:2, ..., 0] = 0.5
    target[2:, ..., 0] = 1.5
    valid = torch.ones(4, 2, 1, dtype=torch.bool)
    return full, history, target, valid


def test_best_constant_is_selected_from_frozen_grid():
    full, history, target, valid = _predictions()
    weight, scores = find_best_constant_weight(
        full, history, target, valid, grid=np.arange(0.0, 1.0001, 0.05)
    )
    assert weight in scores
    assert weight == min(scores, key=lambda key: (scores[key], key))
    # ADE is flat for weights 0.25..0.75 here; the frozen tie-break picks lower w.
    assert weight == 0.25


def test_soft_oracle_beats_best_constant_on_same_cells():
    full, history, target, valid = _predictions()
    weight, _ = find_best_constant_weight(full, history, target, valid)
    constant = weight * full + (1.0 - weight) * history
    oracle_weight = optimal_convex_weight(full, history, target)
    oracle = oracle_weight[..., None] * full + (1.0 - oracle_weight[..., None]) * history
    assert masked_episode_ade(oracle, target, valid).mean() < masked_episode_ade(
        constant, target, valid
    ).mean()


def test_paired_bootstrap_preserves_direction():
    reference = np.linspace(1.0, 2.0, 60)
    candidate = reference * 0.8
    result = paired_relative_bootstrap(reference, candidate, seed=7, n_resamples=500)
    assert np.isclose(result["point_relative"], 0.2)
    assert result["ci80_lower"] > 0


def test_phase0_requires_all_registered_conditions():
    reference = np.tile(np.linspace(1.0, 2.0, 80), (5, 1))
    oracle = reference * 0.75
    result = evaluate_phase0(reference, oracle, seed=11, n_resamples=500)
    assert result["status"] == "STAGE3R_SOFT_HEADROOM_CONFIRMED"
    assert result["positive_pairs"] == 5


def test_weighted_huber_ignores_zero_weight_cells():
    prediction = torch.tensor([0.0, 10.0])
    target = torch.tensor([1.0, -10.0])
    weight = torch.tensor([1.0, 0.0])
    loss = weighted_huber(prediction, target, weight)
    torch.testing.assert_close(loss, torch.tensor(0.5))


def test_pairwise_rank_loss_rewards_correct_vehicle_order():
    score = torch.tensor([[0.1, 0.9, 0.4]])
    target = torch.tensor([[-1.0, 2.0, 0.5]])
    valid = torch.ones_like(score, dtype=torch.bool)
    correct = pairwise_rank_loss(score, target, valid)
    reversed_score = -score
    incorrect = pairwise_rank_loss(reversed_score, target, valid)
    assert correct < incorrect


def test_phase1_requires_predictability_and_router_improvement():
    episodes, horizon, agents = 60, 2, 2
    teacher = np.linspace(0.0, 1.0, episodes * horizon * agents).reshape(
        episodes, horizon, agents
    )
    predicted = np.stack([teacher for _ in range(5)])
    cea = teacher - 0.5
    valid = np.ones_like(teacher, dtype=bool)
    reference = np.ones((5, episodes), dtype=np.float64)
    router = reference * 0.9
    report = evaluate_phase1(
        predicted,
        teacher,
        cea,
        valid,
        reference,
        router,
        seed=17,
        n_resamples=200,
    )
    assert report["status"] == "STAGE3R_VALUE_WEIGHT_PREDICTABLE"
    assert report["positive_pairs"] == 5


def test_joint_power_selects_smallest_passing_candidate():
    generator = np.random.default_rng(5)
    reference = np.stack(
        [1.0 + generator.normal(0.0, 0.02, size=(5, 300)) for _ in range(3)]
    )
    candidate = reference * 0.95 + generator.normal(0.0, 0.005, size=reference.shape)
    report = estimate_joint_power(
        reference,
        candidate,
        n_candidates=[100, 200, 400],
        mdes=0.05,
        target_power=0.8,
        seed=19,
        simulations=300,
    )
    assert report["status"] == "STAGE3R_RSCREEN_POWER_FEASIBLE"
    assert report["selected_episode_count"] in [100, 200, 400]
