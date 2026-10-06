from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin.stage3r_teacher import (  # noqa: E402
    aggregate_episode_vehicle,
    build_consensus_teacher,
    counterfactual_evidence_advantage,
    optimal_convex_weight,
)


def test_optimal_convex_weight_hits_projection_and_clips():
    history = torch.tensor([[[[0.0, 0.0]]]])
    full = torch.tensor([[[[2.0, 0.0]]]])
    target_mid = torch.tensor([[[[1.0, 0.0]]]])
    target_after = torch.tensor([[[[4.0, 0.0]]]])
    target_before = torch.tensor([[[[-1.0, 0.0]]]])

    torch.testing.assert_close(
        optimal_convex_weight(full, history, target_mid),
        torch.tensor([[[0.5]]]),
    )
    assert optimal_convex_weight(full, history, target_after).item() == 1.0
    assert optimal_convex_weight(full, history, target_before).item() == 0.0


def test_optimal_convex_weight_is_finite_for_equal_experts():
    prediction = torch.tensor([[[[0.4, -0.2]]]])
    target = torch.tensor([[[[1.0, 1.0]]]])
    weight = optimal_convex_weight(prediction, prediction, target)
    assert torch.isfinite(weight).all()
    assert weight.item() == 0.0


def test_cea_positive_when_full_is_better():
    target = torch.tensor([[[[1.0, 0.0]]]])
    full = torch.tensor([[[[0.9, 0.0]]]])
    history = torch.tensor([[[[0.0, 0.0]]]])
    assert counterfactual_evidence_advantage(full, history, target).item() > 0


def test_consensus_uses_median_and_separates_rank_from_distill_reliability():
    # Shape: seed, episode, time, agent, xy.
    history = torch.zeros(5, 1, 2, 1, 2)
    full = torch.zeros_like(history)
    full[:, :, :, :, 0] = torch.tensor([0.8, 0.9, 1.0, 1.1, -0.2]).view(5, 1, 1, 1)
    target = torch.ones(1, 2, 1, 2)
    target[..., 1] = 0.0
    valid = torch.ones(1, 2, 1, dtype=torch.bool)

    out = build_consensus_teacher(full, history, target, valid)

    torch.testing.assert_close(out["cew"], out["cew_by_seed"].median(dim=0).values)
    assert out["rank_reliable"][valid].all()
    assert out["distill_weight"].shape == valid.shape
    assert out["winsor_bounds"].shape == (2,)


def test_near_equivalent_experts_keep_effect_but_zero_distill_weight():
    history = torch.zeros(5, 1, 2, 1, 2)
    full = history.clone()
    target = torch.ones(1, 2, 1, 2)
    valid = torch.ones(1, 2, 1, dtype=torch.bool)

    out = build_consensus_teacher(full, history, target, valid)

    assert out["distill_weight"][valid].eq(0).all()
    assert out["valid_for_effect"][valid].all()


def test_aggregate_episode_vehicle_averages_only_valid_times():
    values = torch.tensor([[[1.0, 10.0], [3.0, 20.0], [100.0, 30.0]]])
    valid = torch.tensor([[[True, True], [True, False], [False, False]]])

    mean, agent_valid = aggregate_episode_vehicle(values, valid)

    torch.testing.assert_close(mean, torch.tensor([[2.0, 10.0]]))
    assert torch.equal(agent_valid, torch.tensor([[True, True]]))
