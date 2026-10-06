from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin.stage3b_plus_routing import (  # noqa: E402
    HorizonStateGate,
    Stage3BPlusRouter,
    assert_oracle_route_only,
    audit_direct_router_fairness,
    nonmonotonic_shape_loss,
)


def _inputs(batch: int = 2, horizon: int = 6, agents: int = 4, response_dim: int = 8):
    generator = torch.Generator().manual_seed(42)
    history = torch.randn(batch, 5, agents, 8, generator=generator)
    suffix = torch.randn(batch, 5, agents, 8, generator=generator)
    history[..., 0] = 1
    suffix[..., 0] = 1
    return {
        "history": history,
        "factual_suffix": suffix,
        "factual_actions": torch.randint(0, 5, (batch, horizon), generator=generator),
        "counterfactual_actions": torch.randint(0, 5, (batch, horizon), generator=generator),
        "response_features": torch.randn(batch, horizon, agents, response_dim, generator=generator),
        "regime_probs": torch.softmax(torch.randn(batch, horizon, agents, 4, generator=generator), -1),
    }


def _build(condition: str, architecture: str = "state_full"):
    torch.manual_seed(7)
    return Stage3BPlusRouter(
        condition=condition,
        architecture=architecture,
        feature_dim=8,
        response_feature_dim=8,
        hidden_dim=16,
        object_embedding_dim=16,
        horizon=6,
    )


def test_horizon_state_gate_is_exact_probability_weighting():
    gate = HorizonStateGate(horizon=2, n_regimes=4, n_channels=1)
    with torch.no_grad():
        gate.logits.copy_(torch.logit(torch.tensor([[[0.1, 0.2, 0.3, 0.4]], [[0.8, 0.6, 0.4, 0.2]]])))
    probs = torch.tensor([[[[0.0, 1.0, 0.0, 0.0]], [[0.5, 0.0, 0.0, 0.5]]]])
    weights = gate(probs)
    torch.testing.assert_close(weights, torch.tensor([[[[0.2]], [[0.5]]]]))
    assert bool(((weights >= 0) & (weights <= 1)).all())


@pytest.mark.parametrize("architecture,n_channels", [("state_full", 1), ("state_two_channel", 2)])
def test_route_formula_and_output_shape(architecture: str, n_channels: int):
    model = _build("learned_router", architecture).eval()
    output, trace = model(**_inputs(), return_trace=True)
    assert output.shape == (2, 6, 4, 2)
    assert trace["route_gate"].shape == (2, 6, 4, n_channels)
    if architecture == "state_full":
        torch.testing.assert_close(trace["z_route"], trace["route_gate"] * trace["h_last_seq"])
    else:
        expected = trace["route_gate"][..., :1] * trace["z_base_seq"] + trace["route_gate"][..., 1:] * trace["z_dyn_seq"]
        torch.testing.assert_close(trace["z_route"], expected)


def test_oracle_truth_changes_only_multiplicative_route():
    model = _build("route_only_oracle").eval()
    with torch.no_grad():
        model.state_gates.logits[..., 0].fill_(-2.0)
        model.state_gates.logits[..., 3].fill_(2.0)
    inputs = _inputs()
    labels_a = torch.zeros(2, 6, 4, dtype=torch.long)
    labels_b = torch.full((2, 6, 4), 3, dtype=torch.long)
    _, trace_a = model(**inputs, oracle_labels=labels_a, return_trace=True)
    _, trace_b = model(**inputs, oracle_labels=labels_b, return_trace=True)
    assert_oracle_route_only(trace_a)
    assert_oracle_route_only(trace_b)
    torch.testing.assert_close(trace_a["decoder_response_features"], trace_b["decoder_response_features"])
    torch.testing.assert_close(trace_a["decoder_regime_probs"], trace_b["decoder_regime_probs"])
    torch.testing.assert_close(trace_a["decoder_gate"], trace_b["decoder_gate"])
    assert not torch.equal(trace_a["z_route"], trace_b["z_route"])


def test_direct_and_learned_are_identical_when_gate_is_one():
    learned = _build("learned_router").eval()
    direct = _build("same_information_direct").eval()
    direct.load_state_dict(learned.state_dict())
    learned.compute_state_weights = lambda probs, labels=None: torch.ones(probs.shape[:-1] + (1,))
    direct.compute_state_weights = learned.compute_state_weights
    inputs = _inputs()
    out_learned = learned(**inputs)
    out_direct = direct(**inputs)
    torch.testing.assert_close(out_learned, out_direct, rtol=1e-5, atol=1e-5)
    report = audit_direct_router_fairness({"direct": direct, "learned": learned})
    assert report["within_tolerance"] and report["same_state_keys"]


def test_shape_loss_enforces_registered_nonmonotonic_rankings_and_skips_missing():
    labels = torch.tensor([0, 1, 2, 3]).view(1, 1, 4)
    valid = torch.ones_like(labels, dtype=torch.bool)
    good = torch.tensor([0.9, 0.3, 0.2, 0.8]).view(1, 1, 4, 1)
    bad = torch.tensor([0.2, 0.8, 0.7, 0.1]).view(1, 1, 4, 1)
    assert nonmonotonic_shape_loss(good, labels, valid, role="train").item() == pytest.approx(0.0)
    assert nonmonotonic_shape_loss(bad, labels, valid, role="train").item() > 0
    missing_mid = labels != 2
    loss = nonmonotonic_shape_loss(good, labels, missing_mid, role="train")
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="train"):
        nonmonotonic_shape_loss(good, labels, valid, role="validation")
