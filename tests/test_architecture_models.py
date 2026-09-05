from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.model import (  # noqa: E402
    DynamicGraphTemporalEncoder,
    ObjectTemporalEncoder,
    PairedEffectGRU,
    PairedObjectEffectGRU,
    StaticGraphTemporalEncoder,
)


def _object_batch(batch: int = 4, steps: int = 10) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260904)
    x = torch.randn(batch, steps, 5, 8, generator=generator)
    x[..., 0] = 1.0
    x[..., 1] = 0.0
    x[:, :, 0, 1] = 1.0
    return x


@pytest.mark.parametrize("encoder_type", ["set", "static_graph", "dynamic_graph"])
def test_object_effect_models_return_finite_effect_trajectories(encoder_type: str) -> None:
    history = _object_batch(steps=10)
    suffix = _object_batch(steps=20)
    factual_actions = torch.zeros(4, 20, dtype=torch.long)
    counterfactual_actions = torch.ones(4, 20, dtype=torch.long)
    model = PairedObjectEffectGRU(8, 5, hidden_dim=16, encoder_type=encoder_type)

    output = model(history, suffix, factual_actions, counterfactual_actions)

    assert output.shape == (4, 20, 2)
    assert torch.isfinite(output).all()


def test_row_sorted_effect_model_returns_finite_effect_trajectories() -> None:
    generator = torch.Generator().manual_seed(20260904)
    history = torch.randn(4, 10, 26, generator=generator)
    suffix = torch.randn(4, 20, 26, generator=generator)
    factual_actions = torch.zeros(4, 20, dtype=torch.long)
    counterfactual_actions = torch.ones(4, 20, dtype=torch.long)
    model = PairedEffectGRU(26, 5, hidden_dim=16)

    output = model(history, suffix, factual_actions, counterfactual_actions)

    assert output.shape == (4, 20, 2)
    assert torch.isfinite(output).all()


def test_set_encoder_is_invariant_to_non_ego_row_permutation() -> None:
    x = _object_batch(batch=2, steps=10)
    permuted = x[:, :, [0, 3, 1, 4, 2]]
    encoder = ObjectTemporalEncoder(8, 16).eval()

    with torch.no_grad():
        original = encoder(x)
        changed = encoder(permuted)

    torch.testing.assert_close(original, changed)


@pytest.mark.parametrize(
    "encoder_cls", [StaticGraphTemporalEncoder, DynamicGraphTemporalEncoder]
)
def test_graph_encoders_ignore_features_of_masked_objects(encoder_cls: type) -> None:
    x = _object_batch(batch=2, steps=10)
    x[:, :, 4, 0] = 0.0
    changed = x.clone()
    changed[:, :, 4, 1:] = 10000.0
    encoder = encoder_cls(8, 16).eval()

    with torch.no_grad():
        original_output = encoder(x)
        changed_output = encoder(changed)

    torch.testing.assert_close(original_output, changed_output)


def test_unknown_object_encoder_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown encoder_type"):
        PairedObjectEffectGRU(8, 5, encoder_type="typo")


def test_dynamic_graph_restores_metric_distance_before_radius_gating() -> None:
    encoder = DynamicGraphTemporalEncoder(
        8, 16, radius=60.0, position_scale_xy=(50.0, 20.0)
    )
    normalized_relative_xy = torch.tensor([2.0, 1.0])

    metric_distance = encoder.metric_distance(normalized_relative_xy)

    torch.testing.assert_close(metric_distance, torch.tensor(101.98039))


@pytest.mark.parametrize(
    ("model_factory", "expected"),
    [
        (lambda: PairedEffectGRU(26, 5, 54, 8), 64734),
        (lambda: PairedObjectEffectGRU(8, 5, 48, 8, "set"), 64266),
        (lambda: PairedObjectEffectGRU(8, 5, 42, 8, "dynamic_graph"), 64302),
        (lambda: PairedObjectEffectGRU(8, 5, 42, 8, "static_graph"), 64302),
        (lambda: PairedEffectGRU(26, 5, 62, 8), 83246),
        (lambda: PairedObjectEffectGRU(8, 5, 55, 8, "set"), 83642),
        (lambda: PairedObjectEffectGRU(8, 5, 48, 8, "dynamic_graph"), 83274),
        (lambda: PairedObjectEffectGRU(8, 5, 48, 8, "static_graph"), 83274),
    ],
)
def test_capacity_tier_parameter_counts(model_factory, expected: int) -> None:
    model = model_factory()
    assert sum(parameter.numel() for parameter in model.parameters()) == expected
