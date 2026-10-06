from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.model import (  # noqa: E402
    DynamicGraphTemporalEncoder,
    IdentityTemporalEncoder,
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


def _identity_sequence_batch(batch: int = 2, steps: int = 10) -> torch.Tensor:
    """Create non-ego tracks with deliberately distinct temporal signatures."""
    x = torch.zeros(batch, steps, 5, 8)
    x[..., 0] = 1.0
    time = torch.arange(steps, dtype=torch.float32).view(1, steps, 1)
    tracks = torch.tensor([1.0, 3.0, 7.0, 11.0]).view(1, 1, 4)
    x[:, :, 0, 1] = 100.0
    x[:, :, 1:, 1] = tracks + time
    x[:, :, 1:, 2] = tracks * (time + 1.0)
    x[:, :, 1:, 3] = tracks.square() - time
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


def test_identity_temporal_encoder_uses_cross_time_identity_continuity() -> None:
    x = _identity_sequence_batch()
    whole_sequence = x[:, :, [0, 3, 1, 4, 2]]
    per_frame = x.clone()
    per_frame[:, 1::2] = per_frame[:, 1::2, [0, 3, 1, 4, 2]]
    torch.manual_seed(20260905)
    encoder = IdentityTemporalEncoder(8, 16).eval()

    with torch.no_grad():
        original = encoder(x)
        same_tracks_renamed = encoder(whole_sequence)
        broken_tracks = encoder(per_frame)

    torch.testing.assert_close(original, same_tracks_renamed)
    assert not torch.allclose(original, broken_tracks)


def test_identity_temporal_encoder_ignores_masked_object_features() -> None:
    x = _identity_sequence_batch()
    x[:, :, 4, 0] = 0.0
    changed = x.clone()
    changed[:, :, 4, 1:] = 10000.0
    torch.manual_seed(20260905)
    encoder = IdentityTemporalEncoder(8, 16).eval()

    with torch.no_grad():
        torch.testing.assert_close(encoder(x), encoder(changed))


def test_identity_temporal_encoder_skips_missing_time_steps() -> None:
    x = _identity_sequence_batch(batch=3, steps=8)
    missing_steps = ([0, 1], [3, 4], [6, 7])
    compact_steps = (
        [2, 3, 4, 5, 6, 7],
        [0, 1, 2, 5, 6, 7],
        [0, 1, 2, 3, 4, 5],
    )
    for batch_index, missing in enumerate(missing_steps):
        x[batch_index, missing, :, 0] = 0.0

    torch.manual_seed(20260905)
    encoder = IdentityTemporalEncoder(8, 16).eval()
    with torch.no_grad():
        for batch_index, kept in enumerate(compact_steps):
            full_sequence = encoder(x[batch_index : batch_index + 1])
            compact_sequence = encoder(x[batch_index : batch_index + 1, kept])

            torch.testing.assert_close(full_sequence, compact_sequence)


def test_identity_temporal_encoder_ignores_extreme_nonfinite_missing_features() -> None:
    x = _identity_sequence_batch(batch=3, steps=8)
    for batch_index, missing in enumerate(([0, 1], [3, 4], [6, 7])):
        x[batch_index, missing, :, 0] = 0.0
    changed = x.clone()
    missing = changed[..., 0] == 0
    changed[..., 1][missing] = 1e30
    changed[..., 2][missing] = float("nan")
    changed[..., 3][missing] = float("inf")
    changed[..., 4][missing] = float("-inf")

    torch.manual_seed(20260905)
    encoder = IdentityTemporalEncoder(8, 16).eval()
    with torch.no_grad():
        torch.testing.assert_close(encoder(x), encoder(changed))


def test_identity_temporal_encoder_skips_only_the_missing_vehicle() -> None:
    source = _identity_sequence_batch(batch=1, steps=8)
    target_vehicle = 4
    missing_and_kept = (
        ([0, 1], [2, 3, 4, 5, 6, 7]),
        ([3, 4], [0, 1, 2, 5, 6, 7]),
        ([6, 7], [0, 1, 2, 3, 4, 5]),
    )
    torch.manual_seed(20260905)
    encoder = IdentityTemporalEncoder(8, 16).eval()

    with torch.no_grad():
        for missing, kept in missing_and_kept:
            original = source.clone()
            original[:, missing, target_vehicle, 0] = 0.0

            reference = source.clone()
            reference[:, : len(kept), target_vehicle] = source[:, kept, target_vehicle]
            reference[:, len(kept) :, target_vehicle, 0] = 0.0

            changed = original.clone()
            changed[:, missing, target_vehicle, 1] = 1e30
            changed[:, missing, target_vehicle, 2] = float("nan")
            changed[:, missing, target_vehicle, 3] = float("inf")
            changed[:, missing, target_vehicle, 4] = float("-inf")

            output = encoder(torch.cat([original, reference, changed]))
            torch.testing.assert_close(output[0], output[1])
            torch.testing.assert_close(output[0], output[2])


def test_identity_temporal_encoder_keeps_ego_on_a_separate_branch() -> None:
    x = _identity_sequence_batch()
    ego_swapped_with_non_ego = x[:, :, [1, 0, 2, 3, 4]]
    torch.manual_seed(20260905)
    encoder = IdentityTemporalEncoder(8, 16).eval()

    with torch.no_grad():
        original = encoder(x)
        changed = encoder(ego_swapped_with_non_ego)

    assert not torch.allclose(original, changed)


def test_object_effect_model_supports_identity_temporal_encoder() -> None:
    history = _identity_sequence_batch()
    suffix = _identity_sequence_batch(steps=20)
    factual_actions = torch.zeros(2, 20, dtype=torch.long)
    counterfactual_actions = torch.ones(2, 20, dtype=torch.long)
    model = PairedObjectEffectGRU(8, 5, hidden_dim=16, encoder_type="identity_temporal")

    output = model(history, suffix, factual_actions, counterfactual_actions)

    assert output.shape == (2, 20, 2)
    assert torch.isfinite(output).all()


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
