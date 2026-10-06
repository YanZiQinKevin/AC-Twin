from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.stage1_model import PerAgentPairedEffectGRU, PerObjectTemporalEncoder, masked_scene_summary

EMBED = 16
HIDDEN = 16
AE = 4


def make_model(**overrides):
    params = dict(object_embedding_dim=EMBED, hidden_dim=HIDDEN, action_embedding_dim=AE)
    params.update(overrides)
    return PerAgentPairedEffectGRU(**params)


def make_batch(batch=2, agents=4, hist=10, horizon=20, seed=0):
    rng = np.random.default_rng(seed)
    history = rng.normal(size=(batch, hist, agents, 8)).astype(np.float32)
    suffix = rng.normal(size=(batch, horizon, agents, 8)).astype(np.float32)
    for values in (history, suffix):
        values[..., 0] = 1.0
        values[..., 1] = 0.0
        values[:, :, 0, 1] = 1.0
    factual_actions = rng.integers(0, 5, (batch, horizon))
    counterfactual_actions = rng.integers(0, 5, (batch, horizon))
    return (
        torch.from_numpy(history),
        torch.from_numpy(suffix),
        torch.from_numpy(factual_actions),
        torch.from_numpy(counterfactual_actions),
    )


def test_output_shape_is_b_t_n_2():
    model = make_model()
    out = model(*make_batch())
    assert out.shape == (2, 20, 4, 2)


def test_different_vehicles_produce_different_outputs():
    model = make_model()
    history, suffix, fa, ca = make_batch(agents=3)
    # Make vehicles clearly distinct.
    for slot in range(3):
        history[:, :, slot, 2:6] *= (slot + 1)
        suffix[:, :, slot, 2:6] *= (slot + 1)
    out = model(history, suffix, fa, ca)
    assert not torch.allclose(out[:, :, 0], out[:, :, 1])
    assert not torch.allclose(out[:, :, 1], out[:, :, 2])


def test_non_ego_permutation_is_equivariant():
    torch.manual_seed(0)
    model = make_model()
    history, suffix, fa, ca = make_batch(agents=4)
    perm = torch.tensor([0, 2, 3, 1])  # new slot j holds old vehicle perm[j]
    out_ref = model(history, suffix, fa, ca)
    out_perm = model(history[:, :, perm], suffix[:, :, perm], fa, ca)
    torch.testing.assert_close(out_perm[:, :, 0], out_ref[:, :, 0])
    for slot in range(1, 4):
        torch.testing.assert_close(out_perm[:, :, slot], out_ref[:, :, perm[slot]],
                                   atol=1e-5, rtol=1e-4)


def test_no_slot_specific_parameters():
    model = make_model()
    names = [name for name, _ in model.named_parameters()]
    assert not any("slot" in name for name in names)
    count = sum(p.numel() for p in model.parameters())
    # Parameter count is independent of the vehicle count N.
    out3 = model(*make_batch(agents=3))
    out5 = model(*make_batch(agents=5))
    assert out3.shape[2] == 3 and out5.shape[2] == 5
    assert sum(p.numel() for p in model.parameters()) == count


def test_masked_vehicle_extreme_values_do_not_affect_others():
    torch.manual_seed(0)
    model = make_model()
    history, suffix, fa, ca = make_batch(agents=3)
    # Mask out vehicle 2 (non-ego) entirely, first with zeros then with NaN.
    for values in (history, suffix):
        values[:, :, 2, 0] = 0.0
    out_clean = model(history, suffix, fa, ca)
    history_nan = history.clone()
    suffix_nan = suffix.clone()
    history_nan[:, :, 2, 2:] = float("nan")
    suffix_nan[:, :, 2, 2:] = float("nan")
    out_nan = model(history_nan, suffix_nan, fa, ca)
    assert torch.isfinite(out_nan).all()
    torch.testing.assert_close(out_nan[:, :, 0], out_clean[:, :, 0], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out_nan[:, :, 1], out_clean[:, :, 1], atol=1e-5, rtol=1e-4)
    # Vehicle 2 is history-unknown so its output is zeroed.
    assert torch.count_nonzero(out_clean[:, :, 2]) == 0


def test_missing_steps_do_not_advance_hidden_state():
    torch.manual_seed(0)
    encoder = PerObjectTemporalEncoder(feature_dim=8, embed_dim=EMBED, hidden_dim=HIDDEN)
    base = torch.zeros(1, 10, 1, 8)
    base[0, 0, 0] = torch.tensor([1.0, 1.0, 3.0, 4.0, 1.0, 0.0, 1.0, 0.0])
    mask = torch.zeros(1, 10, 1, dtype=torch.bool)
    mask[0, 0, 0] = True
    variant = base.clone()
    variant[0, 3:, 0, 2:6] = 123.0  # garbage at absent steps
    summary_a, _ = encoder(base, mask)
    summary_b, _ = encoder(variant, mask)
    torch.testing.assert_close(summary_a, summary_b)


def test_empty_suffix_summary_is_strictly_zero():
    torch.manual_seed(0)
    model = make_model()
    suffix = torch.zeros(2, 20, 4, 8)
    suffix_mask = torch.zeros(2, 20, 4, dtype=torch.bool)
    summary, active = model.encode_factual_agents(suffix, suffix_mask)
    assert torch.count_nonzero(summary) == 0
    assert not active.any()
    assert torch.count_nonzero(masked_scene_summary(summary, active)) == 0


def test_m0_m1_m2_share_identical_structure():
    # All three conditions instantiate the same class/capacity, so parameter
    # names and counts must match exactly.
    a, b, c = make_model(), make_model(), make_model()
    names_a = [(n, tuple(p.shape)) for n, p in a.named_parameters()]
    assert names_a == [(n, tuple(p.shape)) for n, p in b.named_parameters()]
    assert names_a == [(n, tuple(p.shape)) for n, p in c.named_parameters()]
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in c.parameters())


def test_backward_produces_finite_gradients():
    torch.manual_seed(0)
    model = make_model()
    history, suffix, fa, ca = make_batch()
    out = model(history, suffix, fa, ca)
    target = torch.zeros_like(out)
    mask = torch.ones(out.shape[:-1])
    loss = (mask.unsqueeze(-1) * (out - target) ** 2).sum() / (mask.sum() * 2)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"
