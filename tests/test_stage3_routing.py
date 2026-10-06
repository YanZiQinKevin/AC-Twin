from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage3_routing as s3route  # noqa: E402
from ac_twin.stage3_routing import Condition  # noqa: E402


def make_inputs(batch=2, agents=5, horizon=20, hist=10, feat=8, d_r=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    history = torch.randn(batch, hist, agents, feat, generator=g)
    factual = torch.randn(batch, 10, agents, feat, generator=g)
    factual_actions = torch.randint(0, 5, (batch, horizon), generator=g)
    counterfactual_actions = torch.randint(0, 5, (batch, horizon), generator=g)
    response_features = torch.randn(batch, horizon, agents, d_r, generator=g)
    regime_probs = torch.softmax(torch.randn(batch, horizon, agents, 4, generator=g), dim=-1)
    for t in (history, factual):
        t[..., 0] = 1.0
        t[..., 1] = 0.0
        t[:, :, 0, 1] = 1.0
    return {
        "history": history,
        "factual_suffix": factual,
        "factual_actions": factual_actions,
        "counterfactual_actions": counterfactual_actions,
        "response_features": response_features,
        "regime_probs": regime_probs,
    }


def build(condition, d_r=16, seed=1234):
    torch.manual_seed(seed)
    return s3route.Stage3Router(
        condition, hidden_dim=32, object_embedding_dim=32,
        response_feature_dim=d_r, gate_hidden_dim=16, seed=seed,
    )


# --- dual-channel decomposition -------------------------------------------------


def test_dual_channel_exact_reconstruction() -> None:
    encoder = s3route.PerObjectTemporalSequenceEncoder(feature_dim=8, embed_dim=16, hidden_dim=24)
    x = torch.randn(2, 10, 5, 8)
    x[..., 0] = 1.0
    seq, valid = encoder(x)
    z_base, z_dyn, h_last = s3route.dual_channel_decompose(seq, valid)
    # z_base + z_dyn reconstructs the terminal valid hidden state exactly.
    torch.testing.assert_close(z_base + z_dyn, h_last, rtol=1e-5, atol=1e-5)


def test_dual_channel_masked_mean_excludes_padding() -> None:
    encoder = s3route.PerObjectTemporalSequenceEncoder(feature_dim=8, embed_dim=16, hidden_dim=24)
    x = torch.randn(1, 10, 5, 8)
    x[..., 0] = 1.0
    # mask out the last 4 suffix steps for all vehicles
    mask = torch.ones(1, 10, 5, dtype=torch.bool)
    mask[:, 6:, :] = False
    seq, valid = encoder(x, mask)
    assert not valid[:, 6:, :].any()
    z_base, z_dyn, h_last = s3route.dual_channel_decompose(seq, valid)
    torch.testing.assert_close(z_base + z_dyn, h_last, rtol=1e-5, atol=1e-5)


# --- Direct vs Dual: only the multiplicative difference -------------------------


def _force_unit_gate(model):
    def unit_gate(q, oracle_gate=None):
        return torch.ones(q.shape[:-1] + (1,), device=q.device)
    model.compute_gate = unit_gate
    return model


def test_direct_and_dual_identical_topology_and_params() -> None:
    dual = build(Condition.DUAL_ROUTER)
    direct = build(Condition.DIRECT)
    # Same architecture -> state dicts interchangeable.
    direct.load_state_dict(dual.state_dict())
    assert s3route.count_parameters(dual) == s3route.count_parameters(direct)


def test_dual_with_unit_gate_equals_direct() -> None:
    """When w=1, z_dual = z_base + z_dyn = z_direct: the ONLY difference is the product."""
    inputs = make_inputs()
    dual = _force_unit_gate(build(Condition.DUAL_ROUTER)).eval()
    direct = _force_unit_gate(build(Condition.DIRECT)).eval()
    direct.load_state_dict(dual.state_dict())
    with torch.no_grad():
        out_dual = dual(**inputs)
        out_direct = direct(**inputs)
    torch.testing.assert_close(out_dual, out_direct, rtol=1e-4, atol=1e-4)


def test_dual_gate_is_nonmonotonic_capable() -> None:
    """No w=1-g or monotonic hard constraint: the gate is a free sigmoid MLP."""
    model = build(Condition.DUAL_ROUTER)
    assert isinstance(model.gate, s3route.GateHead)
    q = torch.randn(2, 20, 5, model.q_dim)
    w = model.gate(q)
    assert w.shape == (2, 20, 5, 1)
    assert ((w >= 0) & (w <= 1)).all()


def test_output_shape_and_finite_all_conditions() -> None:
    inputs = make_inputs()
    for condition in Condition:
        model = build(condition).eval()
        kwargs = dict(inputs)
        if condition == Condition.ORACLE_REFERENCE:
            kwargs["oracle_gate"] = torch.rand(2, 20, 5, 1)
        with torch.no_grad():
            out = model(**kwargs)
        assert out.shape == (2, 20, 5, 2), condition
        assert torch.isfinite(out).all(), condition


# --- oracle leakage ---------------------------------------------------------------


@pytest.mark.parametrize("condition", sorted(s3route.LEARNED_CONDITIONS, key=str))
def test_learned_conditions_reject_oracle_input(condition) -> None:
    inputs = make_inputs()
    model = build(condition)
    with pytest.raises(ValueError, match="oracle"):
        model(**inputs, oracle_gate=torch.rand(2, 20, 5, 1))


def test_oracle_reference_requires_oracle_input() -> None:
    inputs = make_inputs()
    model = build(Condition.ORACLE_REFERENCE)
    with pytest.raises(ValueError, match="oracle"):
        model(**inputs)


def test_oracle_reference_not_in_learned_set() -> None:
    assert Condition.ORACLE_REFERENCE not in s3route.LEARNED_CONDITIONS


# --- seed pairing -----------------------------------------------------------------


def test_seed_pairing_frozen_one_to_one() -> None:
    s3route.validate_seed_pairing()
    assert s3route.PREDICTOR_SEEDS == (101, 202, 303, 404, 505)
    assert s3route.ROUTER_SEEDS == (1101, 1202, 1303, 1404, 1505)
    assert s3route.SEED_PAIRS == (
        (101, 1101), (202, 1202), (303, 1303), (404, 1404), (505, 1505),
    )


def test_seed_pairing_rejects_best_seed_and_cross() -> None:
    # reordered = best-seed selection
    with pytest.raises(ValueError):
        s3route.validate_seed_pairing(
            ((505, 1101), (202, 1202), (303, 1303), (404, 1404), (101, 1505))
        )
    # duplicated router = 5x5 cross leakage
    with pytest.raises(ValueError):
        s3route.validate_seed_pairing(
            ((101, 1101), (202, 1101), (303, 1303), (404, 1404), (505, 1505))
        )
    # wrong count
    with pytest.raises(ValueError):
        s3route.validate_seed_pairing(((101, 1101), (202, 1202)))


# --- capacity audit -----------------------------------------------------------------


def test_capacity_audit_within_one_percent_no_dummy() -> None:
    models = {
        Condition.UNIFORM_SUFFIX: build(Condition.UNIFORM_SUFFIX),
        Condition.DIRECT: build(Condition.DIRECT),
        Condition.DUAL_ROUTER: build(Condition.DUAL_ROUTER),
    }
    report = s3route.audit_capacity(models)
    assert report["within_tolerance"]
    assert report["max_relative_difference"] <= 0.01
    # identical module set -> exactly equal, so no dummy padding was needed
    totals = [report["counts"][c.value]["total"] for c in s3route.CAPACITY_AUDIT_CONDITIONS]
    assert len(set(totals)) == 1


# --- cross-fit folds ------------------------------------------------------------------


def test_crossfit_folds_no_episode_leakage_and_deterministic() -> None:
    n = 40
    rng = np.random.default_rng(0)
    cf_action = rng.integers(0, 5, n)
    cf_duration = rng.choice([1, 3], n)
    episode_seed = np.arange(n) + 5000
    folds = s3route.assign_crossfit_folds(cf_action, cf_duration, episode_seed)
    assert set(np.unique(folds).tolist()) <= {0, 1}
    # each episode in exactly one fold
    assert (np.bincount(folds, minlength=2).sum() == n)
    # deterministic
    folds2 = s3route.assign_crossfit_folds(cf_action, cf_duration, episode_seed)
    assert (folds == folds2).all()
    # roughly balanced within each stratum
    for a in np.unique(cf_action):
        for d in np.unique(cf_duration):
            sel = (cf_action == a) & (cf_duration == d)
            if sel.sum() >= 2:
                counts = np.bincount(folds[sel], minlength=2)
                assert abs(counts[0] - counts[1]) <= 1


def test_crossfit_disjoint_episode_sets() -> None:
    n = 20
    cf_action = np.zeros(n, dtype=int)
    cf_duration = np.ones(n, dtype=int)
    episode_seed = np.arange(n)
    folds = s3route.assign_crossfit_folds(cf_action, cf_duration, episode_seed)
    fold0 = set(episode_seed[folds == 0].tolist())
    fold1 = set(episode_seed[folds == 1].tolist())
    assert fold0.isdisjoint(fold1)
