"""Blocking tests for Stage 4C Plus relative-comparison features (plan sections 3, 14)."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4c_objective as s4co  # noqa: E402
from ac_twin import stage4cp_features as s4cf  # noqa: E402
from ac_twin import stage4cp_protocol as s4cp  # noqa: E402


def _synthetic(G=4, S=2, T=4, seed=0):
    rng = np.random.default_rng(seed)
    N = s4cp.OBJECT_CAPACITY
    A = s4cp.N_ACTIONS
    e2e = rng.normal(size=(S, G, A, T, N, 2)).astype(np.float64) * 0.1
    factual_xy = np.cumsum(rng.normal(size=(G, T, N, 2)) * 0.1, axis=1) + 50.0
    history_last_xy = factual_xy[:, 0, :, :][:, 0, :]  # (G,2) ego
    inference_mask = np.zeros((G, T, N), dtype=bool)
    inference_mask[:, :, 1:3] = True  # two non-ego agents always legal
    inference_mask[..., 0] = False
    action_risks = {k: rng.uniform(0, 1, size=(G, A)) for k in s4cp.RISK_CLASSES}
    distance = np.full((G, N), np.nan)
    relspeed = np.full((G, N), np.nan)
    distance[:, 1:] = rng.uniform(1, 30, size=(G, N - 1))
    relspeed[:, 1:] = rng.uniform(0, 5, size=(G, N - 1))
    geometry = {"distance_to_ego": distance, "relative_speed": relspeed,
                "visible_count": np.full(G, float(N - 1))}
    t_star = rng.integers(10, 16, size=G)
    group_seed = np.arange(1000, 1000 + G)
    return dict(e2e=e2e, factual_xy=factual_xy, history_last_xy=history_last_xy,
                inference_mask=inference_mask, action_risks=action_risks,
                geometry=geometry, t_star=t_star, group_seed=group_seed)


def test_schema_dims():
    assert s4cf.TOKEN_DIM == 34
    assert s4cf.N_TOKEN_CONT == 20
    assert len(s4cf.TOKEN_CAT_NAMES) == 14
    assert s4cf.GLOBAL_DIM == 9
    schema = s4cf.feature_schema()
    assert schema["token_dim"] == 34 and schema["global_dim"] == 9


def test_no_truth_side_parameters():
    """Feature builders must not accept counterfactual truth / regret / oracle (3.1)."""
    forbidden = {"counterfactual_xy", "j_true", "regret", "oracle", "true_response",
                 "true_error", "cea", "cew", "presence_mask", "y_reg", "y_high"}
    for fn in (s4cf.build_relative_tokens, s4cf.compute_predicted_costs):
        params = set(inspect.signature(fn).parameters)
        assert params.isdisjoint(forbidden), (fn.__name__, params & forbidden)


def test_compute_predicted_costs_matches_stage4c_bitwise():
    d = _synthetic()
    j_seeds, j_ens = s4cf.compute_predicted_costs(
        d["e2e"], d["factual_xy"], d["history_last_xy"], d["inference_mask"])
    S, G, A = j_seeds.shape
    # manual reference via the frozen Stage 4C consequence function.
    hist_b = np.broadcast_to(d["history_last_xy"][:, None, :], (G, A, 2))
    mask_b = np.broadcast_to(d["inference_mask"][:, None], (G, A) + d["inference_mask"].shape[1:])
    for s in range(S):
        y = d["factual_xy"][:, None] + d["e2e"][s]
        comp = s4co.trajectory_components(y, hist_b, mask_b)
        ref = s4co.total_cost(comp["interaction"], s4co.efficiency_cost(comp["progress"]))
        np.testing.assert_array_equal(j_seeds[s], ref)
    y_ens = d["factual_xy"][:, None] + d["e2e"].mean(axis=0)
    comp = s4co.trajectory_components(y_ens, hist_b, mask_b)
    ref_ens = s4co.total_cost(comp["interaction"], s4co.efficiency_cost(comp["progress"]))
    np.testing.assert_array_equal(j_ens, ref_ens)


def test_relative_tokens_structure():
    d = _synthetic(G=5)
    out = s4cf.build_relative_tokens(
        d["e2e"], d["factual_xy"], d["history_last_xy"], d["inference_mask"],
        d["action_risks"], d["geometry"], d["t_star"], d["group_seed"])
    G = 5
    assert out["tokens"].shape == (G, s4cp.N_COMPETITORS, s4cf.TOKEN_DIM)
    assert out["global_features"].shape == (G, s4cf.GLOBAL_DIM)
    assert out["competitor_index"].shape == (G, s4cp.N_COMPETITORS)
    j_hat = out["j_hat"]
    a0 = out["reference_action"]
    np.testing.assert_array_equal(a0, np.argmin(j_hat, axis=-1))
    for g in range(G):
        comp = out["competitor_index"][g]
        assert a0[g] not in comp
        assert sorted(comp.tolist()) == sorted(a for a in range(s4cp.N_ACTIONS) if a != a0[g])
        # tokens ordered by frozen ACTION_PLANS order.
        assert list(comp) == [a for a in range(s4cp.N_ACTIONS) if a != a0[g]]


def test_token_continuous_values():
    d = _synthetic(G=3)
    out = s4cf.build_relative_tokens(
        d["e2e"], d["factual_xy"], d["history_last_xy"], d["inference_mask"],
        d["action_risks"], d["geometry"], d["t_star"], d["group_seed"])
    j_hat, a0 = out["j_hat"], out["reference_action"]
    names = list(s4cf.TOKEN_CONT_NAMES)
    djm = names.index("delta_j_mean")
    for g in range(3):
        for k, a in enumerate(out["competitor_index"][g]):
            expected = j_hat[g, a] - j_hat[g, a0[g]]
            assert out["tokens"][g, k, djm] == pytest.approx(expected)
    # selected_risk for COMBINED equals the class risk at a0 (constant across tokens).
    sel_col = names.index("selected_risk_COMBINED_RISK_HEAD")
    comb = d["action_risks"]["COMBINED_RISK_HEAD"]
    for g in range(3):
        assert np.allclose(out["tokens"][g, :, sel_col], comb[g, a0[g]])


def test_global_margins_and_baselines():
    d = _synthetic(G=6)
    out = s4cf.build_relative_tokens(
        d["e2e"], d["factual_xy"], d["history_last_xy"], d["inference_mask"],
        d["action_risks"], d["geometry"], d["t_star"], d["group_seed"])
    j_hat = out["j_hat"]
    order = np.argsort(j_hat, axis=-1, kind="stable")
    rows = np.arange(6)
    margin = j_hat[rows, order[:, 1]] - j_hat[rows, order[:, 0]]
    names = list(s4cf.GLOBAL_NAMES)
    np.testing.assert_allclose(out["global_features"][:, names.index("top1_top2_margin")], margin)
    # MARGIN_ONLY baseline = -(top1_top2_margin).
    np.testing.assert_allclose(out["baselines"][s4cp.BASELINE_MARGIN], -margin)
    # COMBINED_TOP2_MAX = max(combined risk at top1, top2).
    comb = d["action_risks"]["COMBINED_RISK_HEAD"]
    np.testing.assert_allclose(
        out["baselines"][s4cp.BASELINE_COMBINED_TOP2_MAX],
        np.maximum(comb[rows, order[:, 0]], comb[rows, order[:, 1]]))
    # seed vote entropy non-negative, agreement in (0,1].
    assert (out["global_features"][:, names.index("seed_vote_entropy")] >= 0).all()
    sag = out["global_features"][:, names.index("seed_top1_agreement")]
    assert ((sag > 0) & (sag <= 1.0)).all()


def test_invalid_group_rejected_not_padded():
    d = _synthetic(G=4)
    d["geometry"]["distance_to_ego"][2, :] = np.nan  # group 2 loses all distances
    out = s4cf.build_relative_tokens(
        d["e2e"], d["factual_xy"], d["history_last_xy"], d["inference_mask"],
        d["action_risks"], d["geometry"], d["t_star"], d["group_seed"])
    assert out["valid_mask"].tolist() == [True, True, False, True]


def test_scaler_train_only_and_onehot_passthrough():
    d = _synthetic(G=8, seed=1)
    out = s4cf.build_relative_tokens(
        d["e2e"], d["factual_xy"], d["history_last_xy"], d["inference_mask"],
        d["action_risks"], d["geometry"], d["t_star"], d["group_seed"])
    train_idx = np.arange(0, 5)
    scaler = s4cf.FrozenFeatureScaler.fit(out["tokens"][train_idx], out["global_features"][train_idx])
    # means come ONLY from the train subset.
    cont = out["tokens"][train_idx][:, :, s4cf.TOKEN_CONT_IDX].reshape(-1, s4cf.N_TOKEN_CONT)
    np.testing.assert_allclose(scaler.token_cont_mean, cont.mean(axis=0))
    np.testing.assert_allclose(scaler.global_mean, out["global_features"][train_idx].mean(axis=0))
    # transform standardises continuous and leaves one-hots bitwise unchanged.
    transformed = scaler.transform_tokens(out["tokens"])
    assert np.allclose(transformed[:, :, s4cf.TOKEN_CAT_IDX],
                       out["tokens"][:, :, s4cf.TOKEN_CAT_IDX])
    tcont = transformed[train_idx][:, :, s4cf.TOKEN_CONT_IDX].reshape(-1, s4cf.N_TOKEN_CONT)
    assert np.allclose(tcont.mean(axis=0), 0.0, atol=1e-8)
    # save/load roundtrip.
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "scaler.npz"
        scaler.save(p)
        loaded = s4cf.FrozenFeatureScaler.load(p)
        np.testing.assert_allclose(loaded.token_cont_mean, scaler.token_cont_mean)
