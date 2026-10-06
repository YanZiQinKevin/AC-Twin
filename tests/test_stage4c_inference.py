"""Stage 4C inference tests: expansion order, top-2 aggregation, truth whitelist."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4c_inference as s4ci  # noqa: E402
from ac_twin import stage4c_protocol as s4cp  # noqa: E402
from ac_twin import stage4_risk as s4r  # noqa: E402

NA = s4cp.N_ACTIONS


def _group_npz(tmp_path, groups=3):
    """Write a minimal synthetic grouped npz honouring the tensor contract."""
    rng = np.random.default_rng(0)
    capacity, hl, hz, nf = 5, 10, 20, 8
    history = rng.normal(size=(groups, hl, capacity, nf)).astype(np.float32)
    factual = rng.normal(size=(groups, hz, capacity, nf)).astype(np.float32)
    cf = rng.normal(size=(groups, NA, hz, capacity, nf)).astype(np.float32)
    for arr in (history, factual):
        arr[..., 0] = 0.0
        arr[..., :4, 0] = 1.0
    cf[..., 0] = 0.0
    cf[..., :4, 0] = 1.0
    cf_actions = np.tile([a for a, _ in s4cp.ACTION_PLANS], (groups, 1)).astype(np.int64)
    cf_durations = np.tile([d for _, d in s4cp.ACTION_PLANS], (groups, 1)).astype(np.int64)
    factual_actions = rng.integers(0, 5, size=(groups, hz)).astype(np.int64)
    inference_mask = np.zeros((groups, hz, capacity), dtype=bool)
    inference_mask[..., 1:4] = True  # non-ego slots 1..3 legal
    out = tmp_path / s4cp.PURPOSE_DEV / "dev_selection.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, object_history=history, object_factual_future=factual,
        object_counterfactual_futures=cf, cf_actions=cf_actions,
        cf_durations=cf_durations, factual_actions=factual_actions,
        t_star=np.full(groups, 12), episode_seed=np.arange(100, 100 + groups),
        n_mask_cells=np.full(groups, 60), inference_mask=inference_mask)
    return out


def _config(tmp_path, npz):
    # config carries the BASE dev path; load_group_eval adds the per-role suffix.
    base = str(npz).replace("_selection.npz", ".npz")
    return {
        "data_output": base,
        "scaler_path": "experiments/stage3_routing/stage3r/scaler.json",
    }


def test_load_group_eval_expansion_order(tmp_path):
    npz = _group_npz(tmp_path, groups=3)
    bundle = s4ci.load_group_eval(_config(tmp_path, npz), role="development_selection")
    assert bundle["groups"] == 3 and bundle["branches"] == 30
    # branch e = g*10 + a: cf_action/cf_duration must follow the frozen plan order.
    cf_action = bundle["tensors"]["cf_action"].numpy().reshape(3, NA)
    cf_duration = bundle["tensors"]["cf_duration"].numpy().reshape(3, NA)
    for g in range(3):
        assert [int(a) for a in cf_action[g]] == [a for a, _ in s4cp.ACTION_PLANS]
        assert [int(d) for d in cf_duration[g]] == [d for _, d in s4cp.ACTION_PLANS]
    # counterfactual plan for (action,duration) matches materialisation of factual.
    factual_actions = np.load(npz, allow_pickle=True)["factual_actions"]
    cf_plans = bundle["tensors"]["counterfactual_actions"].numpy().reshape(3, NA, 20)
    for g in range(3):
        for a in range(NA):
            act, dur = s4cp.ACTION_PLANS[a]
            assert np.array_equal(
                cf_plans[g, a],
                s4cp.materialize_action_plan(factual_actions[g], act, dur))


def test_aggregate_group_risk_top2_and_margin():
    groups = 2
    rng = np.random.default_rng(1)
    action_scores = {"COMBINED_RISK_HEAD": rng.normal(size=(groups, NA))}
    # predicted cost with known argmin/runner-up per group.
    j_hat = np.tile(np.arange(NA, dtype=float), (groups, 1))  # ascending: a1=0, a2=1
    j_hat[1] = j_hat[1, ::-1]                                  # group1: a1=9, a2=8
    out = s4ci.aggregate_group_risk(action_scores, j_hat)
    r = action_scores["COMBINED_RISK_HEAD"]
    assert out["COMBINED_RISK_HEAD_TOP2_MAX"][0] == pytest.approx(max(r[0, 0], r[0, 1]))
    assert out["COMBINED_RISK_HEAD_TOP2_MAX"][1] == pytest.approx(max(r[1, 9], r[1, 8]))
    assert out["MARGIN_ONLY"][0] == pytest.approx(-(j_hat[0, 1] - j_hat[0, 0]))
    assert out["MARGIN_ONLY"][1] == pytest.approx(-(j_hat[1, 8] - j_hat[1, 9]))


def test_ensemble_predicted_cost_uses_factual_plus_mean_effect(tmp_path):
    npz = _group_npz(tmp_path, groups=2)
    bundle = s4ci.load_group_eval(_config(tmp_path, npz), role="development_selection")
    groups, hz, cap = 2, 20, 5
    # 5 seed pairs of zero effect -> Y_hat_CF == factual; consequences well-defined.
    e2e = np.zeros((5, groups, NA, hz, cap, 2), dtype=np.float32)
    preds = {"e2e": e2e}
    out = s4ci.ensemble_predicted_cost(preds, bundle)
    assert out["Y_hat_CF"].shape == (groups, NA, hz, cap, 2)
    assert out["J_hat"].shape == (groups, NA)
    # zero effect: every action shares the factual trajectory -> identical J per group.
    for g in range(groups):
        assert np.allclose(out["J_hat"][g], out["J_hat"][g][0])


def test_truth_side_params_cannot_enter_risk_interface():
    # build_legal_risk_features refuses a counterfactual-truth keyword.
    with pytest.raises(TypeError):
        s4r.build_legal_risk_features(
            primary_predictions=None, full_predictions=None, history_predictions=None,
            response_predictions=None, pre_intervention_geometry=None,
            action=None, duration=None, valid_mask=None,
            counterfactual_xy=np.zeros((1,)))  # truth-side leak must be a type error


def test_score_actions_deterministic_given_same_features():
    groups = 4
    branches = groups * NA
    rng = np.random.default_rng(2)
    # minimal legal feature set for the direct candidates.
    features = {
        "U_seed_mean": np.abs(rng.normal(size=branches)),
        "D_expert_mean": np.abs(rng.normal(size=branches)),
    }

    class _Fixed:
        def predict(self, matrix):
            return np.zeros(matrix.shape[0])

    frozen = {"COMBINED_RISK_HEAD": _Fixed(), "GEOMETRY_ONLY": _Fixed(),
              "RESPONSE_ONLY": _Fixed()}

    # patch the design-matrix feature lookups to only the features we supplied.
    import ac_twin.stage4_risk as _s4r
    saved = dict(_s4r.CANDIDATE_FEATURES)
    try:
        _s4r.CANDIDATE_FEATURES[_s4r.CANDIDATE_COMBINED] = ("U_seed_mean",)
        _s4r.CANDIDATE_FEATURES[_s4r.BASELINE_GEOMETRY] = ("U_seed_mean",)
        _s4r.CANDIDATE_FEATURES[_s4r.CANDIDATE_RESPONSE] = ("U_seed_mean",)
        a = s4ci.score_actions(features, frozen, groups=groups, n_actions=NA, rng_seed=5)
        b = s4ci.score_actions(features, frozen, groups=groups, n_actions=NA, rng_seed=5)
    finally:
        _s4r.CANDIDATE_FEATURES.clear()
        _s4r.CANDIDATE_FEATURES.update(saved)
    for name in a:
        assert np.array_equal(a[name], b[name])  # identical inputs -> identical scores
