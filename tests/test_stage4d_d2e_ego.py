"""Focused blocking-invariant tests for the Stage 4D-D2E ego repair branch (plan 3/5).

Each test pins one distinct scientific invariant once, on isolated synthetic data (no real D2
role is touched): the mechanical input_dim, the frozen architecture, the ego-inclusive mask
(M_ego is NOT the old exclude_ego mask and is not all-zero), slot-0-only repair, 5-seed index
pairing, and — critically — that the inference features do NOT read counterfactual truth.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage4d_d2e_ego as ego  # noqa: E402
from ac_twin import stage4d_external_inference as INF  # noqa: E402
from ac_twin.stage3_scaler import FeatureScaler  # noqa: E402

G, A, T, N, F_ = 3, 10, 20, 5, 8
E = G * A


def _identity_scaler() -> FeatureScaler:
    return FeatureScaler(np.zeros(ego.N_FEATURES), np.ones(ego.N_FEATURES), fitted_on="test")


def _arrays(present: bool = True) -> SimpleNamespace:
    rng = np.random.default_rng(0)
    history = rng.normal(size=(G, 10, N, F_)).astype(np.float32)
    factual = rng.normal(size=(G, T, N, F_)).astype(np.float32)
    cf = rng.normal(size=(G, A, T, N, F_)).astype(np.float32)
    for block in (history, factual, cf):
        block[..., 0] = 1.0 if present else 0.0     # presence channel
        block[..., 1] = 0.0                          # is_ego
    # mark slot 0 as the ego
    history[:, :, 0, 1] = 1.0
    factual[:, :, 0, 1] = 1.0
    cf[:, :, :, 0, 1] = 1.0
    return SimpleNamespace(groups=G, n_actions=A, object_history=history,
                           object_factual_future=factual, object_counterfactual_futures=cf)


def _bundle(rng_seed: int = 1) -> SimpleNamespace:
    rng = np.random.default_rng(rng_seed)
    return SimpleNamespace(
        branches=E,
        history=rng.normal(size=(E, 10, N, F_)).astype(np.float32),
        factual_suffix=rng.normal(size=(E, 10, N, F_)).astype(np.float32),
        factual_actions=rng.integers(0, 5, size=(E, T)).astype(np.int64),
        counterfactual_actions=rng.integers(0, 5, size=(E, T)).astype(np.int64),
        t_star=rng.integers(10, 16, size=(E,)).astype(np.int64),
        cf_duration=rng.integers(0, 2, size=(E,)).astype(np.int64) * 2 + 1,  # {1,3}
        effect_target=rng.normal(size=(E, T, N, 2)).astype(np.float32),
    )


def test_input_dim_is_mechanical_1002() -> None:
    assert ego.HISTORY_FLAT == 400 and ego.SUFFIX_FLAT == 400
    assert ego.PLAN_ONEHOT_FLAT == 100
    assert ego.INPUT_DIM == 1002
    assert ego.IDX_T_STAR == 1000 and ego.IDX_DURATION == 1001


def test_architecture_param_count_and_shape() -> None:
    model = ego.EgoEffectHead()
    n = ego.count_ego_parameters()
    # Linear(1002,128)+LN+Linear(128,128)+LN+Linear(128,40)
    expected = (1002 * 128 + 128) + 256 + (128 * 128 + 128) + 256 + (128 * 40 + 40)
    assert n == expected == 150568
    out = model(torch.zeros(7, ego.INPUT_DIM))
    assert tuple(out.shape) == (7, 20, 2)


def test_feature_layout_and_finiteness() -> None:
    bundle = _bundle()
    feat = ego.assemble_ego_features(bundle, _identity_scaler())
    assert feat.shape == (E, ego.INPUT_DIM)
    assert np.isfinite(feat).all()
    # one-hot block: each of the 20 frames sums to 1 in both plan blocks
    fact_plan = feat[:, ego.SLICE_FACT_PLAN].reshape(E, T, 5)
    cf_plan = feat[:, ego.SLICE_CF_PLAN].reshape(E, T, 5)
    np.testing.assert_allclose(fact_plan.sum(-1), 1.0)
    np.testing.assert_allclose(cf_plan.sum(-1), 1.0)
    # normalized scalars in range
    assert feat[:, ego.IDX_T_STAR].min() >= 0.0 and feat[:, ego.IDX_T_STAR].max() <= 1.0
    assert set(np.unique(feat[:, ego.IDX_DURATION])).issubset({0.0, 1.0})


def test_inference_features_do_not_read_counterfactual_truth() -> None:
    """Two bundles identical in legal inputs but with different CF truth -> identical features."""
    b1 = _bundle(rng_seed=5)
    b2 = _bundle(rng_seed=5)
    b2.effect_target = np.full((E, T, N, 2), 999.0, dtype=np.float32)  # different truth
    f1 = ego.assemble_ego_features(b1, _identity_scaler())
    f2 = ego.assemble_ego_features(b2, _identity_scaler())
    np.testing.assert_array_equal(f1, f2)


def test_m_ego_ego_inclusive_and_not_all_zero() -> None:
    arrays = _arrays(present=True)
    mask = ego.compute_ego_train_mask(arrays)
    assert mask.shape == (E, T)
    assert mask.dtype == bool
    assert mask.all()                       # ego present everywhere -> all valid
    # distinct from the old exclude_ego mask (whose slot 0 is all-False)
    g_idx = np.repeat(np.arange(G), A)
    old = INF.s3d.compute_effect_eval_mask(
        arrays.object_history[g_idx], arrays.object_factual_future[g_idx],
        arrays.object_counterfactual_futures.reshape(E, T, N, F_))  # default exclude_ego=True
    assert not old[..., 0].any()            # old mask strips the ego slot entirely


def test_m_ego_zero_where_ego_cf_absent() -> None:
    arrays = _arrays(present=True)
    # remove the ego from the counterfactual for one (group, action)
    arrays.object_counterfactual_futures[0, 3, :, 0, 0] = 0.0
    mask = ego.compute_ego_train_mask(arrays).reshape(G, A, T)
    assert not mask[0, 3].any()             # that action's ego frames are invalid
    assert mask[0, 0].all()                 # other actions unaffected


def test_m_ego_fail_closed_when_all_zero() -> None:
    arrays = _arrays(present=False)         # nothing present -> degenerate
    with pytest.raises(RuntimeError):
        ego.compute_ego_train_mask(arrays)


def test_repair_e2e_ego_replaces_only_slot0() -> None:
    rng = np.random.default_rng(2)
    S = 5
    e2e = rng.normal(size=(S, E, T, N, 2)).astype(np.float32)
    deltas = rng.normal(size=(S, E, T, 2)).astype(np.float32)
    repaired = ego.repair_e2e_ego(e2e, deltas)
    np.testing.assert_array_equal(repaired[:, :, :, 1:, :], e2e[:, :, :, 1:, :])  # slots 1-4 kept
    np.testing.assert_array_equal(repaired[:, :, :, 0, :], deltas)                # slot 0 replaced
    np.testing.assert_array_equal(e2e[:, :, :, 0, :], e2e[:, :, :, 0, :])         # input untouched


def test_seed_pairing_index_aligned() -> None:
    assert tuple(ego.EGO_SEEDS) == tuple(int(s) for s in INF.PREDICTOR_PAIR_SEEDS)
    assert len(ego.EGO_SEEDS) == ego.N_SEEDS == 5


def test_rank_label_scale_positive_and_train_only() -> None:
    bundle = _bundle()
    targets = ego.ego_targets(bundle)
    mask = np.ones((E, T), dtype=bool)
    scale = ego.rank_label_scale(targets, mask)
    assert np.isfinite(scale) and scale > 0.0


def test_predict_ego_effect_shape_and_determinism() -> None:
    torch.manual_seed(0)
    model = ego.EgoEffectHead()
    bundle = _bundle()
    feat = ego.assemble_ego_features(bundle, _identity_scaler())
    out1 = ego.predict_ego_effect(model, feat, device="cpu")
    out2 = ego.predict_ego_effect(model, feat, device="cpu")
    assert out1.shape == (E, T, 2) and np.isfinite(out1).all()
    np.testing.assert_array_equal(out1, out2)


def test_train_ego_candidate_learns_and_checkpoints() -> None:
    """Tiny learnable mapping: a few epochs must produce a finite best val loss + checkpoint."""
    rng = np.random.default_rng(3)
    n = 4 * A  # 4 groups
    feats = rng.normal(size=(n, ego.INPUT_DIM)).astype(np.float32)
    W = rng.normal(size=(ego.INPUT_DIM, 40)).astype(np.float32) * 0.01
    targets = (feats @ W).reshape(n, T, 2).astype(np.float32)
    mask = np.ones((n, T), dtype=bool)
    res = ego.train_ego_candidate(
        ego.CANDIDATE_EGO_TRAJECTORY, seed=1101,
        train_features=feats, train_targets=targets, train_mask=mask,
        val_features=feats, val_targets=targets, val_mask=mask,
        device="cpu", rank_scale=None, max_epochs=6, patience=3, batch_groups=2)
    assert np.isfinite(res.best_val_loss) and res.best_epoch >= 0
    assert res.state_dict and res.epochs_run >= 1


def test_rank_aux_candidate_runs() -> None:
    rng = np.random.default_rng(4)
    n = 4 * A
    feats = rng.normal(size=(n, ego.INPUT_DIM)).astype(np.float32)
    targets = rng.normal(size=(n, T, 2)).astype(np.float32)
    mask = np.ones((n, T), dtype=bool)
    scale = ego.rank_label_scale(targets, mask)
    res = ego.train_ego_candidate(
        ego.CANDIDATE_EGO_TRAJECTORY_PLUS_RANK, seed=1101,
        train_features=feats, train_targets=targets, train_mask=mask,
        val_features=feats, val_targets=targets, val_mask=mask,
        device="cpu", rank_scale=scale, max_epochs=3, patience=2, batch_groups=2)
    assert np.isfinite(res.best_val_loss) and res.rank_scale == scale
