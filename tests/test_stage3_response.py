from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage3_response as s3r  # noqa: E402
from ac_twin.stage3_data import FROZEN_THRESHOLDS, N_REGIMES  # noqa: E402


def make_inputs(batch=3, agents=5, horizon=20, hist=10, feat=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    history = torch.randn(batch, hist, agents, feat, generator=g)
    factual = torch.randn(batch, hist, agents, feat, generator=g)
    factual_actions = torch.randint(0, 5, (batch, horizon), generator=g)
    counterfactual_actions = torch.randint(0, 5, (batch, horizon), generator=g)
    cf_action = torch.randint(0, 5, (batch,), generator=g)
    cf_duration = torch.randint(0, 2, (batch,), generator=g) * 2 + 1  # {1,3}
    t_star = torch.randint(10, 16, (batch,), generator=g)
    for t in (history, factual):
        t[..., 0] = 1.0  # presence
        t[..., 1] = 0.0
        t[:, :, 0, 1] = 1.0  # ego
    return {
        "history": history,
        "factual_suffix": factual,
        "factual_actions": factual_actions,
        "counterfactual_actions": counterfactual_actions,
        "cf_action": cf_action,
        "cf_duration": cf_duration,
        "t_star": t_star,
    }


# --- model outputs ------------------------------------------------------------


def test_output_shapes_and_finiteness() -> None:
    model = s3r.Stage3ResponsePredictor(hidden_dim=32, response_feature_dim=16)
    out = model(**make_inputs())
    assert out["log1p_r_hat"].shape == (3, 20, 5)
    assert out["regime_logits"].shape == (3, 20, 5, N_REGIMES)
    assert out["response_features"].shape == (3, 20, 5, 16)
    for key in ("log1p_r_hat", "regime_logits", "response_features"):
        assert torch.isfinite(out[key]).all()


def test_r_hat_nonnegative_after_inverse() -> None:
    model = s3r.Stage3ResponsePredictor(hidden_dim=32, response_feature_dim=16)
    out = model(**make_inputs())
    r_hat = s3r.invert_log1p(out["log1p_r_hat"])
    assert (r_hat >= 0).all()


def test_history_unknown_vehicles_produce_zero_outputs() -> None:
    inputs = make_inputs()
    # make vehicle 3 unknown in history (never present)
    inputs["history"][:, :, 3, :] = 0.0
    inputs["history"][:, :, 3, 0] = 0.0
    model = s3r.Stage3ResponsePredictor(hidden_dim=32, response_feature_dim=16)
    out = model(**inputs)
    assert torch.all(out["log1p_r_hat"][:, :, 3] == 0)
    assert torch.all(out["regime_logits"][:, :, 3, :] == 0)
    assert torch.all(out["response_features"][:, :, 3, :] == 0)


def test_vehicle_permutation_equivariance() -> None:
    inputs = make_inputs()
    model = s3r.Stage3ResponsePredictor(hidden_dim=32, response_feature_dim=16).eval()
    with torch.no_grad():
        out1 = model(**inputs)
        perm = [0, 2, 1, 4, 3]  # keep ego slot semantics by permuting non-ego? use full perm
        perm_inputs = {k: v.clone() for k, v in inputs.items()}
        perm_inputs["history"] = inputs["history"][:, :, perm, :]
        perm_inputs["factual_suffix"] = inputs["factual_suffix"][:, :, perm, :]
        out2 = model(**perm_inputs)
    # Shared per-agent modules => outputs permute with the vehicle axis.
    np.testing.assert_allclose(
        out2["log1p_r_hat"].numpy(),
        out1["log1p_r_hat"].numpy()[:, :, perm],
        rtol=1e-4,
        atol=1e-4,
    )


# --- masked loss ----------------------------------------------------------------


def test_loss_ignores_padding() -> None:
    inputs = make_inputs()
    model = s3r.Stage3ResponsePredictor(hidden_dim=32, response_feature_dim=16)
    out = model(**inputs)
    r_true = torch.rand(3, 20, 5)
    labels = torch.randint(0, N_REGIMES, (3, 20, 5))
    valid = torch.ones(3, 20, 5, dtype=torch.bool)
    weights = torch.ones(N_REGIMES)
    loss_full = s3r.response_masked_loss(out, r_true, labels, valid, weights)
    # Mask out a region and corrupt the truth there; loss must be unchanged.
    valid2 = valid.clone()
    valid2[:, 5:, 2] = False
    r_true2 = r_true.clone()
    r_true2[:, 5:, 2] = 1e6
    labels2 = labels.clone()
    labels2[:, 5:, 2] = 3
    loss_masked = s3r.response_masked_loss(out, r_true2, labels2, valid2, weights)
    # Recompute reference on the kept region only.
    assert torch.isfinite(loss_full)
    assert torch.isfinite(loss_masked)
    # loss_masked should be insensitive to the corrupted (masked-out) values
    loss_masked_clean = s3r.response_masked_loss(out, r_true, labels, valid2, weights)
    assert torch.allclose(loss_masked, loss_masked_clean, atol=1e-5)


def test_loss_matches_manual_smooth_l1_plus_weighted_ce() -> None:
    torch.manual_seed(0)
    out = {
        "log1p_r_hat": torch.tensor([[[0.5, 0.0]]]),
        "regime_logits": torch.tensor([[[[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]]]),
    }
    r_true = torch.tensor([[[1.0, 0.0]]])
    labels = torch.tensor([[[0, 1]]])
    valid = torch.ones(1, 1, 2, dtype=torch.bool)
    weights = torch.tensor([1.0, 2.0, 1.0, 1.0])
    lam = 0.25
    loss = s3r.response_masked_loss(out, r_true, labels, valid, weights, lambda_cls=lam)
    # manual smooth-l1 on log1p
    lp_pred = out["log1p_r_hat"]
    lp_true = torch.log1p(r_true)
    smooth = torch.nn.functional.smooth_l1_loss(lp_pred, lp_true, beta=0.1, reduction="none")
    reg = smooth.mean()
    ce = torch.nn.functional.cross_entropy(
        out["regime_logits"].reshape(-1, 4), labels.reshape(-1), weight=weights, reduction="none"
    )
    manual = reg + lam * ce.mean()
    assert torch.allclose(loss, manual, atol=1e-6)


# --- class weights ---------------------------------------------------------------


def test_class_weights_inverse_frequency_train_only() -> None:
    labels = np.array([[[0, 0, 0, 1]]])  # 3 zero, 1 low
    valid = np.ones((1, 1, 4), dtype=bool)
    w = s3r.compute_class_weights(labels, valid)
    assert w.shape == (4,)
    assert w[1] > w[0]  # rarer class gets higher weight
    assert w[2] == 0 and w[3] == 0  # absent classes -> zero weight


# --- baselines -------------------------------------------------------------------


def _baseline_features(n_eval, horizon, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "n_eval": n_eval,
        "horizon": horizon,
        "eval_shape": (n_eval, horizon, 5),
        "cf_action": rng.integers(0, 5, n_eval),
        "cf_duration": rng.choice([1, 3], n_eval),
        "initial_distance": rng.uniform(0, 30, (n_eval, 5)),  # per-vehicle (E,N)
    }


def test_constant_zero_baseline() -> None:
    feats = _baseline_features(4, 20)
    pred = s3r.ConstantZeroBaseline().fit(None, feats, None).predict(feats)
    assert pred.shape == (4, 20, 5)
    assert (pred == 0).all()


def test_train_mean_by_horizon_baseline() -> None:
    r_true = np.random.default_rng(0).uniform(0, 2, (6, 20, 5))
    valid = np.ones((6, 20, 5), dtype=bool)
    feats = _baseline_features(4, 20)
    model = s3r.TrainMeanByHorizonBaseline().fit(r_true, feats, valid)
    pred = model.predict(feats)
    assert pred.shape == (4, 20, 5)
    # column t equals train mean at horizon t
    np.testing.assert_allclose(pred[0, :, 0], r_true[:, :, :].mean(axis=(0, 2)), rtol=1e-6)


def test_linear_baselines_fit_and_predict() -> None:
    rng = np.random.default_rng(1)
    r_true = rng.uniform(0, 2, (6, 20, 5))
    valid = np.ones((6, 20, 5), dtype=bool)
    train_feats = _baseline_features(6, 20, seed=2)
    eval_feats = _baseline_features(4, 20, seed=3)
    for with_dist in (False, True):
        model = s3r.LinearResponseBaseline(with_distance=with_dist).fit(r_true, train_feats, valid)
        pred = model.predict(eval_feats)
        assert pred.shape == (4, 20, 5)
        assert (pred >= 0).all()
        assert np.isfinite(pred).all()


def test_initial_distance_is_per_vehicle() -> None:
    factual = np.zeros((2, 10, 5, 8), dtype=np.float32)
    factual[..., 0] = 1.0
    factual[:, :, 0, 1] = 1.0  # ego
    factual[:, :, 3:, 0] = 0.0  # vehicles 3,4 absent
    # ego at origin; vehicle 1 at (3,4) -> distance 5; vehicle 2 at (6,8) -> 10
    factual[:, 0, 1, 2:4] = [3.0, 4.0]
    factual[:, 0, 2, 2:4] = [6.0, 8.0]
    d = s3r.make_initial_distance(factual)
    assert d.shape == (2, 5)  # per-vehicle (E,N), not episode-mean (E,)
    np.testing.assert_allclose(d[0], [0.0, 5.0, 10.0, 0.0, 0.0], rtol=1e-5)
    np.testing.assert_allclose(d[1], [0.0, 5.0, 10.0, 0.0, 0.0], rtol=1e-5)


# --- calibration / discrimination -------------------------------------------------


def test_temperature_fit_improves_nll_and_positive() -> None:
    rng = np.random.default_rng(0)
    n = 200
    labels = rng.integers(0, 4, n)
    logits = rng.normal(0, 3, (n, 4))
    logits[np.arange(n), labels] += 5.0  # overconfident-ish
    logits = logits.reshape(n, 1, 1, 4)
    valid = np.ones((n, 1, 1), dtype=bool)
    temp = s3r.fit_temperature(logits, labels.reshape(n, 1, 1), valid)
    assert temp > 0
    # calibrated NLL <= uncalibrated NLL
    probs_raw = s3r.softmax(logits, axis=-1).reshape(n, 4)
    probs_cal = s3r.softmax(logits / temp, axis=-1).reshape(n, 4)
    nll_raw = -np.log(np.clip(probs_raw[np.arange(n), labels], 1e-12, 1)).mean()
    nll_cal = -np.log(np.clip(probs_cal[np.arange(n), labels], 1e-12, 1)).mean()
    assert nll_cal <= nll_raw + 1e-6


def test_temperature_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        s3r.temperature_scale_logits(np.zeros((2, 4)), 0.0)


def test_ece_perfect_calibration_near_zero() -> None:
    # perfectly calibrated: confidence equals empirical accuracy
    rng = np.random.default_rng(0)
    n = 6000
    conf = rng.uniform(0.5, 1.0, n)
    correct = (rng.uniform(0, 1, n) < conf).astype(int)
    probs = np.zeros((n, 1, 1, 4))
    labels = np.zeros((n, 1, 1), dtype=int)
    # set predicted class prob = conf, correctness matches
    for i in range(n):
        probs[i, 0, 0, 0] = conf[i]
        probs[i, 0, 0, 1:] = (1 - conf[i]) / 3
        labels[i, 0, 0] = 0 if correct[i] else 1
    valid = np.ones((n, 1, 1), dtype=bool)
    ece = s3r.expected_calibration_error(probs, labels, valid, n_bins=15, equal_mass=True)
    assert ece < 0.03


def test_macro_brier_perfect_is_zero() -> None:
    labels = np.array([[[0, 1, 2, 3]]])
    probs = np.zeros((1, 1, 4, 4))
    probs[0, 0, np.arange(4), labels[0, 0]] = 1.0
    valid = np.ones((1, 1, 4), dtype=bool)
    assert s3r.macro_brier(probs, labels, valid) == pytest.approx(0.0, abs=1e-9)


def test_pr_auc_perfect_and_degenerate() -> None:
    valid = np.ones((6,), dtype=bool)
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    labels = np.array([1, 1, 1, 0, 0, 0])
    assert s3r.pr_auc(scores, labels, valid) == pytest.approx(1.0, abs=1e-6)
    # reversed ranking -> low
    assert s3r.pr_auc(scores[::-1], labels, valid) < 0.6
    # no positives -> nan
    assert np.isnan(s3r.pr_auc(scores, np.zeros(6, dtype=int), valid))


def test_presence_and_high_vs_lowmid_pr_auc() -> None:
    """PR-AUC supports the two Stage-2-aligned binary tasks."""
    valid = np.ones((8,), dtype=bool)
    regime = np.array([0, 0, 0, 1, 1, 2, 3, 3])  # zero/low/mid/high
    # presence task: zero vs positive
    presence = (regime > 0).astype(int)
    # high vs (low+mid) among positive
    positive = regime > 0
    high_vs = (regime == 3).astype(int)
    scores = np.array([0.05, 0.1, 0.08, 0.5, 0.55, 0.6, 0.9, 0.95])
    auc_presence = s3r.pr_auc(scores, presence, valid)
    assert 0.9 <= auc_presence <= 1.0
    pos_valid = valid & positive
    auc_high = s3r.pr_auc(scores, high_vs, pos_valid)
    assert 0.9 <= auc_high <= 1.0


# --- supervision bundle -----------------------------------------------------------


def test_build_supervision_uses_frozen_thresholds() -> None:
    r_pos = np.random.default_rng(0).uniform(0, 3, (2, 20, 5)).astype(np.float32)
    bundle = s3r.build_supervision(r_pos)
    assert bundle["r_true"].shape == (2, 20, 5)
    assert bundle["regime_labels"].shape == (2, 20, 5)
    assert bundle["regime_labels"].max() < N_REGIMES


# --- cross-lambda selection score (Task 4, repair plan v0.1) -------------------- #


def test_response_loss_sums_separates_reg_ce_and_count() -> None:
    torch.manual_seed(0)
    E, T, N = 2, 3, 4
    outputs = {
        "log1p_r_hat": torch.randn(E, T, N),
        "regime_logits": torch.randn(E, T, N, N_REGIMES),
    }
    r_true = torch.rand(E, T, N)
    labels = torch.randint(0, N_REGIMES, (E, T, N))
    valid = torch.rand(E, T, N) > 0.25
    class_weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
    sums = s3r.response_loss_sums(outputs, r_true, labels, valid, class_weights)
    assert set(sums) == {"reg_sum", "ce_sum", "valid_count"}
    assert float(sums["valid_count"]) == float(valid.sum())
    # reg_sum matches a manual float64 masked reduction
    smooth = torch.nn.functional.smooth_l1_loss(
        outputs["log1p_r_hat"], torch.log1p(r_true.clamp_min(0.0)),
        beta=s3r.SMOOTH_L1_BETA, reduction="none",
    )
    expect = (smooth * valid).sum(dtype=torch.float64)
    assert abs(float(sums["reg_sum"]) - float(expect)) < 1e-9
    # the combined training loss at ANY lambda is recovered from the sums
    for lam in (0.10, 0.25):
        combined = s3r.response_masked_loss(
            outputs, r_true, labels, valid, class_weights, lambda_cls=lam
        )
        recon = (sums["reg_sum"] + lam * sums["ce_sum"]) / sums["valid_count"]
        assert abs(float(combined) - float(recon)) < 1e-6


def test_selection_score_uses_fixed_coefficient_regardless_of_train_lambda() -> None:
    assert s3r.SELECTION_LAMBDA_CLS == 0.25
    # identical (reg, ce) -> identical selection score; no train-lambda parameter exists
    assert s3r.selection_score(0.4, 0.2) == pytest.approx(0.4 + 0.25 * 0.2)
    assert s3r.selection_score(0.1, 0.9) == pytest.approx(0.325)


# --- Stage 3A Plus Task 5: multiclass Brier loss / label smoothing (§4.2/§4.3) --- #


def _masked_loss_fixture(seed=0):
    torch.manual_seed(seed)
    E, T, N = 2, 3, 4
    outputs = {
        "log1p_r_hat": torch.randn(E, T, N),
        "regime_logits": torch.randn(E, T, N, N_REGIMES),
    }
    r_true = torch.rand(E, T, N)
    labels = torch.randint(0, N_REGIMES, (E, T, N))
    valid = torch.rand(E, T, N) > 0.25
    class_weights = torch.tensor([1.0, 2.0, 1.5, 0.5])
    return outputs, r_true, labels, valid, class_weights


def test_multiclass_brier_loss_matches_masked_unweighted_reference() -> None:
    outputs, _, labels, valid, _ = _masked_loss_fixture()
    logits = outputs["regime_logits"]
    value = s3r.multiclass_brier_loss(logits, labels, valid)
    probs = torch.softmax(logits, dim=-1)
    onehot = torch.nn.functional.one_hot(labels.long(), N_REGIMES).to(logits.dtype)
    reference = ((probs - onehot) ** 2).mean(dim=-1)[valid].mean()
    assert torch.allclose(value, reference, atol=1e-6)


def test_multiclass_brier_loss_ignores_invalid_positions_and_is_unweighted() -> None:
    outputs, _, labels, valid, _ = _masked_loss_fixture()
    logits = outputs["regime_logits"]
    # Corrupt invalid positions with enormous logits / flipped labels.
    valid2 = valid.clone()
    valid2[:, 1:, :] = False
    logits2 = logits.clone()
    logits2[:, 1:, :, :] = 1e4
    labels2 = labels.clone()
    labels2[:, 1:, :] = (N_REGIMES - 1) - labels2[:, 1:, :]
    masked = s3r.multiclass_brier_loss(logits2, labels2, valid2)
    clean = s3r.multiclass_brier_loss(logits2, labels2, valid2)
    assert torch.allclose(masked, clean, atol=1e-6)
    # The Brier loss carries no class-weight argument at all: it must equal the
    # plain masked unweighted mean on the valid region regardless of any weighting.
    probs = torch.softmax(logits2, dim=-1)
    onehot = torch.nn.functional.one_hot(labels2.long(), N_REGIMES).to(logits2.dtype)
    reference = ((probs - onehot) ** 2).mean(dim=-1)[valid2].mean()
    assert torch.allclose(masked, reference, atol=1e-6)


def test_response_masked_loss_defaults_reproduce_old_value_bit_for_bit() -> None:
    outputs, r_true, labels, valid, class_weights = _masked_loss_fixture()
    new = s3r.response_masked_loss(
        outputs, r_true, labels, valid, class_weights, lambda_cls=0.25
    )
    # Manual reconstruction of the PRE-Task-5 formula (no Brier, no smoothing).
    valid_f = valid.to(outputs["log1p_r_hat"].dtype)
    denom = valid_f.sum().clamp_min(1.0)
    smooth = torch.nn.functional.smooth_l1_loss(
        outputs["log1p_r_hat"], torch.log1p(r_true.clamp_min(0.0)),
        beta=s3r.SMOOTH_L1_BETA, reduction="none",
    )
    reg = (smooth * valid_f).sum() / denom
    ce = torch.nn.functional.cross_entropy(
        outputs["regime_logits"].reshape(-1, N_REGIMES), labels.reshape(-1).long(),
        weight=class_weights, reduction="none",
    ).reshape(valid.shape)
    cls = (ce * valid_f).sum() / denom
    old = reg + 0.25 * cls
    assert torch.allclose(new, old, atol=1e-12)


def test_response_masked_loss_lambda_brier_adds_exactly() -> None:
    outputs, r_true, labels, valid, class_weights = _masked_loss_fixture()
    base = s3r.response_masked_loss(
        outputs, r_true, labels, valid, class_weights,
        lambda_cls=0.25, lambda_brier=0.0,
    )
    lam_b = 0.1
    with_brier = s3r.response_masked_loss(
        outputs, r_true, labels, valid, class_weights,
        lambda_cls=0.25, lambda_brier=lam_b,
    )
    brier = s3r.multiclass_brier_loss(outputs["regime_logits"], labels, valid)
    assert torch.allclose(with_brier, base + lam_b * brier, atol=1e-9)


def test_response_masked_loss_label_smoothing_changes_ce_only() -> None:
    outputs, r_true, labels, valid, class_weights = _masked_loss_fixture()
    plain = s3r.response_masked_loss(
        outputs, r_true, labels, valid, class_weights,
        lambda_cls=0.25, label_smoothing=0.0,
    )
    smoothed = s3r.response_masked_loss(
        outputs, r_true, labels, valid, class_weights,
        lambda_cls=0.25, label_smoothing=0.05,
    )
    assert not torch.allclose(plain, smoothed)
    # label_smoothing=0 must equal the original (un-smoothed) CE path exactly.
    valid_f = valid.to(outputs["log1p_r_hat"].dtype)
    denom = valid_f.sum().clamp_min(1.0)
    ce = torch.nn.functional.cross_entropy(
        outputs["regime_logits"].reshape(-1, N_REGIMES), labels.reshape(-1).long(),
        weight=class_weights, reduction="none",
    ).reshape(valid.shape)
    cls = (ce * valid_f).sum() / denom
    smooth = torch.nn.functional.smooth_l1_loss(
        outputs["log1p_r_hat"], torch.log1p(r_true.clamp_min(0.0)),
        beta=s3r.SMOOTH_L1_BETA, reduction="none",
    )
    reg = (smooth * valid_f).sum() / denom
    assert torch.allclose(plain, reg + 0.25 * cls, atol=1e-12)
