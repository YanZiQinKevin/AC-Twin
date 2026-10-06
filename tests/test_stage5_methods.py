"""Blocking tests for the Stage 5 published-method adapters (plan v0.3 section 2.2, Task 3).

Plan ``AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3``. Pins the published
formulas in their adapted form: P1 TCP construction (oracle only at train), P2
SelectiveNet loss components, P3 train-all + kNN rejection isolation, P4
fit/calibration disjointness and monotone interval widths, P5 positive finite
temperature, plus the frozen grids and the 0.5x-2.0x Direct capacity band.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage5_methods as s5m  # noqa: E402


def test_method_specs_grids_frozen():
    specs = s5m.METHOD_SPECS
    assert set(specs) == {
        "CONFIDNET_TCP_ADAPTED", "SELECTIVENET_REGRET_ADAPTED",
        "RWR_NN_KNN_REJECT_ADAPTED", "CONFORMAL_REGRESSION_REJECT_ADAPTED",
        "CCTR_DRIVING_CALIBRATION_ADAPTED"}
    assert [c["temperature"] for c in specs["CONFIDNET_TCP_ADAPTED"].grid] == [0.5, 1.0, 2.0]
    assert [c["hidden_width"] for c in specs["SELECTIVENET_REGRET_ADAPTED"].grid] == [64, 96]
    assert [c["k"] for c in specs["RWR_NN_KNN_REJECT_ADAPTED"].grid] == [16, 32, 64]
    assert [c["difficulty_bins"] for c in specs["CONFORMAL_REGRESSION_REJECT_ADAPTED"].grid] == [4, 8]
    assert [(c["hidden_width"], c["depth"]) for c in specs["CCTR_DRIVING_CALIBRATION_ADAPTED"].grid] == [(64, 3), (96, 2)]


# --- P1: TCP construction ------------------------------------------------------------


def test_tempered_probs_and_tcp_target():
    rng = np.random.RandomState(0)
    j_hat = rng.rand(64, 10)
    oracle = rng.randint(0, 10, 64)
    probs = s5m.tempered_action_probs(j_hat, 1.0)
    assert probs.shape == (64, 10)
    assert np.allclose(probs.sum(axis=1), 1.0)
    tcp = s5m.tcp_target(j_hat, oracle, 1.0)
    assert np.allclose(tcp, probs[np.arange(64), oracle])
    # lower temperature sharpens (lower-or-equal oracle prob on average for min-cost oracle)
    oracle_min = np.argmin(j_hat, axis=1)
    tcp_sharp = s5m.tcp_target(j_hat, oracle_min, 0.5)
    tcp_flat = s5m.tcp_target(j_hat, oracle_min, 2.0)
    assert tcp_sharp.mean() > tcp_flat.mean()
    # higher temperature increases the entropy of the distribution
    def ent(p):
        return float(-(p * np.log(p + 1e-12)).sum(1).mean())
    assert ent(s5m.tempered_action_probs(j_hat, 2.0)) > ent(s5m.tempered_action_probs(j_hat, 0.5))


# --- P2: SelectiveNet loss ------------------------------------------------------------


def test_selectivenet_loss_components():
    f = torch.tensor([1.0, 2.0, 3.0, 4.0])
    g = torch.tensor([0.9, 0.8, 0.7, 0.6])
    h = torch.tensor([1.1, 1.9, 3.2, 3.8])
    y = torch.tensor([1.5, 2.0, 2.5, 4.5])
    target_c, lam, alpha = 0.80, 32.0, 0.5
    total, parts = s5m.selectivenet_loss(f, g, h, y, target_c, lam, alpha)
    se = (f - y) ** 2
    sel = (g * se).mean() / g.mean()
    cov_pen = lam * max(0.0, target_c - float(g.mean())) ** 2
    aux = ((h - y) ** 2).mean()
    expected = alpha * (sel + cov_pen) + (1 - alpha) * aux
    assert float(total) == pytest.approx(float(expected), rel=1e-6)
    assert float(parts["selective_risk"]) == pytest.approx(float(sel), rel=1e-6)
    assert float(parts["coverage_penalty"]) == pytest.approx(cov_pen, rel=1e-6)
    assert float(parts["auxiliary"]) == pytest.approx(float(aux), rel=1e-6)


def test_selectivenet_loss_zero_penalty_when_coverage_met():
    f = torch.zeros(8)
    g = torch.full((8,), 0.99)
    h = torch.zeros(8)
    y = torch.zeros(8)
    _, parts = s5m.selectivenet_loss(f, g, h, y, 0.80, 32.0, 0.5)
    assert float(parts["coverage_penalty"]) == 0.0


# --- P3: kNN rejection ------------------------------------------------------------------


def test_knn_rejector_scores_local_residuals():
    reference = np.array([[0.0], [1.0], [2.0], [3.0]])
    residuals = np.array([0.1, 0.2, 0.9, 0.05])
    rej = s5m.KNNRejector(k=2)
    rej.fit(reference, residuals)
    score = rej.score(np.array([[0.05], [2.1]]))
    # nearest two of 0.05 are {0.0, 1.0} -> mean(0.1, 0.2); of 2.1 are {2.0, 3.0} -> mean(0.9, 0.05)
    assert score[0] == pytest.approx((0.1 + 0.2) / 2)
    assert score[1] == pytest.approx((0.9 + 0.05) / 2)
    # k=1 on a reference point returns its own residual
    rej1 = s5m.KNNRejector(k=1)
    rej1.fit(reference, residuals)
    assert rej1.score(np.array([[2.0]]))[0] == pytest.approx(0.9)


def test_knn_rejector_state_is_train_only():
    """The rejector state is fixed at fit time: scoring never sees evaluation truth."""
    rej = s5m.KNNRejector(k=1)
    rej.fit(np.array([[0.0], [10.0]]), np.array([0.0, 1.0]))
    before = rej.score(np.array([[5.0]]))
    after = rej.score(np.array([[5.0]]))
    assert np.array_equal(before, after)


# --- P4: conformal split + Mondrian widths -------------------------------------------------


def test_conformal_fit_mask_disjoint_and_deterministic():
    group_seed = np.arange(1, 501, dtype=np.int64) * 2654435761 % (2 ** 31)
    mask1 = s5m.conformal_fit_mask(group_seed, 0.8)
    mask2 = s5m.conformal_fit_mask(group_seed, 0.8)
    assert np.array_equal(mask1, mask2)
    assert 0.7 < mask1.mean() < 0.9
    fit_idx = set(np.where(mask1)[0].tolist())
    cal_idx = set(np.where(~mask1)[0].tolist())
    assert fit_idx.isdisjoint(cal_idx)


def test_mondrian_conformal_widths_monotone_and_valid():
    rng = np.random.RandomState(3)
    n = 2000
    difficulty = rng.rand(n)
    # residuals genuinely increasing in difficulty
    residuals = 0.05 + 0.5 * difficulty + 0.05 * rng.randn(n)
    residuals = np.abs(residuals)
    state = s5m.MondrianConformalState.fit(residuals, difficulty, n_bins=4, alpha=0.10)
    # monotone non-decreasing widths across difficulty bins
    assert np.all(np.diff(state.qhat) >= 0.0)
    width = state.score(difficulty)
    assert (width > 0).all() and np.isfinite(width).all()
    coverage = float((residuals <= width / 2.0).mean())
    assert coverage >= 0.85  # conformal target 0.90 on the same data it calibrated
    # a harder region gets at least as wide an interval as an easy one
    assert state.score(np.array([0.9]))[0] >= state.score(np.array([0.1]))[0]


# --- P5: temperature -----------------------------------------------------------------------


def test_cctr_sigma_positive_finite():
    torch.manual_seed(0)
    net = s5m.CCTRUncertaintyNet(input_dim=20, hidden_width=64, depth=3)
    x = torch.randn(32, 20)
    mu, sigma = net(x)
    assert mu.shape == (32,) and sigma.shape == (32,)
    assert bool((sigma > 0).all()) and bool(torch.isfinite(sigma).all())


def test_gaussian_nll_prefers_correct_sigma():
    y = torch.tensor([0.0, 0.1, -0.1, 0.05])
    mu = torch.zeros(4)
    good = s5m.gaussian_nll(mu, torch.full((4,), 0.1), y)
    bad = s5m.gaussian_nll(mu, torch.full((4,), 5.0), y)
    assert float(good) < float(bad)


# --- capacity band -------------------------------------------------------------------------


def test_all_grid_items_inside_capacity_band():
    for spec in s5m.METHOD_SPECS.values():
        for config in spec.grid:
            count = s5m.method_param_count(spec.method_id, config)
            lo = 0.5 * s5m.DIRECT_PARAM_COUNT
            hi = 2.0 * s5m.DIRECT_PARAM_COUNT
            assert lo <= count <= hi, (spec.method_id, config, count)


def test_capacity_rejects_oversized():
    with pytest.raises(ValueError):
        s5m.check_capacity("SELECTIVENET_REGRET_ADAPTED", {"hidden_width": 512})
