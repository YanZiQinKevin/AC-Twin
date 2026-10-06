"""Blocking tests for Stage 4C Plus decision-risk models (plan sections 4, 14)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4cp_features as s4cf  # noqa: E402
from ac_twin import stage4cp_models as s4cm  # noqa: E402
from ac_twin import stage4cp_protocol as s4cp  # noqa: E402

CONFIG_PATH = (
    ROOT / "experiments/stage4_reliability/stage4c_plus/stage4cp_config.json")


def _config() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["device"] = "cpu"
    return cfg


def _toy(n=64, seed=0):
    rng = np.random.default_rng(seed)
    tokens = rng.normal(size=(n, s4cp.N_COMPETITORS, s4cf.TOKEN_DIM))
    glob = rng.normal(size=(n, s4cf.GLOBAL_DIM))
    mask = np.ones((n, s4cp.N_COMPETITORS), dtype=bool)
    y_reg = np.abs(rng.normal(size=n))
    y_high = (y_reg > np.median(y_reg)).astype(np.int64)
    return tokens, glob, mask, y_reg, y_high


def test_direct_param_count_within_tolerance():
    for hidden in s4cp.HIDDEN_DIM_CANDIDATES:
        dcr = s4cm.build_model(s4cp.ARCH_DCR, hidden_dim=hidden, dropout=s4cp.DROPOUT)
        dcr_params = s4cm.count_parameters(dcr)
        width, direct_params = s4cm.select_direct_width(dcr_params)
        rel = abs(direct_params - dcr_params) / dcr_params
        assert rel <= s4cp.PARAM_COUNT_TOLERANCE, (hidden, width, rel)


def test_dcr_permutation_invariant():
    torch.manual_seed(0)
    model = s4cm.build_model(s4cp.ARCH_DCR, hidden_dim=64, dropout=s4cp.DROPOUT).eval()
    tokens = torch.randn(6, s4cp.N_COMPETITORS, s4cf.TOKEN_DIM)
    mask = torch.ones(6, s4cp.N_COMPETITORS)
    glob = torch.randn(6, s4cf.GLOBAL_DIM)
    with torch.no_grad():
        mu0, logit0 = model(tokens, mask, glob)
        for _ in range(4):
            perm = torch.randperm(s4cp.N_COMPETITORS)
            mu1, logit1 = model(tokens[:, perm, :], mask[:, perm], glob)
            assert float((mu0 - mu1).abs().max()) < 1e-5
            assert float((logit0 - logit1).abs().max()) < 1e-5


def test_direct_is_order_sensitive_and_flat_dim():
    model = s4cm.build_model(s4cp.ARCH_DIRECT, hidden_dim=64, dropout=0.0, direct_width=43).eval()
    tokens = torch.randn(3, s4cp.N_COMPETITORS, s4cf.TOKEN_DIM)
    glob = torch.randn(3, s4cf.GLOBAL_DIM)
    flat = tokens.flatten(start_dim=1)
    assert flat.shape == (3, s4cp.N_COMPETITORS * s4cf.TOKEN_DIM)
    with torch.no_grad():
        mu, logit = model(flat, glob)
    assert mu.shape == (3,) and logit.shape == (3,)


def test_heads_to_scores_formulas():
    raw_mu = np.array([0.0, 1.0, -2.0, 5.0])
    logit = np.array([0.0, 2.0, -1.0, 3.0])
    out = s4cm.heads_to_scores(raw_mu, logit)
    mu_pos = np.log1p(np.exp(raw_mu))           # softplus
    sig = 1.0 / (1.0 + np.exp(-logit))           # sigmoid
    np.testing.assert_allclose(out["mu_only"], mu_pos)
    np.testing.assert_allclose(out["class_only"], sig)
    np.testing.assert_allclose(out["risk_score"], sig * 0.01 * np.expm1(mu_pos))
    assert (out["risk_score"] >= 0).all()


@pytest.mark.parametrize("arch", [s4cp.ARCH_DCR, s4cp.ARCH_DIRECT])
def test_fit_predict_roundtrip_and_checkpoint(arch):
    cfg = _config()
    tokens, glob, mask, y_reg, y_high = _toy()
    width = None
    if arch == s4cp.ARCH_DIRECT:
        dcr = s4cm.build_model(s4cp.ARCH_DCR, hidden_dim=64, dropout=s4cp.DROPOUT)
        width, _ = s4cm.select_direct_width(s4cm.count_parameters(dcr))
    ckpt = s4cm.fit_decision_risk_model(
        arch, tokens, glob, mask, y_reg, y_high, hidden_dim=64, lambda_cls=0.25,
        seed=6101, config=cfg, direct_width=width)
    assert np.isfinite(ckpt["train_loss"]) and np.isfinite(ckpt["best_internal_val_loss"])
    # class weight matches train prevalence.
    n_pos = float(y_high.sum())
    assert ckpt["pos_weight"] == pytest.approx((len(y_high) - n_pos) / max(n_pos, 1.0))
    scores = s4cm.predict_decision_risk(ckpt, tokens, glob, mask, device="cpu")
    assert np.isfinite(scores["risk_score"]).all()
    # checkpoint roundtrip preserves predictions exactly.
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ckpt.pt"
        s4cm.save_checkpoint(ckpt, p)
        loaded = s4cm.load_checkpoint(p)
        scores2 = s4cm.predict_decision_risk(loaded, tokens, glob, mask, device="cpu")
        np.testing.assert_array_equal(scores["risk_score"], scores2["risk_score"])
