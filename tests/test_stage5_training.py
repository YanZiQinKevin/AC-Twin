"""Blocking tests for the Stage 5 fit/score contract (plan v0.3 section 8, Task 3).

Plan ``AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3``. Pins:
``fit_method(method_id, train_bundle, train_truth, config, seed, output_dir) -> MethodReceipt``
and ``score_method(receipt, legal_bundle) -> [G]`` (higher=riskier) for all five
adapted methods: receipt round-trip, five-seed indexing, scoring determinism, and
truth-free scoring (scores must be generatable after the truth is removed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage5_methods as s5m  # noqa: E402
from ac_twin import stage5_training as s5t  # noqa: E402
from ac_twin.stage5_features import LegalRiskBundle  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402


def make_synthetic(g: int = 256, seed: int = 0):
    rng = np.random.RandomState(seed)
    tokens = rng.randn(g, 9, 34).astype(np.float32)
    globals_ = rng.randn(g, 9).astype(np.float32)
    j_hat_by_seed = rng.rand(5, g, 10).astype(np.float64)
    j_hat = j_hat_by_seed.mean(axis=0)
    selected = np.argmin(j_hat, axis=-1).astype(np.int64)
    group_seed = (np.arange(1, g + 1, dtype=np.int64) * 7919) % (2 ** 31)
    strata = {
        "t_star": rng.randint(0, 3, g).astype(np.int64),
        "interaction_min_distance": (rng.rand(g) * 50),
        "density_count": (rng.rand(g) * 5),
    }
    bundle = LegalRiskBundle(
        tokens=tokens, token_mask=np.ones((g, 9), dtype=bool),
        global_features=globals_, j_hat=j_hat, j_hat_by_seed=j_hat_by_seed,
        selected_action=selected, group_seed=group_seed, scenario_id=group_seed.copy(),
        strata=strata, valid_mask=np.ones(g, dtype=bool))
    truth = {
        "group_seed": group_seed,
        "selected_action": selected,
        "oracle_action": rng.randint(0, 10, g).astype(np.int64),
        "j_true": rng.rand(g, 10),
        "regret": rng.rand(g) * 0.3,
        "high_regret": (rng.rand(g) > 0.8).astype(np.int64),
    }
    return bundle, truth


FIT_KW = dict(device="cpu", max_epochs=3, patience=2)


@pytest.mark.parametrize("method_id,config", [
    ("CONFIDNET_TCP_ADAPTED", {"temperature": 1.0}),
    ("SELECTIVENET_REGRET_ADAPTED", {"hidden_width": 64}),
    ("RWR_NN_KNN_REJECT_ADAPTED", {"k": 32}),
    ("CONFORMAL_REGRESSION_REJECT_ADAPTED", {"difficulty_bins": 4}),
    ("CCTR_DRIVING_CALIBRATION_ADAPTED", {"hidden_width": 64, "depth": 3}),
])
def test_fit_score_roundtrip_all_methods(method_id, config, tmp_path):
    bundle, truth = make_synthetic()
    receipt = s5t.fit_method(method_id, bundle, truth, config, 8101, tmp_path, **FIT_KW)
    assert receipt.method_id == method_id
    assert receipt.seed == 8101
    assert Path(receipt.state_path).is_file()
    assert receipt.state_sha256 == sha256_file(receipt.state_path)
    assert receipt.param_count is None or receipt.param_count > 0
    risk = s5t.score_method(receipt, bundle, device="cpu")
    assert risk.shape == (bundle.group_seed.size,)
    assert np.isfinite(risk).all()
    # determinism: identical repeated scoring
    risk2 = s5t.score_method(receipt, bundle, device="cpu")
    assert np.array_equal(risk, risk2)


def test_five_seed_indexing_produces_distinct_states(tmp_path):
    bundle, truth = make_synthetic()
    r1 = s5t.fit_method("CONFIDNET_TCP_ADAPTED", bundle, truth,
                        {"temperature": 1.0}, 8101, tmp_path / "s1", **FIT_KW)
    r2 = s5t.fit_method("CONFIDNET_TCP_ADAPTED", bundle, truth,
                        {"temperature": 1.0}, 8202, tmp_path / "s2", **FIT_KW)
    assert r1.state_sha256 != r2.state_sha256


def test_p1_risk_is_one_minus_confidence(tmp_path):
    bundle, truth = make_synthetic()
    receipt = s5t.fit_method("CONFIDNET_TCP_ADAPTED", bundle, truth,
                             {"temperature": 1.0}, 8101, tmp_path, **FIT_KW)
    risk = s5t.score_method(receipt, bundle, device="cpu")
    assert risk.min() >= 0.0 and risk.max() <= 1.0


def test_p3_receipt_records_train_reference(tmp_path):
    bundle, truth = make_synthetic(g=128)
    receipt = s5t.fit_method("RWR_NN_KNN_REJECT_ADAPTED", bundle, truth,
                             {"k": 16}, 8101, tmp_path, **FIT_KW)
    assert receipt.notes.get("reference_groups") == 128


def test_p4_receipt_records_disjoint_fit_calibration(tmp_path):
    bundle, truth = make_synthetic(g=256)
    receipt = s5t.fit_method("CONFORMAL_REGRESSION_REJECT_ADAPTED", bundle, truth,
                             {"difficulty_bins": 4}, 8101, tmp_path, **FIT_KW)
    n_fit = receipt.notes.get("fit_groups")
    n_cal = receipt.notes.get("calibration_groups")
    assert n_fit and n_cal and n_fit + n_cal == 256
    assert receipt.notes.get("fit_calibration_disjoint") is True
    # monotone widths frozen in the state
    assert receipt.notes.get("qhat_monotone") is True


def test_scores_need_no_truth(tmp_path):
    """score_method takes only the legal bundle: the truth dict is never an input,
    and scores on a truth-stripped bundle copy are identical."""
    bundle, truth = make_synthetic()
    receipt = s5t.fit_method("SELECTIVENET_REGRET_ADAPTED", bundle, truth,
                             {"hidden_width": 64}, 8101, tmp_path, **FIT_KW)
    stripped = LegalRiskBundle(
        tokens=bundle.tokens, token_mask=bundle.token_mask,
        global_features=bundle.global_features, j_hat=bundle.j_hat,
        j_hat_by_seed=bundle.j_hat_by_seed, selected_action=bundle.selected_action,
        group_seed=bundle.group_seed, scenario_id=bundle.scenario_id,
        strata=dict(bundle.strata), valid_mask=bundle.valid_mask)
    assert np.array_equal(s5t.score_method(receipt, bundle, device="cpu"),
                          s5t.score_method(receipt, stripped, device="cpu"))
