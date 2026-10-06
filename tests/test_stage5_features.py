"""Blocking tests for the Stage 5 unified legal-input layer (plan v0.3 section 3.1, Task 2).

Plan ``AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3``. Pins: LegalRiskBundle
shape/dtype/mask contract, action-order consistency, truth-free score side (the
bundle must build identically from a truth-stripped npz), bitwise rebuild
determinism, and exact numerical parity of the truth side + non-learning baselines
with the frozen Stage 4C-R2 validation gate receipt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4cp_statistics as s4cps  # noqa: E402
from ac_twin import stage5_features as s5f  # noqa: E402
from ac_twin import stage5_protocol as s5p  # noqa: E402

CONFIG_PATH = ROOT / "config/stage5_sota_comparison_v0.3.json"
GATE_REPORT = (
    ROOT / "experiments/stage4_reliability/stage4c_r2/development/validation_gate_report.json")

N_VALIDATION = 1500


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def context():
    sources = s5p.resolve_sources(_config())
    return s5f.build_feature_context(sources, "validation")


@pytest.fixture(scope="module")
def bundle(context):
    return context[0]


@pytest.fixture(scope="module")
def truth():
    sources = s5p.resolve_sources(_config())
    return s5f.compute_group_truth(sources, "validation")


# --- bundle contract ----------------------------------------------------------------


def test_bundle_shapes_and_dtypes(bundle):
    g = N_VALIDATION
    assert bundle.tokens.shape == (g, 9, 34) and bundle.tokens.dtype == np.float32
    assert bundle.token_mask.shape == (g, 9) and bundle.token_mask.dtype == bool
    assert bundle.global_features.shape == (g, 9) and bundle.global_features.dtype == np.float32
    assert bundle.j_hat.shape == (g, 10)
    assert bundle.j_hat_by_seed.shape == (5, g, 10)
    assert bundle.selected_action.shape == (g,) and bundle.selected_action.dtype == np.int64
    assert bundle.group_seed.shape == (g,) and bundle.group_seed.dtype == np.int64
    assert bundle.valid_mask.shape == (g,) and bundle.valid_mask.dtype == bool
    assert np.isfinite(bundle.tokens).all()
    assert np.isfinite(bundle.global_features).all()
    assert np.isfinite(bundle.j_hat).all()


def test_token_mask_complete(bundle):
    assert bundle.token_mask.all()


def test_selected_action_matches_argmin_jhat(bundle):
    assert np.array_equal(bundle.selected_action, np.argmin(bundle.j_hat, axis=-1))
    assert bundle.selected_action.min() >= 0 and bundle.selected_action.max() < 10


def test_group_seeds_unique(bundle):
    assert np.unique(bundle.group_seed).size == bundle.group_seed.size


def test_valid_mask_all_true(bundle):
    assert bundle.valid_mask.all()


def test_strata_inference_legal_only(bundle):
    assert set(bundle.strata) <= {"t_star", "interaction_min_distance", "density_count"}
    assert bundle.strata["t_star"].shape == (N_VALIDATION,)


def test_bundle_carries_no_truth(bundle):
    forbidden = {"j_true", "true_regret", "regret", "oracle_action", "high_regret",
                 "counterfactual_xy", "object_counterfactual_futures"}
    assert not (forbidden & set(vars(bundle)))
    assert not (forbidden & set(bundle.strata))


def test_bitwise_rebuild_repeatable(context):
    sources = s5p.resolve_sources(_config())
    bundle2, _ = s5f.build_feature_context(sources, "validation")
    bundle1 = context[0]
    for field in ("tokens", "token_mask", "global_features", "j_hat", "j_hat_by_seed",
                  "selected_action", "group_seed", "valid_mask"):
        a = getattr(bundle1, field)
        b = getattr(bundle2, field)
        assert np.array_equal(a, b), f"bundle field {field} is not bitwise reproducible"


def test_score_side_runs_without_truth(tmp_path, bundle):
    """The bundle must build identically from a truth-stripped npz (no
    ``object_counterfactual_futures`` key), proving the score side never reads truth."""
    sources = s5p.resolve_sources(_config())
    data_path = Path(sources["validation_data"])
    stripped = tmp_path / "validation_stripped.npz"
    with np.load(data_path, allow_pickle=True) as archive:
        kept = {k: archive[k] for k in archive.files
                if k != "object_counterfactual_futures"}
    assert "object_counterfactual_futures" not in kept
    np.savez(stripped, **kept)
    bundle2, _ = s5f.build_feature_context(sources, "validation", data_path=stripped)
    assert np.array_equal(bundle.tokens, bundle2.tokens)
    assert np.array_equal(bundle.j_hat, bundle2.j_hat)
    assert np.array_equal(bundle.selected_action, bundle2.selected_action)


# --- truth side parity with the frozen R2 validation gate ------------------------------


def test_truth_shapes_and_labels(truth, bundle):
    g = N_VALIDATION
    assert truth["j_true"].shape == (g, 10)
    assert truth["regret"].shape == (g,)
    assert truth["oracle_action"].shape == (g,)
    assert np.array_equal(truth["oracle_action"], np.argmin(truth["j_true"], axis=-1))
    assert np.array_equal(truth["selected_action"], bundle.selected_action)
    assert np.array_equal(truth["group_seed"], bundle.group_seed)
    assert (truth["regret"] >= 0).all()
    assert set(np.unique(truth["high_regret"])) <= {0, 1}
    assert 0.0 < float(truth["high_regret"].mean()) < 1.0


def test_truth_baselines_exact_parity_with_r2_validation_gate(context, truth):
    """Recomputed non-learning baseline curves must equal the frozen R2 validation
    gate receipt exactly (same data, same frozen code path)."""
    _, extras = context
    report = json.loads(GATE_REPORT.read_text(encoding="utf-8"))
    frozen = report["baseline_receipt"]
    regret = truth["regret"]
    for name in ("MARGIN_ONLY", "SEED_VOTE_ENTROPY", "SEED_MARGIN_STD"):
        risk = np.asarray(extras["baselines"][name], dtype=np.float64)
        order = s4cps.order_by_risk(risk)
        aurc = s4cps.aurc_regret(order, regret)
        r80 = s4cps.regret_at_coverage(order, regret, 0.80)
        assert aurc == pytest.approx(frozen["aurc_regret"][name], abs=0, rel=1e-12), name
        assert r80 == pytest.approx(frozen["regret_at_80"][name], abs=0, rel=1e-12), name
