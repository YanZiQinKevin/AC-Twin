from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import run_stage3b as d  # noqa: E402
from ac_twin import stage3b_inputs as s3bi  # noqa: E402
from ac_twin.stage3_routing import (  # noqa: E402
    CAPACITY_AUDIT_CONDITIONS,
    Condition,
    Stage3Router,
    audit_capacity,
)


# --------------------------------------------------------------------------- #
# Frozen structure: condition set, seed pairing, gate-hidden candidates
# --------------------------------------------------------------------------- #
def test_tier1_condition_map_is_exactly_the_four_conditions():
    assert tuple(d.TIER1_CONDITION_MAP) == (
        "uniform_suffix", "direct", "dual_router", "oracle_reference",
    )
    assert d.TIER1_CONDITION_MAP["uniform_suffix"] == Condition.UNIFORM_SUFFIX
    assert d.TIER1_CONDITION_MAP["direct"] == Condition.DIRECT
    assert d.TIER1_CONDITION_MAP["dual_router"] == Condition.DUAL_ROUTER
    assert d.TIER1_CONDITION_MAP["oracle_reference"] == Condition.ORACLE_REFERENCE
    # The config's tier1_conditions must match the driver's map exactly.
    assert tuple(s3bi.TIER1_CONDITIONS) == tuple(d.TIER1_CONDITION_MAP)


def test_pair_seeds_preserves_frozen_one_to_one_pairing():
    expected = [(101, 1101), (202, 1202), (303, 1303), (404, 1404), (505, 1505)]
    assert [d._pair_seeds(i) for i in range(5)] == expected
    with pytest.raises(ValueError):
        d._pair_seeds(5)
    with pytest.raises(ValueError):
        d._pair_seeds(-1)


def test_gate_hidden_candidates_are_exactly_32_and_64():
    assert tuple(s3bi.GATE_HIDDEN_DIM_CANDIDATES) == (32, 64)


def test_select_gate_hidden_mechanical_and_tie_breaks_to_32():
    scored = [
        {"gate_hidden_dim": 64, "validation_effect_ade": 0.40},
        {"gate_hidden_dim": 32, "validation_effect_ade": 0.50},
    ]
    assert d._select_gate_hidden(scored) == 64  # lower ADE wins
    tie = [
        {"gate_hidden_dim": 64, "validation_effect_ade": 0.40},
        {"gate_hidden_dim": 32, "validation_effect_ade": 0.40},
    ]
    assert d._select_gate_hidden(tie) == 32  # deterministic tie -> 32
    with pytest.raises(ValueError, match="not in"):
        d._select_gate_hidden([{"gate_hidden_dim": 128, "validation_effect_ade": 0.1}])
    with pytest.raises(ValueError):
        d._select_gate_hidden([])


def test_capacity_audit_uniform_direct_dual_within_one_percent():
    models = {
        c: Stage3Router(c, hidden_dim=96, response_feature_dim=32, gate_hidden_dim=64, horizon=20)
        for c in CAPACITY_AUDIT_CONDITIONS
    }
    audit = audit_capacity(models)
    assert audit["within_tolerance"]
    assert audit["max_relative_difference"] <= 0.01


# --------------------------------------------------------------------------- #
# Gate T_dev contrast/check assembly (pure statistics on synthetic vectors)
# --------------------------------------------------------------------------- #
def _ade_vecs(base, n_pairs=5, n_ep=60, seed=0):
    rng = np.random.default_rng(seed)
    return [base + rng.normal(0, 0.01, size=n_ep) for _ in range(n_pairs)]


def test_ab_contrasts_signs_bootstrap_and_lopo():
    # dual is best (lowest ADE), direct between, uniform worst, oracle best of all.
    ade = {
        "uniform_suffix": _ade_vecs(1.00, seed=1),
        "direct": _ade_vecs(0.90, seed=2),
        "dual_router": _ade_vecs(0.80, seed=3),
        "oracle_reference": _ade_vecs(0.70, seed=4),
    }
    out = d._ab_contrasts(ade, n_resamples=200, bootstrap_seed=7)
    assert out["c_uniform"] > 0 and out["c_direct"] > 0
    assert out["rel_uniform"] > 0 and out["rel_direct"] > 0
    assert out["oracle_gap"] > 0
    assert np.isfinite(out["rel_uniform_ci80_lower"])
    assert np.isfinite(out["rel_direct_ci80_lower"])
    assert len(out["leave_one_pair_out"]) == 5
    assert out["lopo_direction_stable"]
    assert out["n_episodes"] == 60


def test_ab_contrasts_detects_reversed_direction():
    # dual WORSE than direct/uniform -> contrasts negative, lopo unstable.
    ade = {
        "uniform_suffix": _ade_vecs(0.80, seed=1),
        "direct": _ade_vecs(0.85, seed=2),
        "dual_router": _ade_vecs(1.00, seed=3),
        "oracle_reference": _ade_vecs(0.70, seed=4),
    }
    out = d._ab_contrasts(ade, n_resamples=100, bootstrap_seed=7)
    assert out["c_uniform"] < 0 and out["c_direct"] < 0
    assert not out["lopo_direction_stable"]


def test_six_checks_and_semantics():
    technical_ok = {"reconstruction_ok": True, "ablation_ok": True, "no_oracle_leakage": True}
    good_ab = {"oracle_gap": 0.1, "rel_direct": 0.08, "rel_uniform": 0.09,
               "rel_direct_ci80_lower": 0.01, "rel_uniform_ci80_lower": 0.02}
    checks = d._six_checks(good_ab, technical_ok, audit_ok=True)
    assert all(checks.values())
    # Each individual failure flips the AND.
    assert not all(d._six_checks({**good_ab, "rel_direct": 0.03}, technical_ok, True).values())
    assert not all(d._six_checks({**good_ab, "oracle_gap": -0.1}, technical_ok, True).values())
    assert not all(d._six_checks({**good_ab, "rel_uniform_ci80_lower": -0.5}, technical_ok, True).values())
    assert not all(d._six_checks(good_ab, {**technical_ok, "no_oracle_leakage": False}, True).values())
    assert not all(d._six_checks(good_ab, technical_ok, audit_ok=False).values())


def test_gate_t_dev_statistics_and_semantics():
    # The frozen seven-condition AND evaluator from stage3_statistics.
    from ac_twin import stage3_statistics as s3s
    base = {
        "reconstruction_ok": True, "ablation_ok": True, "no_oracle_leakage": True,
        "oracle_beats_uniform": True, "rel_direct": 0.08, "rel_uniform": 0.09,
        "rel_direct_ci80_lower": 0.01, "rel_uniform_ci80_lower": 0.02,
        "crossfit_direct_not_reversed": True, "audit_ok": True,
    }
    assert s3s.evaluate_gate_t_dev(base)["passed"]
    reversed_cf = {**base, "crossfit_direct_not_reversed": False}
    out = s3s.evaluate_gate_t_dev(reversed_cf)
    assert not out["passed"]
    assert out["status"] == s3s.STATUS_DEV_NOT_READY


# --------------------------------------------------------------------------- #
# smoke integration (12 ep / 1 epoch) — only ENGINEERING_SMOKE_ONLY
# --------------------------------------------------------------------------- #
def test_technical_checks_structural_invariants():
    rng = np.random.default_rng(0)
    n = 3
    tensors = {
        "history": torch.as_tensor(np.abs(rng.normal(size=(n, 10, 4, 8))), dtype=torch.float32),
        "factual_suffix": torch.as_tensor(np.abs(rng.normal(size=(n, 10, 4, 8))), dtype=torch.float32),
        "factual_actions": torch.as_tensor(rng.integers(0, 5, size=(n, 20)), dtype=torch.long),
        "counterfactual_actions": torch.as_tensor(rng.integers(0, 5, size=(n, 20)), dtype=torch.long),
        "response_features": torch.zeros(n, 20, 4, 16),
        "regime_probs": torch.full((n, 20, 4, 4), 0.25),
    }
    checks = d._technical_checks(tensors, "cpu")
    assert checks == {
        "reconstruction_ok": True, "ablation_ok": True, "no_oracle_leakage": True,
    }


def test_technical_checks_use_actual_response_feature_dim():
    # Regression: the frozen predictor emits 32-dim response features; the checks
    # must derive the router width from the tensors, never hardcode it.
    rng = np.random.default_rng(1)
    n = 3
    for rfd in (16, 32, 64):
        tensors = {
            "history": torch.as_tensor(np.abs(rng.normal(size=(n, 10, 5, 8))), dtype=torch.float32),
            "factual_suffix": torch.as_tensor(np.abs(rng.normal(size=(n, 10, 5, 8))), dtype=torch.float32),
            "factual_actions": torch.as_tensor(rng.integers(0, 5, size=(n, 20)), dtype=torch.long),
            "counterfactual_actions": torch.as_tensor(rng.integers(0, 5, size=(n, 20)), dtype=torch.long),
            "response_features": torch.zeros(n, 20, 5, rfd),
            "regime_probs": torch.full((n, 20, 5, 4), 0.25),
        }
        checks = d._technical_checks(tensors, "cpu")
        assert checks["reconstruction_ok"] and checks["ablation_ok"] and checks["no_oracle_leakage"], rfd


def test_smoke_reports_engineering_smoke_only(tmp_path):
    source = ROOT / "data" / "object_centric_stage" / "merge_2000" / "object_twinworld_2000.npz"
    if not source.is_file():
        pytest.skip("development source not present")
    out_dir = tmp_path / "smoke"
    rc = d.main([
        "smoke", "--source", str(source), "--episodes", "12", "--epochs", "1",
        "--output-dir", str(out_dir),
    ])
    assert rc == 0
    report = json.loads((out_dir / "smoke_report.json").read_text(encoding="utf-8"))
    assert report["status"] == d.SMOKE_STATUS == "ENGINEERING_SMOKE_ONLY"
    assert report["episodes"] == 12 and report["epochs"] == 1
    assert set(report["effect_ade"]) == set(s3bi.TIER1_CONDITIONS)
