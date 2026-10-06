"""Engineering-only Stage 5 timing receipt tests."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "code" / "benchmark_stage5_inference.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_stage5_inference", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_summarize_ms_per_group_reports_distribution() -> None:
    module = _load_module()
    result = module.summarize_ms_per_group([10.0, 20.0, 30.0, 40.0], n_groups=10)
    assert result == pytest.approx({
        "median_ms_per_group": 2.5,
        "q1_ms_per_group": 1.75,
        "q3_ms_per_group": 3.25,
        "min_ms_per_group": 1.0,
        "max_ms_per_group": 4.0,
    })


def test_receipt_scope_rejects_scientific_result_fields() -> None:
    module = _load_module()
    clean = {
        "scope": "ENGINEERING_TIMING_ONLY_NO_SCIENTIFIC_RESULTS",
        "benchmark": {
            "AC-Twin Direct": {"median_ms_per_group": 0.1},
            "SELECTIVENET_REGRET_ADAPTED": {"median_ms_per_group": 0.2},
        },
    }
    module.validate_receipt_scope(clean)

    contaminated = dict(clean, aurc_regret=0.1)
    with pytest.raises(ValueError, match="scientific result field"):
        module.validate_receipt_scope(contaminated)


def test_slice_legal_bundle_preserves_seed_axis_and_uses_one_group() -> None:
    module = _load_module()

    @dataclass(frozen=True)
    class Bundle:
        tokens: np.ndarray
        token_mask: np.ndarray
        global_features: np.ndarray
        j_hat: np.ndarray
        j_hat_by_seed: np.ndarray
        selected_action: np.ndarray
        group_seed: np.ndarray
        scenario_id: np.ndarray
        strata: dict[str, np.ndarray]
        valid_mask: np.ndarray

    bundle = Bundle(
        tokens=np.zeros((3, 9, 34)), token_mask=np.ones((3, 9), dtype=bool),
        global_features=np.zeros((3, 9)), j_hat=np.zeros((3, 10)),
        j_hat_by_seed=np.zeros((5, 3, 10)), selected_action=np.arange(3),
        group_seed=np.arange(3), scenario_id=np.arange(3),
        strata={"t_star": np.arange(3)}, valid_mask=np.ones(3, dtype=bool),
    )
    sliced = module.slice_legal_bundle(bundle, 1)
    assert sliced.tokens.shape == (1, 9, 34)
    assert sliced.j_hat_by_seed.shape == (5, 1, 10)
    assert sliced.strata["t_star"].shape == (1,)
