"""Blocking tests for Stage 2B v0.4 model qualification utilities."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage2b_training as st  # noqa: E402


def test_four_of_five_requires_metricwise_direction_not_average_only():
    per_seed = [
        {"spearman": 0.2, "mae_vs_basic": 0.1, "mae_vs_geometry": 0.1},
        {"spearman": 0.2, "mae_vs_basic": 0.1, "mae_vs_geometry": 0.1},
        {"spearman": 0.2, "mae_vs_basic": 0.1, "mae_vs_geometry": 0.1},
        {"spearman": 0.2, "mae_vs_basic": 0.1, "mae_vs_geometry": 0.1},
        {"spearman": -0.1, "mae_vs_basic": -0.1, "mae_vs_geometry": -0.1},
    ]
    assert st.four_of_five_metricwise(per_seed)
    per_seed[3]["mae_vs_geometry"] = -0.1
    assert not st.four_of_five_metricwise(per_seed)


def test_action_duration_baseline_is_six_stratum_one_hot_plus_horizon():
    action = np.array([0, 1, 2])
    duration = np.array([1, 3, 1])
    design = st.action_duration_design(action, duration, time_steps=2)
    assert design.shape == (3, 2, 7)
    assert np.allclose(design[..., :6].sum(axis=-1), 1.0)
    assert np.allclose(design[:, 0, -1], 0.0)
    assert np.allclose(design[:, 1, -1], 1.0)
    assert not np.array_equal(design[0, 0, :6], design[1, 0, :6])


def test_gate_s_requires_all_components_and_cross_seed_consistency():
    assert st.gate_s_status(True, True, True, True) == "GATE_S_PASS"
    assert st.gate_s_status(True, True, True, False) == "GATE_S_FAIL"
    assert st.gate_s_status(True, False, True, True) == "GATE_S_FAIL"


def test_prepare_development_arrays_fits_response_bins_on_train_only():
    e, h, t, n, f = 4, 2, 2, 2, 6
    history = np.zeros((e, h, n, f), dtype=np.float32)
    factual = np.zeros((e, t, n, f), dtype=np.float32)
    counterfactual = np.zeros_like(factual)
    history[..., 0] = factual[..., 0] = counterfactual[..., 0] = 1.0
    # Only non-ego slot 1 matters. Validation responses are deliberately huge.
    counterfactual[0, :, 1, 2] = [0.1, 0.2]
    counterfactual[1, :, 1, 2] = [0.3, 0.4]
    counterfactual[2:, :, 1, 2] = 100.0
    data = {
        "object_history": history,
        "object_factual_future": factual,
        "object_counterfactual_future": counterfactual,
        "split": np.array([0, 0, 1, 1]),
        "factual_actions": np.ones((e, t), dtype=np.int64),
        "counterfactual_actions": np.zeros((e, t), dtype=np.int64),
        "cf_action": np.zeros(e, dtype=np.int64),
        "cf_duration": np.ones(e, dtype=np.int64),
        "episode_seed": np.arange(1, e + 1),
    }
    arrays, thresholds = st.prepare_development_arrays(data, suffix_steps=2)
    assert arrays["indices"]["train"].tolist() == [0, 1]
    assert arrays["indices"]["validation"].tolist() == [2, 3]
    assert thresholds["t2"] < 1.0


def test_training_config_rejects_non_development_source_or_wrong_seeds():
    config = {
        "plan_id": "AC-TWIN-RFER-STAGE2B-INTERSECTION-2026-09-15-V0.4",
        "source_data": "data/object_centric_stage/stage2b_intersection_development_v04/x.npz",
        "source_receipt": "data/object_centric_stage/stage2b_intersection_development_v04/receipt.json",
        "output_dir": "experiments/stage2b_intersection/development_v04/qualification_c0",
        "variant": "C0", "factual_suffix_steps": 10,
        "training_seeds": [11, 22, 33, 44, 55],
        "architecture": {"object_embedding_dim": 64, "hidden_dim": 64,
                         "action_embedding_dim": 8},
        "training": {"batch_size": 128, "max_epochs": 200,
                     "early_stopping_patience": 25, "learning_rate": 0.001,
                     "weight_decay": 0.0001, "gradient_clip_norm": 5.0},
        "statistics": {"bootstrap_resamples": 10000, "confidence_level": 0.95,
                        "bootstrap_seed": 20260915},
        "derangement": {"train_base_seed": 2026091401,
                        "validation_seed": 2026091402},
    }
    st.validate_training_config(config)
    bad = dict(config, source_data="data/confirm.npz")
    with pytest.raises(ValueError, match="development_v04"):
        st.validate_training_config(bad)
    bad = dict(config, training_seeds=[11])
    with pytest.raises(ValueError, match="five seeds"):
        st.validate_training_config(bad)
    for missing in ("source_receipt", "derangement"):
        bad = dict(config)
        bad.pop(missing)
        with pytest.raises(ValueError, match="missing keys"):
            st.validate_training_config(bad)
    bad = dict(config, derangement={"train_base_seed": 7,
                                    "validation_seed": 2026091402})
    with pytest.raises(ValueError, match="derangement"):
        st.validate_training_config(bad)
