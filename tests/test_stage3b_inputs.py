from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin.stage3b_inputs import (  # noqa: E402
    PRODUCTION_PREDICTOR_SEEDS,
    ROUTER_SEEDS,
    freeze_stage3b_inputs,
    load_gatec_features_only,
    validate_stage3b_config,
)


CONFIG_PATH = ROOT / "experiments" / "stage3_routing" / "stage3b" / "stage3b_config.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_stage3b_config_requires_exact_five_seed_pairing():
    cfg = _config()
    checked = validate_stage3b_config(cfg, project_root=ROOT)
    assert tuple(checked["predictor_seeds"]) == PRODUCTION_PREDICTOR_SEEDS
    assert tuple(checked["router_seeds"]) == ROUTER_SEEDS

    bad = copy.deepcopy(cfg)
    bad["predictor_seeds"][-1] = 606
    with pytest.raises(ValueError, match="predictor_seeds"):
        validate_stage3b_config(bad, project_root=ROOT)


def test_stage3b_config_refuses_consumed_qualification_as_development_source():
    cfg = _config()
    cfg["development_source"] = (
        "data/object_centric_stage/stage3a_plus_qualification/"
        "object_twinworld_stage3a_plus_qualification.npz"
    )
    with pytest.raises(ValueError, match="qualification"):
        validate_stage3b_config(cfg, project_root=ROOT)


def test_feature_only_loader_never_returns_counterfactual_truth(tmp_path: Path):
    path = tmp_path / "gatec_merge_id.npz"
    np.savez(
        path,
        object_history=np.zeros((2, 3, 4, 5), dtype=np.float32),
        object_factual_future=np.ones((2, 20, 4, 5), dtype=np.float32),
        object_counterfactual_future=np.full((2, 20, 4, 5), 99.0, dtype=np.float32),
        factual_actions=np.zeros((2, 20), dtype=np.int64),
        counterfactual_actions=np.ones((2, 20), dtype=np.int64),
        cf_action=np.ones(2, dtype=np.int64),
        cf_duration=np.full(2, 2, dtype=np.int64),
        t_star=np.zeros(2, dtype=np.int64),
        episode_seed=np.array([11, 12], dtype=np.int64),
        r_true=np.full((2, 20, 4), 77.0, dtype=np.float32),
        regime_labels=np.full((2, 20, 4), 3, dtype=np.int64),
    )

    features = load_gatec_features_only(path)
    assert set(features) == {
        "object_history",
        "object_factual_suffix",
        "factual_actions",
        "counterfactual_actions",
        "cf_action",
        "cf_duration",
        "t_star",
        "episode_seed",
    }
    assert features["object_factual_suffix"].shape[1] == 10
    assert all("counterfactual_future" not in key for key in features)
    assert "r_true" not in features and "regime_labels" not in features


def test_feature_only_loader_refuses_stage3a_plus_qualification():
    qualification = (
        ROOT
        / "data"
        / "object_centric_stage"
        / "stage3a_plus_qualification"
        / "object_twinworld_stage3a_plus_qualification.npz"
    )
    with pytest.raises(ValueError, match="qualification"):
        load_gatec_features_only(qualification)


def test_input_binding_recomputes_live_receipts_without_modifying_history(tmp_path: Path):
    output = tmp_path / "input_binding_receipt.json"
    receipt = freeze_stage3b_inputs(CONFIG_PATH, output_path=output, project_root=ROOT)

    assert output.is_file()
    payload = receipt["receipt_payload"]
    assert payload["stage3a_plus_terminal_state"] == "RESPONSE_DEV_QUALIFIED_STAGE3A_PLUS"
    assert payload["calibration"]["selected_candidate"] == "CAL_PER_SEED_VECTOR"
    assert payload["calibration"]["fit_split"] == "validation"
    assert payload["calibration"]["data_signature"]
    assert payload["calibration"]["validation_logits_signature"]
    assert tuple(map(int, payload["checkpoints"])) == PRODUCTION_PREDICTOR_SEEDS
    assert len(payload["calibration"]["per_seed_params"]) == 5

