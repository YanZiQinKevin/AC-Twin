from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import collect_stage3_gatec as c3  # noqa: E402
from ac_twin import stage3_protocol as s3p  # noqa: E402
from ac_twin.stage3_data import FROZEN_THRESHOLDS  # noqa: E402


def _config():
    return {
        "plan_id": s3p.PLAN_ID,
        "development_source": "data/object_centric_stage/merge_2000/object_twinworld_2000.npz",
        "response_bins": dict(FROZEN_THRESHOLDS),
        "output_dir": "experiments/stage3_routing/gatec",
        "data_output": "data/object_centric_stage/gatec_merge_id/x.npz",
        "intervention_durations": [1, 3],
        "collection_seed": 20280999,
        "formal_holdouts": {
            s3p.GATEC_ID_NAME: {"durations": [1, 3]},
            s3p.GATEC_UNSEEN_NAME: {"durations": [2]},
        },
    }


def test_id_holdout_accepts_duration_1_3() -> None:
    c3.validate_stage3_collection_config(_config(), s3p.GATEC_ID_NAME)


def test_id_holdout_rejects_unseen_duration() -> None:
    config = _config()
    config["intervention_durations"] = [2]
    with pytest.raises(ValueError, match="intervention_durations"):
        c3.validate_stage3_collection_config(config, s3p.GATEC_ID_NAME)


def test_unseen_holdout_requires_duration_2() -> None:
    config = _config()
    config["intervention_durations"] = [2]
    c3.validate_stage3_collection_config(config, s3p.GATEC_UNSEEN_NAME)
    config["intervention_durations"] = [1, 3]
    with pytest.raises(ValueError, match="intervention_durations"):
        c3.validate_stage3_collection_config(config, s3p.GATEC_UNSEEN_NAME)


def test_rejects_unknown_holdout() -> None:
    with pytest.raises(ValueError, match="unknown"):
        c3.validate_stage3_collection_config(_config(), "gatec_nope")


def test_rejects_sealed_output_path() -> None:
    config = _config()
    config["data_output"] = (
        "data/object_centric_stage/stage2_merge_1200/object_twinworld_stage2_merge_1200.npz"
    )
    with pytest.raises(ValueError, match="sealed"):
        c3.validate_stage3_collection_config(config, s3p.GATEC_ID_NAME)
