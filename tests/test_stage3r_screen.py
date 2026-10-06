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

from ac_twin.stage3r_screen import (  # noqa: E402
    adjudicate_rscreen,
    make_disjoint_seed_schedule,
    validate_rscreen_readiness,
)
import collect_stage3r_screen as collector  # noqa: E402


def test_seed_schedule_is_unique_in_range_and_disjoint():
    historical = np.array([1, 2, 3, 4000], dtype=np.int64)
    schedule = make_disjoint_seed_schedule(
        historical, size=500, entropy=20260912
    )
    assert schedule.shape == (500,)
    assert len(np.unique(schedule)) == 500
    assert schedule.min() >= 1 and schedule.max() < 2_000_000_000
    assert not set(schedule.tolist()) & set(historical.tolist())


def test_readiness_requires_every_frozen_artifact(tmp_path):
    with pytest.raises(RuntimeError, match="readiness incomplete"):
        validate_rscreen_readiness(tmp_path)


def test_collector_rejects_non_rscreen_output():
    config = {"data_output": "data/object_centric_stage/stage2_holdout/x.npz"}
    with pytest.raises(ValueError, match="dedicated stage3r_rscreen"):
        collector.validate_output_path(config)


def test_second_rscreen_adjudication_is_refused(tmp_path):
    report_path = tmp_path / "rscreen_report.json"
    report_path.write_text(json.dumps({"status": "already"}), encoding="utf-8")
    with pytest.raises(FileExistsError, match="already adjudicated"):
        adjudicate_rscreen(
            np.ones((7, 5, 20)),
            ["main", "full", "history", "best_constant", "end_to_end", "direct", "response_only"],
            report_path=report_path,
            seed=1,
            n_resamples=20,
        )


def test_rscreen_stratum_indexing_preserves_pair_by_episode_shape(tmp_path):
    reference = np.ones((6, 5, 60), dtype=np.float64)
    main = np.full((1, 5, 60), 0.9, dtype=np.float64)
    errors = np.concatenate((main, reference), axis=0)
    report = adjudicate_rscreen(
        errors,
        ["main", "full", "history", "best_constant", "end_to_end", "direct", "response_only"],
        report_path=tmp_path / "report.json",
        seed=9,
        n_resamples=50,
        strata={"duration": np.array([1] * 30 + [3] * 30)},
    )
    assert report["status"] == "STAGE3R_READY_FOR_GATE_C_DESIGN"
