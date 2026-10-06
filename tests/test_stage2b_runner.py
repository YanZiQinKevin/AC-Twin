"""Task 1 blocking tests: Stage 2B runner — command surface, the select-profile
flow (mechanical, write-once, no V_evt) and later-phase refusal.

No highway_env import happens here (pytest runs under ``-W error``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import run_stage2b_intersection as runner  # noqa: E402
from ac_twin import stage2b_protocol as sp  # noqa: E402

PILOT_CONFIG = ROOT / "config" / "stage2b_intersection_pilot.json"


def _metrics(weakest=120, support=0.55, positive=330):
    return {
        "accepted_pairs": 600,
        "per_stratum_min": 100,
        "prefix_identity_qa_pass": True,
        "support_rate": support,
        "positive_episodes": positive,
        "median_paired_valid_non_ego": 4.0,
        "bin_episode_counts": {"positive_low": weakest + 10,
                               "positive_mid": weakest + 5,
                               "positive_high": weakest},
    }


@pytest.fixture()
def pilot_container(tmp_path):
    container = json.loads(PILOT_CONFIG.read_text(encoding="utf-8"))
    container["profile_selection_json"] = str(tmp_path / "profile_selection.json")
    cfg_path = tmp_path / "pilot.json"
    cfg_path.write_text(json.dumps(container), encoding="utf-8")
    return container, cfg_path


def test_select_profile_writes_create_once(tmp_path, monkeypatch, pilot_container):
    container, cfg_path = pilot_container
    metrics = {"A": _metrics(weakest=110), "B": _metrics(weakest=150)}
    monkeypatch.setattr(runner, "_profile_metrics",
                        lambda container_, letter: metrics[letter])
    record = runner.cmd_select_profile(cfg_path)
    assert record["status"] == "PROFILE_SELECTED"
    assert record["selected_profile"] == "B"
    assert Path(container["profile_selection_json"]).is_file()
    with pytest.raises(FileExistsError):
        runner.cmd_select_profile(cfg_path)


def test_select_profile_inconclusive_when_unsupported(tmp_path, monkeypatch,
                                                      pilot_container):
    container, cfg_path = pilot_container
    metrics = {"A": _metrics(support=0.05), "B": None}
    monkeypatch.setattr(runner, "_profile_metrics",
                        lambda container_, letter: metrics[letter])
    record = runner.cmd_select_profile(cfg_path)
    assert record["status"] == "INCONCLUSIVE_INTERSECTION_RESPONSE_SUPPORT"
    assert record["selected_profile"] is None


def test_later_phases_refuse_to_run():
    for command in ("fit-models", "gate-s", "power-and-lock", "collect-confirm",
                    "infer-confirm", "unlock-and-adjudicate"):
        with pytest.raises(SystemExit):
            runner.main([command])


def test_collect_development_requires_profile_selection(tmp_path):
    dev = json.loads((ROOT / "config" / "stage2b_intersection_development.json")
                     .read_text(encoding="utf-8"))
    dev["profile_selection_json"] = str(tmp_path / "missing.json")
    cfg_path = tmp_path / "dev.json"
    cfg_path.write_text(json.dumps(dev), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        runner.cmd_collect_development(cfg_path, resume=False)


def test_freeze_v04_profile_b_writes_separate_create_once_receipt(
        tmp_path, monkeypatch):
    source_files = {}
    for name in ("pilot_data", "pilot_receipt", "pilot_manifest",
                 "pilot_schedule", "v03_selection", "v04_plan"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("ascii"))
        source_files[name] = str(path)

    output = tmp_path / "profile_selection_v04.json"
    recovery = {
        "plan_id": "AC-TWIN-RFER-STAGE2B-INTERSECTION-2026-09-15-V0.4",
        "source_pilot_config": str(PILOT_CONFIG),
        "source_files": source_files,
        "profile_selection_json": str(output),
    }
    cfg_path = tmp_path / "recovery.json"
    cfg_path.write_text(json.dumps(recovery), encoding="utf-8")
    metrics = _metrics(weakest=173, support=0.43, positive=258)
    metrics["bin_episode_counts"] = {
        "positive_low": 250, "positive_mid": 240, "positive_high": 173,
    }
    monkeypatch.setattr(runner, "_profile_metrics",
                        lambda container, letter: metrics if letter == "B" else None)

    receipt = runner.cmd_freeze_v04_profile(cfg_path)
    assert receipt["status"] == "V04_PROFILE_B_FROZEN"
    assert receipt["selected_profile"] == "B"
    assert set(receipt["source_sha256"]) == set(source_files)
    assert output.is_file()
    with pytest.raises(FileExistsError):
        runner.cmd_freeze_v04_profile(cfg_path)
