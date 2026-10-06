"""Stage 4B driver tests: one-shot lock ordering, smoke isolation, power->lock gate."""
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

import run_stage4b as driver  # noqa: E402
from ac_twin import stage4b_protocol as s4bp  # noqa: E402


def _config(tmp_path):
    config = json.loads((ROOT / "experiments/stage4_reliability/stage4b/stage4b_config.json")
                        .read_text(encoding="utf-8"))
    config["run_dir"] = str(tmp_path / "stage4b")
    return config


# --- one-shot confirmation lock --------------------------------------------------------------


def test_confirmation_lock_is_single_use(tmp_path):
    lock = tmp_path / "confirmation_used.lock"
    driver._acquire_confirmation_lock(lock)
    assert lock.is_file()
    with pytest.raises(RuntimeError, match="runs once"):
        driver._acquire_confirmation_lock(lock)


def test_adjudicate_refuses_second_run(tmp_path, monkeypatch):
    config = _config(tmp_path)
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "confirmation_used.lock").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(s4bp, "verify_protocol_lock", lambda _rd: {"seed_schedule_sha256": "x"})
    with pytest.raises(FileExistsError, match="runs once"):
        driver.cmd_adjudicate_confirm(config)


# --- smoke isolation ---------------------------------------------------------------------------


def test_smoke_marks_engineering_only_and_writes_only_smoke_dir(tmp_path):
    config = _config(tmp_path)
    report = driver.cmd_smoke(config)
    assert report["status"] == "ENGINEERING_SMOKE_ONLY"
    assert report["whitelist_rejects_truth_input"] is True
    assert report["finalist_loads_and_deterministic"] is True
    run_dir = Path(config["run_dir"])
    # Every artifact the smoke wrote lives under smoke/ and nothing else exists.
    created = [p for p in run_dir.rglob("*") if p.is_file()]
    assert created and all(p.parent.name == "smoke" for p in created)
    assert not (run_dir / "confirmation_used.lock").exists()


# --- power -> lock ordering ---------------------------------------------------------------------


def test_lock_protocol_refused_when_underpowered(tmp_path):
    config = _config(tmp_path)
    run_dir = Path(config["run_dir"])
    power_dir = run_dir / "power"
    power_dir.mkdir(parents=True, exist_ok=True)
    (power_dir / "power_report.json").write_text(
        json.dumps({"selected_n": None,
                    "selection": {"underpowered": True,
                                  "sensitivity_at_max_n": {"1.00": 0.1, "0.75": 0.0, "0.50": 0.0}}}),
        encoding="utf-8")
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to lock"):
        driver.cmd_lock_protocol(config_path)
    assert not (run_dir / "protocol" / "protocol_lock.json").exists()


def test_lock_protocol_requires_power_report_first(tmp_path):
    config = _config(tmp_path)
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="power report first"):
        driver.cmd_lock_protocol(config_path)
