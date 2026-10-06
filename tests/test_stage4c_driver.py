"""Stage 4C driver tests: subcommand coverage, create-once refusal, 30s progress."""
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

import run_stage4c as runner  # noqa: E402
from ac_twin import stage4c_protocol as s4cp  # noqa: E402
from ac_twin.progress import ProgressReporter  # noqa: E402


def test_all_required_subcommands_registered():
    required = {
        "preflight", "lock-development", "smoke", "collect-development",
        "infer-development", "freeze-decision-rule", "adjudicate-development",
        "power", "lock-confirm", "collect-confirm", "infer-confirm",
        "adjudicate-confirm", "verify", "run-all",
    }
    assert required.issubset(set(runner._COMMANDS))


def _tmp_config(tmp_path):
    config = json.loads(
        (ROOT / "experiments/stage4_reliability/stage4c/stage4c_config.json")
        .read_text(encoding="utf-8"))
    config["run_dir"] = str(tmp_path / "run")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_development_lock_is_create_once(tmp_path):
    config_path = _tmp_config(tmp_path)
    run_dir = tmp_path / "run"
    lock_dir = run_dir / "development"
    lock_dir.mkdir(parents=True)
    (lock_dir / "protocol_lock.json").write_text("{}", encoding="utf-8")
    # a pre-existing lock refuses BEFORE any expensive scan/collection.
    with pytest.raises(FileExistsError):
        s4cp.write_development_lock(config_path)


def test_confirm_lock_is_create_once(tmp_path):
    config_path = _tmp_config(tmp_path)
    lock_dir = tmp_path / "run" / "confirm"
    lock_dir.mkdir(parents=True)
    (lock_dir / "protocol_lock.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        s4cp.write_confirm_lock(config_path, selected_n=600)


def test_confirm_lock_rejects_noncandidate_n(tmp_path):
    config_path = _tmp_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        s4cp.build_confirm_lock(config, selected_n=700)  # not in CANDIDATE_NS


def test_progress_reporter_has_30s_heartbeat():
    reporter = ProgressReporter(total=10, every=1, prefix="t", heartbeat_seconds=30.0)
    assert reporter.heartbeat_seconds == 30.0
