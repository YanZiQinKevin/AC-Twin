from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin.stage3r_protocol import (  # noqa: E402
    assert_role_allows_selection,
    transition,
    write_once_json,
)


def test_historical_role_cannot_select():
    with pytest.raises(ValueError, match="historical_only cannot select"):
        assert_role_allows_selection("historical_only")


def test_write_once_json_refuses_existing_scientific_receipt(tmp_path: Path):
    path = tmp_path / "phase0_report.json"
    write_once_json(path, {"status": "FIRST"})
    with pytest.raises(FileExistsError):
        write_once_json(path, {"status": "SECOND"})
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "FIRST"


def test_state_machine_blocks_skipping_phase0():
    with pytest.raises(ValueError, match="invalid Stage 3R transition"):
        transition("SOFTWARE_READY", "PHASE1_READY")


def test_state_machine_accepts_registered_progression():
    assert transition("SOFTWARE_READY", "OOF_COMPLETE") == "OOF_COMPLETE"
    assert transition("OOF_COMPLETE", "PHASE0_PASSED") == "PHASE0_PASSED"
