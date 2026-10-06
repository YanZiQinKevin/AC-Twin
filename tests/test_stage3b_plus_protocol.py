from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin.stage3b_plus_protocol import (  # noqa: E402
    STAGE3B_V01_HASHES,
    require_crossfit_before_gate_t_plus,
    transition,
    validate_data_role,
    verify_stage3b_v01_history,
    write_handoff_state,
    write_once_json,
)


@pytest.mark.parametrize("relative_path,expected", STAGE3B_V01_HASHES.items())
def test_old_stage3b_history_hashes_are_intact(relative_path: str, expected: str):
    result = verify_stage3b_v01_history(ROOT)
    assert result[relative_path] == expected


def test_history_verification_reports_one_clear_blocker(tmp_path: Path):
    for relative_path in STAGE3B_V01_HASHES:
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="immutable Stage 3B v0.1 history mismatch"):
        verify_stage3b_v01_history(tmp_path)


@pytest.mark.parametrize("token", ["stage2_confirmatory", "stage3a_plus/qualification", "gate_c"])
def test_consumed_or_formal_data_cannot_train_or_select(tmp_path: Path, token: str):
    path = tmp_path / token / "source.npz"
    with pytest.raises(ValueError, match="forbidden data role"):
        validate_data_role(path, "train", tmp_path)
    with pytest.raises(ValueError, match="forbidden data role"):
        validate_data_role(path, "selection", tmp_path)


def test_known_development_sets_are_diagnostic_only(tmp_path: Path):
    path = tmp_path / "stage3b" / "development_a.npz"
    validate_data_role(path, "development_diagnostic", tmp_path)
    with pytest.raises(ValueError, match="development_diagnostic"):
        validate_data_role(path, "train", tmp_path)


def test_gate_t_plus_requires_successful_crossfit_receipt(tmp_path: Path):
    with pytest.raises(RuntimeError, match="cross-fit receipt"):
        require_crossfit_before_gate_t_plus(tmp_path)
    receipt = tmp_path / "crossfit" / "crossfit_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"status": "CROSSFIT_COMPLETE", "c_direct_a": 0.1, "c_direct_b": 0.2}), encoding="utf-8")
    assert require_crossfit_before_gate_t_plus(tmp_path)["status"] == "CROSSFIT_COMPLETE"


def test_final_json_is_write_once_but_handoff_is_replaceable(tmp_path: Path):
    final = tmp_path / "gate_t_plus_report.json"
    write_once_json(final, {"state": "first"})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_once_json(final, {"state": "second"})

    handoff = tmp_path / "handoff_state.json"
    write_handoff_state(handoff, {"state": "SOFTWARE_READY"})
    write_handoff_state(handoff, {"state": "SMOKE_PASS"})
    assert json.loads(handoff.read_text(encoding="utf-8"))["state"] == "SMOKE_PASS"


def test_state_machine_accepts_only_registered_next_state():
    assert transition("ACTIVE_IMPLEMENTATION", "SOFTWARE_READY") == "SOFTWARE_READY"
    assert transition("SOFTWARE_READY", "SMOKE_PASS") == "SMOKE_PASS"
    with pytest.raises(ValueError, match="illegal Stage 3B Plus transition"):
        transition("ACTIVE_IMPLEMENTATION", "GATE_T_PLUS_ADJUDICATED")
