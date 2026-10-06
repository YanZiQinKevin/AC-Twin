"""State and write-once protocol primitives for Stage 3R."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


ROLE_RULES = {
    "train": {"split": 0, "may_select": True},
    "validation": {"split": 1, "may_select": True},
    "historical_only": {"split": 2, "may_select": False},
    "r_screen": {"split": None, "may_select": False},
}

TRANSITIONS = {
    "SOFTWARE_READY": {"OOF_COMPLETE"},
    "OOF_COMPLETE": {
        "PHASE0_PASSED",
        "STAGE3R_SOFT_COMPLEMENTARITY_INSUFFICIENT",
    },
    "PHASE0_PASSED": {
        "PHASE1_PASSED",
        "STAGE3R_VALUE_WEIGHT_NOT_PREDICTABLE",
    },
    "PHASE1_PASSED": {"FINALIST_FROZEN"},
    "FINALIST_FROZEN": {"POWER_FROZEN", "STAGE3R_RSCREEN_POWER_INFEASIBLE"},
    "POWER_FROZEN": {"RSCREEN_COLLECTED"},
    "RSCREEN_COLLECTED": {"RSCREEN_INFERRED"},
    "RSCREEN_INFERRED": {
        "STAGE3R_READY_FOR_GATE_C_DESIGN",
        "STAGE3R_DEV_NOT_READY_FOR_CONFIRMATION",
    },
}


def assert_role_allows_selection(role: str) -> None:
    if role not in ROLE_RULES:
        raise ValueError(f"unknown Stage 3R role: {role}")
    if not bool(ROLE_RULES[role]["may_select"]):
        raise ValueError(f"{role} cannot select")


def transition(current: str, requested: str) -> str:
    if requested not in TRANSITIONS.get(current, set()):
        raise ValueError(f"invalid Stage 3R transition: {current} -> {requested}")
    return requested


def write_once_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"scientific artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if target.exists():
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"scientific artifact already exists: {target}")
    os.replace(temporary, target)


__all__ = [
    "ROLE_RULES",
    "TRANSITIONS",
    "assert_role_allows_selection",
    "transition",
    "write_once_json",
]
