"""Stage 3B Plus protocol guards and restart-safe receipts.

This module owns only the scientific boundary conditions.  It deliberately
does not duplicate training, statistics, or reporting code.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


PLAN_ID = "AC-TWIN-RFER-STAGE3B-PLUS-2026-09-10-V0.1"

STAGE3B_V01_HASHES = {
    "experiments/stage3_routing/stage3b/gate_t_dev/gate_t_dev_report.json":
        "D7AA2024859BCA62E247CF00A3A75EE879AECFF696D2937491ABA7E92B688964",
    "experiments/stage3_routing/stage3b/fit_tier1/fit_tier1_results.json":
        "1849474E23BA1D2E8B154836D9E13B17D663FE7253F68478481B2C86EBCD11BC",
    "experiments/stage3_routing/stage3b/input_binding_receipt.json":
        "881A73C3FD9E378A97E16BDF1A09316119272960B3BA0A730F315474A2D14490",
}

STATE_SEQUENCE = (
    "ACTIVE_IMPLEMENTATION",
    "SOFTWARE_READY",
    "SMOKE_PASS",
    "ORACLE_HEADROOM_SCREENED",
    "LEARNED_FINALIST_FROZEN",
    "CROSSFIT_COMPLETE",
    "GATE_T_PLUS_PROTOCOL_FROZEN",
    "GATE_T_PLUS_COLLECTION_COMPLETE",
    "GATE_T_PLUS_ADJUDICATED",
)
TERMINAL_STATES = frozenset(
    {
        "INVALID_STAGE3B_PLUS_TECHNICAL",
        "NO_ROUTE_ONLY_ORACLE_HEADROOM",
        "LEARNED_ROUTER_DEV_INSUFFICIENT",
        "CROSSFIT_DIRECTION_REVERSED",
        "UNDERPOWERED_FOR_GATE_T_PLUS",
        "GATE_T_PLUS_NOT_QUALIFIED",
        "ROUTING_DEV_QUALIFIED_STAGE3B_PLUS",
    }
)

_FORBIDDEN_TOKENS = (
    "stage2_confirmatory",
    "stage3a_plus/qualification",
    "stage3a_plus\\qualification",
    "gate_c",
    "gatec_",
)
_DEVELOPMENT_TOKENS = ("development_a", "development_b", "historical_dev_b")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_stage3b_v01_history(project_root: Path) -> dict[str, str]:
    """Recompute all immutable v0.1 hashes, failing with one scientific blocker."""
    root = Path(project_root).resolve()
    actual: dict[str, str] = {}
    mismatches: list[str] = []
    for relative_path, expected in STAGE3B_V01_HASHES.items():
        path = root / relative_path
        value = _sha256(path) if path.is_file() else "MISSING"
        actual[relative_path] = value
        if value != expected:
            mismatches.append(relative_path)
    if mismatches:
        raise RuntimeError(
            "immutable Stage 3B v0.1 history mismatch: " + ", ".join(mismatches)
        )
    return actual


def validate_data_role(path: Path, role: str, project_root: Path) -> None:
    """Reject consumed/formal data and keep historical A/B diagnostic-only."""
    del project_root  # reserved for future manifest-level checks
    normalised = str(Path(path)).replace("\\", "/").casefold()
    if any(token.replace("\\", "/") in normalised for token in _FORBIDDEN_TOKENS):
        raise ValueError(f"forbidden data role {role!r} for consumed/formal source")
    if any(token in normalised for token in _DEVELOPMENT_TOKENS):
        if role != "development_diagnostic":
            raise ValueError("known A/B sources are development_diagnostic only")
        return
    if role not in {"train", "validation", "selection", "development_diagnostic", "gate_t_plus"}:
        raise ValueError(f"unknown data role: {role}")


def transition(current: str, requested: str) -> str:
    """Return the requested state only when it is the next registered phase."""
    if requested in TERMINAL_STATES:
        return requested
    try:
        current_index = STATE_SEQUENCE.index(current)
    except ValueError as exc:
        raise ValueError(f"unknown Stage 3B Plus state: {current}") from exc
    if current_index + 1 >= len(STATE_SEQUENCE) or STATE_SEQUENCE[current_index + 1] != requested:
        raise ValueError(f"illegal Stage 3B Plus transition: {current} -> {requested}")
    return requested


def write_once_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create a JSON artifact and refuse every overwrite."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(str(target), flags)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite immutable artifact: {target}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_handoff_state(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace the compact, explicitly mutable recovery anchor."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, target)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def require_crossfit_before_gate_t_plus(run_dir: Path) -> dict[str, Any]:
    """Require a successful, signed-stage cross-fit receipt before Gate T+."""
    path = Path(run_dir) / "crossfit" / "crossfit_receipt.json"
    if not path.is_file():
        raise RuntimeError("Gate T+ requires the cross-fit receipt")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "CROSSFIT_COMPLETE":
        raise RuntimeError("Gate T+ requires a successful cross-fit receipt")
    if float(payload.get("c_direct_a", 0.0)) <= 0 or float(payload.get("c_direct_b", 0.0)) <= 0:
        raise RuntimeError("cross-fit receipt must record positive C_direct on A and B")
    return payload


__all__ = [
    "PLAN_ID",
    "STAGE3B_V01_HASHES",
    "STATE_SEQUENCE",
    "TERMINAL_STATES",
    "require_crossfit_before_gate_t_plus",
    "transition",
    "validate_data_role",
    "verify_stage3b_v01_history",
    "write_handoff_state",
    "write_once_json",
]
