"""Stage 3B input binding and truth-free formal-inference loading.

This module is deliberately small.  It turns the completed Stage 3A Plus
freeze into an immutable Stage 3B dependency receipt and provides a loader
that cannot materialize counterfactual targets before a formal used-lock.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ac_twin.checkpointing import canonical_fingerprint
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3_training import verify_checkpoint_integrity


STAGE3B_PLAN_ID = "AC-TWIN-RFER-STAGE3B-2026-09-10-V0.1"
STAGE3_PARENT_PLAN_ID = "AC-TWIN-RFER-STAGE3-2026-09-08-V0.2"
STAGE3A_PLUS_PLAN_ID = "AC-TWIN-RFER-STAGE3A-PLUS-2026-09-09-V0.1"
STAGE3A_PLUS_TERMINAL = "RESPONSE_DEV_QUALIFIED_STAGE3A_PLUS"
PRODUCTION_PREDICTOR_SEEDS = (101, 202, 303, 404, 505)
ROUTER_SEEDS = (1101, 1202, 1303, 1404, 1505)
GATE_HIDDEN_DIM_CANDIDATES = (32, 64)
TIER1_CONDITIONS = (
    "uniform_suffix",
    "direct",
    "dual_router",
    "oracle_reference",
)

_FORBIDDEN_STAGE3B_SOURCE_TOKENS = (
    "stage3a_plus_qualification",
    "stage2_merge_1200",
    "stage2_confirmatory",
    "gatec_merge_id",
    "gatec_merge_unseen_duration",
)

_FEATURE_KEYS = (
    "object_history",
    "object_factual_future",
    "factual_actions",
    "counterfactual_actions",
    "cf_action",
    "cf_duration",
    "t_star",
    "episode_seed",
)


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(path: str | Path, project_root: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def _normalised(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/").casefold()


def _assert_not_consumed_or_formal_source(
    path: str | Path, *, role: str, allow_gatec: bool = False
) -> None:
    normalised = _normalised(path)
    for token in _FORBIDDEN_STAGE3B_SOURCE_TOKENS:
        if allow_gatec and token.startswith("gatec_merge_"):
            continue
        if token in normalised:
            raise ValueError(f"{role} may not use consumed qualification or formal source: {token}")


def validate_stage3b_config(
    config: Mapping[str, Any], *, project_root: str | Path | None = None
) -> dict[str, Any]:
    """Validate the small set of Stage 3B values that may affect the science."""
    root = Path(project_root).resolve() if project_root else _default_project_root()
    checked = dict(config)
    if checked.get("plan_id") != STAGE3B_PLAN_ID:
        raise ValueError("Stage 3B plan_id mismatch")
    if checked.get("parent_plan_id") != STAGE3_PARENT_PLAN_ID:
        raise ValueError("Stage 3 parent plan_id mismatch")
    if tuple(checked.get("predictor_seeds", ())) != PRODUCTION_PREDICTOR_SEEDS:
        raise ValueError("predictor_seeds must be the frozen five production seeds")
    if tuple(checked.get("router_seeds", ())) != ROUTER_SEEDS:
        raise ValueError("router_seeds must preserve the frozen one-to-one pairing")
    if tuple(checked.get("gate_hidden_dim_candidates", ())) != GATE_HIDDEN_DIM_CANDIDATES:
        raise ValueError("gate_hidden_dim_candidates must equal [32, 64]")
    if tuple(checked.get("tier1_conditions", ())) != TIER1_CONDITIONS:
        raise ValueError("tier1_conditions changed")

    for key in ("stage3a_plus_run_dir", "development_source", "historical_dev_b", "run_dir"):
        if not checked.get(key):
            raise ValueError(f"missing Stage 3B config field: {key}")
    _assert_not_consumed_or_formal_source(checked["development_source"], role="development_source")
    _assert_not_consumed_or_formal_source(checked["historical_dev_b"], role="historical_dev_b")
    if not _resolve(checked["development_source"], root).is_file():
        raise FileNotFoundError("Stage 3B development_source missing")
    if not _resolve(checked["historical_dev_b"], root).is_file():
        raise FileNotFoundError("Stage 3B historical_dev_b missing")
    return checked


def load_gatec_features_only(path: str | Path) -> dict[str, np.ndarray]:
    """Load only legal model inputs; never construct or return formal targets."""
    _assert_not_consumed_or_formal_source(path, role="inference", allow_gatec=True)
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        missing = [key for key in _FEATURE_KEYS if key not in archive.files]
        if missing:
            raise ValueError(f"formal inference source missing arrays: {missing}")
        result = {key: np.asarray(archive[key]).copy() for key in _FEATURE_KEYS}
    factual = result.pop("object_factual_future")
    if factual.ndim < 2 or factual.shape[1] < 10:
        raise ValueError("object_factual_future must contain at least ten suffix steps")
    result["object_factual_suffix"] = factual[:, :10].copy()
    return result


def freeze_stage3b_inputs(
    config_path: str | Path,
    *,
    output_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify and bind the exact Stage 3A Plus artifacts used by Stage 3B."""
    root = Path(project_root).resolve() if project_root else _default_project_root()
    config_path = _resolve(config_path, root)
    config = validate_stage3b_config(
        json.loads(config_path.read_text(encoding="utf-8")), project_root=root
    )
    plus_dir = _resolve(config["stage3a_plus_run_dir"], root)
    state_path = plus_dir / "stage3a_plus_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("plan_id") != STAGE3A_PLUS_PLAN_ID or state.get("state") != STAGE3A_PLUS_TERMINAL:
        raise ValueError("Stage 3A Plus has not reached its qualified terminal state")

    freeze_path = plus_dir / "freeze" / "final_freeze_receipt.json"
    freeze_doc = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_payload = freeze_doc.get("receipt_payload", {})
    if freeze_doc.get("receipt_sha256") != canonical_fingerprint(freeze_payload):
        raise ValueError("Stage 3A Plus freeze receipt integrity failure")
    if freeze_payload.get("plan_id") != STAGE3A_PLUS_PLAN_ID:
        raise ValueError("Stage 3A Plus freeze receipt plan mismatch")

    round_no = int(freeze_payload.get("round", -1))
    calibration_path = plus_dir / "development" / f"round{round_no}" / "calibration_selection.json"
    calibration_doc = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration_payload = calibration_doc.get("receipt_payload", {})
    if calibration_doc.get("receipt_sha256") != canonical_fingerprint(calibration_payload):
        raise ValueError("calibration selection receipt integrity failure")
    selected = calibration_payload.get("selected_artifact", {})
    if calibration_payload.get("selected_candidate") != "CAL_PER_SEED_VECTOR":
        raise ValueError("Stage 3B requires the frozen CAL_PER_SEED_VECTOR selection")
    if selected.get("fit_split") != "validation":
        raise ValueError("Stage 3B calibration must be validation-only")
    if not calibration_payload.get("data_signature") or not calibration_payload.get(
        "validation_logits_signature"
    ):
        raise ValueError("calibration selection lacks outer provenance signatures")
    all_params = selected.get("params", {}).get("per_seed", [])
    if len(all_params) < len(PRODUCTION_PREDICTOR_SEEDS):
        raise ValueError("calibration selection lacks per-seed parameters")

    checkpoint_block: dict[str, Any] = {}
    for seed in PRODUCTION_PREDICTOR_SEEDS:
        record = freeze_payload.get("checkpoints", {}).get(str(seed))
        if not record:
            raise ValueError(f"freeze receipt lacks production checkpoint seed {seed}")
        checkpoint_path = _resolve(record["path"], root)
        verify_checkpoint_integrity(checkpoint_path, expected_signature=record["signature"])
        checkpoint_block[str(seed)] = {
            "path": str(checkpoint_path),
            "file_sha256": sha256_file(checkpoint_path),
            "signature": record["signature"],
        }

    payload = {
        "plan_id": STAGE3B_PLAN_ID,
        "artifact_type": "stage3b_input_binding",
        "stage3a_plus_plan_id": STAGE3A_PLUS_PLAN_ID,
        "stage3a_plus_terminal_state": state["state"],
        "stage3a_plus_state_sha256": sha256_file(state_path),
        "stage3a_plus_freeze_file_sha256": sha256_file(freeze_path),
        "stage3a_plus_freeze_receipt_sha256": freeze_doc["receipt_sha256"],
        "development_source_sha256": sha256_file(_resolve(config["development_source"], root)),
        "historical_dev_b_sha256": sha256_file(_resolve(config["historical_dev_b"], root)),
        "seed_pairs": [list(pair) for pair in zip(PRODUCTION_PREDICTOR_SEEDS, ROUTER_SEEDS)],
        "checkpoints": checkpoint_block,
        "calibration": {
            "selection_file": str(calibration_path),
            "selection_file_sha256": sha256_file(calibration_path),
            "selection_receipt_sha256": calibration_doc["receipt_sha256"],
            "selected_candidate": calibration_payload["selected_candidate"],
            "calibration_signature": calibration_payload["candidates"]["CAL_PER_SEED_VECTOR"][
                "calibration_signature"
            ],
            "fit_split": selected["fit_split"],
            "data_signature": calibration_payload["data_signature"],
            "validation_logits_signature": calibration_payload["validation_logits_signature"],
            "per_seed_params": all_params[: len(PRODUCTION_PREDICTOR_SEEDS)],
        },
    }
    receipt = {"receipt_payload": payload, "receipt_sha256": canonical_fingerprint(payload)}
    target = (
        _resolve(output_path, root)
        if output_path is not None
        else _resolve(config["run_dir"], root) / "input_binding_receipt.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != receipt:
            raise FileExistsError("Stage 3B input binding already exists with different content")
        return existing
    fd = os.open(str(target), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, ensure_ascii=False)
    return receipt


__all__ = [
    "PRODUCTION_PREDICTOR_SEEDS",
    "ROUTER_SEEDS",
    "freeze_stage3b_inputs",
    "load_gatec_features_only",
    "validate_stage3b_config",
]
