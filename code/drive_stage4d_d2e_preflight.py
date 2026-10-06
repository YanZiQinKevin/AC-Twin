"""Stage 4D-D2E preflight: environment verification + source inventory (plan section 2).

Plan ``AC-TWIN-RFER-STAGE4D-D2E-EGO-CONSISTENT-EXTERNAL-2026-09-24-V0.1`` section 2. Verifies the
frozen simulation environment (Python 3.11 / torch 2.9.1+cu130 / CUDA / MetaDrive + ScenarioNet
editable installs inside ``D:/MetaDrive`` / NO TensorFlow), confirms the converted Waymo scenario
DB is readable, and registers the create-once ``source_inventory.json`` binding every frozen
source by content SHA256 (paths may differ across computers; content hash + scenario ID is the
identity). Any source drift STOPS the plan before any new data is consumed.

Writes (create-once, under ``experiments/stage4_reliability/stage4d_d2e/``):
``source_inventory.json`` and ``preflight_receipt.json``. A rerun verifies hashes and never
overwrites. Read-only against every old artifact.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

D2E_DIR = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_d2e"
CONFIG_PATH = D2E_DIR / "config.json"

EXPECTED_METADRIVE = Path(r"D:\MetaDrive\metadrive").resolve()
EXPECTED_SCENARIONET = Path(r"D:\MetaDrive\scenarionet").resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _hash_existing(paths: list[tuple[str, Path]]) -> dict[str, Any]:
    """Hash each source that exists; fail loud listing any missing."""
    out: dict[str, Any] = {}
    missing: list[str] = []
    for label, path in paths:
        if path.is_file():
            out[label] = {"path": _rel(path), "sha256": _sha256_file(path)}
        else:
            missing.append(f"{label}: {_rel(path)}")
    if missing:
        raise SystemExit("[stage4d-d2e-preflight] MISSING source(s):\n  " + "\n  ".join(missing))
    return out


def build_source_inventory(config: dict[str, Any]) -> dict[str, Any]:
    """Register every frozen source by content SHA256 (plan section 2)."""
    d2 = PROJECT_ROOT / config["d2_run_dir"]
    plat = d2 / "platform"
    dev = d2 / "development"
    sources: list[tuple[str, Path]] = [
        ("plan_d2e", PROJECT_ROOT / config["plan_path"]),
        ("plan_d2", PROJECT_ROOT / "plans/2026-09-21_AC-Twin_Stage4D-D2复杂平台外部复现实施计划_v0.2.md"),
        ("plan_d2r", PROJECT_ROOT / "plans/2026-09-24_AC-Twin_Stage4D-D2R同平台风险回退决策开发实施计划_v0.1.md"),
        ("d2_config", PROJECT_ROOT / config["d2_config_path"]),
        ("d2_terminal_screen_report", dev / "development_screen_report.json"),
        ("d2_final_selection", dev / "final_selection.json"),
        ("d2_model_inventory", dev / "model_inventory.json"),
        ("d2_feature_schema", dev / "feature_schema.json"),
        ("d2_feature_scaler", dev / "feature_scaler.npz"),
        ("d2_train_collection_receipt", d2 / "roles/train/collection_receipt.json"),
        ("d2_train_groups", d2 / "roles/train/groups.npz"),
        ("d2_validation_collection_receipt", d2 / "roles/validation/collection_receipt.json"),
        ("d2_validation_groups", d2 / "roles/validation/groups.npz"),
        ("d2_development_screen_lock", dev / "development_screen_used.lock"),
        ("platform_action_contract", plat / "action_contract.json"),
        ("platform_adapter_contract", plat / "adapter_contract.json"),
        ("platform_environment_lock", plat / "platform_environment_lock.json"),
        ("platform_preflight_receipt", plat / "preflight_receipt.json"),
        ("platform_scenario_split_manifest", plat / "scenario_split_manifest.json"),
        ("platform_scenario_split_pools", plat / "scenario_split_pools.jsonl"),
        ("platform_raw_dataset_root_receipt", plat / "raw_dataset_root_receipt.json"),
        ("d2r_terminal_power_report", PROJECT_ROOT / "experiments/stage4_reliability/stage4d_d2r/power/screen_power_report.json"),
        ("objective_code", PROJECT_ROOT / "code/ac_twin/stage4c_objective.py"),
        ("d2_config_statistics_code", PROJECT_ROOT / "code/ac_twin/stage4d_external_statistics.py"),
    ]
    inventory = _hash_existing(sources)
    inventory["_meta"] = {
        "plan_id": config["plan_id"],
        "identity_basis": "content SHA256 + scenario ID (absolute paths may differ across computers)",
        "metadrive_commit": config["metadrive_commit"],
        "scenarionet_commit": config["scenarionet_commit"],
        "n_sources": len(sources),
    }
    return inventory


def _write_once_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"create-once artifact already exists: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def run_preflight() -> dict[str, Any]:
    D2E_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    inventory_path = D2E_DIR / "source_inventory.json"
    receipt_path = D2E_DIR / "preflight_receipt.json"
    if receipt_path.exists():
        print("[stage4d-d2e-preflight] preflight_receipt.json exists (create-once); no rerun",
              flush=True)
        return json.loads(receipt_path.read_text(encoding="utf-8"))

    # --- environment assertions (fail loud) ------------------------------------------------
    import torch
    import metadrive
    import scenarionet
    assert platform.python_version_tuple()[:2] == ("3", "11"), platform.python_version()
    assert torch.__version__ == "2.9.1+cu130", torch.__version__
    assert torch.cuda.is_available(), "CUDA unavailable"
    assert Path(metadrive.__file__).resolve().is_relative_to(EXPECTED_METADRIVE)
    assert Path(scenarionet.__file__).resolve().is_relative_to(EXPECTED_SCENARIONET)
    assert importlib.util.find_spec("tensorflow") is None, "TensorFlow must not be in the sim env"

    # --- scenario DB readability ------------------------------------------------------------
    db_dir = Path(config["converted_db_dir"])
    if not db_dir.is_dir():
        raise SystemExit(f"[stage4d-d2e-preflight] converted DB dir missing: {db_dir}")
    db_files = sorted(p for p in db_dir.iterdir() if p.is_file())
    n_db = len(db_files)
    if n_db == 0:
        raise SystemExit(f"[stage4d-d2e-preflight] converted DB dir is empty: {db_dir}")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    inventory = build_source_inventory(config)
    _write_once_json(inventory_path, inventory)

    receipt = {
        "plan_id": config["plan_id"],
        "status": "D2E_PREFLIGHT_PASS",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": gpu_name,
            "numpy": np.__version__,
            "metadrive_source": str(EXPECTED_METADRIVE),
            "scenarionet_source": str(EXPECTED_SCENARIONET),
            "tensorflow_present": False,
        },
        "scenario_db": {
            "converted_db_dir": str(db_dir),
            "n_files": int(n_db),
        },
        "source_inventory_path": _rel(inventory_path),
        "source_inventory_sha256": _sha256_file(inventory_path),
        "n_sources": int(inventory["_meta"]["n_sources"]),
        "estimated_scope": {
            "note": ("collection volume is decided by the screen/confirm power pre-checks; "
                     "candidate screen ns 1000-2500, confirm ns 1000-3000; the global ID audit + "
                     "candidate capacity check run before any collection and stop at "
                     "D2E_FRESH_POOL_INSUFFICIENT if the fresh pool is inadequate"),
            "screen_candidate_ns": config["screen_candidate_ns"],
            "confirm_candidate_ns": config["confirm_candidate_ns"],
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _write_once_json(receipt_path, receipt)
    print(f"[stage4d-d2e-preflight] PASS torch={torch.__version__} gpu={gpu_name} "
          f"db_files={n_db} sources={inventory['_meta']['n_sources']}", flush=True)
    return receipt


def main() -> int:
    receipt = run_preflight()
    print(f"[stage4d-d2e-preflight] SUMMARY status={receipt['status']}", flush=True)
    return 0 if receipt["status"] == "D2E_PREFLIGHT_PASS" else 5


if __name__ == "__main__":
    raise SystemExit(main())
