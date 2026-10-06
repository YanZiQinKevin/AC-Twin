"""Freeze the Stage 4D-D2 adapter contract (create-once), post-D0-X qualification.

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` section 6.3:
``METADRIVE_PLATFORM_QUALIFIED`` -> freeze environment/data/action/adapter contract
before train/validation. The environment lock, scenario-split manifest and action
contract are already frozen; this binds the adapter stack (Map-Frenet / track-ID /
capacity / mask adapter, deterministic meta-controller, 11-branch replay) and the
frozen task contract by content hash so any later code change is detectable.

    sim code/stage4d_freeze_adapter_contract.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402
from ac_twin import stage4c_objective as obj  # noqa: E402
from ac_twin import stage4c_protocol as protocol  # noqa: E402
from ac_twin import stage4d_external_features as F  # noqa: E402
import stage4d_metadrive_adapter as A  # noqa: E402
import collect_stage4d_external as C  # noqa: E402

RUN_DIR = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external"
PLATFORM_DIR = RUN_DIR / "platform"
ADAPTER_CONTRACT_PATH = PLATFORM_DIR / "adapter_contract.json"

CODE_FILES = [
    "code/stage4d_metadrive_adapter.py",
    "code/ac_twin/stage4d_external_features.py",
    "code/collect_stage4d_external.py",
    "code/stage4d_action_contract.py",
    "code/stage4d_pilot_adjudicate.py",
    "code/ac_twin/stage4c_objective.py",
    "code/ac_twin/stage4c_protocol.py",
]


def freeze_adapter_contract() -> dict:
    if ADAPTER_CONTRACT_PATH.is_file():
        raise FileExistsError(f"adapter contract already frozen: {ADAPTER_CONTRACT_PATH}")
    action_contract = PLATFORM_DIR / "action_contract.json"
    if not action_contract.is_file():
        raise FileNotFoundError("action contract must be frozen before the adapter contract")
    receipt = {
        "plan_id": "AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": "bind adapter stack + frozen task contract by hash after D0-X qualification",
        "code_sha256": {p: sha256_file(PROJECT_ROOT / p) for p in CODE_FILES},
        "action_contract_sha256": sha256_file(action_contract),
        "task_contract": {
            "action_plans": [list(p) for p in A.ACTION_PLANS],
            "idle_plan_index": obj.IDLE_PLAN_INDEX,
            "cf_actions": list(protocol.CF_ACTIONS),
            "object_capacity": F.OBJECT_CAPACITY,
            "n_object_features": F.N_OBJECT_FEATURES,
            "feature_names": list(F.FEATURE_NAMES),
            "history_len": F.HISTORY_LEN,
            "future_len": F.FUTURE_LEN,
            "t_star_candidates": list(F.T_STAR_CANDIDATES),
            "native_per_policy": F.NATIVE_PER_POLICY,
            "action_weights": list(F.ACTION_WEIGHTS),
            "role_salt": C.ROLE_SALT,
            "native_dt": A.NATIVE_DT,
        },
        "status": "ADAPTER_CONTRACT_FROZEN",
    }
    write_once_json(ADAPTER_CONTRACT_PATH, receipt)
    print(f"[adapter_contract] FROZEN | files={len(CODE_FILES)} | "
          f"action_contract={receipt['action_contract_sha256'][:12]}…", flush=True)
    return receipt


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    freeze_adapter_contract()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
