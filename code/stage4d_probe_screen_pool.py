"""READ-ONLY probe: development_screen pool counts + pre-collection freeze state.

Prints the frozen split-pool counts for the development_screen role (primary/reserve),
the development_screen quota feasibility, and confirms the contract artifacts that §8.1
step 1 requires to be frozen before screen collection. No writes.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))
from ac_twin import stage4d_external_split as split  # noqa: E402

PLATFORM = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external/platform"
DEV = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external/development"
ROLE = "development_screen"
QUOTA = 4000


def main() -> int:
    rows = [r for r in split.load_split_pools(PLATFORM) if r["role"] == ROLE]
    primary = [r for r in rows if r["pool"] == "primary"]
    reserve = [r for r in rows if r["pool"] == "reserve"]
    print(f"[probe] role={ROLE} primary={len(primary)} reserve={len(reserve)} "
          f"total={len(rows)} quota={QUOTA}", flush=True)
    print(f"[probe] quota feasible (total>=quota): {len(rows) >= QUOTA}", flush=True)
    ids = [str(r['scenario_id']) for r in rows]
    print(f"[probe] screen pool unique ids: {len(set(ids)) == len(ids)} "
          f"({len(set(ids))}/{len(ids)})", flush=True)

    # §8.1 step 1 pre-collection freeze state.
    checks = {
        "scenario_split_manifest.json": PLATFORM / "scenario_split_manifest.json",
        "action_contract.json": PLATFORM / "action_contract.json",
        "adapter_contract.json": PLATFORM / "adapter_contract.json",
        "predictor_stack.json": DEV / "base_models" / "predictor_stack.json",
        "final_selection.json": DEV / "final_selection.json",
    }
    for name, path in checks.items():
        print(f"[probe] frozen {name}: {path.is_file()}", flush=True)

    # Existing roles must not overlap the screen pool.
    roles_dir = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external/roles"
    screen_dir = roles_dir / ROLE
    print(f"[probe] screen role dir already exists (should be False pre-collection): "
          f"{screen_dir.exists()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
