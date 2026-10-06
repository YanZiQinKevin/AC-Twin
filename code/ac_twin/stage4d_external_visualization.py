"""Stage 4D-D2 paper-visualization bypass (plan section 14.2) — NON-adjudicating.

This module implements the preregistered visualization contract. The only step that
must happen *immediately after D0-X qualification* (before any outcome is read) is
freezing the platform candidate pool (section 14.2.A step 1): from the accepted pilot
scenarios, order by ``SHA256(scenario_id || context_id || "platform-visual")``
ascending and take the first ``K_platform=20``. This locks the candidate set; the
selector may later use only render-legibility metrics (never model scores, regret,
oracle action or prediction error). Figure rendering is a separate, non-adjudicating
step that does not affect any gate (section 14.2.D).

    sim code/ac_twin/stage4d_external_visualization.py   # freezes the platform pool once
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

RUN_DIR = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external"
PLATFORM_DIR = RUN_DIR / "platform"
PILOT_DIR = RUN_DIR / "roles" / "pilot"
POOL_PATH = PLATFORM_DIR / "viz_platform_pool.json"

K_PLATFORM = 20
VIZ_SALT = "platform-visual"


def _candidate_hash(scenario_id: str, context_id: str) -> str:
    return hashlib.sha256(f"{scenario_id}||{context_id}||{VIZ_SALT}".encode("utf-8")).hexdigest()


def freeze_platform_pool(pilot_dir: Path = PILOT_DIR, k: int = K_PLATFORM) -> dict:
    """Freeze the K_platform platform-overview candidate pool from accepted pilot groups.

    Create-once. Reads only the pilot groups_meta (scenario_id + t_star context), never
    any outcome. Candidates are ordered by the frozen hash; the pool is the first k.
    """
    if POOL_PATH.is_file():
        raise FileExistsError(f"platform viz pool already frozen: {POOL_PATH}")
    meta = json.loads((pilot_dir / "groups_meta.json").read_text(encoding="utf-8"))
    rows = []
    for m in meta:
        sid = m["scenario_id"]
        cid = str(m["t_star"])  # one t_star context per scenario (plan 4.2)
        rows.append({"scenario_id": sid, "context_id": cid, "hash": _candidate_hash(sid, cid)})
    rows.sort(key=lambda r: r["hash"])
    pool = rows[:k]
    for rank, r in enumerate(pool):
        r["rank"] = rank
    receipt = {
        "plan_id": "AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": "freeze platform-overview candidate pool (legibility-only selection later)",
        "k_platform": k,
        "n_pilot_accepted": len(meta),
        "selection": "SHA256(scenario_id || context_id || 'platform-visual') ascending, first K",
        "pool": pool,
        "pilot_groups_meta_sha256": sha256_file(pilot_dir / "groups_meta.json"),
        "code_sha256": sha256_file(Path(__file__)),
        "status": "VIZ_PLATFORM_POOL_FROZEN",
    }
    write_once_json(POOL_PATH, receipt)
    print(f"[viz_platform_pool] FROZEN | K={k} from {len(meta)} accepted pilot groups", flush=True)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", default=str(PILOT_DIR))
    parser.add_argument("--k", type=int, default=K_PLATFORM)
    args = parser.parse_args(argv)
    freeze_platform_pool(Path(args.pilot_dir), args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
