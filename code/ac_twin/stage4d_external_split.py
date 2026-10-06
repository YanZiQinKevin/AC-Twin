"""Stage 4D-D2 scenario split protocol: role pools over the raw scenario index.

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` sections 3.3
(steps 2-3), 4.2, 4.3 and 12.3 (Task 2 item 1). This module freezes, BEFORE any
referenced shard is converted and before any counterfactual rollout or training,
the complete scenario-role assignment:

* index-level pre-filter (plan 4.2 items 1-2 only: non-degenerate track/map/time
  existence and 5 Hz history/future capacity -- nothing lane- or outcome-level,
  which requires converted data and is checked at collection time);
* deterministic hash ranking by the frozen role salt (plan 4.3);
* contiguous rank segments per role with uniform oversubscription: every
  surviving scenario ID belongs to exactly one role pool; the first ``quota``
  IDs of a pool are its primary list, the remainder its frozen reserve list
  (plan 4.3: reserves fill in hash-rank order, attempted IDs are burned);
* the frozen ``t_star`` candidate-order RULE (plan 4.2). The concrete per-
  scenario ``t_star`` cannot be bound here because eligibility needs converted
  lane data; the rule is frozen now and the chosen value is recorded per
  scenario in the collection receipts.

One scenario ID contributes at most one scenario-context group to exactly one
role (plan Global Constraint 5). Pools never overlap by construction (disjoint
rank segments) and this is pinned by blocking tests.

The pools file (``scenario_split_pools.jsonl``) carries the full per-scenario
assignment; the manifest (``scenario_split_manifest.json``) binds it by content
hash together with every rule constant and the generating code hash.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json
from ac_twin.stage4d_external_protocol import role_rank_key

# --- frozen split constants (plan sections 3.3, 4.2, 4.3) -------------------------

#: Index-level pre-filter (plan 4.2 items 1-2). num_timesteps >= 70 guarantees the
#: worst-case 5 Hz window: t_star=15 -> native frame 30, plus 20 future policy steps
#: at 2 native frames each -> native frame 69 must exist (0-indexed length >= 70).
PREFILTER_MIN_TIMESTEPS = 70
PREFILTER_MIN_TRACKS = 5  # ego + at least 4 non-ego candidates (plan 4.2 item 5)
PREFILTER_MIN_MAP_FEATURES = 10  # non-degenerate map

#: Role order fixes the contiguous rank-segment layout (plan 4.3 table).
ROLE_ORDER = ("smoke", "pilot", "train", "validation", "development_screen", "confirm")

#: The confirm quota is the LARGEST candidate n (plan 4.3: n in {1000, 2000, 3000}
#: chosen later by the mechanical power rule; the pool must absorb the maximum).
CONFIRM_POOL_QUOTA = 3000

#: Frozen t_star candidate-order rule (plan 4.2). The candidate 5 Hz indices are
#: ordered per scenario by this hash; the first eligible candidate wins.
T_STAR_RULE_VERSION = "t_star_hash_order_v1"

POOLS_FILE = "scenario_split_pools.jsonl"
MANIFEST_FILE = "scenario_split_manifest.json"


def t_star_candidate_order(scenario_id: str, role_salt: str, candidates: Sequence[int]) -> list[int]:
    """Frozen per-scenario t_star candidate order (plan 4.2): ascending SHA256 of
    ``role_salt + scenario_id + "|t_star|" + str(t)``. Pure and deterministic."""
    if not candidates:
        raise ValueError("t_star candidates must be non-empty")
    keyed = [
        (
            hashlib.sha256(
                (role_salt + scenario_id + "|t_star|" + str(int(t))).encode("utf-8")
            ).hexdigest(),
            int(t),
        )
        for t in candidates
    ]
    keyed.sort()
    return [t for _key, t in keyed]


def load_scenario_index(index_path: str | Path) -> list[dict[str, Any]]:
    """Load the Task-1 raw scenario index; scenario IDs must be globally unique."""
    records: list[dict[str, Any]] = []
    with open(index_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"empty scenario index: {index_path}")
    ids = [r["scenario_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("scenario index contains duplicate scenario IDs")
    return records


def passes_index_prefilter(record: Mapping[str, Any]) -> bool:
    """Plan 4.2 items 1-2 at index level: non-degenerate tracks/map/time and the
    5 Hz history/future capacity. Reads nothing beyond per-scenario counts."""
    if int(record.get("num_timesteps", 0)) < PREFILTER_MIN_TIMESTEPS:
        return False
    if int(record.get("num_tracks", 0)) < PREFILTER_MIN_TRACKS:
        return False
    if int(record.get("num_map_features", 0)) < PREFILTER_MIN_MAP_FEATURES:
        return False
    if record.get("sdc_track_id") is None:
        return False
    return True


def assign_role_pools(
    records: Sequence[Mapping[str, Any]],
    role_salt: str,
    role_targets: Mapping[str, int],
    *,
    confirm_quota: int = CONFIRM_POOL_QUOTA,
) -> list[dict[str, Any]]:
    """Hash-rank pre-filter survivors and cut contiguous per-role segments.

    Uniform oversubscription: pool sizes are proportional to role quotas with the
    common factor ``floor(N_survivors * quota_r / total_quota)``; the leftover
    survivors extend the confirm reserve (the tail with the largest uncertainty).
    Returns one row per surviving scenario, ordered by global rank.
    """
    survivors = [r for r in records if passes_index_prefilter(r)]
    if not survivors:
        raise ValueError("index pre-filter left no survivors")
    quotas: dict[str, int] = {role: int(role_targets[role]) for role in ROLE_ORDER[:-1]}
    quotas["confirm"] = int(confirm_quota)
    total_quota = sum(quotas.values())
    n_survivors = len(survivors)
    if n_survivors < total_quota:
        raise ValueError(
            f"only {n_survivors} pre-filter survivors for total quota {total_quota}; "
            "pools cannot cover role targets (stop and report)"
        )

    ranked = sorted(
        survivors,
        key=lambda r: (role_rank_key(r["scenario_id"], role_salt), r["scenario_id"]),
    )

    pool_sizes: dict[str, int] = {}
    assigned = 0
    for role in ROLE_ORDER:
        if role == "confirm":
            pool_sizes[role] = n_survivors - assigned  # remainder extends confirm reserve
        else:
            pool_sizes[role] = (n_survivors * quotas[role]) // total_quota
            assigned += pool_sizes[role]
    for role in ROLE_ORDER:
        if pool_sizes[role] < quotas[role]:
            raise ValueError(
                f"pool for role {role} ({pool_sizes[role]}) smaller than its quota "
                f"({quotas[role]}); stop and report"
            )

    rows: list[dict[str, Any]] = []
    cursor = 0
    for role in ROLE_ORDER:
        size = pool_sizes[role]
        for offset in range(size):
            record = ranked[cursor + offset]
            rows.append(
                {
                    "scenario_id": record["scenario_id"],
                    "source_shard": record["source_shard"],
                    "global_rank": cursor + offset,
                    "role": role,
                    "pool": "primary" if offset < quotas[role] else "reserve",
                }
            )
        cursor += size
    return rows


def _split_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for role in ROLE_ORDER:
        role_rows = [r for r in rows if r["role"] == role]
        primary = sum(1 for r in role_rows if r["pool"] == "primary")
        stats[role] = {
            "pool_size": len(role_rows),
            "primary": primary,
            "reserve": len(role_rows) - primary,
            "rank_lo": role_rows[0]["global_rank"] if role_rows else None,
            "rank_hi": role_rows[-1]["global_rank"] if role_rows else None,
        }
    return stats


def _write_pools_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Create-once write of the pools file (atomic via tmp + no-clobber)."""
    if path.exists():
        raise FileExistsError(f"scenario split pools already exist: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    if path.exists():
        tmp.unlink(missing_ok=True)
        raise FileExistsError(f"scenario split pools already exist: {path}")
    tmp.replace(path)


def build_scenario_split_manifest(
    config: Mapping[str, Any], *, code_self_path: str | Path
) -> dict[str, Any]:
    """Freeze the scenario-role split (create-once; plan 3.3 step 3, 4.3).

    Writes ``platform/scenario_split_pools.jsonl`` and
    ``platform/scenario_split_manifest.json``. Both are create-once; a second call
    raises ``FileExistsError``.
    """
    platform_dir = Path(config["run_dir"]) / "platform"
    if not platform_dir.is_absolute():
        platform_dir = Path(__file__).resolve().parents[2] / platform_dir
    index_path = platform_dir / "raw_scenario_index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"raw scenario index missing: {index_path}")

    role_salt = str(config["role_salt"])
    role_targets = {k: int(v) for k, v in dict(config["role_targets"]).items()}
    records = load_scenario_index(index_path)
    rows = assign_role_pools(records, role_salt, role_targets)

    pools_path = platform_dir / POOLS_FILE
    _write_pools_jsonl(pools_path, rows)

    manifest = {
        "plan_id": config["plan_id"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "role_salt": role_salt,
        "source_index_file": "raw_scenario_index.jsonl",
        "source_index_sha256": sha256_file(index_path),
        "source_index_records": len(records),
        "prefilter": {
            "rule": "num_timesteps>=70 AND num_tracks>=5 AND num_map_features>=10 "
                    "AND sdc_track_id present (plan 4.2 items 1-2 at index level)",
            "min_timesteps": PREFILTER_MIN_TIMESTEPS,
            "min_tracks": PREFILTER_MIN_TRACKS,
            "min_map_features": PREFILTER_MIN_MAP_FEATURES,
            "survivors": len(rows),
        },
        "role_order": list(ROLE_ORDER),
        "role_quotas": {**role_targets, "confirm": CONFIRM_POOL_QUOTA},
        "oversubscription_factor": len(rows) / sum(
            [int(role_targets[r]) for r in ROLE_ORDER[:-1]] + [CONFIRM_POOL_QUOTA]
        ),
        "role_stats": _split_stats(rows),
        "t_star_rule": {
            "version": T_STAR_RULE_VERSION,
            "candidates": [int(t) for t in config["t_star_candidates"]],
            "order": "ascending sha256(role_salt + scenario_id + '|t_star|' + str(t))",
            "selection": "first candidate passing plan-4.2 eligibility on converted "
                         "data at collection time; chosen t_star recorded in the "
                         "role collection receipt",
        },
        "pools_file": POOLS_FILE,
        "pools_file_sha256": sha256_file(pools_path),
        "pools_rows": len(rows),
        "code_sha256": sha256_file(code_self_path),
        "status": "SCENARIO_SPLIT_FROZEN",
    }
    write_once_json(platform_dir / MANIFEST_FILE, manifest)
    return manifest


def load_split_pools(platform_dir: str | Path) -> list[dict[str, Any]]:
    """Load the frozen pools file (read-only)."""
    rows: list[dict[str, Any]] = []
    with open(Path(platform_dir) / POOLS_FILE, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("empty scenario split pools file")
    return rows


def verify_scenario_split(config: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive the split from the frozen index and check it against the manifest.

    Blocking invariants: manifest/pools hash binding, zero overlap between roles,
    primary/reserve disjointness, contiguous global ranks, quota coverage.
    """
    platform_dir = Path(config["run_dir"]) / "platform"
    if not platform_dir.is_absolute():
        platform_dir = Path(__file__).resolve().parents[2] / platform_dir
    manifest_path = platform_dir / MANIFEST_FILE
    pools_path = platform_dir / POOLS_FILE
    if not manifest_path.is_file() or not pools_path.is_file():
        raise FileNotFoundError("scenario split manifest/pools missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(pools_path) != manifest["pools_file_sha256"]:
        raise ValueError("pools file hash drift vs manifest")
    index_path = platform_dir / manifest["source_index_file"]
    if sha256_file(index_path) != manifest["source_index_sha256"]:
        raise ValueError("raw scenario index hash drift vs manifest")

    rows = load_split_pools(platform_dir)
    if len(rows) != manifest["pools_rows"]:
        raise ValueError("pools row count drift vs manifest")
    ids = [r["scenario_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("scenario ID appears in more than one pool row")
    if [r["global_rank"] for r in rows] != list(range(len(rows))):
        raise ValueError("global ranks are not contiguous 0..N-1 in row order")
    # re-derive ranks from the frozen salt
    for row in rows:
        expected_rank_key = role_rank_key(row["scenario_id"], str(config["role_salt"]))
        row["_rank_key"] = expected_rank_key
    if [r["_rank_key"] for r in rows] != sorted(r["_rank_key"] for r in rows):
        raise ValueError("pool rows are not in frozen hash-rank order")
    role_targets = {k: int(v) for k, v in dict(config["role_targets"]).items()}
    stats = _split_stats(rows)
    if stats != manifest["role_stats"]:
        raise ValueError("role stats drift vs manifest")
    for role in ROLE_ORDER:
        quota = CONFIRM_POOL_QUOTA if role == "confirm" else role_targets[role]
        if stats[role]["primary"] != quota:
            raise ValueError(f"role {role} primary count {stats[role]['primary']} != quota {quota}")
    return {"rows": len(rows), "roles": stats, "status": "SPLIT_VERIFIED"}


__all__ = [
    "CONFIRM_POOL_QUOTA",
    "MANIFEST_FILE",
    "POOLS_FILE",
    "PREFILTER_MIN_MAP_FEATURES",
    "PREFILTER_MIN_TIMESTEPS",
    "PREFILTER_MIN_TRACKS",
    "ROLE_ORDER",
    "T_STAR_RULE_VERSION",
    "assign_role_pools",
    "build_scenario_split_manifest",
    "load_scenario_index",
    "load_split_pools",
    "passes_index_prefilter",
    "t_star_candidate_order",
    "verify_scenario_split",
]
