"""Repair the Stage 4D-D2 train checkpoint: drop duplicate scenario rows.

The train checkpoint (and finalized groups.npz) carries ONE scenario twice because a
resume re-collected an already-accepted scenario after a checkpoint/ledger desync. The
append-only ledger (the burn-once ground truth) records 4999 UNIQUE accepted scenarios.
This script rebuilds the checkpoint to exactly those 4999 unique groups: for every
duplicated scenario_id it keeps the FIRST (ledgered) occurrence and drops the later
duplicate row(s). Group-axis arrays (first dim == n_rows) are filtered; scalars, canonical
arrays and JSON blobs are carried unchanged. The repair is verified against the backup.

The collector's todo now also excludes already-accepted ids (hardening), so after this
repair a resume collects the ONE remaining unique scenario to reach the 5000 quota.

    sim code/stage4d_repair_train_dup.py            # writes repaired checkpoint
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))
from ac_twin.checkpointing import atomic_save_npz  # noqa: E402

ROLE_DIR = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external/roles/train"
CKPT = ROLE_DIR / "checkpoint.npz"
BACKUP = ROLE_DIR / "_dup_repair_backup_20260923/checkpoint.npz"


def main() -> int:
    if not BACKUP.is_file():
        raise SystemExit(f"backup missing (run the backup step first): {BACKUP}")
    with np.load(CKPT, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    meta = json.loads(str(data["__meta_json__"]))
    sids = [str(m["scenario_id"]) for m in meta]
    n_rows = len(sids)
    counts = Counter(sids)
    dups = {sid for sid, c in counts.items() if c > 1}
    print(f"[repair] rows={n_rows} unique={len(counts)} dup_ids={len(dups)}", flush=True)
    if not dups:
        print("[repair] no duplicates; checkpoint already unique, nothing to do", flush=True)
        return 0

    # Keep the FIRST occurrence of each scenario_id; drop later duplicates.
    seen: set[str] = set()
    keep = np.zeros(n_rows, dtype=bool)
    for i, sid in enumerate(sids):
        if sid in seen:
            continue
        seen.add(sid)
        keep[i] = True
    dropped = np.where(~keep)[0].tolist()
    print(f"[repair] keeping {int(keep.sum())} unique rows; dropping indices {dropped}",
          flush=True)

    # Filter group-axis arrays; carry scalars/canonical/JSON/dunders unchanged.
    out: dict[str, np.ndarray] = {}
    for key, arr in data.items():
        if key == "__meta_json__":
            continue
        if key == "__n_accepted__":
            continue
        if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == n_rows:
            out[key] = arr[keep]
        else:
            out[key] = arr
    out["__n_accepted__"] = np.asarray(int(keep.sum()), dtype=np.int64)
    kept_meta = [m for m, k in zip(meta, keep) if k]
    out["__meta_json__"] = np.asarray(json.dumps(kept_meta))
    atomic_save_npz(CKPT, **out)

    # --- verify against backup: every kept row must equal its source row ---
    with np.load(CKPT, allow_pickle=False) as z:
        new = {k: z[k] for k in z.files}
    with np.load(BACKUP, allow_pickle=False) as z:
        old = {k: z[k] for k in z.files}
    new_meta = json.loads(str(new["__meta_json__"]))
    new_sids = [str(m["scenario_id"]) for m in new_meta]
    assert len(new_sids) == len(set(new_sids)) == int(keep.sum()), "repair not unique"
    assert int(new["__n_accepted__"]) == int(keep.sum())
    keep_idx = np.where(keep)[0]
    for key in ("object_history", "object_factual_future", "object_counterfactual_futures",
                "factual_actions", "inference_mask", "t_star"):
        assert np.array_equal(new[key], old[key][keep_idx]), f"row mismatch in {key}"
    assert str(new["__config_fingerprint__"]) == str(old["__config_fingerprint__"])
    print(f"[repair] OK: checkpoint now {len(new_sids)} unique rows, all kept rows "
          "bit-identical to backup, config fingerprint unchanged", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
