"""READ-ONLY diagnostic: characterize duplicate scenario_ids in collected D2 roles.

Task 3's loader guard rejected the train role for duplicate scenario_ids. This diagnoses
the scale and nature of the duplication introduced across supervisor-resume boundaries:
for each role it reports total/unique/duplicate scenario_ids, the duplicate row indices
(and their gaps, to reveal the respawn pattern), and whether duplicated rows are
BIT-IDENTICAL across every array (deterministic re-simulation) or divergent (corruption).

No writes, no outcome adjudication. Pure structural integrity check.

    sim code/stage4d_diag_duplicates.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLES_DIR = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external/roles"
ARRAY_KEYS = ("object_history", "object_factual_future", "object_counterfactual_futures",
              "factual_actions", "inference_mask", "t_star")


def _diag_role(role: str) -> None:
    role_dir = ROLES_DIR / role
    npz_path = role_dir / "groups.npz"
    meta_path = role_dir / "groups_meta.json"
    if not npz_path.is_file() or not meta_path.is_file():
        print(f"[{role}] groups.npz/groups_meta.json not present; skip", flush=True)
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sids = [str(m["scenario_id"]) for m in meta]
    counts = Counter(sids)
    dups = {sid: c for sid, c in counts.items() if c > 1}
    print(f"[{role}] rows={len(sids)} unique={len(counts)} "
          f"dup_ids={len(dups)} extra_dup_rows={sum(c-1 for c in dups.values())}", flush=True)
    if not dups:
        return
    with np.load(npz_path, allow_pickle=False) as z:
        arrs = {k: z[k] for k in ARRAY_KEYS}
    n_rows = arrs["object_history"].shape[0]
    print(f"[{role}] npz rows={n_rows} (meta rows={len(sids)})", flush=True)
    # For each duplicated id, compare all duplicate rows for bit-identity.
    all_identical = True
    shown = 0
    for sid, c in sorted(dups.items()):
        idx = [i for i, s in enumerate(sids) if s == sid]
        base = {k: arrs[k][idx[0]] for k in ARRAY_KEYS}
        identical = all(
            all(np.array_equal(base[k], arrs[k][j]) for k in ARRAY_KEYS) for j in idx[1:])
        all_identical = all_identical and identical
        if shown < 12:
            gaps = [idx[i + 1] - idx[i] for i in range(len(idx) - 1)]
            print(f"[{role}]   dup {sid}: count={c} indices={idx} gaps={gaps} "
                  f"bit_identical={identical}", flush=True)
            shown += 1
    print(f"[{role}] ALL duplicate rows bit-identical: {all_identical}", flush=True)


def main() -> int:
    for role in ("train", "validation"):
        _diag_role(role)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
