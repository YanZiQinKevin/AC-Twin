"""READ-ONLY deep diagnostic for the single duplicated train scenario 85dfcc7eea4ce69a.

Answers three fix-relevant questions:
  1. Does the train CHECKPOINT (the resume anchor) also carry the duplicate, or only the
     finalized groups.npz? (Determines whether a resume would propagate the defect.)
  2. Are the two duplicated rows the same simulation within the pilot's cross-process
     tolerance (~1e-4), or structurally divergent (different t_star / factual plan)?
  3. Which duplicate row matches the ledger (t_star=13)?

No writes. Structural only.

    sim code/stage4d_diag_dup_detail.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLE_DIR = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external/roles/train"
SID = "85dfcc7eea4ce69a"
ARRAY_KEYS = ("object_history", "object_factual_future", "object_counterfactual_futures")


def main() -> int:
    meta = json.loads((ROLE_DIR / "groups_meta.json").read_text(encoding="utf-8"))
    sids = [str(m["scenario_id"]) for m in meta]
    idx = [i for i, s in enumerate(sids) if s == SID]
    print(f"groups.npz dup indices for {SID}: {idx}", flush=True)
    for i in idx:
        print(f"  meta[{i}]: t_star={meta[i]['t_star']} ego_lane_id={meta[i]['ego_lane_id']} "
              f"eval_track_ids={meta[i]['eval_track_ids']}", flush=True)

    with np.load(ROLE_DIR / "groups.npz", allow_pickle=False) as z:
        npz = {k: z[k] for k in z.files}
    a, b = idx[0], idx[1]
    print(f"t_star rows: {npz['t_star'][a]} vs {npz['t_star'][b]}", flush=True)
    fa_equal = np.array_equal(npz["factual_actions"][a], npz["factual_actions"][b])
    print(f"factual_actions identical: {fa_equal}", flush=True)
    if not fa_equal:
        print(f"  factual_actions[{a}]={npz['factual_actions'][a].tolist()}", flush=True)
        print(f"  factual_actions[{b}]={npz['factual_actions'][b].tolist()}", flush=True)
    for k in ARRAY_KEYS:
        d = np.abs(npz[k][a].astype(np.float64) - npz[k][b].astype(np.float64))
        print(f"  {k}: max_abs_diff={d.max():.6e} mean_abs_diff={d.mean():.6e}", flush=True)

    # Checkpoint internal state (the resume anchor).
    ckpt = ROLE_DIR / "checkpoint.npz"
    if ckpt.is_file():
        with np.load(ckpt, allow_pickle=False) as z:
            n_acc = int(z["__n_accepted__"])
            ck_meta = json.loads(str(z["__meta_json__"]))
        ck_sids = [str(m["scenario_id"]) for m in ck_meta]
        ck_idx = [i for i, s in enumerate(ck_sids) if s == SID]
        print(f"checkpoint __n_accepted__={n_acc} meta_rows={len(ck_sids)} "
              f"unique={len(set(ck_sids))} dup_idx_of_{SID}={ck_idx}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
