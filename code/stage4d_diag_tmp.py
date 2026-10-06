"""READ-ONLY: inspect the stale checkpoint.npz.tmp left by the reaped worker.

The reap killed the worker mid-checkpoint-write, so atomic_save_npz's finally-unlink
never ran and left checkpoint.npz.tmp. The durable checkpoint.npz (3998) is intact.
This reports the .tmp size/mtime and whether it loads as a valid npz (complete vs
partial), to decide delete-vs-recover. No writes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLE = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external/roles/development_screen"
TMP = ROLE / "checkpoint.npz.tmp"
CKPT = ROLE / "checkpoint.npz"


def main() -> int:
    import os
    for p in (CKPT, TMP):
        if p.exists():
            st = p.stat()
            print(f"{p.name}: {st.st_size} bytes, mtime={st.st_mtime:.0f}")
        else:
            print(f"{p.name}: MISSING")
    if TMP.exists():
        try:
            with np.load(TMP, allow_pickle=False) as z:
                n = int(z["__n_accepted__"])
                meta = json.loads(str(z["__meta_json__"]))
            ids = [str(m["scenario_id"]) for m in meta]
            print(f".tmp LOADS OK: __n_accepted__={n} unique={len(set(ids))} "
                  f"dup={len(ids)-len(set(ids))}")
        except Exception as exc:  # noqa: BLE001
            print(f".tmp is PARTIAL/CORRUPT (will not load): {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
