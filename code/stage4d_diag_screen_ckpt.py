"""READ-ONLY: screen checkpoint vs ledger counts (diagnose the os.replace failure state)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLE = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external/roles/development_screen"


def main() -> int:
    with np.load(ROLE / "checkpoint.npz", allow_pickle=False) as z:
        n_acc = int(z["__n_accepted__"])
        meta = json.loads(str(z["__meta_json__"]))
    ck_ids = [str(m["scenario_id"]) for m in meta]
    ledger_ids = []
    with (ROLE / "rejection_ledger.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                ledger_ids.append(str(json.loads(line)["scenario_id"]))
    print(f"checkpoint __n_accepted__={n_acc} meta_rows={len(ck_ids)} unique={len(set(ck_ids))}")
    print(f"ledger lines={len(ledger_ids)} unique={len(set(ledger_ids))}")
    in_ledger_not_ck = [s for s in ledger_ids if s not in set(ck_ids)]
    print(f"ledger-not-in-checkpoint (burned but uncheckpointed)={len(in_ledger_not_ck)}")
    dup_ck = len(ck_ids) - len(set(ck_ids))
    print(f"checkpoint duplicate ids={dup_ck}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
