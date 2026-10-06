"""Read-only integrity check of the confirm collection checkpoint + ledger after the reap."""
import json
import sys
from pathlib import Path

import numpy as np

role = Path("experiments/stage4_reliability/stage4d_d2f/roles/confirm")
ckpt = role / "checkpoint.npz"
ledger = role / "rejection_ledger.jsonl"

with np.load(ckpt, allow_pickle=False) as z:
    keys = sorted(z.files)
    accepted = int(z["accepted_count"]) if "accepted_count" in z.files else None
    print("checkpoint keys:", keys)
    print("checkpoint accepted_count:", accepted)

lines = ledger.read_bytes().splitlines()
print("ledger lines:", len(lines))
ok, bad = 0, 0
for i, raw in enumerate(lines):
    try:
        rec = json.loads(raw)
        ok += 1
        last = rec
    except json.JSONDecodeError:
        bad += 1
        print(f"torn line at {i}: {raw[:80]!r}")
print(f"ledger parsed ok={ok} torn={bad}")
print("last record:", {k: last.get(k) for k in ("outcome", "scenario_id", "reason")})
accepted_ids = sum(1 for raw in lines if b'"accepted"' in raw)
print("ledger accepted records:", accepted_ids)
