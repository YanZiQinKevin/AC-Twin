"""READ-ONLY: dump keys/shapes/dtypes of D2 train/validation npz artifacts.

D2R Phase 0 must rebuild J_hat/J_true/a0/regret/Direct+baseline risk scores from the
frozen D2 predictions + stage4c_objective, aligned by scenario id / group seed / action
slot. This probe only reports the on-disk schema of the ALLOWED roles (train,
validation) and their development prediction/score npz files. It never touches the old
4,000-group screen (forbidden by plan 2.2). No writes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "experiments/stage4_reliability/stage4d_external"

TARGETS = [
    "roles/train/groups.npz",
    "roles/validation/groups.npz",
    "development/train_predictions.npz",
    "development/validation_predictions.npz",
    "development/train_scores.npz",
    "development/validation_scores.npz",
]


def dump(rel: str) -> None:
    p = DEV / rel
    print(f"\n===== {rel} =====")
    if not p.exists():
        print("  MISSING")
        return
    with np.load(p, allow_pickle=False) as z:
        for k in z.files:
            a = z[k]
            print(f"  {k:40s} shape={str(a.shape):22s} dtype={a.dtype}")


def main() -> int:
    for rel in TARGETS:
        dump(rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
