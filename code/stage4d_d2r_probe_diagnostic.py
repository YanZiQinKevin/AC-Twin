"""Read-only probe: dump the Phase 0 diagnostic.npz schema (keys/shapes/dtypes)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

z = np.load(ROOT / "experiments/stage4_reliability/stage4d_d2r/phase0/diagnostic.npz",
            allow_pickle=False)
for k in z.files:
    print(f"{k:40s} {str(z[k].shape):18s} {z[k].dtype}")
