"""READ-ONLY: train/validation t_star distribution vs the frozen stratum split (13).

Sanity check that the frozen family split T_STAR_STRATUM_SPLIT=13 yields non-degenerate
strata (each >=10%) on the already-collected train/validation roles — purely diagnostic,
no writes, does not change the frozen threshold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLES = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external/roles"
SPLIT = 13


def main() -> int:
    for role in ("train", "validation"):
        with np.load(ROLES / role / "groups.npz", allow_pickle=False) as z:
            t = np.asarray(z["t_star"]).ravel()
        lo = float((t < SPLIT).mean())
        hi = float((t >= SPLIT).mean())
        print(f"[{role}] n={t.size} t_star min={t.min()} max={t.max()} "
              f"median={float(np.median(t)):.2f} | <{SPLIT}:{lo:.3f} >={SPLIT}:{hi:.3f} "
              f"| strata>=%10: {lo >= 0.10 and hi >= 0.10}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
