"""Throwaway smoke probe for the D2E ego data path (NOT a frozen artifact).

Loads the old D2 validation bundle, assembles ego features/mask/targets, and prints shapes +
sanity values. Run via: run_stage4d_env.ps1 sim code/stage4d_d2e_probe_ego.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage4d_d2e_ego as ego  # noqa: E402
from ac_twin import stage4d_external_inference as INF  # noqa: E402
from ac_twin.stage3_scaler import FeatureScaler  # noqa: E402


def main() -> int:
    t0 = time.time()
    arrays = INF.load_role_arrays(INF.ROLES_DIR / "validation", "validation")
    print(f"load_role_arrays: {time.time() - t0:.1f}s groups={arrays.groups}", flush=True)
    t0 = time.time()
    bundle = INF.build_branch_bundle(arrays)
    print(f"build_branch_bundle: {time.time() - t0:.1f}s branches={bundle.branches}", flush=True)

    t0 = time.time()
    scaler = FeatureScaler.fit(
        {"history": bundle.history, "factual_suffix": bundle.factual_suffix},
        {"history": bundle.history[..., 0] > 0, "factual_suffix": bundle.factual_suffix[..., 0] > 0},
        fitted_on="validation_probe")
    x = ego.assemble_ego_features(bundle, scaler)
    print(f"assemble_ego_features: {time.time() - t0:.1f}s shape={x.shape} "
          f"finite={np.isfinite(x).all()}", flush=True)

    y = ego.ego_targets(bundle)
    m = ego.compute_ego_train_mask(arrays)
    print(f"targets shape={y.shape} mask shape={m.shape} mask_frac={m.mean():.4f}", flush=True)

    # scientific sanity: the ego final-progress span across the 10 actions (the old bug).
    G, A = arrays.groups, arrays.n_actions
    true_fin = y[:, -1, 0].reshape(G, A)                 # true ego final ds per action
    factual_fin = bundle.factual_xy[:, -1, 0, 0]          # (G,)
    hist_x = bundle.history_last_xy[:, 0]                 # (G,)
    progress_true = true_fin + (factual_fin - hist_x)[:, None]   # (G,A)
    span = progress_true.max(axis=1) - progress_true.min(axis=1)
    print(f"true ten-action ego progress span: median={np.median(span):.4f} m "
          f"mean={span.mean():.4f} m (expect O(7 m) per plan)", flush=True)
    print(f"ego target |ds| final-frame: median={np.median(np.abs(y[:, -1, 0])):.4f} m",
          flush=True)
    print("PROBE OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
