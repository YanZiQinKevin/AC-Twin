"""ENGINEERING_SMOKE_ONLY: prove placeholder-CF scoring == real-CF scoring on REAL screen data.

The screen scoring bundle fills the counterfactual-future slot with a placeholder (the
factual future tiled across the ten actions) because the legal inputs carry no CF. The
frozen scoring functions must be INDEPENDENT of that slot. This smoke builds two RoleArrays
from the SAME real screen groups.npz subset — one with the real CF futures, one with the
placeholder — runs the frozen stack scoring on both, and asserts BIT-IDENTICAL predictions
and risk scores. Any difference means a CF-derived field leaks into scoring (a bug).

Read-only against the collected roles + frozen stack; no freeze / lock / adjudication.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))
from ac_twin import stage4d_external_inference as INF  # noqa: E402

DEV = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external/development"
ROLE_DIR = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external/roles/development_screen"
N_SUBSET = 16
DEVICE = "cuda"
BATCH = 32


def _role_arrays(data: dict, gidx: np.ndarray, cf: np.ndarray, sha: str) -> INF.RoleArrays:
    meta = json.loads((ROLE_DIR / "groups_meta.json").read_text(encoding="utf-8"))
    sids = tuple(str(meta[int(i)]["scenario_id"]) for i in gidx)
    return INF.RoleArrays(
        role="development_screen", groups=int(gidx.size),
        n_actions=int(data["cf_actions"].shape[0]),
        scenario_ids=sids,
        group_seed=_seeds(sids),
        object_history=np.asarray(data["object_history"])[gidx].astype(np.float32),
        object_factual_future=np.asarray(data["object_factual_future"])[gidx].astype(np.float32),
        object_counterfactual_futures=cf,
        factual_actions=np.asarray(data["factual_actions"])[gidx].astype(np.int64),
        inference_mask=np.asarray(data["inference_mask"])[gidx].astype(bool),
        t_star=np.asarray(data["t_star"])[gidx].astype(np.int64),
        cf_actions=np.asarray(data["cf_actions"]).astype(np.int64),
        cf_durations=np.asarray(data["cf_durations"]).astype(np.int64),
        data_sha256=sha,
    )


def _seeds(sids) -> np.ndarray:
    from ac_twin import stage4d_external_training as T
    return np.asarray([T.group_seed_from_scenario_id(s, INF.ROLE_SALT) for s in sids],
                      dtype=np.int64)


def main() -> int:
    with np.load(ROLE_DIR / "groups.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    gidx = np.arange(N_SUBSET, dtype=np.int64)
    real_cf = np.asarray(data["object_counterfactual_futures"])[gidx].astype(np.float32)
    factual = np.asarray(data["object_factual_future"])[gidx].astype(np.float32)
    placeholder_cf = np.repeat(factual[:, None], real_cf.shape[1], axis=1).astype(np.float32)

    stack = INF.PredictorStack.from_dict(json.loads(
        (DEV / "base_models" / "predictor_stack.json").read_text(encoding="utf-8")))

    real = INF.build_branch_bundle(_role_arrays(data, gidx, real_cf, "real"))
    holder = INF.build_branch_bundle(_role_arrays(data, gidx, placeholder_cf, "placeholder"))

    preds_real = INF.run_pair_predictions(stack, real, device=DEVICE, batch_size=BATCH)
    preds_holder = INF.run_pair_predictions(stack, holder, device=DEVICE, batch_size=BATCH)

    worst = 0.0
    ok = True
    for key in ("e2e", "full", "history", "r_hat", "presence_prob", "high_prob"):
        a, b = preds_real[key], preds_holder[key]
        identical = np.array_equal(a, b)
        diff = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
        worst = max(worst, diff)
        ok &= identical
        print(f"[smoke] predictions[{key}] bit-identical={identical} max_abs_diff={diff:.3e}",
              flush=True)

    risk_real = INF.score_action_risks(stack, real, preds_real)
    risk_holder = INF.score_action_risks(stack, holder, preds_holder)
    for key in risk_real:
        a, b = risk_real[key], risk_holder[key]
        identical = np.array_equal(a, b)
        diff = float(np.abs(a - b).max())
        ok &= identical
        print(f"[smoke] risk[{key}] bit-identical={identical} max_abs_diff={diff:.3e}", flush=True)

    print(f"[smoke] placeholder-CF scoring == real-CF scoring: {ok} (worst diff {worst:.3e})",
          flush=True)
    print(f"[smoke] {'PASSED' if ok else 'FAILED'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
