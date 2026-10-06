"""ENGINEERING_SMOKE_ONLY: validate the Stage 4D-D2 inference data path on REAL collected data.

Reads the in-progress train ``checkpoint.npz`` (intermediate collector state — NOT the pilot,
screen or confirm, and no outcome is adjudicated) and verifies that the real MetaDrive/Waymo
Map-Frenet arrays flow through ``stage4d_external_inference`` correctly: schema, both masks,
branch expansion, scaler, and a randomly-initialized forward pass produce finite tensors.

This is an engineering plumbing check only. It trains nothing to convergence, fits no model
that is persisted, and reads no regret/oracle/outcome. It never touches a one-shot resource.

    sim code/stage4d_smoke_real_data.py [--groups 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from ac_twin import stage4d_external_inference as INF  # noqa: E402

RUN_DIR = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external"
CHECKPOINT = RUN_DIR / "roles" / "train" / "checkpoint.npz"


def _load_checkpoint_role(npz_path: Path, max_groups: int) -> INF.RoleArrays:
    with np.load(npz_path, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    meta = json.loads(str(data["__meta_json__"]))
    scenario_ids = tuple(str(m["scenario_id"]) for m in meta)
    n = min(int(max_groups), len(scenario_ids))
    group_seed = np.asarray(
        [INF.T.group_seed_from_scenario_id(sid, INF.ROLE_SALT) for sid in scenario_ids[:n]],
        dtype=np.int64)
    return INF.RoleArrays(
        role="train_smoke", groups=n, n_actions=int(data["n_actions"]),
        scenario_ids=scenario_ids[:n], group_seed=group_seed,
        object_history=data["object_history"][:n].astype(np.float32),
        object_factual_future=data["object_factual_future"][:n].astype(np.float32),
        object_counterfactual_futures=data["object_counterfactual_futures"][:n].astype(np.float32),
        factual_actions=data["factual_actions"][:n].astype(np.int64),
        inference_mask=data["inference_mask"][:n].astype(bool),
        t_star=data["t_star"][:n].astype(np.int64),
        cf_actions=data["cf_actions"].astype(np.int64), cf_durations=data["cf_durations"].astype(np.int64),
        data_sha256="smoke")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=8)
    args = parser.parse_args(argv)
    if not CHECKPOINT.is_file():
        print(f"[smoke] no train checkpoint yet: {CHECKPOINT}", flush=True)
        return 0

    arrays = _load_checkpoint_role(CHECKPOINT, args.groups)
    print(f"[smoke] loaded {arrays.groups} real groups x {arrays.n_actions} actions", flush=True)

    # Real-data schema sanity: finite arrays, channel semantics, both masks.
    for name in ("object_history", "object_factual_future", "object_counterfactual_futures"):
        arr = getattr(arrays, name)
        assert np.isfinite(arr).all(), f"non-finite {name}"
        print(f"[smoke] {name}{arr.shape} finite OK", flush=True)
    # STRUCTURAL invariant: slot 0 is ego everywhere (adapter construction). Hard assert.
    hist = arrays.object_history
    assert (hist[:, :, 0, 1] > 0).all() and (hist[:, :, 1:, 1] <= 0).all(), "is_ego slot-0 contract"
    # DIAGNOSTIC (real-data property, not a bug): presence fullness and (s,d) range.
    print(f"[smoke] history presence fullness={float((hist[..., 0] > 0).mean()):.3f}", flush=True)
    pos_sd = arrays.object_factual_future[..., 2:4]
    print(f"[smoke] (s,d) range s=[{pos_sd[...,0].min():.2f},{pos_sd[...,0].max():.2f}] "
          f"d=[{pos_sd[...,1].min():.2f},{pos_sd[...,1].max():.2f}]", flush=True)

    bundle = INF.build_branch_bundle(arrays)
    E = bundle.branches
    print(f"[smoke] branches E={E}; eval_valid_frac={bundle.valid_eval.mean():.3f} "
          f"infer_valid_frac={bundle.valid_infer.mean():.3f}", flush=True)
    # STRUCTURAL invariants: finite targets, eval mask excludes ego (compute_effect_eval_mask).
    assert np.isfinite(bundle.effect_target).all(), "non-finite effect target"
    assert np.isfinite(bundle.r_true).all(), "non-finite response truth"
    assert bundle.valid_eval[..., 0].sum() == 0, "eval mask must exclude ego"
    # DIAGNOSTIC (data property; the train-support gate owns sufficiency, not this smoke).
    nonzero_effect = float((np.linalg.norm(bundle.effect_target, axis=-1) > 1e-6)
                           [bundle.valid_eval].mean())
    print(f"[smoke] nonzero-effect fraction over valid cells = {nonzero_effect:.3f}", flush=True)

    # Forward-pass plumbing with randomly-initialized models (no training, nothing persisted).
    scaler = INF.fit_scaler(bundle, Path(tempfile_dir()) / "smoke_scaler.json")
    scaled = INF.scaled_model_tensors(bundle, scaler)
    assert np.isfinite(scaled["history"].numpy()).all(), "scaled history non-finite"
    torch.manual_seed(0)
    response = INF.s3r.Stage3ResponsePredictor(
        hidden_dim=INF.RESPONSE_HIDDEN_DIM, response_feature_dim=INF.RESPONSE_FEATURE_DIM).eval()
    expert = INF.Stage3RExpert(condition="full", hidden_dim=INF.EXPERT_HIDDEN_DIM,
                               object_embedding_dim=INF.EXPERT_OBJECT_EMBEDDING_DIM).eval()
    idx = np.arange(min(E, 16))
    with torch.no_grad():
        out = response(**{k: scaled[k][idx] for k in (
            "history", "factual_suffix", "factual_actions", "counterfactual_actions",
            "cf_action", "cf_duration", "t_star")})
        eff = expert(history=scaled["history"][idx], factual_suffix=scaled["factual_suffix"][idx],
                     factual_actions=scaled["factual_actions"][idx],
                     counterfactual_actions=scaled["counterfactual_actions"][idx])
    assert np.isfinite(out["log1p_r_hat"].numpy()).all(), "response output non-finite"
    assert np.isfinite(out["regime_logits"].numpy()).all(), "regime logits non-finite"
    assert np.isfinite(eff.numpy()).all(), "expert effect non-finite"
    print(f"[smoke] forward pass finite: r_hat{tuple(out['log1p_r_hat'].shape)} "
          f"effect{tuple(eff.shape)}", flush=True)
    print("[smoke] ENGINEERING_SMOKE_ONLY PASS: real data flows through the D2 inference path",
          flush=True)
    return 0


def tempfile_dir() -> str:
    import tempfile
    return tempfile.mkdtemp(prefix="stage4d_smoke_")


if __name__ == "__main__":
    raise SystemExit(main())
