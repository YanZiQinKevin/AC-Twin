"""Throwaway D2E scientific probe (NOT a frozen artifact, no gate, no report).

Answers the make-or-break question cheaply before building fit-risk: does replacing e2e slot 0
with the trained ego branch fix the base selector on old validation? Compares a0/IDLE/RANDOM
mean regret for: old D2 (untrained slot 0), kinematic floor, both ego candidates, truth oracle.

Run via: run_stage4d_env.ps1 sim code/stage4d_d2e_probe_repair.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

import torch  # noqa: E402

from ac_twin import stage4d_d2e_ego as ego  # noqa: E402
from ac_twin import stage4d_external_inference as INF  # noqa: E402
from ac_twin import stage4d_external_training as T  # noqa: E402
from ac_twin import stage4cp_features as s4cpf  # noqa: E402
from ac_twin.stage3_scaler import FeatureScaler  # noqa: E402

DEV = ROOT / "experiments/stage4_reliability/stage4d_external/development"
EGO_MODEL_DIR = ego.D2E_DIR / "development/ego_models"


def _load_ego_models(candidate: str) -> dict[int, ego.EgoEffectHead]:
    models = {}
    for seed in ego.EGO_SEEDS:
        payload = torch.load(EGO_MODEL_DIR / f"ego_{candidate}_seed{seed}.pt",
                             map_location="cpu")
        model = ego.EgoEffectHead()
        model.load_state_dict(payload["state_dict"])
        models[seed] = model
    return models


def _kinematic_ego_delta(bundle: INF.BranchBundle) -> np.ndarray:
    """No-learning legal kinematic floor: constant-velocity ego extrapolation (E,T,2)."""
    dt = 0.2  # 5 Hz
    v = bundle.history[:, -1, 0, 4:6].astype(np.float64)     # (E,2) last-frame (v_s, v_d)
    steps = np.arange(1, ego.FUTURE_LEN + 1, dtype=np.float64) * dt  # (T,)
    delta = v[:, None, :] * steps[None, :, None]              # (E,T,2)
    return delta.astype(np.float32)


def _variant_j_hat(bundle, e2e_s_e_t_n2, action_risks):
    predictions = {"e2e": e2e_s_e_t_n2}
    direct = INF.build_direct_inputs(bundle, predictions, action_risks)
    return np.asarray(direct["j_hat"], dtype=np.float64)


def _regrets(bundle, j_hat):
    labels = T.compute_true_labels(bundle.counterfactual_xy, bundle.history_last_xy,
                                   bundle.inference_mask, j_hat)
    return labels


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    t0 = time.time()
    bundle = INF.build_branch_bundle(INF.load_role_arrays(INF.ROLES_DIR / "validation",
                                                          "validation"))
    with np.load(DEV / "validation_predictions.npz", allow_pickle=False) as z:
        e2e_old = np.asarray(z["e2e"], dtype=np.float32)          # (S,E,T,N,2)
        group_seed = np.asarray(z["group_seed"], dtype=np.int64)
    with np.load(DEV / "validation_scores.npz", allow_pickle=False) as z:
        action_risks = {k: np.asarray(z[k], dtype=np.float64) for k in z.files}
    assert np.array_equal(group_seed, bundle.group_seed), "group_seed misalignment"
    print(f"loaded validation e2e {e2e_old.shape} in {time.time() - t0:.1f}s", flush=True)

    scaler = FeatureScaler.load(EGO_MODEL_DIR / "ego_scaler.json")
    feats = ego.assemble_ego_features(bundle, scaler)               # (E,1002)

    variants: dict[str, np.ndarray] = {}
    variants["OLD_D2_untrained_ego"] = e2e_old
    kin = _kinematic_ego_delta(bundle)                              # (E,T,2)
    variants["KINEMATIC_floor"] = ego.repair_e2e_ego(
        e2e_old, np.broadcast_to(kin[None], (ego.N_SEEDS,) + kin.shape))
    for cand in ego.CANDIDATES:
        models = _load_ego_models(cand)
        deltas = np.stack([ego.predict_ego_effect(models[s], feats, device=device)
                           for s in ego.EGO_SEEDS])                 # (S,E,T,2)
        variants[cand] = ego.repair_e2e_ego(e2e_old, deltas)
    # truth oracle (explanation upper bound; NOT deployable)
    true_ego = np.asarray(bundle.effect_target[:, :, 0, :], dtype=np.float32)  # (E,T,2)
    variants["TRUTH_ORACLE_upper_bound"] = ego.repair_e2e_ego(
        e2e_old, np.broadcast_to(true_ego[None], (ego.N_SEEDS,) + true_ego.shape))

    print(f"\n{'variant':30s} {'a0_regret':>10s} {'idle':>10s} {'random':>10s} "
          f"{'a0<=idle':>8s} {'rand-a0':>8s}", flush=True)
    for name, e2e_v in variants.items():
        j_hat = _variant_j_hat(bundle, e2e_v, action_risks)
        labels = _regrets(bundle, j_hat)
        a0, idle, rnd = (labels["base_mean_regret"], labels["idle_mean_regret"],
                         labels["random_mean_regret"])
        print(f"{name:30s} {a0:10.5f} {idle:10.5f} {rnd:10.5f} "
              f"{str(a0 <= idle + 1e-12):>8s} {rnd - a0:8.5f}", flush=True)

    # ego prediction quality: final-progress MAE + within-group progress-ranking Spearman
    print("\n--- ego final-progress MAE + progress-ranking Spearman (group-mean) ---", flush=True)
    true_ds = np.asarray(bundle.effect_target[:, -1, 0, 0], dtype=np.float64)  # (E,) final ds
    G, A = bundle.groups, bundle.n_actions
    true_prog = true_ds.reshape(G, A)
    from scipy.stats import spearmanr  # local import; sim env has scipy
    for name, e2e_v in variants.items():
        pred_ds = np.asarray(e2e_v, dtype=np.float64)[:, :, -1, 0, 0].mean(axis=0)  # ens (E,)
        mae = float(np.abs(pred_ds - true_ds).mean())
        pred_prog = pred_ds.reshape(G, A)
        sp = np.array([spearmanr(pred_prog[g], true_prog[g]).statistic
                       if np.std(pred_prog[g]) > 0 and np.std(true_prog[g]) > 0 else np.nan
                       for g in range(G)])
        print(f"{name:30s} finalMAE={mae:.5f} m  rankSpearman={np.nanmean(sp):.4f}", flush=True)
    print("PROBE OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
