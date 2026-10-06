"""Stage 4D-D2E repaired-prediction pipeline + §3.3 validation metrics.

Plan ``AC-TWIN-RFER-STAGE4D-D2E-EGO-CONSISTENT-EXTERNAL-2026-09-24-V0.1`` sections 3.2-3.3.
This module reuses the frozen D2 base predictions (``{role}_predictions.npz``) and the five
action-risk classes (``{role}_scores.npz``) READ-ONLY — the ego-slot repair provably does not
change the risk features (they aggregate over the ego-excluding ``inference_mask``), so only the
consequence ``J_hat`` changes. The Direct head is retrained on the repaired training predictions
+ ten-action regret; the three standard baselines use the identical repaired ``J_hat``.

Nothing here modifies a frozen D2/D2R artifact; all writes land under
``experiments/stage4_reliability/stage4d_d2e/``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

import torch  # noqa: E402

from ac_twin import stage4d_d2e_ego as ego  # noqa: E402
from ac_twin import stage4d_external_inference as INF  # noqa: E402
from ac_twin import stage4d_external_training as T  # noqa: E402
from ac_twin.stage3_scaler import FeatureScaler  # noqa: E402

D2_DEV = INF.RUN_DIR / "development"
EGO_MODEL_DIR = ego.D2E_DIR / "development" / "ego_models"

#: variant labels for the §3.3 same-table report (plan 3.3 "旧 D2 和运动学基线同表展示").
VARIANT_OLD_D2 = "OLD_D2"
VARIANT_KINEMATIC = "KINEMATIC"
VARIANT_ORACLE = "TRUTH_ORACLE"

_DT_S = 1.0 / 5.0          # 5 Hz sample period (frozen task geometry)


# --- read-only reuse of the frozen D2 base predictions + action risks ---------------------------


def load_persisted_base(role: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load the frozen D2 ``{role}_predictions.npz`` (e2e/full/history) + action risks.

    Read-only. Returns the predictions dict (with the UNREPAIRED e2e) and the five action-risk
    classes ``(G, A)``. The action risks are unaffected by the ego-slot repair (the risk
    features aggregate over the ego-excluding inference mask), so they are reused unchanged.
    """
    with np.load(D2_DEV / f"{role}_predictions.npz", allow_pickle=False) as z:
        predictions = {k: np.asarray(z[k]) for k in z.files}
    with np.load(D2_DEV / f"{role}_scores.npz", allow_pickle=False) as z:
        action_risks = {k: np.asarray(z[k], dtype=np.float64) for k in z.files}
    return predictions, action_risks


def load_bundle(role: str, expected_group_seed: np.ndarray) -> INF.BranchBundle:
    """Build the branch bundle for a role and assert alignment with the persisted predictions."""
    bundle = INF.build_branch_bundle(INF.load_role_arrays(INF.ROLES_DIR / role, role))
    if not np.array_equal(np.asarray(expected_group_seed, dtype=np.int64),
                          np.asarray(bundle.group_seed, dtype=np.int64)):
        raise RuntimeError(f"{role}: persisted predictions group_seed misaligned with bundle")
    return bundle


# --- ego model loading + variant e2e construction ------------------------------------------------


def load_ego_scaler() -> FeatureScaler:
    return FeatureScaler.load(EGO_MODEL_DIR / "ego_scaler.json")


def load_ego_models(candidate: str) -> dict[int, ego.EgoEffectHead]:
    if candidate not in ego.CANDIDATES:
        raise ValueError(f"unknown ego candidate {candidate!r}")
    models: dict[int, ego.EgoEffectHead] = {}
    for seed in ego.EGO_SEEDS:
        payload = torch.load(EGO_MODEL_DIR / f"ego_{candidate}_seed{seed}.pt",
                             map_location="cpu")
        model = ego.EgoEffectHead()
        model.load_state_dict(payload["state_dict"])
        models[seed] = model
    return models


def candidate_e2e(bundle: INF.BranchBundle, e2e_old: np.ndarray, scaler: FeatureScaler,
                  models: Mapping[int, ego.EgoEffectHead], *, device: str,
                  batch_size: int = 4096) -> np.ndarray:
    """The repaired e2e for one candidate: slot 0 <- the index-paired ego branch, slots 1-4 kept."""
    feats = ego.assemble_ego_features(bundle, scaler)
    deltas = np.stack([ego.predict_ego_effect(models[s], feats, device=device,
                                              batch_size=batch_size)
                       for s in ego.EGO_SEEDS])                     # (S,E,T,2)
    return ego.repair_e2e_ego(e2e_old, deltas)


def kinematic_e2e(bundle: INF.BranchBundle, e2e_old: np.ndarray) -> np.ndarray:
    """No-learning legal kinematic floor: constant-velocity ego extrapolation, action-agnostic."""
    v = bundle.history[:, -1, 0, 4:6].astype(np.float64)            # (E,2) last-frame (v_s,v_d)
    steps = np.arange(1, ego.FUTURE_LEN + 1, dtype=np.float64) * _DT_S
    delta = (v[:, None, :] * steps[None, :, None]).astype(np.float32)   # (E,T,2)
    return ego.repair_e2e_ego(
        e2e_old, np.broadcast_to(delta[None], (ego.N_SEEDS,) + delta.shape))


def oracle_e2e(bundle: INF.BranchBundle, e2e_old: np.ndarray) -> np.ndarray:
    """Truth ego-replacement oracle: slot 0 <- TRUE ego displacement. Explanation upper bound only;
    NEVER a deployable candidate, threshold baseline, or screen/confirm input (plan 3.2)."""
    true_ego = np.asarray(bundle.effect_target[:, :, 0, :], dtype=np.float32)   # (E,T,2)
    return ego.repair_e2e_ego(
        e2e_old, np.broadcast_to(true_ego[None], (ego.N_SEEDS,) + true_ego.shape))


def variant_e2e(name: str, bundle: INF.BranchBundle, e2e_old: np.ndarray,
                scaler: FeatureScaler,
                models_by_candidate: Mapping[str, Mapping[int, ego.EgoEffectHead]], *,
                device: str, batch_size: int = 4096) -> np.ndarray:
    """Resolve the e2e for a named §3.3 variant (old / kinematic / a candidate / oracle)."""
    if name == VARIANT_OLD_D2:
        return np.asarray(e2e_old, dtype=np.float32)
    if name == VARIANT_KINEMATIC:
        return kinematic_e2e(bundle, e2e_old)
    if name == VARIANT_ORACLE:
        return oracle_e2e(bundle, e2e_old)
    if name in ego.CANDIDATES:
        return candidate_e2e(bundle, e2e_old, scaler, models_by_candidate[name],
                             device=device, batch_size=batch_size)
    raise ValueError(f"unknown variant {name!r}")


# --- consequence + labels -------------------------------------------------------------------------


def direct_inputs_from_e2e(bundle: INF.BranchBundle, e2e: np.ndarray,
                           action_risks: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Run the frozen ``build_direct_inputs`` on a given e2e (slot-0-repaired or not)."""
    return INF.build_direct_inputs(bundle, {"e2e": e2e}, action_risks)


def true_labels(bundle: INF.BranchBundle, j_hat: np.ndarray) -> dict[str, Any]:
    """Frozen ``compute_true_labels`` (train/validation truth; never screen/confirm)."""
    return T.compute_true_labels(bundle.counterfactual_xy, bundle.history_last_xy,
                                 bundle.inference_mask, j_hat)


def build_direct_role_inputs(bundle: INF.BranchBundle, e2e: np.ndarray,
                             action_risks: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """The Direct train/validation input dict from a (slot-0-repaired) e2e + frozen labels.

    Combines ``build_direct_inputs`` (tokens, global_features, group_seed, j_hat, baselines,
    interaction_min_distance, valid_mask) with ``compute_true_labels`` (y_reg/y_high/regret...)
    computed against the SAME repaired ``j_hat`` — so the Direct head and the three standard
    baselines share the identical repaired underlying predictions (plan 3.2).
    """
    direct = direct_inputs_from_e2e(bundle, e2e, action_risks)
    labels = true_labels(bundle, np.asarray(direct["j_hat"], dtype=np.float64))
    return {
        "tokens": np.asarray(direct["tokens"], dtype=np.float64),
        "global_features": np.asarray(direct["global_features"], dtype=np.float64),
        "group_seed": np.asarray(direct["group_seed"], dtype=np.int64).ravel(),
        "interaction_min_distance": np.asarray(
            direct["interaction_min_distance"], dtype=np.float64).ravel(),
        "valid_mask": np.asarray(direct["valid_mask"], dtype=bool).ravel(),
        "j_hat": np.asarray(direct["j_hat"], dtype=np.float64),
        "j_hat_seeds": np.asarray(direct["j_hat_seeds"], dtype=np.float64),
        "baselines": {k: np.asarray(v, dtype=np.float64).ravel()
                      for k, v in direct["baselines"].items()},
        "y_reg": np.asarray(labels["y_reg"], dtype=np.float64),
        "y_high": np.asarray(labels["y_high"], dtype=np.int64),
        "regret": np.asarray(labels["regret"], dtype=np.float64),
        "idle_regret": np.asarray(labels["idle_regret"], dtype=np.float64),
        "random_regret": np.asarray(labels["random_regret"], dtype=np.float64),
        "j_true": np.asarray(labels["j_true"], dtype=np.float64),
        "base_mean_regret": float(labels["base_mean_regret"]),
        "idle_mean_regret": float(labels["idle_mean_regret"]),
        "random_mean_regret": float(labels["random_mean_regret"]),
    }


def fit_risk_inventory(candidate: str, train_inputs: Mapping[str, np.ndarray],
                       validation_inputs: Mapping[str, np.ndarray], *,
                       device: str,
                       progress: Any = None) -> tuple[T.ModelInventory, Path]:
    """Retrain the frozen Direct head on one candidate's repaired predictions (plan 3.2).

    Writes the per-candidate inventory under ``stage4d_d2e/riskfit_{candidate}/`` (create-once;
    a rerun verifies checkpoint hashes and never retrains). Returns the inventory + its dir.
    """
    run_dir = ego.D2E_DIR / f"riskfit_{candidate}"
    fit_keys = ("tokens", "global_features", "y_reg", "y_high", "group_seed",
                "interaction_min_distance")
    val_keys = ("tokens", "global_features", "y_reg", "y_high", "group_seed")
    train_bundle = {k: train_inputs[k] for k in fit_keys}
    val_bundle = {k: validation_inputs[k] for k in val_keys}
    inventory = T.fit_five_direct_models(train_bundle, val_bundle, run_dir,
                                         device=device, progress=progress)
    return inventory, run_dir


def load_risk_inventory(candidate: str) -> T.ModelInventory:
    """Rebuild a fitted per-candidate Direct inventory from its receipt (read-only, no retrain).

    Mirrors the screen driver's inventory reconstruction: verifies every checkpoint hash against
    the create-once receipt before trusting it.
    """
    import json

    from ac_twin import stage4cp_features as s4cpf
    from ac_twin.stage1_data import sha256_file

    if candidate not in ego.CANDIDATES:
        raise ValueError(f"unknown ego candidate {candidate!r}")
    receipt_path = (ego.D2E_DIR / f"riskfit_{candidate}" / "development" / "models"
                    / "fit_receipt.json")
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"Direct inventory missing for {candidate} (run fit-risk first): {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for seed_text, digest in receipt["checkpoint_sha256"].items():
        path = Path(receipt["checkpoint_paths"][seed_text])
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"D2E Direct checkpoint drifted: {path}")
    sc = receipt["scaler"]
    return T.ModelInventory(
        plan_id=str(receipt["plan_id"]), arch=str(receipt["arch"]),
        seeds=tuple(int(s) for s in receipt["seeds"]),
        param_count=int(receipt["param_count"]), run_dir=str(receipt["run_dir"]),
        checkpoint_paths={int(k): str(v) for k, v in receipt["checkpoint_paths"].items()},
        checkpoint_sha256={int(k): str(v) for k, v in receipt["checkpoint_sha256"].items()},
        scaler=s4cpf.FrozenFeatureScaler(
            np.asarray(sc["token_cont_mean"]), np.asarray(sc["token_cont_std"]),
            np.asarray(sc["global_mean"]), np.asarray(sc["global_std"])),
        pos_weight=float(receipt["pos_weight"]),
        interaction_threshold=float(receipt["interaction_threshold"]),
        train_losses={int(k): float(v) for k, v in receipt["train_losses"].items()},
        validation_losses={int(k): float(v) for k, v in receipt["validation_losses"].items()},
        epochs_run={int(k): int(v) for k, v in receipt["epochs_run"].items()},
        n_train_groups=int(receipt["n_train_groups"]),
        n_validation_groups=int(receipt["n_validation_groups"]),
        written_at=str(receipt["written_at"]),
        verification="existing_inventory_verified_no_overwrite")


# --- §3.3 ego / task metrics -----------------------------------------------------------------------


def _final_progress(bundle: INF.BranchBundle, e2e: np.ndarray) -> np.ndarray:
    """Predicted per-(group,action) ego final progress (metres), ensemble over seeds -> (G,A)."""
    e2e = np.asarray(e2e, dtype=np.float64)
    pred_ds = e2e[:, :, -1, ego.EGO_SLOT, 0].mean(axis=0)             # (E,) ensemble final ds
    G, A = bundle.groups, bundle.n_actions
    factual_fin = bundle.factual_xy[:, -1, ego.EGO_SLOT, 0]            # (G,)
    hist_x = bundle.history_last_xy[:, 0]                              # (G,)
    return pred_ds.reshape(G, A) + (factual_fin - hist_x)[:, None]


def _true_final_progress(bundle: INF.BranchBundle) -> np.ndarray:
    """True per-(group,action) ego final progress (metres) -> (G,A)."""
    G, A = bundle.groups, bundle.n_actions
    cf_fin = bundle.counterfactual_xy[:, :, -1, ego.EGO_SLOT, 0]       # (G,A)
    hist_x = bundle.history_last_xy[:, 0]                              # (G,)
    return cf_fin - hist_x[:, None]


def _spearman_per_group(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """Group-mean within-ten-action Spearman + the tied/constant-progress group fraction."""
    from scipy.stats import rankdata
    G = pred.shape[0]
    sp = np.full(G, np.nan)
    constant = 0
    for g in range(G):
        if np.std(true[g]) < 1e-12:            # constant true progress -> undefined ranking
            constant += 1
            continue
        # Spearman = Pearson on ranks; a zero-variance predicted ranking is degenerate.
        rp = rankdata(pred[g])
        rt = rankdata(true[g])
        if np.std(rp) < 1e-12:
            sp[g] = 0.0
            continue
        sp[g] = float(np.corrcoef(rp, rt)[0, 1])
    return float(np.nanmean(sp)), float(constant / G)


def ego_quality(bundle: INF.BranchBundle, e2e: np.ndarray,
                ego_mask: np.ndarray) -> dict[str, float]:
    """The §3.3 ego-prediction quality metrics for one variant.

    Returns ego final-progress MAE, within-group progress-ranking Spearman (group mean) with the
    tied/constant-progress fraction, ego effect ADE, and non-ego effect ADE. ``ego_mask`` is the
    ego-inclusive training mask ``M_ego`` (E,T) used for the ego ADE; the non-ego ADE uses the
    cf-dependent non-ego ``valid_eval`` cells (slots 1-4) and is identical across slot-0 variants.
    """
    e2e = np.asarray(e2e, dtype=np.float64)
    effect_target = np.asarray(bundle.effect_target, dtype=np.float64)  # (E,T,N,2)
    ens = e2e.mean(axis=0)                                              # (E,T,N,2)

    # final-progress MAE (paired against truth, per group-action)
    pred_prog = _final_progress(bundle, e2e)                            # (G,A)
    true_prog = _true_final_progress(bundle)                            # (G,A)
    final_mae = float(np.abs(pred_prog - true_prog).mean())
    sp, const_frac = _spearman_per_group(pred_prog, true_prog)

    # ego effect ADE over M_ego frames
    ego_err = np.linalg.norm(ens[:, :, ego.EGO_SLOT, :] - effect_target[:, :, ego.EGO_SLOT, :],
                             axis=-1)                                   # (E,T)
    m = np.asarray(ego_mask, dtype=bool)
    ego_ade = float(ego_err[m].mean()) if m.any() else float("nan")

    # non-ego effect ADE over the cf-dependent non-ego valid cells (slots 1-4)
    non_ego = np.zeros(effect_target.shape[:3], dtype=bool)
    non_ego[:, :, 1:] = True
    valid_ne = np.asarray(bundle.valid_eval, dtype=bool) & non_ego      # (E,T,N)
    ne_err = np.linalg.norm(ens - effect_target, axis=-1)               # (E,T,N)
    non_ego_ade = float(ne_err[valid_ne].mean()) if valid_ne.any() else float("nan")

    return {
        "ego_final_progress_mae": final_mae,
        "ego_progress_rank_spearman": sp,
        "progress_constant_group_frac": const_frac,
        "ego_effect_ade": ego_ade,
        "non_ego_effect_ade": non_ego_ade,
    }


__all__ = [
    "D2_DEV", "EGO_MODEL_DIR", "VARIANT_KINEMATIC", "VARIANT_OLD_D2", "VARIANT_ORACLE",
    "build_direct_role_inputs", "candidate_e2e", "direct_inputs_from_e2e", "ego_quality",
    "fit_risk_inventory", "kinematic_e2e", "load_bundle", "load_ego_models", "load_ego_scaler",
    "load_persisted_base", "load_risk_inventory", "oracle_e2e", "true_labels", "variant_e2e",
]
