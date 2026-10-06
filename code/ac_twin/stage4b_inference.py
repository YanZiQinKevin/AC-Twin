"""Stage 4B frozen-model inference, legal features, risk scoring and output receipts.

Plan ``AC-TWIN-RFER-STAGE4B-RISK-CONFIRM-2026-09-15-V0.1`` section 6. Everything here
is inference-time legal: the frozen ``END_TO_END_MIXTURE`` primary predictions, the
Full/History expert disagreement, the Stage 3A Plus response summaries, the
pre-intervention geometry and the action/duration metadata. The true counterfactual
trajectory, the true response, the true error, CEA/CEW and the oracle identity are
NEVER touched here — ``risk_target`` is constructed only at adjudication, after
``confirmation_used.lock``.

The frozen Stage 4A inference primitives are imported from ``run_stage4a`` so the
expert/router/response forward passes are bit-identical to the development screen.
The determinism re-check (plan section 6.3.2) re-runs model pair 0 on the 32 lowest
``episode_seed`` confirm episodes twice and requires elementwise-identical outputs (or
``max abs diff <= 1e-7`` with device/dtype registered when the backend cannot
guarantee bitwise equality).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import run_stage4a as s4a
from ac_twin import stage4_risk as s4r
from ac_twin import stage4b_protocol as s4bp
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3_data import compute_effect_eval_mask
from ac_twin.stage3_scaler import FeatureScaler
from ac_twin.stage3b_bridge import calibrated_regime_probs_single
from ac_twin.stage3r_models import build_router_features
from ac_twin.stage3r_protocol import write_once_json

POSITION_SLICE = s4a.POSITION_SLICE
DETERMINISM_EPISODES = 32
DETERMINISM_PAIR = 0
DETERMINISM_TOL = 1e-7


# --- confirm-set loading ---------------------------------------------------------


def load_confirm_eval(config: Mapping[str, Any], *, project_root: Path) -> dict[str, Any]:
    """Load the collected confirm npz into scaled tensors + geometry.

    Single-role set: there is no train/validation/test split. The counterfactual
    branch is used ONLY to build the eval mask (presence), never an error label —
    the error target is deferred to adjudication. Read-only.
    """
    data_output = s4a._resolve(config["data_output"])
    s4bp.assert_allowed_stage4b_data(data_output)
    with np.load(data_output, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    history = np.asarray(data["object_history"], dtype=np.float32)
    factual_full = np.asarray(data["object_factual_future"], dtype=np.float32)
    counterfactual_full = np.asarray(data["object_counterfactual_future"], dtype=np.float32)
    valid = compute_effect_eval_mask(history, factual_full, counterfactual_full)
    scaler = FeatureScaler.load(s4a._resolve(config["scaler_path"]))
    tensors = {
        "history": torch.as_tensor(history, dtype=torch.float32),
        "factual_suffix": torch.as_tensor(factual_full[:, :10], dtype=torch.float32),
        "factual_actions": torch.as_tensor(data["factual_actions"], dtype=torch.long),
        "counterfactual_actions": torch.as_tensor(data["counterfactual_actions"], dtype=torch.long),
        "cf_action": torch.as_tensor(data["cf_action"], dtype=torch.long),
        "cf_duration": torch.as_tensor(data["cf_duration"], dtype=torch.long),
        "t_star": torch.as_tensor(data["t_star"], dtype=torch.long),
    }
    scaled = dict(tensors)
    scaled["history"] = torch.as_tensor(scaler.transform(tensors["history"].numpy()), dtype=torch.float32)
    scaled["factual_suffix"] = torch.as_tensor(
        scaler.transform(tensors["factual_suffix"].numpy()), dtype=torch.float32
    )
    return {
        "tensors": tensors,
        "scaled": scaled,
        "valid": valid,
        "geometry": s4a._pre_intervention_geometry(history),
        "cf_action": np.asarray(data["cf_action"], dtype=np.int64),
        "cf_duration": np.asarray(data["cf_duration"], dtype=np.int64),
        "episode_seed": np.asarray(data["episode_seed"], dtype=np.int64),
        "factual_xy": factual_full[..., POSITION_SLICE].astype(np.float32),
        "counterfactual_xy": counterfactual_full[..., POSITION_SLICE].astype(np.float32),
        "data_signature": sha256_file(data_output),
    }


# --- frozen multi-pair inference ----------------------------------------------------


def run_pair_inference(
    config: Mapping[str, Any],
    scaled: Mapping[str, torch.Tensor],
    indices: np.ndarray,
    *,
    pair_index: int,
    binding: Mapping[str, Any],
    stage3r_dir: Path,
) -> dict[str, np.ndarray]:
    """Frozen full/history/e2e + response inference for ONE model-seed pair."""
    seed = s4bp.EXPERT_ROUTER_SEEDS[pair_index]
    device = str(config["device"])
    batch_size = int(config["batch_size"])
    full_model = s4a._load_expert(stage3r_dir, pair_index, seed, "full")
    history_model = s4a._load_expert(stage3r_dir, pair_index, seed, "history")
    full = s4a._predict_expert(full_model, scaled, indices, device, batch_size)
    history = s4a._predict_expert(history_model, scaled, indices, device, batch_size)

    response_model = s4a._load_response_predictor(binding, pair_index)
    response = s4a._response_outputs(response_model, scaled, indices, device, batch_size)
    if not np.array_equal(response["episode_indices"], np.asarray(indices, dtype=np.int64)):
        raise RuntimeError("response prediction episode order mismatch")
    regime_probs = calibrated_regime_probs_single(
        binding["calibration"]["per_seed_params"][pair_index], response["regime_logits"]
    )
    presence_prob = 1.0 - regime_probs[..., 0]
    high_prob = regime_probs[..., 3]

    # build_router_features requires episode-aligned tensors; slice the scaled inputs
    # to the requested indices (the predictors above already index_select internally).
    pos = torch.as_tensor(np.asarray(indices, dtype=np.int64), dtype=torch.long)
    sub = {key: scaled[key].index_select(0, pos) for key in (
        "history", "factual_suffix", "factual_actions", "counterfactual_actions")}
    router_features, _, _ = build_router_features(
        history=sub["history"], factual_suffix=sub["factual_suffix"],
        factual_actions=sub["factual_actions"],
        counterfactual_actions=sub["counterfactual_actions"],
        response_features=torch.as_tensor(response["response_features"]),
        regime_probs=torch.as_tensor(regime_probs),
        full_prediction=full, history_prediction=history,
    )
    router = s4a._load_e2e_router(stage3r_dir, pair_index, seed).to(torch.device(device))
    with torch.no_grad():
        weight = router(router_features.to(torch.device(device))).cpu()
    e2e = weight * full + (1.0 - weight) * history
    return {
        "full": full.numpy().astype(np.float32),
        "history": history.numpy().astype(np.float32),
        "e2e": e2e.numpy().astype(np.float32),
        "r_hat": response["r_hat"].astype(np.float32),
        "presence_prob": presence_prob.astype(np.float32),
        "high_prob": high_prob.astype(np.float32),
    }


def determinism_recheck(
    config: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    stage3r_dir: Path,
) -> dict[str, Any]:
    """Re-run pair 0 on the 32 lowest-seed episodes twice; require identical outputs.

    If the backend cannot guarantee bitwise equality, accept ``max abs diff <= 1e-7``
    and register device/dtype/measured diff in the receipt.
    """
    order = np.argsort(bundle["episode_seed"], kind="stable")
    subset = order[:DETERMINISM_EPISODES]
    runs = []
    for _ in range(2):
        runs.append(run_pair_inference(
            config, bundle["scaled"], subset,
            pair_index=DETERMINISM_PAIR, binding=binding, stage3r_dir=stage3r_dir,
        ))
    max_abs = 0.0
    bitwise = True
    for key in runs[0]:
        first, second = runs[0][key], runs[1][key]
        if first.shape != second.shape:
            raise RuntimeError(f"determinism recheck shape mismatch on {key}")
        diff = float(np.max(np.abs(first.astype(np.float64) - second.astype(np.float64))))
        max_abs = max(max_abs, diff)
        if not np.array_equal(first, second):
            bitwise = False
    if not bitwise and max_abs > DETERMINISM_TOL:
        raise RuntimeError(
            f"Stage 4B determinism recheck failed: max abs diff {max_abs:.3e} > {DETERMINISM_TOL}"
        )
    return {
        "pair_index": DETERMINISM_PAIR,
        "n_episodes": int(subset.size),
        "episode_seeds": [int(s) for s in bundle["episode_seed"][subset]],
        "bitwise_identical": bool(bitwise),
        "max_abs_diff": float(max_abs),
        "tolerance": DETERMINISM_TOL,
        "device": str(config["device"]),
        "dtype": "float32",
    }


# --- legal features + risk scoring --------------------------------------------------


def build_features(bundle: Mapping[str, Any], predictions: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Frozen legal risk features for every confirm episode (label-free)."""
    response_predictions = {
        "r_hat": predictions["r_hat"],
        "presence_prob": predictions["presence_prob"],
        "high_prob": predictions["high_prob"],
    }
    return s4r.build_legal_risk_features(
        primary_predictions=predictions["e2e_predictions"],
        full_predictions=predictions["full_predictions"],
        history_predictions=predictions["history_predictions"],
        response_predictions=response_predictions,
        pre_intervention_geometry=bundle["geometry"],
        action=predictions["cf_action"],
        duration=predictions["cf_duration"],
        valid_mask=bundle["valid"],
    )


def score_all_candidates(
    features: Mapping[str, np.ndarray],
    *,
    finalist_model: s4r.FittedRiskModel,
    geometry_model: s4r.FittedRiskModel,
    rng_seed: int,
) -> dict[str, np.ndarray]:
    """Pre-registered risk scores (plan section 6.2); all 'higher = riskier'.

    Primary method ``COMBINED_RISK_HEAD/MLP``; standard baselines ``UNCERTAINTY_ONLY``
    and frozen ``GEOMETRY_ONLY/Ridge``; secondary ablations ``RESPONSE_ONLY/Ridge`` and
    ``DISAGREEMENT_ONLY``; a constant-random ordering is a curve reference only and
    never enters the Gate.
    """
    response_model = _load_response_ridge()
    scores = {
        "COMBINED_RISK_HEAD": finalist_model.predict(
            s4r.design_matrix(features, s4r.CANDIDATE_FEATURES[s4r.CANDIDATE_COMBINED])),
        "UNCERTAINTY_ONLY": s4r.direct_risk_score(s4r.CANDIDATE_UNCERTAINTY, features),
        "GEOMETRY_ONLY": geometry_model.predict(
            s4r.design_matrix(features, s4r.CANDIDATE_FEATURES[s4r.BASELINE_GEOMETRY])),
        "RESPONSE_ONLY": response_model.predict(
            s4r.design_matrix(features, s4r.CANDIDATE_FEATURES[s4r.CANDIDATE_RESPONSE])),
        "DISAGREEMENT_ONLY": s4r.direct_risk_score(s4r.CANDIDATE_DISAGREEMENT, features),
    }
    episodes = scores["COMBINED_RISK_HEAD"].shape[0]
    scores["CONSTANT_RANDOM"] = np.random.default_rng(int(rng_seed)).uniform(size=episodes)
    for name, value in scores.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"non-finite Stage 4B risk score: {name}")
    return scores


def _load_response_ridge() -> s4r.FittedRiskModel:
    path = s4a._resolve(
        "experiments/stage4_reliability/stage4a_v3/risk/models/response_only_ridge.pt")
    return s4r.FittedRiskModel.load(path)


# --- receipts ----------------------------------------------------------------------


def write_prediction_receipt(
    run_dir: Path,
    *,
    predictions_path: Path,
    bundle: Mapping[str, Any],
    lock: Mapping[str, Any],
    determinism: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = {
        "plan_id": s4bp.PLAN_ID,
        "status": "CONFIRM_INFERENCE_COMPLETE",
        "episode_count": int(bundle["episode_seed"].size),
        "data_sha256": bundle["data_signature"],
        "protocol_lock_seed_schedule_sha256": lock["seed_schedule_sha256"],
        "predictions_sha256": sha256_file(predictions_path),
        "determinism_recheck": dict(determinism),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(run_dir / "inference" / "prediction_receipt.json", receipt)
    return receipt


def write_risk_score_receipt(
    run_dir: Path,
    *,
    risk_scores_path: Path,
    scores: Mapping[str, np.ndarray],
    predictions_path: Path,
    finalist_model_path: Path,
    geometry_model_path: Path,
) -> dict[str, Any]:
    receipt = {
        "plan_id": s4bp.PLAN_ID,
        "status": "CONFIRM_RISK_SCORES_FROZEN",
        "scores": sorted(scores),
        "risk_scores_sha256": sha256_file(risk_scores_path),
        "predictions_sha256": sha256_file(predictions_path),
        "finalist_model_sha256": sha256_file(finalist_model_path),
        "geometry_model_sha256": sha256_file(geometry_model_path),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(run_dir / "inference" / "risk_score_receipt.json", receipt)
    return receipt


__all__ = [
    "DETERMINISM_EPISODES",
    "DETERMINISM_PAIR",
    "DETERMINISM_TOL",
    "build_features",
    "determinism_recheck",
    "load_confirm_eval",
    "run_pair_inference",
    "score_all_candidates",
    "write_prediction_receipt",
    "write_risk_score_receipt",
]
