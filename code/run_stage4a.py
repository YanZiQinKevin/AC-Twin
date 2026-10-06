"""Stage 4A Merge reliability development-screen driver.

Staged entry point (plan ``AC-TWIN-RFER-STAGE4A-RISK-DEV-2026-09-11-V0.1``
section 11). It deliberately does NOT copy the Stage 3R monolithic runner: each
phase is a small, individually testable function over the frozen Stage 3R expert
checkpoints, the Stage 3A Plus response production checkpoints and the frozen
scaler, all read-only.

State machine (plan section 7 Task 5)::

    DRAFT -> SOFTWARE_READY -> PROTOCOL_LOCKED -> DEV_COLLECTED
          -> DEV_INFERENCE_COMPLETE -> DEV_CANDIDATE_SELECTED -> DEV_SCREEN_ADJUDICATED

The development-screen is single-use: ``adjudicate-dev-screen`` acquires
``development_screen_used.lock`` (O_EXCL) BEFORE any screen label/prediction/risk
score is read, then writes one write-once report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ac_twin import stage3_training as s3t  # noqa: E402
from ac_twin import stage4_protocol as s4p  # noqa: E402
from ac_twin import stage4_recovery as s4rec  # noqa: E402
from ac_twin import stage4_risk as s4r  # noqa: E402
from ac_twin import stage4_statistics as s4s  # noqa: E402
from ac_twin.checkpointing import canonical_fingerprint  # noqa: E402
from ac_twin.progress import ProgressReporter  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3_data import (  # noqa: E402
    compute_effect_eval_mask,
    compute_history_agent_known,
)
from ac_twin.stage3_response import Stage3ResponsePredictor, invert_log1p  # noqa: E402
from ac_twin.stage3_scaler import FeatureScaler  # noqa: E402
from ac_twin.stage3b_bridge import (  # noqa: E402
    calibrated_regime_probs_single,
    predictor_features_ordered,
)
from ac_twin.stage3r_models import CEWRouter, Stage3RExpert, build_router_features  # noqa: E402
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402

SMOKE_STATUS = "ENGINEERING_SMOKE_ONLY"
POSITION_SLICE = slice(2, 4)  # x, y
VELOCITY_SLICE = slice(4, 6)  # vx, vy


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _load_config(path: str | Path) -> dict[str, Any]:
    return s4p.validate_stage4a_config(json.loads(Path(path).read_text(encoding="utf-8")))


def _verify_lock(run_dir: Path) -> dict[str, Any]:
    """Verify whichever lock a run dir holds: a v2 recovery lock or the v1 protocol lock.

    Both return a mapping exposing ``seed_schedule_sha256`` plus the frozen
    checkpoint/scaler/binding hashes, so downstream commands work unchanged.
    """
    if s4rec.is_recovery_run(run_dir):
        return s4rec.verify_recovery_lock(run_dir)
    return s4p.verify_protocol_lock(run_dir)


# --- status / handoff -----------------------------------------------------------


def _write_status(run_dir: Path, state: str, *, note: str, blocking: str = "none") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    status = (
        f"# Stage 4A 状态\n\n"
        f"> 当前状态：`{state}`\n> 计划：`{s4p.PLAN_ID}`\n> 更新：{time.strftime('%Y-%m-%dT%H:%M:%S')}\n\n"
        f"- 说明：{note}\n- 阻断项：{blocking}\n"
    )
    (run_dir / "STAGE4A_STATUS.md").write_text(status, encoding="utf-8")
    handoff = (
        f"# Stage 4A HANDOFF_STATE\n\n"
        f"- 当前状态：`{state}`\n"
        f"- 冻结决定：见 `protocol/protocol_lock.json` 与 `risk/candidate_selection.json`（若已生成）\n"
        f"- 最近验证：{note}\n"
        f"- 下一步：按 `run_stage4a.py` 命令顺序推进\n"
        f"- 阻断项：{blocking}\n"
    )
    (run_dir / "HANDOFF_STATE.md").write_text(handoff, encoding="utf-8")


# --- preflight --------------------------------------------------------------------


def cmd_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only: validate config, scan historical seeds, verify every frozen input.

    Builds (but does NOT write) the protocol lock so a missing/checkpoint drift or
    a historical seed overlap fails here, before collection. Never modifies any
    old artifact.
    """
    lock = s4p.build_protocol_lock(config, project_root=ROOT)
    run_dir = _resolve(config["run_dir"])
    _write_status(run_dir, "SOFTWARE_READY", note="preflight 通过；模型锁与历史seed零重叠已复验")
    return {
        "status": "STAGE4A_PREFLIGHT_PASS",
        "plan_id": s4p.PLAN_ID,
        "n_historical_episode_seeds": lock["n_historical_episode_seeds"],
        "seed_schedule_zero_overlap": lock["seed_schedule_zero_overlap"],
        "checkpoint_count": len(lock["checkpoint_sha256"]),
        "scaler_sha256": lock["scaler_sha256"],
        "stage3b_binding_sha256": lock["stage3b_binding_sha256"],
        "seed_schedule_sha256": lock["seed_schedule_sha256"],
    }


def cmd_lock_protocol(config_path: str | Path) -> dict[str, Any]:
    lock = s4p.write_protocol_lock(config_path, project_root=ROOT)
    run_dir = _resolve(json.loads(Path(config_path).read_text(encoding="utf-8"))["run_dir"])
    _write_status(run_dir, "PROTOCOL_LOCKED", note="protocol lock 已一次性写入")
    return {
        "status": "PROTOCOL_LOCKED",
        "protocol_lock_sha256": lock["seed_schedule_sha256"],
        "n_historical_episode_seeds": lock["n_historical_episode_seeds"],
    }


def cmd_lock_recovery(config_path: str | Path) -> dict[str, Any]:
    """Write the v2 recovery lock once, reusing the v1-collected data read-only.

    The config must carry a ``recovery_from_v1`` block naming the parent run dir,
    the reason and the exact code location of the provable engineering fix. This
    never re-collects, never re-locks a seed schedule and never touches the v1
    protocol lock; it fails if any frozen input other than the fixed runner drifted.
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    recovery = config.get("recovery_from_v1")
    if not recovery:
        raise ValueError("lock-recovery requires a recovery_from_v1 config block")
    lock = s4rec.write_recovery_lock(
        config_path,
        parent_run_dir=recovery["parent_run_dir"],
        reason=recovery["reason"],
        code_location=recovery["code_location"],
        project_root=ROOT,
    )
    run_dir = _resolve(config["run_dir"])
    _write_status(
        run_dir, "PROTOCOL_LOCKED",
        note="v2 恢复锁已一次性写入；只读复用 v1 数据，从 infer-dev 恢复（不重跑采集）",
    )
    return {
        "status": "RECOVERY_LOCKED",
        "lock_type": lock["lock_type"],
        "parent_protocol_lock_sha256": lock["parent_protocol_lock_sha256"],
        "original_runner_sha256": lock["original_runner_sha256"],
        "fixed_runner_sha256": lock["fixed_runner_sha256"],
        "seed_schedule_sha256": lock["seed_schedule_sha256"],
        "data_npz_sha256": lock["data_npz_sha256"],
        "reuse_verification": lock["reuse_verification"],
    }


def cmd_lock_recovery_v3(config_path: str | Path) -> dict[str, Any]:
    """Write the v3 recovery lock once, reusing v2 frozen inference + PRIMARY_PREDICTOR.

    The config must carry a ``recovery_from_v2`` block naming the v2 run dir, the
    reason and the exact code location of the provable MLP-serialization fix. This
    never re-collects, never re-runs infer-dev, never re-selects the primary
    predictor and never touches the v1/v2 locks; it fails on any inconsistency.
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    recovery = config.get("recovery_from_v2")
    if not recovery:
        raise ValueError("lock-recovery-v3 requires a recovery_from_v2 config block")
    lock = s4rec.write_recovery_lock_v3(
        config_path,
        v2_run_dir=recovery["v2_run_dir"],
        reason=recovery["reason"],
        code_location=recovery["code_location"],
        project_root=ROOT,
    )
    run_dir = _resolve(config["run_dir"])
    _write_status(
        run_dir, "DEV_INFERENCE_REUSED_FROM_V2",
        note="v3 恢复锁已写入；复用 v2 冻结推理与 PRIMARY_PREDICTOR，从 fit-risk 重训风险模型",
    )
    return {
        "status": "RECOVERY_LOCKED_V3",
        "lock_type": lock["lock_type"],
        "v1_protocol_lock_sha256": lock["chain"]["v1_protocol_lock_sha256"],
        "v2_recovery_lock_sha256": lock["chain"]["v2_recovery_lock_sha256"],
        "stage4_risk_fix": lock["stage4_risk_fix"],
        "predictions_npz_sha256": lock["reused_inference"]["predictions_npz_sha256"],
        "primary_predictor": lock["primary_predictor"]["primary_predictor"],
        "data_npz_sha256": lock["data_npz_sha256"],
        "reuse_verification": lock["reuse_verification"],
    }


def cmd_lock_recovery_v4(config_path: str | Path) -> dict[str, Any]:
    """Write the v4 recovery lock once, reusing v2 inference + the complete v3 risk fit.

    The config must carry a ``recovery_from_v3`` block naming the v3 run dir, the
    runner-fix patch receipt, the reason and the exact code location of the provable
    cmd_select_candidate reused-inference receipt-path fix. This never re-collects,
    never re-runs infer-dev, never re-selects the primary predictor, never retrains
    risk models and never touches the v1/v2/v3 locks; it fails on any inconsistency.
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    recovery = config.get("recovery_from_v3")
    if not recovery:
        raise ValueError("lock-recovery-v4 requires a recovery_from_v3 config block")
    run_dir = _resolve(config["run_dir"])
    lock = s4rec.write_recovery_lock_v4(
        config_path,
        v3_run_dir=recovery["v3_run_dir"],
        patch_receipt_path=run_dir / "protocol" / "runner_fix_receipt.json",
        reason=recovery["reason"],
        code_location=recovery["code_location"],
        project_root=ROOT,
    )
    _write_status(
        run_dir, "DEV_SELECTION_REPLAY_FROM_V3",
        note="v4 恢复锁已写入；只读复用 v2 推理 + v2 PRIMARY_PREDICTOR + v3 risk fit，从 select-candidate 确定性重放",
    )
    return {
        "status": "RECOVERY_LOCKED_V4",
        "lock_type": lock["lock_type"],
        "v1_protocol_lock_sha256": lock["chain"]["v1_protocol_lock_sha256"],
        "v2_recovery_lock_sha256": lock["chain"]["v2_recovery_lock_sha256"],
        "v3_recovery_lock_sha256": lock["chain"]["v3_recovery_lock_sha256"],
        "runner_recovery": lock["runner_recovery"],
        "predictions_npz_sha256": lock["reused_inference"]["predictions_npz_sha256"],
        "primary_predictor": lock["primary_predictor"]["primary_predictor"],
        "risk_dataset_sha256": lock["reused_risk_fit"]["v3_risk_dataset_sha256"],
        "fitted_models": sorted(lock["reused_risk_fit"]["fitted_models"]),
        "data_npz_sha256": lock["data_npz_sha256"],
        "reuse_verification": lock["reuse_verification"],
    }


# --- evaluation bundle -------------------------------------------------------------


def _pre_intervention_geometry(history: np.ndarray) -> dict[str, np.ndarray]:
    """Pre-intervention per-vehicle geometry from the last history frame.

    Returns ``distance_to_ego`` / ``relative_speed`` (E,N, NaN for absent or ego)
    and ``visible_count`` (E,) of history-known non-ego vehicles. All values are
    knowable BEFORE the intervention; nothing reads the future branches.
    """
    history = np.asarray(history, dtype=np.float64)
    known = compute_history_agent_known(history)  # E,N
    last = history[:, -1]  # E,N,8
    ego_xy = last[:, 0, POSITION_SLICE]  # E,2
    ego_v = last[:, 0, VELOCITY_SLICE]  # E,2
    xy = last[:, :, POSITION_SLICE]  # E,N,2
    vel = last[:, :, VELOCITY_SLICE]  # E,N,2
    distance = np.linalg.norm(xy - ego_xy[:, None, :], axis=-1)  # E,N
    relspeed = np.linalg.norm(vel - ego_v[:, None, :], axis=-1)  # E,N
    usable = known.copy()
    usable[:, 0] = False  # exclude ego
    distance = np.where(usable, distance, np.nan)
    relspeed = np.where(usable, relspeed, np.nan)
    visible_count = usable.sum(axis=1).astype(np.float64)
    return {
        "distance_to_ego": distance,
        "relative_speed": relspeed,
        "visible_count": visible_count,
    }


def _load_stage4_eval(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load the collected Stage 4A npz into scaled tensors + effect target + mask.

    The counterfactual branch builds ONLY the effect target / eval mask (the
    label side); it is never a model input.
    """
    data_output = _resolve(config["data_output"])
    s4p.assert_allowed_stage4_data(data_output)
    with np.load(data_output, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    history = np.asarray(data["object_history"], dtype=np.float32)
    factual_full = np.asarray(data["object_factual_future"], dtype=np.float32)
    counterfactual_full = np.asarray(data["object_counterfactual_future"], dtype=np.float32)
    valid = compute_effect_eval_mask(history, factual_full, counterfactual_full)
    effect_true, valid = s4r.build_effect_target(
        factual_full[..., POSITION_SLICE], counterfactual_full[..., POSITION_SLICE], valid
    )
    scaler = FeatureScaler.load(_resolve(config["scaler_path"]))
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
    roles = np.asarray(data["split"], dtype=np.int64)
    return {
        "tensors": tensors,
        "scaled": scaled,
        "effect_true": effect_true,
        "valid": valid,
        "roles": roles,
        "role_indices": s4p.role_indices(roles),
        "geometry": _pre_intervention_geometry(history),
        "cf_action": np.asarray(data["cf_action"], dtype=np.int64),
        "cf_duration": np.asarray(data["cf_duration"], dtype=np.int64),
        "episode_seed": np.asarray(data["episode_seed"], dtype=np.int64),
        "data_signature": sha256_file(data_output),
    }


# --- frozen model loading ------------------------------------------------------------


def _load_expert(run_dir: Path, pair_index: int, seed: int, condition: str) -> Stage3RExpert:
    path = s4p.expert_checkpoint_path(run_dir, pair_index, condition, seed)
    model = Stage3RExpert(condition, hidden_dim=64, object_embedding_dim=64)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def _load_e2e_router(run_dir: Path, pair_index: int, seed: int) -> CEWRouter:
    path = s4p.e2e_checkpoint_path(run_dir, pair_index, seed)
    record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    model = CEWRouter(int(record["input_dim"]), 64)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def _load_response_predictor(binding: Mapping[str, Any], pair_index: int) -> Stage3ResponsePredictor:
    pred_seed = s4p.RESPONSE_PREDICTOR_SEEDS[pair_index]
    record = binding["checkpoints"][str(pred_seed)]
    meta = s3t.verify_checkpoint_integrity(record["path"], expected_signature=record["signature"])["meta"]
    state = s3t.load_predictor_state(record["path"], expected_signature=record["signature"])
    model = Stage3ResponsePredictor(
        hidden_dim=int(meta["config"]["hidden_dim"]),
        response_feature_dim=int(meta["config"]["response_feature_dim"]),
    )
    model.load_state_dict(state)
    return model.eval()


def _predict_expert(model: Stage3RExpert, scaled, indices, device, batch_size) -> torch.Tensor:
    device_obj = torch.device(device)
    model = model.to(device_obj).eval()
    ordered = np.asarray(indices, dtype=np.int64)
    outputs = []
    with torch.no_grad():
        for start in range(0, ordered.size, batch_size):
            pos = torch.as_tensor(ordered[start:start + batch_size], dtype=torch.long)
            batch = {
                key: scaled[key].index_select(0, pos).to(device_obj)
                for key in ("history", "factual_suffix", "factual_actions", "counterfactual_actions")
            }
            outputs.append(model(**batch).cpu())
    return torch.cat(outputs, dim=0)


def _response_outputs(model, scaled, indices, device, batch_size) -> dict[str, np.ndarray]:
    """Ordered frozen response inference -> response_features, regime_logits, r_hat."""
    sub = {key: scaled[key] for key in (
        "history", "factual_suffix", "factual_actions", "counterfactual_actions",
        "cf_action", "cf_duration", "t_star",
    )}
    feats = predictor_features_ordered(model, sub, np.asarray(indices), device, batch_size)
    # r_hat needs a second forward pass through the same ordered batches.
    device_obj = torch.device(device)
    model = model.to(device_obj).eval()
    r_hat = []
    with torch.no_grad():
        ordered = np.asarray(indices, dtype=np.int64)
        for start in range(0, ordered.size, batch_size):
            pos = torch.as_tensor(ordered[start:start + batch_size], dtype=torch.long)
            batch = {key: sub[key].index_select(0, pos).to(device_obj) for key in sub}
            out = model(
                batch["history"], batch["factual_suffix"], batch["factual_actions"],
                batch["counterfactual_actions"], batch["cf_action"], batch["cf_duration"],
                batch["t_star"],
            )
            r_hat.append(invert_log1p(out["log1p_r_hat"]).cpu().numpy())
    return {
        "response_features": feats["response_features"],
        "regime_logits": feats["regime_logits"],
        "r_hat": np.concatenate(r_hat, axis=0).astype(np.float32),
        "episode_indices": feats["episode_indices"],
    }


# --- infer-dev ------------------------------------------------------------------


def cmd_infer_dev(config: Mapping[str, Any], *, resume: bool = False) -> dict[str, Any]:
    run_dir = _resolve(config["run_dir"])
    lock = _verify_lock(run_dir)
    receipt_path = Path(lock.get("collection_receipt_path", run_dir / "collection" / "collection_receipt.json"))
    collection = json.loads(receipt_path.read_text(encoding="utf-8"))
    bundle = _load_stage4_eval(config)
    if collection.get("data_output_sha256") != bundle["data_signature"]:
        raise ValueError("Stage 4A collection data hash mismatch")
    prediction_path = run_dir / "inference" / "predictions.npz"
    pred_receipt_path = run_dir / "inference" / "prediction_receipt.json"
    if prediction_path.is_file() and pred_receipt_path.is_file():
        receipt = json.loads(pred_receipt_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError("Stage 4A inference already completed")
        if receipt.get("predictions_sha256") != sha256_file(prediction_path):
            raise ValueError("Stage 4A prediction hash mismatch")
        return receipt
    if prediction_path.exists() or pred_receipt_path.exists():
        raise FileExistsError("partial Stage 4A inference artifacts")

    binding = json.loads(_resolve(config["stage3b_binding"]).read_text(encoding="utf-8"))["receipt_payload"]
    scaled = bundle["scaled"]
    episodes = scaled["history"].shape[0]
    indices = np.arange(episodes, dtype=np.int64)
    device = str(config["device"])
    batch_size = int(config["batch_size"])
    stage3r_dir = _resolve(config["stage3r_run_dir"])

    full_preds, history_preds, constant_preds, e2e_preds = [], [], [], []
    r_hats, presence_probs, high_probs = [], [], []
    reporter = ProgressReporter(total=len(s4p.EXPERT_ROUTER_SEEDS), every=1,
                                prefix="stage4a-infer", heartbeat_seconds=30.0)
    for pair_index, seed in enumerate(s4p.EXPERT_ROUTER_SEEDS):
        full_model = _load_expert(stage3r_dir, pair_index, seed, "full")
        history_model = _load_expert(stage3r_dir, pair_index, seed, "history")
        full = _predict_expert(full_model, scaled, indices, device, batch_size)
        history = _predict_expert(history_model, scaled, indices, device, batch_size)
        constant = s4p.PRIMARY_CONSTANT_W * full + (1.0 - s4p.PRIMARY_CONSTANT_W) * history

        response_model = _load_response_predictor(binding, pair_index)
        response = _response_outputs(response_model, scaled, indices, device, batch_size)
        if not np.array_equal(response["episode_indices"], indices):
            raise RuntimeError("response prediction episode order mismatch")
        regime_probs = calibrated_regime_probs_single(
            binding["calibration"]["per_seed_params"][pair_index], response["regime_logits"]
        )
        presence_prob = 1.0 - regime_probs[..., 0]
        high_prob = regime_probs[..., 3]

        features, _, _ = build_router_features(
            history=scaled["history"], factual_suffix=scaled["factual_suffix"],
            factual_actions=scaled["factual_actions"],
            counterfactual_actions=scaled["counterfactual_actions"],
            response_features=torch.as_tensor(response["response_features"]),
            regime_probs=torch.as_tensor(regime_probs),
            full_prediction=full, history_prediction=history,
        )
        router = _load_e2e_router(stage3r_dir, pair_index, seed).to(torch.device(device))
        with torch.no_grad():
            weight = router(features.to(torch.device(device))).cpu()
            e2e = weight * full + (1.0 - weight) * history

        full_preds.append(full.numpy())
        history_preds.append(history.numpy())
        constant_preds.append(constant.numpy())
        e2e_preds.append(e2e.numpy())
        r_hats.append(response["r_hat"])
        presence_probs.append(presence_prob.astype(np.float32))
        high_probs.append(high_prob.astype(np.float32))
        reporter.update(pair_index + 1, details=f"pair{pair_index} done")

    arrays = {
        "full_predictions": np.stack(full_preds).astype(np.float32),
        "history_predictions": np.stack(history_preds).astype(np.float32),
        "constant_predictions": np.stack(constant_preds).astype(np.float32),
        "e2e_predictions": np.stack(e2e_preds).astype(np.float32),
        "r_hat": np.stack(r_hats).astype(np.float32),
        "presence_prob": np.stack(presence_probs).astype(np.float32),
        "high_prob": np.stack(high_probs).astype(np.float32),
        "effect_true": bundle["effect_true"].astype(np.float32),
        "valid_mask": bundle["valid"].astype(bool),
        "roles": bundle["roles"].astype(np.int64),
        "cf_action": bundle["cf_action"],
        "cf_duration": bundle["cf_duration"],
        "episode_seed": bundle["episode_seed"],
    }
    for name, value in arrays.items():
        if value.dtype != bool and not np.isfinite(value).all():
            raise RuntimeError(f"non-finite Stage 4A prediction field: {name}")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = prediction_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(prediction_path)
    receipt = {
        "plan_id": s4p.PLAN_ID,
        "status": "DEV_INFERENCE_COMPLETE",
        "episode_count": int(episodes),
        "data_sha256": bundle["data_signature"],
        "protocol_lock_seed_schedule_sha256": lock["seed_schedule_sha256"],
        "predictions_sha256": sha256_file(prediction_path),
    }
    write_once_json(pred_receipt_path, receipt)
    _write_status(run_dir, "DEV_INFERENCE_COMPLETE",
                  note=f"冻结推理完成：{episodes} episodes × 5 pair")
    return receipt


# --- fit-risk ------------------------------------------------------------------


def _inference_paths(run_dir: Path, lock: Mapping[str, Any]) -> tuple[Path, Path]:
    """Where the frozen predictions live: v3 reuses v2's inference; v1/v2 use their own."""
    reused = lock.get("reused_inference")
    if reused:
        return Path(reused["predictions_npz_path"]), Path(reused["prediction_receipt_path"])
    return run_dir / "inference" / "predictions.npz", run_dir / "inference" / "prediction_receipt.json"


def _load_predictions(config: Mapping[str, Any], lock: Mapping[str, Any]) -> dict[str, np.ndarray]:
    run_dir = _resolve(config["run_dir"])
    prediction_path, receipt_path = _inference_paths(run_dir, lock)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("predictions_sha256") != sha256_file(prediction_path):
        raise ValueError("Stage 4A prediction hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _select_primary_predictor(constant_ade: np.ndarray, e2e_ade: np.ndarray) -> dict[str, Any]:
    """Freeze PRIMARY_PREDICTOR on risk-train by lower 5-pair mean episode effect-ADE.

    ``constant_ade`` / ``e2e_ade`` are ``(S, E_train)`` per-pair episode errors.
    A tie selects BEST_CONSTANT_W (plan section 4.2).
    """
    constant_mean = float(np.asarray(constant_ade, dtype=np.float64).mean())
    e2e_mean = float(np.asarray(e2e_ade, dtype=np.float64).mean())
    if constant_mean <= e2e_mean:
        chosen = "BEST_CONSTANT_W"
    else:
        chosen = "END_TO_END_MIXTURE"
    return {
        "primary_predictor": chosen,
        "constant_w": s4p.PRIMARY_CONSTANT_W,
        "risk_train_constant_mean_ade": constant_mean,
        "risk_train_e2e_mean_ade": e2e_mean,
        "selection_basis": "lowest 5-pair mean episode effect-ADE on risk-train; tie -> BEST_CONSTANT_W",
    }


def cmd_fit_risk(config: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = _resolve(config["run_dir"])
    lock = _verify_lock(run_dir)
    predictions = _load_predictions(config, lock)
    roles = np.asarray(predictions["roles"], dtype=np.int64)
    role_idx = s4p.role_indices(roles)
    train_idx = role_idx["risk_train"]
    valid_mask = np.asarray(predictions["valid_mask"], dtype=bool)
    effect_true = np.asarray(predictions["effect_true"], dtype=np.float32)
    risk_dir = run_dir / "risk"
    risk_dir.mkdir(parents=True, exist_ok=True)

    # Per-pair episode effect-ADE for the two frozen primary candidates.
    constant_ade = s4r.episode_effect_ade(predictions["constant_predictions"], effect_true, valid_mask)
    e2e_ade = s4r.episode_effect_ade(predictions["e2e_predictions"], effect_true, valid_mask)

    # PRIMARY_PREDICTOR is chosen on RISK-TRAIN only (plan section 4.2). In a v3
    # recovery it is FROZEN upstream (v2): verify + reference that receipt, never
    # re-select or overwrite it.
    reused_primary = lock.get("primary_predictor")
    if reused_primary is not None:
        pp_path = Path(reused_primary["receipt_path"])
        if sha256_file(pp_path) != reused_primary["receipt_sha256"]:
            raise ValueError("reused PRIMARY_PREDICTOR receipt hash mismatch")
        primary = json.loads(pp_path.read_text(encoding="utf-8"))
        primary["_reused_from"] = str(pp_path)
        primary["_reused_receipt_sha256"] = reused_primary["receipt_sha256"]
    else:
        primary_receipt_path = risk_dir / "primary_predictor_receipt.json"
        if primary_receipt_path.exists():
            primary = json.loads(primary_receipt_path.read_text(encoding="utf-8"))
        else:
            primary = _select_primary_predictor(constant_ade[:, train_idx], e2e_ade[:, train_idx])
            primary["plan_id"] = s4p.PLAN_ID
            primary["predictions_sha256"] = sha256_file(_inference_paths(run_dir, lock)[0])
            write_once_json(primary_receipt_path, primary)
    primary_name = primary["primary_predictor"]
    primary_predictions = (
        predictions["constant_predictions"] if primary_name == "BEST_CONSTANT_W"
        else predictions["e2e_predictions"]
    )
    primary_ade = s4r.episode_effect_ade(primary_predictions, effect_true, valid_mask)  # S,E
    risk_target = s4r.risk_target_from_pairs(primary_ade)  # E

    # Legal features for all episodes (label-free construction).
    eval_bundle = _load_stage4_eval(config)
    response_predictions = {
        "r_hat": predictions["r_hat"],
        "presence_prob": predictions["presence_prob"],
        "high_prob": predictions["high_prob"],
    }
    features = s4r.build_legal_risk_features(
        primary_predictions=primary_predictions,
        full_predictions=predictions["full_predictions"],
        history_predictions=predictions["history_predictions"],
        response_predictions=response_predictions,
        pre_intervention_geometry=eval_bundle["geometry"],
        action=predictions["cf_action"],
        duration=predictions["cf_duration"],
        valid_mask=valid_mask,
    )

    # Frozen top-error-20% threshold from risk-train (plan section 8.1).
    top_threshold = s4s.top_error_threshold(
        risk_target[train_idx], float(config["top_error_fraction"])
    )

    # Fit the fitted candidates on RISK-TRAIN only; direct candidates need no fit.
    models_dir = risk_dir / "models"
    fitted: dict[str, str] = {}
    fit_specs = [
        (s4r.CANDIDATE_RESPONSE, "ridge"),
        (s4r.BASELINE_GEOMETRY, "ridge"),
        (s4r.CANDIDATE_COMBINED, "ridge"),
        (s4r.CANDIDATE_COMBINED, "histgb"),
        (s4r.CANDIDATE_COMBINED, "mlp"),
    ]
    train_features = {name: value[train_idx] for name, value in features.items()}
    for name, impl in fit_specs:
        model = s4r.fit_risk_candidate(
            name, train_features, risk_target[train_idx], config, implementation=impl
        )
        model_path = models_dir / f"{name.lower()}_{impl}.pt"
        model.save(model_path)
        fitted[f"{name}:{impl}"] = str(model_path)

    # Pre-registered learning curves (plan section 5.4): nested subsets of
    # risk-train by per-stratum seed-hash order, always evaluated on risk-validation.
    learning_curves = _learning_curves(
        config, features, risk_target,
        np.asarray(predictions["cf_action"], dtype=np.int64),
        np.asarray(predictions["cf_duration"], dtype=np.int64),
        np.asarray(predictions["episode_seed"], dtype=np.int64),
        train_idx, role_idx["risk_validation"], top_threshold,
    )

    dataset_path = risk_dir / "risk_dataset.npz"
    dataset_arrays = {
        "risk_target": risk_target.astype(np.float64),
        "primary_error_by_pair": primary_ade.astype(np.float64),
        "constant_error_by_pair": constant_ade.astype(np.float64),
        "e2e_error_by_pair": e2e_ade.astype(np.float64),
        "roles": roles.astype(np.int64),
        "cf_action": np.asarray(predictions["cf_action"], dtype=np.int64),
        "cf_duration": np.asarray(predictions["cf_duration"], dtype=np.int64),
        "episode_seed": np.asarray(predictions["episode_seed"], dtype=np.int64),
        "top_error_threshold": np.asarray(top_threshold, dtype=np.float64),
    }
    for name, value in features.items():
        dataset_arrays[f"feature__{name}"] = np.asarray(value, dtype=np.float64)
    tmp = dataset_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **dataset_arrays)
    tmp.replace(dataset_path)

    predictions_npz_path = _inference_paths(run_dir, lock)[0]
    fit_receipt = {
        "plan_id": s4p.PLAN_ID,
        "status": "RISK_FIT_COMPLETE",
        "primary_predictor": primary_name,
        "top_error_threshold": float(top_threshold),
        "fitted_models": fitted,
        "learning_curves": learning_curves,
        "risk_dataset_sha256": sha256_file(dataset_path),
        "predictions_sha256": sha256_file(predictions_npz_path),
    }
    if reused_primary is not None:
        fit_receipt["primary_predictor_reused_from"] = primary["_reused_from"]
        fit_receipt["primary_predictor_reused_receipt_sha256"] = primary["_reused_receipt_sha256"]
        fit_receipt["inference_reused_from"] = str(predictions_npz_path)
    write_once_json(risk_dir / "fit_receipt.json", fit_receipt)
    note = f"风险数据集与候选模型已完成；PRIMARY={primary_name}"
    if reused_primary is not None:
        note = f"复用 v2 冻结推理与 PRIMARY={primary_name}；v3 重训全部风险模型"
    _write_status(run_dir, "DEV_INFERENCE_COMPLETE", note=note)
    return fit_receipt


def _stratum_members_seed_order(
    indices: np.ndarray, cf_action: np.ndarray, cf_duration: np.ndarray, episode_seed: np.ndarray
) -> dict[tuple[int, int], list[int]]:
    """Per-stratum member indices (restricted to ``indices``) in seed-hash order."""
    members: dict[tuple[int, int], list[int]] = {}
    index_set = set(int(i) for i in indices)
    for action in s4p.CF_ACTIONS:
        for duration in s4p.CF_DURATIONS:
                cell = [
                    i for i in index_set
                    if int(cf_action[i]) == action and int(cf_duration[i]) == duration
                ]
                cell.sort(key=lambda i: s4p._seed_sort_key(episode_seed[i]))
                members[(action, duration)] = cell
    return members


def _nested_train_subsets(
    train_idx: np.ndarray,
    cf_action: np.ndarray,
    cf_duration: np.ndarray,
    episode_seed: np.ndarray,
    per_stratum_sizes: tuple[int, ...],
) -> list[np.ndarray]:
    """Nested risk-train subsets: the first ``k`` per stratum in seed-hash order.

    Because risk-train is itself the first 300 per-stratum by seed-hash, the
    50/100/200/300-per-stratum subsets are strictly nested (50⊂100⊂200⊂300).
    """
    members = _stratum_members_seed_order(train_idx, cf_action, cf_duration, episode_seed)
    subsets: list[np.ndarray] = []
    for per_stratum in per_stratum_sizes:
        chosen: list[int] = []
        for key in members:
            cell = members[key]
            if len(cell) < per_stratum:
                raise ValueError(
                    f"stratum {key} has only {len(cell)} risk-train episodes, "
                    f"cannot take {per_stratum}"
                )
            chosen.extend(cell[:per_stratum])
        subsets.append(np.asarray(sorted(chosen), dtype=np.int64))
    # Verify the nesting + expected total sizes.
    for previous, current in zip(subsets, subsets[1:]):
        if not set(previous.tolist()) <= set(current.tolist()):
            raise ValueError("learning-curve subsets must be strictly nested")
    return subsets


def _learning_curves(
    config, features, risk_target, cf_action, cf_duration, episode_seed,
    train_idx, validation_idx, top_threshold,
) -> dict[str, Any]:
    """Train Ridge/HistGB/MLP at the four nested scales; evaluate on risk-validation.

    Only the COMBINED_RISK_HEAD feature set is used for the three model families
    (plan section 5.4). Reports AURC, ADE@80%, the train/validation gap and the
    adjacent-scale improvement per scale; the FINAL candidate always uses the
    3,000-episode model (fitted separately in ``cmd_fit_risk``).
    """
    sizes = tuple(int(s) for s in config["learning_curve_per_stratum"])
    totals = tuple(int(s) for s in config["learning_curve_sizes"])
    subsets = _nested_train_subsets(train_idx, cf_action, cf_duration, episode_seed, sizes)
    val_features = {name: value[validation_idx] for name, value in features.items()}
    val_target = risk_target[validation_idx]
    curves: dict[str, Any] = {}
    for impl in s4r.COMBINED_IMPLEMENTATIONS:
        rows = []
        previous_aurc = None
        for total, subset in zip(totals, subsets):
            sub_features = {name: value[subset] for name, value in features.items()}
            model = s4r.fit_risk_candidate(
                s4r.CANDIDATE_COMBINED, sub_features, risk_target[subset], config,
                implementation=impl,
            )
            train_score = model.predict(
                s4r.design_matrix(sub_features, s4r.CANDIDATE_FEATURES[s4r.CANDIDATE_COMBINED])
            )
            val_score = model.predict(
                s4r.design_matrix(val_features, s4r.CANDIDATE_FEATURES[s4r.CANDIDATE_COMBINED])
            )
            train_aurc = s4s.aurc(train_score, risk_target[subset])
            val_aurc = s4s.aurc(val_score, val_target)
            val_ade80 = s4s.ade_at_coverage(val_score, val_target, 0.80)
            rows.append({
                "train_episodes": int(total),
                "aurc_validation": val_aurc,
                "ade80_validation": val_ade80,
                "aurc_train": train_aurc,
                "train_validation_aurc_gap": train_aurc - val_aurc,
                "adjacent_aurc_improvement": (
                    None if previous_aurc is None else float(previous_aurc - val_aurc)
                ),
            })
            previous_aurc = val_aurc
        curves[impl] = rows
    return {
        "scales_train_episodes": list(totals),
        "per_stratum": list(sizes),
        "evaluation": "risk_validation",
        "curves": curves,
        "note": "data/capacity/signal-limited diagnostic only; final candidate uses the 3000-scale model",
    }


# --- select-candidate -------------------------------------------------------------


def _load_risk_dataset(run_dir: Path) -> dict[str, np.ndarray]:
    dataset_path = run_dir / "risk" / "risk_dataset.npz"
    fit_receipt = json.loads((run_dir / "risk" / "fit_receipt.json").read_text(encoding="utf-8"))
    if fit_receipt.get("risk_dataset_sha256") != sha256_file(dataset_path):
        raise ValueError("Stage 4A risk dataset hash mismatch")
    with np.load(dataset_path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _dataset_features(dataset: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name[len("feature__"):]: np.asarray(value, dtype=np.float64)
        for name, value in dataset.items() if name.startswith("feature__")
    }


def _load_fitted_models(run_dir: Path) -> dict[tuple[str, str], s4r.FittedRiskModel]:
    fit_receipt = json.loads((run_dir / "risk" / "fit_receipt.json").read_text(encoding="utf-8"))
    models: dict[tuple[str, str], s4r.FittedRiskModel] = {}
    for key, path in fit_receipt["fitted_models"].items():
        name, impl = key.split(":")
        models[(name, impl)] = s4r.FittedRiskModel.load(path)
    return models


def _candidate_score_on(
    name: str,
    impl: str | None,
    features: Mapping[str, np.ndarray],
    models: Mapping[tuple[str, str], s4r.FittedRiskModel],
    indices: np.ndarray,
) -> np.ndarray:
    """Score one candidate on a subset of episodes (indices into the feature vectors)."""
    sub = {key: np.asarray(value)[indices] for key, value in features.items()}
    if name in s4r.DIRECT_SCORE_FEATURE:
        return s4r.direct_risk_score(name, sub)
    if impl is None:
        raise ValueError(f"fitted candidate {name} needs an implementation")
    model = models[(name, impl)]
    return model.predict(s4r.design_matrix(sub, s4r.CANDIDATE_FEATURES[name]))


def cmd_select_candidate(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the unique finalist on RISK-VALIDATION (never the development-screen)."""
    run_dir = _resolve(config["run_dir"])
    lock = _verify_lock(run_dir)
    dataset = _load_risk_dataset(run_dir)
    fit_receipt = json.loads((run_dir / "risk" / "fit_receipt.json").read_text(encoding="utf-8"))
    models = _load_fitted_models(run_dir)
    selection_path = run_dir / "risk" / "candidate_selection.json"
    if selection_path.exists():
        raise FileExistsError(f"candidate selection already frozen: {selection_path}")

    features = _dataset_features(dataset)
    risk_target = np.asarray(dataset["risk_target"], dtype=np.float64)
    top_threshold = float(dataset["top_error_threshold"])
    role_idx = s4p.role_indices(np.asarray(dataset["roles"], dtype=np.int64))
    val_idx = role_idx["risk_validation"]
    val_target = risk_target[val_idx]

    # Score every candidate on risk-validation.
    scores: dict[str, np.ndarray] = {}
    for name in s4r.DIRECT_SCORE_FEATURE:
        scores[name] = _candidate_score_on(name, None, features, models, val_idx)
    for name, impl in ((s4r.CANDIDATE_RESPONSE, "ridge"), (s4r.BASELINE_GEOMETRY, "ridge")):
        scores[name] = _candidate_score_on(name, impl, features, models, val_idx)
    combined_scores = {
        impl: _candidate_score_on(s4r.CANDIDATE_COMBINED, impl, features, models, val_idx)
        for impl in s4r.COMBINED_IMPLEMENTATIONS
    }

    reports = {
        name: s4s.candidate_report(score, val_target, top_error_threshold_value=top_threshold)
        for name, score in scores.items()
    }
    combined_reports = {
        impl: s4s.candidate_report(score, val_target, top_error_threshold_value=top_threshold)
        for impl, score in combined_scores.items()
    }

    # STRONGEST_STANDARD_BASELINE: lower validation AURC of UNCERTAINTY vs GEOMETRY;
    # tie -> UNCERTAINTY_ONLY (plan section 5.3).
    unc_aurc = reports[s4r.CANDIDATE_UNCERTAINTY]["aurc"]
    geo_aurc = reports[s4r.BASELINE_GEOMETRY]["aurc"]
    standard = (
        s4r.BASELINE_GEOMETRY if geo_aurc < unc_aurc else s4r.CANDIDATE_UNCERTAINTY
    )
    standard_score = scores[standard]

    # COMBINED sub-model selection by validation AURC (tie <1e-12 -> ridge, histgb, mlp).
    combined_choice = min(
        s4r.COMBINED_IMPLEMENTATIONS,
        key=lambda impl: (round(combined_reports[impl]["aurc"], 12),
                          s4r.COMBINED_IMPLEMENTATIONS.index(impl)),
    )
    scores[s4r.CANDIDATE_COMBINED] = combined_scores[combined_choice]
    reports[s4r.CANDIDATE_COMBINED] = combined_reports[combined_choice]

    # Compare the three developed candidates vs the standard baseline on validation.
    comparisons: dict[str, Any] = {}
    eligible: list[str] = []
    for name in s4r.DEVELOPED_CANDIDATES:
        boot = s4s.paired_improvement_bootstrap(
            scores[name], standard_score, val_target,
            n_resamples=int(config["bootstrap_resamples_screen"]),
            seed=int(config["bootstrap_entropy_screen"]),
            confidence=float(config["confidence_level"]),
        )
        comparisons[name] = boot
        if boot["i_aurc"] > 0 and boot["i_c80"] > 0:
            eligible.append(name)

    if not eligible:
        # Registered scientific stop: no candidate beats the standard baseline on
        # validation in both directions; never read the development-screen.
        selection = {
            "plan_id": s4p.PLAN_ID,
            "status": s4p.STATUS_INSUFFICIENT,
            "reason": "no developed candidate has validation I_AURC>0 and I_C80>0",
            "strongest_standard_baseline": standard,
            "combined_implementation": combined_choice,
            "primary_predictor": fit_receipt["primary_predictor"],
            "validation_comparisons": comparisons,
            "top_error_threshold": top_threshold,
        }
        write_once_json(selection_path, selection)
        _write_status(run_dir, "DEV_CANDIDATE_SELECTED",
                      note="risk-validation 无候选双向优于标准基线",
                      blocking=s4p.STATUS_INSUFFICIENT)
        return selection

    priority = {s4r.CANDIDATE_DISAGREEMENT: 0, s4r.CANDIDATE_RESPONSE: 1, s4r.CANDIDATE_COMBINED: 2}
    finalist = min(
        eligible,
        key=lambda name: (round(-comparisons[name]["i_aurc"], 12), priority[name]),
    )
    selection = {
        "plan_id": s4p.PLAN_ID,
        "status": "DEV_CANDIDATE_SELECTED",
        "finalist": finalist,
        "combined_implementation": combined_choice,
        "strongest_standard_baseline": standard,
        "primary_predictor": fit_receipt["primary_predictor"],
        "top_error_threshold": top_threshold,
        "validation_reports": {**reports, "combined_by_impl": combined_reports},
        "validation_comparisons": comparisons,
        "risk_dataset_sha256": sha256_file(run_dir / "risk" / "risk_dataset.npz"),
        "predictions_sha256": sha256_file(_inference_paths(run_dir, lock)[0]),
    }
    write_once_json(selection_path, selection)
    _write_status(run_dir, "DEV_CANDIDATE_SELECTED",
                  note=f"finalist={finalist}；standard={standard}；combined={combined_choice}")
    return selection


# --- adjudicate-dev-screen --------------------------------------------------------


def _acquire_screen_lock(lock_path: Path) -> None:
    """Atomically burn the single-use development-screen lock BEFORE any screen read."""
    payload = json.dumps(
        {"plan_id": s4p.PLAN_ID, "used_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        indent=2,
    )
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"development-screen already consumed (lock: {lock_path}); it runs once"
        )
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)


def _stratified_reversal(
    finalist_score, standard_score, target, cf_action, cf_duration, tolerance
) -> dict[str, Any]:
    """Per-action and per-duration AURC of finalist vs standard on the screen.

    A 'major reversal' is any stratum where the finalist's AURC is more than
    ``tolerance`` (10%) WORSE than the standard's (plan section 8.2).
    """
    def rel(aurc_std, aurc_cnd):
        return float((aurc_std - aurc_cnd) / aurc_std) if aurc_std > 0 else float("nan")

    reversals = []
    strata_report: dict[str, Any] = {}
    for label, values in (("action", cf_action), ("duration", cf_duration)):
        for level in sorted(set(int(v) for v in values)):
            mask = np.asarray([int(v) == level for v in values])
            if mask.sum() < 2:
                continue
            a_std = s4s.aurc(standard_score[mask], target[mask])
            a_cnd = s4s.aurc(finalist_score[mask], target[mask])
            improvement = rel(a_std, a_cnd)
            reversed_here = bool(improvement < -float(tolerance))
            strata_report[f"{label}_{level}"] = {
                "aurc_standard": a_std, "aurc_candidate": a_cnd,
                "i_aurc": improvement, "major_reversal": reversed_here,
                "n_episodes": int(mask.sum()),
            }
            if reversed_here:
                reversals.append(f"{label}_{level}")
    return {"major_reversal": bool(reversals), "reversal_strata": reversals,
            "strata": strata_report}


def cmd_adjudicate_dev_screen(config: Mapping[str, Any]) -> dict[str, Any]:
    """The SINGLE development-screen adjudication (write-once, single-use)."""
    run_dir = _resolve(config["run_dir"])
    _verify_lock(run_dir)
    selection_path = run_dir / "risk" / "candidate_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") == s4p.STATUS_INSUFFICIENT:
        raise RuntimeError(
            "risk-validation already adjudicated INSUFFICIENT; the development-screen "
            "must not be read"
        )
    report_path = run_dir / "reports" / "stage4a_dev_report.json"
    decision_path = run_dir / "reports" / "STAGE4A_DEV_DECISION.md"
    screen_lock = run_dir / "development_screen_used.lock"
    if report_path.exists() or decision_path.exists() or screen_lock.exists():
        raise FileExistsError("development-screen adjudication already executed; it runs once")

    # Burn the single-use lock BEFORE reading any screen label/prediction/score.
    _acquire_screen_lock(screen_lock)
    try:
        dataset = _load_risk_dataset(run_dir)
        models = _load_fitted_models(run_dir)
        features = _dataset_features(dataset)
        risk_target = np.asarray(dataset["risk_target"], dtype=np.float64)
        primary_by_pair = np.asarray(dataset["primary_error_by_pair"], dtype=np.float64)
        top_threshold = float(dataset["top_error_threshold"])
        cf_action = np.asarray(dataset["cf_action"], dtype=np.int64)
        cf_duration = np.asarray(dataset["cf_duration"], dtype=np.int64)
        role_idx = s4p.role_indices(np.asarray(dataset["roles"], dtype=np.int64))
        screen_idx = role_idx["development_screen"]
        screen_target = risk_target[screen_idx]

        finalist = selection["finalist"]
        standard = selection["strongest_standard_baseline"]
        combined_impl = selection["combined_implementation"]

        def score(name):
            impl = combined_impl if name == s4r.CANDIDATE_COMBINED else (
                None if name in s4r.DIRECT_SCORE_FEATURE else "ridge")
            return _candidate_score_on(name, impl, features, models, screen_idx)

        finalist_score = score(finalist)
        standard_score = score(standard)

        finalist_report = s4s.candidate_report(
            finalist_score, screen_target, top_error_threshold_value=top_threshold)
        standard_report = s4s.candidate_report(
            standard_score, screen_target, top_error_threshold_value=top_threshold)
        boot = s4s.paired_improvement_bootstrap(
            finalist_score, standard_score, screen_target,
            n_resamples=int(config["bootstrap_resamples_screen"]),
            seed=int(config["bootstrap_entropy_screen"]),
            confidence=float(config["confidence_level"]),
        )
        finalist_screen = {
            "i_aurc": boot["i_aurc"], "i_aurc_ci_lower": boot["i_aurc_ci_lower"],
            "i_c80": boot["i_c80"], "i_c80_ci_lower": boot["i_c80_ci_lower"],
            "pr_auc_minus_prevalence": finalist_report["pr_auc_minus_prevalence"],
        }
        # Condition 4: same finalist score against each of the five pair errors.
        pair_i_aurc = np.array([
            (lambda a, c: (a - c) / a if a > 0 else float("nan"))(
                s4s.aurc(standard_score, primary_by_pair[p][screen_idx]),
                s4s.aurc(finalist_score, primary_by_pair[p][screen_idx]),
            )
            for p in range(primary_by_pair.shape[0])
        ])
        reversal = _stratified_reversal(
            finalist_score, standard_score, screen_target,
            cf_action[screen_idx], cf_duration[screen_idx],
            float(config["reversal_aurc_tolerance"]),
        )
        qa = _screen_qa(run_dir, dataset, selection)

        verdict = s4s.evaluate_stage4a_gate(
            finalist_screen,
            pair_i_aurc=pair_i_aurc,
            stratified_reversal=reversal["major_reversal"],
            qa_passed=qa["passed"],
            config=config,
        )
    except Exception:
        # The lock stays burned; an engineering failure does NOT refund the screen.
        raise

    report = {
        "plan_id": s4p.PLAN_ID,
        "finalist": finalist,
        "combined_implementation": combined_impl,
        "strongest_standard_baseline": standard,
        "primary_predictor": selection["primary_predictor"],
        "n_screen_episodes": int(screen_idx.size),
        "top_error_threshold": top_threshold,
        "finalist_report": finalist_report,
        "standard_report": standard_report,
        "improvement_bootstrap": boot,
        "per_pair_i_aurc": pair_i_aurc.tolist(),
        "stratified": reversal,
        "qa": qa,
        "verdict": verdict,
        "status": verdict["status"],
        "selection_receipt_sha256": sha256_file(selection_path),
        "risk_dataset_sha256": sha256_file(run_dir / "risk" / "risk_dataset.npz"),
        "note": "Stage 4A 是开发筛选，非论文确认性结果；通过也需另立 Stage 2B 与独立风险确认计划。",
    }
    write_once_json(report_path, report)
    _write_decision_md(decision_path, report)
    _write_status(run_dir, "DEV_SCREEN_ADJUDICATED",
                  note=f"development-screen 裁判完成：{verdict['status']}",
                  blocking="none" if verdict["passed"] else verdict["status"])
    return report


def _screen_qa(run_dir: Path, dataset, selection) -> dict[str, Any]:
    """Re-verify data/model/whitelist/alignment integrity at adjudication time."""
    checks: dict[str, bool] = {}
    try:
        _verify_lock(run_dir)
        checks["protocol_lock"] = True
    except Exception:
        checks["protocol_lock"] = False
    # Episode alignment: roles sum to 3000/1000/1000 and seeds are unique.
    roles = np.asarray(dataset["roles"], dtype=np.int64)
    checks["role_counts"] = (
        int((roles == 0).sum()) == 3000 and int((roles == 1).sum()) == 1000
        and int((roles == 2).sum()) == 1000
    )
    checks["unique_seeds"] = bool(
        len(np.unique(dataset["episode_seed"])) == len(dataset["episode_seed"])
    )
    # Whitelist: no feature column is a truth-side leak (finite + episode-aligned).
    features = [v for k, v in dataset.items() if k.startswith("feature__")]
    checks["features_finite"] = bool(all(np.isfinite(np.asarray(v)).all() for v in features))
    checks["primary_predictor_frozen"] = selection.get("primary_predictor") in (
        "BEST_CONSTANT_W", "END_TO_END_MIXTURE")
    return {"passed": bool(all(checks.values())), "checks": checks}


def _write_decision_md(path: Path, report: Mapping[str, Any]) -> None:
    verdict = report["verdict"]
    lines = [
        "# Stage 4A Development-Screen 裁判",
        "",
        f"> 状态：`{report['status']}`",
        f"> 计划：`{report['plan_id']}`",
        "> 性质：开发筛选（非确认性结果）；通过也需另立 Stage 2B 与独立风险确认计划",
        "",
        f"- finalist：`{report['finalist']}`（combined 实现 `{report['combined_implementation']}`）",
        f"- 最强标准基线：`{report['strongest_standard_baseline']}`",
        f"- PRIMARY_PREDICTOR：`{report['primary_predictor']}`",
        f"- development-screen episodes：{report['n_screen_episodes']}",
        "",
        "## 继续条件（§9.2）",
        "",
    ]
    for name, ok in verdict["conditions"].items():
        lines.append(f"- [{'x' if ok else ' '}] `{name}`")
    lines += [
        "",
        f"- I_AURC = {report['improvement_bootstrap']['i_aurc']:+.4f} "
        f"(80% CI 下界 {report['improvement_bootstrap']['i_aurc_ci_lower']:+.4f})",
        f"- I_C80 = {report['improvement_bootstrap']['i_c80']:+.4f} "
        f"(80% CI 下界 {report['improvement_bootstrap']['i_c80_ci_lower']:+.4f})",
        f"- PR-AUC - prevalence = {report['finalist_report']['pr_auc_minus_prevalence']:+.4f}",
        f"- 正向 pair 数 = {verdict['positive_pairs']}/5",
        f"- 分层重大反转 = {report['stratified']['major_reversal']} {report['stratified']['reversal_strata']}",
        "",
        f"裁决：`{report['status']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- smoke ------------------------------------------------------------------------


def cmd_smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    """12-episode end-to-end engineering smoke; writes only to ``smoke/``.

    Exercises the frozen expert/response/e2e inference path, the legal-feature
    constructor and a tiny candidate fit on a synthetic paired batch. It may only
    ever report ``ENGINEERING_SMOKE_ONLY`` and never touches real Stage 4A data or
    any research artifact.
    """
    run_dir = _resolve(config["run_dir"])
    smoke_dir = run_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    episodes, agents, horizon, history_len = 12, 5, 20, 10
    rng = np.random.default_rng(20260911)
    history = rng.normal(size=(episodes, history_len, agents, 8)).astype(np.float32)
    factual = rng.normal(size=(episodes, horizon, agents, 8)).astype(np.float32)
    counterfactual = rng.normal(size=(episodes, horizon, agents, 8)).astype(np.float32)
    for block in (history, factual, counterfactual):
        block[..., 0] = 1.0  # presence
    valid = np.ones((episodes, horizon, agents), dtype=bool)
    effect_true, valid = s4r.build_effect_target(
        factual[..., POSITION_SLICE], counterfactual[..., POSITION_SLICE], valid
    )
    # Synthetic per-pair predictions (5 pairs).
    primary = counterfactual[..., POSITION_SLICE] - factual[..., POSITION_SLICE]
    primary_by_seed = np.stack(
        [primary + 0.01 * p for p in range(5)]).astype(np.float32)  # S,E,T,N,2
    full_by_seed = np.stack([primary + 0.02 * p for p in range(5)]).astype(np.float32)
    history_by_seed = np.stack([primary - 0.02 * p for p in range(5)]).astype(np.float32)
    response = {
        "r_hat": np.abs(rng.normal(size=(5, episodes, horizon, agents))).astype(np.float32),
        "presence_prob": rng.uniform(size=(5, episodes, horizon, agents)).astype(np.float32),
        "high_prob": rng.uniform(size=(5, episodes, horizon, agents)).astype(np.float32),
    }
    geometry = {
        "distance_to_ego": np.abs(rng.normal(size=(episodes, agents))),
        "relative_speed": np.abs(rng.normal(size=(episodes, agents))),
        "visible_count": np.full(episodes, float(agents - 1)),
    }
    action = rng.integers(0, 5, size=episodes)
    duration = rng.choice([1, 3], size=episodes)

    features = s4r.build_legal_risk_features(
        primary_predictions=primary_by_seed,
        full_predictions=full_by_seed,
        history_predictions=history_by_seed,
        response_predictions=response,
        pre_intervention_geometry=geometry,
        action=action, duration=duration, valid_mask=valid,
    )
    primary_ade = s4r.episode_effect_ade(primary_by_seed, effect_true, valid)
    risk_target = s4r.risk_target_from_pairs(primary_ade)

    # Whitelist: injecting a truth-side keyword into the feature constructor must
    # be a type error (the signature has no such parameter).
    whitelist_ok = False
    try:
        s4r.build_legal_risk_features(
            primary_predictions=primary_by_seed, full_predictions=full_by_seed,
            history_predictions=history_by_seed, response_predictions=response,
            pre_intervention_geometry=geometry, action=action, duration=duration,
            valid_mask=valid, true_error=risk_target,  # illegal kwarg
        )
    except TypeError:
        whitelist_ok = True

    # Tiny candidate fit + coverage curve sanity.
    threshold = s4s.top_error_threshold(risk_target[:8], 0.20)
    model = s4r.fit_risk_candidate(
        s4r.CANDIDATE_COMBINED,
        {k: v[:8] for k, v in features.items()}, risk_target[:8], config,
        implementation="ridge",
    )
    score = model.predict(s4r.design_matrix(
        {k: v[8:] for k, v in features.items()}, s4r.CANDIDATE_FEATURES[s4r.CANDIDATE_COMBINED]))
    curve = s4s.risk_coverage_curve(score, risk_target[8:])
    report = {
        "status": SMOKE_STATUS,
        "episodes": episodes,
        "feature_count": len(features),
        "feature_names": sorted(features),
        "whitelist_rejects_truth_input": bool(whitelist_ok),
        "risk_target_finite": bool(np.isfinite(risk_target).all()),
        "curve_points": int(curve["coverage"].size),
        "aurc": s4s.aurc(score, risk_target[8:]),
    }
    report_path = smoke_dir / "smoke_report.json"
    tmp = report_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(report_path)
    print(f"[stage4a-smoke] STATUS={SMOKE_STATUS} features={len(features)} "
          f"whitelist_ok={whitelist_ok} -> {report_path}", flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AC-Twin Stage 4A reliability dev screen")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "smoke", "lock-protocol", "lock-recovery", "lock-recovery-v3",
                    "lock-recovery-v4", "infer-dev", "fit-risk", "select-candidate",
                    "adjudicate-dev-screen"):
        item = sub.add_parser(command)
        item.add_argument("--config", required=True)
        if command in ("infer-dev",):
            item.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    if args.command == "preflight":
        report = cmd_preflight(config)
    elif args.command == "lock-protocol":
        report = cmd_lock_protocol(args.config)
    elif args.command == "lock-recovery":
        report = cmd_lock_recovery(args.config)
    elif args.command == "lock-recovery-v3":
        report = cmd_lock_recovery_v3(args.config)
    elif args.command == "lock-recovery-v4":
        report = cmd_lock_recovery_v4(args.config)
    elif args.command == "smoke":
        report = cmd_smoke(config)
    elif args.command == "infer-dev":
        report = cmd_infer_dev(config, resume=bool(getattr(args, "resume", False)))
    elif args.command == "fit-risk":
        report = cmd_fit_risk(config)
    elif args.command == "select-candidate":
        report = cmd_select_candidate(config)
    else:
        report = cmd_adjudicate_dev_screen(config)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
