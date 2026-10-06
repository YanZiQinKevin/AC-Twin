"""Stage 3B non-monotonic routing development driver (plan AC-TWIN-RFER-STAGE3B-2026-09-10-V0.1).

Subcommands:

- ``preflight``           read-only: validate config, re-verify the Stage 3B input
                          binding receipt against the live Stage 3A Plus freeze, the
                          five production checkpoints and the frozen calibration, and
                          re-affirm the Uniform/Direct/Dual capacity audit;
- ``smoke``               a 12-episode / 1-epoch self-contained pass through the
                          predictor->router bridge, all four Tier-1 conditions and the
                          effect-ADE contrasts; may only report ``ENGINEERING_SMOKE_ONLY``;
- ``screen-gate-hidden``  seed-pair 1 mechanical screen of gate hidden dim {32, 64}
                          by validation effect-ADE (tie -> 32); writes a single-use
                          receipt (LONG TASK — started manually by the user);
- ``fit-tier1-five``      train the four Tier-1 conditions (Uniform / Same-information
                          Direct / Dual / Oracle-reference) for all five frozen seed-pairs
                          at the screened gate hidden dim (LONG TASK — user started);
- ``evaluate-gate-t-dev`` the SINGLE Gate T_dev feasibility adjudication over the two
                          historical development sets A (merge split 2) and B
                          (replacement dev); runs once, never a confirmation (LONG TASK);
- ``crossfit``            the fixed 2-fold cross-fit sensitivity re-train of Direct and
                          Dual on out-of-fold features (LONG TASK — user started);
- ``budget``              12 ep/1 epoch router budget re-check against the approved band;
- ``readiness``           print the pre-launch readiness checklist.

Only ``preflight``/``smoke``/``budget``/``readiness`` run automatically at engineering
scale. The long tasks write resumable checkpoints, never overwrite existing receipts
without ``--resume``, and are started by the user in PowerShell.
"""
from __future__ import annotations

import argparse
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

import run_stage3a as s3a  # noqa: E402  (reuse tested loading/batching helpers)
from ac_twin import stage3_training as s3t  # noqa: E402
from ac_twin import stage3b_bridge as bridge  # noqa: E402
from ac_twin import stage3b_inputs as s3bi  # noqa: E402
from ac_twin.progress import ProgressReporter  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3_data import (  # noqa: E402
    compute_effect_eval_mask,
    compute_response_pos,
    split_role_indices,
)
from ac_twin.stage3_response import Stage3ResponsePredictor  # noqa: E402
from ac_twin.stage3_routing import (  # noqa: E402
    CAPACITY_AUDIT_CONDITIONS,
    Condition,
    Stage3Router,
    audit_capacity,
    validate_seed_pairing,
)
from ac_twin import stage3_statistics as s3s  # noqa: E402

SMOKE_STATUS = "ENGINEERING_SMOKE_ONLY"
STAGE3B_CONTRACT_ID = "AC-TWIN-STAGE3B-CONTRACT-2026-09-10-V0.1"
GATE_T_DEV_STATUS_PASS = "ROUTING_DEV_QUALIFIED"
GATE_T_DEV_STATUS_FAIL = s3s.STATUS_DEV_NOT_READY

# Map the short Tier-1 config names to the router Condition enum.
TIER1_CONDITION_MAP = {
    "uniform_suffix": Condition.UNIFORM_SUFFIX,
    "direct": Condition.DIRECT,
    "dual_router": Condition.DUAL_ROUTER,
    "oracle_reference": Condition.ORACLE_REFERENCE,
}
# Default router backbone width: the frozen final response config and the widest
# benchmarked router config (only the GATE hidden dim is searched in Stage 3B).
DEFAULT_ROUTER_HIDDEN_DIM = 96


class RefusedOverwriteError(RuntimeError):
    """A phase artifact already exists and the caller did not pass --resume."""


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Config + binding
# --------------------------------------------------------------------------- #
def _load_config(args) -> dict[str, Any]:
    config = json.loads(_resolve(args.config).read_text(encoding="utf-8"))
    return s3bi.validate_stage3b_config(config, project_root=ROOT)


def _load_binding(run_dir: Path) -> dict[str, Any]:
    receipt = json.loads((run_dir / "input_binding_receipt.json").read_text(encoding="utf-8"))
    return receipt["receipt_payload"]


def _predictor_config(binding, pair_index: int = 0) -> dict[str, Any]:
    """Read the frozen predictor's hidden/response dims and lambda_cls from meta."""
    pred_seed, _ = _pair_seeds(pair_index)
    record = binding["checkpoints"][str(pred_seed)]
    meta = s3t.verify_checkpoint_integrity(record["path"], expected_signature=record["signature"])["meta"]
    cfg = meta["config"]
    return {"hidden_dim": int(cfg["hidden_dim"]),
            "response_feature_dim": int(cfg["response_feature_dim"]),
            "lambda_cls": float(cfg["lambda_cls"])}


def _pair_seeds(pair_index: int) -> tuple[int, int]:
    validate_seed_pairing()
    if not 0 <= pair_index < len(s3bi.PRODUCTION_PREDICTOR_SEEDS):
        raise ValueError("pair_index must be in 0..4")
    return (
        int(s3bi.PRODUCTION_PREDICTOR_SEEDS[pair_index]),
        int(s3bi.ROUTER_SEEDS[pair_index]),
    )


# --------------------------------------------------------------------------- #
# Development bundle + router tensor assembly
# --------------------------------------------------------------------------- #
def _load_dev(config) -> dict[str, Any]:
    """Load the development source, fit/reuse the train-only scaler, scale tensors.

    Also retains the full 20-step factual/counterfactual futures (supervision
    only): the router's effect target and oracle gate are built from them.
    """
    source = _resolve(config["development_source"])
    bundle = s3a.load_development_tensors(source)
    with np.load(source, allow_pickle=True) as archive:
        bundle["factual_full"] = np.asarray(archive["object_factual_future"], dtype=np.float32)
        bundle["counterfactual_full"] = np.asarray(archive["object_counterfactual_future"], dtype=np.float32)
        bundle["episode_seed"] = np.asarray(archive["episode_seed"], dtype=np.int64)
    scaler = s3a.prepare_scaler(bundle["tensors"], bundle["roles"], _resolve(config["run_dir"]))
    bundle["scaler"] = scaler
    bundle["scaled"] = s3a.scale_tensors(bundle["tensors"], scaler)
    return bundle


def _keep_subset(roles) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Restrict to train+validation (split 2 development test is never routed on).

    Returns ``keep`` (sorted absolute episode indices) and local role index arrays
    addressing positions within the subset.
    """
    train = np.asarray(roles["train"])
    val = np.asarray(roles["validation"])
    keep = np.sort(np.concatenate([train, val]))
    position = {int(ep): i for i, ep in enumerate(keep)}
    local = {
        "train": np.array([position[int(ep)] for ep in train], dtype=np.int64),
        "validation": np.array([position[int(ep)] for ep in val], dtype=np.int64),
    }
    return keep, local


def _router_base_tensors(bundle, keep: np.ndarray) -> dict[str, torch.Tensor]:
    """Shared per-episode router inputs/supervision over the ``keep`` subset."""
    scaled, tensors = bundle["scaled"], bundle["tensors"]
    # Full-horizon factual/counterfactual build the effect target and oracle gate
    # (supervision only; never a model input).
    factual_full = np.asarray(bundle["factual_full"])  # (N,20,N,F)
    counterfactual_full = np.asarray(bundle["counterfactual_full"])
    r_true = np.asarray(tensors["r_true"].numpy())  # (N,20,N)
    effect_target = bridge.build_effect_target(factual_full, counterfactual_full)
    base = {
        "history": scaled["history"][keep],
        "factual_suffix": scaled["factual_suffix"][keep],
        "factual_actions": tensors["factual_actions"][keep],
        "counterfactual_actions": tensors["counterfactual_actions"][keep],
        "cf_action": tensors["cf_action"][keep],
        "cf_duration": tensors["cf_duration"][keep],
        "t_star": tensors["t_star"][keep],
        "effect_target": torch.as_tensor(effect_target[keep], dtype=torch.float32),
        "valid": tensors["valid"][keep],
        "oracle_gate": torch.as_tensor(bridge.make_oracle_gate(r_true)[keep], dtype=torch.float32),
    }
    return base


def _pair_feature_tensors(
    bundle, binding, pair_index: int, keep: np.ndarray, device: str, batch_size: int
) -> dict[str, torch.Tensor]:
    """Frozen-predictor response features + calibrated regime probs over ``keep``."""
    pred_seed, _ = _pair_seeds(pair_index)
    record = binding["checkpoints"][str(pred_seed)]
    state = s3t.load_predictor_state(record["path"], expected_signature=record["signature"])
    meta = s3t.verify_checkpoint_integrity(record["path"], expected_signature=record["signature"])["meta"]
    cfg = meta["config"]
    model = Stage3ResponsePredictor(
        hidden_dim=int(cfg["hidden_dim"]), response_feature_dim=int(cfg["response_feature_dim"])
    )
    model.load_state_dict(state)
    # Ordered frozen inference over the keep subset (train+validation only).
    scaled = bundle["scaled"]
    sub = {key: scaled[key][keep] for key in (
        "history", "factual_suffix", "factual_actions", "counterfactual_actions",
        "cf_action", "cf_duration", "t_star",
    )}
    # predictor needs the same keys as _response_loss_fn; supply the truth fields
    # as harmless placeholders (unused in eval-mode feature extraction).
    for key in ("r_true", "regime_labels", "valid"):
        sub[key] = bundle["tensors"][key][keep]
    feats = bridge.predictor_features_ordered(model, sub, np.arange(len(keep)), device, batch_size)
    per_seed_params = binding["calibration"]["per_seed_params"]
    regime_probs = bridge.calibrated_regime_probs_single(
        per_seed_params[pair_index], feats["regime_logits"]
    )
    return {
        "response_features": torch.as_tensor(feats["response_features"], dtype=torch.float32),
        "regime_probs": torch.as_tensor(regime_probs, dtype=torch.float32),
    }


def _assemble_router_tensors(base, pair_feats) -> dict[str, torch.Tensor]:
    tensors = dict(base)
    tensors.update(pair_feats)
    return tensors


# --------------------------------------------------------------------------- #
# Router losses
# --------------------------------------------------------------------------- #
def _router_loss_fn(condition: Condition):
    def loss_fn(model, batch):
        kwargs = dict(
            history=batch["history"], factual_suffix=batch["factual_suffix"],
            factual_actions=batch["factual_actions"],
            counterfactual_actions=batch["counterfactual_actions"],
            response_features=batch["response_features"], regime_probs=batch["regime_probs"],
        )
        if condition == Condition.ORACLE_REFERENCE:
            kwargs["oracle_gate"] = batch["oracle_gate"]
        effect = model(**kwargs)
        return bridge.router_masked_mse(effect, batch["effect_target"], batch["valid"])
    return loss_fn


def _router_sums_fn(condition: Condition):
    def sums_fn(model, batch):
        kwargs = dict(
            history=batch["history"], factual_suffix=batch["factual_suffix"],
            factual_actions=batch["factual_actions"],
            counterfactual_actions=batch["counterfactual_actions"],
            response_features=batch["response_features"], regime_probs=batch["regime_probs"],
        )
        if condition == Condition.ORACLE_REFERENCE:
            kwargs["oracle_gate"] = batch["oracle_gate"]
        effect = model(**kwargs)
        return bridge.router_masked_mse_sums(effect, batch["effect_target"], batch["valid"])
    return sums_fn


def _router_selection_fn(reg: float, ce: float) -> float:
    """Router selection uses the globally-aggregated masked effect-MSE (ce==0)."""
    return float(reg)


# --------------------------------------------------------------------------- #
# Training one router (resumable, no-overwrite)
# --------------------------------------------------------------------------- #
def _train_router(
    config,
    tensors,
    local_roles,
    condition: Condition,
    pair_index: int,
    gate_hidden_dim: int,
    run_dir: Path,
    resume: bool,
    *,
    data_signature: str,
    scaler_signature: str,
    progress_prefix: str,
) -> dict[str, Any]:
    """Train one router (condition x seed-pair) with checkpoint + failure exit.

    Existing artifacts are never overwritten: a COMPLETE run is reused under
    --resume (with integrity re-verification); any pre-existing artifact without
    --resume refuses the phase.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_seed, router_seed = _pair_seeds(pair_index)
    run_name = f"{condition.value}_g{gate_hidden_dim}_pair{pair_index}_r{router_seed}"
    ckpt = run_dir / f"{run_name}.ckpt"
    state_path = run_dir / f"{run_name}.state.json"
    if state_path.is_file():
        prior = json.loads(state_path.read_text(encoding="utf-8"))
        if not resume:
            raise RefusedOverwriteError(
                f"{state_path} already exists (state={prior.get('state')}); "
                "re-run with --resume to reuse/continue, never overwrite"
            )
        if prior.get("state") == "COMPLETE":
            s3t.verify_checkpoint_integrity(
                run_dir / f"{run_name}.pt", expected_signature=prior["signature"]
            )
            return prior
    _atomic_write_json(state_path, {"run": run_name, "state": "IN_PROGRESS", "pair_index": pair_index})
    try:
        s3t.seed_all(router_seed)
        merged_config = {
            "model": {
                "condition": condition.value,
                "hidden_dim": int(config.get("router_hidden_dim", DEFAULT_ROUTER_HIDDEN_DIM)),
                "response_feature_dim": int(tensors["response_features"].shape[-1]),
                "gate_hidden_dim": int(gate_hidden_dim),
            },
            "training": {
                "max_epochs": int(config["max_epochs"]),
                "early_stopping_patience": int(config["early_stopping_patience"]),
                "learning_rate": float(config["learning_rate"]),
                "batch_size": int(config["batch_size"]),
            },
            "seed_pair": [pred_seed, router_seed],
        }
        fingerprint = s3t.compute_run_fingerprint(
            data_signature=data_signature,
            split_name="train",
            config=merged_config,
            scaler_signature=scaler_signature,
            class_weights=np.zeros(4, dtype=np.float64),  # router has no class weights
            contract_id=STAGE3B_CONTRACT_ID,
        )
        train_batches = s3a.make_batches(
            tensors, local_roles["train"], int(config["batch_size"]), seed=router_seed
        )
        val_batches = [
            batch for _, batch in s3a.make_eval_batches(tensors, local_roles["validation"], int(config["batch_size"]))
        ]
        model = Stage3Router(
            condition,
            hidden_dim=int(config.get("router_hidden_dim", DEFAULT_ROUTER_HIDDEN_DIM)),
            response_feature_dim=int(tensors["response_features"].shape[-1]),
            gate_hidden_dim=int(gate_hidden_dim),
            horizon=20,
            seed=router_seed,
        )
        result = s3t.train_model(
            model, _router_loss_fn(condition), train_batches, val_batches,
            seed=router_seed,
            max_epochs=int(config["max_epochs"]),
            early_stopping_patience=int(config["early_stopping_patience"]),
            learning_rate=float(config["learning_rate"]),
            device=config.get("device", "cpu"),
            progress_prefix=progress_prefix,
            checkpoint_path=ckpt,
            resume=resume,
            run_fingerprint=fingerprint,
            val_sums_fn=_router_sums_fn(condition),
            selection_fn=_router_selection_fn,
        )
        meta = {"run": run_name, "condition": condition.value, "pair_index": pair_index,
                "seed_pair": [pred_seed, router_seed], "config": merged_config["model"],
                "run_fingerprint": fingerprint}
        record = s3t.save_predictor_checkpoint(run_dir / f"{run_name}.pt", result["state_dict"], meta=meta)
        summary = {
            "run": run_name, "state": "COMPLETE", "condition": condition.value,
            "pair_index": pair_index, "seed_pair": [pred_seed, router_seed],
            "gate_hidden_dim": int(gate_hidden_dim),
            "config": dict(merged_config["model"]),
            "best_val_loss": result["best_val_loss"],
            "validation_selection_score": result.get("validation_selection_score"),
            "epochs_run": result["epochs_run"],
            "signature": record["signature"], "checkpoint": record["path"],
        }
        _atomic_write_json(state_path, summary)
        return summary
    except Exception as exc:
        _atomic_write_json(state_path, {
            "run": run_name, "state": "FAILED", "pair_index": pair_index, "error": repr(exc),
        })
        raise


def _router_effect_ade(
    model, tensors, indices, device, batch_size, condition: Condition
) -> np.ndarray:
    """Per-episode effect-ADE over the requested episodes (ordered inference)."""
    model.to(device)
    model.eval()
    preds, targets, masks = [], [], []
    with torch.no_grad():
        for _, batch in s3a.make_eval_batches(tensors, np.asarray(indices), batch_size):
            batch = {k: v.to(device) for k, v in batch.items()}
            kwargs = dict(
                history=batch["history"], factual_suffix=batch["factual_suffix"],
                factual_actions=batch["factual_actions"],
                counterfactual_actions=batch["counterfactual_actions"],
                response_features=batch["response_features"], regime_probs=batch["regime_probs"],
            )
            if condition == Condition.ORACLE_REFERENCE:
                kwargs["oracle_gate"] = batch["oracle_gate"]
            effect = model(**kwargs)
            preds.append(effect.cpu().numpy())
            targets.append(batch["effect_target"].cpu().numpy())
            masks.append(batch["valid"].cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    target = np.concatenate(targets, axis=0)
    mask = np.concatenate(masks, axis=0)
    return s3s.effect_ade_per_episode(pred, target, mask)


def _load_trained_router(run_dir: Path, summary, tensors, condition: Condition, device):
    # Prefer the summary's stored model config; fall back to the checkpoint meta
    # so a run whose state predates the config field (or a legacy resume) still loads.
    cfg = summary.get("config")
    if cfg is None:
        cfg = s3t.verify_checkpoint_integrity(
            summary["checkpoint"], expected_signature=summary["signature"]
        )["meta"]["config"]
    model = Stage3Router(
        condition,
        hidden_dim=int(cfg["hidden_dim"]),
        response_feature_dim=int(cfg["response_feature_dim"]),
        gate_hidden_dim=int(cfg["gate_hidden_dim"]),
        horizon=20,
        seed=int(summary["seed_pair"][1]),
    )
    state = s3t.load_predictor_state(summary["checkpoint"], expected_signature=summary["signature"])
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Historical development eval sources A (merge split 2) and B (replacement dev)
# --------------------------------------------------------------------------- #
def _prepare_eval_npz(path: Path, scaler) -> dict[str, Any]:
    """Load an eval npz as a router-ready bundle, scaled transform-only.

    Used for the two historical development diagnostics A and B. The counterfactual
    branch builds ONLY the supervision target / oracle gate; it is never a model
    input. Returns scaled tensors, full futures, r_true/valid and roles.
    """
    s3a.assert_not_sealed_stage2(path)
    with np.load(path, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    required = (
        "object_history", "object_factual_future", "object_counterfactual_future",
        "factual_actions", "counterfactual_actions", "cf_action", "cf_duration", "t_star",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"eval source missing arrays: {missing}")
    factual_full = np.asarray(data["object_factual_future"], dtype=np.float32)
    counterfactual_full = np.asarray(data["object_counterfactual_future"], dtype=np.float32)
    r_pos = compute_response_pos(factual_full, counterfactual_full)
    eval_mask = compute_effect_eval_mask(
        data["object_history"], factual_full, counterfactual_full
    )
    tensors = {
        "history": torch.as_tensor(data["object_history"], dtype=torch.float32),
        "factual_suffix": torch.as_tensor(factual_full[:, :10], dtype=torch.float32),
        "factual_actions": torch.as_tensor(data["factual_actions"], dtype=torch.long),
        "counterfactual_actions": torch.as_tensor(data["counterfactual_actions"], dtype=torch.long),
        "cf_action": torch.as_tensor(data["cf_action"], dtype=torch.long),
        "cf_duration": torch.as_tensor(data["cf_duration"], dtype=torch.long),
        "t_star": torch.as_tensor(data["t_star"], dtype=torch.long),
        "r_true": torch.as_tensor(r_pos, dtype=torch.float32),
        "valid": torch.as_tensor(eval_mask, dtype=torch.bool),
    }
    n = int(len(data["cf_action"]))
    roles = split_role_indices(data["split"]) if "split" in data else {"eval": np.arange(n)}
    return {
        "tensors": tensors,
        "scaled": s3a.scale_tensors(tensors, scaler),
        "factual_full": factual_full,
        "counterfactual_full": counterfactual_full,
        "roles": roles,
        "data_signature": sha256_file(path),
    }


def _eval_router_tensors(eval_bundle, binding, pair_index, indices, device, batch_size):
    """Build router input/supervision tensors for an eval index set (ordered)."""
    indices = np.asarray(indices)
    pred_seed, _ = _pair_seeds(pair_index)
    record = binding["checkpoints"][str(pred_seed)]
    meta = s3t.verify_checkpoint_integrity(record["path"], expected_signature=record["signature"])["meta"]
    state = s3t.load_predictor_state(record["path"], expected_signature=record["signature"])
    model = Stage3ResponsePredictor(
        hidden_dim=int(meta["config"]["hidden_dim"]),
        response_feature_dim=int(meta["config"]["response_feature_dim"]),
    )
    model.load_state_dict(state)
    scaled = eval_bundle["scaled"]
    sub = {key: scaled[key] for key in (
        "history", "factual_suffix", "factual_actions", "counterfactual_actions",
        "cf_action", "cf_duration", "t_star",
    )}
    feats = bridge.predictor_features_ordered(model, sub, indices, device, batch_size)
    regime_probs = bridge.calibrated_regime_probs_single(
        binding["calibration"]["per_seed_params"][pair_index], feats["regime_logits"]
    )
    r_true = eval_bundle["tensors"]["r_true"].numpy()
    effect_target = bridge.build_effect_target(
        eval_bundle["factual_full"], eval_bundle["counterfactual_full"]
    )
    return {
        "history": scaled["history"][indices],
        "factual_suffix": scaled["factual_suffix"][indices],
        "factual_actions": eval_bundle["tensors"]["factual_actions"][indices],
        "counterfactual_actions": eval_bundle["tensors"]["counterfactual_actions"][indices],
        "cf_action": eval_bundle["tensors"]["cf_action"][indices],
        "cf_duration": eval_bundle["tensors"]["cf_duration"][indices],
        "t_star": eval_bundle["tensors"]["t_star"][indices],
        "response_features": torch.as_tensor(feats["response_features"], dtype=torch.float32),
        "regime_probs": torch.as_tensor(regime_probs, dtype=torch.float32),
        "effect_target": torch.as_tensor(effect_target[indices], dtype=torch.float32),
        "valid": eval_bundle["tensors"]["valid"][indices],
        "oracle_gate": torch.as_tensor(bridge.make_oracle_gate(r_true)[indices], dtype=torch.float32),
    }


# --------------------------------------------------------------------------- #
# preflight (read-only, automatic)
# --------------------------------------------------------------------------- #
def cmd_preflight(args) -> int:
    config = _load_config(args)
    run_dir = _resolve(config["run_dir"])
    # Re-verify the whole binding against the live Stage 3A Plus freeze. This
    # recomputes every signature and returns the existing receipt unchanged.
    receipt = s3bi.freeze_stage3b_inputs(args.config, project_root=ROOT)
    payload = receipt["receipt_payload"]
    for seed in s3bi.PRODUCTION_PREDICTOR_SEEDS:
        record = payload["checkpoints"][str(seed)]
        s3t.verify_checkpoint_integrity(record["path"], expected_signature=record["signature"])
    # Re-affirm the Uniform/Direct/Dual capacity audit at the widest gate hidden.
    models = {
        c: Stage3Router(c, hidden_dim=int(config.get("router_hidden_dim", DEFAULT_ROUTER_HIDDEN_DIM)),
                        response_feature_dim=32, gate_hidden_dim=64, horizon=20)
        for c in CAPACITY_AUDIT_CONDITIONS
    }
    audit = audit_capacity(models)
    print(f"[stage3b-preflight] binding_receipt={receipt['receipt_sha256'][:16]} "
          f"checkpoints=5 verified calib={payload['calibration']['selected_candidate']}", flush=True)
    print(f"[stage3b-preflight] capacity within_tol={audit['within_tolerance']} "
          f"max_rel_diff={audit['max_relative_difference']:.4f}", flush=True)
    print(f"[stage3b-preflight] dev_source_sha={payload['development_source_sha256'][:16]} "
          f"OK", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# smoke (12 ep / 1 epoch, automatic; only ENGINEERING_SMOKE_ONLY)
# --------------------------------------------------------------------------- #
def cmd_smoke(args) -> int:
    npz_path = _resolve(args.source)
    if not npz_path.is_file():
        print(f"smoke: development source missing: {npz_path}", flush=True)
        return 1
    limit = int(args.episodes)
    with np.load(npz_path, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    for key in ("object_history", "object_factual_future", "object_counterfactual_future",
                "factual_actions", "counterfactual_actions"):
        data[key] = data[key][:limit]
    for key in ("cf_action", "cf_duration", "episode_seed", "t_star", "split"):
        if key in data:
            data[key] = data[key][:limit]
    reporter = ProgressReporter(total=4, every=1, prefix="stage3b-smoke", heartbeat_seconds=30.0)

    def T(a, d=torch.float32):
        return torch.as_tensor(np.asarray(a), dtype=d)

    history = T(data["object_history"])
    factual_full = T(data["object_factual_future"])
    counterfactual_full = T(data["object_counterfactual_future"])
    factual_suffix = factual_full[:, :10]
    factual_actions = T(data["factual_actions"], torch.long)
    counterfactual_actions = T(data["counterfactual_actions"], torch.long)
    cf_action = T(data["cf_action"], torch.long)
    cf_duration = T(data["cf_duration"], torch.long)
    t_star = T(data["t_star"], torch.long)
    r_pos = compute_response_pos(factual_full.numpy(), counterfactual_full.numpy())
    eval_mask = compute_effect_eval_mask(
        data["object_history"], data["object_factual_future"], data["object_counterfactual_future"]
    )
    effect_target = T(bridge.build_effect_target(factual_full.numpy(), counterfactual_full.numpy()))
    valid = T(eval_mask, torch.bool)
    oracle_gate = T(bridge.make_oracle_gate(r_pos))
    reporter.update(1, details="data-loaded")

    # Tiny predictor -> features (self-contained; not the frozen checkpoints).
    torch.manual_seed(101)
    predictor = Stage3ResponsePredictor(hidden_dim=32, response_feature_dim=16)
    predictor.eval()
    with torch.no_grad():
        out = predictor(history, factual_suffix, factual_actions, counterfactual_actions,
                        cf_action, cf_duration, t_star)
    response_features = out["response_features"].detach()
    regime_probs = torch.softmax(out["regime_logits"].detach(), dim=-1)
    reporter.update(2, details="features-ok")

    tensors = dict(
        history=history, factual_suffix=factual_suffix, factual_actions=factual_actions,
        counterfactual_actions=counterfactual_actions, cf_action=cf_action,
        cf_duration=cf_duration, t_star=t_star, response_features=response_features,
        regime_probs=regime_probs, effect_target=effect_target, valid=valid,
        oracle_gate=oracle_gate,
    )
    all_idx = np.arange(limit)
    ade: dict[str, np.ndarray] = {}
    for name in s3bi.TIER1_CONDITIONS:
        condition = TIER1_CONDITION_MAP[name]
        torch.manual_seed(1101)
        router = Stage3Router(condition, hidden_dim=32, response_feature_dim=16,
                              gate_hidden_dim=32, horizon=20, seed=1101)
        opt = torch.optim.Adam(router.parameters(), lr=1e-3)
        loss_fn = _router_loss_fn(condition)
        router.train()
        for _ in range(max(1, int(args.epochs))):
            batch = {k: v for k, v in tensors.items()}
            opt.zero_grad()
            loss = loss_fn(router, batch)
            loss.backward()
            opt.step()
        ade[name] = _router_effect_ade(router, tensors, all_idx, "cpu", limit, condition)
    reporter.update(3, details="routers-trained")

    c_uniform = float(np.nanmean(ade["uniform_suffix"] - ade["dual_router"]))
    c_direct = float(np.nanmean(ade["direct"] - ade["dual_router"]))
    oracle_gap = float(np.nanmean(ade["uniform_suffix"] - ade["oracle_reference"]))
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": SMOKE_STATUS,
        "episodes": limit, "epochs": int(args.epochs),
        "effect_ade": {k: [float(x) for x in v] for k, v in ade.items()},
        "smoke_c_uniform_point": c_uniform, "smoke_c_direct_point": c_direct,
        "smoke_oracle_gap_point": oracle_gap,
    }
    report_path = output_dir / "smoke_report.json"
    tmp = report_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(report_path)
    reporter.update(4, details=f"C_direct={c_direct:+.4f}")
    print(f"[stage3b-smoke] STATUS={SMOKE_STATUS} | report={report_path} | "
          f"C_direct={c_direct:+.4f} oracle_gap={oracle_gap:+.4f}", flush=True)
    print("[stage3b-smoke] SUCCESS: bridge + 4 Tier-1 forwards + effect-ADE OK "
          "(engineering only, no scientific meaning)", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# Shared long-task context
# --------------------------------------------------------------------------- #
def _dev_context(config, device) -> dict[str, Any]:
    """Load dev bundle, restrict to train+validation, and build shared base tensors."""
    bundle = _load_dev(config)
    keep, local = _keep_subset(bundle["roles"])
    base = _router_base_tensors(bundle, keep)
    return {
        "bundle": bundle, "keep": keep, "local": local, "base": base,
        "data_signature": bundle["data_signature"],
        "scaler_signature": bundle["scaler"].signature(),
    }


def _pair_router_tensors(ctx, binding, pair_index, device, batch_size):
    feats = _pair_feature_tensors(ctx["bundle"], binding, pair_index, ctx["keep"], device, batch_size)
    return _assemble_router_tensors(ctx["base"], feats)


# --------------------------------------------------------------------------- #
# screen-gate-hidden (LONG TASK): pair-1 mechanical screen of {32, 64}
# --------------------------------------------------------------------------- #
def _select_gate_hidden(scored: list[Mapping[str, Any]]) -> int:
    """Mechanically select the gate hidden dim by validation effect-ADE.

    Lowest pair-1 validation effect-ADE wins; a deterministic tie selects 32
    (ascending gate_hidden_dim is the secondary sort key).
    """
    if not scored:
        raise ValueError("no gate-hidden candidates were scored")
    for entry in scored:
        if int(entry["gate_hidden_dim"]) not in s3bi.GATE_HIDDEN_DIM_CANDIDATES:
            raise ValueError(f"gate_hidden_dim {entry['gate_hidden_dim']} not in "
                             f"{s3bi.GATE_HIDDEN_DIM_CANDIDATES}")
    ranked = sorted(scored, key=lambda s: (s["validation_effect_ade"], s["gate_hidden_dim"]))
    return int(ranked[0]["gate_hidden_dim"])


def cmd_screen_gate_hidden(args) -> int:
    config = _load_config(args)
    run_dir = _resolve(config["run_dir"])
    binding = _load_binding(run_dir)
    screen_dir = run_dir / "screen_gate_hidden"
    screen_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = screen_dir / "gate_hidden_selection.json"
    if receipt_path.is_file():
        print(f"[stage3b-screen] REFUSED: {receipt_path} already exists; the gate-hidden "
              "selection is immutable", flush=True)
        return 3
    device = config.get("device", "cpu")
    ctx = _dev_context(config, device)
    tensors = _pair_router_tensors(ctx, binding, 0, device, int(config["batch_size"]))
    candidates = list(s3bi.GATE_HIDDEN_DIM_CANDIDATES)
    reporter = ProgressReporter(total=len(candidates), every=1, prefix="3B-screen", heartbeat_seconds=30.0)
    scored = []
    try:
        for index, gh in enumerate(candidates, start=1):
            summary = _train_router(
                config, tensors, ctx["local"], Condition.DUAL_ROUTER, 0, gh,
                screen_dir, args.resume,
                data_signature=ctx["data_signature"], scaler_signature=ctx["scaler_signature"],
                progress_prefix=f"3B-screen:g{gh}",
            )
            model = _load_trained_router(screen_dir, summary, tensors, Condition.DUAL_ROUTER, device)
            ade = _router_effect_ade(
                model, tensors, ctx["local"]["validation"], device, int(config["batch_size"]),
                Condition.DUAL_ROUTER,
            )
            val_ade = float(np.nanmean(ade))
            scored.append({"gate_hidden_dim": int(gh), "validation_effect_ade": val_ade,
                           "run": summary["run"], "signature": summary["signature"]})
            reporter.update(index, details=f"g{gh} valADE={val_ade:.4f}")
    except RefusedOverwriteError as exc:
        print(f"[stage3b-screen] REFUSED: {exc}", flush=True)
        return 3
    # Mechanical selection on validation effect-ADE only; deterministic tie -> 32.
    selected_gh = _select_gate_hidden(scored)
    selected = next(s for s in scored if s["gate_hidden_dim"] == selected_gh)
    receipt = {
        "plan_id": s3bi.STAGE3B_PLAN_ID,
        "screen_seed_pair": list(_pair_seeds(0)),
        "candidates": scored,
        "selected_gate_hidden_dim": selected["gate_hidden_dim"],
        "selection_basis": "lowest pair-1 validation effect-ADE; deterministic tie selects 32",
        "selected_on": "validation",
    }
    fd = os.open(str(receipt_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, ensure_ascii=False)
    print(f"[stage3b-screen] selected gate_hidden_dim={selected['gate_hidden_dim']} -> {receipt_path}",
          flush=True)
    return 0


def _load_selected_gate_hidden(run_dir: Path) -> int:
    receipt_path = run_dir / "screen_gate_hidden" / "gate_hidden_selection.json"
    if not receipt_path.is_file():
        raise ValueError("screen_gate_hidden/gate_hidden_selection.json missing; run screen-gate-hidden first")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    gh = int(receipt["selected_gate_hidden_dim"])
    if gh not in s3bi.GATE_HIDDEN_DIM_CANDIDATES:
        raise ValueError(f"selected gate_hidden_dim {gh} not in {s3bi.GATE_HIDDEN_DIM_CANDIDATES}")
    return gh


# --------------------------------------------------------------------------- #
# fit-tier1-five (LONG TASK): four Tier-1 conditions x five seed-pairs
# --------------------------------------------------------------------------- #
def cmd_fit_tier1_five(args) -> int:
    config = _load_config(args)
    run_dir = _resolve(config["run_dir"])
    binding = _load_binding(run_dir)
    gate_hidden = _load_selected_gate_hidden(run_dir)
    fit_dir = run_dir / "fit_tier1"
    fit_dir.mkdir(parents=True, exist_ok=True)
    device = config.get("device", "cpu")
    ctx = _dev_context(config, device)
    conditions = [TIER1_CONDITION_MAP[name] for name in s3bi.TIER1_CONDITIONS]
    n_pairs = len(s3bi.PRODUCTION_PREDICTOR_SEEDS)
    reporter = ProgressReporter(total=n_pairs * len(conditions), every=1,
                                prefix="3B-tier1", heartbeat_seconds=30.0)
    per_condition: dict[str, list] = {c.value: [] for c in conditions}
    done = 0
    for pair_index in range(n_pairs):
        tensors = _pair_router_tensors(ctx, binding, pair_index, device, int(config["batch_size"]))
        for condition in conditions:
            try:
                summary = _train_router(
                    config, tensors, ctx["local"], condition, pair_index, gate_hidden,
                    fit_dir, args.resume,
                    data_signature=ctx["data_signature"], scaler_signature=ctx["scaler_signature"],
                    progress_prefix=f"3B-tier1:{condition.value}:p{pair_index}",
                )
            except RefusedOverwriteError as exc:
                print(f"[stage3b-fit-tier1] REFUSED: {exc}", flush=True)
                return 3
            s3t.verify_checkpoint_integrity(summary["checkpoint"], expected_signature=summary["signature"])
            per_condition[condition.value].append({
                "pair_index": pair_index, "seed_pair": summary["seed_pair"],
                "signature": summary["signature"], "checkpoint": summary["checkpoint"],
                "best_val_loss": summary["best_val_loss"], "config": summary["config"],
            })
            done += 1
            reporter.update(done, details=f"{condition.value} p{pair_index} val={summary['best_val_loss']:.4f}")
    # Re-affirm capacity audit at the trained gate hidden dim. The router's
    # response_feature_dim equals the frozen predictor's response_feature_dim.
    rfd = _predictor_config(binding)["response_feature_dim"]
    models = {
        c: Stage3Router(c, hidden_dim=int(config.get("router_hidden_dim", DEFAULT_ROUTER_HIDDEN_DIM)),
                        response_feature_dim=rfd, gate_hidden_dim=gate_hidden, horizon=20)
        for c in CAPACITY_AUDIT_CONDITIONS
    }
    audit = audit_capacity(models)
    _atomic_write_json(fit_dir / "fit_tier1_results.json", {
        "plan_id": s3bi.STAGE3B_PLAN_ID,
        "gate_hidden_dim": gate_hidden,
        "conditions": s3bi.TIER1_CONDITIONS,
        "per_condition": per_condition,
        "capacity_audit": audit,
        "development_data_signature": ctx["data_signature"],
    })
    print(f"[stage3b-fit-tier1] conditions={len(conditions)} pairs={n_pairs} "
          f"capacity_within_tol={audit['within_tolerance']} -> {fit_dir}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# Gate T_dev helpers (Task 3)
# --------------------------------------------------------------------------- #
def _technical_checks(tensors, device) -> dict[str, bool]:
    """Live micro-checks of the structural invariants behind Gate T_dev 'technical_ok'.

    - reconstruction_ok: z_base + z_dyn reconstructs the last-valid hidden state.
    - ablation_ok: Direct's z_route equals Dual's z_route when Dual's gate is 1
      (the only structural difference is the explicit multiplication).
    - no_oracle_leakage: a learned condition refuses an oracle gate input.
    """
    from ac_twin.stage3_routing import dual_channel_decompose

    batch = {k: v[:2].to(device) for k, v in tensors.items()}
    # Derive the router dims from the ACTUAL tensors (the frozen predictor emits
    # response_features of its own width, e.g. 32 — never hardcode it).
    rfd = int(tensors["response_features"].shape[-1])
    horizon = int(tensors["factual_actions"].shape[1])
    agents = int(tensors["history"].shape[2])

    def _router(condition):
        return Stage3Router(condition, hidden_dim=32, response_feature_dim=rfd,
                            gate_hidden_dim=32, horizon=horizon, seed=1101).to(device)

    # reconstruction on the factual sequence encoder path
    router = _router(Condition.DUAL_ROUTER)
    h_seq, valid = router.factual_sequence_encoder(batch["factual_suffix"], None)
    z_base, z_dyn, h_last = dual_channel_decompose(h_seq, valid)
    reconstruction_ok = bool(torch.allclose(z_base + z_dyn, h_last, atol=1e-5))

    kwargs = dict(
        history=batch["history"], factual_suffix=batch["factual_suffix"],
        factual_actions=batch["factual_actions"], counterfactual_actions=batch["counterfactual_actions"],
        response_features=batch["response_features"], regime_probs=batch["regime_probs"],
    )
    # ablation: Direct vs Dual share everything; with Dual's gate pinned to 1 they
    # match exactly (the ONLY structural difference is the explicit multiply).
    direct = _router(Condition.DIRECT)
    dual = _router(Condition.DUAL_ROUTER)
    dual.load_state_dict(direct.state_dict())
    # Pin BOTH gates to 1 (instance attribute shadowing compute_gate, matching the
    # established test_stage3_routing._force_unit_gate pattern): then z_route and
    # the decoder's w input coincide, so Direct and Dual outputs must match exactly.
    unit = lambda q, oracle_gate=None: torch.ones(q.shape[:-1] + (1,), device=q.device)
    with torch.no_grad():
        direct.eval()
        dual.eval()
        direct.compute_gate = unit
        dual.compute_gate = unit
        out_direct = direct(**kwargs)
        out_dual_w1 = dual(**kwargs)
    ablation_ok = bool(torch.allclose(out_direct, out_dual_w1, atol=1e-5))

    # no oracle leakage: a learned condition must refuse an oracle gate.
    try:
        with torch.no_grad():
            direct(**kwargs, oracle_gate=torch.rand(2, horizon, agents, 1, device=device))
        no_oracle_leakage = False
    except ValueError:
        no_oracle_leakage = True
    return {
        "reconstruction_ok": reconstruction_ok,
        "ablation_ok": ablation_ok,
        "no_oracle_leakage": no_oracle_leakage,
    }


def _pair_mean_ade(per_pair_vectors: list[np.ndarray]) -> np.ndarray:
    """Average per-episode ADE over the five seed-pairs -> (E,)."""
    stacked = np.stack([np.asarray(v, dtype=np.float64) for v in per_pair_vectors], axis=0)
    return np.nanmean(stacked, axis=0)


def _ab_contrasts(ade_by_condition: Mapping[str, list[np.ndarray]], *,
                  n_resamples: int, bootstrap_seed: int, confidence: float = 0.80) -> dict[str, Any]:
    """Episode-level paired contrasts for one development set (A or B).

    ``ade_by_condition`` maps condition name -> list of five per-pair episode-ADE
    vectors. Per episode, ADE is averaged over the five pairs, then the two
    contrasts and their 80% paired bootstrap CIs are computed, plus the oracle
    gap and leave-one-pair-out direction stability.
    """
    from ac_twin.stage1_statistics import episode_bootstrap

    ade = {name: _pair_mean_ade(vecs) for name, vecs in ade_by_condition.items()}
    ade_uniform, ade_dual = ade["uniform_suffix"], ade["dual_router"]
    ade_direct, ade_oracle = ade["direct"], ade["oracle_reference"]

    def rel_stat(ref, cand):
        return s3s.make_relative_contrast_statistic(ref, cand)

    def paired_stat(ref, cand):
        return s3s.make_paired_contrast_statistic(ref, cand)

    boot_u = episode_bootstrap(len(ade_uniform), rel_stat(ade_uniform, ade_dual),
                               n_resamples=n_resamples, confidence_level=confidence, seed=bootstrap_seed)
    boot_d = episode_bootstrap(len(ade_direct), rel_stat(ade_direct, ade_dual),
                               n_resamples=n_resamples, confidence_level=confidence, seed=bootstrap_seed)
    rel_uniform = float(np.nanmean(ade_uniform - ade_dual) / np.nanmean(ade_uniform))
    rel_direct = float(np.nanmean(ade_direct - ade_dual) / np.nanmean(ade_direct))
    c_uniform = float(np.nanmean(ade_uniform - ade_dual))
    c_direct = float(np.nanmean(ade_direct - ade_dual))
    oracle_gap = float(np.nanmean(ade_uniform - ade_oracle))

    # leave-one-pair-out direction stability for both contrasts
    lopo = []
    n_pairs = len(ade_by_condition["dual_router"])
    for drop in range(n_pairs):
        kept = {
            name: _pair_mean_ade([v for i, v in enumerate(vecs) if i != drop])
            for name, vecs in ade_by_condition.items()
        }
        cu = float(np.nanmean(kept["uniform_suffix"] - kept["dual_router"]))
        cd = float(np.nanmean(kept["direct"] - kept["dual_router"]))
        lopo.append({"dropped_pair": drop, "c_uniform": cu, "c_direct": cd,
                     "direction_stable": cu > 0 and cd > 0})
    return {
        "ade": {k: float(np.nanmean(v)) for k, v in ade.items()},
        "c_uniform": c_uniform, "c_direct": c_direct,
        "rel_uniform": rel_uniform, "rel_direct": rel_direct,
        "rel_uniform_ci80_lower": float(boot_u["ci_lower"]),
        "rel_direct_ci80_lower": float(boot_d["ci_lower"]),
        "oracle_gap": oracle_gap,
        "leave_one_pair_out": lopo,
        "lopo_direction_stable": all(e["direction_stable"] for e in lopo),
        "n_episodes": int(len(ade_uniform)),
    }


def _six_checks(ab: Mapping[str, Any], technical: Mapping[str, bool], audit_ok: bool) -> dict[str, bool]:
    """The six non-crossfit Gate T_dev conditions for one development set."""
    return {
        "technical_ok": bool(technical["reconstruction_ok"]) and bool(technical["ablation_ok"]) and bool(technical["no_oracle_leakage"]),
        "oracle_beats_uniform": float(ab["oracle_gap"]) > 0.0,
        "dual_vs_direct_point_ge5": float(ab["rel_direct"]) >= s3s.MIN_SUBSTANTIVE_REL,
        "dual_vs_uniform_point_ge5": float(ab["rel_uniform"]) >= s3s.MIN_SUBSTANTIVE_REL,
        "ci80_lower_gt0": float(ab["rel_direct_ci80_lower"]) > 0.0 and float(ab["rel_uniform_ci80_lower"]) > 0.0,
        "audit_ok": bool(audit_ok),
    }


def _evaluate_on_source(config, binding, fit, eval_bundle, eval_indices, device) -> dict[str, list[np.ndarray]]:
    """Per-condition list of five per-pair episode-ADE vectors on one eval source."""
    batch_size = int(config["batch_size"])
    ade_by_condition: dict[str, list[np.ndarray]] = {name: [] for name in s3bi.TIER1_CONDITIONS}
    n_pairs = len(s3bi.PRODUCTION_PREDICTOR_SEEDS)
    for pair_index in range(n_pairs):
        tensors = _eval_router_tensors(eval_bundle, binding, pair_index, eval_indices, device, batch_size)
        for name in s3bi.TIER1_CONDITIONS:
            condition = TIER1_CONDITION_MAP[name]
            summary = fit["per_condition"][condition.value][pair_index]
            model = _load_trained_router(_resolve(config["run_dir"]) / "fit_tier1", summary, tensors, condition, device)
            ade = _router_effect_ade(model, tensors, np.arange(len(eval_indices)), device, batch_size, condition)
            ade_by_condition[name].append(ade)
    return ade_by_condition


def cmd_evaluate_gate_t_dev(args) -> int:
    """The single Gate T_dev feasibility adjudication over development sets A and B.

    Computes the six non-crossfit conditions per set and adjudicates on "both pass".
    If the six pass on both, the report is left in ``AWAITING_CROSSFIT`` and the user
    runs ``crossfit`` to supply the seventh condition; otherwise Gate T_dev fails
    (``STAGE3_DEV_NOT_READY_FOR_CONFIRMATION``) and the single adjudication is final.
    """
    config = _load_config(args)
    run_dir = _resolve(config["run_dir"])
    binding = _load_binding(run_dir)
    fit_path = run_dir / "fit_tier1" / "fit_tier1_results.json"
    if not fit_path.is_file():
        raise ValueError("fit_tier1/fit_tier1_results.json missing; run fit-tier1-five first")
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    report_path = run_dir / "gate_t_dev" / "gate_t_dev_report.json"
    main_path = run_dir / "gate_t_dev" / "gate_t_dev_main_report.json"
    if report_path.is_file() or main_path.is_file():
        print(f"[stage3b-gate-t] REFUSED: Gate T_dev already executed; it runs once.", flush=True)
        return 3
    device = config.get("device", "cpu")
    dev = s3a.load_development_tensors(_resolve(config["development_source"]))
    scaler = s3a.prepare_scaler(dev["tensors"], dev["roles"], run_dir)
    # A = merge split 2 (development_test); B = replacement dev (all-eval).
    eval_a = _prepare_eval_npz(_resolve(config["development_source"]), scaler)
    idx_a = eval_a["roles"]["development_test"]
    eval_b = _prepare_eval_npz(_resolve(config["historical_dev_b"]), scaler)
    idx_b = eval_b["roles"].get("eval", np.arange(len(eval_b["tensors"]["cf_action"])))

    audit_ok = bool(fit.get("capacity_audit", {}).get("within_tolerance", False))
    reporter = ProgressReporter(total=2, every=1, prefix="3B-gateT", heartbeat_seconds=30.0)
    results = {}
    technical = None
    for label, eval_bundle, indices in (("A", eval_a, idx_a), ("B", eval_b, idx_b)):
        ade_by_condition = _evaluate_on_source(config, binding, fit, eval_bundle, indices, device)
        if technical is None:
            sample = _eval_router_tensors(eval_bundle, binding, 0, indices[:2], device, int(config["batch_size"]))
            technical = _technical_checks(sample, device)
        ab = _ab_contrasts(ade_by_condition, n_resamples=int(args.n_resamples),
                           bootstrap_seed=int(args.bootstrap_seed))
        ab["checks"] = _six_checks(ab, technical, audit_ok)
        ab["data_signature"] = eval_bundle["data_signature"]
        results[label] = ab
        reporter.update(1 if label == "A" else 2,
                        details=f"{label}: rel_direct={ab['rel_direct']:.3f} rel_uniform={ab['rel_uniform']:.3f}")

    six_pass = {label: all(r["checks"].values()) for label, r in results.items()}
    both_pass = all(six_pass.values())
    out = {
        "plan_id": s3bi.STAGE3B_PLAN_ID,
        "gate": "T_dev",
        "sets": results,
        "six_checks_pass": six_pass,
        "technical": technical,
        "audit_ok": audit_ok,
        "n_resamples": int(args.n_resamples),
        "confidence_level": 0.80,
        "note": "A/B 分别计算并各自裁判，episode 不合并；开发门非确认结果。",
    }
    gate_dir = run_dir / "gate_t_dev"
    gate_dir.mkdir(parents=True, exist_ok=True)
    if both_pass:
        out["status"] = "AWAITING_CROSSFIT"
        _atomic_write_json(main_path, out)
        print(f"[stage3b-gate-t] six checks pass on A and B -> AWAITING_CROSSFIT; "
              f"run `crossfit` to complete condition 6. main_report={main_path}", flush=True)
        return 0
    # Final single-use failure verdict (cross-fit never runs).
    out["status"] = GATE_T_DEV_STATUS_FAIL
    out["passed"] = False
    out["verdict"] = {"passed": False, "checks": {f"six_{k}": v for k, v in six_pass.items()},
                      "status": GATE_T_DEV_STATUS_FAIL}
    _atomic_write_json(report_path, out)
    print(f"[stage3b-gate-t] FAIL: six-check pass A={six_pass['A']} B={six_pass['B']} -> "
          f"{GATE_T_DEV_STATUS_FAIL}. report={report_path}", flush=True)
    return 2


# --------------------------------------------------------------------------- #
# crossfit (LONG TASK): fixed 2-fold out-of-fold Direct/Dual sensitivity
# --------------------------------------------------------------------------- #
def _response_tensor_dict(bundle) -> dict[str, torch.Tensor]:
    """Scaled predictor inputs + response supervision over ALL episodes."""
    scaled, tensors = bundle["scaled"], bundle["tensors"]
    rt = {k: scaled[k] for k in (
        "history", "factual_suffix", "factual_actions", "counterfactual_actions",
        "cf_action", "cf_duration", "t_star",
    )}
    rt["r_true"] = tensors["r_true"]
    rt["regime_labels"] = tensors["regime_labels"]
    rt["valid"] = tensors["valid"]
    return rt


def _train_fold_predictor(config, bundle, binding, fit_idx, val_idx, pair_index, device):
    """Train a response predictor on one fold to make out-of-fold features.

    Uses the frozen production config (hidden/response dims and lambda_cls) read
    from the bound checkpoint meta; early stopping on the development validation
    split. Intermediate and deterministic given the predictor seed; not separately
    checkpointed (the cross-fit ROUTERS are the resumable deliverable).
    """
    from ac_twin.stage3_response import selection_score

    pred_seed, _ = _pair_seeds(pair_index)
    pcfg = _predictor_config(binding, pair_index)
    rt = _response_tensor_dict(bundle)
    class_weights = s3a.compute_class_weights(
        rt["regime_labels"][fit_idx].numpy(), rt["valid"][fit_idx].numpy()
    )
    weights_t = torch.as_tensor(class_weights)
    batches = s3a.make_batches(rt, np.asarray(fit_idx), int(config["batch_size"]), seed=pred_seed)
    val_batches = [b for _, b in s3a.make_eval_batches(rt, np.asarray(val_idx), int(config["batch_size"]))]
    s3t.seed_all(pred_seed)
    model = Stage3ResponsePredictor(
        hidden_dim=pcfg["hidden_dim"], response_feature_dim=pcfg["response_feature_dim"]
    )
    result = s3t.train_model(
        model, s3a._response_loss_fn(weights_t, lambda_cls=pcfg["lambda_cls"]),
        batches, val_batches, seed=pred_seed,
        max_epochs=int(config["max_epochs"]),
        early_stopping_patience=int(config["early_stopping_patience"]),
        learning_rate=float(config["learning_rate"]),
        device=device,
        progress_prefix=f"3B-crossfit-pred:p{pair_index}",
        val_sums_fn=s3a._response_sums_fn(weights_t),
        selection_fn=selection_score,
    )
    model.load_state_dict(result["state_dict"])
    model.to(device)
    model.eval()
    return model


def _outof_fold_train_features(config, bundle, binding, pair_index, device):
    """Out-of-fold response features + calibrated regime probs for all train episodes.

    Retrains the pair's predictor on each cross-fit fold and generates features for
    the held-out fold, then concatenates to cover every train episode exactly once.
    """
    roles = bundle["roles"]
    train_idx = np.asarray(roles["train"])
    tensors = bundle["tensors"]
    splits = s3t.build_crossfit_splits(
        train_idx,
        tensors["cf_action"].numpy(), tensors["cf_duration"].numpy(), bundle["episode_seed"],
    )
    fold0, fold1 = np.asarray(splits["fold0"]), np.asarray(splits["fold1"])
    val_idx = np.asarray(roles["validation"])
    scaled = bundle["scaled"]
    sub = {key: scaled[key] for key in (
        "history", "factual_suffix", "factual_actions", "counterfactual_actions",
        "cf_action", "cf_duration", "t_star",
    )}
    feats_by_episode: dict[int, dict[str, np.ndarray]] = {}
    # predictor fit on fold0 -> features for fold1, and vice versa.
    for fit_fold, target_fold in ((fold0, fold1), (fold1, fold0)):
        model = _train_fold_predictor(config, bundle, binding, fit_fold, val_idx, pair_index, device)
        out = bridge.predictor_features_ordered(
            model, sub, target_fold, device, int(config["batch_size"])
        )
        for pos, ep in enumerate(target_fold):
            feats_by_episode[int(ep)] = {
                "response_features": out["response_features"][pos],
                "regime_logits": out["regime_logits"][pos],
            }
    rf = np.stack([feats_by_episode[int(ep)]["response_features"] for ep in train_idx], axis=0)
    lg = np.stack([feats_by_episode[int(ep)]["regime_logits"] for ep in train_idx], axis=0)
    probs = bridge.calibrated_regime_probs_single(
        binding["calibration"]["per_seed_params"][pair_index], lg
    )
    return {
        "response_features": torch.as_tensor(rf, dtype=torch.float32),
        "regime_probs": torch.as_tensor(probs, dtype=torch.float32),
        "train_index_order": train_idx,
    }


def cmd_crossfit(args) -> int:  # long task
    """Fixed 2-fold cross-fit sensitivity re-train of Direct and Dual (plan §4.6).

    Requires the Gate T_dev main report in ``AWAITING_CROSSFIT``. Retrains Direct
    and Dual on out-of-fold train features, computes ``C_direct`` on development
    sets A and B, then runs the full seven-condition Gate T_dev adjudication ONCE.
    Any ``C_direct <= 0`` mechanically stops at STAGE3_DEV_NOT_READY_FOR_CONFIRMATION.
    """
    config = _load_config(args)
    run_dir = _resolve(config["run_dir"])
    binding = _load_binding(run_dir)
    gate_dir = run_dir / "gate_t_dev"
    main_path = gate_dir / "gate_t_dev_main_report.json"
    report_path = gate_dir / "gate_t_dev_report.json"
    if report_path.is_file():
        print("[stage3b-crossfit] REFUSED: Gate T_dev already adjudicated; it runs once.", flush=True)
        return 3
    if not main_path.is_file():
        print("[stage3b-crossfit] REFUSED: run `evaluate-gate-t-dev` first (main report missing).",
              flush=True)
        return 3
    main_report = json.loads(main_path.read_text(encoding="utf-8"))
    if main_report.get("status") != "AWAITING_CROSSFIT":
        print(f"[stage3b-crossfit] REFUSED: main report status={main_report.get('status')}",
              flush=True)
        return 3

    device = config.get("device", "cpu")
    gate_hidden = _load_selected_gate_hidden(run_dir)
    bundle = _load_dev(config)
    keep, local = _keep_subset(bundle["roles"])
    base = _router_base_tensors(bundle, keep)
    val_idx_abs = np.asarray(bundle["roles"]["validation"])
    n_pairs = len(s3bi.PRODUCTION_PREDICTOR_SEEDS)

    # Eval sources A/B (production-predictor features; A/B are out-of-sample for them).
    scaler = bundle["scaler"]
    eval_a = _prepare_eval_npz(_resolve(config["development_source"]), scaler)
    idx_a = eval_a["roles"]["development_test"]
    eval_b = _prepare_eval_npz(_resolve(config["historical_dev_b"]), scaler)
    idx_b = eval_b["roles"].get("eval", np.arange(len(eval_b["tensors"]["cf_action"])))

    crossfit_dir = run_dir / "crossfit"
    crossfit_dir.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter(total=n_pairs * 2, every=1, prefix="3B-crossfit", heartbeat_seconds=30.0)
    # Per set, per condition: list of five per-pair episode-ADE vectors.
    cf_ade = {"A": {"direct": [], "dual_router": []}, "B": {"direct": [], "dual_router": []}}
    done = 0
    for pair_index in range(n_pairs):
        oof = _outof_fold_train_features(config, bundle, binding, pair_index, device)
        # Build router tensors over keep: train rows use out-of-fold features,
        # validation rows use the production predictor's features.
        prod = _pair_feature_tensors(bundle, binding, pair_index, keep, device, int(config["batch_size"]))
        tensors = dict(base)
        rf = prod["response_features"].numpy().copy()
        rp = prod["regime_probs"].numpy().copy()
        train_pos = local["train"]
        rf[train_pos] = oof["response_features"].numpy()
        rp[train_pos] = oof["regime_probs"].numpy()
        tensors["response_features"] = torch.as_tensor(rf, dtype=torch.float32)
        tensors["regime_probs"] = torch.as_tensor(rp, dtype=torch.float32)

        for condition in (Condition.DIRECT, Condition.DUAL_ROUTER):
            try:
                summary = _train_router(
                    config, tensors, local, condition, pair_index, gate_hidden,
                    crossfit_dir, args.resume,
                    data_signature=bundle["data_signature"], scaler_signature=bundle["scaler"].signature(),
                    progress_prefix=f"3B-crossfit:{condition.value}:p{pair_index}",
                )
            except RefusedOverwriteError as exc:
                print(f"[stage3b-crossfit] REFUSED: {exc}", flush=True)
                return 3
            model = _load_trained_router(crossfit_dir, summary, tensors, condition, device)
            for label, eval_bundle, indices in (("A", eval_a, idx_a), ("B", eval_b, idx_b)):
                ev_tensors = _eval_router_tensors(eval_bundle, binding, pair_index, indices, device, int(config["batch_size"]))
                ade = _router_effect_ade(model, ev_tensors, np.arange(len(indices)), device, int(config["batch_size"]), condition)
                key = "direct" if condition == Condition.DIRECT else "dual_router"
                cf_ade[label][key].append(ade)
            done += 1
            reporter.update(done, details=f"{condition.value} p{pair_index}")

    # C_direct per set (episode-level, averaged over pairs).
    c_direct = {}
    for label in ("A", "B"):
        direct_ep = _pair_mean_ade(cf_ade[label]["direct"])
        dual_ep = _pair_mean_ade(cf_ade[label]["dual_router"])
        c_direct[label] = float(np.nanmean(direct_ep - dual_ep))
    not_reversed = bool(c_direct["A"] > 0.0 and c_direct["B"] > 0.0)

    # Final seven-condition Gate T_dev adjudication per set; both must pass.
    verdicts = {}
    for label in ("A", "B"):
        ab = main_report["sets"][label]
        results = {
            "reconstruction_ok": main_report["technical"]["reconstruction_ok"],
            "ablation_ok": main_report["technical"]["ablation_ok"],
            "no_oracle_leakage": main_report["technical"]["no_oracle_leakage"],
            "oracle_beats_uniform": float(ab["oracle_gap"]) > 0.0,
            "rel_direct": float(ab["rel_direct"]),
            "rel_uniform": float(ab["rel_uniform"]),
            "rel_direct_ci80_lower": float(ab["rel_direct_ci80_lower"]),
            "rel_uniform_ci80_lower": float(ab["rel_uniform_ci80_lower"]),
            "crossfit_direct_not_reversed": not_reversed,
            "audit_ok": bool(main_report["audit_ok"]),
        }
        verdicts[label] = s3s.evaluate_gate_t_dev(results)
    passed = bool(verdicts["A"]["passed"] and verdicts["B"]["passed"])
    final = {
        "plan_id": s3bi.STAGE3B_PLAN_ID,
        "gate": "T_dev",
        "main_report": main_report,
        "crossfit": {
            "c_direct": c_direct,
            "crossfit_direct_not_reversed": not_reversed,
            "folds": 2,
            "conditions": ["direct", "dual_router"],
        },
        "verdicts": verdicts,
        "passed": passed,
        "status": GATE_T_DEV_STATUS_PASS if passed else GATE_T_DEV_STATUS_FAIL,
        "note": "Gate T_dev 只执行一次；A/B 均通过方判通过。开发门非确认结果。",
    }
    fd = os.open(str(report_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2, ensure_ascii=False)
    print(f"[stage3b-crossfit] C_direct A={c_direct['A']:+.4f} B={c_direct['B']:+.4f} "
          f"not_reversed={not_reversed} Gate_T_dev passed={passed} -> {final['status']}", flush=True)
    return 0 if passed else 2


# --------------------------------------------------------------------------- #
# budget (Task 4): 12 ep / 1 epoch router budget re-check against the approved band
# --------------------------------------------------------------------------- #
# Approved Stage 3 routing budget (Stage 3 v0.2 §5.3, user-confirmed 2026-09-09):
# router-timing point 15 GPU h, interval [7.5, 30] h. Re-use it if the recomputed
# Stage 3B + cross-fit estimate stays within +/-25%; only escalate if it exceeds.
APPROVED_ROUTER_POINT_H = 15.0
APPROVED_ROUTER_INTERVAL = (7.5, 30.0)
BUDGET_TOLERANCE = 0.25


def cmd_budget(args) -> int:
    config = _load_config(args)
    device = "cuda" if (config.get("device") == "cuda" and torch.cuda.is_available()) else "cpu"
    npz_path = _resolve(config["development_source"])
    with np.load(npz_path, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    n = int(args.episodes)
    for key in ("object_history", "object_factual_future", "object_counterfactual_future",
                "factual_actions", "counterfactual_actions"):
        data[key] = data[key][:n]
    history = torch.as_tensor(data["object_history"], dtype=torch.float32)
    factual_suffix = torch.as_tensor(data["object_factual_future"][:, :10], dtype=torch.float32)
    factual_actions = torch.as_tensor(data["factual_actions"], dtype=torch.long)
    counterfactual_actions = torch.as_tensor(data["counterfactual_actions"], dtype=torch.long)
    agents = history.shape[2]
    rfd = 32
    steps = factual_actions.shape[1]
    tensors = {
        "history": history, "factual_suffix": factual_suffix,
        "factual_actions": factual_actions, "counterfactual_actions": counterfactual_actions,
        "response_features": torch.zeros(n, steps, agents, rfd),
        "regime_probs": torch.full((n, steps, agents, 4), 0.25),
        "effect_target": torch.zeros(n, steps, agents, 2),
        "valid": torch.ones(n, steps, agents, dtype=torch.bool),
    }
    router = Stage3Router(
        Condition.DUAL_ROUTER,
        hidden_dim=int(config.get("router_hidden_dim", DEFAULT_ROUTER_HIDDEN_DIM)),
        response_feature_dim=rfd, gate_hidden_dim=int(s3bi.GATE_HIDDEN_DIM_CANDIDATES[-1]),
        horizon=20, seed=int(s3bi.ROUTER_SEEDS[0]),
    )
    loss_fn = _router_loss_fn(Condition.DUAL_ROUTER)
    device_obj = torch.device(device)
    router.to(device_obj)
    opt = torch.optim.Adam(router.parameters(), lr=1e-3)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    router.train()
    started = time.perf_counter()
    for _ in range(max(1, int(args.epochs))):
        batch = {k: v.to(device_obj) for k, v in tensors.items()}
        opt.zero_grad()
        loss = loss_fn(router, batch)
        loss.backward()
        opt.step()
    epoch_s = (time.perf_counter() - started) / max(int(args.epochs), 1)
    peak_gb = (torch.cuda.max_memory_allocated() / 1024 ** 3) if device == "cuda" else 0.0

    train_episodes = int(config.get("train_episodes", 1400))
    scale = train_episodes / max(n, 1)
    per_run_s = epoch_s * scale * int(config["max_epochs"])
    tier1_runs = len(s3bi.TIER1_CONDITIONS) * len(s3bi.PRODUCTION_PREDICTOR_SEEDS)
    crossfit_runs = 2 * len(s3bi.PRODUCTION_PREDICTOR_SEEDS)
    screen_runs = len(s3bi.GATE_HIDDEN_DIM_CANDIDATES)
    total_runs = tier1_runs + crossfit_runs + screen_runs
    hours = per_run_s * total_runs / 3600.0
    lo, hi = APPROVED_ROUTER_INTERVAL
    point = APPROVED_ROUTER_POINT_H
    within = (lo * (1 - BUDGET_TOLERANCE) <= hours <= hi * (1 + BUDGET_TOLERANCE)) and (
        abs(hours - point) <= BUDGET_TOLERANCE * point
    )
    review = {
        "plan_id": s3bi.STAGE3B_PLAN_ID,
        "device": device,
        "benchmark": {"episodes": n, "epochs": int(args.epochs),
                      "router_epoch_seconds": epoch_s, "peak_memory_gb": peak_gb,
                      "date": time.strftime("%Y-%m-%d")},
        "extrapolation": {"train_episodes": train_episodes, "max_epochs": int(config["max_epochs"]),
                          "per_run_seconds": per_run_s,
                          "tier1_runs": tier1_runs, "crossfit_runs": crossfit_runs,
                          "screen_runs": screen_runs, "total_runs": total_runs},
        "stage3b_plus_crossfit_hours": hours,
        "approved": {"point_hours": point, "interval_hours": [lo, hi], "tolerance": BUDGET_TOLERANCE},
        "within_approved_budget": bool(within),
        "verdict": ("REUSE_APPROVED_BUDGET" if within else "ESCALATE_TO_USER"),
    }
    out_path = _resolve(args.output)
    _atomic_write_json(out_path, review)
    print(f"[stage3b-budget] device={device} epoch={epoch_s:.3f}s/{n}ep "
          f"estimate={hours:.2f}h (approved point {point}h, interval {lo}-{hi}h) "
          f"within={within} -> {review['verdict']}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# readiness checklist
# --------------------------------------------------------------------------- #
READINESS_CHECKLIST = [
    "Stage 3B 计划 AC-TWIN-RFER-STAGE3B-2026-09-10-V0.1 已批准（软件准备范围）",
    "Stage 3B 输入绑定 receipt 已冻结并可只读复验（5 checkpoint + 校准来源）",
    "feature-only 正式推理 loader 就绪；CF future/R/label 注入即失败",
    "路由器只用 split 0/1 训练；历史 A/B 仅作非确认 Gate T_dev 诊断",
    "五组 predictor-router 一一配对，禁 best-seed / 5x5",
    "Uniform/Direct/Dual 三口径参数差<=1%（容量审计通过）",
    "gate hidden dim 仅 {32,64}，pair-1 validation 机械选择，并列取 32",
    "Tier 1 只训练 Uniform/Direct/Dual/Oracle-reference 四条件",
    "Direct/Dual 唯一结构差异为 w 是否乘 z_dyn（w=1 等价已验证）",
    "12 ep/1 epoch smoke 通过，状态仅 ENGINEERING_SMOKE_ONLY",
    "router 预算复核落在已批准 15h / [7.5,30]h ±25% 内",
    "全量回归 pytest -W error 通过；长任务由用户手动启动",
]


def cmd_readiness(args) -> int:
    print("[stage3b-readiness] 启动前 checklist：", flush=True)
    for index, item in enumerate(READINESS_CHECKLIST, start=1):
        print(f"  {index:2d}. [ ] {item}", flush=True)
    print("[stage3b-readiness] 全部确认后，由用户手动启动 screen/fit-tier1/gate-t/crossfit。",
          flush=True)
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = "experiments/stage3_routing/stage3b/stage3b_config.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AC-Twin Stage 3B routing development driver")
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="read-only config + binding + capacity verification")
    preflight.add_argument("--config", default=DEFAULT_CONFIG)
    preflight.set_defaults(func=cmd_preflight)

    smoke = sub.add_parser("smoke", help="12-episode/1-epoch engineering smoke only")
    smoke.add_argument("--source", default="data/object_centric_stage/merge_2000/object_twinworld_2000.npz")
    smoke.add_argument("--episodes", type=int, default=12)
    smoke.add_argument("--epochs", type=int, default=1)
    smoke.add_argument("--output-dir", default="experiments/stage3_routing/stage3b/smoke")
    smoke.set_defaults(func=cmd_smoke)

    screen = sub.add_parser("screen-gate-hidden", help="pair-1 gate hidden screen (LONG, user-started)")
    screen.add_argument("--config", default=DEFAULT_CONFIG)
    screen.add_argument("--resume", action="store_true")
    screen.set_defaults(func=cmd_screen_gate_hidden)

    fit = sub.add_parser("fit-tier1-five", help="four Tier-1 conditions x five pairs (LONG, user-started)")
    fit.add_argument("--config", default=DEFAULT_CONFIG)
    fit.add_argument("--resume", action="store_true")
    fit.set_defaults(func=cmd_fit_tier1_five)

    gate = sub.add_parser("evaluate-gate-t-dev", help="single Gate T_dev A/B adjudication (LONG, user-started)")
    gate.add_argument("--config", default=DEFAULT_CONFIG)
    gate.add_argument("--n-resamples", type=int, default=2000)
    gate.add_argument("--bootstrap-seed", type=int, default=20260910)
    gate.set_defaults(func=cmd_evaluate_gate_t_dev)

    crossfit = sub.add_parser("crossfit", help="fixed 2-fold Direct/Dual sensitivity (LONG, user-started)")
    crossfit.add_argument("--config", default=DEFAULT_CONFIG)
    crossfit.add_argument("--resume", action="store_true")
    crossfit.set_defaults(func=cmd_crossfit)

    budget = sub.add_parser("budget", help="12 ep/1 epoch router budget re-check")
    budget.add_argument("--config", default=DEFAULT_CONFIG)
    budget.add_argument("--episodes", type=int, default=12)
    budget.add_argument("--epochs", type=int, default=1)
    budget.add_argument("--output", default="experiments/stage3_routing/stage3b/BUDGET_REVIEW.json")
    budget.set_defaults(func=cmd_budget)

    ready = sub.add_parser("readiness", help="print the pre-launch readiness checklist")
    ready.set_defaults(func=cmd_readiness)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
