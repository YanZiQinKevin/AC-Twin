"""Stage 3 unified training: seed-pairs, cross-fit, checkpoints and compute budget.

Training is deterministic per seed and enforces the frozen one-to-one
predictor<->router pairing. The main training path may use a frozen full-train
response predictor to generate train response features, but this in-sample risk
is registered and a fixed 2-fold cross-fit sensitivity re-trains Direct and Dual
once on out-of-fold features. All long training jobs are started manually by the
user; this module only prepares code, checkpoints and the compute budget.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch

from ac_twin.progress import ProgressReporter
from ac_twin.stage3_routing import (
    SEED_PAIRS,
    Condition,
    Stage3Router,
    assign_crossfit_folds,
    validate_seed_pairing,
)


def seed_all(seed: int) -> None:
    """Seed python/numpy/torch (incl. CUDA) for deterministic training.

    Repair plan v0.1 §2.2: strict deterministic algorithms (a non-deterministic
    op raises instead of silently warning) and deterministic cuDNN. Callers must
    invoke this BEFORE constructing any model/optimizer/random tensor, and set
    CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment for CUDA runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Code/contract identifier bound into every run fingerprint (item 5). Bumping it
# invalidates all prior checkpoints so a stale run can never be silently resumed.
STAGE3A_CONTRACT_ID = "AC-TWIN-STAGE3A-REPAIR-CONTRACT-2026-09-09-V0.1"


def compute_run_fingerprint(
    *,
    data_signature: str,
    split_name: str,
    config: Mapping[str, Any],
    scaler_signature: str,
    class_weights: np.ndarray | list,
    contract_id: str = STAGE3A_CONTRACT_ID,
) -> str:
    """SHA256 run identity covering data, split, full config, scaler and weights.

    A resume is only accepted when every one of these matches the checkpoint,
    so a run cannot be continued against different data, a different split, a
    changed config, a re-fit scaler or recomputed class weights.
    """
    digest = hashlib.sha256()
    digest.update(str(data_signature).encode("utf-8"))
    digest.update(str(split_name).encode("utf-8"))
    canonical_config = json.dumps(config, sort_keys=True, default=str)
    digest.update(canonical_config.encode("utf-8"))
    digest.update(str(scaler_signature).encode("utf-8"))
    digest.update(np.asarray(class_weights, dtype=np.float64).tobytes())
    digest.update(str(contract_id).encode("utf-8"))
    return digest.hexdigest()


# Contract id for the select-final receipt (distinct from per-run fingerprints).
SELECT_FINAL_CONTRACT_ID = STAGE3A_CONTRACT_ID + "-SELECT-FINAL"


def compute_selection_receipt_signature(
    *,
    final_config: Mapping[str, Any],
    strongest_baselines: Mapping[str, Any],
    checkpoint_signatures: list,
    selection_scores: list,
    upstream_sha256: str,
    data_signature: str,
    contract_id: str = SELECT_FINAL_CONTRACT_ID,
) -> str:
    """SHA256 over the frozen selection: final config + frozen baselines + the
    five checkpoint signatures + the five per-seed selection scores + the
    upstream fit-five results file hash + data + contract.

    The gate recomputes this from the on-disk checkpoints, the on-disk upstream
    results file and the receipt's stored fields; any edit to the receipt's
    config/baselines/scores, any swap of checkpoints, or any post-selection edit
    of the fit-five results changes the recomputed value and is detected.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps(final_config, sort_keys=True, default=str).encode("utf-8"))
    digest.update(json.dumps(strongest_baselines, sort_keys=True, default=str).encode("utf-8"))
    for sig in sorted(str(s) for s in checkpoint_signatures):
        digest.update(sig.encode("utf-8"))
    for score in selection_scores:
        digest.update(f"{float(score):.17g}".encode("utf-8"))
    digest.update(str(upstream_sha256).encode("utf-8"))
    digest.update(str(data_signature).encode("utf-8"))
    digest.update(str(contract_id).encode("utf-8"))
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Checkpoint signatures
# --------------------------------------------------------------------------- #
def checkpoint_signature(state_dict: Mapping[str, torch.Tensor]) -> str:
    """SHA256 over a model's parameters (names + shapes + values)."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().astype(np.float64).tobytes())
    return digest.hexdigest()


def five_checkpoint_signatures(state_dicts: list[Mapping[str, torch.Tensor]]) -> dict[str, Any]:
    """Verify exactly five distinct checkpoints and return their signatures."""
    if len(state_dicts) != 5:
        raise ValueError("expected exactly five checkpoints (one per seed-pair)")
    signatures = [checkpoint_signature(sd) for sd in state_dicts]
    if len(set(signatures)) != 5:
        raise ValueError("the five seed-pair checkpoints must be distinct")
    return {"count": 5, "signatures": signatures, "all_distinct": True}


# --------------------------------------------------------------------------- #
# Predictor checkpoint persistence + integrity (item 6)
# --------------------------------------------------------------------------- #
def save_predictor_checkpoint(
    path: str | Path,
    state_dict: Mapping[str, torch.Tensor],
    *,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically persist a predictor state_dict with its signature and metadata.

    Returns the integrity record ``{signature, path, meta}``; the signature is
    recomputed on save so the on-disk artefact is self-describing.
    """
    signature = checkpoint_signature(state_dict)
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
        "signature": signature,
        "meta": dict(meta),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return {"signature": signature, "path": str(path), "meta": dict(meta)}


def verify_checkpoint_integrity(path: str | Path, *, expected_signature: str | None = None) -> dict[str, Any]:
    """Reload a predictor checkpoint and recompute its signature (integrity check).

    Raises if the recomputed signature differs from the stored one, or from
    ``expected_signature`` when given. Returns the verified record.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    recomputed = checkpoint_signature(payload["state_dict"])
    stored = payload.get("signature")
    if stored is not None and recomputed != stored:
        raise ValueError(f"checkpoint integrity failure (stored {stored} vs recomputed {recomputed})")
    if expected_signature is not None and recomputed != expected_signature:
        raise ValueError(
            f"checkpoint signature {recomputed} does not match expected {expected_signature}"
        )
    return {"signature": recomputed, "path": str(path), "meta": payload.get("meta", {})}


def load_predictor_state(path: str | Path, *, expected_signature: str | None = None) -> Mapping[str, torch.Tensor]:
    """Verify then return a predictor checkpoint's state_dict."""
    verify_checkpoint_integrity(path, expected_signature=expected_signature)
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return payload["state_dict"]


# --------------------------------------------------------------------------- #
# Seed-pair enforcement
# --------------------------------------------------------------------------- #
def seed_pair_for_index(pair_index: int) -> tuple[int, int]:
    """Return the frozen (predictor_seed, router_seed) for a pair index 0..4."""
    validate_seed_pairing()
    if not 0 <= pair_index < len(SEED_PAIRS):
        raise ValueError("pair_index must be in 0..4 (no best-seed, no 5x5 cross)")
    return SEED_PAIRS[pair_index]


# --------------------------------------------------------------------------- #
# Cross-fit (2-fold out-of-fold response features)
# --------------------------------------------------------------------------- #
def crossfit_fold_masks(
    cf_action: np.ndarray,
    cf_duration: np.ndarray,
    episode_seed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (fold0_mask, fold1_mask) booleans with no episode overlap."""
    folds = assign_crossfit_folds(cf_action, cf_duration, episode_seed)
    fold0 = folds == 0
    fold1 = folds == 1
    if (fold0 & fold1).any():
        raise RuntimeError("cross-fit folds overlap")
    if not (fold0 | fold1).all():
        raise RuntimeError("cross-fit folds do not cover all episodes")
    return fold0, fold1


def build_crossfit_splits(
    train_indices: np.ndarray,
    cf_action: np.ndarray,
    cf_duration: np.ndarray,
    episode_seed: np.ndarray,
) -> dict[str, np.ndarray]:
    """Split the train role into two out-of-fold halves for cross-fit.

    A predictor trained on fold A generates out-of-fold response features for
    fold B and vice versa; the concatenation covers every train episode exactly
    once with no episode appearing in both a predictor's fit and its own
    feature generation.
    """
    train_indices = np.asarray(train_indices)
    fold0, fold1 = crossfit_fold_masks(
        cf_action[train_indices], cf_duration[train_indices], episode_seed[train_indices]
    )
    return {
        "fold0": train_indices[fold0],
        "fold1": train_indices[fold1],
    }


# --------------------------------------------------------------------------- #
# Training checkpoint / resume
# --------------------------------------------------------------------------- #
def save_train_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    best_state: Mapping[str, torch.Tensor] | None,
    history: list[dict],
    run_fingerprint: str,
    epochs_without_improvement: int = 0,
) -> None:
    """Atomically save a resumable training checkpoint."""
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val": float(best_val),
        "best_state": best_state,
        "history": history,
        "run_fingerprint": run_fingerprint,
        "epochs_without_improvement": int(epochs_without_improvement),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_train_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    run_fingerprint: str,
) -> dict[str, Any]:
    """Load a resumable checkpoint; refuse a fingerprint mismatch (wrong run)."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("run_fingerprint") != run_fingerprint:
        raise ValueError(
            "checkpoint run_fingerprint does not match this run; refusing to resume"
        )
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    return payload


# --------------------------------------------------------------------------- #
# Generic masked-effect training loop (response predictor and router)
# --------------------------------------------------------------------------- #
def global_validation_metrics(
    model: torch.nn.Module,
    val_batches: Iterable[Mapping[str, torch.Tensor]],
    sums_fn: Callable[[torch.nn.Module, Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]],
    selection_fn: Callable[[float, float], float],
    device: str | torch.device,
) -> dict[str, float]:
    """Globally aggregated validation metrics (repair plan v0.1 §2.4).

    Numerators and denominators are summed over ALL valid agent-times before
    dividing, so the result is invariant to batch size, tail-batch size and
    episode permutation (float64 accumulation; tolerance 1e-7 holds).
    """
    reg_sum = 0.0
    ce_sum = 0.0
    count = 0.0
    with torch.no_grad():
        for batch in val_batches:
            batch = {k: v.to(device) for k, v in batch.items()}
            sums = sums_fn(model, batch)
            reg_sum += float(sums["reg_sum"])
            ce_sum += float(sums["ce_sum"])
            count += float(sums["valid_count"])
    reg = reg_sum / max(count, 1.0)
    ce = ce_sum / max(count, 1.0)
    return {
        "validation_reg_loss": reg,
        "validation_ce_loss": ce,
        "validation_selection_score": float(selection_fn(reg, ce)),
    }


def train_model(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.nn.Module, Mapping[str, torch.Tensor]], torch.Tensor],
    batches: list[Mapping[str, torch.Tensor]],
    val_batches: list[Mapping[str, torch.Tensor]],
    *,
    seed: int,
    max_epochs: int,
    early_stopping_patience: int,
    learning_rate: float,
    device: str = "cpu",
    weight_decay: float = 0.0,
    progress_prefix: str = "train",
    heartbeat_seconds: float = 30.0,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    run_fingerprint: str | None = None,
    val_sums_fn: Callable[[torch.nn.Module, Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]] | None = None,
    selection_fn: Callable[[float, float], float] | None = None,
) -> dict[str, Any]:
    """Train with early stopping on validation loss; return best state + history.

    The loop emits a plain-text progress line at least every ``heartbeat_seconds``
    even when few epochs complete. Model selection uses the validation loss only.
    When ``checkpoint_path`` is set, a resumable checkpoint is written after every
    epoch; with ``resume=True`` the loop continues from the saved epoch (the
    ``run_fingerprint`` must match, guarding against resuming the wrong run).
    Stage 3A Plus Task 5 (§4.3): ``weight_decay`` is threaded into Adam; the
    default ``0.0`` reproduces the original optimiser exactly.
    """
    seed_all(seed)
    device_obj = torch.device(device)
    model.to(device_obj)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    fingerprint = run_fingerprint or f"{progress_prefix}-seed{seed}"
    reporter = ProgressReporter(
        total=max_epochs, every=1, prefix=progress_prefix, heartbeat_seconds=heartbeat_seconds
    )
    best_val = float("inf")
    best_state = None
    best_metrics: dict[str, float] | None = None
    epochs_without_improvement = 0
    history: list[dict] = []
    start_epoch = 0
    if resume and checkpoint_path is not None and Path(checkpoint_path).is_file():
        payload = load_train_checkpoint(
            checkpoint_path, model=model, optimizer=optimizer, run_fingerprint=fingerprint
        )
        best_val = float(payload["best_val"])
        best_state = payload["best_state"]
        history = list(payload["history"])
        start_epoch = int(payload["epoch"]) + 1
        # Restore the early-stopping counter so a resumed run stops at exactly
        # the same epoch an uninterrupted run would (item 5).
        epochs_without_improvement = int(payload.get("epochs_without_improvement", 0))
        if val_sums_fn is not None and history:
            # Rebuild the best-epoch breakdown so a resumed run that never
            # improves again still reports its selection-score decomposition.
            best_entry = min(history, key=lambda h: h["val_loss"])
            best_metrics = {
                "validation_reg_loss": best_entry["val_reg_loss"],
                "validation_ce_loss": best_entry["val_ce_loss"],
                "validation_selection_score": best_entry["val_loss"],
            }
        reporter.update(start_epoch, details=f"resumed best={best_val:.4f}", force=True)

    for epoch in range(start_epoch, max_epochs):
        model.train()
        train_loss = 0.0
        for batch in batches:
            batch = {k: v.to(device_obj) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach())
        train_loss /= max(len(batches), 1)

        model.eval()
        if val_sums_fn is not None:
            # Repair plan v0.1 §2.4: global sum/sum-count aggregation so the
            # selection score never depends on batch size or tail batches.
            metrics = global_validation_metrics(model, val_batches, val_sums_fn, selection_fn, device_obj)
            val_loss = metrics["validation_selection_score"]
            history.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_reg_loss": metrics["validation_reg_loss"],
                "val_ce_loss": metrics["validation_ce_loss"],
            })
        else:
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_batches:
                    batch = {k: v.to(device_obj) for k, v in batch.items()}
                    val_loss += float(loss_fn(model, batch))
            val_loss /= max(len(val_batches), 1)
            metrics = None
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        reporter.update(epoch + 1, details=f"val_loss={val_loss:.4f} best={best_val:.4f}")
        reporter.heartbeat(epoch + 1, details=f"val_loss={val_loss:.4f}")
        if checkpoint_path is not None:
            save_train_checkpoint(
                checkpoint_path,
                model=model, optimizer=optimizer, epoch=epoch, best_val=best_val,
                best_state=best_state, history=history, run_fingerprint=fingerprint,
                epochs_without_improvement=epochs_without_improvement,
            )
        if epochs_without_improvement >= early_stopping_patience:
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out = {
        "state_dict": best_state,
        "best_val_loss": float(best_val),
        "epochs_run": len(history),
        "history": history,
        "signature": checkpoint_signature(best_state),
        "seed": int(seed),
    }
    if best_metrics is not None:
        # Breakdown persisted alongside the selection score (plan v0.1 §2.3).
        out.update(best_metrics)
    return out


# --------------------------------------------------------------------------- #
# Compute budget (section 5.3 + expanded user requirements)
# --------------------------------------------------------------------------- #
# Frozen bounded search space (v0.2 registered; do not expand).
SEARCH_HIDDEN_DIMS = (64, 96)
SEARCH_RESPONSE_FEATURE_DIMS = (32, 64)
SEARCH_LAMBDA_CLS = (0.10, 0.25)
MAX_FINALIST_CONFIGS = 2
N_SEED_PAIRS = 5

# Uncertainty band applied to every extrapolated GPU-hour figure.
UNCERTAINTY_LOW = 0.5
UNCERTAINTY_HIGH = 2.0

REQUIRED_BUDGET_FIELDS = (
    "benchmarks",
    "extrapolation",
    "search_space",
    "stage3a",
    "stage3b",
    "crossfit",
    "formal",
    "retry_policy",
    "totals",
)


def search_space_configs() -> list[dict[str, Any]]:
    """The 8 bounded Stage 3A candidate configs (hidden_dim x rfd x lambda_cls)."""
    return [
        {"hidden_dim": h, "response_feature_dim": r, "lambda_cls": l}
        for h in SEARCH_HIDDEN_DIMS
        for r in SEARCH_RESPONSE_FEATURE_DIMS
        for l in SEARCH_LAMBDA_CLS
    ]


def _device_labels(device: str) -> dict[str, str]:
    """Unit labels so CPU time/memory is never reported as GPU-hours/VRAM."""
    if str(device).lower() == "cuda":
        return {"hours": "GPU 小时", "memory": "峰值显存(VRAM)", "memory_kind": "VRAM"}
    return {"hours": "CPU 小时", "memory": "峰值内存(CPU RSS)", "memory_kind": "CPU_RSS"}


def build_budget_estimate(
    benchmarks: Mapping[str, Any],
    *,
    train_episodes: int,
    benchmark_episodes: int,
    assumed_epochs: int,
    checkpoint_gb_per_run: float,
    gatec_id_episodes: int = 1600,
    gatec_unseen_episodes: int = 800,
    n_bootstrap: int = 10000,
    n_bootstrap_endpoints: int = 8,
) -> dict[str, Any]:
    """Compute the structured budget estimate from full-epoch benchmarks.

    ``benchmarks`` carries a ``response`` and a ``router`` sub-benchmark, each
    with ``epoch_seconds``/``peak_memory_gb``/``device`` measured over a full
    train epoch at the widest candidate config and the formal batch size, plus
    optional ``collection_episodes_per_second``/``inference_episodes_per_second``/
    ``bootstrap_resamples_per_second`` rates for the formal-stage estimate.

    Stage 3A uses the response timing; cross-fit and Stage 3B use the router
    timing (never the response timing). Timing scales linearly with episode
    count; the widest config makes every per-config figure an upper bound.
    Units follow the benchmark device — CPU runs are reported as CPU hours and
    CPU RSS, never GPU-hours/VRAM. Every device-hour carries a [x0.5, x2.0]
    uncertainty band; totals report both the point (典型) and high-band (最坏).
    """
    response = benchmarks["response"]
    router = benchmarks["router"]
    device = str(response.get("device", "cpu")).lower()
    labels = _device_labels(device)

    scale = train_episodes / max(benchmark_episodes, 1)
    resp_per_seed_seconds = float(response["epoch_seconds"]) * scale * assumed_epochs
    router_per_seed_seconds = float(router["epoch_seconds"]) * scale * assumed_epochs
    resp_mem = float(response["peak_memory_gb"])
    router_mem = float(router["peak_memory_gb"])

    def band(hours: float) -> dict[str, float]:
        return {"point": hours, "low": hours * UNCERTAINTY_LOW, "high": hours * UNCERTAINTY_HIGH}

    def hours(seconds: float) -> float:
        return seconds / 3600.0

    # --- Stage 3A (response timing) ------------------------------------------
    configs = search_space_configs()
    per_config = [
        {
            **cfg,
            "per_seed_seconds": resp_per_seed_seconds,
            "per_seed_hours": hours(resp_per_seed_seconds),
            "peak_memory_gb": resp_mem,
        }
        for cfg in configs
    ]
    n_screen = len(configs)
    n_five = MAX_FINALIST_CONFIGS * N_SEED_PAIRS
    stage3a = {
        "per_config": per_config,
        "single_seed_screen": {
            "configs": n_screen, "runs": n_screen,
            "device_hours": band(hours(resp_per_seed_seconds * n_screen)),
            "storage_gb": checkpoint_gb_per_run * n_screen,
        },
        "five_seed_finalists": {
            "configs": MAX_FINALIST_CONFIGS, "runs": n_five,
            "device_hours": band(hours(resp_per_seed_seconds * n_five)),
            "storage_gb": checkpoint_gb_per_run * n_five,
        },
        "total_runs": n_screen + n_five,
        "total_device_hours": band(hours(resp_per_seed_seconds * (n_screen + n_five))),
        "storage_gb": checkpoint_gb_per_run * (n_screen + n_five),
    }

    # --- cross-fit (router timing): Direct + Dual, five pairs each -----------
    crossfit_runs = 2 * N_SEED_PAIRS
    crossfit = {
        "runs": crossfit_runs,
        "device_hours": band(hours(router_per_seed_seconds * crossfit_runs)),
        "storage_gb": checkpoint_gb_per_run * crossfit_runs,
    }

    # --- Stage 3B (router timing), two-tier per section 5.3 ------------------
    tier1_conditions = ["uniform_suffix", "direct", "dual_router", "oracle_reference"]
    tier2_conditions = [
        "no_suffix", "shuffled_suffix", "response_head_no_routing",
        "random_gate", "static_gate", "whole_suffix_soft_gate",
    ]
    tier1_runs = len(tier1_conditions) * N_SEED_PAIRS
    tier2_runs = len(tier2_conditions) * N_SEED_PAIRS
    stage3b = {
        "tier1": {
            "conditions": tier1_conditions, "runs": tier1_runs,
            "device_hours": band(hours(router_per_seed_seconds * tier1_runs)),
            "storage_gb": checkpoint_gb_per_run * tier1_runs,
        },
        "tier2": {
            "conditions": tier2_conditions, "runs": tier2_runs,
            "device_hours": band(hours(router_per_seed_seconds * tier2_runs)),
            "storage_gb": checkpoint_gb_per_run * tier2_runs,
        },
        "conditions": len(tier1_conditions) + len(tier2_conditions),
        "runs": tier1_runs + tier2_runs,
        "device_hours": band(hours(router_per_seed_seconds * (tier1_runs + tier2_runs))),
        "storage_gb": checkpoint_gb_per_run * (tier1_runs + tier2_runs),
    }

    # --- formal stage (pre-lock estimate; never auto-started) ----------------
    collection_rate = float(benchmarks.get("collection_episodes_per_second", 20.0))
    inference_rate = float(benchmarks.get("inference_episodes_per_second", 50.0))
    bootstrap_rate = float(benchmarks.get("bootstrap_resamples_per_second", 2000.0))
    formal = {
        "note": (
            "PRE-LOCK ESTIMATE: formal collection, full frozen inference and the "
            "10,000x bootstrap are frozen only after Gate R_dev + Gate T_dev + "
            "Gate C_dev + power all pass; the single unlock is not estimated."
        ),
        "collection": {
            "sets": ["gatec_merge_id", "gatec_merge_unseen_duration"],
            "episodes": {"gatec_merge_id": int(gatec_id_episodes), "gatec_merge_unseen_duration": int(gatec_unseen_episodes)},
            "episodes_per_second": collection_rate,
            "device_hours": band(hours((gatec_id_episodes + gatec_unseen_episodes) / collection_rate)),
        },
        "inference": {
            "episodes": int(gatec_id_episodes + gatec_unseen_episodes),
            "episodes_per_second": inference_rate,
            "device_hours": band(hours((gatec_id_episodes + gatec_unseen_episodes) / inference_rate)),
        },
        "bootstrap": {
            "n_bootstrap": int(n_bootstrap),
            "n_endpoints": int(n_bootstrap_endpoints),
            "resamples_per_second": bootstrap_rate,
            "device_hours": band(hours(n_bootstrap * n_bootstrap_endpoints / bootstrap_rate)),
        },
    }
    formal_total_hours = (
        formal["collection"]["device_hours"]["point"]
        + formal["inference"]["device_hours"]["point"]
        + formal["bootstrap"]["device_hours"]["point"]
    )
    formal["total_device_hours"] = band(formal_total_hours)

    # --- totals -----------------------------------------------------------------
    total_runs = stage3a["total_runs"] + crossfit["runs"] + stage3b["runs"]
    train_device_hours = (
        stage3a["total_device_hours"]["point"]
        + crossfit["device_hours"]["point"]
        + stage3b["device_hours"]["point"]
    )
    overall_point = train_device_hours + formal_total_hours
    estimate = {
        "device": device,
        "unit_labels": labels,
        "benchmarks": {k: dict(v) for k, v in benchmarks.items() if isinstance(v, Mapping)},
        "extrapolation": {
            "method": (
                "epoch_seconds scales linearly from the benchmark episode count to "
                "the full train split; per-seed time = full_epoch_seconds x assumed_epochs; "
                "Stage 3A uses the response benchmark, cross-fit and Stage 3B use the "
                "router benchmark; the widest config gives an upper bound for all configs."
            ),
            "train_episodes": int(train_episodes),
            "benchmark_episodes": int(benchmark_episodes),
            "assumed_epochs": int(assumed_epochs),
            "response_per_seed_seconds": resp_per_seed_seconds,
            "router_per_seed_seconds": router_per_seed_seconds,
            "uncertainty_low": UNCERTAINTY_LOW,
            "uncertainty_high": UNCERTAINTY_HIGH,
        },
        "search_space": {
            "hidden_dim": list(SEARCH_HIDDEN_DIMS),
            "response_feature_dim": list(SEARCH_RESPONSE_FEATURE_DIMS),
            "lambda_cls": list(SEARCH_LAMBDA_CLS),
            "max_finalist_configs": MAX_FINALIST_CONFIGS,
            "n_seed_pairs": N_SEED_PAIRS,
        },
        "stage3a": stage3a,
        "stage3b": stage3b,
        "crossfit": crossfit,
        "formal": formal,
        "retry_policy": {
            "checkpoint_every_epoch": True,
            "resume_command": "add --resume to the same command; the run_fingerprint guards against resuming the wrong run",
            "failure_exit": "a failed run writes a failure summary and exits non-zero; re-run with --resume",
            "max_retries": 3,
        },
        "totals": {
            "total_runs": total_runs,
            "train_device_hours": band(train_device_hours),
            "overall_device_hours": band(overall_point),
            "typical_hours": overall_point,
            "worst_hours": overall_point * UNCERTAINTY_HIGH,
            "storage_gb": checkpoint_gb_per_run * total_runs,
            "peak_memory_gb": max(resp_mem, router_mem),
        },
    }
    return estimate


def _fmt_band(band: Mapping[str, float]) -> str:
    return f"{band['point']:.2f} h (range {band['low']:.2f}–{band['high']:.2f} h)"


def generate_compute_budget(estimate: Mapping[str, Any], output_path: str | Path) -> str:
    """Render COMPUTE_BUDGET.md from a structured estimate.

    ``estimate`` must provide every key in ``REQUIRED_BUDGET_FIELDS`` (see
    :func:`build_budget_estimate`). The budget is produced before any multi-seed
    long training and is confirmed by the user separately.
    """
    missing = [k for k in REQUIRED_BUDGET_FIELDS if k not in estimate]
    if missing:
        raise ValueError(f"compute budget is missing required fields: {missing}")
    labels = estimate.get("unit_labels", _device_labels(estimate.get("device", "cpu")))
    H = labels["hours"]      # "GPU 小时" or "CPU 小时"
    M = labels["memory"]     # "峰值显存(VRAM)" or "峰值内存(CPU RSS)"
    device = estimate.get("device", "?")
    benches = estimate["benchmarks"]
    resp = benches["response"]
    extra = estimate["extrapolation"]
    s3a = estimate["stage3a"]
    s3b = estimate["stage3b"]
    cf = estimate["crossfit"]
    formal = estimate["formal"]
    totals = estimate["totals"]
    retry = estimate["retry_policy"]

    def fb(band):
        return _fmt_band(band)

    lines = [
        "# AC-Twin Stage 3 COMPUTE_BUDGET",
        "",
        "> 状态：`ENGINEERING_ESTIMATE`。多 seed 长训练前由完整 epoch benchmark 生成；",
        "> 须用户单独确认预算后，才由用户手动启动任何正式/长任务。",
        f"> 设备：`{device}`。时间单位 = {H}；显存单位 = {M}。",
        "",
        "## 0. 估算来源与外推",
        "",
        f"- benchmark 来源：{resp.get('source', 'full-epoch benchmark')}",
        f"- benchmark 设备 / 日期：{device} / {resp.get('date', '?')}",
        f"- response 实测 epoch_seconds（{extra['benchmark_episodes']} episode）：{resp['epoch_seconds']:.3f} s，{M}：{resp['peak_memory_gb']:.3f} GB",
        f"- router 实测 epoch_seconds：{benches['router']['epoch_seconds']:.3f} s，{M}：{benches['router']['peak_memory_gb']:.3f} GB",
        f"- 外推方法：{extra['method']}",
        f"- 外推至 train={extra['train_episodes']} episode：response 单 seed（{extra['assumed_epochs']} epoch）={extra['response_per_seed_seconds']:.0f} s，"
        f"router 单 seed={extra['router_per_seed_seconds']:.0f} s",
        f"- 不确定范围：所有{H} ×[{extra['uncertainty_low']}, {extra['uncertainty_high']}]",
        "",
        "## 1. 每个配置、每个 seed 的预计时长与显存",
        "",
        f"| hidden_dim | resp_feat_dim | λ_cls | 单 seed 时长 | {M} |",
        "|---:|---:|---:|---:|---:|",
    ]
    for cfg in s3a["per_config"]:
        lines.append(
            f"| {cfg['hidden_dim']} | {cfg['response_feature_dim']} | {cfg['lambda_cls']} "
            f"| {cfg['per_seed_hours']:.2f} h | {cfg['peak_memory_gb']:.2f} GB |"
        )
    lines += [
        "",
        "## 2. Stage 3A 分项预算（response 计时）",
        "",
        "### 2a. 单 seed 有界筛选",
        "",
        f"- 配置数 / 训练次数：{s3a['single_seed_screen']['configs']} / {s3a['single_seed_screen']['runs']}",
        f"- {H}：{fb(s3a['single_seed_screen']['device_hours'])}",
        f"- 存储：{s3a['single_seed_screen']['storage_gb']:.2f} GB",
        "",
        "### 2b. 入围配置补五 seed",
        "",
        f"- 入围配置 / 训练次数：{s3a['five_seed_finalists']['configs']} / {s3a['five_seed_finalists']['runs']}",
        f"- {H}：{fb(s3a['five_seed_finalists']['device_hours'])}",
        f"- 存储：{s3a['five_seed_finalists']['storage_gb']:.2f} GB",
        "",
        f"**Stage 3A 合计：{s3a['total_runs']} 次训练，{fb(s3a['total_device_hours'])}，"
        f"存储 {s3a['storage_gb']:.2f} GB。**",
        "",
        "## 3. cross-fit 额外成本（router 计时）",
        "",
        f"- 训练次数（Direct+Dual 各五组）：{cf['runs']}",
        f"- {H}：{fb(cf['device_hours'])}",
        f"- 存储：{cf['storage_gb']:.2f} GB",
        "",
        "## 4. Stage 3B 分项预算（router 计时，仅登记，未启动）",
        "",
        "### 4a. tier-1 主条件（Uniform/Direct/Dual + Oracle reference）",
        "",
        f"- 条件：{', '.join(s3b['tier1']['conditions'])}",
        f"- 训练次数（{len(s3b['tier1']['conditions'])} 条件 × 5 seed-pair）：{s3b['tier1']['runs']}",
        f"- {H}：{fb(s3b['tier1']['device_hours'])}",
        "",
        "### 4b. tier-2 诊断条件（开发门通过后补）",
        "",
        f"- 条件：{', '.join(s3b['tier2']['conditions'])}",
        f"- 训练次数（{len(s3b['tier2']['conditions'])} 条件 × 5 seed-pair）：{s3b['tier2']['runs']}",
        f"- {H}：{fb(s3b['tier2']['device_hours'])}",
        "",
        f"**Stage 3B 合计：{s3b['runs']} 次训练，{fb(s3b['device_hours'])}，存储 {s3b['storage_gb']:.2f} GB。**",
        "",
        "## 5. 正式集合（冻结前估算，不自动启动）",
        "",
        f"- 说明：{formal['note']}",
        f"- 采集：{formal['collection']['sets'][0]}={formal['collection']['episodes']['gatec_merge_id']} ep，"
        f"{formal['collection']['sets'][1]}={formal['collection']['episodes']['gatec_merge_unseen_duration']} ep，"
        f"{H}：{fb(formal['collection']['device_hours'])}",
        f"- 全量冻结推理：{formal['inference']['episodes']} ep，{H}：{fb(formal['inference']['device_hours'])}",
        f"- 10,000× bootstrap（{formal['bootstrap']['n_endpoints']} 端点）：{H}：{fb(formal['bootstrap']['device_hours'])}",
        f"- 正式集合小计：{fb(formal['total_device_hours'])}",
        "",
        "## 6. 失败、重试与断点续跑策略",
        "",
        f"- 每 epoch 写断点：{retry['checkpoint_every_epoch']}",
        f"- 断点续跑：{retry['resume_command']}",
        f"- 失败退出：{retry['failure_exit']}",
        f"- 最大重试：{retry['max_retries']}",
        "",
        "## 7. 总计",
        "",
        f"- 总训练次数：{totals['total_runs']}",
        f"- 训练 {H}：{fb(totals['train_device_hours'])}",
        f"- 含正式集合的 {H}：{fb(totals['overall_device_hours'])}",
        f"- 典型总时长：{totals['typical_hours']:.2f} h",
        f"- 最坏总时长：{totals['worst_hours']:.2f} h",
        f"- 存储需求：{totals['storage_gb']:.2f} GB",
        f"- {M}（单任务峰值）：{totals['peak_memory_gb']:.2f} GB",
        "",
    ]
    text = "\n".join(lines)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(output_path)
    return text
