"""Stage 3R command-line driver."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin.stage3r_models import (  # noqa: E402
    CEWRouter,
    SameInformationDirect,
    Stage3RExpert,
    build_router_features,
    match_direct_hidden_width,
    stage3r_forward,
)
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402
from ac_twin.stage3r_statistics import (  # noqa: E402
    evaluate_phase0,
    evaluate_phase1,
    estimate_joint_power,
    masked_episode_ade,
    pairwise_rank_loss,
    weighted_huber,
)
from ac_twin.stage3r_teacher import (  # noqa: E402
    aggregate_episode_vehicle,
    build_consensus_teacher,
    optimal_convex_weight,
)
from ac_twin.stage3r_screen import (  # noqa: E402
    adjudicate_rscreen,
    make_disjoint_seed_schedule,
    validate_rscreen_readiness,
)


PLAN_ID = "AC-TWIN-RFER-STAGE3R-2026-09-10-V0.1"
FROZEN_PREDICTOR_SEEDS = [101, 202, 303, 404, 505]
FROZEN_EXPERT_ROUTER_SEEDS = [1101, 1202, 1303, 1404, 1505]


def masked_expert_mse(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[:-1] != valid.shape:
        raise ValueError("expert prediction, target, and valid shapes do not align")
    mask = valid.to(prediction.dtype).unsqueeze(-1)
    denominator = (mask.sum() * prediction.shape[-1]).clamp_min(1.0)
    return (((prediction - target) ** 2) * mask).sum() / denominator


def validate_oof_coverage(
    train_indices: np.ndarray, holdout_parts: list[np.ndarray]
) -> None:
    expected = np.sort(np.asarray(train_indices, dtype=np.int64))
    observed = np.sort(
        np.concatenate([np.asarray(part, dtype=np.int64) for part in holdout_parts])
    )
    if observed.shape != expected.shape or not np.array_equal(observed, expected):
        raise ValueError("every train episode must appear exactly once in OOF holdouts")


def expert_loss(
    model: Stage3RExpert, batch: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    prediction = model(
        history=batch["history"],
        factual_suffix=batch["factual_suffix"],
        factual_actions=batch["factual_actions"],
        counterfactual_actions=batch["counterfactual_actions"],
    )
    return masked_expert_mse(prediction, batch["effect_target"], batch["valid"])


def predict_expert_ordered(
    model: Stage3RExpert,
    tensors: Mapping[str, torch.Tensor],
    indices: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    device_obj = torch.device(device)
    model = model.to(device_obj).eval()
    ordered = np.asarray(indices, dtype=np.int64)
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, ordered.size, int(batch_size)):
            positions = torch.as_tensor(
                ordered[start : start + int(batch_size)], dtype=torch.long
            )
            batch = {
                key: tensors[key].index_select(0, positions).to(device_obj)
                for key in (
                    "history",
                    "factual_suffix",
                    "factual_actions",
                    "counterfactual_actions",
                )
            }
            outputs.append(model(**batch).cpu())
    if not outputs:
        raise ValueError("ordered expert prediction requires at least one index")
    return torch.cat(outputs, dim=0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _outer_progress(phase: str, completed: int, total: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 0.0)
    rate = completed / elapsed if completed and elapsed else 0.0
    eta = (total - completed) / rate if rate else 0.0
    percent = 100.0 * completed / max(total, 1)
    print(
        f"[{phase}] {completed}/{total} {percent:.1f}% "
        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
        flush=True,
    )


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def validate_config(
    config: Mapping[str, Any], *, project_root: Path = ROOT
) -> dict[str, Any]:
    validated = dict(config)
    if validated.get("plan_id") != PLAN_ID:
        raise ValueError("unexpected Stage 3R plan_id")
    if (
        validated.get("train_split"),
        validated.get("validation_split"),
        validated.get("historical_split"),
    ) != (0, 1, 2):
        raise ValueError("frozen split roles must be train=0 validation=1 historical=2")
    if int(validated.get("k_folds", -1)) != 5:
        raise ValueError("Stage 3R requires exactly five OOF folds")
    if list(validated.get("response_predictor_seeds", [])) != FROZEN_PREDICTOR_SEEDS:
        raise ValueError("response predictor seeds differ from frozen pairing")
    if list(validated.get("expert_router_seeds", [])) != FROZEN_EXPERT_ROUTER_SEEDS:
        raise ValueError("expert/router seeds differ from frozen pairing")
    if list(validated.get("lambda_cew", [])) != [0.25, 1.0]:
        raise ValueError("lambda_cew candidate set drifted")
    if list(validated.get("lambda_rank", [])) != [0.0, 0.1]:
        raise ValueError("lambda_rank candidate set drifted")
    source = _resolve(project_root, validated["development_source"])
    run_dir = _resolve(project_root, validated["run_dir"])
    binding = _resolve(project_root, validated["stage3b_binding"])
    if run_dir == source.parent or source.is_relative_to(run_dir):
        raise ValueError("run_dir cannot contain the development source")
    validated["_source_path"] = source
    validated["_run_dir_path"] = run_dir
    validated["_binding_path"] = binding
    return validated


def preflight(
    config: Mapping[str, Any], *, project_root: Path = ROOT
) -> dict[str, Any]:
    validated = validate_config(config, project_root=project_root)
    source = validated["_source_path"]
    binding = validated["_binding_path"]
    if not source.is_file():
        raise FileNotFoundError(source)
    if not binding.is_file():
        raise FileNotFoundError(binding)
    binding_payload = json.loads(binding.read_text(encoding="utf-8"))["receipt_payload"]
    if binding_payload.get("stage3a_plus_terminal_state") != "RESPONSE_DEV_QUALIFIED_STAGE3A_PLUS":
        raise ValueError("Stage 3A Plus binding is not qualified")
    if binding_payload.get("seed_pairs") != [
        [predictor, router]
        for predictor, router in zip(
            FROZEN_PREDICTOR_SEEDS, FROZEN_EXPERT_ROUTER_SEEDS, strict=True
        )
    ]:
        raise ValueError("Stage 3A Plus binding seed pairs drifted")
    return {
        "status": "STAGE3R_PREFLIGHT_PASS",
        "development_source_exists": True,
        "binding_exists": True,
        "historical_split_not_selectable": True,
        "gate_c_authorized": False,
    }


def engineering_smoke(
    *, device: str = "cpu", episodes: int = 2, horizon: int = 4
) -> dict[str, Any]:
    torch.manual_seed(20260910)
    device_obj = torch.device(device)
    agents = 3
    history = torch.randn(episodes, 5, agents, 8, device=device_obj)
    suffix_a = torch.randn(episodes, 5, agents, 8, device=device_obj)
    suffix_b = torch.randn(episodes, 5, agents, 8, device=device_obj)
    history[..., 0] = 1.0
    suffix_a[..., 0] = 1.0
    suffix_b[..., 0] = 1.0
    factual_actions = torch.randint(0, 5, (episodes, horizon), device=device_obj)
    counterfactual_actions = torch.randint(0, 5, (episodes, horizon), device=device_obj)

    torch.manual_seed(31)
    full_model = Stage3RExpert(
        "full", feature_dim=8, object_embedding_dim=16, hidden_dim=16
    ).to(device_obj).eval()
    torch.manual_seed(31)
    history_model = Stage3RExpert(
        "history", feature_dim=8, object_embedding_dim=16, hidden_dim=16
    ).to(device_obj).eval()
    with torch.no_grad():
        full = full_model(history, suffix_a, factual_actions, counterfactual_actions)
        history_a = history_model(history, suffix_a, factual_actions, counterfactual_actions)
        history_b = history_model(history, suffix_b, factual_actions, counterfactual_actions)
        features = torch.randn(episodes, horizon, agents, 48, device=device_obj)
        router = CEWRouter(48, 64).to(device_obj).eval()
        direct_width = match_direct_hidden_width(48, 64)
        direct = SameInformationDirect(48, direct_width).to(device_obj).eval()
        routed, route_trace = stage3r_forward(
            "cew_distilled_router",
            full,
            history_a,
            features,
            0.5,
            router=router,
            return_trace=True,
        )
        direct_output = stage3r_forward(
            "same_information_direct",
            full,
            history_a,
            features,
            0.5,
            direct=direct,
        )
    finite = all(
        bool(torch.isfinite(value).all())
        for value in (full, history_a, routed, direct_output)
    )
    weight = route_trace["weight"]
    return {
        "status": "ENGINEERING_SMOKE_ONLY",
        "finite": finite,
        "full_shape": list(full.shape),
        "history_suffix_invariant": bool(torch.equal(history_a, history_b)),
        "router_convex": bool(((weight >= 0) & (weight <= 1)).all()),
        "direct_hidden_dim": direct_width,
    }


def _load_or_train_expert(
    *,
    condition: str,
    seed: int,
    label: str,
    tensors: Mapping[str, torch.Tensor],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: Mapping[str, Any],
    output_dir: Path,
    resume: bool,
) -> tuple[Stage3RExpert, dict[str, Any]]:
    import run_stage3a as stage3a
    from ac_twin.stage3_training import train_model

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{label}_{condition}_seed{seed}"
    model_path = output_dir / f"{stem}.pt"
    state_path = output_dir / f"{stem}.json"
    train_checkpoint = output_dir / f"{stem}.train.ckpt"
    torch.manual_seed(int(seed))
    model = Stage3RExpert(
        condition,
        hidden_dim=int(config["expert_hidden_dim"]),
        object_embedding_dim=int(config["expert_hidden_dim"]),
    )
    if resume and model_path.is_file() and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("checkpoint_sha256") != _sha256(model_path):
            raise ValueError(f"completed expert checkpoint hash mismatch: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        return model, state
    if model_path.exists() or state_path.exists():
        raise FileExistsError(f"partial non-resumable expert artifact: {stem}")

    train_batches = stage3a.make_batches(
        dict(tensors),
        np.asarray(train_indices, dtype=np.int64),
        int(config["batch_size"]),
        seed=int(seed),
    )
    validation_batches = [
        batch
        for _, batch in stage3a.make_eval_batches(
            dict(tensors),
            np.asarray(validation_indices, dtype=np.int64),
            int(config["batch_size"]),
        )
    ]
    result = train_model(
        model,
        expert_loss,
        train_batches,
        validation_batches,
        seed=int(seed),
        max_epochs=int(config["max_epochs"]),
        early_stopping_patience=int(config["early_stopping_patience"]),
        learning_rate=float(config["learning_rate"]),
        device=str(config["device"]),
        progress_prefix=f"stage3r-{label}-{condition}-s{seed}",
        checkpoint_path=train_checkpoint,
        resume=bool(resume),
        run_fingerprint=f"{PLAN_ID}:{label}:{condition}:{seed}",
    )
    model.load_state_dict(result["state_dict"])
    temporary = model_path.with_suffix(".pt.tmp")
    torch.save(result["state_dict"], temporary)
    temporary.replace(model_path)
    state = {
        "plan_id": PLAN_ID,
        "label": label,
        "condition": condition,
        "seed": int(seed),
        "train_episode_count": int(len(train_indices)),
        "validation_episode_count": int(len(validation_indices)),
        "best_val_loss": float(result["best_val_loss"]),
        "epochs_run": int(result["epochs_run"]),
        "signature": str(result["signature"]),
        "checkpoint_sha256": _sha256(model_path),
    }
    write_once_json(state_path, state)
    return model, state


def run_oof_experts(
    config: Mapping[str, Any], *, project_root: Path = ROOT, resume: bool = False
) -> dict[str, Any]:
    import run_stage3b as legacy
    from ac_twin.stage3r_models import make_oof_folds

    validated = validate_config(config, project_root=project_root)
    preflight(config, project_root=project_root)
    run_dir = validated["_run_dir_path"]
    receipt_path = run_dir / "oof" / "oof_receipt.json"
    if receipt_path.is_file():
        if not resume:
            raise FileExistsError(f"OOF receipt already exists: {receipt_path}")
        return json.loads(receipt_path.read_text(encoding="utf-8"))

    bundle = legacy._load_dev(dict(config))
    episode_count = int(bundle["tensors"]["history"].shape[0])
    all_indices = np.arange(episode_count, dtype=np.int64)
    tensors = legacy._router_base_tensors(bundle, all_indices)
    train_indices = np.sort(np.asarray(bundle["roles"]["train"], dtype=np.int64))
    validation_indices = np.sort(
        np.asarray(bundle["roles"]["validation"], dtype=np.int64)
    )
    with np.load(validated["_source_path"], allow_pickle=True) as archive:
        episode_seed = np.asarray(archive["episode_seed"], dtype=np.int64)
        split = np.asarray(archive["split"], dtype=np.int8)
    folds = make_oof_folds(
        episode_seed,
        split,
        k=int(config["k_folds"]),
        entropy=int(config["fold_entropy"]),
    )
    validate_oof_coverage(
        train_indices, [fold.holdout_indices for fold in folds]
    )
    position = {int(episode): index for index, episode in enumerate(train_indices)}
    target_shape = tuple(tensors["effect_target"].shape[1:])
    oof_full = torch.empty((5, len(train_indices), *target_shape), dtype=torch.float32)
    oof_history = torch.empty_like(oof_full)
    validation_full = torch.empty(
        (5, len(validation_indices), *target_shape), dtype=torch.float32
    )
    validation_history = torch.empty_like(validation_full)

    total = 5 * (len(folds) * 2 + 2)
    completed = 0
    started = time.monotonic()
    states: list[dict[str, Any]] = []
    checkpoint_dir = run_dir / "oof" / "checkpoints"
    for pair_index, seed in enumerate(FROZEN_EXPERT_ROUTER_SEEDS):
        for fold in folds:
            fold_positions = np.asarray(
                [position[int(item)] for item in fold.holdout_indices], dtype=np.int64
            )
            for condition, destination in (
                ("full", oof_full),
                ("history", oof_history),
            ):
                label = f"pair{pair_index}_fold{fold.index}"
                model, state = _load_or_train_expert(
                    condition=condition,
                    seed=int(seed),
                    label=label,
                    tensors=tensors,
                    train_indices=fold.train_indices,
                    validation_indices=validation_indices,
                    config=config,
                    output_dir=checkpoint_dir,
                    resume=resume,
                )
                prediction = predict_expert_ordered(
                    model,
                    tensors,
                    fold.holdout_indices,
                    device=str(config["device"]),
                    batch_size=int(config["batch_size"]),
                )
                destination[pair_index, fold_positions] = prediction
                states.append(state)
                completed += 1
                _outer_progress("stage3r-oof", completed, total, started)

        for condition, destination in (
            ("full", validation_full),
            ("history", validation_history),
        ):
            model, state = _load_or_train_expert(
                condition=condition,
                seed=int(seed),
                label=f"pair{pair_index}_fulltrain",
                tensors=tensors,
                train_indices=train_indices,
                validation_indices=validation_indices,
                config=config,
                output_dir=checkpoint_dir,
                resume=resume,
            )
            destination[pair_index] = predict_expert_ordered(
                model,
                tensors,
                validation_indices,
                device=str(config["device"]),
                batch_size=int(config["batch_size"]),
            )
            states.append(state)
            completed += 1
            _outer_progress("stage3r-oof", completed, total, started)

    train_target = tensors["effect_target"][train_indices].cpu()
    train_valid = tensors["valid"][train_indices].cpu().bool()
    teacher = build_consensus_teacher(
        oof_full, oof_history, train_target, train_valid
    )
    oof_path = run_dir / "oof" / "oof_teacher.npz"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        oof_path,
        full_by_seed=oof_full.numpy(),
        history_by_seed=oof_history.numpy(),
        target=train_target.numpy(),
        valid=train_valid.numpy(),
        train_indices=train_indices,
        episode_seed=episode_seed[train_indices],
        cew=teacher["cew"].numpy(),
        cea=teacher["cea"].numpy(),
        cea_winsorized=teacher["cea_winsorized"].numpy(),
        rank_reliable=teacher["rank_reliable"].numpy(),
        distill_weight=teacher["distill_weight"].numpy(),
        disagreement_threshold=np.asarray(
            float(teacher["disagreement_threshold"]), dtype=np.float64
        ),
        winsor_bounds=teacher["winsor_bounds"].numpy(),
    )
    validation_path = run_dir / "oof" / "validation_predictions.npz"
    np.savez_compressed(
        validation_path,
        full_by_seed=validation_full.numpy(),
        history_by_seed=validation_history.numpy(),
        target=tensors["effect_target"][validation_indices].cpu().numpy(),
        valid=tensors["valid"][validation_indices].cpu().numpy(),
        validation_indices=validation_indices,
        episode_seed=episode_seed[validation_indices],
    )
    receipt = {
        "plan_id": PLAN_ID,
        "status": "OOF_COMPLETE",
        "k": int(config["k_folds"]),
        "train_split": 0,
        "oof_episode_count": int(len(train_indices)),
        "validation_episode_count": int(len(validation_indices)),
        "train_holdout_overlap": 0,
        "expert_run_count": int(len(states)),
        "oof_teacher_sha256": _sha256(oof_path),
        "validation_predictions_sha256": _sha256(validation_path),
        "checkpoint_states": states,
    }
    write_once_json(receipt_path, receipt)
    return receipt


def oof_engineering_smoke(
    config: Mapping[str, Any], *, project_root: Path = ROOT
) -> dict[str, Any]:
    """Exercise the real-data expert training path without creating research receipts."""
    import run_stage3b as legacy

    validate_config(config, project_root=project_root)
    preflight(config, project_root=project_root)
    bundle = legacy._load_dev(dict(config))
    episode_count = int(bundle["tensors"]["history"].shape[0])
    tensors = legacy._router_base_tensors(
        bundle, np.arange(episode_count, dtype=np.int64)
    )
    train_indices = np.sort(np.asarray(bundle["roles"]["train"], dtype=np.int64))[:12]
    validation_indices = np.sort(
        np.asarray(bundle["roles"]["validation"], dtype=np.int64)
    )[:12]
    if len(train_indices) < 12 or len(validation_indices) < 12:
        raise ValueError("OOF smoke requires at least 12 train and validation episodes")

    smoke_config = dict(config)
    smoke_config.update(
        {
            "max_epochs": 1,
            "early_stopping_patience": 1,
            "batch_size": 6,
        }
    )
    with tempfile.TemporaryDirectory(prefix="stage3r_oof_smoke_") as temporary:
        checkpoint_dir = Path(temporary)
        full_model, full_state = _load_or_train_expert(
            condition="full",
            seed=1101,
            label="smoke",
            tensors=tensors,
            train_indices=train_indices,
            validation_indices=validation_indices,
            config=smoke_config,
            output_dir=checkpoint_dir,
            resume=False,
        )
        history_model, history_state = _load_or_train_expert(
            condition="history",
            seed=1101,
            label="smoke",
            tensors=tensors,
            train_indices=train_indices,
            validation_indices=validation_indices,
            config=smoke_config,
            output_dir=checkpoint_dir,
            resume=False,
        )
        full_prediction = predict_expert_ordered(
            full_model,
            tensors,
            validation_indices,
            device=str(smoke_config["device"]),
            batch_size=6,
        )
        history_prediction = predict_expert_ordered(
            history_model,
            tensors,
            validation_indices,
            device=str(smoke_config["device"]),
            batch_size=6,
        )

    full_parameters = sum(parameter.numel() for parameter in full_model.parameters())
    history_parameters = sum(
        parameter.numel() for parameter in history_model.parameters()
    )
    return {
        "status": "ENGINEERING_SMOKE_ONLY",
        "train_episode_count": int(len(train_indices)),
        "validation_episode_count": int(len(validation_indices)),
        "full_shape": list(full_prediction.shape),
        "history_shape": list(history_prediction.shape),
        "finite": bool(
            torch.isfinite(full_prediction).all()
            and torch.isfinite(history_prediction).all()
        ),
        "parameter_count_equal": full_parameters == history_parameters,
        "full_epochs_run": int(full_state["epochs_run"]),
        "history_epochs_run": int(history_state["epochs_run"]),
    }


def phase0_from_validation_arrays(
    full_by_seed: np.ndarray,
    history_by_seed: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    constant_grid: list[float] | None = None,
) -> dict[str, Any]:
    full = torch.as_tensor(full_by_seed, dtype=torch.float32)
    history = torch.as_tensor(history_by_seed, dtype=torch.float32)
    truth = torch.as_tensor(target, dtype=torch.float32)
    mask = torch.as_tensor(valid, dtype=torch.bool)
    if full.shape != history.shape or full.ndim != 5 or full.shape[0] != 5:
        raise ValueError("validation expert predictions must have shape 5,E,T,N,2")
    if full.shape[1:] != truth.shape or truth.shape[:-1] != mask.shape:
        raise ValueError("validation truth and mask do not align")
    grid = constant_grid or [round(index * 0.05, 2) for index in range(21)]
    constant_scores: dict[float, float] = {}
    constant_errors: dict[float, np.ndarray] = {}
    expanded_truth = truth.unsqueeze(0).expand_as(full)
    expanded_mask = mask.unsqueeze(0).expand(full.shape[:-1])
    for raw_weight in grid:
        weight = round(float(raw_weight), 10)
        prediction = weight * full + (1.0 - weight) * history
        errors = torch.stack(
            [
                masked_episode_ade(prediction[pair], truth, mask)
                for pair in range(5)
            ]
        ).numpy()
        constant_errors[weight] = errors
        constant_scores[weight] = float(errors.mean())
    best_weight = min(
        constant_scores, key=lambda item: (constant_scores[item], item)
    )

    oracle_weight = optimal_convex_weight(full, history, expanded_truth)
    oracle_prediction = (
        oracle_weight.unsqueeze(-1) * full
        + (1.0 - oracle_weight.unsqueeze(-1)) * history
    )
    oracle_errors = torch.stack(
        [
            masked_episode_ade(oracle_prediction[pair], truth, mask)
            for pair in range(5)
        ]
    ).numpy()
    report = evaluate_phase0(
        constant_errors[best_weight],
        oracle_errors,
        seed=int(seed),
        n_resamples=int(n_resamples),
    )
    report.update(
        {
            "best_constant_w": float(best_weight),
            "constant_scores": {
                str(weight): score for weight, score in constant_scores.items()
            },
            "soft_oracle_mean_ade": float(oracle_errors.mean()),
            "best_constant_mean_ade": float(
                constant_errors[best_weight].mean()
            ),
            "cew_mean": float(oracle_weight[expanded_mask].mean()),
            "cew_std": float(oracle_weight[expanded_mask].std()),
        }
    )
    return report


def run_phase0(
    config: Mapping[str, Any], *, project_root: Path = ROOT
) -> dict[str, Any]:
    validated = validate_config(config, project_root=project_root)
    run_dir = validated["_run_dir_path"]
    receipt_path = run_dir / "oof" / "oof_receipt.json"
    validation_path = run_dir / "oof" / "validation_predictions.npz"
    if not receipt_path.is_file() or not validation_path.is_file():
        raise RuntimeError("OOF_COMPLETE receipt and validation predictions required")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "OOF_COMPLETE":
        raise RuntimeError("OOF receipt is not complete")
    if receipt.get("validation_predictions_sha256") != _sha256(validation_path):
        raise ValueError("validation prediction hash mismatch")
    report_path = run_dir / "phase0" / "phase0_report.json"
    if report_path.exists():
        raise FileExistsError(f"Phase 0 already adjudicated: {report_path}")
    with np.load(validation_path, allow_pickle=False) as archive:
        report = phase0_from_validation_arrays(
            archive["full_by_seed"],
            archive["history_by_seed"],
            archive["target"],
            archive["valid"],
            n_resamples=int(config["bootstrap_resamples_dev"]),
            seed=20260910,
            constant_grid=list(config["constant_weight_grid"]),
        )
    report["plan_id"] = PLAN_ID
    report["validation_predictions_sha256"] = _sha256(validation_path)
    write_once_json(report_path, report)
    return report


def _load_binding_payload(validated: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(validated["_binding_path"].read_text(encoding="utf-8"))[
        "receipt_payload"
    ]


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _response_feature_archive(
    config: Mapping[str, Any],
    validated: Mapping[str, Any],
    bundle: Mapping[str, Any],
    binding: Mapping[str, Any],
    pair_index: int,
    *,
    resume: bool,
) -> Path:
    import run_stage3b as legacy

    path = validated["_run_dir_path"] / "router" / f"response_pair{pair_index}.npz"
    state_path = path.with_suffix(".json")
    if path.is_file() and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"response feature archive exists: {path}")
        if state.get("sha256") != _sha256(path):
            raise ValueError(f"response feature archive hash mismatch: {path}")
        return path
    if path.exists() or state_path.exists():
        raise FileExistsError(f"partial response feature archive: pair {pair_index}")
    train = legacy._outof_fold_train_features(
        dict(config),
        bundle,
        dict(binding),
        pair_index,
        str(config["device"]),
    )
    validation_indices = np.sort(
        np.asarray(bundle["roles"]["validation"], dtype=np.int64)
    )
    validation = legacy._pair_feature_tensors(
        bundle,
        dict(binding),
        pair_index,
        validation_indices,
        str(config["device"]),
        int(config["batch_size"]),
    )
    _atomic_npz(
        path,
        train_response_features=train["response_features"].numpy(),
        train_regime_probs=train["regime_probs"].numpy(),
        train_indices=np.asarray(train["train_index_order"], dtype=np.int64),
        validation_response_features=validation["response_features"].numpy(),
        validation_regime_probs=validation["regime_probs"].numpy(),
        validation_indices=validation_indices,
    )
    write_once_json(
        state_path,
        {
            "plan_id": PLAN_ID,
            "pair_index": int(pair_index),
            "train_feature_role": "two_fold_out_of_fold",
            "validation_feature_role": "frozen_production_predictor",
            "sha256": _sha256(path),
        },
    )
    return path


def router_objective(
    model: CEWRouter,
    features: torch.Tensor,
    full: torch.Tensor,
    history: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    teacher_cew: torch.Tensor,
    teacher_cea: torch.Tensor,
    rank_reliable: torch.Tensor,
    distill_weight: torch.Tensor,
    *,
    lambda_cew: float,
    lambda_rank: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = model(features).squeeze(-1)
    mixed = weight.unsqueeze(-1) * full + (1.0 - weight.unsqueeze(-1)) * history
    effect = masked_expert_mse(mixed, target, valid)
    cew_loss = weighted_huber(weight, teacher_cew, distill_weight)
    score_vehicle, score_valid = aggregate_episode_vehicle(weight, rank_reliable)
    cea_vehicle, cea_valid = aggregate_episode_vehicle(teacher_cea, rank_reliable)
    rank_loss = pairwise_rank_loss(
        score_vehicle, cea_vehicle, score_valid & cea_valid
    )
    return effect + float(lambda_cew) * cew_loss + float(lambda_rank) * rank_loss, weight


def _train_router_model(
    *,
    features: torch.Tensor,
    full: torch.Tensor,
    history: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    teacher_cew: torch.Tensor,
    teacher_cea: torch.Tensor,
    rank_reliable: torch.Tensor,
    distill_weight: torch.Tensor,
    validation_features: torch.Tensor,
    validation_full: torch.Tensor,
    validation_history: torch.Tensor,
    validation_target: torch.Tensor,
    validation_valid: torch.Tensor,
    lambda_cew: float,
    lambda_rank: float,
    seed: int,
    config: Mapping[str, Any],
    condition: str,
    output_path: Path,
    resume: bool,
) -> dict[str, Any]:
    state_path = output_path.with_suffix(".json")
    if output_path.is_file() and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(output_path)
        if state.get("sha256") != _sha256(output_path):
            raise ValueError(f"router checkpoint hash mismatch: {output_path}")
        return state
    if output_path.exists() or state_path.exists():
        raise FileExistsError(f"partial router artifact: {output_path.stem}")

    torch.manual_seed(int(seed))
    model = CEWRouter(int(features.shape[-1]), int(config["router_hidden_dim"]))
    device = torch.device(str(config["device"]))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    generator = np.random.default_rng(int(seed))
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epochs_run = 0
    started = time.monotonic()
    batch_size = int(config["batch_size"])
    for epoch in range(int(config["max_epochs"])):
        order = generator.permutation(features.shape[0])
        model.train()
        for start in range(0, len(order), batch_size):
            positions = torch.as_tensor(order[start : start + batch_size], dtype=torch.long)
            batch = [
                value.index_select(0, positions).to(device)
                for value in (
                    features,
                    full,
                    history,
                    target,
                    valid,
                    teacher_cew,
                    teacher_cea,
                    rank_reliable,
                    distill_weight,
                )
            ]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = router_objective(
                model,
                *batch,
                lambda_cew=lambda_cew,
                lambda_rank=lambda_rank,
            )
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_weight = model(validation_features.to(device)).squeeze(-1)
            validation_prediction = (
                validation_weight.unsqueeze(-1) * validation_full.to(device)
                + (1.0 - validation_weight.unsqueeze(-1))
                * validation_history.to(device)
            )
            validation_errors = masked_episode_ade(
                validation_prediction,
                validation_target.to(device),
                validation_valid.to(device),
            )
            validation_loss = float(validation_errors.mean().cpu())
        epochs_run = epoch + 1
        if validation_loss < best_loss - 1e-10:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        _outer_progress(
            f"stage3r-router-{condition}-s{seed}",
            epochs_run,
            int(config["max_epochs"]),
            started,
        )
        if epochs_without_improvement >= int(config["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("router training produced no checkpoint")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".pt.tmp")
    torch.save(best_state, temporary)
    temporary.replace(output_path)
    state = {
        "plan_id": PLAN_ID,
        "condition": condition,
        "seed": int(seed),
        "lambda_cew": float(lambda_cew),
        "lambda_rank": float(lambda_rank),
        "input_dim": int(features.shape[-1]),
        "epochs_run": int(epochs_run),
        "best_validation_ade": float(best_loss),
        "checkpoint": str(output_path),
        "sha256": _sha256(output_path),
    }
    write_once_json(state_path, state)
    return state


def _router_development_context(
    config: Mapping[str, Any], *, project_root: Path = ROOT
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    import run_stage3b as legacy

    validated = validate_config(config, project_root=project_root)
    bundle = legacy._load_dev(dict(config))
    binding = _load_binding_payload(validated)
    train_indices = np.sort(np.asarray(bundle["roles"]["train"], dtype=np.int64))
    validation_indices = np.sort(
        np.asarray(bundle["roles"]["validation"], dtype=np.int64)
    )
    return validated, bundle, binding, train_indices, validation_indices


def run_train_router_candidates(
    config: Mapping[str, Any], *, project_root: Path = ROOT, resume: bool = False
) -> dict[str, Any]:
    import run_stage3b as legacy

    validated, bundle, binding, train_indices, validation_indices = (
        _router_development_context(config, project_root=project_root)
    )
    phase0 = json.loads(
        (validated["_run_dir_path"] / "phase0" / "phase0_report.json").read_text(
            encoding="utf-8"
        )
    )
    if phase0.get("status") != "STAGE3R_SOFT_HEADROOM_CONFIRMED":
        raise RuntimeError("Phase 0 did not authorize router development")
    summary_path = validated["_run_dir_path"] / "router" / "training_summary.json"
    if summary_path.is_file():
        if not resume:
            raise FileExistsError(summary_path)
        return json.loads(summary_path.read_text(encoding="utf-8"))
    base = legacy._router_base_tensors(
        bundle, np.arange(bundle["tensors"]["history"].shape[0], dtype=np.int64)
    )
    with np.load(validated["_run_dir_path"] / "oof" / "oof_teacher.npz") as teacher_archive:
        oof_full = torch.as_tensor(teacher_archive["full_by_seed"], dtype=torch.float32)
        oof_history = torch.as_tensor(teacher_archive["history_by_seed"], dtype=torch.float32)
        teacher_cew = torch.as_tensor(teacher_archive["cew"], dtype=torch.float32)
        teacher_cea = torch.as_tensor(teacher_archive["cea_winsorized"], dtype=torch.float32)
        rank_reliable = torch.as_tensor(teacher_archive["rank_reliable"], dtype=torch.bool)
        distill_weight = torch.as_tensor(teacher_archive["distill_weight"], dtype=torch.float32)
        if not np.array_equal(teacher_archive["train_indices"], train_indices):
            raise ValueError("OOF train order differs from development train role")
    with np.load(validated["_run_dir_path"] / "oof" / "validation_predictions.npz") as validation_archive:
        validation_full = torch.as_tensor(validation_archive["full_by_seed"], dtype=torch.float32)
        validation_history = torch.as_tensor(validation_archive["history_by_seed"], dtype=torch.float32)
        validation_target = torch.as_tensor(validation_archive["target"], dtype=torch.float32)
        validation_valid = torch.as_tensor(validation_archive["valid"], dtype=torch.bool)
        if not np.array_equal(validation_archive["validation_indices"], validation_indices):
            raise ValueError("validation prediction order differs from role order")

    records: list[dict[str, Any]] = []
    candidate_specs = [
        ("cew_distilled_router", float(lambda_cew), float(lambda_rank), False)
        for lambda_cew in config["lambda_cew"]
        for lambda_rank in config["lambda_rank"]
    ] + [
        ("end_to_end_mixture", 0.0, 0.0, False),
        ("response_only_router", 0.0, 0.0, True),
    ]
    for pair_index, seed in enumerate(FROZEN_EXPERT_ROUTER_SEEDS):
        feature_path = _response_feature_archive(
            config,
            validated,
            bundle,
            binding,
            pair_index,
            resume=resume,
        )
        with np.load(feature_path) as response_archive:
            if not np.array_equal(response_archive["train_indices"], train_indices):
                raise ValueError("response OOF train order mismatch")
            train_features, train_response_only, metadata = build_router_features(
                history=base["history"][train_indices],
                factual_suffix=base["factual_suffix"][train_indices],
                factual_actions=base["factual_actions"][train_indices],
                counterfactual_actions=base["counterfactual_actions"][train_indices],
                response_features=torch.as_tensor(response_archive["train_response_features"]),
                regime_probs=torch.as_tensor(response_archive["train_regime_probs"]),
                full_prediction=oof_full[pair_index],
                history_prediction=oof_history[pair_index],
            )
            validation_features, validation_response_only, validation_metadata = build_router_features(
                history=base["history"][validation_indices],
                factual_suffix=base["factual_suffix"][validation_indices],
                factual_actions=base["factual_actions"][validation_indices],
                counterfactual_actions=base["counterfactual_actions"][validation_indices],
                response_features=torch.as_tensor(response_archive["validation_response_features"]),
                regime_probs=torch.as_tensor(response_archive["validation_regime_probs"]),
                full_prediction=validation_full[pair_index],
                history_prediction=validation_history[pair_index],
            )
        if metadata != validation_metadata:
            raise ValueError("train and validation router feature contracts differ")
        for condition, lambda_cew, lambda_rank, response_only in candidate_specs:
            name = (
                f"{condition}_lc{lambda_cew:g}_lr{lambda_rank:g}_pair{pair_index}_s{seed}"
            )
            record = _train_router_model(
                features=train_response_only if response_only else train_features,
                full=oof_full[pair_index],
                history=oof_history[pair_index],
                target=base["effect_target"][train_indices],
                valid=base["valid"][train_indices],
                teacher_cew=teacher_cew,
                teacher_cea=teacher_cea,
                rank_reliable=rank_reliable,
                distill_weight=distill_weight,
                validation_features=(
                    validation_response_only if response_only else validation_features
                ),
                validation_full=validation_full[pair_index],
                validation_history=validation_history[pair_index],
                validation_target=validation_target,
                validation_valid=validation_valid,
                lambda_cew=lambda_cew,
                lambda_rank=lambda_rank,
                seed=int(seed),
                config=config,
                condition=condition,
                output_path=validated["_run_dir_path"] / "router" / "checkpoints" / f"{name}.pt",
                resume=resume,
            )
            record["pair_index"] = pair_index
            record["response_only_features"] = bool(response_only)
            records.append(record)
    summary = {
        "plan_id": PLAN_ID,
        "status": "ROUTER_CANDIDATES_COMPLETE",
        "best_constant_w": float(phase0["best_constant_w"]),
        "records": records,
    }
    write_once_json(summary_path, summary)
    return summary


def _select_candidate_records(records: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for record in records:
        if record["condition"] == "cew_distilled_router":
            key = (float(record["lambda_cew"]), float(record["lambda_rank"]))
            grouped.setdefault(key, []).append(record)
    if len(grouped) != 4 or any(len(items) != 5 for items in grouped.values()):
        raise ValueError("four complete five-pair CEW candidates required")
    chosen_key = min(
        grouped,
        key=lambda key: (
            np.mean([item["best_validation_ade"] for item in grouped[key]]),
            key[1],
            key[0],
            f"lc{key[0]:g}_lr{key[1]:g}",
        ),
    )
    return {"lambda_cew": chosen_key[0], "lambda_rank": chosen_key[1]}, grouped[chosen_key]


def run_phase1(
    config: Mapping[str, Any], *, project_root: Path = ROOT
) -> dict[str, Any]:
    import run_stage3b as legacy

    validated, bundle, _, train_indices, validation_indices = _router_development_context(
        config, project_root=project_root
    )
    summary_path = validated["_run_dir_path"] / "router" / "training_summary.json"
    report_path = validated["_run_dir_path"] / "phase1" / "phase1_report.json"
    if report_path.exists():
        raise FileExistsError(f"Phase 1 already adjudicated: {report_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    chosen, chosen_records = _select_candidate_records(summary["records"])
    base = legacy._router_base_tensors(
        bundle, np.arange(bundle["tensors"]["history"].shape[0], dtype=np.int64)
    )
    with np.load(validated["_run_dir_path"] / "oof" / "validation_predictions.npz") as archive:
        full = torch.as_tensor(archive["full_by_seed"], dtype=torch.float32)
        history = torch.as_tensor(archive["history_by_seed"], dtype=torch.float32)
        target = torch.as_tensor(archive["target"], dtype=torch.float32)
        valid = torch.as_tensor(archive["valid"], dtype=torch.bool)
    teacher = build_consensus_teacher(full, history, target, valid)
    predicted_weights: list[np.ndarray] = []
    router_errors: list[np.ndarray] = []
    constant_errors: list[np.ndarray] = []
    for pair_index, record in enumerate(sorted(chosen_records, key=lambda item: item["pair_index"])):
        feature_path = validated["_run_dir_path"] / "router" / f"response_pair{pair_index}.npz"
        with np.load(feature_path) as response_archive:
            features, _, _ = build_router_features(
                history=base["history"][validation_indices],
                factual_suffix=base["factual_suffix"][validation_indices],
                factual_actions=base["factual_actions"][validation_indices],
                counterfactual_actions=base["counterfactual_actions"][validation_indices],
                response_features=torch.as_tensor(response_archive["validation_response_features"]),
                regime_probs=torch.as_tensor(response_archive["validation_regime_probs"]),
                full_prediction=full[pair_index],
                history_prediction=history[pair_index],
            )
        model = CEWRouter(int(record["input_dim"]), int(config["router_hidden_dim"]))
        model.load_state_dict(torch.load(record["checkpoint"], map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            weight = model(features).squeeze(-1)
            prediction = weight.unsqueeze(-1) * full[pair_index] + (1.0 - weight.unsqueeze(-1)) * history[pair_index]
            constant = float(summary["best_constant_w"]) * full[pair_index] + (1.0 - float(summary["best_constant_w"])) * history[pair_index]
        predicted_weights.append(weight.numpy())
        router_errors.append(masked_episode_ade(prediction, target, valid).numpy())
        constant_errors.append(masked_episode_ade(constant, target, valid).numpy())
    report = evaluate_phase1(
        np.stack(predicted_weights),
        teacher["cew"].numpy(),
        teacher["cea"].numpy(),
        valid.numpy(),
        np.stack(constant_errors),
        np.stack(router_errors),
        seed=20260911,
        n_resamples=int(config["bootstrap_resamples_dev"]),
    )
    report.update(
        {
            "plan_id": PLAN_ID,
            "provisional_candidate": chosen,
            "best_constant_w": float(summary["best_constant_w"]),
            "diagnostic_mean_validation_ade": {
                condition: float(
                    np.mean(
                        [
                            item["best_validation_ade"]
                            for item in summary["records"]
                            if item["condition"] == condition
                        ]
                    )
                )
                for condition in ("end_to_end_mixture", "response_only_router")
            },
        }
    )
    write_once_json(report_path, report)
    return report


def run_select_finalist(
    config: Mapping[str, Any], *, project_root: Path = ROOT
) -> dict[str, Any]:
    validated = validate_config(config, project_root=project_root)
    report = json.loads(
        (validated["_run_dir_path"] / "phase1" / "phase1_report.json").read_text(
            encoding="utf-8"
        )
    )
    if report.get("status") != "STAGE3R_VALUE_WEIGHT_PREDICTABLE":
        raise RuntimeError("Phase 1 did not authorize finalist selection")
    summary = json.loads(
        (validated["_run_dir_path"] / "router" / "training_summary.json").read_text(
            encoding="utf-8"
        )
    )
    chosen, records = _select_candidate_records(summary["records"])
    input_dims = {int(record["input_dim"]) for record in records}
    if len(input_dims) != 1:
        raise ValueError("finalist input dimensions differ across pairs")
    input_dim = input_dims.pop()
    receipt = {
        "plan_id": PLAN_ID,
        "status": "STAGE3R_FINALIST_FROZEN",
        "selected": chosen,
        "best_constant_w": float(summary["best_constant_w"]),
        "input_dim": input_dim,
        "direct_hidden_width": match_direct_hidden_width(
            input_dim, int(config["router_hidden_dim"])
        ),
        "checkpoints": sorted(records, key=lambda item: item["pair_index"]),
        "phase1_report_sha256": _sha256(
            validated["_run_dir_path"] / "phase1" / "phase1_report.json"
        ),
    }
    write_once_json(
        validated["_run_dir_path"] / "finalist" / "finalist_receipt.json", receipt
    )
    return receipt


def _train_direct_model(
    *,
    features: torch.Tensor,
    full: torch.Tensor,
    history: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    validation_features: torch.Tensor,
    validation_full: torch.Tensor,
    validation_history: torch.Tensor,
    validation_target: torch.Tensor,
    validation_valid: torch.Tensor,
    constant_w: float,
    seed: int,
    config: Mapping[str, Any],
    output_path: Path,
    resume: bool,
) -> dict[str, Any]:
    state_path = output_path.with_suffix(".json")
    if output_path.is_file() and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(output_path)
        if state.get("sha256") != _sha256(output_path):
            raise ValueError(f"Direct checkpoint hash mismatch: {output_path}")
        return state
    if output_path.exists() or state_path.exists():
        raise FileExistsError(f"partial Direct artifact: {output_path.stem}")
    input_dim = int(features.shape[-1])
    hidden_width = match_direct_hidden_width(input_dim, int(config["router_hidden_dim"]))
    torch.manual_seed(int(seed))
    model = SameInformationDirect(input_dim, hidden_width)
    device = torch.device(str(config["device"]))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    generator = np.random.default_rng(int(seed))
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    epochs_run = 0
    batch_size = int(config["batch_size"])
    started = time.monotonic()
    for epoch in range(int(config["max_epochs"])):
        model.train()
        order = generator.permutation(features.shape[0])
        for start in range(0, len(order), batch_size):
            positions = torch.as_tensor(order[start : start + batch_size], dtype=torch.long)
            batch_features = features.index_select(0, positions).to(device)
            batch_full = full.index_select(0, positions).to(device)
            batch_history = history.index_select(0, positions).to(device)
            batch_target = target.index_select(0, positions).to(device)
            batch_valid = valid.index_select(0, positions).to(device)
            base = float(constant_w) * batch_full + (1.0 - float(constant_w)) * batch_history
            prediction = base + model(batch_features)
            loss = masked_expert_mse(prediction, batch_target, batch_valid)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            base = float(constant_w) * validation_full.to(device) + (
                1.0 - float(constant_w)
            ) * validation_history.to(device)
            prediction = base + model(validation_features.to(device))
            validation_loss = float(
                masked_episode_ade(
                    prediction,
                    validation_target.to(device),
                    validation_valid.to(device),
                ).mean().cpu()
            )
        epochs_run = epoch + 1
        if validation_loss < best_loss - 1e-10:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
        _outer_progress(
            f"stage3r-direct-s{seed}",
            epochs_run,
            int(config["max_epochs"]),
            started,
        )
        if wait >= int(config["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("Direct training produced no checkpoint")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".pt.tmp")
    torch.save(best_state, temporary)
    temporary.replace(output_path)
    state = {
        "plan_id": PLAN_ID,
        "condition": "same_information_direct",
        "seed": int(seed),
        "input_dim": input_dim,
        "hidden_width": hidden_width,
        "epochs_run": int(epochs_run),
        "best_validation_ade": float(best_loss),
        "checkpoint": str(output_path),
        "sha256": _sha256(output_path),
    }
    write_once_json(state_path, state)
    return state


def run_train_direct(
    config: Mapping[str, Any], *, project_root: Path = ROOT, resume: bool = False
) -> dict[str, Any]:
    import run_stage3b as legacy

    validated, bundle, _, train_indices, validation_indices = _router_development_context(
        config, project_root=project_root
    )
    finalist = json.loads(
        (validated["_run_dir_path"] / "finalist" / "finalist_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    summary_path = validated["_run_dir_path"] / "direct" / "direct_summary.json"
    if summary_path.is_file():
        if not resume:
            raise FileExistsError(summary_path)
        return json.loads(summary_path.read_text(encoding="utf-8"))
    base = legacy._router_base_tensors(
        bundle, np.arange(bundle["tensors"]["history"].shape[0], dtype=np.int64)
    )
    with np.load(validated["_run_dir_path"] / "oof" / "oof_teacher.npz") as archive:
        train_full = torch.as_tensor(archive["full_by_seed"], dtype=torch.float32)
        train_history = torch.as_tensor(archive["history_by_seed"], dtype=torch.float32)
    with np.load(validated["_run_dir_path"] / "oof" / "validation_predictions.npz") as archive:
        validation_full = torch.as_tensor(archive["full_by_seed"], dtype=torch.float32)
        validation_history = torch.as_tensor(archive["history_by_seed"], dtype=torch.float32)
        validation_target = torch.as_tensor(archive["target"], dtype=torch.float32)
        validation_valid = torch.as_tensor(archive["valid"], dtype=torch.bool)
    records = []
    for pair_index, seed in enumerate(FROZEN_EXPERT_ROUTER_SEEDS):
        with np.load(validated["_run_dir_path"] / "router" / f"response_pair{pair_index}.npz") as response:
            train_features, _, _ = build_router_features(
                history=base["history"][train_indices],
                factual_suffix=base["factual_suffix"][train_indices],
                factual_actions=base["factual_actions"][train_indices],
                counterfactual_actions=base["counterfactual_actions"][train_indices],
                response_features=torch.as_tensor(response["train_response_features"]),
                regime_probs=torch.as_tensor(response["train_regime_probs"]),
                full_prediction=train_full[pair_index],
                history_prediction=train_history[pair_index],
            )
            validation_features, _, _ = build_router_features(
                history=base["history"][validation_indices],
                factual_suffix=base["factual_suffix"][validation_indices],
                factual_actions=base["factual_actions"][validation_indices],
                counterfactual_actions=base["counterfactual_actions"][validation_indices],
                response_features=torch.as_tensor(response["validation_response_features"]),
                regime_probs=torch.as_tensor(response["validation_regime_probs"]),
                full_prediction=validation_full[pair_index],
                history_prediction=validation_history[pair_index],
            )
        record = _train_direct_model(
            features=train_features,
            full=train_full[pair_index],
            history=train_history[pair_index],
            target=base["effect_target"][train_indices],
            valid=base["valid"][train_indices],
            validation_features=validation_features,
            validation_full=validation_full[pair_index],
            validation_history=validation_history[pair_index],
            validation_target=validation_target,
            validation_valid=validation_valid,
            constant_w=float(finalist["best_constant_w"]),
            seed=int(seed),
            config=config,
            output_path=validated["_run_dir_path"] / "direct" / "checkpoints" / f"direct_pair{pair_index}_s{seed}.pt",
            resume=resume,
        )
        record["pair_index"] = pair_index
        records.append(record)
    summary = {
        "plan_id": PLAN_ID,
        "status": "DIRECT_COMPLETE",
        "best_constant_w": float(finalist["best_constant_w"]),
        "records": records,
    }
    write_once_json(summary_path, summary)
    return summary


def _load_router_for_record(record: Mapping[str, Any], config: Mapping[str, Any]) -> CEWRouter:
    model = CEWRouter(int(record["input_dim"]), int(config["router_hidden_dim"]))
    model.load_state_dict(torch.load(record["checkpoint"], map_location="cpu", weights_only=True))
    return model.eval()


def run_power_rscreen(
    config: Mapping[str, Any], *, project_root: Path = ROOT
) -> dict[str, Any]:
    import run_stage3b as legacy

    validated, bundle, _, _, validation_indices = _router_development_context(
        config, project_root=project_root
    )
    power_path = validated["_run_dir_path"] / "power" / "power_report.json"
    if power_path.exists():
        raise FileExistsError(f"power already frozen: {power_path}")
    finalist = json.loads(
        (validated["_run_dir_path"] / "finalist" / "finalist_receipt.json").read_text(encoding="utf-8")
    )
    training = json.loads(
        (validated["_run_dir_path"] / "router" / "training_summary.json").read_text(encoding="utf-8")
    )
    direct = json.loads(
        (validated["_run_dir_path"] / "direct" / "direct_summary.json").read_text(encoding="utf-8")
    )
    base = legacy._router_base_tensors(
        bundle, np.arange(bundle["tensors"]["history"].shape[0], dtype=np.int64)
    )
    with np.load(validated["_run_dir_path"] / "oof" / "validation_predictions.npz") as archive:
        full = torch.as_tensor(archive["full_by_seed"], dtype=torch.float32)
        history = torch.as_tensor(archive["history_by_seed"], dtype=torch.float32)
        target = torch.as_tensor(archive["target"], dtype=torch.float32)
        valid = torch.as_tensor(archive["valid"], dtype=torch.bool)
    errors = {name: [] for name in (
        "main", "full", "history", "best_constant", "end_to_end", "direct", "response_only"
    )}
    main_records = sorted(finalist["checkpoints"], key=lambda item: item["pair_index"])
    direct_records = sorted(direct["records"], key=lambda item: item["pair_index"])
    for pair_index in range(5):
        with np.load(validated["_run_dir_path"] / "router" / f"response_pair{pair_index}.npz") as response:
            features, response_only_features, _ = build_router_features(
                history=base["history"][validation_indices],
                factual_suffix=base["factual_suffix"][validation_indices],
                factual_actions=base["factual_actions"][validation_indices],
                counterfactual_actions=base["counterfactual_actions"][validation_indices],
                response_features=torch.as_tensor(response["validation_response_features"]),
                regime_probs=torch.as_tensor(response["validation_regime_probs"]),
                full_prediction=full[pair_index],
                history_prediction=history[pair_index],
            )
        condition_records = {
            condition: next(
                item for item in training["records"]
                if item["condition"] == condition and int(item["pair_index"]) == pair_index
            )
            for condition in ("end_to_end_mixture", "response_only_router")
        }
        with torch.no_grad():
            main_weight = _load_router_for_record(main_records[pair_index], config)(features).squeeze(-1)
            e2e_weight = _load_router_for_record(condition_records["end_to_end_mixture"], config)(features).squeeze(-1)
            response_weight = _load_router_for_record(condition_records["response_only_router"], config)(response_only_features).squeeze(-1)
            constant = float(finalist["best_constant_w"])
            constant_prediction = constant * full[pair_index] + (1.0 - constant) * history[pair_index]
            main_prediction = main_weight.unsqueeze(-1) * full[pair_index] + (1.0 - main_weight.unsqueeze(-1)) * history[pair_index]
            e2e_prediction = e2e_weight.unsqueeze(-1) * full[pair_index] + (1.0 - e2e_weight.unsqueeze(-1)) * history[pair_index]
            response_prediction = response_weight.unsqueeze(-1) * full[pair_index] + (1.0 - response_weight.unsqueeze(-1)) * history[pair_index]
            direct_model = SameInformationDirect(
                int(direct_records[pair_index]["input_dim"]),
                int(direct_records[pair_index]["hidden_width"]),
            )
            direct_model.load_state_dict(torch.load(direct_records[pair_index]["checkpoint"], map_location="cpu", weights_only=True))
            direct_model.eval()
            direct_prediction = constant_prediction + direct_model(features)
        predictions = {
            "main": main_prediction,
            "full": full[pair_index],
            "history": history[pair_index],
            "best_constant": constant_prediction,
            "end_to_end": e2e_prediction,
            "direct": direct_prediction,
            "response_only": response_prediction,
        }
        for name, prediction in predictions.items():
            errors[name].append(masked_episode_ade(prediction, target, valid).numpy())
    stacked = {name: np.stack(values) for name, values in errors.items()}
    baseline_names = ["full", "history", "best_constant", "end_to_end", "direct", "response_only"]
    references = np.stack([stacked[name] for name in baseline_names])
    candidates = np.stack([stacked["main"] for _ in baseline_names])
    report = estimate_joint_power(
        references,
        candidates,
        n_candidates=[400, 600, 800, 1000, 1200, 1500, 1800, 2000],
        mdes=float(config["rscreen_mdes"]),
        target_power=float(config["rscreen_joint_power"]),
        seed=20260912,
        simulations=1000,
    )
    report.update(
        {
            "plan_id": PLAN_ID,
            "baseline_order": baseline_names,
            "observed_validation_relative": {
                name: float(1.0 - stacked["main"].mean() / stacked[name].mean())
                for name in baseline_names
            },
            "finalist_receipt_sha256": _sha256(
                validated["_run_dir_path"] / "finalist" / "finalist_receipt.json"
            ),
        }
    )
    effects_path = validated["_run_dir_path"] / "power" / "validation_effects.npz"
    _atomic_npz(effects_path, **stacked)
    report["validation_effects_sha256"] = _sha256(effects_path)
    write_once_json(power_path, report)
    return report


def run_freeze_rscreen(
    config: Mapping[str, Any], *, project_root: Path = ROOT
) -> dict[str, Any]:
    from ac_twin import stage3_data as s3d

    validated = validate_config(config, project_root=project_root)
    run_dir = validated["_run_dir_path"]
    rscreen_dir = run_dir / "rscreen"
    paths = {
        "config": rscreen_dir / "rscreen_config.json",
        "schedule": rscreen_dir / "rscreen_seed_schedule.npy",
        "schedule_doc": rscreen_dir / "rscreen_seed_schedule.json",
        "model_lock": rscreen_dir / "model_lock.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("R-screen freeze artifacts already exist")
    power_path = run_dir / "power" / "power_report.json"
    finalist_path = run_dir / "finalist" / "finalist_receipt.json"
    power = json.loads(power_path.read_text(encoding="utf-8"))
    if power.get("status") != "STAGE3R_RSCREEN_POWER_FEASIBLE":
        raise RuntimeError("R-screen power is not feasible")
    n_pairs = int(power["selected_episode_count"])
    historical = s3d.scan_all_episode_seeds(project_root / "data")
    schedule = make_disjoint_seed_schedule(
        historical,
        size=n_pairs * 15,
        entropy=20260912,
    )
    schedule_sha = hashlib.sha256(schedule.tobytes()).hexdigest()
    code_paths = [
        ROOT / "code" / "run_stage3r.py",
        ROOT / "code" / "collect_stage3r_screen.py",
        ROOT / "code" / "ac_twin" / "stage3r_models.py",
        ROOT / "code" / "ac_twin" / "stage3r_statistics.py",
        ROOT / "code" / "ac_twin" / "stage3r_screen.py",
    ]
    model_lock = {
        "plan_id": PLAN_ID,
        "status": "STAGE3R_RSCREEN_MODEL_LOCKED",
        "finalist_receipt_sha256": _sha256(finalist_path),
        "oof_receipt_sha256": _sha256(run_dir / "oof" / "oof_receipt.json"),
        "router_training_summary_sha256": _sha256(
            run_dir / "router" / "training_summary.json"
        ),
        "direct_summary_sha256": _sha256(run_dir / "direct" / "direct_summary.json"),
        "code_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
            for path in code_paths
        },
    }
    write_once_json(paths["model_lock"], model_lock)
    temporary_schedule = paths["schedule"].with_suffix(".tmp.npy")
    temporary_schedule.parent.mkdir(parents=True, exist_ok=True)
    np.save(temporary_schedule, schedule)
    temporary_schedule.replace(paths["schedule"])
    generation = dict(s3d.MERGE_GENERATION_CONTRACT)
    generation.pop("max_attempt_multiplier", None)
    screen_config = {
        "experiment_name": "stage3r_rscreen",
        "analysis_status": "stage3r_rscreen_development_test",
        "purpose": "stage3r_rscreen_development_test",
        "plan_id": PLAN_ID,
        "run_dir": str(run_dir),
        "scene": "merge",
        "env_id": "merge-v0",
        "n_pairs": n_pairs,
        "collection_seed": 20260912,
        "seed_schedule_size": int(schedule.size),
        "max_attempts": int(schedule.size),
        **generation,
        "checkpoint_every_accepted": 25,
        "output_dir": str(rscreen_dir / "collection"),
        "data_output": str(
            ROOT
            / "data"
            / "object_centric_stage"
            / "stage3r_rscreen"
            / "object_twinworld_stage3r_rscreen.npz"
        ),
        "rejection_ledger": str(
            ROOT
            / "data"
            / "object_centric_stage"
            / "stage3r_rscreen"
            / "rejection_ledger.jsonl"
        ),
        "collection_receipt": str(rscreen_dir / "collection_receipt.json"),
        "historical_data_root": str(ROOT / "data"),
        "power_report_sha256": _sha256(power_path),
        "model_lock_sha256": _sha256(paths["model_lock"]),
        "finalist_receipt_sha256": _sha256(finalist_path),
        "seed_schedule_sha256": schedule_sha,
    }
    write_once_json(
        paths["schedule_doc"],
        {
            "plan_id": PLAN_ID,
            "entropy": 20260912,
            "size": int(schedule.size),
            "schedule_sha256": schedule_sha,
            "historical_seed_count": int(historical.size),
            "historical_overlap_count": 0,
        },
    )
    write_once_json(paths["config"], screen_config)
    return validate_rscreen_readiness(run_dir)


def _load_stage3r_expert_checkpoint(
    run_dir: Path, pair_index: int, seed: int, condition: str
) -> Stage3RExpert:
    path = (
        run_dir
        / "oof"
        / "checkpoints"
        / f"pair{pair_index}_fulltrain_{condition}_seed{seed}.pt"
    )
    model = Stage3RExpert(
        condition,
        hidden_dim=64,
        object_embedding_dim=64,
    )
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def run_infer_rscreen(
    config: Mapping[str, Any], *, project_root: Path = ROOT, resume: bool = False
) -> dict[str, Any]:
    import run_stage3b as legacy

    validated = validate_config(config, project_root=project_root)
    run_dir = validated["_run_dir_path"]
    validate_rscreen_readiness(run_dir)
    screen_config = json.loads(
        (run_dir / "rscreen" / "rscreen_config.json").read_text(encoding="utf-8")
    )
    data_path = Path(screen_config["data_output"])
    collection_receipt_path = run_dir / "rscreen" / "collection_receipt.json"
    collection_receipt = json.loads(collection_receipt_path.read_text(encoding="utf-8"))
    if collection_receipt.get("data_sha256") != _sha256(data_path):
        raise ValueError("R-screen collection data hash mismatch")
    prediction_path = run_dir / "rscreen" / "predictions.npz"
    receipt_path = run_dir / "rscreen" / "inference_receipt.json"
    if prediction_path.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError("R-screen inference already completed")
        if receipt.get("predictions_sha256") != _sha256(prediction_path):
            raise ValueError("R-screen prediction hash mismatch")
        return receipt
    if prediction_path.exists() or receipt_path.exists():
        raise FileExistsError("partial R-screen inference artifacts")

    dev_bundle = legacy._load_dev(dict(config))
    eval_bundle = legacy._prepare_eval_npz(data_path, dev_bundle["scaler"])
    binding = _load_binding_payload(validated)
    indices = np.arange(eval_bundle["tensors"]["history"].shape[0], dtype=np.int64)
    base = legacy._router_base_tensors(eval_bundle, indices)
    finalist = json.loads(
        (run_dir / "finalist" / "finalist_receipt.json").read_text(encoding="utf-8")
    )
    training = json.loads(
        (run_dir / "router" / "training_summary.json").read_text(encoding="utf-8")
    )
    direct = json.loads(
        (run_dir / "direct" / "direct_summary.json").read_text(encoding="utf-8")
    )
    condition_names = [
        "main",
        "full",
        "history",
        "best_constant",
        "end_to_end",
        "direct",
        "response_only",
    ]
    errors = {name: [] for name in condition_names}
    weights = {name: [] for name in ("main", "end_to_end", "response_only")}
    for pair_index, seed in enumerate(FROZEN_EXPERT_ROUTER_SEEDS):
        full_model = _load_stage3r_expert_checkpoint(
            run_dir, pair_index, seed, "full"
        )
        history_model = _load_stage3r_expert_checkpoint(
            run_dir, pair_index, seed, "history"
        )
        full_prediction = predict_expert_ordered(
            full_model,
            base,
            indices,
            device=str(config["device"]),
            batch_size=int(config["batch_size"]),
        )
        history_prediction = predict_expert_ordered(
            history_model,
            base,
            indices,
            device=str(config["device"]),
            batch_size=int(config["batch_size"]),
        )
        eval_inputs = legacy._eval_router_tensors(
            eval_bundle,
            binding,
            pair_index,
            indices,
            str(config["device"]),
            int(config["batch_size"]),
        )
        features, response_only_features, _ = build_router_features(
            history=base["history"],
            factual_suffix=base["factual_suffix"],
            factual_actions=base["factual_actions"],
            counterfactual_actions=base["counterfactual_actions"],
            response_features=eval_inputs["response_features"],
            regime_probs=eval_inputs["regime_probs"],
            full_prediction=full_prediction,
            history_prediction=history_prediction,
        )
        main_record = finalist["checkpoints"][pair_index]
        e2e_record = next(
            item for item in training["records"]
            if item["condition"] == "end_to_end_mixture" and int(item["pair_index"]) == pair_index
        )
        response_record = next(
            item for item in training["records"]
            if item["condition"] == "response_only_router" and int(item["pair_index"]) == pair_index
        )
        direct_record = direct["records"][pair_index]
        with torch.no_grad():
            main_weight = _load_router_for_record(main_record, config)(features).squeeze(-1)
            e2e_weight = _load_router_for_record(e2e_record, config)(features).squeeze(-1)
            response_weight = _load_router_for_record(response_record, config)(response_only_features).squeeze(-1)
            constant_w = float(finalist["best_constant_w"])
            constant_prediction = constant_w * full_prediction + (1.0 - constant_w) * history_prediction
            main_prediction = main_weight.unsqueeze(-1) * full_prediction + (1.0 - main_weight.unsqueeze(-1)) * history_prediction
            e2e_prediction = e2e_weight.unsqueeze(-1) * full_prediction + (1.0 - e2e_weight.unsqueeze(-1)) * history_prediction
            response_prediction = response_weight.unsqueeze(-1) * full_prediction + (1.0 - response_weight.unsqueeze(-1)) * history_prediction
            direct_model = SameInformationDirect(
                int(direct_record["input_dim"]), int(direct_record["hidden_width"])
            )
            direct_model.load_state_dict(torch.load(direct_record["checkpoint"], map_location="cpu", weights_only=True))
            direct_model.eval()
            direct_prediction = constant_prediction + direct_model(features)
        predictions = {
            "main": main_prediction,
            "full": full_prediction,
            "history": history_prediction,
            "best_constant": constant_prediction,
            "end_to_end": e2e_prediction,
            "direct": direct_prediction,
            "response_only": response_prediction,
        }
        for name in condition_names:
            errors[name].append(
                masked_episode_ade(
                    predictions[name], base["effect_target"], base["valid"]
                ).numpy()
            )
        weights["main"].append(main_weight.numpy())
        weights["end_to_end"].append(e2e_weight.numpy())
        weights["response_only"].append(response_weight.numpy())
        _outer_progress("stage3r-rscreen-inference", pair_index + 1, 5, time.monotonic() - 1.0)
    _atomic_npz(
        prediction_path,
        errors=np.stack([np.stack(errors[name]) for name in condition_names]),
        condition_names=np.asarray(condition_names),
        main_weight=np.stack(weights["main"]),
        end_to_end_weight=np.stack(weights["end_to_end"]),
        response_only_weight=np.stack(weights["response_only"]),
        cf_action=base["cf_action"].numpy(),
        cf_duration=base["cf_duration"].numpy(),
        episode_seed=np.load(data_path, allow_pickle=True)["episode_seed"],
    )
    receipt = {
        "plan_id": PLAN_ID,
        "status": "RSCREEN_INFERENCE_COMPLETE",
        "episode_count": int(len(indices)),
        "condition_count": len(condition_names),
        "data_sha256": _sha256(data_path),
        "model_lock_sha256": _sha256(run_dir / "rscreen" / "model_lock.json"),
        "predictions_sha256": _sha256(prediction_path),
    }
    write_once_json(receipt_path, receipt)
    return receipt


def run_adjudicate_rscreen(
    config: Mapping[str, Any], *, project_root: Path = ROOT
) -> dict[str, Any]:
    validated = validate_config(config, project_root=project_root)
    run_dir = validated["_run_dir_path"]
    validate_rscreen_readiness(run_dir)
    prediction_path = run_dir / "rscreen" / "predictions.npz"
    receipt = json.loads(
        (run_dir / "rscreen" / "inference_receipt.json").read_text(encoding="utf-8")
    )
    if receipt.get("predictions_sha256") != _sha256(prediction_path):
        raise ValueError("R-screen inference hash mismatch")
    amendment_path = run_dir / "rscreen" / "adjudication_code_amendment.json"
    with np.load(prediction_path) as archive:
        return adjudicate_rscreen(
            archive["errors"],
            archive["condition_names"].tolist(),
            report_path=run_dir / "rscreen" / "rscreen_report.json",
            seed=20260913,
            n_resamples=int(config["bootstrap_resamples_rscreen"]),
            strata={
                "cf_action": archive["cf_action"],
                "cf_duration": archive["cf_duration"],
            },
            provenance={
                "predictions_sha256": _sha256(prediction_path),
                "inference_receipt_sha256": _sha256(
                    run_dir / "rscreen" / "inference_receipt.json"
                ),
                "model_lock_sha256": _sha256(
                    run_dir / "rscreen" / "model_lock.json"
                ),
                "adjudication_code_amendment_sha256": _sha256(amendment_path),
            },
        )


def _load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AC-Twin Stage 3R")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "preflight",
        "smoke",
        "oof-smoke",
        "phase1-predictability",
        "select-finalist",
        "power-rscreen",
        "freeze-rscreen",
        "adjudicate-rscreen",
    ):
        item = subparsers.add_parser(command)
        item.add_argument("--config", required=True)
    oof = subparsers.add_parser("oof-experts")
    oof.add_argument("--config", required=True)
    oof.add_argument("--resume", action="store_true")
    phase0 = subparsers.add_parser("phase0-headroom")
    phase0.add_argument("--config", required=True)
    routers = subparsers.add_parser("train-router-candidates")
    routers.add_argument("--config", required=True)
    routers.add_argument("--resume", action="store_true")
    direct = subparsers.add_parser("train-direct")
    direct.add_argument("--config", required=True)
    direct.add_argument("--resume", action="store_true")
    inference = subparsers.add_parser("infer-rscreen")
    inference.add_argument("--config", required=True)
    inference.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    if args.command == "preflight":
        report = preflight(config)
    elif args.command == "smoke":
        validated = validate_config(config)
        report = engineering_smoke(device=str(validated.get("device", "cpu")))
    elif args.command == "oof-smoke":
        report = oof_engineering_smoke(config)
    elif args.command == "oof-experts":
        report = run_oof_experts(config, resume=bool(args.resume))
    elif args.command == "phase0-headroom":
        report = run_phase0(config)
    elif args.command == "train-router-candidates":
        report = run_train_router_candidates(config, resume=bool(args.resume))
    elif args.command == "phase1-predictability":
        report = run_phase1(config)
    elif args.command == "select-finalist":
        report = run_select_finalist(config)
    elif args.command == "train-direct":
        report = run_train_direct(config, resume=bool(args.resume))
    elif args.command == "power-rscreen":
        report = run_power_rscreen(config)
    elif args.command == "freeze-rscreen":
        report = run_freeze_rscreen(config)
    elif args.command == "infer-rscreen":
        report = run_infer_rscreen(config, resume=bool(args.resume))
    else:
        report = run_adjudicate_rscreen(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
