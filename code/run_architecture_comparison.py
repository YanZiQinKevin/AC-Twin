"""Fair capacity-controlled comparison of traffic interaction encoders."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.model import PairedEffectGRU, PairedObjectEffectGRU
from ac_twin.checkpointing import canonical_fingerprint
from ac_twin.progress import format_duration
from ac_twin.statistics import paired_bootstrap_mean_difference, trajectory_errors
from run_suffix_ablation import anchor_xy, make_deranged_match

LEGAL_ENCODERS = {"row_sorted_gru", "set", "static_graph", "dynamic_graph"}
LEGAL_SUFFIX_SOURCES = {"same_episode", "shuffled"}
CAPACITY_TARGETS = {"65k": 65_000, "83k": 83_000}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/architecture_comparison.json")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed condition/seed artifacts after validating their signatures.",
    )
    return parser.parse_args(argv)


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_signature(condition: dict, config: dict, dataset_sha256: str) -> str:
    signature_payload = {
        "condition": condition,
        "dataset_sha256": dataset_sha256,
        "factual_suffix_steps": config["factual_suffix_steps"],
        "shuffle_seed": config["shuffle_seed"],
        "model": config["model"],
        "statistics": config["statistics"],
    }
    return canonical_fingerprint(signature_payload)


def write_run_metadata(file_path: Path, signature: str, summary: dict) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = file_path.with_suffix(file_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"signature": signature, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    temporary.replace(file_path)


def load_run_metadata(file_path: Path, expected_signature: str) -> dict:
    payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
    if payload.get("signature") != expected_signature:
        raise ValueError(f"Run metadata does not match the current experiment: {file_path}")
    summary = payload.get("summary", {})
    if summary.get("status") != "complete":
        raise ValueError(f"Run metadata is not complete: {file_path}")
    return payload


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def build_model(condition: dict, config: dict, position_scale_xy=(1.0, 1.0)) -> nn.Module:
    encoder_type = condition["encoder_type"]
    hidden_dim = int(condition["hidden_dim"])
    embedding_dim = int(config["model"].get("action_embedding_dim", 8))
    if encoder_type == "row_sorted_gru":
        return PairedEffectGRU(26, 5, hidden_dim, embedding_dim)
    return PairedObjectEffectGRU(
        8,
        5,
        hidden_dim,
        embedding_dim,
        encoder_type,
        float(config.get("graph_radius_m", 60.0)),
        position_scale_xy,
    )


def initialize_model(condition: dict, config: dict, seed: int,
                     position_scale_xy=(1.0, 1.0)) -> nn.Module:
    seed_all(seed)
    return build_model(condition, config, position_scale_xy)


def validate_config(config: dict) -> None:
    seeds = config["model"]["training_seeds"]
    if not config.get("smoke_test", False) and (len(seeds) != 5 or len(set(seeds)) != 5):
        raise ValueError("Confirmatory experiments require five unique training seeds")
    conditions = config["conditions"]
    if not conditions or not all(isinstance(condition, dict) for condition in conditions):
        raise ValueError("conditions must be non-empty condition objects")
    names = [condition["name"] for condition in conditions]
    if len(names) != len(set(names)):
        raise ValueError("condition names must be unique")
    for condition in conditions:
        encoder_type = condition["encoder_type"]
        suffix_source = condition["suffix_source"]
        capacity_tier = condition["capacity_tier"]
        if encoder_type not in LEGAL_ENCODERS:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        if suffix_source not in LEGAL_SUFFIX_SOURCES:
            raise ValueError(f"Unknown suffix_source: {suffix_source}")
        if capacity_tier not in CAPACITY_TARGETS:
            raise ValueError(f"Unknown capacity_tier: {capacity_tier}")
        parameters = sum(parameter.numel() for parameter in build_model(condition, config).parameters())
        target = CAPACITY_TARGETS[capacity_tier]
        if abs(parameters - target) / target > 0.015:
            raise ValueError(
                f"{condition['name']} parameter count {parameters} deviates by more than 1.5% "
                f"from tier target {target}"
            )


def legacy_condition_specs(config: dict) -> list[dict]:
    hidden_dim = int(config["model"]["hidden_dim"])
    mapping = {
        "row_sorted_gru": ("row_sorted_gru", "same_episode"),
        "stable_id_set": ("set", "same_episode"),
        "dynamic_graph": ("dynamic_graph", "same_episode"),
        "dynamic_graph_shuffled": ("dynamic_graph", "shuffled"),
    }
    return [
        {
            "name": name,
            "encoder_type": mapping[name][0],
            "hidden_dim": hidden_dim,
            "capacity_tier": "legacy",
            "suffix_source": mapping[name][1],
        }
        for name in config["conditions"]
    ]


def normalize_objects(history, factual, train_indices):
    anchor = history[:, -1, 0, 2:4].copy()

    def transform(values):
        result = values.copy()
        mask = result[..., 0] > 0
        result[..., 2] = np.where(mask, result[..., 2] - anchor[:, None, None, 0], 0)
        result[..., 3] = np.where(mask, result[..., 3] - anchor[:, None, None, 1], 0)
        return result

    history, factual = transform(history), transform(factual)
    both = np.concatenate([history[train_indices], factual[train_indices]], axis=1)
    mask = both[..., 0] > 0
    mean = np.zeros(8, np.float32)
    std = np.ones(8, np.float32)
    for feature in range(2, 6):
        mean[feature] = both[..., feature][mask].mean()
        std[feature] = max(both[..., feature][mask].std(), 1e-6)
    for values in (history, factual):
        for feature in range(2, 6):
            values[..., feature] = np.where(
                values[..., 0] > 0,
                (values[..., feature] - mean[feature]) / std[feature],
                0,
            )
    return history, factual, mean, std


def train_one(model, history, suffix, factual_actions, counterfactual_actions, targets,
              train_indices, validation_indices, test_indices, config, seed, device,
              progress_callback: Callable[[dict], None] | None = None):
    seed_all(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    loss_function = nn.MSELoss()

    def loader(indices, shuffle):
        dataset = TensorDataset(*[
            torch.from_numpy(array[indices])
            for array in (history, suffix, factual_actions, counterfactual_actions, targets)
        ])
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle,
                          generator=generator if shuffle else None, num_workers=0)

    best_state = None
    best_loss = float("inf")
    patience = 0
    log = []
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for hb, sb, fab, cab, yb in loader(train_indices, True):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(hb.float().to(device), sb.float().to(device),
                               fab.long().to(device), cab.long().to(device))
            loss = loss_function(prediction, yb.float().to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            train_total += loss.detach().item() * len(hb)
            train_count += len(hb)
        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.no_grad():
            for hb, sb, fab, cab, yb in loader(validation_indices, False):
                validation_loss = loss_function(
                    model(hb.float().to(device), sb.float().to(device),
                          fab.long().to(device), cab.long().to(device)),
                    yb.float().to(device),
                )
                validation_total += validation_loss.detach().item() * len(hb)
                validation_count += len(hb)
        mean_validation_loss = validation_total / validation_count
        log.append({"epoch": epoch, "train_mse": train_total / train_count,
                    "val_mse": mean_validation_loss})
        if mean_validation_loss < best_loss - 1e-7:
            best_loss = mean_validation_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if progress_callback is not None:
            progress_callback({
                **log[-1],
                "max_epochs": int(config["max_epochs"]),
                "best_val_mse": float(best_loss),
                "patience": patience,
            })
        if patience >= config["early_stopping_patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.no_grad():
        for hb, sb, fab, cab, _ in loader(test_indices, False):
            predictions.append(model(hb.float().to(device), sb.float().to(device),
                                     fab.long().to(device), cab.long().to(device)).cpu().numpy())
    return model, np.concatenate(predictions), log


def prepare_inputs(config, data):
    split = data["split"]
    train_indices = np.where(split == 0)[0]
    factual_actions = data["factual_actions"]
    target = data["effect_xy"].astype(np.float32)
    target_mean = target[train_indices].mean((0, 1))
    target_std = target[train_indices].std((0, 1))
    target_std[target_std < 1e-6] = 1
    observation_shape = tuple(data["obs_shape"])
    raw_history = data["history"].astype(np.float32)
    raw_factual = data["factual_future"].astype(np.float32)
    anchor = raw_history.reshape(len(split), raw_history.shape[1], *observation_shape)[:, -1, 0, 1:3]
    row_history = anchor_xy(raw_history, anchor, observation_shape)
    row_factual = anchor_xy(raw_factual, anchor, observation_shape)
    observations = np.concatenate([
        row_history[train_indices].reshape(-1, 25),
        row_factual[train_indices].reshape(-1, 25),
    ])
    observation_mean = observations.mean(0)
    observation_std = observations.std(0)
    observation_std[observation_std < 1e-6] = 1
    row_history = (row_history - observation_mean) / observation_std
    row_factual = (row_factual - observation_mean) / observation_std
    row_history = np.concatenate([row_history, np.ones((*row_history.shape[:2], 1), np.float32)], -1)
    suffix_steps = config["factual_suffix_steps"]
    row_suffix = np.zeros((len(split), target.shape[1], 26), np.float32)
    row_suffix[:, :suffix_steps] = np.concatenate([
        row_factual[:, :suffix_steps],
        np.ones((len(split), suffix_steps, 1), np.float32),
    ], -1)
    object_history, object_factual, _, object_std = normalize_objects(
        data["object_history"].astype(np.float32),
        data["object_factual_future"].astype(np.float32),
        train_indices,
    )
    object_suffix = np.zeros_like(object_factual)
    object_suffix[:, :suffix_steps] = object_factual[:, :suffix_steps]
    mapping, _ = make_deranged_match(
        split, factual_actions[:, 0], data["cf_action"], data["cf_duration"], config["shuffle_seed"]
    )
    return {
        "split": split,
        "target": target,
        "normalized_target": ((target - target_mean) / target_std).astype(np.float32),
        "target_mean": target_mean,
        "target_std": target_std,
        "factual_actions": factual_actions,
        "counterfactual_actions": data["counterfactual_actions"],
        "row_history": row_history,
        "row_suffix": row_suffix,
        "object_history": object_history,
        "object_suffix": object_suffix,
        "object_position_scale_xy": object_std[2:4],
        "shuffle_mapping": mapping,
    }


def contrast_pairs(condition_specs):
    if any(condition["capacity_tier"] == "legacy" for condition in condition_specs):
        return [("row_sorted_gru", "stable_id_set"),
                ("stable_id_set", "dynamic_graph"),
                ("dynamic_graph_shuffled", "dynamic_graph")]
    pairs = []
    for tier in CAPACITY_TARGETS:
        names = {
            (condition["encoder_type"], condition["suffix_source"]): condition["name"]
            for condition in condition_specs if condition["capacity_tier"] == tier
        }
        requested = [
            (("row_sorted_gru", "same_episode"), ("set", "same_episode")),
            (("set", "shuffled"), ("set", "same_episode")),
            (("set", "same_episode"), ("static_graph", "same_episode")),
            (("static_graph", "same_episode"), ("dynamic_graph", "same_episode")),
            (("dynamic_graph", "shuffled"), ("dynamic_graph", "same_episode")),
        ]
        pairs.extend((names[reference], names[candidate]) for reference, candidate in requested
                     if reference in names and candidate in names)
    return pairs


def main() -> None:
    args = parse_args()
    config = json.loads(path(args.config).read_text(encoding="utf-8"))
    if all(isinstance(condition, dict) for condition in config["conditions"]):
        validate_config(config)
        condition_specs = config["conditions"]
    else:
        condition_specs = legacy_condition_specs(config)
    source_path = path(config["source_data"])
    dataset_sha256 = sha256_file(source_path)
    data = np.load(source_path, allow_pickle=True)
    prepared = prepare_inputs(config, data)
    output_dir = path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    split = prepared["split"]
    train_indices = np.where(split == 0)[0]
    validation_indices = np.where(split == 1)[0]
    test_indices = np.where(split == 2)[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    print(
        f"[train] device={device} conditions={len(condition_specs)} "
        f"seeds={len(config['model']['training_seeds'])}",
        flush=True,
    )
    results, errors, predictions = {}, {}, {}
    total_runs = len(condition_specs) * len(config["model"]["training_seeds"])
    run_number = 0
    for condition in condition_specs:
        name = condition["name"]
        condition_predictions, runs = [], []
        for seed in config["model"]["training_seeds"]:
            run_number += 1
            model = initialize_model(
                condition, config, seed, prepared["object_position_scale_xy"]
            )
            if condition["encoder_type"] == "row_sorted_gru":
                history, suffix = prepared["row_history"], prepared["row_suffix"]
            else:
                history, suffix = prepared["object_history"], prepared["object_suffix"]
            if condition["suffix_source"] == "shuffled":
                suffix = suffix[prepared["shuffle_mapping"]]
            checkpoint_path = output_dir / f"{name}_seed{seed}.pt"
            training_path = output_dir / f"{name}_seed{seed}_training.json"
            prediction_path = output_dir / f"{name}_seed{seed}_prediction.npy"
            metadata_path = output_dir / f"{name}_seed{seed}_run.json"
            signature = run_signature(condition, config, dataset_sha256)
            existing_artifacts = [
                candidate for candidate in
                (checkpoint_path, training_path, prediction_path, metadata_path)
                if candidate.exists()
            ]
            if existing_artifacts and not args.resume:
                raise FileExistsError(
                    f"Run artifacts already exist for {name} seed={seed}; "
                    "rerun with --resume or move them aside."
                )

            if metadata_path.exists() and args.resume:
                metadata = load_run_metadata(metadata_path, signature)
                missing = [
                    candidate for candidate in
                    (checkpoint_path, training_path, prediction_path)
                    if not candidate.exists()
                ]
                if missing:
                    raise FileNotFoundError(
                        f"Completed run metadata is missing artifacts: {missing}"
                    )
                log = json.loads(training_path.read_text(encoding="utf-8"))
                normalized_prediction = np.load(prediction_path, allow_pickle=False)
                model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
                run_summary = dict(metadata["summary"])
                print(
                    f"[train {run_number:02d}/{total_runs:02d}] {name} seed={seed} "
                    "resumed=complete",
                    flush=True,
                )
            else:
                if existing_artifacts:
                    print(
                        f"[train {run_number:02d}/{total_runs:02d}] {name} seed={seed} "
                        "resume=incomplete_retrain",
                        flush=True,
                    )
                else:
                    print(
                        f"[train {run_number:02d}/{total_runs:02d}] {name} seed={seed}",
                        flush=True,
                    )
                started = time.perf_counter()
                progress_every = int(config.get("progress_every_epochs", 5))

                def report_epoch(record: dict) -> None:
                    epoch = int(record["epoch"])
                    max_epochs = int(record["max_epochs"])
                    if epoch != 1 and epoch != max_epochs and epoch % progress_every != 0:
                        return
                    elapsed = time.perf_counter() - started
                    eta = elapsed * (max_epochs - epoch) / epoch if epoch > 0 else None
                    print(
                        f"[train {run_number:02d}/{total_runs:02d}] {name} seed={seed} | "
                        f"epoch={epoch}/{max_epochs} | "
                        f"train={record['train_mse']:.6f} | val={record['val_mse']:.6f} | "
                        f"best={record['best_val_mse']:.6f} | "
                        f"elapsed={format_duration(elapsed)} | ETA={format_duration(eta)}",
                        flush=True,
                    )

                model, normalized_prediction, log = train_one(
                    model, history, suffix, prepared["factual_actions"],
                    prepared["counterfactual_actions"], prepared["normalized_target"],
                    train_indices, validation_indices, test_indices, config["model"], seed,
                    device, progress_callback=report_epoch,
                )
                runtime_seconds = time.perf_counter() - started
                prediction = normalized_prediction * prepared["target_std"] + prepared["target_mean"]
                ade, fde = trajectory_errors(prediction, prepared["target"][test_indices])
                best_record = min(log, key=lambda record: record["val_mse"])
                run_summary = {
                    "status": "complete",
                    "seed": seed,
                    "FDE": float(fde.mean()),
                    "ADE": float(ade.mean()),
                    "epochs_run": len(log),
                    "best_epoch": best_record["epoch"],
                    "best_val_mse": best_record["val_mse"],
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "runtime_seconds": runtime_seconds,
                }
                torch.save(model.state_dict(), checkpoint_path)
                training_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
                np.save(prediction_path, normalized_prediction, allow_pickle=False)
                write_run_metadata(metadata_path, signature, run_summary)

            prediction = normalized_prediction * prepared["target_std"] + prepared["target_mean"]
            ade, fde = trajectory_errors(prediction, prepared["target"][test_indices])
            condition_predictions.append(prediction)
            run_summary["FDE"] = float(fde.mean())
            run_summary["ADE"] = float(ade.mean())
            runs.append(run_summary)
        predictions[name] = np.stack(condition_predictions)
        seed_errors = [trajectory_errors(prediction, prepared["target"][test_indices])[1]
                       for prediction in condition_predictions]
        errors[name] = np.mean(seed_errors, axis=0)
        results[name] = {
            "encoder_type": condition["encoder_type"],
            "capacity_tier": condition["capacity_tier"],
            "suffix_source": condition["suffix_source"],
            "FDE": float(errors[name].mean()), "runs": runs,
        }
    contrasts = {}
    for reference, candidate in contrast_pairs(condition_specs):
        contrasts[f"{reference}_minus_{candidate}"] = paired_bootstrap_mean_difference(
            errors[reference], errors[candidate],
            n_resamples=config["statistics"]["bootstrap_resamples"],
            confidence_level=config["statistics"]["confidence_level"],
            seed=config["shuffle_seed"] + len(contrasts),
        )
    report = {"experiment": config["experiment_name"], "device": str(device),
              "n_test": len(test_indices), "results": results, "contrasts": contrasts}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / "predictions.npz",
                        target=prepared["target"][test_indices], **predictions)
    print(json.dumps(report["contrasts"], indent=2))


if __name__ == "__main__":
    main()
