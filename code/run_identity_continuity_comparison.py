"""Capacity-matched identity continuity controls with isolated evaluation data."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.checkpointing import canonical_fingerprint
from ac_twin.identity_controls import transform_object_order
from ac_twin.model import PairedObjectEffectGRU
from ac_twin.progress import format_duration
from ac_twin.statistics import hierarchical_paired_bootstrap_mean_difference, trajectory_errors
from run_architecture_comparison import (
    load_run_metadata, path, seed_all, sha256_file, train_one, write_run_metadata,
)

REQUIRED_CONDITIONS = {
    ("stable_temporal", "stable", "identity_temporal"),
    ("distance_temporal", "distance", "identity_temporal"),
    ("permuted_temporal", "permuted", "identity_temporal"),
    ("set_pool", "stable", "set"),
}
CAPACITY_TARGETS = {"65k": 65_000, "83k": 83_000}
FEATURE_NAMES = ["presence", "is_ego", "x", "y", "vx", "vy", "cos_heading", "sin_heading"]
PRIMARY_CONTRASTS = [
    ("permuted_temporal", "stable_temporal"),
    ("distance_temporal", "stable_temporal"),
]
SECONDARY_CONTRASTS = [("set_pool", "stable_temporal")]
REQUIRED_QA = ("fairness", "object_multiset", "ego_slot", "permutation",
               "data_consistency", "capacity", "finite_values", "resume", "holdout_collection")


def _paired_error_arrays(reference: dict, candidate: dict, error_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Validate an aligned [training seed, test episode] error pair."""
    reference_error = np.asarray(reference[error_key], dtype=np.float64)
    candidate_error = np.asarray(candidate[error_key], dtype=np.float64)
    if reference_error.shape != candidate_error.shape or reference_error.ndim != 2:
        raise ValueError("Paired seed-by-episode errors must have identical [S,N] shape")
    return reference_error, candidate_error


def _hierarchical_contrast(
    reference_name: str,
    candidate_name: str,
    reference: dict,
    candidate: dict,
    statistics: Mapping,
    *,
    bootstrap_seed: int,
) -> dict:
    reference_seeds = list(reference.get("training_seeds", []))
    candidate_seeds = list(candidate.get("training_seeds", []))
    if reference_seeds != candidate_seeds:
        raise ValueError("Paired conditions must use identical training seed order")

    output = {
        "reference_condition": reference_name,
        "candidate_condition": candidate_name,
        "training_seeds": reference_seeds,
        "independent_units": ["training_seed", "test_episode"],
    }
    metric_pairs = {
        "ade": _paired_error_arrays(reference, candidate, "ade_by_seed_episode"),
        "fde": _paired_error_arrays(reference, candidate, "fde_by_seed_episode"),
    }
    for metric, (reference_error, candidate_error) in metric_pairs.items():
        if reference_error.shape[0] != len(reference_seeds):
            raise ValueError("training_seeds must align with the first error-array dimension")
        if min(reference_error.shape) < 2:
            output.update({
                "status": "unavailable",
                "reason": "At least two training seeds and two test episodes are required for hierarchical bootstrap",
            })
            return output
        output[metric] = hierarchical_paired_bootstrap_mean_difference(
            reference_error, candidate_error,
            n_resamples=statistics["bootstrap_resamples"],
            confidence_level=statistics["confidence_level"],
            seed=bootstrap_seed,
        )
    output["status"] = "available"
    return output


def build_hierarchical_contrast_report(results: Mapping[str, dict], statistics: Mapping) -> dict[str, dict]:
    """Summarize the pre-registered controls with seed×episode paired intervals."""
    if not isinstance(statistics, Mapping):
        raise ValueError("statistics must provide bootstrap_resamples and confidence_level")
    required_statistics = {"bootstrap_resamples", "confidence_level"}
    if not required_statistics <= set(statistics):
        raise ValueError("statistics must provide bootstrap_resamples and confidence_level")

    report = {}
    for tier_index, tier in enumerate(CAPACITY_TARGETS):
        condition_results = {
            condition: results[f"{condition}_{tier}"]
            for condition in ("stable_temporal", "permuted_temporal", "distance_temporal", "set_pool")
            if f"{condition}_{tier}" in results
        }
        if not condition_results:
            continue
        if len(condition_results) != len(REQUIRED_CONDITIONS):
            raise ValueError(f"Capacity tier {tier} is missing fair-comparison conditions")
        tier_report = {"primary": {}, "secondary": {}}
        for group, contrasts in (("primary", PRIMARY_CONTRASTS), ("secondary", SECONDARY_CONTRASTS)):
            for contrast_index, (reference_name, candidate_name) in enumerate(contrasts):
                name = f"{reference_name}_minus_{candidate_name}"
                tier_report[group][name] = _hierarchical_contrast(
                    reference_name, candidate_name,
                    condition_results[reference_name], condition_results[candidate_name], statistics,
                    bootstrap_seed=20260905 + tier_index * 100 + contrast_index,
                )
        report[tier] = tier_report
    return report


def validate_identity_config(config: dict) -> None:
    conditions = config.get("conditions", [])
    fields = {"name", "order_mode", "encoder_type"}
    if (not isinstance(conditions, list) or len(conditions) != 4
            or any(not isinstance(c, dict) or set(c) != fields for c in conditions)
            or {(c["name"], c["order_mode"], c["encoder_type"]) for c in conditions}
            != REQUIRED_CONDITIONS):
        raise ValueError("conditions must contain exactly the four fair condition specifications")
    tiers = config.get("capacity_tiers", [])
    allowed = [{"65k", "83k"}]
    if config.get("smoke_test", False):
        allowed.append({"65k"})
    if (not isinstance(tiers, list) or len(set(tiers)) != len(tiers)
            or set(tiers) not in allowed):
        raise ValueError("capacity_tiers requires 65k and 83k; smoke may use only 65k")
    model = config.get("model", {})
    seeds = model.get("training_seeds", [])
    if not seeds or len(set(seeds)) != len(seeds) or any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ValueError("training_seeds must be unique non-negative integers (formal: five unique seeds)")
    if not config.get("smoke_test", False) and seeds != [11, 22, 33, 44, 55]:
        raise ValueError("Formal training_seeds must be the five unique seeds [11, 22, 33, 44, 55] in order")
    for key in ("batch_size", "max_epochs", "early_stopping_patience", "action_embedding_dim"):
        if not isinstance(model.get(key, 1), int) or model.get(key, 1) <= 0:
            raise ValueError(f"model.{key} must be a positive integer")
    for key, default in (("progress_every_epochs", 5), ("factual_suffix_steps", 0)):
        if not isinstance(config.get(key, default), int) or config.get(key, default) <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(config.get("permutation_seed", 0), int) or config.get("permutation_seed", 0) < 0:
        raise ValueError("permutation_seed must be a non-negative integer")
    limits = config.get("sample_limits", {})
    if limits and not config.get("smoke_test", False):
        raise ValueError("sample_limits are only allowed for smoke tests")
    if (not isinstance(limits, dict) or set(limits) - {"train", "validation", "test"}
            or any(not isinstance(n, int) or n <= 0 for n in limits.values())):
        raise ValueError("sample_limits must have positive train/validation/test limits")
    if not config.get("smoke_test", False) and not config.get("frozen_test_data"):
        raise ValueError("Formal comparisons require frozen_test_data; development_test is smoke-only")
    if not config.get("smoke_test", False):
        statistics = config.get("statistics")
        if not isinstance(statistics, Mapping):
            raise ValueError("Formal statistics must specify the frozen bootstrap settings")
        if (not isinstance(statistics.get("bootstrap_resamples"), int)
                or statistics["bootstrap_resamples"] != 10_000):
            raise ValueError("Formal statistics.bootstrap_resamples must be 10000")
        if statistics.get("confidence_level") != 0.95:
            raise ValueError("Formal statistics.confidence_level must be 0.95")


def preflight_identity_run(config: dict) -> dict:
    """Read-only data and output checks, before constructing models or writing files.

    Legacy collector JSON scalars have object dtype. Only trusted local collector
    archives may be supplied: reading those metadata scalars requires pickle.
    No trajectory errors or test performance are inspected here.
    """
    validate_identity_config(config)
    formal = not config.get("smoke_test", False)
    frozen_path = path(config["frozen_test_data"]) if config.get("frozen_test_data") else None
    if frozen_path is not None and not frozen_path.is_file():
        raise FileNotFoundError(f"冻结数据缺失 (frozen test data missing): {frozen_path}")
    source_path = path(config["source_data"])
    result = {"source_data_sha256": sha256_file(source_path),
              "frozen_test_data_sha256": sha256_file(frozen_path) if frozen_path else None}
    if not formal:
        return result

    descriptors = []
    for filename in (source_path, frozen_path):
        with np.load(filename, allow_pickle=True) as archive:
            required = {"config_json", "object_features_json", "object_capacity", "episode_seed",
                        "object_history", "object_factual_future", "effect_xy"}
            if not required <= set(archive.files):
                raise ValueError(f"Data consistency metadata/arrays missing: {required - set(archive.files)}")
            metadata = json.loads(archive["config_json"].item())
            features = json.loads(archive["object_features_json"].item())
            if features != FEATURE_NAMES:
                raise ValueError("Object feature list must match the frozen eight features")
            capacity = int(archive["object_capacity"].item())
            history, future, target = (archive[key].shape for key in (
                "object_history", "object_factual_future", "effect_xy"))
            if len(history) != 4 or len(future) != 4 or history[-1] != 8 or future[-1] != 8:
                raise ValueError("Object feature tensors must have [N,T,K,8] shape")
            if history[2] != capacity or future[2] != capacity:
                raise ValueError("Object capacity metadata does not match tensors")
            if history[1] != metadata.get("history_len"):
                raise ValueError("Object history length does not match metadata")
            if future[1] != metadata.get("horizon") or target != (history[0], future[1], 2):
                raise ValueError("Object horizon does not match metadata/target")
            if future[0] != history[0] or history[0] == 0:
                raise ValueError("Object data sample counts must match and be non-empty")
            seeds = archive["episode_seed"]
            if seeds.shape != (history[0],) or not np.issubdtype(seeds.dtype, np.integer):
                raise ValueError("episode seed must contain one integer environment seed per pair")
            collection_seed = metadata.get("collection_seed")
            if not isinstance(collection_seed, int) or isinstance(collection_seed, bool):
                raise ValueError("collection seed metadata is required")
            if metadata.get("env_id") != f"{config['scene']}-v0":
                raise ValueError("Collection scene does not match the formal config")
            descriptors.append({"collection_seed": collection_seed, "episode_seeds": set(seeds.tolist()),
                                "capacity": capacity, "history": history[1], "horizon": future[1]})
    source, frozen = descriptors
    if source["collection_seed"] == frozen["collection_seed"]:
        raise ValueError("Frozen collection seed overlaps development collection seed")
    # collect_pilot uses episode_seed directly in env.reset(seed=...), so equality
    # has the same meaning across both collections, including source split 2.
    if source["episode_seeds"] & frozen["episode_seeds"]:
        raise ValueError("Frozen episode seed set overlaps development episode seed set")
    for field in ("capacity", "history", "horizon"):
        if source[field] != frozen[field]:
            raise ValueError(f"Frozen {field} differs from source {field}")
    result["data_consistency"] = True
    result["signature"] = run_signature(
        {}, config, result["source_data_sha256"], result["frozen_test_data_sha256"],
        capacity_tier="all", seed=0)
    output = path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        manifest_path = output / "experiment_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Formal output directory has no experiment signature")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") != result["signature"]:
            raise ValueError("Formal output directory signature does not match current experiment")
    return result


def decide_identity_continuity(merge_report: dict | None, roundabout_report: dict | None) -> str:
    """Conservative, pure two-scene decision; QA evidence is supplied after review.

    Both capacities must agree. A positive primary CI also needs at least four
    of five positive seed means and positive leave-one-seed-out means. Missing
    statistics never count as evidence of no difference.
    """
    if merge_report is None or roundabout_report is None:
        return "PENDING_SECOND_SCENE"
    reports = (merge_report, roundabout_report)
    for scene, report in zip(("merge", "roundabout"), reports):
        if (report.get("scene") != scene or report.get("evaluation_mode") != "frozen_test"
                or not report.get("frozen_test_data_sha256")
                or not isinstance(report.get("qa"), Mapping)
                or any(report.get("qa", {}).get(key) is not True for key in REQUIRED_QA)):
            return "QA_FAIL"
    if merge_report["frozen_test_data_sha256"] == roundabout_report["frozen_test_data_sha256"]:
        return "QA_FAIL"
    scene_support, distance_support, pool_unreliable = [], [], []
    try:
        for report in reports:
            seeds = report["training_seeds"]
            if seeds != [11, 22, 33, 44, 55]:
                return "INCONCLUSIVE"
            primary_support, distance, pool = [], [], []
            for tier in CAPACITY_TARGETS:
                tier_report = report["hierarchical_paired_contrasts"][tier]
                for group, contrasts in (("primary", PRIMARY_CONTRASTS), ("secondary", SECONDARY_CONTRASTS)):
                    for reference, candidate in contrasts:
                        contrast = tier_report[group][f"{reference}_minus_{candidate}"]
                        metric = contrast["fde"]
                        per_seed = np.asarray(metric["per_seed_mean_differences"], dtype=float)
                        lo, hi, mean = (float(metric[key]) for key in (
                            "ci_lower", "ci_upper", "mean_paired_difference_reference_minus_candidate"))
                        if (contrast["status"] != "available" or contrast["training_seeds"] != seeds
                                or metric["independent_units"] != ["training_seed", "test_episode"]
                                or metric["seed_count"] != 5 or per_seed.shape != (5,)
                                or metric["episode_count"] != report["n_frozen_test"]
                                or metric["episode_count"] < 2
                                or metric["bootstrap_resamples"] != 10000 or metric["confidence_level"] != 0.95
                                or not np.isfinite([lo, hi, mean, *per_seed]).all() or lo > hi):
                            return "INCONCLUSIVE"
                        positive = lo > 0 and mean > 0
                        robust = (np.count_nonzero(per_seed > 0) >= 4
                                  and np.all((per_seed.sum() - per_seed) / 4 > 0))
                        if positive and not robust:
                            return "INCONCLUSIVE"
                        if group == "primary":
                            primary_support.append(positive)
                            if reference == "distance_temporal":
                                distance.append(positive)
                        else:
                            pool.append(lo <= 0)
            scene_support.append(all(primary_support))
            distance_support.append(all(distance))
            pool_unreliable.append(all(pool))
    except (KeyError, TypeError, ValueError, OverflowError):
        return "INCONCLUSIVE"
    if all(pool_unreliable):
        return "PREFER_SET_POOL"
    if all(scene_support):
        return "PASS_IDENTITY_CONTINUITY"
    if any(scene_support):
        return "SCENE_LIMITED_IDENTITY_EVIDENCE"
    if all(distance_support):
        return "DISTANCE_ONLY_DIFFERENCE_NOT_IDENTITY"
    return "INCONCLUSIVE"


def build_capacity_matched_model(
    *, encoder_type: str, target_parameters: int, feature_dim: int,
    action_dim: int, action_embedding_dim: int,
) -> tuple[nn.Module, int, int]:
    if encoder_type not in {"identity_temporal", "set"}:
        raise ValueError(f"Unsupported encoder_type: {encoder_type}")
    if target_parameters <= 0:
        raise ValueError("target_parameters must be positive")
    candidates = []
    # Meta models expose exact parameter sizes without allocating 249 models' weights.
    with torch.device("meta"):
        for hidden_dim in range(8, 257):
            candidate = PairedObjectEffectGRU(feature_dim, action_dim, hidden_dim,
                                              action_embedding_dim, encoder_type)
            count = sum(p.numel() for p in candidate.parameters())
            candidates.append((abs(count - target_parameters), hidden_dim, count))
    deviation, hidden_dim, count = min(candidates)
    if deviation / target_parameters > 0.015:
        raise ValueError(f"Closest capacity {count} differs from {target_parameters} by more than 1.5%")
    model = PairedObjectEffectGRU(feature_dim, action_dim, hidden_dim,
                                 action_embedding_dim, encoder_type)
    return model, hidden_dim, count


def fit_object_normalizer(
    history: np.ndarray, factual: np.ndarray, train_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    if len(train_indices) == 0:
        raise ValueError("Normalizer needs non-empty train_indices")
    anchors = history[train_indices, -1, 0, 2:4]
    values = np.concatenate([history[train_indices], factual[train_indices]], axis=1).astype(np.float32)
    values[..., 2:4] -= anchors[:, None, None]
    mask = values[..., 0] > 0
    if not mask.any():
        raise ValueError("Normalizer needs visible training objects")
    observed = values[..., 2:6][mask]
    if not np.isfinite(observed).all():
        raise ValueError("Visible training features must be finite")
    mean, std = np.zeros(8, np.float32), np.ones(8, np.float32)
    mean[2:6] = observed.mean(axis=0)
    std[2:6] = np.maximum(observed.std(axis=0), 1e-6)
    return {"mean": mean, "std": std}


def apply_object_normalizer(
    values: np.ndarray, anchors: np.ndarray, normalizer: dict[str, np.ndarray],
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    mask = result[..., 0] > 0
    result[..., 2:4] -= np.asarray(anchors)[:, None, None]
    result[..., 2:6] = (result[..., 2:6] - normalizer["mean"][2:6]) / normalizer["std"][2:6]
    result[..., 2:6] = np.where(mask[..., None], result[..., 2:6], 0)
    return result


def _prepare_arrays(config, data, normalizer, target_mean, target_std):
    history = np.asarray(data["object_history"], dtype=np.float32)
    factual = np.asarray(data["object_factual_future"], dtype=np.float32)
    target = np.asarray(data["effect_xy"], dtype=np.float32)
    if (history.ndim != 4 or factual.ndim != 4 or history.shape[-1] != 8
            or factual.shape[-1] != 8 or history.shape[2] != factual.shape[2]
            or len(history) != len(factual) or len(history) != len(target)
            or factual.shape[1] != target.shape[1]):
        raise ValueError("Object data must share [N,T,K,8] dimensions and target horizon")
    suffix_steps = config["factual_suffix_steps"]
    if suffix_steps > factual.shape[1]:
        raise ValueError("factual_suffix_steps exceeds available factual horizon")
    anchors = history[:, -1, 0, 2:4]
    conditions = {}
    for condition in config["conditions"]:
        mode = condition["order_mode"]
        ordered_history = transform_object_order(history, mode, data["episode_seed"],
                                                  config.get("permutation_seed", 0))
        ordered_factual = transform_object_order(factual, mode, data["episode_seed"],
                                                  config.get("permutation_seed", 0),
                                                  start_step=history.shape[1])
        normalized_history = apply_object_normalizer(ordered_history, anchors, normalizer)
        suffix = apply_object_normalizer(ordered_factual, anchors, normalizer)
        suffix[:, suffix_steps:] = 0
        conditions[condition["name"]] = {"history": normalized_history, "suffix": suffix}
    return {"conditions": conditions, "target": target,
            "normalized_target": ((target - target_mean) / target_std).astype(np.float32),
            "factual_actions": np.asarray(data["factual_actions"], dtype=np.int64),
            "counterfactual_actions": np.asarray(data["counterfactual_actions"], dtype=np.int64),
            "episode_seed": np.asarray(data["episode_seed"]).copy()}


def prepare_identity_inputs(config: dict, data: Mapping[str, np.ndarray]) -> dict:
    validate_identity_config(config)
    split = np.asarray(data["split"])
    train = np.flatnonzero(split == 0)
    validation = np.flatnonzero(split == 1)
    limits = config.get("sample_limits", {})
    train = train[:limits.get("train")]
    validation = validation[:limits.get("validation")]
    if len(train) == 0 or len(validation) == 0:
        raise ValueError("Source data needs non-empty train and validation splits")
    normalizer = fit_object_normalizer(data["object_history"], data["object_factual_future"], train)
    target = np.asarray(data["effect_xy"], dtype=np.float32)
    target_mean = target[train].mean((0, 1))
    target_std = target[train].std((0, 1))
    target_std[target_std < 1e-6] = 1
    prepared = _prepare_arrays(config, data, normalizer, target_mean, target_std)
    if config.get("frozen_test_data"):
        with np.load(path(config["frozen_test_data"]), allow_pickle=False) as archive:
            evaluation_data = {key: archive[key] for key in (
                "object_history", "object_factual_future", "effect_xy", "episode_seed",
                "factual_actions", "counterfactual_actions")}
        test = np.arange(len(evaluation_data["episode_seed"]))
        evaluation_mode, test_index_source = "frozen_test", "frozen_test_data"
    else:
        test = np.flatnonzero(split == 2)[:limits.get("test")]
        evaluation_data = {key: np.asarray(data[key])[test] for key in (
            "object_history", "object_factual_future", "effect_xy", "episode_seed",
            "factual_actions", "counterfactual_actions")}
        evaluation_mode, test_index_source = "development_test", "source_data"
    if not len(test):
        raise ValueError("Evaluation data needs non-empty test indices")
    evaluation = _prepare_arrays(config, evaluation_data, normalizer, target_mean, target_std)
    prepared.update({"normalizer": normalizer, "target_mean": target_mean, "target_std": target_std,
                     "feature_names": FEATURE_NAMES.copy(), "split": split.copy(),
                     "train_indices": train, "validation_indices": validation, "test_indices": test,
                     "evaluation": evaluation, "evaluation_mode": evaluation_mode,
                     "test_index_source": test_index_source})
    return prepared


def run_signature(condition: dict, config: dict, source_sha256: str,
                  frozen_sha256: str | None = None, *, capacity_tier: str, seed: int) -> str:
    return canonical_fingerprint({
        "config": config, "condition": condition, "capacity_tier": capacity_tier,
        "seed": seed, "source_sha256": source_sha256, "frozen_sha256": frozen_sha256,
        "implementation": {name: sha256_file(ROOT / "code" / name) for name in (
            "run_identity_continuity_comparison.py", "run_architecture_comparison.py",
            "ac_twin/model.py", "ac_twin/identity_controls.py", "ac_twin/statistics.py")},
    })


def run_experiment(config: dict, *, resume: bool = False, device: str | None = None) -> dict:
    """Train/reuse each full condition/seed run; incomplete runs restart that seed."""
    preflight = preflight_identity_run(config)
    source_path = path(config["source_data"])
    source_sha256 = preflight["source_data_sha256"]
    frozen_sha256 = preflight["frozen_test_data_sha256"]
    with np.load(source_path, allow_pickle=False) as archive:
        source = {key: archive[key] for key in (
            "object_history", "object_factual_future", "effect_xy", "episode_seed", "split",
            "factual_actions", "counterfactual_actions")}
    prepared = prepare_identity_inputs(config, source)
    evaluation = prepared["evaluation"]
    output = path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    # Check every existing signature before writing anything or training any seed.
    jobs = []
    for tier in config["capacity_tiers"]:
        for condition in config["conditions"]:
            name = f"{condition['name']}_{tier}"
            for seed in config["model"]["training_seeds"]:
                stem = f"{name}_seed{seed}"
                files = {"model": output / f"{stem}.pt", "log": output / f"{stem}_training.json",
                         "prediction": output / f"{stem}_prediction.npy", "metadata": output / f"{stem}_run.json"}
                signature = run_signature(condition, config, source_sha256, frozen_sha256,
                                          capacity_tier=tier, seed=seed)
                existing = any(file.exists() for file in files.values())
                if existing and not resume:
                    raise FileExistsError(f"Run artifacts already exist for {stem}; use --resume")
                completed = None
                if existing:
                    if not files["metadata"].is_file():
                        raise ValueError(f"Run artifacts have no signature to match: {stem}")
                    metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))
                    if metadata.get("signature") != signature:
                        raise ValueError(f"Run metadata does not match the current experiment: {stem}")
                    if metadata.get("summary", {}).get("status") == "complete":
                        missing = [str(file) for file in files.values() if not file.is_file()]
                        if missing:
                            raise FileNotFoundError(f"Completed run is missing artifacts: {missing}")
                        completed = load_run_metadata(files["metadata"], signature)["summary"]
                jobs.append((tier, condition, name, seed, files, signature, completed))

    if not config.get("smoke_test", False):
        (output / "experiment_manifest.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    np.savez_compressed(output / "normalization.npz", **prepared["normalizer"],
                        target_mean=prepared["target_mean"], target_std=prepared["target_std"],
                        train_indices=prepared["train_indices"],
                        validation_indices=prepared["validation_indices"], test_indices=prepared["test_indices"],
                        test_index_source=np.asarray(prepared["test_index_source"]))
    # Append the evaluation set only for the existing trainer's evaluation loader.
    # All optimizer and early-stopping indices continue to refer exclusively to source.
    test_offset = len(prepared["target"])
    training_test_indices = test_offset + np.arange(len(evaluation["target"]))
    combined = {key: np.concatenate([prepared[key], evaluation[key]], axis=0) for key in (
        "factual_actions", "counterfactual_actions", "normalized_target")}
    results, predictions = {}, {}
    for tier, condition, name, seed, files, signature, completed in jobs:
        seed_all(seed)
        model, hidden_dim, parameter_count = build_capacity_matched_model(
            encoder_type=condition["encoder_type"], target_parameters=CAPACITY_TARGETS[tier],
            feature_dim=8, action_dim=5,
            action_embedding_dim=int(config["model"].get("action_embedding_dim", 8)))
        prefix = f"[scene={config['scene']}] [tier={tier}] [condition={condition['name']}] [seed={seed}]"
        if completed is not None:
            model.load_state_dict(torch.load(files["model"], map_location="cpu", weights_only=True))
            normalized_prediction = np.load(files["prediction"], allow_pickle=False)
            if normalized_prediction.shape != evaluation["normalized_target"].shape or not np.isfinite(normalized_prediction).all():
                raise ValueError(f"Saved prediction does not match expected test shape/values: {files['prediction']}")
            summary = dict(completed)
            print(f"{prefix} resumed=complete", flush=True)
        else:
            if files["metadata"].exists():
                print(f"{prefix} resumed=incomplete restart_seed=true", flush=True)
            write_run_metadata(files["metadata"], signature, {"status": "running", "seed": seed})
            started = time.perf_counter()

            def report_epoch(record):
                epoch, max_epochs = int(record["epoch"]), int(record["max_epochs"])
                every = config.get("progress_every_epochs", 5)
                if epoch not in (1, max_epochs) and epoch % every:
                    return
                elapsed = time.perf_counter() - started
                eta = elapsed * (max_epochs - epoch) / epoch
                print(f"{prefix} epoch={epoch}/{max_epochs} train={record['train_mse']:.6f} "
                      f"val={record['val_mse']:.6f} elapsed={format_duration(elapsed)} "
                      f"eta={format_duration(eta)}", flush=True)

            source_input = prepared["conditions"][condition["name"]]
            test_input = evaluation["conditions"][condition["name"]]
            history = np.concatenate([source_input["history"], test_input["history"]])
            suffix = np.concatenate([source_input["suffix"], test_input["suffix"]])
            model, normalized_prediction, log = train_one(
                model, history, suffix, combined["factual_actions"], combined["counterfactual_actions"],
                combined["normalized_target"], prepared["train_indices"], prepared["validation_indices"],
                training_test_indices, config["model"], seed, device, progress_callback=report_epoch)
            best = log[0]
            for record in log[1:]:
                if record["val_mse"] < best["val_mse"] - 1e-7:
                    best = record
            summary = {"status": "complete", "seed": seed, "parameters": parameter_count,
                       "hidden_dim": hidden_dim, "epochs_run": len(log), "best_epoch": best["epoch"],
                       "best_val_mse": best["val_mse"], "runtime_seconds": time.perf_counter() - started}
            torch.save(model.state_dict(), files["model"])
            files["log"].write_text(json.dumps(log, indent=2), encoding="utf-8")
            np.save(files["prediction"], normalized_prediction, allow_pickle=False)

        prediction = normalized_prediction * prepared["target_std"] + prepared["target_mean"]
        ade, fde = trajectory_errors(prediction, evaluation["target"])
        summary.update(ADE=float(ade.mean()), FDE=float(fde.mean()))
        if completed is None:
            write_run_metadata(files["metadata"], signature, summary)
        result = results.setdefault(name, {
            **condition, "capacity_tier": tier, "target_parameters": CAPACITY_TARGETS[tier],
            "hidden_dim": hidden_dim, "parameters": parameter_count, "training_seeds": [],
            "runs": [], "fde_by_seed_episode": [], "ade_by_seed_episode": []})
        result["training_seeds"].append(seed)
        result["runs"].append(summary)
        result["fde_by_seed_episode"].append(fde.tolist())
        result["ade_by_seed_episode"].append(ade.tolist())
        predictions.setdefault(name, []).append(prediction)
        if config.get("smoke_test", False):
            print(f"{prefix} complete ADE={summary['ADE']:.6f} FDE={summary['FDE']:.6f}", flush=True)
        else:
            print(f"{prefix} complete (test metrics sealed until final report)", flush=True)

    for result in results.values():
        result["ADE"] = float(np.mean(result["ade_by_seed_episode"]))
        result["FDE"] = float(np.mean(result["fde_by_seed_episode"]))
    hierarchical_contrasts = build_hierarchical_contrast_report(
        results,
        config.get("statistics", {"bootstrap_resamples": 10_000, "confidence_level": 0.95}),
    )
    report = {
        "experiment": config["experiment_name"], "scene": config["scene"], "device": str(device),
        "evaluation_mode": prepared["evaluation_mode"], "test_index_source": prepared["test_index_source"],
        "source_data": str(source_path), "source_data_sha256": source_sha256,
        "frozen_test_data": str(path(config["frozen_test_data"])) if frozen_sha256 else None,
        "frozen_test_data_sha256": frozen_sha256, "test_dataset_sha256": frozen_sha256 or source_sha256,
        "n_source": len(source["episode_seed"]), "n_train": len(prepared["train_indices"]),
        "n_validation": len(prepared["validation_indices"]), "n_test": len(evaluation["target"]),
        "n_frozen_test": len(evaluation["target"]) if frozen_sha256 else 0,
        "config_signature": canonical_fingerprint(config), "config": config,
        "training_seeds": config["model"]["training_seeds"], "feature_names": FEATURE_NAMES,
        "train_indices": prepared["train_indices"].tolist(),
        "validation_indices": prepared["validation_indices"].tolist(),
        "test_indices": prepared["test_indices"].tolist(),
        "test_episode_seeds": evaluation["episode_seed"].tolist(),
        "results": results, "predictions_file": "predictions.npz",
        "normalization_file": "normalization.npz",
        "hierarchical_paired_contrasts": hierarchical_contrasts,
        "decision": "PENDING_SECOND_SCENE" if not config.get("smoke_test", False) else "SMOKE_ONLY",
        "qa": {key: preflight.get(key, "PENDING") for key in REQUIRED_QA},
    }
    np.savez_compressed(output / "predictions.npz", target=evaluation["target"],
                        training_seeds=np.asarray(config["model"]["training_seeds"]),
                        episode_seed=evaluation["episode_seed"],
                        **{name: np.stack(values) for name, values in predictions.items()})
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/identity_continuity_smoke.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Reuse completed runs with matching signatures (default: enabled)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = json.loads(path(args.config).read_text(encoding="utf-8"))
    report = run_experiment(config, resume=args.resume, device=args.device)
    print(f"Saved {len(report['results'])} conditions to {path(config['output_dir']) / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
