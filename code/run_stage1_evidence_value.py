"""Stage 1 evidence-value audit: M0/M1/M2 per-agent effect prediction.

Implements the frozen Stage 1 software contract. Only the factual-suffix data
source differs across the three fair conditions; the model class, capacity,
optimizer and budget are identical. The counterfactual future is used only for
targets, response truth and masks -- never as a model input. The independent
statistical unit is the episode; training seeds are averaged per agent-time.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Avoid the torch/MKL duplicate-OpenMP abort (OMP Error #15) when numpy LAPACK
# solvers run while torch is resident on this machine; sequential MKL is
# deterministic and does not change results.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.checkpointing import canonical_fingerprint
from ac_twin.progress import format_duration
from ac_twin.stage1_data import (
    DERANGEMENT_BASE_SEED,
    OBJECT_FEATURES,
    TEST_DERANGEMENT_SEED,
    VALIDATION_DERANGEMENT_SEED,
    apply_feature_normalizer,
    apply_suffix_window,
    build_derangement,
    compute_anchors,
    compute_history_agent_known,
    compute_target_effect,
    compute_valid_mask,
    derangement_manifest,
    fit_feature_normalizer,
    fit_target_normalizer,
    sha256_file,
    suffix_presence_mask,
    validate_source_contract,
    visible_count_signature,
)
from ac_twin.stage1_model import PerAgentPairedEffectGRU
from ac_twin.stage1_statistics import (
    GATE_BLOCKED,
    GATE_S_PASS,
    OUTCOME_GO,
    aggregate_bins,
    assign_response_bin,
    build_trend_design,
    episode_bootstrap,
    evaluate_gate_a,
    evaluate_gate_b,
    evaluate_gate_s,
    fit_response_bins,
    make_spearman_statistic,
    make_trend_coefficient_statistic,
    make_v_evt_low_high_statistic,
    make_v_evt_statistic,
    position_errors,
    resolve_stage1_outcome,
    response_magnitudes,
    ridge_fit,
    ridge_predict,
    seed_mean,
    spearman,
    value_metrics,
)

CONDITIONS = (
    {"name": "no_suffix", "suffix_mode": "none"},
    {"name": "uniform_suffix", "suffix_mode": "correct"},
    {"name": "shuffled_suffix", "suffix_mode": "shuffled"},
)
SUFFIX_MODES = {c["name"]: c["suffix_mode"] for c in CONDITIONS}
ALLOWED_OUTPUT_ROOT = "experiments/stage1_evidence_value"
FORMAL_SEEDS = [11, 22, 33, 44, 55]
CODE_FILES = (
    "run_stage1_evidence_value.py",
    "ac_twin/stage1_model.py",
    "ac_twin/stage1_data.py",
    "ac_twin/stage1_statistics.py",
)
SMOKE_DECISION = "SMOKE_ONLY"
GATE_B_MIN_EPISODES = 30


def path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# --- configuration -----------------------------------------------------------

def validate_stage1_config(config: Mapping) -> None:
    required = {"experiment_name", "scene", "source_data", "output_dir",
                "evaluation_mode", "conditions", "derangement", "model",
                "statistics", "progress_every_epochs"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Stage 1 config missing keys: {sorted(missing)}")
    if config["evaluation_mode"] != "development_test":
        raise ValueError("evaluation_mode must be development_test")
    if "frozen_test_data" in config:
        raise ValueError("Stage 1 forbids frozen_test_data (no confirmatory holdout here)")

    conditions = config["conditions"]
    if (not isinstance(conditions, list) or len(conditions) != 3
            or any(not isinstance(c, dict) for c in conditions)):
        raise ValueError("conditions must be exactly three condition objects")
    names = [c.get("name") for c in conditions]
    if len(set(names)) != 3 or set(names) != set(SUFFIX_MODES):
        raise ValueError("conditions must be exactly no_suffix/uniform_suffix/shuffled_suffix")
    for c in conditions:
        if c.get("suffix_mode") != SUFFIX_MODES[c["name"]]:
            raise ValueError(f"condition {c['name']} has wrong suffix_mode")

    output_dir = str(config["output_dir"]).replace("\\", "/")
    # The three canonical configs use repo-relative paths under the fixed root;
    # absolute paths (e.g. unit-test tmp dirs) are allowed through.
    if not os.path.isabs(str(config["output_dir"])) and not output_dir.startswith(ALLOWED_OUTPUT_ROOT):
        raise ValueError(f"output_dir must be under {ALLOWED_OUTPUT_ROOT}")

    smoke = bool(config.get("smoke_test", False))
    seeds = config["model"].get("training_seeds")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("model.training_seeds must be unique")
    if not smoke and seeds != FORMAL_SEEDS:
        raise ValueError("Formal configs require exactly the five seeds [11,22,33,44,55]")

    statistics = config["statistics"]
    if not smoke:
        if statistics.get("bootstrap_resamples") != 10000:
            raise ValueError("Formal statistics.bootstrap_resamples must be 10000")
        if statistics.get("confidence_level") != 0.95:
            raise ValueError("Formal statistics.confidence_level must be 0.95")

    limits = config.get("sample_limits")
    if smoke:
        if (not isinstance(limits, Mapping)
                or not {"train", "validation", "test"} <= set(limits)
                or any(not isinstance(v, int) or v <= 0 for v in limits.values())):
            raise ValueError("smoke requires positive integer train/validation/test sample_limits")
    elif limits:
        raise ValueError("sample_limits are only allowed for smoke tests")

    derangement = config["derangement"]
    for key, expected in (("base_seed", DERANGEMENT_BASE_SEED),
                          ("validation_seed", VALIDATION_DERANGEMENT_SEED),
                          ("test_seed", TEST_DERANGEMENT_SEED)):
        if derangement.get(key) != expected:
            raise ValueError(f"derangement.{key} must be {expected}")


def run_signature(config: Mapping, condition: str, seed: int, source_sha256: str,
                  validation_manifest_sha: str, test_manifest_sha: str) -> str:
    return canonical_fingerprint({
        "config": config,
        "condition": condition,
        "seed": seed,
        "source_sha256": source_sha256,
        "validation_manifest_sha256": validation_manifest_sha,
        "test_manifest_sha256": test_manifest_sha,
        "implementation": {name: sha256_file(ROOT / "code" / name) for name in CODE_FILES},
    })


# --- data preparation ----------------------------------------------------------

def split_indices(config: Mapping, split: np.ndarray) -> dict[str, np.ndarray]:
    limits = config.get("sample_limits", {})
    train = np.flatnonzero(split == 0)[:limits.get("train")]
    validation = np.flatnonzero(split == 1)[:limits.get("validation")]
    test = np.flatnonzero(split == 2)[:limits.get("test")]
    if len(train) == 0 or len(validation) == 0 or len(test) == 0:
        raise ValueError("Each split must be non-empty")
    return {"train": train, "validation": validation, "test": test}


def prepare_stage1_arrays(config: Mapping, data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    history = np.asarray(data["object_history"], dtype=np.float32)
    factual = np.asarray(data["object_factual_future"], dtype=np.float32)
    counterfactual = np.asarray(data["object_counterfactual_future"], dtype=np.float32)
    split = np.asarray(data["split"])
    indices = split_indices(config, split)

    suffix_steps = int(config["factual_suffix_steps"])
    anchors = compute_anchors(history)
    normalizer = fit_feature_normalizer(history, factual, indices["train"])
    norm_history = apply_feature_normalizer(history, anchors, normalizer)
    norm_factual = apply_feature_normalizer(factual, anchors, normalizer)
    windowed_factual = apply_suffix_window(norm_factual, suffix_steps)
    suffix_mask = suffix_presence_mask(factual, suffix_steps)

    target = compute_target_effect(factual, counterfactual)
    valid_mask = compute_valid_mask(history, factual, counterfactual)
    target_mean, target_std = fit_target_normalizer(target, valid_mask, indices["train"])
    norm_target = ((target - target_mean) / target_std).astype(np.float32)

    history_known = compute_history_agent_known(history)
    visible_count = visible_count_signature(history_known)
    r_pos, r_vel = response_magnitudes(counterfactual, factual)

    return {
        "indices": indices,
        "split": split,
        "norm_history": norm_history,
        "windowed_factual": windowed_factual,
        "suffix_mask": suffix_mask,
        "target": target,
        "norm_target": norm_target,
        "target_mean": target_mean,
        "target_std": target_std,
        "valid_mask": valid_mask,
        "history_known": history_known,
        "visible_count": visible_count,
        "r_pos": r_pos,
        "r_vel": r_vel,
        "factual_actions": np.asarray(data["factual_actions"], dtype=np.int64),
        "counterfactual_actions": np.asarray(data["counterfactual_actions"], dtype=np.int64),
        "cf_action": np.asarray(data["cf_action"]),
        "cf_duration": np.asarray(data["cf_duration"]),
        "episode_seed": np.asarray(data["episode_seed"]),
        "object_factual_future": factual,
        "normalizer": normalizer,
        "anchors": anchors,
    }


def _split_derangement_inputs(arrays: Mapping, indices: np.ndarray):
    return (
        arrays["cf_action"][indices],
        arrays["cf_duration"][indices],
        arrays["factual_actions"][indices, 0],
        arrays["visible_count"][indices],
        arrays["episode_seed"][indices],
    )


def build_split_derangement(arrays: Mapping, indices: np.ndarray, split_label: int,
                            seed: int) -> tuple[np.ndarray, dict]:
    inputs = _split_derangement_inputs(arrays, indices)
    donor, tier = build_derangement(*inputs, seed=seed)
    cf_action, cf_duration, _first, visible_count, episode_seed = inputs
    manifest = derangement_manifest(donor, tier, split_label, cf_action, cf_duration,
                                    visible_count, episode_seed)
    return donor, manifest


def condition_suffix(arrays: Mapping, indices: np.ndarray, condition: str,
                     donor_local: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Build (suffix_features, suffix_mask) for the given condition and episodes."""
    mode = SUFFIX_MODES[condition]
    if mode == "none":
        features = np.zeros_like(arrays["windowed_factual"][indices])
        mask = np.zeros_like(arrays["suffix_mask"][indices])
        return features, mask
    if mode == "correct":
        return arrays["windowed_factual"][indices], arrays["suffix_mask"][indices]
    if donor_local is None:
        raise ValueError("shuffled_suffix requires a derangement donor mapping")
    donor_global = indices[np.asarray(donor_local)]
    return arrays["windowed_factual"][donor_global], arrays["suffix_mask"][donor_global]


# --- training -------------------------------------------------------------------

def masked_mse(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    squared = (pred - target) ** 2
    mask = valid_mask.to(pred.dtype).unsqueeze(-1)
    denominator = mask.sum() * 2.0
    if denominator.item() < 1.0:
        raise ValueError("Batch has no valid agent-time for the loss")
    return (squared * mask).sum() / denominator


def _masked_mse_numpy(pred: np.ndarray, target: np.ndarray, valid_mask: np.ndarray) -> float:
    squared = ((pred - target) ** 2).sum(-1)
    mask = valid_mask.astype(np.float64)
    if mask.sum() < 1.0:
        return float("nan")
    return float((squared * mask).sum() / (mask.sum() * 2.0))


def train_stage1_run(model: nn.Module, arrays: Mapping, indices: Mapping, condition: str,
                     train_donor_for_epoch: Callable[[int], np.ndarray],
                     validation_donor: np.ndarray | None, test_donor: np.ndarray | None,
                     model_config: Mapping, seed: int, device: torch.device,
                     progress_callback: Callable[[dict], None] | None = None) -> tuple[nn.Module, np.ndarray, np.ndarray, list]:
    """Train one condition×seed; select best epoch on validation masked MSE only.

    The recipient batch order is generated from a generator seeded by ``seed``
    and is identical across conditions and epochs; only the M2 donor mapping
    changes per epoch.
    """
    seed_all(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config["learning_rate"],
                                  weight_decay=model_config["weight_decay"])
    batch_size = int(model_config["batch_size"])
    max_epochs = int(model_config["max_epochs"])
    patience_limit = int(model_config["early_stopping_patience"])
    train_idx = indices["train"]
    generator = torch.Generator().manual_seed(seed)

    val_idx = indices["validation"]
    val_inputs = _model_inputs(arrays, val_idx, condition, validation_donor)
    test_idx = indices["test"]
    test_inputs = _model_inputs(arrays, test_idx, condition, test_donor)

    def run_split(inputs, shuffle, epoch):
        n = len(inputs["history"])
        order = torch.randperm(n, generator=generator).tolist() if shuffle else list(range(n))
        total, count = 0.0, 0
        model.train(shuffle)
        for start in range(0, n, batch_size):
            sel = order[start:start + batch_size]
            history = torch.from_numpy(inputs["history"][sel]).float().to(device)
            suffix = torch.from_numpy(inputs["suffix"][sel]).float().to(device)
            suffix_mask = torch.from_numpy(inputs["suffix_mask"][sel]).to(device)
            fa = torch.from_numpy(inputs["factual_actions"][sel]).long().to(device)
            ca = torch.from_numpy(inputs["counterfactual_actions"][sel]).long().to(device)
            target = torch.from_numpy(inputs["norm_target"][sel]).float().to(device)
            valid = torch.from_numpy(inputs["valid_mask"][sel]).to(device)
            if shuffle:
                optimizer.zero_grad(set_to_none=True)
            pred = model(history, suffix, fa, ca, suffix_mask=suffix_mask)
            loss = masked_mse(pred, target, valid)
            if shuffle:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total += loss.detach().item() * len(sel)
            count += len(sel)
        return total / max(count, 1)

    best_state, best_loss, patience, log = None, float("inf"), 0, []
    for epoch in range(1, max_epochs + 1):
        train_donor = train_donor_for_epoch(epoch)
        train_inputs = _model_inputs(arrays, train_idx, condition, train_donor)
        train_loss = run_split(train_inputs, True, epoch)
        with torch.no_grad():
            val_loss = run_split(val_inputs, False, epoch)
        log.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        if val_loss < best_loss - 1e-7:
            best_loss, patience = val_loss, 0
            best_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        else:
            patience += 1
        if progress_callback is not None:
            progress_callback({**log[-1], "max_epochs": max_epochs,
                               "best_val_mse": float(best_loss), "patience": patience})
        if patience >= patience_limit:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prediction = _predict(model, val_inputs, batch_size, device)
        test_prediction = _predict(model, test_inputs, batch_size, device)
    return model, val_prediction, test_prediction, log


def _model_inputs(arrays: Mapping, indices: np.ndarray, condition: str,
                  donor_local: np.ndarray | None) -> dict[str, np.ndarray]:
    suffix, suffix_mask = condition_suffix(arrays, indices, condition, donor_local)
    return {
        "history": arrays["norm_history"][indices],
        "suffix": suffix,
        "suffix_mask": suffix_mask,
        "factual_actions": arrays["factual_actions"][indices],
        "counterfactual_actions": arrays["counterfactual_actions"][indices],
        "norm_target": arrays["norm_target"][indices],
        "valid_mask": arrays["valid_mask"][indices],
    }


def _predict(model: nn.Module, inputs: Mapping, batch_size: int, device: torch.device) -> np.ndarray:
    n = len(inputs["history"])
    outputs = []
    for start in range(0, n, batch_size):
        sel = slice(start, min(start + batch_size, n))
        pred = model(
            torch.from_numpy(inputs["history"][sel]).float().to(device),
            torch.from_numpy(inputs["suffix"][sel]).float().to(device),
            torch.from_numpy(inputs["factual_actions"][sel]).long().to(device),
            torch.from_numpy(inputs["counterfactual_actions"][sel]).long().to(device),
            suffix_mask=torch.from_numpy(inputs["suffix_mask"][sel]).to(device),
        )
        outputs.append(pred.cpu().numpy())
    return np.concatenate(outputs, axis=0)


# --- Gate analysis (formal runs only) ------------------------------------------

def _nonego_valid_mask(arrays: Mapping, indices: np.ndarray) -> np.ndarray:
    mask = arrays["valid_mask"][indices].copy()
    mask[:, :, 0] = False  # exclude ego (slot 0) from Gate A/B main analysis
    return mask


def _initial_distance_to_ego(history: np.ndarray) -> np.ndarray:
    presence = history[..., 0] > 0
    ego_last = history[:, -1, 0, 2:4]
    episodes, _, agents, _ = history.shape
    distance = np.full((episodes, agents), np.nan)
    for e in range(episodes):
        for i in range(agents):
            present = np.nonzero(presence[e, :, i])[0]
            if len(present):
                distance[e, i] = np.linalg.norm(history[e, present[-1], i, 2:4] - ego_last[e])
    return distance


def _broadcast_covariates(arrays: Mapping, indices: np.ndarray) -> dict[str, np.ndarray]:
    n = len(indices)
    T = arrays["target"].shape[1]
    N = arrays["target"].shape[2]
    history_full = arrays["_history_full"]
    distance = _initial_distance_to_ego(history_full[indices])  # n,N
    presence = (history_full[indices][..., 0] > 0)
    visible_ratio = presence.mean(axis=1)  # n,N
    visible_count = arrays["visible_count"][indices]  # n
    horizon = np.broadcast_to(np.arange(T)[None, :, None], (n, T, N)).astype(np.float64)
    return {
        "horizon_index": horizon,
        "cf_action": np.broadcast_to(arrays["cf_action"][indices][:, None, None], (n, T, N)),
        "cf_duration": np.broadcast_to(arrays["cf_duration"][indices][:, None, None], (n, T, N)).astype(np.float64),
        "initial_distance": np.broadcast_to(distance[:, None, :], (n, T, N)),
        "history_visible_ratio": np.broadcast_to(visible_ratio[:, None, :], (n, T, N)),
        "visible_count": np.broadcast_to(visible_count[:, None, None], (n, T, N)).astype(np.float64),
    }


def analyze_stage1(arrays: Mapping, indices: Mapping, predictions: dict, config: Mapping,
                   manifests: Mapping | None = None) -> dict:
    """Compute Gate S/A/B from saved validation + test predictions (world coords)."""
    statistics = config["statistics"]
    n_resamples = int(statistics["bootstrap_resamples"])
    confidence = float(statistics["confidence_level"])
    epsilon = float(statistics["eta_epsilon_m"])
    zero_tol = float(statistics["zero_response_tolerance_m"])
    quantiles = tuple(statistics["positive_response_quantiles"])
    stat_seed = int(statistics.get("bootstrap_seed", 20260907))
    test_idx = indices["test"]
    val_idx = indices["validation"]

    # Response bins are fitted on split-0 valid non-ego agent-times only.
    train_idx = indices["train"]
    train_nonego = _nonego_valid_mask(arrays, train_idx)
    thresholds = fit_response_bins(arrays["r_pos"][train_idx][train_nonego], zero_tol, quantiles)

    test_nonego = _nonego_valid_mask(arrays, test_idx)
    bin_id_test = assign_response_bin(arrays["r_pos"][test_idx], thresholds)
    r_pos_test = arrays["r_pos"][test_idx]
    target_test = arrays["target"][test_idx]

    # Per-seed and seed-mean errors per condition on test (seeds are not samples).
    errors = {}
    errors_by_seed = {}
    for condition in SUFFIX_MODES:
        per_seed = np.stack([
            position_errors(predictions[condition][s]["test"], target_test)
            for s in range(len(predictions[condition]))
        ])
        errors_by_seed[condition] = per_seed
        errors[condition] = seed_mean(per_seed)
    l_no = errors["no_suffix"]
    l_suffix = errors["uniform_suffix"]
    l_shuffled = errors["shuffled_suffix"]

    bins = aggregate_bins(l_no, l_suffix, l_shuffled, bin_id_test, test_nonego)
    metrics = value_metrics(l_suffix, l_shuffled, l_no, epsilon)

    low_bins, high_bins = (0, 1), (3,)
    v_low_stat = make_v_evt_statistic(l_suffix, l_shuffled, bin_id_test, test_nonego, low_bins)
    v_low = episode_bootstrap(len(test_idx), v_low_stat, n_resamples, confidence, stat_seed + 1)
    diff_stat = make_v_evt_low_high_statistic(l_suffix, l_shuffled, bin_id_test, test_nonego,
                                              low_bins, high_bins)
    v_diff = episode_bootstrap(len(test_idx), diff_stat, n_resamples, confidence, stat_seed + 2)
    spearman_stat = make_spearman_statistic(r_pos_test, metrics["eta_evt"], test_nonego)
    spearman_result = episode_bootstrap(len(test_idx), spearman_stat, n_resamples, confidence,
                                        stat_seed + 3)

    # Covariate trend model (continuous standardized on train only).
    cov_test = _broadcast_covariates(arrays, test_idx)
    cov_train = _broadcast_covariates(arrays, train_idx)
    standardizer = {}
    for name in ("horizon_index", "cf_duration", "initial_distance",
                 "history_visible_ratio", "visible_count"):
        values = cov_train[name][train_nonego]
        standardizer[name] = {"mean": float(values.mean()), "std": float(max(values.std(), 1e-6))}
    X, y, feature_names, log_col = build_trend_design(
        metrics["eta_evt"], r_pos_test, cov_test["horizon_index"], cov_test["cf_action"],
        cov_test["cf_duration"], cov_test["initial_distance"],
        cov_test["history_visible_ratio"], cov_test["visible_count"], test_nonego, standardizer)
    row_episode = np.nonzero(test_nonego)[0]
    trend_stat = make_trend_coefficient_statistic(X, y, log_col, row_episode, len(test_idx))
    trend = episode_bootstrap(len(test_idx), trend_stat, n_resamples, confidence, stat_seed + 4)

    # Gate B: high-response absolute Delta = L_no - L_suffix.
    high_mask = test_nonego & (bin_id_test == 3)
    high_episodes = np.unique(np.nonzero(high_mask)[0])
    episode_axis = np.arange(len(test_idx))[:, None, None]
    if len(high_episodes) >= 2:
        def delta_stat(idx):
            per_episode = [metrics["Delta"][high_mask & (episode_axis == high_episodes[i])].mean()
                           for i in idx]
            return float(np.mean(per_episode))
        delta_boot = episode_bootstrap(len(high_episodes), delta_stat, n_resamples, confidence, stat_seed + 5)
        gate_b = evaluate_gate_b(GATE_S_PASS, delta_boot["ci_upper"], delta_boot["point"],
                                 len(high_episodes), GATE_B_MIN_EPISODES)
        gate_b_detail = delta_boot
    else:
        gate_b = evaluate_gate_b(GATE_S_PASS, None, None, len(high_episodes), GATE_B_MIN_EPISODES)
        gate_b_detail = {"n_episodes": int(len(high_episodes))}

    # Gate S (uses train/validation + frozen validation manifest only).
    gate_s, gate_s_detail = _evaluate_gate_s(arrays, indices, predictions, config, thresholds)

    # Gate A robustness: leave-one-seed/action/episode-out, high-error deletion, manifest QA.
    cf_action_test = arrays["cf_action"][test_idx]
    robustness = _gate_a_robustness(
        errors_by_seed, l_suffix, l_shuffled, bin_id_test, test_nonego, cf_action_test,
        low_bins, high_bins, n_resamples, confidence, stat_seed)
    manifest_qa_pass, manifest_qa_detail = _manifest_qa_pass(manifests)

    gate_a_conditions = {
        "low_v_evt_positive": bool(np.isfinite(v_low["ci_lower"]) and v_low["ci_lower"] > 0),
        "low_minus_high_positive": bool(np.isfinite(v_diff["ci_lower"]) and v_diff["ci_lower"] > 0),
        "spearman_positive": bool(np.isfinite(spearman_result["ci_lower"]) and spearman_result["ci_lower"] > 0),
        "trend_coefficient_positive": bool(np.isfinite(trend["ci_lower"]) and trend["ci_lower"] > 0),
        "directions_consistent": bool(
            np.nanmean(metrics["D_evt"][test_nonego]) > 0 and np.nanmean(metrics["eta_evt"][test_nonego]) < 0),
        "not_single_seed": robustness["not_single_seed"],
        "not_single_action_type": robustness["not_single_action_type"],
        "not_high_influence_episodes": robustness["not_high_influence_episodes"],
        "manifest_qa_pass": manifest_qa_pass,
    }
    gate_a_status = evaluate_gate_a(gate_s, all(gate_a_conditions.values()) if gate_s == GATE_S_PASS else None)
    if gate_s != GATE_S_PASS:
        gate_b = evaluate_gate_b(gate_s, None, None, None, GATE_B_MIN_EPISODES)
        outcome = resolve_stage1_outcome(gate_s, GATE_BLOCKED)
    else:
        outcome = resolve_stage1_outcome(gate_s, gate_a_status)

    return {
        "response_bins": thresholds,
        "bins": bins,
        "v_evt_low": v_low,
        "v_evt_low_minus_high": v_diff,
        "spearman_r_eta": spearman_result,
        "trend_log_r": {**trend, "feature_names": feature_names},
        "per_seed_low_minus_high": robustness["per_seed_low_minus_high"],
        "robustness": robustness,
        "manifest_qa_pass": manifest_qa_pass,
        "manifest_qa_detail": manifest_qa_detail,
        "gate_a_conditions": gate_a_conditions,
        "gate_s": gate_s,
        "gate_s_detail": gate_s_detail,
        "gate_a": gate_a_status,
        "gate_b": gate_b,
        "gate_b_detail": gate_b_detail,
        "outcome": outcome,
    }


def _evaluate_gate_s(arrays: Mapping, indices: Mapping, predictions: dict,
                     config: Mapping, thresholds: Mapping) -> tuple[str, dict]:
    """Gate S model-sufficiency checks; strictly train/validation data only."""
    val_idx = indices["validation"]
    val_nonego = _nonego_valid_mask(arrays, val_idx)
    r_pos_val = arrays["r_pos"][val_idx]
    statistics = config["statistics"]
    stat_seed = int(statistics.get("bootstrap_seed", 20260907))
    n_resamples = int(statistics["bootstrap_resamples"])
    confidence = float(statistics["confidence_level"])

    # S1: finite losses + non-degenerate validation outputs + identical parameter
    #     counts + ego positive control (validation only, seed-mean, strict direction).
    finite_losses = all(
        np.isfinite([r["val_mse"] for r in predictions[c][s]["log"]]).all()
        for c in SUFFIX_MODES for s in range(len(predictions[c]))
    )
    m1_val = seed_mean(np.stack([predictions["uniform_suffix"][s]["val"]
                                 for s in range(len(predictions["uniform_suffix"]))]))
    m0_val = seed_mean(np.stack([predictions["no_suffix"][s]["val"]
                                 for s in range(len(predictions["no_suffix"]))]))
    non_degenerate = bool(
        np.isfinite(m1_val).all()
        and np.count_nonzero(m1_val) > 0
        and not np.allclose(m1_val, m1_val.flat[0])
        and not all(np.allclose(m1_val[:, :, i], m1_val[:, :, 1])
                    for i in range(2, m1_val.shape[2]))
    )
    parameter_counts = [int(predictions[c][0]["parameters"]) for c in SUFFIX_MODES]
    params_equal = len(set(parameter_counts)) == 1

    # Ego positive control on validation: mean ego error of M1 must be strictly
    # below M0. Report the point difference and an episode-cluster bootstrap CI.
    target_val = arrays["target"][val_idx]
    ego_mask = arrays["valid_mask"][val_idx].copy()
    ego_mask[:, :, 1:] = False
    err_m1 = position_errors(m1_val, target_val)
    err_m0 = position_errors(m0_val, target_val)
    per_ep_m1 = _episode_group_means(err_m1, ego_mask)
    per_ep_m0 = _episode_group_means(err_m0, ego_mask)
    ego_diff = per_ep_m1 - per_ep_m0  # negative means M1 (with suffix) is better
    ego_valid = np.isfinite(ego_diff)
    n_ego_episodes = int(ego_valid.sum())

    def ego_stat(idx):
        return float(np.nanmean(ego_diff[idx]))
    if n_ego_episodes >= 2:
        ego_boot = episode_bootstrap(n_ego_episodes, lambda idx: ego_stat(idx),
                                     n_resamples, confidence, stat_seed + 41)
        ego_point = float(np.nanmean(ego_diff))
        ego_ci = (ego_boot["ci_lower"], ego_boot["ci_upper"])
    else:
        ego_point, ego_ci = float("nan"), (float("nan"), float("nan"))
    ego_control = bool(n_ego_episodes >= 2 and ego_point < 0)
    s1 = bool(finite_losses and non_degenerate and params_equal and ego_control)
    s1_detail = {"finite_losses": bool(finite_losses),
                 "non_degenerate": bool(non_degenerate),
                 "params_equal": bool(params_equal),
                 "parameter_counts": parameter_counts,
                 "ego_control_direction": bool(ego_control),
                 "ego_m1_minus_m0_point": ego_point,
                 "ego_m1_minus_m0_ci95": [float(ego_ci[0]), float(ego_ci[1])],
                 "ego_n_episodes": n_ego_episodes}

    # S2: per-agent response magnitude predictability on validation non-ego.
    s2, s2_detail = _gate_s2(arrays, val_idx, val_nonego, predictions, config, r_pos_val)

    # S3: factual-suffix channel is used (prediction + representation evidence).
    s3, s3_detail = _gate_s3(arrays, indices, predictions, config)

    status = evaluate_gate_s(s1, s2, s3)
    return status, {"S1": bool(s1), "S2": bool(s2), "S3": bool(s3),
                    "data_scope": "train_validation_only",
                    "S1_detail": s1_detail,
                    "S2_detail": s2_detail, "S3_detail": s3_detail}


def _gate_s2(arrays: Mapping, val_idx, val_nonego, predictions, config, r_pos_val):
    statistics = config["statistics"]
    n_resamples = int(statistics["bootstrap_resamples"])
    confidence = float(statistics["confidence_level"])
    stat_seed = int(statistics.get("bootstrap_seed", 20260907))
    # R_hat from M1 (seed-mean) on validation.
    r_hat_model = seed_mean(np.stack([
        np.linalg.norm(predictions["uniform_suffix"][s]["val"], axis=-1)
        for s in range(len(predictions["uniform_suffix"]))
    ]))
    cov_val = _broadcast_covariates(arrays, val_idx)
    # Baselines fit on split-0, applied to validation.
    train_idx = arrays["_indices_train"]
    train_nonego = _nonego_valid_mask(arrays, train_idx)
    r_pos_train = arrays["r_pos"][train_idx]
    cov_train = _broadcast_covariates(arrays, train_idx)

    def build_X(indices, cov, use_distance):
        fa = arrays["factual_actions"][indices]
        ca = arrays["counterfactual_actions"][indices]
        action_diff = np.broadcast_to(np.abs(fa - ca)[:, :, None].astype(np.float64),
                                      cov["horizon_index"].shape)
        cols = [np.ones_like(cov["horizon_index"]), action_diff, cov["horizon_index"]]
        if use_distance:
            cols.append(cov["initial_distance"])
        return np.column_stack([c.reshape(-1) for c in cols])

    X_train2 = build_X(train_idx, cov_train, False)
    X_train3 = build_X(train_idx, cov_train, True)
    X_val2 = build_X(val_idx, cov_val, False)
    X_val3 = build_X(val_idx, cov_val, True)
    y_train = r_pos_train[train_nonego]
    train_rows = train_nonego.reshape(-1)
    beta2 = ridge_fit(X_train2[train_rows], y_train, lam=1e-3)
    beta3 = ridge_fit(X_train3[train_rows], y_train, lam=1e-3)
    pred2 = ridge_predict(X_val2, beta2).reshape(r_pos_val.shape)
    pred3 = ridge_predict(X_val3, beta3).reshape(r_pos_val.shape)

    def mae(arr):
        return np.abs(arr - r_pos_val)[val_nonego].mean()

    # Spearman of model R_hat vs true R with episode bootstrap.
    spearman_stat = make_spearman_statistic(r_hat_model, r_pos_val, val_nonego)
    spear = episode_bootstrap(len(val_idx), spearman_stat, n_resamples, confidence, stat_seed + 11)

    def mae_diff_stat(model_mae_arr, base_arr):
        # Difference is (model - baseline); a lower model MAE gives a negative
        # value, so the pass rule is "CI upper < 0" (M1 beats the baseline).
        base = np.abs(base_arr - r_pos_val)
        mod = np.abs(model_mae_arr - r_pos_val)
        base_rows = _episode_group_means(base, val_nonego)
        mod_rows = _episode_group_means(mod, val_nonego)

        def stat(indices):
            return (mod_rows[indices] - base_rows[indices]).mean()
        return stat

    model_mae = mae(r_hat_model)
    baseline2_mae = mae(pred2)
    baseline3_mae = mae(pred3)
    diff2 = episode_bootstrap(len(val_idx), mae_diff_stat(r_hat_model, pred2),
                              n_resamples, confidence, stat_seed + 12)
    diff3 = episode_bootstrap(len(val_idx), mae_diff_stat(r_hat_model, pred3),
                              n_resamples, confidence, stat_seed + 13)
    spearman_ok = bool(np.isfinite(spear["ci_lower"]) and spear["ci_lower"] > 0)
    mae_ok = bool(np.isfinite(diff2["ci_upper"]) and diff2["ci_upper"] < 0
                  and np.isfinite(diff3["ci_upper"]) and diff3["ci_upper"] < 0)
    passed = bool(spearman_ok and mae_ok)
    return passed, {"spearman": spear, "model_mae": float(model_mae),
                    "baseline_action_horizon_mae": float(baseline2_mae),
                    "baseline_action_horizon_distance_mae": float(baseline3_mae),
                    "mae_diff_vs_baseline2": diff2, "mae_diff_vs_baseline3": diff3}


def _episode_group_means(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-episode mean of a masked agent-time array."""
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    sums = np.zeros(values.shape[0])
    counts = np.zeros(values.shape[0])
    np.add.at(sums, np.nonzero(mask)[0], values[mask])
    np.add.at(counts, np.nonzero(mask)[0], 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / np.maximum(counts, 1.0)


def _per_episode_bin_means(values: np.ndarray, bin_id: np.ndarray, valid_mask: np.ndarray,
                           bins: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-episode mean of ``values`` over valid agent-times in ``bins``.

    Returns (per_episode_mean over all episodes, eligible episode indices,
    total valid agent-time count). Episodes with no valid agent-time in the bins
    are excluded from the eligible set (not silently dropped mid-bootstrap).
    """
    values = np.asarray(values, dtype=np.float64)
    sel = np.asarray(valid_mask, dtype=bool) & np.isin(bin_id, list(bins))
    sums = np.zeros(values.shape[0])
    counts = np.zeros(values.shape[0])
    np.add.at(sums, np.nonzero(sel)[0], values[sel])
    np.add.at(counts, np.nonzero(sel)[0], 1.0)
    eligible = np.nonzero(counts > 0)[0]
    per_episode = np.full(values.shape[0], np.nan)
    per_episode[eligible] = sums[eligible] / counts[eligible]
    return per_episode, eligible, int(sel.sum())


def _make_bin_mean_diff_statistic(diff: np.ndarray, bin_id: np.ndarray, valid_mask: np.ndarray,
                                  bins: tuple[int, ...]):
    """Episode statistic: mean of per-episode (diff) within a response-bin set."""
    per_episode, eligible, _ = _per_episode_bin_means(diff, bin_id, valid_mask, bins)

    def statistic(indices: np.ndarray) -> float:
        episodes = eligible[indices]
        return float(np.nanmean(per_episode[episodes]))

    return statistic, len(eligible)


def _low_high_diff_scalar(suf_low: float, shuf_low: float, suf_high: float, shuf_high: float) -> float:
    """V_evt_low - V_evt_high from summed errors (positive = value decays with R)."""
    if shuf_low <= 0 or shuf_high <= 0:
        return float("nan")
    return (1.0 - suf_low / shuf_low) - (1.0 - suf_high / shuf_high)


def _low_high_sums(l_suffix: np.ndarray, l_shuffled: np.ndarray, bin_id: np.ndarray,
                   valid_mask: np.ndarray, low_bins: tuple[int, ...], high_bins: tuple[int, ...]):
    """Per-episode summed errors within the low and high response groups."""

    def group_sums(errors, mask):
        sums = np.zeros(errors.shape[0], dtype=np.float64)
        np.add.at(sums, np.nonzero(mask)[0], np.asarray(errors, dtype=np.float64)[mask])
        return sums

    low_mask = valid_mask & np.isin(bin_id, list(low_bins))
    high_mask = valid_mask & np.isin(bin_id, list(high_bins))
    return (group_sums(l_suffix, low_mask), group_sums(l_shuffled, low_mask),
            group_sums(l_suffix, high_mask), group_sums(l_shuffled, high_mask))


def _low_high_diff_bootstrap(suf_low, shuf_low, suf_high, shuf_high, included,
                             n_resamples, confidence, seed):
    """Episode-cluster bootstrap of V_evt_low - V_evt_high over a subset of episodes."""
    included = np.asarray(included, dtype=np.int64)

    def stat(idx):
        epi = included[idx]
        return _low_high_diff_scalar(suf_low[epi].sum(), shuf_low[epi].sum(),
                                     suf_high[epi].sum(), shuf_high[epi].sum())

    result = episode_bootstrap(len(included), stat, n_resamples, confidence, seed)
    result["n_episodes"] = int(len(included))
    return result


def _gate_a_robustness(errors_by_seed, l_suffix, l_shuffled, bin_id, valid_mask, cf_action,
                       low_bins, high_bins, n_resamples, confidence, stat_seed):
    """Gate A robustness: leave-one-seed/action/episode-out and high-error deletion."""
    n_seeds = errors_by_seed["uniform_suffix"].shape[0]
    low_bins = tuple(low_bins)
    high_bins = tuple(high_bins)

    # Per-seed per-episode sums (for leave-one-seed-out).
    per_seed_sums = []
    for s in range(n_seeds):
        per_seed_sums.append(_low_high_sums(
            errors_by_seed["uniform_suffix"][s], errors_by_seed["shuffled_suffix"][s],
            bin_id, valid_mask, low_bins, high_bins))
    sl_s = np.stack([p[0] for p in per_seed_sums])  # S,E
    hl_s = np.stack([p[1] for p in per_seed_sums])
    sh_s = np.stack([p[2] for p in per_seed_sums])
    hh_s = np.stack([p[3] for p in per_seed_sums])
    total_sl, total_hl, total_sh, total_hh = sl_s.sum(0), hl_s.sum(0), sh_s.sum(0), hh_s.sum(0)

    per_seed_direction = [
        _low_high_diff_scalar(sl_s[s].sum(), hl_s[s].sum(), sh_s[s].sum(), hh_s[s].sum())
        for s in range(n_seeds)
    ]
    leave_one_seed_out = []
    for s in range(n_seeds):
        leave_one_seed_out.append(_low_high_diff_scalar(
            (total_sl - sl_s[s]).sum(), (total_hl - hl_s[s]).sum(),
            (total_sh - sh_s[s]).sum(), (total_hh - hh_s[s]).sum()))
    not_single_seed = bool(all(np.isfinite(leave_one_seed_out))
                           and all(d > 0 for d in leave_one_seed_out))

    # Seed-mean per-episode sums for action/episode robustness.
    sl_m, hl_m, sh_m, hh_m = _low_high_sums(l_suffix, l_shuffled, bin_id, valid_mask,
                                            low_bins, high_bins)
    # Leave-one-action-type-out.
    cf_action = np.asarray(cf_action)
    action_detail = {}
    for action in sorted(np.unique(cf_action).tolist()):
        keep = cf_action != action
        diff = _low_high_diff_scalar(sl_m[keep].sum(), hl_m[keep].sum(),
                                     sh_m[keep].sum(), hh_m[keep].sum())
        action_detail[int(action)] = {"low_minus_high": float(diff),
                                      "n_episodes": int(keep.sum()),
                                      "n_episodes_removed": int((~keep).sum())}
    not_single_action_type = bool(action_detail) and all(
        np.isfinite(v["low_minus_high"]) and v["low_minus_high"] > 0 for v in action_detail.values())

    # Leave-one-episode-out (any reversal fails).
    Sl, Hl, Sh, Hh = sl_m.sum(), hl_m.sum(), sh_m.sum(), hh_m.sum()
    has_valid = valid_mask.any(axis=(1, 2))
    eligible_ep = np.nonzero(has_valid)[0]
    loeo_diffs = []
    for e in eligible_ep:
        loeo_diffs.append(_low_high_diff_scalar(Sl - sl_m[e], Hl - hl_m[e],
                                                Sh - sh_m[e], Hh - hh_m[e]))
    loeo_diffs = np.asarray(loeo_diffs)
    loeo_ok = bool(len(loeo_diffs) and np.isfinite(loeo_diffs).all() and (loeo_diffs > 0).all())

    # Predefined high-error ranking score and fixed 1%/5% deletions.
    score = _episode_group_means(l_suffix + l_shuffled, valid_mask)
    high_influence = {"leave_one_episode_out_min": float(loeo_diffs.min()) if len(loeo_diffs) else float("nan"),
                      "leave_one_episode_out_n_reversed": int((loeo_diffs <= 0).sum()) if len(loeo_diffs) else 0}
    deletion_ok = True
    for label, frac in (("full", 0.0), ("drop_top_1pct", 0.01), ("drop_top_5pct", 0.05)):
        if frac == 0.0:
            included = eligible_ep
        else:
            k = max(1, int(np.floor(frac * len(eligible_ep))))
            order = eligible_ep[np.argsort(-score[eligible_ep])]
            included = np.setdiff1d(eligible_ep, order[:k])
        boot = _low_high_diff_bootstrap(sl_m, hl_m, sh_m, hh_m, included,
                                        n_resamples, confidence, stat_seed + 61)
        high_influence[label] = boot
        if frac > 0.0 and not (np.isfinite(boot["point"]) and boot["point"] > 0):
            deletion_ok = False
    not_high_influence_episodes = bool(loeo_ok and deletion_ok)

    return {
        "per_seed_low_minus_high": [float(d) for d in per_seed_direction],
        "leave_one_seed_out_low_minus_high": [float(d) for d in leave_one_seed_out],
        "not_single_seed": not_single_seed,
        "leave_one_action_type_out": action_detail,
        "not_single_action_type": not_single_action_type,
        "high_influence_episode": high_influence,
        "not_high_influence_episodes": not_high_influence_episodes,
    }


def _manifest_qa_pass(manifests: Mapping | None):
    """Manifest QA is an explicit necessary condition for Gate A."""
    if manifests is None:
        return False, {}
    detail = {}
    ok = True
    for split_name in ("validation", "test"):
        qa = manifests.get(split_name, {}).get("qa", {})
        detail[split_name] = qa
        ok = ok and all(int(qa.get(key, -1)) == 0 for key in
                        ("self_match", "same_seed", "cross_split", "duplicate_donor"))
    return bool(ok), detail


def _gate_s3(arrays: Mapping, indices: Mapping, predictions: dict, config: Mapping):
    """S3: the factual-suffix channel is used by M1 (fixed-checkpoint suffix swap).

    Prediction evidence compares ONE M1 checkpoint run twice on validation
    (correct suffix vs frozen donor suffix); it never reads M2 weights or errors.
    Representation evidence uses the read-only factual-summary probe with
    episode-cluster bootstrap CIs against two baselines.
    """
    val_idx = indices["validation"]
    val_nonego = _nonego_valid_mask(arrays, val_idx)
    target_val = arrays["target"][val_idx]
    statistics = config["statistics"]
    n_resamples = int(statistics["bootstrap_resamples"])
    confidence = float(statistics["confidence_level"])
    stat_seed = int(statistics.get("bootstrap_seed", 20260907))

    m1 = predictions["uniform_suffix"]
    if "val_suffix_swap_correct" not in m1[0]:
        return False, {"reason": "fixed-M1 suffix-swap predictions not captured"}
    # Seed-mean the two M1-only passes first (seeds are not statistical samples).
    correct_world = seed_mean(np.stack([e["val_suffix_swap_correct"] for e in m1]))
    shuffled_world = seed_mean(np.stack([e["val_suffix_swap_shuffled"] for e in m1]))
    l_correct = position_errors(correct_world, target_val)
    l_wrong = position_errors(shuffled_world, target_val)
    diff = l_wrong - l_correct  # positive => correct suffix beats wrong suffix

    change = np.abs(shuffled_world - correct_world)
    nonzero_change = bool(np.count_nonzero(change[val_nonego]) > 0)
    # Vehicle-specificity: the per-agent mean change must not be identical for all.
    change_mag = np.linalg.norm(change, axis=-1)  # E,T,N
    agent_change = (np.where(val_nonego, change_mag, 0.0).sum(axis=(0, 1))
                    / np.maximum(val_nonego.sum(axis=(0, 1)), 1))  # N
    vehicle_specific = bool(agent_change.size > 1 and not np.allclose(agent_change, agent_change[0]))

    thresholds = arrays["_response_thresholds"]
    bin_id_val = assign_response_bin(arrays["r_pos"][val_idx], thresholds)
    gap_stats = {}
    gap_ok = False
    for name, bins in (("zero", (0,)), ("positive_low", (1,)), ("positive_mid", (2,))):
        statistic, n_eligible = _make_bin_mean_diff_statistic(diff, bin_id_val, val_nonego, bins)
        _, _, n_agent_time = _per_episode_bin_means(diff, bin_id_val, val_nonego, bins)
        if n_eligible < 2:
            gap_stats[name] = None
            continue
        boot = episode_bootstrap(n_eligible, statistic, n_resamples, confidence, stat_seed + 21)
        boot["n_episodes"] = n_eligible
        boot["n_agent_time"] = n_agent_time
        gap_stats[name] = boot
        if np.isfinite(boot["ci_lower"]) and boot["ci_lower"] > 0:
            gap_ok = True
    prediction_evidence = bool(nonzero_change and vehicle_specific and gap_ok)

    # Representation evidence: ridge probe from factual summary -> motion summary.
    representation_evidence, rep_detail = _gate_s3_probe(arrays, indices, predictions, config)

    passed = bool(prediction_evidence and representation_evidence)
    return passed, {"prediction_evidence": bool(prediction_evidence),
                    "nonzero_prediction_change": bool(nonzero_change),
                    "vehicle_specific_change": bool(vehicle_specific),
                    "method": "fixed_m1_checkpoint_suffix_swap",
                    "low_mid_bin_gaps": gap_stats,
                    "representation_evidence": bool(representation_evidence),
                    "representation_detail": rep_detail}


def _gate_s3_probe(arrays: Mapping, indices: Mapping, predictions: dict, config: Mapping):
    if "factual_summaries" not in predictions.get("uniform_suffix", [{}])[0]:
        return False, {"reason": "factual summaries not captured"}
    statistics = config["statistics"]
    n_resamples = int(statistics["bootstrap_resamples"])
    confidence = float(statistics["confidence_level"])
    stat_seed = int(statistics.get("bootstrap_seed", 20260907))
    train_idx = indices["train"]
    val_idx = indices["validation"]

    def motion_summary(factual):
        # per-agent: total displacement dx,dy; mean velocity vx,vy; final velocity vx,vy
        presence = factual[..., 0] > 0                      # E,T,N
        count = presence.sum(axis=1)                        # E,N
        disp = factual[:, -1, :, 2:4] - factual[:, 0, :, 2:4]
        vel_sum = (factual[..., 4:6] * presence[..., None]).sum(axis=1)  # E,N,2
        mean_vel = np.where(count[..., None] > 0, vel_sum / np.maximum(count, 1)[..., None], 0.0)
        final_vel = factual[:, -1, :, 4:6]
        return np.concatenate([disp, mean_vel, final_vel], axis=-1)  # E,N,6

    m1 = predictions["uniform_suffix"]
    train_summ = np.stack([e["factual_summaries"]["train"] for e in m1]).mean(0)
    val_summ = np.stack([e["factual_summaries"]["val"] for e in m1]).mean(0)
    y_train = motion_summary(arrays["object_factual_future"][train_idx])  # E_tr,N,6
    y_val = motion_summary(arrays["object_factual_future"][val_idx])      # E_val,N,6
    train_active = np.stack([e["factual_active"]["train"] for e in m1]).any(0)
    val_active = np.stack([e["factual_active"]["val"] for e in m1]).any(0)  # E_val,N

    X_train = train_summ[train_active]
    Y_train = y_train[train_active]
    if len(X_train) < X_train.shape[1] or val_active.sum() < 2:
        return False, {"reason": "insufficient active agents for probe"}

    n_val, N, hidden = val_summ.shape
    Xv = val_summ.reshape(-1, hidden)

    def fit_predict(Xtr, Ytr):
        beta = ridge_fit(np.column_stack([np.ones(len(Xtr)), Xtr]), Ytr, lam=1e-3)
        pred = ridge_predict(np.column_stack([np.ones(len(Xv)), Xv]), beta)
        return pred.reshape(n_val, N, 6)

    pred_probe = fit_predict(X_train, Y_train)
    rng = np.random.default_rng(stat_seed + 31)
    pred_shuffled = fit_predict(X_train, Y_train[rng.permutation(len(Y_train))])
    pred_scene = np.broadcast_to(Y_train.mean(0), (n_val, N, 6))

    se_probe = ((pred_probe - y_val) ** 2).mean(-1)      # E_val,N
    se_shuffled = ((pred_shuffled - y_val) ** 2).mean(-1)
    se_scene = ((pred_scene - y_val) ** 2).mean(-1)

    def per_episode(se):
        count = val_active.sum(axis=1)
        out = np.full(n_val, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[count > 0] = (se * val_active).sum(axis=1)[count > 0] / count[count > 0]
        return out

    pe_probe, pe_shuffled, pe_scene = per_episode(se_probe), per_episode(se_shuffled), per_episode(se_scene)
    diff_shuffled = pe_probe - pe_shuffled   # negative => probe beats shuffled labels
    diff_scene = pe_probe - pe_scene         # negative => probe beats scene mean
    eligible = np.nonzero(np.isfinite(diff_shuffled) & np.isfinite(diff_scene))[0]
    n_eligible = len(eligible)

    def paired_boot(diff, seed_offset):
        if n_eligible < 2:
            return {"point": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
                    "n_episodes": int(n_eligible)}
        stat = lambda idx: float(np.nanmean(diff[eligible[idx]]))
        return episode_bootstrap(n_eligible, stat, n_resamples, confidence, stat_seed + seed_offset)

    boot_shuffled = paired_boot(diff_shuffled, 32)
    boot_scene = paired_boot(diff_scene, 33)
    probe_mse = float(np.nanmean(pe_probe))
    passed = bool(n_eligible >= 2
                  and np.isfinite(boot_shuffled["ci_upper"]) and boot_shuffled["ci_upper"] < 0
                  and np.isfinite(boot_scene["ci_upper"]) and boot_scene["ci_upper"] < 0)
    return passed, {
        "probe_mse": probe_mse,
        "n_episodes": int(n_eligible),
        "probe_minus_shuffled_label": boot_shuffled,
        "probe_minus_scene_mean": boot_scene,
        "bootstrap_resamples": n_resamples,
    }


# --- orchestration ---------------------------------------------------------------

def preflight_stage1(config: Mapping) -> dict:
    """Read-only contract/signature checks before any output directory is made."""
    validate_stage1_config(config)
    source_path = path(config["source_data"])
    if not source_path.is_file():
        raise FileNotFoundError(f"source data missing: {source_path}")
    source_sha256 = sha256_file(source_path)
    with np.load(source_path, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    summary = validate_source_contract(data)
    return {"source_sha256": source_sha256, "data": data, "contract": summary}


def run_stage1(config: Mapping, *, resume: bool = False, device: str | None = None,
               reanalyze_only: bool = False) -> dict:
    """Train/resume each condition×seed, then compute Gate S/A/B.

    ``reanalyze_only`` reloads the sealed predictions and recomputes the analysis
    with the current code WITHOUT retraining; it requires every run to be
    complete and intentionally does not enforce the code-hash signature (the
    whole point is to re-run a corrected analysis on predictions sealed under
    the training code). It never writes model weights, logs or predictions.
    """
    preflight = preflight_stage1(config)
    data = preflight["data"]
    source_sha256 = preflight["source_sha256"]
    arrays = prepare_stage1_arrays(config, data)
    arrays["_history_full"] = np.asarray(data["object_history"], dtype=np.float32)
    arrays["_indices_train"] = arrays["indices"]["train"]
    indices = arrays["indices"]

    # Frozen validation/test derangements; train is epoch-wise.
    validation_donor, validation_manifest = build_split_derangement(
        arrays, indices["validation"], 1, config["derangement"]["validation_seed"])
    test_donor, test_manifest = build_split_derangement(
        arrays, indices["test"], 2, config["derangement"]["test_seed"])
    validation_manifest_sha = canonical_fingerprint(validation_manifest)
    test_manifest_sha = canonical_fingerprint(test_manifest)

    output = path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    base_seed = int(config["derangement"]["base_seed"])

    def train_donor_for_epoch(epoch: int) -> np.ndarray:
        donor, _ = build_split_derangement(arrays, indices["train"], 0, base_seed + epoch)
        return donor

    # Response thresholds are needed by Gate S3; fit on train split-0 now.
    train_nonego = _nonego_valid_mask(arrays, indices["train"])
    statistics = config["statistics"]
    thresholds = fit_response_bins(
        arrays["r_pos"][indices["train"]][train_nonego],
        float(statistics["zero_response_tolerance_m"]),
        tuple(statistics["positive_response_quantiles"]))
    arrays["_response_thresholds"] = thresholds

    smoke = bool(config.get("smoke_test", False))
    seeds = config["model"]["training_seeds"]
    predictions: dict[str, list] = {c: [] for c in SUFFIX_MODES}
    jobs = []
    for condition in SUFFIX_MODES:
        for seed in seeds:
            stem = f"{condition}_seed{seed}"
            files = {
                "model": output / f"{stem}.pt",
                "log": output / f"{stem}_training.json",
                "val_prediction": output / f"{stem}_val_prediction.npy",
                "test_prediction": output / f"{stem}_test_prediction.npy",
                "metadata": output / f"{stem}_run.json",
            }
            signature = run_signature(config, condition, seed, source_sha256,
                                      validation_manifest_sha, test_manifest_sha)
            jobs.append((condition, seed, stem, files, signature))

    # Validate every existing signature before writing or training anything.
    completed_map = {}
    for condition, seed, stem, files, signature in jobs:
        existing = any(f.exists() for f in files.values())
        completed = None
        if reanalyze_only:
            # Require complete sealed runs; do not enforce the code-hash signature.
            if not files["metadata"].is_file():
                raise ValueError(f"Reanalysis needs a complete run, missing metadata: {stem}")
            metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))
            if metadata.get("summary", {}).get("status") != "complete":
                raise ValueError(f"Reanalysis needs a complete run: {stem}")
            missing = [str(f) for f in files.values() if not f.is_file()]
            if missing:
                raise FileNotFoundError(f"Completed run is missing artifacts: {missing}")
            completed = metadata["summary"]
        elif existing:
            if not resume:
                raise FileExistsError(f"Run artifacts already exist for {stem}; use --resume")
            if not files["metadata"].is_file():
                raise ValueError(f"Run artifacts have no signature to match: {stem}")
            metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))
            if metadata.get("signature") != signature:
                raise ValueError(f"Run metadata does not match current experiment: {stem}")
            if metadata.get("summary", {}).get("status") == "complete":
                missing = [str(f) for f in files.values() if not f.is_file()]
                if missing:
                    raise FileNotFoundError(f"Completed run is missing artifacts: {missing}")
                completed = metadata["summary"]
        completed_map[stem] = completed

    for condition, seed, stem, files, signature in jobs:
        prefix = f"[scene={config['scene']}] [condition={condition}] [seed={seed}]"
        model = PerAgentPairedEffectGRU(
            feature_dim=len(OBJECT_FEATURES), action_dim=5,
            object_embedding_dim=int(config["model"]["object_embedding_dim"]),
            hidden_dim=int(config["model"]["hidden_dim"]),
            action_embedding_dim=int(config["model"]["action_embedding_dim"]))
        parameter_count = sum(p.numel() for p in model.parameters())
        completed = completed_map[stem]
        if completed is not None:
            model.load_state_dict(torch.load(files["model"], map_location="cpu", weights_only=True))
            model = model.to(device)
            val_prediction = np.load(files["val_prediction"], allow_pickle=False)
            test_prediction = np.load(files["test_prediction"], allow_pickle=False)
            log = json.loads(files["log"].read_text(encoding="utf-8"))
            print(f"{prefix} resumed=complete", flush=True)
        else:
            if files["metadata"].exists():
                print(f"{prefix} resumed=incomplete restart_seed=true", flush=True)
            _write_metadata(files["metadata"], signature, {"status": "running", "seed": seed})
            started = time.perf_counter()

            def report_epoch(record):
                epoch, max_epochs = int(record["epoch"]), int(record["max_epochs"])
                every = int(config.get("progress_every_epochs", 5))
                if epoch not in (1, max_epochs) and epoch % every:
                    return
                elapsed = time.perf_counter() - started
                eta = elapsed * (max_epochs - epoch) / epoch
                print(f"{prefix} epoch={epoch}/{max_epochs} train={record['train_mse']:.6f} "
                      f"val={record['val_mse']:.6f} elapsed={format_duration(elapsed)} "
                      f"eta={format_duration(eta)}", flush=True)

            model, val_prediction, test_prediction, log = train_stage1_run(
                model, arrays, indices, condition, train_donor_for_epoch,
                validation_donor, test_donor, config["model"], seed, device,
                progress_callback=report_epoch)
            best = min(log, key=lambda record: record["val_mse"])
            summary = {"status": "complete", "seed": seed, "parameters": parameter_count,
                       "epochs_run": len(log), "best_epoch": best["epoch"],
                       "best_val_mse": best["val_mse"],
                       "runtime_seconds": time.perf_counter() - started}
            torch.save(model.state_dict(), files["model"])
            files["log"].write_text(json.dumps(log, indent=2), encoding="utf-8")
            np.save(files["val_prediction"], val_prediction, allow_pickle=False)
            np.save(files["test_prediction"], test_prediction, allow_pickle=False)
            _write_metadata(files["metadata"], signature, summary)

        # De-normalize predictions to world coordinates; every downstream metric
        # (errors, bins, Gate S/A/B) is defined in world coordinates.
        entry = {"val": _denormalize_effect(val_prediction, arrays),
                 "test": _denormalize_effect(test_prediction, arrays),
                 "log": log, "parameters": parameter_count}
        if condition == "uniform_suffix":
            entry["factual_summaries"], entry["factual_active"] = _capture_factual_summaries(
                model, arrays, indices)
            # Fixed-M1 suffix swap: correct vs frozen donor suffix through ONE checkpoint.
            swap_correct, swap_shuffled = _suffix_swap_inference(
                model, arrays, indices, validation_donor, config, seed, device,
                output, config["scene"], source_sha256, validation_manifest_sha)
            entry["val_suffix_swap_correct"] = swap_correct
            entry["val_suffix_swap_shuffled"] = swap_shuffled
        predictions[condition].append(entry)
        if smoke:
            print(f"{prefix} complete", flush=True)
        else:
            print(f"{prefix} complete (test metrics sealed until final report)", flush=True)

    # Persist shared artifacts.
    np.savez_compressed(output / "normalization.npz", **arrays["normalizer"],
                        target_mean=arrays["target_mean"], target_std=arrays["target_std"],
                        train_indices=indices["train"], validation_indices=indices["validation"],
                        test_indices=indices["test"],
                        feature_names=np.asarray(list(OBJECT_FEATURES)),
                        source_data_sha256=np.asarray(source_sha256))
    (output / "response_bins.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    (output / "validation_manifest.json").write_text(
        json.dumps(validation_manifest, indent=2), encoding="utf-8")
    (output / "test_manifest.json").write_text(
        json.dumps(test_manifest, indent=2), encoding="utf-8")

    if smoke:
        analysis = {"decision": SMOKE_DECISION}
    else:
        analysis = analyze_stage1(arrays, indices, predictions, config,
                                  manifests={"validation": validation_manifest,
                                             "test": test_manifest})

    report = {
        "experiment": config["experiment_name"], "scene": config["scene"],
        "device": str(device), "evaluation_mode": config["evaluation_mode"],
        "smoke_test": smoke,
        "reanalyze_only": bool(reanalyze_only),
        "reanalyze_note": ("Analysis recomputed from sealed predictions with corrected code; "
                           "no retraining performed.") if reanalyze_only else None,
        "protocol_fix_version": "AC-TWIN-RFER-STAGE1-CLOSURE-2026-09-07-V1",
        "training_performed": bool((not reanalyze_only) and (not smoke)),
        "checkpoint_modified": False,
        "analysis_scope": ("sealed_weights_validation_gate_s_development_test_gate_a_b"
                           if reanalyze_only else ("smoke_dataflow_only" if smoke
                                                   else "train_and_evaluate")),
        "sealed_artifact_hash_manifest":
            "experiments/stage1_evidence_value/stage1_sealed_artifact_hashes_before_protocol_fix.json",
        "gate_s_data_scope": "train_validation_only",
        "claim_boundary": ("development_only_not_confirmatory; "
                           "V_evt_R_relationship_is_conditional_heterogeneity_not_causal; "
                           "no_dual_edged_wording_unless_gate_b_supported; "
                           "router_forbidden_until_confirmatory_gate_a"),
        "source_data": str(source_path := path(config["source_data"])),
        "source_data_sha256": source_sha256,
        "validation_manifest_sha256": validation_manifest_sha,
        "test_manifest_sha256": test_manifest_sha,
        "n_train": len(indices["train"]), "n_validation": len(indices["validation"]),
        "n_test": len(indices["test"]),
        "config_signature": canonical_fingerprint(config), "config": config,
        "training_seeds": seeds, "feature_names": list(OBJECT_FEATURES),
        "response_bins": thresholds,
        "qa": {"validation_manifest": validation_manifest["qa"],
               "test_manifest": test_manifest["qa"]},
        "analysis": analysis,
        "decision": SMOKE_DECISION if smoke else analysis.get("outcome"),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _denormalize_effect(pred_normalized: np.ndarray, arrays: Mapping) -> np.ndarray:
    """Map normalized effect predictions back to world coordinates (meters)."""
    return np.asarray(pred_normalized, dtype=np.float64) * arrays["target_std"] + arrays["target_mean"]


def _capture_factual_summaries(model, arrays, indices):
    model.eval()
    device = next(model.parameters()).device
    summaries, actives = {}, {}
    for name in ("train", "validation"):
        idx = indices[name]
        suffix, suffix_mask = condition_suffix(arrays, idx, "uniform_suffix", None)
        with torch.no_grad():
            summary, active = model.encode_factual_agents(
                torch.from_numpy(suffix).float().to(device),
                torch.from_numpy(suffix_mask).to(device))
        key = "train" if name == "train" else "val"
        summaries[key] = summary.cpu().numpy()
        actives[key] = active.cpu().numpy()
    return summaries, actives


def _suffix_swap_inference(model, arrays, indices, validation_donor, config, seed, device,
                           output, scene, source_sha256, validation_manifest_sha):
    """Fixed-M1 suffix-swap inference on validation: correct vs frozen donor suffix.

    Both passes use the SAME M1 checkpoint; only the suffix features and suffix
    mask are swapped to the frozen validation-manifest donor. The correct-suffix
    pass must reproduce the sealed M1 validation prediction within a frozen
    tolerance, otherwise we stop. Returns (correct_world, shuffled_world).
    No optimizer is created, no backward is called, no training log is written.
    """
    model.eval()
    # Use the checkpoint's own device so a CPU-loaded model never meets CUDA inputs.
    device = next(model.parameters()).device
    val_idx = indices["validation"]
    batch_size = int(config["model"]["batch_size"])
    with torch.no_grad():
        # Pass 1: correct episode suffix through M1.
        correct_inputs = _model_inputs(arrays, val_idx, "uniform_suffix", None)
        correct_norm = _predict(model, correct_inputs, batch_size, device)
        # Consistency check against the sealed M1 validation prediction.
        sealed_path = output / f"uniform_suffix_seed{seed}_val_prediction.npy"
        sealed_norm = np.load(sealed_path, allow_pickle=False)
        if correct_norm.shape != sealed_norm.shape:
            raise RuntimeError(f"Suffix-swap shape mismatch for seed {seed}")
        max_abs_diff = float(np.abs(correct_norm - sealed_norm).max())
        matches = bool(np.allclose(correct_norm, sealed_norm, atol=1e-4, rtol=1e-3))
        if not matches:
            raise RuntimeError(
                f"Suffix-swap correct pass does not match sealed M1 validation prediction "
                f"for seed {seed} (max abs diff {max_abs_diff:.3e})")
        # Pass 2: identical H/AF/ACF; suffix features+mask swapped to the frozen donor.
        shuffled_inputs = _model_inputs(arrays, val_idx, "shuffled_suffix", validation_donor)
        shuffled_norm = _predict(model, shuffled_inputs, batch_size, device)
    if not np.isfinite(shuffled_norm).all():
        raise RuntimeError(f"Suffix-swap shuffled prediction is non-finite for seed {seed}")

    stem = f"uniform_suffix_seed{seed}"
    swap_path = output / f"{stem}_val_prediction_with_frozen_shuffled_suffix.npy"
    np.save(swap_path, shuffled_norm, allow_pickle=False)
    normalization_path = output / "normalization.npz"
    normalization_sha = (sha256_file(normalization_path) if normalization_path.is_file()
                         else canonical_fingerprint({k: v.tolist() for k, v in
                                                     arrays["normalizer"].items()}))
    donor = np.asarray(validation_donor)
    metadata = {
        "scene": scene,
        "seed": int(seed),
        "condition": "uniform_suffix",
        "intervention": "fixed_m1_checkpoint_suffix_swap",
        "checkpoint_path": f"{stem}.pt",
        "checkpoint_sha256": sha256_file(output / f"{stem}.pt"),
        "source_data_sha256": source_sha256,
        "normalization_sha256": normalization_sha,
        "validation_manifest_sha256": validation_manifest_sha,
        "implementation": {name: sha256_file(ROOT / "code" / name) for name in CODE_FILES},
        "inference_device": str(device),
        "training_performed": False,
        "checkpoint_modified": False,
        "correct_suffix_matches_sealed_val": matches,
        "correct_suffix_max_abs_diff_vs_sealed": max_abs_diff,
        "output_shape": list(shuffled_norm.shape),
        "output_finite": bool(np.isfinite(shuffled_norm).all()),
        "mask_qa": {
            "self_match": int((donor == np.arange(len(donor))).sum()),
            "n_validation": int(len(donor)),
            "correct_suffix_mask_count": int(correct_inputs["suffix_mask"].sum()),
            "shuffled_suffix_mask_count": int(shuffled_inputs["suffix_mask"].sum()),
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path = output / f"{stem}_val_suffix_swap_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return _denormalize_effect(correct_norm, arrays), _denormalize_effect(shuffled_norm, arrays)


def _write_metadata(file_path: Path, signature: str, summary: dict) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = file_path.with_suffix(file_path.suffix + ".tmp")
    temporary.write_text(json.dumps({"signature": signature, "summary": summary}, indent=2),
                         encoding="utf-8")
    temporary.replace(file_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/stage1_evidence_value_smoke.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed runs whose signatures match exactly")
    parser.add_argument("--reanalyze-only", action="store_true",
                        help="Recompute Gate S/A/B from sealed predictions without retraining")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = json.loads(path(args.config).read_text(encoding="utf-8"))
    report = run_stage1(config, resume=args.resume or args.reanalyze_only, device=args.device,
                        reanalyze_only=args.reanalyze_only)
    print(f"decision={report['decision']} report={path(config['output_dir']) / 'report.json'}",
          flush=True)


if __name__ == "__main__":
    main()
