"""Stage 2B v0.4 training and Gate-S qualification utilities.

The module is deliberately separate from the collector/protocol layer.  It
may read only the fresh development train/validation roles; confirm data is
never accepted here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ac_twin.stage1_data import (
    apply_feature_normalizer,
    apply_suffix_window,
    compute_anchors,
    compute_history_agent_known,
    compute_target_effect,
    compute_valid_mask,
    fit_feature_normalizer,
    fit_target_normalizer,
    suffix_presence_mask,
    visible_count_signature,
)
from ac_twin.stage1_statistics import (
    assign_response_bin,
    episode_bootstrap,
    fit_response_bins,
    make_spearman_statistic,
    position_errors,
    response_magnitudes,
    ridge_fit,
    ridge_predict,
    seed_mean,
    spearman,
)


GATE_S_PASS = "GATE_S_PASS"
GATE_S_FAIL = "GATE_S_FAIL"
V04_PLAN_ID = "AC-TWIN-RFER-STAGE2B-INTERSECTION-2026-09-15-V0.4"
FORMAL_SEEDS = [11, 22, 33, 44, 55]


def validate_training_config(config: Mapping) -> None:
    required = {"plan_id", "source_data", "source_receipt", "output_dir", "variant",
                "factual_suffix_steps", "training_seeds", "architecture",
                "training", "derangement", "statistics"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"training config missing keys: {sorted(missing)}")
    if config["plan_id"] != V04_PLAN_ID:
        raise ValueError(f"plan_id must be {V04_PLAN_ID}")
    if "stage2b_intersection_development_v04" not in str(config["source_data"]):
        raise ValueError("source_data must be the fresh development_v04 dataset")
    if list(config["training_seeds"]) != FORMAL_SEEDS:
        raise ValueError("formal qualification requires the five seeds [11,22,33,44,55]")
    if config["variant"] not in ("C0", "C1"):
        raise ValueError("variant must be C0 or C1")
    expected = ((64, 64, 8) if config["variant"] == "C0" else (96, 96, 8))
    arch = config["architecture"]
    actual = (int(arch["object_embedding_dim"]), int(arch["hidden_dim"]),
              int(arch["action_embedding_dim"]))
    if actual != expected:
        raise ValueError(f"{config['variant']} architecture must be {expected}")
    training = config["training"]
    frozen = {"batch_size": 128, "max_epochs": 200,
              "early_stopping_patience": 25, "learning_rate": 0.001,
              "weight_decay": 0.0001, "gradient_clip_norm": 5.0}
    for key, value in frozen.items():
        if float(training[key]) != float(value):
            raise ValueError(f"training.{key} must equal {value}")
    if int(config["factual_suffix_steps"]) != 10:
        raise ValueError("factual_suffix_steps must equal 10")
    derangement = config["derangement"]
    if (int(derangement.get("train_base_seed", -1)) != 2026091401
            or int(derangement.get("validation_seed", -1)) != 2026091402):
        raise ValueError("derangement seeds must match the frozen v0.4 contract")
    statistics = config["statistics"]
    if (int(statistics["bootstrap_resamples"]) != 10000
            or float(statistics["confidence_level"]) != 0.95):
        raise ValueError("Gate S requires 10000 bootstrap resamples at 95% confidence")


def four_of_five_metricwise(
        per_seed: Sequence[Mapping[str, float]],
        metrics: Sequence[str] = ("spearman", "mae_vs_basic", "mae_vs_geometry"),
        ) -> bool:
    """True only when every named metric is positive for at least 4/5 seeds."""
    if len(per_seed) != 5:
        return False
    return all(sum(float(row[name]) > 0.0 for row in per_seed) >= 4
               for name in metrics)


def action_duration_design(action: np.ndarray, duration: np.ndarray,
                           *, time_steps: int) -> np.ndarray:
    """Six-stratum one-hot plus normalized horizon, shape E x T x 7."""
    action = np.asarray(action, dtype=np.int64)
    duration = np.asarray(duration, dtype=np.int64)
    if action.shape != duration.shape or action.ndim != 1:
        raise ValueError("action and duration must be aligned one-dimensional arrays")
    if time_steps < 1:
        raise ValueError("time_steps must be positive")
    duration_index = np.where(duration == 1, 0, np.where(duration == 3, 1, -1))
    if ((action < 0) | (action > 2) | (duration_index < 0)).any():
        raise ValueError("expected actions {0,1,2} and durations {1,3}")
    stratum = action * 2 + duration_index
    one_hot = np.eye(6, dtype=np.float64)[stratum]
    one_hot = np.broadcast_to(one_hot[:, None, :],
                              (len(action), int(time_steps), 6)).copy()
    horizon = np.linspace(0.0, 1.0, int(time_steps), dtype=np.float64)
    horizon = np.broadcast_to(horizon[None, :, None],
                              (len(action), int(time_steps), 1))
    return np.concatenate([one_hot, horizon], axis=-1)


def gate_s_status(s1: bool, s2: bool, s3: bool,
                  cross_seed_consistency: bool) -> str:
    return (GATE_S_PASS if all((s1, s2, s3, cross_seed_consistency))
            else GATE_S_FAIL)


def prepare_development_arrays(data: Mapping[str, np.ndarray],
                               *, suffix_steps: int = 10
                               ) -> tuple[dict, dict]:
    """Prepare Stage 2B arrays; every fitted statistic uses split-0 only."""
    history = np.asarray(data["object_history"], dtype=np.float32)
    factual = np.asarray(data["object_factual_future"], dtype=np.float32)
    counterfactual = np.asarray(data["object_counterfactual_future"], dtype=np.float32)
    split = np.asarray(data["split"], dtype=np.int64)
    if not set(np.unique(split).tolist()) <= {0, 1}:
        raise ValueError("Stage 2B development accepts train/validation only")
    indices = {"train": np.flatnonzero(split == 0),
               "validation": np.flatnonzero(split == 1)}
    if not len(indices["train"]) or not len(indices["validation"]):
        raise ValueError("Stage 2B development needs non-empty train and validation")

    anchors = compute_anchors(history)
    normalizer = fit_feature_normalizer(history, factual, indices["train"])
    norm_history = apply_feature_normalizer(history, anchors, normalizer)
    norm_factual = apply_feature_normalizer(factual, anchors, normalizer)
    windowed_factual = apply_suffix_window(norm_factual, int(suffix_steps))
    suffix_mask = suffix_presence_mask(factual, int(suffix_steps))
    target = compute_target_effect(factual, counterfactual)
    valid_mask = compute_valid_mask(history, factual, counterfactual)
    target_mean, target_std = fit_target_normalizer(
        target, valid_mask, indices["train"])
    norm_target = ((target - target_mean) / target_std).astype(np.float32)
    history_known = compute_history_agent_known(history)
    visible_count = visible_count_signature(history_known)
    r_pos, r_vel = response_magnitudes(counterfactual, factual)

    train_mask = valid_mask[indices["train"]].copy()
    train_mask[:, :, 0] = False
    thresholds = fit_response_bins(
        r_pos[indices["train"]][train_mask], 1e-4, (1.0 / 3.0, 2.0 / 3.0))
    arrays = {
        "indices": indices, "split": split,
        "norm_history": norm_history,
        "windowed_factual": windowed_factual,
        "suffix_mask": suffix_mask,
        "target": target, "norm_target": norm_target,
        "target_mean": target_mean, "target_std": target_std,
        "valid_mask": valid_mask,
        "history_known": history_known, "visible_count": visible_count,
        "r_pos": r_pos, "r_vel": r_vel,
        "factual_actions": np.asarray(data["factual_actions"], dtype=np.int64),
        "counterfactual_actions": np.asarray(data["counterfactual_actions"], dtype=np.int64),
        "cf_action": np.asarray(data["cf_action"], dtype=np.int64),
        "cf_duration": np.asarray(data["cf_duration"], dtype=np.int64),
        "episode_seed": np.asarray(data["episode_seed"], dtype=np.int64),
        "object_factual_future": factual,
        "normalizer": normalizer, "anchors": anchors,
        "_history_full": history, "_indices_train": indices["train"],
        "_response_thresholds": thresholds,
    }
    return arrays, thresholds


def _nonego_mask(arrays: Mapping[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    mask = np.asarray(arrays["valid_mask"])[indices].copy()
    mask[:, :, 0] = False
    return mask


def _episode_means(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    sums = (values * mask).sum(axis=(1, 2))
    counts = mask.sum(axis=(1, 2))
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    out[counts > 0] = sums[counts > 0] / counts[counts > 0]
    return out


def _initial_distance(history: np.ndarray) -> np.ndarray:
    presence = history[..., 0] > 0
    ego = history[:, -1, 0, 2:4]
    out = np.full((history.shape[0], history.shape[2]), np.nan, dtype=np.float64)
    for e in range(history.shape[0]):
        for n in range(history.shape[2]):
            seen = np.flatnonzero(presence[e, :, n])
            if seen.size:
                out[e, n] = np.linalg.norm(history[e, seen[-1], n, 2:4] - ego[e])
    return out


def _baseline_design(arrays: Mapping[str, np.ndarray], indices: np.ndarray,
                     *, geometry: bool) -> np.ndarray:
    e = len(indices)
    t, n = arrays["r_pos"].shape[1:]
    base = action_duration_design(arrays["cf_action"][indices],
                                  arrays["cf_duration"][indices],
                                  time_steps=t)
    design = np.broadcast_to(base[:, :, None, :], (e, t, n, base.shape[-1])).copy()
    if geometry:
        distance = _initial_distance(arrays["_history_full"][indices])
        finite = np.isfinite(distance)
        fill = float(np.median(distance[finite])) if finite.any() else 0.0
        distance = np.where(finite, distance, fill)
        distance = np.broadcast_to(distance[:, None, :, None], (e, t, n, 1))
        design = np.concatenate([design, distance], axis=-1)
    return design


def _bootstrap_mean(values: np.ndarray, *, n_resamples: int,
                    confidence: float, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {"point": float("nan"), "ci_lower": float("nan"),
                "ci_upper": float("nan"), "n_episodes": int(len(values))}
    result = episode_bootstrap(
        len(values), lambda idx: float(values[idx].mean()),
        n_resamples, confidence, seed)
    result["n_episodes"] = int(len(values))
    return result


def _vehicle_specific(change: np.ndarray, mask: np.ndarray) -> bool:
    magnitude = np.linalg.norm(change, axis=-1)
    for e in range(magnitude.shape[0]):
        for t in range(magnitude.shape[1]):
            values = magnitude[e, t][mask[e, t]]
            if values.size >= 2 and not np.allclose(values, values[0]):
                return True
    return False


def evaluate_gate_s(arrays: Mapping[str, np.ndarray], predictions: Mapping[str, list],
                    thresholds: Mapping[str, float], *, n_resamples: int = 10000,
                    confidence: float = 0.95, bootstrap_seed: int = 20260915
                    ) -> dict:
    """Evaluate the v0.4 Gate S on development train/validation only."""
    val_idx = arrays["indices"]["validation"]
    train_idx = arrays["indices"]["train"]
    val_mask = _nonego_mask(arrays, val_idx)
    train_mask = _nonego_mask(arrays, train_idx)
    target_val = arrays["target"][val_idx]
    r_val = arrays["r_pos"][val_idx]
    r_train = arrays["r_pos"][train_idx]

    finite_losses = all(
        np.isfinite([row["val_mse"] for row in entry["log"]]).all()
        and np.isfinite(entry["val"]).all()
        for condition in ("no_suffix", "uniform_suffix", "shuffled_suffix")
        for entry in predictions[condition])
    parameter_counts = [int(entry["parameters"])
                        for condition in ("no_suffix", "uniform_suffix", "shuffled_suffix")
                        for entry in predictions[condition]]
    m1_seed = np.stack([entry["val"] for entry in predictions["uniform_suffix"]])
    m1_mean = seed_mean(m1_seed)
    nondegenerate = bool(np.count_nonzero(m1_mean) > 0
                         and not np.allclose(m1_mean, m1_mean.flat[0]))
    s1 = bool(finite_losses and len(set(parameter_counts)) == 1 and nondegenerate)

    basic_train = _baseline_design(arrays, train_idx, geometry=False)
    basic_val = _baseline_design(arrays, val_idx, geometry=False)
    geom_train = _baseline_design(arrays, train_idx, geometry=True)
    geom_val = _baseline_design(arrays, val_idx, geometry=True)

    def fit_baseline(x_train: np.ndarray, x_val: np.ndarray) -> np.ndarray:
        beta = ridge_fit(x_train[train_mask], r_train[train_mask], lam=1e-3)
        pred = ridge_predict(x_val.reshape(-1, x_val.shape[-1]), beta)
        return np.maximum(pred.reshape(r_val.shape), 0.0)

    pred_basic = fit_baseline(basic_train, basic_val)
    pred_geometry = fit_baseline(geom_train, geom_val)
    r_hat_seed = np.linalg.norm(m1_seed, axis=-1)
    r_hat = seed_mean(r_hat_seed)
    spear = episode_bootstrap(
        len(val_idx), make_spearman_statistic(r_hat, r_val, val_mask),
        n_resamples, confidence, bootstrap_seed + 1)
    basic_error = np.abs(pred_basic - r_val)
    geometry_error = np.abs(pred_geometry - r_val)
    model_error = np.abs(r_hat - r_val)
    improve_basic = (_episode_means(basic_error, val_mask)
                     - _episode_means(model_error, val_mask))
    improve_geometry = (_episode_means(geometry_error, val_mask)
                        - _episode_means(model_error, val_mask))
    boot_basic = _bootstrap_mean(improve_basic, n_resamples=n_resamples,
                                 confidence=confidence, seed=bootstrap_seed + 2)
    boot_geometry = _bootstrap_mean(improve_geometry, n_resamples=n_resamples,
                                    confidence=confidence, seed=bootstrap_seed + 3)
    s2_mean = bool(spear["ci_lower"] > 0 and boot_basic["ci_lower"] > 0
                   and boot_geometry["ci_lower"] > 0)
    per_seed_s2 = []
    for seed_pred in r_hat_seed:
        seed_error = np.abs(seed_pred - r_val)
        per_seed_s2.append({
            "spearman": spearman(seed_pred[val_mask], r_val[val_mask]),
            "mae_vs_basic": float(np.nanmean(
                _episode_means(basic_error, val_mask)
                - _episode_means(seed_error, val_mask))),
            "mae_vs_geometry": float(np.nanmean(
                _episode_means(geometry_error, val_mask)
                - _episode_means(seed_error, val_mask))),
        })
    s2_seed_consistency = four_of_five_metricwise(per_seed_s2)
    s2 = bool(s2_mean and s2_seed_consistency)

    bins = assign_response_bin(r_val, thresholds)
    correct_seed = np.stack([
        entry["val_suffix_swap_correct"] for entry in predictions["uniform_suffix"]])
    wrong_seed = np.stack([
        entry["val_suffix_swap_shuffled"] for entry in predictions["uniform_suffix"]])
    correct_mean = seed_mean(correct_seed)
    wrong_mean = seed_mean(wrong_seed)
    diff_mean = (position_errors(wrong_mean, target_val)
                 - position_errors(correct_mean, target_val))
    region_defs = {"zero": 0, "positive_low": 1, "positive_mid": 2}
    region_reports = {}
    eligible_regions = []
    for offset, (name, bin_value) in enumerate(region_defs.items(), start=10):
        region_mask = val_mask & (bins == bin_value)
        boot = _bootstrap_mean(_episode_means(diff_mean, region_mask),
                               n_resamples=n_resamples, confidence=confidence,
                               seed=bootstrap_seed + offset)
        seed_points = []
        for correct, wrong in zip(correct_seed, wrong_seed):
            diff = position_errors(wrong, target_val) - position_errors(correct, target_val)
            seed_points.append(float(np.nanmean(_episode_means(diff, region_mask))))
        positive_seeds = int(sum(np.isfinite(v) and v > 0 for v in seed_points))
        region_reports[name] = {"bootstrap": boot, "per_seed_point": seed_points,
                                "positive_seed_count": positive_seeds,
                                "n_agent_time": int(region_mask.sum())}
        if np.isfinite(boot["ci_lower"]) and boot["ci_lower"] > 0 and positive_seeds >= 4:
            eligible_regions.append(name)
    change = wrong_mean - correct_mean
    nonzero_change = bool(np.any(np.abs(change[val_mask]) > 0))
    vehicle_specific = _vehicle_specific(change, val_mask)
    s3 = bool(nonzero_change and vehicle_specific and eligible_regions)
    cross_seed = bool(s2_seed_consistency and eligible_regions)
    status = gate_s_status(s1, s2, s3, cross_seed)
    return {
        "status": status,
        "data_scope": "development_train_validation_only",
        "S1": {"pass": s1, "finite_losses": finite_losses,
               "nondegenerate": nondegenerate,
               "parameter_counts": parameter_counts,
               "equal_parameters": len(set(parameter_counts)) == 1},
        "S2": {"pass": s2, "spearman": spear,
               "mae_improvement_vs_action_duration": boot_basic,
               "mae_improvement_vs_action_duration_geometry": boot_geometry,
               "per_seed": per_seed_s2,
               "four_of_five_metricwise": s2_seed_consistency},
        "S3": {"pass": s3, "nonzero_prediction_change": nonzero_change,
               "vehicle_specific_change": vehicle_specific,
               "eligible_regions": eligible_regions, "regions": region_reports},
        "cross_seed_consistency": cross_seed,
    }
