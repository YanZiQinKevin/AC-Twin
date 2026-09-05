#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from ac_twin.model import OneStepDeltaMLP
from ac_twin.statistics import (
    decision_from_bootstrap,
    paired_bootstrap_mean_difference,
    trajectory_errors,
)
from run_abduction_pilot import train_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run factual-evidence source ablations.")
    parser.add_argument("--config", default="config/suffix_evidence_ablation.json")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def anchor_xy(
    sequence: np.ndarray,
    anchor: np.ndarray,
    obs_shape: tuple[int, int],
) -> np.ndarray:
    n, steps, _ = sequence.shape
    rows, features = obs_shape
    mat = np.asarray(sequence, dtype=np.float32).reshape(n, steps, rows, features).copy()
    present = mat[:, :, :, 0] > 0
    x = mat[:, :, :, 1] - anchor[:, None, None, 0]
    y = mat[:, :, :, 2] - anchor[:, None, None, 1]
    mat[:, :, :, 1] = np.where(present, x, 0.0)
    mat[:, :, :, 2] = np.where(present, y, 0.0)
    return mat.reshape(n, steps, rows * features)


def make_deranged_match(
    split: np.ndarray,
    factual_action_0: np.ndarray,
    cf_action: np.ndarray,
    duration: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = len(split)
    mapping = np.full(n, -1, dtype=np.int64)
    exact_match_count = 0

    broad_keys = sorted(set(zip(split.tolist(), cf_action.tolist(), duration.tolist())))
    for split_value, cf_value, duration_value in broad_keys:
        broad_idx = np.where(
            (split == split_value) & (cf_action == cf_value) & (duration == duration_value)
        )[0]
        remaining: list[int] = []
        for factual_value in sorted(np.unique(factual_action_0[broad_idx]).tolist()):
            subgroup = broad_idx[factual_action_0[broad_idx] == factual_value]
            subgroup = rng.permutation(subgroup)
            if len(subgroup) >= 2:
                shift = int(rng.integers(1, len(subgroup)))
                mapped = np.roll(subgroup, shift)
                mapping[subgroup] = mapped
                exact_match_count += len(subgroup)
            else:
                remaining.extend(subgroup.tolist())

        for idx in remaining:
            candidates = broad_idx[broad_idx != idx]
            if len(candidates) == 0:
                candidates = np.where((split == split_value) & (np.arange(n) != idx))[0]
            if len(candidates) == 0:
                raise RuntimeError("Cannot construct a non-self shuffled suffix mapping.")
            mapping[idx] = int(candidates[int(rng.integers(0, len(candidates)))])

    if np.any(mapping < 0) or np.any(mapping == np.arange(n)):
        raise RuntimeError("Shuffled suffix mapping contains missing or self matches.")
    if not np.all(split[mapping] == split):
        raise RuntimeError("Shuffled suffix mapping crossed dataset splits.")
    stats = {
        "n": int(n),
        "self_matches": int(np.sum(mapping == np.arange(n))),
        "cross_split_matches": int(np.sum(split[mapping] != split)),
        "cf_action_match_rate": float(np.mean(cf_action[mapping] == cf_action)),
        "duration_match_rate": float(np.mean(duration[mapping] == duration)),
        "factual_action_0_match_rate": float(
            np.mean(factual_action_0[mapping] == factual_action_0)
        ),
        "exact_four_field_match_count": int(exact_match_count),
    }
    return mapping, stats


def train_residual_model(
    current_obs: np.ndarray,
    next_obs: np.ndarray,
    actions: np.ndarray,
    split: np.ndarray,
    action_dim: int,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[OneStepDeltaMLP, np.ndarray, list[dict[str, float]]]:
    seed = int(cfg["seed"])
    set_seed(seed)
    obs_dim = current_obs.shape[-1]
    model = OneStepDeltaMLP(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=int(cfg["hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    loss_fn = nn.MSELoss()

    def loader_for(split_value: int, shuffle: bool) -> DataLoader:
        episode_idx = np.where(split == split_value)[0]
        x = current_obs[episode_idx].reshape(-1, obs_dim)
        y = (next_obs[episode_idx] - current_obs[episode_idx]).reshape(-1, obs_dim)
        a = actions[episode_idx].reshape(-1)
        ds = TensorDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(a).long(),
            torch.from_numpy(y).float(),
        )
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            ds,
            batch_size=int(cfg["batch_size"]),
            shuffle=shuffle,
            generator=generator if shuffle else None,
            num_workers=0,
        )

    train_loader = loader_for(0, True)
    val_loader = loader_for(1, False)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    training_log: list[dict[str, float]] = []
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        train_total = 0.0
        train_n = 0
        for obs_b, action_b, delta_b in train_loader:
            obs_b = obs_b.to(device)
            action_b = action_b.to(device)
            delta_b = delta_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(obs_b, action_b)
            loss = loss_fn(pred, delta_b)
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach().cpu()) * len(obs_b)
            train_n += len(obs_b)

        model.eval()
        val_total = 0.0
        val_n = 0
        with torch.no_grad():
            for obs_b, action_b, delta_b in val_loader:
                obs_b = obs_b.to(device)
                action_b = action_b.to(device)
                delta_b = delta_b.to(device)
                loss = loss_fn(model(obs_b, action_b), delta_b)
                val_total += float(loss.detach().cpu()) * len(obs_b)
                val_n += len(obs_b)
        train_loss = train_total / max(train_n, 1)
        val_loss = val_total / max(val_n, 1)
        training_log.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= int(cfg["early_stopping_patience"]):
            break

    if best_state is None:
        raise RuntimeError("Residual model training produced no checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    flat_current = current_obs.reshape(-1, obs_dim)
    flat_actions = actions.reshape(-1)
    pred_parts = []
    with torch.no_grad():
        for start in range(0, len(flat_current), int(cfg["batch_size"])):
            end = min(start + int(cfg["batch_size"]), len(flat_current))
            pred_parts.append(
                model(
                    torch.from_numpy(flat_current[start:end]).float().to(device),
                    torch.from_numpy(flat_actions[start:end]).long().to(device),
                ).cpu().numpy()
            )
    predicted_delta = np.concatenate(pred_parts).reshape(current_obs.shape)
    residual = (next_obs - current_obs) - predicted_delta
    return model, residual.astype(np.float32), training_log


def make_condition_evidence(
    condition: str,
    factual_evidence: np.ndarray,
    residual_evidence: np.ndarray,
    shuffled_mapping: np.ndarray,
) -> tuple[np.ndarray, bool]:
    n, horizon, dim = factual_evidence.shape
    zeros = np.zeros((n, horizon, dim + 1), dtype=np.float32)
    if condition == "history_only":
        return zeros, False

    source = residual_evidence if condition == "residual_innovation_20" else factual_evidence
    visible = np.ones((n, horizon, 1), dtype=np.float32)
    full = np.concatenate([source, visible], axis=-1)
    if condition == "full_20" or condition == "residual_innovation_20":
        return full, True
    if condition == "shuffled_full":
        return full[shuffled_mapping], True
    if condition == "last_frame":
        out = zeros.copy()
        out[:, -1] = full[:, -1]
        return out, True
    if condition.startswith("prefix_"):
        length = int(condition.split("_")[1])
        out = zeros.copy()
        out[:, :length] = full[:, :length]
        return out, True
    raise ValueError(f"Unknown condition: {condition}")


def aggregate_condition_errors(
    predictions: list[np.ndarray], target: np.ndarray
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    ade_by_seed = []
    fde_by_seed = []
    for pred in predictions:
        ade, fde = trajectory_errors(pred, target)
        ade_by_seed.append(ade)
        fde_by_seed.append(fde)
    episode_ade = np.mean(np.stack(ade_by_seed), axis=0)
    episode_fde = np.mean(np.stack(fde_by_seed), axis=0)
    metrics = {
        "effect_ADE_at_20": float(episode_ade.mean()),
        "effect_FDE_at_20": float(episode_fde.mean()),
        "effect_ADE_sd_across_training_seeds": float(
            np.std([x.mean() for x in ade_by_seed], ddof=1)
        ),
        "effect_FDE_sd_across_training_seeds": float(
            np.std([x.mean() for x in fde_by_seed], ddof=1)
        ),
        "n_training_seeds": int(len(predictions)),
    }
    return metrics, episode_ade, episode_fde


def write_results(report: dict[str, Any], path: Path) -> None:
    primary = report["primary_analysis"]
    decision = report["decision"]
    metrics = report["condition_metrics"]
    ordered = report["conditions"]
    reference = report["primary_reference_condition"]
    candidate = report["primary_candidate_condition"]
    lines = [
        "# 事实证据来源消融实验结果",
        "",
        f"运行时间：{report['run_timestamp']}",
        "",
        "## 确认性结论",
        "",
        f"判定：**{decision['verdict']}**。`{candidate}`相对`{reference}`，"
        f"effect-FDE@20改善 {decision['relative_improvement_percent']:.2f}%。",
        "",
        f"episode级配对均值差（{reference} - {candidate}）为 "
        f"{primary['mean_paired_difference_reference_minus_candidate']:.4f} m，"
        f"95% CI [{primary['ci_lower']:.4f}, {primary['ci_upper']:.4f}]，"
        f"n={primary['n']}。",
        "",
        "## 全部条件",
        "",
        "| 条件 | effect-ADE@20 (m) | effect-FDE@20 (m) | seed间FDE SD |",
        "|---|---:|---:|---:|",
    ]
    for condition in ordered:
        m = metrics[condition]
        lines.append(
            f"| {condition} | {m['effect_ADE_at_20']:.4f} | {m['effect_FDE_at_20']:.4f} | "
            f"{m['effect_FDE_sd_across_training_seeds']:.4f} |"
        )
    lines += [
        "",
        "## 解释",
        "",
        f"- `{candidate}`优于`{reference}`才支持同episode事实证据，而不是单纯增加输入长度。",
        "- 所有推断以episode为独立单位；五个训练seed不增加n。",
        "",
        "详细对比置信区间、打乱匹配质量和残差模型信息见 `report.json`。",
    ]
    if "residual_innovation_20" in ordered:
        lines.insert(
            -3,
            "- `residual_innovation_20`是一阶动力学残差代理，不等于真实外生噪声或完成了结构因果溯因。",
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(project_path(args.config).read_text(encoding="utf-8"))
    data_path = project_path(config["source_data"])
    output_dir = project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(data_path, allow_pickle=True)

    history_raw = np.asarray(data["history"], dtype=np.float32)
    factual_raw = np.asarray(data["factual_future"], dtype=np.float32)
    target_raw = np.asarray(data["effect_xy"], dtype=np.float32)
    factual_actions = np.asarray(data["factual_actions"], dtype=np.int64)
    counterfactual_actions = np.asarray(data["counterfactual_actions"], dtype=np.int64)
    split = np.asarray(data["split"], dtype=np.int8)
    cf_action = np.asarray(data["cf_action"], dtype=np.int64)
    duration = np.asarray(data["cf_duration"], dtype=np.int64)
    obs_shape = tuple(np.asarray(data["obs_shape"], dtype=np.int64).tolist())
    if len(obs_shape) != 2 or obs_shape[1] < 3:
        raise RuntimeError(f"Unsupported observation shape: {obs_shape}")

    train_idx = np.where(split == 0)[0]
    val_idx = np.where(split == 1)[0]
    test_idx = np.where(split == 2)[0]
    anchor = history_raw.reshape(*history_raw.shape[:2], *obs_shape)[:, -1, 0, 1:3]
    history_anchored = anchor_xy(history_raw, anchor, obs_shape)
    factual_anchored = anchor_xy(factual_raw, anchor, obs_shape)
    obs_dim = history_anchored.shape[-1]
    train_obs = np.concatenate(
        [history_anchored[train_idx].reshape(-1, obs_dim), factual_anchored[train_idx].reshape(-1, obs_dim)],
        axis=0,
    )
    obs_mean = train_obs.mean(axis=0)
    obs_std = train_obs.std(axis=0)
    obs_std[obs_std < 1e-6] = 1.0
    history_norm = (history_anchored - obs_mean.reshape(1, 1, -1)) / obs_std.reshape(1, 1, -1)
    factual_norm = (factual_anchored - obs_mean.reshape(1, 1, -1)) / obs_std.reshape(1, 1, -1)
    history_with_mask = np.concatenate(
        [history_norm, np.ones((*history_norm.shape[:2], 1), dtype=np.float32)], axis=-1
    ).astype(np.float32)

    current_norm = np.concatenate([history_norm[:, -1:], factual_norm[:, :-1]], axis=1)
    action_dim = int(max(factual_actions.max(), counterfactual_actions.max()) + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    conditions = [str(x) for x in config["conditions"]]
    if "residual_innovation_20" in conditions:
        print(f"[residual] training one-step factual model on device={device}", flush=True)
        residual_model, residual_raw, residual_log = train_residual_model(
            current_norm,
            factual_norm,
            factual_actions,
            split,
            action_dim,
            config["residual_model"],
            device,
        )
        residual_mean = residual_raw[train_idx].mean(axis=(0, 1))
        residual_std = residual_raw[train_idx].std(axis=(0, 1))
        residual_std[residual_std < 1e-6] = 1.0
        residual_norm = (
            residual_raw - residual_mean.reshape(1, 1, -1)
        ) / residual_std.reshape(1, 1, -1)
        torch.save(
            {
                "model_state_dict": residual_model.state_dict(),
                "config": config["residual_model"],
                "obs_mean": obs_mean,
                "obs_std": obs_std,
                "residual_mean": residual_mean,
                "residual_std": residual_std,
            },
            output_dir / "factual_residual_model.pt",
        )
        (output_dir / "training_factual_residual_model.json").write_text(
            json.dumps(residual_log, indent=2), encoding="utf-8"
        )
        residual_metadata = {
            "representation": "one_step_factual_delta_residual_proxy",
            "trained_on": "train_split_factual_transitions_only",
            "epochs_run": int(len(residual_log)),
            "best_val_mse_normalized": float(min(x["val_mse"] for x in residual_log)),
            "not_claimed_as": "identified_exogenous_noise",
        }
    else:
        residual_norm = np.zeros_like(factual_norm, dtype=np.float32)
        residual_metadata = {
            "status": "not_trained",
            "reason": "No configured condition uses residual evidence.",
        }

    shuffle_mapping, shuffle_stats = make_deranged_match(
        split,
        factual_actions[:, 0],
        cf_action,
        duration,
        int(config["shuffle_seed"]),
    )
    np.save(output_dir / "shuffled_suffix_mapping.npy", shuffle_mapping)

    target_mean = target_raw[train_idx].mean(axis=(0, 1))
    target_std = target_raw[train_idx].std(axis=(0, 1))
    target_std[target_std < 1e-6] = 1.0
    target_norm = (target_raw - target_mean.reshape(1, 1, 2)) / target_std.reshape(1, 1, 2)

    effect_cfg = config["effect_model"]
    seeds = [int(x) for x in effect_cfg["training_seeds"]]
    predictions: dict[str, list[np.ndarray]] = {condition: [] for condition in conditions}
    training_runs: dict[str, list[dict[str, Any]]] = {condition: [] for condition in conditions}
    for condition in conditions:
        evidence, use_suffix = make_condition_evidence(
            condition,
            factual_norm,
            residual_norm,
            shuffle_mapping,
        )
        for seed in seeds:
            print(f"[effect] condition={condition} seed={seed}", flush=True)
            model, pred, log = train_one(
                history=history_with_mask,
                factual_suffix=evidence,
                factual_actions=factual_actions,
                counterfactual_actions=counterfactual_actions,
                target=target_norm,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                obs_dim=obs_dim + 1,
                action_dim=action_dim,
                model_cfg=effect_cfg,
                seed=seed,
                use_factual_suffix=use_suffix,
                target_mean=target_mean,
                target_std=target_std,
                device=device,
            )
            predictions[condition].append(pred)
            ade, fde = trajectory_errors(pred, target_raw[test_idx])
            training_runs[condition].append(
                {
                    "seed": seed,
                    "effect_ADE_at_20": float(ade.mean()),
                    "effect_FDE_at_20": float(fde.mean()),
                    "epochs_run": int(len(log)),
                    "best_val_mse_normalized": float(min(x["val_mse"] for x in log)),
                }
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "condition": condition,
                    "seed": seed,
                    "effect_model_config": effect_cfg,
                    "obs_mean": obs_mean,
                    "obs_std": obs_std,
                    "target_mean": target_mean,
                    "target_std": target_std,
                },
                output_dir / f"{condition}_seed{seed}.pt",
            )
            (output_dir / f"training_{condition}_seed{seed}.json").write_text(
                json.dumps(log, indent=2), encoding="utf-8"
            )
            print(
                f"[effect-result] {condition} seed={seed} ADE={ade.mean():.4f} FDE={fde.mean():.4f}",
                flush=True,
            )

    condition_metrics: dict[str, Any] = {}
    episode_errors: dict[str, dict[str, np.ndarray]] = {}
    for condition in conditions:
        metrics, ade, fde = aggregate_condition_errors(
            predictions[condition], target_raw[test_idx]
        )
        condition_metrics[condition] = metrics
        episode_errors[condition] = {"ade": ade, "fde": fde}

    stats_cfg = config["statistics"]
    primary_reference = str(config.get("primary_reference_condition", "shuffled_full"))
    primary_candidate = str(config.get("primary_candidate_condition", "full_20"))
    if primary_reference not in episode_errors or primary_candidate not in episode_errors:
        raise RuntimeError(
            "Primary contrast conditions must both appear in the configured conditions: "
            f"{primary_reference}, {primary_candidate}"
        )
    primary = paired_bootstrap_mean_difference(
        episode_errors[primary_reference]["fde"],
        episode_errors[primary_candidate]["fde"],
        n_resamples=int(stats_cfg["bootstrap_resamples"]),
        confidence_level=float(stats_cfg["confidence_level"]),
        seed=int(config["shuffle_seed"]) + 1,
    )
    decision = decision_from_bootstrap(
        primary,
        minimum_improvement_percent=float(stats_cfg["minimum_practical_improvement_percent"]),
        strong_improvement_percent=float(stats_cfg["strong_improvement_percent"]),
    )

    exploratory_contrasts: dict[str, Any] = {}
    default_exploratory = [
        name for name in conditions if name not in {primary_reference, primary_candidate}
    ]
    exploratory_references = [
        str(x) for x in config.get("exploratory_reference_conditions", default_exploratory)
    ]
    for reference in exploratory_references:
        if reference not in episode_errors:
            raise RuntimeError(f"Unknown exploratory reference condition: {reference}")
        exploratory_contrasts[f"{reference}_minus_{primary_candidate}"] = paired_bootstrap_mean_difference(
            episode_errors[reference]["fde"],
            episode_errors[primary_candidate]["fde"],
            n_resamples=int(stats_cfg["bootstrap_resamples"]),
            confidence_level=float(stats_cfg["confidence_level"]),
            seed=int(config["shuffle_seed"]) + 10 + len(exploratory_contrasts),
        )

    import datetime as dt

    report = {
        "experiment": config["experiment_name"],
        "run_timestamp": dt.datetime.now().astimezone().isoformat(),
        "status": "complete",
        "source_data": str(data_path),
        "device": str(device),
        "conditions": conditions,
        "dataset_counts": {
            "total": int(len(split)),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "independent_unit": "episode_pair",
        "coordinate_transform": config["coordinate_transform"],
        "shuffle_matching": shuffle_stats,
        "residual_model": residual_metadata,
        "condition_metrics": condition_metrics,
        "training_runs": training_runs,
        "primary_contrast": stats_cfg["primary_contrast"],
        "primary_reference_condition": primary_reference,
        "primary_candidate_condition": primary_candidate,
        "primary_analysis": primary,
        "decision": decision,
        "exploratory_contrasts_against_candidate": exploratory_contrasts,
        "limitations": [
            "Each replication run contains one simulator scene family; generalization is assessed only after cross-run synthesis.",
            "Shuffled suffixes are action/duration matched and episode-relative, but cannot match every latent scene property.",
            f"Only the {primary_reference} versus {primary_candidate} contrast is confirmatory; all other contrasts are exploratory.",
        ],
    }
    if "residual_innovation_20" in conditions:
        report["limitations"].append(
            "Residual innovations contain model misspecification and are not identified exogenous variables."
        )
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_results(report, output_dir / "RESULTS.md")
    prediction_payload: dict[str, Any] = {
        "test_indices": test_idx,
        "target_effect": target_raw[test_idx],
        "cf_action": cf_action[test_idx],
        "cf_duration": duration[test_idx],
        "shuffle_mapping": shuffle_mapping,
    }
    for condition in conditions:
        prediction_payload[f"predictions_{condition}"] = np.stack(predictions[condition])
        prediction_payload[f"episode_ade_{condition}"] = episode_errors[condition]["ade"]
        prediction_payload[f"episode_fde_{condition}"] = episode_errors[condition]["fde"]
    np.savez_compressed(output_dir / "test_predictions_and_errors.npz", **prediction_payload)
    print(f"[complete] report={report_path}")
    print(json.dumps({"primary_analysis": primary, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
