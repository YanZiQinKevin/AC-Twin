#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from ac_twin.model import PairedEffectGRU
from ac_twin.statistics import (
    decision_from_bootstrap,
    paired_bootstrap_mean_difference,
    trajectory_errors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the factual-suffix abduction pilot.")
    parser.add_argument("--config", default="config/pilot_2000.json")
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


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "sd_across_training_seeds": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n_training_seeds": int(len(arr)),
    }


def make_loader(
    history: np.ndarray,
    factual_suffix: np.ndarray,
    factual_actions: np.ndarray,
    counterfactual_actions: np.ndarray,
    target: np.ndarray,
    idx: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(history[idx]).float(),
        torch.from_numpy(factual_suffix[idx]).float(),
        torch.from_numpy(factual_actions[idx]).long(),
        torch.from_numpy(counterfactual_actions[idx]).long(),
        torch.from_numpy(target[idx]).float(),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def train_one(
    *,
    history: np.ndarray,
    factual_suffix: np.ndarray,
    factual_actions: np.ndarray,
    counterfactual_actions: np.ndarray,
    target: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    obs_dim: int,
    action_dim: int,
    model_cfg: dict[str, Any],
    seed: int,
    use_factual_suffix: bool,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: torch.device,
) -> tuple[PairedEffectGRU, np.ndarray, list[dict[str, float]]]:
    set_seed(seed)
    model = PairedEffectGRU(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=int(model_cfg["hidden_dim"]),
        action_embedding_dim=int(model_cfg["action_embedding_dim"]),
        use_factual_suffix=use_factual_suffix,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg["learning_rate"]),
        weight_decay=float(model_cfg["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    train_loader = make_loader(
        history,
        factual_suffix,
        factual_actions,
        counterfactual_actions,
        target,
        train_idx,
        int(model_cfg["batch_size"]),
        True,
        seed,
    )
    val_loader = make_loader(
        history,
        factual_suffix,
        factual_actions,
        counterfactual_actions,
        target,
        val_idx,
        int(model_cfg["batch_size"]),
        False,
        seed,
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    history_log: list[dict[str, float]] = []

    for epoch in range(1, int(model_cfg["max_epochs"]) + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for hist_b, fact_b, fa_b, cfa_b, target_b in train_loader:
            hist_b = hist_b.to(device)
            fact_b = fact_b.to(device)
            fa_b = fa_b.to(device)
            cfa_b = cfa_b.to(device)
            target_b = target_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(hist_b, fact_b, fa_b, cfa_b)
            loss = loss_fn(pred, target_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_sum += float(loss.detach().cpu()) * len(hist_b)
            train_count += len(hist_b)

        model.eval()
        val_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for hist_b, fact_b, fa_b, cfa_b, target_b in val_loader:
                hist_b = hist_b.to(device)
                fact_b = fact_b.to(device)
                fa_b = fa_b.to(device)
                cfa_b = cfa_b.to(device)
                target_b = target_b.to(device)
                pred = model(hist_b, fact_b, fa_b, cfa_b)
                loss = loss_fn(pred, target_b)
                val_sum += float(loss.detach().cpu()) * len(hist_b)
                val_count += len(hist_b)

        train_loss = train_sum / max(train_count, 1)
        val_loss = val_sum / max(val_count, 1)
        history_log.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= int(model_cfg["early_stopping_patience"]):
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a model state.")
    model.load_state_dict(best_state)
    model.eval()
    test_loader = make_loader(
        history,
        factual_suffix,
        factual_actions,
        counterfactual_actions,
        target,
        test_idx,
        int(model_cfg["batch_size"]),
        False,
        seed,
    )
    predictions_norm = []
    with torch.no_grad():
        for hist_b, fact_b, fa_b, cfa_b, _ in test_loader:
            pred = model(
                hist_b.to(device),
                fact_b.to(device),
                fa_b.to(device),
                cfa_b.to(device),
            )
            predictions_norm.append(pred.cpu().numpy())
    predictions_norm_arr = np.concatenate(predictions_norm, axis=0)
    predictions = predictions_norm_arr * target_std.reshape(1, 1, 2) + target_mean.reshape(1, 1, 2)
    return model, predictions.astype(np.float32), history_log


def action_pair_mean_baseline(
    target_raw: np.ndarray,
    factual_actions: np.ndarray,
    counterfactual_actions: np.ndarray,
    durations: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    grouped: dict[tuple[int, int, int], list[np.ndarray]] = defaultdict(list)
    for idx in train_idx:
        key = (
            int(factual_actions[idx, 0]),
            int(counterfactual_actions[idx, 0]),
            int(durations[idx]),
        )
        grouped[key].append(target_raw[idx])
    means = {key: np.mean(np.stack(values), axis=0) for key, values in grouped.items()}
    fallback = target_raw[train_idx].mean(axis=0)
    predictions = []
    for idx in test_idx:
        key = (
            int(factual_actions[idx, 0]),
            int(counterfactual_actions[idx, 0]),
            int(durations[idx]),
        )
        predictions.append(means.get(key, fallback))
    return np.stack(predictions).astype(np.float32)


def model_metrics(pred: np.ndarray, target: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    ade, fde = trajectory_errors(pred, target)
    return (
        {
            "effect_ADE_at_20": float(ade.mean()),
            "effect_FDE_at_20": float(fde.mean()),
        },
        ade,
        fde,
    )


def stratified_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    cf_action: np.ndarray,
    durations: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for action in sorted(np.unique(cf_action).tolist()):
        for duration in sorted(np.unique(durations).tolist()):
            mask = (cf_action == action) & (durations == duration)
            if not np.any(mask):
                continue
            metrics, _, _ = model_metrics(pred[mask], target[mask])
            result[f"action_{int(action)}_duration_{int(duration)}"] = {
                "n": int(mask.sum()),
                **metrics,
            }
    return result


def stratified_error_metrics(
    ade: np.ndarray,
    fde: np.ndarray,
    cf_action: np.ndarray,
    durations: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for action in sorted(np.unique(cf_action).tolist()):
        for duration in sorted(np.unique(durations).tolist()):
            mask = (cf_action == action) & (durations == duration)
            if not np.any(mask):
                continue
            result[f"action_{int(action)}_duration_{int(duration)}"] = {
                "n": int(mask.sum()),
                "effect_ADE_at_20": float(np.asarray(ade)[mask].mean()),
                "effect_FDE_at_20": float(np.asarray(fde)[mask].mean()),
            }
    return result


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    primary = report["primary_analysis"]
    decision = report["decision"]
    aggregate = report["aggregate_test_metrics"]
    lines = [
        "# pilot_2000 实验结果",
        "",
        f"运行时间：{report['run_timestamp']}",
        "",
        "## 结论",
        "",
        f"判定：**{decision['verdict']}**。事实后缀相对历史模型的 effect-FDE@20 改善为 "
        f"{decision['relative_improvement_percent']:.2f}%。",
        "",
        f"episode 级配对均值差（history - suffix）为 {primary['mean_paired_difference_reference_minus_candidate']:.4f} m，"
        f"{primary['confidence_level'] * 100:.0f}% CI [{primary['ci_lower']:.4f}, {primary['ci_upper']:.4f}]，"
        f"n={primary['n']} 个独立 test episodes。",
        "",
        "## 汇总结果",
        "",
        "| 模型 | effect-ADE@20 (m) | effect-FDE@20 (m) |",
        "|---|---:|---:|",
    ]
    for name in ["zero_effect", "action_pair_mean", "history_only", "factual_suffix"]:
        metrics = aggregate[name]
        lines.append(
            f"| {name} | {metrics['effect_ADE_at_20']:.4f} | {metrics['effect_FDE_at_20']:.4f} |"
        )
    history_strata = report["stratified_test_metrics"]["history_only"]
    suffix_strata = report["stratified_test_metrics"]["factual_suffix"]
    action_by_id = {int(value): key for key, value in report["action_mapping"].items()}
    improved_strata = 0
    lines += [
        "",
        "## 动作分层",
        "",
        "| 反事实动作 | 持续步数 | n | history FDE | suffix FDE | 改善 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, history_metrics in history_strata.items():
        parts = key.split("_")
        action_id = int(parts[1])
        duration = int(parts[3])
        suffix_metrics = suffix_strata[key]
        history_fde = float(history_metrics["effect_FDE_at_20"])
        suffix_fde = float(suffix_metrics["effect_FDE_at_20"])
        improvement = 100.0 * (history_fde - suffix_fde) / history_fde
        if improvement > 0:
            improved_strata += 1
        lines.append(
            f"| {action_by_id.get(action_id, str(action_id))} | {duration} | {history_metrics['n']} | "
            f"{history_fde:.4f} | {suffix_fde:.4f} | {improvement:.2f}% |"
        )
    lines += [
        "",
        f"事实后缀在 {improved_strata}/{len(history_strata)} 个动作×持续时间子组中降低了终点误差。"
        "该结果不是所有干预类型都一致，分层差异属于下一轮必须解释的失败边界。",
    ]
    lines += [
        "",
        "## 统计口径",
        "",
        "- 独立单位：episode；时间点和训练随机种子均不作为额外 n。",
        "- 主指标：预先固定的 effect-FDE@20。",
        "- 推断：每个 test episode 在五个训练 seed 下分别计算误差，再对误差求平均并做配对 bootstrap。",
        "- 次要指标和分层结果为探索性描述，本轮不作多重显著性声明。",
        "",
        "## 解释边界",
        "",
        "- 若结果通过，只说明事实后缀在本 Highway-env 回溯任务中提供了独立预测信息。",
        "- 不能据此声称恢复了真实道路因果效应，也不能把结果外推为在线规划能力。",
        "- 下一步需要检查不同效应大小、动作类型和第二平台，并实现真正的创新量反演。",
        "",
        "详细数值、五个训练 seed、分层结果和数据完整性信息见 `report.json`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = project_path(config["paths"]["data"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(data_path, allow_pickle=True)
    history_raw = np.asarray(data["history"], dtype=np.float32)
    factual_suffix_raw = np.asarray(data["factual_future"], dtype=np.float32)
    factual_actions = np.asarray(data["factual_actions"], dtype=np.int64)
    counterfactual_actions = np.asarray(data["counterfactual_actions"], dtype=np.int64)
    target_raw = np.asarray(data["effect_xy"], dtype=np.float32)
    split = np.asarray(data["split"], dtype=np.int8)
    durations = np.asarray(data["cf_duration"], dtype=np.int64)
    cf_action = np.asarray(data["cf_action"], dtype=np.int64)
    factual_ego_xy = np.asarray(data["factual_ego_xy"], dtype=np.float32)
    counterfactual_ego_xy = np.asarray(data["counterfactual_ego_xy"], dtype=np.float32)
    collection_stats = json.loads(str(data["collection_stats_json"].item()))

    train_idx = np.where(split == 0)[0]
    val_idx = np.where(split == 1)[0]
    test_idx = np.where(split == 2)[0]
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise RuntimeError("One or more dataset splits are empty.")

    obs_dim = history_raw.shape[-1]
    action_dim = int(max(factual_actions.max(), counterfactual_actions.max()) + 1)
    train_obs = np.concatenate(
        [
            history_raw[train_idx].reshape(-1, obs_dim),
            factual_suffix_raw[train_idx].reshape(-1, obs_dim),
        ],
        axis=0,
    )
    obs_mean = train_obs.mean(axis=0)
    obs_std = train_obs.std(axis=0)
    obs_std[obs_std < 1e-6] = 1.0
    history = (history_raw - obs_mean.reshape(1, 1, -1)) / obs_std.reshape(1, 1, -1)
    factual_suffix = (
        factual_suffix_raw - obs_mean.reshape(1, 1, -1)
    ) / obs_std.reshape(1, 1, -1)

    target_mean = target_raw[train_idx].mean(axis=(0, 1))
    target_std = target_raw[train_idx].std(axis=(0, 1))
    target_std[target_std < 1e-6] = 1.0
    target = (target_raw - target_mean.reshape(1, 1, 2)) / target_std.reshape(1, 1, 2)

    model_cfg = config["model"]
    seeds = [int(seed) for seed in model_cfg["training_seeds"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    predictions: dict[str, list[np.ndarray]] = {"history_only": [], "factual_suffix": []}
    seed_metrics: dict[str, list[dict[str, Any]]] = {"history_only": [], "factual_suffix": []}
    for seed in seeds:
        for model_name, use_suffix in [("history_only", False), ("factual_suffix", True)]:
            print(f"[train] seed={seed} model={model_name} device={device}", flush=True)
            model, pred, training_log = train_one(
                history=history,
                factual_suffix=factual_suffix,
                factual_actions=factual_actions,
                counterfactual_actions=counterfactual_actions,
                target=target,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                obs_dim=obs_dim,
                action_dim=action_dim,
                model_cfg=model_cfg,
                seed=seed,
                use_factual_suffix=use_suffix,
                target_mean=target_mean,
                target_std=target_std,
                device=device,
            )
            metrics, ade, fde = model_metrics(pred, target_raw[test_idx])
            predictions[model_name].append(pred)
            seed_metrics[model_name].append(
                {
                    "seed": seed,
                    **metrics,
                    "best_val_mse_normalized": float(min(x["val_mse"] for x in training_log)),
                    "epochs_run": int(len(training_log)),
                }
            )
            checkpoint_path = output_dir / f"{model_name}_seed{seed}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "seed": seed,
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "model_config": model_cfg,
                    "obs_mean": obs_mean,
                    "obs_std": obs_std,
                    "target_mean": target_mean,
                    "target_std": target_std,
                },
                checkpoint_path,
            )
            (output_dir / f"training_{model_name}_seed{seed}.json").write_text(
                json.dumps(training_log, indent=2), encoding="utf-8"
            )
            print(
                f"[result] seed={seed} model={model_name} "
                f"effect_ADE={metrics['effect_ADE_at_20']:.4f} "
                f"effect_FDE={metrics['effect_FDE_at_20']:.4f}",
                flush=True,
            )

    ensemble_predictions = {
        name: np.mean(np.stack(values, axis=0), axis=0)
        for name, values in predictions.items()
    }
    zero_pred = np.zeros_like(target_raw[test_idx])
    pair_mean_pred = action_pair_mean_baseline(
        target_raw,
        factual_actions,
        counterfactual_actions,
        durations,
        train_idx,
        test_idx,
    )

    aggregate_metrics: dict[str, Any] = {}
    per_episode_errors: dict[str, dict[str, np.ndarray]] = {}
    for name, pred in {
        "zero_effect": zero_pred,
        "action_pair_mean": pair_mean_pred,
    }.items():
        metrics, ade, fde = model_metrics(pred, target_raw[test_idx])
        aggregate_metrics[name] = metrics
        per_episode_errors[name] = {"ade": ade, "fde": fde}

    ensemble_metrics: dict[str, Any] = {}
    for name, model_predictions in predictions.items():
        ade_by_seed = []
        fde_by_seed = []
        for pred in model_predictions:
            ade, fde = trajectory_errors(pred, target_raw[test_idx])
            ade_by_seed.append(ade)
            fde_by_seed.append(fde)
        mean_episode_ade = np.mean(np.stack(ade_by_seed), axis=0)
        mean_episode_fde = np.mean(np.stack(fde_by_seed), axis=0)
        aggregate_metrics[name] = {
            "effect_ADE_at_20": float(mean_episode_ade.mean()),
            "effect_FDE_at_20": float(mean_episode_fde.mean()),
        }
        per_episode_errors[name] = {
            "ade": mean_episode_ade,
            "fde": mean_episode_fde,
        }
        ensemble_metrics[name] = model_metrics(
            ensemble_predictions[name], target_raw[test_idx]
        )[0]

    stats_cfg = config["statistics"]
    primary = paired_bootstrap_mean_difference(
        per_episode_errors["history_only"]["fde"],
        per_episode_errors["factual_suffix"]["fde"],
        n_resamples=int(stats_cfg["bootstrap_resamples"]),
        confidence_level=float(stats_cfg["confidence_level"]),
        seed=int(config["collection_seed"]) + 2,
    )
    decision = decision_from_bootstrap(
        primary,
        minimum_improvement_percent=float(stats_cfg["minimum_practical_improvement_percent"]),
        strong_improvement_percent=float(stats_cfg["strong_improvement_percent"]),
    )

    action_mapping = json.loads(str(data["action_mapping_json"].item()))
    model_seed_summary: dict[str, Any] = {}
    for model_name, entries in seed_metrics.items():
        model_seed_summary[model_name] = {
            "effect_ADE_at_20": summarize([float(x["effect_ADE_at_20"]) for x in entries]),
            "effect_FDE_at_20": summarize([float(x["effect_FDE_at_20"]) for x in entries]),
            "runs": entries,
        }

    import datetime as dt

    report = {
        "experiment": config["experiment_name"],
        "run_timestamp": dt.datetime.now().astimezone().isoformat(),
        "status": "complete",
        "device": str(device),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "dataset_path": str(data_path),
        "dataset_counts": {
            "total": int(len(split)),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "collection_integrity": collection_stats,
        "action_mapping": action_mapping,
        "independent_unit": "episode_pair",
        "primary_endpoint": stats_cfg["primary_endpoint"],
        "aggregate_test_metrics": aggregate_metrics,
        "aggregate_definition": (
            "For learned models, compute each test-episode error for every training seed, "
            "then average errors across seeds. Training seeds are not treated as independent n."
        ),
        "ensemble_prediction_metrics_descriptive_only": ensemble_metrics,
        "training_seed_summary": model_seed_summary,
        "primary_analysis": primary,
        "decision": decision,
        "stratified_test_metrics": {
            name: stratified_error_metrics(
                per_episode_errors[name]["ade"],
                per_episode_errors[name]["fde"],
                cf_action[test_idx],
                durations[test_idx],
            )
            for name in ["history_only", "factual_suffix"]
        },
        "limitations": [
            "Highway-env seed-and-replay is not a universal full simulator snapshot.",
            "The primary target is the ego effect because nearby-agent row identity is not stable across branches.",
            "This retrospective task uses a factual future that is unavailable to online planning.",
            "Secondary endpoints are exploratory and receive no confirmatory multiplicity claim in this pilot.",
        ],
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(report, output_dir / "RESULTS.md")
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        test_indices=test_idx,
        target_effect=target_raw[test_idx],
        history_only_predictions=np.stack(predictions["history_only"]),
        factual_suffix_predictions=np.stack(predictions["factual_suffix"]),
        zero_effect_predictions=zero_pred,
        action_pair_mean_predictions=pair_mean_pred,
        cf_action=cf_action[test_idx],
        cf_duration=durations[test_idx],
        factual_ego_xy=factual_ego_xy[test_idx],
        counterfactual_ego_xy=counterfactual_ego_xy[test_idx],
    )
    print(f"[complete] report={report_path}")
    print(json.dumps({"primary_analysis": primary, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
