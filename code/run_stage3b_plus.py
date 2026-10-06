"""Thin, restart-safe Stage 3B Plus command driver."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import run_stage3b as legacy  # noqa: E402
from ac_twin.stage3b_bridge import router_masked_mse  # noqa: E402
from ac_twin.stage3b_plus_protocol import (  # noqa: E402
    require_crossfit_before_gate_t_plus,
    verify_stage3b_v01_history,
    write_handoff_state,
    write_once_json,
)
from ac_twin.stage3b_plus_routing import (  # noqa: E402
    Stage3BPlusRouter,
    assert_oracle_route_only,
    audit_direct_router_fairness,
    nonmonotonic_shape_loss,
)
from ac_twin.stage3_data import compute_regime_labels  # noqa: E402


CONFIG_PATH = ROOT / "experiments" / "stage3_routing" / "stage3b_plus" / "stage3b_plus_config.json"
LEGACY_CONFIG_PATH = ROOT / "experiments" / "stage3_routing" / "stage3b" / "stage3b_config.json"
LEGACY_RUN_DIR = ROOT / "experiments" / "stage3_routing" / "stage3b"


def _progress(phase: str, completed: int, total: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 0.0)
    fraction = completed / max(total, 1)
    eta = elapsed * (total - completed) / completed if completed else 0.0
    print(
        f"phase={phase} completed={completed}/{total} percent={fraction * 100:.1f}% "
        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
        flush=True,
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def screen_oracle_headroom(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen two-setting, three-seed route-only oracle screen."""
    evaluated: dict[str, dict[str, Any]] = {}
    for architecture, metrics in results.items():
        a_values = [float(value) for value in metrics["a_rel_improvements"]]
        b_values = [float(value) for value in metrics["b_rel_improvements"]]
        if len(a_values) != 3 or len(b_values) != 3:
            raise ValueError("oracle headroom screen requires exactly three registered seeds")
        a_point = sum(a_values) / len(a_values)
        b_point = sum(b_values) / len(b_values)
        passed = (
            a_point >= 0.05
            and b_point >= 0.05
            and sum(value > 0 for value in a_values) >= 2
            and sum(value > 0 for value in b_values) >= 2
        )
        evaluated[architecture] = {
            **metrics,
            "a_point": a_point,
            "b_point": b_point,
            "passed": passed,
        }
    eligible = [name for name, metrics in evaluated.items() if metrics["passed"]]
    if not eligible:
        return {
            "passed": False,
            "status": "NO_ROUTE_ONLY_ORACLE_HEADROOM",
            "architectures": evaluated,
        }
    selected = min(
        eligible,
        key=lambda name: (
            float(evaluated[name]["validation_ade"]),
            0 if name == "state_full" else 1,
        ),
    )
    return {
        "passed": True,
        "status": "ORACLE_HEADROOM_SCREENED",
        "selected_architecture": selected,
        "architectures": evaluated,
    }


def select_shape_lambda(validation_ade: dict[float, float]) -> float:
    """Lowest validation ADE wins; an exact tie deterministically selects zero."""
    if set(validation_ade) != {0.0, 0.05}:
        raise ValueError("shape lambda screen must contain exactly {0.0, 0.05}")
    return min(validation_ade, key=lambda value: (float(validation_ade[value]), float(value)))


def evaluate_learned_dev_screen(contrasts: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Frozen A/B x Uniform/Direct 5% and positive-CI AND screen."""
    expected = {"a_uniform", "a_direct", "b_uniform", "b_direct"}
    if set(contrasts) != expected:
        raise ValueError("learned development screen requires four registered contrasts")
    checks = {
        name: float(value["point"]) >= 0.05 and float(value["ci80_lower"]) > 0.0
        for name, value in contrasts.items()
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "status": "LEARNED_DEV_SCREEN_PASSED" if passed else "LEARNED_ROUTER_DEV_INSUFFICIENT",
        "checks": checks,
        "contrasts": contrasts,
    }


def cmd_preflight(_: argparse.Namespace) -> int:
    started = time.monotonic()
    _progress("preflight", 0, 5, started)
    hashes = verify_stage3b_v01_history(ROOT)
    _progress("preflight", 1, 5, started)
    config = _load_json(CONFIG_PATH)
    if config.get("plan_id") != "AC-TWIN-RFER-STAGE3B-PLUS-2026-09-10-V0.1":
        raise RuntimeError("Stage 3B Plus config plan_id mismatch")
    _progress("preflight", 2, 5, started)
    if not LEGACY_CONFIG_PATH.is_file() or not (LEGACY_RUN_DIR / "input_binding_receipt.json").is_file():
        raise FileNotFoundError("Stage 3B development binding is unavailable")
    _progress("preflight", 3, 5, started)
    free_gb = shutil.disk_usage(ROOT).free / (1024 ** 3)
    if free_gb < 2.0:
        raise RuntimeError(f"insufficient free disk space: {free_gb:.2f} GiB")
    _progress("preflight", 4, 5, started)
    print(
        f"python={sys.version.split()[0]} torch={torch.__version__} "
        f"cuda_available={torch.cuda.is_available()} free_disk_gib={free_gb:.1f} "
        f"history_files={len(hashes)}",
        flush=True,
    )
    _progress("preflight", 5, 5, started)
    print("PREFLIGHT_PASS", flush=True)
    return 0


def _smoke_output_allowed(path: Path) -> bool:
    resolved = path.resolve()
    registered = (ROOT / "experiments" / "stage3_routing" / "stage3b_plus" / "smoke").resolve()
    system_temp = Path(tempfile.gettempdir()).resolve()
    return resolved == registered or system_temp in resolved.parents


def _smoke_tensors(episodes: int, device: str):
    old_config = _load_json(LEGACY_CONFIG_PATH)
    binding = legacy._load_binding(LEGACY_RUN_DIR)
    bundle = legacy._load_dev(old_config)
    keep, _ = legacy._keep_subset(bundle["roles"])
    selected = keep[: int(episodes)]
    if len(selected) != int(episodes):
        raise ValueError("smoke requested more development episodes than available")
    base = legacy._router_base_tensors(bundle, selected)
    features = legacy._pair_feature_tensors(
        bundle, binding, 0, selected, device=device, batch_size=max(1, min(32, episodes))
    )
    tensors = legacy._assemble_router_tensors(base, features)
    tensors["regime_labels"] = bundle["tensors"]["regime_labels"][selected]
    return tensors


def _smoke_forward(model: Stage3BPlusRouter, tensors, condition: str):
    kwargs = {
        "history": tensors["history"],
        "factual_suffix": tensors["factual_suffix"],
        "factual_actions": tensors["factual_actions"],
        "counterfactual_actions": tensors["counterfactual_actions"],
        "response_features": tensors["response_features"],
        "regime_probs": tensors["regime_probs"],
        "return_trace": True,
    }
    if condition == "route_only_oracle":
        kwargs["oracle_labels"] = tensors["regime_labels"]
    return model(**kwargs)


def cmd_smoke(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if not _smoke_output_allowed(output_dir):
        raise ValueError("smoke output must be the registered smoke directory or system temp")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    started = time.monotonic()
    conditions = (
        "uniform_suffix", "same_information_direct", "learned_router", "route_only_oracle"
    )
    _progress("smoke-load-real-data", 0, len(conditions) + 2, started)
    tensors = _smoke_tensors(args.episodes, args.device)
    _progress("smoke-load-real-data", 1, len(conditions) + 2, started)

    outputs: dict[str, float] = {}
    models: dict[str, Stage3BPlusRouter] = {}
    roundtrip = True
    for index, condition in enumerate(conditions):
        checkpoint = checkpoint_dir / f"{condition}.pt"
        torch.manual_seed(1101)
        model = Stage3BPlusRouter(
            condition=condition,
            architecture="state_full",
            feature_dim=int(tensors["history"].shape[-1]),
            response_feature_dim=int(tensors["response_features"].shape[-1]),
            hidden_dim=16,
            object_embedding_dim=16,
            horizon=int(tensors["factual_actions"].shape[1]),
        )
        if args.resume and checkpoint.is_file():
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        else:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(int(args.epochs)):
                optimizer.zero_grad(set_to_none=True)
                prediction, trace = _smoke_forward(model, tensors, condition)
                loss = router_masked_mse(prediction, tensors["effect_target"], tensors["valid"])
                if condition in {"same_information_direct", "learned_router"}:
                    loss = loss + 0.05 * nonmonotonic_shape_loss(
                        trace["route_gate"], tensors["regime_labels"], tensors["valid"], role="train"
                    )
                loss.backward()
                optimizer.step()
            torch.save(model.state_dict(), checkpoint)
        reloaded = Stage3BPlusRouter(
            condition=condition,
            architecture="state_full",
            feature_dim=int(tensors["history"].shape[-1]),
            response_feature_dim=int(tensors["response_features"].shape[-1]),
            hidden_dim=16,
            object_embedding_dim=16,
            horizon=int(tensors["factual_actions"].shape[1]),
        )
        reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        reloaded.eval()
        with torch.no_grad():
            prediction, trace = _smoke_forward(reloaded, tensors, condition)
        if condition == "route_only_oracle":
            assert_oracle_route_only(trace)
        outputs[condition] = float(
            router_masked_mse(prediction, tensors["effect_target"], tensors["valid"]).item()
        )
        models[condition] = reloaded
        _progress("smoke-train-forward", index + 2, len(conditions) + 2, started)
        if args.interrupt_after_checkpoint and not args.resume and index == 0:
            write_handoff_state(
                output_dir / "handoff_state.json",
                {"state": "SMOKE_INTERRUPTED_RECOVERABLE", "next": "rerun smoke --resume"},
            )
            print("SMOKE_INTERRUPTED_RECOVERABLE", flush=True)
            return 75

    direct = models["same_information_direct"]
    learned = models["learned_router"]
    fairness = audit_direct_router_fairness({"direct": direct, "learned": learned})
    crossfit_path = output_dir / "crossfit" / "crossfit_receipt.json"
    write_handoff_state(
        crossfit_path,
        {"status": "CROSSFIT_COMPLETE", "c_direct_a": 0.01, "c_direct_b": 0.01, "smoke": True},
    )
    crossfit = require_crossfit_before_gate_t_plus(output_dir)
    report = {
        "state": "ENGINEERING_SMOKE_ONLY",
        "episodes": int(args.episodes),
        "epochs": int(args.epochs),
        "resumed": bool(args.resume),
        "checkpoint_roundtrip": bool(roundtrip),
        "losses": outputs,
        "fairness": fairness,
        "temporary_gate": {"crossfit_receipt_seen": crossfit["status"] == "CROSSFIT_COMPLETE"},
    }
    write_handoff_state(output_dir / "smoke_report.json", report)
    write_handoff_state(
        output_dir / "handoff_state.json",
        {"state": "ENGINEERING_SMOKE_ONLY", "next": "oracle-headroom"},
    )
    verify_stage3b_v01_history(ROOT)
    _progress("smoke-complete", len(conditions) + 2, len(conditions) + 2, started)
    print("ENGINEERING_SMOKE_ONLY", flush=True)
    return 0


def _zero_response_tensors(bundle, indices: np.ndarray, response_dim: int = 32):
    base = legacy._router_base_tensors(bundle, indices)
    episodes, steps, agents = base["valid"].shape
    base["response_features"] = torch.zeros(episodes, steps, agents, response_dim)
    base["regime_probs"] = torch.zeros(episodes, steps, agents, 4)
    if "regime_labels" in bundle["tensors"]:
        labels = bundle["tensors"]["regime_labels"]
    else:
        labels = torch.as_tensor(
            compute_regime_labels(bundle["tensors"]["r_true"].numpy()), dtype=torch.long
        )
    base["regime_labels"] = labels[indices]
    return base


def _plus_loss(condition: str, shape_lambda: float = 0.0, shape_margin: float = 0.10):
    def loss_fn(model, batch):
        kwargs = {
            "history": batch["history"],
            "factual_suffix": batch["factual_suffix"],
            "factual_actions": batch["factual_actions"],
            "counterfactual_actions": batch["counterfactual_actions"],
            "response_features": batch["response_features"],
            "regime_probs": batch["regime_probs"],
        }
        if condition == "route_only_oracle":
            kwargs["oracle_labels"] = batch["regime_labels"]
        if shape_lambda > 0 and condition in {"same_information_direct", "learned_router"}:
            kwargs["return_trace"] = True
            prediction, trace = model(**kwargs)
            shape = nonmonotonic_shape_loss(
                trace["route_gate"], batch["regime_labels"], batch["valid"],
                margin=shape_margin, role="train",
            )
        else:
            prediction = model(**kwargs)
            shape = prediction.sum() * 0.0
        return router_masked_mse(
            prediction, batch["effect_target"], batch["valid"]
        ) + float(shape_lambda) * shape
    return loss_fn


def _plus_sums(condition: str):
    def sums_fn(model, batch):
        kwargs = {
            "history": batch["history"],
            "factual_suffix": batch["factual_suffix"],
            "factual_actions": batch["factual_actions"],
            "counterfactual_actions": batch["counterfactual_actions"],
            "response_features": batch["response_features"],
            "regime_probs": batch["regime_probs"],
        }
        if condition == "route_only_oracle":
            kwargs["oracle_labels"] = batch["regime_labels"]
        prediction = model(**kwargs)
        return legacy.bridge.router_masked_mse_sums(
            prediction, batch["effect_target"], batch["valid"]
        )
    return sums_fn


def _train_plus_router(
    *, config: dict[str, Any], tensors, roles, condition: str, architecture: str,
    router_seed: int, run_dir: Path, resume: bool, data_signature: str,
    scaler_signature: str, shape_lambda: float = 0.0, shape_margin: float = 0.10,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    lambda_suffix = "" if float(shape_lambda) == 0.0 else f"_ls{shape_lambda:g}".replace(".", "p")
    name = f"{architecture}_{condition}_r{router_seed}{lambda_suffix}"
    final_path = run_dir / f"{name}.pt"
    state_path = run_dir / f"{name}.state.json"
    if state_path.is_file():
        state = _load_json(state_path)
        if state.get("state") == "COMPLETE" and resume:
            legacy.s3t.verify_checkpoint_integrity(
                final_path, expected_signature=state["signature"]
            )
            return state
        if not resume:
            raise FileExistsError(f"{name} exists; use --resume")

    legacy.s3t.seed_all(router_seed)
    model = Stage3BPlusRouter(
        condition=condition,
        architecture=architecture,
        feature_dim=int(tensors["history"].shape[-1]),
        response_feature_dim=int(tensors["response_features"].shape[-1]),
        hidden_dim=int(config["router_hidden_dim"]),
        object_embedding_dim=int(config["router_hidden_dim"]),
        horizon=int(tensors["factual_actions"].shape[1]),
    )
    model_config = {
        "condition": condition,
        "architecture": architecture,
        "hidden_dim": int(config["router_hidden_dim"]),
        "response_feature_dim": int(tensors["response_features"].shape[-1]),
        "horizon": int(tensors["factual_actions"].shape[1]),
        "shape_lambda": float(shape_lambda),
        "shape_margin": float(shape_margin),
    }
    fingerprint = legacy.s3t.compute_run_fingerprint(
        data_signature=data_signature,
        split_name="train",
        config={"model": model_config, "training": {
            key: config[key] for key in (
                "max_epochs", "early_stopping_patience", "batch_size", "learning_rate"
            )
        }},
        scaler_signature=scaler_signature,
        class_weights=np.zeros(4),
        contract_id="AC-TWIN-STAGE3B-PLUS-ORACLE-2026-09-10-V0.1",
    )
    train_batches = legacy.s3a.make_batches(
        tensors, roles["train"], int(config["batch_size"]), seed=router_seed
    )
    val_batches = [
        batch for _, batch in legacy.s3a.make_eval_batches(
            tensors, roles["validation"], int(config["batch_size"])
        )
    ]
    write_handoff_state(state_path, {"state": "IN_PROGRESS", "fingerprint": fingerprint})
    result = legacy.s3t.train_model(
        model,
        _plus_loss(condition, shape_lambda, shape_margin),
        train_batches,
        val_batches,
        seed=router_seed,
        max_epochs=int(config["max_epochs"]),
        early_stopping_patience=int(config["early_stopping_patience"]),
        learning_rate=float(config["learning_rate"]),
        device=str(config["device"]),
        progress_prefix=f"3B+oracle:{name}",
        checkpoint_path=run_dir / f"{name}.train.ckpt",
        resume=resume,
        run_fingerprint=fingerprint,
        val_sums_fn=_plus_sums(condition),
        selection_fn=lambda reg, ce: float(reg),
    )
    record = legacy.s3t.save_predictor_checkpoint(
        final_path,
        result["state_dict"],
        meta={"config": model_config, "router_seed": router_seed, "fingerprint": fingerprint},
    )
    summary = {
        "state": "COMPLETE",
        "name": name,
        "condition": condition,
        "architecture": architecture,
        "router_seed": router_seed,
        "checkpoint": record["path"],
        "signature": record["signature"],
        "best_val_loss": result["best_val_loss"],
        "epochs_run": result["epochs_run"],
        "config": model_config,
    }
    write_handoff_state(state_path, summary)
    return summary


def _load_plus_router(summary: dict[str, Any], device: str) -> Stage3BPlusRouter:
    cfg = summary["config"]
    model = Stage3BPlusRouter(
        condition=cfg["condition"], architecture=cfg["architecture"],
        hidden_dim=int(cfg["hidden_dim"]), object_embedding_dim=int(cfg["hidden_dim"]),
        response_feature_dim=int(cfg["response_feature_dim"]), horizon=int(cfg["horizon"]),
    )
    state = legacy.s3t.load_predictor_state(
        summary["checkpoint"], expected_signature=summary["signature"]
    )
    model.load_state_dict(state)
    return model.to(device).eval()


def _effect_ade_and_gates(model, tensors, condition: str, device: str, batch_size: int):
    predictions, targets, masks, gates, labels = [], [], [], [], []
    indices = np.arange(len(tensors["history"]))
    with torch.no_grad():
        for _, batch in legacy.s3a.make_eval_batches(tensors, indices, batch_size):
            batch = {key: value.to(device) for key, value in batch.items()}
            kwargs = {
                "history": batch["history"], "factual_suffix": batch["factual_suffix"],
                "factual_actions": batch["factual_actions"],
                "counterfactual_actions": batch["counterfactual_actions"],
                "response_features": batch["response_features"],
                "regime_probs": batch["regime_probs"], "return_trace": True,
            }
            if condition == "route_only_oracle":
                kwargs["oracle_labels"] = batch["regime_labels"]
            output, trace = model(**kwargs)
            if condition == "route_only_oracle":
                assert_oracle_route_only(trace)
            predictions.append(output.cpu().numpy())
            targets.append(batch["effect_target"].cpu().numpy())
            masks.append(batch["valid"].cpu().numpy())
            gates.append(trace["route_gate"].cpu().numpy())
            labels.append(batch["regime_labels"].cpu().numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    mask = np.concatenate(masks)
    gate = np.concatenate(gates)
    label = np.concatenate(labels)
    ade = legacy.s3s.effect_ade_per_episode(prediction, target, mask)
    gate_means = {}
    for regime in range(4):
        selected = mask & (label == regime)
        gate_means[str(regime)] = (
            gate[selected].mean(axis=0).tolist() if selected.any() else None
        )
    return ade, gate_means


def _relative_bootstrap(reference: np.ndarray, candidate: np.ndarray, seed: int):
    keep = np.isfinite(reference) & np.isfinite(candidate)
    reference, candidate = reference[keep], candidate[keep]
    point = float((reference.mean() - candidate.mean()) / reference.mean())
    rng = np.random.default_rng(seed)
    values = np.empty(2000)
    for index in range(len(values)):
        selected = rng.integers(0, len(reference), len(reference))
        ref = reference[selected].mean()
        values[index] = (ref - candidate[selected].mean()) / ref
    return {
        "point": point,
        "ci80_lower": float(np.quantile(values, 0.10)),
        "ci80_upper": float(np.quantile(values, 0.90)),
    }


def cmd_oracle_headroom(args: argparse.Namespace) -> int:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    verify_stage3b_v01_history(ROOT)
    config = _load_json(CONFIG_PATH)
    old_config = _load_json(LEGACY_CONFIG_PATH)
    bundle = legacy._load_dev(old_config)
    roles = bundle["roles"]
    all_indices = np.arange(len(bundle["tensors"]["history"]))
    train_tensors = _zero_response_tensors(bundle, all_indices)
    data_signature = bundle["data_signature"]
    scaler_signature = bundle["scaler"].signature()

    a_indices = np.asarray(roles["development_test"])
    a_tensors = _zero_response_tensors(bundle, a_indices)
    b_bundle = legacy._prepare_eval_npz(
        ROOT / old_config["historical_dev_b"], bundle["scaler"]
    )
    b_indices = np.arange(len(b_bundle["tensors"]["history"]))
    b_tensors = _zero_response_tensors(b_bundle, b_indices)

    output_dir = ROOT / "experiments" / "stage3_routing" / "stage3b_plus" / "oracle_headroom"
    checkpoint_dir = output_dir / "checkpoints"
    seeds = tuple(int(value) for value in config["oracle_screen_seeds"])
    results: dict[str, dict[str, Any]] = {}
    total = 2 * 2 * len(seeds)
    completed = 0
    started = time.monotonic()
    for architecture in config["architectures"]:
        per_condition: dict[str, list[dict[str, Any]]] = {
            "uniform_suffix": [], "route_only_oracle": []
        }
        for condition in ("uniform_suffix", "route_only_oracle"):
            for router_seed in seeds:
                summary = _train_plus_router(
                    config=config, tensors=train_tensors, roles=roles,
                    condition=condition, architecture=architecture,
                    router_seed=router_seed, run_dir=checkpoint_dir, resume=args.resume,
                    data_signature=data_signature, scaler_signature=scaler_signature,
                )
                model = _load_plus_router(summary, str(config["device"]))
                a_ade, a_gates = _effect_ade_and_gates(
                    model, a_tensors, condition, str(config["device"]), int(config["batch_size"])
                )
                b_ade, b_gates = _effect_ade_and_gates(
                    model, b_tensors, condition, str(config["device"]), int(config["batch_size"])
                )
                val_tensors = {key: value[roles["validation"]] for key, value in train_tensors.items()}
                val_ade, _ = _effect_ade_and_gates(
                    model, val_tensors, condition, str(config["device"]), int(config["batch_size"])
                )
                per_condition[condition].append({
                    "seed": router_seed, "summary": summary,
                    "a_ade": a_ade, "b_ade": b_ade, "validation_ade": val_ade,
                    "a_gate_means": a_gates, "b_gate_means": b_gates,
                })
                completed += 1
                _progress("oracle-headroom", completed, total, started)

        a_rel, b_rel = [], []
        for uniform, oracle in zip(
            per_condition["uniform_suffix"], per_condition["route_only_oracle"]
        ):
            a_rel.append(float((uniform["a_ade"].mean() - oracle["a_ade"].mean()) / uniform["a_ade"].mean()))
            b_rel.append(float((uniform["b_ade"].mean() - oracle["b_ade"].mean()) / uniform["b_ade"].mean()))
        uniform_a = np.stack([item["a_ade"] for item in per_condition["uniform_suffix"]]).mean(0)
        oracle_a = np.stack([item["a_ade"] for item in per_condition["route_only_oracle"]]).mean(0)
        uniform_b = np.stack([item["b_ade"] for item in per_condition["uniform_suffix"]]).mean(0)
        oracle_b = np.stack([item["b_ade"] for item in per_condition["route_only_oracle"]]).mean(0)
        results[architecture] = {
            "validation_ade": float(np.mean([
                item["validation_ade"].mean() for item in per_condition["route_only_oracle"]
            ])),
            "a_rel_improvements": a_rel,
            "b_rel_improvements": b_rel,
            "a_bootstrap": _relative_bootstrap(uniform_a, oracle_a, 20260910),
            "b_bootstrap": _relative_bootstrap(uniform_b, oracle_b, 20260911),
            "runs": {
                condition: [
                    {
                        "seed": item["seed"], "summary": item["summary"],
                        "a_effect_ade": float(item["a_ade"].mean()),
                        "b_effect_ade": float(item["b_ade"].mean()),
                        "validation_effect_ade": float(item["validation_ade"].mean()),
                        "a_gate_means": item["a_gate_means"],
                        "b_gate_means": item["b_gate_means"],
                    }
                    for item in entries
                ]
                for condition, entries in per_condition.items()
            },
        }
    decision = screen_oracle_headroom(results)
    write_handoff_state(output_dir / "oracle_headroom_report.json", decision)
    state = "ORACLE_HEADROOM_SCREENED" if decision["passed"] else decision["status"]
    write_handoff_state(
        ROOT / "experiments" / "stage3_routing" / "stage3b_plus" / "handoff_state.json",
        {"state": state, "completed_artifact": str(output_dir / "oracle_headroom_report.json"),
         "next": "screen-learned" if decision["passed"] else "stop"},
    )
    print(f"ORACLE_HEADROOM_STATUS={state}", flush=True)
    return 0 if decision["passed"] else 2


def _pair_tensors(bundle, binding, pair_index: int, indices: np.ndarray, config):
    base = legacy._router_base_tensors(bundle, indices)
    features = legacy._pair_feature_tensors(
        bundle, binding, pair_index, indices, str(config["device"]), int(config["batch_size"])
    )
    base.update(features)
    base["regime_labels"] = bundle["tensors"]["regime_labels"][indices]
    return base


def _eval_pair_tensors(eval_bundle, binding, pair_index: int, indices: np.ndarray, config):
    tensors = legacy._eval_router_tensors(
        eval_bundle, binding, pair_index, indices,
        str(config["device"]), int(config["batch_size"]),
    )
    labels = compute_regime_labels(eval_bundle["tensors"]["r_true"].numpy())
    tensors["regime_labels"] = torch.as_tensor(labels[indices], dtype=torch.long)
    return tensors


def _selected_architecture() -> str:
    report = _load_json(
        ROOT / "experiments" / "stage3_routing" / "stage3b_plus" /
        "oracle_headroom" / "oracle_headroom_report.json"
    )
    if not report.get("passed"):
        raise RuntimeError("learned routing is blocked because oracle headroom did not pass")
    return str(report["selected_architecture"])


def cmd_screen_learned(args: argparse.Namespace) -> int:
    verify_stage3b_v01_history(ROOT)
    config = _load_json(CONFIG_PATH)
    old_config = _load_json(LEGACY_CONFIG_PATH)
    architecture = _selected_architecture()
    output_dir = ROOT / "experiments" / "stage3_routing" / "stage3b_plus" / "learned_screen"
    receipt_path = output_dir / "finalist_receipt.json"
    if receipt_path.is_file():
        if not args.resume:
            raise FileExistsError("finalist receipt exists; use --resume to verify/reuse")
        receipt = _load_json(receipt_path)
        print(f"LEARNED_FINALIST_FROZEN lambda={receipt['shape_lambda']}", flush=True)
        return 0

    bundle = legacy._load_dev(old_config)
    binding = legacy._load_binding(LEGACY_RUN_DIR)
    indices = np.arange(len(bundle["tensors"]["history"]))
    tensors = _pair_tensors(bundle, binding, 0, indices, config)
    scores: dict[float, float] = {}
    summaries: dict[str, Any] = {}
    for shape_lambda in (0.0, 0.05):
        summary = _train_plus_router(
            config=config, tensors=tensors, roles=bundle["roles"],
            condition="learned_router", architecture=architecture,
            router_seed=1101, run_dir=output_dir / "candidates", resume=args.resume,
            data_signature=bundle["data_signature"], scaler_signature=bundle["scaler"].signature(),
            shape_lambda=shape_lambda, shape_margin=float(config["shape_margin"]),
        )
        model = _load_plus_router(summary, str(config["device"]))
        validation = {key: value[bundle["roles"]["validation"]] for key, value in tensors.items()}
        ade, gate_means = _effect_ade_and_gates(
            model, validation, "learned_router", str(config["device"]), int(config["batch_size"])
        )
        scores[shape_lambda] = float(np.nanmean(ade))
        summaries[str(shape_lambda)] = {
            "validation_effect_ade": scores[shape_lambda],
            "validation_gate_means": gate_means,
            "training": summary,
        }
    selected_lambda = select_shape_lambda(scores)
    receipt = {
        "status": "LEARNED_FINALIST_FROZEN",
        "architecture": architecture,
        "shape_lambda": selected_lambda,
        "shape_margin": float(config["shape_margin"]),
        "selection_role": "validation",
        "pair_index": 0,
        "candidates": summaries,
        "oracle_report_sha256": legacy.sha256_file(
            ROOT / "experiments" / "stage3_routing" / "stage3b_plus" /
            "oracle_headroom" / "oracle_headroom_report.json"
        ),
    }
    write_once_json(receipt_path, receipt)
    write_handoff_state(
        ROOT / "experiments" / "stage3_routing" / "stage3b_plus" / "handoff_state.json",
        {"state": "LEARNED_FINALIST_FROZEN", "architecture": architecture,
         "shape_lambda": selected_lambda, "next": "fit-five"},
    )
    print(f"LEARNED_FINALIST_FROZEN architecture={architecture} lambda={selected_lambda}", flush=True)
    return 0


def cmd_fit_five(args: argparse.Namespace) -> int:
    verify_stage3b_v01_history(ROOT)
    base_dir = ROOT / "experiments" / "stage3_routing" / "stage3b_plus"
    finalist = _load_json(base_dir / "learned_screen" / "finalist_receipt.json")
    if finalist.get("status") != "LEARNED_FINALIST_FROZEN":
        raise RuntimeError("fit-five requires the frozen learned finalist")
    receipt_path = base_dir / "fit_five" / "fit_five_receipt.json"
    if receipt_path.is_file():
        if not args.resume:
            raise FileExistsError("fit-five receipt exists; use --resume to verify/reuse")
        print("FIT_FIVE_COMPLETE", flush=True)
        return 0

    config = _load_json(CONFIG_PATH)
    old_config = _load_json(LEGACY_CONFIG_PATH)
    bundle = legacy._load_dev(old_config)
    binding = legacy._load_binding(LEGACY_RUN_DIR)
    indices = np.arange(len(bundle["tensors"]["history"]))
    conditions = (
        "uniform_suffix", "same_information_direct", "learned_router", "route_only_oracle"
    )
    summaries: dict[str, dict[str, Any]] = {condition: {} for condition in conditions}
    total = len(conditions) * 5
    completed = 0
    started = time.monotonic()
    for pair_index, router_seed in enumerate(config["router_seeds"]):
        tensors = _pair_tensors(bundle, binding, pair_index, indices, config)
        for condition in conditions:
            shape_lambda = (
                float(finalist["shape_lambda"])
                if condition in {"same_information_direct", "learned_router"}
                else 0.0
            )
            summary = _train_plus_router(
                config=config, tensors=tensors, roles=bundle["roles"],
                condition=condition, architecture=str(finalist["architecture"]),
                router_seed=int(router_seed), run_dir=base_dir / "fit_five" / "checkpoints",
                resume=args.resume, data_signature=bundle["data_signature"],
                scaler_signature=bundle["scaler"].signature(), shape_lambda=shape_lambda,
                shape_margin=float(finalist["shape_margin"]),
            )
            summaries[condition][str(pair_index)] = summary
            completed += 1
            _progress("fit-five", completed, total, started)
    for pair_index in range(5):
        direct = _load_plus_router(summaries["same_information_direct"][str(pair_index)], "cpu")
        learned = _load_plus_router(summaries["learned_router"][str(pair_index)], "cpu")
        audit = audit_direct_router_fairness({"direct": direct, "learned": learned})
        if not audit["within_tolerance"] or not audit["same_state_keys"]:
            raise RuntimeError(f"Direct/Learned fairness audit failed for pair {pair_index}")
    receipt = {
        "status": "FIT_FIVE_COMPLETE",
        "architecture": finalist["architecture"],
        "shape_lambda": finalist["shape_lambda"],
        "seed_pairs": [list(pair) for pair in zip(config["predictor_seeds"], config["router_seeds"])],
        "runs": summaries,
    }
    write_once_json(receipt_path, receipt)
    print("FIT_FIVE_COMPLETE", flush=True)
    return 0


def cmd_evaluate_dev_screen(args: argparse.Namespace) -> int:
    verify_stage3b_v01_history(ROOT)
    base_dir = ROOT / "experiments" / "stage3_routing" / "stage3b_plus"
    report_path = base_dir / "learned_screen" / "dev_screen_report.json"
    if report_path.is_file():
        raise FileExistsError("development screen report already exists and is not rerun")
    fit = _load_json(base_dir / "fit_five" / "fit_five_receipt.json")
    config = _load_json(CONFIG_PATH)
    old_config = _load_json(LEGACY_CONFIG_PATH)
    bundle = legacy._load_dev(old_config)
    binding = legacy._load_binding(LEGACY_RUN_DIR)
    a_indices = np.asarray(bundle["roles"]["development_test"])
    b_bundle = legacy._prepare_eval_npz(ROOT / old_config["historical_dev_b"], bundle["scaler"])
    b_indices = np.arange(len(b_bundle["tensors"]["history"]))
    per_setting: dict[str, dict[str, list[np.ndarray]]] = {
        "a": {condition: [] for condition in fit["runs"]},
        "b": {condition: [] for condition in fit["runs"]},
    }
    for pair_index in range(5):
        a_tensors = _pair_tensors(bundle, binding, pair_index, a_indices, config)
        b_tensors = _eval_pair_tensors(b_bundle, binding, pair_index, b_indices, config)
        for condition, runs in fit["runs"].items():
            model = _load_plus_router(runs[str(pair_index)], str(config["device"]))
            a_ade, _ = _effect_ade_and_gates(
                model, a_tensors, condition, str(config["device"]), int(config["batch_size"])
            )
            b_ade, _ = _effect_ade_and_gates(
                model, b_tensors, condition, str(config["device"]), int(config["batch_size"])
            )
            per_setting["a"][condition].append(a_ade)
            per_setting["b"][condition].append(b_ade)
        _progress("evaluate-dev-screen", pair_index + 1, 5, time.monotonic())

    contrasts: dict[str, dict[str, float]] = {}
    endpoint_means: dict[str, dict[str, float]] = {"a": {}, "b": {}}
    for setting in ("a", "b"):
        ensemble = {
            condition: np.stack(values).mean(axis=0)
            for condition, values in per_setting[setting].items()
        }
        for condition, values in ensemble.items():
            endpoint_means[setting][condition] = float(np.nanmean(values))
        for reference in ("uniform_suffix", "same_information_direct"):
            label = "uniform" if reference == "uniform_suffix" else "direct"
            contrasts[f"{setting}_{label}"] = _relative_bootstrap(
                ensemble[reference], ensemble["learned_router"],
                int(args.bootstrap_seed) + (0 if setting == "a" else 10) + (0 if label == "uniform" else 1),
            )
    decision = evaluate_learned_dev_screen(contrasts)
    report = {**decision, "effect_ade": endpoint_means, "n_resamples": 2000,
              "bootstrap_seed": int(args.bootstrap_seed)}
    write_once_json(report_path, report)
    state = "LEARNED_DEV_SCREEN_PASSED" if decision["passed"] else decision["status"]
    write_handoff_state(
        base_dir / "handoff_state.json",
        {"state": state, "completed_artifact": str(report_path),
         "next": "crossfit" if decision["passed"] else "stop"},
    )
    print(f"LEARNED_DEV_SCREEN_STATUS={state}", flush=True)
    return 0 if decision["passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AC-Twin Stage 3B Plus driver")
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight", help="verify immutable history and runtime inputs")
    preflight.set_defaults(func=cmd_preflight)
    smoke = sub.add_parser("smoke", help="run the real-data engineering smoke chain")
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--episodes", type=int, default=12)
    smoke.add_argument("--epochs", type=int, default=1)
    smoke.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    smoke.add_argument("--resume", action="store_true")
    smoke.add_argument("--interrupt-after-checkpoint", action="store_true", help=argparse.SUPPRESS)
    smoke.set_defaults(func=cmd_smoke)
    oracle = sub.add_parser("oracle-headroom", help="run the bounded route-only oracle screen")
    oracle.add_argument("--resume", action="store_true")
    oracle.set_defaults(func=cmd_oracle_headroom)
    learned = sub.add_parser("screen-learned", help="select the frozen shape-loss lambda on validation")
    learned.add_argument("--resume", action="store_true")
    learned.set_defaults(func=cmd_screen_learned)
    fit = sub.add_parser("fit-five", help="train the frozen four-condition, five-pair matrix")
    fit.add_argument("--resume", action="store_true")
    fit.set_defaults(func=cmd_fit_five)
    evaluate = sub.add_parser("evaluate-dev-screen", help="apply the one-time A/B learned-router screen")
    evaluate.add_argument("--n-resamples", type=int, choices=(2000,), default=2000)
    evaluate.add_argument("--bootstrap-seed", type=int, choices=(20260910,), default=20260910)
    evaluate.set_defaults(func=cmd_evaluate_dev_screen)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"STAGE3B_PLUS_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
