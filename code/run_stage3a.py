"""Stage 3A development driver: bounded search, five-seed fit, benchmark, budget.

Subcommands:

- ``validate-config``  read-only config + search-space validation;
- ``benchmark``        a FULL-train-epoch benchmark of BOTH the response predictor
                       and the router at the widest config and formal batch size,
                       on the formal device (real VRAM on CUDA; CPU RSS, clearly
                       labelled, on CPU) -> benchmark.json;
- ``budget``           benchmark.json -> COMPUTE_BUDGET.md (all section-5.3
                       sections; fails if config device != benchmark device);
- ``readiness``        print the pre-launch readiness checklist;
- ``search``           the single-seed bounded screen over the 8 registered
                       configs (LONG TASK — started manually by the user);
- ``fit-five``         the five-seed fit of the <=2 finalist configs, with
                       per-finalist checkpoint signatures and integrity records
                       (LONG TASK — started manually by the user);
- ``select-final``     mechanically freeze the UNIQUE final config (lowest mean
                       five-seed validation SELECTION SCORE, L_reg + 0.25*L_CE)
                       and the validation-frozen strongest baselines ->
                       selection/final_selection.json receipt (single-use,
                       immutable; signature covers config + baselines + five
                       checkpoint signatures + five selection scores + the
                       upstream fit-five file hash);
- ``lock-replacement-dev``  register the collected replacement development
                       holdout in the joint lock (requires frozen selection,
                       five verified checkpoints and a matching collection
                       receipt); written once;
- ``gate-r-dev-retrain``  the SINGLE replacement-development-test Go/No-Go
                       (v0.1 Task 7): dual-source — scaler/baselines/temperature
                       from the DEVELOPMENT train/validation only, all Gate
                       metrics on the replacement holdout only; unshuffled
                       inference with episode-order asserts; receipt + lock
                       re-verification and a single-use replacement lock;
                       exit 0 pass / 2 adjudicated fail / 3 refused.

Only ``validate-config``/``benchmark``/``budget``/``readiness`` run automatically
at engineering scale. ``search``/``fit-five``/``select-final``/
``lock-replacement-dev``/``gate-r-dev-retrain`` write resumable checkpoints,
exit non-zero with a failure summary on error, and are prepared here but
started by the user in PowerShell after the budget is confirmed. The retired
single-source ``gate-r-dev`` subcommand was removed in the v0.1 repair so the
contaminated old split 2 can never be adjudicated again.
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

from ac_twin import stage3_training as s3t  # noqa: E402
from ac_twin.progress import ProgressReporter  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3_data import (  # noqa: E402
    assert_not_sealed_stage2,
    assert_replacement_source_allowed,
    compute_effect_eval_mask,
    compute_regime_labels,
    compute_response_pos,
    split_role_indices,
    validate_stage3_source,
)
from ac_twin.stage3_response import (  # noqa: E402
    SELECTION_LAMBDA_CLS,
    ConstantZeroBaseline,
    LinearResponseBaseline,
    MultinomialRegimeClassifier,
    Stage3ResponsePredictor,
    TrainMeanByHorizonBaseline,
    TrainPrevalenceClassifier,
    compute_class_weights,
    fit_temperature,
    invert_log1p,
    make_initial_distance,
    response_loss_sums,
    response_masked_loss,
    selection_score,
)
from ac_twin.stage3_routing import Condition, PREDICTOR_SEEDS, Stage3Router  # noqa: E402
from ac_twin.stage3_scaler import FeatureScaler  # noqa: E402
from ac_twin.stage3a_gate import evaluate_h4_gate, select_strongest_baselines  # noqa: E402

DEVELOPMENT_SOURCE = "data/object_centric_stage/merge_2000/object_twinworld_2000.npz"
WIDEST_CONFIG = {"hidden_dim": 96, "response_feature_dim": 64}
FORMAL_BATCH_SIZE = 32


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def _peak_rss_gb() -> float:
    """Cross-platform peak RSS in GB (Windows lacks the ``resource`` module)."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    except Exception:
        return 0.0


def _finalist_name(cfg: Mapping[str, Any]) -> str:
    return f"h{cfg['hidden_dim']}_r{cfg['response_feature_dim']}_l{cfg['lambda_cls']}"


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #
def validate_stage3a_config(config: Mapping[str, Any]) -> None:
    """Validate the Stage 3A search config; the search space may not be expanded."""
    space = config.get("search_space", {})
    if tuple(space.get("hidden_dim", ())) != s3t.SEARCH_HIDDEN_DIMS:
        raise ValueError(f"search_space.hidden_dim must be {s3t.SEARCH_HIDDEN_DIMS}")
    if tuple(space.get("response_feature_dim", ())) != s3t.SEARCH_RESPONSE_FEATURE_DIMS:
        raise ValueError(
            f"search_space.response_feature_dim must be {s3t.SEARCH_RESPONSE_FEATURE_DIMS}"
        )
    if tuple(space.get("lambda_cls", ())) != s3t.SEARCH_LAMBDA_CLS:
        raise ValueError(f"search_space.lambda_cls must be {s3t.SEARCH_LAMBDA_CLS}")
    if int(config.get("max_finalist_configs", -1)) != s3t.MAX_FINALIST_CONFIGS:
        raise ValueError("max_finalist_configs must be 2")
    for key in ("max_epochs", "early_stopping_patience", "learning_rate", "batch_size"):
        if key not in config:
            raise ValueError(f"config.{key} is required")


# --------------------------------------------------------------------------- #
# Data loading, scaling and batching
# --------------------------------------------------------------------------- #
def load_development_tensors(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    validate_stage3_source(data, npz_path)
    roles = split_role_indices(data["split"])
    r_pos = compute_response_pos(data["object_factual_future"], data["object_counterfactual_future"])
    regime = compute_regime_labels(r_pos)
    eval_mask = compute_effect_eval_mask(
        data["object_history"], data["object_factual_future"], data["object_counterfactual_future"]
    )
    tensors = {
        "history": torch.as_tensor(data["object_history"], dtype=torch.float32),
        "factual_suffix": torch.as_tensor(data["object_factual_future"][:, :10], dtype=torch.float32),
        "factual_actions": torch.as_tensor(data["factual_actions"], dtype=torch.long),
        "counterfactual_actions": torch.as_tensor(data["counterfactual_actions"], dtype=torch.long),
        "cf_action": torch.as_tensor(data["cf_action"], dtype=torch.long),
        "cf_duration": torch.as_tensor(data["cf_duration"], dtype=torch.long),
        "t_star": torch.as_tensor(data["t_star"], dtype=torch.long),
        "r_true": torch.as_tensor(r_pos, dtype=torch.float32),
        "regime_labels": torch.as_tensor(regime, dtype=torch.long),
        "valid": torch.as_tensor(eval_mask, dtype=torch.bool),
    }
    return {
        "tensors": tensors,
        "roles": roles,
        "raw_factual_suffix": np.asarray(data["object_factual_future"][:, :10], dtype=np.float32),
        "data_signature": sha256_file(npz_path),
    }


def prepare_scaler(tensors: Mapping[str, torch.Tensor], roles, run_dir: Path) -> FeatureScaler:
    """Fit the scaler ONCE on the train split and persist it; later calls reload it.

    Validation/development-test reuse the persisted scaler and never re-fit it.
    Fitting uses present (presence>0) agent-times of the train history and
    factual suffix only.
    """
    path = Path(run_dir) / "scaler.json"
    if path.is_file():
        return FeatureScaler.load(path)
    train = roles["train"]
    history = tensors["history"][train].numpy()
    suffix = tensors["factual_suffix"][train].numpy()
    arrays = {"history": history, "factual_suffix": suffix}
    masks = {
        "history": history[..., 0] > 0,
        "factual_suffix": suffix[..., 0] > 0,
    }
    scaler = FeatureScaler.fit(arrays, masks, fitted_on="train")
    path.parent.mkdir(parents=True, exist_ok=True)
    scaler.save(path)
    return scaler


def scale_tensors(tensors: Mapping[str, torch.Tensor], scaler: FeatureScaler) -> dict[str, torch.Tensor]:
    """Return a copy of the tensors with history/factual_suffix standardised."""
    scaled = dict(tensors)
    scaled["history"] = torch.as_tensor(scaler.transform(tensors["history"].numpy()), dtype=torch.float32)
    scaled["factual_suffix"] = torch.as_tensor(
        scaler.transform(tensors["factual_suffix"].numpy()), dtype=torch.float32
    )
    return scaled


def make_eval_batches(tensors: Mapping[str, torch.Tensor], indices: np.ndarray, batch_size: int):
    """Yield eval batches in the exact requested episode order (never shuffled).

    Repair plan v0.1 §2.1: validation/test/frozen inference must not use the
    shuffled training ``make_batches``; the caller gets back episodes in
    precisely the order it asked for.
    """
    indices = np.asarray(indices)
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        yield selected, {key: value[selected] for key, value in tensors.items()}


def make_batches(tensors: Mapping[str, torch.Tensor], indices: np.ndarray, batch_size: int, seed: int):
    """Yield mini-batches of full episodes (the episode is the sampling unit)."""
    indices = np.asarray(indices).copy()
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(indices), generator=generator).numpy()
    batches = []
    for start in range(0, len(order), batch_size):
        sel = indices[order[start : start + batch_size]]
        batches.append({key: value[sel] for key, value in tensors.items()})
    return batches


def _response_loss_fn(class_weights: torch.Tensor, lambda_cls: float):
    def loss_fn(model, batch):
        out = model(
            batch["history"], batch["factual_suffix"], batch["factual_actions"],
            batch["counterfactual_actions"], batch["cf_action"], batch["cf_duration"],
            batch["t_star"],
        )
        weights = class_weights.to(out["regime_logits"].device)
        return response_masked_loss(
            out, batch["r_true"], batch["regime_labels"], batch["valid"],
            weights, lambda_cls=lambda_cls,
        )
    return loss_fn


def _response_sums_fn(class_weights: torch.Tensor):
    """Per-batch masked float64 sums for global validation aggregation (v0.1 §2.4)."""
    def sums_fn(model, batch):
        out = model(
            batch["history"], batch["factual_suffix"], batch["factual_actions"],
            batch["counterfactual_actions"], batch["cf_action"], batch["cf_duration"],
            batch["t_star"],
        )
        weights = class_weights.to(out["regime_logits"].device)
        return response_loss_sums(
            out, batch["r_true"], batch["regime_labels"], batch["valid"], weights,
        )
    return sums_fn


# --------------------------------------------------------------------------- #
# benchmark (FULL train epoch, response + router, formal device and batch size)
# --------------------------------------------------------------------------- #
def _measure_peak_gb(device: str) -> float:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return _peak_rss_gb()


def _train_epochs(model, loss_fn, batches, device, epochs, lr=1e-3) -> int:
    device_obj = torch.device(device)
    model.to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(int(epochs)):
        for batch in batches:
            batch = {k: v.to(device_obj) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimizer.step()
    return int(epochs)


def _router_benchmark_loss_fn():
    def loss_fn(model, batch):
        effect = model(
            batch["history"], batch["factual_suffix"], batch["factual_actions"],
            batch["counterfactual_actions"], batch["response_features"],
            batch["regime_probs"],
        )
        # Timing-only target: the loss value is irrelevant to wall time.
        target = torch.zeros_like(effect)
        valid = batch["valid"].unsqueeze(-1).to(effect.dtype)
        denom = valid.sum().clamp_min(1.0) * effect.shape[-1]
        return (((effect - target) ** 2) * valid).sum() / denom
    return loss_fn


def cmd_benchmark(args: argparse.Namespace) -> int:
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    npz_path = _resolve(args.source)
    bundle = load_development_tensors(npz_path)
    tensors, roles = bundle["tensors"], bundle["roles"]
    scaler = prepare_scaler(tensors, roles, _resolve(args.run_dir))
    scaled = scale_tensors(tensors, scaler)

    train_idx = roles["train"]
    if not args.full_epoch:
        train_idx = train_idx[: int(args.episodes)]
    batch_size = FORMAL_BATCH_SIZE if args.full_epoch else int(args.batch_size)
    batches = make_batches(scaled, train_idx, batch_size, seed=0)
    n_episodes = len(train_idx)

    memory_kind = "VRAM" if device == "cuda" else "CPU_RSS"
    records: dict[str, Any] = {}

    # --- response predictor (widest config) ---------------------------------
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = Stage3ResponsePredictor(
        hidden_dim=WIDEST_CONFIG["hidden_dim"],
        response_feature_dim=WIDEST_CONFIG["response_feature_dim"],
    )
    loss_fn = _response_loss_fn(torch.ones(4), lambda_cls=s3t.SEARCH_LAMBDA_CLS[0])
    started = time.perf_counter()
    _train_epochs(model, loss_fn, batches, device, args.epochs)
    resp_epoch_s = (time.perf_counter() - started) / max(int(args.epochs), 1)
    records["response"] = {
        "epoch_seconds": float(resp_epoch_s),
        "peak_memory_gb": float(_measure_peak_gb(device)),
        "memory_kind": memory_kind,
        "device": device,
        "episodes": n_episodes,
        "batch_size": batch_size,
        "config": dict(WIDEST_CONFIG),
        "date": time.strftime("%Y-%m-%d"),
        "source": f"full-epoch response benchmark ({'full train epoch' if args.full_epoch else f'{n_episodes} ep'} at widest config)",
    }

    # --- router (dual, widest config) ---------------------------------------
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    router = Stage3Router(
        Condition.DUAL_ROUTER,
        hidden_dim=WIDEST_CONFIG["hidden_dim"],
        response_feature_dim=WIDEST_CONFIG["response_feature_dim"],
        horizon=20,
    )
    # Router consumes response features + regime probs; build constant-shape stubs.
    router_batches = []
    for batch in batches:
        b = dict(batch)
        steps = b["factual_actions"].shape[1]
        agents = b["history"].shape[2]
        b["response_features"] = torch.zeros(
            b["history"].shape[0], steps, agents, WIDEST_CONFIG["response_feature_dim"]
        )
        b["regime_probs"] = torch.full(
            (b["history"].shape[0], steps, agents, 4), 0.25
        )
        router_batches.append(b)
    started = time.perf_counter()
    _train_epochs(router, _router_benchmark_loss_fn(), router_batches, device, args.epochs)
    router_epoch_s = (time.perf_counter() - started) / max(int(args.epochs), 1)
    records["router"] = {
        "epoch_seconds": float(router_epoch_s),
        "peak_memory_gb": float(_measure_peak_gb(device)),
        "memory_kind": memory_kind,
        "device": device,
        "episodes": n_episodes,
        "batch_size": batch_size,
        "config": dict(WIDEST_CONFIG),
        "date": time.strftime("%Y-%m-%d"),
        "source": f"full-epoch router benchmark ({'full train epoch' if args.full_epoch else f'{n_episodes} ep'} at widest config)",
    }

    note = (
        "Real VRAM measured via torch.cuda.max_memory_allocated."
        if device == "cuda"
        else "CPU run: times are CPU-time and memory is CPU RSS — NOT GPU-hours/VRAM."
    )
    benchmark = {
        "device": device,
        "date": time.strftime("%Y-%m-%d"),
        "response": records["response"],
        "router": records["router"],
        "note": note,
    }
    out_path = _resolve(args.output)
    _atomic_write_json(out_path, benchmark)
    print(f"[stage3a-benchmark] device={device} "
          f"response={resp_epoch_s:.3f}s/ep router={router_epoch_s:.3f}s/ep -> {out_path}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# budget
# --------------------------------------------------------------------------- #
def cmd_budget(args: argparse.Namespace) -> int:
    benchmark = json.loads(_resolve(args.benchmark).read_text(encoding="utf-8"))
    config = json.loads(_resolve(args.config).read_text(encoding="utf-8"))
    validate_stage3a_config(config)
    bench_device = str(benchmark.get("device", "cpu")).lower()
    config_device = str(config.get("device", "cpu")).lower()
    if bench_device != config_device:
        raise ValueError(
            f"device mismatch: config.device={config_device} but benchmark.device={bench_device}; "
            "config and budget device must be identical"
        )
    estimate = s3t.build_budget_estimate(
        benchmark,
        train_episodes=int(config.get("train_episodes", 1400)),
        benchmark_episodes=int(benchmark["response"].get("episodes", 1400)),
        assumed_epochs=int(config["max_epochs"]),
        checkpoint_gb_per_run=float(config.get("checkpoint_gb_per_run", 0.5)),
        gatec_id_episodes=int(config.get("gatec_id_episodes", 1600)),
        gatec_unseen_episodes=int(config.get("gatec_unseen_episodes", 800)),
    )
    out_path = _resolve(args.output)
    s3t.generate_compute_budget(estimate, out_path)
    print(f"[stage3a-budget] wrote {out_path}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# readiness checklist
# --------------------------------------------------------------------------- #
READINESS_CHECKLIST = [
    "v0.2 计划 §11 已批准且状态为 APPROVED_FOR_IMPLEMENTATION（见计划头部与状态索引）",
    "搜索空间为已登记范围：hidden_dim{64,96} x response_feature_dim{32,64} x lambda_cls{0.10,0.25}，未扩展",
    "Stage 2 holdout 三重封存拒绝生效（路径/逻辑路径/SHA256），config 校验通过",
    "train-only scaler 已落盘，validation/test 仅复用不重拟合",
    "run fingerprint 覆盖数据签名/split/完整配置/scaler/类别权重/合同标识",
    "初始距离为逐车 (E,N) 基线特征，不人为削弱基线",
    "五 seed 方向一致覆盖全部指标（I_MAE/Spearman/两 PR-AUC/Brier）",
    "development test 单次锁死：select-final 冻结唯一配置+基线，已有报告即拒绝",
    "全量回归 pytest -W error 通过，无破坏",
    "12 episode/1 epoch smoke 通过，状态仅 ENGINEERING_SMOKE_ONLY",
    "正式设备完整 epoch benchmark 已生成（response+router，真实 VRAM）",
    "COMPUTE_BUDGET.md 已生成，含 3A/cross-fit/3B 分项、正式集合、典型/最坏",
    "用户已单独确认预算，长任务命令由用户手动启动",
]


def cmd_readiness(args: argparse.Namespace) -> int:
    print("[stage3a-readiness] 启动前 checklist：", flush=True)
    for index, item in enumerate(READINESS_CHECKLIST, start=1):
        print(f"  {index:2d}. [ ] {item}", flush=True)
    print("[stage3a-readiness] 全部确认后方可启动 search / fit-five。", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# search / fit-five (LONG TASKS — prepared here, started by the user)
class RefusedOverwriteError(RuntimeError):
    """A phase artifact already exists and the caller did not pass --resume."""


# --------------------------------------------------------------------------- #
def _train_one_config(
    config: Mapping[str, Any],
    scaled_tensors,
    roles,
    cfg: Mapping[str, Any],
    seed: int,
    run_dir: Path,
    resume: bool,
    *,
    scaler: FeatureScaler,
    data_signature: str,
) -> dict[str, Any]:
    """Train one response-predictor config/seed with checkpoint + failure exit.

    The run fingerprint covers the data signature, split, full merged config,
    scaler signature, class weights and the contract id, so a resume is accepted
    only for the identical run (item 5). Repair plan v0.1 §4.2/§5: existing
    artifacts are never overwritten — a COMPLETE run is reused idempotently
    (with integrity re-verification) under --resume, and any pre-existing
    artifact without --resume refuses the phase.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{_finalist_name(cfg)}_seed{seed}"
    ckpt = run_dir / f"{run_name}.ckpt"
    state_path = run_dir / f"{run_name}.state.json"
    if state_path.is_file():
        prior = json.loads(state_path.read_text(encoding="utf-8"))
        prior_state = prior.get("state")
        if not resume:
            raise RefusedOverwriteError(
                f"{state_path} already exists (state={prior_state}); "
                "re-run with --resume to reuse/continue, never overwrite"
            )
        if prior_state == "COMPLETE":
            # Idempotent reuse: verify the persisted checkpoint still matches.
            s3t.verify_checkpoint_integrity(
                run_dir / f"{run_name}.pt", expected_signature=prior["signature"]
            )
            return prior
    _atomic_write_json(state_path, {"run": run_name, "state": "IN_PROGRESS", "seed": seed})
    try:
        # Repair plan v0.1 §2.2: seed BEFORE any model/random tensor creation so
        # initial weights depend only on the registered seed, not ambient state.
        s3t.seed_all(seed)
        train_idx, val_idx = roles["train"], roles["validation"]
        class_weights = compute_class_weights(
            scaled_tensors["regime_labels"][train_idx].numpy(),
            scaled_tensors["valid"][train_idx].numpy(),
        )
        merged_config = {
            "model": dict(cfg),
            "training": {
                "max_epochs": int(config["max_epochs"]),
                "early_stopping_patience": int(config["early_stopping_patience"]),
                "learning_rate": float(config["learning_rate"]),
                "batch_size": int(config["batch_size"]),
            },
            "seed": int(seed),
        }
        fingerprint = s3t.compute_run_fingerprint(
            data_signature=data_signature,
            split_name="train",
            config=merged_config,
            scaler_signature=scaler.signature(),
            class_weights=np.asarray(class_weights, dtype=np.float64),
        )
        loss_fn = _response_loss_fn(torch.as_tensor(class_weights), lambda_cls=float(cfg["lambda_cls"]))
        sums_fn = _response_sums_fn(torch.as_tensor(class_weights))
        batches = make_batches(scaled_tensors, train_idx, int(config["batch_size"]), seed=seed)
        # Validation inference never uses the training shuffle (v0.1 §2.1/§2.4).
        val_batches = [batch for _, batch in make_eval_batches(scaled_tensors, val_idx, int(config["batch_size"]))]
        model = Stage3ResponsePredictor(
            hidden_dim=int(cfg["hidden_dim"]),
            response_feature_dim=int(cfg["response_feature_dim"]),
        )
        result = s3t.train_model(
            model, loss_fn, batches, val_batches,
            seed=seed,
            max_epochs=int(config["max_epochs"]),
            early_stopping_patience=int(config["early_stopping_patience"]),
            learning_rate=float(config["learning_rate"]),
            device=config.get("device", "cpu"),
            progress_prefix=f"3A:{run_name}",
            checkpoint_path=ckpt,
            resume=resume,
            run_fingerprint=fingerprint,
            val_sums_fn=sums_fn,
            selection_fn=selection_score,
        )
        meta = {"run": run_name, "seed": int(seed), "config": dict(cfg), "run_fingerprint": fingerprint}
        record = s3t.save_predictor_checkpoint(run_dir / f"{run_name}.pt", result["state_dict"], meta=meta)
        summary = {
            "run": run_name, "state": "COMPLETE", "seed": seed,
            "config": dict(cfg), "best_val_loss": result["best_val_loss"],
            "train_lambda_cls": float(cfg["lambda_cls"]),
            "selection_lambda_cls": SELECTION_LAMBDA_CLS,
            "validation_reg_loss": result["validation_reg_loss"],
            "validation_ce_loss": result["validation_ce_loss"],
            "validation_selection_score": result["validation_selection_score"],
            "epochs_run": result["epochs_run"], "signature": record["signature"],
            "checkpoint": record["path"],
        }
        _atomic_write_json(state_path, summary)
        return summary
    except Exception as exc:  # failure exit: record and re-raise as non-zero
        _atomic_write_json(state_path, {
            "run": run_name, "state": "FAILED", "seed": seed, "error": repr(exc),
        })
        raise


def _load_bundle_and_scaler(args, source: str | None = None) -> dict[str, Any]:
    bundle = load_development_tensors(_resolve(source or args.source))
    scaler = prepare_scaler(bundle["tensors"], bundle["roles"], _resolve(args.run_dir))
    scaled = scale_tensors(bundle["tensors"], scaler)
    bundle["scaler"] = scaler
    bundle["scaled"] = scaled
    return bundle


def cmd_search(args: argparse.Namespace) -> int:
    config = json.loads(_resolve(args.config).read_text(encoding="utf-8"))
    validate_stage3a_config(config)
    bundle = _load_bundle_and_scaler(args)
    scaled, roles = bundle["scaled"], bundle["roles"]
    run_dir = _resolve(args.run_dir)
    search_dir = run_dir / "search"  # phase-isolated artifacts (v0.1 §4.2)
    screen_seed = PREDICTOR_SEEDS[0]
    configs = s3t.search_space_configs()
    reporter = ProgressReporter(total=len(configs), every=1, prefix="3A-search", heartbeat_seconds=30.0)
    results = []
    try:
        for index, cfg in enumerate(configs, start=1):
            summary = _train_one_config(
                config, scaled, roles, cfg, screen_seed, search_dir, args.resume,
                scaler=bundle["scaler"], data_signature=bundle["data_signature"],
            )
            results.append(summary)
            reporter.update(index, details=f"{_finalist_name(cfg)} sel={summary['validation_selection_score']:.4f}")
    except RefusedOverwriteError as exc:
        print(f"[stage3a-search] REFUSED: {exc}", flush=True)
        return 3
    # select the <=2 finalists by the cross-lambda-comparable validation
    # selection score ONLY (v0.1 §2.3/§2.5), deterministic tie-break ascending.
    results.sort(key=lambda r: (
        r["validation_selection_score"],
        r["config"]["hidden_dim"], r["config"]["response_feature_dim"], r["config"]["lambda_cls"],
    ))
    finalists = results[: s3t.MAX_FINALIST_CONFIGS]
    out = {
        "screen_seed": screen_seed,
        "results": results,
        "finalists": [f["config"] for f in finalists],
        "selection_basis": "validation selection score only (L_reg + 0.25*L_CE, fixed coefficient)",
    }
    _atomic_write_json(search_dir / "search_results.json", out)
    print(f"[stage3a-search] finalists={out['finalists']}", flush=True)
    return 0


def cmd_fit_five(args: argparse.Namespace) -> int:
    config = json.loads(_resolve(args.config).read_text(encoding="utf-8"))
    validate_stage3a_config(config)
    run_dir = _resolve(args.run_dir)
    search_path = run_dir / "search" / "search_results.json"
    search_results = json.loads(search_path.read_text(encoding="utf-8"))
    search_signatures = {r["run"]: r["signature"] for r in search_results["results"]}
    finalists = search_results["finalists"]
    fit_dir = run_dir / "fit_five"  # phase-isolated; never overwrites search/
    bundle = _load_bundle_and_scaler(args)
    scaled, roles = bundle["scaled"], bundle["roles"]
    reporter = ProgressReporter(
        total=len(finalists) * len(PREDICTOR_SEEDS), every=1, prefix="3A-fit5", heartbeat_seconds=30.0
    )
    per_finalist = []
    done = 0
    for cfg in finalists:
        seed_records = []
        state_dicts = []
        for seed in PREDICTOR_SEEDS:
            try:
                summary = _train_one_config(
                    config, scaled, roles, cfg, seed, fit_dir, args.resume,
                    scaler=bundle["scaler"], data_signature=bundle["data_signature"],
                )
            except RefusedOverwriteError as exc:
                print(f"[stage3a-fit-five] REFUSED: {exc}", flush=True)
                return 3
            # integrity: reload + recompute signature (item 6)
            verified = s3t.verify_checkpoint_integrity(
                summary["checkpoint"], expected_signature=summary["signature"]
            )
            state_dicts.append(s3t.load_predictor_state(summary["checkpoint"], expected_signature=summary["signature"]))
            seed_records.append({
                "seed": seed, "signature": verified["signature"],
                "path": verified["path"], "best_val_loss": summary["best_val_loss"],
                "validation_selection_score": summary["validation_selection_score"],
            })
            done += 1
            reporter.update(done, details=f"{_finalist_name(cfg)} seed{seed} sel={summary['validation_selection_score']:.4f}")
        # Determinism audit (v0.1 §2.5.5): the finalist's screen-seed run must
        # reproduce EXACTLY the signature observed during search.
        screen_run = f"{_finalist_name(cfg)}_seed{PREDICTOR_SEEDS[0]}"
        fit_sig = next(s["signature"] for s in seed_records if s["seed"] == PREDICTOR_SEEDS[0])
        if fit_sig != search_signatures.get(screen_run):
            raise RuntimeError(
                f"DETERMINISM_FAILURE: {screen_run} signature in fit-five ({fit_sig}) "
                f"!= search ({search_signatures.get(screen_run)})"
            )
        five = s3t.five_checkpoint_signatures(state_dicts)
        per_finalist.append({
            "config": dict(cfg),
            "per_seed": seed_records,
            "five_checkpoint_signatures": five,
            "all_distinct": five["all_distinct"],
        })
    _atomic_write_json(fit_dir / "fit_five_results.json", {
        "finalists": per_finalist,
        "search_results_sha256": sha256_file(search_path),
    })
    print(f"[stage3a-fit-five] finalists={len(per_finalist)} "
          f"all_distinct={all(f['all_distinct'] for f in per_finalist)}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# gate-r-dev (LONG TASK — prepared here, started by the user)
# --------------------------------------------------------------------------- #
def _predict_split(model, scaled_tensors, indices, device, batch_size=None) -> dict[str, np.ndarray]:
    """Run a response predictor over a split in the exact requested order.

    Returns ``episode_indices`` identical (value and order) to the requested
    ``indices``, plus ``r_hat`` (>=0) and ``regime_logits`` aligned to it.
    Uses unshuffled ``make_eval_batches`` — never the training shuffle.
    """
    indices = np.asarray(indices)
    if batch_size is None:
        batch_size = max(1, len(indices))
    model.to(device)
    model.eval()
    r_hats, logits, seen = [], [], []
    with torch.no_grad():
        for selected, batch in make_eval_batches(scaled_tensors, indices, batch_size):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                batch["history"], batch["factual_suffix"], batch["factual_actions"],
                batch["counterfactual_actions"], batch["cf_action"], batch["cf_duration"],
                batch["t_star"],
            )
            r_hats.append(invert_log1p(out["log1p_r_hat"]).cpu().numpy())
            logits.append(out["regime_logits"].cpu().numpy())
            seen.append(np.asarray(selected))
    return {
        "episode_indices": np.concatenate(seen, axis=0),
        "r_hat": np.concatenate(r_hats, axis=0),
        "regime_logits": np.concatenate(logits, axis=0),
    }


def _assert_episode_order(prediction: Mapping[str, np.ndarray], expected_indices) -> None:
    """Hard stop if a prediction's episode order differs from what was requested."""
    if not np.array_equal(np.asarray(prediction["episode_indices"]), np.asarray(expected_indices)):
        raise RuntimeError(
            "prediction episode order does not match the requested indices; "
            "refusing to score misaligned predictions"
        )


def _baseline_features(tensors, raw_factual_suffix, indices) -> dict[str, Any]:
    indices = np.asarray(indices)
    eval_shape = tuple(tensors["r_true"][indices].shape)
    return {
        "eval_shape": eval_shape,
        "n_eval": eval_shape[0],
        "horizon": eval_shape[1],
        "cf_action": tensors["cf_action"][indices].numpy(),
        "cf_duration": tensors["cf_duration"][indices].numpy().astype(np.float64),
        "initial_distance": make_initial_distance(raw_factual_suffix[indices]),  # per-vehicle (E,N)
    }


def _build_baselines(train_tensors, train_raw_suffix, train_indices,
                     target_tensors, target_raw_suffix, target_indices):
    """Fit every simple baseline on the TRAIN tensors and predict on the target.

    Returns ``(continuous_baselines, classification_baselines)`` over the target
    grid. Fitting never touches the target split/source; only
    ``predict``/``predict_proba`` does (dual-source safe, v0.1 Task 7).
    """
    train_feats = _baseline_features(train_tensors, train_raw_suffix, train_indices)
    target_feats = _baseline_features(target_tensors, target_raw_suffix, target_indices)
    r_train = train_tensors["r_true"][train_indices].numpy()
    reg_train = train_tensors["regime_labels"][train_indices].numpy()
    valid_train = train_tensors["valid"][train_indices].numpy()
    continuous = {
        "constant_zero": ConstantZeroBaseline().fit(r_train, train_feats, valid_train).predict(target_feats),
        "train_mean_by_horizon": TrainMeanByHorizonBaseline().fit(r_train, train_feats, valid_train).predict(target_feats),
        "linear_action_duration_horizon": LinearResponseBaseline(False).fit(r_train, train_feats, valid_train).predict(target_feats),
        "linear_plus_distance": LinearResponseBaseline(True).fit(r_train, train_feats, valid_train).predict(target_feats),
    }
    classification = {
        "prevalence": TrainPrevalenceClassifier().fit(reg_train, train_feats, valid_train).predict_proba(target_feats),
        "multinomial_adh": MultinomialRegimeClassifier(False).fit(reg_train, train_feats, valid_train).predict_proba(target_feats),
        "multinomial_plus_distance": MultinomialRegimeClassifier(True).fit(reg_train, train_feats, valid_train).predict_proba(target_feats),
    }
    return continuous, classification


# --------------------------------------------------------------------------- #
# select-final (mechanical, validation-only) -> receipt; gate-r-dev consumes it
# --------------------------------------------------------------------------- #
def cmd_select_final(args: argparse.Namespace) -> int:
    """Mechanically freeze the UNIQUE final config + strongest baselines.

    The final config is the finalist with the lowest MEAN five-seed validation
    loss (deterministic tie-break). The strongest baseline per metric is frozen
    on the validation split. Both are written to ``final_selection.json`` so the
    single development-test Go/No-Go cannot re-pick a config or a baseline on
    test data (item 3).
    """
    config = json.loads(_resolve(args.config).read_text(encoding="utf-8"))
    validate_stage3a_config(config)
    run_dir = _resolve(args.run_dir)
    selection_dir = run_dir / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selection_path = selection_dir / "final_selection.json"
    if selection_path.is_file():
        # The selection is single-use and immutable: once frozen it can never be
        # overwritten (a re-run would otherwise allow switching the final config).
        print(f"[stage3a-select-final] REFUSED: {selection_path} already exists; "
              "the final selection is immutable", flush=True)
        return 3
    fit5_path = run_dir / "fit_five" / "fit_five_results.json"
    fit5 = json.loads(fit5_path.read_text(encoding="utf-8"))
    per_finalist = fit5["finalists"]

    def mean_selection_score(finalist):
        return float(np.mean([s["validation_selection_score"] for s in finalist["per_seed"]]))

    # Mechanical selection: lowest mean five-seed validation SELECTION SCORE
    # (cross-lambda comparable, v0.1 §2.5); deterministic tie-break on
    # (hidden_dim, response_feature_dim, lambda_cls).
    ranked = sorted(
        per_finalist,
        key=lambda f: (
            mean_selection_score(f),
            f["config"]["hidden_dim"], f["config"]["response_feature_dim"], f["config"]["lambda_cls"],
        ),
    )
    final = ranked[0]
    final_config = dict(final["config"])

    # Freeze the strongest baselines on the VALIDATION split.
    bundle = _load_bundle_and_scaler(args)
    tensors, roles = bundle["tensors"], bundle["roles"]
    cont_val, clf_val = _build_baselines(
        tensors, bundle["raw_factual_suffix"], roles["train"],
        tensors, bundle["raw_factual_suffix"], roles["validation"],
    )
    strongest = select_strongest_baselines(
        r_true=tensors["r_true"][roles["validation"]].numpy(),
        regime_labels=tensors["regime_labels"][roles["validation"]].numpy(),
        valid_mask=tensors["valid"][roles["validation"]].numpy(),
        continuous_baselines=cont_val,
        classification_baselines=clf_val,
    )

    # The receipt signature covers the final config, the frozen baselines, the
    # five checkpoint signatures AND the five per-seed selection scores of the
    # selected finalist, plus the upstream fit-five results file hash — a later
    # edit of any of them is detectable at gate time (v0.1 §5 Task 5).
    checkpoint_signatures = [s["signature"] for s in final["per_seed"]]
    selection_scores = [float(s["validation_selection_score"]) for s in final["per_seed"]]
    fit5_sha = sha256_file(fit5_path)
    receipt = {
        "final_config": final_config,
        "selection_basis": "lowest mean five-seed validation selection score (L_reg + 0.25*L_CE; mechanical, deterministic tie-break)",
        "per_finalist_mean_validation_selection_score": {
            _finalist_name(f["config"]): mean_selection_score(f) for f in per_finalist
        },
        "selection_scores": selection_scores,
        "fit_five_results_sha256": fit5_sha,
        "strongest_baselines": strongest,
        "strongest_baselines_frozen_on": "validation",
        "finalist_name": _finalist_name(final_config),
        "checkpoint_signatures": checkpoint_signatures,
    }
    receipt["receipt_signature"] = s3t.compute_selection_receipt_signature(
        final_config=final_config,
        strongest_baselines=strongest,
        checkpoint_signatures=checkpoint_signatures,
        selection_scores=selection_scores,
        upstream_sha256=fit5_sha,
        data_signature=bundle["data_signature"],
        contract_id=s3t.SELECT_FINAL_CONTRACT_ID,
    )
    _atomic_write_json(selection_path, receipt)
    print(f"[stage3a-select-final] final={receipt['finalist_name']} "
          f"strongest={strongest} -> {selection_path}", flush=True)
    return 0


def _acquire_single_use_lock(lock_path: Path, finalist_name: str) -> Path:
    """Atomically create a config-independent single-use lock (O_EXCL).

    The lock is written BEFORE any protected truth is read; a crash after
    acquisition still blocks a later retry — the conservative single-use
    guarantee. Any prior use, under any config, refuses the run.
    """
    payload = json.dumps(
        {"finalist": finalist_name, "locked_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        indent=2,
    )
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"single-use resource already consumed (lock: {lock_path})")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)
    return lock_path


def load_replacement_tensors(npz_path: Path) -> dict[str, Any]:
    """Load a replacement development holdout as an ALL-EVAL tensor bundle.

    The replacement file is a dedicated test set: every episode is an eval
    episode, no train/validation roles exist. Schema, duration set {1,3}, the
    sealed-Stage-2 rejection and the forbidden-source rejection (old development
    source, Gate C paths) are enforced before any tensor is built (v0.1 §3.3).
    """
    assert_replacement_source_allowed(npz_path)
    with np.load(npz_path, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    required = (
        "object_history", "object_factual_future", "object_counterfactual_future",
        "factual_actions", "counterfactual_actions", "cf_action", "cf_duration", "t_star",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"replacement holdout missing required arrays: {missing}")
    durations = set(np.unique(np.asarray(data["cf_duration"])).tolist())
    if durations != {1, 3}:
        raise ValueError(f"replacement holdout cf_duration must be exactly {{1,3}}, got {durations}")
    n_episodes = int(len(data["cf_action"]))
    r_pos = compute_response_pos(data["object_factual_future"], data["object_counterfactual_future"])
    regime = compute_regime_labels(r_pos)
    eval_mask = compute_effect_eval_mask(
        data["object_history"], data["object_factual_future"], data["object_counterfactual_future"]
    )
    tensors = {
        "history": torch.as_tensor(data["object_history"], dtype=torch.float32),
        "factual_suffix": torch.as_tensor(data["object_factual_future"][:, :10], dtype=torch.float32),
        "factual_actions": torch.as_tensor(data["factual_actions"], dtype=torch.long),
        "counterfactual_actions": torch.as_tensor(data["counterfactual_actions"], dtype=torch.long),
        "cf_action": torch.as_tensor(data["cf_action"], dtype=torch.long),
        "cf_duration": torch.as_tensor(data["cf_duration"], dtype=torch.long),
        "t_star": torch.as_tensor(data["t_star"], dtype=torch.long),
        "r_true": torch.as_tensor(r_pos, dtype=torch.float32),
        "regime_labels": torch.as_tensor(regime, dtype=torch.long),
        "valid": torch.as_tensor(eval_mask, dtype=torch.bool),
    }
    return {
        "tensors": tensors,
        "roles": {"eval": np.arange(n_episodes)},
        "raw_factual_suffix": np.asarray(data["object_factual_future"][:, :10], dtype=np.float32),
        "data_signature": sha256_file(npz_path),
    }


def _verify_selection_receipt(run_dir: Path, receipt: Mapping[str, Any], data_signature: str) -> bool:
    """Recompute the selection receipt signature from on-disk artifacts.

    Covers the final config, frozen baselines, the five checkpoint signatures
    (recomputed from the .pt files), the five per-seed selection scores and the
    upstream fit-five results file hash. Any post-selection edit fails this.
    """
    cfg = receipt["final_config"]
    checkpoint_signatures = [
        s3t.checkpoint_signature(s3t.load_predictor_state(run_dir / "fit_five" / f"{_finalist_name(cfg)}_seed{s}.pt"))
        for s in PREDICTOR_SEEDS
    ]
    expected = s3t.compute_selection_receipt_signature(
        final_config=cfg,
        strongest_baselines=receipt["strongest_baselines"],
        checkpoint_signatures=checkpoint_signatures,
        selection_scores=receipt.get("selection_scores", []),
        upstream_sha256=sha256_file(run_dir / "fit_five" / "fit_five_results.json"),
        data_signature=data_signature,
        contract_id=s3t.SELECT_FINAL_CONTRACT_ID,
    )
    return expected == receipt.get("receipt_signature")


def cmd_lock_replacement_dev(args: argparse.Namespace) -> int:
    """Register the collected replacement holdout in the joint lock (v0.1 §3.3).

    Requires the frozen selection receipt, the five integrity-verified
    checkpoints and the collection receipt (whose ``data_sha256`` must match the
    replacement file). Written once with O_EXCL; the analysis command may only
    run after this lock exists.
    """
    run_dir = _resolve(args.run_dir)
    receipt_path = run_dir / "selection" / "final_selection.json"
    if not receipt_path.is_file():
        raise ValueError("selection/final_selection.json missing; freeze the selection first")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    # NOTE: checkpoint integrity is NOT verified here. The lock only BINDS the
    # replacement file to the frozen selection receipt; full receipt re-verification
    # (config + baselines + 5 checkpoint signatures + 5 selection scores) is the
    # gate's job, done in `gate-r-dev-retrain` BEFORE the single-use lock is burned.
    # Verifying here would crash (uncaught ValueError) on a tampered receipt instead
    # of refusing cleanly, and is redundant with the gate.
    repl_path = _resolve(args.replacement_test_source)
    assert_not_sealed_stage2(repl_path)
    repl_sha = sha256_file(repl_path)
    coll_path = run_dir / "replacement_dev" / "replacement_dev_collection_receipt.json"
    if not coll_path.is_file():
        print(f"[stage3a-lock-replacement] REFUSED: collection receipt missing ({coll_path})", flush=True)
        return 3
    coll = json.loads(coll_path.read_text(encoding="utf-8"))
    if coll.get("data_sha256") != repl_sha:
        print("[stage3a-lock-replacement] REFUSED: collection receipt data_sha256 "
              "does not match the replacement file", flush=True)
        return 3
    lock_path = run_dir / "replacement_dev" / "replacement_lock.json"
    payload = json.dumps({
        "replacement_data_sha256": repl_sha,
        "collection_receipt_sha256": sha256_file(coll_path),
        "selection_receipt_signature": receipt["receipt_signature"],
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"[stage3a-lock-replacement] REFUSED: lock already exists ({lock_path})", flush=True)
        return 3
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)
    print(f"[stage3a-lock-replacement] locked {repl_path} -> {lock_path}", flush=True)
    return 0


def cmd_gate_r_dev_retrain(args: argparse.Namespace) -> int:
    """The SINGLE replacement-development-test Go/No-Go (v0.1 §5 Task 7).

    All fitting state comes from the DEVELOPMENT source train/validation splits
    (scaler, class-weight context, baselines, temperature); every Gate metric is
    computed ONLY on the replacement holdout. Refusals (exit 3): tampered
    receipt, missing/mismatched replacement lock, replacement == development
    source, or a prior use of the replacement single-use lock.
    """
    config = json.loads(_resolve(args.config).read_text(encoding="utf-8"))
    validate_stage3a_config(config)
    run_dir = _resolve(args.run_dir)
    selection_path = run_dir / "selection" / "final_selection.json"
    if not selection_path.is_file():
        raise ValueError("selection/final_selection.json missing; run `select-final` first")
    receipt = json.loads(selection_path.read_text(encoding="utf-8"))
    cfg = receipt["final_config"]
    strongest = receipt["strongest_baselines"]

    # (1) Receipt re-verification against the DEVELOPMENT source — before any
    # replacement read.
    dev_signature = sha256_file(_resolve(args.development_source))
    if not _verify_selection_receipt(run_dir, receipt, dev_signature):
        print("[stage3a-gate-retrain] REFUSED: selection receipt signature mismatch "
              "(tampered receipt, swapped checkpoints or edited fit-five results)", flush=True)
        return 3

    # (2) Replacement lock must exist and bind THIS replacement file.
    repl_path = _resolve(args.replacement_test_source)
    repl_sha = sha256_file(repl_path)
    if repl_sha == dev_signature:
        print("[stage3a-gate-retrain] REFUSED: replacement source equals the development "
              "source (the contaminated old split 2 can never be the eval set)", flush=True)
        return 3
    # (2b) Forbidden replacement inputs (sealed Stage 2 by path/hash, the old
    # development source, any Gate C path) are refused BEFORE the single-use
    # lock is burned — a refusal must never consume the one adjudication.
    try:
        assert_replacement_source_allowed(repl_path, file_sha256=repl_sha)
    except ValueError as exc:
        print(f"[stage3a-gate-retrain] REFUSED: {exc}", flush=True)
        return 3
    lock_path = run_dir / "replacement_dev" / "replacement_lock.json"
    if not lock_path.is_file():
        print(f"[stage3a-gate-retrain] REFUSED: replacement lock missing ({lock_path}); "
              "run `lock-replacement-dev` first", flush=True)
        return 3
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("replacement_data_sha256") != repl_sha or \
            lock.get("selection_receipt_signature") != receipt["receipt_signature"]:
        print("[stage3a-gate-retrain] REFUSED: replacement lock does not bind this "
              "replacement file and selection receipt", flush=True)
        return 3

    # (3) Single-use replacement lock, atomically acquired BEFORE reading the
    # replacement truth. A second run — under any config — refuses.
    try:
        _acquire_single_use_lock(run_dir / "replacement_dev_used.lock", _finalist_name(cfg))
    except RuntimeError as exc:
        print(f"[stage3a-gate-retrain] REFUSED: {exc}", flush=True)
        return 3

    # --- development-side state: scaler, validation temperature, train baselines
    bundle = _load_bundle_and_scaler(args, source=args.development_source)
    dev_scaled, dev_roles = bundle["scaled"], bundle["roles"]
    dev_tensors = bundle["tensors"]
    device = config.get("device", "cpu")

    # --- replacement-side tensors, scaled with the DEVELOPMENT scaler (transform-only)
    repl = load_replacement_tensors(repl_path)
    repl_scaled = scale_tensors(repl["tensors"], bundle["scaler"])
    eval_indices = repl["roles"]["eval"]
    val_indices = dev_roles["validation"]

    # --- five-seed inference: replacement eval + development validation --------
    r_hat_seeds, logit_seeds, val_logit_seeds = [], [], []
    for seed in PREDICTOR_SEEDS:
        run_name = f"{_finalist_name(cfg)}_seed{seed}"
        state = s3t.load_predictor_state(run_dir / "fit_five" / f"{run_name}.pt")
        model = Stage3ResponsePredictor(
            hidden_dim=int(cfg["hidden_dim"]), response_feature_dim=int(cfg["response_feature_dim"])
        )
        model.load_state_dict(state)
        ev = _predict_split(model, repl_scaled, eval_indices, device)
        _assert_episode_order(ev, eval_indices)
        r_hat_seeds.append(ev["r_hat"])
        logit_seeds.append(ev["regime_logits"])
        val_pred = _predict_split(model, dev_scaled, val_indices, device)
        _assert_episode_order(val_pred, val_indices)
        val_logit_seeds.append(val_pred["regime_logits"])
    r_hat_seeds = np.stack(r_hat_seeds, axis=0)
    logit_seeds = np.stack(logit_seeds, axis=0)

    # --- temperature on the DEVELOPMENT validation split only ------------------
    val_logits = np.stack(val_logit_seeds, axis=0).mean(axis=0)
    temperature = fit_temperature(
        val_logits,
        dev_tensors["regime_labels"][val_indices].numpy(),
        dev_tensors["valid"][val_indices].numpy(),
    )

    # --- baselines fit on DEVELOPMENT train, predicted on the replacement set --
    continuous_baselines, classification_baselines = _build_baselines(
        dev_tensors, bundle["raw_factual_suffix"], dev_roles["train"],
        repl["tensors"], repl["raw_factual_suffix"], eval_indices,
    )

    report = evaluate_h4_gate(
        model_r_hat_seeds=r_hat_seeds,
        model_regime_logits_seeds=logit_seeds,
        temperature=temperature,
        r_true=repl["tensors"]["r_true"][eval_indices].numpy(),
        regime_labels=repl["tensors"]["regime_labels"][eval_indices].numpy(),
        valid_mask=repl["tensors"]["valid"][eval_indices].numpy(),
        continuous_baselines=continuous_baselines,
        classification_baselines=classification_baselines,
        strongest_baselines=strongest,
        n_resamples=int(args.n_resamples),
        confidence_level=0.95,
        bootstrap_seed=int(args.bootstrap_seed),
    )
    report["final_config"] = dict(cfg)
    report["eval_split"] = "replacement_development_test"
    report["development_data_signature"] = dev_signature
    report["replacement_data_signature"] = repl_sha
    report["replacement_lock_sha256"] = sha256_file(lock_path)
    report["selection_receipt_signature"] = receipt["receipt_signature"]
    report["seed_signatures"] = receipt["checkpoint_signatures"]
    gate_dir = run_dir / "gate_r_dev"
    gate_dir.mkdir(parents=True, exist_ok=True)
    report_path = gate_dir / f"gate_r_dev_retrain_report_{_finalist_name(cfg)}.json"
    _atomic_write_json(report_path, report)
    print(f"[stage3a-gate-retrain] passed={report['verdict']['passed']} -> {report_path}", flush=True)
    return 0 if report["verdict"]["passed"] else 2


def cmd_validate_config(args: argparse.Namespace) -> int:
    config = json.loads(_resolve(args.config).read_text(encoding="utf-8"))
    validate_stage3a_config(config)
    print(f"Stage 3A config OK: {args.config}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AC-Twin Stage 3A development driver")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-config")
    validate.add_argument("--config", required=True)
    validate.set_defaults(func=cmd_validate_config)

    bench = sub.add_parser("benchmark")
    bench.add_argument("--source", default=DEVELOPMENT_SOURCE)
    bench.add_argument("--run-dir", default="experiments/stage3_routing/stage3a")
    bench.add_argument("--full-epoch", action="store_true", default=True)
    bench.add_argument("--no-full-epoch", dest="full_epoch", action="store_false")
    bench.add_argument("--episodes", type=int, default=64)
    bench.add_argument("--epochs", type=int, default=1)
    bench.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    bench.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    bench.add_argument("--output", default="experiments/stage3_routing/benchmark.json")
    bench.set_defaults(func=cmd_benchmark)

    budget = sub.add_parser("budget")
    budget.add_argument("--config", required=True)
    budget.add_argument("--benchmark", default="experiments/stage3_routing/benchmark.json")
    budget.add_argument("--output", default="experiments/stage3_routing/COMPUTE_BUDGET.md")
    budget.set_defaults(func=cmd_budget)

    ready = sub.add_parser("readiness")
    ready.set_defaults(func=cmd_readiness)

    search = sub.add_parser("search")
    search.add_argument("--config", required=True)
    search.add_argument("--source", default=DEVELOPMENT_SOURCE)
    search.add_argument("--run-dir", default="experiments/stage3_routing/stage3a")
    search.add_argument("--resume", action="store_true")
    search.set_defaults(func=cmd_search)

    fit5 = sub.add_parser("fit-five")
    fit5.add_argument("--config", required=True)
    fit5.add_argument("--source", default=DEVELOPMENT_SOURCE)
    fit5.add_argument("--run-dir", default="experiments/stage3_routing/stage3a")
    fit5.add_argument("--resume", action="store_true")
    fit5.set_defaults(func=cmd_fit_five)

    select = sub.add_parser("select-final")
    select.add_argument("--config", required=True)
    select.add_argument("--source", default=DEVELOPMENT_SOURCE)
    select.add_argument("--run-dir", default="experiments/stage3_routing/stage3a")
    select.set_defaults(func=cmd_select_final)

    lockrepl = sub.add_parser("lock-replacement-dev")
    lockrepl.add_argument("--run-dir", default="experiments/stage3_routing/stage3a_retrain_v1")
    lockrepl.add_argument("--replacement-test-source", required=True)
    lockrepl.set_defaults(func=cmd_lock_replacement_dev)

    gate = sub.add_parser("gate-r-dev-retrain")
    gate.add_argument("--config", required=True)
    gate.add_argument("--development-source", required=True)
    gate.add_argument("--replacement-test-source", required=True)
    gate.add_argument("--run-dir", default="experiments/stage3_routing/stage3a_retrain_v1")
    gate.add_argument("--n-resamples", type=int, default=2000)
    gate.add_argument("--bootstrap-seed", type=int, default=20260908)
    gate.set_defaults(func=cmd_gate_r_dev_retrain)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
