"""Truth-free latency benchmark for the Stage 5 paper comparison.

This utility reuses only the frozen Stage 4C-R2 development-validation legal
inputs and the already selected checkpoints.  It never opens a truth file,
computes a scientific metric, changes model selection, or writes scores.  Model
loading and input preparation happen before timing; each timed call produces a
five-seed ensemble risk vector and materialises it on the host.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

CONFIG_PATH = ROOT / "config" / "stage5_sota_comparison_v0.3.json"
STAGE = ROOT / "experiments" / "stage5_sota_comparison"
DEFAULT_OUTPUT = (
    ROOT / "paper" / "tables" / "table_stage5_sota"
    / "stage5_inference_timing_receipt.json"
)
SCOPE = "ENGINEERING_TIMING_ONLY_NO_SCIENTIFIC_RESULTS"
FORBIDDEN_RECEIPT_TOKENS = (
    "aurc", "regret", "pr_auc", "spearman", "oracle", "truth",
    "selected_action", "scientific_metric",
)


def summarize_ms_per_group(samples_ms: list[float], n_groups: int) -> dict[str, float]:
    """Summarise repeated whole-bundle wall times as milliseconds per group."""
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")
    values = np.asarray(samples_ms, dtype=np.float64) / float(n_groups)
    if values.size < 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("need at least two finite non-negative timing samples")
    return {
        "median_ms_per_group": float(np.median(values)),
        "q1_ms_per_group": float(np.quantile(values, 0.25)),
        "q3_ms_per_group": float(np.quantile(values, 0.75)),
        "min_ms_per_group": float(values.min()),
        "max_ms_per_group": float(values.max()),
    }


def validate_receipt_scope(receipt: dict[str, Any]) -> None:
    """Fail closed if a timing receipt accidentally carries scientific results."""
    if receipt.get("scope") != SCOPE:
        raise ValueError("timing receipt has the wrong scope")

    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                lowered = str(key).lower()
                # Keys immediately below ``benchmark`` are method identifiers, not
                # receipt fields (for example SELECTIVENET_REGRET_ADAPTED).
                is_method_identifier = path == "benchmark."
                if (not is_method_identifier
                        and any(token in lowered for token in FORBIDDEN_RECEIPT_TOKENS)):
                    raise ValueError(f"scientific result field is forbidden: {path}{key}")
                walk(child, f"{path}{key}.")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}{index}.")

    walk(receipt)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def benchmark_callable(
    function: Callable[[], np.ndarray],
    *,
    n_groups: int,
    warmup: int,
    repeats: int,
    device: str,
) -> dict[str, float]:
    """Time a prepared scorer; loading and preprocessing are outside this call."""
    if warmup < 1 or repeats < 2:
        raise ValueError("warmup must be >=1 and repeats must be >=2")
    for _ in range(warmup):
        result = np.asarray(function())
        if result.shape != (n_groups,) or not np.isfinite(result).all():
            raise RuntimeError("prepared scorer returned an invalid risk vector")
    _sync(device)
    samples: list[float] = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter_ns()
        result = np.asarray(function())
        _sync(device)
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
        if result.shape != (n_groups,) or not np.isfinite(result).all():
            raise RuntimeError("timed scorer returned an invalid risk vector")
    summary = summarize_ms_per_group(samples, n_groups)
    summary["repeat_count"] = int(repeats)
    return summary


def slice_legal_bundle(bundle: Any, n_groups: int) -> Any:
    """Take a deterministic leading legal-input slice without touching truth."""
    total = int(np.asarray(bundle.group_seed).size)
    if n_groups <= 0 or n_groups > total:
        raise ValueError(f"n_groups must be within [1,{total}]")
    index = slice(0, int(n_groups))
    return dataclasses.replace(
        bundle,
        tokens=np.asarray(bundle.tokens)[index],
        token_mask=np.asarray(bundle.token_mask)[index],
        global_features=np.asarray(bundle.global_features)[index],
        j_hat=np.asarray(bundle.j_hat)[index],
        j_hat_by_seed=np.asarray(bundle.j_hat_by_seed)[:, index],
        selected_action=np.asarray(bundle.selected_action)[index],
        group_seed=np.asarray(bundle.group_seed)[index],
        scenario_id=np.asarray(bundle.scenario_id)[index],
        strata={key: np.asarray(value)[index] for key, value in bundle.strata.items()},
        valid_mask=np.asarray(bundle.valid_mask)[index],
    )


def _load_config_and_bundle() -> tuple[dict[str, Any], dict[str, Path], Any, str]:
    import torch
    from ac_twin import stage5_features as s5feat
    from ac_twin import stage5_protocol as s5prot

    config = s5prot.load_config(CONFIG_PATH)
    sources = s5prot.resolve_sources(config)
    device = str(config.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    bundle, _ = s5feat.build_feature_context(sources, "validation")
    if not bool(np.asarray(bundle.valid_mask).all()):
        raise RuntimeError("validation legal-input bundle contains invalid groups")
    return config, sources, bundle, device


def _prepared_actwin(sources: dict[str, Path], bundle: Any, device: str):
    import torch
    from ac_twin import stage4cp_models as s4cm
    from ac_twin import stage4cr2_training as s4cr2t
    from ac_twin import stage5_selection as s5sel

    inventory = s5sel._actwin_inventory(sources)
    tokens = inventory.scaler.transform_tokens(np.asarray(bundle.tokens, dtype=np.float64))
    globals_ = inventory.scaler.transform_global(
        np.asarray(bundle.global_features, dtype=np.float64))
    n = int(tokens.shape[0])
    mask = np.ones((n, tokens.shape[1]), dtype=bool)
    xt = torch.as_tensor(tokens, dtype=torch.float32, device=device)
    xg = torch.as_tensor(globals_, dtype=torch.float32, device=device)
    xm = torch.as_tensor(mask, dtype=torch.float32, device=device)
    loaded = []
    for seed in inventory.seeds:
        checkpoint = s4cm.load_checkpoint(inventory.checkpoint_paths[int(seed)])
        model = s4cm._load_model_from_checkpoint(checkpoint).to(device).eval()
        loaded.append((checkpoint, model))

    def score() -> np.ndarray:
        rows = []
        with torch.no_grad():
            for checkpoint, model in loaded:
                raw_mu, logit = s4cm._forward(model, str(checkpoint["arch"]), xt, xm, xg)
                rows.append(s4cm.heads_to_scores(
                    raw_mu.cpu().numpy(), logit.cpu().numpy())["risk_score"])
        return np.mean(np.stack(rows), axis=0)

    reference = s5sel._score_actwin(inventory, bundle, device).risk_ensemble
    if not np.allclose(score(), reference, rtol=1e-7, atol=1e-8):
        raise RuntimeError("prepared AC-Twin scorer does not match the frozen scoring path")
    return score, int(inventory.param_count)


def _method_receipts(method_id: str, config_key: str):
    from ac_twin.stage5_training import MethodReceipt

    ledger = json.loads((STAGE / "development" / "models" / "fit_receipts.json")
                        .read_text(encoding="utf-8"))
    rows = []
    for seed in (8101, 8202, 8303, 8404, 8505):
        key = f"{method_id}|{config_key}|{seed}"
        rows.append(MethodReceipt(**ledger[key]))
    return rows


def _prepared_method(method_id: str, config_key: str, bundle: Any, device: str):
    import torch
    from ac_twin import stage5_methods as s5m
    from ac_twin import stage5_training as s5t

    receipts = _method_receipts(method_id, config_key)
    prepared: list[Callable[[], np.ndarray]] = []
    for receipt in receipts:
        state = s5t._load_state(receipt)
        config = state["config"]

        if method_id == "CONFIDNET_TCP_ADAPTED":
            x = s5t._apply_scaler(s5m.p1_inputs(bundle, config), state["scaler"]["x"])
            xt = torch.from_numpy(x).to(device)
            model = s5m.build_model(method_id, config, receipt.seed)
            model.load_state_dict(state["model_state"])
            model.to(device).eval()

            def run(model=model, xt=xt):
                with torch.no_grad():
                    return (1.0 - model(xt)).cpu().numpy().astype(np.float64)

        elif method_id in ("SELECTIVENET_REGRET_ADAPTED", "RWR_NN_KNN_REJECT_ADAPTED"):
            tt, tm, tg = s5t._encoder_tensors(bundle, state["scaler"], device)
            model = s5m.build_model(method_id, config, receipt.seed)
            model.load_state_dict(state["model_state"])
            model.to(device).eval()
            if method_id == "SELECTIVENET_REGRET_ADAPTED":

                def run(model=model, tt=tt, tm=tm, tg=tg):
                    with torch.no_grad():
                        return (1.0 - model(tt, tm, tg)[1]).cpu().numpy().astype(np.float64)

            else:
                rejector = s5m.KNNRejector.from_state_dict(state["rejector"])

                def run(model=model, tt=tt, tm=tm, tg=tg, rejector=rejector):
                    with torch.no_grad():
                        rep = model.representation(tt, tm, tg).cpu().numpy().astype(np.float64)
                    return rejector.score(rep).astype(np.float64)

        elif method_id == "CONFORMAL_REGRESSION_REJECT_ADAPTED":
            difficulty = s5m.difficulty_from_bundle(bundle)
            mondrian = s5m.MondrianConformalState.from_state_dict(state["mondrian"])

            def run(difficulty=difficulty, mondrian=mondrian):
                return mondrian.score(difficulty).astype(np.float64)

        elif method_id == "CCTR_DRIVING_CALIBRATION_ADAPTED":
            x = s5t._apply_scaler(s5m.p5_inputs(bundle), state["scaler"]["x"])
            xt = torch.from_numpy(x).to(device)
            model = s5m.build_model(method_id, config, receipt.seed)
            model.load_state_dict(state["model_state"])
            model.to(device).eval()

            def run(model=model, xt=xt):
                with torch.no_grad():
                    return model(xt)[1].cpu().numpy().astype(np.float64)

        else:
            raise ValueError(f"unsupported selected method: {method_id}")
        prepared.append(run)

    def ensemble() -> np.ndarray:
        return np.mean(np.stack([run() for run in prepared]), axis=0)

    reference = np.mean(np.stack([
        s5t.score_method(receipt, bundle, device=device) for receipt in receipts
    ]), axis=0)
    if not np.allclose(ensemble(), reference, rtol=1e-7, atol=1e-8):
        raise RuntimeError(f"prepared scorer parity failed for {method_id}")
    return ensemble, int(receipts[0].param_count)


def _hardware(device: str) -> dict[str, Any]:
    import torch
    record: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": device,
    }
    if device.startswith("cuda"):
        record.update({
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
        })
    return record


def run_benchmark(output: Path, warmup: int, repeats: int,
                  batch_size: int = 1) -> dict[str, Any]:
    from ac_twin.stage1_data import sha256_file

    _, sources, full_bundle, device = _load_config_and_bundle()
    bundle = slice_legal_bundle(full_bundle, int(batch_size))
    selection = json.loads((STAGE / "development" / "final_method_selection.json")
                           .read_text(encoding="utf-8"))
    n_groups = int(np.asarray(bundle.group_seed).size)
    methods: dict[str, dict[str, Any]] = {}

    prepared: list[tuple[str, Callable[[], np.ndarray], int]] = []
    ac_fn, ac_params = _prepared_actwin(sources, bundle, device)
    prepared.append(("AC-Twin Direct", ac_fn, ac_params))
    for method_id, row in selection["selected"].items():
        function, params = _prepared_method(method_id, row["config_key"], bundle, device)
        prepared.append((method_id, function, params))

    total = len(prepared)
    for index, (name, function, params) in enumerate(prepared, start=1):
        print(f"[stage5-timing] {index}/{total} | {name} | warmup={warmup}, "
              f"repeats={repeats}", flush=True)
        row = benchmark_callable(function, n_groups=n_groups, warmup=warmup,
                                 repeats=repeats, device=device)
        row["param_count_per_seed"] = int(params)
        row["ensemble_seeds"] = 5
        methods[name] = row
        print(f"[stage5-timing] {name} median={row['median_ms_per_group']:.6f} "
              "ms/group", flush=True)

    source_hashes = {
        "validation_legal_input": sha256_file(sources["validation_data"]),
        "r2_model_inventory": sha256_file(sources["model_inventory"]),
        "stage5_final_selection": sha256_file(
            STAGE / "development" / "final_method_selection.json"),
    }
    receipt = {
        "scope": SCOPE,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "definition": (
            "five-seed ensemble risk-head scoring after model/input preload; "
            "device-to-host score materialisation included; shared upstream predictor excluded"
        ),
        "data_role": "stage4c_r2_development_validation_legal_inputs_only",
        "timing_batch_size": int(batch_size),
        "benchmark_group_seed": [int(x) for x in np.asarray(bundle.group_seed)],
        "n_groups": n_groups,
        "warmup_repeats": int(warmup),
        "timed_repeats": int(repeats),
        "hardware": _hardware(device),
        "source_sha256": source_hashes,
        "benchmark": methods,
    }
    validate_receipt_scope(receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)
    print(f"[stage5-timing] COMPLETE | {total}/{total} methods | receipt={output}", flush=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="number of legal groups per timed call; paper table uses 1")
    args = parser.parse_args()
    run_benchmark(args.output, args.warmup, args.repeats, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
