"""M1 supervised-baseline ablation runner (plan AC-TWIN-RFER-M1-SUPERVISED-ABLATION-2026-10-07-V0.2).

Subcommands: preflight / prepare / fit / evaluate / report / verify / run-all.

Every phase prints plain-text terminal progress (phase, completed/total, percent,
elapsed, ETA where meaningful) and ends with a success/failure summary. The score side
never reads counterfactual truth; validation truth is opened exactly once, AFTER all
fifteen checkpoints and the protocol are frozen, inside ``evaluate``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from ac_twin import m1_ablation_features as m1f
from ac_twin import m1_ablation_protocol as m1p
from ac_twin import m1_ablation_statistics as m1st
from ac_twin import m1_ablation_training as m1t
from ac_twin import stage4cp_features as s4cpf
from ac_twin import stage4cr2_training as s4cr2t
from ac_twin.stage1_data import sha256_file

_STARTED = time.perf_counter()


def _log(phase: str, message: str) -> None:
    elapsed = time.perf_counter() - _STARTED
    print(f"[m1 {phase} {elapsed:8.1f}s] {message}", flush=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True,
                                      ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _handoff(out: Path, **fields: Any) -> None:
    payload = {
        "plan_id": m1p.PLAN_ID,
        "analysis_status": m1p.ANALYSIS_STATUS,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **fields,
    }
    lines = ["# M1 ablation HANDOFF", ""]
    for key, value in payload.items():
        lines.append(f"- **{key}**: {value}")
    (out / "HANDOFF_STATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- frozen R2 reference ---------------------------------------------------------------------


def _frozen_inventory(sources: Mapping[str, Any]) -> s4cr2t.ModelInventory:
    """Rebuild the frozen R2 ModelInventory from hash-verified sources (read-only)."""
    receipt = _read_json(Path(sources["model_inventory"]))
    sp = receipt["scaler"]
    return s4cr2t.ModelInventory(
        plan_id=str(receipt["plan_id"]), arch=str(receipt["arch"]),
        seeds=tuple(int(s) for s in receipt["seeds"]),
        param_count=int(receipt["param_count"]),
        run_dir=str(Path(sources["model_inventory"]).parent),
        checkpoint_paths={int(s): str(sources[f"direct_checkpoint_{s}"])
                          for s in m1p.MODEL_SEEDS},
        checkpoint_sha256={int(s): sha256_file(sources[f"direct_checkpoint_{s}"])
                           for s in m1p.MODEL_SEEDS},
        scaler=s4cpf.FrozenFeatureScaler(
            np.asarray(sp["token_cont_mean"]), np.asarray(sp["token_cont_std"]),
            np.asarray(sp["global_mean"]), np.asarray(sp["global_std"])),
        pos_weight=float(receipt["pos_weight"]),
        interaction_threshold=float(receipt["interaction_threshold"]),
        main_indices=tuple(int(i) for i in receipt["main_indices"]),
        monitor_indices=tuple(int(i) for i in receipt["monitor_indices"]),
        train_losses={int(k): float(v) for k, v in receipt["train_losses"].items()},
        monitor_losses={int(k): float(v) for k, v in receipt["monitor_losses"].items()},
        n_train_groups=int(receipt["n_train_groups"]),
        written_at=str(receipt["written_at"]))


def _device_candidates(config: Mapping[str, Any]) -> list[str]:
    import torch
    device = str(config.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return [device] if device == "cpu" else [device, "cpu"]


# --- P0: preflight ----------------------------------------------------------------------------


def cmd_preflight(config: Mapping[str, Any]) -> None:
    out = m1p.assert_output_isolation(config)
    out.mkdir(parents=True, exist_ok=True)
    (out / "preflight").mkdir(parents=True, exist_ok=True)
    _log("preflight", "config valid; resolving the six-file whitelist + receipts")
    sources = m1p.resolve_sources(config)
    hashes = dict(m1p.resolve_sources.last_hashes)  # type: ignore[attr-defined]
    _log("preflight", f"{len(hashes)} sources hash-match the frozen R2 receipts")

    inventory_records: dict[str, Any] = {}
    for key, path in sorted(sources.items()):
        inventory_records[key] = {"path": m1p._normalised(Path(path).relative_to(m1p.ROOT))
                                  if Path(path).is_relative_to(m1p.ROOT) else str(path),
                                  "sha256": hashes[key]}
    _write_json(out / "preflight" / "source_inventory.json", {
        "plan_id": m1p.PLAN_ID, "whitelist": inventory_records,
        "binding": "final_selection.json + model_inventory.json hashes (verified live)"})

    # group identity checks (in-memory only; never persisted)
    train_seed = m1p.source_group_seed(sources, "train")
    val_seed = m1p.source_group_seed(sources, "validation")
    if train_seed.size != 5000 or val_seed.size != 1500:
        raise RuntimeError(f"role size drift: train={train_seed.size} val={val_seed.size}")
    if np.unique(train_seed).size != train_seed.size \
            or np.unique(val_seed).size != val_seed.size:
        raise RuntimeError("group seeds are not unique within a role")
    if set(train_seed.tolist()) & set(val_seed.tolist()):
        raise RuntimeError("train/validation group seeds overlap")
    for role in ("train", "validation"):
        with np.load(sources[f"{role}_data"], allow_pickle=True) as archive:
            if int(archive["n_actions"]) != 10:
                raise RuntimeError(f"{role}: ten-action contract drift")
            if int(archive["accepted_groups"]) != int(archive["episode_seed"].size):
                raise RuntimeError(f"{role}: accepted-group count drift")
    _log("preflight", "roles: train=5000 validation=1500, unique, disjoint, ten-action OK")

    import platform
    import torch
    environment = {
        "python": sys.version.split()[0], "executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "numpy": np.__version__,
    }
    if torch.cuda.is_available():
        environment["gpu"] = torch.cuda.get_device_name(0)
        environment["cuda"] = torch.version.cuda
    import scipy, sklearn
    environment["scipy"] = scipy.__version__
    environment["sklearn"] = sklearn.__version__
    _write_json(out / "preflight" / "environment.json", environment)
    _log("preflight", f"environment: {environment}")

    protected = m1p.hash_protected_files()
    _write_json(out / "preflight" / "protected_hashes.json", {
        "plan_id": m1p.PLAN_ID, "scope": "paper/manuscript artifacts + manuscript-linked "
        "figure/table files (read-only protection)", "hashes": protected})
    _log("preflight", f"protected manuscript hashes recorded: {len(protected)} files")

    # read-only frozen-model scoring stability on legal inputs (NO truth access)
    _log("preflight", "building validation legal features (score side only, no truth)")
    built = m1f.build_legal_role(sources, "validation", expected_groups=1500)
    inventory = _frozen_inventory(sources)
    bundle = {"tokens": built["tokens"], "global_features": built["global_features"],
              "group_seed": built["group_seed"]}
    device = _device_candidates(config)[0]
    first = s4cr2t.score_models(inventory, bundle, device=device)
    second = s4cr2t.score_models(inventory, bundle, device=device)
    if not np.array_equal(first.risk_ensemble, second.risk_ensemble):
        raise RuntimeError("frozen Direct repeated-inference scores are not stable")
    _log("preflight", f"frozen Direct read-back stable on device={device}; "
                      "scaler/schema/binding verified (no truth read)")

    policy = {
        "plan_id": m1p.PLAN_ID,
        "decision": "B2-b: caches store source_row_index + group_id bound to source "
                    "relpath/SHA256/identity digest; raw group_seed never persisted",
        "forbidden_npz_keys": list(m1p.FORBIDDEN_NPZ_KEYS),
        "seed_json_hints": list(m1p.SEED_JSON_HINTS),
        "expected_new_scanner_seeds": 0,
        "verification": "verify phase scans every new NPZ/JSON in the namespace with "
                        "the mirrored historical extraction rules",
    }
    _write_json(out / "preflight" / "cache_identity_policy.json", policy)
    _handoff(out, phase="P0_PREFLIGHT_PASS", next="prepare",
             completed="preflight/{source_inventory,environment,protected_hashes,"
                       "cache_identity_policy}.json",
             blockers="none")
    _log("preflight", "SUCCESS: sources legal/complete, output isolated, "
                      "frozen model loadable")


# --- P2a: prepare ------------------------------------------------------------------------------


def cmd_prepare(config: Mapping[str, Any]) -> None:
    out = m1p.assert_output_isolation(config)
    sources = m1p.resolve_sources(config)
    _log("prepare", "building train/validation legal feature caches (score side only)")
    for role, expected in (("train", 5000), ("validation", 1500)):
        built = m1f.build_legal_role(sources, role, expected_groups=expected)
        m1f.save_legal_cache(out / "features" / f"{role}_legal.npz", role,
                             Path(sources[f"{role}_data"]), built)
        _log("prepare", f"{role} legal cache written ({expected} groups, identity-bound)")
    _log("prepare", "building train labels (truth side, separate cache)")
    labels = m1f.build_role_labels(sources, "train")
    m1f.save_label_cache(out / "features" / "train_labels.npz", "train",
                         Path(sources["train_data"]), labels)
    _log("prepare", "train labels written (5000 groups)")

    train_cache = m1f.load_legal_cache(out / "features" / "train_legal.npz",
                                       role="train",
                                       source_path=Path(sources["train_data"]),
                                       expected_rows=5000)
    main_idx, monitor_idx = s4cr2t.internal_monitor_split(train_cache["group_seed"])
    split_payload = {
        "plan_id": m1p.PLAN_ID,
        "split_rule": "stage4cr2_training.internal_monitor_split(group_seed), "
                      "dev_seed_entropy=2026092880, frac=0.10",
        "role": "train",
        "source_relpath": m1p.TRAIN_DATA,
        "source_sha256": sha256_file(Path(sources["train_data"])),
        "source_identity_sha256": m1p.identity_digest(train_cache["group_seed"]),
        "main_row_index": [int(i) for i in main_idx],
        "monitor_row_index": [int(i) for i in monitor_idx],
    }
    m1p.assert_json_seed_free(split_payload, context="split.json")
    _write_json(out / "split.json", split_payload)
    _log("prepare", f"split fixed: main={main_idx.size} monitor={monitor_idx.size}")

    full_scaler = s4cpf.FrozenFeatureScaler.fit(
        train_cache["tokens"][main_idx], train_cache["global_features"][main_idx])
    full_scaler.save(out / "scalers" / "full.npz")
    b3_scaler = m1f.fit_baseline3_scaler(train_cache["baseline3"][main_idx])
    from ac_twin.checkpointing import atomic_save_npz
    atomic_save_npz(out / "scalers" / "baseline3.npz",
                    mean=b3_scaler["mean"], std=b3_scaler["std"])
    _log("prepare", "scalers fit on the 4,500 main rows only (full + baseline3)")

    # engineering smoke: synthetic backward/restore/output check (no validation signal)
    rng = np.random.default_rng(0)
    xs = rng.normal(size=(64, m1f.FLAT_DIM))
    yr = rng.normal(size=64)
    yh = (rng.random(64) > 0.5).astype(np.float64)
    smoke = m1t.fit_arm_seed("FULL_MATCHED", 8101, xs[:48], yr[:48], yh[:48],
                             xs[48:], yr[48:], yh[48:], pos_weight=1.0,
                             device="cpu")
    if not np.isfinite(smoke["train_loss"]):
        raise RuntimeError("engineering smoke produced a non-finite loss")
    _log("prepare", f"engineering smoke OK (synthetic 3-epoch-equivalent fit, "
                    f"loss={smoke['train_loss']:.4f})")

    lock = {
        "plan_id": m1p.PLAN_ID,
        "config_sha256": m1p.config_signature(config),
        "code_hashes": m1p.code_hashes(),
        "source_hashes": dict(m1p.resolve_sources.last_hashes),  # type: ignore[attr-defined]
        "split": {"main": int(main_idx.size), "monitor": int(monitor_idx.size)},
        "scaler_sha256": {
            "full": sha256_file(out / "scalers" / "full.npz"),
            "baseline3": sha256_file(out / "scalers" / "baseline3.npz")},
        "arms": list(m1p.ARMS), "model_seeds": list(m1p.MODEL_SEEDS),
        "contrasts": [list(c) for c in m1p.CONTRASTS],
        "bootstrap": {"draws": m1p.BOOTSTRAP_DRAWS, "entropy": m1p.BOOTSTRAP_ENTROPY,
                      "main_ci_quantiles": list(m1p.MAIN_CI_QUANTILES)},
        "protected_manuscript_hashes": m1p.hash_protected_files(),
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    m1p.assert_json_seed_free(lock, context="protocol_lock.json")
    m1p.create_protocol_lock(out / "protocol_lock.json", lock)
    _handoff(out, phase="P2_PREPARE_LOCKED", next="fit",
             completed="features caches, train labels, split.json, scalers, protocol_lock.json",
             blockers="none")
    _log("prepare", "SUCCESS: protocol locked; analysis design frozen")


# --- P2b: fit -----------------------------------------------------------------------------------


def _load_train_side(out: Path, sources: Mapping[str, Any]) -> dict[str, Any]:
    train_cache = m1f.load_legal_cache(out / "features" / "train_legal.npz", role="train",
                                       source_path=Path(sources["train_data"]),
                                       expected_rows=5000)
    labels = m1f.load_label_cache(out / "features" / "train_labels.npz", role="train",
                                  source_path=Path(sources["train_data"]),
                                  expected_rows=5000)
    split = _read_json(out / "split.json")
    main_idx = np.asarray(split["main_row_index"], dtype=np.int64)
    monitor_idx = np.asarray(split["monitor_row_index"], dtype=np.int64)
    full_scaler = s4cpf.FrozenFeatureScaler.load(out / "scalers" / "full.npz")
    with np.load(out / "scalers" / "baseline3.npz") as archive:
        b3_scaler = {"mean": archive["mean"], "std": archive["std"]}
    arm_inputs = {
        arm: m1f.arm_flat_inputs(
            arm, train_cache["tokens"], train_cache["global_features"],
            train_cache["baseline3"], full_scaler=full_scaler,
            baseline3_scaler=b3_scaler)
        for arm in m1p.ARMS}
    return {"labels": labels, "main_idx": main_idx, "monitor_idx": monitor_idx,
            "arm_inputs": arm_inputs}


def cmd_fit(config: Mapping[str, Any]) -> None:
    out = m1p.assert_output_isolation(config)
    m1p.verify_protocol_lock(out / "protocol_lock.json")
    sources = m1p.resolve_sources(config)
    side = _load_train_side(out, sources)
    labels, main_idx, monitor_idx = side["labels"], side["main_idx"], side["monitor_idx"]
    n_main = int(main_idx.size)
    n_pos = float(labels["y_high"][main_idx].sum())
    pos_weight = (n_main - n_pos) / max(n_pos, 1.0)
    _log("fit", f"15 fits (5 seeds x 3 arms), pos_weight={pos_weight:.6f} from main only")
    device = _device_candidates(config)[0]
    m1t.fit_all_arms(
        side["arm_inputs"], main_idx=main_idx, monitor_idx=monitor_idx,
        y_reg=labels["y_reg"], y_high=labels["y_high"], pos_weight=pos_weight,
        model_root=out / "models", device=device,
        progress=lambda msg: _log("fit", msg))
    _handoff(out, phase="P2_FIT_COMPLETE", next="evaluate",
             completed="models/{FULL_MATCHED,BASELINE3_MATCHED,NO_RISK_MATCHED}/seed*.pt "
                       "+ model_inventory.json (15 checkpoints)",
             blockers="none")
    _log("fit", "SUCCESS: all 15 checkpoints frozen")


# --- P3: evaluate --------------------------------------------------------------------------------


def cmd_evaluate(config: Mapping[str, Any]) -> None:
    out = m1p.assert_output_isolation(config)
    m1p.verify_protocol_lock(out / "protocol_lock.json")
    sources = m1p.resolve_sources(config)
    inventory = _read_json(out / "models" / m1t.MODEL_INVENTORY_NAME)
    val_cache = m1f.load_legal_cache(out / "features" / "validation_legal.npz",
                                     role="validation",
                                     source_path=Path(sources["validation_data"]),
                                     expected_rows=1500)
    full_scaler = s4cpf.FrozenFeatureScaler.load(out / "scalers" / "full.npz")
    with np.load(out / "scalers" / "baseline3.npz") as archive:
        b3_scaler = {"mean": archive["mean"], "std": archive["std"]}
    device = _device_candidates(config)[0]

    scores: dict[str, np.ndarray] = {}
    scores_by_seed: dict[str, np.ndarray] = {}
    _log("evaluate", "scoring the three new arms on validation (frozen checkpoints only)")
    for arm in m1p.ARMS:
        flat = m1f.arm_flat_inputs(
            arm, val_cache["tokens"], val_cache["global_features"],
            val_cache["baseline3"], full_scaler=full_scaler,
            baseline3_scaler=b3_scaler)
        checkpoints = m1t.load_arm_checkpoints(inventory, arm)
        result = m1t.score_arm_checkpoints(checkpoints, flat, device=device)
        scores[arm] = result["risk_ensemble"]
        scores_by_seed[arm] = result["risk_by_seed"]
        _log("evaluate", f"{arm}: 5-seed ensemble scores complete")

    # frozen historical reference: score with the ORIGINAL scaler/checkpoints; the
    # continuity tolerance (1e-8 vs the R2 gate report) is applied after the truth
    # side is opened; exceeding it is an engineering localisation stop, never a
    # silent pass and never a retrain (plan P3).
    _log("evaluate", "scoring FULL_FROZEN_REFERENCE (original frozen weights + scaler)")
    frozen = _frozen_inventory(sources)
    bundle = {"tokens": val_cache["tokens"],
              "global_features": val_cache["global_features"],
              "group_seed": val_cache["group_seed"]}
    reference = s4cr2t.score_models(frozen, bundle, device=device)
    _log("evaluate", f"frozen reference scored on device={device}")
    scores["FULL_FROZEN_REFERENCE"] = np.asarray(reference.risk_ensemble,
                                                 dtype=np.float64)
    for i, name in enumerate(m1p.BASELINE3_ORDER):
        scores[name] = np.asarray(val_cache["baseline3"][:, i], dtype=np.float64)

    row_index = np.arange(1500, dtype=np.int64)
    score_arrays = {f"score_{method}": np.asarray(values, dtype=np.float64)
                    for method, values in scores.items()}
    score_arrays.update({
        "source_row_index": row_index,
        "group_id": m1p.make_group_ids("validation", row_index),
        "role": np.asarray("validation"),
        "source_relpath": np.asarray(m1p.VALIDATION_DATA),
        "source_sha256": np.asarray(sha256_file(Path(sources["validation_data"]))),
        "source_identity_sha256": np.asarray(
            m1p.identity_digest(val_cache["group_seed"])),
    })
    m1p.assert_cache_keys_legal(list(score_arrays.keys()), context="validation scores")
    from ac_twin.checkpointing import atomic_save_npz
    atomic_save_npz(out / "scores" / "validation_scores.npz", **score_arrays)
    _log("evaluate", "validation scores frozen (7 methods x 1500 groups)")

    # ---- truth side: opened exactly once, after every checkpoint is frozen ----
    _log("evaluate", "reading validation truth (labels/evaluation only)")
    truth = m1f.build_role_labels(sources, "validation")
    m1f.save_label_cache(out / "evaluation" / "validation_truth.npz", "validation",
                         Path(sources["validation_data"]), truth)
    regret = truth["regret"]

    _log("evaluate", "computing absolute metrics + C1-C5 contrasts + paired bootstrap")
    bootstrap = m1st.paired_bootstrap(
        scores, regret, progress=lambda msg: _log("evaluate", msg))
    absolute_rows = m1st.absolute_metrics_table(bootstrap)
    contrast_rows = m1st.contrast_table(bootstrap)
    heuristics = {name: scores[name] for name in m1p.HEURISTIC_METHODS}
    per_seed_rows = m1st.per_seed_contrasts(scores_by_seed, heuristics, regret)
    pr_rows = m1st.high_regret_pr_by_method(scores, truth["high_regret"])

    names = bootstrap["names"]
    header = ",".join(["draw"] + names)
    lines = [header]
    draws = bootstrap["draws"]
    # full draws stay in the NPZ (4 MB); the CSV keeps a bounded human-auditable head.
    for i in range(min(200, draws.shape[0])):
        lines.append(",".join([str(i)] + [f"{v:.10g}" for v in draws[i]]))
    (out / "evaluation").mkdir(parents=True, exist_ok=True)
    (out / "evaluation" / "paired_draws.csv").write_text("\n".join(lines) + "\n",
                                                         encoding="utf-8")
    atomic_save_npz(out / "evaluation" / "paired_draws.npz",
                    names=np.asarray(names), draws=draws,
                    point=np.asarray(bootstrap["point"], dtype=np.float64))
    _write_csv(out / "evaluation" / "absolute_metrics.csv",
               absolute_rows, ("method", m1st.METRIC_AURC, m1st.METRIC_R80))
    _write_csv(out / "evaluation" / "contrasts.csv", contrast_rows,
               ("contrast", "metric", "form", "method", "reference", "point",
                "ci99_lower", "ci99_upper", "ci95_lower", "ci95_upper",
                "invalid_draws", "status"))
    _write_csv(out / "evaluation" / "per_seed.csv", per_seed_rows,
               ("contrast", "metric", "seed_index", "improvement", "absdiff",
                "direction"))

    gate = _read_json(Path(sources["validation_gate_report"]))
    continuity = m1st.continuity_check(absolute_rows, gate)
    _write_json(out / "evaluation" / "continuity.json", continuity)
    if not continuity["passed"]:
        raise RuntimeError(
            "M1_BLOCKED_ENGINEERING: frozen-reference continuity exceeds 1e-8 "
            f"against the R2 gate report: {continuity['checks']}")
    _log("evaluate", "continuity vs R2 validation_gate_report within 1e-8: PASS")

    _write_json(out / "evaluation" / "bootstrap_receipt.json", {
        "plan_id": m1p.PLAN_ID, "draws": bootstrap["n_draws"],
        "entropy": bootstrap["entropy"],
        "main_family": "C1-C5 AURC contrasts, per-contrast 99% CI (0.005/0.995), "
                       "Bonferroni family-95%",
        "secondary_family": "C1-C5 Regret@80 contrasts, same intervals, never upgraded",
        "uncorrected_95": "labelled uncorrected; narrative uses the family 99% CI",
        "pr_auc_minus_prevalence_descriptive": pr_rows,
        "recompute": "run_m1_ablation.py evaluate --config <config>"})
    _handoff(out, phase="P3_EVALUATION_COMPLETE", next="report",
             completed="scores/validation_scores.npz, evaluation/{paired_draws,"
                       "absolute_metrics,contrasts,per_seed,continuity,bootstrap_receipt}",
             blockers="none")
    _log("evaluate", "SUCCESS: metrics, contrasts and bootstrap complete")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(_csv_cell(row.get(c)) for c in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


# --- P4: report -----------------------------------------------------------------------------------


def cmd_report(config: Mapping[str, Any]) -> None:
    out = m1p.assert_output_isolation(config)
    m1p.verify_protocol_lock(out / "protocol_lock.json")
    absolute_rows = _read_csv(out / "evaluation" / "absolute_metrics.csv")
    contrast_rows = _read_csv(out / "evaluation" / "contrasts.csv")
    per_seed_rows = _read_csv(out / "evaluation" / "per_seed.csv")
    continuity = _read_json(out / "evaluation" / "continuity.json")
    receipt = _read_json(out / "evaluation" / "bootstrap_receipt.json")
    val_truth = m1f.load_label_cache(
        out / "evaluation" / "validation_truth.npz", role="validation",
        source_path=m1p.ROOT / m1p.VALIDATION_DATA, expected_rows=1500)
    with np.load(out / "scores" / "validation_scores.npz", allow_pickle=False) as archive:
        scores = {key[len("score_"):]: np.asarray(archive[key], dtype=np.float64)
                  for key in archive.files if key.startswith("score_")}
    regret = val_truth["regret"]

    # --- main table ---
    active = m1f.active_input_dims()
    info_sets = {
        "FULL_MATCHED": "complete 315-dim input",
        "BASELINE3_MATCHED": "three standard baseline scalars only",
        "NO_RISK_MATCHED": "complete input minus 5 explicit risk channels",
        "FULL_FROZEN_REFERENCE": "original frozen Direct (historical continuity)",
        "MARGIN_ONLY": "fixed heuristic", "SEED_VOTE_ENTROPY": "fixed heuristic",
        "SEED_MARGIN_STD": "fixed heuristic"}
    table_lines = [
        "method,information_set,active_dims,nominal_params,aurc_regret,regret_at_80,"
        "pr_auc_minus_prevalence"]
    pr = receipt["pr_auc_minus_prevalence_descriptive"]
    for row in absolute_rows:
        method = row["method"]
        dims = active.get(method, "")
        params = m1p.EXPECTED_TRAINABLE_PARAMETERS if method in m1p.ARMS else (
            m1p.EXPECTED_TRAINABLE_PARAMETERS if method == "FULL_FROZEN_REFERENCE" else 0)
        table_lines.append(
            f"{method},\"{info_sets[method]}\",{dims},{params},"
            f"{row[m1st.METRIC_AURC]:.10g},{row[m1st.METRIC_R80]:.10g},{pr[method]:.10g}")
    (out / "report").mkdir(parents=True, exist_ok=True)
    (out / "report" / "table_M1.csv").write_text("\n".join(table_lines) + "\n",
                                                 encoding="utf-8")

    # --- figure: three new arms + the frozen main heuristic ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for method, style in (("FULL_MATCHED", "-"), ("BASELINE3_MATCHED", "--"),
                          ("NO_RISK_MATCHED", "-."), ("SEED_VOTE_ENTROPY", ":")):
        curve = m1st.risk_coverage_curve(scores[method], regret)
        ax.plot(curve["coverage"], curve["risk"], style, label=method, linewidth=1.4)
    ax.axvline(m1p.COVERAGE_MAIN, color="grey", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("coverage"); ax.set_ylabel("selective regret (mean retained)")
    ax.set_title("M1 development risk-coverage curves (validation, "
                 "ADAPTIVE_DEVELOPMENT_ONLY)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out / "report" / "figure_M1_curve.png", dpi=200)
    plt.close(fig)
    _log("report", "table_M1.csv + figure_M1_curve.png written")

    # --- narrative report with the frozen conclusion mapping (plan section 8) ---
    report = _render_report(absolute_rows, contrast_rows, per_seed_rows, continuity,
                            receipt, active)
    (out / "report" / "M1_report.md").write_text(report, encoding="utf-8")
    _handoff(out, phase="M1_DIAGNOSTIC_COMPLETE / "
                        "MANUSCRIPT_UPDATE_AWAITING_AUTHOR_CONFIRMATION",
             next="author review; NO manuscript edit without separate confirmation",
             completed="report/{M1_report.md,table_M1.csv,figure_M1_curve.png}",
             blockers="none")
    _log("report", "SUCCESS: M1 report delivered (manuscript untouched)")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    rows: list[dict[str, Any]] = []
    for line in lines[1:]:
        cells = line.split(",")
        row: dict[str, Any] = {}
        for key, cell in zip(header, cells):
            try:
                row[key] = float(cell)
            except ValueError:
                row[key] = cell
        rows.append(row)
    return rows


def _fmt_interval(row: Mapping[str, Any]) -> str:
    if row["ci99_lower"] in ("", None):
        return "NOT_IDENTIFIABLE"
    return f"[{float(row['ci99_lower']):.4f}, {float(row['ci99_upper']):.4f}]"


def _render_report(absolute_rows, contrast_rows, per_seed_rows, continuity, receipt,
                   active) -> str:
    abs_map = {r["method"]: r for r in absolute_rows}

    def contrast(cid: str, metric: str, form: str) -> Mapping[str, Any]:
        return next(r for r in contrast_rows
                    if r["contrast"] == cid and r["metric"] == metric
                    and r["form"] == form)

    lines = [
        "# M1 监督对照与特征消融：开发性诊断报告",
        "",
        f"- 计划 ID：`{m1p.PLAN_ID}`",
        f"- 性质：Merge 开发性组件诊断，`{m1p.ANALYSIS_STATUS}`；非确认、非新预注册",
        "- 数据：R2 train 5,000 组（main 4,500 / monitor 500）与 validation 1,500 组；"
        "未采集新数据，未创建 screen/power/confirm",
        f"- bootstrap：{receipt['draws']} 次整组配对重抽样，entropy "
        f"{receipt['entropy']}；主家族 C1–C5 AURC 对比 99% 区间（0.005/0.995，"
        "Bonferroni 家族 95%），次指标家族同口径；另附未校正 95% 仅供参考",
        f"- 连续性核验：旧冻结 Direct 与三项启发式相对 R2 validation_gate_report "
        f"全部在 1e-8 内（passed={continuity['passed']}）",
        "- **正文编辑边界：无论结果如何，本报告不写回摘要、正文、补充稿、Word 或"
        "稿件图表；任何修稿须作者另行明确确认。**",
        "",
        "## 1. 绝对指标（validation 1,500 组，配对同一套 draws）",
        "",
        "| 方法 | 信息集 | 活跃维度 | 名义参数 | AURC-regret | Regret@80 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for method in m1p.ALL_METHODS:
        row = abs_map[method]
        params = m1p.EXPECTED_TRAINABLE_PARAMETERS if method in (
            *m1p.ARMS, "FULL_FROZEN_REFERENCE") else 0
        lines.append(
            f"| {method} | {'完整315维' if method in ('FULL_MATCHED','FULL_FROZEN_REFERENCE') else '仅三基线标量' if method == 'BASELINE3_MATCHED' else '屏蔽五类风险通道' if method == 'NO_RISK_MATCHED' else '固定启发式'} "
            f"| {active.get(method, '—')} | {params} | "
            f"{row[m1st.METRIC_AURC]:.6f} | {row[m1st.METRIC_R80]:.6f} |")
    lines += [
        "",
        "## 2. C1–C5 主对比（改善率 = 1 − 方法/参照；99% 为家族校正区间）",
        "",
        "| 对比 | 指标 | 点估计 | 99% CI（家族） | 95% CI（未校正） | 绝对差（点） | 判定 |",
        "| --- | --- | ---: | --- | --- | ---: | --- |",
    ]
    verdicts: dict[str, str] = {}
    for cid, _, _ in m1p.CONTRASTS:
        for metric in (m1st.METRIC_AURC, m1st.METRIC_R80):
            imp = contrast(cid, metric, "improvement")
            diff = contrast(cid, metric, "absdiff")
            if imp["status"] == "NOT_IDENTIFIABLE":
                verdict = "NOT_IDENTIFIABLE"
            elif float(imp["ci99_lower"]) > 0:
                verdict = "区间下界>0"
            elif float(imp["ci99_upper"]) < 0:
                verdict = "区间上界<0（受反证）"
            else:
                verdict = "区间跨零（未建立）"
            if metric == m1st.METRIC_AURC:
                verdicts[cid] = verdict
            lines.append(
                f"| {cid} | {metric} | {float(imp['point']):.4f} | {_fmt_interval(imp)} | "
                f"[{float(imp['ci95_lower']):.4f}, {float(imp['ci95_upper']):.4f}] | "
                f"{float(diff['point']):.6f} | {verdict} |")
    lines += [
        "",
        "## 3. 五 seed 方向（点估计，描述性）",
        "",
        "| 对比 | 指标 | 正/负/零 | 范围 | 中位数 |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for cid, _, _ in m1p.CONTRASTS:
        for metric in (m1st.METRIC_AURC, m1st.METRIC_R80):
            vals = [float(r["improvement"]) for r in per_seed_rows
                    if r["contrast"] == cid and r["metric"] == metric]
            pos = sum(v > 0 for v in vals)
            neg = sum(v < 0 for v in vals)
            zero = len(vals) - pos - neg
            lines.append(
                f"| {cid} | {metric} | {pos}/{neg}/{zero} | "
                f"[{min(vals):.4f}, {max(vals):.4f}] | {float(np.median(vals)):.4f} |")
    lines += [
        "",
        "## 4. 结论映射（冻结口径，§8）",
        "",
        f"- C1（完整输入 vs 三基线监督组合，AURC）：{verdicts['C1']}",
        f"- C2（显式风险通道条件增量，AURC）：{verdicts['C2']}",
        f"- C3（监督组合 vs 原冻结主启发式 SEED_VOTE_ENTROPY，AURC）：{verdicts['C3']}",
        f"- C4（监督组合 vs MARGIN_ONLY，AURC）：{verdicts['C4']}",
        f"- C5（监督组合 vs SEED_MARGIN_STD，AURC）：{verdicts['C5']}",
        "",
        "解释边界：区间条件于已训练模型与历史开发数据；即使区间全为正，也仅为"
        "开发性增量证据。不得把改善率相减去归因原确认 26.0% 的构成；C1 跨零不"
        "证明 FULL 与 BASELINE3 等效（本方案无等效/非劣效设计）；任何负结果或"
        "不稳定均如实报告，不追加候选或调整训练挽救。",
    ]
    return "\n".join(lines) + "\n"


# --- verify / run-all ---------------------------------------------------------------------------


def cmd_verify(config: Mapping[str, Any]) -> None:
    out = m1p.assert_output_isolation(config)
    lock = m1p.verify_protocol_lock(out / "protocol_lock.json")
    _log("verify", "protocol lock: code + protected manuscript hashes intact")
    drifted = m1p.verify_protected_hashes(lock["protected_manuscript_hashes"])
    if drifted:
        raise RuntimeError(f"protected manuscript files drifted: {drifted}")
    inventory = _read_json(out / "models" / m1t.MODEL_INVENTORY_NAME)
    if len(inventory["checkpoints"]) != 15:
        raise RuntimeError(f"expected 15 checkpoints, got {len(inventory['checkpoints'])}")
    for arm in m1p.ARMS:
        m1t.load_arm_checkpoints(inventory, arm)
    _log("verify", "15 checkpoints hash-match the frozen inventory")
    sources = m1p.resolve_sources(config)
    val = m1f.load_legal_cache(out / "features" / "validation_legal.npz",
                               role="validation",
                               source_path=Path(sources["validation_data"]),
                               expected_rows=1500)
    del val
    with np.load(out / "scores" / "validation_scores.npz", allow_pickle=False) as archive:
        m1p.assert_cache_keys_legal(list(archive.files), context="validation scores")
        for key in archive.files:
            if key.startswith("score_") and not np.isfinite(archive[key]).all():
                raise RuntimeError(f"non-finite scores: {key}")
    # historical-scanner compatibility: every new NPZ/JSON in the namespace is seed-free
    offenders: list[str] = []
    for path in sorted(out.rglob("*")):
        if path.suffix == ".npz" and m1p.scanner_npz_seeds(path):
            offenders.append(str(path))
        elif path.suffix == ".json" and m1p.scanner_json_seeds(_read_json(path)):
            offenders.append(str(path))
    if offenders:
        raise RuntimeError(f"new namespace carries scanner-visible group seeds: {offenders}")
    if list(out.rglob("*seed_schedule*.npy")):
        raise RuntimeError("forbidden *seed_schedule*.npy in the new namespace")
    continuity = _read_json(out / "evaluation" / "continuity.json")
    if not continuity["passed"]:
        raise RuntimeError("continuity check not passed")
    _handoff(out, phase="M1_DIAGNOSTIC_COMPLETE / "
                        "MANUSCRIPT_UPDATE_AWAITING_AUTHOR_CONFIRMATION",
             next="author review; NO manuscript edit without separate confirmation",
             completed="verify: hashes, groups, scores, tables, identity contract",
             blockers="none")
    _log("verify", "SUCCESS: hashes, counts, scores, identity contract and "
                   "terminal state consistent")


def cmd_run_all(config: Mapping[str, Any]) -> None:
    out = m1p.assert_output_isolation(config)
    if not (out / "protocol_lock.json").is_file():
        cmd_prepare(config)
    cmd_fit(config)
    cmd_evaluate(config)
    cmd_report(config)
    cmd_verify(config)
    _log("run-all", "SUCCESS: M1 pipeline complete "
                    "(M1_DIAGNOSTIC_COMPLETE / manuscript untouched)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "preflight", "prepare", "fit", "evaluate", "report", "verify", "run-all"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = m1p.load_config(args.config)
    handlers = {
        "preflight": cmd_preflight, "prepare": cmd_prepare, "fit": cmd_fit,
        "evaluate": cmd_evaluate, "report": cmd_report, "verify": cmd_verify,
        "run-all": cmd_run_all,
    }
    handlers[args.command](config)


if __name__ == "__main__":
    main()
