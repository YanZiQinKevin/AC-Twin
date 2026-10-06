"""Stage 5 mechanical validation selection (plan v0.3 section 3.3, Task 4).

Plan ``AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3``. The selection is a
pure function of VALIDATION metrics — it never touches the comparison holdout
(which does not exist until after the full freeze):

1. exclude candidates with non-finite outputs, degenerate (constant) rankings,
   truth leakage or seed inconsistency;
2. per method, the lowest validation ``AURC_REGRET`` wins;
3. ties within 1e-12 break by lower ``REGRET@80``, fewer parameters, faster
   single-group inference, then lexicographic config key;
4. ``STRONGEST_PUBLISHED_BASELINE`` is the UNIQUE lowest-AURC winner across the
   five adapted methods (same tie chain, lexicographic method ID last);
5. ``final_method_selection.json`` is create-once and records EVERY candidate.

The module tail carries the Phase D development-freeze orchestration (plan 6.1):
fit every (method, grid config, seed) on the train role, evaluate all candidates
on validation ONCE, freeze the selections, and write the create-once artifacts
(``method_search_space.json``, ``final_method_selection.json``,
``validation_comparison_report.json``, ``validation_risks_selected.npz``).
"""
from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ac_twin.stage3r_protocol import write_once_json

TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CandidateRecord:
    """One (method, grid config) validation outcome — validation metrics ONLY."""
    method_id: str
    config_key: str
    config: dict[str, Any]
    aurc_regret: float
    regret_at_80: float
    param_count: int
    mean_inference_ms: float
    valid: bool
    exclusion_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "config_key": self.config_key,
            "config": dict(self.config),
            "aurc_regret": float(self.aurc_regret),
            "regret_at_80": float(self.regret_at_80),
            "param_count": int(self.param_count),
            "mean_inference_ms": float(self.mean_inference_ms),
            "valid": bool(self.valid),
            "exclusion_reason": self.exclusion_reason,
        }


def check_candidate_validity(risk_by_seed: np.ndarray) -> tuple[bool, str]:
    """Validity screen (plan 3.3 step 1): finite + non-degenerate five-seed risks."""
    risks = np.asarray(risk_by_seed, dtype=np.float64)
    if risks.ndim != 2:
        return False, "risk_by_seed must be (seeds, groups)"
    if not np.isfinite(risks).all():
        return False, "non-finite risk outputs"
    for i in range(risks.shape[0]):
        if float(risks[i].std()) <= 1e-12:
            return False, "degenerate (constant) risk ranking"
    if float(risks.mean(axis=0).std()) <= 1e-12:
        return False, "degenerate (constant) ensemble ranking"
    return True, ""


def _sort_key(candidate: CandidateRecord) -> tuple:
    return (round(float(candidate.aurc_regret) / TIE_TOLERANCE),
            round(float(candidate.regret_at_80) / TIE_TOLERANCE),
            int(candidate.param_count),
            float(candidate.mean_inference_ms),
            str(candidate.config_key))


def select_config_per_method(
        candidates: list[CandidateRecord]) -> dict[str, CandidateRecord]:
    """The unique winning config per method (mechanical; losers are kept by the caller)."""
    winners: dict[str, CandidateRecord] = {}
    method_ids = sorted({c.method_id for c in candidates})
    for method_id in method_ids:
        valid = [c for c in candidates if c.method_id == method_id and c.valid]
        if not valid:
            raise ValueError(
                f"method {method_id!r} has no valid candidate "
                f"(all excluded: {[c.exclusion_reason for c in candidates if c.method_id == method_id]})")
        winners[method_id] = min(valid, key=_sort_key)
    return winners


def select_strongest_published_baseline(
        winners: dict[str, CandidateRecord]) -> str:
    """The UNIQUE lowest-validation-AURC winner across methods (frozen forever after)."""
    if not winners:
        raise ValueError("no selected methods — cannot freeze the strongest baseline")
    key = min(winners.values(),
              key=lambda c: (round(float(c.aurc_regret) / TIE_TOLERANCE),
                             round(float(c.regret_at_80) / TIE_TOLERANCE),
                             int(c.param_count),
                             float(c.mean_inference_ms),
                             str(c.method_id)))
    return str(key.method_id)


def write_final_method_selection(path, candidates: list[CandidateRecord],
                                 winners: dict[str, CandidateRecord],
                                 strongest: str) -> None:
    """Create-once freeze of the FULL selection audit trail (winners AND losers)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"final_method_selection.json already exists: {path}")
    if strongest not in winners:
        raise ValueError(f"strongest {strongest!r} is not a selected method")
    payload = {
        "rule": ("per method: lowest validation AURC_REGRET; ties within 1e-12 by lower "
                 "REGRET@80, fewer params, faster inference, lexicographic config key; "
                 "STRONGEST_PUBLISHED_BASELINE = unique lowest-AURC winner across methods "
                 "(same chain, lexicographic method ID last); validation only, the "
                 "comparison holdout did not exist at selection time"),
        "tie_tolerance": TIE_TOLERANCE,
        "strongest_published_baseline": strongest,
        "selected": {m: c.to_dict() for m, c in sorted(winners.items())},
        "candidates": [c.to_dict() for c in sorted(
            candidates, key=lambda c: (c.method_id, c.config_key))],
    }
    write_once_json(path, payload)


__all__ = [
    "CandidateRecord", "TIE_TOLERANCE", "check_candidate_validity",
    "select_config_per_method", "select_strongest_published_baseline",
    "write_final_method_selection",
]


# --- Phase D development freeze orchestration (plan 6.1) -------------------------------------
#
# Long-running entry point: trains every (method, grid config, seed) on the train
# role, evaluates every candidate on validation once, freezes the mechanical
# selections. Terminal plain-text progress (phase, done/total, %, elapsed, ETA,
# at least one update per 30 s, final summary) per the project long-task rule.

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "stage5_sota_comparison_v0.3.json"
DEV_DIR_NAME = "development"
SEARCH_SPACE_NAME = "method_search_space.json"
FINAL_SELECTION_NAME = "final_method_selection.json"
VALIDATION_REPORT_NAME = "validation_comparison_report.json"
VALIDATION_RISKS_NAME = "validation_risks_selected.npz"
FIT_LEDGER_NAME = "fit_receipts.json"
TSTAR_SPLIT = 13  # frozen Stage 4C stratum split (stage4cr2_statistics.TSTAR_SPLIT)


class TerminalProgress:
    """Plain-text long-task progress: phase, done/total, %, elapsed, ETA."""

    def __init__(self, phase: str, total: int):
        self.phase = phase
        self.total = max(int(total), 1)
        self.done = 0
        self.t0 = time.perf_counter()
        self._last = 0.0

    def update(self, note: str = "", force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last < 5.0:
            return
        self._last = now
        elapsed = now - self.t0
        pct = 100.0 * self.done / self.total
        eta = elapsed * (self.total - self.done) / self.done if self.done else 0.0
        print(f"[stage5-dev] {self.phase}: {self.done}/{self.total} ({pct:.1f}%) "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s {note}", flush=True)

    def step(self, note: str = "") -> None:
        self.done += 1
        self.update(note)

    def finish(self, status: str) -> None:
        elapsed = time.perf_counter() - self.t0
        print(f"[stage5-dev] {self.phase}: DONE status={status} "
              f"total={self.done}/{self.total} elapsed={elapsed:.0f}s", flush=True)


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _load_config() -> dict[str, Any]:
    from ac_twin import stage5_protocol as s5prot
    return s5prot.validate_config(
        json.loads(CONFIG_PATH.read_text(encoding="utf-8")))


def _actwin_inventory(sources: dict) -> Any:
    """Rebuild the frozen R2 ModelInventory from hash-verified Stage 5 sources."""
    from ac_twin import stage4cp_features as s4cpf
    from ac_twin import stage4cr2_training as s4cr2t
    from ac_twin import stage5_protocol as s5prot
    from ac_twin.stage1_data import sha256_file
    receipt = json.loads(Path(sources["model_inventory"]).read_text(encoding="utf-8"))
    checkpoint_paths = {int(s): str(sources[f"direct_checkpoint_{s}"])
                        for s in s5prot.MODEL_SEEDS}
    checkpoint_sha = {int(s): sha256_file(sources[f"direct_checkpoint_{s}"])
                      for s in s5prot.MODEL_SEEDS}
    sp = receipt["scaler"]
    return s4cr2t.ModelInventory(
        plan_id=str(receipt["plan_id"]), arch=str(receipt["arch"]),
        seeds=tuple(int(s) for s in receipt["seeds"]),
        param_count=int(receipt["param_count"]),
        run_dir=str(Path(sources["model_inventory"]).parent),
        checkpoint_paths=checkpoint_paths, checkpoint_sha256=checkpoint_sha,
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
        written_at=str(receipt["written_at"]),
        verification="stage5_reload_sha256_verified")


def _score_actwin(inventory: Any, bundle: Any, device: str) -> Any:
    from ac_twin import stage4cr2_training as s4cr2t
    mapping = {"tokens": np.asarray(bundle.tokens, dtype=np.float64),
               "global_features": np.asarray(bundle.global_features, dtype=np.float64),
               "group_seed": np.asarray(bundle.group_seed, dtype=np.int64)}
    return s4cr2t.score_models(inventory, mapping, device=device)


def build_validation_quantities(actwin_scores: Any, sota_ensemble: np.ndarray,
                                sota_by_seed: np.ndarray, bundle: Any,
                                truth: dict, interaction_threshold: float
                                ) -> dict[str, Any]:
    """The power/report quantity mapping on the validation role (truth-joined)."""
    return {
        "group_seed": np.asarray(bundle.group_seed, dtype=np.int64),
        "regret": np.asarray(truth["regret"], dtype=np.float64),
        "y_high": np.asarray(truth["high_regret"], dtype=np.int64),
        "idle_regret": np.asarray(truth["idle_regret"], dtype=np.float64),
        "random_regret": np.asarray(truth["random_regret"], dtype=np.float64),
        "a_star": np.asarray(truth["oracle_action"], dtype=np.int64),
        "tstar_layer": (np.asarray(bundle.strata["t_star"], dtype=np.int64)
                        >= TSTAR_SPLIT).astype(np.int64),
        "interaction_layer": (np.asarray(bundle.strata["interaction_min_distance"],
                                         dtype=np.float64)
                              > float(interaction_threshold)).astype(np.int64),
        "actwin_ensemble": np.asarray(actwin_scores.risk_ensemble, dtype=np.float64),
        "actwin_by_seed": np.asarray(actwin_scores.risk_by_seed, dtype=np.float64),
        "sota_ensemble": np.asarray(sota_ensemble, dtype=np.float64),
        "sota_by_seed": np.asarray(sota_by_seed, dtype=np.float64),
    }


def run_development_freeze(progress: bool = True) -> dict[str, Any]:
    """Phase D (plan 6.1): train, evaluate on validation once, freeze everything.

    Create-once: if ``final_method_selection.json`` exists the freeze is already
    terminal and is reloaded, never recomputed. Resume after a crash reuses the
    per-fit ledger entries (each entry is write-once; a drifted state file is a
    hard stop, never a silent refit).
    """
    from ac_twin import stage4cp_statistics as s4cs
    from ac_twin import stage5_features as s5feat
    from ac_twin import stage5_methods as s5m
    from ac_twin import stage5_power as s5p
    from ac_twin import stage5_protocol as s5prot
    from ac_twin import stage5_statistics as s5st
    from ac_twin import stage5_training as s5t
    from ac_twin.stage1_data import sha256_file

    checked = _load_config()
    dev_dir = ROOT / checked["stage_root"] / DEV_DIR_NAME
    models_dir = dev_dir / "models"
    final_path = dev_dir / FINAL_SELECTION_NAME
    report_path = dev_dir / VALIDATION_REPORT_NAME
    risks_path = dev_dir / VALIDATION_RISKS_NAME
    ledger_path = models_dir / FIT_LEDGER_NAME

    if final_path.exists():
        print("[stage5-dev] final_method_selection.json exists (create-once); "
              "the development freeze is already terminal — reloading", flush=True)
        return json.loads(final_path.read_text(encoding="utf-8"))

    t_start = time.perf_counter()
    sources = s5prot.resolve_sources(checked)
    device = str(checked.get("device", "cuda"))
    try:
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"
    seeds = [int(s) for s in checked["model_seeds"]]

    phase = TerminalProgress("load bundles/truth", 4)
    train_bundle, _ = s5feat.build_feature_context(sources, "train")
    train_truth = s5feat.compute_group_truth(sources, "train")
    phase.step("train")
    val_bundle, val_extras = s5feat.build_feature_context(sources, "validation")
    val_truth = s5feat.compute_group_truth(sources, "validation")
    phase.step("validation")
    if not (bool(train_bundle.valid_mask.all()) and bool(val_bundle.valid_mask.all())):
        raise RuntimeError("non-valid groups present — the R2 roles are fully valid; "
                           "any invalid group is a contract drift (fail closed)")
    phase.finish("bundles_ok")

    # --- AC-Twin validation scoring + bitwise parity with the frozen R2 gate ----
    print(f"[stage5-dev] scoring AC-Twin on validation (device={device}) + "
          "R2-gate parity check", flush=True)
    inventory = _actwin_inventory(sources)
    gate = json.loads((Path(sources["model_inventory"]).parent
                       / "validation_gate_report.json").read_text(encoding="utf-8"))
    actwin = None
    for candidate_device in (device, "cpu"):
        scores = _score_actwin(inventory, val_bundle, candidate_device)
        m = s5st.evaluate_group_metrics(scores.risk_ensemble, scores.risk_by_seed,
                                        val_truth)
        d_aurc = abs(m["aurc_regret"] - float(gate["metrics"]["aurc_regret_ensemble"]))
        d_r80 = abs(m["regret_at_80"] - float(gate["metrics"]["regret80_ensemble"]))
        if d_aurc <= 1e-12 and d_r80 <= 1e-12:
            actwin = scores
            print(f"[stage5-dev] parity OK on device={candidate_device} "
                  f"(|dAURC|={d_aurc:.2e}, |dR80|={d_r80:.2e})", flush=True)
            break
        print(f"[stage5-dev] parity miss on device={candidate_device} "
              f"(|dAURC|={d_aurc:.2e}, |dR80|={d_r80:.2e}); trying next caliber",
              flush=True)
    if actwin is None:
        raise RuntimeError("AC-Twin validation scoring cannot reproduce the frozen R2 "
                           "gate quantities on any device caliber (fail closed)")

    # --- frozen search space (create-once, BEFORE any training) ------------------
    dev_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    space = {"plan_id": checked["plan_id"],
             "frozen_before_training": True,
             "methods": {mid: {"grid": [dict(c) for c in spec.grid],
                               "param_counts": [s5m.method_param_count(mid, c)
                                                for c in spec.grid]}
                         for mid, spec in s5m.METHOD_SPECS.items()}}
    write_once_json(dev_dir / SEARCH_SPACE_NAME, space)

    # --- train all (method, config, seed) -----------------------------------------
    total_fits = sum(len(spec.grid) for spec in s5m.METHOD_SPECS.values()) * len(seeds)
    ledger: dict[str, Any] = {}
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        print(f"[stage5-dev] resuming with {len(ledger)} ledgered fits", flush=True)
    receipts: dict[tuple[str, int, int], s5t.MethodReceipt] = {}
    fit_phase = TerminalProgress("fit methods", total_fits)
    for method_id, spec in s5m.METHOD_SPECS.items():
        for grid_index, config in enumerate(spec.grid):
            out_dir = models_dir / method_id / f"grid{grid_index}"
            for seed in seeds:
                key = f"{method_id}|grid{grid_index}|{seed}"
                entry = ledger.get(key)
                if entry is not None:
                    state_path = Path(entry["state_path"])
                    if not state_path.is_file() \
                            or sha256_file(state_path) != entry["state_sha256"]:
                        raise RuntimeError(f"ledgered state drifted for {key}: {state_path}")
                    receipts[(method_id, grid_index, seed)] = s5t.MethodReceipt(**entry)
                else:
                    receipt = s5t.fit_method(
                        method_id, train_bundle, train_truth, config, seed, out_dir,
                        device=device)
                    ledger[key] = dataclasses.asdict(receipt)
                    tmp = ledger_path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(ledger, indent=1), encoding="utf-8")
                    tmp.replace(ledger_path)
                    receipts[(method_id, grid_index, seed)] = receipt
                fit_phase.step(key)
    fit_phase.finish("all_fits_complete")

    # --- one-shot validation evaluation of every candidate -------------------------
    candidates: list[CandidateRecord] = []
    metrics_by_key: dict[str, Any] = {}
    risks_by_key: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    raw_by_method: dict[str, Any] = {}
    eval_total = sum(len(spec.grid) for spec in s5m.METHOD_SPECS.values())
    eval_phase = TerminalProgress("validation evaluation", eval_total)
    for method_id, spec in s5m.METHOD_SPECS.items():
        for grid_index, config in enumerate(spec.grid):
            key = f"{method_id}#grid{grid_index}"
            seed_receipts = [receipts[(method_id, grid_index, s)] for s in seeds]
            t0 = time.perf_counter()
            by_seed = np.stack([s5t.score_method(r, val_bundle, device=device)
                                for r in seed_receipts])
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            ensemble = by_seed.mean(axis=0)
            valid, reason = check_candidate_validity(by_seed)
            metrics = s5st.evaluate_group_metrics(ensemble, by_seed, val_truth)
            metrics_by_key[key] = metrics
            risks_by_key[key] = (ensemble, by_seed)
            raws = [s5t.raw_outputs_method(r, val_bundle, device=device)
                    for r in seed_receipts]
            if raws[0] is not None:
                raw_by_method[key] = {k: np.mean([r[k] for r in raws], axis=0)
                                      for k in raws[0]}
            candidates.append(CandidateRecord(
                method_id=method_id, config_key=f"grid{grid_index}",
                config=dict(config),
                aurc_regret=float(metrics["aurc_regret"]),
                regret_at_80=float(metrics["regret_at_80"]),
                param_count=int(seed_receipts[0].param_count),
                mean_inference_ms=float(elapsed_ms / val_bundle.group_seed.size),
                valid=bool(valid), exclusion_reason=reason))
            eval_phase.step(key)
    eval_phase.finish("validation_evaluated")

    # --- mechanical freeze -----------------------------------------------------------
    winners = select_config_per_method(candidates)
    strongest = select_strongest_published_baseline(winners)
    write_final_method_selection(final_path, candidates, winners, strongest)
    print(f"[stage5-dev] STRONGEST_PUBLISHED_BASELINE = {strongest} "
          f"(config {winners[strongest].config_key}, "
          f"validation AURC {winners[strongest].aurc_regret:.6f})", flush=True)

    # --- validation comparison report (create-once) ------------------------------------
    from ac_twin import stage4cr2_protocol as s4cr2
    interaction_threshold = float(json.loads(
        Path(sources["final_selection"]).read_text(encoding="utf-8"))
        ["interaction_threshold"])
    actwin_metrics = s5st.evaluate_group_metrics(
        actwin.risk_ensemble, actwin.risk_by_seed, val_truth)
    baseline_metrics = {}
    for name in s4cr2.STRONGEST_BASELINE_CANDIDATES:
        risk = np.asarray(val_extras["baselines"][name], dtype=np.float64)
        tiled = np.tile(risk, (len(seeds), 1))
        baseline_metrics[name] = s5st.evaluate_group_metrics(risk, tiled, val_truth)

    skey = f"{strongest}#{winners[strongest].config_key}"
    sota_ensemble, sota_by_seed = risks_by_key[skey]
    quantities = build_validation_quantities(actwin, sota_ensemble, sota_by_seed,
                                             val_bundle, val_truth,
                                             interaction_threshold)

    def vector_fn(index: np.ndarray) -> np.ndarray:
        return s5p.compute_quantity_vector(index, quantities)

    boot = s4cs.paired_group_bootstrap(
        vector_fn, int(val_bundle.group_seed.size),
        n_resamples=int(checked["power_bootstrap_draws"]),
        seed=int(checked["power_bootstrap_entropy"]), confidence=0.95)
    q05 = np.quantile(boot["draws"], 0.05, axis=0)
    q95 = np.quantile(boot["draws"], 0.95, axis=0)
    point = boot["point"]
    material = s5p.check_materiality(point[s5p.Q_I_AURC], point[s5p.Q_I_REGRET80])
    tost = {
        "i_aurc": s5st.tost_equivalent(q05[s5p.Q_I_AURC], q95[s5p.Q_I_AURC],
                                       tuple(checked["tost_i_aurc"])),
        "i_regret80": s5st.tost_equivalent(q05[s5p.Q_I_REGRET80], q95[s5p.Q_I_REGRET80],
                                           tuple(checked["tost_i_regret80"])),
        "label": "DESCRIPTIVE on validation; the binding TOST is the holdout adjudication",
    }

    # persist the frozen validation risk/truth arrays for the power stage (create-once)
    if risks_path.exists():
        raise FileExistsError(f"{risks_path} already exists (create-once)")
    npz_payload = {
        "group_seed": quantities["group_seed"],
        "regret": quantities["regret"], "y_high": quantities["y_high"],
        "idle_regret": quantities["idle_regret"],
        "random_regret": quantities["random_regret"], "a_star": quantities["a_star"],
        "tstar_layer": quantities["tstar_layer"],
        "interaction_layer": quantities["interaction_layer"],
        "actwin_ensemble": quantities["actwin_ensemble"],
        "actwin_by_seed": quantities["actwin_by_seed"],
    }
    for method_id, winner in sorted(winners.items()):
        ens, by_seed = risks_by_key[f"{method_id}#{winner.config_key}"]
        npz_payload[f"method_ensemble__{method_id}"] = ens
        npz_payload[f"method_by_seed__{method_id}"] = by_seed
    tmp_npz = risks_path.with_suffix(".tmp.npz")
    np.savez(tmp_npz, **npz_payload)
    tmp_npz.replace(risks_path)

    calibration = {}
    for method_id, winner in sorted(winners.items()):
        key = f"{method_id}#{winner.config_key}"
        calibration[method_id] = s5st.native_calibration(
            method_id, raw_by_method.get(key), val_truth)

    report = {
        "plan_id": checked["plan_id"],
        "n_validation_groups": int(val_bundle.group_seed.size),
        "scope": "ADAPTIVE_DEVELOPMENT_ONLY (validation; the comparison holdout does "
                 "not exist yet)",
        "strongest_published_baseline": strongest,
        "selected": {m: c.to_dict() for m, c in sorted(winners.items())},
        "candidates": [c.to_dict() for c in sorted(
            candidates, key=lambda c: (c.method_id, c.config_key))],
        "candidate_metrics": {k: _jsonify(v) for k, v in sorted(metrics_by_key.items())},
        "actwin_metrics": _jsonify(actwin_metrics),
        "internal_baseline_metrics": _jsonify(baseline_metrics),
        "native_calibration_selected": _jsonify(calibration),
        "comparison_vs_strongest": {
            "quantity_names": list(s5p.QUANTITY_NAMES),
            "point": _jsonify(point),
            "ci95_lower": _jsonify(boot["ci_lower"]),
            "ci95_upper": _jsonify(boot["ci_upper"]),
            "q05": _jsonify(q05), "q95": _jsonify(q95),
            "bootstrap": {"n_resamples": int(checked["power_bootstrap_draws"]),
                          "seed": int(checked["power_bootstrap_entropy"]),
                          "confidence": 0.95,
                          "pairing": "whole-group; same groups, same selected action, "
                                     "same model-seed index"},
        },
        "materiality": {
            "shrink": float(checked["materiality_shrink"]),
            "i_aurc_min": float(checked["materiality_i_aurc_min"]),
            "i_regret80_min": float(checked["materiality_i_regret80_min"]),
            "point_i_aurc": float(point[s5p.Q_I_AURC]),
            "point_i_regret80": float(point[s5p.Q_I_REGRET80]),
            "material": bool(material),
            "terminal_if_not_material": s5p.STATUS_NOT_MATERIAL,
        },
        "tost_validation_descriptive": tost,
        "bindings": {
            "config_sha256": sha256_file(CONFIG_PATH),
            "r2_final_selection_sha256": sha256_file(sources["final_selection"]),
            "method_search_space_sha256": sha256_file(dev_dir / SEARCH_SPACE_NAME),
            "final_method_selection_sha256": sha256_file(final_path),
            "validation_risks_sha256": sha256_file(risks_path),
            "fit_ledger_sha256": sha256_file(ledger_path),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(report_path, report)
    total = time.perf_counter() - t_start
    print(f"[stage5-dev] development freeze COMPLETE: strongest={strongest}, "
          f"material={material}, i_aurc={point[s5p.Q_I_AURC]:.4f}, "
          f"i_regret80={point[s5p.Q_I_REGRET80]:.4f}, elapsed={total:.0f}s",
          flush=True)
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    report = run_development_freeze()
    material = report.get("materiality", {}).get("material")
    print(f"[stage5-dev] summary: strongest={report.get('strongest_published_baseline')} "
          f"material={material}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
