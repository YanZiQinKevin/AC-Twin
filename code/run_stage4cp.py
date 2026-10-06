"""Stage 4C Plus orchestration runner.

Plan ``AC-TWIN-RFER-STAGE4C-PLUS-DECISION-AWARE-RISK-2026-09-16-V0.1`` section 13. The
runner ONLY orchestrates: it wires the collector, the grouped frozen inference (reused
bit-identically from Stage 4C), the feature/model/training modules, the statistics and
the power module. It never re-implements their science. One-shot resources (the
development screen, the confirm set, the locks, the create-once reports) are never
overwritten, and a second adjudication is refused.

Subcommands: preflight, phase0, lock-development, smoke, collect-train-validation,
train-candidates, select-finalist, collect-screen, infer-screen, adjudicate-screen,
power, lock-confirm, collect-confirm, infer-confirm, adjudicate-confirm, verify, run-all.
"""
from __future__ import annotations

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

import run_stage4a as s4a  # noqa: E402  (frozen pre-intervention geometry helper)
from ac_twin import stage4c_objective as s4co  # noqa: E402
from ac_twin import stage4cp_features as s4cf  # noqa: E402
from ac_twin import stage4cp_models as s4cm  # noqa: E402
from ac_twin import stage4cp_protocol as s4cp  # noqa: E402
from ac_twin import stage4cp_statistics as s4cs  # noqa: E402
from ac_twin.checkpointing import atomic_save_npz  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402

ROOT = PROJECT_ROOT
CONFIG_PATH = ROOT / "experiments/stage4_reliability/stage4c_plus/stage4cp_config.json"

# Phase 0 quantity-vector layout (bootstrap CI): [I_AURC, I_REGRET80, PR_c, PR_margin].
_P0_I_AURC = 0
_P0_I_REGRET80 = 1
_P0_PR_CAND = 2
_P0_PR_MARGIN = 3


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value)


def _load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def _run_dir(config: Mapping[str, Any]) -> Path:
    return _resolve(config["run_dir"])


# --- preflight ------------------------------------------------------------------------


def cmd_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify frozen sources, model lock, both seed schedules and zero overlap."""
    checked = s4cp.validate_config(config)
    sources = s4cp.inventory_frozen_sources()
    model_lock = s4cp.inventory_model_lock(checked)
    historical = s4cp.scan_historical_group_seeds(
        [_resolve(checked["historical_data_root"]), ROOT / "experiments"])
    dev = s4cp.make_group_seed_schedule(s4cp.DEV_SEED_ENTROPY, s4cp.DEV_SCHEDULE_SIZE)
    confirm = s4cp.make_group_seed_schedule(s4cp.CONFIRM_SEED_ENTROPY, s4cp.CONFIRM_SCHEDULE_SIZE)
    from ac_twin.stage2_protocol import validate_seed_schedule
    validate_seed_schedule(dev, historical)
    validate_seed_schedule(confirm, historical)
    if np.intersect1d(dev, confirm).size:
        raise ValueError("dev and confirm schedules overlap")
    return {
        "status": "PREFLIGHT_OK",
        "n_frozen_sources": len(sources["stage4_sha256"]),
        "n_checkpoints": len(model_lock["checkpoint_sha256"]),
        "n_historical_episode_seeds": int(historical.size),
        "dev_schedule_sha256": hashlib.sha256(dev.tobytes()).hexdigest(),
        "confirm_schedule_sha256": hashlib.sha256(confirm.tobytes()).hexdigest(),
        "device": str(checked["device"]),
    }


# --- Phase 0: one exploratory OOF feasibility screen on the old 500-group selection ------


def _load_phase0_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load the Stage 4C selection data + frozen predictions + action risks (Phase 0 only).

    These are the ONLY Stage 4C products Phase 0 may read (plan section 5.1). The screen
    products are permanently forbidden; the guard enforces this.
    """
    checked = s4cp.validate_config(config)
    data_path = _resolve(checked["phase0_selection_data"])
    pred_path = _resolve(checked["phase0_predictions"])
    risk_path = _resolve(checked["phase0_risk_scores"])
    for p in (data_path, pred_path, risk_path):
        s4cp.assert_allowed_stage4cp_data(p, allow_phase0_selection=True)
    with np.load(data_path, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    with np.load(pred_path, allow_pickle=True) as z:
        predictions = {k: z[k] for k in z.files}
    with np.load(risk_path, allow_pickle=True) as z:
        risks = {k: z[k] for k in z.files}
    return {
        "data": data, "predictions": predictions, "risks": risks,
        "data_sha256": sha256_file(data_path),
        "predictions_sha256": sha256_file(pred_path),
        "risk_scores_sha256": sha256_file(risk_path),
    }


def _phase0_true_regret(data: Mapping[str, np.ndarray], j_hat: np.ndarray) -> np.ndarray:
    """True decision regret from the frozen consequence function (selection truth is legal).

    Phase 0 reads selection truth because the 500-group selection is NOT a one-shot
    adjudication set (plan section 5.1); it is never reused after Phase 0.
    """
    counterfactual_xy = np.asarray(data["object_counterfactual_futures"], dtype=np.float64)[..., 2:4]
    history_last_xy = np.asarray(data["object_history"], dtype=np.float64)[:, -1, 0, 2:4]
    inference_mask = np.asarray(data["inference_mask"], dtype=bool)
    groups, n_actions = counterfactual_xy.shape[0], counterfactual_xy.shape[1]
    hist_b = np.broadcast_to(history_last_xy[:, None, :], (groups, n_actions, 2))
    mask_b = np.broadcast_to(inference_mask[:, None, :, :], (groups, n_actions) + inference_mask.shape[1:])
    comp_true = s4co.trajectory_components(counterfactual_xy, hist_b, mask_b)
    j_true = s4co.total_cost(comp_true["interaction"], s4co.efficiency_cost(comp_true["progress"]))
    rows = np.arange(groups)
    a_hat = np.argmin(j_hat, axis=-1)
    a_star = np.argmin(j_true, axis=-1)
    return s4co.clamp_regret(j_true[rows, a_hat] - j_true[rows, a_star])


def _phase0_quantity_vector(
    index: np.ndarray,
    cand_score: np.ndarray,
    margin_score: np.ndarray,
    regret: np.ndarray,
    high_label: np.ndarray,
) -> np.ndarray:
    cand = cand_score[index]
    marg = margin_score[index]
    reg = regret[index]
    lab = high_label[index]
    order_c = s4cs.order_by_risk(cand)
    order_m = s4cs.order_by_risk(marg)
    a_c = s4cs.aurc_regret(order_c, reg)
    a_m = s4cs.aurc_regret(order_m, reg)
    r80_c = s4cs.regret_at_coverage(order_c, reg, s4cp.COVERAGE_MAIN)
    r80_m = s4cs.regret_at_coverage(order_m, reg, s4cp.COVERAGE_MAIN)
    return np.array([
        s4cs.improvement(a_m, a_c),
        s4cs.improvement(r80_m, r80_c),
        s4cs.high_regret_pr(cand, lab),
        s4cs.high_regret_pr(marg, lab),
    ])


def _phase0_fold_assignment(n_groups: int, n_folds: int, entropy: int) -> np.ndarray:
    """Deterministic whole-group fold assignment (group never crosses folds)."""
    generator = np.random.default_rng(int(entropy))
    permutation = generator.permutation(int(n_groups))
    folds = np.empty(int(n_groups), dtype=np.int64)
    splits = np.array_split(permutation, int(n_folds))
    for fold_id, idx in enumerate(splits):
        folds[idx] = fold_id
    return folds


def _phase0_oof_candidate(
    arch: str,
    tokens: np.ndarray,
    global_features: np.ndarray,
    token_mask: np.ndarray,
    y_reg: np.ndarray,
    y_high: np.ndarray,
    folds: np.ndarray,
    config: Mapping[str, Any],
    *,
    hidden_dim: int,
    lambda_cls: float,
    init_seeds: tuple[int, ...],
    direct_width: int | None,
) -> dict[str, np.ndarray]:
    """Out-of-fold risk scores for one architecture (3-seed mean per held-out group).

    Each group's score comes from models trained ONLY on the other folds (no train-fold
    re-entry). Scaler and class weight are refit per outer fold on that fold's train part.
    """
    n = tokens.shape[0]
    n_folds = int(folds.max()) + 1
    per_seed_scores = np.full((len(init_seeds), n), np.nan, dtype=np.float64)
    for fold_id in range(n_folds):
        test_idx = np.where(folds == fold_id)[0]
        train_idx = np.where(folds != fold_id)[0]
        if np.intersect1d(test_idx, train_idx).size:
            raise RuntimeError("Phase 0 train-fold re-entry detected")
        scaler = s4cf.FrozenFeatureScaler.fit(tokens[train_idx], global_features[train_idx])
        x_tok_tr = scaler.transform_tokens(tokens[train_idx])
        x_glob_tr = scaler.transform_global(global_features[train_idx])
        x_tok_te = scaler.transform_tokens(tokens[test_idx])
        x_glob_te = scaler.transform_global(global_features[test_idx])
        for s_i, seed in enumerate(init_seeds):
            ckpt = s4cm.fit_decision_risk_model(
                arch, x_tok_tr, x_glob_tr, token_mask[train_idx],
                y_reg[train_idx], y_high[train_idx],
                hidden_dim=hidden_dim, lambda_cls=lambda_cls, seed=int(seed),
                config=config, direct_width=direct_width)
            scores = s4cm.predict_decision_risk(
                ckpt, x_tok_te, x_glob_te, token_mask[test_idx],
                device="cpu")
            per_seed_scores[s_i, test_idx] = scores["risk_score"]
    if not np.isfinite(per_seed_scores).all():
        raise RuntimeError(f"Phase 0 OOF produced non-finite score for {arch}")
    return {
        "oof_score": per_seed_scores.mean(axis=0),   # (n,) 3-seed mean
        "oof_score_by_seed": per_seed_scores,        # (S,n)
    }


def _phase0_evaluate_candidate(
    arch: str,
    oof_score: np.ndarray,
    margin_score: np.ndarray,
    regret: np.ndarray,
    high_label: np.ndarray,
    folds: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the six Phase 0 Go/No-Go conditions (plan section 5.3) to one candidate."""
    checked = s4cp.validate_config(config)
    n = oof_score.shape[0]
    boot = s4cs.paired_group_bootstrap(
        lambda idx: _phase0_quantity_vector(idx, oof_score, margin_score, regret, high_label),
        n, n_resamples=int(checked["phase0_bootstrap_resamples"]),
        seed=int(checked["phase0_bootstrap_entropy"]),
        confidence=float(checked["confidence_screen"]))
    point, ci_lower = boot["point"], boot["ci_lower"]

    # condition 4: per-fold I_AURC direction (held-out fold only, vs MARGIN_ONLY).
    n_folds = int(folds.max()) + 1
    fold_i_aurc = []
    for fold_id in range(n_folds):
        idx = np.where(folds == fold_id)[0]
        a_c = s4cs.aurc_regret(s4cs.order_by_risk(oof_score[idx]), regret[idx])
        a_m = s4cs.aurc_regret(s4cs.order_by_risk(margin_score[idx]), regret[idx])
        fold_i_aurc.append(s4cs.improvement(a_m, a_c))
    folds_positive = int(sum(1 for v in fold_i_aurc if v > 0.0))

    # condition 5: trim 1% / 5% aggregate I_AURC.
    trims = {}
    for frac in s4cp.GATE_TRIM_FRACTIONS:
        keep = s4cs.trim_high_regret(regret, frac)
        a_c = s4cs.aurc_regret(s4cs.order_by_risk(oof_score[keep]), regret[keep])
        a_m = s4cs.aurc_regret(s4cs.order_by_risk(margin_score[keep]), regret[keep])
        trims[f"{frac:.2f}"] = s4cs.improvement(a_m, a_c)

    conditions = {
        "i_aurc_regret": bool(point[_P0_I_AURC] >= s4cp.PHASE0_I_AURC_MIN
                              and ci_lower[_P0_I_AURC] > 0.0),
        "i_regret80": bool(point[_P0_I_REGRET80] >= s4cp.PHASE0_I_REGRET80_MIN
                           and ci_lower[_P0_I_REGRET80] > 0.0),
        "pr_auc": bool(point[_P0_PR_CAND] > 0.0 and point[_P0_PR_CAND] >= point[_P0_PR_MARGIN]),
        "folds_positive": bool(folds_positive >= s4cp.PHASE0_MIN_FOLDS_POSITIVE),
        "trims_positive": bool(all(v > 0.0 for v in trims.values())),
    }
    return {
        "arch": arch,
        "point": {"i_aurc_regret": float(point[_P0_I_AURC]),
                  "i_regret80": float(point[_P0_I_REGRET80]),
                  "pr_auc_minus_prevalence_cand": float(point[_P0_PR_CAND]),
                  "pr_auc_minus_prevalence_margin": float(point[_P0_PR_MARGIN])},
        "ci_lower": {"i_aurc_regret": float(ci_lower[_P0_I_AURC]),
                     "i_regret80": float(ci_lower[_P0_I_REGRET80])},
        "fold_i_aurc": [float(v) for v in fold_i_aurc],
        "folds_positive": folds_positive,
        "trim_i_aurc": {k: float(v) for k, v in trims.items()},
        "conditions": conditions,
        "passed": bool(all(conditions.values())),
        "bootstrap_resamples": int(checked["phase0_bootstrap_resamples"]),
        "confidence": float(checked["confidence_screen"]),
    }


def run_phase0(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the one exploratory 5-fold OOF feasibility screen (plan section 5).

    Marked ``EXPLORATORY_OOF_NOT_CONFIRMATORY``. Reads ONLY the Stage 4C selection
    products; never the screen. Writes create-once ``phase0_report.json`` +
    ``oof_predictions.npz`` + ``fold_assignment.npy``; on a pass also writes
    ``phase0_pass_receipt.json`` naming the retained architectures.
    """
    checked = s4cp.validate_config(config)
    phase0_dir = _run_dir(checked) / "phase0"
    phase0_dir.mkdir(parents=True, exist_ok=True)
    report_path = phase0_dir / "phase0_report.json"
    if report_path.exists():
        raise FileExistsError("phase0_report.json already exists (create-once)")

    inputs = _load_phase0_inputs(checked)
    data, predictions, risks = inputs["data"], inputs["predictions"], inputs["risks"]

    object_history = np.asarray(data["object_history"], dtype=np.float32)
    factual_xy = np.asarray(data["object_factual_future"], dtype=np.float64)[..., 2:4]
    history_last_xy = object_history[:, -1, 0, 2:4].astype(np.float64)
    inference_mask = np.asarray(data["inference_mask"], dtype=bool)
    t_star = np.asarray(data["t_star"], dtype=np.int64)
    group_seed = np.asarray(data["episode_seed"], dtype=np.int64)
    geometry = s4a._pre_intervention_geometry(object_history)
    action_risks = {k: np.asarray(risks[k], dtype=np.float64) for k in s4cp.RISK_CLASSES}

    bundle = s4cf.build_relative_tokens(
        predictions["e2e"], factual_xy, history_last_xy, inference_mask,
        action_risks, geometry, t_star, group_seed)
    j_hat = bundle["j_hat"]
    regret = _phase0_true_regret(data, j_hat)
    y_reg = np.log1p(regret / s4cp.HIGH_REGRET_THRESHOLD)
    y_high = (regret > s4cp.HIGH_REGRET_THRESHOLD).astype(np.int64)

    # Restrict to feature-valid groups (no zero-padding; plan section 3.2).
    valid = np.asarray(bundle["valid_mask"], dtype=bool)
    keep = np.where(valid)[0]
    tokens = bundle["tokens"][keep]
    global_features = bundle["global_features"][keep]
    token_mask = bundle["token_mask"][keep]
    regret_v = regret[keep]
    y_reg_v = y_reg[keep]
    y_high_v = y_high[keep]
    group_seed_v = group_seed[keep]
    margin_score = bundle["baselines"][s4cp.BASELINE_MARGIN][keep]
    n_valid = int(keep.size)

    folds = _phase0_fold_assignment(n_valid, int(checked["phase0_n_folds"]),
                                    int(checked["phase0_fold_entropy"]))

    # Direct width mechanically matches the DCR param count at the Phase 0 hidden dim.
    dcr_params = s4cm.count_parameters(s4cm.build_model(
        s4cp.ARCH_DCR, hidden_dim=int(checked["phase0_hidden_dim"]),
        dropout=float(checked["dropout"])))
    direct_width, direct_params = s4cm.select_direct_width(dcr_params)
    param_audit = {
        "dcr_param_count": int(dcr_params),
        "direct_width": int(direct_width),
        "direct_param_count": int(direct_params),
        "rel_diff": float(abs(direct_params - dcr_params) / dcr_params),
        "tolerance": float(checked["param_count_tolerance"]),
        "within_tolerance": bool(abs(direct_params - dcr_params) / dcr_params
                                 <= float(checked["param_count_tolerance"])),
    }

    results: dict[str, Any] = {}
    oof_store: dict[str, np.ndarray] = {}
    for arch in (s4cp.ARCH_DCR, s4cp.ARCH_DIRECT):
        width = None if arch == s4cp.ARCH_DCR else direct_width
        oof = _phase0_oof_candidate(
            arch, tokens, global_features, token_mask, y_reg_v, y_high_v, folds, checked,
            hidden_dim=int(checked["phase0_hidden_dim"]),
            lambda_cls=float(checked["phase0_lambda_cls"]),
            init_seeds=tuple(int(s) for s in checked["phase0_init_seeds"]),
            direct_width=width)
        oof_store[f"oof_score__{arch}"] = oof["oof_score"]
        for s_i, seed in enumerate(checked["phase0_init_seeds"]):
            oof_store[f"oof_score__{arch}__seed{int(seed)}"] = oof["oof_score_by_seed"][s_i]
        results[arch] = _phase0_evaluate_candidate(
            arch, oof["oof_score"], margin_score, regret_v, y_high_v, folds, checked)

    dcr_pass = results[s4cp.ARCH_DCR]["passed"]
    direct_pass = results[s4cp.ARCH_DIRECT]["passed"]
    if dcr_pass:
        retained = [s4cp.ARCH_DCR, s4cp.ARCH_DIRECT]  # Direct mandatory same-info control
    elif direct_pass:
        retained = [s4cp.ARCH_DIRECT]
    else:
        retained = []
    status = ("PHASE0_PASSED" if retained else s4cp.STATUS_FEASIBILITY_NOT_SUPPORTED)

    # persist OOF artefacts (create-once).
    fold_path = phase0_dir / "fold_assignment.npy"
    fold_tmp = fold_path.with_name(fold_path.name + ".tmp")
    with open(fold_tmp, "wb") as handle:
        np.save(handle, folds)
    Path(fold_tmp).replace(fold_path)
    atomic_save_npz(phase0_dir / "oof_predictions.npz",
                    group_seed=group_seed_v, regret=regret_v, y_high=y_high_v,
                    margin_score=margin_score, **oof_store)

    report = {
        "plan_id": s4cp.PLAN_ID,
        "marker": "EXPLORATORY_OOF_NOT_CONFIRMATORY",
        "role": "phase0_oof_on_stage4c_development_selection",
        "status": status,
        "n_selection_groups": int(len(group_seed)),
        "n_valid_groups": n_valid,
        "n_dropped_non_finite": int(len(group_seed) - n_valid),
        "selection_data_sha256": inputs["data_sha256"],
        "predictions_sha256": inputs["predictions_sha256"],
        "risk_scores_sha256": inputs["risk_scores_sha256"],
        "high_regret_threshold": s4cp.HIGH_REGRET_THRESHOLD,
        "n_folds": int(checked["phase0_n_folds"]),
        "init_seeds": [int(s) for s in checked["phase0_init_seeds"]],
        "hidden_dim": int(checked["phase0_hidden_dim"]),
        "lambda_cls": float(checked["phase0_lambda_cls"]),
        "param_audit": param_audit,
        "feature_schema": s4cf.feature_schema(),
        "candidates": results,
        "retained_architectures": retained,
        "phase0_code_sha256": s4cp.phase0_code_hashes(),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(report_path, report)

    if retained:
        receipt = {
            "plan_id": s4cp.PLAN_ID,
            "status": status,
            "retained_architectures": retained,
            "direct_is_mandatory_control": bool(dcr_pass),
            "param_audit": param_audit,
            "phase0_report_sha256": sha256_file(report_path),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        write_once_json(phase0_dir / "phase0_pass_receipt.json", receipt)
    return report


def cmd_phase0(config: Mapping[str, Any]) -> dict[str, Any]:
    return run_phase0(config)


# --- not-yet-implemented stages (filled in Tasks 2-5) --------------------------------------


def _not_implemented(name: str) -> None:
    raise NotImplementedError(f"Stage 4C Plus subcommand {name!r} is implemented in a later task")


# --- verify / run-all ------------------------------------------------------------------------


def cmd_verify(config: Mapping[str, Any]) -> dict[str, Any]:
    return cmd_preflight(config)


def cmd_run_all(config_path: str | Path) -> dict[str, Any]:
    config = _load_config(config_path)
    log: dict[str, Any] = {}
    log["preflight"] = cmd_preflight(config)
    log["phase0"] = cmd_phase0(config)
    if log["phase0"]["status"] != "PHASE0_PASSED":
        log["halted"] = log["phase0"]["status"]
        return log
    log["halted"] = "AWAITING_LATER_TASKS"
    return log


_COMMANDS = (
    "preflight", "phase0", "lock-development", "smoke", "collect-train-validation",
    "train-candidates", "select-finalist", "collect-screen", "infer-screen",
    "adjudicate-screen", "power", "lock-confirm", "collect-confirm", "infer-confirm",
    "adjudicate-confirm", "verify", "run-all",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4C Plus decision-aware risk runner")
    parser.add_argument("command", choices=_COMMANDS)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--selected-n", type=int, default=None, help="required for lock-confirm")
    args = parser.parse_args(argv)
    config = _load_config(args.config)

    if args.command == "preflight":
        out = cmd_preflight(config)
    elif args.command == "phase0":
        out = cmd_phase0(config)
    elif args.command == "verify":
        out = cmd_verify(config)
    elif args.command == "run-all":
        out = cmd_run_all(args.config)
    elif args.command == "lock-confirm" and args.selected_n is None:
        raise ValueError("lock-confirm requires --selected-n")
    else:
        _not_implemented(args.command)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
