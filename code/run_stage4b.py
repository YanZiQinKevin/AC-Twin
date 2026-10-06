"""Stage 4B independent Merge risk confirmation driver.

Staged entry point (plan ``AC-TWIN-RFER-STAGE4B-RISK-CONFIRM-2026-09-15-V0.1``
section 10). It mirrors the Stage 4A driver's small-phase structure but has NO
fit-risk / select-candidate step: the finalist, baselines, threshold, coverage and
features are all frozen upstream. It only dispatches preflight / power /
lock-protocol / smoke / infer-confirm / adjudicate-confirm; model and statistics
implementations live in the ``ac_twin.stage4b_*`` modules.

State machine (plan section 10)::

    DRAFT -> SOFTWARE_READY -> POWERED -> PROTOCOL_LOCKED -> CONFIRM_COLLECTED
          -> CONFIRM_INFERENCE_COMPLETE -> CONFIRM_ADJUDICATED

The confirmation set is single-use. Gate Q integrity checks that need no truth label
run BEFORE ``confirmation_used.lock`` is burned (a failure there is recoverable and
does NOT consume the one-shot resource); the lock is burned once with ``O_EXCL`` and
only then is the true ``risk_target`` constructed and the single 10,000-fold paired
bootstrap adjudicated.
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

import run_stage4a as s4a  # noqa: E402  (frozen inference primitives)
from ac_twin import stage4_risk as s4r  # noqa: E402
from ac_twin import stage4_statistics as s4s  # noqa: E402
from ac_twin import stage4b_inference as s4bi  # noqa: E402
from ac_twin import stage4b_power as s4bpw  # noqa: E402
from ac_twin import stage4b_protocol as s4bp  # noqa: E402
from ac_twin import stage4b_statistics as s4bs  # noqa: E402
from ac_twin.progress import ProgressReporter  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402

SMOKE_STATUS = "ENGINEERING_SMOKE_ONLY"
POSITION_SLICE = s4a.POSITION_SLICE


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _load_config(path: str | Path) -> dict[str, Any]:
    return s4bp.validate_stage4b_config(json.loads(Path(path).read_text(encoding="utf-8")))


# --- status / handoff -----------------------------------------------------------


def _write_status(run_dir: Path, state: str, *, note: str, blocking: str = "none") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    status = (
        f"# Stage 4B 状态\n\n"
        f"> 当前状态：`{state}`\n> 计划：`{s4bp.PLAN_ID}`\n> 更新：{time.strftime('%Y-%m-%dT%H:%M:%S')}\n\n"
        f"- 说明：{note}\n- 阻断项：{blocking}\n"
    )
    (run_dir / "STAGE4B_STATUS.md").write_text(status, encoding="utf-8")
    handoff = (
        f"# Stage 4B HANDOFF_STATE\n\n"
        f"- 当前状态：`{state}`\n"
        f"- 冻结决定：finalist=`{s4bp.FINALIST}/{s4bp.COMBINED_IMPLEMENTATION}`，"
        f"primary=`{s4bp.PRIMARY_PREDICTOR}`，基线=`{s4bp.STANDARD_BASELINE}`+`{s4bp.SECOND_BASELINE}`，"
        f"阈值=`{s4bp.TOP_ERROR_THRESHOLD}`（均冻结自 Stage 4A，不重训/不重选/不重算）\n"
        f"- 最近验证：{note}\n"
        f"- 下一步：按 `run_stage4b.py` 命令顺序推进\n"
        f"- 阻断项：{blocking}\n"
    )
    (run_dir / "HANDOFF_STATE.md").write_text(handoff, encoding="utf-8")


# --- preflight ------------------------------------------------------------------


def cmd_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only: validate config, prove seed zero-overlap, verify every frozen input.

    Builds (but does NOT write) the candidate schedule and re-hashes every frozen
    Stage 4A / model / scaler / binding artifact, so a missing or drifted input fails
    here, before power or collection. Never modifies any old artifact.
    """
    checked = s4bp.validate_stage4b_config(config)
    historical = s4bp.scan_historical_seeds(
        [_resolve(checked["historical_data_root"]), ROOT / "experiments"]
    )
    schedule = s4bp.make_candidate_seed_schedule()
    from ac_twin.stage2_protocol import validate_seed_schedule

    validate_seed_schedule(schedule, historical)
    model_lock = s4bp.inventory_model_lock(checked)
    run_dir = _resolve(checked["run_dir"])
    _write_status(run_dir, "SOFTWARE_READY",
                  note="preflight 通过；Stage 4A 冻结链/模型/scaler hash 与历史seed零重叠已复验")
    return {
        "status": "STAGE4B_PREFLIGHT_PASS",
        "plan_id": s4bp.PLAN_ID,
        "n_historical_episode_seeds": int(historical.size),
        "seed_schedule_zero_overlap": True,
        "checkpoint_count": len(model_lock["checkpoint_sha256"]),
        "stage4a_artifacts_verified": sorted(model_lock["stage4a_sha256"]),
        "scaler_sha256": model_lock["scaler_sha256"],
        "seed_schedule_sha256": __import__("hashlib").sha256(schedule.tobytes()).hexdigest(),
    }


# --- power ------------------------------------------------------------------------


def cmd_power(config: Mapping[str, Any], config_path: str | Path) -> dict[str, Any]:
    """Run the create-once joint power analysis (plan section 5).

    Gate-0 self-check vs the frozen Stage 4A report runs before any Monte-Carlo draw.
    If the 75%-shrink joint power is below target even at the max candidate n, the
    stage terminates ``STAGE4B_CONFIRMATION_UNDERPOWERED`` and no lock is built.
    """
    run_dir = _resolve(config["run_dir"])
    report_path = run_dir / "power" / "power_report.json"
    if report_path.exists():
        raise FileExistsError(f"Stage 4B power report already exists: {report_path}")
    report = s4bpw.run_power(config, project_root=ROOT)
    write_once_json(report_path, report)
    if report["selection"]["underpowered"]:
        _write_status(run_dir, "POWERED",
                      note="功效不足：75%收缩在最大候选n下联合功效<0.80",
                      blocking=s4bp.STATUS_UNDERPOWERED)
    else:
        _write_status(run_dir, "POWERED",
                      note=f"功效达标：机械选定 selected_n={report['selected_n']}")
    return report


# --- lock-protocol ------------------------------------------------------------------


def cmd_lock_protocol(config_path: str | Path) -> dict[str, Any]:
    """Write the confirm protocol lock once, embedding the power-chosen ``selected_n``."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    checked = s4bp.validate_stage4b_config(config)
    run_dir = _resolve(checked["run_dir"])
    report_path = run_dir / "power" / "power_report.json"
    if not report_path.is_file():
        raise RuntimeError("lock-protocol requires the create-once power report first")
    power = json.loads(report_path.read_text(encoding="utf-8"))
    selected_n = power.get("selected_n")
    if selected_n is None:
        raise RuntimeError(
            f"power selected no size ({s4bp.STATUS_UNDERPOWERED}); refusing to lock")
    lock = s4bp.write_protocol_lock(config_path, selected_n=int(selected_n), project_root=ROOT)
    _write_readiness_checklist(run_dir, lock, power)
    _write_status(run_dir, "PROTOCOL_LOCKED",
                  note=f"protocol lock 已一次性写入；selected_n={selected_n}")
    return {
        "status": "PROTOCOL_LOCKED",
        "selected_n": int(selected_n),
        "per_stratum": int(lock["per_stratum"]),
        "protocol_lock_seed_schedule_sha256": lock["seed_schedule_sha256"],
        "n_historical_episode_seeds": lock["n_historical_episode_seeds"],
    }


def _write_readiness_checklist(run_dir: Path, lock: Mapping[str, Any], power: Mapping[str, Any]) -> None:
    lines = [
        "# Stage 4B READINESS_CHECKLIST",
        "",
        f"> 计划：`{s4bp.PLAN_ID}`",
        f"> 更新：{time.strftime('%Y-%m-%dT%H:%M:%S')}",
        "",
        "- [x] 功效报告 create-once 已写；75% 收缩为唯一阻断裁判",
        f"- [x] 机械选定 `selected_n = {lock['selected_n']}`（每层 {lock['per_stratum']}）",
        f"- [x] 候选 schedule 零重叠（历史 seed 数 {lock['n_historical_episode_seeds']}）",
        "- [x] Stage 4A 冻结链（recovery lock / selection / finalist / geometry / scaler / risk_dataset）hash 全部匹配",
        f"- [x] 冻结阈值 `top_error_threshold = {s4bp.TOP_ERROR_THRESHOLD}`（不重算）",
        "",
        "## n=5000 三收缩场景联合功效（并列）",
        "",
    ]
    for scenario, power_val in power["selection"]["sensitivity_at_max_n"].items():
        lines.append(f"- {scenario} 收缩：{power_val:.4f}")
    lines += [
        "",
        "50% 收缩仅作选择性乐观风险提示，不设人工暂停或二次批准点。",
    ]
    (run_dir / "READINESS_CHECKLIST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- smoke -------------------------------------------------------------------------


def cmd_smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    """12-episode engineering smoke; writes only to ``smoke/`` -> ENGINEERING_SMOKE_ONLY.

    Exercises the legal-feature constructor, the frozen finalist/geometry model load +
    predict in THIS environment (same input -> same output), the whitelist type-error
    guard and a tiny risk-coverage curve on a synthetic paired batch. Never touches
    real Stage 4B data or any research artifact.
    """
    run_dir = _resolve(config["run_dir"])
    smoke_dir = run_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    episodes, agents, horizon, history_len = 12, 5, 20, 10
    rng = np.random.default_rng(20260917)
    history = rng.normal(size=(episodes, history_len, agents, 8)).astype(np.float32)
    factual = rng.normal(size=(episodes, horizon, agents, 8)).astype(np.float32)
    counterfactual = rng.normal(size=(episodes, horizon, agents, 8)).astype(np.float32)
    for block in (history, factual, counterfactual):
        block[..., 0] = 1.0
    valid = np.ones((episodes, horizon, agents), dtype=bool)
    primary = counterfactual[..., POSITION_SLICE] - factual[..., POSITION_SLICE]
    primary_by_seed = np.stack([primary + 0.01 * p for p in range(5)]).astype(np.float32)
    full_by_seed = np.stack([primary + 0.02 * p for p in range(5)]).astype(np.float32)
    history_by_seed = np.stack([primary - 0.02 * p for p in range(5)]).astype(np.float32)
    response = {
        "r_hat": np.abs(rng.normal(size=(5, episodes, horizon, agents))).astype(np.float32),
        "presence_prob": rng.uniform(size=(5, episodes, horizon, agents)).astype(np.float32),
        "high_prob": rng.uniform(size=(5, episodes, horizon, agents)).astype(np.float32),
    }
    geometry = {
        "distance_to_ego": np.abs(rng.normal(size=(episodes, agents))),
        "relative_speed": np.abs(rng.normal(size=(episodes, agents))),
        "visible_count": np.full(episodes, float(agents - 1)),
    }
    action = rng.integers(0, 5, size=episodes)
    duration = rng.choice([1, 3], size=episodes)

    features = s4r.build_legal_risk_features(
        primary_predictions=primary_by_seed, full_predictions=full_by_seed,
        history_predictions=history_by_seed, response_predictions=response,
        pre_intervention_geometry=geometry, action=action, duration=duration,
        valid_mask=valid,
    )

    whitelist_ok = False
    try:
        s4r.build_legal_risk_features(
            primary_predictions=primary_by_seed, full_predictions=full_by_seed,
            history_predictions=history_by_seed, response_predictions=response,
            pre_intervention_geometry=geometry, action=action, duration=duration,
            valid_mask=valid, true_error=np.ones(episodes),  # illegal kwarg
        )
    except TypeError:
        whitelist_ok = True

    # Frozen finalist loads in this environment and is deterministic on same input.
    frozen = s4bp.STAGE4A_FROZEN
    finalist = s4r.FittedRiskModel.load(ROOT / frozen["finalist_risk_model"][0])
    geometry_model = s4r.FittedRiskModel.load(ROOT / frozen["geometry_risk_model"][0])
    matrix = s4r.design_matrix(features, s4r.CANDIDATE_FEATURES[s4r.CANDIDATE_COMBINED])
    score_a = finalist.predict(matrix)
    score_b = finalist.predict(matrix)
    finalist_deterministic = bool(np.array_equal(score_a, score_b))
    geo_score = geometry_model.predict(
        s4r.design_matrix(features, s4r.CANDIDATE_FEATURES[s4r.BASELINE_GEOMETRY]))
    unc_score = s4r.direct_risk_score(s4r.CANDIDATE_UNCERTAINTY, features)

    risk_target = s4r.risk_target_from_pairs(
        s4r.episode_effect_ade(primary_by_seed, s4r.build_effect_target(
            factual[..., POSITION_SLICE], counterfactual[..., POSITION_SLICE], valid)[0], valid))
    curve = s4s.risk_coverage_curve(score_a, risk_target)
    report = {
        "status": SMOKE_STATUS,
        "episodes": episodes,
        "feature_count": len(features),
        "whitelist_rejects_truth_input": bool(whitelist_ok),
        "finalist_loads_and_deterministic": finalist_deterministic,
        "finalist_score_finite": bool(np.isfinite(score_a).all()),
        "geometry_score_finite": bool(np.isfinite(geo_score).all()),
        "uncertainty_score_finite": bool(np.isfinite(unc_score).all()),
        "risk_target_finite": bool(np.isfinite(risk_target).all()),
        "curve_points": int(curve["coverage"].size),
    }
    report_path = smoke_dir / "smoke_report.json"
    tmp = report_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(report_path)
    print(f"[stage4b-smoke] STATUS={SMOKE_STATUS} features={len(features)} "
          f"whitelist_ok={whitelist_ok} finalist_deterministic={finalist_deterministic} "
          f"-> {report_path}", flush=True)
    return report


# --- infer-confirm ------------------------------------------------------------------


def _binding(config: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_resolve(config["stage3b_binding"]).read_text(encoding="utf-8"))["receipt_payload"]


def cmd_infer_confirm(config: Mapping[str, Any], *, resume: bool = False) -> dict[str, Any]:
    run_dir = _resolve(config["run_dir"])
    lock = s4bp.verify_protocol_lock(run_dir)
    receipt_path = _resolve(config["collection_receipt"])
    collection = json.loads(receipt_path.read_text(encoding="utf-8"))
    bundle = s4bi.load_confirm_eval(config, project_root=ROOT)
    if collection.get("data_output_sha256") != bundle["data_signature"]:
        raise ValueError("Stage 4B collection data hash mismatch")

    prediction_path = run_dir / "inference" / "predictions.npz"
    pred_receipt_path = run_dir / "inference" / "prediction_receipt.json"
    risk_scores_path = run_dir / "inference" / "risk_scores.npz"
    risk_receipt_path = run_dir / "inference" / "risk_score_receipt.json"

    if risk_receipt_path.is_file() and risk_scores_path.is_file():
        receipt = json.loads(risk_receipt_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError("Stage 4B inference already completed")
        if receipt.get("risk_scores_sha256") != sha256_file(risk_scores_path):
            raise ValueError("Stage 4B risk-score hash mismatch")
        return receipt
    for partial in (prediction_path, pred_receipt_path, risk_scores_path, risk_receipt_path):
        if partial.exists():
            raise FileExistsError(f"partial Stage 4B inference artifact: {partial}")

    checked = s4bp.validate_stage4b_config(config)
    binding = _binding(checked)
    stage3r_dir = _resolve(checked["stage3r_run_dir"])
    scaled = bundle["scaled"]
    episodes = scaled["history"].shape[0]
    indices = np.arange(episodes, dtype=np.int64)

    # Determinism re-check BEFORE the full pass (plan section 6.3.2).
    determinism = s4bi.determinism_recheck(
        checked, bundle, binding=binding, stage3r_dir=stage3r_dir)

    full_preds, history_preds, e2e_preds = [], [], []
    r_hats, presence_probs, high_probs = [], [], []
    reporter = ProgressReporter(total=len(s4bp.EXPERT_ROUTER_SEEDS), every=1,
                                prefix="stage4b-infer", heartbeat_seconds=30.0)
    for pair_index in range(len(s4bp.EXPERT_ROUTER_SEEDS)):
        out = s4bi.run_pair_inference(
            checked, scaled, indices, pair_index=pair_index,
            binding=binding, stage3r_dir=stage3r_dir)
        full_preds.append(out["full"])
        history_preds.append(out["history"])
        e2e_preds.append(out["e2e"])
        r_hats.append(out["r_hat"])
        presence_probs.append(out["presence_prob"])
        high_probs.append(out["high_prob"])
        reporter.update(pair_index + 1, details=f"pair{pair_index} done")

    arrays = {
        "full_predictions": np.stack(full_preds).astype(np.float32),
        "history_predictions": np.stack(history_preds).astype(np.float32),
        "e2e_predictions": np.stack(e2e_preds).astype(np.float32),
        "r_hat": np.stack(r_hats).astype(np.float32),
        "presence_prob": np.stack(presence_probs).astype(np.float32),
        "high_prob": np.stack(high_probs).astype(np.float32),
        "valid_mask": bundle["valid"].astype(bool),
        "cf_action": bundle["cf_action"],
        "cf_duration": bundle["cf_duration"],
        "episode_seed": bundle["episode_seed"],
    }
    for name, value in arrays.items():
        if value.dtype != bool and not np.isfinite(value).all():
            raise RuntimeError(f"non-finite Stage 4B prediction field: {name}")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = prediction_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(prediction_path)
    s4bi.write_prediction_receipt(
        run_dir, predictions_path=prediction_path, bundle=bundle,
        lock=lock, determinism=determinism)

    # Legal features + frozen risk scores (still pre-lock; no truth label read).
    features = s4bi.build_features(bundle, arrays)
    frozen = s4bp.STAGE4A_FROZEN
    finalist_path = ROOT / frozen["finalist_risk_model"][0]
    geometry_path = ROOT / frozen["geometry_risk_model"][0]
    finalist = s4r.FittedRiskModel.load(finalist_path)
    geometry_model = s4r.FittedRiskModel.load(geometry_path)
    scores = s4bi.score_all_candidates(
        features, finalist_model=finalist, geometry_model=geometry_model,
        rng_seed=int(checked["collection_seed"]))
    tmp = risk_scores_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{k: v.astype(np.float64) for k, v in scores.items()})
    tmp.replace(risk_scores_path)
    score_receipt = s4bi.write_risk_score_receipt(
        run_dir, risk_scores_path=risk_scores_path, scores=scores,
        predictions_path=prediction_path,
        finalist_model_path=finalist_path, geometry_model_path=geometry_path)

    _write_status(run_dir, "CONFIRM_INFERENCE_COMPLETE",
                  note=f"冻结推理+风险评分完成：{episodes} episodes × 5 pair；"
                       f"确定性复验 bitwise={determinism['bitwise_identical']}")
    return score_receipt


# --- adjudicate-confirm -------------------------------------------------------------


def _acquire_confirmation_lock(lock_path: Path) -> None:
    """Atomically burn the single-use confirmation lock BEFORE any truth-side read."""
    payload = json.dumps(
        {"plan_id": s4bp.PLAN_ID, "used_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        indent=2,
    )
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"confirmation set already consumed (lock: {lock_path}); it runs once")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)


def _confirm_qa_prelock(
    config: Mapping[str, Any],
    run_dir: Path,
    bundle: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
    scores: Mapping[str, np.ndarray],
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Gate Q integrity checks that need NO truth label — run BEFORE burning the lock.

    A failure here is recoverable without consuming the one-shot resource, so this
    raises (aborting adjudication) rather than burning ``confirmation_used.lock``.
    """
    checked = s4bp.validate_stage4b_config(config)
    checks: dict[str, bool] = {}

    # (4) Stage 4A recovery lock / selection / models / scaler / Stage 4B code+config.
    try:
        s4bp.verify_protocol_lock(run_dir)
        checks["locked_inputs_hash_match"] = True
    except Exception:
        checks["locked_inputs_hash_match"] = False
    frozen_ok = True
    try:
        s4bp.inventory_frozen_stage4a()
    except Exception:
        frozen_ok = False
    checks["stage4a_frozen_chain"] = frozen_ok

    # (1) seed zero overlap: confirm seeds disjoint from historical AND within schedule.
    episode_seed = np.asarray(bundle["episode_seed"], dtype=np.int64)
    historical = s4bp.scan_historical_seeds(
        [_resolve(checked["historical_data_root"]), ROOT / "experiments"])
    schedule = np.load(run_dir / "protocol" / "candidate_seed_schedule.npy").astype(np.int64)
    checks["seed_unique"] = bool(len(np.unique(episode_seed)) == episode_seed.size)
    checks["seed_disjoint_from_history"] = bool(
        np.intersect1d(episode_seed, historical).size == 0)
    checks["seed_within_schedule"] = bool(set(episode_seed.tolist()) <= set(schedule.tolist()))

    # (2) ten-stratum balance + episode-order / output consistency.
    per_cell = int(lock["per_stratum"])
    try:
        s4bp.assert_balanced_strata(bundle["cf_action"], bundle["cf_duration"], per_cell)
        checks["strata_balanced"] = True
    except Exception:
        checks["strata_balanced"] = False
    checks["episode_order_consistent"] = bool(
        np.array_equal(np.asarray(predictions["episode_seed"], dtype=np.int64), episode_seed))
    checks["prediction_shapes"] = bool(
        predictions["e2e_predictions"].shape[0] == len(s4bp.EXPERT_ROUTER_SEEDS)
        and predictions["e2e_predictions"].shape[1] == episode_seed.size)

    # (3) prefix / identity / mask / capacity QA from the collection receipt.
    collection = json.loads(_resolve(checked["collection_receipt"]).read_text(encoding="utf-8"))
    checks["prefix_identity_clean"] = bool(
        float(collection.get("prefix_max_abs_diff", 1.0)) <= 1e-6
        and float(collection.get("identity_max_abs_diff", 1.0)) <= 1e-6)
    valid = np.asarray(predictions["valid_mask"], dtype=bool)
    checks["mask_has_valid_cell"] = bool((valid.sum(axis=(1, 2)) > 0).all())
    checks["data_hash_match"] = bool(
        collection.get("data_output_sha256") == bundle["data_signature"])

    # (5) whitelist: the feature constructor refuses truth-side kwargs by signature.
    import inspect

    params = set(inspect.signature(s4r.build_legal_risk_features).parameters)
    forbidden = {"true_error", "true_response", "counterfactual_xy", "cea", "cew", "oracle"}
    checks["whitelist_no_truth_param"] = bool(forbidden.isdisjoint(params))
    checks["scores_finite"] = bool(
        all(np.isfinite(np.asarray(v)).all() for k, v in scores.items() if k != "CONSTANT_RANDOM"))

    return {"passed": bool(all(checks.values())), "checks": checks}


def cmd_adjudicate_confirm(config: Mapping[str, Any]) -> dict[str, Any]:
    """The SINGLE confirmation adjudication (write-once, single-use)."""
    run_dir = _resolve(config["run_dir"])
    lock = s4bp.verify_protocol_lock(run_dir)
    report_path = run_dir / "reports" / "stage4b_confirmatory_report.json"
    decision_path = run_dir / "reports" / "STAGE4B_CONFIRMATORY_DECISION.md"
    confirm_lock = run_dir / "confirmation_used.lock"
    if report_path.exists() or decision_path.exists() or confirm_lock.exists():
        raise FileExistsError("Stage 4B confirmation already executed; it runs once")

    prediction_path = run_dir / "inference" / "predictions.npz"
    risk_scores_path = run_dir / "inference" / "risk_scores.npz"
    pred_receipt = json.loads((run_dir / "inference" / "prediction_receipt.json").read_text(encoding="utf-8"))
    score_receipt = json.loads((run_dir / "inference" / "risk_score_receipt.json").read_text(encoding="utf-8"))
    if pred_receipt.get("predictions_sha256") != sha256_file(prediction_path):
        raise ValueError("Stage 4B predictions hash mismatch")
    if score_receipt.get("risk_scores_sha256") != sha256_file(risk_scores_path):
        raise ValueError("Stage 4B risk-scores hash mismatch")

    bundle = s4bi.load_confirm_eval(config, project_root=ROOT)
    with np.load(prediction_path, allow_pickle=False) as z:
        predictions = {key: z[key] for key in z.files}
    with np.load(risk_scores_path, allow_pickle=False) as z:
        scores = {key: z[key] for key in z.files}

    # Gate Q pre-lock: recoverable integrity checks; abort WITHOUT burning on failure.
    qa = _confirm_qa_prelock(config, run_dir, bundle, predictions, scores, lock)
    if not qa["passed"]:
        raise RuntimeError(
            "Stage 4B Gate Q pre-lock integrity check failed (lock NOT burned): "
            + json.dumps(qa["checks"]))

    # Burn the single-use lock BEFORE constructing any truth-side label.
    _acquire_confirmation_lock(confirm_lock)
    try:
        effect_true, valid = s4r.build_effect_target(
            bundle["factual_xy"], bundle["counterfactual_xy"],
            np.asarray(predictions["valid_mask"], dtype=bool))
        primary_by_pair = s4r.episode_effect_ade(
            predictions["e2e_predictions"], effect_true, valid)  # (5, E)
        risk_target = s4r.risk_target_from_pairs(primary_by_pair)  # (E,)

        inputs = {
            "comb_score": np.asarray(scores["COMBINED_RISK_HEAD"], dtype=np.float64),
            "unc_score": np.asarray(scores["UNCERTAINTY_ONLY"], dtype=np.float64),
            "geo_score": np.asarray(scores["GEOMETRY_ONLY"], dtype=np.float64),
            "risk_target": risk_target,
            "primary_error_by_pair": primary_by_pair,
            "cf_action": np.asarray(predictions["cf_action"], dtype=np.int64),
            "cf_duration": np.asarray(predictions["cf_duration"], dtype=np.int64),
            "top_error_threshold": float(config["top_error_threshold"]),
        }
        boot = s4bs.confirm_paired_bootstrap(
            inputs,
            n_resamples=int(config["bootstrap_resamples_confirm"]),
            seed=int(config["bootstrap_entropy_confirm"]),
        )
        verdict = s4bs.evaluate_gate_risk_confirm(
            boot["point"], boot["ci_lower"], qa_passed=True)
    except Exception:
        # The lock stays burned; an engineering failure does NOT refund the confirm set.
        raise

    point = boot["point"]
    ci_lower = boot["ci_lower"]
    ci_upper = boot["ci_upper"]
    report = {
        "plan_id": s4bp.PLAN_ID,
        "finalist": s4bp.FINALIST,
        "combined_implementation": s4bp.COMBINED_IMPLEMENTATION,
        "primary_predictor": s4bp.PRIMARY_PREDICTOR,
        "standard_baseline": s4bp.STANDARD_BASELINE,
        "second_baseline": s4bp.SECOND_BASELINE,
        "n_confirm_episodes": int(risk_target.size),
        "selected_n": int(lock["selected_n"]),
        "top_error_threshold": float(config["top_error_threshold"]),
        "quantities": {
            "point": point.tolist(),
            "ci_lower": ci_lower.tolist(),
            "ci_upper": ci_upper.tolist(),
        },
        "secondary": _secondary_metrics(scores, risk_target, primary_by_pair, predictions, config),
        "qa": qa,
        "verdict": verdict,
        "status": verdict["status"],
        "bootstrap": {
            "n_resamples": boot["n_resamples"],
            "seed": boot["bootstrap_seed"],
            "confidence_level": boot["confidence_level"],
        },
        "predictions_sha256": sha256_file(prediction_path),
        "risk_scores_sha256": sha256_file(risk_scores_path),
        "data_sha256": bundle["data_signature"],
        "note": "Stage 4B 是 Merge 域内独立确认；不承担跨场景外部有效性。",
    }
    write_once_json(report_path, report)
    _write_decision_md(decision_path, report)
    _write_status(run_dir, "CONFIRM_ADJUDICATED",
                  note=f"确认裁判完成：{verdict['status']}",
                  blocking="none" if verdict["passed"] else verdict["status"])
    return report


def _secondary_metrics(scores, risk_target, primary_by_pair, predictions, config) -> dict[str, Any]:
    """Secondary (non-Gate) metrics: ADE@90, ablations, deciles, top-error trims."""
    threshold = float(config["top_error_threshold"])
    comb = np.asarray(scores["COMBINED_RISK_HEAD"], dtype=np.float64)
    out: dict[str, Any] = {
        "ade@90": s4s.ade_at_coverage(comb, risk_target, 0.90),
        "ablations": {},
    }
    for name in ("RESPONSE_ONLY", "DISAGREEMENT_ONLY", "GEOMETRY_ONLY", "UNCERTAINTY_ONLY"):
        score = np.asarray(scores[name], dtype=np.float64)
        out["ablations"][name] = s4s.candidate_report(
            score, risk_target, top_error_threshold_value=threshold)
    # Risk deciles vs true error.
    order = np.argsort(comb, kind="stable")
    deciles = np.array_split(order, 10)
    out["decile_mean_error"] = [float(risk_target[d].mean()) for d in deciles]
    # Sensitivity: drop top 1% / 5% true-error episodes.
    for trim in (0.01, 0.05):
        k = int(round((1.0 - trim) * risk_target.size))
        keep = np.argsort(risk_target, kind="stable")[:k]
        out[f"i_aurc_trim_top_{int(trim*100)}pct"] = float(
            (s4s.aurc(np.asarray(scores["UNCERTAINTY_ONLY"])[keep], risk_target[keep])
             - s4s.aurc(comb[keep], risk_target[keep]))
            / s4s.aurc(np.asarray(scores["UNCERTAINTY_ONLY"])[keep], risk_target[keep]))
    return out


def _write_decision_md(path: Path, report: Mapping[str, Any]) -> None:
    verdict = report["verdict"]
    point = report["quantities"]["point"]
    ci_lower = report["quantities"]["ci_lower"]
    names = [
        "I_AURC vs UNCERTAINTY", "I_C80 vs UNCERTAINTY",
        "I_AURC vs GEOMETRY", "I_C80 vs GEOMETRY", "PR-AUC - prevalence",
    ]
    lines = [
        "# Stage 4B 独立风险确认裁判",
        "",
        f"> 状态：`{report['status']}`",
        f"> 计划：`{report['plan_id']}`",
        "> 性质：Merge 域内独立确认（confirmatory）；不承担跨场景外部有效性",
        "",
        f"- finalist：`{report['finalist']}/{report['combined_implementation']}`",
        f"- 主预测器：`{report['primary_predictor']}`",
        f"- 基线：`{report['standard_baseline']}` + `{report['second_baseline']}`",
        f"- 确认 episodes：{report['n_confirm_episodes']}（selected_n={report['selected_n']}）",
        "",
        "## 主确认条件（§8.2 AND）",
        "",
    ]
    for name, ok in verdict["conditions"].items():
        lines.append(f"- [{'x' if ok else ' '}] `{name}`")
    lines.append("")
    for i, label in enumerate(names):
        lines.append(f"- {label} = {point[i]:+.4f}（95% CI 下界 {ci_lower[i]:+.4f}）")
    lines += [
        "",
        f"- 正向 pair 数 = {verdict['positive_pairs']}/5",
        f"- 分层最小 I_AURC = {verdict['strata_min_i_aurc']:+.4f}（下限 -0.10）",
        "",
        f"裁决：`{report['status']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AC-Twin Stage 4B independent risk confirmation")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "power", "lock-protocol", "smoke",
                    "infer-confirm", "adjudicate-confirm", "verify"):
        item = sub.add_parser(command)
        item.add_argument("--config", required=True)
        if command in ("infer-confirm",):
            item.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    if args.command == "preflight":
        report = cmd_preflight(config)
    elif args.command == "power":
        report = cmd_power(config, args.config)
    elif args.command == "lock-protocol":
        report = cmd_lock_protocol(args.config)
    elif args.command == "smoke":
        report = cmd_smoke(config)
    elif args.command == "infer-confirm":
        report = cmd_infer_confirm(config, resume=bool(getattr(args, "resume", False)))
    elif args.command == "verify":
        report = s4bp.verify_protocol_lock(_resolve(config["run_dir"]))
    else:
        report = cmd_adjudicate_confirm(config)
    if args.command != "verify":
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print(json.dumps({"status": "LOCK_VERIFIED",
                          "seed_schedule_sha256": report["seed_schedule_sha256"]},
                         indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
