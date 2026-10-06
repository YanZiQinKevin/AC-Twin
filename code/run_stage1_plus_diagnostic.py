"""Stage 1 Plus read-only shape diagnostic and power-analysis runner.

This runner is a *development-only*, read-only diagnostic. It reuses the sealed
Stage 1 merge development-test predictions and ground truth to decide whether
the development Gate A signal is driven by response *presence* (zero vs
positive), a within-positive magnitude trend, or a high-response rebound, and to
estimate confirmatory power for Stage 2 sample-size planning.

Hard boundaries (frozen):
- No training, no resume, no optimizer, no gradient computation, no checkpoint
  loading, no re-inference, no new data collection, no router.
- It never imports the Stage 1 training module and never modifies any Stage 1
  sealed artifact; outputs go only to the Stage 1 Plus output directory.
- It does not re-judge Stage 1 Gate S/A/B and never emits a confirmatory claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.stage1_data import (  # noqa: E402
    compute_history_agent_known,
    compute_target_effect,
    compute_valid_mask,
    visible_count_signature,
)
from ac_twin import stage1_statistics as _s1  # noqa: E402
from ac_twin import stage1_plus_statistics as _plus  # noqa: E402

FROZEN_SCENE = "merge"
FROZEN_EXPERIMENT = "stage1_plus_merge_shape_and_power"
FROZEN_ANALYSIS_STATUS = "DEVELOPMENT_DIAGNOSTIC_ONLY"
FROZEN_TRAINING_SEEDS = [11, 22, 33, 44, 55]
FROZEN_CONDITIONS = ["no_suffix", "uniform_suffix", "shuffled_suffix"]
FROZEN_BOOTSTRAP_RESAMPLES = 10000
FROZEN_CONFIDENCE_LEVEL = 0.95
FROZEN_ETA_EPSILON_M = 0.0001
FROZEN_POWER_MONTE_CARLO = 10000
FROZEN_CANDIDATE_EPISODE_COUNTS = [300, 450, 600, 750, 900, 1200, 1500, 1800, 2400]
FROZEN_MINIMUM_BIN_EPISODES = 30
FROZEN_DROP_FRACTIONS = [0.01, 0.05]
FROZEN_POWER_EFFECT_MULTIPLIERS = [1.0, 0.75, 0.5]

REPRO_TOLERANCE = 1e-10
BIN_NAMES = ("zero", "positive_low", "positive_mid", "positive_high")
CLAIM_BOUNDARY = (
    "development_diagnostic_only; does_not_reopen_or_rejudge_stage1_gate_a; "
    "does_not_confirm_monotonic_decay_or_u_shape; no_mechanism_or_causal_claim; "
    "stage2_and_router_remain_blocked_pending_user_approval"
)


def _resolve(path_value: str) -> Path:
    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage1_plus_config(config: Mapping[str, Any]) -> None:
    """Validate the frozen Stage 1 Plus config; raise ValueError before output."""
    if config.get("scene") != FROZEN_SCENE:
        raise ValueError(f"scene must be {FROZEN_SCENE!r}; only Merge is analyzed")
    if config.get("experiment_name") != FROZEN_EXPERIMENT:
        raise ValueError(f"experiment_name must be {FROZEN_EXPERIMENT!r}")
    if config.get("analysis_status") != FROZEN_ANALYSIS_STATUS:
        raise ValueError(f"analysis_status must be {FROZEN_ANALYSIS_STATUS!r}")
    if list(config.get("training_seeds", [])) != FROZEN_TRAINING_SEEDS:
        raise ValueError(f"training_seeds must be {FROZEN_TRAINING_SEEDS}")
    if list(config.get("conditions", [])) != FROZEN_CONDITIONS:
        raise ValueError(f"conditions must be {FROZEN_CONDITIONS}")

    for field in ("source_data", "stage1_config", "stage1_output_dir",
                  "sealed_hash_manifest", "output_dir"):
        if not config.get(field):
            raise ValueError(f"missing required path field: {field}")

    output_dir = _resolve(config["output_dir"]).resolve()
    stage1_dir = _resolve(config["stage1_output_dir"]).resolve()
    if output_dir == stage1_dir or stage1_dir in output_dir.parents:
        raise ValueError(
            f"output_dir {output_dir} must not equal or be inside stage1_output_dir {stage1_dir}")

    statistics = config.get("statistics", {})
    if int(statistics.get("bootstrap_resamples", -1)) != FROZEN_BOOTSTRAP_RESAMPLES:
        raise ValueError(f"bootstrap_resamples must be {FROZEN_BOOTSTRAP_RESAMPLES}")
    if float(statistics.get("confidence_level", -1)) != FROZEN_CONFIDENCE_LEVEL:
        raise ValueError(f"confidence_level must be {FROZEN_CONFIDENCE_LEVEL}")
    if float(statistics.get("eta_epsilon_m", -1)) != FROZEN_ETA_EPSILON_M:
        raise ValueError(f"eta_epsilon_m must be {FROZEN_ETA_EPSILON_M}")
    if int(statistics.get("power_monte_carlo_replicates", -1)) != FROZEN_POWER_MONTE_CARLO:
        raise ValueError(f"power_monte_carlo_replicates must be {FROZEN_POWER_MONTE_CARLO}")
    if list(statistics.get("candidate_episode_counts", [])) != FROZEN_CANDIDATE_EPISODE_COUNTS:
        raise ValueError(f"candidate_episode_counts must be {FROZEN_CANDIDATE_EPISODE_COUNTS}")
    if int(statistics.get("minimum_bin_episodes", -1)) != FROZEN_MINIMUM_BIN_EPISODES:
        raise ValueError(f"minimum_bin_episodes must be {FROZEN_MINIMUM_BIN_EPISODES}")
    if list(statistics.get("drop_high_error_episode_fractions", [])) != FROZEN_DROP_FRACTIONS:
        raise ValueError(f"drop_high_error_episode_fractions must be {FROZEN_DROP_FRACTIONS}")
    if list(statistics.get("power_effect_multipliers", [])) != FROZEN_POWER_EFFECT_MULTIPLIERS:
        raise ValueError(f"power_effect_multipliers must be {FROZEN_POWER_EFFECT_MULTIPLIERS}")


def verify_sealed_inputs(config: Mapping[str, Any]) -> dict[str, str]:
    """Verify sealed Stage 1 inputs against the pre-fix hash manifest.

    Covers source data, the Stage 1 merge config, response bins, test manifest
    and the 15 test predictions (the outdated pre-fix ``report.json`` hash is not
    used). Raises ValueError on the first mismatch before any output is written.
    """
    scene = config["scene"]
    manifest = json.loads(_resolve(config["sealed_hash_manifest"]).read_text(encoding="utf-8"))
    stage1_dir = _resolve(config["stage1_output_dir"])
    checks: dict[str, tuple[Path, str]] = {
        "source_data": (_resolve(config["source_data"]), manifest["source_data"][scene]),
        "stage1_config": (_resolve(config["stage1_config"]), manifest["configs"][scene]),
        "response_bins": (stage1_dir / "response_bins.json", manifest[scene]["response_bins.json"]),
        "test_manifest": (stage1_dir / "test_manifest.json", manifest[scene]["test_manifest.json"]),
    }
    for condition in config["conditions"]:
        for seed in config["training_seeds"]:
            name = f"{condition}_seed{seed}_test_prediction.npy"
            checks[name] = (stage1_dir / name, manifest[scene][name])

    actual: dict[str, str] = {}
    for label, (file_path, expected) in checks.items():
        if not file_path.exists():
            raise ValueError(f"sealed input missing: {file_path}")
        digest = _sha256_file(file_path)
        if digest != expected:
            raise ValueError(
                f"sealed input hash mismatch for {label}: {file_path}\n"
                f"  expected {expected}\n  actual   {digest}")
        actual[label] = digest
    return actual


# --- read-only data assembly (mirrors the frozen Stage 1 analysis path) ----------


def _initial_distance_to_ego(history: np.ndarray) -> np.ndarray:
    presence = history[..., 0] > 0
    ego_last = history[:, -1, 0, 2:4]
    episodes, _, agents, _ = history.shape
    distance = np.full((episodes, agents), np.nan)
    for e in range(episodes):
        for i in range(agents):
            present = np.nonzero(presence[e, :, i])[0]
            if len(present):
                distance[e, i] = np.linalg.norm(history[e, present[-1], i, 2:4] - ego_last[e])
    return distance


def _broadcast_covariates(history_full, cf_action, cf_duration, visible_count,
                          n_future, n_agents, indices):
    n = len(indices)
    distance = _initial_distance_to_ego(history_full[indices])
    presence = (history_full[indices][..., 0] > 0)
    visible_ratio = presence.mean(axis=1)
    vcount = visible_count[indices]
    horizon = np.broadcast_to(np.arange(n_future)[None, :, None], (n, n_future, n_agents)).astype(np.float64)
    return {
        "horizon_index": horizon,
        "cf_action": np.broadcast_to(cf_action[indices][:, None, None], (n, n_future, n_agents)),
        "cf_duration": np.broadcast_to(cf_duration[indices][:, None, None], (n, n_future, n_agents)).astype(np.float64),
        "initial_distance": np.broadcast_to(distance[:, None, :], (n, n_future, n_agents)),
        "history_visible_ratio": np.broadcast_to(visible_ratio[:, None, :], (n, n_future, n_agents)),
        "visible_count": np.broadcast_to(vcount[:, None, None], (n, n_future, n_agents)).astype(np.float64),
    }


def load_analysis_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load sealed predictions + ground truth and assemble the analysis arrays.

    Read-only: loads ``.npy`` predictions and the source ``.npz``; never touches
    ``.pt`` checkpoints, optimizers or any training/inference path.
    """
    stage1_dir = _resolve(config["stage1_output_dir"])
    seeds = list(config["training_seeds"])
    conditions = list(config["conditions"])

    data = np.load(_resolve(config["source_data"]), allow_pickle=True)
    history = np.asarray(data["object_history"], dtype=np.float32)
    factual = np.asarray(data["object_factual_future"], dtype=np.float32)
    counterfactual = np.asarray(data["object_counterfactual_future"], dtype=np.float32)
    split = np.asarray(data["split"])
    cf_action = np.asarray(data["cf_action"])
    cf_duration = np.asarray(data["cf_duration"])

    test_idx = np.flatnonzero(split == 2)
    train_idx = np.flatnonzero(split == 0)

    target = compute_target_effect(factual, counterfactual)
    valid_mask = compute_valid_mask(history, factual, counterfactual)
    r_pos, _ = _s1.response_magnitudes(counterfactual, factual)
    visible_count = visible_count_signature(compute_history_agent_known(history))

    thresholds = json.loads((stage1_dir / "response_bins.json").read_text(encoding="utf-8"))

    def nonego(indices):
        mask = valid_mask[indices].copy()
        mask[:, :, 0] = False
        return mask

    test_nonego = nonego(test_idx)
    train_nonego = nonego(train_idx)
    bin_id_test = _s1.assign_response_bin(r_pos[test_idx], thresholds)
    target_test = target[test_idx]

    normalization = np.load(stage1_dir / "normalization.npz")
    target_mean = normalization["target_mean"].astype(np.float64)
    target_std = normalization["target_std"].astype(np.float64)

    def denorm(pred):
        return np.asarray(pred, dtype=np.float64) * target_std + target_mean

    errors_by_seed = {}
    for condition in conditions:
        per_seed = np.stack([
            _s1.position_errors(
                denorm(np.load(stage1_dir / f"{condition}_seed{seed}_test_prediction.npy",
                               allow_pickle=False)),
                target_test)
            for seed in seeds
        ])
        errors_by_seed[condition] = per_seed

    n_future = target.shape[1]
    n_agents = target.shape[2]
    cov_test = _broadcast_covariates(history, cf_action, cf_duration, visible_count,
                                     n_future, n_agents, test_idx)
    cov_train = _broadcast_covariates(history, cf_action, cf_duration, visible_count,
                                      n_future, n_agents, train_idx)
    standardizer = {}
    for name in ("horizon_index", "cf_duration", "initial_distance",
                 "history_visible_ratio", "visible_count"):
        values = cov_train[name][train_nonego]
        standardizer[name] = {"mean": float(values.mean()), "std": float(max(values.std(), 1e-6))}

    return {
        "seeds": seeds,
        "conditions": conditions,
        "n_train": int(len(train_idx)),
        "n_validation": int((split == 1).sum()),
        "n_development_test": int(len(test_idx)),
        "errors_by_seed": errors_by_seed,
        "r_pos_test": r_pos[test_idx],
        "bin_id_test": bin_id_test,
        "valid_mask_test": test_nonego,
        "cf_action_test": cf_action[test_idx],
        "covariates_test": cov_test,
        "train_standardizer": standardizer,
        "response_bins": thresholds,
    }


def build_power_statistics(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Build the multiplicity-aware frozen statistic closures for power analysis."""
    l_suffix = _s1.seed_mean(inputs["errors_by_seed"]["uniform_suffix"])
    l_shuffled = _s1.seed_mean(inputs["errors_by_seed"]["shuffled_suffix"])
    bin_id = inputs["bin_id_test"]
    valid = inputs["valid_mask_test"]
    r_pos = inputs["r_pos_test"]
    eta = np.log(l_suffix + _plus.ETA_EPSILON_M) - np.log(l_shuffled + _plus.ETA_EPSILON_M)
    n_episodes = valid.shape[0]

    original_low = valid & np.isin(bin_id, [0, 1])
    bin0 = valid & (bin_id == 0)
    bin3 = valid & (bin_id == 3)
    positive = valid & (bin_id >= 1)

    stand = inputs["train_standardizer"]
    cov = inputs["covariates_test"]
    X_all, y_all, _, log_col = _s1.build_trend_design(
        eta, r_pos, cov["horizon_index"], cov["cf_action"], cov["cf_duration"],
        cov["initial_distance"], cov["history_visible_ratio"], cov["visible_count"],
        valid, stand)
    rows_all = np.nonzero(valid)[0]
    X_pos, y_pos, _, pos_log_col = _s1.build_trend_design(
        eta, r_pos, cov["horizon_index"], cov["cf_action"], cov["cf_duration"],
        cov["initial_distance"], cov["history_visible_ratio"], cov["visible_count"],
        positive, stand)
    rows_pos = np.nonzero(positive)[0]

    return {
        "v_evt_low": _plus.make_group_v_evt_statistic(l_suffix, l_shuffled, original_low),
        "v_evt_low_minus_high": _plus.make_group_contrast_statistic(l_suffix, l_shuffled, original_low, bin3),
        "spearman_r_eta": _plus.make_weighted_spearman_statistic(r_pos, eta, valid),
        "trend_log_r": _plus.make_weighted_trend_statistic(X_all, y_all, log_col, rows_all, n_episodes),
        "c_presence": _plus.make_group_contrast_statistic(l_suffix, l_shuffled, bin0, positive),
        "positive_only_spearman": _plus.make_weighted_spearman_statistic(r_pos, eta, positive),
        "positive_only_trend": _plus.make_weighted_trend_statistic(X_pos, y_pos, pos_log_col, rows_pos, n_episodes),
        "c_rebound": _plus.make_rebound_statistic(l_suffix, l_shuffled, bin_id, valid),
    }


# --- hash conservation -----------------------------------------------------------


def snapshot_conservation_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    """Hash every file under the Stage 1 merge dir plus source data and config.

    Used before and after the run to prove no Stage 1 artifact was modified.
    """
    stage1_dir = _resolve(config["stage1_output_dir"])
    snapshot: dict[str, str] = {}
    for file_path in sorted(stage1_dir.rglob("*")):
        if file_path.is_file():
            snapshot[f"stage1_merge/{file_path.name}"] = _sha256_file(file_path)
    snapshot["source_data"] = _sha256_file(_resolve(config["source_data"]))
    snapshot["stage1_config"] = _sha256_file(_resolve(config["stage1_config"]))
    return snapshot


# --- atomic writers ----------------------------------------------------------------


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=target.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _atomic_write_json(target: Path, obj: Any) -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    # Validate round-trip before replacing the target.
    json.loads(text)
    _atomic_write_bytes(target, text.encode("utf-8"))


def _atomic_write_text(target: Path, text: str) -> None:
    _atomic_write_bytes(target, text.encode("utf-8"))


# --- report + decision document -----------------------------------------------------


def _reproduction_block(diag: Mapping[str, Any], report_analysis: Mapping[str, Any]) -> dict:
    mapping = {
        "v_evt_low": report_analysis["v_evt_low"]["point"],
        "v_evt_low_minus_high": report_analysis["v_evt_low_minus_high"]["point"],
        "spearman_r_eta": report_analysis["spearman_r_eta"]["point"],
        "trend_log_r": report_analysis["trend_log_r"]["point"],
    }
    block = {}
    for name, reference in mapping.items():
        computed = float(diag["reproduction"][name])
        block[name] = {
            "computed": computed,
            "reference": float(reference),
            "abs_diff": abs(computed - float(reference)),
            "within_tolerance": bool(abs(computed - float(reference)) <= REPRO_TOLERANCE),
        }
    bin_block = {}
    for name in BIN_NAMES:
        computed = float(diag["bins"][name]["V_evt"])
        reference = float(report_analysis["bins"][name]["V_evt"])
        bin_block[name] = {
            "computed": computed,
            "reference": reference,
            "abs_diff": abs(computed - reference),
            "within_tolerance": bool(abs(computed - reference) <= REPRO_TOLERANCE),
        }
    all_ok = all(v["within_tolerance"] for v in block.values()) and \
        all(v["within_tolerance"] for v in bin_block.values())
    return {"gate_a_point_estimates": block, "bin_v_evt": bin_block,
            "tolerance": REPRO_TOLERANCE, "all_within_tolerance": bool(all_ok)}


def run_diagnostic(config: Mapping[str, Any]) -> dict[str, Any]:
    """Execute the full read-only diagnostic and return (report, decision_md)."""
    started = time.time()
    statistics = config["statistics"]
    inputs = load_analysis_inputs(config)
    report_reference = json.loads(
        (_resolve(config["stage1_output_dir"]) / "report.json").read_text(encoding="utf-8"))
    report_analysis = report_reference["analysis"]

    diagnostics = _plus.compute_shape_diagnostics(
        l_no_by_seed=inputs["errors_by_seed"]["no_suffix"],
        l_suffix_by_seed=inputs["errors_by_seed"]["uniform_suffix"],
        l_shuffled_by_seed=inputs["errors_by_seed"]["shuffled_suffix"],
        r_pos=inputs["r_pos_test"],
        bin_id=inputs["bin_id_test"],
        valid_mask=inputs["valid_mask_test"],
        cf_action=inputs["cf_action_test"],
        covariates=inputs["covariates_test"],
        train_standardizer=inputs["train_standardizer"],
        bootstrap_resamples=int(statistics["bootstrap_resamples"]),
        confidence_level=float(statistics["confidence_level"]),
        bootstrap_seed=int(statistics["bootstrap_seed"]))

    power_statistics = build_power_statistics(inputs)
    power = _plus.estimate_power_curve(
        episode_statistics=power_statistics,
        candidate_episode_counts=list(statistics["candidate_episode_counts"]),
        monte_carlo_replicates=int(statistics["power_monte_carlo_replicates"]),
        confidence_level=float(statistics["confidence_level"]),
        seed=int(statistics["power_seed"]))

    reproduction = _reproduction_block(diagnostics, report_analysis)
    if not reproduction["all_within_tolerance"]:
        raise RuntimeError(
            "Stage 1 point estimates could not be reproduced within "
            f"{REPRO_TOLERANCE}: {json.dumps(reproduction, indent=2)}")

    report = {
        "experiment": config["experiment_name"],
        "analysis_status": FROZEN_ANALYSIS_STATUS,
        "scene": FROZEN_SCENE,
        "training_performed": False,
        "inference_performed": False,
        "stage1_gate_a_unchanged": True,
        "response_bins": inputs["response_bins"],
        "n_train": inputs["n_train"],
        "n_validation": inputs["n_validation"],
        "n_development_test": inputs["n_development_test"],
        "seed_order": inputs["seeds"],
        "bin_summaries": diagnostics["bins"],
        "reproduction": reproduction,
        "zero_dominance": diagnostics["zero_dominance"],
        "presence_diagnostic": {"c_presence": diagnostics["zero_dominance"]["c_presence"]},
        "positive_only_decay_diagnostic": diagnostics["positive_only"],
        "rebound_diagnostic": diagnostics["rebound"],
        "power_analysis": power,
        "shape_flags": diagnostics["shape_flags"],
        "shape_recommendation": diagnostics["shape_recommendation"],
        "claim_boundary": CLAIM_BOUNDARY,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    return report


def _fmt_ci(entry: Mapping[str, Any]) -> str:
    return (f"{entry['point']:.4f} [{entry['ci_lower']:.4f}, {entry['ci_upper']:.4f}]")


def render_decision_markdown(report: Mapping[str, Any], config: Mapping[str, Any],
                             test_summary: Mapping[str, Any]) -> str:
    d = report
    zd = d["zero_dominance"]
    po = d["positive_only_decay_diagnostic"]
    rb = d["rebound_diagnostic"]
    flags = d["shape_flags"]
    rec = d["shape_recommendation"]
    power = d["power_analysis"]
    candidates = power["candidate_episode_counts"]

    lines = []
    A = lines.append
    A("# Stage 1 Plus 轻量形状诊断与功效分析 — 决策文档")
    A("")
    A("> 文档状态：**DEVELOPMENT_DIAGNOSTIC_ONLY**  ")
    A("> 计划编号：`AC-TWIN-RFER-STAGE1-PLUS-2026-09-07-V1`  ")
    A("> 上位研究计划：`AC-TWIN-RFER-2026-09-07-V1`  ")
    A("> 日期：2026-09-07  ")
    A(f"> claim_boundary：`{d['claim_boundary']}`")
    A("")
    A("---")
    A("")
    A("## 1. 性质与边界")
    A("")
    A("- 本诊断仅使用已多次查看的 Merge development-test 开发数据，**不构成确认性结论**。")
    A("- 不重新判定 Stage 1 Gate S/A/B；原书面状态（merge=`DEVELOPMENT_GO_TO_NEW_HOLDOUT`）保持不变。")
    A("- Stage 2 与 router 仍处阻塞，等待用户审批；本诊断不授权采集、训练或 router。")
    A(f"- 运行标记：`training_performed={d['training_performed']}`、`inference_performed={d['inference_performed']}`、`stage1_gate_a_unchanged={d['stage1_gate_a_unchanged']}`。")
    A("")
    A("## 2. 输入完整性")
    A("")
    A(f"- 场景：`{d['scene']}`；种子顺序 `{d['seed_order']}`；development-test episode 数 = {d['n_development_test']}。")
    A(f"- split：train={d['n_train']} / validation={d['n_validation']} / development_test={d['n_development_test']}。")
    A("- 源数据、Stage 1 配置、response bins、test manifest 与 15 个 test prediction 的 SHA256 已在运行前核验通过，明细见 `input_hashes.json`；运行前后 Stage 1 merge 目录全部文件 hash 守恒。")
    A("")
    A("## 3. 原 Stage 1 复现（点估计，容差 1e-10）")
    A("")
    A("| 量 | 本次计算 | report.json | abs diff | 通过 |")
    A("|---|---:|---:|---:|:---:|")
    for name, cell in d["reproduction"]["gate_a_point_estimates"].items():
        A(f"| `{name}` | {cell['computed']:.12f} | {cell['reference']:.12f} | {cell['abs_diff']:.2e} | {'✓' if cell['within_tolerance'] else '✗'} |")
    A("")
    A("| 分箱 V_evt | 本次计算 | report.json | abs diff | 通过 |")
    A("|---|---:|---:|---:|:---:|")
    for name, cell in d["reproduction"]["bin_v_evt"].items():
        A(f"| {name} | {cell['computed']:.12f} | {cell['reference']:.12f} | {cell['abs_diff']:.2e} | {'✓' if cell['within_tolerance'] else '✗'} |")
    A("")
    A("原报告 CI 端点按协议原样引用，不要求以新诊断 bootstrap 种子重现。")
    A("")
    A("## 4. D1 zero/positive 存在性")
    A("")
    A(f"- zero agent-time 占比 = {zd['zero_agent_time_share']:.4f}")
    A(f"- zero 占 original_low 打乱后缀误差和比例 = {zd['zero_share_of_original_low_shuffled_error_sum']:.4f}")
    A(f"- zero 占 original_low 正确后缀误差和比例 = {zd['zero_share_of_original_low_suffix_error_sum']:.4f}")
    A(f"- `V_evt_zero` = {_fmt_ci(zd['v_evt_zero'])}")
    A(f"- `V_evt_positive_all` = {_fmt_ci(zd['v_evt_positive_all'])}")
    A(f"- `C_presence` = {_fmt_ci(zd['c_presence'])}")
    A(f"- leave-one-seed-out `C_presence` = {[round(v,4) for v in zd['leave_one_seed_out_c_presence']]}")
    A(f"- **PRESENCE_DIFFERENCE_SUPPORTED = {flags['PRESENCE_DIFFERENCE_SUPPORTED']}**")
    A("")
    A("## 5. D2 仅正响应幅度趋势")
    A("")
    A(f"- positive-only `Spearman(R, eta_evt)` = {_fmt_ci(po['spearman'])}")
    A(f"- positive-only 趋势 `log1p(R)` 系数 = {_fmt_ci(po['trend_log_r'])}")
    A(f"- `V_evt_positive_low - V_evt_positive_high` = {_fmt_ci(po['v_evt_low_minus_high'])}")
    A(f"- leave-one-seed-out spearman = {[round(v,4) for v in po['leave_one_seed_out']['spearman']]}")
    A(f"- leave-one-seed-out trend = {[round(v,4) for v in po['leave_one_seed_out']['trend_log_r']]}")
    A(f"- leave-one-seed-out low-high = {[round(v,4) for v in po['leave_one_seed_out']['v_evt_low_minus_high']]}")
    A(f"- **POSITIVE_MAGNITUDE_DECAY_SUPPORTED = {flags['POSITIVE_MAGNITUDE_DECAY_SUPPORTED']}**")
    A("")
    A("## 6. D3 高响应反弹")
    A("")
    A(f"- `C_high_low` = {_fmt_ci(rb['c_high_low'])}")
    A(f"- `C_high_mid` = {_fmt_ci(rb['c_high_mid'])}")
    A(f"- `C_rebound` = min 对比 = {_fmt_ci(rb['c_rebound'])}")
    A(f"- leave-one-seed-out `C_rebound` = {[round(v,4) for v in rb['leave_one_seed_out_c_rebound']]}")
    A("- leave-one-action-type-out `C_rebound`：")
    for action, cell in sorted(rb["leave_one_action_type_out"].items()):
        A(f"  - action {action}: C_rebound={cell['c_rebound']:.4f}（保留 episode {cell['n_episodes']}，移除 {cell['n_episodes_removed']}）")
    A(f"- leave-one-episode-out：min = {rb['leave_one_episode_out_min']:.4f}，反转数 = {rb['leave_one_episode_out_n_reversed']}")
    for label, cell in rb["high_error_deletion"].items():
        A(f"- 高误差删除 {label}：C_rebound = {cell['c_rebound']:.4f}（保留 {cell['n_episodes']} episode）")
    A(f"- 各正响应箱独立 episode 数 = {rb['bin_episode_counts']}")
    A(f"- **ROBUST_HIGH_REBOUND_SUPPORTED = {flags['ROBUST_HIGH_REBOUND_SUPPORTED']}**")
    A("")
    A("## 7. 完整分箱表")
    A("")
    A("| 响应箱 | V_evt | rho_no | meanL_no | meanL_suffix | meanL_shuffled | agent-time | episode |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in BIN_NAMES:
        b = d["bin_summaries"][name]
        A(f"| {name} | {b['V_evt']:.6f} | {b['rho_no']:.6f} | {b['mean_l_no']:.3f} | "
          f"{b['mean_l_suffix']:.3f} | {b['mean_l_shuffled']:.3f} | {b['n_agent_time']} | {b['n_episodes']} |")
    A("")
    A("## 8. 功效分析（单项 + 联合）")
    A("")
    A(f"- 候选 episode 数：{candidates}；Monte Carlo 复制 = {power['monte_carlo_replicates']}；"
      f"bootstrap 池 = {power['pool_episodes']}；z = {power['z']}")
    A("- 注意：本功效基于开发数据经验分布的近似，不保证确认性结果，仅用于避免沿用 n=300 的无功效设计。")
    A("")
    A("### 8.1 原 Gate A 四个数值条件单项功效（empirical）")
    A("")
    header = "| n | " + " | ".join(power["gate_a_numeric_statistics"]) + " | joint |"
    A(header)
    A("|" + "---|" * (len(power["gate_a_numeric_statistics"]) + 2))
    for n in candidates:
        row = [f"{d['power_analysis']['per_candidate'][n]['single'][g]['empirical']:.3f}"
               for g in power["gate_a_numeric_statistics"]]
        joint = d["power_analysis"]["per_candidate"][n]["joint_numeric"]["empirical"]
        A(f"| {n} | " + " | ".join(row) + f" | {joint:.3f} |")
    A("")
    A("### 8.2 联合功效（三效应情景）")
    A("")
    A("| n | empirical | 75% | 50% |")
    A("|---|---:|---:|---:|")
    for n in candidates:
        j = d["power_analysis"]["per_candidate"][n]["joint_numeric"]
        A(f"| {n} | {j['empirical']:.3f} | {j['75pct']:.3f} | {j['50pct']:.3f} |")
    A("")
    A("### 8.3 最弱条件（empirical）")
    A("")
    for n in candidates:
        w = power["weakest_gate_a_condition"][n]
        A(f"- n={n}：`{w['condition']}` = {w['power']:.3f}")
    A("")
    A("### 8.4 样本量建议（按情景）")
    A("")
    A("| 情景 | smallest_n 单项均≥0.90 | smallest_n 联合≥0.80 | smallest_n 联合≥0.90 |")
    A("|---|---:|---:|---:|")
    for scen, cell in power["sample_size_summary"].items():
        A(f"| {scen} | {cell['smallest_n_single_each_ge_0.90']} | "
          f"{cell['smallest_n_joint_numeric_ge_0.80']} | {cell['smallest_n_joint_numeric_ge_0.90']} |")
    A("")
    A("## 9. 机械状态")
    A("")
    A(f"- `PRESENCE_DIFFERENCE_SUPPORTED` = {flags['PRESENCE_DIFFERENCE_SUPPORTED']}")
    A(f"- `POSITIVE_MAGNITUDE_DECAY_SUPPORTED` = {flags['POSITIVE_MAGNITUDE_DECAY_SUPPORTED']}")
    A(f"- `ROBUST_HIGH_REBOUND_SUPPORTED` = {flags['ROBUST_HIGH_REBOUND_SUPPORTED']}")
    A(f"- **唯一 shape_recommendation = `{rec}`**")
    A("")
    A("## 10. 对 Stage 2 的建议")
    A("")
    A(_recommendation_guidance(rec))
    A("")
    A("## 11. 禁止主张")
    A("")
    A("- 不确认 U 形，也不确认逐箱单调衰减；")
    A("- 不提出碰撞、换道或“事件身份”机制结论；")
    A("- 不把 R 与证据价值的关系表述为因果作用；")
    A("- 不输出 `GATE_A_CONFIRMED`，不授权 Stage 2 采集或 router；")
    A("- 不使用“双刃剑 / dual-edged evidence”表述（Gate B 仍为 NOT_SUPPORTED）。")
    A("")
    A("## 12. 验证")
    A("")
    A(f"- 完整测试：{test_summary.get('full_suite', '见 QA')}；")
    A(f"- 运行前后 Stage 1 封存产物 hash 守恒：{test_summary.get('hash_conserved', '见 input_hashes.json')}")
    A(f"- `training_performed={d['training_performed']}`、`inference_performed={d['inference_performed']}`。")
    A(f"- 运行耗时 {d['elapsed_seconds']:.1f}s；Python {platform.python_version()}。")
    A("")
    return "\n".join(lines)


def _recommendation_guidance(recommendation: str) -> str:
    if recommendation == _plus.REVISE_NONMONOTONIC:
        return ("开发数据显示高响应反弹稳健、正响应内部无单调幅度衰减。建议在新确认集**采集前**"
                "正式修订 H2 为**非单调响应依赖**并预注册非单调形状；这不表示 U 形已被确认。"
                "是否修订上位计划与 Stage 2 由用户审批决定。")
    if recommendation == _plus.KEEP_ORIGINAL:
        return ("开发数据支持原 H2 的衰减方向且未见稳健反弹。建议保留原 H2，"
                "但 Stage 2 必须完整报告四箱曲线并加入 high-rebound 反证检查。")
    if recommendation == _plus.SHRINK_TO_PRESENCE:
        return ("开发数据仅支持“未响应/已响应”的存在性异质性，正响应内部无稳定幅度规律。"
                "建议把主张收缩为 response-presence heterogeneity，并重新评审 router 是否仍有充分动机。")
    return ("开发数据无法支持任何稳定形状。Stage 2 不应直接采集；"
            "由用户决定停止或重新制定问题。")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 Plus read-only diagnostic")
    parser.add_argument("--config", required=True, help="Path to stage1_plus config JSON")
    return parser.parse_args(argv)


def execute_and_write(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify inputs, run the diagnostic, prove hash conservation, write outputs.

    This is the post-validation flow. ``main`` runs the frozen config validation
    first; tests may call this directly with a reduced in-memory statistics
    configuration (smaller resample counts) pointed at a temporary output dir.
    """
    input_hashes = verify_sealed_inputs(config)
    hashes_before = snapshot_conservation_hashes(config)

    report = run_diagnostic(config)

    hashes_after = snapshot_conservation_hashes(config)
    if hashes_before != hashes_after:
        raise RuntimeError("Stage 1 sealed artifacts changed during the read-only run")
    report["hash_conservation"] = {"before_after_identical": True,
                                   "n_files": len(hashes_before)}
    report["input_hashes"] = input_hashes
    report["config_sha256"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    report["source_data_sha256"] = input_hashes["source_data"]

    body = json.dumps(report, sort_keys=True, ensure_ascii=False).encode("utf-8")
    report["output_signature"] = hashlib.sha256(body).hexdigest()

    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_dir / "input_hashes.json", {
        "experiment": config["experiment_name"],
        "input_hashes": input_hashes,
        "config_sha256": report["config_sha256"],
        "hash_conservation": report["hash_conservation"],
    })
    _atomic_write_json(output_dir / "report.json", report)

    decision_md = render_decision_markdown(report, config, {
        "full_suite": "见本目录 QA / 完整回归",
        "hash_conserved": "运行前后一致（before_after_identical=True）",
    })
    _atomic_write_text(output_dir / "STAGE1_PLUS_DIAGNOSTIC_DECISION_2026-09-07.md", decision_md)
    return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = json.loads(_resolve(args.config).read_text(encoding="utf-8"))
    # All validation happens before any output is created.
    validate_stage1_plus_config(config)
    return execute_and_write(config)


if __name__ == "__main__":
    main()
