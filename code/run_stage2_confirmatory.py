"""Stage 2 four-phase confirmatory runner.

Phases:
- ``create-lock``: write the protocol lock exactly once (refuses to overwrite).
- ``verify-lock``: read-only re-verification of every locked artifact hash.
- ``infer``: verify the lock, then run frozen-checkpoint inference; writes only
  predictions, manifests and a prediction receipt. It never prints or writes any
  Gate/V_evt/CI quantity.
- ``unlock-and-analyze``: the single unblinding. Verifies the lock + data + all
  manifests + all predictions + receipt hashes, computes Gate Q, runs the frozen
  statistics once, and atomically writes one ``report.json`` and one
  ``STAGE2_CONFIRMATORY_DECISION.md``. An existing report forbids re-unlocking.

For the smoke config the unlock writes an ``ENGINEERING_SMOKE_ONLY`` artifact and
executes no formal Gate S*/A/B.
"""
from __future__ import annotations

import os
# Avoid the torch/MKL duplicate-OpenMP abort (OMP Error #15) when NumPy LAPACK
# solvers run while torch is resident; sequential MKL is deterministic and does
# not change results. Must be set before the first numpy/torch import.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage2_protocol as sp  # noqa: E402
from ac_twin import stage2_inference as si  # noqa: E402
from ac_twin import stage2_statistics as ss  # noqa: E402
from ac_twin.checkpointing import canonical_fingerprint  # noqa: E402

REPORT_NAME = "report.json"
DECISION_NAME = "STAGE2_CONFIRMATORY_DECISION.md"


def _load_config(config_path: str | Path) -> dict:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


# --- output signature (canonical, recomposable from the on-disk JSON) ---------------


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _normalize(obj: Any) -> Any:
    """JSON round-trip: dict keys become strings, NumPy scalars become Python."""
    return json.loads(json.dumps(obj, ensure_ascii=False, default=_json_scalar))


def _signature(normalized_obj: Mapping[str, Any]) -> str:
    body = json.dumps(normalized_obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _atomic_write_json(target: Path, obj: Mapping[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)


def _atomic_write_text(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)


def _signed_report(report: Mapping[str, Any]) -> dict:
    normalized = _normalize(report)
    normalized["output_signature"] = _signature(normalized)
    return normalized


# --- Gate Q (technical validity) -----------------------------------------------------


def gate_q(config: Mapping[str, Any], output_dir: Path, lock: Mapping[str, Any],
           expected_lock_sha256: str) -> tuple[bool, dict]:
    detail: dict[str, Any] = {}
    try:
        sp.verify_protocol_lock(output_dir)
        detail["lock_artifact_hashes"] = True
    except Exception as exc:  # noqa: BLE001
        detail["lock_artifact_hashes"] = False
        detail["lock_error"] = str(exc)
    detail["expected_lock_sha256_match"] = bool(lock["lock_sha256"] == expected_lock_sha256)

    schedule = np.load(output_dir / "seed_schedule.npy")
    historical = sp.scan_historical_episode_seeds(ROOT / "data")
    try:
        sp.validate_seed_schedule(schedule, historical)
        detail["seed_schedule_valid_zero_overlap"] = True
    except ValueError:
        detail["seed_schedule_valid_zero_overlap"] = False

    receipt_path = sp._resolve(config["collection_receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    detail["accepted_pairs_matches"] = bool(
        int(receipt["accepted_pairs"]) == int(config["n_pairs"]))
    detail["prefix_consistency"] = bool(float(receipt["prefix_max_abs_diff"]) <= 1e-6)
    detail["identity_replay"] = bool(float(receipt["identity_max_abs_diff"]) <= 1e-6)
    detail["no_training"] = bool(
        receipt.get("training_performed", False) is False)

    pred_receipt = json.loads((output_dir / "prediction_receipt.json").read_text(encoding="utf-8"))
    detail["inference_no_training"] = bool(
        pred_receipt.get("training_performed", False) is False
        and pred_receipt.get("checkpoint_modified", True) is False)

    predictions_ok, manifests_ok = True, True
    for name, expected in pred_receipt["prediction_sha256"].items():
        if not (output_dir / name).is_file() or sp.sha256_file(output_dir / name) != expected:
            predictions_ok = False
    for name, expected in pred_receipt["manifest_sha256"].items():
        if not (output_dir / name).is_file() or sp.sha256_file(output_dir / name) != expected:
            manifests_ok = False
    detail["predictions_hash_verified"] = bool(predictions_ok)
    detail["manifests_hash_verified"] = bool(manifests_ok)
    detail["data_output_hash_verified"] = bool(
        sp.sha256_file(sp._resolve(config["data_output"])) == pred_receipt["data_output_sha256"])

    return bool(all(v for k, v in detail.items() if isinstance(v, bool))), detail


# --- phases ---------------------------------------------------------------------------


def cmd_create_lock(config_path: str | Path) -> dict:
    config = _load_config(config_path)
    output_dir = sp._resolve(config["output_dir"])
    lock = sp.write_protocol_lock(config_path, output_dir)
    print(f"[create-lock] plan={lock['plan_id']} schedule={lock['seed_schedule_size']} "
          f"historical_seeds={lock['n_historical_episode_seeds']} "
          f"checkpoints={len(lock['checkpoint_sha256'])}", flush=True)
    print(f"[create-lock] lock_sha256={lock['lock_sha256']}", flush=True)
    return lock


def cmd_verify_lock(config_path: str | Path) -> dict:
    config = _load_config(config_path)
    output_dir = sp._resolve(config["output_dir"])
    lock = sp.verify_protocol_lock(output_dir)
    schedule = np.load(output_dir / "seed_schedule.npy")
    historical = sp.scan_historical_episode_seeds(ROOT / "data")
    sp.validate_seed_schedule(schedule, historical)
    print(f"[verify-lock] historical_episode_seeds={lock['n_historical_episode_seeds']} "
          f"candidate_schedule={len(schedule)} zero_overlap=True "
          f"checkpoints_ok={len(lock['checkpoint_sha256'])}", flush=True)
    print("[verify-lock] PROTOCOL_LOCK_VERIFIED", flush=True)
    return lock


def cmd_infer(config_path: str | Path, device: str) -> dict:
    config = _load_config(config_path)
    sp.validate_stage2_config(config)
    output_dir = sp._resolve(config["output_dir"])
    lock = sp.verify_protocol_lock(output_dir)
    receipt = si.run_inference(config, device=device, output_dir=output_dir, lock=lock)
    n_pred = len(receipt["prediction_sha256"])
    n_manifest = len(receipt["manifest_sha256"])
    # Never print Gate/V_evt/CI quantities here.
    print(f"[infer] predictions={n_pred} manifests={n_manifest} "
          f"episodes={receipt['n_episodes']} device={device}", flush=True)
    if "verification" in receipt:
        print(f"[infer] {receipt['verification']}", flush=True)
    return receipt


def _build_truth(config: Mapping[str, Any]) -> dict:
    with np.load(sp._resolve(config["data_output"]), allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    normalizer = si.load_frozen_normalizers(sp._resolve(config["normalization_npz"]))
    arrays = si.prepare_holdout_arrays(data, normalizer,
                                       factual_suffix_steps=int(config["factual_suffix_steps"]))
    return {
        "r_pos": arrays["r_pos"], "target": arrays["target"],
        "valid_mask": arrays["valid_mask"], "cf_action": arrays["cf_action"],
        "cf_duration": arrays["cf_duration"], "episode_seed": arrays["episode_seed"],
        "history": np.asarray(data["object_history"], dtype=np.float32),
        "factual": np.asarray(data["object_factual_future"], dtype=np.float32),
        "factual_actions": arrays["factual_actions"],
        "counterfactual_actions": arrays["counterfactual_actions"],
        "visible_count": arrays["visible_count"],
    }


def _load_predictions(output_dir: Path) -> dict:
    def load(name):
        return np.load(output_dir / name, allow_pickle=False)
    seeds = list(sp.TRAINING_SEEDS)
    main = {c: np.stack([load(f"{c}_seed{s}_prediction.npy") for s in seeds])
            for c in sp.CONDITIONS}
    m1_swap = np.stack([load(f"uniform_suffix_seed{s}_primary_swap_prediction.npy")
                        for s in seeds])
    alt = {int(d): np.stack([load(f"shuffled_suffix_seed{s}_donor{d}_prediction.npy")
                             for s in seeds]) for d in sp.ALTERNATE_DONOR_SEEDS}
    return {
        "main": main, "m1_swap": m1_swap, "alt_donor": alt,
        "factual_summaries": load("uniform_suffix_factual_summaries_seed_mean.npy"),
        "factual_active": load("uniform_suffix_factual_active.npy"),
    }


def _load_manifests(output_dir: Path) -> dict:
    primary = json.loads((output_dir /
                          f"confirmatory_manifest_donor{sp.PRIMARY_DERANGEMENT_SEED}.json")
                         .read_text(encoding="utf-8"))
    alternates = {int(d): json.loads(
        (output_dir / f"confirmatory_manifest_donor{d}.json").read_text(encoding="utf-8"))
        for d in sp.ALTERNATE_DONOR_SEEDS}
    return {"primary": primary, "alternates": alternates}


def _load_frozen_params(output_dir: Path) -> dict:
    with np.load(output_dir / "frozen_gate_s_star_params.npz") as archive:
        return {key: archive[key] for key in archive.files}


def cmd_unlock_and_analyze(config_path: str | Path, expected_lock_sha256: str) -> dict:
    config = _load_config(config_path)
    sp.validate_stage2_config(config)
    output_dir = sp._resolve(config["output_dir"])
    report_path = output_dir / REPORT_NAME
    if report_path.exists():
        raise FileExistsError(
            f"report already exists; a second unblinding is forbidden: {report_path}")
    lock = sp.verify_protocol_lock(output_dir)
    if lock["lock_sha256"] != expected_lock_sha256:
        raise ValueError("expected-lock-sha256 does not match the locked protocol")

    gate_q_pass, gate_q_detail = gate_q(config, output_dir, lock, expected_lock_sha256)
    started = time.time()

    if config["analysis_status"] == sp.STATUS_SMOKE:
        report = {
            "experiment_name": config["experiment_name"],
            "analysis_status": sp.STATUS_SMOKE,
            "plan_id": config["plan_id"],
            "note": ("Engineering smoke only: validates the pipeline mechanics. "
                     "No formal Gate S*/A/B is executed and no scientific claim is made."),
            "gate_q": gate_q_detail,
            "gate_q_pass": gate_q_pass,
            "lock_sha256": lock["lock_sha256"],
            "elapsed_seconds": round(time.time() - started, 3),
        }
        signed = _signed_report(report)
        _atomic_write_json(report_path, signed)
        decision = _render_smoke_decision(signed)
        _atomic_write_text(output_dir / DECISION_NAME, decision)
        print(f"[unlock] status={sp.STATUS_SMOKE} gate_q_pass={gate_q_pass}", flush=True)
        return signed

    # Full confirmatory analysis (single unblinding).
    truth = _build_truth(config)
    predictions = _load_predictions(output_dir)
    frozen_params = _load_frozen_params(output_dir)
    manifest = _load_manifests(output_dir)
    report = ss.compute_stage2_statistics(truth, predictions, frozen_params, manifest, config)
    final_status = ss.judge_gate_a(report, gate_q_pass=gate_q_pass)
    gate_b_status = ss.judge_gate_b(report)

    full_report = {
        "experiment_name": config["experiment_name"],
        "analysis_status": config["analysis_status"],
        "plan_id": config["plan_id"],
        "claim_boundary": ("confirmatory_holdout_single_unblinding; "
                           "nonmonotonic_response_regime_heterogeneity_only; "
                           "not_u_shape_not_causal_not_mechanism; "
                           "no_dual_edged_wording_unless_gate_b_supported"),
        "gate_q": gate_q_detail,
        "gate_q_pass": gate_q_pass,
        "lock_sha256": lock["lock_sha256"],
        "statistics": report,
        "final_status": final_status,
        "gate_b": gate_b_status,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    signed = _signed_report(full_report)
    _atomic_write_json(report_path, signed)
    decision = _render_decision(signed)
    _atomic_write_text(output_dir / DECISION_NAME, decision)
    print(f"[unlock] final_status={final_status} gate_b={gate_b_status}", flush=True)
    return signed


# --- decision documents -----------------------------------------------------------------


def _fmt(entry: Mapping[str, Any]) -> str:
    return (f"{entry['point']:.4f} [{entry['ci_lower']:.4f}, {entry['ci_upper']:.4f}]")


def _render_smoke_decision(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 2 Confirmatory — ENGINEERING_SMOKE_ONLY",
        "",
        f"- analysis_status: `{report['analysis_status']}`",
        "- 本输出仅验证管线机械正确性；**未执行任何正式 Gate S*/A/B**，不构成科学结论。",
        f"- gate_q_pass: `{report['gate_q_pass']}`",
        f"- lock_sha256: `{report['lock_sha256']}`",
        f"- output_signature: `{report['output_signature']}`",
        "",
    ]
    return "\n".join(lines)


def _render_decision(report: Mapping[str, Any]) -> str:
    s = report["statistics"]
    lines = []
    A = lines.append
    A("# Stage 2 确认性 Holdout — 机械裁判决策文档")
    A("")
    A(f"> 计划编号：`{report['plan_id']}`  ")
    A(f"> claim_boundary：`{report['claim_boundary']}`")
    A("")
    A(f"- lock_sha256：`{report['lock_sha256']}`")
    A(f"- Gate Q 通过：`{report['gate_q_pass']}`")
    A(f"- output_signature：`{report['output_signature']}`")
    A("")
    A("## Gate Q 明细")
    A("")
    for key, value in report["gate_q"].items():
        A(f"- `{key}` = {value}")
    A("")
    A("## 估计规模")
    A("")
    A(f"- accepted episodes = {s['n_episodes']}；有效非自车 agent-time = {s['n_agent_time_valid']}")
    A(f"- 各正响应箱独立 episode 数 = {s['bin_episode_counts']}")
    A("")
    A("## 四箱 V_evt")
    A("")
    A("| 响应箱 | V_evt | agent-time | episodes |")
    A("|---|---:|---:|---:|")
    for name in ss.BIN_NAMES:
        b = s["bins"][name]
        v = b["V_evt"] if b["V_evt"] is not None else float("nan")
        A(f"| {name} | {v:.6f} | {b['n_agent_time']} | {b['n_episodes']} |")
    A("")
    A("## 主对比（裁判 CI 仅落在三个光滑对比上）")
    A("")
    A("| 量 | 点估计 [95% CI] | 0.05 筛选 |")
    A("|---|---:|:---:|")
    A(f"| `C_presence` | {_fmt(s['c_presence'])} | {s['point_screen_checks']['c_presence']} |")
    A(f"| `C_high_low` | {_fmt(s['c_high_low'])} | {s['point_screen_checks']['c_high_low']} |")
    A(f"| `C_high_mid` | {_fmt(s['c_high_mid'])} | {s['point_screen_checks']['c_high_mid']} |")
    A(f"| `V_zero` | {_fmt(s['v_zero'])} | — |")
    A(f"| `V_high` | {_fmt(s['v_high'])} | — |")
    A(f"| `V_positive_all` | {_fmt(s['v_positive_all'])} | — |")
    A("")
    A(f"- `C_rebound`（无裁判 CI 的摘要点估计）= {s['c_rebound_point']:.4f}")
    A("")
    A("## 稳健性（均不得反转为负）")
    A("")
    rob = s["robustness"]
    for group, checks in rob["checks"].items():
        A(f"- {group}: {checks}")
    A(f"- 合并 all_positive = {rob['all_positive']}")
    A("")
    A("## Alternate donor 敏感性（点估计）")
    A("")
    for donor_seed, contrasts in s["alternate_donor"].items():
        A(f"- donor {donor_seed}: "
          + ", ".join(f"{k}={v:.4f}" for k, v in contrasts.items()))
    A("")
    A("## Gate S* / Gate B")
    A("")
    gs = s["gate_s_star"]
    A(f"- S1*={gs['S1']} S2*={gs['S2']} S3*={gs['S3']} -> pass={gs['pass']}")
    A(f"- Gate B = `{report['gate_b']}`")
    A("")
    A("## 次分析（secondary_only，无 Gate A 方向要求）")
    A("")
    sec = s["secondary"]
    A(f"- positive-only Spearman = {_fmt(sec['positive_only_spearman'])}")
    A(f"- positive-only trend log1p(R) = {_fmt(sec['positive_only_trend_log_r'])}")
    A(f"- 原 V_low - V_high = {_fmt(sec['original_v_evt_low_minus_high'])}")
    A("")
    A("## 唯一机械状态")
    A("")
    A(f"**`{report['final_status']}`**")
    A("")
    return "\n".join(lines)


# --- CLI -------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 confirmatory runner")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("create-lock", "verify-lock", "infer", "unlock-and-analyze"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "infer":
            p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        if name == "unlock-and-analyze":
            p.add_argument("--expected-lock-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Any:
    args = parse_args(argv)
    config_path = sp._resolve(args.config)
    if args.command == "create-lock":
        return cmd_create_lock(config_path)
    if args.command == "verify-lock":
        return cmd_verify_lock(config_path)
    if args.command == "infer":
        return cmd_infer(config_path, args.device)
    if args.command == "unlock-and-analyze":
        return cmd_unlock_and_analyze(config_path, args.expected_lock_sha256)
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
