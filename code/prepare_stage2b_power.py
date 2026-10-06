"""Stage 2B v0.4 confirm sample-size power analysis.

Numeric effects come only from the fresh development validation split.  The
v0.3 profile-B pilot is used separately for conservative positive-bin support
sensitivity and never contributes a model error or value estimate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage2b_power as bp
from ac_twin import stage2b_training as bt
from ac_twin.stage1_data import compute_target_effect, compute_valid_mask, sha256_file
from ac_twin.stage1_statistics import assign_response_bin, position_errors, response_magnitudes
from ac_twin.stage1_plus_statistics import make_group_contrast_statistic, make_group_v_evt_statistic
from ac_twin.stage2_protocol import estimate_stage2_joint_power


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_once_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"power report is create-once: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        return {name: archive[name] for name in archive.files}


def _verify_receipt(data_path: Path, receipt_path: Path, expected: int) -> dict:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("partial", True) or int(receipt.get("accepted_pairs", -1)) != expected:
        raise ValueError(f"incomplete source receipt: {receipt_path}")
    if receipt.get("data_output_sha256") != sha256_file(data_path):
        raise ValueError(f"source hash mismatch: {data_path}")
    return receipt


def build_validation_inputs(data: dict[str, np.ndarray], qualification: Path
                            ) -> tuple[dict[str, object], np.ndarray]:
    arrays, thresholds = bt.prepare_development_arrays(data, suffix_steps=10)
    val = arrays["indices"]["validation"]
    target = arrays["target"][val]
    valid = arrays["valid_mask"][val].copy()
    valid[:, :, 0] = False
    bins = assign_response_bin(arrays["r_pos"][val], thresholds)

    errors: dict[str, np.ndarray] = {}
    for condition in ("uniform_suffix", "shuffled_suffix"):
        per_seed = []
        for seed in bt.FORMAL_SEEDS:
            pred_norm = np.load(qualification / f"{condition}_seed{seed}_val_prediction.npy",
                                allow_pickle=False)
            pred = pred_norm * arrays["target_std"] + arrays["target_mean"]
            per_seed.append(position_errors(pred, target))
        errors[condition] = np.mean(np.stack(per_seed), axis=0)
    l_suffix = errors["uniform_suffix"]
    l_shuffled = errors["shuffled_suffix"]
    masks = {value: valid & (bins == value) for value in range(4)}
    positive = valid & (bins >= 1)
    statistics = {
        "c_presence": make_group_contrast_statistic(l_suffix, l_shuffled,
                                                     masks[0], positive),
        "v_zero": make_group_v_evt_statistic(l_suffix, l_shuffled, masks[0]),
        "c_high_low": make_group_contrast_statistic(l_suffix, l_shuffled,
                                                     masks[3], masks[1]),
        "c_high_mid": make_group_contrast_statistic(l_suffix, l_shuffled,
                                                     masks[3], masks[2]),
        "v_high": make_group_v_evt_statistic(l_suffix, l_shuffled, masks[3]),
    }
    return {"statistics": statistics, "thresholds": thresholds,
            "bin_id": bins, "valid": valid}, val


def pilot_support_flags(data: dict[str, np.ndarray], thresholds: dict) -> np.ndarray:
    factual = np.asarray(data["object_factual_future"], dtype=np.float32)
    counterfactual = np.asarray(data["object_counterfactual_future"], dtype=np.float32)
    history = np.asarray(data["object_history"], dtype=np.float32)
    r_pos, _ = response_magnitudes(counterfactual, factual)
    valid = compute_valid_mask(history, factual, counterfactual)
    valid[:, :, 0] = False
    return bp.positive_episode_flags(assign_response_bin(r_pos, thresholds), valid)


def run(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("plan_id") != bt.V04_PLAN_ID:
        raise ValueError("wrong Stage 2B v0.4 plan_id")
    candidates = tuple(int(n) for n in cfg["candidate_episode_counts"])
    if candidates != bp.CONFIRM_CANDIDATES or any(n % 6 for n in candidates):
        raise ValueError("confirm candidates must be 1200/1800/2400/3600 and divisible by 6")
    if (int(cfg["monte_carlo_replicates"]) != 10000
            or float(cfg["confidence_level"]) != 0.95
            or float(cfg["effect_shrinkage"]) != 0.75
            or float(cfg["practical_point_screen"]) != 0.05):
        raise ValueError("power contract drift")

    dev_path, dev_receipt = resolve(cfg["development_data"]), resolve(cfg["development_receipt"])
    pilot_path, pilot_receipt = resolve(cfg["pilot_data"]), resolve(cfg["pilot_receipt"])
    qualification = resolve(cfg["qualification_dir"])
    gate_path = qualification / "gate_s_report.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != bt.GATE_S_PASS or gate.get("variant") != "C0":
        raise RuntimeError("C0 Gate S must pass before power analysis")
    _verify_receipt(dev_path, dev_receipt, 6000)
    _verify_receipt(pilot_path, pilot_receipt, 600)
    inputs, _ = build_validation_inputs(_load_npz(dev_path), qualification)
    validation_flags = bp.positive_episode_flags(inputs["bin_id"], inputs["valid"])
    pilot_flags = pilot_support_flags(_load_npz(pilot_path), inputs["thresholds"])

    numeric: dict[int, dict] = {}
    joint: dict[int, float] = {}
    seed = int(cfg["power_seed"])
    for offset, n in enumerate(candidates):
        print(f"[Stage2B-power] numeric candidate={n} ({offset + 1}/{len(candidates)})",
              flush=True)
        result = estimate_stage2_joint_power(
            statistics=inputs["statistics"], n=n,
            monte_carlo_replicates=int(cfg["monte_carlo_replicates"]),
            seed=seed + offset, confidence_level=float(cfg["confidence_level"]),
            point_screen=float(cfg["practical_point_screen"]))
        numeric[n] = result
        joint[n] = float(result["scenarios"]["75pct"]["joint"])
        print(f"[Stage2B-power] candidate={n} joint75={joint[n]:.4f}", flush=True)

    validation_support = bp.estimate_support_power(
        validation_flags, candidates, replicates=int(cfg["monte_carlo_replicates"]),
        min_count=int(cfg["min_positive_bin_episodes"]), seed=seed + 100)
    pilot_support = bp.estimate_support_power(
        pilot_flags, candidates, replicates=int(cfg["monte_carlo_replicates"]),
        min_count=int(cfg["min_positive_bin_episodes"]), seed=seed + 200)
    selected = bp.select_candidate_n(
        joint, validation_support, pilot_support,
        numeric_target=float(cfg["numeric_joint_power_target"]),
        support_target=float(cfg["positive_bin_support_target"]))

    weakest: dict[str, object] = {}
    for n in candidates:
        cells = {"numeric_joint_75pct": joint[n]}
        cells.update({f"validation_{k}": v for k, v in validation_support[n].items()})
        cells.update({f"pilot_{k}": v for k, v in pilot_support[n].items()})
        key = min(cells, key=cells.get)
        weakest[str(n)] = {"condition": key, "probability": float(cells[key])}

    report = {
        "plan_id": bt.V04_PLAN_ID,
        "status": ("V04_CONFIRM_POWER_FEASIBLE" if selected is not None
                   else "STAGE2B_CONFIRMATION_UNDERPOWERED"),
        "selected_episode_count": selected,
        "candidate_episode_counts": list(candidates),
        "numeric_power": {str(k): v for k, v in numeric.items()},
        "numeric_joint_75pct": {str(k): v for k, v in joint.items()},
        "validation_positive_bin_support": {str(k): v for k, v in validation_support.items()},
        "pilot_positive_bin_support_sensitivity": {str(k): v for k, v in pilot_support.items()},
        "binding_condition": weakest,
        "pilot_numeric_effects_used": False,
        "pilot_note": "Pilot B contributes response-bin occupancy sensitivity only; no model/value estimate.",
        "thresholds": inputs["thresholds"],
        "sources": {
            "development_data_sha256": sha256_file(dev_path),
            "development_receipt_sha256": sha256_file(dev_receipt),
            "pilot_data_sha256": sha256_file(pilot_path),
            "pilot_receipt_sha256": sha256_file(pilot_receipt),
            "gate_s_report_sha256": sha256_file(gate_path),
        },
        "implementation_sha256": {
            "runner": sha256_file(Path(__file__)),
            "power": sha256_file(ROOT / "code/ac_twin/stage2b_power.py"),
            "stage2_protocol": sha256_file(ROOT / "code/ac_twin/stage2_protocol.py"),
            "stage1_plus_statistics": sha256_file(ROOT / "code/ac_twin/stage1_plus_statistics.py"),
        },
        "config_sha256": sha256_file(config_path),
    }
    output = resolve(cfg["output"])
    write_once_json(output, report)
    print(f"[Stage2B-power] status={report['status']} selected={selected}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(resolve(args.config))


if __name__ == "__main__":
    main()
