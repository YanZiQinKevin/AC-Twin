"""Frozen R-screen scheduling, readiness and single development adjudication."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ac_twin.stage3r_statistics import paired_relative_bootstrap


PLAN_ID = "AC-TWIN-RFER-STAGE3R-2026-09-10-V0.1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_disjoint_seed_schedule(
    historical: np.ndarray, *, size: int, entropy: int
) -> np.ndarray:
    historical_set = set(np.asarray(historical, dtype=np.int64).tolist())
    generator = np.random.default_rng(np.random.SeedSequence(int(entropy)))
    selected: list[int] = []
    seen: set[int] = set()
    while len(selected) < int(size):
        candidates = generator.integers(
            1, 2_000_000_000, size=max(1024, int(size)), dtype=np.int64
        )
        for raw in candidates:
            value = int(raw)
            if value not in historical_set and value not in seen:
                selected.append(value)
                seen.add(value)
                if len(selected) == int(size):
                    break
    return np.asarray(selected, dtype=np.int64)


def validate_rscreen_readiness(run_dir: str | Path) -> dict[str, object]:
    root = Path(run_dir)
    paths = {
        "power_report": root / "power" / "power_report.json",
        "screen_config": root / "rscreen" / "rscreen_config.json",
        "seed_schedule": root / "rscreen" / "rscreen_seed_schedule.npy",
        "model_lock": root / "rscreen" / "model_lock.json",
        "finalist": root / "finalist" / "finalist_receipt.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"R-screen readiness incomplete: missing {missing}")
    config = json.loads(paths["screen_config"].read_text(encoding="utf-8"))
    if config.get("plan_id") != PLAN_ID:
        raise ValueError("R-screen plan_id mismatch")
    expected = {
        "power_report_sha256": paths["power_report"],
        "model_lock_sha256": paths["model_lock"],
        "finalist_receipt_sha256": paths["finalist"],
    }
    for key, path in expected.items():
        if config.get(key) != sha256_file(path):
            raise ValueError(f"R-screen hash mismatch: {key}")
    schedule = np.load(paths["seed_schedule"]).astype(np.int64)
    schedule_hash = hashlib.sha256(schedule.tobytes()).hexdigest()
    if config.get("seed_schedule_sha256") != schedule_hash:
        raise ValueError("R-screen seed schedule hash mismatch")
    if int(config.get("seed_schedule_size", -1)) != int(schedule.size):
        raise ValueError("R-screen seed schedule size mismatch")
    return {
        "status": "RSCREEN_READINESS_PASS",
        "n_pairs": int(config["n_pairs"]),
        "schedule_size": int(schedule.size),
        "hashes_valid": True,
    }


def adjudicate_rscreen(
    errors: np.ndarray,
    condition_names: Sequence[str],
    *,
    report_path: str | Path,
    seed: int,
    n_resamples: int,
    strata: Mapping[str, np.ndarray] | None = None,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    path = Path(report_path)
    if path.exists():
        raise FileExistsError(f"R-screen already adjudicated: {path}")
    values = np.asarray(errors, dtype=np.float64)
    names = list(condition_names)
    required = [
        "main",
        "full",
        "history",
        "best_constant",
        "end_to_end",
        "direct",
        "response_only",
    ]
    if names != required or values.ndim != 3 or values.shape[:2] != (7, 5):
        raise ValueError("R-screen errors must be ordered 7 conditions by 5 pairs by episode")
    contrasts: dict[str, dict[str, float]] = {}
    main = values[0]
    for index, name in enumerate(names[1:], start=1):
        contrasts[name] = paired_relative_bootstrap(
            values[index].mean(axis=0),
            main.mean(axis=0),
            seed=int(seed) + index,
            n_resamples=int(n_resamples),
        )
    strongest_index = 1 + int(np.argmin(values[1:].mean(axis=(1, 2))))
    strongest = names[strongest_index]
    pair_relative = 1.0 - main.mean(axis=1) / values[strongest_index].mean(axis=1)
    positive_pairs = int((pair_relative > 0).sum())
    material_reversal = False
    stratum_reports: dict[str, dict[str, float]] = {}
    if strata:
        for label, raw in strata.items():
            labels = np.asarray(raw)
            if labels.shape != (values.shape[2],):
                raise ValueError(f"stratum {label} does not align with episodes")
            for value in np.unique(labels):
                selected = labels == value
                if int(selected.sum()) < 30:
                    continue
                result = paired_relative_bootstrap(
                    values[strongest_index][:, selected].mean(axis=0),
                    main[:, selected].mean(axis=0),
                    seed=int(seed) + 100 + len(stratum_reports),
                    n_resamples=int(n_resamples),
                )
                key = f"{label}={value}"
                stratum_reports[key] = result
                if result["point_relative"] <= -0.02 and result["ci80_upper"] < 0:
                    material_reversal = True
    passed = (
        all(
            item["point_relative"] >= 0.05 and item["ci80_lower"] > 0
            for item in contrasts.values()
        )
        and positive_pairs >= 4
        and not material_reversal
    )
    report = {
        "plan_id": PLAN_ID,
        "status": (
            "STAGE3R_READY_FOR_GATE_C_DESIGN"
            if passed
            else "STAGE3R_DEV_NOT_READY_FOR_CONFIRMATION"
        ),
        "passed": bool(passed),
        "contrasts": contrasts,
        "strongest_baseline": strongest,
        "positive_pairs_against_strongest": positive_pairs,
        "pair_relative_against_strongest": pair_relative.tolist(),
        "material_stratum_reversal": material_reversal,
        "strata": stratum_reports,
        "provenance": dict(provenance or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return report


__all__ = [
    "PLAN_ID",
    "adjudicate_rscreen",
    "make_disjoint_seed_schedule",
    "sha256_file",
    "validate_rscreen_readiness",
]
