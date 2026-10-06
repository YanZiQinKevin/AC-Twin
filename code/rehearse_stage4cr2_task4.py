"""Dress rehearsal for the Stage 4C-R2 one-shot screen at FULL 8,000-group scale.

Runs the exact Task-4 adjudication chain (freeze -> burn -> assemble -> load -> gates)
in a THROWAWAY tmp run dir with synthetic score/truth arrays. Nothing touches the real
stage4c_r2 namespace. Not a pytest test (full-scale IO); a one-off engineering rehearsal.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "tests"))

from test_stage4cr2_screen import (  # noqa: E402
    _config, _fake_features, _fake_score_fn, _screen_segment_seeds,
    _truth_bundle, _write_data_npz, _write_final_selection,
    _write_train_support_reports)
from test_stage4cr2_statistics import _passing_quantities  # noqa: E402
from ac_twin import stage4cp_protocol as s4cpp  # noqa: E402
from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402
from ac_twin import stage4cr2_statistics as s4cr2st  # noqa: E402
from ac_twin.checkpointing import atomic_save_npz  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402


def _write_minimal_inference_side(config, groups, seeds):
    """Minimal predictor-side artifacts (hash chain only reads the digests)."""
    paths = s4cr2st.screen_artifact_paths(config)
    seeds = np.asarray(seeds, dtype=np.int64)
    atomic_save_npz(paths["predictions"], group_seed=seeds)
    atomic_save_npz(paths["scores"], group_seed=seeds,
                    **{k: np.zeros((groups, 1)) for k in s4cpp.RISK_CLASSES})
    data_sha = sha256_file(paths["data_output"])
    paths["collection_receipt"].write_text(json.dumps({
        "plan_id": s4cr2.PLAN_ID, "role": "development_screen",
        "status": "COLLECTION_COMPLETE", "accepted_groups": int(groups),
        "data_output_sha256": data_sha,
    }, indent=2), encoding="utf-8")
    paths["inference_receipt"].write_text(json.dumps({
        "plan_id": s4cr2.PLAN_ID, "role": "development_screen",
        "status": "INFERENCE_COMPLETE", "groups": int(groups),
        "data_sha256": data_sha,
        "predictions_sha256": sha256_file(paths["predictions"]),
        "scores_sha256": sha256_file(paths["scores"]),
    }, indent=2), encoding="utf-8")
    paths["determinism_receipt"].write_text(json.dumps({
        "plan_id": s4cr2.PLAN_ID, "role": "development_screen",
        "status": "DETERMINISM_VERIFIED", "n_groups": 32,
        "bitwise_identical": True, "max_abs_diff": 0.0, "data_sha256": data_sha,
    }, indent=2), encoding="utf-8")


def _fabricated_bootstrap(quantities, n=200):
    point = s4cr2st.compute_screen_quantity_vector(
        np.arange(int(quantities["n_groups"]), dtype=np.int64), quantities)
    draws = np.tile(point, (n, 1))
    alpha = (1.0 - s4cr2.CONFIDENCE_SCREEN) / 2.0
    return s4cr2st.BootstrapResult(
        point=point, ci_lower=np.quantile(draws, alpha, axis=0),
        ci_upper=np.quantile(draws, 1.0 - alpha, axis=0), draws=draws,
        n_resamples=n, seed=s4cr2.BOOTSTRAP_ENTROPY,
        confidence=s4cr2.CONFIDENCE_SCREEN)


def main() -> int:
    groups = s4cr2.DEV_SCREEN_GROUPS
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        config = _config(tmp_path)
        paths = s4cr2st.screen_artifact_paths(config)
        paths["dev_dir"].mkdir(parents=True, exist_ok=True)
        seeds = _screen_segment_seeds(config, groups)
        _write_final_selection(paths["dev_dir"])
        _write_train_support_reports(paths["dev_dir"])
        _write_data_npz(config, groups, seeds)
        _write_minimal_inference_side(config, groups, seeds)

        started = time.perf_counter()
        npz_scores = s4cr2st.freeze_screen_direct_scores(
            config, features=_fake_features(groups, seeds), score_fn=_fake_score_fn,
            device="cpu")
        s4cr2st.burn_screen_lock(config)
        npz_quantities = s4cr2st.assemble_screen_group_quantities(
            config, bundle=_truth_bundle(groups, seeds))
        elapsed = time.perf_counter() - started
        print(f"freeze+burn+assemble at {groups} groups: {elapsed:.1f}s")
        print(f"scores npz: {npz_scores.stat().st_size / 1e6:.1f} MB; "
              f"quantities npz: {npz_quantities.stat().st_size / 1e6:.1f} MB")

        quantities = s4cr2st.load_screen_group_quantities(
            npz_quantities, paths["quantities_receipt"])
        # The synthetic truth is random noise, so gates are NOT expected to pass; what
        # must hold is that the chain runs end-to-end at full scale, the qa flags and
        # stratum counts are exact, and the terminal state is one of the plan-11 set.
        decision = s4cr2st.evaluate_screen_gates(
            quantities, _fabricated_bootstrap(quantities))
        print("terminal state (synthetic truth):", decision.status)
        print("stratum counts:", decision.task["stratum_counts"])
        print("qa:", quantities["qa"])
        assert decision.task["stratum_counts"] == {
            "tstar_low": 4000, "tstar_high": 4000,
            "interaction_low": 4000, "interaction_high": 4000}
        assert decision.status in (
            s4cr2.STATUS_DEV_READY, s4cr2.STATUS_LINE_CLOSED,
            s4cr2.STATUS_TASK_DEGENERATE, s4cr2.STATUS_MODEL_INSUFFICIENT,
            s4cr2.STATUS_INVALID)

        good = _passing_quantities()
        good_decision = s4cr2st.evaluate_screen_gates(good, _fabricated_bootstrap(good))
        assert good_decision.status == s4cr2.STATUS_DEV_READY, good_decision.status
        print("full-scale passing fixture -> STAGE4CR2_DEV_READY_FOR_CONFIRMATION OK")
    print("DRESS REHEARSAL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
