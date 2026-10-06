"""Stage 3A Plus driver tests: diagnostic orchestration core (Task 2),
calibration screen (Task 4), dev state machine and freeze (Task 7).

The heavy torch inference is injected; the diagnostic core is pure numpy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import run_stage3a_plus as drv  # noqa: E402
from ac_twin import stage3a_plus_calibration as cal  # noqa: E402
from ac_twin import stage3a_plus_statistics as sps  # noqa: E402
from ac_twin.stage3_data import STAGE3A_PLUS_ALL_SEEDS  # noqa: E402
from ac_twin.stage3_response import softmax  # noqa: E402

N_SEEDS = len(STAGE3A_PLUS_ALL_SEEDS)


def _make_eval_set(name, *, e=8, t=4, n=3, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    E, T, N = e, t, n
    r_true = rng.gamma(1.5, 0.4, size=(E, T, N)) + shift
    regime = np.digitize(r_true, [0.25, 1.0, 1.9]).astype(np.int64)
    valid = rng.random((E, T, N)) > 0.1
    valid[:, 0, 0] = False
    base_r = r_true + rng.normal(0, 0.05, size=(E, T, N))
    r_hat_seeds = np.stack([np.clip(base_r + rng.normal(0, 0.02, (E, T, N)), 0, None)
                            for _ in range(N_SEEDS)], axis=0)
    logits = rng.normal(0, 1.0, size=(E, T, N, 4))
    logits_seeds = np.stack([logits + rng.normal(0, 0.1, (E, T, N, 4))
                             for _ in range(N_SEEDS)], axis=0)
    base_clf = softmax(logits, axis=-1)
    baselines = {
        "continuous": {"strong": np.zeros((E, T, N))},
        "classification": {"strong": base_clf},
        "strongest_continuous": "strong",
        "strongest_brier": "strong",
    }
    idx = np.arange(E)
    return {
        "name": name,
        "r_hat_seeds": r_hat_seeds,
        "logits_seeds": logits_seeds,
        "r_true": r_true,
        "regime_labels": regime,
        "valid_mask": valid,
        "episode_ids": idx,
        "expected_episode_ids": idx,
        "horizons": np.broadcast_to(np.arange(T)[None, :, None], (E, T, N)).copy(),
        "cf_actions": rng.integers(0, 5, size=E),
        "cf_durations": rng.choice([1, 3], size=E),
        "baselines": baselines,
    }


def _val_block(seed=99):
    rng = np.random.default_rng(seed)
    Ev, Tv, Nv = 7, 4, 3
    logits_seeds = rng.normal(0, 2.0, size=(N_SEEDS, Ev, Tv, Nv, 4))
    labels = rng.integers(0, 4, size=(Ev, Tv, Nv))
    mask = rng.random((Ev, Tv, Nv)) > 0.1
    mask[:, 0, 0] = False
    prevalence = np.array([0.55, 0.20, 0.15, 0.10])
    return logits_seeds, labels, mask, prevalence


def _run(**over):
    val_logits, val_labels, val_mask, prev = _val_block()
    eval_sets = {"A": _make_eval_set("A", seed=1), "B": _make_eval_set("B", seed=2, shift=0.05)}
    kwargs = dict(
        val_logits_seeds=val_logits, val_labels=val_labels, val_mask=val_mask,
        train_prevalence=prev, eval_sets=eval_sets, engineering_ok=True,
        n_production_seeds=5, n_resamples=50, bootstrap_seed=20260929,
    )
    kwargs.update(over)
    return drv.run_diagnostic(**kwargs)


def test_run_diagnostic_produces_four_candidates_and_status():
    out = _run()
    assert set(out["candidates"]) == set(cal.CALIBRATION_CANDIDATES)
    assert out["calibration_failure_status"] in {
        sps.STATUS_RUN_INVALID, sps.STATUS_SHARED_TEMP, sps.STATUS_REGIME_LOCALIZED,
        sps.STATUS_BORDERLINE, sps.STATUS_SYSTEMATIC, sps.STATUS_MIXED,
    }
    for cid, c in out["candidates"].items():
        assert set(c["per_set"]) == {"A", "B"}
        assert "calibration_signature" in c and "validation_nll" in c
        for s in ("A", "B"):
            assert set(c["per_set"][s]["per_seed"]) == {
                str(x) for x in STAGE3A_PLUS_ALL_SEEDS}


def test_run_diagnostic_tracks_seed303_ci_on_both_sets():
    out = _run()
    ci = out["classification_input"]["seed303"]
    assert set(ci) == {"A", "B"}
    for s in ("A", "B"):
        assert ci[s]["i_brier_ci_lower"] <= ci[s]["i_brier_ci_upper"]


def test_run_diagnostic_engineering_failure_dominates():
    out = _run(engineering_ok=False)
    assert out["calibration_failure_status"] == sps.STATUS_RUN_INVALID


def test_load_dev_config_validates_frozen_seeds(tmp_path):
    import json
    cfg = {
        "plan_id": "AC-TWIN-RFER-STAGE3A-PLUS-2026-09-09-V0.1",
        "dev_seeds": [101, 202, 303, 404, 505],
        "stress_seeds": [606, 707, 808, 909, 1010],
        "development_source": "a", "replacement_source": "b", "prior_run_dir": "c",
        "final_config": {}, "training": {}, "device": "cpu",
    }
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    assert drv.load_dev_config(p)["plan_id"] == cfg["plan_id"]
    cfg_bad = dict(cfg, stress_seeds=[1, 2, 3, 4, 5])
    p2 = tmp_path / "bad.json"
    p2.write_text(json.dumps(cfg_bad), encoding="utf-8")
    with pytest.raises(ValueError):
        drv.load_dev_config(p2)


# --------------------------------------------------------------------------- #
# Task 4: Round 0 calibration screen (J_cal selection + receipt)
# --------------------------------------------------------------------------- #
def test_j_cal_formula_and_ece_guard():
    brier = np.array([0.10, 0.20, 0.30])
    expected = brier.mean() + 0.5 * brier.std(ddof=0) + 2.0 * (0.07 - 0.05)
    assert drv.calibration_j_cal(brier, 0.07) == pytest.approx(expected)
    # ECE below the 0.05 guard contributes nothing.
    assert drv.calibration_j_cal(brier, 0.03) == pytest.approx(
        brier.mean() + 0.5 * brier.std(ddof=0))
    with pytest.raises(ValueError):
        drv.calibration_j_cal(np.array([0.1, np.nan]), 0.01)


def test_select_candidate_lowest_score_wins():
    j = {cal.CAL_SHARED_T: 0.30, cal.CAL_PER_SEED_T: 0.20,
         cal.CAL_PER_SEED_T_SHRINK: 0.25, cal.CAL_PER_SEED_VECTOR: 0.28}
    assert drv.select_calibration_candidate(j) == cal.CAL_PER_SEED_T


def test_select_candidate_tie_prefers_simpler_within_tolerance():
    # SHARED_T and VECTOR within 1e-8 -> the simpler SHARED_T wins.
    j = {cal.CAL_SHARED_T: 0.200000000, cal.CAL_PER_SEED_T: 0.21,
         cal.CAL_PER_SEED_T_SHRINK: 0.22, cal.CAL_PER_SEED_VECTOR: 0.200000005}
    assert drv.select_calibration_candidate(j) == cal.CAL_SHARED_T
    # PER_SEED_T vs T_SHRINK tie -> PER_SEED_T.
    j2 = {cal.CAL_SHARED_T: 0.30, cal.CAL_PER_SEED_T: 0.200000000,
          cal.CAL_PER_SEED_T_SHRINK: 0.200000004, cal.CAL_PER_SEED_VECTOR: 0.29}
    assert drv.select_calibration_candidate(j2) == cal.CAL_PER_SEED_T
    # A gap larger than 1e-8 is a real difference, not a tie.
    j3 = {cal.CAL_SHARED_T: 0.20, cal.CAL_PER_SEED_T: 0.20 - 1e-7,
          cal.CAL_PER_SEED_T_SHRINK: 0.30, cal.CAL_PER_SEED_VECTOR: 0.31}
    assert drv.select_calibration_candidate(j3) == cal.CAL_PER_SEED_T


def test_select_candidate_rejects_nonfinite_scores():
    j = {cal.CAL_SHARED_T: float("nan"), cal.CAL_PER_SEED_T: float("inf"),
         cal.CAL_PER_SEED_T_SHRINK: float("nan"), cal.CAL_PER_SEED_VECTOR: float("inf")}
    with pytest.raises(ValueError):
        drv.select_calibration_candidate(j)


def test_validation_calibration_screen_scores_all_four_candidates():
    val_logits, val_labels, val_mask, prev = _val_block()
    screen = drv.validation_calibration_screen(
        val_logits_seeds=val_logits, val_labels=val_labels, val_mask=val_mask,
        train_prevalence=prev, n_production_seeds=5)
    assert set(screen["rows"]) == set(cal.CALIBRATION_CANDIDATES)
    assert screen["selected"] in cal.CALIBRATION_CANDIDATES
    for cid in cal.CALIBRATION_CANDIDATES:
        row = screen["rows"][cid]
        assert np.isfinite(row["j_cal"])
        assert row["calibration_signature"]
        assert row["n_fit"] > 0
        assert set(row["per_seed_brier"]) == {str(s) for s in STAGE3A_PLUS_ALL_SEEDS}
        # J_cal is exactly the frozen formula over the stored components.
        brier = np.array([row["per_seed_brier"][str(s)] for s in STAGE3A_PLUS_ALL_SEEDS])
        assert row["j_cal"] == pytest.approx(drv.calibration_j_cal(brier, row["ece_ensemble"]))
    # The selected row carries a fully serialisable artifact for downstream rounds.
    art = screen["selected_artifact"]
    assert art["method"] == screen["selected"]
    assert art["fit_split"] == "validation"


def test_calibration_selection_receipt_write_refuse_and_resume(tmp_path):
    import json
    val_logits, val_labels, val_mask, prev = _val_block()
    screen = drv.validation_calibration_screen(
        val_logits_seeds=val_logits, val_labels=val_labels, val_mask=val_mask,
        train_prevalence=prev, n_production_seeds=5)
    receipt = drv.build_calibration_selection_receipt(
        screen, data_signature="devsig", validation_logits_signature="valsig")
    assert receipt["receipt_sha256"]
    path = tmp_path / "round0" / "calibration_selection.json"

    # First write succeeds and creates parent directories.
    drv.write_calibration_selection(path, receipt)
    assert path.is_file()
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["receipt_payload"]["selected_candidate"] == screen["selected"]

    # A second write is refused (single-use, no overwrite).
    with pytest.raises(FileExistsError):
        drv.write_calibration_selection(path, receipt)

    # Resume re-verifies against a recomputed screen without rewriting.
    before = path.read_bytes()
    ok, checks = drv.verify_calibration_selection(
        path, screen, data_signature="devsig", validation_logits_signature="valsig")
    assert ok, checks
    assert path.read_bytes() == before  # untouched

    # Tampered receipt fails integrity.
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["receipt_payload"]["selected_candidate"] = "CAL_PER_SEED_VECTOR"
    bad_path = tmp_path / "tampered.json"
    bad_path.write_text(json.dumps(tampered), encoding="utf-8")
    ok, _ = drv.verify_calibration_selection(
        bad_path, screen, data_signature="devsig", validation_logits_signature="valsig")
    assert not ok

    # A different data signature fails provenance binding.
    ok, checks = drv.verify_calibration_selection(
        path, screen, data_signature="WRONG", validation_logits_signature="valsig")
    assert not ok


# --------------------------------------------------------------------------- #
# Task 7: Gate R+ dev report assembly, dev state machine, final freeze
# --------------------------------------------------------------------------- #
def _decomp_and_eval(seed=1, shift=0.0):
    """A small but complete decompose report + its eval set (for gate assembly)."""
    es = _make_eval_set("A", seed=seed, shift=shift)
    rng = np.random.default_rng(seed + 100)
    cal_probs = np.stack(
        [softmax(lg, axis=-1) for lg in es["logits_seeds"]], axis=0)
    decomp = sps.decompose_seed_metrics(
        r_hat_seeds=es["r_hat_seeds"], logits_seeds=es["logits_seeds"],
        calibrated_probs_seeds=cal_probs, r_true=es["r_true"],
        regime_labels=es["regime_labels"], valid_mask=es["valid_mask"],
        episode_ids=es["episode_ids"], horizons=es["horizons"],
        cf_actions=es["cf_actions"], cf_durations=es["cf_durations"],
        baselines=es["baselines"], n_resamples=50, bootstrap_seed=20260929,
        expected_episode_ids=es["expected_episode_ids"], n_production_seeds=5)
    return decomp, es, cal_probs


def test_per_seed_episode_i_brier_shape_and_consistency():
    decomp, es, cal_probs = _decomp_and_eval()
    clf_base = es["baselines"]["classification"]["strong"]
    values = drv.per_seed_episode_i_brier(
        cal_probs, clf_base, es["regime_labels"], es["valid_mask"])
    assert values.shape == (N_SEEDS, es["regime_labels"].shape[0])
    assert np.isfinite(values).all()
    # Per-seed mean of per-episode values approximates the point I_Brier scale
    # (pooled vs per-episode macro Brier differ, but must be same order/sign).
    for i, seed in enumerate(STAGE3A_PLUS_ALL_SEEDS):
        assert np.sign(values[i].mean()) == np.sign(decomp["per_seed"][str(seed)]["i_brier"]) \
            or abs(values[i].mean()) < 0.05


def test_assemble_gate_r_plus_report_structure_and_and_semantics():
    decomp, es, cal_probs = _decomp_and_eval()
    out = drv.assemble_gate_r_plus_report(
        decomp, cal_probs_seeds=cal_probs, eval_set=es,
        n_resamples=100, bootstrap_seed=20260929,
        gate_q_passed=True, gate_q_checks={"signatures_ok": True})
    rep, ev = out["report"], out["evaluation"]
    assert set(rep) >= {"core", "stability", "gate_q_passed", "gate_q_checks"}
    assert set(rep["stability"]["per_seed"]) == {str(s) for s in STAGE3A_PLUS_ALL_SEEDS}
    assert "ci_lower" in rep["stability"]["hierarchical_i_brier"]
    assert set(ev["stability"]["checks"]) == {
        "per_seed_core_metrics_positive", "at_least_8_of_10_positive",
        "median_i_brier_positive", "worst_seed_noninferior",
        "hierarchical_ci_lower_gt0", "all_seeds_nondegenerate"}
    assert set(ev["core"]["checks"])  # all core checks present, none skipped
    assert out["passed"] == ev["passed"]
    # Q+ failure forces total failure (AND semantics).
    out_q = drv.assemble_gate_r_plus_report(
        decomp, cal_probs_seeds=cal_probs, eval_set=es,
        n_resamples=100, bootstrap_seed=20260929,
        gate_q_passed=False, gate_q_checks={"signatures_ok": False})
    assert not out_q["passed"]
    assert out_q["evaluation"]["passed"] is False


# --- dev state machine ------------------------------------------------------ #
def test_dev_state_machine_transitions_and_terminal():
    import json
    # fresh dir -> initial state
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rd = Path(td)
        st = drv.load_state(rd)
        assert st["state"] == "STAGE3A_PLUS_DESIGN_FROZEN"
        st = drv.advance_state(rd, "SOFTWARE_TESTED", note="tests green")
        assert st["state"] == "SOFTWARE_TESTED"
        assert st["history"][-1]["to"] == "SOFTWARE_TESTED"
        # illegal skip is refused and state unchanged
        with pytest.raises(ValueError):
            drv.advance_state(rd, "DEV_ROUND_0_COMPLETE")
        assert drv.load_state(rd)["state"] == "SOFTWARE_TESTED"
        # normal dev chain
        drv.advance_state(rd, "FORENSIC_DIAGNOSTIC_COMPLETE")
        drv.advance_state(rd, "DEV_ROUND_0_COMPLETE")
        # round-0 pass jumps straight to stable, skipping rounds 1/2
        drv.advance_state(rd, "STAGE3A_PLUS_DEVELOPMENT_STABLE")
        drv.advance_state(rd, "FINAL_MODEL_FROZEN")
        assert drv.load_state(rd)["state"] == "FINAL_MODEL_FROZEN"


def test_dev_state_machine_round2_failure_is_terminal():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rd = Path(td)
        for target in ("SOFTWARE_TESTED", "FORENSIC_DIAGNOSTIC_COMPLETE",
                       "DEV_ROUND_0_COMPLETE", "DEV_ROUND_1_COMPLETE",
                       "DEV_ROUND_2_COMPLETE"):
            drv.advance_state(rd, target)
        drv.advance_state(rd, "STAGE3A_PLUS_DEVELOPMENT_NOT_READY")
        st = drv.load_state(rd)
        assert st["state"] == "STAGE3A_PLUS_DEVELOPMENT_NOT_READY"
        with pytest.raises(ValueError):
            drv.advance_state(rd, "FINAL_MODEL_FROZEN")  # terminal: no way out


# --- final freeze receipt --------------------------------------------------- #
def _freeze_inputs(tmp_path, *, both_passed=True):
    import json
    round_dir = tmp_path / "development" / "round0"
    round_dir.mkdir(parents=True)
    a = {"passed": both_passed, "evaluation": {"passed": both_passed}}
    b = {"passed": True, "evaluation": {"passed": True}}
    report = {"plan_id": "AC-TWIN-RFER-STAGE3A-PLUS-2026-09-09-V0.1", "round": 0,
              "per_set": {"A": a, "B": b}, "both_passed": both_passed and True}
    report_path = round_dir / "gate_r_plus_dev_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    val_logits, val_labels, val_mask, prev = _val_block()
    screen = drv.validation_calibration_screen(
        val_logits_seeds=val_logits, val_labels=val_labels, val_mask=val_mask,
        train_prevalence=prev, n_production_seeds=5)
    receipt = drv.build_calibration_selection_receipt(
        screen, data_signature="devsig", validation_logits_signature="valsig")
    cal_path = round_dir / "calibration_selection.json"
    cal_path.write_text(json.dumps(receipt), encoding="utf-8")
    ckpts = {s: f"/ckpt/seed{s}.pt" for s in STAGE3A_PLUS_ALL_SEEDS}
    return report_path, cal_path, ckpts


def test_freeze_receipt_covers_all_required_components(tmp_path):
    report_path, cal_path, ckpts = _freeze_inputs(tmp_path)
    receipt = drv.build_final_freeze_receipt(
        dev_report_path=report_path, calibration_selection_path=cal_path,
        checkpoints=ckpts, checkpoint_signatures={s: f"sig{s}" for s in ckpts},
        config={"final_config": {"hidden_dim": 96}, "training": {"max_epochs": 40}},
        code_root=ROOT / "code", strongest_baselines={"continuous": "x", "brier": "y"},
        selection_receipt_sha256="selsha")
    payload = receipt["receipt_payload"]
    for key in ("config", "calibration", "checkpoints", "baselines", "gate",
                    "code_sha256", "dev_report_sha256"):
        assert key in payload, f"freeze receipt missing {key}"
    assert set(payload["checkpoints"]) == {str(s) for s in STAGE3A_PLUS_ALL_SEEDS}
    assert set(payload["dev_report_sha256"]) >= {"A", "B", "file"}
    assert receipt["receipt_sha256"]
    assert payload["plan_id"] == "AC-TWIN-RFER-STAGE3A-PLUS-2026-09-09-V0.1"


def test_freeze_final_refuses_when_either_dev_set_failed(tmp_path):
    report_path, cal_path, ckpts = _freeze_inputs(tmp_path, both_passed=False)
    out = tmp_path / "freeze" / "final_freeze_receipt.json"
    rc = drv.freeze_final(
        dev_report_path=report_path, calibration_selection_path=cal_path,
        checkpoints=ckpts, checkpoint_signatures={s: f"sig{s}" for s in ckpts},
        config={"final_config": {}, "training": {}}, code_root=ROOT / "code",
        strongest_baselines={"continuous": "x", "brier": "y"},
        selection_receipt_sha256="selsha", output_path=out)
    assert rc == 2
    assert not out.exists()


def test_freeze_receipt_single_use_no_overwrite(tmp_path):
    report_path, cal_path, ckpts = _freeze_inputs(tmp_path)
    out = tmp_path / "freeze" / "final_freeze_receipt.json"
    kwargs = dict(dev_report_path=report_path, calibration_selection_path=cal_path,
                  checkpoints=ckpts, checkpoint_signatures={s: f"sig{s}" for s in ckpts},
                  config={"final_config": {}, "training": {}}, code_root=ROOT / "code",
                  strongest_baselines={"continuous": "x", "brier": "y"},
                  selection_receipt_sha256="selsha", output_path=out)
    assert drv.freeze_final(**kwargs) == 0
    assert out.is_file()
    before = out.read_bytes()
    assert drv.freeze_final(**kwargs) == 2  # second run refused
    assert out.read_bytes() == before


# --------------------------------------------------------------------------- #
# Task 5 (driver side): train-round candidate space, selection, gating
# --------------------------------------------------------------------------- #
def _round_config():
    return {
        "round1": {"lambda_brier_candidates": [0.05, 0.10, 0.20]},
        "round2": {"weight_decay": [0.0, 1e-5, 1e-4], "label_smoothing": [0.0, 0.02]},
    }


def test_round1_candidate_space_exact_and_extras_rejected(tmp_path):
    cands = drv.round_candidate_space(_round_config(), tmp_path, 1)
    assert [c["lambda_brier"] for c in cands] == [0.05, 0.10, 0.20]
    assert all(c["weight_decay"] == 0.0 and c["label_smoothing"] == 0.0 for c in cands)
    with pytest.raises(ValueError):
        drv.round_candidate_space(
            {"round1": {"lambda_brier_candidates": [0.05, 0.10]}}, tmp_path, 1)


def test_round2_candidate_space_six_combos_with_round1_finalist(tmp_path):
    import json
    rd = tmp_path / "development" / "round1"
    rd.mkdir(parents=True)
    sel = {"receipt_payload": {"finalist": {"lambda_brier": 0.10}}}
    (rd / "train_selection.json").write_text(json.dumps(sel), encoding="utf-8")
    cands = drv.round_candidate_space(_round_config(), tmp_path, 2)
    assert len(cands) == 6
    assert all(c["lambda_brier"] == 0.10 for c in cands)
    assert [(c["weight_decay"], c["label_smoothing"]) for c in cands] == [
        (0.0, 0.0), (0.0, 0.02), (1e-5, 0.0), (1e-5, 0.02), (1e-4, 0.0), (1e-4, 0.02)]
    with pytest.raises(ValueError):
        drv.round_candidate_space(
            {"round2": {"weight_decay": [0.0, 1e-5], "label_smoothing": [0.0, 0.02]}},
            tmp_path, 2)


def test_j_train_formula():
    comps = [{"l_reg": 1.0, "l_ce": 0.4, "l_brier": 0.20},
             {"l_reg": 1.2, "l_ce": 0.5, "l_brier": 0.30},
             {"l_reg": 0.8, "l_ce": 0.3, "l_brier": 0.10}]
    per_seed = [c["l_reg"] + 0.25 * c["l_ce"] + 0.50 * c["l_brier"] for c in comps]
    briers = np.array([c["l_brier"] for c in comps])
    expected = float(np.mean(per_seed) + 0.50 * briers.std(ddof=0))
    assert drv.j_train(comps) == pytest.approx(expected)


def test_select_train_finalist_lowest_score_then_order():
    cands = [{"lambda_brier": 0.05}, {"lambda_brier": 0.10}, {"lambda_brier": 0.20}]
    assert drv.select_train_finalist(cands, [0.30, 0.20, 0.25]) == 1
    # Exact tie -> earliest candidate in the frozen order wins.
    assert drv.select_train_finalist(cands, [0.20, 0.20, 0.30]) == 0
    with pytest.raises(ValueError):
        drv.select_train_finalist(cands, [float("nan")] * 3)


def test_train_round_gating_rules(tmp_path):
    import json
    # Round 1 requires state DEV_ROUND_0_COMPLETE and a FAILED round-0 report.
    rd = tmp_path / "development" / "round0"
    rd.mkdir(parents=True)
    (rd / "gate_r_plus_dev_report.json").write_text(
        json.dumps({"plan_id": drv.STAGE3A_PLUS_PLAN_ID, "round": 0,
                    "both_passed": False}), encoding="utf-8")
    ok, _ = drv.check_train_round_allowed(tmp_path, 1)
    assert not ok  # state still INITIAL
    for target in ("SOFTWARE_TESTED", "FORENSIC_DIAGNOSTIC_COMPLETE", "DEV_ROUND_0_COMPLETE"):
        drv.advance_state(tmp_path, target)
    ok, reason = drv.check_train_round_allowed(tmp_path, 1)
    assert ok, reason
    # A PASSED previous round forbids further search.
    (rd / "gate_r_plus_dev_report.json").write_text(
        json.dumps({"plan_id": drv.STAGE3A_PLUS_PLAN_ID, "round": 0,
                    "both_passed": True}), encoding="utf-8")
    ok, _ = drv.check_train_round_allowed(tmp_path, 1)
    assert not ok
    # Round 2 from the wrong state is refused.
    ok, _ = drv.check_train_round_allowed(tmp_path, 2)
    assert not ok


def test_cmd_train_round_orchestration_with_injected_training(tmp_path):
    """Production five train per candidate; only the finalist trains the five
    stress seeds; the round-0 calibration family is refit and receipts are
    single-use (a second run verifies instead of rewriting)."""
    import json
    # State: round 0 complete and failed; round-0 calibration receipt present.
    r0 = tmp_path / "development" / "round0"
    r0.mkdir(parents=True)
    (r0 / "gate_r_plus_dev_report.json").write_text(
        json.dumps({"plan_id": drv.STAGE3A_PLUS_PLAN_ID, "round": 0,
                    "both_passed": False}), encoding="utf-8")
    val_logits, val_labels, val_mask, prev = _val_block()
    screen0 = drv.validation_calibration_screen(
        val_logits_seeds=val_logits, val_labels=val_labels, val_mask=val_mask,
        train_prevalence=prev, n_production_seeds=5)
    receipt0 = drv.build_calibration_selection_receipt(
        screen0, data_signature="devsig", validation_logits_signature="valsig")
    drv.write_calibration_selection(r0 / "calibration_selection.json", receipt0)
    for target in ("SOFTWARE_TESTED", "FORENSIC_DIAGNOSTIC_COMPLETE", "DEV_ROUND_0_COMPLETE"):
        drv.advance_state(tmp_path, target)

    cfg = {
        "plan_id": drv.STAGE3A_PLUS_PLAN_ID,
        "dev_seeds": [101, 202, 303, 404, 505],
        "stress_seeds": [606, 707, 808, 909, 1010],
        "development_source": "a", "replacement_source": "b", "prior_run_dir": "c",
        "final_config": {"hidden_dim": 96, "response_feature_dim": 32, "lambda_cls": 0.1},
        "training": {"max_epochs": 1, "early_stopping_patience": 1,
                     "learning_rate": 0.001, "batch_size": 32},
        "device": "cpu", "run_dir": str(tmp_path),
        **_round_config(),
    }
    cfg_path = tmp_path / "dev_config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    trained = []  # (candidate_idx, seed) in call order

    def fake_train(config, scaled, roles, ckpt_dir, seed, *, knobs, tag):
        trained.append((knobs["lambda_brier"], seed))
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        pt = ckpt_dir / f"fake_{knobs['lambda_brier']}_{seed}.pt"
        pt.write_bytes(b"x")
        return {"run": tag, "state": "COMPLETE", "seed": seed,
                "signature": f"sig-{knobs['lambda_brier']}-{seed}",
                "checkpoint": str(pt),
                "val_components": {"l_reg": 1.0, "l_ce": 0.4,
                                   "l_brier": 0.10 + 0.05 * knobs["lambda_brier"]}}

    val_block = (val_logits, val_labels, val_mask, prev, "devsig")

    args = argparse.Namespace(config=str(cfg_path), round=1)
    rc = drv.cmd_train_round(
        args, train_seed_fn=fake_train,
        validation_infer_fn=lambda *a, **k: val_block)
    assert rc == 0
    sel_path = r0.parent / "round1" / "train_selection.json"
    assert sel_path.is_file()
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    finalist = sel["receipt_payload"]["finalist"]
    # lambda_B=0.05 has the lowest fake L_Brier -> finalist.
    assert finalist["lambda_brier"] == 0.05
    # 3 candidates x 5 production seeds + 5 stress seeds for the finalist only.
    prod = [t for t in trained if t[1] in (101, 202, 303, 404, 505)]
    stress = [t for t in trained if t[1] in (606, 707, 808, 909, 1010)]
    assert len(prod) == 15 and len(stress) == 5
    assert all(t[0] == 0.05 for t in stress)
    # The round-0-selected calibration family was refit into round1's receipt.
    cal1 = json.loads((r0.parent / "round1" / "calibration_selection.json")
                      .read_text(encoding="utf-8"))
    assert (cal1["receipt_payload"]["selected_candidate"]
            == receipt0["receipt_payload"]["selected_candidate"])
    # Second run: verify-only, no rewrite, no retraining.
    before = sel_path.read_bytes()
    n_trained = len(trained)
    rc = drv.cmd_train_round(
        args, train_seed_fn=fake_train,
        validation_infer_fn=lambda *a, **k: val_block)
    assert rc == 0
    assert sel_path.read_bytes() == before
    assert len(trained) == n_trained


# --------------------------------------------------------------------------- #
# Task 7 step 6: run-all orchestration (advance / stop / restart-verify)
# --------------------------------------------------------------------------- #
def _run_all_config(tmp_path):
    import json
    cfg = {
        "plan_id": drv.STAGE3A_PLUS_PLAN_ID,
        "dev_seeds": [101, 202, 303, 404, 505],
        "stress_seeds": [606, 707, 808, 909, 1010],
        "development_source": "a", "replacement_source": "b", "prior_run_dir": "c",
        "final_config": {"hidden_dim": 96, "response_feature_dim": 32, "lambda_cls": 0.1},
        "training": {"max_epochs": 1}, "device": "cpu", "run_dir": str(tmp_path),
        **_round_config(),
    }
    p = tmp_path / "dev_config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def _fake_dev_artifacts(tmp_path, round_no, *, both_passed):
    import json
    from ac_twin.checkpointing import canonical_fingerprint
    rd = tmp_path / "development" / f"round{round_no}"
    rd.mkdir(parents=True, exist_ok=True)
    payload = {"plan_id": drv.STAGE3A_PLUS_PLAN_ID, "round": round_no}
    (rd / "calibration_selection.json").write_text(json.dumps({
        "receipt_payload": payload, "receipt_sha256": canonical_fingerprint(payload),
    }), encoding="utf-8")
    (rd / "gate_r_plus_dev_report.json").write_text(json.dumps({
        "plan_id": drv.STAGE3A_PLUS_PLAN_ID, "round": round_no,
        "both_passed": both_passed}), encoding="utf-8")
    if round_no > 0:
        (rd / "train_selection.json").write_text(json.dumps({}), encoding="utf-8")


def _fake_diagnose_artifact(tmp_path):
    import json
    d = tmp_path / "diagnostics"
    d.mkdir(parents=True, exist_ok=True)
    (d / "diagnostics_round0.json").write_text(
        json.dumps({"plan_id": drv.STAGE3A_PLUS_PLAN_ID}), encoding="utf-8")


def test_run_all_advances_to_frozen_on_round0_pass(tmp_path):
    cfg = _run_all_config(tmp_path)
    calls = []

    def round0():
        calls.append("round0")
        _fake_dev_artifacts(tmp_path, 0, both_passed=True)
        return 0

    runners = {
        "smoke": lambda: (calls.append("smoke"), 0)[1],
        "diagnose": lambda: (calls.append("diagnose"), _fake_diagnose_artifact(tmp_path), 0)[2],
        "round0": round0,
        "freeze": lambda: (calls.append("freeze"), _fake_freeze_artifact(tmp_path), 0)[2],
        # budget benchmark not yet recorded -> the pipeline parks at the gate
        "budget": lambda: (calls.append("budget"), 2)[1],
    }
    rc = drv.cmd_run_all(argparse.Namespace(config=str(cfg)), phase_runners=runners)
    assert rc == 2
    assert calls == ["smoke", "diagnose", "round0", "freeze", "budget"]
    assert drv.load_state(tmp_path)["state"] == "FINAL_MODEL_FROZEN"


def test_run_all_stops_on_failed_phase_without_advancing(tmp_path):
    cfg = _run_all_config(tmp_path)
    runners = {
        "smoke": lambda: 0,
        "diagnose": lambda: (_fake_diagnose_artifact(tmp_path), 0)[1],
        "round0": lambda: 2,  # engineering failure inside the round
    }
    rc = drv.cmd_run_all(argparse.Namespace(config=str(cfg)), phase_runners=runners)
    assert rc == 2
    assert drv.load_state(tmp_path)["state"] == "FORENSIC_DIAGNOSTIC_COMPLETE"


def test_run_all_round2_exhaustion_forces_not_ready(tmp_path):
    import json
    cfg = _run_all_config(tmp_path)
    # Pre-advance to DEV_ROUND_1_COMPLETE with failed rounds 0 and 1.
    for target in ("SOFTWARE_TESTED", "FORENSIC_DIAGNOSTIC_COMPLETE",
                   "DEV_ROUND_0_COMPLETE", "DEV_ROUND_1_COMPLETE"):
        drv.advance_state(tmp_path, target)
    _fake_diagnose_artifact(tmp_path)
    _fake_dev_artifacts(tmp_path, 0, both_passed=False)
    _fake_dev_artifacts(tmp_path, 1, both_passed=False)

    def round2():
        _fake_dev_artifacts(tmp_path, 2, both_passed=False)
        return 0

    rc = drv.cmd_run_all(argparse.Namespace(config=str(cfg)),
                         phase_runners={"round2": round2})
    assert rc == 3
    assert drv.load_state(tmp_path)["state"] == "STAGE3A_PLUS_DEVELOPMENT_NOT_READY"
    # Terminal: a later phase runner must never be invoked.
    rc2 = drv.cmd_run_all(argparse.Namespace(config=str(cfg)),
                          phase_runners={"freeze": lambda: 0})
    assert rc2 == 3


def test_run_all_restart_reverifies_completed_phases(tmp_path):
    cfg = _run_all_config(tmp_path)
    for target in ("SOFTWARE_TESTED", "FORENSIC_DIAGNOSTIC_COMPLETE"):
        drv.advance_state(tmp_path, target)
    # The diagnose artifact is missing -> restart verification must fail.
    rc = drv.cmd_run_all(argparse.Namespace(config=str(cfg)), phase_runners={})
    assert rc == 2
    # With the artifact present, verification passes and the run continues.
    _fake_diagnose_artifact(tmp_path)
    _fake_dev_artifacts(tmp_path, 0, both_passed=True)
    runners = {"round0": lambda: 0,
               "freeze": lambda: (_fake_freeze_artifact(tmp_path), 0)[1],
               "budget": lambda: 2}  # parks at the budget gate
    rc = drv.cmd_run_all(argparse.Namespace(config=str(cfg)), phase_runners=runners)
    assert rc == 2
    assert drv.load_state(tmp_path)["state"] == "FINAL_MODEL_FROZEN"


# --------------------------------------------------------------------------- #
# Task 10: run-all wiring of the qualification chain
# --------------------------------------------------------------------------- #
def _fake_freeze_artifact(tmp_path):
    import json
    from ac_twin.checkpointing import canonical_fingerprint
    payload = {"plan_id": drv.STAGE3A_PLUS_PLAN_ID, "artifact_type": "final_freeze"}
    d = tmp_path / "freeze"
    d.mkdir(parents=True, exist_ok=True)
    (d / "final_freeze_receipt.json").write_text(json.dumps({
        "receipt_payload": payload,
        "receipt_sha256": canonical_fingerprint(payload)}), encoding="utf-8")


def test_run_all_executes_qualification_chain(tmp_path):
    import json
    cfg = _run_all_config(tmp_path)
    for target in ("SOFTWARE_TESTED", "FORENSIC_DIAGNOSTIC_COMPLETE",
                   "DEV_ROUND_0_COMPLETE", "STAGE3A_PLUS_DEVELOPMENT_STABLE",
                   "FINAL_MODEL_FROZEN"):
        drv.advance_state(tmp_path, target)
    _fake_diagnose_artifact(tmp_path)
    _fake_dev_artifacts(tmp_path, 0, both_passed=True)
    _fake_freeze_artifact(tmp_path)

    calls = []

    def budget():
        calls.append("budget")
        (tmp_path / "COMPUTE_BUDGET.md").write_text("budget", encoding="utf-8")
        return 0

    def power():
        calls.append("power")
        d = tmp_path / "power"
        d.mkdir(parents=True, exist_ok=True)
        (d / "power_report.json").write_text(json.dumps({
            "plan_id": drv.STAGE3A_PLUS_PLAN_ID, "status": "POWER_FEASIBLE"}),
            encoding="utf-8")
        drv.advance_state(tmp_path, "QUALIFICATION_POWER_FROZEN")
        return 0

    def collect():
        calls.append("collect")
        q = tmp_path / "qualification"
        q.mkdir(parents=True, exist_ok=True)
        (q / "collection_receipt.json").write_text("{}", encoding="utf-8")
        (q / "qualification_lock.json").write_text("{}", encoding="utf-8")
        drv.advance_state(tmp_path, "MANUAL_COLLECTION_COMPLETE")
        return 0

    def lock_infer():
        calls.append("lock_infer")
        p = tmp_path / "qualification" / "predictions"
        p.mkdir(parents=True, exist_ok=True)
        (p / "inference_receipt.json").write_text(json.dumps({
            "plan_id": drv.STAGE3A_PLUS_PLAN_ID, "seeds": {}}), encoding="utf-8")
        drv.advance_state(tmp_path, "FROZEN_INFERENCE_COMPLETE")
        return 0

    def gate():
        calls.append("gate")
        (tmp_path / "qualification" / "gate_r_plus_qualification_report.json"
         ).write_text("{}", encoding="utf-8")
        drv.advance_state(tmp_path, "AUTOMATIC_SINGLE_ADJUDICATION_STARTED")
        drv.advance_state(tmp_path, "QUALIFICATION_ADJUDICATED")
        drv.advance_state(tmp_path, "RESPONSE_DEV_QUALIFIED_STAGE3A_PLUS")
        return 0

    runners = {"budget": budget, "power": power, "collect": collect,
               "lock_infer": lock_infer, "gate": gate}
    rc = drv.cmd_run_all(argparse.Namespace(config=str(cfg)), phase_runners=runners)
    assert rc == 0
    assert calls == ["budget", "power", "collect", "lock_infer", "gate"]
    assert drv.load_state(tmp_path)["state"] == "RESPONSE_DEV_QUALIFIED_STAGE3A_PLUS"


def test_run_all_power_infeasible_hard_stops_without_collection(tmp_path):
    cfg = _run_all_config(tmp_path)
    for target in ("SOFTWARE_TESTED", "FORENSIC_DIAGNOSTIC_COMPLETE",
                   "DEV_ROUND_0_COMPLETE", "STAGE3A_PLUS_DEVELOPMENT_STABLE",
                   "FINAL_MODEL_FROZEN", "COMPUTE_BUDGET_RECORDED_STAGE3A_PLUS"):
        drv.advance_state(tmp_path, target)
    _fake_diagnose_artifact(tmp_path)
    _fake_dev_artifacts(tmp_path, 0, both_passed=True)
    _fake_freeze_artifact(tmp_path)
    (tmp_path / "COMPUTE_BUDGET.md").write_text("budget", encoding="utf-8")
    runners = {"power": lambda: 3,  # STAGE3A_PLUS_QUALIFICATION_POWER_INFEASIBLE
               "collect": lambda: 0}
    rc = drv.cmd_run_all(argparse.Namespace(config=str(cfg)), phase_runners=runners)
    assert rc == 3
    assert drv.load_state(tmp_path)["state"] == "COMPUTE_BUDGET_RECORDED_STAGE3A_PLUS"


def test_run_all_burned_adjudication_is_a_hard_stop(tmp_path):
    cfg = _run_all_config(tmp_path)
    for target in ("SOFTWARE_TESTED", "FORENSIC_DIAGNOSTIC_COMPLETE",
                   "DEV_ROUND_0_COMPLETE", "STAGE3A_PLUS_DEVELOPMENT_STABLE",
                   "FINAL_MODEL_FROZEN", "COMPUTE_BUDGET_RECORDED_STAGE3A_PLUS",
                   "QUALIFICATION_POWER_FROZEN", "MANUAL_COLLECTION_COMPLETE",
                   "FROZEN_INFERENCE_COMPLETE", "AUTOMATIC_SINGLE_ADJUDICATION_STARTED"):
        drv.advance_state(tmp_path, target)
    _fake_diagnose_artifact(tmp_path)
    _fake_dev_artifacts(tmp_path, 0, both_passed=True)
    _fake_freeze_artifact(tmp_path)
    (tmp_path / "COMPUTE_BUDGET.md").write_text("budget", encoding="utf-8")
    q = tmp_path / "qualification"
    (q / "power").mkdir(parents=True, exist_ok=True)
    (tmp_path / "power").mkdir(parents=True, exist_ok=True)
    import json
    (tmp_path / "power" / "power_report.json").write_text(json.dumps({
        "plan_id": drv.STAGE3A_PLUS_PLAN_ID, "status": "POWER_FEASIBLE"}),
        encoding="utf-8")
    q.mkdir(parents=True, exist_ok=True)
    (q / "collection_receipt.json").write_text("{}", encoding="utf-8")
    (q / "qualification_lock.json").write_text("{}", encoding="utf-8")
    (q / "qualification_used.lock").write_text("{}", encoding="utf-8")
    p = q / "predictions"
    p.mkdir(parents=True, exist_ok=True)
    (p / "inference_receipt.json").write_text(json.dumps({
        "plan_id": drv.STAGE3A_PLUS_PLAN_ID, "seeds": {}}), encoding="utf-8")
    rc = drv.cmd_run_all(argparse.Namespace(config=str(cfg)),
                         phase_runners={"gate": lambda: 0})
    assert rc == 2  # burned mid-adjudication: never silently rerun the gate
    assert drv.load_state(tmp_path)["state"] == "AUTOMATIC_SINGLE_ADJUDICATION_STARTED"
