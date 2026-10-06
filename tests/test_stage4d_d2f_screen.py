"""Blocking-invariant tests for the D2F screen chain (plan sections 3.4-3.5).

The distinct invariants: (1) the gate literals are D2 section 8.2/8.3 VERBATIM with only the
completeness n moved to the D2F ``selected_n`` and the D2F terminal-state priority (task fail
-> D2F_EXTERNAL_DEV_TASK_NOT_READY; risk fail -> D2F_EXTERNAL_DEV_RISK_NOT_READY; integrity ->
INCONCLUSIVE_D2F_EXTERNAL_INVALID; both pass -> D2F_DEV_READY_FOR_CONFIRM_POWER); (2) the
score freeze is truth-free (it runs with the sealed truth absent) and the one-shot ordering is
fail-closed (no lock burn before the freeze, no truth before the lock, no second adjudication);
(3) the selected ego checkpoints and the Direct checkpoints pair one-to-one by content hash.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage4d_d2f_protocol as PROT  # noqa: E402
from ac_twin import stage4d_d2f_statistics as D2FSTAT  # noqa: E402
from ac_twin import stage4d_external_statistics as D2STAT  # noqa: E402
from ac_twin import stage4d_external_training as T  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402

G = 48
BASELINES = ("MARGIN_ONLY", "SEED_VOTE_ENTROPY", "SEED_MARGIN_STD")


def _quantities(*, high_regret: float = 0.30, i_aurc: float = 0.20,
                qa_ok: bool = True, baseline_ok: bool = True) -> dict:
    rng = np.random.default_rng(0)
    regret = rng.uniform(0.005, 0.06, size=G)
    j_true = np.abs(rng.normal(size=(G, 10))) + 0.05
    j_true[np.arange(G), rng.integers(0, 4, size=G)] -= 0.04
    qa = {key: qa_ok for key in D2STAT.QA_KEYS}
    return {
        "group_seed": np.arange(G, dtype=np.int64) + 1,
        "regret": regret,
        "y_high": (regret > 0.01).astype(np.int64),
        "random_regret": regret + 0.02,
        "idle_regret": regret + 0.01,
        "ensemble_risk": regret + 0.005,
        "risk_by_seed": regret[None] + np.full((len(T.MODEL_SEEDS), G), 0.005),
        "J_hat": j_true + 0.001,
        "J_true": j_true,
        "a_star": np.argmin(j_true, axis=-1),
        "tstar_layer": (np.arange(G) % 2).astype(np.int64),
        "interaction_layer": ((np.arange(G) // 2) % 2).astype(np.int64),
        "t_star": np.where(np.arange(G) % 2 == 0, 10, 14).astype(np.int64),
        "interaction_min_distance": rng.uniform(1.0, 30.0, size=G),
        "reactive": (np.arange(G) % 10) < 4,
        "baseline_risks": {name: rng.permutation(regret) + 0.02
                           for name in D2STAT.ALL_BASELINE_NAMES},
        "strongest_baseline": "SEED_MARGIN_STD",
        "n_groups": G,
        "interaction_threshold": 11.53877460000361,
        "qa": qa,
        "projection_nondegenerate": True,
        "eval_mask_nondegenerate": True,
        "_high_regret_override": high_regret,
        "_i_aurc_override": i_aurc,
        "_baseline_ok": baseline_ok,
    }


def _bootstrap(quantities: dict, *, n_resamples: int = 64) -> D2STAT.BootstrapResult:
    point = np.full(D2STAT.N_QUANTITIES, 0.0)
    point[D2STAT.Q_I_AURC] = quantities["_i_aurc_override"]
    point[D2STAT.Q_I_REGRET80] = 0.12
    point[D2STAT.Q_PR] = 0.15
    point[D2STAT.Q_SPEARMAN_BRANCH] = 0.50
    point[D2STAT.Q_BASE_VS_RANDOM] = 0.30
    point[D2STAT.Q_AURC_BASELINE] = 0.10 if quantities["_baseline_ok"] else 0.0
    point[D2STAT.Q_R80_BASELINE] = 0.08
    point[D2STAT.Q_BASE_MEAN_REGRET] = 0.05
    point[D2STAT.Q_IDLE_MEAN_REGRET] = 0.10
    point[D2STAT.Q_RANDOM_MEAN_REGRET] = 0.09
    point[D2STAT.Q_HIGH_REGRET_FRAC] = quantities["_high_regret_override"]
    for i in range(D2STAT.N_MODEL_SEEDS):
        point[D2STAT.Q_SEED_I_AURC + i] = 0.10
        point[D2STAT.Q_SEED_I_R80 + i] = 0.06
    for bi in range(3):
        slot = D2STAT.Q_BASELINE_CURVE + 3 * bi
        point[slot:slot + 3] = (0.11, 0.09, 0.05)
    point[D2STAT.Q_AURC_MODEL] = 0.04
    point[D2STAT.Q_R80_MODEL] = 0.03
    ci_lower = point - 0.01
    ci_upper = point + 0.01
    draws = np.tile(point, (n_resamples, 1))
    return D2STAT.BootstrapResult(point=point, ci_lower=ci_lower, ci_upper=ci_upper,
                                  draws=draws, n_resamples=n_resamples,
                                  seed=PROT.D2F_SCREEN_BOOTSTRAP_ENTROPY, confidence=0.80)


# --- 1. gate literals + terminal-state priority ----------------------------------------------------------


def test_both_gates_pass_at_selected_n() -> None:
    q = _quantities()
    decision = D2FSTAT.evaluate_d2f_screen_gates(q, _bootstrap(q), screen_groups=G)
    assert decision.status == D2FSTAT.STATUS_DEV_READY == "D2F_DEV_READY_FOR_CONFIRM_POWER"
    assert decision.passed and decision.task["passed"] and decision.risk["passed"]
    assert all(decision.task["conditions"].values())
    assert all(decision.risk["conditions"].values())


def test_completeness_uses_selected_n_not_d2_4000() -> None:
    q = _quantities()
    decision = D2FSTAT.evaluate_d2f_screen_gates(q, _bootstrap(q), screen_groups=G + 1)
    assert not decision.task["conditions"]["c1_completeness_and_qa"]
    assert decision.status == D2FSTAT.STATUS_TASK_NOT_READY


def test_task_fail_priority_over_risk() -> None:
    q = _quantities(high_regret=0.10)                            # c2 fails; risk would pass
    decision = D2FSTAT.evaluate_d2f_screen_gates(q, _bootstrap(q), screen_groups=G)
    assert not decision.task["passed"] and decision.risk["passed"]
    assert decision.status == D2FSTAT.STATUS_TASK_NOT_READY == "D2F_EXTERNAL_DEV_TASK_NOT_READY"


def test_risk_fail_after_task_pass() -> None:
    q = _quantities(i_aurc=0.01)                                 # r1 fails; task passes
    decision = D2FSTAT.evaluate_d2f_screen_gates(q, _bootstrap(q), screen_groups=G)
    assert decision.task["passed"] and not decision.risk["passed"]
    assert decision.status == D2FSTAT.STATUS_RISK_NOT_READY == "D2F_EXTERNAL_DEV_RISK_NOT_READY"


def test_integrity_failure_is_invalid_even_when_gates_pass() -> None:
    q = _quantities(qa_ok=False)
    decision = D2FSTAT.evaluate_d2f_screen_gates(q, _bootstrap(q), screen_groups=G)
    assert decision.status == D2FSTAT.STATUS_INVALID == "INCONCLUSIVE_D2F_EXTERNAL_INVALID"
    q2 = _quantities(baseline_ok=False)                          # degenerate denominator
    decision2 = D2FSTAT.evaluate_d2f_screen_gates(q2, _bootstrap(q2), screen_groups=G)
    assert decision2.status == D2FSTAT.STATUS_INVALID


def test_gate_condition_literals_match_d2_except_c1_n() -> None:
    """The c2-c8/r1-r6 condition booleans are D2 section 8.2/8.3 verbatim (gate parity)."""
    q = _quantities()
    boot = _bootstrap(q)
    d2 = D2STAT.evaluate_screen_gates(q, boot)
    d2f = D2FSTAT.evaluate_d2f_screen_gates(q, boot, screen_groups=G)
    d2_task = dict(d2.task["conditions"])
    d2f_task = dict(d2f.task["conditions"])
    assert d2_task.pop("c1_completeness_and_qa") is False        # D2 requires 4000 groups
    assert d2f_task.pop("c1_completeness_and_qa") is True        # D2F requires selected_n
    assert d2_task == d2f_task, "every other task condition is bitwise the D2 literal"
    assert d2.risk["conditions"] == d2f.risk["conditions"]
    # D2F never reuses the D2 terminal states.
    assert d2f.status != d2.status or d2f.status == D2FSTAT.STATUS_DEV_READY


# --- 2. the one-shot chain on a synthetic workspace --------------------------------------------------------


def _screen_workspace(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    """Synthetic D2F + D2E chain past the Phase S freeze (collection replaced by fixtures)."""
    d2f = tmp_path / "d2f"
    d2e = tmp_path / "d2e"
    monkeypatch.setattr(D2FSTAT, "D2F_DIR", d2f)
    monkeypatch.setattr(D2FSTAT, "D2E_DIR", d2e)
    monkeypatch.setattr(D2FSTAT, "D2_DIR", tmp_path / "d2")

    selected = "EGO_TRAJECTORY_PLUS_RANK"
    ego_dir = d2e / "development" / "ego_models"
    ego_dir.mkdir(parents=True)
    ego_rel: dict[str, str] = {}
    binding_entries = []
    for seed in ("1101", "1202", "1303", "1404", "1505"):
        path = ego_dir / f"ego_{selected}_seed{seed}.pt"
        path.write_bytes(f"ego-{seed}".encode())
        ego_rel[seed] = f"development/ego_models/{path.name}"
        binding_entries.append({"path": str(path), "sha256": sha256_file(path),
                                "label": "ego"})
    (d2e / "development" / "final_selection.json").write_text(json.dumps({
        "selected_candidate": selected, "ego_checkpoints": ego_rel,
        "risk_run_dir": f"riskfit_{selected}"}), encoding="utf-8")
    (d2e / "development" / "validation_report.json").write_text(json.dumps({
        "status": "D2E_INTERFACE_REPAIR_QUALIFIED",
        "candidate_gate": {selected: {"strongest_baseline": "SEED_MARGIN_STD"}}}),
        encoding="utf-8")
    models_dir = d2e / f"riskfit_{selected}" / "development" / "models"
    models_dir.mkdir(parents=True)
    ckpt_paths, ckpt_sha = {}, {}
    for seed in ("9101", "9202", "9303", "9404", "9505"):
        path = models_dir / f"direct_seed{seed}.pt"
        path.write_bytes(f"direct-{seed}".encode())
        ckpt_paths[seed] = str(path)
        ckpt_sha[seed] = sha256_file(path)
    (models_dir / "fit_receipt.json").write_text(json.dumps({
        "interaction_threshold": 11.53877460000361,
        "checkpoint_paths": ckpt_paths, "checkpoint_sha256": ckpt_sha}), encoding="utf-8")

    (d2f).mkdir(parents=True, exist_ok=True)
    (d2f / PROT.SOURCE_BINDING_NAME).write_text(json.dumps(
        {"plan_id": PROT.PLAN_ID, "expanded": {"checkpoints": binding_entries}}),
        encoding="utf-8")
    (d2f / PROT.SCREEN_MANIFEST_NAME).write_text(json.dumps(
        {"plan_id": PROT.PLAN_ID, "selected_n": G}), encoding="utf-8")
    (d2f / PROT.SCREEN_LOCK_NAME).write_text(json.dumps({"plan_id": PROT.PLAN_ID}),
                                            encoding="utf-8")

    rng = np.random.default_rng(3)
    split = d2f / "development" / "screen_split"
    split.mkdir(parents=True)
    factual = rng.normal(size=(G, 20, 5, 8)).astype(np.float32)
    factual[..., 0] = 1.0                                       # presence channel
    cf = (factual[:, None] + rng.normal(scale=0.5, size=(G, 10, 20, 5, 8))).astype(np.float32)
    evaluation_mask = np.ones((G, 20, 5), dtype=bool)
    np.savez(split / "sealed_truth.npz",
             group_seed=np.arange(G, dtype=np.int64) + 1,
             object_counterfactual_futures=cf,
             object_factual_future=factual,
             evaluation_mask=evaluation_mask,
             history_last_xy=rng.normal(size=(G, 2)),
             scenario_ids_json=np.asarray(json.dumps([f"s-{i}" for i in range(G)])))
    np.savez(split / "inference_inputs.npz", group_seed=np.arange(G, dtype=np.int64) + 1)
    return D2FSTAT.d2f_screen_artifact_paths()


def _scores() -> dict:
    rng = np.random.default_rng(5)
    by_seed = rng.uniform(0.0, 0.1, size=(D2STAT.N_MODEL_SEEDS, G))
    return {
        "group_seed": np.arange(G, dtype=np.int64) + 1,
        "t_star": np.where(np.arange(G) % 2 == 0, 10, 14).astype(np.int64),
        "interaction_min_distance": rng.uniform(1.0, 30.0, size=G),
        "j_hat": np.abs(rng.normal(size=(G, 10))),
        "ensemble_risk": by_seed.mean(axis=0),
        "risk_by_seed": by_seed,
        "mu_only_ensemble": by_seed.mean(axis=0),
        "class_only_ensemble": by_seed.mean(axis=0),
        "valid_mask": np.ones(G, dtype=bool),
        "baselines": {name: rng.uniform(0.0, 0.1, size=G) for name in D2STAT.ALL_BASELINE_NAMES},
    }


def test_score_freeze_is_truth_free_and_create_once(tmp_path, monkeypatch) -> None:
    paths = _screen_workspace(tmp_path, monkeypatch)
    assert not (paths["split_dir"] / "moved_truth.npz").exists()
    truth_away = paths["sealed_truth"].rename(paths["split_dir"] / "moved_truth.npz")
    with pytest.raises(ValueError, match="elementwise identical"):
        D2FSTAT.freeze_d2f_screen_scores(
            None, scores=_scores(), determinism={"bitwise_identical": False})
    npz_path = D2FSTAT.freeze_d2f_screen_scores(
        None, scores=_scores(), determinism={"bitwise_identical": True, "n_groups": 32})
    assert npz_path.is_file(), "the score freeze never needs the sealed truth"
    again = D2FSTAT.freeze_d2f_screen_scores(
        None, scores=_scores(), determinism={"bitwise_identical": True})
    assert again == npz_path, "a rerun verifies and reloads (create-once)"
    truth_away.rename(paths["sealed_truth"])


def test_lock_and_adjudication_ordering_is_fail_closed(tmp_path, monkeypatch) -> None:
    paths = _screen_workspace(tmp_path, monkeypatch)
    with pytest.raises(Exception):
        D2FSTAT.burn_d2f_screen_lock(None)                      # no frozen scores yet
    with pytest.raises(ValueError, match="lock is not burned"):
        D2FSTAT.assert_d2f_screen_adjudication_open(None)
    D2FSTAT.freeze_d2f_screen_scores(
        None, scores=_scores(), determinism={"bitwise_identical": True})
    D2FSTAT.burn_d2f_screen_lock(None)
    assert paths["lock"].is_file()
    D2FSTAT.burn_d2f_screen_lock(None)                          # re-verify, never overwrite

    monkeypatch.setattr(D2STAT, "paired_group_bootstrap",
                        lambda quantities, *, seed=0: _bootstrap(
                            _quantities(), n_resamples=8))
    decision = D2FSTAT.adjudicate_d2f_screen(None)
    assert decision.n_groups == G
    assert paths["report"].is_file()
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["plan_id"] == PROT.PLAN_ID
    assert report["bootstrap"]["seed"] == PROT.D2F_SCREEN_BOOTSTRAP_ENTROPY
    with pytest.raises(FileExistsError, match="second D2F screen adjudication"):
        D2FSTAT.adjudicate_d2f_screen(None)


def test_checkpoint_pairing_detects_drift(tmp_path, monkeypatch) -> None:
    paths = _screen_workspace(tmp_path, monkeypatch)
    selection = D2FSTAT._load_frozen_d2e_selection(paths)
    assert selection["strongest_standard_baseline"] == "SEED_MARGIN_STD"
    assert D2FSTAT._verify_checkpoint_pairing(selection) is True
    ego_file = D2FSTAT.D2E_DIR / "development" / "ego_models" \
        / f"ego_EGO_TRAJECTORY_PLUS_RANK_seed1101.pt"
    ego_file.write_bytes(b"tampered")
    assert D2FSTAT._verify_checkpoint_pairing(selection) is False
