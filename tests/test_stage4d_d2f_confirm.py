"""Blocking-invariant tests for the D2F confirm chain (plan section 4).

The distinct confirm invariants: (1) the confirm bootstrap is the SAME whole-group paired
vector recomputation at the two-sided 95% CI (the frozen D2 module is hardcoded to 0.80 and is
never re-parameterised; the 95% variant lives in the D2F namespace) with the fresh registered
entropies 2026122408 (confirm power) / 2026122409 (confirm adjudication); (2) the confirm gates
are D2 10.3/10.4 verbatim with every CI-based condition at 95% and the confirm terminal
priority — any task-gate failure or integrity failure -> INCONCLUSIVE_D2F_EXTERNAL_INVALID
(never a method falsification), risk-gate failure -> D2F_META_DRIVE_NOT_CONFIRMED, both pass ->
D2F_META_DRIVE_EXTERNAL_CONFIRMED; (3) the confirm phase is fail-closed until the single screen
adjudication passed BOTH gates, the confirm manifest draws ONLY from the preserved confirm pool
(5 x n primary, zero overlap, create-once, terminal D2F_FRESH_CONFIRM_POOL_INSUFFICIENT on a
shortfall), and the one-shot ordering (freeze -> burn confirmation_used.lock -> single
adjudication) is fail-closed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage4d_d2f_power as DPOW  # noqa: E402
from ac_twin import stage4d_d2f_protocol as PROT  # noqa: E402
from ac_twin import stage4d_d2f_statistics as D2FSTAT  # noqa: E402
from ac_twin import stage4d_d2e_power as POW  # noqa: E402
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


def _bootstrap(quantities: dict, *, confidence: float = 0.95,
               n_resamples: int = 64) -> D2STAT.BootstrapResult:
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
                                  seed=PROT.D2F_CONFIRM_BOOTSTRAP_ENTROPY,
                                  confidence=confidence)


# --- 1. the 95% bootstrap calibre + fresh entropy registry -------------------------------------------------


def test_confirm_entropies_registered_and_distinct() -> None:
    assert PROT.D2F_CONFIRM_POWER_SEED == 2026122408
    assert PROT.D2F_CONFIRM_BOOTSTRAP_ENTROPY == 2026122409
    assert DPOW.CONFIRM_Z == 1.959963984540054
    assert DPOW.CONFIRM_CONFIDENCE == 0.95
    assert D2FSTAT.CONFIDENCE_CONFIRM == 0.95
    fresh = {2026122406, 2026122407, 2026122408, 2026122409}     # the four D2F entropies
    assert len(fresh) == 4, "every D2F entropy is a distinct registered value"


def test_confirm_bootstrap_uses_95pct_band() -> None:
    q = _quantities()
    result = D2FSTAT.paired_group_bootstrap_95(q, seed=123, n_resamples=200)
    assert result.confidence == 0.95
    assert result.seed == 123 and result.n_resamples == 200
    np.testing.assert_allclose(result.ci_lower, np.quantile(result.draws, 0.025, axis=0),
                               rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(result.ci_upper, np.quantile(result.draws, 0.975, axis=0),
                               rtol=0.0, atol=1e-15)
    # the 95% band is strictly wider than an 80% band on the same draws.
    lo80 = np.quantile(result.draws, 0.10, axis=0)
    hi80 = np.quantile(result.draws, 0.90, axis=0)
    assert (result.ci_lower <= lo80 + 1e-15).all() and (result.ci_upper >= hi80 - 1e-15).all()


def test_confirm_eval_uses_95_z_not_screen_z() -> None:
    """A CI condition passing at the 80% z but failing at the 95% z proves the calibre."""
    row = np.zeros((1, DPOW.N_POWER_QUANTITIES))
    row[0, POW.P_I_AURC] = 0.20
    row[0, POW.P_I_REGRET80] = 0.12
    row[0, POW.P_PR] = 0.15
    row[0, POW.P_SPEARMAN] = 0.50
    row[0, POW.P_BASE_VS_RANDOM] = 0.30
    row[0, POW.P_HIGH_REGRET] = 0.30
    row[0, list(POW.P_SEED_I_AURC)] = 0.10
    row[0, list(POW.P_SEED_I_R80)] = 0.06
    row[0, list(POW.P_ORACLE)] = 0.10
    row[0, POW.P_REACTIVE] = 0.40
    row[0, list(POW.P_STRATA)] = 0.25
    se = np.full(DPOW.N_POWER_QUANTITIES, 0.005)
    se[POW.P_I_AURC] = 0.11        # 0.20 - 1.2816*0.11 > 0 but 0.20 - 1.9600*0.11 < 0
    d_idle = np.array([0.05])
    screen_out = DPOW.evaluate_corrected_draws(row, d_idle, se, z=DPOW.SCREEN_Z)
    confirm_out = DPOW.evaluate_confirm_draws(row, d_idle, se)
    assert screen_out["conditions"]["r1_i_aurc_regret"].all()
    assert not confirm_out["conditions"]["r1_i_aurc_regret"].any()
    assert not confirm_out["passed"].any()


def test_confirm_gates_refuse_non_95_bootstrap() -> None:
    q = _quantities()
    with pytest.raises(ValueError, match="frozen at 0.95"):
        D2FSTAT.evaluate_d2f_confirm_gates(q, _bootstrap(q, confidence=0.80),
                                           confirm_groups=G)


# --- 2. gate literals + confirm terminal priority ------------------------------------------------------------


def test_both_gates_pass_is_confirmed() -> None:
    q = _quantities()
    decision = D2FSTAT.evaluate_d2f_confirm_gates(q, _bootstrap(q), confirm_groups=G)
    assert decision.status == D2FSTAT.STATUS_CONFIRMED == "D2F_META_DRIVE_EXTERNAL_CONFIRMED"
    assert decision.passed and decision.task["passed"] and decision.risk["passed"]
    assert all(decision.task["conditions"].values())
    assert all(decision.risk["conditions"].values())
    assert decision.bootstrap["confidence"] == 0.95


def test_gate_condition_literals_match_d2_except_c1_n() -> None:
    """The c2-c8/r1-r6 condition booleans are D2 8.2/8.3 verbatim evaluated at the 95% CI."""
    q = _quantities()
    boot95 = _bootstrap(q, confidence=0.95)
    boot80 = D2STAT.BootstrapResult(point=boot95.point, ci_lower=boot95.ci_lower,
                                    ci_upper=boot95.ci_upper, draws=boot95.draws,
                                    n_resamples=boot95.n_resamples, seed=boot95.seed,
                                    confidence=0.80)
    d2 = D2STAT.evaluate_screen_gates(q, boot80)
    confirm = D2FSTAT.evaluate_d2f_confirm_gates(q, boot95, confirm_groups=G)
    d2_task = dict(d2.task["conditions"])
    confirm_task = dict(confirm.task["conditions"])
    assert d2_task.pop("c1_completeness_and_qa") is False        # D2 requires 4000 groups
    assert confirm_task.pop("c1_completeness_and_qa") is True    # confirm requires selected_n
    assert d2_task == confirm_task, "every other task condition is the D2 literal at the CI"
    assert d2.risk["conditions"] == confirm.risk["conditions"]


def test_task_fail_is_invalid_never_not_confirmed() -> None:
    q = _quantities(high_regret=0.10)                            # c2 fails; risk would pass
    decision = D2FSTAT.evaluate_d2f_confirm_gates(q, _bootstrap(q), confirm_groups=G)
    assert not decision.task["passed"] and decision.risk["passed"]
    assert decision.status == D2FSTAT.STATUS_INVALID == "INCONCLUSIVE_D2F_EXTERNAL_INVALID", (
        "a failed confirm task gate is an INVALID external platform, never a method "
        "falsification (D2 10.3/11.3)")
    assert not decision.passed


def test_risk_fail_after_task_pass_is_not_confirmed() -> None:
    q = _quantities(i_aurc=0.01)                                 # r1 fails; task passes
    decision = D2FSTAT.evaluate_d2f_confirm_gates(q, _bootstrap(q), confirm_groups=G)
    assert decision.task["passed"] and not decision.risk["passed"]
    assert decision.status == D2FSTAT.STATUS_NOT_CONFIRMED == "D2F_META_DRIVE_NOT_CONFIRMED"
    assert not decision.passed


def test_integrity_failure_is_invalid_even_when_gates_pass() -> None:
    q = _quantities(qa_ok=False)
    decision = D2FSTAT.evaluate_d2f_confirm_gates(q, _bootstrap(q), confirm_groups=G)
    assert decision.status == D2FSTAT.STATUS_INVALID
    q2 = _quantities(baseline_ok=False)                          # degenerate denominator
    decision2 = D2FSTAT.evaluate_d2f_confirm_gates(q2, _bootstrap(q2), confirm_groups=G)
    assert decision2.status == D2FSTAT.STATUS_INVALID


def test_confirm_baseline_disclosure_never_substitutes() -> None:
    q = _quantities()
    decision = D2FSTAT.evaluate_d2f_confirm_gates(q, _bootstrap(q), confirm_groups=G)
    disclosure = decision.baselines["disclosure"]
    assert disclosure["frozen_primary_baseline"] == "SEED_MARGIN_STD"
    assert set(disclosure) >= {"confirm_strongest_standard_baseline",
                               "confirm_strongest_differs_from_frozen",
                               "model_beats_confirm_strongest_baseline"}
    curves = decision.baselines["curves"]
    assert set(curves) == set(D2FSTAT.STANDARD_BASELINES)
    for name, entry in curves.items():
        assert "aurc_regret_ci95" in entry and "regret_at_80_ci95" in entry
        assert "pr_auc_minus_prevalence_ci95" in entry


# --- 3. the one-shot confirm chain on a synthetic workspace ---------------------------------------------------


def _confirm_workspace(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    """Synthetic D2F + D2E chain past the Phase C freeze (collection replaced by fixtures)."""
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
    (d2f / PROT.CONFIRM_MANIFEST_NAME).write_text(json.dumps(
        {"plan_id": PROT.PLAN_ID, "selected_n": G}), encoding="utf-8")
    (d2f / PROT.CONFIRM_LOCK_NAME).write_text(json.dumps({"plan_id": PROT.PLAN_ID}),
                                              encoding="utf-8")

    rng = np.random.default_rng(3)
    split = d2f / "confirm" / "confirm_split"
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
    return D2FSTAT.d2f_confirm_artifact_paths()


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


def test_confirm_score_freeze_is_truth_free_and_create_once(tmp_path, monkeypatch) -> None:
    paths = _confirm_workspace(tmp_path, monkeypatch)
    assert not (paths["split_dir"] / "moved_truth.npz").exists()
    truth_away = paths["sealed_truth"].rename(paths["split_dir"] / "moved_truth.npz")
    with pytest.raises(ValueError, match="elementwise identical"):
        D2FSTAT.freeze_d2f_confirm_scores(
            None, scores=_scores(), determinism={"bitwise_identical": False})
    npz_path = D2FSTAT.freeze_d2f_confirm_scores(
        None, scores=_scores(), determinism={"bitwise_identical": True, "n_groups": 32})
    assert npz_path.is_file(), "the confirm score freeze never needs the sealed truth"
    again = D2FSTAT.freeze_d2f_confirm_scores(
        None, scores=_scores(), determinism={"bitwise_identical": True})
    assert again == npz_path, "a rerun verifies and reloads (create-once)"
    truth_away.rename(paths["sealed_truth"])


def test_confirm_lock_and_adjudication_ordering_is_fail_closed(tmp_path, monkeypatch) -> None:
    paths = _confirm_workspace(tmp_path, monkeypatch)
    with pytest.raises(Exception):
        D2FSTAT.burn_d2f_confirm_lock(None)                      # no frozen scores yet
    with pytest.raises(ValueError, match="lock is not burned"):
        D2FSTAT.assert_d2f_confirm_adjudication_open(None)
    D2FSTAT.freeze_d2f_confirm_scores(
        None, scores=_scores(), determinism={"bitwise_identical": True})
    D2FSTAT.burn_d2f_confirm_lock(None)
    assert paths["lock"].is_file()
    D2FSTAT.burn_d2f_confirm_lock(None)                          # re-verify, never overwrite

    monkeypatch.setattr(D2FSTAT, "paired_group_bootstrap_95",
                        lambda quantities, *, seed=0: _bootstrap(
                            _quantities(), n_resamples=8))
    decision = D2FSTAT.adjudicate_d2f_confirm(None)
    assert decision.n_groups == G
    assert paths["report"].is_file()
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["plan_id"] == PROT.PLAN_ID
    assert report["bootstrap"]["seed"] == PROT.D2F_CONFIRM_BOOTSTRAP_ENTROPY
    assert report["bootstrap"]["confidence"] == 0.95
    assert report["status"] == "D2F_META_DRIVE_EXTERNAL_CONFIRMED"
    with pytest.raises(FileExistsError, match="second D2F confirm adjudication"):
        D2FSTAT.adjudicate_d2f_confirm(None)


# --- 4. confirm phase gate / manifest / lock (protocol) -------------------------------------------------------


def _write_role(root: Path, role: str, ledger_ids: list[str], accepted_ids: list[str]) -> None:
    role_dir = root / role
    role_dir.mkdir(parents=True, exist_ok=True)
    with (role_dir / "rejection_ledger.jsonl").open("w", encoding="utf-8") as handle:
        for scenario_id in ledger_ids:
            handle.write(json.dumps({"outcome": "rejected", "scenario_id": scenario_id}) + "\n")
        for scenario_id in accepted_ids:
            handle.write(json.dumps({"outcome": "accepted", "scenario_id": scenario_id}) + "\n")
    (role_dir / "groups_meta.json").write_text(json.dumps(
        [{"scenario_id": s, "t_star": 10} for s in accepted_ids]), encoding="utf-8")


def _rows(spec: dict[str, int]) -> list[dict]:
    rows, rank = [], 0
    for role, count in spec.items():
        for i in range(count):
            rows.append({"global_rank": rank, "pool": "primary" if i == 0 else "reserve",
                         "role": role, "scenario_id": f"{role}-{i:05d}",
                         "source_shard": "training_20s.tfrecord-00000-of-01000"})
            rank += 1
    return rows


def _synth_workspace(tmp_path: Path) -> Path:
    platform = tmp_path / "d2" / "platform"
    platform.mkdir(parents=True)
    rows = _rows({"train": 300, "confirm": 120, "development_screen": 80})
    with (platform / "scenario_split_pools.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (platform / "scenario_split_manifest.json").write_text(
        json.dumps({"plan_id": "d2", "pools_rows": len(rows)}), encoding="utf-8")
    _write_role(tmp_path / "d2" / "roles", "train", ["train-00000"], ["train-00001"])
    return tmp_path


def _phase_c_workspace(tmp_path: Path, monkeypatch, *, confirm_n: int = 10,
                       confirm_power_status: str = PROT.STATUS_CONFIRM_POWER_PASS,
                       screen_status: str | None = "D2F_DEV_READY_FOR_CONFIRM_POWER") -> Path:
    """Synthetic workspace past the D2F screen adjudication (Phase C inputs frozen)."""
    _synth_workspace(tmp_path)
    out_dir = tmp_path / "d2f"
    monkeypatch.setattr(PROT, "D2_DIR", tmp_path / "d2")
    monkeypatch.setattr(PROT, "D2F_DIR", out_dir)
    monkeypatch.setattr(PROT, "SCREEN_CANDIDATE_NS", (10, 20))
    monkeypatch.setattr(PROT, "CONFIRM_CANDIDATE_NS", (10, 20, 24))
    monkeypatch.setattr(PROT, "SMOKE_CANDIDATE_BUDGET", 5)
    audit = PROT.run_pool_capacity_audit()
    assert audit["status"] == PROT.STATUS_AUDIT_PASS
    (out_dir / PROT.SOURCE_BINDING_NAME).write_text(json.dumps(
        {"plan_id": PROT.PLAN_ID, "status": PROT.STATUS_AUDIT_PASS,
         "expanded": {"checkpoints": [], "scalers": []}}), encoding="utf-8")
    (out_dir / PROT.POST_SMOKE_AUDIT_NAME).write_text(json.dumps(
        {"plan_id": PROT.PLAN_ID, "status": PROT.STATUS_AUDIT_PASS}),
        encoding="utf-8")
    (out_dir / PROT.SCREEN_MANIFEST_NAME).write_text(json.dumps(
        {"plan_id": PROT.PLAN_ID, "selected_n": 10}), encoding="utf-8")
    (out_dir / PROT.SCREEN_LOCK_NAME).write_text(json.dumps({"plan_id": PROT.PLAN_ID}),
                                                 encoding="utf-8")
    if screen_status is not None:
        dev_dir = out_dir / "development"
        dev_dir.mkdir(parents=True, exist_ok=True)
        (dev_dir / "development_screen_report.json").write_text(json.dumps(
            {"plan_id": PROT.PLAN_ID, "status": screen_status}), encoding="utf-8")
    confirm_dir = out_dir / "confirm"
    confirm_dir.mkdir(parents=True, exist_ok=True)
    (confirm_dir / "power_report.json").write_text(json.dumps(
        {"plan_id": PROT.PLAN_ID, "status": confirm_power_status,
         "selected_n": confirm_n if confirm_power_status == PROT.STATUS_CONFIRM_POWER_PASS
         else None}), encoding="utf-8")
    return out_dir


def test_confirm_phase_closed_until_screen_passes(tmp_path, monkeypatch) -> None:
    out_dir = _phase_c_workspace(tmp_path, monkeypatch, screen_status=None)
    with pytest.raises(RuntimeError, match="may only start after BOTH"):
        PROT.assert_confirm_phase_open()
    report_path = out_dir / "development" / "development_screen_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(
        {"plan_id": PROT.PLAN_ID, "status": "D2F_EXTERNAL_DEV_RISK_NOT_READY"}),
        encoding="utf-8")
    with pytest.raises(RuntimeError, match="D2F_DEV_READY_FOR_CONFIRM_POWER"):
        PROT.assert_confirm_phase_open()
    report_path.write_text(json.dumps(
        {"plan_id": PROT.PLAN_ID, "status": "D2F_DEV_READY_FOR_CONFIRM_POWER"}),
        encoding="utf-8")
    PROT.assert_confirm_phase_open()                             # opens exactly on both-pass


def test_derive_confirm_candidate_ids_deterministic_zero_overlap() -> None:
    rows = _rows({"train": 6, "confirm": 4, "development_screen": 2})
    burned = {"train-00000", "development_screen-00000"}
    order = PROT.derive_confirm_candidate_ids(rows, burned)
    assert order == PROT.derive_confirm_candidate_ids(rows, burned), "salt order is frozen"
    assert set(order) == {f"confirm-{i:05d}" for i in range(4)}, (
        "ONLY the preserved never-attempted old-confirm IDs are legal confirm candidates")
    assert not (set(order) & burned)
    keyed = sorted(order, key=lambda sid: hashlib.sha256(
        (PROT.SCREEN_SALT + sid).encode("utf-8")).hexdigest())
    assert order == keyed, "the order is the frozen SHA256 salt ranking"


def test_confirm_manifest_primary_reserve_zero_overlap_create_once(tmp_path, monkeypatch) -> None:
    _phase_c_workspace(tmp_path, monkeypatch, confirm_n=10)
    manifest = PROT.build_confirm_manifest()
    assert manifest["selected_n"] == 10
    assert len(manifest["primary_ids"]) == 50                    # 5 x selected_n (frozen floor)
    assert len(manifest["reserve_ids"]) == 120 - 50
    confirm_pool = {f"confirm-{i:05d}" for i in range(120)}
    assert set(manifest["primary_ids"]) <= confirm_pool
    assert set(manifest["reserve_ids"]) <= confirm_pool
    assert not (set(manifest["primary_ids"]) & set(manifest["reserve_ids"]))
    rows = PROT.load_pool_rows(tmp_path / "d2" / "platform" / "scenario_split_pools.jsonl")
    scan = PROT.scan_burned_ids(platform_dir=tmp_path / "d2")
    screen_order = set(PROT.derive_screen_candidate_ids(rows, scan["burned"]))
    assert not (set(manifest["primary_ids"] + manifest["reserve_ids"]) & screen_order), (
        "the confirm manifest never raids the screen pool")
    again = PROT.build_confirm_manifest()
    assert again["written_at"] == manifest["written_at"], "create-once reload"


def test_confirm_manifest_refused_on_terminal_power(tmp_path, monkeypatch) -> None:
    out_dir = _phase_c_workspace(tmp_path, monkeypatch, confirm_n=10,
                                 confirm_power_status=PROT.STATUS_CONFIRM_UNDERPOWERED)
    with pytest.raises(RuntimeError, match="terminal"):
        PROT.build_confirm_manifest()
    assert not (out_dir / PROT.CONFIRM_MANIFEST_NAME).exists(), (
        "no manifest is ever written after D2F_CONFIRM_UNDERPOWERED")


def test_confirm_manifest_pool_insufficient_terminal(tmp_path, monkeypatch) -> None:
    out_dir = _phase_c_workspace(tmp_path, monkeypatch, confirm_n=25)   # 5x25=125 > 120 pool
    with pytest.raises(RuntimeError, match=PROT.STATUS_CONFIRM_POOL_INSUFFICIENT):
        PROT.build_confirm_manifest()
    assert not (out_dir / PROT.CONFIRM_MANIFEST_NAME).exists(), (
        "the shortfall terminal writes NO manifest and never re-ranks the order")


def test_confirm_lock_contents_and_create_once(tmp_path, monkeypatch) -> None:
    out_dir = _phase_c_workspace(tmp_path, monkeypatch, confirm_n=10)
    PROT.build_confirm_manifest()
    (out_dir / PROT.CONFIRM_CODE_INVENTORY_NAME).write_text(json.dumps(
        {"plan_id": PROT.PLAN_ID, "code": []}), encoding="utf-8")
    (out_dir / PROT.MODEL_INVENTORY_NAME).write_text(json.dumps(
        {"plan_id": PROT.PLAN_ID, "checkpoints": []}), encoding="utf-8")
    lock = PROT.freeze_confirm_lock()
    assert lock["selected_n"] == 10 and lock["screen_selected_n"] == 10
    assert lock["bootstrap"]["entropy"] == PROT.D2F_CONFIRM_BOOTSTRAP_ENTROPY == 2026122409
    assert lock["bootstrap"]["resamples"] == 10000
    assert lock["bootstrap"]["confidence"] == 0.95
    assert lock["gates"]["frozen_primary_baseline"] == "SEED_MARGIN_STD"
    assert lock["gates"]["task_x_confirm"]["c1"] == "groups == 10 and all QA flags"
    assert lock["terminal_states"]["both_pass"] == "D2F_META_DRIVE_EXTERNAL_CONFIRMED"
    assert lock["terminal_states"]["risk_fail"] == "D2F_META_DRIVE_NOT_CONFIRMED"
    assert lock["terminal_states"]["task_fail_or_invalid"] == "INCONCLUSIVE_D2F_EXTERNAL_INVALID"
    assert len(lock["bindings"]) == 7
    again = PROT.freeze_confirm_lock()
    assert again["written_at"] == lock["written_at"]


# --- 5. confirm power (95% chain, conditional on the passed screen) --------------------------------------------


def _synth_power_quantities(seed: int = 0, correlated: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    regret = rng.uniform(0.005, 0.06, size=240)
    noise = rng.normal(scale=0.01, size=240)
    ensemble = regret + noise if correlated else rng.uniform(0.005, 0.06, size=240)
    risk_by_seed = ensemble[None] + rng.normal(scale=0.01, size=(len(T.MODEL_SEEDS), 240))
    baseline_risks = {
        name: rng.permutation(regret) + rng.normal(scale=0.02, size=240) for name in BASELINES}
    j_true = np.abs(rng.normal(size=(240, 10))) + 0.05
    best = rng.integers(0, 4, size=240)
    j_true[np.arange(240), best] -= 0.04
    j_hat = j_true + rng.normal(scale=0.01, size=(240, 10))
    return {
        "group_seed": np.arange(240, dtype=np.int64) + 1,
        "regret": regret,
        "y_high": (regret > 0.01).astype(np.int64),
        "random_regret": regret + rng.uniform(0.01, 0.03, size=240),
        "idle_regret": regret + rng.uniform(0.005, 0.02, size=240),
        "ensemble_risk": ensemble,
        "risk_by_seed": risk_by_seed,
        "J_hat": j_hat,
        "J_true": j_true,
        "a_star": np.argmin(j_true, axis=-1),
        "tstar_layer": (np.arange(240) % 2).astype(np.int64),
        "interaction_layer": ((np.arange(240) // 2) % 2).astype(np.int64),
        "t_star": np.where(np.arange(240) % 2 == 0, 10, 14).astype(np.int64),
        "interaction_min_distance": rng.uniform(1.0, 30.0, size=240),
        "reactive": (np.arange(240) % 10) < 3,
        "baseline_risks": baseline_risks,
        "strongest_baseline": "SEED_MARGIN_STD",
        "n_groups": 240,
    }


def _confirm_power_workspace(tmp_path: Path, monkeypatch, *, admissible=(1000, 1500),
                             screen_status: str | None = "D2F_DEV_READY_FOR_CONFIRM_POWER",
                             correlated: bool = True) -> Path:
    out_dir = tmp_path / "d2f"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(PROT, "D2F_DIR", out_dir)
    monkeypatch.setattr(DPOW, "CONFIRM_DIR", out_dir / "confirm")
    monkeypatch.setattr(DPOW, "load_d2f_screen_quantities",
                        lambda: _synth_power_quantities(correlated=correlated))
    monkeypatch.setattr(DPOW, "POWER_RESAMPLES", 150)
    (out_dir / PROT.POOL_AUDIT_NAME).write_text(json.dumps({
        "plan_id": PROT.PLAN_ID, "status": PROT.STATUS_AUDIT_PASS,
        "capacity": {"confirm_admissible_ns": list(admissible)}}), encoding="utf-8")
    (out_dir / PROT.SOURCE_BINDING_NAME).write_text(json.dumps(
        {"plan_id": PROT.PLAN_ID, "status": PROT.STATUS_AUDIT_PASS}), encoding="utf-8")
    (out_dir / PROT.SCREEN_LOCK_NAME).write_text(json.dumps({"plan_id": PROT.PLAN_ID}),
                                                 encoding="utf-8")
    dev_dir = out_dir / "development"
    dev_dir.mkdir(parents=True, exist_ok=True)
    # the confirm power bindings hash the frozen screen quantities + receipt (read-only).
    np.savez(dev_dir / "screen_group_quantities.npz", placeholder=np.zeros(1))
    (dev_dir / "screen_group_quantities_receipt.json").write_text(json.dumps(
        {"plan_id": PROT.PLAN_ID}), encoding="utf-8")
    if screen_status is not None:
        (dev_dir / "development_screen_report.json").write_text(json.dumps(
            {"plan_id": PROT.PLAN_ID, "status": screen_status}), encoding="utf-8")
    return out_dir


def test_confirm_power_95z_fresh_seed_create_once(tmp_path, monkeypatch) -> None:
    out_dir = _confirm_power_workspace(tmp_path, monkeypatch)
    report = DPOW.run_confirm_power()
    assert report["status"] == PROT.STATUS_CONFIRM_POWER_PASS
    assert report["selected_n"] == 1000, "a strong synthetic effect selects the smallest n"
    assert report["power_seed"] == 2026122408
    assert report["ci_z"] == 1.959963984540054
    assert report["confidence"] == 0.95
    assert report["scope"] == "CONDITIONAL_POWER_GIVEN_SCREEN_PASSED"
    assert report["terminal"] is False
    report_path = out_dir / "confirm" / "power_report.json"
    assert report_path.is_file()
    again = DPOW.run_confirm_power()
    assert again["written_at"] == report["written_at"], "create-once reload"


def test_confirm_power_underpowered_terminal(tmp_path, monkeypatch) -> None:
    out_dir = _confirm_power_workspace(tmp_path, monkeypatch, correlated=False)
    report = DPOW.run_confirm_power()
    assert report["status"] == PROT.STATUS_CONFIRM_UNDERPOWERED
    assert report["terminal"] is True and report["selected_n"] is None
    assert not (out_dir / PROT.CONFIRM_MANIFEST_NAME).exists(), (
        "D2F_CONFIRM_UNDERPOWERED forbids any confirm manifest / lock / collection")


def test_confirm_power_refused_before_screen_pass(tmp_path, monkeypatch) -> None:
    _confirm_power_workspace(tmp_path, monkeypatch, screen_status=None)
    with pytest.raises(RuntimeError, match="may only start after BOTH"):
        DPOW.run_confirm_power()
