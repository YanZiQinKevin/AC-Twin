"""Blocking tests for the Stage 4D-D2 pilot adjudication's pure decision math.

Plan sections 6.2/6.3: the seven-criterion gate and mechanical adjudication. These
pin the regret/oracle/reactive computation and the adjudication routing, using
synthetic groups (no MetaDrive, no converted data, no one-shot resources).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import stage4d_metadrive_adapter as A
import stage4d_pilot_adjudicate as ADJ
from ac_twin import stage4c_objective as obj
from ac_twin import stage4d_external_features as F


NON_IDLE_PLANS = [a for a in range(F.N_ACTIONS) if a != obj.IDLE_PLAN_INDEX]


def _groups(n=20, seed=0, idle_suboptimal_frac=1.0, reactive=True):
    """Synthetic group dict satisfying the structural contract with a controllable oracle.

    Non-ego are placed far laterally (d=100) so interaction severity ~ 0 and the
    oracle is decided by efficiency (ego progress) alone. For each group the oracle
    plan is chosen deterministically: idle-suboptimal groups cycle over the non-IDLE
    plans (giving oracle diversity); idle-optimal groups use IDLE. When reactive, one
    CF branch moves non-ego slot 1 well beyond tolerance.
    """
    G = n
    hist = np.zeros((G, F.HISTORY_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES))
    fact = np.zeros((G, F.FUTURE_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES))
    cf = np.zeros((G, F.N_ACTIONS, F.FUTURE_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES))
    for arr in (hist, fact, cf):
        arr[..., 0] = 1.0  # presence
    n_idle_bad = int(round(idle_suboptimal_frac * G))

    # non-ego: far laterally, identical across branches unless reactive
    for slot in range(1, F.OBJECT_CAPACITY):
        fact[:, :, slot, ADJ.CH_S] = 5.0 + np.linspace(0, 10, F.FUTURE_LEN)[None, :]
        fact[:, :, slot, ADJ.CH_D] = 100.0
    for a in range(F.N_ACTIONS):
        for slot in range(1, F.OBJECT_CAPACITY):
            cf[:, a, :, slot, ADJ.CH_S] = fact[:, :, slot, ADJ.CH_S]
            cf[:, a, :, slot, ADJ.CH_D] = fact[:, :, slot, ADJ.CH_D]
    if reactive:
        cf[:, 0, -1, 1, ADJ.CH_S] += 5.0

    # factual ego progress
    fact[:, :, 0, ADJ.CH_S] = np.linspace(0, 20, F.FUTURE_LEN)[None, :]
    # per-group oracle plan via ego progress (final s); oracle plan gets +30
    for g in range(G):
        p = obj.IDLE_PLAN_INDEX if g >= n_idle_bad else NON_IDLE_PLANS[g % len(NON_IDLE_PLANS)]
        for a in range(F.N_ACTIONS):
            extra = 30.0 if a == p else float(a)
            cf[g, a, :, 0, ADJ.CH_S] = np.linspace(0, 10 + extra, F.FUTURE_LEN)
    return {
        "object_history": hist,
        "object_factual_future": fact,
        "object_counterfactual_futures": cf,
        "inference_mask": np.ones((G, F.FUTURE_LEN, F.OBJECT_CAPACITY), dtype=bool),
        "t_star": np.full(G, 12, dtype=np.int64),
        "cf_actions": np.asarray([a for a, _ in A.ACTION_PLANS]),
        "cf_durations": np.asarray([d for _, d in A.ACTION_PLANS]),
    }


def _meta(n, prefix="s"):
    return [{"scenario_id": f"{prefix}{i}"} for i in range(n)]


def test_compute_group_quantities_shapes_and_idle_regret():
    g = _groups(n=10, idle_suboptimal_frac=1.0)
    q = ADJ.compute_group_quantities(g)
    assert q["J_true"].shape == (10, F.N_ACTIONS)
    assert q["a_star"].shape == (10,)
    assert q["always_idle_regret"].shape == (10,)
    # every group was made idle-suboptimal -> regret exceeds threshold
    assert (q["always_idle_regret"] > ADJ.REGRET_MEANINGFUL).all()
    assert q["reactive"].all()


def test_idle_regret_zero_when_idle_is_optimal():
    g = _groups(n=6, idle_suboptimal_frac=0.0)
    q = ADJ.compute_group_quantities(g)
    # IDLE is the oracle in every group -> zero regret everywhere (clamped)
    assert (q["always_idle_regret"] == 0.0).all()


def test_criterion4_reactive_threshold():
    g = _groups(n=10, reactive=True)
    q = ADJ.compute_group_quantities(g)
    assert q["reactive"].mean() == 1.0
    g2 = _groups(n=10, reactive=False)
    q2 = ADJ.compute_group_quantities(g2)
    assert q2["reactive"].mean() == 0.0


def _write_role_dir(tmp_path, name, groups, meta, ledger_rows):
    d = tmp_path / name
    d.mkdir()
    from ac_twin.checkpointing import atomic_save_npz
    atomic_save_npz(d / "groups.npz", **groups)
    (d / "groups_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with open(d / "rejection_ledger.jsonl", "w", encoding="utf-8") as f:
        for r in ledger_rows:
            f.write(json.dumps(r) + "\n")
    return d


def test_adjudication_all_pass(tmp_path):
    n = ADJ_quota = 20
    groups = _groups(n=n, idle_suboptimal_frac=1.0, reactive=True)
    meta = _meta(n)
    ledger = [{"scenario_id": "x", "outcome": "rejected", "reason": "no_left_lane_at_t_star"}]
    a = _write_role_dir(tmp_path, "pilot_a", groups, meta, ledger)
    b = _write_role_dir(tmp_path, "pilot_b", groups, meta, ledger)  # identical replay
    report = ADJ.adjudicate(a, b, quota=ADJ_quota)
    assert report["verdict"] == "ALL_PASS"
    assert report["terminal_status"] == "METADRIVE_PLATFORM_QUALIFIED"
    assert all(r["pass"] for r in report["criteria"].values())


def test_adjudication_technical_fail_routing(tmp_path):
    n = 20
    groups = _groups(n=n, idle_suboptimal_frac=1.0, reactive=True)
    meta = _meta(n)
    # duplicate scenario id -> criterion 2 (technical) fails
    meta[1]["scenario_id"] = meta[0]["scenario_id"]
    a = _write_role_dir(tmp_path, "pilot_a", groups, meta, [])
    b = _write_role_dir(tmp_path, "pilot_b", groups, meta, [])  # identical replay
    report = ADJ.adjudicate(a, b, quota=n)
    assert report["verdict"] == "TECHNICAL_FAIL"
    assert report["terminal_status"] == "METADRIVE_TECHNICAL_CONTRACT_FAILED / SUMO_FALLBACK_ELIGIBLE"
    assert report["criteria"]["2"]["pass"] is False


def test_adjudication_task_fail_routing(tmp_path):
    n = 20
    # not reactive and idle is optimal -> criteria 4 and 5 fail (task support)
    groups = _groups(n=n, idle_suboptimal_frac=0.0, reactive=False)
    meta = _meta(n)
    a = _write_role_dir(tmp_path, "pilot_a", groups, meta, [])
    b = _write_role_dir(tmp_path, "pilot_b", groups, meta, [])  # identical replay
    report = ADJ.adjudicate(a, b, quota=n)
    assert report["criteria"]["4"]["pass"] is False
    assert report["criteria"]["5"]["pass"] is False
    assert report["criteria"]["6"]["pass"] is True
    assert report["verdict"] == "TASK_FAIL"
    assert report["terminal_status"] == "STAGE4D_EXTERNAL_TASK_INSUFFICIENT"


def test_adjudication_outcome_rejection_reason_out_of_set(tmp_path):
    n = 20
    groups = _groups(n=n, idle_suboptimal_frac=1.0, reactive=True)
    meta = _meta(n)
    ledger = [{"scenario_id": "x", "outcome": "rejected", "reason": "regret_too_low"}]
    a = _write_role_dir(tmp_path, "pilot_a", groups, meta, ledger)
    report = ADJ.adjudicate(a, None, quota=n)
    assert report["criteria"]["7"]["pass"] is False
