"""Blocking tests for the Stage 4D-D2 action-contract measurement (synthetic).

Plan section 13: the contract freeze reads ONLY ego kinematics + known override
intents (tracking correctness). These pin the measurement math and the create-once
freeze guard, with no MetaDrive and no converted data.
"""
from __future__ import annotations

import numpy as np
import pytest

import stage4d_action_contract as AC
import stage4d_metadrive_adapter as A
from ac_twin import stage4d_external_features as F


def _synthetic_groups(n_groups=4):
    """Build a (G,10,20,5,8) CF tensor where each override produces its intended
    ego displacement: LANE_LEFT +d, LANE_RIGHT -d, FASTER +v_s, SLOWER -v_s. The
    factual ego moves at a steady 5 m/s so every maneuver's feasible subset is
    non-empty (moving ego, room ahead)."""
    G = n_groups
    cf = np.zeros((G, F.N_ACTIONS, F.FUTURE_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES))
    cf[..., 0] = 1.0  # presence
    factual = np.zeros((G, F.FUTURE_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES))
    factual[..., 0] = 1.0
    factual[:, :, 0, AC.CH_S] = np.linspace(0, 20, F.FUTURE_LEN)  # ego advances 20 m
    factual[:, :, 0, AC.CH_VS] = np.linspace(5.0, 6.0, F.FUTURE_LEN)  # moving, mild rise
    cf_actions = np.asarray([a for a, _ in A.ACTION_PLANS])
    for a in range(F.N_ACTIONS):
        action = int(cf_actions[a])
        for t in range(F.FUTURE_LEN):
            frac = t / (F.FUTURE_LEN - 1)
            cf[:, a, t, 0, AC.CH_S] = np.linspace(0, 20, F.FUTURE_LEN)[t]
            cf[:, a, t, 0, AC.CH_VS] = 5.0
            if action == AC.LANE_LEFT:
                cf[:, a, t, 0, AC.CH_D] = +3.5 * frac      # move left (+d)
            elif action == AC.LANE_RIGHT:
                cf[:, a, t, 0, AC.CH_D] = -3.5 * frac      # move right (-d)
            elif action == AC.FASTER:
                cf[:, a, t, 0, AC.CH_VS] = 5.0 + 3.0 * frac  # speed up (peak +3)
            elif action == AC.SLOWER:
                cf[:, a, t, 0, AC.CH_VS] = 5.0 - 3.0 * frac  # slow down (peak -3)
    return {
        "object_counterfactual_futures": cf,
        "object_factual_future": factual,
        "cf_actions": cf_actions,
    }


def test_measure_detects_intended_tracking():
    stats = AC._measure(_synthetic_groups())
    assert stats["lane_change"]["certified"] and stats["lane_change"]["success_frac"] == 1.0
    assert stats["lane_change_left"]["success_frac"] == 1.0
    assert stats["lane_change_right"]["success_frac"] == 1.0
    assert stats["faster"]["success_frac"] == 1.0
    assert stats["slower"]["success_frac"] == 1.0
    assert stats["tracking_ok"] is True
    assert stats["factual_ego_progress_median_m"] > 10.0


def test_measure_detects_failed_lane_change():
    groups = _synthetic_groups()
    groups["object_counterfactual_futures"][:, :, :, 0, AC.CH_D] = 0.0  # no LC happens
    stats = AC._measure(groups)
    assert stats["lane_change_left"]["success_frac"] == 0.0
    assert stats["lane_change_right"]["success_frac"] == 0.0
    assert stats["lane_change"]["certified"] is False
    assert stats["tracking_ok"] is False


def test_measure_respects_feasibility_gating():
    """A stopped ego (v_star=0, no factual motion) makes SLOWER/LC infeasible; the
    measure must report them uncertified rather than counting them as failures."""
    groups = _synthetic_groups()
    # freeze the ego: no factual motion and zero speed -> LC and SLOWER infeasible
    groups["object_factual_future"][:, :, 0, AC.CH_S] = 0.0
    groups["object_factual_future"][:, :, 0, AC.CH_VS] = 0.0
    stats = AC._measure(groups)
    assert stats["slower"]["feasible_n"] == 0
    assert stats["slower"]["success_frac"] is None
    assert stats["lane_change"]["feasible_n"] == 0
    assert stats["tracking_ok"] is False


def test_freeze_is_create_once(tmp_path, monkeypatch):
    import stage4d_action_contract as mod
    contract_path = tmp_path / "action_contract.json"
    monkeypatch.setattr(mod, "CONTRACT_PATH", contract_path)
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    from ac_twin.checkpointing import atomic_save_npz
    atomic_save_npz(smoke / "groups.npz", **_synthetic_groups())
    receipt = mod.freeze_contract(smoke)
    assert receipt["tracking_ok"] and receipt["status"] == "ACTION_CONTRACT_FROZEN"
    with pytest.raises(FileExistsError):
        mod.freeze_contract(smoke)
