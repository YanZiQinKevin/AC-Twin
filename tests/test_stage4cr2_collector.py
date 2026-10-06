"""Stage 4C-R2 collector tests: complete-group science, role isolation, resume, guards."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import collect_stage4cr2_merge_multiaction as coll  # noqa: E402
from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402

CONFIG_PATH = ROOT / "experiments/stage4_reliability/stage4c_r2/stage4cr2_config.json"


def _config(tmp_path):
    """The frozen config with run/data roots redirected into a tmp purpose dir."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["run_dir"] = str(tmp_path / "run")
    config["data_root"] = str(
        tmp_path / "data" / "object_centric_stage" / s4cr2.PURPOSE_DEV)
    return config


def _fake_probe(config):
    capacity = int(config["object_capacity"])
    action_ids = np.arange(5, dtype=np.int64)
    action_probs = np.full(5, 0.2, dtype=np.float64)
    return action_ids, action_probs, capacity


def _fingerprints(seed, capacity):
    rng = np.random.default_rng(int(seed))
    return [{
        "entity_id": i, "class": "car", "route": (),
        "position_xy": [float(rng.normal()), float(rng.normal())],
        "heading": 0.0, "speed": 1.0,
    } for i in range(capacity)]


def make_fake_rollout(config, *, future_scale=1.0):
    """Deterministic fake branch rollout; shared prefix across branches of a group.

    ``future_scale`` multiplies the future-frame positions only (channels 2:4): it
    changes every downstream consequence/regret value WITHOUT changing presence,
    finiteness, shapes or termination — the outcome-independence probe.
    """
    capacity = int(config["object_capacity"])
    horizon = int(config["horizon"])
    calls = []

    def fake(env_id, env_cfg, seed, prefix_actions, future_actions, cap):
        calls.append((int(seed), tuple(int(a) for a in prefix_actions),
                      tuple(int(a) for a in future_actions)))
        prefix_actions = np.asarray(prefix_actions)
        future_actions = np.asarray(future_actions)
        t_star = len(prefix_actions)
        prng = np.random.default_rng((int(seed) * 100003 + int(prefix_actions.sum())) % (2**31))
        prefix = prng.normal(size=(t_star + 1, capacity, 8)).astype(np.float32)
        frng = np.random.default_rng(
            (int(seed) * 7919 + int(prefix_actions.sum()) * 131
             + int(future_actions.sum()) * 17 + len(future_actions)) % (2**31))
        future = prng.normal(size=(horizon, capacity, 8)).astype(np.float32)
        future[..., 2:4] *= float(future_scale)
        for frames in (prefix, future):
            frames[..., 0] = 0.0
            frames[:, :4, 0] = 1.0  # slots 0..3 present, slot 4 absent
        diag = {k: np.zeros((len(prefix), capacity), dtype=np.float32)
                for k in ("lane_index", "target_lane_index", "commanded_accel",
                          "commanded_steering", "crashed")}
        fdiag = {k: np.zeros((horizon, capacity), dtype=np.float32)
                 for k in ("lane_index", "target_lane_index", "commanded_accel",
                           "commanded_steering", "crashed")}
        return {
            "prefix_frames": prefix, "future_frames": future,
            "terminated": False, "truncated": False,
            "termination_reason": "", "first_contact_step": -1,
            "diag_prefix": diag, "diag_future": fdiag,
            "fingerprints": _fingerprints(seed, capacity),
            "static_metadata": [],
        }
    fake.calls = calls
    return fake


def _one_group(config, fake, seed=12345):
    t_star = 12
    prefix = np.full(t_star, 4, dtype=np.int64)
    factual_plan = np.full(int(config["horizon"]), 1, dtype=np.int64)
    return coll.collect_group(seed, t_star, prefix, factual_plan, config, fake), t_star, prefix, factual_plan


# --- ten-action completeness + shared pre-intervention state ---------------------------


def test_ten_action_group_completeness_and_shared_prefix(tmp_path):
    config = _config(tmp_path)
    fake = make_fake_rollout(config)
    group, t_star, prefix, factual_plan = _one_group(config, fake)
    # 11 calls: 1 factual + 10 counterfactual in the frozen ACTION_PLANS order.
    assert len(fake.calls) == 11
    assert fake.calls[0][2] == tuple(int(a) for a in factual_plan)
    expected_cf = [tuple(int(x) for x in s4cr2.materialize_action_plan(factual_plan, a, d))
                   for a, d in s4cr2.ACTION_PLANS]
    assert [c[2] for c in fake.calls[1:]] == expected_cf
    ref = group["factual"]["prefix_frames"]
    for b in group["cf"]:
        assert np.array_equal(b["prefix_frames"], ref)
    verdict = coll.validate_complete_group(group, config)
    assert verdict["accepted"], verdict
    assert verdict["cf_futures"].shape == (s4cr2.N_ACTIONS, 20, 5, 8)
    assert [int(a) for a in verdict["cf_actions"]] == [a for a, _ in s4cr2.ACTION_PLANS]


# --- whole-group rejection on engineering invariants only -------------------------------


def test_abnormal_termination_rejects_whole_group(tmp_path):
    config = _config(tmp_path)
    _, _, _, factual_plan = _one_group(config, make_fake_rollout(config))
    base_fake = make_fake_rollout(config)

    def terminating(env_id, env_cfg, seed, prefix_actions, future_actions, cap):
        out = base_fake(env_id, env_cfg, seed, prefix_actions, future_actions, cap)
        target = tuple(int(x) for x in s4cr2.materialize_action_plan(factual_plan, 3, 3))
        if tuple(int(x) for x in future_actions) == target:
            out["terminated"] = True
        return out

    group, _, _, _ = _one_group(config, terminating, seed=999)
    verdict = coll.validate_complete_group(group, config)
    assert not verdict["accepted"]
    assert verdict["reason"] == "prefix_terminated"
    assert verdict["failed_branch"] == "cf_7"  # (FASTER,3) is plan index 7


def test_prefix_mismatch_rejects_group(tmp_path):
    config = _config(tmp_path)
    _, _, _, factual_plan = _one_group(config, make_fake_rollout(config))
    base_fake = make_fake_rollout(config)

    def drifted(env_id, env_cfg, seed, prefix_actions, future_actions, cap):
        out = base_fake(env_id, env_cfg, seed, prefix_actions, future_actions, cap)
        target = tuple(int(x) for x in s4cr2.materialize_action_plan(factual_plan, 0, 1))
        if tuple(int(x) for x in future_actions) == target:
            out["prefix_frames"] = out["prefix_frames"] + 1e-3  # counterfactual prefix drift
        return out

    group, _, _, _ = _one_group(config, drifted, seed=555)
    verdict = coll.validate_complete_group(group, config)
    assert not verdict["accepted"]
    assert verdict["reason"] == "object_prefix_mismatch"


def test_present_set_mismatch_rejects_group(tmp_path):
    config = _config(tmp_path)
    _, _, _, factual_plan = _one_group(config, make_fake_rollout(config))
    base_fake = make_fake_rollout(config)

    def mismatch(env_id, env_cfg, seed, prefix_actions, future_actions, cap):
        out = base_fake(env_id, env_cfg, seed, prefix_actions, future_actions, cap)
        target = tuple(int(x) for x in s4cr2.materialize_action_plan(factual_plan, 0, 1))
        if tuple(int(x) for x in future_actions) == target:
            out["future_frames"][:, 3, 0] = 0.0  # drop a present entity mid-horizon
        return out

    group, _, _, _ = _one_group(config, mismatch, seed=556)
    verdict = coll.validate_complete_group(group, config)
    assert not verdict["accepted"]
    assert verdict["reason"] == "present_set_mismatch"


def test_observation_shape_or_capacity_failure_rejects_group(tmp_path):
    config = _config(tmp_path)
    base_fake = make_fake_rollout(config)

    def wrong_capacity(env_id, env_cfg, seed, prefix_actions, future_actions, cap):
        out = base_fake(env_id, env_cfg, seed, prefix_actions, future_actions, cap)
        out["future_frames"] = out["future_frames"][:, :4, :]  # capacity violation
        return out

    group, _, _, _ = _one_group(config, wrong_capacity, seed=557)
    verdict = coll.validate_complete_group(group, config)
    assert not verdict["accepted"]
    assert verdict["reason"] == "observation_shape"


def test_non_finite_observation_rejects_group(tmp_path):
    config = _config(tmp_path)
    base_fake = make_fake_rollout(config)

    def non_finite(env_id, env_cfg, seed, prefix_actions, future_actions, cap):
        out = base_fake(env_id, env_cfg, seed, prefix_actions, future_actions, cap)
        out["future_frames"][0, 1, 2] = np.nan
        return out

    group, _, _, _ = _one_group(config, non_finite, seed=558)
    verdict = coll.validate_complete_group(group, config)
    assert not verdict["accepted"]
    assert verdict["reason"] == "non_finite"


def test_wrong_length_branch_rejects_group(tmp_path):
    config = _config(tmp_path)
    base_fake = make_fake_rollout(config)

    def short(env_id, env_cfg, seed, prefix_actions, future_actions, cap):
        out = base_fake(env_id, env_cfg, seed, prefix_actions, future_actions, cap)
        if len(out["future_frames"]) == int(config["horizon"]):
            out["future_frames"] = out["future_frames"][:-1]  # missing branch steps
        return out

    group, _, _, _ = _one_group(config, short, seed=559)
    verdict = coll.validate_complete_group(group, config)
    assert not verdict["accepted"]
    assert verdict["reason"] == "factual_future_short"


# --- outcome-independent rejection (plan section 5.3) ------------------------------------


def test_rejection_is_outcome_independent(tmp_path):
    """Changing every downstream consequence value changes NO acceptance decision."""
    config = _config(tmp_path)
    decisions = {}
    for scale in (1.0, 1e-4, 1e4):
        fake = make_fake_rollout(config, future_scale=scale)
        verdicts = []
        for seed in (7001, 7002, 7003, 7004):
            group, _, _, _ = _one_group(config, fake, seed=seed)
            verdict = coll.validate_complete_group(group, config)
            verdicts.append((bool(verdict["accepted"]), str(verdict["reason"])))
        decisions[scale] = verdicts
    assert decisions[1.0] == decisions[1e-4] == decisions[1e4]
    # the validator interface takes only the raw group + config: no truth-side channel.
    assert list(inspect.signature(coll.validate_complete_group).parameters) == ["group", "config"]
    # the rejection vocabulary contains no outcome-based reason.
    forbidden = ("regret", "oracle", "collision", "response", "error", "cost", "margin")
    for reason in coll.REASONS:
        assert not any(word in reason for word in forbidden), reason


# --- role isolation, quotas and the protocol data guard -----------------------------------


def test_role_quota_is_frozen_per_role(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="quota"):
        coll.collect_role(config, "development_train", 4999,
                          branch_rollout=make_fake_rollout(config), probe_fn=_fake_probe)
    with pytest.raises(ValueError, match="quota"):
        coll.collect_role(config, "smoke", 11,
                          branch_rollout=make_fake_rollout(config), probe_fn=_fake_probe)
    with pytest.raises(ValueError, match="role"):
        coll.collect_role(config, "confirm", 12,
                          branch_rollout=make_fake_rollout(config), probe_fn=_fake_probe)


def test_role_isolation_fixed_schedule_segments(tmp_path):
    """A formal role may consume ONLY its own frozen schedule segment."""
    config = _config(tmp_path)
    checked = s4cr2.validate_config(config)
    schedule = s4cr2.build_seed_schedule(checked)
    v_lo, v_hi = s4cr2.VALIDATION_SEGMENT
    foreign = schedule[v_lo:v_lo + 8]  # validation seeds are illegal for the train role
    with pytest.raises(ValueError, match="segment"):
        coll.collect_role(config, "development_train", s4cr2.DEV_TRAIN_GROUPS,
                          schedule=foreign,
                          branch_rollout=make_fake_rollout(config), probe_fn=_fake_probe)
    # the derived per-role slices are disjoint and in frozen order.
    t_lo, t_hi = s4cr2.TRAIN_SEGMENT
    s_lo, s_hi = s4cr2.SCREEN_SEGMENT
    assert np.intersect1d(schedule[t_lo:t_hi], schedule[v_lo:v_hi]).size == 0
    assert np.intersect1d(schedule[t_lo:t_hi], schedule[s_lo:s_hi]).size == 0


def test_protocol_guard_refuses_foreign_data_root(tmp_path):
    config = _config(tmp_path)
    config["data_root"] = str(
        tmp_path / "data" / "object_centric_stage" / "stage4c_merge_multiaction_dev")
    with pytest.raises(ValueError):
        coll.collect_role(config, "smoke", s4cr2.SMOKE_GROUPS,
                          branch_rollout=make_fake_rollout(config), probe_fn=_fake_probe)


# --- collection, ledger vocabulary, duplicate seeds, resume --------------------------------


def test_smoke_collection_marks_engineering_smoke_only(tmp_path):
    config = _config(tmp_path)
    receipt = coll.collect_role(config, "smoke", s4cr2.SMOKE_GROUPS,
                                branch_rollout=make_fake_rollout(config),
                                probe_fn=_fake_probe)
    assert receipt.status == "ENGINEERING_SMOKE_ONLY"
    assert receipt.accepted_groups == s4cr2.SMOKE_GROUPS
    assert not receipt.partial
    data_output, ledger_path, receipt_path = coll.role_paths(config, "smoke")
    assert data_output.is_file() and ledger_path.is_file() and receipt_path.is_file()
    assert "smoke" in data_output.parent.name  # smoke never lands in the formal files
    with np.load(data_output, allow_pickle=True) as z:
        assert z["object_counterfactual_futures"].shape == (12, 10, 20, 5, 8)
        assert len(np.unique(z["episode_seed"])) == 12
    # every ledger decision uses a reason from the frozen engineering vocabulary.
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert record["decision"] in ("accepted", "rejected")
        if record["decision"] == "rejected":
            assert record["reason"] in coll.REASONS
    # smoke artifacts never mix into the formal role files.
    data_root = Path(config["data_root"])
    assert not (data_root / "object_twinworld_stage4cr2_train.npz").exists()


def test_duplicate_seed_rejected_not_double_accepted(tmp_path):
    config = _config(tmp_path)
    schedule = coll.role_schedule(s4cr2.validate_config(config), "smoke")
    duplicated = np.concatenate([schedule[:1], schedule[:13]]).astype(np.int64)
    receipt = coll.collect_role(config, "smoke", s4cr2.SMOKE_GROUPS, schedule=duplicated,
                                branch_rollout=make_fake_rollout(config), probe_fn=_fake_probe)
    assert receipt.accepted_groups == s4cr2.SMOKE_GROUPS
    assert receipt.skip_counts.get("duplicate_seed", 0) == 1
    data_output, _, _ = coll.role_paths(config, "smoke")
    with np.load(data_output, allow_pickle=True) as z:
        seeds = np.asarray(z["episode_seed"])
    assert len(np.unique(seeds)) == seeds.size


def test_checkpoint_resume_is_idempotent(tmp_path):
    config = _config(tmp_path)
    receipt1 = coll.collect_role(config, "smoke", s4cr2.SMOKE_GROUPS,
                                 branch_rollout=make_fake_rollout(config),
                                 probe_fn=_fake_probe, stop_after_accepted=5)
    assert receipt1.partial and receipt1.accepted_groups == 5
    data_output, _, _ = coll.role_paths(config, "smoke")
    assert not data_output.exists()  # only the atomic partial checkpoint exists
    receipt2 = coll.collect_role(config, "smoke", s4cr2.SMOKE_GROUPS,
                                 branch_rollout=make_fake_rollout(config),
                                 probe_fn=_fake_probe)
    assert not receipt2.partial and receipt2.accepted_groups == 12
    with np.load(data_output, allow_pickle=True) as z:
        seeds = np.asarray(z["episode_seed"])
    assert len(np.unique(seeds)) == seeds.size == 12  # resume never re-accepts a group
    # a third call verifies the completed dataset and never overwrites it.
    sha_before = receipt2.data_output_sha256
    receipt3 = coll.collect_role(config, "smoke", s4cr2.SMOKE_GROUPS,
                                 branch_rollout=make_fake_rollout(config),
                                 probe_fn=_fake_probe)
    assert receipt3.verification == "existing_final_dataset_verified_no_overwrite"
    assert receipt3.data_output_sha256 == sha_before
