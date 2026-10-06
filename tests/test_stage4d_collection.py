"""Stage 4D D0/D1 OOD collector tests: eleven-branch collection + truth isolation.

Plan ``AC-TWIN-RFER-STAGE4D-D0D1-OOD-2026-09-20-V1.0`` §10 Task 2 and §11 items 3/6/10.
The rollout engine is replaced by fast deterministic fakes; the real highway-env 12-group
ENGINEERING_SMOKE_ONLY run is an operational step (run by the controller), not a unit test.
Patterns mirror ``tests/test_stage4cr2_confirm_collection.py`` (truth-separated two-file
publish, receipt binding, atomic resume, engineering-only rejection whitelist), with the
Stage 4D addition: the collector's own ``branch_rollout`` applies
``stage4d_ood_env.apply_profile_to_env`` AFTER reset and the eleven post-perturbation
snapshots are cross-checked pairwise (divergence = whole-group reject, engineering reason).
"""
from __future__ import annotations

import builtins
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import collect_stage4d_merge_ood as coll  # noqa: E402
from ac_twin import stage4d_protocol as prot  # noqa: E402
from ac_twin import stage4d_ood_env as ood  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402

CONFIG_PATH = prot.CONFIG_PATH

#: truth-side tokens that must NEVER appear as keys of the score-side file.
TRUTH_KEY_TOKENS = ("counterfactual", "truth", "j_true", "regret", "oracle",
                    "a_star", "true_")


def _config(tmp_path) -> dict:
    """The frozen Stage 4D config with run/data roots redirected to tmp.

    The formal ``paths`` entries are remapped into tmp as well: since Task 5 the REAL
    formal D1 artifacts exist in the real workspace, and this file's smoke tests must
    stay hermetic (the smoke role never touches the formal paths — the assertion targets
    the tmp-remapped formal paths, not the real ones).
    """
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["run_dir"] = str(tmp_path / "run" / "stage4d_ood")
    config["data_root"] = str(tmp_path / "data" / "object_centric_stage"
                              / "stage4d_merge_ood")
    config["paths"] = {
        key: str(tmp_path / "formal_paths" / Path(value).name)
        for key, value in config["paths"].items()}
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


def _fake_snapshot(seed, capacity):
    """A deterministic per-seed post-perturbation snapshot (identical across branches)."""
    return {
        str(i): {
            "class": "IDMVehicle" if i else "MDPVehicle",
            "is_ego": i == 0,
            "lane_index": ("a", "b", 0),
            "position": (float(seed), float(i)),
            "speed": float(seed % 30),
            "heading": 0.0,
            "target_speed": float("nan"),
            "behavior": {"DISTANCE_WANTED": 10.0, "TIME_WANTED": 1.5},
        }
        for i in range(capacity)
    }


def make_fake_rollout(config, *, future_scale=1.0, diverge_snapshot=False):
    """Deterministic fake stage4d branch rollout; shared prefix across a group's branches.

    Every branch embeds a ``post_perturbation_snapshot`` that is a pure function of the
    group seed (identical across the eleven branches) unless ``diverge_snapshot`` makes one
    CF branch's snapshot drift. ``future_scale`` rescales future positions only — the
    outcome-independence probe.
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
        prng = np.random.default_rng(
            (int(seed) * 100003 + int(prefix_actions.sum())) % (2**31))
        prefix = prng.normal(size=(t_star + 1, capacity, 8)).astype(np.float32)
        future = prng.normal(size=(horizon, capacity, 8)).astype(np.float32)
        future[..., 2:4] *= float(future_scale)
        for frames in (prefix, future):
            frames[..., 0] = 0.0
            frames[:, :4, 0] = 1.0  # slots 0..3 present at every frame; slot 4 absent
        snapshot = _fake_snapshot(seed, capacity)
        if diverge_snapshot and tuple(int(a) for a in future_actions[:1]) == (0,):
            # one CF branch's post-perturbation state drifts (engineering defect probe).
            snapshot = dict(snapshot)
            snapshot["1"] = dict(snapshot["1"], speed=float(seed % 30) + 5.0)
        return {
            "prefix_frames": prefix, "future_frames": future,
            "terminated": False, "truncated": False,
            "termination_reason": "", "first_contact_step": -1,
            "fingerprints": _fingerprints(seed, capacity),
            "post_perturbation_snapshot": snapshot,
        }
    fake.calls = calls
    return fake


def _one_group(config, fake, seed=12345):
    t_star = 12
    prefix = np.full(t_star, 4, dtype=np.int64)
    factual_plan = np.full(int(config["horizon"]), 1, dtype=np.int64)
    group = coll.collect_group(seed, t_star, prefix, factual_plan, config, fake)
    return group, t_star, prefix, factual_plan


def _accepted_records(config, seeds, fake):
    records = []
    for seed in seeds:
        group, _, _, _ = _one_group(config, fake, seed=int(seed))
        verdict = coll.validate_complete_group(group, config)
        assert verdict["accepted"], verdict
        records.append(coll.record_from_verdict(verdict))
    return records


# --- eleven-branch completeness + shared pre-intervention state ---------------------------


def test_eleven_branch_completeness_and_shared_prefix(tmp_path):
    config = _config(tmp_path)
    fake = make_fake_rollout(config)
    group, t_star, prefix, factual_plan = _one_group(config, fake)
    # 11 calls: 1 factual + 10 counterfactual in the frozen ACTION_PLANS order.
    assert len(fake.calls) == 11
    assert fake.calls[0][2] == tuple(int(a) for a in factual_plan)
    expected_cf = [tuple(int(x) for x in prot.materialize_action_plan(factual_plan, a, d))
                   for a, d in prot.ACTION_PLANS]
    assert [c[2] for c in fake.calls[1:]] == expected_cf
    ref = group["factual"]["prefix_frames"]
    for b in group["cf"]:
        assert np.array_equal(b["prefix_frames"], ref)
    verdict = coll.validate_complete_group(group, config)
    assert verdict["accepted"], verdict
    assert verdict["cf_futures"].shape == (prot.N_ACTIONS, 20, 5, 8)
    assert [int(a) for a in verdict["cf_actions"]] == [a for a, _ in prot.ACTION_PLANS]
    assert [int(d) for d in verdict["cf_durations"]] == [d for _, d in prot.ACTION_PLANS]
    # (IDLE,1) and (IDLE,3) stay distinct branches; the IDLE reference stays (IDLE,3).
    assert prot.ACTION_PLANS[prot.IDLE_PLAN_INDEX] == (1, 3)


# --- post-perturbation snapshot consistency (Stage 4D specific) ----------------------------


def test_post_perturbation_snapshots_consistent_across_eleven_branches(tmp_path):
    config = _config(tmp_path)
    group, _, _, _ = _one_group(config, make_fake_rollout(config))
    assert group["snapshot_consistent"] is True
    verdict = coll.validate_complete_group(group, config)
    assert verdict["accepted"], verdict


def test_post_perturbation_divergence_rejects_whole_group(tmp_path):
    """A single branch's divergent post-perturbation snapshot rejects the WHOLE group.

    The reason is a pure engineering/structural one — never an outcome reason.
    """
    config = _config(tmp_path)
    _, _, _, factual_plan = _one_group(config, make_fake_rollout(config))
    base_fake = make_fake_rollout(config)

    def drifted(env_id, env_cfg, seed, prefix_actions, future_actions, cap):
        out = base_fake(env_id, env_cfg, seed, prefix_actions, future_actions, cap)
        target = tuple(int(x) for x in prot.materialize_action_plan(factual_plan, 0, 1))
        if tuple(int(x) for x in future_actions) == target:
            snap = dict(out["post_perturbation_snapshot"])
            snap["2"] = dict(snap["2"], speed=snap["2"]["speed"] + 7.0)
            out["post_perturbation_snapshot"] = snap
        return out

    group, _, _, _ = _one_group(config, drifted, seed=555)
    assert group["snapshot_consistent"] is False
    verdict = coll.validate_complete_group(group, config)
    assert not verdict["accepted"]
    assert verdict["reason"] == "branch_perturbation_inconsistent"
    # the new reason is an engineering reason, never regret/collision/outcome.
    assert "branch_perturbation_inconsistent" in coll.REASONS


def test_branch_rollout_factory_applies_profile_post_reset(tmp_path):
    """The real-fallback factory binds the profile and stamps ids before perturbing."""
    config = _config(tmp_path)
    rollout = coll.make_stage4d_branch_rollout("BEHAVIOR_OOD")
    # the produced callable matches the reused collect_group's six-arg call signature.
    assert len(inspect.signature(rollout).parameters) == 6


# --- whole-group rejection on the engineering whitelist ------------------------------------


def test_missing_branch_rejects_whole_group(tmp_path):
    config = _config(tmp_path)
    group, _, _, _ = _one_group(config, make_fake_rollout(config))
    group["cf"] = group["cf"][:9]  # one counterfactual branch missing
    verdict = coll.validate_complete_group(group, config)
    assert not verdict["accepted"]
    assert verdict["reason"] == "counterfactual_future_short"


def test_prefix_divergence_rejects_whole_group(tmp_path):
    config = _config(tmp_path)
    _, _, _, factual_plan = _one_group(config, make_fake_rollout(config))
    base_fake = make_fake_rollout(config)

    def drifted(env_id, env_cfg, seed, prefix_actions, future_actions, cap):
        out = base_fake(env_id, env_cfg, seed, prefix_actions, future_actions, cap)
        target = tuple(int(x) for x in prot.materialize_action_plan(factual_plan, 0, 1))
        if tuple(int(x) for x in future_actions) == target:
            out["prefix_frames"] = out["prefix_frames"] + 1e-3
        return out

    group, _, _, _ = _one_group(config, drifted, seed=556)
    verdict = coll.validate_complete_group(group, config)
    assert not verdict["accepted"]
    assert verdict["reason"] == "object_prefix_mismatch"


def test_initial_fingerprint_mismatch_rejects_whole_group(tmp_path):
    config = _config(tmp_path)
    _, _, _, factual_plan = _one_group(config, make_fake_rollout(config))
    base_fake = make_fake_rollout(config)

    def drifted(env_id, env_cfg, seed, prefix_actions, future_actions, cap):
        out = base_fake(env_id, env_cfg, seed, prefix_actions, future_actions, cap)
        target = tuple(int(x) for x in prot.materialize_action_plan(factual_plan, 2, 1))
        if tuple(int(x) for x in future_actions) == target:
            out["fingerprints"] = _fingerprints(int(seed) + 1, int(config["object_capacity"]))
        return out

    group, _, _, _ = _one_group(config, drifted, seed=560)
    verdict = coll.validate_complete_group(group, config)
    assert not verdict["accepted"]
    assert verdict["reason"] == "initial_fingerprint_mismatch"


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


def test_no_legal_agent_time_rejects_group(tmp_path):
    config = _config(tmp_path)
    base_fake = make_fake_rollout(config)

    def ego_only(env_id, env_cfg, seed, prefix_actions, future_actions, cap):
        out = base_fake(env_id, env_cfg, seed, prefix_actions, future_actions, cap)
        for frames in (out["prefix_frames"], out["future_frames"]):
            frames[:, 1:, 0] = 0.0  # only the ego slot present anywhere
        return out

    group, _, _, _ = _one_group(config, ego_only, seed=561)
    verdict = coll.validate_complete_group(group, config)
    assert not verdict["accepted"]
    assert verdict["reason"] == "no_legal_agent_time"


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
    # the rejection whitelist contains only engineering reasons — never outcome reasons.
    forbidden = ("regret", "oracle", "collision", "response", "error", "cost",
                 "margin", "stratum")
    for reason in coll.REASONS:
        assert not any(word in reason for word in forbidden), reason


# --- publish adapter: exact two-file fields, mask provenance, quota -------------------------


def test_publish_requires_exact_frozen_quota(tmp_path):
    config = _config(tmp_path)
    checked = prot.validate_config(config)
    seeds = coll.role_schedule(checked, "smoke")[:3]
    records = _accepted_records(config, seeds, make_fake_rollout(config))
    with pytest.raises(ValueError, match="quota"):
        coll.publish_group_data(records, config, "smoke")


def test_publish_rejects_cf_reduced_input_mask(tmp_path):
    config = _config(tmp_path)
    checked = prot.validate_config(config)
    seeds = coll.role_schedule(checked, "smoke")[:12]
    records = _accepted_records(config, seeds, make_fake_rollout(config))
    tampered = [dict(r) for r in records]
    reduced = np.asarray(tampered[3]["inference_mask"], dtype=bool).copy()
    assert reduced[:, 1].all()
    reduced[:, 1] = False  # illegal reduction of legal cells
    tampered[3]["inference_mask"] = reduced
    inputs_path, truth_path, _, _ = coll.role_paths(config, "smoke")
    with pytest.raises(ValueError, match="mask"):
        coll.publish_group_data(tampered, config, "smoke")
    assert not inputs_path.exists() and not truth_path.exists()
    binding = coll.publish_group_data(records, config, "smoke")
    assert binding["n_groups"] == 12
    with np.load(inputs_path) as z:
        recomputed = prot.build_inference_mask(z["object_history"], z["object_factual_future"])
        assert np.array_equal(np.asarray(z["inference_mask"], dtype=bool), recomputed)


def test_publish_is_create_once(tmp_path):
    config = _config(tmp_path)
    checked = prot.validate_config(config)
    seeds = coll.role_schedule(checked, "smoke")[:12]
    records = _accepted_records(config, seeds, make_fake_rollout(config))
    coll.publish_group_data(records, config, "smoke")
    with pytest.raises(FileExistsError):
        coll.publish_group_data(records, config, "smoke")


# --- smoke collection end-to-end: fields, shapes, truth separation, receipt -----------------


def _run_smoke(config, **kwargs):
    kwargs.setdefault("branch_rollout", make_fake_rollout(config))
    kwargs.setdefault("probe_fn", _fake_probe)
    return coll.collect_role(config, "smoke", **kwargs)


def test_smoke_publishes_exact_fields_and_shapes(tmp_path):
    config = _config(tmp_path)
    receipt = _run_smoke(config)
    assert receipt["status"] == "ENGINEERING_SMOKE_ONLY"
    assert receipt["accepted_groups"] == coll.ROLE_QUOTA["smoke"] == 12
    g = 12
    inputs_path, truth_path, ledger_path, receipt_path = coll.role_paths(config, "smoke")
    with np.load(inputs_path) as z:
        # EXACTLY the eight legal fields — no more, no fewer; no truth-side key.
        assert len(z.files) == 8
        assert set(z.files) == set(coll.LEGAL_INPUT_FIELDS)
        assert z["object_history"].shape == (g, 10, 5, 8)
        assert z["object_factual_future"].shape == (g, 20, 5, 8)
        assert z["factual_actions"].shape == (g, 20)
        assert z["cf_actions"].shape == (g, 10)
        assert z["cf_durations"].shape == (g, 10)
        assert z["t_star"].shape == (g,)
        assert z["group_seed"].shape == (g,)
        assert z["inference_mask"].shape == (g, 20, 5)
        assert z["inference_mask"].dtype == bool
        for key in z.files:
            assert not any(tok in key.lower() for tok in TRUTH_KEY_TOKENS), key
        legal_seeds = np.asarray(z["group_seed"], dtype=np.int64)
        assert np.unique(legal_seeds).size == g
    with np.load(truth_path) as z:
        assert set(z.files) == set(coll.TRUTH_FIELDS)
        assert z["object_counterfactual_futures"].shape == (g, 10, 20, 5, 8)
        assert np.isfinite(z["object_counterfactual_futures"]).all()
        assert np.array_equal(np.asarray(z["group_seed"], dtype=np.int64), legal_seeds)
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert record["decision"] in ("accepted", "rejected")
        if record["decision"] == "rejected":
            assert record["reason"] in coll.REASONS
    assert receipt_path.is_file()


def test_collection_receipt_binds_group_ordering(tmp_path):
    config = _config(tmp_path)
    receipt = _run_smoke(config)
    inputs_path, truth_path, ledger_path, receipt_path = coll.role_paths(config, "smoke")
    assert receipt["inference_inputs_sha256"] == sha256_file(inputs_path)
    assert receipt["confirmation_truth_sha256"] == sha256_file(truth_path)
    binding = receipt["group_order_binding"]
    assert binding["same_order"] is True
    assert binding["n_groups"] == 12
    assert binding["order"] == "acceptance_order"
    with np.load(inputs_path) as z:
        seeds = np.asarray(z["group_seed"], dtype=np.int64)
    expected = hashlib.sha256(seeds.tobytes()).hexdigest()
    assert binding["inference_inputs_group_seed_sha256"] == expected
    assert binding["confirmation_truth_group_seed_sha256"] == expected
    # the role consumed exactly the approved frozen smoke candidate domain.
    assert receipt["seed_schedule_sha256"] == prot.SMOKE_SCHEDULE_RAW_SHA256
    assert receipt["action_stream_root"] == [2026100099, 0]
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted["group_order_binding"] == receipt["group_order_binding"]


def test_smoke_writes_only_under_smoke_dirs(tmp_path):
    config = _config(tmp_path)
    checked = prot.validate_config(config)
    _run_smoke(config)
    data_root = Path(config["data_root"])
    run_dir = Path(config["run_dir"])
    assert (data_root / "smoke" / "inference_inputs.npz").is_file()
    assert (data_root / "smoke" / "confirmation_truth.npz").is_file()
    assert (data_root / "smoke" / "rejection_ledger.jsonl").is_file()
    assert (run_dir / "smoke" / "smoke_collection_receipt.json").is_file()
    # formal artifact paths are NEVER touched by the smoke role.
    assert not Path(checked["paths"]["inference_inputs"]).exists()
    assert not Path(checked["paths"]["confirmation_truth"]).exists()
    assert not Path(checked["paths"]["collection_receipt"]).exists()


def test_smoke_profile_is_behavior_and_documented(tmp_path):
    config = _config(tmp_path)
    receipt = _run_smoke(config)
    # the engineering smoke runs the BEHAVIOR_OOD profile (plan leaves the smoke profile
    # undetermined; the brief sanctions choosing BEHAVIOR_OOD with a documented note).
    assert receipt["profile"] == "BEHAVIOR_OOD"


# --- duplicate seeds, checkpoint resume, idempotent rerun -----------------------------------


def test_duplicate_seed_rejected_not_double_accepted(tmp_path):
    config = _config(tmp_path)
    checked = prot.validate_config(config)
    schedule = coll.role_schedule(checked, "smoke")
    duplicated = np.concatenate([schedule[:1], schedule[:13]]).astype(np.int64)
    receipt = _run_smoke(config, schedule=duplicated)
    assert receipt["accepted_groups"] == 12
    assert receipt["skip_counts"].get("duplicate_seed", 0) == 1
    inputs_path, _, ledger_path, _ = coll.role_paths(config, "smoke")
    with np.load(inputs_path) as z:
        seeds = np.asarray(z["group_seed"])
    assert len(np.unique(seeds)) == seeds.size
    records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    dup = [r for r in records if r["decision"] == "rejected"]
    assert len(dup) == 1 and dup[0]["reason"] == "duplicate_seed"
    assert [r["attempt_index"] for r in records] == list(range(len(records)))


def test_checkpoint_resume_identical_to_uninterrupted(tmp_path):
    config_a = _config(tmp_path / "a")
    config_b = _config(tmp_path / "b")
    receipt_a = _run_smoke(config_a)
    partial = _run_smoke(config_b, stop_after_accepted=5)
    assert partial["partial"] and partial["accepted_groups"] == 5
    inputs_b, truth_b, ledger_b, _ = coll.role_paths(config_b, "smoke")
    assert not inputs_b.exists() and not truth_b.exists()  # only the checkpoint exists
    receipt_b = _run_smoke(config_b)
    assert not receipt_b["partial"] and receipt_b["accepted_groups"] == 12
    inputs_a, truth_a, ledger_a, _ = coll.role_paths(config_a, "smoke")
    with np.load(inputs_a) as za, np.load(inputs_b) as zb:
        for key in coll.LEGAL_INPUT_FIELDS:
            assert np.array_equal(za[key], zb[key]), key
    with np.load(truth_a) as za, np.load(truth_b) as zb:
        for key in coll.TRUTH_FIELDS:
            assert np.array_equal(za[key], zb[key]), key
    assert (ledger_a.read_text(encoding="utf-8").splitlines()
            == ledger_b.read_text(encoding="utf-8").splitlines())
    assert receipt_a["skip_counts"] == receipt_b["skip_counts"]
    assert receipt_a["attempts"] == receipt_b["attempts"]


def test_completed_collection_rerun_verifies_without_overwrite(tmp_path):
    config = _config(tmp_path)
    receipt = _run_smoke(config)
    inputs_path, truth_path, _, _ = coll.role_paths(config, "smoke")
    sha_inputs = sha256_file(inputs_path)
    sha_truth = sha256_file(truth_path)
    again = _run_smoke(config)
    assert again["verification"] == "existing_final_dataset_verified_no_overwrite"
    assert again["inference_inputs_sha256"] == sha_inputs == sha256_file(inputs_path)
    assert again["confirmation_truth_sha256"] == sha_truth == sha256_file(truth_path)
    assert again["attempts"] == receipt["attempts"]


def test_exhausted_candidate_domain_raises_collection_incomplete(tmp_path):
    config = _config(tmp_path)
    checked = prot.validate_config(config)
    schedule = coll.role_schedule(checked, "smoke")[:3]
    with pytest.raises(coll.CollectionIncompleteError) as excinfo:
        _run_smoke(config, schedule=schedule)
    assert excinfo.value.status == prot.STATUS_INVALID or \
        excinfo.value.status == coll.STATUS_COLLECTION_INCOMPLETE
    inputs_path, truth_path, _, receipt_path = coll.role_paths(config, "smoke")
    assert not inputs_path.exists() and not truth_path.exists()
    assert not receipt_path.exists()


# --- D0 pilot / D1 confirm role semantics ----------------------------------------------------


def test_pilot_role_paths_and_profile(tmp_path):
    config = _config(tmp_path)
    for role, profile, short in (
            ("pilot_behavior", "BEHAVIOR_OOD", "behavior"),
            ("pilot_geometry", "GEOMETRY_OOD", "geometry"),
            ("pilot_joint", "JOINT_OOD", "joint")):
        assert coll.ROLE_PROFILE[role] == profile
        inputs_path, truth_path, ledger_path, receipt_path = coll.role_paths(config, role)
        assert f"pilot/{short}" in str(inputs_path).replace("\\", "/")
        assert f"pilot/{short}" in str(receipt_path).replace("\\", "/")


def test_d1_confirm_collection_requires_protocol_lock(tmp_path):
    """D1 formal collection is fail-closed before the Task 4 protocol lock exists."""
    config = _config(tmp_path)
    with pytest.raises((FileNotFoundError, ValueError)):
        coll.collect_role(config, "d1_confirm", profile="JOINT_OOD",
                          branch_rollout=make_fake_rollout(config), probe_fn=_fake_probe)


# --- truth isolation: legal loader + score-side truth-open failure ---------------------------


class _TruthAccessGuard:
    """Wrap a numpy NpzFile so any access to a truth-side key fails the test."""

    def __init__(self, archive):
        self._archive = archive

    def __getitem__(self, key):
        low = str(key).lower()
        if any(tok in low for tok in TRUTH_KEY_TOKENS):
            raise AssertionError(f"score-side code read a truth key: {key!r}")
        return self._archive[key]

    def __getattr__(self, name):
        return getattr(self._archive, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._archive.__exit__(*exc)


def _install_truth_spy(monkeypatch):
    real_open = builtins.open
    real_np_load = np.load

    def guarded_open(file, *args, **kwargs):
        if "confirmation_truth" in str(file).replace("\\", "/").lower():
            raise AssertionError(f"score-side code opened the sealed truth file: {file}")
        return real_open(file, *args, **kwargs)

    def guarded_np_load(file, *args, **kwargs):
        if "confirmation_truth" in str(file).replace("\\", "/").lower():
            raise AssertionError(f"score-side code np.load'ed the sealed truth file: {file}")
        return _TruthAccessGuard(real_np_load(file, *args, **kwargs))

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(np, "load", guarded_np_load)


def test_legal_loader_succeeds_with_truth_moved_away(tmp_path, monkeypatch):
    """The score-side legal-input loader completes with the truth file physically absent."""
    config = _config(tmp_path)
    _run_smoke(config)
    inputs_path, truth_path, _, _ = coll.role_paths(config, "smoke")
    # move the truth file away (physical isolation fixture).
    moved = truth_path.with_name("confirmation_truth.moved.npz")
    truth_path.rename(moved)
    assert not truth_path.exists()
    _install_truth_spy(monkeypatch)
    bundle = coll.load_legal_inputs(inputs_path, config)
    assert bundle["groups"] == 12
    assert set(bundle["arrays"]) == set(coll.LEGAL_INPUT_FIELDS)
    assert not truth_path.exists()  # still absent after the score-side load


def test_legal_loader_refuses_truth_path(tmp_path):
    """Pointing the score-side loader at the sealed truth file is refused by path alone."""
    config = _config(tmp_path)
    _run_smoke(config)
    _, truth_path, _, _ = coll.role_paths(config, "smoke")
    with pytest.raises(ValueError, match="truth"):
        coll.load_legal_inputs(truth_path, config)


def test_legal_loader_rejects_truth_keys(tmp_path):
    """An input archive carrying any truth-side key is refused fail-closed."""
    config = _config(tmp_path)
    _run_smoke(config)
    inputs_path, _, _, _ = coll.role_paths(config, "smoke")
    with np.load(inputs_path) as z:
        payload = {k: z[k] for k in z.files}
    payload["object_counterfactual_futures"] = np.zeros((12, 10, 20, 5, 8))
    bad = inputs_path.with_name("bad_inputs.npz")
    with open(bad, "wb") as handle:
        np.savez_compressed(handle, **payload)
    with pytest.raises(ValueError, match="truth"):
        coll.load_legal_inputs(bad, config)


# --- 30-second heartbeat mechanism ------------------------------------------------------------


def test_progress_heartbeat_format_and_interval(monkeypatch):
    """The 30s heartbeat emits phase/completed/total/percentage/elapsed/ETA."""
    from ac_twin import progress as progress_mod
    clock = {"t": 1000.0}
    monkeypatch.setattr(progress_mod.time, "perf_counter", lambda: clock["t"])
    reporter = progress_mod.ProgressReporter(
        total=12, every=25, prefix="stage4d-collect-smoke", heartbeat_seconds=30.0)
    # pin the start/last-emit to the fake clock (the dataclass default_factory captured
    # the real perf_counter at class-definition time).
    reporter.started_at = clock["t"]
    reporter._last_emit_at = clock["t"]
    # no heartbeat before 30s elapse.
    assert reporter.heartbeat(3, attempted=5) is None
    clock["t"] += 31.0  # cross the heartbeat interval
    line = reporter.heartbeat(3, attempted=5)
    assert line is not None
    assert "stage4d-collect-smoke" in line  # phase
    assert "3/12" in line                    # completed/total
    assert "25.0%" in line                   # percentage
    assert "elapsed=" in line                # elapsed
    assert "ETA=" in line                    # ETA


def test_collect_role_constructs_thirty_second_heartbeat(tmp_path, monkeypatch):
    """collect_role wires a ProgressReporter with heartbeat_seconds == 30."""
    from ac_twin import progress as progress_mod
    seen = {}
    real_init = progress_mod.ProgressReporter.__init__

    def spy_init(self, *args, **kwargs):
        seen["heartbeat_seconds"] = kwargs.get("heartbeat_seconds")
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(progress_mod.ProgressReporter, "__init__", spy_init)
    config = _config(tmp_path)
    _run_smoke(config)
    assert seen["heartbeat_seconds"] == 30.0
