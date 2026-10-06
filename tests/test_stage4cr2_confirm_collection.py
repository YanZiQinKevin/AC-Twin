"""Stage 4C-R2 independent-confirmation collector tests.

Plan ``AC-TWIN-RFER-STAGE4CR2-CONFIRM-2026-09-17-V0.1`` sections 3.1, 4.2, 5.1,
5.2, 10 and Task 2. The rollout engine is replaced by fast deterministic fakes;
the real highway-env 12-group engineering smoke belongs to the operational phase
(run later by the controller), not to unit tests. Patterns mirror
``tests/test_stage4cr2_collector.py`` (R2), whose rollout engine fake contract is
reused here.
"""
from __future__ import annotations

import hashlib
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

import collect_stage4cr2_merge_confirm as coll  # noqa: E402
from ac_twin import stage4cr2_confirm_protocol as prot  # noqa: E402
from ac_twin.checkpointing import canonical_fingerprint  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402

CONFIG_PATH = prot.CONFIG_PATH

#: truth-side tokens that must NEVER appear as keys of the score-side file
#: (mirrors stage4cr2_confirm_inference.FORBIDDEN_INPUT_KEY_TOKENS).
TRUTH_KEY_TOKENS = ("counterfactual", "truth", "j_true", "regret", "oracle",
                    "a_star", "true_")


def _config(tmp_path) -> dict:
    """The frozen confirm config with run/data/protocol roots redirected to tmp."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    run_dir = tmp_path / "run" / "stage4cr2_confirm"
    data_root = (tmp_path / "data" / "object_centric_stage"
                 / "stage4cr2_merge_multiaction_confirm")
    config["run_dir"] = str(run_dir)
    config["data_root"] = str(data_root)
    paths = dict(config["paths"])
    paths["protocol_dir"] = str(run_dir / "protocol")
    paths["protocol_lock"] = str(run_dir / "protocol" / "protocol_lock.json")
    paths["collection_receipt"] = str(run_dir / "collection" / "collection_receipt.json")
    paths["inference_inputs"] = str(data_root / "inference_inputs.npz")
    paths["confirmation_truth"] = str(data_root / "confirmation_truth.npz")
    paths["rejection_ledger"] = str(data_root / "rejection_ledger.jsonl")
    config["paths"] = paths
    return config


def _write_minimal_lock(config) -> Path:
    """Write a minimal stand-in protocol lock at the (tmp-redirected) lock path.

    The real formal lock is created ONLY by the controller in the operational
    phase via ``prot.create_protocol_lock``; unit tests never burn it. This
    stand-in carries exactly the fields the collector's ordering guard reads:
    plan id, running-config signature, locked confirm schedule raw hash and the
    locked code hash of the collector file.
    """
    checked = prot.validate_config(config)
    lock_path = Path(checked["paths"]["protocol_lock"])
    payload = {
        "plan_id": prot.PLAN_ID,
        "config_signature": canonical_fingerprint(json.loads(json.dumps(checked))),
        "confirm_schedule_raw_sha256": prot.CONFIRM_SCHEDULE_RAW_SHA256,
        "code_sha256": {
            "collect_stage4cr2_merge_confirm.py": sha256_file(
                CODE / "collect_stage4cr2_merge_confirm.py"),
        },
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    return lock_path


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


def make_fake_rollout(config, *, future_scale=1.0, crash=False):
    """Deterministic fake branch rollout; shared prefix across branches of a group.

    ``future_scale`` multiplies future-frame positions only (channels 2:4): it
    changes every downstream consequence/regret value WITHOUT changing presence,
    finiteness, shapes or termination — the outcome-independence probe.
    ``crash=True`` marks a legal collision (crashed diag + first_contact_step)
    with NO termination — the legal-collision inclusion probe.
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
        diag = {k: np.zeros((len(prefix), capacity), dtype=np.float32)
                for k in ("lane_index", "target_lane_index", "commanded_accel",
                          "commanded_steering", "crashed")}
        fdiag = {k: np.zeros((horizon, capacity), dtype=np.float32)
                 for k in ("lane_index", "target_lane_index", "commanded_accel",
                           "commanded_steering", "crashed")}
        first_contact = -1
        if crash:
            fdiag["crashed"][3:, 1] = 1.0  # legal collision mid-horizon
            first_contact = 3
        return {
            "prefix_frames": prefix, "future_frames": future,
            "terminated": False, "truncated": False,
            "termination_reason": "collision" if crash else "",
            "first_contact_step": first_contact,
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
    group = coll.collect_group(seed, t_star, prefix, factual_plan, config, fake)
    return group, t_star, prefix, factual_plan


def _accepted_records(config, seeds, fake):
    """Drive the reused low-level group machinery into publish-ready records."""
    records = []
    for seed in seeds:
        group, _, _, _ = _one_group(config, fake, seed=int(seed))
        verdict = coll.validate_complete_group(group, config)
        assert verdict["accepted"], verdict
        records.append(coll.record_from_verdict(verdict))
    return records


# --- ten-action completeness + shared pre-intervention state (reused low-level) ----------


def test_ten_action_group_completeness_and_shared_prefix(tmp_path):
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
    # (IDLE,1) and (IDLE,3) stay distinct branches.
    assert (1, 1) in [tuple(p) for p in prot.ACTION_PLANS]
    assert prot.ACTION_PLANS[prot.IDLE_PLAN_INDEX] == (1, 3)


# --- whole-group rejection on the §5.2 engineering whitelist ----------------------------


def test_missing_branch_rejects_whole_group(tmp_path):
    config = _config(tmp_path)
    group, _, _, _ = _one_group(config, make_fake_rollout(config))
    group["cf"] = group["cf"][:9]  # one counterfactual branch missing
    verdict = coll.validate_complete_group(group, config)
    assert not verdict["accepted"]
    assert verdict["reason"] == "counterfactual_future_short"


def test_abnormal_termination_rejects_whole_group(tmp_path):
    config = _config(tmp_path)
    _, _, _, factual_plan = _one_group(config, make_fake_rollout(config))
    base_fake = make_fake_rollout(config)

    def terminating(env_id, env_cfg, seed, prefix_actions, future_actions, cap):
        out = base_fake(env_id, env_cfg, seed, prefix_actions, future_actions, cap)
        target = tuple(int(x) for x in prot.materialize_action_plan(factual_plan, 3, 3))
        if tuple(int(x) for x in future_actions) == target:
            out["terminated"] = True
        return out

    group, _, _, _ = _one_group(config, terminating, seed=999)
    verdict = coll.validate_complete_group(group, config)
    assert not verdict["accepted"]
    assert verdict["reason"] == "prefix_terminated"
    assert verdict["failed_branch"] == "cf_7"  # (FASTER,3) is plan index 7


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

    group, _, _, _ = _one_group(config, drifted, seed=555)
    verdict = coll.validate_complete_group(group, config)
    assert not verdict["accepted"]
    assert verdict["reason"] == "object_prefix_mismatch"


def test_per_frame_entity_mismatch_rejects_whole_group(tmp_path):
    """Present sets are compared by stable ID across ALL frames, not only at reset.

    The mismatch is introduced only from frame 5 onward: the reset frame and the
    first four frames match exactly, so a reset-only comparison would accept.
    """
    config = _config(tmp_path)
    _, _, _, factual_plan = _one_group(config, make_fake_rollout(config))
    base_fake = make_fake_rollout(config)

    def mismatch(env_id, env_cfg, seed, prefix_actions, future_actions, cap):
        out = base_fake(env_id, env_cfg, seed, prefix_actions, future_actions, cap)
        target = tuple(int(x) for x in prot.materialize_action_plan(factual_plan, 0, 1))
        if tuple(int(x) for x in future_actions) == target:
            out["future_frames"][5:, 3, 0] = 0.0  # drop a present entity mid-horizon
        return out

    group, _, _, _ = _one_group(config, mismatch, seed=556)
    verdict = coll.validate_complete_group(group, config)
    assert not verdict["accepted"]
    assert verdict["reason"] == "present_set_mismatch"


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


# --- legal collisions are INCLUDED; rejection is outcome-independent (§5.2) --------------


def test_legal_collision_is_not_rejected(tmp_path):
    config = _config(tmp_path)
    fake = make_fake_rollout(config, crash=True)
    group, _, _, _ = _one_group(config, fake, seed=424242)
    # the collision really happened in the factual branch (and every CF branch).
    assert group["factual"]["first_contact_step"] >= 0
    assert not group["factual"]["terminated"]
    verdict = coll.validate_complete_group(group, config)
    assert verdict["accepted"], verdict


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
    forbidden = ("regret", "oracle", "collision", "response", "error", "cost",
                 "margin", "stratum")
    for reason in coll.REASONS:
        assert not any(word in reason for word in forbidden), reason


# --- publish adapter: exact §5.1 fields, mask provenance, quota --------------------------


def test_publish_requires_exact_frozen_quota(tmp_path):
    config = _config(tmp_path)
    checked = prot.validate_config(config)
    seeds = prot.build_smoke_seed_schedule(checked)[:3]
    records = _accepted_records(config, seeds, make_fake_rollout(config))
    with pytest.raises(ValueError, match="quota"):
        coll.publish_group_data(records, config, "smoke")


def test_publish_rejects_cf_reduced_input_mask(tmp_path):
    """The adapter refuses any input mask not generated by the legal contract.

    The tampered record's mask drops legal history/factual cells — exactly what
    an illegal reduction from a counterfactual mask would look like. The publish
    adapter must reject it BEFORE writing anything, and accept the untampered
    records (whose mask equals an independent history/factual-only recomputation).
    """
    config = _config(tmp_path)
    checked = prot.validate_config(config)
    seeds = prot.build_smoke_seed_schedule(checked)[:12]
    records = _accepted_records(config, seeds, make_fake_rollout(config))
    tampered = [dict(r) for r in records]
    reduced = np.asarray(tampered[3]["inference_mask"], dtype=bool).copy()
    assert reduced[:, 1].all()  # slot 1 is a legal cell under the fake contract
    reduced[:, 1] = False       # illegal reduction of legal cells
    tampered[3]["inference_mask"] = reduced
    inputs_path, truth_path, _, _ = coll.role_paths(config, "smoke")
    with pytest.raises(ValueError, match="mask"):
        coll.publish_group_data(tampered, config, "smoke")
    assert not inputs_path.exists() and not truth_path.exists()  # nothing written
    # untampered records publish cleanly; the published mask equals the
    # independent history/factual-only recomputation (never a CF reduction).
    binding = coll.publish_group_data(records, config, "smoke")
    assert binding["n_groups"] == 12
    with np.load(inputs_path) as z:
        recomputed = prot.build_inference_mask(z["object_history"], z["object_factual_future"])
        assert np.array_equal(np.asarray(z["inference_mask"], dtype=bool), recomputed)


def test_publish_is_create_once(tmp_path):
    config = _config(tmp_path)
    checked = prot.validate_config(config)
    seeds = prot.build_smoke_seed_schedule(checked)[:12]
    records = _accepted_records(config, seeds, make_fake_rollout(config))
    coll.publish_group_data(records, config, "smoke")
    with pytest.raises(FileExistsError):
        coll.publish_group_data(records, config, "smoke")


# --- smoke collection end-to-end: fields, shapes, truth separation, receipt binding ------


def _run_smoke(config):
    return coll.collect_confirmation(config, "smoke",
                                     branch_rollout=make_fake_rollout(config),
                                     probe_fn=_fake_probe)


def test_smoke_publishes_exact_fields_and_shapes(tmp_path):
    config = _config(tmp_path)
    receipt = _run_smoke(config)
    assert receipt["status"] == "ENGINEERING_SMOKE_ONLY"
    assert receipt["accepted_groups"] == prot.SMOKE_ACCEPTED_GROUPS
    g = prot.SMOKE_ACCEPTED_GROUPS
    inputs_path, truth_path, ledger_path, receipt_path = coll.role_paths(config, "smoke")
    with np.load(inputs_path) as z:
        # EXACTLY the eight §5.1 legal fields — no more, no fewer.
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
        # the score-side file contains NO counterfactual-truth field.
        for key in z.files:
            assert not any(tok in key.lower() for tok in TRUTH_KEY_TOKENS), key
        legal_seeds = np.asarray(z["group_seed"], dtype=np.int64)
        assert np.unique(legal_seeds).size == g
    with np.load(truth_path) as z:
        assert set(z.files) == set(coll.TRUTH_FIELDS)
        assert z["object_counterfactual_futures"].shape == (g, 10, 20, 5, 8)
        assert np.isfinite(z["object_counterfactual_futures"]).all()
        # same-order group binding and integrity copies match the legal file.
        assert np.array_equal(np.asarray(z["group_seed"], dtype=np.int64), legal_seeds)
    with np.load(inputs_path) as zi, np.load(truth_path) as zt:
        for key in ("t_star", "factual_actions", "cf_actions", "cf_durations",
                    "inference_mask"):
            assert np.array_equal(zi[key], zt[key]), key
    # ledger decisions use only the frozen engineering vocabulary.
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
    assert receipt["rejection_ledger"] == str(ledger_path)
    assert receipt["receipt_path"] == str(receipt_path)
    binding = receipt["group_order_binding"]
    assert binding["same_order"] is True
    assert binding["n_groups"] == prot.SMOKE_ACCEPTED_GROUPS
    assert binding["order"] == "acceptance_order"
    with np.load(inputs_path) as z:
        seeds = np.asarray(z["group_seed"], dtype=np.int64)
    expected = hashlib.sha256(seeds.tobytes()).hexdigest()
    assert binding["inference_inputs_group_seed_sha256"] == expected
    assert binding["confirmation_truth_group_seed_sha256"] == expected
    # the role consumed exactly the approved frozen smoke candidate domain.
    assert receipt["seed_schedule_sha256"] == prot.SMOKE_SCHEDULE_RAW_SHA256
    assert receipt["action_stream_root"] == [2026093100, 12]
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted["group_order_binding"] == receipt["group_order_binding"]
    assert persisted["confirmation_truth_sha256"] == receipt["confirmation_truth_sha256"]


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
    # formal artifact paths and the formal protocol lock are NEVER touched.
    assert not Path(checked["paths"]["inference_inputs"]).exists()
    assert not Path(checked["paths"]["confirmation_truth"]).exists()
    assert not Path(checked["paths"]["rejection_ledger"]).exists()
    assert not Path(checked["paths"]["collection_receipt"]).exists()
    assert not Path(checked["paths"]["protocol_lock"]).exists()


# --- action stream: frozen roots and R2's original sampling order (§4.2) -----------------


def test_action_stream_root_and_r2_sampling_order(tmp_path):
    config = _config(tmp_path)
    assert prot.ACTION_STREAM_ROOTS["confirm"] == (2026093012, 0)
    assert prot.ACTION_STREAM_ROOTS["smoke"] == (2026093100, 12)
    fake = make_fake_rollout(config)
    coll.collect_confirmation(config, "smoke", branch_rollout=fake, probe_fn=_fake_probe)
    # independently regenerate the expected stream: default_rng(SeedSequence(
    # [smoke_entropy, 12])), drawing t_star, then prefix, then the factual plan.
    action_ids = np.arange(5, dtype=np.int64)
    probs = np.full(5, 0.2, dtype=np.float64)
    rng = np.random.default_rng(np.random.SeedSequence([2026093100, 12]))
    expected = []
    for _ in range(prot.SMOKE_ACCEPTED_GROUPS):
        t_star = int(rng.integers(10, 16))
        prefix = rng.choice(action_ids, size=t_star, p=probs)
        plan = rng.choice(action_ids, size=20, p=probs)
        expected.append((t_star, prefix, plan))
    inputs_path, _, _, _ = coll.role_paths(config, "smoke")
    with np.load(inputs_path) as z:
        assert [int(v) for v in z["t_star"]] == [e[0] for e in expected]
        for row, e in zip(z["factual_actions"], expected):
            assert np.array_equal(np.asarray(row), e[2])
    for i, e in enumerate(expected):
        assert fake.calls[11 * i][1] == tuple(int(a) for a in e[1])


# --- duplicate seeds, checkpoint resume, idempotent rerun ---------------------------------


def test_duplicate_seed_rejected_not_double_accepted(tmp_path):
    config = _config(tmp_path)
    checked = prot.validate_config(config)
    schedule = prot.build_smoke_seed_schedule(checked)
    duplicated = np.concatenate([schedule[:1], schedule[:13]]).astype(np.int64)
    receipt = coll.collect_confirmation(config, "smoke", schedule=duplicated,
                                        branch_rollout=make_fake_rollout(config),
                                        probe_fn=_fake_probe)
    assert receipt["accepted_groups"] == prot.SMOKE_ACCEPTED_GROUPS
    assert receipt["skip_counts"].get("duplicate_seed", 0) == 1
    inputs_path, _, ledger_path, _ = coll.role_paths(config, "smoke")
    with np.load(inputs_path) as z:
        seeds = np.asarray(z["group_seed"])
    assert len(np.unique(seeds)) == seeds.size
    # the ledger preserves every attempt seed, reason and cursor.
    records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    dup = [r for r in records if r["decision"] == "rejected"]
    assert len(dup) == 1 and dup[0]["reason"] == "duplicate_seed"
    assert dup[0]["group_seed"] == int(schedule[0])
    assert dup[0]["attempt_index"] == 1  # the candidate cursor of the rejected attempt
    assert [r["attempt_index"] for r in records] == list(range(len(records)))


def test_checkpoint_resume_identical_to_uninterrupted(tmp_path):
    config_a = _config(tmp_path / "a")
    config_b = _config(tmp_path / "b")
    receipt_a = _run_smoke(config_a)
    partial = coll.collect_confirmation(config_b, "smoke",
                                        branch_rollout=make_fake_rollout(config_b),
                                        probe_fn=_fake_probe, stop_after_accepted=5)
    assert partial["partial"] and partial["accepted_groups"] == 5
    inputs_b, truth_b, ledger_b, _ = coll.role_paths(config_b, "smoke")
    assert not inputs_b.exists() and not truth_b.exists()  # only the checkpoint exists
    receipt_b = coll.collect_confirmation(config_b, "smoke",
                                          branch_rollout=make_fake_rollout(config_b),
                                          probe_fn=_fake_probe)
    assert not receipt_b["partial"] and receipt_b["accepted_groups"] == 12
    inputs_a, truth_a, ledger_a, _ = coll.role_paths(config_a, "smoke")
    # schedule/RNG/cursor resume produced results identical to uninterrupted collection.
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
    assert (receipt_a["group_order_binding"]["inference_inputs_group_seed_sha256"]
            == receipt_b["group_order_binding"]["inference_inputs_group_seed_sha256"])


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
    # the ledger was not rewritten or appended to by the verifying rerun.
    assert again["attempts"] == receipt["attempts"]


def test_exhausted_candidate_domain_raises_collection_incomplete(tmp_path):
    config = _config(tmp_path)
    checked = prot.validate_config(config)
    schedule = prot.build_smoke_seed_schedule(checked)[:3]
    with pytest.raises(coll.CollectionIncompleteError) as excinfo:
        coll.collect_confirmation(config, "smoke", schedule=schedule,
                                  branch_rollout=make_fake_rollout(config),
                                  probe_fn=_fake_probe)
    assert excinfo.value.status == prot.STATUS_COLLECTION_INCOMPLETE
    assert prot.STATUS_COLLECTION_INCOMPLETE in str(excinfo.value)
    inputs_path, truth_path, ledger_path, receipt_path = coll.role_paths(config, "smoke")
    # nothing final is published and no receipt is written on the incomplete path.
    assert not inputs_path.exists() and not truth_path.exists()
    assert not receipt_path.exists()
    records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    for i, record in enumerate(records):
        assert record["attempt_index"] == i
        assert record["group_seed"] == int(schedule[i])
        assert record["decision"] == "accepted"


# --- confirm role: formal-lock ordering guard + full-quota two-file publish ---------------


def test_confirm_collection_requires_formal_protocol_lock(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(FileNotFoundError, match="protocol lock"):
        coll.collect_confirmation(config, "confirm",
                                  branch_rollout=make_fake_rollout(config),
                                  probe_fn=_fake_probe)


def test_confirm_collection_rejects_lock_drift(tmp_path):
    config = _config(tmp_path)
    lock_path = _write_minimal_lock(config)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["confirm_schedule_raw_sha256"] = "0" * 64  # drift toward a foreign domain
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    with pytest.raises(ValueError, match="schedule"):
        coll.collect_confirmation(config, "confirm",
                                  branch_rollout=make_fake_rollout(config),
                                  probe_fn=_fake_probe)


def test_confirm_collection_full_quota_two_file_publish(tmp_path):
    config = _config(tmp_path)
    config["checkpoint_every_accepted"] = 0  # keep the unit test fast; resume is covered above
    lock_path = _write_minimal_lock(config)
    lock_bytes = lock_path.read_bytes()
    checked = prot.validate_config(config)
    receipt = coll.collect_confirmation(config, "confirm",
                                        branch_rollout=make_fake_rollout(config),
                                        probe_fn=_fake_probe)
    assert receipt["status"] == "COLLECTED"
    assert not receipt["partial"]
    assert receipt["accepted_groups"] == prot.CONFIRM_ACCEPTED_GROUPS == 2000
    assert receipt["attempts"] == 2000  # the all-accept fake rejects nothing
    assert receipt["counterfactual_branches"] == 20000
    assert receipt["seed_schedule_sha256"] == prot.CONFIRM_SCHEDULE_RAW_SHA256
    assert receipt["action_stream_root"] == [2026093012, 0]
    inputs_path, truth_path, ledger_path, receipt_path = coll.role_paths(config, "confirm")
    assert inputs_path.is_file() and truth_path.is_file() and receipt_path.is_file()
    expected_seeds = prot.build_confirm_seed_schedule(checked)[:2000]
    with np.load(inputs_path) as z:
        assert set(z.files) == set(coll.LEGAL_INPUT_FIELDS)
        assert np.array_equal(np.asarray(z["group_seed"], dtype=np.int64), expected_seeds)
        for key in z.files:
            assert not any(tok in key.lower() for tok in TRUTH_KEY_TOKENS), key
    with np.load(truth_path) as z:
        assert z["object_counterfactual_futures"].shape == (2000, 10, 20, 5, 8)
        assert np.array_equal(np.asarray(z["group_seed"], dtype=np.int64), expected_seeds)
    records = ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(records) == 2000
    # collection never touched the formal lock, and never entered the smoke dirs.
    assert lock_path.read_bytes() == lock_bytes
    assert not (Path(config["data_root"]) / "smoke").exists()
    assert not (Path(config["run_dir"]) / "smoke").exists()
