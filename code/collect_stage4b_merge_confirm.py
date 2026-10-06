"""Stage 4B Merge risk confirmation collector (``stage4_merge_risk_confirm``).

Replicates the frozen Stage 2 / Stage 4A Merge paired-world generation semantics
(prefix consistency, complete-horizon and identity-replay acceptance) with the
Stage 4B-specific changes (plan section 4):

- single confirmation role: ALL accepted episodes enter one confirm set; there is NO
  train/validation/test split, so no ``assign_stratum_roles`` and no ``split`` column;
- exact ten-stratum balancing: ``selected_n / 10`` accepted episodes per
  ``(cf_action, cf_duration)`` cell, where ``selected_n`` is read from the locked
  protocol (chosen mechanically by the power analysis — never set by hand here);
- a 12,000-seed candidate schedule (entropy ``2026091701``) consumed in frozen order;
  if it exhausts before all strata fill the stage terminates
  ``STAGE4B_COLLECTION_INCOMPLETE`` (a registered hard stop — never silently top up).

Accept/reject uses ONLY pre-intervention-knowable stratum quota and simulation
validity; it never reads the realized response, effect error, which expert is better,
a collision label or any risk score. Rollout-injectable so tests need no highway_env.
"""
from __future__ import annotations

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

import collect_stage2_holdout as engine  # noqa: E402
from ac_twin import stage2_protocol as sp  # noqa: E402
from ac_twin import stage4b_protocol as s4bp  # noqa: E402
from ac_twin.checkpointing import (  # noqa: E402
    canonical_fingerprint,
    load_collection_checkpoint,
    save_collection_checkpoint,
)
from ac_twin.progress import ProgressReporter  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402

HISTORY_LEN = engine.HISTORY_LEN
FUTURE_LEN = engine.FUTURE_LEN
N_OBJECT_FEATURES = engine.N_OBJECT_FEATURES

_STACKED_FIELDS = engine._STACKED_FIELDS
_SCALAR_INT_FIELDS = engine._SCALAR_INT_FIELDS
_SCALAR_BOOL_FIELDS = engine._SCALAR_BOOL_FIELDS
_JSON_FIELDS = engine._JSON_FIELDS

_REASONS = tuple(engine._REASONS) + ("stratum_quota_full",)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _empty_state() -> dict[str, Any]:
    state: dict[str, Any] = {field: [] for field in _STACKED_FIELDS}
    state.update({field: [] for field in _SCALAR_INT_FIELDS})
    state.update({field: [] for field in _SCALAR_BOOL_FIELDS})
    state.update({field: [] for field in _JSON_FIELDS})
    state["accepted_strata"] = Counter()
    state["skip_counts"] = Counter({reason: 0 for reason in _REASONS})
    state["attempts"] = 0
    state["prefix_max_abs_diff"] = 0.0
    state["identity_max_abs_diff"] = 0.0
    return state


_pack_state = engine._pack_state
_restore_state = engine._restore_state


def _all_strata(config: Mapping[str, Any]) -> list[tuple[int, int]]:
    return [
        (int(a), int(d))
        for a in config["cf_actions"]
        for d in config["intervention_durations"]
    ]


def _select_stratum(state, eligible, stratum_rng, quota):
    """Pick the least-filled eligible stratum under its quota; None if all full."""
    under = [key for key in eligible if state["accepted_strata"][key] < quota]
    if not under:
        return None
    min_count = min(state["accepted_strata"][key] for key in under)
    least = [key for key in under if state["accepted_strata"][key] == min_count]
    return least[int(stratum_rng.integers(0, len(least)))]


def _all_full(state, strata, quota) -> bool:
    return all(state["accepted_strata"][key] >= quota for key in strata)


def collect_stage4b(
    config: Mapping[str, Any],
    *,
    resume: bool = False,
    branch_rollout: Callable | None = None,
    probe_fn: Callable | None = None,
    stop_after_accepted: int | None = None,
) -> dict[str, Any]:
    """Collect the Stage 4B Merge risk confirmation set; returns the receipt.

    Requires a verified Stage 4B protocol lock (which carries ``selected_n``). The
    final NPZ is the completion marker; a temporary NPZ + receipt are validated
    before the atomic publish. ``stop_after_accepted`` is a test hook.
    """
    checked = s4bp.validate_stage4b_config(config)
    run_dir = _resolve(checked["run_dir"])
    lock = s4bp.verify_protocol_lock(run_dir)  # raises on missing/drift

    schedule_path = run_dir / "protocol" / "candidate_seed_schedule.npy"
    schedule = np.load(schedule_path).astype(np.int64)
    if hashlib.sha256(schedule.tobytes()).hexdigest() != lock["seed_schedule_sha256"]:
        raise RuntimeError("Stage 4B candidate seed schedule bytes changed since lock")
    if schedule.shape[0] < int(checked["max_attempts"]):
        raise RuntimeError("Stage 4B seed schedule smaller than max_attempts")

    branch_rollout = branch_rollout or engine.rollout_branch_real
    probe_fn = probe_fn or engine._real_probe
    action_ids, action_probs, capacity = probe_fn(config)
    action_ids = np.asarray(action_ids, dtype=np.int64)
    action_probs = np.asarray(action_probs, dtype=np.float64)

    n_pairs = int(lock["selected_n"])
    quota = int(lock["per_stratum"])
    horizon = int(checked["horizon"])
    history_len = int(checked["history_len"])
    max_attempts = int(checked["max_attempts"])
    determinism_checks = int(checked["determinism_checks"])
    dt = 1.0 / float(checked["policy_frequency"])

    data_output = _resolve(checked["data_output"])
    ledger_path = _resolve(checked["rejection_ledger"])
    receipt_path = _resolve(checked["collection_receipt"])
    partial_path = data_output.with_suffix(".partial.npz")

    _, action_rng, stratum_rng = sp._spawn_streams(int(checked["collection_seed"]))
    all_strata = _all_strata(checked)

    state = _empty_state()
    resuming = False
    if data_output.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["verification"] = "existing_final_dataset_verified_no_overwrite"
        return receipt
    if partial_path.exists():
        rng_state, metadata, arrays = load_collection_checkpoint(partial_path, dict(config))
        state = _restore_state(arrays, metadata)
        action_rng.bit_generator.state = rng_state["action_rng"]
        stratum_rng.bit_generator.state = rng_state["stratum_rng"]
        resuming = True

    data_output.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_handle = open(ledger_path, "a" if resuming else "w", encoding="utf-8")

    checkpoint_every = int(checked.get("checkpoint_every_accepted", 50))
    attempts = int(state["attempts"])
    accepted = len(state["object_history"])
    reporter = ProgressReporter(
        total=n_pairs, every=checkpoint_every, prefix="stage4b-collect", heartbeat_seconds=30.0
    )

    def write_ledger(record: dict[str, Any]) -> None:
        ledger_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        ledger_handle.flush()

    def checkpoint() -> None:
        metadata, arrays = _pack_state(state)
        save_collection_checkpoint(
            partial_path,
            dict(config),
            {"action_rng": action_rng.bit_generator.state,
             "stratum_rng": stratum_rng.bit_generator.state},
            metadata,
            arrays,
        )

    try:
        while not _all_full(state, all_strata, quota) and attempts < max_attempts:
            episode_seed = int(schedule[attempts])
            attempt_index = attempts
            attempts += 1
            state["attempts"] = attempts

            t_star = int(action_rng.integers(int(checked["t_star_low"]),
                                             int(checked["t_star_high"]) + 1))
            prefix_actions = action_rng.choice(action_ids, size=t_star, p=action_probs).astype(np.int64)
            factual_plan = action_rng.choice(action_ids, size=horizon, p=action_probs).astype(np.int64)

            eligible = [key for key in all_strata if key[0] != int(factual_plan[0])]
            chosen = _select_stratum(state, eligible, stratum_rng, quota)
            if chosen is None:
                state["skip_counts"]["stratum_quota_full"] += 1
                write_ledger({
                    "attempt_index": int(attempt_index), "episode_seed": int(episode_seed),
                    "decision": "rejected", "reason": "stratum_quota_full",
                })
                continue
            cf_action, cf_duration = chosen
            cf_duration = min(cf_duration, horizon)
            counterfactual_plan = factual_plan.copy()
            counterfactual_plan[:cf_duration] = int(cf_action)

            f = branch_rollout(config["env_id"], engine._env_config(config), episode_seed,
                               prefix_actions, factual_plan, capacity)
            c = branch_rollout(config["env_id"], engine._env_config(config), episode_seed,
                               prefix_actions, counterfactual_plan, capacity)

            def ledger_base(decision: str, reason: str) -> dict[str, Any]:
                return {
                    "attempt_index": int(attempt_index), "episode_seed": int(episode_seed),
                    "decision": decision, "reason": reason,
                    "cf_action": int(cf_action), "cf_duration": int(cf_duration),
                    "t_star": int(t_star),
                    "factual_terminated": bool(f["terminated"]),
                    "factual_truncated": bool(f["truncated"]),
                    "counterfactual_terminated": bool(c["terminated"]),
                    "counterfactual_truncated": bool(c["truncated"]),
                    "factual_first_contact_step": int(f["first_contact_step"]),
                    "counterfactual_first_contact_step": int(c["first_contact_step"]),
                }

            reason = None
            if (f["terminated"] or f["truncated"] or c["terminated"] or c["truncated"]
                    or len(f["prefix_frames"]) != t_star + 1
                    or len(c["prefix_frames"]) != t_star + 1):
                reason = "prefix_terminated"
            else:
                prefix_diff = float(np.max(np.abs(f["prefix_frames"] - c["prefix_frames"])))
                state["prefix_max_abs_diff"] = max(state["prefix_max_abs_diff"], prefix_diff)
                if prefix_diff > 1e-6:
                    reason = "object_prefix_mismatch"
            if reason is None and len(f["future_frames"]) != horizon:
                reason = "factual_future_short"
            if reason is None and len(c["future_frames"]) != horizon:
                reason = "counterfactual_future_short"
            if reason is None and not (
                    np.isfinite(f["future_frames"]).all()
                    and np.isfinite(c["future_frames"]).all()):
                reason = "non_finite"
            if reason is None and not engine._fingerprints_consistent(
                    f["fingerprints"], c["fingerprints"]):
                reason = "initial_fingerprint_mismatch"
            history = f["prefix_frames"][-history_len:]
            if reason is None and history.shape[0] != history_len:
                reason = "short_history"

            if reason is not None:
                state["skip_counts"][reason] += 1
                write_ledger(ledger_base("rejected", reason))
                continue

            if accepted < determinism_checks:
                check = branch_rollout(config["env_id"], engine._env_config(config),
                                       episode_seed, prefix_actions, factual_plan, capacity)
                if (check["prefix_frames"].shape != f["prefix_frames"].shape
                        or check["future_frames"].shape != f["future_frames"].shape):
                    raise RuntimeError("identity replay changed trajectory shape")
                identity_diff = max(
                    float(np.max(np.abs(check["prefix_frames"] - f["prefix_frames"]))),
                    float(np.max(np.abs(check["future_frames"] - f["future_frames"]))),
                )
                state["identity_max_abs_diff"] = max(state["identity_max_abs_diff"], identity_diff)
                if identity_diff > 1e-6:
                    raise RuntimeError(
                        f"identity replay failed determinism check: {identity_diff}")

            state["object_history"].append(history.astype(np.float32))
            state["object_factual_future"].append(f["future_frames"].astype(np.float32))
            state["object_counterfactual_future"].append(c["future_frames"].astype(np.float32))
            state["factual_actions"].append(factual_plan.astype(np.int64))
            state["counterfactual_actions"].append(counterfactual_plan.astype(np.int64))
            state["cf_action"].append(int(cf_action))
            state["cf_duration"].append(int(cf_duration))
            state["episode_seed"].append(int(episode_seed))
            state["t_star"].append(int(t_star))

            for prefix, branch in (("factual", f), ("counterfactual", c)):
                hist_diag = {k: v[-history_len:] for k, v in branch["diag_prefix"].items()}
                fut_diag = branch["diag_future"]
                lane = np.concatenate([hist_diag["lane_index"], fut_diag["lane_index"]])
                target_lane = np.concatenate([hist_diag["target_lane_index"], fut_diag["target_lane_index"]])
                cmd_accel = np.concatenate([hist_diag["commanded_accel"], fut_diag["commanded_accel"]])
                cmd_steer = np.concatenate([hist_diag["commanded_steering"], fut_diag["commanded_steering"]])
                crashed = np.concatenate([hist_diag["crashed"], fut_diag["crashed"]])
                state[f"{prefix}_lane_index"].append(lane.astype(np.int32))
                state[f"{prefix}_target_lane_index"].append(target_lane.astype(np.int32))
                state[f"{prefix}_cmd_accel"].append(cmd_accel.astype(np.float32))
                state[f"{prefix}_cmd_steer"].append(cmd_steer.astype(np.float32))
                state[f"{prefix}_crashed"].append(crashed.astype(bool))
                state[f"{prefix}_terminated"].append(bool(branch["terminated"]))
                state[f"{prefix}_truncated"].append(bool(branch["truncated"]))
                state[f"{prefix}_first_contact_step"].append(int(branch["first_contact_step"]))
                state[f"{prefix}_termination_reason"].append(str(branch["termination_reason"]))
                state[f"{prefix}_fingerprints_json"].append(
                    json.dumps(branch["fingerprints"], ensure_ascii=False))

            state["initial_fingerprint_consistent"].append(True)
            state["vehicle_metadata_json"].append(json.dumps(f["static_metadata"], ensure_ascii=False))
            state["accepted_strata"][(int(cf_action), int(cf_duration))] += 1
            accepted += 1
            write_ledger(ledger_base("accepted", ""))
            reporter.update(accepted, attempted=attempts)
            reporter.heartbeat(accepted, attempted=attempts)

            if stop_after_accepted is not None and accepted >= stop_after_accepted:
                checkpoint()
                ledger_handle.close()
                return _receipt(checked, state, attempts, accepted, quota, partial=True, lock=lock)
            if checkpoint_every > 0 and accepted % checkpoint_every == 0:
                checkpoint()
    finally:
        if not ledger_handle.closed:
            ledger_handle.close()

    if not _all_full(state, all_strata, quota):
        receipt = _receipt(checked, state, attempts, accepted, quota, partial=True, lock=lock)
        receipt["status"] = s4bp.STATUS_COLLECTION_INCOMPLETE
        receipt["incomplete_strata"] = {
            f"action_{a}_duration_{d}": int(quota - state["accepted_strata"][(a, d)])
            for (a, d) in all_strata if state["accepted_strata"][(a, d)] < quota
        }
        raise CollectionIncompleteError(json.dumps(receipt, indent=2, ensure_ascii=False))

    checkpoint()
    _publish(checked, state, schedule, lock, capacity, dt, quota)
    partial_path.unlink(missing_ok=True)
    return _receipt(checked, state, attempts, accepted, quota, partial=False, lock=lock)


class CollectionIncompleteError(RuntimeError):
    """The 12,000-seed schedule exhausted before all ten strata filled (hard stop)."""


# --- publish + receipt -------------------------------------------------------------


def _build_payload(config, state, capacity, dt) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for field in _STACKED_FIELDS:
        payload[field] = np.stack(state[field])
    for field in _SCALAR_INT_FIELDS:
        payload[field] = np.asarray(state[field], dtype=np.int64)
    for field in _SCALAR_BOOL_FIELDS:
        payload[field] = np.asarray(state[field], dtype=bool)
    for prefix in ("factual", "counterfactual"):
        for channel in ("cmd_accel", "cmd_steer"):
            values = payload[f"{prefix}_{channel}"]
            payload[f"{prefix}_{channel}_mask"] = np.isfinite(values)
    payload.update(engine._derive_accel_fields(state, dt))
    # Stage 4B has a SINGLE confirm role: no split column is written. The ten-stratum
    # balance (selected_n/10 per cell) is the only structural invariant.
    payload["object_capacity"] = np.asarray(capacity, dtype=np.int64)
    payload["object_features_json"] = np.asarray(
        json.dumps(["presence", "is_ego", "x", "y", "vx", "vy",
                    "cos_heading", "sin_heading"]), dtype=object)
    for field in _JSON_FIELDS:
        payload[field] = np.asarray([str(v) for v in state[field]], dtype=object)
    return payload


def _publish(config, state, schedule, lock, capacity, dt, quota) -> None:
    data_output = _resolve(config["data_output"])
    receipt_path = _resolve(config["collection_receipt"])
    accepted = len(state["object_history"])
    payload = _build_payload(config, state, capacity, dt)

    receipt = _receipt(config, state, int(state["attempts"]), accepted, quota,
                       partial=False, lock=lock)
    receipt["data_output_sha256_pending"] = True

    tmp_npz = data_output.with_name(data_output.name + ".tmp")
    tmp_receipt = receipt_path.with_name(receipt_path.name + ".tmp")
    with open(tmp_npz, "wb") as handle:
        np.savez_compressed(handle, **payload)
    with np.load(tmp_npz, allow_pickle=True) as z:
        n = accepted
        assert z["object_history"].shape == (n, HISTORY_LEN, capacity, N_OBJECT_FEATURES)
        assert z["object_factual_future"].shape == (n, FUTURE_LEN, capacity, N_OBJECT_FEATURES)
        assert z["object_counterfactual_future"].shape == (n, FUTURE_LEN, capacity, N_OBJECT_FEATURES)
        assert len(np.unique(z["episode_seed"])) == n
        for field in ("object_history", "object_factual_future", "object_counterfactual_future"):
            if not np.isfinite(z[field]).all():
                raise RuntimeError(f"final NPZ field {field} is non-finite")
        # Ten-stratum balance must hold exactly (selected_n/10 per cell).
        s4bp.assert_balanced_strata(
            np.asarray(z["cf_action"]), np.asarray(z["cf_duration"]), quota
        )
    receipt["data_output_sha256"] = sha256_file(tmp_npz)
    receipt.pop("data_output_sha256_pending", None)
    tmp_receipt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(tmp_receipt).replace(receipt_path)
    Path(tmp_npz).replace(data_output)


def _receipt(config, state, attempts, accepted, quota, *, partial, lock) -> dict[str, Any]:
    return {
        "experiment_name": config["experiment_name"],
        "analysis_status": config["analysis_status"],
        "purpose": s4bp.PURPOSE,
        "plan_id": config["plan_id"],
        "partial": bool(partial),
        "requested_pairs": int(lock["selected_n"]),
        "accepted_pairs": int(accepted),
        "attempts": int(attempts),
        "collection_seed": int(config["collection_seed"]),
        "seed_entropy": s4bp.SEED_ENTROPY,
        "strata_per_cell": int(quota),
        "selected_n": int(lock["selected_n"]),
        "skip_counts": {str(k): int(v) for k, v in state["skip_counts"].items()},
        "accepted_strata": {f"action_{a}_duration_{d}": int(c)
                            for (a, d), c in sorted(state["accepted_strata"].items())},
        "prefix_max_abs_diff": float(state["prefix_max_abs_diff"]),
        "identity_max_abs_diff": float(state["identity_max_abs_diff"]),
        "identity_replay_checks": int(config["determinism_checks"]),
        "protocol_lock_seed_schedule_sha256": lock["seed_schedule_sha256"],
        "environment": sp._environment_versions(),
        "config_signature": canonical_fingerprint(json.loads(json.dumps(config))),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect Stage 4B Merge risk confirm set")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    try:
        receipt = collect_stage4b(config, resume=bool(args.resume))
    except CollectionIncompleteError as exc:
        print(f"[stage4b-collect] {s4bp.STATUS_COLLECTION_INCOMPLETE}: {exc}", flush=True)
        return 4
    print(
        f"[stage4b-collect] accepted={receipt.get('accepted_pairs')} "
        f"attempts={receipt.get('attempts')} partial={receipt.get('partial', False)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
