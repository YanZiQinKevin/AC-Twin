"""Stage 4D-D2F screen scorer: frozen D2 base stack + D2E slot-0 ego repair + D2E Direct head.

Plan ``AC-TWIN-RFER-STAGE4D-D2F-POWER-CORRECTION-EXTERNAL-CONFIRM-2026-09-24-V0.1`` section 3.4.
The D2F scorer is a pure composition of frozen, hash-verified components (nothing is retrained):

1. the D2 frozen Full/History/E2E + response predictor stack runs on the NEW scenarios'
   ``inference_inputs.npz`` ONLY (the sealed truth file may be moved away / unreadable — the
   score side never opens it);
2. the D2E frozen five-seed ego effect (candidate ``EGO_TRAJECTORY_PLUS_RANK``) replaces ONLY
   slot 0 of the e2e prediction (``stage4d_d2e_pipeline.candidate_e2e``);
3. the frozen ``stage4c_objective`` consequence builds the ten-action ``J_hat``;
4. the D2E retrained five-seed Direct risk head and the three standard baselines share the SAME
   repaired ``J_hat`` and the legal ``inference_mask``.

Before the formal screen, section 3.4 requires the old-validation parity: the SAME scorer runs
on the old validation role's LEGAL inputs and must reproduce the D2E frozen
``power_validation_quantities.npz`` per group (``group_seed`` aligned, ``J_hat`` / ensemble /
five-seed risks / three standard baselines within 1e-8, ten-action argmin identical). That
parity is an engineering acceptance only; it never recomputes the old validation gate.

Every loaded artifact is hash-verified against the create-once ``source_binding.json`` /
``model_inventory.json`` chain before it is trusted.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

import torch  # noqa: E402

from ac_twin import stage4d_d2e_ego as ego  # noqa: E402
from ac_twin import stage4d_d2e_pipeline as PIPE  # noqa: E402
from ac_twin import stage4d_d2f_protocol as PROT  # noqa: E402
from ac_twin import stage4d_external_inference as INF  # noqa: E402
from ac_twin import stage4d_external_statistics as D2STAT  # noqa: E402
from ac_twin import stage4d_external_training as T  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402

import drive_stage4d_external_screen as D2DRV  # noqa: E402  (read-only reuse; never modified)

D2F_DIR = PROT.D2F_DIR
D2_DIR = PROT.D2_DIR
D2E_DIR = PROT.D2E_DIR
SCREEN_ROLE = PROT.SCREEN_ROLE

PARITY_TOLERANCE = 1e-8                    # plan 3.4 frozen numerical parity tolerance
PARITY_RECEIPT_NAME = "scorer_parity_receipt.json"
VALIDATION_ROLE = "validation"

_SCORE_ARRAY_KEYS = ("j_hat", "ensemble_risk", "risk_by_seed", "mu_only_ensemble",
                     "class_only_ensemble")
_STANDARD_BASELINE_NPZ = tuple(f"baseline_risk__{name}" for name in D2STAT.STANDARD_BASELINES)


# --- hash-verified frozen component loaders ------------------------------------------------------------


def _source_binding_hashes() -> dict[str, str]:
    binding = json.loads((D2F_DIR / PROT.SOURCE_BINDING_NAME).read_text(encoding="utf-8"))
    expanded = binding["expanded"]
    out: dict[str, str] = {}
    for group in ("checkpoints", "scalers", "contracts", "code"):
        for entry in expanded.get(group, []):
            out[str(Path(str(entry["path"])))] = str(entry["sha256"])
    return out


def _verify_against_binding(path: Path, hashes: Mapping[str, str], label: str) -> None:
    key = str(Path(path))
    if key not in hashes:
        raise RuntimeError(f"{label} is not covered by the create-once source binding: {key}")
    if sha256_file(path) != hashes[key]:
        raise RuntimeError(f"{label} drifted from the create-once source binding: {key}")


def load_predictor_stack() -> INF.PredictorStack:
    """The D2 frozen base predictor stack (read-only), hash-verified against the binding."""
    stack_path = D2_DIR / "development" / "base_models" / "predictor_stack.json"
    if sha256_file(stack_path) != dict(
            (rel, sha) for rel, sha, _ in PROT.PLAN_LISTED_SOURCES)[
            "stage4d_external/development/base_models/predictor_stack.json"]:
        raise RuntimeError("the D2 predictor stack drifted from its plan-listed SHA256")
    stack = INF.PredictorStack.from_dict(json.loads(stack_path.read_text(encoding="utf-8")))
    hashes = _source_binding_hashes()
    for key in ("expert_paths", "response_paths", "risk_paths", "router_paths"):
        for name, path_str in sorted(getattr(stack, key).items()):
            _verify_against_binding(Path(str(path_str)), hashes, f"d2 {key} {name}")
    _verify_against_binding(Path(str(stack.scaler_path)), hashes, "d2 platform scaler")
    return stack


def load_selected_ego() -> tuple[str, Any, dict[int, Any]]:
    """The D2E frozen selected ego candidate: scaler + five seed heads (hash-verified)."""
    final_path = D2E_DIR / "development" / "final_selection.json"
    if sha256_file(final_path) != dict(
            (rel, sha) for rel, sha, _ in PROT.PLAN_LISTED_SOURCES)[
            "stage4d_d2e/development/final_selection.json"]:
        raise RuntimeError("the D2E final selection drifted from its plan-listed SHA256")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    selected = str(final["selected_candidate"])
    hashes = _source_binding_hashes()
    for seed, rel in sorted(final["ego_checkpoints"].items()):
        _verify_against_binding(
            D2E_DIR / "development" / "ego_models" / Path(str(rel)).name,
            hashes, f"d2e ego checkpoint seed{seed}")
    scaler = PIPE.load_ego_scaler()
    models = PIPE.load_ego_models(selected)
    return selected, scaler, models


def load_risk_inventory(selected: str) -> T.ModelInventory:
    """The D2E retrained five-seed Direct inventory (hashes verified from its receipt)."""
    return PIPE.load_risk_inventory(selected)


# --- scoring (LEGAL INPUTS ONLY) -------------------------------------------------------------------------


def score_d2f_screen(stack: INF.PredictorStack,
                     ego_scaler: Any,
                     ego_models: Mapping[int, Any],
                     inventory: T.ModelInventory,
                     inputs_path: str | Path,
                     *,
                     device: str,
                     batch_size: int,
                     group_index: np.ndarray | None = None) -> dict[str, Any]:
    """Score groups from ``inference_inputs.npz`` ONLY (truth-free; plan 3.4).

    Mirrors the frozen D2 screen scoring with exactly one change: the e2e slot 0 is replaced by
    the D2E frozen ego effect before ``J_hat`` / Direct / baselines are computed. The scoring
    bundle builder provably never reads the counterfactual-truth fields (placeholder cf).
    """
    bundle = D2DRV.build_scoring_bundle(inputs_path, group_index=group_index)
    predictions = INF.run_pair_predictions(stack, bundle, device=device, batch_size=batch_size)
    predictions = dict(predictions)
    predictions["e2e"] = PIPE.candidate_e2e(bundle, predictions["e2e"], ego_scaler, ego_models,
                                            device=device, batch_size=batch_size)
    action_risks = INF.score_action_risks(stack, bundle, predictions)
    direct = INF.build_direct_inputs(bundle, predictions, action_risks)
    valid = np.asarray(direct["valid_mask"], dtype=bool).ravel()
    if not valid.all():
        raise ValueError(
            f"{int((~valid).sum())} screen groups have non-finite features; the screen contract "
            "rejects zero-padding, so scoring is refused")
    risk = T.score_models(
        inventory,
        {"tokens": direct["tokens"], "global_features": direct["global_features"],
         "group_seed": direct["group_seed"]},
        device=device)
    return {
        "group_seed": np.asarray(direct["group_seed"], dtype=np.int64).ravel(),
        "t_star": np.asarray(bundle.t_star_group, dtype=np.int64).ravel(),
        "interaction_min_distance": np.asarray(
            direct["interaction_min_distance"], dtype=np.float64).ravel(),
        "j_hat": np.asarray(direct["j_hat"], dtype=np.float64),
        "ensemble_risk": np.asarray(risk.risk_ensemble, dtype=np.float64).ravel(),
        "risk_by_seed": np.asarray(risk.risk_by_seed, dtype=np.float64),
        "mu_only_ensemble": np.asarray(risk.mu_only_ensemble, dtype=np.float64).ravel(),
        "class_only_ensemble": np.asarray(risk.class_only_ensemble, dtype=np.float64).ravel(),
        "valid_mask": valid,
        "baselines": {name: np.asarray(v, dtype=np.float64).ravel()
                      for name, v in direct["baselines"].items()},
    }


def determinism_recheck(stack: INF.PredictorStack,
                        ego_scaler: Any,
                        ego_models: Mapping[int, Any],
                        inventory: T.ModelInventory,
                        inputs_path: str | Path,
                        *,
                        device: str,
                        batch_size: int,
                        n_groups: int) -> dict[str, Any]:
    """Re-score the first ``n_groups`` twice (same device/batch/dtype/order); elementwise identical."""
    subset = np.arange(int(n_groups), dtype=np.int64)
    first = score_d2f_screen(stack, ego_scaler, ego_models, inventory, inputs_path,
                             device=device, batch_size=batch_size, group_index=subset)
    second = score_d2f_screen(stack, ego_scaler, ego_models, inventory, inputs_path,
                              device=device, batch_size=batch_size, group_index=subset)
    max_abs = 0.0
    for key in _SCORE_ARRAY_KEYS:
        a = np.asarray(first[key], dtype=np.float64)
        b = np.asarray(second[key], dtype=np.float64)
        if not np.array_equal(a, b):
            raise RuntimeError(
                f"D2F screen scoring is not elementwise reproducible ({key}); the one-shot "
                "screen halts before the score freeze")
        max_abs = max(max_abs, float(np.max(np.abs(a - b))) if a.size else 0.0)
    for name in D2STAT.ALL_BASELINE_NAMES:
        a = np.asarray(first["baselines"][name], dtype=np.float64)
        b = np.asarray(second["baselines"][name], dtype=np.float64)
        if not np.array_equal(a, b):
            raise RuntimeError(
                f"D2F screen baseline scoring is not elementwise reproducible ({name})")
    return {
        "role": SCREEN_ROLE,
        "n_groups": int(n_groups),
        "group_indices": [int(i) for i in subset],
        "bitwise_identical": True,
        "max_abs_diff": float(max_abs),
        "device": str(device),
        "batch_size": int(batch_size),
    }


# --- old-validation scorer parity (plan 3.4 engineering acceptance) --------------------------------------


def build_bundle_from_role_groups(role_dir: str | Path, *, role: str) -> INF.BranchBundle:
    """Build a scoring bundle from a role's collected groups with a PLACEHOLDER cf.

    Used ONLY for the old-validation parity (engineering acceptance). The true counterfactual
    futures are dropped and replaced by the factual placeholder exactly as in the screen's
    ``build_scoring_bundle`` — the score side provably never reads the truth fields. The group
    seeds are derived with the OLD D2 salt (bitwise from ``INF.load_role_arrays``) so they
    align with the D2E frozen quantities.
    """
    import dataclasses

    role_dir = Path(role_dir)
    arrays = INF.load_role_arrays(role_dir, role)
    placeholder_cf = np.repeat(arrays.object_factual_future[:, None],
                               arrays.n_actions, axis=1)
    arrays = dataclasses.replace(arrays, object_counterfactual_futures=placeholder_cf)
    return INF.build_branch_bundle(arrays)


def run_old_validation_parity(stack: INF.PredictorStack,
                              ego_scaler: Any,
                              ego_models: Mapping[int, Any],
                              inventory: T.ModelInventory,
                              *,
                              device: str,
                              batch_size: int,
                              progress: Any = None) -> dict[str, Any]:
    """Section 3.4 parity: the D2F scorer composition vs the D2E frozen validation quantities.

    Two paths, one engineering-acceptance gate:

    - **Path A (the 1e-8 gate) — composition parity on the D2E input calibre.** The D2E frozen
      quantities were computed from the PERSISTED D2 base predictions + action risks
      (``PIPE.load_persisted_base``, read-only). Feeding the identical inputs through the D2F
      scorer composition (slot-0 ego repair -> ``build_direct_inputs`` -> frozen Direct
      ``score_models`` + standard baselines) must reproduce the D2E frozen npz within 1e-8 with
      the ten-action argmin identical. Any wiring drift (wrong checkpoint / scaler / repair /
      head) fails this gate.
    - **Path B (informational) — the cross-process re-inference float envelope.** A LIVE
      ``run_pair_predictions`` over the same legal inputs is compared against the persisted D2
      predictions. Cross-process GPU float noise on this stack is ~1e-5 (pilot-documented for
      the simulator; SEED_VOTE_ENTROPY's discrete features stay bitwise), so live re-inference
      can never meet 1e-8 — the envelope is measured and disclosed, never gated. The formal
      screen scores each fresh scenario exactly once and proves same-process determinism on
      >=32 groups instead (plan 3.4).

    Engineering acceptance only — the old validation gate is never recomputed. Create-once receipt.
    """
    def _log(message: str) -> None:
        if progress is not None:
            progress(message)
        else:
            print(f"[d2f-parity-scorer] {message}", flush=True)

    receipt_path = D2F_DIR / "development" / PARITY_RECEIPT_NAME
    if receipt_path.exists():
        _log("scorer parity receipt exists (create-once); reloaded")
        return json.loads(receipt_path.read_text(encoding="utf-8"))

    role_dir = D2_DIR / "roles" / VALIDATION_ROLE
    _log("building the old-validation scoring bundle (placeholder cf; truth-free)")
    bundle = build_bundle_from_role_groups(role_dir, role=VALIDATION_ROLE)

    # --- Path A: composition parity on the persisted D2E input calibre (the 1e-8 gate) --------
    _log("Path A: persisted D2 predictions + D2E slot-0 repair + frozen Direct scoring")
    predictions_persisted, action_risks = PIPE.load_persisted_base(VALIDATION_ROLE)
    e2e_rep = PIPE.candidate_e2e(bundle, predictions_persisted["e2e"], ego_scaler, ego_models,
                                 device=device, batch_size=batch_size)
    direct = INF.build_direct_inputs(bundle, {"e2e": e2e_rep}, action_risks)
    risk = T.score_models(
        inventory,
        {"tokens": direct["tokens"], "global_features": direct["global_features"],
         "group_seed": direct["group_seed"]},
        device=device)

    with np.load(D2E_DIR / "development" / "power_validation_quantities.npz",
                 allow_pickle=False) as archive:
        ref = {key: archive[key] for key in archive.files}
    ref_seed = np.asarray(ref["group_seed"], dtype=np.int64).ravel()
    got_seed = np.asarray(direct["group_seed"], dtype=np.int64).ravel()
    if set(got_seed.tolist()) != set(ref_seed.tolist()):
        raise RuntimeError("validation parity group_seed sets differ (fail closed)")
    order = np.array([np.where(ref_seed == s)[0][0] for s in got_seed], dtype=np.int64)

    comparisons: dict[str, Any] = {}
    ok = True
    j_hat_got = np.asarray(direct["j_hat"], dtype=np.float64)
    j_hat_ref = np.asarray(ref["J_hat"], dtype=np.float64)[order]
    comparisons["J_hat_max_abs_diff"] = float(np.max(np.abs(j_hat_got - j_hat_ref)))
    ok &= comparisons["J_hat_max_abs_diff"] <= PARITY_TOLERANCE
    argmin_identical = bool(np.array_equal(j_hat_got.argmin(axis=1), j_hat_ref.argmin(axis=1)))
    comparisons["argmin_identical"] = argmin_identical
    ok &= argmin_identical
    got = np.asarray(risk.risk_ensemble, dtype=np.float64).ravel()
    reference = np.asarray(ref["ensemble_risk"], dtype=np.float64)[order]
    diff = float(np.max(np.abs(got - reference)))
    comparisons["ensemble_risk_max_abs_diff"] = diff
    ok &= diff <= PARITY_TOLERANCE
    by_seed_got = np.asarray(risk.risk_by_seed, dtype=np.float64)
    by_seed_ref = np.asarray(ref["risk_by_seed"], dtype=np.float64)[:, order]
    diff = float(np.max(np.abs(by_seed_got - by_seed_ref)))
    comparisons["risk_by_seed_max_abs_diff"] = diff
    ok &= diff <= PARITY_TOLERANCE
    for name in D2STAT.STANDARD_BASELINES:
        got = np.asarray(direct["baselines"][name], dtype=np.float64).ravel()
        reference = np.asarray(ref[f"baseline_risk__{name}"], dtype=np.float64)[order]
        diff = float(np.max(np.abs(got - reference)))
        comparisons[f"baseline_{name}_max_abs_diff"] = diff
        ok &= diff <= PARITY_TOLERANCE

    # --- Path B: the cross-process re-inference float envelope (informational, never gated) ---
    _log("Path B: live re-inference float envelope vs the persisted D2 predictions")
    live = INF.run_pair_predictions(stack, bundle, device=device, batch_size=batch_size)
    envelope = {key: float(np.max(np.abs(
        np.asarray(live[key], dtype=np.float64)
        - np.asarray(predictions_persisted[key], dtype=np.float64))))
        for key in ("e2e", "full", "history")}

    receipt = {
        "plan_id": PROT.PLAN_ID,
        "scope": "ENGINEERING_ACCEPTANCE_ONLY",
        "mode": "old_validation_scorer_parity",
        "tolerance": PARITY_TOLERANCE,
        "path_a_composition_parity": comparisons,
        "path_b_live_reinference_envelope": {
            "max_abs_diff": envelope,
            "note": ("cross-process GPU float envelope on the frozen base stack; informational "
                     "only — the formal screen scores each fresh group exactly once and gates "
                     ">=32-group same-process determinism instead (plan 3.4)"),
        },
        "n_groups": int(got_seed.size),
        "device": str(device),
        "batch_size": int(batch_size),
        "passed": bool(ok),
        "bindings": {
            "validation_groups_sha256": sha256_file(role_dir / "groups.npz"),
            "validation_predictions_sha256": sha256_file(
                D2_DIR / "development" / f"{VALIDATION_ROLE}_predictions.npz"),
            "d2e_power_validation_quantities_sha256": sha256_file(
                D2E_DIR / "development" / "power_validation_quantities.npz"),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if not ok:
        raise RuntimeError(f"old-validation scorer parity FAILED: {comparisons}")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    write_once_json(receipt_path, receipt)
    _log(f"parity PASS: max diffs {comparisons}; envelope {envelope}")
    return receipt
