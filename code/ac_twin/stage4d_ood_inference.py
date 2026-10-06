"""Stage 4D D1 frozen inference + scoring adapter: truth-isolated, cinf-reuse, thin.

Plan ``AC-TWIN-RFER-STAGE4D-D0D1-OOD-2026-09-20-V1.0`` §3.2, §6.4, §10 Task 5 and §13
(controller ruling, Task 4 review Important #1). The score side reads ONLY the published
legal-input file (``collection/inference_inputs.npz``); the sealed
``confirmation_truth.npz`` is NEVER opened — the collector's truth-isolated loader refuses
the truth path by name and screens every archive key fail-closed, and scoring completes
with the truth file physically moved away (Global Constraint 6). ``confirmation_used.lock``
must NOT exist at any score-side step (it is Task 6's one-shot unseal); its presence
refuses inference, the determinism recheck and the freeze.

Reuse policy (read-only, NEVER re-implemented here): the score math is the frozen Stage
4C-R2 chain — ``stage4cr2_confirm_inference`` ("cinf") supplies the legal-input bundle
expansion, the frozen Direct model inventory loader, the prediction validator, the
group-level CONSTANT_RANDOM helper and the row-count-context alignment helpers;
``stage4c_inference`` / ``stage4cp_features`` / ``stage4cr2_training`` supply the frozen
predictor / feature / scoring functions; ``stage4cr2_confirm_statistics`` supplies the
Direct row-context alignment. cinf's role/path/config contract is bound to the R2-confirm
namespace (its ``prot.assert_allowed_confirm_data`` guard and its frozen action-stream
roots), so cinf's monolithic ``infer_and_score`` CANNOT be called for this stage — this
module is the thin adapter the controller ruling mandates: it reuses cinf's building
blocks verbatim and re-binds ONLY the stage-specific orchestration (paths, the D1
action-stream root [2026110994, 4] for the action-level CONSTANT_RANDOM stream, receipts).

The frozen engine config is the stage4cr2 confirm config, loaded read-only and bound to
the stage4cr2 lock chain: the stage4cr2 recovery lock file is hash-pinned by this stage's
§2.1 source table, the chain resolves cinf's frozen hash, and the config signature must
match the stage4cr2 protocol lock. No ``fit``, no optimizer, no calibration, no training
or selection entrypoint of any old stage is called; every model stays in ``eval()``.

Determinism (plan §6.4 step 2): the fixed leading >=32 groups are re-run twice under the
SAME device/batch/dtype/call order; the three row-count-context-dependent forwards (the
unbatched e2e router mixture, the single-CPU-forward COMBINED MLP, the single-forward
Direct head) are recomputed in the formal forward context via zero-padding (GEMM/MLP row
independence — exactly the stage4cr2 recovery-lock-r1 repair pattern), so the comparison
is a true bitwise one: the two runs must be elementwise identical AND equal the formal
artifacts' leading slices. No tolerance is ever substituted for identity.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

import collect_stage4d_merge_ood as collector  # this stage's truth-isolated loader
from ac_twin import stage4cr2_confirm_inference as cinf  # frozen R2-confirm scoring, read-only
from ac_twin import stage4cr2_confirm_protocol as r2confirm  # frozen chain, read-only
from ac_twin import stage4cr2_confirm_statistics as r2stats  # frozen, read-only
from ac_twin import stage4c_inference as s4ci  # frozen predictor chain, read-only
from ac_twin import stage4cp_features as s4cpf  # frozen features, read-only
from ac_twin import stage4cp_protocol as s4cpp  # RISK_CLASSES, read-only
from ac_twin import stage4cr2_training as s4cr2t  # frozen Direct scoring, read-only
from ac_twin import stage4d_confirm_protocol as confirm_prot
from ac_twin import stage4d_protocol as prot
from ac_twin.checkpointing import atomic_save_npz
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

ROOT = prot.ROOT
CONFIG_PATH = prot.CONFIG_PATH

PLAN_ID = prot.PLAN_ID
STATUS_INFERENCE = "INFERENCE_COMPLETE"

#: the only role this module serves: the formal D1 confirm collection.
D1_ROLE = "d1_confirm"

#: the plan-fixed determinism recheck width (plan §6.4: at least 32 groups).
DETERMINISM_GROUPS = 32

#: the frozen CONSTANT_RANDOM salt (Stage 4C-R2 confirm contract), read-only.
CONSTANT_RANDOM_SALT = r2confirm.CONSTANT_RANDOM_SALT

#: exactly the legal input keys of the published inference_inputs.npz (== cinf's contract).
LEGAL_INPUT_KEYS = cinf.LEGAL_INPUT_KEYS

#: cinf's prediction-side arrays persisted in predictions.npz (plus group binding ids).
PREDICTION_KEYS = cinf.PREDICTION_KEYS


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load + validate the frozen Stage 4D config."""
    return prot.validate_config(_read_json(Path(config_path)))


# --- frozen engine binding (the stage4cr2 confirm chain, reused read-only) --------------------


def load_engine_config(engine_config_path: str | Path | None = None) -> dict[str, Any]:
    """The frozen stage4cr2 confirm engine config, validated and lock-chain-bound.

    Read-only: the stage4cr2 recovery lock file must equal this stage's pinned §2.1
    source hash; it binds the stage4cr2 protocol lock, whose embedded config signature
    must match the validated engine config; and the chain-resolved code set must bind the
    CURRENT on-disk hashes of cinf / the confirm protocol / the confirm statistics. Any
    drift fails closed before any scoring.
    """
    engine_path = Path(engine_config_path or r2confirm.CONFIG_PATH)
    engine = r2confirm.validate_config(_read_json(engine_path))
    recovery_path = _resolve(confirm_prot.STAGE4CR2_RECOVERY_LOCK)
    if not recovery_path.is_file():
        raise prot.PreflightError(f"stage4cr2 recovery lock missing: {recovery_path}")
    pinned = prot.STAGE4D_PINNED_SOURCES[confirm_prot.STAGE4CR2_RECOVERY_LOCK]
    if sha256_file(recovery_path) != pinned:
        raise prot.PreflightError(
            "stage4cr2 recovery lock drifted from this stage's pinned source hash")
    lock_path = _resolve(confirm_prot.STAGE4CR2_PROTOCOL_LOCK)
    recovery = _read_json(recovery_path)
    if str(recovery.get("binds_previous_lock_sha256", "")) != sha256_file(lock_path):
        raise prot.PreflightError(
            "stage4cr2 recovery lock does not bind the stage4cr2 protocol lock")
    lock = _read_json(lock_path)
    if lock.get("config_signature") != r2confirm._config_signature(engine):
        raise prot.PreflightError(
            "the frozen stage4cr2 engine config drifted from its protocol lock")
    chain = r2confirm.resolve_locked_code_hashes(lock_path)
    for rel in ("ac_twin/stage4cr2_confirm_inference.py",
                "ac_twin/stage4cr2_confirm_protocol.py",
                "ac_twin/stage4cr2_confirm_statistics.py"):
        if chain.get(rel) != sha256_file(ROOT / "code" / rel):
            raise prot.PreflightError(
                f"the reused frozen scoring chain drifted from the stage4cr2 lock "
                f"chain: {rel}")
    return engine


# --- path contract ---------------------------------------------------------------------------


def inference_artifact_paths(checked: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve every D1 score-side path from the frozen Stage 4D config."""
    paths = checked["paths"]
    predictions = _resolve(paths["inference_predictions"])
    return {
        "inputs": _resolve(paths["inference_inputs"]),
        "truth": _resolve(paths["confirmation_truth"]),  # NEVER opened; refusal anchor
        "collection_receipt": _resolve(paths["collection_receipt"]),
        "predictions": predictions,
        "risk_scores": _resolve(paths["inference_risk_scores"]),
        "inference_receipt": predictions.parent / "inference_receipt.json",
        "determinism_receipt": _resolve(paths["determinism_receipt"]),
        "score_freeze_receipt": _resolve(paths["score_freeze_receipt"]),
        "used_lock": _resolve(paths["confirmation_used_lock"]),
        "protocol_lock": _resolve(checked["run_dir"]) / "protocol" / "protocol_lock.json",
    }


def _assert_confirmation_unused(paths: Mapping[str, Path]) -> None:
    """The one-shot unseal lock must NOT exist on the score side (Task 6 only)."""
    if paths["used_lock"].exists():
        raise ValueError(
            f"confirmation_used.lock already exists ({paths['used_lock']}); the "
            "one-shot unseal is Task 6 only — the score side must refuse to run")


# --- constant-random streams (mirrors cinf exactly, with THIS stage's D1 root) ----------------


def constant_random_seed_d1(checked: Mapping[str, Any]) -> int:
    """Action-level CONSTANT_RANDOM stream seed for D1 — mirrors cinf's construction.

    Exactly ``stage4cr2_confirm_inference._constant_random_seed`` with the SAME frozen
    salt but THIS stage's frozen d1_confirm action-stream root ([2026110994, 4]):
    ``SeedSequence([*action_stream_root, CONSTANT_RANDOM_SALT]).generate_state(1)[0]``.
    Deterministic and never gate-relevant.
    """
    root = prot.action_stream_root(checked, D1_ROLE)
    return int(np.random.SeedSequence(
        [*root, CONSTANT_RANDOM_SALT]).generate_state(1)[0])


# --- frozen inference + scoring (plan §6.4 step 1) ----------------------------------------------


def _resolve_hooks(run_inference_fn, build_features_fn, load_models_fn,
                   score_actions_fn, build_tokens_fn, score_fn) -> dict[str, Any]:
    """Default to the same frozen chain cinf uses (read-only reuse)."""
    return {
        "run_inference_fn": run_inference_fn or s4ci.run_grouped_frozen_inference,
        "build_features_fn": build_features_fn or s4ci.build_action_risk_features,
        "load_models_fn": load_models_fn or s4ci.load_frozen_models,
        "score_actions_fn": score_actions_fn or s4ci.score_actions,
        "build_tokens_fn": build_tokens_fn or s4cpf.build_relative_tokens,
        "score_fn": score_fn or s4cr2t.score_models,
        "real_chain": run_inference_fn is None,
    }


def infer_and_score_d1(
    config: Mapping[str, Any],
    *,
    run_inference_fn: Callable | None = None,
    build_features_fn: Callable | None = None,
    load_models_fn: Callable | None = None,
    score_actions_fn: Callable | None = None,
    build_tokens_fn: Callable | None = None,
    score_fn: Callable | None = None,
    inventory: s4cr2t.ModelInventory | None = None,
    device: str | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Run the frozen predictor/risk/Direct chain over the D1 legal inputs (truth-blind).

    Mirrors cinf's ``infer_and_score`` step-for-step on the frozen building blocks;
    writes create-once ``predictions.npz`` (per-pair predictor outputs + group binding
    ids), ``risk_scores.npz`` (five Direct seeds, ensemble, MU/CLASS ablations and all
    baselines) and a create-once inference receipt. A rerun verifies hashes and never
    overwrites. No label, regret, oracle action, counterfactual truth or adaptive
    statistic is read or computed; the scaler is the frozen train-only one (never refit).
    """
    checked = prot.validate_config(config)
    paths = inference_artifact_paths(checked)
    _assert_confirmation_unused(paths)
    hooks = _resolve_hooks(run_inference_fn, build_features_fn, load_models_fn,
                           score_actions_fn, build_tokens_fn, score_fn)
    engine = load_engine_config()
    device = str(device or engine.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    batch_size = int(batch_size or engine.get("batch_size", 32))

    pred_path = paths["predictions"]
    score_path = paths["risk_scores"]
    receipt_path = paths["inference_receipt"]
    if receipt_path.exists():
        receipt = _read_json(receipt_path)
        for key, path in (("predictions_sha256", pred_path),
                          ("risk_scores_sha256", score_path)):
            if not path.is_file() or sha256_file(path) != receipt[key]:
                raise ValueError(f"inference artifact drifted from its receipt: {path}")
        return {**receipt, "verification": "existing_inference_verified_no_overwrite"}
    if pred_path.exists() or score_path.exists():
        raise RuntimeError(
            "inference artifacts exist without their create-once receipt: "
            f"{pred_path}, {score_path}")

    legal = collector.load_legal_inputs(paths["inputs"], checked)
    bundle = cinf._bundle_from_arrays(
        legal["arrays"], config=engine, data_signature=legal["data_signature"],
        role=D1_ROLE)
    binding = _read_json(_resolve(engine["stage3b_binding"]))["receipt_payload"]
    stage3r_dir = _resolve(engine["stage3r_run_dir"])
    groups = int(bundle["groups"])
    n_actions = int(bundle["n_actions"])

    predictions = hooks["run_inference_fn"](
        bundle, binding=binding, stage3r_dir=stage3r_dir,
        device=device, batch_size=batch_size)
    cinf._validate_predictions(predictions, groups=groups, n_actions=n_actions)
    features = hooks["build_features_fn"](bundle, predictions)
    frozen_models = hooks["load_models_fn"]()
    rng_seed = constant_random_seed_d1(checked)
    scored = hooks["score_actions_fn"](
        features, frozen_models, groups=groups, n_actions=n_actions, rng_seed=rng_seed)
    action_risks: dict[str, np.ndarray] = {}
    for name in s4cpp.RISK_CLASSES:
        value = np.asarray(scored[name], dtype=np.float64)
        if value.shape != (groups, n_actions) or not np.isfinite(value).all():
            raise RuntimeError(f"non-finite D1 action risk score: {name}")
        action_risks[name] = value

    tokens_bundle = hooks["build_tokens_fn"](
        np.asarray(predictions["e2e"], dtype=np.float64), bundle["factual_xy"],
        bundle["history_last_xy"], bundle["inference_mask"], action_risks,
        bundle["geometry_group"], bundle["t_star"], bundle["group_seed"])
    valid = np.asarray(tokens_bundle["valid_mask"], dtype=bool).ravel()
    if not valid.all():
        raise ValueError(
            f"{int((~valid).sum())} {D1_ROLE} groups have non-finite features; the "
            "contract rejects zero-padding, so scoring is refused")

    inventory = inventory if inventory is not None else cinf.load_frozen_model_inventory(
        engine)
    scores = hooks["score_fn"](
        inventory,
        {"tokens": np.asarray(tokens_bundle["tokens"], dtype=np.float64),
         "global_features": np.asarray(tokens_bundle["global_features"], dtype=np.float64),
         "group_seed": np.asarray(bundle["group_seed"], dtype=np.int64)},
        device=device)
    if tuple(int(s) for s in scores.seeds) != tuple(int(s) for s in prot.MODEL_SEEDS):
        raise RuntimeError("Direct scores must come from the five frozen seeds")
    by_seed = np.asarray(scores.risk_by_seed, dtype=np.float64)
    ensemble = np.asarray(scores.risk_ensemble, dtype=np.float64)
    if by_seed.shape != (len(prot.MODEL_SEEDS), groups) or ensemble.shape != (groups,):
        raise RuntimeError("Direct score shapes must align as (5, G) and (G,)")
    if not np.isfinite(by_seed).all() or not np.isfinite(ensemble).all():
        raise RuntimeError("non-finite Direct D1 score")
    if not np.allclose(ensemble, by_seed.mean(axis=0), rtol=0.0, atol=1e-15):
        raise RuntimeError("the ensemble must be exactly the arithmetic mean of five seeds")

    group_seed = np.asarray(bundle["group_seed"], dtype=np.int64)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(
        pred_path,
        **{k: np.asarray(predictions[k], dtype=np.float32) for k in PREDICTION_KEYS},
        group_seed=group_seed,
        t_star=np.asarray(bundle["t_star"], dtype=np.int64),
        cf_actions=np.tile([a for a, _ in prot.ACTION_PLANS], (groups, 1)).astype(np.int64),
        cf_durations=np.tile([d for _, d in prot.ACTION_PLANS], (groups, 1)).astype(
            np.int64),
    )
    score_payload: dict[str, np.ndarray] = {
        "group_seed": group_seed,
        "risk_by_seed": by_seed,
        "risk_ensemble": ensemble,
        "mu_only_ensemble": np.asarray(scores.mu_only_ensemble, dtype=np.float64),
        "class_only_ensemble": np.asarray(scores.class_only_ensemble, dtype=np.float64),
        "model_seeds": np.asarray(scores.seeds, dtype=np.int64),
        "constant_random": cinf._constant_random_group_scores(group_seed),
        "t_star": np.asarray(bundle["t_star"], dtype=np.int64),
        "interaction_min_distance": np.asarray(
            tokens_bundle["interaction_min_distance"], dtype=np.float64),
        "reference_action": np.asarray(tokens_bundle["reference_action"], dtype=np.int64),
        "valid_mask": valid,
    }
    for name, risk in tokens_bundle["baselines"].items():
        value = np.asarray(risk, dtype=np.float64).ravel()
        if value.shape != (groups,) or not np.isfinite(value).all():
            raise RuntimeError(f"non-finite D1 baseline risk: {name}")
        score_payload[f"baseline_{name}"] = value
    atomic_save_npz(score_path, **score_payload)

    r2_dev = _resolve(engine["r2_development_dir"])
    receipt = {
        "plan_id": PLAN_ID,
        "role": D1_ROLE,
        "status": STATUS_INFERENCE,
        "groups": groups,
        "n_actions": n_actions,
        "branches": int(bundle["branches"]),
        "n_valid": int(valid.sum()),
        "data_sha256": str(bundle["data_signature"]),
        "input_path": str(paths["inputs"]),
        "predictions_path": str(pred_path),
        "predictions_sha256": sha256_file(pred_path),
        "risk_scores_path": str(score_path),
        "risk_scores_sha256": sha256_file(score_path),
        "receipt_path": str(receipt_path),
        "pair_indices": [0, 1, 2, 3, 4],
        "model_seeds": [int(s) for s in prot.MODEL_SEEDS],
        "risk_classes": list(s4cpp.RISK_CLASSES),
        "binding_sha256": sha256_file(_resolve(engine["stage3b_binding"])),
        "trajectory_scaler_sha256": sha256_file(_resolve(engine["scaler_path"])),
        "feature_scaler_sha256": sha256_file(r2_dev / "feature_scaler.npz"),
        "model_inventory_sha256": sha256_file(r2_dev / "model_inventory.json"),
        "final_selection_sha256": sha256_file(r2_dev / "final_selection.json"),
        "engine_binding": {
            "engine_config_path": str(r2confirm.CONFIG_PATH),
            "stage4cr2_protocol_lock_sha256": sha256_file(
                _resolve(confirm_prot.STAGE4CR2_PROTOCOL_LOCK)),
            "stage4cr2_recovery_lock_sha256": sha256_file(
                _resolve(confirm_prot.STAGE4CR2_RECOVERY_LOCK)),
            "cinf_sha256": sha256_file(
                ROOT / "code" / "ac_twin" / "stage4cr2_confirm_inference.py"),
        },
        "constant_random_rng_seed": int(rng_seed),
        "constant_random_streams": {
            "action_level": {
                "derivation": "int(SeedSequence([*action_stream_root, "
                              "constant_random_salt]).generate_state(1)[0])",
                "action_stream_root": [int(v) for v in prot.action_stream_root(
                    checked, D1_ROLE)],
                "constant_random_salt": int(CONSTANT_RANDOM_SALT),
                "rng_seed": int(rng_seed),
                "mirrors": ("stage4cr2_confirm_inference._constant_random_seed (this "
                            "stage's frozen d1_confirm root [2026110994, 4] substituted; "
                            "the salt is the frozen Stage 4C-R2 confirm salt)"),
            },
            "group_level": {
                "derivation": "per group: SeedSequence([int(group_seed)])"
                              ".generate_state(1)[0] / float(2**32) — NO salt",
                "mirrors": "stage4cr2_statistics._constant_random_scores",
            },
        },
        "ensemble_definition": prot.ENSEMBLE_DEFINITION,
        "device": device,
        "dtype": "float32",
        "batch_size": batch_size,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(receipt_path, receipt)
    return receipt


# --- hash-chain verification (score side) --------------------------------------------------------


def _verify_inference_chain(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Re-hash every score-side artifact against its create-once receipts (fail-closed).

    Mirrors the stage4cr2 confirm chain verification on this stage's paths. The sealed
    truth is NEVER opened — only its collection-receipt-recorded hash is carried through.
    """
    if not paths["inference_receipt"].is_file():
        raise ValueError(
            f"the D1 inference receipt is missing: {paths['inference_receipt']}")
    inference = _read_json(paths["inference_receipt"])
    for key, path in (("predictions_sha256", paths["predictions"]),
                      ("risk_scores_sha256", paths["risk_scores"])):
        if not path.is_file() or sha256_file(path) != inference.get(key):
            raise ValueError(f"D1 inference artifact drifted from its receipt: {path}")
    if not paths["inputs"].is_file() \
            or sha256_file(paths["inputs"]) != inference.get("data_sha256"):
        raise ValueError("D1 legal inputs drifted from the inference receipt")
    if not paths["collection_receipt"].is_file():
        raise ValueError(
            f"the D1 collection receipt is missing: {paths['collection_receipt']}")
    collection = _read_json(paths["collection_receipt"])
    if collection.get("status") != collector.STATUS_COLLECTED:
        raise ValueError(
            f"the collection receipt status must be {collector.STATUS_COLLECTED!r}; got "
            f"{collection.get('status')!r}")
    if collection.get("inference_inputs_sha256") != inference.get("data_sha256"):
        raise ValueError("collection receipt does not bind the inferred legal inputs")
    expected_truth = str(collection.get("confirmation_truth_sha256", ""))
    if not expected_truth:
        raise ValueError("collection receipt lost the sealed-truth hash binding")
    inference["collection_receipt_sha256"] = sha256_file(paths["collection_receipt"])
    inference["inference_receipt_sha256"] = sha256_file(paths["inference_receipt"])
    inference["collection_accepted_groups"] = int(collection.get("accepted_groups", -1))
    inference["expected_confirmation_truth_sha256"] = expected_truth
    binding = collection.get("group_order_binding", {})
    inference["collection_same_order"] = bool(binding.get("same_order"))
    return inference


def _verify_determinism(paths: Mapping[str, Path], inference: Mapping[str, Any]
                        ) -> dict[str, Any]:
    """The >=32-group repeated-inference/scoring receipt must exist, pass and bind."""
    if not paths["determinism_receipt"].is_file():
        raise ValueError(
            "the determinism receipt is missing; run the fixed leading-groups repeated "
            "inference/scoring recheck before the score freeze")
    determinism = _read_json(paths["determinism_receipt"])
    if not determinism.get("bitwise_identical"):
        raise ValueError("the repeated inference/scoring was not bitwise identical")
    bindings = determinism.get("bindings", {})
    expected = {
        "predictions_sha256": inference["predictions_sha256"],
        "risk_scores_sha256": inference["risk_scores_sha256"],
        "inference_receipt_sha256": inference["inference_receipt_sha256"],
        "inference_inputs_sha256": inference["data_sha256"],
        "collection_receipt_sha256": inference["collection_receipt_sha256"],
    }
    for key, value in expected.items():
        if bindings.get(key) != value:
            raise ValueError(
                f"determinism receipt is not bound to the formal artifacts ({key})")
    return determinism


def _assert_lock_binding_d1(checked: Mapping[str, Any], paths: Mapping[str, Path]
                            ) -> dict[str, Any]:
    """Cheap local guard: the formal lock must bind THIS config and the current code set.

    The extended code set is resolved through the §13 recovery chain
    (``confirm_prot.resolve_locked_extended_code_hashes``); a broken chain or any
    uncovered drift fails closed. The full source/seed re-verification stays with the
    runner's lock phases.
    """
    lock_path = paths["protocol_lock"]
    if not lock_path.is_file():
        raise ValueError(
            f"the formal protocol lock is missing; freeze requires it first: {lock_path}")
    payload = _read_json(lock_path)
    if payload.get("plan_id") != PLAN_ID:
        raise ValueError(f"protocol lock plan_id must be {PLAN_ID!r}")
    if payload.get("config_signature") != prot._config_signature(checked):
        raise ValueError("running config does not match the locked configuration")
    locked_base = payload.get("code_sha256")
    if not isinstance(locked_base, Mapping) \
            or prot.protocol_code_hashes() != dict(locked_base):
        raise ValueError("locked base code SHA256 set mismatch (code drift or tampering)")
    if confirm_prot.confirm_code_hashes() != \
            confirm_prot.resolve_locked_extended_code_hashes(lock_path):
        raise ValueError(
            "the extended locked code set does not bind the current code (code drift "
            "or tampering; no valid recovery-lock chain covers it)")
    return payload


# --- determinism recheck (plan §6.4 step 2; the stage4cr2 recovery-r1 context pattern) -----------


def _recheck_chain_once_d1(
    checked: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    context_rows: int,
    direct_context_rows: int,
    hooks: Mapping[str, Any],
    device: str,
    batch_size: int,
    binding: Mapping[str, Any],
    stage3r_dir: Path,
    inventory: Any,
) -> dict[str, Any]:
    """Run the frozen predictor/risk/Direct chain once over a leading-group sub-bundle.

    Mirrors ``stage4cr2_confirm_statistics._recheck_chain_once`` with THIS stage's D1
    action-level CONSTANT_RANDOM seed. With the real chain, the three
    row-count-context-dependent forwards are recomputed in the formal context exactly as
    the stage4cr2 recovery-lock-r1 repair prescribes: the unbatched e2e router mixture
    via ``cinf._shape_aligned_router_e2e`` and the single-CPU-forward COMBINED MLP via
    ``cinf._context_aligned_combined_score`` (``context_rows`` = the formal branch
    count), and the single-forward Direct head via
    ``r2stats._context_aligned_direct_scores`` (``direct_context_rows`` = the formal
    group count) — GEMM/MLP row independence; no tolerance is ever substituted for
    bitwise identity.
    """
    groups = int(bundle["groups"])
    predictions = hooks["run_inference_fn"](
        bundle, binding=binding, stage3r_dir=stage3r_dir,
        device=device, batch_size=batch_size)
    cinf._validate_predictions(predictions, groups=groups, n_actions=prot.N_ACTIONS)
    predictions = dict(predictions)
    if hooks["real_chain"]:
        branches = int(bundle["branches"])
        e2e_aligned = np.empty_like(np.asarray(predictions["e2e"], dtype=np.float32))
        for pair_index in range(len(prot.EXPERT_ROUTER_SEEDS)):
            full_pair = np.asarray(
                predictions["full"][pair_index], dtype=np.float32).reshape(
                branches, prot.HORIZON, prot.OBJECT_CAPACITY, 2)
            history_pair = np.asarray(
                predictions["history"][pair_index], dtype=np.float32).reshape(
                branches, prot.HORIZON, prot.OBJECT_CAPACITY, 2)
            e2e_pair = cinf._shape_aligned_router_e2e(
                bundle, full_pair, history_pair, pair_index=pair_index,
                binding=binding, stage3r_dir=stage3r_dir, device=device,
                batch_size=batch_size, context_rows=context_rows)
            e2e_aligned[pair_index] = e2e_pair.reshape(
                groups, prot.N_ACTIONS, prot.HORIZON, prot.OBJECT_CAPACITY, 2)
        predictions["e2e"] = e2e_aligned
    features = hooks["build_features_fn"](bundle, predictions)
    frozen_models = hooks["load_models_fn"]()
    scored = dict(hooks["score_actions_fn"](
        features, frozen_models, groups=groups, n_actions=prot.N_ACTIONS,
        rng_seed=constant_random_seed_d1(checked)))
    if hooks["real_chain"]:
        combined_model = frozen_models["COMBINED_RISK_HEAD"]
        if str(combined_model.implementation) == "mlp":
            scored["COMBINED_RISK_HEAD"] = cinf._context_aligned_combined_score(
                features, combined_model, context_rows=context_rows,
                groups=groups, n_actions=prot.N_ACTIONS)
    action_risks = {name: np.asarray(scored[name], dtype=np.float64)
                    for name in s4cpp.RISK_CLASSES}
    tokens = hooks["build_tokens_fn"](
        np.asarray(predictions["e2e"], dtype=np.float64), bundle["factual_xy"],
        bundle["history_last_xy"], bundle["inference_mask"], action_risks,
        bundle["geometry_group"], bundle["t_star"], bundle["group_seed"])
    scores = r2stats._context_aligned_direct_scores(
        hooks["score_fn"], inventory,
        np.asarray(tokens["tokens"], dtype=np.float64),
        np.asarray(tokens["global_features"], dtype=np.float64),
        np.asarray(bundle["group_seed"], dtype=np.int64),
        context_rows=direct_context_rows, device=device)
    return {
        "predictions": predictions,
        "action_risks": action_risks,
        "tokens": tokens,
        "scores": scores,
    }


def run_determinism_recheck_d1(
    config: Mapping[str, Any],
    *,
    n_groups: int | None = None,
    run_inference_fn: Callable | None = None,
    build_features_fn: Callable | None = None,
    load_models_fn: Callable | None = None,
    score_actions_fn: Callable | None = None,
    build_tokens_fn: Callable | None = None,
    score_fn: Callable | None = None,
    inventory: Any = None,
    device: str | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Repeat inference+scoring on the fixed leading groups; require bitwise identity.

    Plan §6.4 step 2. Two independent runs over the first ``min(DETERMINISM_GROUPS, G)``
    accepted groups (``n_groups`` is a test-only override; the operational run never
    passes it) under the SAME device/batch/dtype/call order must be elementwise
    identical, AND the first run must equal the corresponding LEADING SLICES of the
    formal ``predictions.npz`` / ``risk_scores.npz`` (all five pair predictor outputs,
    all five risk classes, all five Direct seed scores, all baselines, strata inputs) —
    never a two-small-batches-only comparison. Writes the create-once determinism
    receipt; a rerun re-verifies the bindings and never overwrites.
    """
    checked = prot.validate_config(config)
    paths = inference_artifact_paths(checked)
    _assert_confirmation_unused(paths)
    inference = _verify_inference_chain(paths)
    receipt_path = paths["determinism_receipt"]
    if receipt_path.exists():
        receipt = _read_json(receipt_path)
        _verify_determinism(paths, inference)
        return {**receipt, "verification": "existing_determinism_verified_no_overwrite"}

    legal = collector.load_legal_inputs(paths["inputs"], checked)
    groups_total = int(legal["groups"])
    n = min(int(n_groups) if n_groups is not None else DETERMINISM_GROUPS, groups_total)
    sliced = {key: value[:n] for key, value in legal["arrays"].items()}
    engine = load_engine_config()
    bundle = cinf._bundle_from_arrays(
        sliced, config=engine, data_signature=legal["data_signature"],
        role="determinism_recheck")
    with np.load(paths["predictions"], allow_pickle=False) as archive:
        formal_predictions = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(paths["risk_scores"], allow_pickle=False) as archive:
        formal_scores = {key: np.asarray(archive[key]) for key in archive.files}
    context_rows = int(formal_predictions["e2e"].shape[1]) * prot.N_ACTIONS
    direct_context_rows = int(np.asarray(formal_scores["risk_by_seed"]).shape[1])
    if direct_context_rows != int(formal_predictions["e2e"].shape[1]):
        raise ValueError(
            "the formal predictions/risk_scores group counts diverge; the formal "
            "artifacts are inconsistent")

    device = str(device or engine.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    batch_size = int(batch_size or engine.get("batch_size", 32))
    hooks = _resolve_hooks(run_inference_fn, build_features_fn, load_models_fn,
                           score_actions_fn, build_tokens_fn, score_fn)
    binding = _read_json(_resolve(engine["stage3b_binding"]))["receipt_payload"]
    stage3r_dir = _resolve(engine["stage3r_run_dir"])
    if inventory is None and hooks["real_chain"]:
        inventory = cinf.load_frozen_model_inventory(engine)

    first = _recheck_chain_once_d1(
        checked, bundle, context_rows=context_rows,
        direct_context_rows=direct_context_rows, hooks=hooks, device=device,
        batch_size=batch_size, binding=binding, stage3r_dir=stage3r_dir,
        inventory=inventory)
    second = _recheck_chain_once_d1(
        checked, bundle, context_rows=context_rows,
        direct_context_rows=direct_context_rows, hooks=hooks, device=device,
        batch_size=batch_size, binding=binding, stage3r_dir=stage3r_dir,
        inventory=inventory)

    # --- two independent runs: elementwise identical -------------------------------------
    for key in PREDICTION_KEYS:
        if not np.array_equal(np.asarray(first["predictions"][key]),
                              np.asarray(second["predictions"][key])):
            raise RuntimeError(
                f"repeated D1 inference is not elementwise reproducible ({key}); "
                "the one-shot confirmation halts before the lock is burned")
    for name in s4cpp.RISK_CLASSES:
        if not np.array_equal(first["action_risks"][name], second["action_risks"][name]):
            raise RuntimeError(
                f"repeated D1 action scoring is not elementwise reproducible ({name})")
    for key in ("risk_ensemble", "risk_by_seed", "mu_only_ensemble",
                "class_only_ensemble"):
        if not np.array_equal(np.asarray(getattr(first["scores"], key)),
                              np.asarray(getattr(second["scores"], key))):
            raise RuntimeError(
                f"repeated D1 Direct scoring is not elementwise reproducible ({key})")

    # --- run one vs the formal artifacts' leading slices ----------------------------------
    for key in PREDICTION_KEYS:
        formal_slice = np.asarray(formal_predictions[key])[:, :n]
        if not np.array_equal(np.asarray(first["predictions"][key]), formal_slice):
            raise RuntimeError(
                f"the recheck prediction {key!r} does not equal the formal "
                "predictions' leading slice (formal-output cross-check failed)")
    score_slices = {
        "risk_by_seed": np.asarray(first["scores"].risk_by_seed, dtype=np.float64),
        "risk_ensemble": np.asarray(first["scores"].risk_ensemble, dtype=np.float64),
        "mu_only_ensemble": np.asarray(first["scores"].mu_only_ensemble,
                                       dtype=np.float64),
        "class_only_ensemble": np.asarray(first["scores"].class_only_ensemble,
                                          dtype=np.float64),
    }
    for key, mine in score_slices.items():
        formal_slice = np.asarray(formal_scores[key])
        formal_slice = formal_slice[:, :n] if formal_slice.ndim == 2 else formal_slice[:n]
        if not np.array_equal(mine, formal_slice):
            raise RuntimeError(
                f"the recheck Direct score {key!r} does not equal the formal "
                "risk scores' leading slice")
    tokens = first["tokens"]
    seeds_slice = np.asarray(bundle["group_seed"], dtype=np.int64)
    formal_seed = np.asarray(formal_scores["group_seed"], dtype=np.int64)
    if not np.array_equal(seeds_slice, formal_seed[:n]):
        raise RuntimeError("the recheck group order diverges from the formal scores")
    for name, risk in tokens["baselines"].items():
        if not np.array_equal(np.asarray(risk, dtype=np.float64),
                              np.asarray(formal_scores[f"baseline_{name}"])[:n]):
            raise RuntimeError(
                f"the recheck baseline {name!r} does not equal the formal slice")
    if not np.array_equal(cinf._constant_random_group_scores(seeds_slice),
                          np.asarray(formal_scores["constant_random"])[:n]):
        raise RuntimeError("the recheck CONSTANT_RANDOM slice drifted from the formal")
    for key, mine in (("t_star", np.asarray(bundle["t_star"], dtype=np.int64)),
                      ("interaction_min_distance",
                       np.asarray(tokens["interaction_min_distance"], dtype=np.float64)),
                      ("reference_action",
                       np.asarray(tokens["reference_action"], dtype=np.int64)),
                      ("valid_mask", np.asarray(tokens["valid_mask"], dtype=bool))):
        if not np.array_equal(mine, np.asarray(formal_scores[key])[:n]):
            raise RuntimeError(
                f"the recheck {key!r} slice drifted from the formal risk scores")

    receipt = {
        "plan_id": PLAN_ID,
        "role": D1_ROLE,
        "bitwise_identical": True,
        "n_groups_rechecked": int(n),
        "group_indices": list(range(n)),
        "context_rows": int(context_rows),
        "direct_context_rows": int(direct_context_rows),
        "context_note": (
            "the frozen e2e router forward, the COMBINED MLP forward and the Direct "
            "five-seed scoring forward are row-count-context dependent; the recheck "
            "reproduces the formal forward contexts (context_rows = formal branch "
            "count; direct_context_rows = formal group count) via zero-padding "
            "(GEMM/MLP row independence), so the comparison is a true bitwise one"),
        "blocks_compared": [
            "predictions(6 keys, 5 pairs)", "action_risks(5 classes)",
            "direct_scores(ensemble, 5 seeds, MU_ONLY, CLASS_ONLY)",
            "baselines(4)", "constant_random", "t_star", "interaction_min_distance",
            "reference_action", "valid_mask", "group_seed",
        ],
        "bindings": {
            "predictions_sha256": inference["predictions_sha256"],
            "risk_scores_sha256": inference["risk_scores_sha256"],
            "inference_receipt_sha256": sha256_file(paths["inference_receipt"]),
            "inference_inputs_sha256": inference["data_sha256"],
            "collection_receipt_sha256": inference["collection_receipt_sha256"],
        },
        "device": device,
        "dtype": "float32",
        "batch_size": batch_size,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(receipt_path, receipt)
    return receipt


# --- score freeze (plan §6.4 step 3) --------------------------------------------------------------


def freeze_scores_d1(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze every score-side artifact; write the create-once score freeze receipt.

    Refuses unless: the formal protocol lock (with its §13 recovery chain) binds this
    config and the current code set; the collection/inference receipts exist and
    hash-bind the published inputs, predictions and risk scores; the determinism recheck
    passed and is bound to the same artifacts; the frozen n matches the score width; and
    ``confirmation_used.lock`` does NOT exist. The sealed truth is NEVER opened here;
    only its receipt-recorded hash is carried into the receipt. A rerun re-verifies
    everything and never overwrites.
    """
    checked = prot.validate_config(config)
    paths = inference_artifact_paths(checked)
    # A FIRST freeze requires the one-shot unseal lock to be absent. Re-verifying an
    # existing create-once receipt AFTER the lock burned is legal (the Task 6 judge
    # re-verifies the frozen score side around the burn) and must not be blocked by the
    # lock's presence — the create-once receipt already prevents any re-freeze.
    if not paths["score_freeze_receipt"].exists():
        _assert_confirmation_unused(paths)
    lock = _assert_lock_binding_d1(checked, paths)
    inference = _verify_inference_chain(paths)
    determinism = _verify_determinism(paths, inference)
    with np.load(paths["risk_scores"], allow_pickle=False) as archive:
        groups = int(np.asarray(archive["group_seed"]).ravel().size)
    if inference["collection_accepted_groups"] != groups:
        raise ValueError("collection accepted group count does not match the scores")
    if int(lock.get("selected_n", -1)) != groups:
        raise ValueError("the frozen selected n does not match the score width")
    if not inference["collection_same_order"]:
        raise ValueError("the collection receipt does not attest same-order two-file "
                         "group binding")

    receipt_path = paths["score_freeze_receipt"]
    recovery_locks = confirm_prot.recovery_lock_paths(paths["protocol_lock"])
    bindings = {
        "inference_inputs_sha256": inference["data_sha256"],
        "predictions_sha256": inference["predictions_sha256"],
        "risk_scores_sha256": inference["risk_scores_sha256"],
        "inference_receipt_sha256": sha256_file(paths["inference_receipt"]),
        "determinism_receipt_sha256": sha256_file(paths["determinism_receipt"]),
        "collection_receipt_sha256": inference["collection_receipt_sha256"],
        "expected_confirmation_truth_sha256":
            inference["expected_confirmation_truth_sha256"],
        "protocol_lock_sha256": sha256_file(paths["protocol_lock"]),
        "recovery_locks": {path.name: sha256_file(path) for path in recovery_locks},
    }
    if receipt_path.exists():
        receipt = _read_json(receipt_path)
        # The data/score bindings are immutable and must match exactly.
        stored = dict(receipt.get("bindings", {}))
        stored_recovery = stored.pop("recovery_locks", {})
        current = dict(bindings)
        current_recovery = current.pop("recovery_locks", {})
        if receipt.get("plan_id") != PLAN_ID or stored != current:
            raise ValueError(
                "the score freeze receipt no longer matches the on-disk frozen "
                "artifacts (tampering or drift)")
        # The §13 recovery chain may only GROW by appending valid locks after the freeze
        # (e.g. a post-freeze Task 6 adjudication-code lock); the frozen prefix is
        # immutable, so every lock frozen into this receipt must still be present and
        # byte-identical, and the current (extended) chain must resolve to the on-disk set.
        for name, digest in stored_recovery.items():
            if current_recovery.get(name) != digest:
                raise ValueError(
                    "a recovery lock frozen into the score freeze receipt drifted or was "
                    "removed (tampering)")
        if confirm_prot.confirm_code_hashes() != \
                confirm_prot.resolve_locked_extended_code_hashes(paths["protocol_lock"]):
            raise ValueError(
                "the current recovery chain does not resolve to the on-disk extended "
                "code set (code drift or tampering)")
        return {**receipt, "verification": "existing_score_freeze_verified_no_overwrite"}
    receipt = {
        "plan_id": PLAN_ID,
        "role": D1_ROLE,
        "n_groups": int(groups),
        "bindings": bindings,
        "model_seeds": [int(s) for s in prot.MODEL_SEEDS],
        "ensemble_definition": prot.ENSEMBLE_DEFINITION,
        "primary_baseline": prot.PRIMARY_BASELINE,
        "standard_baselines": list(prot.STANDARD_BASELINES),
        "reference_baselines": list(prot.REFERENCE_BASELINES),
        "determinism_rechecked_groups": int(determinism["n_groups_rechecked"]),
        "bootstrap": {
            "entropy": prot.D1_BOOTSTRAP_ENTROPY,
            "resamples": prot.BOOTSTRAP_RESAMPLES,
            "confidence": prot.CONFIDENCE,
        },
        "constant_random_salt": int(CONSTANT_RANDOM_SALT),
        "engine_binding": dict(inference.get("engine_binding", {})),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(receipt_path, receipt)
    return receipt


__all__ = [
    "CONFIG_PATH",
    "CONSTANT_RANDOM_SALT",
    "D1_ROLE",
    "DETERMINISM_GROUPS",
    "LEGAL_INPUT_KEYS",
    "PLAN_ID",
    "PREDICTION_KEYS",
    "STATUS_INFERENCE",
    "constant_random_seed_d1",
    "freeze_scores_d1",
    "infer_and_score_d1",
    "inference_artifact_paths",
    "load_config",
    "load_engine_config",
    "run_determinism_recheck_d1",
]
