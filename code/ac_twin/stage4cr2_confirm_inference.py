"""Stage 4C-R2 independent-confirmation inference: truth-isolated loading + frozen scoring.

Plan ``AC-TWIN-RFER-STAGE4CR2-CONFIRM-2026-09-17-V0.1`` sections 2.2, 3.3, 5.1, 6.4 and 9
Task 1. The score side reads ONLY the published legal-input file (``inference_inputs.npz``,
plan section 5.1); the sealed ``confirmation_truth.npz`` is NEVER opened by any function in
this module — the input path is refused by name, every archive key is screened against
truth-side tokens, and the returned bundle carries no counterfactual-truth array. The ten
actions / 20-step horizon / 10-step history are the fixed contract; the action count is
NEVER derived from counterfactual-truth shapes.

The frozen chain is REUSED read-only: ``stage4c_inference.run_grouped_frozen_inference`` /
``build_action_risk_features`` / ``score_actions`` / ``load_frozen_models``,
``stage4cp_features.build_relative_tokens`` and ``stage4cr2_training.score_models``. The
frozen ``ModelInventory`` is constructed read-only from R2's hash-pinned
``model_inventory.json`` / ``final_selection.json`` / ``feature_scaler.npz`` (paths
relocated first, then hash-verified). No ``fit``, no optimizer, no calibration, no
training/selection entrypoint of any old stage is called.

The adapter consistency check (plan section 2.2) verifies this module's predictions,
action scores, features and Direct scores against R2's on-disk train products for the
fixed first 32 R2 train groups, using ONLY legal inputs and on-disk predictions — no
counterfactual truth, no labels, no new scientific metric.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

import run_stage4a as s4a
from ac_twin import stage4_protocol as s4p  # EXPERT_ROUTER_SEEDS, read-only
from ac_twin import stage4_risk as s4r  # design matrix + candidate contract, read-only
from ac_twin import stage4c_inference as s4ci
from ac_twin import stage4cp_features as s4cpf
from ac_twin import stage4cp_protocol as s4cpp  # RISK_CLASSES, read-only
from ac_twin import stage4cr2_confirm_protocol as prot
from ac_twin import stage4cr2_protocol as s4cr2  # frozen R2 constants, read-only
from ac_twin import stage4cr2_training as s4cr2t
from ac_twin.checkpointing import atomic_save_npz
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3_scaler import FeatureScaler
from ac_twin.stage3b_bridge import calibrated_regime_probs_single
from ac_twin.stage3r_models import build_router_features
from ac_twin.stage3r_protocol import write_once_json

ROOT = prot.ROOT
CONFIG_PATH = prot.CONFIG_PATH

POSITION_SLICE = s4a.POSITION_SLICE

STATUS_INFERENCE = "INFERENCE_COMPLETE"
STATUS_SMOKE = "ENGINEERING_SMOKE_ONLY"

#: prediction-side arrays persisted in ``predictions.npz`` (plus group binding ids).
PREDICTION_KEYS = ("e2e", "full", "history", "r_hat", "presence_prob", "high_prob")

#: exactly the legal input keys of the published §5.1 inference_inputs.npz.
LEGAL_INPUT_KEYS = (
    "object_history", "object_factual_future", "factual_actions",
    "cf_actions", "cf_durations", "t_star", "group_seed", "inference_mask",
)

#: archive-key tokens that mark truth-side content; any input key containing one is
#: refused fail-closed, and the score side never reads such a key from any archive.
FORBIDDEN_INPUT_KEY_TOKENS = (
    "counterfactual", "truth", "j_true", "regret", "oracle", "a_star", "true_",
)

#: R2 train archive legal-key mapping (R2 name -> this stage's legal name). The R2
#: archive's ``object_counterfactual_futures`` key is NEVER accessed (plan section 2.2).
_R2_TRAIN_LEGAL_KEYS = {
    "object_history": "object_history",
    "object_factual_future": "object_factual_future",
    "factual_actions": "factual_actions",
    "cf_actions": "cf_actions",
    "cf_durations": "cf_durations",
    "t_star": "t_star",
    "episode_seed": "group_seed",
    "inference_mask": "inference_mask",
}

#: R2's frozen curve-reference salt (must equal stage4cr2_inference._CURVE_REFERENCE_SALT;
#: pinned by the inference tests). Used ONLY to reproduce R2's train-role action-score
#: stream inside the adapter consistency check — never in this stage's own products.
_R2_CURVE_REFERENCE_SALT = 17


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load + validate the frozen confirm-stage config."""
    return prot.validate_config(_read_json(Path(config_path)))


# --- path contract ----------------------------------------------------------------------


def role_input_path(config: Mapping[str, Any], role: str) -> Path:
    """The role's legal-input file (NEVER the sealed truth)."""
    checked = prot.validate_config(config)
    role = str(role)
    if role == "confirm":
        return _resolve(checked["paths"]["inference_inputs"])
    if role == "smoke":
        return _resolve(checked["data_root"]) / "smoke" / "inference_inputs.npz"
    raise ValueError(f"unknown Stage 4C-R2 confirm inference role: {role!r}")


def role_truth_path(config: Mapping[str, Any], role: str) -> Path:
    """The role's sealed truth file — registered so score-side code can REFUSE it."""
    checked = prot.validate_config(config)
    role = str(role)
    if role == "confirm":
        return _resolve(checked["paths"]["confirmation_truth"])
    if role == "smoke":
        return _resolve(checked["data_root"]) / "smoke" / "confirmation_truth.npz"
    raise ValueError(f"unknown Stage 4C-R2 confirm inference role: {role!r}")


def role_inference_paths(config: Mapping[str, Any], role: str) -> tuple[Path, Path, Path]:
    """Resolve (predictions_npz, risk_scores_npz, inference_receipt) for ``role``."""
    checked = prot.validate_config(config)
    role = str(role)
    if role == "confirm":
        paths = checked["paths"]
        return (_resolve(paths["inference_predictions"]),
                _resolve(paths["inference_risk_scores"]),
                _resolve(paths["inference_receipt"]))
    if role == "smoke":
        base = _resolve(checked["run_dir"]) / "smoke"
        return (base / "smoke_predictions.npz", base / "smoke_risk_scores.npz",
                base / "smoke_inference_receipt.json")
    raise ValueError(f"unknown Stage 4C-R2 confirm inference role: {role!r}")


# --- truth-isolated legal-input loading ---------------------------------------------------


def _assert_not_truth_path(path: Path, checked: Mapping[str, Any]) -> None:
    normalized = str(path).replace("\\", "/").lower()
    if "confirmation_truth" in normalized or "truth" in Path(path).name.lower():
        raise ValueError(
            f"the score side must never open the sealed truth file: {path}")
    for role in prot.CONFIRM_DATA_ROLES:
        if path == role_truth_path(checked, role):
            raise ValueError(f"score-side input resolved to the sealed truth file: {path}")


def _screen_archive_keys(names: list[str]) -> None:
    for name in names:
        lowered = str(name).lower()
        if any(token in lowered for token in FORBIDDEN_INPUT_KEY_TOKENS):
            raise ValueError(
                f"score-side input archive contains a truth-side key {name!r}; the "
                "legal input file carries no counterfactual truth (plan section 5.1)")


def _read_legal_npz(
    input_path: Path,
    key_map: Mapping[str, str],
    *,
    screen_keys: bool = True,
) -> dict[str, np.ndarray]:
    """Read ONLY the legal keys of an input archive; truth keys are never accessed.

    ``screen_keys=True`` (the published §5.1 legal input) rejects any archive that
    carries truth-side keys at all. ``screen_keys=False`` is used ONLY for the R2 train
    archive in the adapter consistency check: that dev archive legitimately contains
    counterfactual truth, which is simply never accessed (the truth-access spy in the
    tests proves no truth key is touched).
    """
    with np.load(input_path, allow_pickle=True) as archive:
        if screen_keys:
            _screen_archive_keys(list(archive.files))
        missing = [key for key in key_map if key not in archive.files]
        if missing:
            raise ValueError(
                f"legal input archive {input_path} missing required keys: {missing}")
        return {legal: np.asarray(archive[published]) for published, legal in key_map.items()}


def _bundle_from_arrays(
    data: Mapping[str, np.ndarray],
    *,
    config: Mapping[str, Any],
    data_signature: str,
    role: str,
) -> dict[str, Any]:
    """Expand legal group arrays to the E = G*10 branch bundle (group-major, action-minor).

    Mirrors the frozen R2/Stage 4C expansion bit-for-bit (scaler transform, materialised
    counterfactual plans, pre-intervention geometry) — WITHOUT any counterfactual-truth
    array. The action axis is the fixed ten-action contract; it is never derived from a
    counterfactual-truth shape.
    """
    checked = prot.validate_config(config)
    history = np.asarray(data["object_history"], dtype=np.float32)
    factual = np.asarray(data["object_factual_future"], dtype=np.float32)
    factual_actions = np.asarray(data["factual_actions"], dtype=np.int64)
    cf_actions = np.asarray(data["cf_actions"], dtype=np.int64)
    cf_durations = np.asarray(data["cf_durations"], dtype=np.int64)
    t_star = np.asarray(data["t_star"], dtype=np.int64).ravel()
    group_seed = np.asarray(data["group_seed"], dtype=np.int64).ravel()
    inference_mask = np.asarray(data["inference_mask"], dtype=bool)

    groups = history.shape[0]
    expected_history = (prot.HISTORY_LEN, prot.OBJECT_CAPACITY, prot.N_OBJECT_FEATURES)
    expected_future = (prot.HORIZON, prot.OBJECT_CAPACITY, prot.N_OBJECT_FEATURES)
    if history.shape[1:] != expected_history:
        raise ValueError(f"object_history must be (G,{expected_history}); got {history.shape}")
    if factual.shape != (groups,) + expected_future:
        raise ValueError(
            f"object_factual_future must be (G,{expected_future}); got {factual.shape}")
    if factual_actions.shape != (groups, prot.HORIZON):
        raise ValueError(f"factual_actions must be (G,{prot.HORIZON})")
    if cf_actions.shape != (groups, prot.N_ACTIONS) \
            or cf_durations.shape != (groups, prot.N_ACTIONS):
        raise ValueError(
            f"cf_actions/cf_durations must be (G,{prot.N_ACTIONS}) — the fixed "
            "ten-action contract, never read from counterfactual-truth shapes")
    if inference_mask.shape != (groups, prot.HORIZON, prot.OBJECT_CAPACITY):
        raise ValueError("inference_mask must be (G,horizon,capacity)")
    if t_star.shape != (groups,) or group_seed.shape != (groups,):
        raise ValueError("t_star/group_seed must be (G,)")
    if np.unique(group_seed).size != groups:
        raise ValueError("group seeds must be unique within an input file")
    if not np.isfinite(history).all() or not np.isfinite(factual).all():
        raise ValueError("non-finite legal input array")

    n_actions = prot.N_ACTIONS  # fixed contract
    g_idx = np.repeat(np.arange(groups), n_actions)
    branches = groups * n_actions

    hist_b = history[g_idx]
    fact_b = factual[g_idx]
    factual_suffix = fact_b[:, :prot.FACTUAL_SUFFIX_STEPS]
    factual_actions_b = factual_actions[g_idx]
    cf_actions_b = cf_actions.reshape(branches)
    cf_durations_b = cf_durations.reshape(branches)
    t_star_b = t_star[g_idx]
    cf_plans = np.stack([
        prot.materialize_action_plan(factual_actions[g], int(cf_actions[g, a]),
                                     int(cf_durations[g, a]))
        for g in range(groups) for a in range(n_actions)
    ]).astype(np.int64)
    valid_b = inference_mask[g_idx]

    scaler = FeatureScaler.load(_resolve(checked["scaler_path"]))
    tensors = {
        "history": torch.as_tensor(hist_b, dtype=torch.float32),
        "factual_suffix": torch.as_tensor(factual_suffix, dtype=torch.float32),
        "factual_actions": torch.as_tensor(factual_actions_b, dtype=torch.long),
        "counterfactual_actions": torch.as_tensor(cf_plans, dtype=torch.long),
        "cf_action": torch.as_tensor(cf_actions_b, dtype=torch.long),
        "cf_duration": torch.as_tensor(cf_durations_b, dtype=torch.long),
        "t_star": torch.as_tensor(t_star_b, dtype=torch.long),
    }
    scaled = dict(tensors)
    scaled["history"] = torch.as_tensor(scaler.transform(hist_b), dtype=torch.float32)
    scaled["factual_suffix"] = torch.as_tensor(
        scaler.transform(factual_suffix), dtype=torch.float32)

    geometry_group = s4a._pre_intervention_geometry(history)
    geometry = {
        "distance_to_ego": geometry_group["distance_to_ego"][g_idx],
        "relative_speed": geometry_group["relative_speed"][g_idx],
        "visible_count": geometry_group["visible_count"][g_idx],
    }
    last = history[:, -1]  # (G,5,8)
    ego_xy = last[:, 0, POSITION_SLICE]
    dist = np.linalg.norm(last[:, :, POSITION_SLICE] - ego_xy[:, None, :], axis=-1)
    present = last[..., 0] > 0
    present[:, 0] = False  # exclude ego
    density_count = ((dist <= 30.0) & present).sum(axis=1).astype(np.float64)
    return {
        "role": role,
        "groups": int(groups),
        "n_actions": int(n_actions),
        "branches": int(branches),
        "tensors": tensors,
        "scaled": scaled,
        "valid": valid_b,
        "geometry": geometry,
        "geometry_group": geometry_group,
        "group_seed": group_seed,
        "t_star": t_star,
        "inference_mask": inference_mask,
        "density_count": density_count,
        "history_last_xy": history[:, -1, 0, POSITION_SLICE].astype(np.float64),
        "factual_xy": factual[..., POSITION_SLICE].astype(np.float64),
        # NO counterfactual_xy, NO truth-side array — adjudication-side only, and never
        # in this module (plan sections 2.2, 5.1).
        "data_signature": data_signature,
    }


def load_inference_bundle(input_path: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Load a published §5.1 legal-input file into the truth-isolated score bundle.

    The sealed truth file is refused by path name; the archive is screened for
    truth-side keys fail-closed; only the eight legal keys are read. Every data path
    passes the protocol namespace guard first.
    """
    checked = prot.validate_config(config)
    input_path = _resolve(input_path)
    _assert_not_truth_path(input_path, checked)
    prot.assert_allowed_confirm_data(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"legal inference inputs missing: {input_path}")
    data = _read_legal_npz(input_path, {key: key for key in LEGAL_INPUT_KEYS})
    return _bundle_from_arrays(data, config=checked,
                               data_signature=sha256_file(input_path), role="confirm")


# --- frozen model inventory (read-only; paths relocated first, then hash-verified) ---------


def load_frozen_model_inventory(config: Mapping[str, Any]) -> s4cr2t.ModelInventory:
    """Rebuild the frozen Direct ModelInventory from R2's hash-pinned receipts.

    Read-only: verifies the pinned hashes of ``final_selection.json`` /
    ``model_inventory.json`` / ``feature_scaler.npz``, relocates the five checkpoint
    paths under the R2 development ``models/`` dir, hash-verifies each against BOTH
    receipts, and cross-checks the scaler against the pinned ``feature_scaler.npz``.
    The scaler is LOADED, never fit; no training or selection entrypoint is called.
    """
    checked = prot.validate_config(config)
    r2_dev = _resolve(checked["r2_development_dir"])

    # Hash-verify ONLY the three pinned sources this function consumes (the full §2.1
    # ten-file verification lives in preflight / verify_r2_sources).
    for name in ("final_selection.json", "model_inventory.json", "feature_scaler.npz"):
        path = r2_dev / name
        if not path.is_file():
            raise prot.PreflightError(f"missing R2 authoritative source: {path}")
        digest = sha256_file(path)
        if digest != prot.R2_PINNED_SOURCES[name]:
            raise prot.PreflightError(
                f"R2 authoritative source hash drift: {name}; expected "
                f"{prot.R2_PINNED_SOURCES[name]}, got {digest}")

    final = _read_json(r2_dev / "final_selection.json")
    if final.get("status") != "VALIDATION_QUALIFIED":
        raise ValueError(
            f"the frozen R2 selection must be VALIDATION_QUALIFIED; got "
            f"{final.get('status')!r}")
    if str(final.get("architecture")) != prot.ARCH_DIRECT:
        raise ValueError(f"frozen architecture drifted from {prot.ARCH_DIRECT}")
    if [int(s) for s in final.get("model_seeds", [])] != list(prot.MODEL_SEEDS):
        raise ValueError("final_selection model seeds drifted from the frozen five")
    if int(final.get("param_count", -1)) != prot.EXPECTED_TRAINABLE_PARAMETERS:
        raise ValueError("final_selection param count drifted from 15,740")
    if str(final.get("strongest_standard_baseline")) != prot.PRIMARY_BASELINE:
        raise ValueError(
            f"the validation-frozen primary baseline must be {prot.PRIMARY_BASELINE!r}")
    if float(final.get("interaction_threshold", float("nan"))) != float(
            checked["interaction_threshold"]):
        raise ValueError("final_selection interaction threshold drifted from the config")
    if str(final.get("primary_score_definition", "")) != prot.PRIMARY_SCORE_DEFINITION:
        raise ValueError("final_selection primary score definition drifted")

    payload = _read_json(r2_dev / "model_inventory.json")
    scaler = s4cpf.FrozenFeatureScaler.load(r2_dev / "feature_scaler.npz")
    embedded = payload["scaler"]
    for key in ("token_cont_mean", "token_cont_std", "global_mean", "global_std"):
        if not np.array_equal(np.asarray(embedded[key], dtype=np.float64),
                              np.asarray(getattr(scaler, key), dtype=np.float64)):
            raise ValueError(
                f"model_inventory embedded scaler disagrees with the pinned "
                f"feature_scaler.npz on {key}")

    # Relocate the five checkpoints under <r2_dev>/models/ first, then hash-verify
    # against BOTH final_selection.json and model_inventory.json (cross-checked).
    checkpoint_paths: dict[int, str] = {}
    checkpoint_sha256: dict[int, str] = {}
    for seed_text, record in final["checkpoints"].items():
        expected_final = str(record["sha256"])
        expected_inventory = str(payload["checkpoint_sha256"][seed_text])
        if expected_final != expected_inventory:
            raise prot.PreflightError(
                f"final_selection/model_inventory checkpoint hash mismatch for seed "
                f"{seed_text}")
        relocated = r2_dev / "models" / Path(record["path"]).name
        if not relocated.is_file():
            raise prot.PreflightError(f"missing frozen Direct checkpoint: {relocated}")
        digest = sha256_file(relocated)
        if digest != expected_final:
            raise prot.PreflightError(
                f"frozen Direct checkpoint hash drift: {relocated}; expected "
                f"{expected_final}, got {digest}")
        checkpoint_paths[int(seed_text)] = str(relocated)
        checkpoint_sha256[int(seed_text)] = digest
    if sorted(checkpoint_paths) != sorted(int(s) for s in prot.MODEL_SEEDS):
        raise prot.PreflightError(
            "final_selection checkpoints do not cover the frozen five seeds")
    return s4cr2t.ModelInventory(
        plan_id=str(payload["plan_id"]),
        arch=str(payload["arch"]),
        seeds=tuple(int(s) for s in payload["seeds"]),
        param_count=int(payload["param_count"]),
        run_dir=str(r2_dev.parent),
        checkpoint_paths=checkpoint_paths,
        checkpoint_sha256=checkpoint_sha256,
        scaler=scaler,
        pos_weight=float(payload["pos_weight"]),
        interaction_threshold=float(payload["interaction_threshold"]),
        main_indices=tuple(int(i) for i in payload["main_indices"]),
        monitor_indices=tuple(int(i) for i in payload["monitor_indices"]),
        train_losses={int(k): float(v) for k, v in payload["train_losses"].items()},
        monitor_losses={int(k): float(v) for k, v in payload["monitor_losses"].items()},
        n_train_groups=int(payload["n_train_groups"]),
        written_at=str(payload["written_at"]),
        verification="loaded_read_only_hash_verified_no_fit",
    )


# --- frozen scoring ----------------------------------------------------------------------------


def _validate_predictions(
    predictions: Mapping[str, np.ndarray], *, groups: int, n_actions: int
) -> None:
    """Fail closed on any missing/non-finite/mis-shaped predictor output."""
    for key in PREDICTION_KEYS:
        if key not in predictions:
            raise RuntimeError(f"frozen inference output missing key: {key}")
        arr = np.asarray(predictions[key])
        if not np.isfinite(arr.astype(np.float64)).all():
            raise RuntimeError(f"non-finite confirm-stage predictor output: {key}")
    e2e = np.asarray(predictions["e2e"])
    if e2e.ndim != 6 or e2e.shape[1] != groups or e2e.shape[2] != n_actions \
            or e2e.shape[-1] != 2:
        raise RuntimeError(f"e2e must be (S,G,A,T,N,2); got {e2e.shape}")


def _constant_random_seed(checked: Mapping[str, Any], role: str) -> int:
    """Action-level CONSTANT_RANDOM stream seed — mirrors R2's frozen construction.

    Exactly ``stage4cr2_inference._curve_reference_seed`` with the confirm salt:
    ``SeedSequence([*action_stream_root, CONSTANT_RANDOM_SALT]).generate_state(1)[0]``
    where the root is the role's frozen action stream root ([2026093012, 0] confirm,
    [2026093100, 12] smoke). Deterministic and never gate-relevant.
    """
    root = prot.action_stream_root(checked, role)
    return int(np.random.SeedSequence(
        [*root, prot.CONSTANT_RANDOM_SALT]).generate_state(1)[0])


def _constant_random_group_scores(group_seed: np.ndarray) -> np.ndarray:
    """Group-level CONSTANT_RANDOM scores — mirrors R2's frozen construction exactly.

    Exactly ``stage4cr2_statistics._constant_random_scores``:
    ``SeedSequence([int(seed)]).generate_state(1)[0] / 2**32`` per group — a pure
    function of the unique group seed with NO salt. Deterministic across reruns,
    introduces no new frozen constant, never gate-relevant.
    """
    out = np.empty(group_seed.size, dtype=np.float64)
    for i, seed in enumerate(group_seed):
        out[i] = np.random.SeedSequence([int(seed)]).generate_state(1)[0] / float(2**32)
    return out


def infer_and_score(
    config: Mapping[str, Any],
    role: str,
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
    """Run the frozen predictor/risk/Direct chain over one role's legal inputs.

    Reads ONLY the role's legal-input file. Writes create-once ``predictions.npz``
    (per-pair predictor outputs + group binding ids) and ``risk_scores.npz`` (five
    Direct seeds, ensemble, MU/CLASS ablations and all baselines), then a create-once
    inference receipt. A rerun verifies hashes and never overwrites. No label, regret,
    oracle action, counterfactual truth or adaptive statistic is read or computed; the
    scaler is the frozen train-only one (never refit); all models stay in ``eval()``.
    """
    checked = prot.validate_config(config)
    role = str(role)
    if role not in prot.CONFIRM_DATA_ROLES:
        raise ValueError(f"unknown Stage 4C-R2 confirm inference role: {role!r}")
    run_inference_fn = run_inference_fn or s4ci.run_grouped_frozen_inference
    build_features_fn = build_features_fn or s4ci.build_action_risk_features
    load_models_fn = load_models_fn or s4ci.load_frozen_models
    score_actions_fn = score_actions_fn or s4ci.score_actions
    build_tokens_fn = build_tokens_fn or s4cpf.build_relative_tokens
    score_fn = score_fn or s4cr2t.score_models
    device = str(device or checked.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    batch_size = int(batch_size or checked.get("batch_size", 32))

    pred_path, score_path, receipt_path = role_inference_paths(checked, role)
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

    bundle = load_inference_bundle(role_input_path(checked, role), checked)
    binding = _read_json(_resolve(checked["stage3b_binding"]))["receipt_payload"]
    stage3r_dir = _resolve(checked["stage3r_run_dir"])
    groups = int(bundle["groups"])
    n_actions = int(bundle["n_actions"])

    predictions = run_inference_fn(
        bundle, binding=binding, stage3r_dir=stage3r_dir,
        device=device, batch_size=batch_size)
    _validate_predictions(predictions, groups=groups, n_actions=n_actions)
    features = build_features_fn(bundle, predictions)
    frozen_models = load_models_fn()
    rng_seed = _constant_random_seed(checked, role)
    scored = score_actions_fn(
        features, frozen_models, groups=groups, n_actions=n_actions, rng_seed=rng_seed)
    action_risks: dict[str, np.ndarray] = {}
    for name in s4cpp.RISK_CLASSES:
        value = np.asarray(scored[name], dtype=np.float64)
        if value.shape != (groups, n_actions) or not np.isfinite(value).all():
            raise RuntimeError(f"non-finite confirm-stage action risk score: {name}")
        action_risks[name] = value

    tokens_bundle = build_tokens_fn(
        np.asarray(predictions["e2e"], dtype=np.float64), bundle["factual_xy"],
        bundle["history_last_xy"], bundle["inference_mask"], action_risks,
        bundle["geometry_group"], bundle["t_star"], bundle["group_seed"])
    valid = np.asarray(tokens_bundle["valid_mask"], dtype=bool).ravel()
    if not valid.all():
        raise ValueError(
            f"{int((~valid).sum())} {role} groups have non-finite features; the "
            "contract rejects zero-padding, so scoring is refused")

    inventory = inventory if inventory is not None else load_frozen_model_inventory(checked)
    scores = score_fn(
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
        raise RuntimeError("non-finite Direct confirm score")
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
        cf_durations=np.tile([d for _, d in prot.ACTION_PLANS], (groups, 1)).astype(np.int64),
    )
    score_payload: dict[str, np.ndarray] = {
        "group_seed": group_seed,
        "risk_by_seed": by_seed,
        "risk_ensemble": ensemble,
        "mu_only_ensemble": np.asarray(scores.mu_only_ensemble, dtype=np.float64),
        "class_only_ensemble": np.asarray(scores.class_only_ensemble, dtype=np.float64),
        "model_seeds": np.asarray(scores.seeds, dtype=np.int64),
        "constant_random": _constant_random_group_scores(group_seed),
        "t_star": np.asarray(bundle["t_star"], dtype=np.int64),
        "interaction_min_distance": np.asarray(
            tokens_bundle["interaction_min_distance"], dtype=np.float64),
        "reference_action": np.asarray(tokens_bundle["reference_action"], dtype=np.int64),
        "valid_mask": valid,
    }
    for name, risk in tokens_bundle["baselines"].items():
        value = np.asarray(risk, dtype=np.float64).ravel()
        if value.shape != (groups,) or not np.isfinite(value).all():
            raise RuntimeError(f"non-finite confirm-stage baseline risk: {name}")
        score_payload[f"baseline_{name}"] = value
    atomic_save_npz(score_path, **score_payload)

    r2_dev = _resolve(checked["r2_development_dir"])
    receipt = {
        "plan_id": prot.PLAN_ID,
        "role": role,
        "status": STATUS_SMOKE if role == "smoke" else STATUS_INFERENCE,
        "groups": groups,
        "n_actions": n_actions,
        "branches": int(bundle["branches"]),
        "n_valid": int(valid.sum()),
        "data_sha256": str(bundle["data_signature"]),
        "input_path": str(role_input_path(checked, role)),
        "predictions_path": str(pred_path),
        "predictions_sha256": sha256_file(pred_path),
        "risk_scores_path": str(score_path),
        "risk_scores_sha256": sha256_file(score_path),
        "receipt_path": str(receipt_path),
        "pair_indices": [0, 1, 2, 3, 4],
        "model_seeds": [int(s) for s in prot.MODEL_SEEDS],
        "risk_classes": list(s4cpp.RISK_CLASSES),
        "binding_sha256": sha256_file(_resolve(checked["stage3b_binding"])),
        "trajectory_scaler_sha256": sha256_file(_resolve(checked["scaler_path"])),
        "feature_scaler_sha256": sha256_file(r2_dev / "feature_scaler.npz"),
        "model_inventory_sha256": sha256_file(r2_dev / "model_inventory.json"),
        "final_selection_sha256": sha256_file(r2_dev / "final_selection.json"),
        "constant_random_rng_seed": int(rng_seed),
        "constant_random_streams": {
            "action_level": {
                "derivation": "int(SeedSequence([*action_stream_root, "
                              "constant_random_salt]).generate_state(1)[0])",
                "action_stream_root": [int(v) for v in prot.action_stream_root(checked, role)],
                "constant_random_salt": int(prot.CONSTANT_RANDOM_SALT),
                "rng_seed": int(rng_seed),
                "mirrors": "stage4cr2_inference._curve_reference_seed (salt substituted)",
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


# --- adapter consistency check (plan section 2.2; R2 train first 32 groups, legal only) -------


def _r2_train_curve_reference_seed() -> int:
    """R2's frozen train-role CONSTANT_RANDOM stream seed (read-only reproduction).

    Equals ``stage4cr2_inference._curve_reference_seed(r2_config, 'development_train')``
    (pinned by the inference tests); used ONLY inside the adapter consistency check so
    the recomputed action scores are comparable to R2's on-disk train scores.
    """
    return int(np.random.SeedSequence(
        [s4cr2.DEV_SEED_ENTROPY, s4cr2.TRAIN_SEGMENT[0],
         _R2_CURVE_REFERENCE_SALT]).generate_state(1)[0])


def _compare_prediction_arrays(
    ours: Mapping[str, np.ndarray], reference: Mapping[str, np.ndarray]
) -> float:
    """Elementwise-identity comparison of the six predictor outputs; fail closed."""
    max_abs = 0.0
    for key in PREDICTION_KEYS:
        if key not in ours or key not in reference:
            raise RuntimeError(f"adapter comparison missing predictor key: {key}")
        mine = np.asarray(ours[key])
        ref = np.asarray(reference[key])
        if mine.shape != ref.shape:
            raise RuntimeError(
                f"adapter prediction shape mismatch on {key}: {mine.shape} vs {ref.shape}")
        diff = float(np.max(np.abs(mine.astype(np.float64) - ref.astype(np.float64)))) \
            if mine.size else 0.0
        max_abs = max(max_abs, diff)
        if not np.array_equal(mine, ref):
            raise RuntimeError(
                f"adapter prediction {key!r} is not elementwise identical to R2's "
                f"on-disk train predictions (max abs diff {diff:.3e})")
    return max_abs


def _shape_aligned_router_e2e(
    bundle: Mapping[str, Any],
    full: np.ndarray,
    history: np.ndarray,
    *,
    pair_index: int,
    binding: Mapping[str, Any],
    stage3r_dir: Path,
    device: str,
    batch_size: int,
    context_rows: int,
) -> np.ndarray:
    """Reproduce one pair's e2e router mixture in R2's ORIGINAL forward context.

    The frozen ``stage4b_inference.run_pair_inference`` runs the e2e router forward
    UNBATCHED over all requested branches, so the cuBLAS kernel (hence the float32
    accumulation order) depends on the total row count M. R2's on-disk train
    predictions were produced in a single forward at M = G_train*10. GEMM output rows
    depend only on their own input row given a fixed kernel, so zero-padding the
    router input to the original M reproduces R2's first-E rows bit-for-bit (probe
    verified on all keys). Every other predictor output is batched at ``batch_size``
    and is already bitwise identical without padding. Uses ONLY the same frozen
    primitives as ``run_pair_inference``; no frozen file is modified.
    """
    seed = s4p.EXPERT_ROUTER_SEEDS[pair_index]
    scaled = bundle["scaled"]
    branches = int(bundle["branches"])
    if int(context_rows) < branches:
        raise RuntimeError("router context rows cannot be smaller than the branch count")
    indices = np.arange(branches, dtype=np.int64)
    response = s4a._response_outputs(
        s4a._load_response_predictor(binding, pair_index), scaled, indices,
        device, batch_size)
    regime_probs = calibrated_regime_probs_single(
        binding["calibration"]["per_seed_params"][pair_index], response["regime_logits"])
    pos = torch.as_tensor(indices, dtype=torch.long)
    sub = {key: scaled[key].index_select(0, pos) for key in (
        "history", "factual_suffix", "factual_actions", "counterfactual_actions")}
    full_t = torch.as_tensor(np.ascontiguousarray(full, dtype=np.float32))
    history_t = torch.as_tensor(np.ascontiguousarray(history, dtype=np.float32))
    router_features, _, _ = build_router_features(
        history=sub["history"], factual_suffix=sub["factual_suffix"],
        factual_actions=sub["factual_actions"],
        counterfactual_actions=sub["counterfactual_actions"],
        response_features=torch.as_tensor(response["response_features"]),
        regime_probs=torch.as_tensor(regime_probs),
        full_prediction=full_t, history_prediction=history_t)
    padded = torch.zeros((int(context_rows),) + tuple(router_features.shape[1:]),
                         dtype=router_features.dtype)
    padded[:branches] = router_features
    router = s4a._load_e2e_router(stage3r_dir, pair_index, seed).to(torch.device(device))
    with torch.no_grad():
        weight = router(padded.to(torch.device(device))).cpu()[:branches]
    return (weight * full_t + (1.0 - weight) * history_t).numpy().astype(np.float32)


def _context_aligned_combined_score(
    features: Mapping[str, np.ndarray],
    combined_model: s4r.FittedRiskModel,
    *,
    context_rows: int,
    groups: int,
    n_actions: int,
) -> np.ndarray:
    """COMBINED_RISK_HEAD score in R2's original CPU-forward context.

    The frozen COMBINED finalist is a torch MLP whose ``predict`` runs the whole
    design matrix in ONE CPU forward, so the BLAS kernel (and float32 accumulation
    order) depends on the row count. R2's on-disk train scores were produced at
    E = G_train*10 rows. Per-row MLP outputs depend only on their own input row given
    a fixed kernel, so zero-padding the standardized matrix to the original row count
    reproduces R2's first-E scores bit-for-bit (probe-verified). Uses ONLY the frozen
    model's own ``_standardize``/estimator/target statistics; nothing is refit.
    """
    matrix = s4r.design_matrix(features, s4r.CANDIDATE_FEATURES[s4r.CANDIDATE_COMBINED])
    standardized = combined_model._standardize(matrix)
    rows = int(standardized.shape[0])
    if int(context_rows) < rows:
        raise RuntimeError("combined-score context rows cannot be smaller than E")
    padded = np.zeros((int(context_rows), standardized.shape[1]), dtype=np.float64)
    padded[:rows] = standardized
    raw = combined_model.estimator.predict(padded)[:rows]
    score = np.asarray(raw, dtype=np.float64) * combined_model.target_std \
        + combined_model.target_mean
    return score.reshape(groups, n_actions)


def verify_adapter_consistency(
    config: Mapping[str, Any],
    n_groups: int = 32,
    *,
    run_inference_fn: Callable | None = None,
    build_features_fn: Callable | None = None,
    load_models_fn: Callable | None = None,
    score_actions_fn: Callable | None = None,
    device: str | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Verify this module reproduces R2's frozen implementation on the fixed first-32
    R2 train groups.

    Reads ONLY the legal input keys of the R2 train archive (its counterfactual-truth
    key is never accessed) plus R2's on-disk train predictions/scores. Recomputes the
    full frozen chain — predictions, action risk scores, relative-comparison features
    and Direct scores — and requires elementwise identity with R2's on-disk products.
    No CF truth, no label, no regret, no new scientific metric.
    """
    checked = prot.validate_config(config)
    n_groups = int(n_groups)
    if n_groups != int(checked["adapter_consistency_groups"]):
        raise ValueError(
            f"the adapter consistency check is fixed at "
            f"{checked['adapter_consistency_groups']} groups")
    prot.verify_r2_sources(_resolve(checked["r2_development_dir"]))
    run_inference_fn = run_inference_fn or s4ci.run_grouped_frozen_inference
    build_features_fn = build_features_fn or s4ci.build_action_risk_features
    load_models_fn = load_models_fn or s4ci.load_frozen_models
    score_actions_fn = score_actions_fn or s4ci.score_actions
    device = str(device or checked.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    batch_size = int(batch_size or checked.get("batch_size", 32))

    r2_dev = _resolve(checked["r2_development_dir"])
    train_data_path = _resolve(checked["r2_train_data"])
    if not train_data_path.is_file():
        raise FileNotFoundError(f"R2 train data missing: {train_data_path}")
    data = {key: value[:n_groups]
            for key, value in _read_legal_npz(
                train_data_path, _R2_TRAIN_LEGAL_KEYS, screen_keys=False).items()}
    bundle = _bundle_from_arrays(
        data, config=checked, data_signature=sha256_file(train_data_path),
        role="adapter_consistency")
    binding = _read_json(_resolve(checked["stage3b_binding"]))["receipt_payload"]
    stage3r_dir = _resolve(checked["stage3r_run_dir"])

    predictions = run_inference_fn(
        bundle, binding=binding, stage3r_dir=stage3r_dir,
        device=device, batch_size=batch_size)
    _validate_predictions(predictions, groups=n_groups, n_actions=prot.N_ACTIONS)

    ref_predictions_path = r2_dev / "train_predictions.npz"
    if not ref_predictions_path.is_file():
        raise FileNotFoundError(f"R2 on-disk train predictions missing: {ref_predictions_path}")
    with np.load(ref_predictions_path, allow_pickle=True) as archive:
        ref_predictions = {key: archive[key][:, :n_groups] for key in PREDICTION_KEYS}
        ref_group_seed = np.asarray(archive["group_seed"][:n_groups], dtype=np.int64)
    if not np.array_equal(ref_group_seed, bundle["group_seed"]):
        raise RuntimeError("adapter check group order misaligned with R2's on-disk train")

    # e2e is the router mixture; the frozen router forward is unbatched, so R2's
    # on-disk e2e lives in the M = G_train*10 kernel context. Recompute e2e per pair
    # in that exact context (zero-padded router input; GEMM row independence) so the
    # comparison is a true bitwise one — no tolerance is substituted for identity.
    branches = int(bundle["branches"])
    predictions = dict(predictions)
    e2e_aligned = np.empty_like(np.asarray(predictions["e2e"], dtype=np.float32))
    horizon, agents = prot.HORIZON, prot.OBJECT_CAPACITY
    with np.load(ref_predictions_path, allow_pickle=True) as archive:
        context_rows = int(archive["e2e"].shape[1]) * prot.N_ACTIONS
    for pair_index in range(len(prot.EXPERT_ROUTER_SEEDS)):
        full_pair = np.asarray(predictions["full"][pair_index], dtype=np.float32).reshape(
            branches, horizon, agents, 2)
        history_pair = np.asarray(
            predictions["history"][pair_index], dtype=np.float32).reshape(
            branches, horizon, agents, 2)
        e2e_pair = _shape_aligned_router_e2e(
            bundle, full_pair, history_pair, pair_index=pair_index, binding=binding,
            stage3r_dir=stage3r_dir, device=device, batch_size=batch_size,
            context_rows=context_rows)
        e2e_aligned[pair_index] = e2e_pair.reshape(
            n_groups, prot.N_ACTIONS, horizon, agents, 2)
    predictions["e2e"] = e2e_aligned
    max_abs = _compare_prediction_arrays(predictions, ref_predictions)

    features = build_features_fn(bundle, predictions)
    frozen_models = load_models_fn()
    scored = dict(score_actions_fn(
        features, frozen_models, groups=n_groups, n_actions=prot.N_ACTIONS,
        rng_seed=_r2_train_curve_reference_seed()))
    combined_model = frozen_models["COMBINED_RISK_HEAD"]
    if str(combined_model.implementation) == "mlp":
        # the MLP finalist's single CPU forward is row-count-context dependent;
        # reproduce R2's original E=G_train*10 kernel context for a true bitwise check.
        scored["COMBINED_RISK_HEAD"] = _context_aligned_combined_score(
            features, combined_model, context_rows=context_rows,
            groups=n_groups, n_actions=prot.N_ACTIONS)
    ref_scores_path = r2_dev / "train_scores.npz"
    if not ref_scores_path.is_file():
        raise FileNotFoundError(f"R2 on-disk train scores missing: {ref_scores_path}")
    with np.load(ref_scores_path, allow_pickle=True) as archive:
        ref_scores = {name: np.asarray(archive[name])[:n_groups] for name in s4cpp.RISK_CLASSES}
    action_risks: dict[str, np.ndarray] = {}
    for name in s4cpp.RISK_CLASSES:
        mine = np.asarray(scored[name], dtype=np.float64)
        ref = np.asarray(ref_scores[name], dtype=np.float64)
        if mine.shape != ref.shape or not np.array_equal(mine, ref):
            diff = float(np.max(np.abs(mine - ref))) if mine.shape == ref.shape else -1.0
            raise RuntimeError(
                f"adapter action score {name!r} is not elementwise identical to R2's "
                f"on-disk train scores (max abs diff {diff:.3e})")
        action_risks[name] = mine

    ours_tokens = s4cpf.build_relative_tokens(
        np.asarray(predictions["e2e"], dtype=np.float64), bundle["factual_xy"],
        bundle["history_last_xy"], bundle["inference_mask"], action_risks,
        bundle["geometry_group"], bundle["t_star"], bundle["group_seed"])
    ref_tokens = s4cpf.build_relative_tokens(
        np.asarray(ref_predictions["e2e"], dtype=np.float64), bundle["factual_xy"],
        bundle["history_last_xy"], bundle["inference_mask"], ref_scores,
        bundle["geometry_group"], bundle["t_star"], bundle["group_seed"])
    for key in ("tokens", "global_features", "j_hat", "j_hat_seeds",
                "interaction_min_distance"):
        if not np.array_equal(np.asarray(ours_tokens[key]), np.asarray(ref_tokens[key])):
            raise RuntimeError(f"adapter feature {key!r} drifted from the R2 implementation")
    if not np.array_equal(np.asarray(ours_tokens["reference_action"]),
                          np.asarray(ref_tokens["reference_action"])):
        raise RuntimeError("adapter reference actions drifted from the R2 implementation")

    inventory = load_frozen_model_inventory(checked)
    ours_direct = s4cr2t.score_models(
        inventory, {"tokens": ours_tokens["tokens"],
                    "global_features": ours_tokens["global_features"],
                    "group_seed": bundle["group_seed"]}, device=device)
    ref_direct = s4cr2t.score_models(
        inventory, {"tokens": ref_tokens["tokens"],
                    "global_features": ref_tokens["global_features"],
                    "group_seed": bundle["group_seed"]}, device=device)
    for key in ("risk_ensemble", "risk_by_seed", "mu_only_ensemble", "class_only_ensemble"):
        if not np.array_equal(np.asarray(getattr(ours_direct, key)),
                              np.asarray(getattr(ref_direct, key))):
            raise RuntimeError(f"adapter Direct score {key!r} drifted from the R2 pipeline")
    if not np.isfinite(ours_direct.risk_ensemble).all():
        raise RuntimeError("non-finite adapter Direct ensemble score")

    return {
        "plan_id": prot.PLAN_ID,
        "status": "ADAPTER_CONSISTENCY_VERIFIED",
        "n_groups": n_groups,
        "group_seeds": [int(s) for s in bundle["group_seed"]],
        "r2_train_data_sha256": sha256_file(train_data_path),
        "r2_train_predictions_sha256": sha256_file(ref_predictions_path),
        "r2_train_scores_sha256": sha256_file(ref_scores_path),
        "predictions_bitwise_identical": True,
        "action_scores_bitwise_identical": True,
        "features_bitwise_identical": True,
        "direct_scores_bitwise_identical": True,
        "max_abs_diff": float(max_abs),
        "router_context_rows": int(context_rows),
        "router_context_note": (
            "the frozen e2e router forward is unbatched; R2's on-disk train e2e lives "
            "in the M=G_train*10 cuBLAS kernel context, reproduced here by zero-padding "
            "the router input (GEMM row independence, probe-verified bitwise); every "
            "other predictor output is batched and bitwise identical without padding"),
        "device": device,
        "dtype": "float32",
        "batch_size": batch_size,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


__all__ = [
    "CONFIG_PATH",
    "FORBIDDEN_INPUT_KEY_TOKENS",
    "LEGAL_INPUT_KEYS",
    "PREDICTION_KEYS",
    "POSITION_SLICE",
    "STATUS_INFERENCE",
    "STATUS_SMOKE",
    "infer_and_score",
    "load_config",
    "load_frozen_model_inventory",
    "load_inference_bundle",
    "role_inference_paths",
    "role_input_path",
    "role_truth_path",
    "verify_adapter_consistency",
]
