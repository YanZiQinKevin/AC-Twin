"""Stage 4C-R2 frozen grouped inference + repeated-inference determinism.

Plan ``AC-TWIN-RFER-STAGE4C-R2-DECISION-RISK-SCALING-2026-09-16-V0.2`` sections 4, 6, 9.3.
The frozen five-seed-pair predictor/risk chain is REUSED bit-identically from Stage 4C
(``code/ac_twin/stage4c_inference.py``): the group's ten counterfactual branches are
flattened to ``E = G*10`` branches (group-major, action-minor) and scored by the same
``run_pair_inference`` forward passes; the five action-level risk classes
(``stage4cp_protocol.RISK_CLASSES``) are produced by the same frozen risk models, whose
checkpoints resolve through the receipt-pinned explicit paths in
``stage4c_protocol.STAGE4_FROZEN`` (never guessed from directories).

This layer produces ONLY the raw arrays needed downstream to construct the frozen
9x34-token + 9-global feature contract (``stage4cp_features.build_relative_tokens``):
per-seed predictor outputs (``e2e`` etc.), the five action-level risk classes and the
group geometry/presence arrays loaded from the collected data. It computes NO labels, NO
regret, NO oracle action and NO adaptive statistic (no scaler, class weight or
threshold); true counterfactual trajectories stay inside the bundle for the deferred
adjudication side and are never persisted by this layer.

Every data path passes the Task 1 three-layer guard (``assert_allowed_stage4cr2_data``)
before any read. Predictions/scores NPZs and inference receipts are create-once; a rerun
verifies hashes and never overwrites.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

import collect_stage4cr2_merge_multiaction as collector  # noqa: E402
import run_stage4a as s4a  # noqa: E402  (frozen pre-intervention geometry helper)
from ac_twin import stage4b_inference as s4bi  # noqa: E402
from ac_twin import stage4c_inference as s4ci  # noqa: E402
from ac_twin import stage4cp_protocol as s4cpp  # noqa: E402  (RISK_CLASSES, read-only)
from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402
from ac_twin.checkpointing import atomic_save_npz  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3_scaler import FeatureScaler  # noqa: E402
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402

CONFIG_PATH = ROOT / "experiments/stage4_reliability/stage4c_r2/stage4cr2_config.json"

POSITION_SLICE = s4a.POSITION_SLICE
DETERMINISM_PAIR = s4ci.DETERMINISM_PAIR  # 0

STATUS_INFERENCE = "INFERENCE_COMPLETE"
STATUS_SMOKE = "ENGINEERING_SMOKE_ONLY"
STATUS_DETERMINISM = "DETERMINISM_VERIFIED"

#: prediction-side arrays persisted in ``{role}_predictions.npz`` (plus group binding ids).
PREDICTION_KEYS = ("e2e", "full", "history", "r_hat", "presence_prob", "high_prob")

#: salt distinguishing the curve-reference CONSTANT_RANDOM stream from the collector's
#: action-sampling stream; deterministic and never gate-relevant.
_CURVE_REFERENCE_SALT = 17


@dataclass(frozen=True)
class InferenceReceipt:
    """One role's frozen-inference receipt (mirrors the create-once JSON on disk)."""
    plan_id: str
    role: str
    status: str
    groups: int
    n_actions: int
    branches: int
    data_sha256: str
    predictions_path: str
    predictions_sha256: str
    scores_path: str
    scores_sha256: str
    receipt_path: str
    pair_indices: tuple[int, ...]
    risk_classes: tuple[str, ...]
    binding_sha256: str
    scaler_sha256: str
    device: str
    dtype: str
    written_at: str
    verification: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, verification: str = "") -> "InferenceReceipt":
        data = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        data.pop("verification", None)
        for key in ("pair_indices", "risk_classes"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(verification=verification, **data)


@dataclass(frozen=True)
class DeterminismReceipt:
    """Result of the repeated-inference elementwise-identity check (plan section 9.3)."""
    plan_id: str
    role: str
    status: str
    pair_index: int
    n_groups: int
    group_indices: tuple[int, ...]
    group_seeds: tuple[int, ...]
    bitwise_identical: bool
    max_abs_diff: float
    device: str
    dtype: str
    data_sha256: str
    verified_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value)


def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load + validate the frozen R2 config."""
    return s4cr2.validate_config(json.loads(Path(config_path).read_text(encoding="utf-8")))


def role_inference_paths(config: Mapping[str, Any], role: str) -> tuple[Path, Path, Path]:
    """Resolve (predictions_npz, scores_npz, inference_receipt) for ``role``."""
    checked = s4cr2.validate_config(config)
    role = str(role)
    if role not in collector.ROLES:
        raise ValueError(f"unknown Stage 4C-R2 inference role: {role}")
    prefix = collector.ROLE_PREFIX[role]
    run_dir = _resolve(checked["run_dir"])
    if role == "smoke":
        base = run_dir / "smoke"
        return (base / "smoke_predictions.npz", base / "smoke_scores.npz",
                base / "smoke_inference_receipt.json")
    base = run_dir / "development"
    return (base / f"{prefix}_predictions.npz", base / f"{prefix}_scores.npz",
            base / f"{prefix}_inference_receipt.json")


def _binding(config: Mapping[str, Any]) -> dict[str, Any]:
    """The Stage 3A Plus input-binding receipt payload (explicit path from the config)."""
    return json.loads(_resolve(config["stage3b_binding"]).read_text(encoding="utf-8"))["receipt_payload"]


def _curve_reference_seed(checked: Mapping[str, Any], role: str) -> int:
    """Deterministic seed for the curve-reference CONSTANT_RANDOM scores (never gated)."""
    root = collector.action_stream_root(checked, role)
    return int(np.random.SeedSequence([*root, _CURVE_REFERENCE_SALT]).generate_state(1)[0])


# --- grouped data loading + branch expansion ------------------------------------------------


def load_role_bundle(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    """Load a collected R2 group npz into scaled branch tensors + geometry.

    Mirrors the frozen Stage 4C ``load_group_eval`` expansion (branch ``e = g*10 + a``)
    against the R2 namespace. The true counterfactual trajectories are loaded ONLY for
    the deferred adjudication side; they are never fed into a risk feature and never
    persisted by this layer. Read-only; the data path passes the Task 1 guard first.
    """
    checked = s4cr2.validate_config(config)
    role = str(role)
    if role not in collector.ROLES:
        raise ValueError(f"unknown Stage 4C-R2 inference role: {role}")
    data_output, _, _ = collector.role_paths(checked, role)
    s4cr2.assert_allowed_stage4cr2_data(data_output, role=role)
    if not data_output.is_file():
        raise FileNotFoundError(f"collected role data missing: {data_output}")
    with np.load(data_output, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}

    history = np.asarray(data["object_history"], dtype=np.float32)        # (G,10,5,8)
    factual = np.asarray(data["object_factual_future"], dtype=np.float32)  # (G,20,5,8)
    cf = np.asarray(data["object_counterfactual_futures"], dtype=np.float32)  # (G,10,20,5,8)
    cf_actions = np.asarray(data["cf_actions"], dtype=np.int64)            # (G,10)
    cf_durations = np.asarray(data["cf_durations"], dtype=np.int64)        # (G,10)
    factual_actions = np.asarray(data["factual_actions"], dtype=np.int64)  # (G,20)
    t_star = np.asarray(data["t_star"], dtype=np.int64)                    # (G,)
    episode_seed = np.asarray(data["episode_seed"], dtype=np.int64)        # (G,)
    inference_mask = np.asarray(data["inference_mask"], dtype=bool)        # (G,20,5)

    groups = history.shape[0]
    n_actions = cf.shape[1]
    if n_actions != s4cr2.N_ACTIONS:
        raise ValueError(f"expected {s4cr2.N_ACTIONS} actions per group, got {n_actions}")

    # Expand to E = G*10 branches (group-major, action-minor): branch e = g*10 + a.
    g_idx = np.repeat(np.arange(groups), n_actions)
    branches = groups * n_actions

    hist_b = history[g_idx]                      # (E,10,5,8)
    fact_b = factual[g_idx]                      # (E,20,5,8)
    factual_suffix = fact_b[:, :10]              # (E,10,5,8)
    factual_actions_b = factual_actions[g_idx]   # (E,20)
    cf_actions_b = cf_actions.reshape(branches)  # (E,)
    cf_durations_b = cf_durations.reshape(branches)  # (E,)
    t_star_b = t_star[g_idx]                     # (E,)
    # materialised counterfactual action plan per branch.
    cf_plans = np.stack([
        s4cr2.materialize_action_plan(factual_actions[g], int(cf_actions[g, a]),
                                      int(cf_durations[g, a]))
        for g in range(groups) for a in range(n_actions)
    ]).astype(np.int64)                          # (E,20)
    valid_b = inference_mask[g_idx]              # (E,20,5) shared across the 10 actions

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

    geometry_group = s4a._pre_intervention_geometry(history)  # per-group (G,...)
    geometry = {
        "distance_to_ego": geometry_group["distance_to_ego"][g_idx],
        "relative_speed": geometry_group["relative_speed"][g_idx],
        "visible_count": geometry_group["visible_count"][g_idx],
    }
    # interaction density: non-ego vehicles within 30 m of the ego at history last frame.
    last = history[:, -1]  # (G,5,8)
    ego_xy = last[:, 0, POSITION_SLICE]  # (G,2)
    dist = np.linalg.norm(last[:, :, POSITION_SLICE] - ego_xy[:, None, :], axis=-1)  # (G,5)
    present = last[..., 0] > 0
    present[:, 0] = False  # exclude ego
    density_count = ((dist <= 30.0) & present).sum(axis=1).astype(np.float64)  # (G,)
    return {
        "role": role,
        "groups": groups,
        "n_actions": n_actions,
        "branches": branches,
        "tensors": tensors,
        "scaled": scaled,
        "valid": valid_b,
        "geometry": geometry,                # branch-level (E,...) for the risk features
        "geometry_group": geometry_group,    # group-level (G,...) for build_relative_tokens
        "episode_seed": episode_seed,
        "t_star": t_star,
        "inference_mask": inference_mask,
        "density_count": density_count,
        "history_last_xy": history[:, -1, 0, POSITION_SLICE].astype(np.float64),  # (G,2)
        "factual_xy": factual[..., POSITION_SLICE].astype(np.float64),           # (G,20,5,2)
        # adjudication-side ONLY (never a feature input, never persisted here):
        "counterfactual_xy": cf[..., POSITION_SLICE].astype(np.float64),         # (G,10,20,5,2)
        "data_signature": sha256_file(data_output),
    }


# --- frozen grouped inference -----------------------------------------------------------------


def _validate_predictions(
    predictions: Mapping[str, np.ndarray], *, groups: int, n_actions: int
) -> None:
    """Fail closed on any missing/non-finite/mis-shaped predictor output."""
    for key in PREDICTION_KEYS:
        if key not in predictions:
            raise RuntimeError(f"frozen inference output missing key: {key}")
        arr = np.asarray(predictions[key])
        if not np.isfinite(arr.astype(np.float64)).all():
            raise RuntimeError(f"non-finite Stage 4C-R2 predictor output: {key}")
    e2e = np.asarray(predictions["e2e"])
    if e2e.ndim != 6 or e2e.shape[1] != groups or e2e.shape[2] != n_actions \
            or e2e.shape[-1] != 2:
        raise RuntimeError(f"e2e must be (S,G,A,T,N,2); got {e2e.shape}")


def infer_role(
    config: Mapping[str, Any],
    role: str,
    *,
    run_inference_fn: Callable | None = None,
    build_features_fn: Callable | None = None,
    load_models_fn: Callable | None = None,
    score_actions_fn: Callable | None = None,
    device: str | None = None,
    batch_size: int | None = None,
) -> InferenceReceipt:
    """Run the frozen predictor/risk chain over one role and persist the raw arrays.

    Writes create-once ``{role}_predictions.npz`` (per-seed predictor outputs + group
    binding ids) and ``{role}_scores.npz`` (exactly the five action-level risk classes +
    group seeds), then a create-once inference receipt. No label, regret, oracle action
    or adaptive statistic is computed. A rerun verifies hashes and never overwrites.
    """
    checked = s4cr2.validate_config(config)
    role = str(role)
    if role not in collector.ROLES:
        raise ValueError(f"unknown Stage 4C-R2 inference role: {role}")
    run_inference_fn = run_inference_fn or s4ci.run_grouped_frozen_inference
    build_features_fn = build_features_fn or s4ci.build_action_risk_features
    load_models_fn = load_models_fn or s4ci.load_frozen_models
    score_actions_fn = score_actions_fn or s4ci.score_actions
    device = str(device or checked.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    batch_size = int(batch_size or checked.get("batch_size", 32))

    pred_path, score_path, receipt_path = role_inference_paths(checked, role)
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for key, path in (("predictions_sha256", pred_path), ("scores_sha256", score_path)):
            if not path.is_file() or sha256_file(path) != receipt[key]:
                raise ValueError(f"inference artifact drifted from its receipt: {path}")
        return InferenceReceipt.from_dict(
            receipt, verification="existing_inference_verified_no_overwrite")
    if pred_path.exists() or score_path.exists():
        raise RuntimeError(
            "inference artifacts exist without their create-once receipt: "
            f"{pred_path}, {score_path}")

    bundle = load_role_bundle(checked, role)
    binding = _binding(checked)
    stage3r_dir = _resolve(checked["stage3r_run_dir"])
    predictions = run_inference_fn(
        bundle, binding=binding, stage3r_dir=stage3r_dir,
        device=device, batch_size=batch_size)
    _validate_predictions(predictions, groups=int(bundle["groups"]),
                          n_actions=int(bundle["n_actions"]))
    features = build_features_fn(bundle, predictions)
    frozen_models = load_models_fn()
    scored = score_actions_fn(
        features, frozen_models, groups=int(bundle["groups"]),
        n_actions=int(bundle["n_actions"]),
        rng_seed=_curve_reference_seed(checked, role))
    action_scores: dict[str, np.ndarray] = {}
    for name in s4cpp.RISK_CLASSES:
        value = np.asarray(scored[name], dtype=np.float64)
        if value.shape != (int(bundle["groups"]), int(bundle["n_actions"])) \
                or not np.isfinite(value).all():
            raise RuntimeError(f"non-finite Stage 4C-R2 action risk score: {name}")
        action_scores[name] = value

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    group_seed = np.asarray(bundle["episode_seed"], dtype=np.int64)
    atomic_save_npz(
        pred_path,
        **{k: np.asarray(predictions[k], dtype=np.float32) for k in PREDICTION_KEYS},
        group_seed=group_seed,
        t_star=np.asarray(bundle["t_star"], dtype=np.int64),
        cf_actions=np.tile([a for a, _ in s4cr2.ACTION_PLANS],
                           (int(bundle["groups"]), 1)).astype(np.int64),
        cf_durations=np.tile([d for _, d in s4cr2.ACTION_PLANS],
                             (int(bundle["groups"]), 1)).astype(np.int64),
    )
    atomic_save_npz(score_path, group_seed=group_seed, **action_scores)
    receipt = InferenceReceipt(
        plan_id=s4cr2.PLAN_ID,
        role=role,
        status=STATUS_SMOKE if role == "smoke" else STATUS_INFERENCE,
        groups=int(bundle["groups"]),
        n_actions=int(bundle["n_actions"]),
        branches=int(bundle["branches"]),
        data_sha256=str(bundle["data_signature"]),
        predictions_path=str(pred_path),
        predictions_sha256=sha256_file(pred_path),
        scores_path=str(score_path),
        scores_sha256=sha256_file(score_path),
        receipt_path=str(receipt_path),
        pair_indices=(0, 1, 2, 3, 4),
        risk_classes=tuple(s4cpp.RISK_CLASSES),
        binding_sha256=sha256_file(_resolve(checked["stage3b_binding"])),
        scaler_sha256=sha256_file(_resolve(checked["scaler_path"])),
        device=device,
        dtype="float32",
        written_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    write_once_json(receipt_path, receipt.to_dict())
    return receipt


# --- repeated-inference determinism (plan section 9.3, step 2) -------------------------------


def verify_repeated_inference(
    role: str,
    group_indices: np.ndarray,
    *,
    config: Mapping[str, Any] | None = None,
    bundle: Mapping[str, Any] | None = None,
    pair_inference_fn: Callable | None = None,
    device: str | None = None,
    batch_size: int | None = None,
) -> DeterminismReceipt:
    """Re-run frozen pair 0 twice over ``group_indices``; require elementwise identity.

    Raises ``RuntimeError`` unless every predictor output is bitwise identical across the
    two runs. ``group_indices`` are positional indices into the role's groups (the caller
    fixes which groups — for the screen this is the frozen 32-group set).
    """
    checked = s4cr2.validate_config(config) if config is not None else load_config()
    role = str(role)
    if role not in collector.ROLES:
        raise ValueError(f"unknown Stage 4C-R2 inference role: {role}")
    pair_inference_fn = pair_inference_fn or s4bi.run_pair_inference
    device = str(device or checked.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    batch_size = int(batch_size or checked.get("batch_size", 32))
    bundle = bundle if bundle is not None else load_role_bundle(checked, role)

    group_indices = np.asarray(group_indices, dtype=np.int64)
    groups = int(bundle["groups"])
    if group_indices.ndim != 1 or group_indices.size == 0:
        raise ValueError("group_indices must be a non-empty 1-D array")
    if np.unique(group_indices).size != group_indices.size:
        raise ValueError("group_indices must not contain duplicates")
    if int(group_indices.min()) < 0 or int(group_indices.max()) >= groups:
        raise ValueError(f"group_indices out of range [0, {groups})")

    n_actions = int(bundle["n_actions"])
    branch_subset = (group_indices[:, None] * n_actions
                     + np.arange(n_actions)[None, :]).ravel()
    binding = _binding(checked)
    stage3r_dir = _resolve(checked["stage3r_run_dir"])
    forward_config = {"device": device, "batch_size": batch_size}
    runs = [
        pair_inference_fn(forward_config, bundle["scaled"], branch_subset,
                          pair_index=DETERMINISM_PAIR, binding=binding,
                          stage3r_dir=stage3r_dir)
        for _ in range(2)
    ]
    if set(runs[0]) != set(runs[1]):
        raise RuntimeError("repeated inference returned different output keys")
    max_abs = 0.0
    for key in runs[0]:
        first = np.asarray(runs[0][key])
        second = np.asarray(runs[1][key])
        if first.shape != second.shape:
            raise RuntimeError(f"repeated inference shape mismatch on {key}")
        diff = float(np.max(np.abs(first.astype(np.float64) - second.astype(np.float64)))) \
            if first.size else 0.0
        max_abs = max(max_abs, diff)
        if not np.array_equal(first, second):
            raise RuntimeError(
                f"Stage 4C-R2 repeated inference is not elementwise identical "
                f"(key {key!r}, max abs diff {diff:.3e})")
    return DeterminismReceipt(
        plan_id=s4cr2.PLAN_ID,
        role=role,
        status=STATUS_DETERMINISM,
        pair_index=DETERMINISM_PAIR,
        n_groups=int(group_indices.size),
        group_indices=tuple(int(i) for i in group_indices),
        group_seeds=tuple(int(s) for s in np.asarray(bundle["episode_seed"])[group_indices]),
        bitwise_identical=True,
        max_abs_diff=float(max_abs),
        device=device,
        dtype="float32",
        data_sha256=str(bundle["data_signature"]),
        verified_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


__all__ = [
    "CONFIG_PATH",
    "DETERMINISM_PAIR",
    "DeterminismReceipt",
    "InferenceReceipt",
    "PREDICTION_KEYS",
    "STATUS_DETERMINISM",
    "STATUS_INFERENCE",
    "STATUS_SMOKE",
    "infer_role",
    "load_config",
    "load_role_bundle",
    "role_inference_paths",
    "verify_repeated_inference",
]
