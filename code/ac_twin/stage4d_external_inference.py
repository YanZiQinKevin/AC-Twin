"""Stage 4D-D2 in-platform base-predictor retraining + frozen production inference.

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` sections 7.2, 7.3, 7.5
and 12.4 (Task 3). This module RE-TRAINS the frozen Stage 3A Plus / Stage 3R / R2 base
pipeline on the MetaDrive/Waymo train role and then applies it as a frozen stack:

    1. Full predictor        (Stage3RExpert, condition="full")
    2. History predictor     (Stage3RExpert, condition="history")
    3. End-to-end predictor  (CEWRouter, end_to_end_mixture lambda_cew=0 lambda_rank=0)
    4. response predictor/features (Stage3ResponsePredictor h96_r32_l0.1) + validation
       calibration (per-seed vector scaling)
    5. the three FITTED risk classes (COMBINED_RISK_HEAD mlp, GEOMETRY_ONLY ridge,
       RESPONSE_ONLY ridge); UNCERTAINTY_ONLY / DISAGREEMENT_ONLY are unfitted direct
       aggregations reused read-only.

It is a platform-specific retraining, NOT a zero-shot transfer of the Merge checkpoints
(plan Global Constraint 2). Every architecture, loss, optimizer, epoch cap, early-stopping
rule and input binding is mechanically migrated from the frozen upstream modules, imported
READ-ONLY (plan Global Constraint 1 — nothing upstream is modified).

Permitted platform differences (plan 7.2): a platform train-only scaler, explicit SI/numeric
conversions, new-platform checkpoints and new-platform model seeds. No new layers, no wider
hidden width, no loss swap, no map encoder, no post-screen model edits.

Early-stopping monitor (plan 7.5): every supervised component trains on the 5,000 train
groups and early-stops / selects its checkpoint on the 1,500 validation role ("validation
仅用于 early stopping、机械 checkpoint"). The scaler and the response class weight are fit on
TRAIN only; calibration is fit on validation only (plan 7.5 "非阻断 calibration"). The plan
registers NO out-of-fold split (7.2 trains all five components on the 5,000 train groups and
7.5 allows train truth for supervision); if in-sample optimism collapsed the train regret
signal, the plan-7.5 train-support gate fails closed to STAGE4D_EXTERNAL_TRAIN_SUPPORT_INSUFFICIENT.

Coordinate note (plan 5.5): the D2 collector stores Map-Frenet channels
[presence, is_ego, s, d, v_s, v_d, cos_rel, sin_rel]; POSITION_SLICE=[2,3] is therefore (s,d)
in metres, which is exactly where the upstream models/readers expect the position channels
(POSITION_FEATURES=(2,3)). The consequence/objective is translation-invariant, so (s,d)
metres drop in directly (plan 5.5 scales 5.0/2.0 m). All architectures are coordinate-agnostic
8-dim->effect networks retrained here with a refit train-only scaler.

Masks (frozen, two distinct masks replicated from upstream): base-predictor TRAINING and
risk-head FITTING use the cf-dependent ``compute_effect_eval_mask`` (run_stage3a /
run_stage4a.cmd_fit_risk); risk SCORING and the Direct consequence use the collector's
cf-independent ``build_inference_mask`` (stage4c_inference / stage4cr2_inference).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage3_data as s3d  # noqa: E402
from ac_twin import stage3_response as s3r
from ac_twin import stage3_training as s3t
from ac_twin import stage4_risk as s4risk
from ac_twin import stage4c_objective as s4co
from ac_twin import stage4cp_features as s4cpf
from ac_twin import stage4cp_protocol as s4cp
from ac_twin import stage4d_external_training as T
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3_scaler import FeatureScaler
from ac_twin.stage3a_plus_calibration import fit_vector_scaling
from ac_twin.stage3b_bridge import calibrated_regime_probs_single, predictor_features_ordered
from ac_twin.stage3r_models import CEWRouter, Stage3RExpert, build_router_features
from ac_twin.stage3r_protocol import write_once_json
from ac_twin.stage4c_protocol import materialize_action_plan
from ac_twin.checkpointing import atomic_save_npz

import run_stage4a as s4a  # noqa: E402  frozen Stage 4A helpers (POSITION_SLICE, _pre_intervention_geometry)

RUN_DIR = ROOT / "experiments/stage4_reliability/stage4d_external"
ROLES_DIR = RUN_DIR / "roles"

PLAN_ID = T.PLAN_ID
ROLE_SALT = "AC-TWIN-D2-V0.1|"  # must equal collect_stage4d_external.ROLE_SALT (bound by test)

# --- frozen task geometry (identical to Merge/R2) ---------------------------------------------
N_ACTIONS = T.N_ACTIONS              # 10
HISTORY_LEN = 10
FUTURE_LEN = 20
OBJECT_CAPACITY = 5
N_FEATURES = 8
POSITION_SLICE = s4a.POSITION_SLICE  # slice(2,4) -> D2 (s, d) in metres
PRIMARY_CONSTANT_W = 0.60            # stage4_protocol.PRIMARY_CONSTANT_W (frozen)

# --- frozen seed pairing (plan 7.3; index-aligned, no 5x5 cross) ------------------------------
RESPONSE_PREDICTOR_SEEDS = T.RESPONSE_PREDICTOR_SEEDS   # (101,202,303,404,505)
PREDICTOR_PAIR_SEEDS = T.PREDICTOR_PAIR_SEEDS           # (1101,1202,1303,1404,1505)
N_SEEDS = len(RESPONSE_PREDICTOR_SEEDS)

# --- frozen training mechanics (plan 7.2 mechanical migration) ---------------------------------
# response predictor (R2-frozen h96_r32_l0.1; weight_decay / lambda_brier are the defaults 0.0)
RESPONSE_HIDDEN_DIM = 96
RESPONSE_FEATURE_DIM = 32
RESPONSE_LAMBDA_CLS = 0.1
# experts / router (Stage 3R frozen)
EXPERT_HIDDEN_DIM = 64
EXPERT_OBJECT_EMBEDDING_DIM = 64
ROUTER_HIDDEN_DIM = 64
ROUTER_INPUT_DIM = 70
# shared base-predictor optimizer/schedule (Stage 3A/3R frozen values)
BASE_MAX_EPOCHS = 40
BASE_EARLY_STOP_PATIENCE = 6
BASE_LEARNING_RATE = 1e-3
BASE_BATCH_SIZE = 32
BASE_WEIGHT_DECAY = 0.0

# risk-head fitting (Stage 4A frozen): COMBINED=mlp (frozen internal defaults), GEO/RESP=ridge
RISK_FIT_CONFIG = {"ridge_alpha": 1.0, "winsor_lower_q": 0.01, "winsor_upper_q": 0.99}
COMBINED_IMPLEMENTATION = "mlp"

FIT_COMPONENTS = ("response", "expert_full", "expert_history", "router")


# --- verbatim frozen loss helpers ---------------------------------------------------------------
# ``masked_expert_mse`` is reproduced verbatim from run_stage3r.py:56 (frozen); it is
# reimplemented here rather than imported so this module does not pull the heavy Stage 3R
# driver (stage3r_teacher / stage3r_screen) into the production path. A test asserts
# bit-exact equality against run_stage3r.masked_expert_mse on random tensors.


def masked_expert_mse(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[:-1] != valid.shape:
        raise ValueError("prediction/target/valid shapes must align as E,T,N,2 / E,T,N,2 / E,T,N")
    mask = valid.to(prediction.dtype).unsqueeze(-1)
    denominator = (mask.sum() * prediction.shape[-1]).clamp_min(1.0)
    return (((prediction - target) ** 2) * mask).sum() / denominator


def router_mixture_mse(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """end_to_end_mixture router objective with lambda_cew=0 / lambda_rank=0.

    This is EXACTLY run_stage3r.router_objective under the frozen ("end_to_end_mixture",
    0.0, 0.0) candidate: the CEW-distillation and pairwise-rank terms are multiplied by 0,
    reducing the objective to ``masked_expert_mse(weight*full + (1-weight)*history, target,
    valid)``. No teacher/rank tensors are needed because both weights are zero.
    """
    weight = model(batch["features"]).squeeze(-1)                 # E,T,N
    mixed = weight.unsqueeze(-1) * batch["full"] + (1.0 - weight.unsqueeze(-1)) * batch["history"]
    return masked_expert_mse(mixed, batch["effect_target"], batch["valid"])


def _expert_loss(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """run_stage3r.expert_loss verbatim (masked effect MSE on the four model inputs)."""
    prediction = model(
        history=batch["history"],
        factual_suffix=batch["factual_suffix"],
        factual_actions=batch["factual_actions"],
        counterfactual_actions=batch["counterfactual_actions"],
    )
    return masked_expert_mse(prediction, batch["effect_target"], batch["valid"])


# --- data loading + branch expansion ------------------------------------------------------------


@dataclass(frozen=True)
class RoleArrays:
    """Raw collected arrays for one D2 role plus the derived group binding ids."""
    role: str
    groups: int
    n_actions: int
    scenario_ids: tuple[str, ...]
    group_seed: np.ndarray          # (G,) int64 from scenario_id
    object_history: np.ndarray      # (G,10,5,8) float32
    object_factual_future: np.ndarray  # (G,20,5,8) float32
    object_counterfactual_futures: np.ndarray  # (G,10,20,5,8) float32
    factual_actions: np.ndarray     # (G,20) int64
    inference_mask: np.ndarray      # (G,20,5) bool (collector's build_inference_mask)
    t_star: np.ndarray              # (G,) int64
    cf_actions: np.ndarray          # (10,) int64 canonical
    cf_durations: np.ndarray        # (10,) int64 canonical
    data_sha256: str


def load_role_arrays(role_dir: str | Path, role: str) -> RoleArrays:
    """Load a collected D2 role (``groups.npz`` + ``groups_meta.json``) read-only.

    The D2 npz stores no integer episode seed; the group binding id is derived from the
    scenario ID via ``stage4d_external_training.group_seed_from_scenario_id`` (a pure
    function of the scenario ID, identical across branches and reruns).
    """
    role_dir = Path(role_dir)
    npz_path = role_dir / "groups.npz"
    meta_path = role_dir / "groups_meta.json"
    if not npz_path.is_file():
        raise FileNotFoundError(f"collected D2 role data missing: {npz_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"collected D2 role meta missing: {meta_path}")
    with np.load(npz_path, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    scenario_ids = tuple(str(m["scenario_id"]) for m in meta)
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError(f"role {role} has duplicate scenario_ids")

    history = np.asarray(data["object_history"], dtype=np.float32)
    factual = np.asarray(data["object_factual_future"], dtype=np.float32)
    cf = np.asarray(data["object_counterfactual_futures"], dtype=np.float32)
    groups = int(history.shape[0])
    n_actions = int(cf.shape[1])
    if n_actions != N_ACTIONS:
        raise ValueError(f"expected {N_ACTIONS} actions per group, got {n_actions}")
    if not (factual.shape[0] == cf.shape[0] == groups == len(scenario_ids)):
        raise ValueError("role arrays and meta must align on the group axis")

    group_seed = np.asarray(
        [T.group_seed_from_scenario_id(sid, ROLE_SALT) for sid in scenario_ids],
        dtype=np.int64)
    return RoleArrays(
        role=str(role), groups=groups, n_actions=n_actions, scenario_ids=scenario_ids,
        group_seed=group_seed, object_history=history, object_factual_future=factual,
        object_counterfactual_futures=cf,
        factual_actions=np.asarray(data["factual_actions"], dtype=np.int64),
        inference_mask=np.asarray(data["inference_mask"], dtype=bool),
        t_star=np.asarray(data["t_star"], dtype=np.int64),
        cf_actions=np.asarray(data["cf_actions"], dtype=np.int64).ravel(),
        cf_durations=np.asarray(data["cf_durations"], dtype=np.int64).ravel(),
        data_sha256=sha256_file(npz_path),
    )


@dataclass(frozen=True)
class BranchBundle:
    """Branch-expanded (E = G*10, group-major action-minor) tensors + masks + targets."""
    role: str
    groups: int
    n_actions: int
    branches: int
    group_seed: np.ndarray           # (G,)
    scenario_ids: tuple[str, ...]
    # branch-level model inputs (E,...)
    history: np.ndarray              # (E,10,5,8) raw
    factual_suffix: np.ndarray       # (E,10,5,8) raw
    factual_actions: np.ndarray      # (E,20)
    counterfactual_actions: np.ndarray  # (E,20)
    cf_action: np.ndarray            # (E,)
    cf_duration: np.ndarray          # (E,)
    t_star: np.ndarray               # (E,)
    # training/eval side (RAW units)
    valid_eval: np.ndarray           # (E,20,5) compute_effect_eval_mask (cf-dependent)
    r_true: np.ndarray               # (E,20,5) compute_response_pos
    regime_labels: np.ndarray        # (E,20,5)
    effect_target: np.ndarray        # (E,20,5,2) cf_xy - factual_xy
    # inference side (group-level + collector mask)
    inference_mask: np.ndarray       # (G,20,5) cf-independent collector mask
    valid_infer: np.ndarray          # (E,20,5) inference_mask expanded to branches
    t_star_group: np.ndarray         # (G,) group-level t_star
    history_last_xy: np.ndarray      # (G,2)
    factual_xy: np.ndarray           # (G,20,5,2)
    counterfactual_xy: np.ndarray    # (G,10,20,5,2) TRUE (adjudication/fit only)
    geometry_group: dict[str, np.ndarray]   # group-level (G,N)/(G,)
    geometry_branch: dict[str, np.ndarray]  # branch-level (E,N)/(E,)
    data_sha256: str


def build_branch_bundle(arrays: RoleArrays) -> BranchBundle:
    """Expand group arrays to E = G*10 branches and build all masks/targets.

    Branch ``e = g*10 + a`` (group-major, action-minor), matching the frozen R2/Stage 4C
    expansion. The counterfactual plan is rebuilt with the same canonical
    ``materialize_action_plan`` the collector used. The cf-dependent eval mask / response
    truth / effect target are computed fresh from the RAW arrays (run_stage3a parity); the
    collector's cf-independent inference mask is carried through untouched.
    """
    groups, n_actions = arrays.groups, arrays.n_actions
    branches = groups * n_actions
    g_idx = np.repeat(np.arange(groups), n_actions)                # (E,) group of each branch

    history_b = arrays.object_history[g_idx]                       # (E,10,5,8)
    factual_b = arrays.object_factual_future[g_idx]                # (E,20,5,8)
    cf_b = arrays.object_counterfactual_futures.reshape(
        branches, FUTURE_LEN, OBJECT_CAPACITY, N_FEATURES)         # (E,20,5,8)
    factual_suffix = factual_b[:, :HISTORY_LEN]                    # (E,10,5,8)
    factual_actions_b = arrays.factual_actions[g_idx]              # (E,20)
    cf_actions_b = np.tile(arrays.cf_actions, groups)              # (E,) action of plan a
    cf_durations_b = np.tile(arrays.cf_durations, groups)          # (E,)
    t_star_b = arrays.t_star[g_idx]                                # (E,)

    cf_plans = np.stack([
        materialize_action_plan(arrays.factual_actions[g], int(arrays.cf_actions[a]),
                                int(arrays.cf_durations[a]))
        for g in range(groups) for a in range(n_actions)
    ]).astype(np.int64)                                            # (E,20)

    # cf-dependent evaluation mask + response truth + effect target (RAW, run_stage3a parity)
    valid_eval = s3d.compute_effect_eval_mask(history_b, factual_b, cf_b)   # (E,20,5)
    r_true = s3d.compute_response_pos(factual_b, cf_b)                       # (E,20,5)
    regime_labels = s3d.compute_regime_labels(r_true)                        # (E,20,5)
    factual_xy_b = factual_b[..., POSITION_SLICE].astype(np.float32)         # (E,20,5,2)
    cf_xy_b = cf_b[..., POSITION_SLICE].astype(np.float32)                   # (E,20,5,2)
    effect_target, _ = s4risk.build_effect_target(factual_xy_b, cf_xy_b, valid_eval)

    geometry_group = s4a._pre_intervention_geometry(arrays.object_history)   # group-level
    geometry_branch = {
        "distance_to_ego": geometry_group["distance_to_ego"][g_idx],
        "relative_speed": geometry_group["relative_speed"][g_idx],
        "visible_count": geometry_group["visible_count"][g_idx],
    }
    return BranchBundle(
        role=arrays.role, groups=groups, n_actions=n_actions, branches=branches,
        group_seed=arrays.group_seed, scenario_ids=arrays.scenario_ids,
        history=history_b, factual_suffix=factual_suffix,
        factual_actions=factual_actions_b, counterfactual_actions=cf_plans,
        cf_action=cf_actions_b, cf_duration=cf_durations_b, t_star=t_star_b,
        valid_eval=valid_eval, r_true=r_true, regime_labels=regime_labels,
        effect_target=effect_target,
        inference_mask=arrays.inference_mask, valid_infer=arrays.inference_mask[g_idx],
        t_star_group=arrays.t_star,
        history_last_xy=arrays.object_history[:, -1, 0, POSITION_SLICE].astype(np.float64),
        factual_xy=arrays.object_factual_future[..., POSITION_SLICE].astype(np.float64),
        counterfactual_xy=arrays.object_counterfactual_futures[..., POSITION_SLICE].astype(np.float64),
        geometry_group=geometry_group, geometry_branch=geometry_branch,
        data_sha256=arrays.data_sha256,
    )


# --- scaler (fit on TRAIN only) ------------------------------------------------------------------


def fit_scaler(train: BranchBundle, path: str | Path) -> FeatureScaler:
    """Fit the platform train-only FeatureScaler on train history+factual_suffix presence.

    Reuses the frozen ``FeatureScaler`` unchanged (it scales only motion channels (2,3,4,5)
    = D2 s,d,v_s,v_d). Create-once: a rerun reloads and verifies the persisted scaler.
    """
    path = Path(path)
    if path.is_file():
        return FeatureScaler.load(path)
    arrays = {"history": train.history, "factual_suffix": train.factual_suffix}
    masks = {
        "history": train.history[..., 0] > 0,
        "factual_suffix": train.factual_suffix[..., 0] > 0,
    }
    scaler = FeatureScaler.fit(arrays, masks, fitted_on="train")
    path.parent.mkdir(parents=True, exist_ok=True)
    scaler.save(path)
    return scaler


def scaled_model_tensors(bundle: BranchBundle, scaler: FeatureScaler) -> dict[str, torch.Tensor]:
    """The scaled branch model-input tensor dict (history/factual_suffix standardised)."""
    return {
        "history": torch.as_tensor(scaler.transform(bundle.history), dtype=torch.float32),
        "factual_suffix": torch.as_tensor(scaler.transform(bundle.factual_suffix), dtype=torch.float32),
        "factual_actions": torch.as_tensor(bundle.factual_actions, dtype=torch.long),
        "counterfactual_actions": torch.as_tensor(bundle.counterfactual_actions, dtype=torch.long),
        "cf_action": torch.as_tensor(bundle.cf_action, dtype=torch.long),
        "cf_duration": torch.as_tensor(bundle.cf_duration, dtype=torch.long),
        "t_star": torch.as_tensor(bundle.t_star, dtype=torch.long),
    }


# --- resumable training drivers (frozen ``train_model`` loop) ------------------------------------


def _make_train_batches(tensors: Mapping[str, torch.Tensor], indices: np.ndarray,
                        batch_size: int, seed: int) -> list[dict[str, torch.Tensor]]:
    """Shuffled full-episode batches (run_stage3a.make_batches parity)."""
    indices = np.asarray(indices).copy()
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(indices), generator=generator).numpy()
    batches = []
    for start in range(0, len(order), batch_size):
        sel = indices[order[start:start + batch_size]]
        batches.append({key: value[sel] for key, value in tensors.items()})
    return batches


def _make_eval_batches(tensors: Mapping[str, torch.Tensor], indices: np.ndarray,
                       batch_size: int) -> list[dict[str, torch.Tensor]]:
    """Ordered (never shuffled) eval batches (run_stage3a.make_eval_batches parity)."""
    indices = np.asarray(indices)
    out = []
    for start in range(0, len(indices), batch_size):
        sel = indices[start:start + batch_size]
        out.append({key: value[sel] for key, value in tensors.items()})
    return out


def _fingerprint(component: str, seed: int, data_sha256: str) -> str:
    return f"stage4d-d2-{component}-seed{seed}-{data_sha256[:16]}"


def fit_response_predictors(train: BranchBundle, validation: BranchBundle,
                            scaler: FeatureScaler, model_dir: str | Path, *,
                            device: str, progress: Callable[[str], None] | None = None
                            ) -> dict[int, str]:
    """Train the five response predictors (seeds 101..505), early-stop on validation.

    Returns ``{seed: checkpoint_path}``. Checkpoints use the frozen wrapped
    ``save_predictor_checkpoint`` format (state_dict + signature + meta.config). Create-once
    per checkpoint; resumable via the frozen ``train_model`` checkpoint.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    def _tensors(bundle: BranchBundle) -> dict[str, torch.Tensor]:
        t = scaled_model_tensors(bundle, scaler)
        t["r_true"] = torch.as_tensor(bundle.r_true, dtype=torch.float32)
        t["regime_labels"] = torch.as_tensor(bundle.regime_labels, dtype=torch.long)
        t["valid"] = torch.as_tensor(bundle.valid_eval, dtype=torch.bool)
        return t

    train_t = _tensors(train)
    val_t = _tensors(validation)
    train_idx = np.arange(train.branches, dtype=np.int64)
    val_idx = np.arange(validation.branches, dtype=np.int64)
    class_weights = s3r.compute_class_weights(
        train_t["regime_labels"].numpy(), train_t["valid"].numpy())
    cw = torch.as_tensor(class_weights)

    def loss_fn(model, batch):
        out = model(batch["history"], batch["factual_suffix"], batch["factual_actions"],
                    batch["counterfactual_actions"], batch["cf_action"],
                    batch["cf_duration"], batch["t_star"])
        weights = cw.to(out["regime_logits"].device)
        return s3r.response_masked_loss(out, batch["r_true"], batch["regime_labels"],
                                        batch["valid"], weights, lambda_cls=RESPONSE_LAMBDA_CLS)

    def sums_fn(model, batch):
        out = model(batch["history"], batch["factual_suffix"], batch["factual_actions"],
                    batch["counterfactual_actions"], batch["cf_action"],
                    batch["cf_duration"], batch["t_star"])
        weights = cw.to(out["regime_logits"].device)
        return s3r.response_loss_sums(out, batch["r_true"], batch["regime_labels"],
                                      batch["valid"], weights)

    paths: dict[int, str] = {}
    for seed in RESPONSE_PREDICTOR_SEEDS:
        target = model_dir / f"response_h{RESPONSE_HIDDEN_DIM}_r{RESPONSE_FEATURE_DIM}_seed{seed}.pt"
        if target.is_file():
            s3t.verify_checkpoint_integrity(target)  # create-once: never retrain
            paths[int(seed)] = str(target)
            continue
        batches = _make_train_batches(train_t, train_idx, BASE_BATCH_SIZE, seed)
        val_batches = _make_eval_batches(val_t, val_idx, BASE_BATCH_SIZE)
        s3t.seed_all(seed)
        model = s3r.Stage3ResponsePredictor(hidden_dim=RESPONSE_HIDDEN_DIM,
                                            response_feature_dim=RESPONSE_FEATURE_DIM)
        result = s3t.train_model(
            model, loss_fn, batches, val_batches, seed=int(seed),
            max_epochs=BASE_MAX_EPOCHS, early_stopping_patience=BASE_EARLY_STOP_PATIENCE,
            learning_rate=BASE_LEARNING_RATE, device=device, weight_decay=BASE_WEIGHT_DECAY,
            progress_prefix=f"d2-response:{seed}",
            checkpoint_path=model_dir / f".response_seed{seed}.ckpt", resume=True,
            run_fingerprint=_fingerprint("response", int(seed), train.data_sha256),
            val_sums_fn=sums_fn, selection_fn=s3r.selection_score)
        meta = {"run": f"stage4d_d2_response_seed{seed}", "seed": int(seed),
                "config": {"hidden_dim": RESPONSE_HIDDEN_DIM,
                           "response_feature_dim": RESPONSE_FEATURE_DIM,
                           "lambda_cls": RESPONSE_LAMBDA_CLS},
                "plan_id": PLAN_ID}
        record = s3t.save_predictor_checkpoint(target, result["state_dict"], meta=meta)
        paths[int(seed)] = str(record["path"])
        if progress is not None:
            progress(f"response seed={seed} best_val_loss={result['best_val_loss']:.6f} "
                     f"epochs={result.get('epochs_run', '?')}")
    return paths


def fit_experts(train: BranchBundle, validation: BranchBundle, scaler: FeatureScaler,
                model_dir: str | Path, *, device: str,
                progress: Callable[[str], None] | None = None) -> dict[tuple[int, str], str]:
    """Train the five Full/History expert pairs (seeds 1101..1505), early-stop on validation.

    Returns ``{(seed, condition): checkpoint_path}`` with plain ``state_dict`` checkpoints.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    def _tensors(bundle: BranchBundle) -> dict[str, torch.Tensor]:
        t = scaled_model_tensors(bundle, scaler)
        t["effect_target"] = torch.as_tensor(bundle.effect_target, dtype=torch.float32)
        t["valid"] = torch.as_tensor(bundle.valid_eval, dtype=torch.bool)
        return t

    train_t = _tensors(train)
    val_t = _tensors(validation)
    train_idx = np.arange(train.branches, dtype=np.int64)
    val_idx = np.arange(validation.branches, dtype=np.int64)

    paths: dict[tuple[int, str], str] = {}
    for seed in PREDICTOR_PAIR_SEEDS:
        for condition in ("full", "history"):
            target = model_dir / f"expert_{condition}_seed{seed}.pt"
            if target.is_file():
                paths[(int(seed), condition)] = str(target)  # create-once
                continue
            batches = _make_train_batches(train_t, train_idx, BASE_BATCH_SIZE, seed)
            val_batches = _make_eval_batches(val_t, val_idx, BASE_BATCH_SIZE)
            s3t.seed_all(seed)
            model = Stage3RExpert(condition=condition, hidden_dim=EXPERT_HIDDEN_DIM,
                                  object_embedding_dim=EXPERT_OBJECT_EMBEDDING_DIM)
            result = s3t.train_model(
                model, _expert_loss, batches, val_batches, seed=int(seed),
                max_epochs=BASE_MAX_EPOCHS, early_stopping_patience=BASE_EARLY_STOP_PATIENCE,
                learning_rate=BASE_LEARNING_RATE, device=device, weight_decay=BASE_WEIGHT_DECAY,
                progress_prefix=f"d2-expert-{condition}:{seed}",
                checkpoint_path=model_dir / f".expert_{condition}_seed{seed}.ckpt", resume=True,
                run_fingerprint=_fingerprint(f"expert-{condition}", int(seed), train.data_sha256))
            torch.save({k: v.detach().cpu() for k, v in result["state_dict"].items()}, target)
            paths[(int(seed), condition)] = str(target)
            if progress is not None:
                progress(f"expert {condition} seed={seed} best_val_loss="
                         f"{result['best_val_loss']:.6f}")
    return paths


# --- frozen model loading + forward passes --------------------------------------------------------


def _load_response_model(path: str | Path) -> s3r.Stage3ResponsePredictor:
    meta = s3t.verify_checkpoint_integrity(path)["meta"]
    cfg = meta["config"]
    model = s3r.Stage3ResponsePredictor(hidden_dim=int(cfg["hidden_dim"]),
                                        response_feature_dim=int(cfg["response_feature_dim"]))
    model.load_state_dict(s3t.load_predictor_state(path))
    return model.eval()


def _load_expert_model(path: str | Path, condition: str) -> Stage3RExpert:
    model = Stage3RExpert(condition=condition, hidden_dim=EXPERT_HIDDEN_DIM,
                          object_embedding_dim=EXPERT_OBJECT_EMBEDDING_DIM)
    model.load_state_dict(torch.load(Path(path), map_location="cpu", weights_only=True))
    return model.eval()


def _load_router_model(path: str | Path) -> CEWRouter:
    meta_path = Path(path).with_suffix(".json")
    input_dim = int(json.loads(meta_path.read_text(encoding="utf-8"))["input_dim"])
    model = CEWRouter(input_dim=input_dim, hidden_dim=ROUTER_HIDDEN_DIM)
    model.load_state_dict(torch.load(Path(path), map_location="cpu", weights_only=True))
    return model.eval()


def _predict_expert(model: Stage3RExpert, scaled: Mapping[str, torch.Tensor],
                    indices: np.ndarray, device: str, batch_size: int) -> torch.Tensor:
    """run_stage4a._predict_expert verbatim -> (E,T,N,2) effect prediction."""
    device_obj = torch.device(device)
    model = model.to(device_obj).eval()
    ordered = np.asarray(indices, dtype=np.int64)
    outputs = []
    with torch.no_grad():
        for start in range(0, ordered.size, batch_size):
            pos = torch.as_tensor(ordered[start:start + batch_size], dtype=torch.long)
            batch = {key: scaled[key].index_select(0, pos).to(device_obj)
                     for key in ("history", "factual_suffix", "factual_actions",
                                 "counterfactual_actions")}
            outputs.append(model(**batch).cpu())
    return torch.cat(outputs, dim=0)


def _response_outputs(model: s3r.Stage3ResponsePredictor, scaled: Mapping[str, torch.Tensor],
                      indices: np.ndarray, device: str, batch_size: int) -> dict[str, np.ndarray]:
    """run_stage4a._response_outputs verbatim -> response_features, regime_logits, r_hat."""
    sub = {key: scaled[key] for key in (
        "history", "factual_suffix", "factual_actions", "counterfactual_actions",
        "cf_action", "cf_duration", "t_star")}
    feats = predictor_features_ordered(model, sub, np.asarray(indices), device, batch_size)
    device_obj = torch.device(device)
    model = model.to(device_obj).eval()
    r_hat = []
    with torch.no_grad():
        ordered = np.asarray(indices, dtype=np.int64)
        for start in range(0, ordered.size, batch_size):
            pos = torch.as_tensor(ordered[start:start + batch_size], dtype=torch.long)
            batch = {key: sub[key].index_select(0, pos).to(device_obj) for key in sub}
            out = model(batch["history"], batch["factual_suffix"], batch["factual_actions"],
                        batch["counterfactual_actions"], batch["cf_action"],
                        batch["cf_duration"], batch["t_star"])
            r_hat.append(s3r.invert_log1p(out["log1p_r_hat"]).cpu().numpy())
    return {"response_features": feats["response_features"],
            "regime_logits": feats["regime_logits"],
            "r_hat": np.concatenate(r_hat, axis=0).astype(np.float32),
            "episode_indices": feats["episode_indices"]}


# --- response calibration (fit on VALIDATION only, plan 7.5) -------------------------------------


def fit_calibration(response_paths: Mapping[int, str], validation: BranchBundle,
                    scaler: FeatureScaler, out_path: str | Path, *, device: str,
                    batch_size: int) -> dict[int, dict[str, list[float]]]:
    """Fit per-seed vector-scaling calibration on the validation regime logits.

    For each response seed, fit ``z'_c = softplus(a_c) z_c + b_c`` minimising validation NLL
    against the TRUE regime labels over the cf-dependent eval mask (plan 7.5 allows
    validation-fitted, non-blocking calibration). Create-once; a rerun reloads the receipt.
    """
    out_path = Path(out_path)
    if out_path.is_file():
        return {int(k): v for k, v in
                json.loads(out_path.read_text(encoding="utf-8"))["per_seed_params"].items()}
    scaled = scaled_model_tensors(validation, scaler)
    indices = np.arange(validation.branches, dtype=np.int64)
    labels = np.asarray(validation.regime_labels, dtype=np.int64)
    mask = np.asarray(validation.valid_eval, dtype=bool)
    per_seed: dict[int, dict[str, list[float]]] = {}
    for i, seed in enumerate(RESPONSE_PREDICTOR_SEEDS):
        model = _load_response_model(response_paths[int(seed)])
        out = _response_outputs(model, scaled, indices, device, batch_size)
        artifact = fit_vector_scaling(
            np.asarray(out["regime_logits"], dtype=np.float64), labels, mask)
        params = artifact.params
        per_seed[int(seed)] = {"a": [float(x) for x in params["a"]],
                               "b": [float(x) for x in params["b"]]}
    payload = {"plan_id": PLAN_ID, "method": "CAL_PER_SEED_VECTOR",
               "fit_split": "validation", "seeds": [int(s) for s in RESPONSE_PREDICTOR_SEEDS],
               "per_seed_params": {str(k): v for k, v in per_seed.items()},
               "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_once_json(out_path, payload)
    return per_seed


# --- router (End-to-end predictor) ---------------------------------------------------------------


def _pair_pre_router_outputs(response_model, full_model, history_model, calib_params,
                             scaled, indices, device, batch_size):
    """Per-pair response (calibrated) + expert predictions used to build router features."""
    full = _predict_expert(full_model, scaled, indices, device, batch_size)          # (E,T,N,2)
    history = _predict_expert(history_model, scaled, indices, device, batch_size)    # (E,T,N,2)
    resp = _response_outputs(response_model, scaled, indices, device, batch_size)
    regime_probs = calibrated_regime_probs_single(
        calib_params, np.asarray(resp["regime_logits"], dtype=np.float64))          # (E,T,N,4)
    return full, history, resp, regime_probs


def fit_routers(train: BranchBundle, validation: BranchBundle, scaler: FeatureScaler,
                response_paths: Mapping[int, str],
                expert_paths: Mapping[tuple[int, str], str],
                calibration: Mapping[int, Mapping[str, Any]], model_dir: str | Path, *,
                device: str, batch_size: int,
                progress: Callable[[str], None] | None = None) -> dict[int, str]:
    """Train the five End-to-end routers (seeds 1101..1505), early-stop on validation.

    For pair ``i`` the router blends the pair's full/history expert predictions with weight
    from the pair's calibrated response features; the objective is the frozen
    end_to_end_mixture (lambda_cew=0, lambda_rank=0) pure masked-MSE mixture. Returns
    ``{seed: checkpoint_path}`` (+ a sibling ``.json`` recording ``input_dim``).
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    train_scaled = scaled_model_tensors(train, scaler)
    val_scaled = scaled_model_tensors(validation, scaler)
    train_idx = np.arange(train.branches, dtype=np.int64)
    val_idx = np.arange(validation.branches, dtype=np.int64)
    effect_tr = torch.as_tensor(train.effect_target, dtype=torch.float32)
    valid_tr = torch.as_tensor(train.valid_eval, dtype=torch.bool)
    effect_val = torch.as_tensor(validation.effect_target, dtype=torch.float32)
    valid_val = torch.as_tensor(validation.valid_eval, dtype=torch.bool)

    paths: dict[int, str] = {}
    for i, seed in enumerate(PREDICTOR_PAIR_SEEDS):
        target = model_dir / f"router_e2e_seed{seed}.pt"
        if target.is_file():
            paths[int(seed)] = str(target)  # create-once
            continue
        response_model = _load_response_model(response_paths[int(RESPONSE_PREDICTOR_SEEDS[i])])
        full_model = _load_expert_model(expert_paths[(int(seed), "full")], "full")
        history_model = _load_expert_model(expert_paths[(int(seed), "history")], "history")
        calib = calibration[int(RESPONSE_PREDICTOR_SEEDS[i])]

        def _router_tensors(scaled, indices, effect, valid):
            full, history, resp, regime_probs = _pair_pre_router_outputs(
                response_model, full_model, history_model, calib, scaled, indices,
                device, batch_size)
            sub = {k: scaled[k].index_select(0, torch.as_tensor(indices, dtype=torch.long))
                   for k in ("history", "factual_suffix", "factual_actions",
                             "counterfactual_actions")}
            features, _, feat_meta = build_router_features(
                history=sub["history"], factual_suffix=sub["factual_suffix"],
                factual_actions=sub["factual_actions"],
                counterfactual_actions=sub["counterfactual_actions"],
                response_features=torch.as_tensor(resp["response_features"], dtype=torch.float32),
                regime_probs=torch.as_tensor(regime_probs, dtype=torch.float32),
                full_prediction=full, history_prediction=history)
            return {"features": features.float(), "full": full.float(),
                    "history": history.float(),
                    "effect_target": effect[indices].float(), "valid": valid[indices]}, feat_meta

        train_router, feat_meta = _router_tensors(train_scaled, train_idx, effect_tr, valid_tr)
        val_router, _ = _router_tensors(val_scaled, val_idx, effect_val, valid_val)
        if int(feat_meta["input_dim"]) != ROUTER_INPUT_DIM:
            raise RuntimeError(f"router input_dim {feat_meta['input_dim']} != {ROUTER_INPUT_DIM}")

        batches = _make_train_batches(train_router, np.arange(train.branches), BASE_BATCH_SIZE, seed)
        val_batches = _make_eval_batches(val_router, np.arange(validation.branches), BASE_BATCH_SIZE)
        s3t.seed_all(seed)
        model = CEWRouter(input_dim=ROUTER_INPUT_DIM, hidden_dim=ROUTER_HIDDEN_DIM)
        result = s3t.train_model(
            model, router_mixture_mse, batches, val_batches, seed=int(seed),
            max_epochs=BASE_MAX_EPOCHS, early_stopping_patience=BASE_EARLY_STOP_PATIENCE,
            learning_rate=BASE_LEARNING_RATE, device=device, weight_decay=BASE_WEIGHT_DECAY,
            progress_prefix=f"d2-router:{seed}",
            checkpoint_path=model_dir / f".router_seed{seed}.ckpt", resume=True,
            run_fingerprint=_fingerprint("router", int(seed), train.data_sha256))
        torch.save({k: v.detach().cpu() for k, v in result["state_dict"].items()}, target)
        write_once_json(target.with_suffix(".json"),
                        {"input_dim": ROUTER_INPUT_DIM, "hidden_dim": ROUTER_HIDDEN_DIM,
                         "seed": int(seed), "plan_id": PLAN_ID,
                         "objective": "end_to_end_mixture lambda_cew=0 lambda_rank=0"})
        paths[int(seed)] = str(target)
        if progress is not None:
            progress(f"router seed={seed} best_val_loss={result['best_val_loss']:.6f}")
    return paths


# --- risk-head fitting (COMBINED mlp / GEOMETRY ridge / RESPONSE ridge) ---------------------------


@dataclass(frozen=True)
class PrimarySelection:
    primary_predictor: str          # "BEST_CONSTANT_W" | "END_TO_END_MIXTURE"
    constant_w: float
    train_constant_mean_ade: float
    train_e2e_mean_ade: float

    def to_dict(self) -> dict[str, Any]:
        return {"primary_predictor": self.primary_predictor, "constant_w": self.constant_w,
                "train_constant_mean_ade": self.train_constant_mean_ade,
                "train_e2e_mean_ade": self.train_e2e_mean_ade,
                "selection_basis": "lowest 5-pair mean episode effect-ADE on train; tie -> BEST_CONSTANT_W"}


def _select_primary(constant_ade: np.ndarray, e2e_ade: np.ndarray) -> PrimarySelection:
    """run_stage4a._select_primary_predictor verbatim (tie -> BEST_CONSTANT_W)."""
    constant_mean = float(np.asarray(constant_ade, dtype=np.float64).mean())
    e2e_mean = float(np.asarray(e2e_ade, dtype=np.float64).mean())
    chosen = "BEST_CONSTANT_W" if constant_mean <= e2e_mean else "END_TO_END_MIXTURE"
    return PrimarySelection(chosen, PRIMARY_CONSTANT_W, constant_mean, e2e_mean)


def fit_risk_heads(predictions: Mapping[str, np.ndarray], bundle: BranchBundle,
                   model_dir: str | Path, *, progress: Callable[[str], None] | None = None
                   ) -> dict[str, Any]:
    """Fit the three fitted risk classes on the train role + freeze the primary predictor.

    ``predictions`` carries the per-seed flat branch predictions (S,E,T,N,2 /
    S,E,T,N) from :func:`run_pair_predictions` on the TRAIN role. The primary predictor
    (constant 0.60/0.40 blend vs e2e) is frozen on train by lower mean episode effect-ADE;
    the risk target is its 5-pair mean episode ADE (Stage 4A parity, cf-dependent eval
    mask). Returns paths + the primary-selection receipt. Create-once.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = model_dir / "risk_fit_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        return {"primary": PrimarySelection(**{k: receipt["primary"][k] for k in
                                               ("primary_predictor", "constant_w",
                                                "train_constant_mean_ade", "train_e2e_mean_ade")}),
                "paths": receipt["paths"], "verification": "existing_risk_fit_verified_no_overwrite"}

    valid = np.asarray(bundle.valid_eval, dtype=bool)                 # (E,T,N) cf-dependent
    effect_true = np.asarray(bundle.effect_target, dtype=np.float32)  # (E,T,N,2)
    full = np.asarray(predictions["full"], dtype=np.float64)          # (S,E,T,N,2)
    history = np.asarray(predictions["history"], dtype=np.float64)
    e2e = np.asarray(predictions["e2e"], dtype=np.float64)
    constant = PRIMARY_CONSTANT_W * full + (1.0 - PRIMARY_CONSTANT_W) * history

    constant_ade = s4risk.episode_effect_ade(constant, effect_true, valid)   # (S,E)
    e2e_ade = s4risk.episode_effect_ade(e2e, effect_true, valid)             # (S,E)
    primary = _select_primary(constant_ade, e2e_ade)
    primary_predictions = constant if primary.primary_predictor == "BEST_CONSTANT_W" else e2e
    primary_ade = s4risk.episode_effect_ade(primary_predictions, effect_true, valid)
    risk_target = s4risk.risk_target_from_pairs(primary_ade)                 # (E,)

    response_predictions = {"r_hat": predictions["r_hat"],
                            "presence_prob": predictions["presence_prob"],
                            "high_prob": predictions["high_prob"]}
    features = s4risk.build_legal_risk_features(
        primary_predictions=primary_predictions, full_predictions=full,
        history_predictions=history, response_predictions=response_predictions,
        pre_intervention_geometry=bundle.geometry_branch,
        action=np.asarray(bundle.cf_action, dtype=np.int64),
        duration=np.asarray(bundle.cf_duration, dtype=np.float64),
        valid_mask=valid)

    fits = [("COMBINED_RISK_HEAD", COMBINED_IMPLEMENTATION),
            ("GEOMETRY_ONLY", None), ("RESPONSE_ONLY", None)]
    paths: dict[str, str] = {}
    for name, impl in fits:
        model = s4risk.fit_risk_candidate(name, features, risk_target, RISK_FIT_CONFIG,
                                          implementation=impl)
        target = model_dir / f"risk_{name.lower()}.pt"
        model.save(target)
        paths[name] = str(target)
        if progress is not None:
            progress(f"risk head {name} ({model.implementation}) fitted")
    receipt = {"plan_id": PLAN_ID, "primary": primary.to_dict(), "paths": paths,
               "risk_fit_config": dict(RISK_FIT_CONFIG),
               "combined_implementation": COMBINED_IMPLEMENTATION,
               "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_once_json(receipt_path, receipt)
    return {"primary": primary, "paths": paths, "verification": "fitted"}


# --- production inference: per-pair predictions ---------------------------------------------------


@dataclass(frozen=True)
class PredictorStack:
    """The frozen D2 predictor chain: checkpoints + scaler + calibration + risk models."""
    scaler_path: str
    response_paths: dict[int, str]
    expert_paths: dict[tuple[int, str], str]
    router_paths: dict[int, str]
    calibration: dict[int, dict[str, list[float]]]
    risk_paths: dict[str, str]
    primary_predictor: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scaler_path": self.scaler_path,
            "response_paths": {str(k): v for k, v in self.response_paths.items()},
            "expert_paths": {f"{k[0]}|{k[1]}": v for k, v in self.expert_paths.items()},
            "router_paths": {str(k): v for k, v in self.router_paths.items()},
            "calibration": {str(k): v for k, v in self.calibration.items()},
            "risk_paths": dict(self.risk_paths),
            "primary_predictor": self.primary_predictor,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PredictorStack":
        expert_paths = {}
        for key, value in payload["expert_paths"].items():
            seed_text, condition = str(key).split("|")
            expert_paths[(int(seed_text), condition)] = str(value)
        return cls(
            scaler_path=str(payload["scaler_path"]),
            response_paths={int(k): str(v) for k, v in payload["response_paths"].items()},
            expert_paths=expert_paths,
            router_paths={int(k): str(v) for k, v in payload["router_paths"].items()},
            calibration={int(k): v for k, v in payload["calibration"].items()},
            risk_paths={str(k): str(v) for k, v in payload["risk_paths"].items()},
            primary_predictor=str(payload["primary_predictor"]))


def run_pair_predictions(stack: PredictorStack, bundle: BranchBundle, *,
                         device: str, batch_size: int) -> dict[str, np.ndarray]:
    """Run the frozen five seed pairs over all branches -> flat (S,E,...) predictions.

    Mirrors stage4b_inference.run_pair_inference per pair (experts -> calibrated response ->
    router -> e2e blend), then stacks over S. Returns e2e/full/history (S,E,T,N,2) and
    r_hat/presence_prob/high_prob (S,E,T,N). No truth is read.
    """
    scaler = FeatureScaler.load(stack.scaler_path)
    scaled = scaled_model_tensors(bundle, scaler)
    indices = np.arange(bundle.branches, dtype=np.int64)

    per_seed: dict[str, list[np.ndarray]] = {
        k: [] for k in ("e2e", "full", "history", "r_hat", "presence_prob", "high_prob")}
    for i in range(N_SEEDS):
        seed_pair = int(PREDICTOR_PAIR_SEEDS[i])
        seed_resp = int(RESPONSE_PREDICTOR_SEEDS[i])
        full_model = _load_expert_model(stack.expert_paths[(seed_pair, "full")], "full")
        history_model = _load_expert_model(stack.expert_paths[(seed_pair, "history")], "history")
        response_model = _load_response_model(stack.response_paths[seed_resp])
        router_model = _load_router_model(stack.router_paths[seed_pair]).to(torch.device(device))
        calib = stack.calibration[seed_resp]

        full, history, resp, regime_probs = _pair_pre_router_outputs(
            response_model, full_model, history_model, calib, scaled, indices, device, batch_size)
        sub = {k: scaled[k].index_select(0, torch.as_tensor(indices, dtype=torch.long))
               for k in ("history", "factual_suffix", "factual_actions", "counterfactual_actions")}
        router_features, _, _ = build_router_features(
            history=sub["history"], factual_suffix=sub["factual_suffix"],
            factual_actions=sub["factual_actions"],
            counterfactual_actions=sub["counterfactual_actions"],
            response_features=torch.as_tensor(resp["response_features"], dtype=torch.float32),
            regime_probs=torch.as_tensor(regime_probs, dtype=torch.float32),
            full_prediction=full, history_prediction=history)
        with torch.no_grad():
            weight = router_model(router_features.to(torch.device(device))).cpu()  # (E,T,N,1)
        e2e = weight * full + (1.0 - weight) * history                          # (E,T,N,2)

        per_seed["full"].append(full.numpy())
        per_seed["history"].append(history.numpy())
        per_seed["e2e"].append(e2e.numpy().astype(np.float32))
        per_seed["r_hat"].append(np.asarray(resp["r_hat"], dtype=np.float32))
        per_seed["presence_prob"].append((1.0 - regime_probs[..., 0]).astype(np.float32))
        per_seed["high_prob"].append(regime_probs[..., 3].astype(np.float32))

    return {key: np.stack(values).astype(np.float32) for key, values in per_seed.items()}


def score_action_risks(stack: PredictorStack, bundle: BranchBundle,
                       predictions: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The five action-level risk classes, each ``(G, A)`` (higher = riskier).

    Mirrors stage4c_inference.build_action_risk_features + score_actions against the D2
    fitted risk models, using the collector's cf-INdependent inference mask (Stage 4C/R2
    scoring parity). COMBINED/GEOMETRY/RESPONSE use the D2-fitted models;
    UNCERTAINTY/DISAGREEMENT are the unfitted direct aggregations (reused read-only).
    """
    groups, n_actions = bundle.groups, bundle.n_actions

    def flatten(key):  # predictions are already flat (S,E,T,N,...) -> float64
        return np.asarray(predictions[key], dtype=np.float64)

    risk_models = {name: s4risk.FittedRiskModel.load(stack.risk_paths[name])
                   for name in ("COMBINED_RISK_HEAD", "GEOMETRY_ONLY", "RESPONSE_ONLY")}
    features = s4risk.build_legal_risk_features(
        primary_predictions=flatten("e2e") if stack.primary_predictor == "END_TO_END_MIXTURE"
        else flatten("full") * PRIMARY_CONSTANT_W + flatten("history") * (1.0 - PRIMARY_CONSTANT_W),
        full_predictions=flatten("full"), history_predictions=flatten("history"),
        response_predictions={"r_hat": flatten("r_hat"), "presence_prob": flatten("presence_prob"),
                              "high_prob": flatten("high_prob")},
        pre_intervention_geometry=bundle.geometry_branch,
        action=np.asarray(bundle.cf_action, dtype=np.int64),
        duration=np.asarray(bundle.cf_duration, dtype=np.float64),
        valid_mask=np.asarray(bundle.valid_infer, dtype=bool))  # cf-independent inference mask

    def to_ga(score):
        return np.asarray(score, dtype=np.float64).reshape(groups, n_actions)

    scores = {
        "COMBINED_RISK_HEAD": to_ga(risk_models["COMBINED_RISK_HEAD"].predict(
            s4risk.design_matrix(features, s4risk.CANDIDATE_FEATURES[s4risk.CANDIDATE_COMBINED]))),
        "UNCERTAINTY_ONLY": to_ga(s4risk.direct_risk_score(s4risk.CANDIDATE_UNCERTAINTY, features)),
        "GEOMETRY_ONLY": to_ga(risk_models["GEOMETRY_ONLY"].predict(
            s4risk.design_matrix(features, s4risk.CANDIDATE_FEATURES[s4risk.BASELINE_GEOMETRY]))),
        "RESPONSE_ONLY": to_ga(risk_models["RESPONSE_ONLY"].predict(
            s4risk.design_matrix(features, s4risk.CANDIDATE_FEATURES[s4risk.CANDIDATE_RESPONSE]))),
        "DISAGREEMENT_ONLY": to_ga(s4risk.direct_risk_score(s4risk.CANDIDATE_DISAGREEMENT, features)),
    }
    for name, value in scores.items():
        if value.shape != (groups, n_actions) or not np.isfinite(value).all():
            raise RuntimeError(f"non-finite D2 action risk score: {name}")
    return {name: scores[name] for name in s4cp.RISK_CLASSES}


def build_direct_inputs(bundle: BranchBundle, predictions: Mapping[str, np.ndarray],
                        action_risks: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Assemble the frozen 9x34-token + 9-global Direct bundle via ``build_relative_tokens``.

    ``predictions`` are the flat (S,E,...) arrays; they are reshaped to (S,G,A,T,N,2) for
    the consequence. Returns the ``build_relative_tokens`` bundle (tokens, global_features,
    valid_mask, j_hat, j_hat_seeds, reference_action, baselines, interaction_min_distance).
    """
    groups, n_actions = bundle.groups, bundle.n_actions
    horizon, agents = FUTURE_LEN, OBJECT_CAPACITY
    e2e = np.asarray(predictions["e2e"], dtype=np.float64).reshape(
        (N_SEEDS, groups, n_actions, horizon, agents, 2))
    return s4cpf.build_relative_tokens(
        e2e=e2e, factual_xy=bundle.factual_xy, history_last_xy=bundle.history_last_xy,
        inference_mask=bundle.inference_mask, action_risks=action_risks,
        geometry=bundle.geometry_group, t_star=bundle.t_star_group,
        group_seed=bundle.group_seed)


__all__ = [
    "BASE_BATCH_SIZE", "BASE_EARLY_STOP_PATIENCE", "BASE_LEARNING_RATE", "BASE_MAX_EPOCHS",
    "BASE_WEIGHT_DECAY", "BranchBundle", "COMBINED_IMPLEMENTATION", "EXPERT_HIDDEN_DIM",
    "EXPERT_OBJECT_EMBEDDING_DIM", "FIT_COMPONENTS", "N_ACTIONS", "N_SEEDS", "PLAN_ID",
    "POSITION_SLICE", "PREDICTOR_PAIR_SEEDS", "PRIMARY_CONSTANT_W", "PrimarySelection",
    "PredictorStack", "RESPONSE_FEATURE_DIM", "RESPONSE_HIDDEN_DIM", "RESPONSE_LAMBDA_CLS",
    "RESPONSE_PREDICTOR_SEEDS", "ROLE_SALT", "ROUTER_HIDDEN_DIM", "ROUTER_INPUT_DIM",
    "RISK_FIT_CONFIG", "RoleArrays", "build_branch_bundle", "build_direct_inputs",
    "fit_calibration", "fit_experts", "fit_response_predictors", "fit_risk_heads",
    "fit_routers", "fit_scaler", "load_role_arrays", "masked_expert_mse",
    "router_mixture_mse", "run_pair_predictions", "scaled_model_tensors", "score_action_risks",
]
