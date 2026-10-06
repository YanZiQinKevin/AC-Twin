"""Stage 4D-D2 in-platform isomorphic retraining: Direct risk model, train-support gate,
one-shot validation baseline selection and freeze.

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` sections 7.4-7.6 and
12.4 (Task 3). This module RE-TRAINS the fixed ``SAME_INFORMATION_DIRECT`` decision-risk
model on the MetaDrive/Waymo train role. It is a platform-specific retraining, NOT a
zero-shot transfer of the Merge checkpoints (plan Global Constraint 2).

Mechanical migration (plan 7.2): the Direct architecture, loss, optimizer, epoch cap,
early-stopping rule (patience + best-state restore) and the 34+9 input binding are reused
READ-ONLY from the frozen Stage 4C Plus / R2 chain (``stage4cp_models`` /
``stage4cp_features`` / ``stage4c_objective`` / ``stage4cp_statistics``). The ONLY
permitted platform differences (plan 7.2) are: a platform train-only scaler, explicit
SI/numeric conversions, new-platform checkpoints and new-platform model seeds. No new
layers, no wider hidden width, no loss swap, no map encoder, no post-screen model edits.

Early-stopping monitor (plan 7.5/7.6, a deliberate D2 choice distinct from R2): the D2
validation role is the early-stopping and checkpoint-selection monitor — "validation 仅用于
early stopping、机械 checkpoint、唯一主基线和非阻断 calibration" and "为每个模型 seed 机械选择
最低冻结 validation loss checkpoint". The scaler and class weight stay TRAIN-ONLY
(plan 7.5); only epoch selection reads validation. The 4,000-group development screen and
the confirm set remain the untouched one-shot holdouts.

The frozen shared machinery is imported read-only; nothing upstream is modified (plan
Global Constraint 1). The 34-dim token / 9-dim global NAMES and ORDER are bound by reusing
``stage4cp_features.feature_schema`` verbatim (plan 7.4).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn

from ac_twin import stage4c_objective as s4co
from ac_twin import stage4cp_features as s4cpf
from ac_twin import stage4cp_models as s4cm
from ac_twin import stage4cp_protocol as s4cp
from ac_twin import stage4cp_statistics as s4cs
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

ROOT = Path(__file__).resolve().parents[2]

PLAN_ID = "AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2"

# --- frozen task / Direct contract (plan 7.4; mirrors the R2/4C-Plus frozen values) -------
N_ACTIONS = len(s4co.ACTION_PLANS)             # 10
N_COMPETITORS = N_ACTIONS - 1                  # 9
TOKEN_DIM = s4cpf.TOKEN_DIM                    # 34
GLOBAL_DIM = s4cpf.GLOBAL_DIM                  # 9
DIRECT_FLAT_DIM = N_COMPETITORS * TOKEN_DIM + GLOBAL_DIM  # 315
ARCH_DIRECT = s4cp.ARCH_DIRECT                 # "SAME_INFORMATION_DIRECT"
DIRECT_HIDDEN_WIDTH = 43
DROPOUT = 0.10
LAMBDA_CLS = 0.25
EXPECTED_TRAINABLE_PARAMETERS = 15740          # EXACT; preflight fails otherwise
IDLE_PLAN_INDEX = s4co.IDLE_PLAN_INDEX         # 3 -> (1,3)

# --- frozen Direct training mechanics (plan 7.4) -------------------------------------------
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_BATCH_SIZE = 128
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 12

# --- frozen labels / gates (plan 7.5, 7.6) ---------------------------------------------------
HIGH_REGRET_THRESHOLD = 0.01
TRAIN_MIN_HIGH_REGRET_GROUPS = 750
TRAIN_MIN_NONZERO_REGRET_GROUPS = 1250
VALIDATION_I_AURC_MIN = 0.05
VALIDATION_I_REGRET80_MIN = 0.03
VALIDATION_MIN_POSITIVE_SEEDS = 4
BASELINE_TIE_TOLERANCE = 1e-12
COVERAGE_MAIN = s4cp.COVERAGE_MAIN             # 0.80

# --- frozen role sizing + new-platform model seeds (plan 4.3, 7.3) ---------------------------
DEV_TRAIN_GROUPS = 5000
DEV_VALIDATION_GROUPS = 1500
DEV_SCREEN_GROUPS = 4000
MODEL_SEEDS = (9101, 9202, 9303, 9404, 9505)               # Direct risk seeds (plan 7.3)
RESPONSE_PREDICTOR_SEEDS = (101, 202, 303, 404, 505)       # response predictor
PREDICTOR_PAIR_SEEDS = (1101, 1202, 1303, 1404, 1505)      # Full/History/E2E pair

#: continuous feature columns constant BY CONTRACT, exempt from the plan-7.5 zero-variance
#: check ("已知结构常量字段不因零方差失败，但必须在 schema 中显式声明"). ``visible_vehicle_count``
#: is identically OBJECT_CAPACITY-1 = 4 for every accepted group (capacity-5 object
#: observation, slot 0 ego + 4 non-ego; acceptance requires all four non-ego present), so
#: the column is structurally constant. Declared explicitly here per plan 7.5.
CONSTANT_CONTRACT_CONTINUOUS: tuple[str, ...] = ("visible_vehicle_count",)

# --- baselines (plan 7.6; the three eligible standard baselines, frozen order) --------------
BASELINE_MARGIN = s4cp.BASELINE_MARGIN                         # "MARGIN_ONLY"
BASELINE_SEED_VOTE_ENTROPY = s4cp.BASELINE_SEED_VOTE_ENTROPY   # "SEED_VOTE_ENTROPY"
BASELINE_SEED_MARGIN_STD = s4cp.BASELINE_SEED_MARGIN_STD       # "SEED_MARGIN_STD"
BASELINE_COMBINED_TOP2_MAX = s4cp.BASELINE_COMBINED_TOP2_MAX   # historical reference only
STRONGEST_BASELINE_CANDIDATES = (
    BASELINE_MARGIN, BASELINE_SEED_VOTE_ENTROPY, BASELINE_SEED_MARGIN_STD)

# --- mechanical terminal states (plan 15, verbatim) -------------------------------------------
STATUS_TRAIN_SUPPORT_OK = "TRAIN_SUPPORT_OK"
STATUS_TRAIN_SUPPORT_INSUFFICIENT = "STAGE4D_EXTERNAL_TRAIN_SUPPORT_INSUFFICIENT"
STATUS_VALIDATION_QUALIFIED = "STAGE4D_EXTERNAL_VALIDATION_QUALIFIED"
STATUS_VALIDATION_NOT_QUALIFIED = "STAGE4D_EXTERNAL_VALIDATION_NOT_QUALIFIED"
STATUS_SCHEMA_INVALID = "INCONCLUSIVE_STAGE4D_EXTERNAL_SCHEMA_INVALID"

FIT_RECEIPT_NAME = "fit_receipt.json"
FREEZE_ARTIFACTS = (
    "feature_schema.json",
    "feature_scaler.npz",
    "model_inventory.json",
    "validation_gate_report.json",
    "final_selection.json",
)
PRIMARY_SCORE_DEFINITION = (
    "risk_score = sigmoid(logit_high) * 0.01 * expm1(softplus(raw_mu)); "
    "ensemble = arithmetic mean of the five model seeds' risk_score")


def group_seed_from_scenario_id(scenario_id: str, role_salt: str) -> int:
    """Deterministic per-group seed from the scenario ID (new platform binding, frozen).

    The D2 collector stores no integer episode seed; the group's stable identity is its
    ``scenario_id``. This maps it to a positive int in R2's seed range [1, 2**31-2] via
    SHA256(role_salt + "group|" + scenario_id). It is a pure function of the scenario ID,
    identical across every branch and every rerun, and links the predictions/scores/truth
    of one group. Uniqueness within a role is asserted by the caller.
    """
    digest = hashlib.sha256((role_salt + "group|" + str(scenario_id)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 2) + 1


def count_direct_parameters() -> int:
    """Recount the frozen Direct architecture; raise unless EXACTLY 15,740."""
    model = s4cm.build_model(
        ARCH_DIRECT, hidden_dim=DIRECT_HIDDEN_WIDTH, dropout=DROPOUT,
        direct_width=DIRECT_HIDDEN_WIDTH, n_competitors=N_COMPETITORS,
        token_dim=TOKEN_DIM, global_dim=GLOBAL_DIM)
    count = s4cm.count_parameters(model)
    if count != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            f"Direct trainable parameters {count} != frozen {EXPECTED_TRAINABLE_PARAMETERS}")
    return count


# --- truth-side labels (legal for train/validation; NEVER read for screen/confirm) ------------


def compute_true_labels(
    counterfactual_xy: np.ndarray,
    history_last_xy: np.ndarray,
    inference_mask: np.ndarray,
    j_hat: np.ndarray,
) -> dict[str, Any]:
    """True consequences and decision labels for one role (train/validation only).

    ``counterfactual_xy`` (G,A,T,N,2) is the platform TRUE counterfactual futures in the
    common Map-Frenet frame (s,d channels used as x,y; the consequence function is
    translation-invariant). ``j_hat`` (G,A) is the ensemble predicted consequence (from
    ``build_relative_tokens``). Returns regret/oracle/idle/random and the Direct
    supervision targets ``y_reg``/``y_high``. No outcome is used to accept/reject.
    """
    true_xy = np.asarray(counterfactual_xy, dtype=np.float64)
    groups, n_actions = true_xy.shape[0], true_xy.shape[1]
    hist_b = np.broadcast_to(history_last_xy[:, None, :], (groups, n_actions, 2))
    mask_b = np.broadcast_to(inference_mask[:, None, :, :],
                             (groups, n_actions) + inference_mask.shape[1:])
    comp_true = s4co.trajectory_components(true_xy, hist_b, mask_b)
    j_true = s4co.total_cost(comp_true["interaction"],
                             s4co.efficiency_cost(comp_true["progress"]))
    j_hat = np.asarray(j_hat, dtype=np.float64)
    rows = np.arange(groups)
    a_hat = np.argmin(j_hat, axis=-1)
    a_star = np.argmin(j_true, axis=-1)
    j_true_star = j_true[rows, a_star]
    regret = s4co.clamp_regret(j_true[rows, a_hat] - j_true_star)
    idle_regret = s4co.clamp_regret(j_true[:, IDLE_PLAN_INDEX] - j_true_star)
    random_regret = s4co.clamp_regret(j_true.mean(axis=-1) - j_true_star)
    return {
        "j_true": j_true,
        "a_hat": a_hat,
        "a_star": a_star,
        "regret": regret,
        "idle_regret": idle_regret,
        "random_regret": random_regret,
        "oracle_action": a_star,
        "base_mean_regret": float(regret.mean()),
        "idle_mean_regret": float(idle_regret.mean()),
        "random_mean_regret": float(random_regret.mean()),
        "y_reg": np.log1p(regret / HIGH_REGRET_THRESHOLD),
        "y_high": (regret > HIGH_REGRET_THRESHOLD).astype(np.int64),
    }


# --- train support hard gate (plan 7.5; fail-closed) -------------------------------------------


def evaluate_train_support_gate(train: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the plan-7.5 minimum train-support conditions to the 5,000 train groups.

    Any failure yields terminal ``STAGE4D_EXTERNAL_TRAIN_SUPPORT_INSUFFICIENT``; thresholds
    are frozen, no top-up collection, validation is never merged into train. Conditions
    (plan 7.5): regret>0.01 >= 750 groups; nonzero regret >= 1,250 groups; every output
    target finite; main continuous features non-degenerate (no zero-variance column
    outside the explicitly declared structural constants).
    """
    regret = np.asarray(train["regret"], dtype=np.float64).ravel()
    y_reg = np.asarray(train["y_reg"], dtype=np.float64).ravel()
    y_high = np.asarray(train["y_high"], dtype=np.float64).ravel()
    tokens = np.asarray(train["tokens"], dtype=np.float64)
    global_features = np.asarray(train["global_features"], dtype=np.float64)
    groups = int(tokens.shape[0])

    # completeness: every group carries all ten actions and aligned labels.
    c_complete = bool(
        groups == DEV_TRAIN_GROUPS
        and regret.size == groups and y_reg.size == groups and y_high.size == groups
        and global_features.shape[0] == groups
        and tokens.shape[1:] == (N_COMPETITORS, TOKEN_DIM)
        and global_features.shape[1] == GLOBAL_DIM)

    high_regret_groups = int((regret > HIGH_REGRET_THRESHOLD).sum())
    nonzero_regret_groups = int((regret > 0.0).sum())
    c_high = high_regret_groups >= TRAIN_MIN_HIGH_REGRET_GROUPS
    c_nonzero = nonzero_regret_groups >= TRAIN_MIN_NONZERO_REGRET_GROUPS

    # every output target finite.
    c_finite_targets = bool(np.isfinite(y_reg).all() and np.isfinite(y_high).all())

    # main continuous features finite and non-degenerate (structural constants exempt).
    cont_names = list(s4cpf.TOKEN_CONT_NAMES) + list(s4cpf.GLOBAL_NAMES)
    token_cont = tokens[:, :, s4cpf.TOKEN_CONT_IDX].reshape(-1, s4cpf.N_TOKEN_CONT)
    finite = bool(np.isfinite(token_cont).all() and np.isfinite(global_features).all())
    variances = np.concatenate([token_cont.var(axis=0), global_features.var(axis=0)])
    exempt = {i for i, name in enumerate(cont_names) if name in CONSTANT_CONTRACT_CONTINUOUS}
    zero_var = [cont_names[i] for i in range(len(cont_names))
                if variances[i] <= 1e-24 and i not in exempt]
    c_features = finite and not zero_var

    conditions = {
        "all_complete_10_action": c_complete,
        "high_regret_groups_ge_750": bool(c_high),
        "nonzero_regret_groups_ge_1250": bool(c_nonzero),
        "output_targets_finite": c_finite_targets,
        "continuous_features_finite_with_variance": bool(c_features),
    }
    passed = bool(all(conditions.values()))
    return {
        "plan_id": PLAN_ID,
        "role": "train",
        "conditions": conditions,
        "passed": passed,
        "status": (STATUS_TRAIN_SUPPORT_OK if passed else STATUS_TRAIN_SUPPORT_INSUFFICIENT),
        "values": {
            "groups": groups,
            "high_regret_groups": high_regret_groups,
            "nonzero_regret_groups": nonzero_regret_groups,
            "zero_variance_continuous_columns": zero_var,
        },
    }


# --- fixed five-seed Direct training (validation is the early-stopping/checkpoint monitor) -----


@dataclass(frozen=True)
class ModelInventory:
    """The five frozen Direct checkpoints + the train-only statistics."""
    plan_id: str
    arch: str
    seeds: tuple[int, ...]
    param_count: int
    run_dir: str
    checkpoint_paths: dict[int, str]
    checkpoint_sha256: dict[int, str]
    scaler: s4cpf.FrozenFeatureScaler
    pos_weight: float
    interaction_threshold: float
    train_losses: dict[int, float]
    validation_losses: dict[int, float]
    epochs_run: dict[int, int]
    n_train_groups: int
    n_validation_groups: int
    written_at: str
    verification: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "arch": self.arch,
            "seeds": [int(s) for s in self.seeds],
            "param_count": int(self.param_count),
            "run_dir": self.run_dir,
            "checkpoint_paths": {str(k): v for k, v in self.checkpoint_paths.items()},
            "checkpoint_sha256": {str(k): v for k, v in self.checkpoint_sha256.items()},
            "scaler": {
                "token_cont_mean": self.scaler.token_cont_mean.tolist(),
                "token_cont_std": self.scaler.token_cont_std.tolist(),
                "global_mean": self.scaler.global_mean.tolist(),
                "global_std": self.scaler.global_std.tolist(),
            },
            "pos_weight": float(self.pos_weight),
            "interaction_threshold": float(self.interaction_threshold),
            "train_losses": {str(k): float(v) for k, v in self.train_losses.items()},
            "validation_losses": {str(k): float(v) for k, v in self.validation_losses.items()},
            "epochs_run": {str(k): int(v) for k, v in self.epochs_run.items()},
            "n_train_groups": int(self.n_train_groups),
            "n_validation_groups": int(self.n_validation_groups),
            "written_at": self.written_at,
        }


@dataclass(frozen=True)
class RiskScores:
    """Five-seed scores on a bundle; ``risk_ensemble`` is the arithmetic mean."""
    group_seed: np.ndarray
    risk_ensemble: np.ndarray   # (G,) == risk_by_seed.mean(axis=0), exactly
    risk_by_seed: np.ndarray    # (5,G)
    mu_only_ensemble: np.ndarray
    class_only_ensemble: np.ndarray
    seeds: tuple[int, ...]


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value)


def _model_dir(run_dir: str | Path) -> Path:
    return _resolve(run_dir) / "development" / "models"


def _fit_direct_seed(
    seed: int,
    train: Mapping[str, np.ndarray],
    validation: Mapping[str, np.ndarray],
    *,
    pos_weight: float,
    device: str,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train ONE frozen Direct seed; early-stop on VALIDATION loss (plan 7.5/7.6).

    Mirrors the frozen R2/4C-Plus loop mechanics (AdamW, SmoothL1 + lambda_cls * weighted
    BCE on train, patience + best-state restore), but the early-stopping/checkpoint
    monitor is the D2 validation role — the plan's deliberate choice (validation is used
    for early stopping and the mechanical lowest-validation-loss checkpoint). The best
    state is the lowest-validation-loss checkpoint.
    """
    dev = torch.device(device)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**31 - 1))

    model = s4cm.build_model(
        ARCH_DIRECT, hidden_dim=DIRECT_HIDDEN_WIDTH, dropout=DROPOUT,
        direct_width=DIRECT_HIDDEN_WIDTH).to(dev)

    x_tr_tok = torch.as_tensor(train["x_tok"], dtype=torch.float32)
    x_tr_glob = torch.as_tensor(train["x_glob"], dtype=torch.float32)
    y_tr_r = torch.as_tensor(train["y_reg"], dtype=torch.float32)
    y_tr_h = torch.as_tensor(train["y_high"], dtype=torch.float32)
    x_val_tok = torch.as_tensor(validation["x_tok"], dtype=torch.float32).to(dev)
    x_val_glob = torch.as_tensor(validation["x_glob"], dtype=torch.float32).to(dev)
    y_val_r = torch.as_tensor(validation["y_reg"], dtype=torch.float32).to(dev)
    y_val_h = torch.as_tensor(validation["y_high"], dtype=torch.float32).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    smooth_l1 = nn.SmoothL1Loss()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(
        float(pos_weight), dtype=torch.float32, device=dev))

    def total_loss(x_tok, x_glob, y_r, y_h) -> float:
        model.eval()
        with torch.no_grad():
            raw_mu, logit = model(x_tok.flatten(start_dim=1), x_glob)
            mu_pos = torch.nn.functional.softplus(raw_mu)
            loss = smooth_l1(mu_pos, y_r) + LAMBDA_CLS * bce(logit, y_h)
        return float(loss.item())

    n_train = x_tr_tok.shape[0]
    best_state = None
    best_val = float("inf")
    wait = 0
    started = time.perf_counter()
    batch_generator = torch.Generator().manual_seed(int(seed))
    epochs_run = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(n_train, generator=batch_generator).numpy()
        for start in range(0, n_train, TRAIN_BATCH_SIZE):
            sel = order[start:start + TRAIN_BATCH_SIZE]
            optimizer.zero_grad()
            raw_mu, logit = model(
                x_tr_tok[sel].flatten(start_dim=1).to(dev),
                x_tr_glob[sel].to(dev))
            mu_pos = torch.nn.functional.softplus(raw_mu)
            loss = smooth_l1(mu_pos, y_tr_r[sel].to(dev)) \
                + LAMBDA_CLS * bce(logit, y_tr_h[sel].to(dev))
            loss.backward()
            optimizer.step()
        epochs_run = epoch + 1
        val_loss = total_loss(x_val_tok, x_val_glob, y_val_r, y_val_h)
        if val_loss < best_val - 1e-10:
            best_val = val_loss
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if progress is not None:
            progress(f"seed={seed} epoch={epochs_run}/{MAX_EPOCHS} "
                     f"validation_loss={val_loss:.6f} best={best_val:.6f} "
                     f"elapsed={time.perf_counter() - started:.1f}s")
        if wait >= EARLY_STOP_PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    train_loss = total_loss(x_tr_tok.to(dev), x_tr_glob.to(dev),
                            y_tr_r.to(dev), y_tr_h.to(dev))
    return {
        "arch": ARCH_DIRECT,
        "hidden_dim": DIRECT_HIDDEN_WIDTH,
        "direct_width": DIRECT_HIDDEN_WIDTH,
        "lambda_cls": LAMBDA_CLS,
        "seed": int(seed),
        "pos_weight": float(pos_weight),
        "dropout": DROPOUT,
        "token_dim": TOKEN_DIM,
        "global_dim": GLOBAL_DIM,
        "n_competitors": N_COMPETITORS,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "train_loss": float(train_loss),
        "best_validation_loss": float(best_val),
        "epochs_run": int(epochs_run),
        "n_train_groups": int(n_train),
    }


def fit_five_direct_models(
    train_bundle: Mapping[str, np.ndarray],
    validation_bundle: Mapping[str, np.ndarray],
    run_dir: str | Path,
    *,
    device: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> ModelInventory:
    """Train the five frozen Direct seeds on the train role; keep ALL checkpoints.

    ``train_bundle``/``validation_bundle`` carry raw (unscaled) ``tokens`` (G,9,34),
    ``global_features`` (G,9), labels ``y_reg``/``y_high`` (G,), ``group_seed`` (G,); the
    train bundle also carries ``interaction_min_distance`` (G,). The scaler and the class
    weight are fit on the TRAIN role ONLY (plan 7.5); each seed early-stops on the
    VALIDATION role and its checkpoint is the lowest-validation-loss state (plan 7.6
    step 1). Create-once: a rerun with the receipt present verifies checkpoint hashes and
    never retrains or overwrites.
    """
    param_count = count_direct_parameters()
    device = str(device or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model_dir = _model_dir(run_dir)
    receipt_path = model_dir / FIT_RECEIPT_NAME
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for seed_text, expected in receipt["checkpoint_sha256"].items():
            path = Path(receipt["checkpoint_paths"][seed_text])
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"Direct checkpoint drifted from its fit receipt: {path}")
        scaler_payload = receipt["scaler"]
        return ModelInventory(
            plan_id=str(receipt["plan_id"]), arch=str(receipt["arch"]),
            seeds=tuple(int(s) for s in receipt["seeds"]),
            param_count=int(receipt["param_count"]), run_dir=str(_resolve(run_dir)),
            checkpoint_paths={int(k): str(v) for k, v in receipt["checkpoint_paths"].items()},
            checkpoint_sha256={int(k): str(v) for k, v in receipt["checkpoint_sha256"].items()},
            scaler=s4cpf.FrozenFeatureScaler(
                np.asarray(scaler_payload["token_cont_mean"]),
                np.asarray(scaler_payload["token_cont_std"]),
                np.asarray(scaler_payload["global_mean"]),
                np.asarray(scaler_payload["global_std"])),
            pos_weight=float(receipt["pos_weight"]),
            interaction_threshold=float(receipt["interaction_threshold"]),
            train_losses={int(k): float(v) for k, v in receipt["train_losses"].items()},
            validation_losses={int(k): float(v) for k, v in receipt["validation_losses"].items()},
            epochs_run={int(k): int(v) for k, v in receipt["epochs_run"].items()},
            n_train_groups=int(receipt["n_train_groups"]),
            n_validation_groups=int(receipt["n_validation_groups"]),
            written_at=str(receipt["written_at"]),
            verification="existing_inventory_verified_no_overwrite")

    def _arrays(bundle: Mapping[str, np.ndarray], role: str):
        tokens = np.asarray(bundle["tokens"], dtype=np.float64)
        global_features = np.asarray(bundle["global_features"], dtype=np.float64)
        y_reg = np.asarray(bundle["y_reg"], dtype=np.float64).ravel()
        y_high = np.asarray(bundle["y_high"], dtype=np.float64).ravel()
        group_seed = np.asarray(bundle["group_seed"], dtype=np.int64).ravel()
        n = tokens.shape[0]
        if not (global_features.shape[0] == y_reg.shape[0] == y_high.shape[0]
                == group_seed.shape[0] == n):
            raise ValueError(f"{role} bundle arrays must align on the group axis")
        for name, arr in (("tokens", tokens), ("global_features", global_features),
                          ("y_reg", y_reg), ("y_high", y_high)):
            if not np.isfinite(arr).all():
                raise ValueError(f"non-finite {role} bundle array: {name}")
        return tokens, global_features, y_reg, y_high, group_seed, n

    tokens, global_features, y_reg, y_high, group_seed, n_train = _arrays(
        train_bundle, "train")
    v_tokens, v_global, v_y_reg, v_y_high, v_seed, n_val = _arrays(
        validation_bundle, "validation")
    distance = np.asarray(train_bundle["interaction_min_distance"], dtype=np.float64).ravel()
    if distance.shape[0] != n_train or not np.isfinite(distance).all():
        raise ValueError("non-finite train interaction_min_distance")

    # Train-only statistics: the scaler and class weight are fit on the TRAIN role ONLY;
    # the interaction threshold is the train-only median (plan 7.5, 8.2).
    scaler = s4cpf.FrozenFeatureScaler.fit(tokens, global_features)
    pos_weight = (n_train - float(y_high.sum())) / max(float(y_high.sum()), 1.0)
    interaction_threshold = float(np.median(distance))

    train = {"x_tok": scaler.transform_tokens(tokens),
             "x_glob": scaler.transform_global(global_features),
             "y_reg": y_reg, "y_high": y_high}
    validation = {"x_tok": scaler.transform_tokens(v_tokens),
                  "x_glob": scaler.transform_global(v_global),
                  "y_reg": v_y_reg, "y_high": v_y_high}

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: dict[int, str] = {}
    checkpoint_sha256: dict[int, str] = {}
    train_losses: dict[int, float] = {}
    validation_losses: dict[int, float] = {}
    epochs_run: dict[int, int] = {}
    for seed in MODEL_SEEDS:
        target = model_dir / f"direct_seed{int(seed)}.pt"
        if target.exists():
            raise FileExistsError(
                f"Direct checkpoint already exists without a fit receipt: {target}; "
                "delete the incomplete models/ directory to re-run the fit")
        checkpoint = _fit_direct_seed(int(seed), train, validation, pos_weight=pos_weight,
                                      device=device, progress=progress)
        if int(checkpoint["seed"]) != int(seed):
            raise RuntimeError(f"trainer returned the wrong seed: {checkpoint['seed']}")
        if str(checkpoint["arch"]) != ARCH_DIRECT:
            raise RuntimeError(f"D2 trains only {ARCH_DIRECT}")
        for key in ("train_loss", "best_validation_loss"):
            if not np.isfinite(float(checkpoint[key])):
                raise RuntimeError(f"non-finite {key} for seed {seed}")
        s4cm.save_checkpoint(checkpoint, target)
        checkpoint_paths[int(seed)] = str(target)
        checkpoint_sha256[int(seed)] = sha256_file(target)
        train_losses[int(seed)] = float(checkpoint["train_loss"])
        validation_losses[int(seed)] = float(checkpoint["best_validation_loss"])
        epochs_run[int(seed)] = int(checkpoint["epochs_run"])

    inventory = ModelInventory(
        plan_id=PLAN_ID, arch=ARCH_DIRECT, seeds=tuple(int(s) for s in MODEL_SEEDS),
        param_count=int(param_count), run_dir=str(_resolve(run_dir)),
        checkpoint_paths=checkpoint_paths, checkpoint_sha256=checkpoint_sha256,
        scaler=scaler, pos_weight=float(pos_weight),
        interaction_threshold=float(interaction_threshold),
        train_losses=train_losses, validation_losses=validation_losses,
        epochs_run=epochs_run, n_train_groups=int(n_train),
        n_validation_groups=int(n_val),
        written_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    write_once_json(receipt_path, inventory.to_dict())
    return inventory


def score_models(
    inventory: ModelInventory,
    bundle: Mapping[str, np.ndarray],
    *,
    device: str | None = None,
) -> RiskScores:
    """Score a bundle with the five frozen checkpoints; ensemble = arithmetic mean.

    The bundle's raw features are standardised with the TRAIN-frozen scaler carried by the
    inventory (never refit). The ensemble is exactly ``risk_by_seed.mean(axis=0)``; never
    median, never seed-picked, never validation-weighted.
    """
    device = str(device or "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    tokens = np.asarray(bundle["tokens"], dtype=np.float64)
    global_features = np.asarray(bundle["global_features"], dtype=np.float64)
    group_seed = np.asarray(bundle["group_seed"], dtype=np.int64).ravel()
    n = tokens.shape[0]
    if not np.isfinite(tokens).all() or not np.isfinite(global_features).all():
        raise ValueError("score_models requires finite features (drop invalid groups first)")

    x_tok = inventory.scaler.transform_tokens(tokens)
    x_glob = inventory.scaler.transform_global(global_features)
    token_mask = np.ones((n, N_COMPETITORS), dtype=bool)

    per_seed = np.empty((len(inventory.seeds), n), dtype=np.float64)
    mu_only = np.empty_like(per_seed)
    class_only = np.empty_like(per_seed)
    for i, seed in enumerate(inventory.seeds):
        checkpoint = s4cm.load_checkpoint(inventory.checkpoint_paths[int(seed)])
        scores = s4cm.predict_decision_risk(checkpoint, x_tok, x_glob, token_mask,
                                            device=device)
        risk = np.asarray(scores["risk_score"], dtype=np.float64)
        if risk.shape != (n,) or not np.isfinite(risk).all():
            raise RuntimeError(f"non-finite risk_score from seed {seed}")
        per_seed[i] = risk
        mu_only[i] = np.asarray(scores["mu_only"], dtype=np.float64)
        class_only[i] = np.asarray(scores["class_only"], dtype=np.float64)
    return RiskScores(
        group_seed=group_seed,
        risk_ensemble=per_seed.mean(axis=0),  # THE frozen ensemble definition
        risk_by_seed=per_seed,
        mu_only_ensemble=mu_only.mean(axis=0),
        class_only_ensemble=class_only.mean(axis=0),
        seeds=tuple(int(s) for s in inventory.seeds),
    )


# --- mechanical strongest-baseline selection + one-shot validation gate (plan 7.6) -------------


@dataclass(frozen=True)
class BaselineReceipt:
    """The mechanical strongest-standard-baseline choice + the full audit trail."""
    plan_id: str
    strongest: str
    n_groups: int
    aurc_regret: dict[str, float]
    regret_at_80: dict[str, float]
    pr_auc_minus_prevalence: dict[str, float]
    tie_break_order: tuple[str, ...]
    tie_tolerance: float
    historical_reference_aurc: dict[str, float]  # COMBINED_TOP2_MAX, never eligible

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "strongest_standard_baseline": self.strongest,
            "n_groups": int(self.n_groups),
            "aurc_regret": {k: float(v) for k, v in self.aurc_regret.items()},
            "regret_at_80": {k: float(v) for k, v in self.regret_at_80.items()},
            "pr_auc_minus_prevalence": {k: float(v)
                                        for k, v in self.pr_auc_minus_prevalence.items()},
            "tie_break_order": list(self.tie_break_order),
            "tie_tolerance": float(self.tie_tolerance),
            "historical_reference_aurc": {k: float(v)
                                          for k, v in self.historical_reference_aurc.items()},
        }


@dataclass(frozen=True)
class ValidationGateReport:
    """The single validation adjudication outcome (plan 7.6)."""
    plan_id: str
    n_groups: int
    strongest_baseline: str
    baseline: BaselineReceipt
    conditions: dict[str, bool]
    metrics: dict[str, Any]
    interaction_threshold: float
    bindings: dict[str, Any]
    passed: bool
    status: str
    written_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "n_groups": int(self.n_groups),
            "strongest_standard_baseline": self.strongest_baseline,
            "baseline_receipt": self.baseline.to_dict(),
            "conditions": {k: bool(v) for k, v in self.conditions.items()},
            "metrics": self.metrics,
            "interaction_threshold": float(self.interaction_threshold),
            "bindings": self.bindings,
            "passed": bool(self.passed),
            "status": self.status,
            "written_at": self.written_at,
        }


def _aurc(risk: np.ndarray, regret: np.ndarray) -> float:
    return s4cs.aurc_regret(s4cs.order_by_risk(risk), regret)


def _regret80(risk: np.ndarray, regret: np.ndarray) -> float:
    return s4cs.regret_at_coverage(s4cs.order_by_risk(risk), regret, COVERAGE_MAIN)


def select_strongest_standard_baseline(
    validation: Mapping[str, np.ndarray]
) -> BaselineReceipt:
    """Unique lowest validation AURC_REGRET among the three eligible baselines.

    Ties within 1e-12 break in the frozen listed order (MARGIN_ONLY, SEED_VOTE_ENTROPY,
    SEED_MARGIN_STD). COMBINED_TOP2_MAX is recorded as historical reference and is NEVER
    eligible; the screen may never re-select (plan 7.6 step 2).
    """
    regret = np.asarray(validation["regret"], dtype=np.float64).ravel()
    y_high = np.asarray(validation["y_high"], dtype=np.int64).ravel()
    risks = validation["baseline_risks"]
    if not np.isfinite(regret).all():
        raise ValueError("validation regret must be finite for baseline selection")

    aurc: dict[str, float] = {}
    r80: dict[str, float] = {}
    pr: dict[str, float] = {}
    for name in STRONGEST_BASELINE_CANDIDATES:
        risk = np.asarray(risks[name], dtype=np.float64).ravel()
        if risk.shape != regret.shape or not np.isfinite(risk).all():
            raise ValueError(f"baseline risk {name!r} must be finite and group-aligned")
        aurc[name] = _aurc(risk, regret)
        r80[name] = _regret80(risk, regret)
        pr[name] = s4cs.high_regret_pr(risk, y_high)

    strongest = STRONGEST_BASELINE_CANDIDATES[0]
    for name in STRONGEST_BASELINE_CANDIDATES[1:]:
        if aurc[name] < aurc[strongest] - BASELINE_TIE_TOLERANCE:
            strongest = name

    historical: dict[str, float] = {}
    if BASELINE_COMBINED_TOP2_MAX in risks:
        combined = np.asarray(risks[BASELINE_COMBINED_TOP2_MAX], dtype=np.float64).ravel()
        historical[BASELINE_COMBINED_TOP2_MAX] = _aurc(combined, regret)

    return BaselineReceipt(
        plan_id=PLAN_ID, strongest=str(strongest), n_groups=int(regret.size),
        aurc_regret=aurc, regret_at_80=r80, pr_auc_minus_prevalence=pr,
        tie_break_order=tuple(STRONGEST_BASELINE_CANDIDATES),
        tie_tolerance=BASELINE_TIE_TOLERANCE, historical_reference_aurc=historical)


def evaluate_validation_gate(
    scores: RiskScores, truth: Mapping[str, Any]
) -> ValidationGateReport:
    """Adjudicate the plan-7.6 step-3 validation conditions exactly once (fail-closed).

    ``scores`` is the five-seed :class:`RiskScores` on the validation role; ``truth``
    carries regret, y_high, the baseline risks, the frozen baseline receipt and the group
    seeds. Conditions: Direct ensemble vs the frozen main baseline has
    ``I_AURC_REGRET>=0.05`` and ``I_REGRET@80>=0.03``, and at least 4/5 Direct seeds are
    positive on both improvements. A non-positive baseline denominator yields NaN
    improvements, which fail — never an epsilon, never an absolute difference.
    """
    ensemble = np.asarray(scores.risk_ensemble, dtype=np.float64).ravel()
    by_seed = np.asarray(scores.risk_by_seed, dtype=np.float64)
    score_seeds = np.asarray(scores.group_seed, dtype=np.int64).ravel()

    regret = np.asarray(truth["regret"], dtype=np.float64).ravel()
    risks = truth["baseline_risks"]
    receipt = truth["baseline_receipt"]
    if isinstance(receipt, Mapping):
        receipt = BaselineReceipt(
            plan_id=str(receipt["plan_id"]),
            strongest=str(receipt["strongest_standard_baseline"]),
            n_groups=int(receipt["n_groups"]),
            aurc_regret={k: float(v) for k, v in receipt["aurc_regret"].items()},
            regret_at_80={k: float(v) for k, v in receipt["regret_at_80"].items()},
            pr_auc_minus_prevalence={k: float(v)
                                     for k, v in receipt["pr_auc_minus_prevalence"].items()},
            tie_break_order=tuple(receipt["tie_break_order"]),
            tie_tolerance=float(receipt["tie_tolerance"]),
            historical_reference_aurc={
                k: float(v) for k, v in receipt["historical_reference_aurc"].items()})
    truth_seeds = np.asarray(truth["group_seed"], dtype=np.int64).ravel()
    groups = regret.size

    finite_inputs = bool(
        np.isfinite(ensemble).all() and np.isfinite(by_seed).all()
        and np.isfinite(regret).all()
        and by_seed.shape == (len(MODEL_SEEDS), groups)
        and ensemble.shape == (groups,))
    if not finite_inputs:
        conditions = {key: False for key in (
            "c1_i_aurc_regret", "c2_i_regret80", "c3_seeds_positive",
            "c4_finite_matching")}
        nan = float("nan")
        return ValidationGateReport(
            plan_id=PLAN_ID, n_groups=int(groups),
            strongest_baseline=receipt.strongest, baseline=receipt,
            conditions=conditions,
            metrics={"aurc_regret_ensemble": nan, "aurc_regret_baseline": nan,
                     "i_aurc_regret": nan, "regret80_ensemble": nan,
                     "regret80_baseline": nan, "i_regret80": nan,
                     "seed_i_aurc": [], "seed_i_regret80": [],
                     "seeds_i_aurc_positive": 0, "seeds_i_regret80_positive": 0},
            interaction_threshold=float(truth.get("interaction_threshold", nan)),
            bindings=dict(truth.get("bindings", {})),
            passed=False, status=STATUS_VALIDATION_NOT_QUALIFIED,
            written_at=time.strftime("%Y-%m-%dT%H:%M:%S"))

    base_risk = np.asarray(risks[receipt.strongest], dtype=np.float64).ravel()
    aurc_base = _aurc(base_risk, regret)
    aurc_ens = _aurc(ensemble, regret)
    i_aurc = s4cs.improvement(aurc_base, aurc_ens)
    r80_base = _regret80(base_risk, regret)
    r80_ens = _regret80(ensemble, regret)
    i_r80 = s4cs.improvement(r80_base, r80_ens)

    seed_i_aurc: list[float] = []
    seed_i_r80: list[float] = []
    for i in range(by_seed.shape[0]):
        seed_i_aurc.append(s4cs.improvement(aurc_base, _aurc(by_seed[i], regret)))
        seed_i_r80.append(s4cs.improvement(r80_base, _regret80(by_seed[i], regret)))
    n_seed_aurc_pos = int(sum(1 for v in seed_i_aurc if v > 0.0))
    n_seed_r80_pos = int(sum(1 for v in seed_i_r80 if v > 0.0))

    matching = bool(
        np.isfinite(base_risk).all()
        and np.isfinite([aurc_base, aurc_ens, r80_base, r80_ens]).all()
        and score_seeds.shape == truth_seeds.shape
        and np.array_equal(score_seeds, truth_seeds)
        and np.allclose(ensemble, by_seed.mean(axis=0), rtol=0.0, atol=1e-15)
        and tuple(int(s) for s in scores.seeds) == tuple(int(s) for s in MODEL_SEEDS))

    conditions = {
        "c1_i_aurc_regret": bool(i_aurc >= VALIDATION_I_AURC_MIN),
        "c2_i_regret80": bool(i_r80 >= VALIDATION_I_REGRET80_MIN),
        "c3_seeds_positive": bool(n_seed_aurc_pos >= VALIDATION_MIN_POSITIVE_SEEDS
                                  and n_seed_r80_pos >= VALIDATION_MIN_POSITIVE_SEEDS),
        "c4_finite_matching": bool(matching),
    }
    passed = bool(all(conditions.values()))
    metrics = {
        "aurc_regret_ensemble": float(aurc_ens),
        "aurc_regret_baseline": float(aurc_base),
        "i_aurc_regret": float(i_aurc),
        "regret80_ensemble": float(r80_ens),
        "regret80_baseline": float(r80_base),
        "i_regret80": float(i_r80),
        "seed_i_aurc": [float(v) for v in seed_i_aurc],
        "seed_i_regret80": [float(v) for v in seed_i_r80],
        "seeds_i_aurc_positive": n_seed_aurc_pos,
        "seeds_i_regret80_positive": n_seed_r80_pos,
    }
    return ValidationGateReport(
        plan_id=PLAN_ID, n_groups=int(groups),
        strongest_baseline=receipt.strongest, baseline=receipt,
        conditions=conditions, metrics=metrics,
        interaction_threshold=float(truth.get("interaction_threshold", float("nan"))),
        bindings=dict(truth.get("bindings", {})),
        passed=passed,
        status=(STATUS_VALIDATION_QUALIFIED if passed else STATUS_VALIDATION_NOT_QUALIFIED),
        written_at=time.strftime("%Y-%m-%dT%H:%M:%S"))


def freeze_final_selection(report: ValidationGateReport, inventory: ModelInventory,
                           dev_dir: str | Path) -> Path:
    """Write the five create-once freeze artifacts exactly once; return the final path.

    Refuses a failed gate and any second write. ``final_selection.json`` binds the five
    checkpoints (path + SHA256), the unique strongest baseline with the full
    offline-auditable metrics trail, the interaction threshold, the exact parameter count,
    the primary-score definition and the code hashes (plan 7.6 step 4).
    """
    if not report.passed:
        raise ValueError(
            f"cannot freeze a failed validation gate (status {report.status}); "
            "the terminal state must be recorded instead and D2 stops before the screen")
    dev_dir = _resolve(dev_dir)
    targets = {name: dev_dir / name for name in FREEZE_ARTIFACTS}
    for name, path in targets.items():
        if path.exists():
            raise FileExistsError(
                f"Stage 4D-D2 freeze artifact already exists (create-once): {path}")

    for seed, path in inventory.checkpoint_paths.items():
        digest = sha256_file(Path(path))
        if digest != inventory.checkpoint_sha256[int(seed)]:
            raise ValueError(f"checkpoint drifted since the fit: {path}")

    schema = dict(s4cpf.feature_schema())
    schema["plan_id"] = PLAN_ID
    schema["reused_feature_module"] = "code/ac_twin/stage4cp_features.py"
    schema["reused_feature_module_sha256"] = sha256_file(
        ROOT / "code/ac_twin/stage4cp_features.py")

    write_once_json(targets["feature_schema.json"], schema)
    inventory.scaler.save(targets["feature_scaler.npz"])
    write_once_json(targets["model_inventory.json"], inventory.to_dict())
    write_once_json(targets["validation_gate_report.json"], report.to_dict())

    bindings = dict(report.bindings)
    final = {
        "plan_id": PLAN_ID,
        "status": report.status,
        "architecture": ARCH_DIRECT,
        "param_count": int(inventory.param_count),
        "model_seeds": [int(s) for s in inventory.seeds],
        "ensemble_definition": "arithmetic_mean_of_five_seeds",
        "primary_score_definition": PRIMARY_SCORE_DEFINITION,
        "checkpoints": {
            str(seed): {"path": str(inventory.checkpoint_paths[int(seed)]),
                        "sha256": str(inventory.checkpoint_sha256[int(seed)])}
            for seed in inventory.seeds
        },
        "strongest_standard_baseline": report.strongest_baseline,
        "baseline_validation_metrics": {
            name: {
                "aurc_regret": float(report.baseline.aurc_regret[name]),
                "regret_at_80": float(report.baseline.regret_at_80[name]),
                "pr_auc_minus_prevalence": float(
                    report.baseline.pr_auc_minus_prevalence[name]),
            }
            for name in STRONGEST_BASELINE_CANDIDATES
        },
        "baseline_tie_break_order": list(STRONGEST_BASELINE_CANDIDATES),
        "baseline_tie_tolerance": BASELINE_TIE_TOLERANCE,
        "historical_reference_aurc": {
            k: float(v) for k, v in report.baseline.historical_reference_aurc.items()},
        "interaction_threshold": float(inventory.interaction_threshold),
        "interaction_threshold_source":
            "train preintervention_min_distance median (train-only)",
        "bindings": bindings,
        "code_sha256": {
            "ac_twin/stage4d_external_training.py": sha256_file(
                ROOT / "code/ac_twin/stage4d_external_training.py"),
        },
        "validation_gate_report_sha256": sha256_file(
            targets["validation_gate_report.json"]),
        "model_inventory_sha256": sha256_file(targets["model_inventory.json"]),
        "feature_scaler_sha256": sha256_file(targets["feature_scaler.npz"]),
        "feature_schema_sha256": sha256_file(targets["feature_schema.json"]),
        "n_validation_groups": int(report.n_groups),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(targets["final_selection.json"], final)
    return targets["final_selection.json"]


__all__ = [
    "ARCH_DIRECT",
    "BASELINE_COMBINED_TOP2_MAX",
    "BASELINE_MARGIN",
    "BASELINE_SEED_MARGIN_STD",
    "BASELINE_SEED_VOTE_ENTROPY",
    "BASELINE_TIE_TOLERANCE",
    "CONSTANT_CONTRACT_CONTINUOUS",
    "COVERAGE_MAIN",
    "DEV_SCREEN_GROUPS",
    "DEV_TRAIN_GROUPS",
    "DEV_VALIDATION_GROUPS",
    "DIRECT_FLAT_DIM",
    "DIRECT_HIDDEN_WIDTH",
    "DROPOUT",
    "EARLY_STOP_PATIENCE",
    "EXPECTED_TRAINABLE_PARAMETERS",
    "FIT_RECEIPT_NAME",
    "FREEZE_ARTIFACTS",
    "GLOBAL_DIM",
    "HIGH_REGRET_THRESHOLD",
    "IDLE_PLAN_INDEX",
    "LAMBDA_CLS",
    "LEARNING_RATE",
    "MAX_EPOCHS",
    "MODEL_SEEDS",
    "N_ACTIONS",
    "N_COMPETITORS",
    "PLAN_ID",
    "PREDICTOR_PAIR_SEEDS",
    "PRIMARY_SCORE_DEFINITION",
    "RESPONSE_PREDICTOR_SEEDS",
    "STATUS_SCHEMA_INVALID",
    "STATUS_TRAIN_SUPPORT_INSUFFICIENT",
    "STATUS_TRAIN_SUPPORT_OK",
    "STATUS_VALIDATION_NOT_QUALIFIED",
    "STATUS_VALIDATION_QUALIFIED",
    "STRONGEST_BASELINE_CANDIDATES",
    "TOKEN_DIM",
    "TRAIN_BATCH_SIZE",
    "TRAIN_MIN_HIGH_REGRET_GROUPS",
    "TRAIN_MIN_NONZERO_REGRET_GROUPS",
    "VALIDATION_I_AURC_MIN",
    "VALIDATION_I_REGRET80_MIN",
    "VALIDATION_MIN_POSITIVE_SEEDS",
    "WEIGHT_DECAY",
    "BaselineReceipt",
    "ModelInventory",
    "RiskScores",
    "ValidationGateReport",
    "compute_true_labels",
    "count_direct_parameters",
    "evaluate_train_support_gate",
    "evaluate_validation_gate",
    "fit_five_direct_models",
    "freeze_final_selection",
    "group_seed_from_scenario_id",
    "score_models",
    "select_strongest_standard_baseline",
]
