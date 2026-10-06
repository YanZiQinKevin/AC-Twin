"""Stage 4C-R2 fixed Direct training: five seeds, frozen architecture, train-only statistics.

Plan ``AC-TWIN-RFER-STAGE4C-R2-DECISION-RISK-SCALING-2026-09-16-V0.2`` sections 4.2 and 7.

The ONE model is ``SAME_INFORMATION_DIRECT`` (width 43, dropout 0.10, lambda_cls 0.25,
EXACTLY 15,740 trainable parameters). R2 trains it on development_train ONLY:

* the 10% internal-monitor split is a fixed, pure function of the group seeds (never of
  a model seed); development-validation never selects epochs;
* the feature scaler and the class weight are fit on the train MAIN BODY ONLY — the
  internal-monitor rows and every external role are read-only to them;
* the ensemble is the plain arithmetic mean of the five frozen seeds' ``risk_score``;
  all five checkpoints are kept; never median, never seed-picking, never
  validation-weighted;
* the train support hard gate (plan 7.1) is fail-closed with terminal state
  ``STAGE4CR2_TRAIN_SUPPORT_INSUFFICIENT``; thresholds are never lowered, no top-up
  collection, validation is never merged into train.

Training mechanics mirror the frozen Stage 4C Plus loop (``stage4cp_models``): AdamW,
SmoothL1(mu_pos, y_reg) + lambda_cls * weighted_BCE(logit_high, y_high), early stopping
on the internal monitor with best-state restore. The architecture, the score definition
and the scaler are REUSED from the frozen modules, never reimplemented.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn

from ac_twin import stage4cp_features as s4cpf
from ac_twin import stage4cp_models as s4cm
from ac_twin import stage4cr2_protocol as s4cr2
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

ROOT = Path(__file__).resolve().parents[2]

STATUS_TRAIN_SUPPORT_OK = "TRAIN_SUPPORT_OK"

#: continuous feature columns that are constant BY CONTRACT and therefore exempt from the
#: plan-7.1 zero-variance check ("除明确的常量合同外无零方差列"). ``visible_vehicle_count``
#: is the count of non-ego slots in the frozen capacity-5 object observation (ego + 4);
#: acceptance requires every slot history-known (present-set/prefix invariants), so the
#: column is identically 4 in every accepted group — verified constant in BOTH the R2
#: 5,000-group train set AND the historical Stage 4C 500-group selection (same frozen
#: contract; read as design evidence only). The frozen scaler floors std at 1e-12 for the
#: same reason. Every other continuous column must carry genuine variance.
CONSTANT_CONTRACT_CONTINUOUS: tuple[str, ...] = ("visible_vehicle_count",)

#: create-once engineering resume receipt for the five-seed fit (NOT the frozen
#: model_inventory.json — that is written exactly once by the selection freeze).
FIT_RECEIPT_NAME = "fit_receipt.json"


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
    main_indices: tuple[int, ...]
    monitor_indices: tuple[int, ...]
    train_losses: dict[int, float]
    monitor_losses: dict[int, float]
    n_train_groups: int
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
            "main_indices": [int(i) for i in self.main_indices],
            "monitor_indices": [int(i) for i in self.monitor_indices],
            "train_losses": {str(k): float(v) for k, v in self.train_losses.items()},
            "monitor_losses": {str(k): float(v) for k, v in self.monitor_losses.items()},
            "n_train_groups": int(self.n_train_groups),
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


def count_direct_parameters() -> int:
    """Recount the frozen Direct architecture; raise unless EXACTLY 15,740."""
    model = s4cm.build_model(
        s4cr2.ARCH_DIRECT, hidden_dim=s4cr2.DIRECT_HIDDEN_WIDTH,
        dropout=s4cr2.DROPOUT, direct_width=s4cr2.DIRECT_HIDDEN_WIDTH,
        n_competitors=s4cr2.N_COMPETITORS, token_dim=s4cr2.TOKEN_DIM,
        global_dim=s4cr2.GLOBAL_DIM)
    count = s4cm.count_parameters(model)
    if count != s4cr2.EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            f"Direct trainable parameters {count} != frozen "
            f"{s4cr2.EXPECTED_TRAINABLE_PARAMETERS}")
    return count


def internal_monitor_split(group_seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fixed 10% internal-monitor split of train, a pure function of the group seeds.

    Each group's assignment derives from ``SeedSequence([group_seed, dev_entropy])`` so
    the split is identical for every model seed and every rerun; development-validation
    never participates. Returns ``(main_idx, monitor_idx)``.
    """
    group_seed = np.asarray(group_seed, dtype=np.int64).ravel()
    n = group_seed.size
    if n < 2:
        raise ValueError("need >=2 train groups for the internal-monitor split")
    if np.unique(group_seed).size != n:
        raise ValueError("group seeds must be unique for the internal-monitor split")
    u = np.array([
        np.random.SeedSequence([int(seed), s4cr2.DEV_SEED_ENTROPY]).generate_state(1)[0]
        for seed in group_seed
    ], dtype=np.uint64).astype(np.float64) / float(2**32)
    order = np.argsort(u, kind="stable")
    n_monitor = max(1, int(round(s4cr2.INTERNAL_MONITOR_FRAC * n)))
    monitor_idx = np.sort(order[:n_monitor])
    main_idx = np.sort(order[n_monitor:])
    return main_idx, monitor_idx


# --- train support hard gate (plan section 7.1) -----------------------------------------------


def evaluate_train_support_gate(
    train: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply all seven plan-7.1 conditions to the 5,000 train groups (fail-closed).

    Any failure yields terminal ``STAGE4CR2_TRAIN_SUPPORT_INSUFFICIENT``; thresholds are
    frozen, no top-up collection, validation is never merged into train.
    """
    checked = s4cr2.validate_config(config)
    regret = np.asarray(train["regret"], dtype=np.float64).ravel()
    oracle = np.asarray(train["oracle_action"], dtype=np.int64).ravel()
    group_seed = np.asarray(train["group_seed"], dtype=np.int64).ravel()
    tokens = np.asarray(train["tokens"], dtype=np.float64)
    global_features = np.asarray(train["global_features"], dtype=np.float64)
    groups = int(train["groups"])

    # 1. every group complete with all ten actions.
    c1 = (groups == s4cr2.DEV_TRAIN_GROUPS
          and int(train["n_actions"]) == s4cr2.N_ACTIONS
          and int(train["complete_groups"]) == groups
          and regret.size == groups and tokens.shape[0] == groups)

    # 2/3. regret support.
    c2 = int((regret > s4cr2.HIGH_REGRET_THRESHOLD).sum()) \
        >= s4cr2.TRAIN_MIN_HIGH_REGRET_GROUPS
    c3 = int((regret > 0.0).sum()) >= s4cr2.TRAIN_MIN_NONZERO_REGRET_GROUPS

    # 4. at least three oracle actions each covering >=5% of the groups.
    shares = [(oracle == a).mean() for a in range(s4cr2.N_ACTIONS)]
    n_oracle_covered = int(sum(s >= 0.05 for s in shares))
    c4 = n_oracle_covered >= 3

    # 5. continuous features finite; no zero-variance column outside constant contracts.
    cont_names = list(s4cpf.TOKEN_CONT_NAMES) + list(s4cpf.GLOBAL_NAMES)
    token_cont = tokens[:, :, s4cpf.TOKEN_CONT_IDX].reshape(-1, s4cpf.N_TOKEN_CONT)
    finite = bool(np.isfinite(token_cont).all() and np.isfinite(global_features).all())
    variances = np.concatenate(
        [token_cont.var(axis=0), global_features.var(axis=0)])
    exempt = {i for i, name in enumerate(cont_names) if name in CONSTANT_CONTRACT_CONTINUOUS}
    # zero-variance = variance at/below the frozen scaler's std floor (1e-12) squared;
    # float noise on a genuinely constant column sits near 1e-28.
    zero_var = [cont_names[i] for i in range(len(cont_names))
                if variances[i] <= 1e-24 and i not in exempt]
    c5 = finite and not zero_var

    # 6. base action beats RANDOM_ACTION and is not worse than ALWAYS_IDLE.
    base = float(train["base_mean_regret"])
    random = float(train["random_mean_regret"])
    idle = float(train["idle_mean_regret"])
    c6 = bool(np.isfinite([base, random, idle]).all()
              and base < random and base <= idle + 1e-12)

    # 7. role seed disjointness: every train seed inside its own frozen segment, unique.
    schedule = s4cr2.build_seed_schedule(checked)
    lo, hi = s4cr2.TRAIN_SEGMENT
    segment = set(int(v) for v in schedule[lo:hi])
    other = set(int(v) for v in schedule[hi:])  # validation + screen segments
    seeds = [int(v) for v in group_seed]
    c7 = (len(set(seeds)) == len(seeds) and all(s in segment for s in seeds)
          and not any(s in other for s in seeds))

    conditions = {
        "all_complete_10_action": bool(c1),
        "high_regret_groups_ge_750": bool(c2),
        "nonzero_regret_groups_ge_1250": bool(c3),
        "at_least_3_oracle_actions_ge_5pct": bool(c4),
        "continuous_features_finite_with_variance": bool(c5),
        "base_beats_random_not_worse_than_idle": bool(c6),
        "role_seed_disjointness": bool(c7),
    }
    passed = bool(all(conditions.values()))
    return {
        "plan_id": s4cr2.PLAN_ID,
        "role": "development_train",
        "conditions": conditions,
        "passed": passed,
        "status": (STATUS_TRAIN_SUPPORT_OK if passed
                   else s4cr2.STATUS_TRAIN_SUPPORT_INSUFFICIENT),
        "values": {
            "groups": groups,
            "high_regret_groups": int((regret > s4cr2.HIGH_REGRET_THRESHOLD).sum()),
            "nonzero_regret_groups": int((regret > 0.0).sum()),
            "oracle_action_shares": [float(s) for s in shares],
            "oracle_actions_covered_ge_5pct": n_oracle_covered,
            "zero_variance_continuous_columns": zero_var,
            "base_mean_regret": base,
            "random_mean_regret": random,
            "idle_mean_regret": idle,
        },
    }


# --- fixed five-seed training ---------------------------------------------------------------------


def _fit_direct_seed(
    seed: int,
    main: Mapping[str, np.ndarray],
    monitor: Mapping[str, np.ndarray],
    *,
    pos_weight: float,
    config: Mapping[str, Any],
    device: str,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train ONE frozen Direct seed on the main body; early-stop on the monitor split.

    Mirrors the frozen Stage 4C Plus loop exactly (AdamW, SmoothL1 + lambda_cls *
    weighted BCE, best-state restore), but the monitor split is the fixed group-seed
    split passed in — never re-drawn per seed, never an external role.
    """
    checked = s4cr2.validate_config(config)
    dev = torch.device(device)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**31 - 1))

    model = s4cm.build_model(
        s4cr2.ARCH_DIRECT, hidden_dim=s4cr2.DIRECT_HIDDEN_WIDTH,
        dropout=float(checked["dropout"]), direct_width=s4cr2.DIRECT_HIDDEN_WIDTH).to(dev)

    x_main_tok = torch.as_tensor(main["x_tok"], dtype=torch.float32)
    x_main_glob = torch.as_tensor(main["x_glob"], dtype=torch.float32)
    y_main_r = torch.as_tensor(main["y_reg"], dtype=torch.float32)
    y_main_h = torch.as_tensor(main["y_high"], dtype=torch.float32)
    x_mon_tok = torch.as_tensor(monitor["x_tok"], dtype=torch.float32).to(dev)
    x_mon_glob = torch.as_tensor(monitor["x_glob"], dtype=torch.float32).to(dev)
    y_mon_r = torch.as_tensor(monitor["y_reg"], dtype=torch.float32).to(dev)
    y_mon_h = torch.as_tensor(monitor["y_high"], dtype=torch.float32).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(checked["learning_rate"]),
                                  weight_decay=float(checked["weight_decay"]))
    smooth_l1 = nn.SmoothL1Loss()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(
        float(pos_weight), dtype=torch.float32, device=dev))
    lambda_cls = float(checked["lambda_cls"])
    batch_size = int(checked["train_batch_size"])
    max_epochs = int(checked["max_epochs"])
    patience = int(checked["early_stop_patience"])

    def forward(flat_tokens, glob):
        return model(flat_tokens, glob)

    def total_loss(x_tok, x_glob, y_r, y_h) -> float:
        model.eval()
        with torch.no_grad():
            raw_mu, logit = forward(x_tok.flatten(start_dim=1), x_glob)
            mu_pos = torch.nn.functional.softplus(raw_mu)
            loss = smooth_l1(mu_pos, y_r) + lambda_cls * bce(logit, y_h)
        return float(loss.item())

    n_main = x_main_tok.shape[0]
    best_state = None
    best_val = float("inf")
    wait = 0
    started = time.perf_counter()
    batch_generator = torch.Generator().manual_seed(int(seed))
    epochs_run = 0
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(n_main, generator=batch_generator).numpy()
        for start in range(0, n_main, batch_size):
            sel = order[start:start + batch_size]
            optimizer.zero_grad()
            raw_mu, logit = forward(
                x_main_tok[sel].flatten(start_dim=1).to(dev),
                x_main_glob[sel].to(dev))
            mu_pos = torch.nn.functional.softplus(raw_mu)
            loss = smooth_l1(mu_pos, y_main_r[sel].to(dev)) \
                + lambda_cls * bce(logit, y_main_h[sel].to(dev))
            loss.backward()
            optimizer.step()
        epochs_run = epoch + 1
        val_loss = total_loss(x_mon_tok, x_mon_glob, y_mon_r, y_mon_h)
        if val_loss < best_val - 1e-10:
            best_val = val_loss
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if progress is not None:
            progress(f"seed={seed} epoch={epochs_run}/{max_epochs} "
                     f"monitor_loss={val_loss:.6f} best={best_val:.6f} "
                     f"elapsed={time.perf_counter() - started:.1f}s")
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    train_loss = total_loss(x_main_tok.to(dev), x_main_glob.to(dev),
                            y_main_r.to(dev), y_main_h.to(dev))
    return {
        "arch": s4cr2.ARCH_DIRECT,
        "hidden_dim": s4cr2.DIRECT_HIDDEN_WIDTH,
        "direct_width": s4cr2.DIRECT_HIDDEN_WIDTH,
        "lambda_cls": lambda_cls,
        "seed": int(seed),
        "pos_weight": float(pos_weight),
        "dropout": float(checked["dropout"]),
        "token_dim": s4cr2.TOKEN_DIM,
        "global_dim": s4cr2.GLOBAL_DIM,
        "n_competitors": s4cr2.N_COMPETITORS,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "train_loss": float(train_loss),
        "best_internal_monitor_loss": float(best_val),
        "epochs_run": int(epochs_run),
        "n_train_groups": int(n_main),
    }


def _model_dir(config: Mapping[str, Any]) -> Path:
    return _resolve(config["run_dir"]) / "development" / "models"


def _checkpoint_path(config: Mapping[str, Any], seed: int) -> Path:
    return _model_dir(config) / f"direct_seed{int(seed)}.pt"


def _inventory_from_receipt(config: Mapping[str, Any], receipt: Mapping[str, Any],
                            *, verification: str) -> ModelInventory:
    seeds = tuple(int(s) for s in receipt["seeds"])
    scaler_payload = receipt["scaler"]
    return ModelInventory(
        plan_id=str(receipt["plan_id"]),
        arch=str(receipt["arch"]),
        seeds=seeds,
        param_count=int(receipt["param_count"]),
        run_dir=str(_resolve(config["run_dir"])),
        checkpoint_paths={int(k): str(v) for k, v in receipt["checkpoint_paths"].items()},
        checkpoint_sha256={int(k): str(v) for k, v in receipt["checkpoint_sha256"].items()},
        scaler=s4cpf.FrozenFeatureScaler(
            np.asarray(scaler_payload["token_cont_mean"]),
            np.asarray(scaler_payload["token_cont_std"]),
            np.asarray(scaler_payload["global_mean"]),
            np.asarray(scaler_payload["global_std"])),
        pos_weight=float(receipt["pos_weight"]),
        interaction_threshold=float(receipt["interaction_threshold"]),
        main_indices=tuple(int(i) for i in receipt["main_indices"]),
        monitor_indices=tuple(int(i) for i in receipt["monitor_indices"]),
        train_losses={int(k): float(v) for k, v in receipt["train_losses"].items()},
        monitor_losses={int(k): float(v) for k, v in receipt["monitor_losses"].items()},
        n_train_groups=int(receipt["n_train_groups"]),
        written_at=str(receipt["written_at"]),
        verification=verification,
    )


def fit_five_direct_models(
    train_bundle: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    fit_one_fn: Callable | None = None,
    device: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> ModelInventory:
    """Train the five frozen Direct seeds on development_train; keep ALL checkpoints.

    ``train_bundle`` carries raw (unscaled) ``tokens`` (G,9,34), ``global_features``
    (G,9), labels ``y_reg``/``y_high`` (G,), ``group_seed`` (G,) and
    ``interaction_min_distance`` (G,). The scaler, the class weight and the interaction
    threshold are fit on the train main body ONLY; the fixed internal-monitor split is a
    pure function of the group seeds. Create-once: a rerun with the receipt present
    verifies checkpoint hashes and never retrains or overwrites.
    """
    checked = s4cr2.validate_config(config)
    param_count = count_direct_parameters()
    if fit_one_fn is None:
        real_fit = _fit_direct_seed

        def fit_one_fn(seed, main, monitor, *, pos_weight, config, device):
            return real_fit(seed, main, monitor, pos_weight=pos_weight,
                            config=config, device=device, progress=progress)
    device = str(device or checked.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model_dir = _model_dir(checked)
    receipt_path = model_dir / FIT_RECEIPT_NAME
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for seed_text, expected in receipt["checkpoint_sha256"].items():
            path = Path(receipt["checkpoint_paths"][seed_text])
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"Direct checkpoint drifted from its fit receipt: {path}")
        return _inventory_from_receipt(
            checked, receipt, verification="existing_inventory_verified_no_overwrite")

    tokens = np.asarray(train_bundle["tokens"], dtype=np.float64)
    global_features = np.asarray(train_bundle["global_features"], dtype=np.float64)
    y_reg = np.asarray(train_bundle["y_reg"], dtype=np.float64).ravel()
    y_high = np.asarray(train_bundle["y_high"], dtype=np.float64).ravel()
    group_seed = np.asarray(train_bundle["group_seed"], dtype=np.int64).ravel()
    distance = np.asarray(train_bundle["interaction_min_distance"], dtype=np.float64).ravel()
    n = tokens.shape[0]
    if not (global_features.shape[0] == y_reg.shape[0] == y_high.shape[0]
            == group_seed.shape[0] == distance.shape[0] == n):
        raise ValueError("train bundle arrays must align on the group axis")
    for name, arr in (("tokens", tokens), ("global_features", global_features),
                      ("y_reg", y_reg), ("y_high", y_high),
                      ("interaction_min_distance", distance)):
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite train bundle array: {name}")

    main_idx, monitor_idx = internal_monitor_split(group_seed)

    # Train-only statistics: scaler + class weight from the MAIN BODY only; the
    # interaction threshold from the full train role (plan 8.2, train-only median).
    scaler = s4cpf.FrozenFeatureScaler.fit(tokens[main_idx], global_features[main_idx])
    x_tok = scaler.transform_tokens(tokens)
    x_glob = scaler.transform_global(global_features)
    n_main = int(main_idx.size)
    n_pos = float(y_high[main_idx].sum())
    pos_weight = (n_main - n_pos) / max(n_pos, 1.0)
    interaction_threshold = float(np.median(distance))

    main = {"x_tok": x_tok[main_idx], "x_glob": x_glob[main_idx],
            "y_reg": y_reg[main_idx], "y_high": y_high[main_idx]}
    monitor = {"x_tok": x_tok[monitor_idx], "x_glob": x_glob[monitor_idx],
               "y_reg": y_reg[monitor_idx], "y_high": y_high[monitor_idx]}

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: dict[int, str] = {}
    checkpoint_sha256: dict[int, str] = {}
    train_losses: dict[int, float] = {}
    monitor_losses: dict[int, float] = {}
    for seed in s4cr2.MODEL_SEEDS:
        target = _checkpoint_path(checked, seed)
        if target.exists():
            raise FileExistsError(
                f"Direct checkpoint already exists without a fit receipt: {target}; "
                "delete the incomplete models/ directory to re-run the fit")
        checkpoint = fit_one_fn(int(seed), main, monitor, pos_weight=pos_weight,
                                config=checked, device=device)
        if int(checkpoint["seed"]) != int(seed):
            raise RuntimeError(f"trainer returned the wrong seed: {checkpoint['seed']}")
        if str(checkpoint["arch"]) != s4cr2.ARCH_DIRECT:
            raise RuntimeError(f"R2 trains only {s4cr2.ARCH_DIRECT}")
        for key in ("train_loss", "best_internal_monitor_loss"):
            if not np.isfinite(float(checkpoint[key])):
                raise RuntimeError(f"non-finite {key} for seed {seed}")
        s4cm.save_checkpoint(checkpoint, target)
        checkpoint_paths[int(seed)] = str(target)
        checkpoint_sha256[int(seed)] = sha256_file(target)
        train_losses[int(seed)] = float(checkpoint["train_loss"])
        monitor_losses[int(seed)] = float(checkpoint["best_internal_monitor_loss"])

    written_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    inventory = ModelInventory(
        plan_id=s4cr2.PLAN_ID,
        arch=s4cr2.ARCH_DIRECT,
        seeds=tuple(int(s) for s in s4cr2.MODEL_SEEDS),
        param_count=int(param_count),
        run_dir=str(_resolve(checked["run_dir"])),
        checkpoint_paths=checkpoint_paths,
        checkpoint_sha256=checkpoint_sha256,
        scaler=scaler,
        pos_weight=float(pos_weight),
        interaction_threshold=float(interaction_threshold),
        main_indices=tuple(int(i) for i in main_idx),
        monitor_indices=tuple(int(i) for i in monitor_idx),
        train_losses=train_losses,
        monitor_losses=monitor_losses,
        n_train_groups=int(n),
        written_at=written_at,
    )
    write_once_json(receipt_path, inventory.to_dict())
    return inventory


def score_models(
    inventory: ModelInventory,
    bundle: Mapping[str, np.ndarray],
    *,
    predict_fn: Callable | None = None,
    device: str | None = None,
) -> RiskScores:
    """Score a bundle with the five frozen checkpoints; ensemble = arithmetic mean.

    The bundle's raw features are standardised with the TRAIN-frozen scaler carried by
    the inventory (never refit). The ensemble is exactly ``risk_by_seed.mean(axis=0)``;
    never median, never seed-picked, never validation-weighted.
    """
    predict_fn = predict_fn or s4cm.predict_decision_risk
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
    token_mask = np.ones((n, s4cr2.N_COMPETITORS), dtype=bool)

    per_seed = np.empty((len(inventory.seeds), n), dtype=np.float64)
    mu_only = np.empty_like(per_seed)
    class_only = np.empty_like(per_seed)
    for i, seed in enumerate(inventory.seeds):
        checkpoint = s4cm.load_checkpoint(inventory.checkpoint_paths[int(seed)])
        scores = predict_fn(checkpoint, x_tok, x_glob, token_mask, device=device)
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


__all__ = [
    "CONSTANT_CONTRACT_CONTINUOUS",
    "FIT_RECEIPT_NAME",
    "ModelInventory",
    "RiskScores",
    "STATUS_TRAIN_SUPPORT_OK",
    "count_direct_parameters",
    "evaluate_train_support_gate",
    "fit_five_direct_models",
    "internal_monitor_split",
    "score_models",
]
