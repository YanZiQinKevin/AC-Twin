"""Stage 4D-D2 unique development-screen orchestration driver (plan section 8.1).

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` section 8.1, the one-shot
sequence, each step create-once and resumable:

    split        groups.npz -> inference_inputs.npz + sealed_truth.npz (physical separation)
    score        frozen-stack predictions + five risk classes + 9x34 Direct tokens + five Direct
                 seed scores, computed from ``inference_inputs.npz`` ONLY (the scoring process is
                 never handed the sealed-truth path, so it is structurally unable to read truth)
    determinism  >=32 groups re-scored twice on the same device/batch/dtype/order, elementwise
                 identical (plan 8.1 step 4)
    freeze       screen_scores.npz + receipt, create-once (step 5)
    lock         development_screen_used.lock, write-once (step 6)
    adjudicate   ONLY NOW the adjudicator opens sealed_truth.npz, assembles the per-group table
                 (plan 9.1) and runs the ONE 10,000-draw 80% paired bootstrap + gates (steps 7-8)

Any gate failure is the terminal state ``STAGE4D_EXTERNAL_DEV_NOT_READY`` (plan 8.4/15): no
screen rerun, no model change, no confirm. All science/statistics live in
``stage4d_external_statistics`` (torch-free); this driver only orchestrates and runs the frozen
torch stack read-only. A plain-text heartbeat is emitted at least every 30 s during every phase
(the long-running progress contract).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from ac_twin import stage4d_external_inference as INF  # noqa: E402
from ac_twin import stage4d_external_statistics as STATS  # noqa: E402
from ac_twin import stage4d_external_training as T  # noqa: E402
from ac_twin import stage4cp_features as s4cpf  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402

import stage4d_split_screen_truth as SPLIT  # noqa: E402

RUN_DIR = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external"
CONFIG_PATH = RUN_DIR / "config.json"
SCREEN_ROLE = "development_screen"

_SCORE_ARRAY_KEYS = ("j_hat", "ensemble_risk", "risk_by_seed", "mu_only_ensemble",
                     "class_only_ensemble")


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _load_stack(paths: Mapping[str, Path]) -> INF.PredictorStack:
    if not paths["predictor_stack"].is_file():
        raise SystemExit(f"[stage4d-screen] predictor stack missing: {paths['predictor_stack']}")
    return INF.PredictorStack.from_dict(
        json.loads(paths["predictor_stack"].read_text(encoding="utf-8")))


def _load_inventory(paths: Mapping[str, Path]) -> T.ModelInventory:
    """Rebuild the frozen Direct ModelInventory from fit_receipt.json; verify checkpoints."""
    receipt_path = paths["direct_fit_receipt"]
    if not receipt_path.is_file():
        raise SystemExit(f"[stage4d-screen] Direct fit receipt missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for seed_text, digest in receipt["checkpoint_sha256"].items():
        path = Path(receipt["checkpoint_paths"][seed_text])
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"frozen Direct checkpoint drifted: {path}")
    scaler_payload = receipt["scaler"]
    return T.ModelInventory(
        plan_id=str(receipt["plan_id"]), arch=str(receipt["arch"]),
        seeds=tuple(int(s) for s in receipt["seeds"]),
        param_count=int(receipt["param_count"]), run_dir=str(receipt["run_dir"]),
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


# --- scoring bundle from LEGAL INPUTS ONLY ----------------------------------------------------------


def build_scoring_bundle(
    inputs_path: str | Path, *, group_index: np.ndarray | None = None
) -> INF.BranchBundle:
    """Build the branch bundle from ``inference_inputs.npz`` ONLY (never the sealed truth).

    The legal inputs carry no counterfactual future, so a shape-correct placeholder
    (the factual future tiled across the ten actions) fills the ``object_counterfactual_futures``
    slot purely so the frozen ``build_branch_bundle`` expansion runs. The scoring functions
    (``run_pair_predictions`` / ``score_action_risks`` / ``build_direct_inputs``) provably never
    read the cf-derived fields (``counterfactual_xy`` / ``valid_eval`` / ``r_true`` /
    ``regime_labels`` / ``effect_target``); the placeholder is the observed factual trajectory,
    not any truth. ``group_index`` selects a subset of groups (for the determinism recheck).
    """
    inputs_path = Path(inputs_path)
    with np.load(inputs_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    scenario_ids = json.loads(np.asarray(data["scenario_ids_json"]).item())
    groups = int(np.asarray(data["object_history"]).shape[0])
    n_actions = int(np.asarray(data["n_actions"]).item())
    factual = np.asarray(data["object_factual_future"], dtype=np.float32)
    placeholder_cf = np.repeat(factual[:, None], n_actions, axis=1)  # (G,10,20,5,8) placeholder

    arrays = INF.RoleArrays(
        role=SCREEN_ROLE, groups=groups, n_actions=n_actions,
        scenario_ids=tuple(scenario_ids),
        group_seed=np.asarray(data["group_seed"], dtype=np.int64),
        object_history=np.asarray(data["object_history"], dtype=np.float32),
        object_factual_future=factual,
        object_counterfactual_futures=placeholder_cf,  # placeholder; never read by scoring
        factual_actions=np.asarray(data["factual_actions"], dtype=np.int64),
        inference_mask=np.asarray(data["inference_mask"], dtype=bool),
        t_star=np.asarray(data["t_star"], dtype=np.int64),
        cf_actions=np.asarray(data["cf_actions"], dtype=np.int64).ravel(),
        cf_durations=np.asarray(data["cf_durations"], dtype=np.int64).ravel(),
        data_sha256=sha256_file(inputs_path),
    )
    bundle = INF.build_branch_bundle(arrays)
    if group_index is not None:
        bundle = _subset_bundle(bundle, np.asarray(group_index, dtype=np.int64))
    return bundle


def _subset_bundle(bundle: INF.BranchBundle, group_index: np.ndarray) -> INF.BranchBundle:
    """Slice a BranchBundle to a subset of groups (branch axis = group-major action-minor)."""
    g = np.asarray(group_index, dtype=np.int64)
    a = bundle.n_actions
    branch_index = np.concatenate([np.arange(int(i) * a, int(i) * a + a) for i in g])

    def _branch(x):
        return np.asarray(getattr(bundle, x))[branch_index]

    geometry_group = {k: np.asarray(v)[g] for k, v in bundle.geometry_group.items()}
    geometry_branch = {k: np.asarray(v)[branch_index] for k, v in bundle.geometry_branch.items()}
    return INF.BranchBundle(
        role=bundle.role, groups=int(g.size), n_actions=a, branches=int(g.size * a),
        group_seed=np.asarray(bundle.group_seed)[g],
        scenario_ids=tuple(bundle.scenario_ids[int(i)] for i in g),
        history=_branch("history"), factual_suffix=_branch("factual_suffix"),
        factual_actions=_branch("factual_actions"),
        counterfactual_actions=_branch("counterfactual_actions"),
        cf_action=_branch("cf_action"), cf_duration=_branch("cf_duration"),
        t_star=_branch("t_star"),
        valid_eval=_branch("valid_eval"), r_true=_branch("r_true"),
        regime_labels=_branch("regime_labels"), effect_target=_branch("effect_target"),
        inference_mask=np.asarray(bundle.inference_mask)[g],
        valid_infer=_branch("valid_infer"),
        t_star_group=np.asarray(bundle.t_star_group)[g],
        history_last_xy=np.asarray(bundle.history_last_xy)[g],
        factual_xy=np.asarray(bundle.factual_xy)[g],
        counterfactual_xy=np.asarray(bundle.counterfactual_xy)[g],
        geometry_group=geometry_group, geometry_branch=geometry_branch,
        data_sha256=bundle.data_sha256,
    )


def score_screen(
    stack: INF.PredictorStack,
    inventory: T.ModelInventory,
    inputs_path: str | Path,
    *,
    device: str,
    batch_size: int,
    group_index: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run the frozen stack + Direct ensemble over the screen groups -> score dict.

    Reads ONLY ``inference_inputs.npz`` (via ``build_scoring_bundle``); never the sealed truth.
    Refuses (no zero-padding) if any group has non-finite features.
    """
    bundle = build_scoring_bundle(inputs_path, group_index=group_index)
    predictions = INF.run_pair_predictions(stack, bundle, device=device, batch_size=batch_size)
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


def determinism_recheck(
    stack: INF.PredictorStack,
    inventory: T.ModelInventory,
    inputs_path: str | Path,
    *,
    device: str,
    batch_size: int,
    n_groups: int,
) -> dict[str, Any]:
    """Re-score the first ``n_groups`` twice (same device/batch/dtype/order); elementwise identical."""
    subset = np.arange(int(n_groups), dtype=np.int64)
    first = score_screen(stack, inventory, inputs_path, device=device,
                         batch_size=batch_size, group_index=subset)
    second = score_screen(stack, inventory, inputs_path, device=device,
                          batch_size=batch_size, group_index=subset)
    max_abs = 0.0
    keys = list(_SCORE_ARRAY_KEYS) + [f"baseline__{k}" for k in STATS.ALL_BASELINE_NAMES]
    for key in _SCORE_ARRAY_KEYS:
        a = np.asarray(first[key], dtype=np.float64)
        b = np.asarray(second[key], dtype=np.float64)
        if not np.array_equal(a, b):
            raise RuntimeError(
                f"screen scoring is not elementwise reproducible ({key}); the one-shot screen "
                "halts before the score freeze")
        max_abs = max(max_abs, float(np.max(np.abs(a - b))) if a.size else 0.0)
    for name in STATS.ALL_BASELINE_NAMES:
        a = np.asarray(first["baselines"][name], dtype=np.float64)
        b = np.asarray(second["baselines"][name], dtype=np.float64)
        if not np.array_equal(a, b):
            raise RuntimeError(
                f"screen baseline scoring is not elementwise reproducible ({name})")
    return {
        "role": SCREEN_ROLE,
        "n_groups": int(n_groups),
        "group_indices": [int(i) for i in subset],
        "bitwise_identical": True,
        "max_abs_diff": float(max_abs),
        "device": str(device),
        "batch_size": int(batch_size),
    }


# --- heartbeat (terminal-safe progress, >= every 30 s) ----------------------------------------------


class _Phase:
    """Plain-text phase progress: start/end lines + a 15 s watchdog while a phase blocks."""

    def __init__(self, total_phases: int) -> None:
        self.total = int(total_phases)
        self.index = 0
        self.t0 = time.perf_counter()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, name: str, detail: str = "") -> None:
        self.index += 1
        self._phase_t0 = time.perf_counter()
        pct = 100.0 * (self.index - 1) / self.total
        print(f"[stage4d-screen] phase {self.index}/{self.total} ({pct:.0f}%) START {name} "
              f"{detail} elapsed={time.perf_counter()-self.t0:.1f}s", flush=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._watchdog, args=(name,), daemon=True)
        self._thread.start()

    def _watchdog(self, name: str) -> None:
        while not self._stop.wait(15.0):  # 15 s < the 30 s contract
            elapsed = time.perf_counter() - self._phase_t0
            print(f"[stage4d-screen] phase {self.index}/{self.total} RUNNING {name} "
                  f"phase_elapsed={elapsed:.1f}s total_elapsed={time.perf_counter()-self.t0:.1f}s",
                  flush=True)

    def done(self, name: str, detail: str = "") -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        pct = 100.0 * self.index / self.total
        print(f"[stage4d-screen] phase {self.index}/{self.total} ({pct:.0f}%) DONE {name} "
              f"{detail} phase_elapsed={time.perf_counter()-self._phase_t0:.1f}s", flush=True)


# --- the one-shot sequence ----------------------------------------------------------------------------


def run_screen(config: Mapping[str, Any], *, device: str, batch_size: int,
               n_determinism: int | None = None) -> int:
    paths = STATS.screen_artifact_paths(config)
    n_det = int(n_determinism or config.get("determinism_check_groups", 32))
    expect_groups = int(config.get("screen_groups", STATS.DEV_SCREEN_GROUPS))
    phase = _Phase(total_phases=6)

    phase.start("split", f"role={SCREEN_ROLE}")
    SPLIT.split_screen_truth(paths["role_dir"], paths["split_dir"],
                             role_salt=str(config["role_salt"]), expect_groups=expect_groups)
    phase.done("split", f"inputs={paths['inference_inputs'].name} truth=sealed")

    # The scoring path is handed ONLY the inputs path; it is structurally unable to read truth.
    stack = _load_stack(paths)
    inventory = _load_inventory(paths)

    phase.start("score", f"groups={expect_groups} device={device}")
    scores = score_screen(stack, inventory, paths["inference_inputs"],
                          device=device, batch_size=batch_size)
    phase.done("score", f"groups={scores['group_seed'].size}")

    phase.start("determinism", f"recheck_groups={n_det}")
    determinism = determinism_recheck(stack, inventory, paths["inference_inputs"],
                                      device=device, batch_size=batch_size, n_groups=n_det)
    phase.done("determinism", f"bitwise={determinism['bitwise_identical']}")

    phase.start("freeze", "screen_scores.npz create-once")
    STATS.freeze_screen_scores(config, scores=scores, determinism=determinism)
    phase.done("freeze", f"scores={paths['scores_npz'].name}")

    phase.start("lock", "development_screen_used.lock write-once")
    STATS.burn_screen_lock(config)
    phase.done("lock", f"lock={paths['lock'].name}")

    phase.start("adjudicate", "ONE bootstrap + gates (opens sealed truth)")
    decision = STATS.adjudicate_screen(config)
    phase.done("adjudicate", f"status={decision.status}")

    total = time.perf_counter() - phase.t0
    print(f"[stage4d-screen] COMPLETE status={decision.status} passed={decision.passed} "
          f"total_elapsed={total:.1f}s", flush=True)
    print(f"[stage4d-screen] gate_task={decision.task['conditions']}", flush=True)
    print(f"[stage4d-screen] gate_risk={decision.risk['conditions']}", flush=True)
    if decision.passed:
        print(f"[stage4d-screen] {decision.status}: proceed to power (Task 5)", flush=True)
        return 0
    print(f"[stage4d-screen] TERMINAL {decision.status}: no rerun, no model change, no confirm",
          flush=True)
    return 6 if decision.status == STATS.STATUS_INVALID_ANALYSIS else 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4D-D2 unique development-screen driver")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--determinism-groups", type=int, default=None)
    args = parser.parse_args(argv)
    config = _load_config()
    device = str(args.device or config.get("device", "cuda"))
    if device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    return run_screen(config, device=device, batch_size=args.batch_size,
                      n_determinism=args.determinism_groups)


if __name__ == "__main__":
    raise SystemExit(main())
