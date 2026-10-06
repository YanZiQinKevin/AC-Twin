"""D2F plan section 3.4: ONE targeted parity localization (read-only, writes nothing).

The plan's original contract: run the D2F scorer on the old validation's LEGAL INPUTS
(live end-to-end inference) and reproduce the D2E frozen power_validation_quantities.npz
within 1e-8 with the ten-action argmin identical. The existing receipt only proves the
composition chain on persisted predictions (Path A, bitwise 0.0); the live path (Path B)
showed a float envelope. This script localizes, under the SAME model/input/device/dtype/
batch caliber as the original D2 persistence run (device=cuda, batch_size=32), exactly
where the difference arises and whether the final J_hat / five-seed risks / ensemble /
three baselines / argmin change.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ac_twin import stage4d_d2e_pipeline as PIPE  # noqa: E402
from ac_twin import stage4d_d2f_inference as D2FINF  # noqa: E402
from ac_twin import stage4d_d2f_protocol as PROT  # noqa: E402
from ac_twin import stage4d_external_inference as INF  # noqa: E402
from ac_twin import stage4d_external_statistics as D2STAT  # noqa: E402
from ac_twin import stage4d_external_training as T  # noqa: E402

D2_DIR = PROT.D2_DIR
D2E_DIR = PROT.D2E_DIR
DEVICE = "cuda"
PERSIST_BATCH = 32          # D2 persistence caliber (drive_stage4d_external_task3 default)
CHAIN_BATCH = 4096          # D2E downstream caliber used by the frozen Path A receipt
PRED_KEYS = ("e2e", "full", "history", "r_hat", "presence_prob", "high_prob")


def _max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def main() -> int:
    print("[localize] torch backend flags:",
          {"cudnn.benchmark": torch.backends.cudnn.benchmark,
           "cudnn.deterministic": torch.backends.cudnn.deterministic,
           "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
           "matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
           "float32_matmul_precision": torch.get_float32_matmul_precision(),
           "torch": torch.__version__, "device": DEVICE}, flush=True)

    t0 = time.perf_counter()
    stack = D2FINF.load_predictor_stack()
    selected, ego_scaler, ego_models = D2FINF.load_selected_ego()
    inventory = D2FINF.load_risk_inventory(selected)
    bundle = D2FINF.build_bundle_from_role_groups(D2_DIR / "roles" / "validation", role="validation")
    print(f"[localize] stack+bundle loaded ({bundle.groups} groups) in {time.perf_counter()-t0:.1f}s",
          flush=True)

    with np.load(D2_DIR / "development" / "validation_predictions.npz", allow_pickle=False) as z:
        persisted = {k: z[k] for k in z.files}

    # --- layer 1: base predictor forward pass, at the persistence batch caliber ----------------
    t0 = time.perf_counter()
    live32 = INF.run_pair_predictions(stack, bundle, device=DEVICE, batch_size=PERSIST_BATCH)
    print(f"[localize] live inference batch={PERSIST_BATCH} in {time.perf_counter()-t0:.1f}s", flush=True)
    env32 = {k: _max_abs(live32[k], persisted[k]) for k in PRED_KEYS}
    print(f"[localize] ENVELOPE batch={PERSIST_BATCH} (persistence caliber): {env32}", flush=True)

    # --- layer 1b: same process, same caliber, second pass -> same-process determinism ---------
    live32b = INF.run_pair_predictions(stack, bundle, device=DEVICE, batch_size=PERSIST_BATCH)
    det = {k: _max_abs(live32b[k], live32[k]) for k in PRED_KEYS}
    print(f"[localize] same-process repeat batch={PERSIST_BATCH}: {det}", flush=True)

    # --- layer 1c: batch-caliber sensitivity (the receipt's Path B used 4096) -------------------
    live4096 = INF.run_pair_predictions(stack, bundle, device=DEVICE, batch_size=CHAIN_BATCH)
    env4096 = {k: _max_abs(live4096[k], persisted[k]) for k in PRED_KEYS}
    cross = {k: _max_abs(live4096[k], live32[k]) for k in PRED_KEYS}
    print(f"[localize] ENVELOPE batch={CHAIN_BATCH}: {env4096}", flush=True)
    print(f"[localize] batch 32-vs-4096 (same process): {cross}", flush=True)

    # --- layers 2-4: full D2F scorer chain on the batch-32 live predictions ---------------------
    t0 = time.perf_counter()
    predictions = dict(live32)
    predictions["e2e"] = PIPE.candidate_e2e(bundle, predictions["e2e"], ego_scaler, ego_models,
                                            device=DEVICE, batch_size=CHAIN_BATCH)
    action_risks = INF.score_action_risks(stack, bundle, predictions)
    direct = INF.build_direct_inputs(bundle, predictions, action_risks)
    risk = T.score_models(inventory,
                          {"tokens": direct["tokens"], "global_features": direct["global_features"],
                           "group_seed": direct["group_seed"]},
                          device=DEVICE)
    print(f"[localize] downstream chain in {time.perf_counter()-t0:.1f}s", flush=True)

    with np.load(D2E_DIR / "development" / "power_validation_quantities.npz",
                 allow_pickle=False) as z:
        ref = {k: z[k] for k in z.files}
    ref_seed = np.asarray(ref["group_seed"], dtype=np.int64).ravel()
    got_seed = np.asarray(direct["group_seed"], dtype=np.int64).ravel()
    order = np.array([np.where(ref_seed == s)[0][0] for s in got_seed], dtype=np.int64)

    j_got = np.asarray(direct["j_hat"], dtype=np.float64)
    j_ref = np.asarray(ref["J_hat"], dtype=np.float64)[order]
    final = {
        "J_hat_max_abs_diff": _max_abs(j_got, j_ref),
        "argmin_identical": bool(np.array_equal(j_got.argmin(axis=1), j_ref.argmin(axis=1))),
        "n_argmin_mismatch": int((j_got.argmin(axis=1) != j_ref.argmin(axis=1)).sum()),
        "ensemble_risk_max_abs_diff": _max_abs(np.asarray(risk.risk_ensemble).ravel(),
                                               np.asarray(ref["ensemble_risk"])[order]),
        "risk_by_seed_max_abs_diff": _max_abs(np.asarray(risk.risk_by_seed),
                                              np.asarray(ref["risk_by_seed"])[:, order]),
    }
    for name in D2STAT.STANDARD_BASELINES:
        final[f"baseline_{name}_max_abs_diff"] = _max_abs(
            np.asarray(direct["baselines"][name]).ravel(),
            np.asarray(ref[f"baseline_risk__{name}"])[order])
    tol = D2FINF.PARITY_TOLERANCE
    final["passes_1e-8_end_to_end"] = bool(
        final["argmin_identical"]
        and all(v <= tol for k, v in final.items()
                if k.endswith("max_abs_diff") and isinstance(v, float)))
    print(f"[localize] FINAL end-to-end vs D2E frozen npz (tolerance {tol}): {final}", flush=True)
    print("[localize] DONE (read-only; nothing written)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
