"""Stage 4D-D2 action-contract freeze (post-smoke, tracking correctness only).

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` sections 5 and 13.
After the 12-group engineering smoke (``ENGINEERING_SMOKE_ONLY``) this measures
*tracking correctness* of the frozen meta-action controller on the smoke groups and
freezes ``action_contract.json`` (create-once). It reads ONLY ego kinematics
(s, d, v_s) and the known per-branch override intents -- never regret, oracle ranking,
collision outcome, or any consequence value, so the contract cannot be outcome-tuned.

Run inside the frozen sim env:

    powershell.exe -NoProfile -ExecutionPolicy Bypass -File code\\run_stage4d_env.ps1 `
        sim code/stage4d_action_contract.py --smoke-dir <smoke role dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402
import stage4d_metadrive_adapter as A  # noqa: E402

RUN_DIR = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external"
PLATFORM_DIR = RUN_DIR / "platform"
CONTRACT_PATH = PLATFORM_DIR / "action_contract.json"

# feature channel indices (frozen schema, see stage4d_external_features.FEATURE_NAMES)
CH_S, CH_D, CH_VS = 2, 3, 4
# intent indices
LANE_LEFT, LANE_RIGHT, FASTER, SLOWER = 0, 2, 3, 4

# frozen tracking-correctness success definitions (engineering; not outcome-derived)
LANE_CHANGE_MIN_DISPLACEMENT_M = 1.5   # |Delta d| that counts as a real lane change
SPEED_RESPONSE_MIN_MS = 1.0            # peak |Delta v_s| that counts as a speed response
# feasibility gates (structural, from the pre-outcome anchor v_star and the nominal
# factual branch only): a maneuver can only be certified where it is physically possible.
LC_MIN_PROGRESS_M = 2.0        # ego must advance >= this in the factual branch to lane-change
FASTER_FEAS_MIN_RISE = 0.5     # factual ego peak speed rise (m/s) proving room to accelerate
SLOWER_FEAS_MIN_VSTAR = 2.0    # ego must be moving (m/s) to be able to slow down
FEASIBLE_SUCCESS_MIN_FRAC = 0.5  # controller must succeed on >= this fraction of feasible groups
MIN_FEASIBLE_GROUPS = 1        # need at least this many feasible groups to certify a maneuver


def _measure(groups: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Tracking-correctness statistics from the smoke tensors (ego slot 0).

    The meta-action can only be *tracked* where it is physically feasible: a stopped
    ego cannot SLOWER, and a boxed-in ego cannot FASTER or change lane. Feasibility is
    read ONLY from the pre-outcome anchor (``v_star`` = ego speed at the first future
    frame) and the nominal factual branch (its longitudinal progress / speed change),
    never from counterfactual outcomes. The certification gate is the success fraction
    on each maneuver's feasible subset.
    """
    cf = groups["object_counterfactual_futures"]  # (G,10,20,5,8)
    cf_actions = groups["cf_actions"]            # (10,)
    factual = groups["object_factual_future"]    # (G,20,5,8)
    n_groups = cf.shape[0]

    v_star = factual[:, 0, 0, CH_VS]                            # (G,) anchor speed
    factual_progress = factual[:, -1, 0, CH_S] - factual[:, 0, 0, CH_S]
    factual_peak_rise = factual[:, :, 0, CH_VS].max(axis=-1) - v_star  # room to accelerate
    feas = {
        "lane_change": factual_progress >= LC_MIN_PROGRESS_M,
        "faster": factual_peak_rise >= FASTER_FEAS_MIN_RISE,
        "slower": v_star >= SLOWER_FEAS_MIN_VSTAR,
    }

    # per-branch ego displacement / peak speed deviation
    disp = cf[:, :, -1, 0, CH_D] - cf[:, :, 0, 0, CH_D]         # (G,10) +d = left
    peak_rise = cf[:, :, :, 0, CH_VS].max(axis=-1) - cf[:, :, 0, 0, CH_VS]
    peak_drop = cf[:, :, 0, 0, CH_VS] - cf[:, :, :, 0, CH_VS].min(axis=-1)

    def _gate(success_mask_per_branch, feas_mask):
        """success_mask_per_branch: (G,) OR-reduced over that intent's branches."""
        n_feas = int(feas_mask.sum())
        if n_feas < MIN_FEASIBLE_GROUPS:
            return {"feasible_n": n_feas, "success_n": 0, "success_frac": None,
                    "certified": False}
        n_succ = int((success_mask_per_branch & feas_mask).sum())
        frac = n_succ / n_feas
        return {"feasible_n": n_feas, "success_n": n_succ, "success_frac": float(frac),
                "certified": bool(frac >= FEASIBLE_SUCCESS_MIN_FRAC)}

    def _intent_mask(intent):
        return np.array([int(a) == intent for a in cf_actions])  # (10,)

    def _or_over_intent(metric_per_branch, intent, comparator):
        m = _intent_mask(intent)                                  # (10,)
        sel = comparator(metric_per_branch[:, m])                 # (G, n_intent) bool
        return sel.any(axis=-1)                                   # (G,)

    left_ok = _or_over_intent(disp, LANE_LEFT, lambda x: x >= LANE_CHANGE_MIN_DISPLACEMENT_M)
    right_ok = _or_over_intent(disp, LANE_RIGHT, lambda x: x <= -LANE_CHANGE_MIN_DISPLACEMENT_M)
    faster_ok = _or_over_intent(peak_rise, FASTER, lambda x: x >= SPEED_RESPONSE_MIN_MS)
    slower_ok = _or_over_intent(peak_drop, SLOWER, lambda x: x >= SPEED_RESPONSE_MIN_MS)

    # lane change is certified if either direction tracks on feasible groups
    lc_feas = feas["lane_change"]
    lc_n = int(lc_feas.sum())
    lc_succ = int(((left_ok | right_ok) & lc_feas).sum())
    lane_change = {"feasible_n": lc_n, "success_n": lc_succ,
                   "success_frac": (lc_succ / lc_n) if lc_n >= MIN_FEASIBLE_GROUPS else None,
                   "certified": bool(lc_n >= MIN_FEASIBLE_GROUPS
                                     and (lc_succ / lc_n) >= FEASIBLE_SUCCESS_MIN_FRAC)}

    result = {
        "n_groups": int(n_groups),
        "lane_change": lane_change,
        "lane_change_left": _gate(left_ok, lc_feas),
        "lane_change_right": _gate(right_ok, lc_feas),
        "faster": _gate(faster_ok, feas["faster"]),
        "slower": _gate(slower_ok, feas["slower"]),
        "factual_ego_progress_median_m": float(np.median(factual_progress)),
        "v_star_median_ms": float(np.median(v_star)),
        "thresholds": {
            "lane_change_min_displacement_m": LANE_CHANGE_MIN_DISPLACEMENT_M,
            "speed_response_min_ms": SPEED_RESPONSE_MIN_MS,
            "lc_min_progress_m": LC_MIN_PROGRESS_M,
            "faster_feas_min_rise": FASTER_FEAS_MIN_RISE,
            "slower_feas_min_vstar": SLOWER_FEAS_MIN_VSTAR,
            "feasible_success_min_frac": FEASIBLE_SUCCESS_MIN_FRAC,
        },
    }
    result["tracking_ok"] = bool(
        lane_change["certified"] and result["faster"]["certified"]
        and result["slower"]["certified"])
    return result


def freeze_contract(smoke_dir: Path) -> dict[str, Any]:
    """Measure tracking on the smoke output and freeze action_contract.json once."""
    groups_path = smoke_dir / "groups.npz"
    if not groups_path.is_file():
        raise FileNotFoundError(f"smoke groups not found: {groups_path}")
    if CONTRACT_PATH.is_file():
        raise FileExistsError(f"action contract already frozen: {CONTRACT_PATH}")
    with np.load(groups_path, allow_pickle=False) as z:
        groups = {k: z[k] for k in z.files}
    stats = _measure(groups)
    tracking_ok = stats["tracking_ok"]

    receipt = {
        "plan_id": "AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": "freeze meta-action controller contract from smoke tracking correctness",
        "smoke_dir": str(smoke_dir),
        "smoke_groups_sha256": sha256_file(groups_path),
        "controller": dict(A.DEFAULT_CONTRACT.__dict__),
        "tracking_stats": stats,
        "tracking_ok": tracking_ok,
        "code_sha256": sha256_file(Path(__file__)),
        "status": "ACTION_CONTRACT_FROZEN" if tracking_ok else "ACTION_CONTRACT_TRACKING_FAILED",
    }
    write_once_json(CONTRACT_PATH, receipt)
    lc, fa, sl = stats["lane_change"], stats["faster"], stats["slower"]

    def _fmt(m):
        return "n/a" if m["success_frac"] is None else f"{m['success_n']}/{m['feasible_n']}={m['success_frac']:.2f}"

    print(f"[action_contract] {'FROZEN' if tracking_ok else 'TRACKING_FAILED'} | "
          f"LC={_fmt(lc)} FASTER={_fmt(fa)} SLOWER={_fmt(sl)} | "
          f"v_star_median={stats['v_star_median_ms']:.2f}m/s "
          f"progress_median={stats['factual_ego_progress_median_m']:.1f}m", flush=True)
    return receipt


def measure_only(smoke_dir: Path) -> dict[str, Any]:
    """Measure tracking correctness WITHOUT freezing (pre-freeze gain iteration).

    Engineering aid: the contract freeze is create-once, so controller gains are
    validated here first; ``freeze_contract`` is only run once this reports the
    frozen thresholds met.
    """
    groups_path = smoke_dir / "groups.npz"
    if not groups_path.is_file():
        raise FileNotFoundError(f"smoke groups not found: {groups_path}")
    with np.load(groups_path, allow_pickle=False) as z:
        groups = {k: z[k] for k in z.files}
    stats = _measure(groups)
    lc = stats["lane_change"]
    fa = stats["faster"]
    sl = stats["slower"]

    def _fmt(m):
        return "n/a" if m["success_frac"] is None else f"{m['success_n']}/{m['feasible_n']}={m['success_frac']:.2f}"

    print(f"[action_contract:measure] tracking_ok={stats['tracking_ok']} | "
          f"LC={_fmt(lc)} (L={_fmt(stats['lane_change_left'])} R={_fmt(stats['lane_change_right'])}) | "
          f"FASTER={_fmt(fa)} SLOWER={_fmt(sl)} | "
          f"v_star_median={stats['v_star_median_ms']:.2f}m/s "
          f"progress_median={stats['factual_ego_progress_median_m']:.1f}m", flush=True)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", default=str(RUN_DIR / "roles" / "smoke"))
    parser.add_argument("--measure-only", action="store_true",
                        help="measure tracking correctness without freezing (pre-freeze)")
    args = parser.parse_args(argv)
    if args.measure_only:
        return 0 if measure_only(Path(args.smoke_dir))["tracking_ok"] else 1
    receipt = freeze_contract(Path(args.smoke_dir))
    return 0 if receipt["tracking_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
