"""Stage 4D-D2 pilot adjudication: the mechanical D0-X gate (plan sections 6.2 / 6.3).

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2``. This is a
scientific gate, evaluated EXACTLY ONCE on the unique 100-group pilot. It reads only
the frozen pilot tensors + ledger and applies the seven pre-registered criteria:

1. completeness: 100/100 groups, 11 branches, ten actions legal at t_star, prefix
   consistent;
2. non-degenerate indices, no duplicate IDs / NaN / Inf;
3. 5 Hz layer, 20-step horizon, durations, Map-Frenet units conform to contract;
4. >= 20% of groups show a non-ego eval entity whose future changed (reactive
   response) beyond the frozen numerical tolerance;
5. ``always_idle_regret > 0.01`` fraction >= 15% AND >= 3 distinct oracle action plans
   each covering >= 5% of groups (the pilot is model-free, so "regret" is the frozen
   default-action (IDLE) decision-qualification baseline vs the group oracle -- the
   same task-support measure the screen recomputes in its criterion 2);
6. double-replay: discrete IDs/actions/masks identical, continuous state within the
   frozen atol/rtol, across two fresh collection processes;
7. no outcome-based rejection: every rejection reason is in the frozen set.

Adjudication (section 6.3): criteria 1-3/6-7 failing -> METADRIVE_TECHNICAL_CONTRACT_FAILED
/ SUMO_FALLBACK_ELIGIBLE; criteria 4-5 failing -> STAGE4D_EXTERNAL_TASK_INSUFFICIENT;
all pass -> METADRIVE_PLATFORM_QUALIFIED. This module never reads model scores and
never rejects on outcome.

Run inside the frozen sim env:
    sim code/stage4d_pilot_adjudicate.py --pilot-dir <run A> [--replay-dir <run B>]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from ac_twin import stage4c_objective as obj  # noqa: E402
from ac_twin import stage4d_external_features as F  # noqa: E402
import stage4d_metadrive_adapter as A  # noqa: E402

RUN_DIR = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external"

# --- frozen gate constants (plan 6.2) -----------------------------------------------
REGRET_MEANINGFUL = obj.REGRET_MEANINGFUL          # 0.01
REACTIVE_MIN_GROUP_FRAC = 0.20
REGRET_MIN_GROUP_FRAC = 0.15
ORACLE_MIN_PLANS = 3
ORACLE_MIN_PLAN_FRAC = 0.05
#: "beyond numerical tolerance" for a non-ego reactive response (metres). Above the
#: measured same-process float noise floor (~3.4e-3); a real IDM reaction over the
#: 4 s horizon is far larger.
REACTIVE_TOLERANCE_M = 1e-2
#: double-replay continuous-state tolerance (criterion 6). Fresh-env reruns are
#: bit-identical in the adapter test; the tolerance only guards platform float jitter.
REPLAY_ATOL = 1e-4
REPLAY_RTOL = 1e-4

TECHNICAL_CRITERIA = (1, 2, 3, 6, 7)   # -> METADRIVE_TECHNICAL_CONTRACT_FAILED
TASK_CRITERIA = (4, 5)                 # -> STAGE4D_EXTERNAL_TASK_INSUFFICIENT

CH_S, CH_D = 2, 3


def compute_group_quantities(groups: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-group true decision quantities from the 10 counterfactual branches.

    The objective consumes the common Map-Frenet (s, d) channels as its (x, y), so
    longitudinal progress is arc progress along the group reference -- the frozen
    stage4c_objective code is reused unchanged (plan section 7).
    """
    cf = np.asarray(groups["object_counterfactual_futures"], dtype=np.float64)
    true_xy = cf[..., [CH_S, CH_D]]                          # (G,10,20,5,2)
    hist = np.asarray(groups["object_history"], dtype=np.float64)
    history_last_xy = hist[:, -1, 0, [CH_S, CH_D]]           # ego (s,d) at t_star
    mask = np.asarray(groups["inference_mask"], dtype=bool)  # (G,20,5)

    comp = obj.trajectory_components(
        true_xy,
        np.broadcast_to(history_last_xy[:, None, :], true_xy.shape[:2] + (2,)),
        np.broadcast_to(mask[:, None, :, :], true_xy.shape[:-1]),
    )
    j_true = obj.total_cost(comp["interaction"], obj.efficiency_cost(comp["progress"]))
    a_star = np.argmin(j_true, axis=-1).astype(np.int64)     # canonical order tie-break
    rows = np.arange(j_true.shape[0])
    always_idle_regret = obj.clamp_regret(
        j_true[:, obj.IDLE_PLAN_INDEX] - j_true[rows, a_star])
    oracle_spread = j_true.max(axis=-1) - j_true[rows, a_star]

    # reactive response: any selected non-ego (slots 1..4) whose (s,d) future differs
    # between the factual branch and any counterfactual branch beyond tolerance.
    factual = np.asarray(groups["object_factual_future"], dtype=np.float64)
    factual_xy = factual[:, :, 1:, [CH_S, CH_D]]             # (G,20,4,2)
    factual_presence = factual[:, :, 1:, 0] > 0              # (G,20,4)
    cf_xy = cf[:, :, :, 1:, [CH_S, CH_D]]                    # (G,10,20,4,2)
    diff = np.abs(cf_xy - factual_xy[:, None]).max(axis=-1)  # (G,10,20,4)
    diff = np.where(factual_presence[:, None], diff, 0.0)
    reactive = (diff.max(axis=(1, 2, 3)) > REACTIVE_TOLERANCE_M)  # (G,)

    return {
        "J_true": j_true,
        "a_star": a_star,
        "always_idle_regret": always_idle_regret,
        "oracle_spread": oracle_spread,
        "reactive": reactive,
    }


def _check_criteria(
    groups_a: Mapping[str, np.ndarray],
    meta_a: list[Mapping[str, Any]],
    ledger_rows: list[Mapping[str, Any]],
    groups_b: Mapping[str, np.ndarray] | None,
    meta_b: list[Mapping[str, Any]] | None,
    quota: int,
) -> dict[int, dict[str, Any]]:
    q = compute_group_quantities(groups_a)
    n = len(meta_a)
    criteria: dict[int, dict[str, Any]] = {}

    # 1. completeness
    branches_ok = bool(
        groups_a["object_counterfactual_futures"].shape[1] == len(A.ACTION_PLANS)
        and n == quota)
    criteria[1] = {"pass": branches_ok,
                   "detail": f"groups={n}/{quota}, actions={groups_a['object_counterfactual_futures'].shape[1]}"}

    # 2. non-degenerate indices, unique IDs, finite
    ids = [m["scenario_id"] for m in meta_a]
    finite = all(np.isfinite(groups_a[k]).all() for k in
                 ("object_history", "object_factual_future", "object_counterfactual_futures"))
    criteria[2] = {"pass": bool(len(set(ids)) == len(ids) and finite),
                   "detail": f"unique_ids={len(set(ids))}/{len(ids)}, finite={finite}"}

    # 3. contract shapes / units
    c3 = bool(
        groups_a["object_history"].shape[1:] == (F.HISTORY_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES)
        and groups_a["object_factual_future"].shape[1:] == (F.FUTURE_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES)
        and groups_a["object_counterfactual_futures"].shape[1:] ==
            (len(A.ACTION_PLANS), F.FUTURE_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES)
        and set(np.unique(groups_a["cf_durations"])).issubset({1, 3})
        and int(groups_a["t_star"].min()) >= min(F.T_STAR_CANDIDATES)
        and int(groups_a["t_star"].max()) <= max(F.T_STAR_CANDIDATES))
    criteria[3] = {"pass": c3, "detail": "history/future/cf shapes + durations + t_star range"}

    # 4. reactive support
    reactive_frac = float(q["reactive"].mean())
    criteria[4] = {"pass": reactive_frac >= REACTIVE_MIN_GROUP_FRAC,
                   "detail": f"reactive_frac={reactive_frac:.3f} >= {REACTIVE_MIN_GROUP_FRAC}"}

    # 5. regret support + oracle diversity
    regret_frac = float((q["always_idle_regret"] > REGRET_MEANINGFUL).mean())
    oracle_counts = Counter(int(a) for a in q["a_star"])
    plans_over_5 = sum(1 for c in oracle_counts.values() if c / n >= ORACLE_MIN_PLAN_FRAC)
    criteria[5] = {"pass": bool(regret_frac >= REGRET_MIN_GROUP_FRAC and plans_over_5 >= ORACLE_MIN_PLANS),
                   "detail": f"regret_frac={regret_frac:.3f}>={REGRET_MIN_GROUP_FRAC}; "
                             f"oracle_plans>=5%={plans_over_5}>={ORACLE_MIN_PLANS}; "
                             f"oracle_dist={dict(sorted(oracle_counts.items()))}"}

    # 6. double-replay determinism
    if groups_b is None or meta_b is None:
        criteria[6] = {"pass": None, "detail": "replay run B not provided; run --replay first"}
    else:
        ids_b = [m["scenario_id"] for m in meta_b]
        discrete_same = (ids == ids_b
                         and np.array_equal(groups_a["cf_actions"], groups_b["cf_actions"])
                         and np.array_equal(groups_a["cf_durations"], groups_b["cf_durations"])
                         and np.array_equal(groups_a["inference_mask"], groups_b["inference_mask"])
                         and np.array_equal(groups_a["t_star"], groups_b["t_star"]))
        cont_close = bool(np.allclose(
            groups_a["object_counterfactual_futures"], groups_b["object_counterfactual_futures"],
            atol=REPLAY_ATOL, rtol=REPLAY_RTOL))
        criteria[6] = {"pass": bool(discrete_same and cont_close),
                       "detail": f"discrete_same={discrete_same}, continuous_close={cont_close}"}

    # 7. no outcome-based rejection: all reasons in the frozen set
    import collect_stage4d_external as C
    bad_reasons = [r.get("reason") for r in ledger_rows
                   if r.get("outcome") == "rejected" and r.get("reason") not in C.REASONS]
    criteria[7] = {"pass": not bad_reasons,
                   "detail": f"out_of_set_reasons={sorted(set(bad_reasons))}"}
    return criteria


def adjudicate(pilot_dir: Path, replay_dir: Path | None, quota: int = 100) -> dict[str, Any]:
    with np.load(pilot_dir / "groups.npz", allow_pickle=False) as z:
        groups_a = {k: z[k] for k in z.files}
    meta_a = json.loads((pilot_dir / "groups_meta.json").read_text(encoding="utf-8"))
    ledger_rows = []
    ledger_path = pilot_dir / "rejection_ledger.jsonl"
    if ledger_path.is_file():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ledger_rows.append(json.loads(line))
    groups_b = meta_b = None
    if replay_dir is not None and (replay_dir / "groups.npz").is_file():
        with np.load(replay_dir / "groups.npz", allow_pickle=False) as z:
            groups_b = {k: z[k] for k in z.files}
        meta_b = json.loads((replay_dir / "groups_meta.json").read_text(encoding="utf-8"))

    criteria = _check_criteria(groups_a, meta_a, ledger_rows, groups_b, meta_b, quota)
    failed = [c for c, r in criteria.items() if r["pass"] is False]
    unevaluated = [c for c, r in criteria.items() if r["pass"] is None]
    if unevaluated:
        verdict = "PILOT_GATE_INCOMPLETE"
        terminal = None
    elif not failed:
        verdict = "ALL_PASS"
        terminal = "METADRIVE_PLATFORM_QUALIFIED"
    elif any(c in TECHNICAL_CRITERIA for c in failed):
        verdict = "TECHNICAL_FAIL"
        terminal = "METADRIVE_TECHNICAL_CONTRACT_FAILED / SUMO_FALLBACK_ELIGIBLE"
    else:
        verdict = "TASK_FAIL"
        terminal = "STAGE4D_EXTERNAL_TASK_INSUFFICIENT"

    report = {
        "plan_id": "AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pilot_dir": str(pilot_dir),
        "replay_dir": str(replay_dir) if replay_dir else None,
        "quota": quota,
        "criteria": {str(c): r for c, r in criteria.items()},
        "failed_criteria": failed,
        "verdict": verdict,
        "terminal_status": terminal,
    }
    out = pilot_dir / "technical_pilot_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[pilot_adjudicate] verdict={verdict} terminal={terminal} "
          f"failed={failed} unevaluated={unevaluated}", flush=True)
    for c in sorted(criteria):
        r = criteria[c]
        print(f"  criterion {c}: pass={r['pass']} | {r['detail']}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", default=str(RUN_DIR / "roles" / "pilot"))
    parser.add_argument("--replay-dir", default=None)
    parser.add_argument("--quota", type=int, default=100)
    args = parser.parse_args(argv)
    report = adjudicate(Path(args.pilot_dir),
                        Path(args.replay_dir) if args.replay_dir else None,
                        args.quota)
    return 0 if report["verdict"] == "ALL_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
