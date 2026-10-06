"""Render non-adjudicating Stage 4D-D2F paper images from frozen artifacts.

Usage (frozen simulation environment):
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File code/run_stage4d_env.ps1 sim code/render_stage4d_d2f_paper_figures.py

This script never fits a model, samples a scientific episode, or changes a Gate.
It reads the already-adjudicated confirmation table only for qualitative-case
selection and reads the already-frozen pilot pool only for platform legibility.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import pickle
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage4d_external_features as F  # noqa: E402
from ac_twin.stage4d_paper_visualization import (  # noqa: E402
    frenet_to_world,
    peak_step,
    platform_legibility,
    select_cases,
    world_to_ego_plot,
)

PILOT_DIR = ROOT / "experiments/stage4_reliability/stage4d_external/roles/pilot"
POOL_PATH = ROOT / "experiments/stage4_reliability/stage4d_external/platform/viz_platform_pool.json"
D2F_DIR = ROOT / "experiments/stage4_reliability/stage4d_d2f"
OUT_DIR = D2F_DIR / "paper_visuals"
DB_DIR = Path(r"D:\MetaDrive\scenarionet_waymo_training_20s")
ROLE_SALT = "AC-TWIN-D2-V0.1|"
BASELINE_KEY = "baseline_risk__SEED_MARGIN_STD"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _heartbeat(phase: str, stop: threading.Event, start: float) -> None:
    while not stop.wait(25):
        print(f"[paper-viz] {phase}: still running, elapsed={time.monotonic()-start:.0f}s", flush=True)


def with_heartbeat(phase: str, fn):
    start, stop = time.monotonic(), threading.Event()
    print(f"[paper-viz] {phase}: started", flush=True)
    worker = threading.Thread(target=_heartbeat, args=(phase, stop, start), daemon=True)
    worker.start()
    try:
        result = fn()
        print(f"[paper-viz] {phase}: complete in {time.monotonic()-start:.1f}s", flush=True)
        return result
    finally:
        stop.set()
        worker.join(timeout=1)


def _scenario_lookup() -> dict[str, tuple[int, str]]:
    with (DB_DIR / "dataset_summary.pkl").open("rb") as stream:
        summary = pickle.load(stream)
    result = {}
    for index, file_name in enumerate(summary):
        sid = str(summary[file_name].get("scenario_id", file_name.replace("sd_waymo_v1.2_", "").replace(".pkl", "")))
        result[sid] = (index, file_name)
    return result


def load_scenario(sid: str, lookup: dict[str, tuple[int, str]]):
    from metadrive.scenario.utils import read_scenario_data

    index, file_name = lookup[sid]
    return index, read_scenario_data(str(DB_DIR / file_name), centralize=True)


def local_to_world(frame, features: np.ndarray) -> np.ndarray:
    return frenet_to_world(frame.reference, features[..., 2], features[..., 3], origin_s=frame.origin_s)


def to_plot(frame, xy: np.ndarray) -> np.ndarray:
    return world_to_ego_plot(xy, frame.reference, origin_s=frame.origin_s)


def _all_visible_xy(*arrays: np.ndarray) -> np.ndarray:
    points = []
    for features in arrays:
        mask = features[..., 0] > .5
        points.append(features[..., 2:4][mask])
    return np.concatenate(points, axis=0)


def _limits(frame, *feature_arrays: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    pts = []
    for features in feature_arrays:
        xy = to_plot(frame, local_to_world(frame, features))
        pts.append(xy[features[..., 0] > .5])
    all_xy = np.concatenate(pts, axis=0)
    low, high = np.min(all_xy, axis=0) - 12, np.max(all_xy, axis=0) + 12
    # Preserve geometric scale and provide adequate visual breathing room.
    return (float(low[0]), float(high[0])), (float(low[1]), float(high[1]))


def draw_map(ax, scenario, frame, xlim, ylim):
    ax.set_facecolor("#f8fafc")
    features = scenario.get("map_features", {})
    drawn = 0
    for item in features.values():
        if not isinstance(item, dict):
            continue
        poly = np.asarray(item.get("polyline", []), dtype=np.float64)
        if poly.ndim != 2 or len(poly) < 2 or poly.shape[1] < 2:
            continue
        poly = to_plot(frame, poly[:, :2])
        if (poly[:, 0].max() < xlim[0] or poly[:, 0].min() > xlim[1]
                or poly[:, 1].max() < ylim[0] or poly[:, 1].min() > ylim[1]):
            continue
        typ = str(item.get("type", ""))
        color = "#94a3b8" if "LANE" in typ else "#cbd5e1"
        lw = 1.0 if "LANE" in typ else .65
        ax.plot(poly[:, 0], poly[:, 1], color=color, lw=lw, alpha=.7, zorder=1)
        drawn += 1
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_xlabel("Forward (m; ego-heading frame)", fontsize=9)
    ax.set_ylabel("Left (m)", fontsize=9)
    ax.tick_params(labelsize=8)
    return drawn


def draw_tracks(ax, frame, history, future, *, ego_color, mark_peak=None):
    past = to_plot(frame, local_to_world(frame, history))
    fut = to_plot(frame, local_to_world(frame, future))
    for slot in range(5):
        pmask = history[:, slot, 0] > .5
        fmask = future[:, slot, 0] > .5
        if pmask.any():
            ax.plot(past[pmask, slot, 0], past[pmask, slot, 1], color="#64748b",
                    lw=2.0 if slot == 0 else 1.0, alpha=.85, zorder=4)
        if fmask.any():
            color = ego_color if slot == 0 else "#38bdf8"
            ax.plot(fut[fmask, slot, 0], fut[fmask, slot, 1], color=color,
                    lw=3.0 if slot == 0 else 1.1, alpha=1 if slot == 0 else .65, zorder=5)
            ax.scatter(fut[fmask, slot, 0][-1], fut[fmask, slot, 1][-1],
                       s=27 if slot == 0 else 10, color=color, zorder=6)
        if pmask.any():
            label = "EGO" if slot == 0 else f"V{slot}"
            ax.scatter(past[pmask, slot, 0][-1], past[pmask, slot, 1][-1],
                       s=62 if slot == 0 else 28, color="#0f172a" if slot == 0 else "#0284c7", zorder=7)
            ax.annotate(label, past[pmask, slot][-1], xytext=(4, 4),
                        textcoords="offset points", fontsize=7, weight="bold", zorder=8)
    if mark_peak is not None and future[mark_peak, 0, 0] > .5:
        ax.scatter(*fut[mark_peak, 0], marker="*", s=185, facecolor="#facc15",
                   edgecolor="#0f172a", linewidth=.8, zorder=9)


def _validate_terminal() -> dict:
    report_path = D2F_DIR / "confirm/confirm_report.json"
    lock_path = D2F_DIR / "confirm/confirmation_used.lock"
    if not lock_path.is_file():
        raise RuntimeError("D2F confirmation lock absent; qualitative confirmation case access forbidden")
    report = read_json(report_path)
    if report.get("status") != "D2F_META_DRIVE_EXTERNAL_CONFIRMED":
        raise RuntimeError(f"D2F terminal status not confirmed: {report.get('status')}")
    return report


def _select_platform(pilot, meta, pool):
    index_by_id = {row["scenario_id"]: i for i, row in enumerate(meta)}
    rows = []
    for candidate in pool["pool"]:
        i = index_by_id[candidate["scenario_id"]]
        if str(meta[i]["t_star"]) != candidate["context_id"]:
            raise RuntimeError("frozen pilot context mismatch")
        values = platform_legibility(pilot["object_history"][i])
        rows.append({**candidate, **values, "pilot_index": i})
    rows.sort(key=lambda row: (-row["score"], row["hash"]))
    return rows[0], rows


def _render_3d_platform(sid: str, pilot_meta: dict, scenario, scenario_index: int, dest: Path) -> dict:
    """Replay the same pilot prefix in MetaDrive's own Panda3D renderer."""
    import stage4d_metadrive_adapter as A
    from metadrive.constants import DEFAULT_AGENT

    cls = A.make_env_class()
    # Dynamics are copied from the frozen adapter; only use_render/window/UI
    # differ. This replay is for an image and is never a scientific observation.
    cfg = dict(
        data_directory=str(DB_DIR), start_scenario_index=0,
        num_scenarios=len(_scenario_lookup()), sequential_seed=False,
        agent_policy=cls.ACTWIN_EGO_POLICY, reactive_traffic=True,
        filter_overlapping_car=False, no_static_vehicles=False, no_light=False,
        store_map=True, store_data=True, use_render=True,
        window_size=(1600, 900), show_interface=False, manual_control=False,
        horizon=None, decision_repeat=5,
        physics_world_step_size=A.NATIVE_DT / 5.0,
        show_policy_mark=False, show_fps=False, out_of_route_done=False,
        crash_vehicle_done=False, crash_object_done=False,
        crash_human_done=False, relax_out_of_road_done=True,
        max_lateral_dist=1e6, truncate_as_terminate=False,
    )
    ctx = A.prepare_group_context(scenario, sid, ROLE_SALT)
    if ctx.t_star != int(pilot_meta["t_star"]) or ctx.eval_track_ids != pilot_meta["eval_track_ids"]:
        raise RuntimeError("pilot replay context does not match frozen collection")
    env = cls(cfg)
    try:
        env.reset(seed=int(scenario_index))
        ego_obj_id = env.engine.agent_manager.agent_to_object(DEFAULT_AGENT)
        policy = env.engine.get_policy(ego_obj_id)
        policy.set_group_context(dict(
            t_star=ctx.t_star,
            sdc_track=env.engine.data_manager.current_scenario["tracks"][str(
                env.engine.data_manager.current_scenario["metadata"]["sdc_id"])],
            ego_lane_id=ctx.ego_lane_id, adjacency=ctx.adjacency,
            target_lanes=ctx.target_lanes, plan=np.asarray(ctx.factual_plan, dtype=np.int64),
            contract=ctx.contract,
        ))
        for _ in range(2 * ctx.t_star):
            env.step([0.0, 0.0])
        env.render()
        # Panda3D's PNG writer on this Windows build cannot handle the Chinese
        # characters in the project path. Capture to an ASCII-only temporary
        # directory on the same drive, then copy the genuine image into the
        # paper output directory. This changes no pixels.
        with tempfile.TemporaryDirectory(prefix="ac_twin_viz_", dir=str(DB_DIR.parent)) as temp_dir:
            ascii_capture = Path(temp_dir) / "metadrive_capture.png"
            from panda3d.core import Filename, PNMImage
            shot = PNMImage()
            ok = bool(env.engine.win.getScreenshot(shot))
            print(f"[paper-viz] Panda capture ok={ok} size={shot.getXSize()}x{shot.getYSize()}", flush=True)
            if ok:
                written = bool(shot.write(Filename.from_os_specific(str(ascii_capture))))
                print(f"[paper-viz] Panda PNG write={written}", flush=True)
            if not ascii_capture.is_file():
                raise RuntimeError("MetaDrive/Panda3D capture did not create a PNG")
            shutil.copyfile(ascii_capture, dest)
    finally:
        env.close()
    if not dest.is_file() or dest.stat().st_size < 20_000:
        raise RuntimeError("MetaDrive renderer did not produce a substantive PNG")
    with Image.open(dest) as image:
        shape = list(image.size)
    return {"capture_size": shape, "camera": "MetaDrive main camera at t_star", "render_mode": "onscreen"}


def _save_platform_figure(raw_3d: Path, scenario, frame, history, factual, intervention, dest_png, dest_pdf):
    with Image.open(raw_3d) as image:
        rgb = np.asarray(image.convert("RGB"))
    xlim, ylim = _limits(frame, history, factual, intervention)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8),
                             gridspec_kw={"width_ratios": [1.05, 1]})
    fig.subplots_adjust(left=.04, right=.98, bottom=.17, top=.88, wspace=.16)
    axes[0].imshow(rgb)
    axes[0].axis("off")
    axes[0].set_title("A  MetaDrive / Waymo scenario at intervention time", fontsize=11, weight="bold")
    draw_map(axes[1], scenario, frame, xlim, ylim)
    draw_tracks(axes[1], frame, history, factual, ego_color="#16a34a")
    branch_xy = to_plot(frame, local_to_world(frame, intervention))
    mask = intervention[:, 0, 0] > .5
    axes[1].plot(branch_xy[mask, 0, 0], branch_xy[mask, 0, 1], color="#ef4444",
                 lw=2, ls="--", label="One counterfactual action", zorder=6)
    axes[1].set_title("B  Same scenario: map + factual / counterfactual tracks",
                      fontsize=11, weight="bold")
    axes[1].legend(loc="lower left", fontsize=7, framealpha=.9)
    max_sep = float(np.max(np.linalg.norm(factual[:, 0, 2:4] - intervention[:, 0, 2:4], axis=1)))
    fig.text(.5, .06, "Best-rendered of 20 hash-selected pilot scenarios; visual legibility only. "
             "Blue: other vehicles  |  Green: factual ego  |  Red dashed: fixed LEFT × 1 intervention.\n"
             f"The ego branches nearly overlap at this red light (max Map-Frenet separation {max_sep:.3f} m); "
             "this platform panel does not illustrate a treatment effect.",
             ha="center", fontsize=8, va="center")
    fig.savefig(dest_png, dpi=600, facecolor="white")
    fig.savefig(dest_pdf, facecolor="white")
    plt.close(fig)


def _case_row(axes, label, scenario, frame, history, factual, chosen, oracle,
              action_plans, a_hat, a_star, regret, direct_pct, baseline_pct):
    xlim, ylim = _limits(frame, history, factual, chosen, oracle)
    chosen_xy = local_to_world(frame, chosen)[:, 0]
    oracle_xy = local_to_world(frame, oracle)[:, 0]
    peak = peak_step(chosen_xy, oracle_xy)
    branches = [factual, chosen, oracle]
    titles = ["Factual context", "Model-selected action", "Oracle action"]
    colors = ["#64748b", "#dc2626", "#16a34a"]
    for col, (ax, branch, title, color) in enumerate(zip(axes, branches, titles, colors)):
        draw_map(ax, scenario, frame, xlim, ylim)
        draw_tracks(ax, frame, history, branch, ego_color=color,
                    mark_peak=peak if col > 0 else None)
        if col == 0:
            subtitle = f"t* context; factual continuation"
        else:
            index = a_hat if col == 1 else a_star
            action, duration = action_plans[index]
            names = ["LEFT", "IDLE", "RIGHT", "FASTER", "SLOWER"]
            subtitle = f"{names[action]} × {duration} step(s)"
        ax.set_title(f"{title}\n{subtitle}", fontsize=9, weight="bold")
    caption = (f"{label} | true regret={regret:.3f} | Direct risk pct={direct_pct:.1%} | "
               f"frozen baseline pct={baseline_pct:.1%} | peak separation step={peak+1}/20")
    return peak, caption


def _save_case_figure(case_rows, dest_png, dest_pdf, k_success):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 6.6))
    fig.subplots_adjust(left=.045, right=.99, bottom=.17, top=.84, hspace=.7, wspace=.2)
    peaks, captions = [], []
    for row, item in enumerate(case_rows):
        peak, caption = _case_row(axes[row], **item)
        peaks.append(peak)
        captions.append(caption)
        if row == 0:
            for ax in axes[row]:
                # Both rows use the same coordinate definition. Suppressing
                # duplicate x labels leaves a clear band for the second-row
                # case caption without changing any trajectory geometry.
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
    fig.text(.045, .96, captions[0], ha="left", va="top", fontsize=9, weight="bold")
    fig.text(.045, .54, captions[1], ha="left", va="top", fontsize=9, weight="bold")
    fig.text(.5, .05,
             f"Illustrative best-of-{k_success} eligible confirmation case (top), selected post-adjudication "
             "by a preregistered rule; not representative or used for inference/model selection.\n"
             "Bottom: mechanically selected worst false-negative. Stars mark peak ego-branch separation.",
             ha="center", fontsize=8, va="center")
    fig.savefig(dest_png, dpi=600, facecolor="white")
    fig.savefig(dest_pdf, facecolor="white")
    plt.close(fig)
    return peaks


def main() -> int:
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-revision", type=int,
                        help="Reuse the create-once selection, write improved layout in a new directory")
    args = parser.parse_args()
    start = time.monotonic()
    _validate_terminal()
    if not DB_DIR.is_dir():
        raise FileNotFoundError(DB_DIR)
    original_receipt_path = OUT_DIR / "selection_receipt.json"
    prior_selection = read_json(original_receipt_path) if args.layout_revision is not None else None
    if args.layout_revision is not None:
        if args.layout_revision < 2:
            raise ValueError("layout revision must be >=2")
        OUT_DIR = D2F_DIR / f"paper_visuals_layout_v{args.layout_revision}"
    receipt_path = OUT_DIR / ("render_receipt.json" if args.layout_revision else "selection_receipt.json")
    if receipt_path.exists():
        raise FileExistsError(f"create-once selection receipt already exists: {receipt_path}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[paper-viz] phase 1/4: frozen selection and source checks", flush=True)
    pilot = np.load(PILOT_DIR / "groups.npz", allow_pickle=False)
    pilot_meta = read_json(PILOT_DIR / "groups_meta.json")
    pool = read_json(POOL_PATH)
    if pool["k_platform"] != 20 or len(pool["pool"]) != 20:
        raise RuntimeError("platform candidate pool is not frozen K=20")
    platform, ranked_platform = _select_platform(pilot, pilot_meta, pool)
    quantities_path = D2F_DIR / "confirm/confirm_group_quantities.npz"
    quantities = np.load(quantities_path, allow_pickle=False)
    confirm_meta = read_json(D2F_DIR / "roles/confirm/groups_meta.json")
    scenario_ids = json.loads(str(quantities["scenario_ids_json"]))
    if scenario_ids != [m["scenario_id"] for m in confirm_meta]:
        raise RuntimeError("confirmation scenario ID alignment failure")
    ids = [f"{m['scenario_id']}|{m['t_star']}" for m in confirm_meta]
    selected = select_cases(quantities["regret"], quantities["ensemble_risk"],
                            quantities[BASELINE_KEY], quantities["a_hat"],
                            quantities["a_star"], ids)
    k_success, k_failure = len(selected["success_indices"]), len(selected["failure_indices"])
    print(f"[paper-viz] platform={platform['scenario_id']} | success K={k_success} | "
          f"failure K={k_failure}", flush=True)
    if not k_success or not k_failure:
        raise RuntimeError("prespecified illustrative success/failure candidates unavailable")
    if prior_selection is not None:
        frozen = prior_selection["confirmation_cases"]["chosen"]
        if (platform["scenario_id"] != prior_selection["platform"]["selected_scenario_id"]
                or confirm_meta[selected["success_index"]]["scenario_id"] != frozen[0]["scenario_id"]
                or confirm_meta[selected["failure_index"]]["scenario_id"] != frozen[1]["scenario_id"]):
            raise RuntimeError("layout revision attempted to change a create-once selected scenario")
    lookup = with_heartbeat("load ScenarioNet index", _scenario_lookup)

    print("[paper-viz] phase 2/4: actual MetaDrive 3D replay", flush=True)
    platform_index, platform_scenario = load_scenario(platform["scenario_id"], lookup)
    pilot_i = platform["pilot_index"]
    raw_3d = OUT_DIR / "raw_metadrive_platform.png"
    if prior_selection is None:
        camera = with_heartbeat("MetaDrive 3D screenshot", lambda: _render_3d_platform(
            platform["scenario_id"], pilot_meta[pilot_i], platform_scenario, platform_index, raw_3d))
    else:
        shutil.copyfile(original_receipt_path.parent / "raw_metadrive_platform.png", raw_3d)
        camera = {key: prior_selection["platform"][key] for key in
                  ("capture_size", "camera", "render_mode")}
        print("[paper-viz] reused original genuine MetaDrive 3D screenshot", flush=True)

    print("[paper-viz] phase 3/4: map and qualitative trajectory panels", flush=True)
    platform_frame = F.build_group_frame(platform_scenario, int(pilot_meta[pilot_i]["t_star"]),
                                         pilot_meta[pilot_i]["ego_lane_id"])
    p_png = OUT_DIR / "Fig_D2_metadrive_platform_and_branch.png"
    p_pdf = OUT_DIR / "Fig_D2_metadrive_platform_and_branch.pdf"
    with_heartbeat("platform figure", lambda: _save_platform_figure(
        raw_3d, platform_scenario, platform_frame,
        pilot["object_history"][pilot_i], pilot["object_factual_future"][pilot_i],
        pilot["object_counterfactual_futures"][pilot_i, 0], p_png, p_pdf))

    confirm = np.load(D2F_DIR / "roles/confirm/groups.npz", allow_pickle=False)
    actions = json.loads(str(confirm["action_plans_json"]))
    case_rows = []
    chosen_info = []
    for tag, index in (("Success, best-of-K", selected["success_index"]),
                       ("Failure, worst false-negative", selected["failure_index"])):
        meta = confirm_meta[index]
        _scenario_index, scenario = load_scenario(meta["scenario_id"], lookup)
        frame = F.build_group_frame(scenario, int(meta["t_star"]), meta["ego_lane_id"])
        a_hat, a_star = int(quantities["a_hat"][index]), int(quantities["a_star"][index])
        record = dict(
            label=tag, scenario=scenario, frame=frame,
            history=confirm["object_history"][index],
            factual=confirm["object_factual_future"][index],
            chosen=confirm["object_counterfactual_futures"][index, a_hat],
            oracle=confirm["object_counterfactual_futures"][index, a_star],
            action_plans=actions, a_hat=a_hat, a_star=a_star,
            regret=float(quantities["regret"][index]),
            direct_pct=float(selected["direct_percentile"][index]),
            baseline_pct=float(selected["baseline_percentile"][index]),
        )
        case_rows.append(record)
        chosen_info.append(dict(
            kind=tag, group_index=index, scenario_id=meta["scenario_id"],
            context_id=str(meta["t_star"]), action_selected=actions[a_hat],
            action_oracle=actions[a_star], regret=record["regret"],
            direct_risk_percentile=record["direct_pct"],
            baseline_risk_percentile=record["baseline_pct"],
        ))
    c_png = OUT_DIR / "Fig_D5_best_of_k_success_and_failure.png"
    c_pdf = OUT_DIR / "Fig_D5_best_of_k_success_and_failure.pdf"
    peaks = with_heartbeat("success/failure figure", lambda: _save_case_figure(
        case_rows, c_png, c_pdf, k_success))
    for record, peak in zip(chosen_info, peaks):
        record["t_peak_1_based"] = int(peak + 1)

    print("[paper-viz] phase 4/4: image validation and create-once receipt", flush=True)
    images = {}
    for path in (raw_3d, p_png, p_pdf, c_png, c_pdf):
        if not path.is_file() or path.stat().st_size < 20_000:
            raise RuntimeError(f"missing or implausibly small figure: {path}")
        images[path.name] = dict(sha256=sha256(path), bytes=path.stat().st_size)
    source_lock = read_json(ROOT / "experiments/stage4_reliability/stage4d_external/platform/platform_environment_lock.json")
    receipt = {
        "purpose": "non-adjudicating paper visualization; does not change D2F Gate",
        "source_terminal_status": "D2F_META_DRIVE_EXTERNAL_CONFIRMED",
        "plan_selection_contract": "D2 v0.2 section 14.2, inherited by D2F section 6",
        "platform": {
            "k_platform": 20,
            "pool_ordered_hashes": [r["hash"] for r in pool["pool"]],
            "legibility_ranked": ranked_platform,
            "selected_scenario_id": platform["scenario_id"],
            "selected_context_id": platform["context_id"],
            "legibility_formula": "100*visible_entities + 2*min(min_separation_m,12) + min(min_margin_m,10) - 10*occluded_pairs_lt3m; fixed s[-30,30],d[-15,15]",
            "intervention_branch": actions[0],
            **camera,
        },
        "confirmation_cases": {
            "k_success": k_success, "k_false_negative": k_failure,
            "success_candidate_id_hashes_ordered": [hashlib.sha256(ids[i].encode()).hexdigest() for i in selected["success_indices"]],
            "failure_candidate_id_hashes_ordered": [hashlib.sha256(ids[i].encode()).hexdigest() for i in selected["failure_indices"]],
            "risk_percentile_definition": "empirical upper rank; top20 via >= pooled 80th quantile",
            "success_criteria": "regret>.01, Direct top20%, baseline not top20%, a_hat!=a_star",
            "failure_criteria": "regret top5%, Direct not top20%",
            "chosen": chosen_info,
        },
        "metadrive_commit": source_lock["repos"]["metadrive_commit"],
        "scenarionet_commit": source_lock["repos"]["scenarionet_commit"],
        "source_sha256": {
            "platform_pool": sha256(POOL_PATH),
            "pilot_groups": sha256(PILOT_DIR / "groups.npz"),
            "pilot_meta": sha256(PILOT_DIR / "groups_meta.json"),
            "confirm_quantities": sha256(quantities_path),
            "confirm_groups": sha256(D2F_DIR / "roles/confirm/groups.npz"),
            "confirm_meta": sha256(D2F_DIR / "roles/confirm/groups_meta.json"),
        },
        "code_sha256": {"render_script": sha256(Path(__file__)),
                        "selection_module": sha256(ROOT / "code/ac_twin/stage4d_paper_visualization.py")},
        "images": images,
        "caption_boundary": "Qualitative best-of-K success is nonrepresentative; full confirm results alone support statistical inference.",
    }
    if prior_selection is not None:
        receipt["layout_revision_of_selection_receipt_sha256"] = sha256(original_receipt_path)
        receipt["layout_revision_boundary"] = "Same selected scenarios and values; rigidly rotated map view and changed typography/layout only."
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"[paper-viz] SUCCESS 5 images + receipt | elapsed={time.monotonic()-start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[paper-viz] FAILED {type(error).__name__}: {error}", flush=True)
        raise
