"""Build the full AC-Twin Fig. 5 from frozen MetaDrive/Waymo artifacts.

The figure preserves the original narrative composition: a genuine MetaDrive
view and matching map/branch panel above two frozen qualitative case triptychs.
The platform scene is selected only for paper visualization from the already
frozen K=20 pool, using a deterministic hierarchy fixed in this script.  It is
not an experimental observation, model-selection input, or inferential result.

Read-only with respect to every frozen artifact. Loads the ScenarioNet scenario
through MetaDrive, so it must run under the frozen simulation interpreter:

    C:\\ProgramData\\anaconda3\\envs\\ac_twin_d2_sim_py311\\python.exe \
        paper/figures/fig5/build_fig5.py

Writes (inside paper/figures/fig5/):
    Fig5_metadrive_external_validation.svg / .png / .pdf
    Fig5_selected_platform_raw.png
    Fig5_render_receipt.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# matplotlib's PDF font embedding is chatty at INFO level; keep the run readable.
logging.getLogger("fontTools").setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

D2F = ROOT / "experiments/stage4_reliability/stage4d_d2f"
PILOT_DIR = ROOT / "experiments/stage4_reliability/stage4d_external/roles/pilot"
POOL_PATH = (ROOT / "experiments/stage4_reliability/stage4d_external/platform"
             / "viz_platform_pool.json")
ARCHIVE = ROOT / "paper/figures/stage4d_d2f"
RECEIPT = ARCHIVE / "provenance/selection_receipt.json"
RAW_3D = ARCHIVE / "platform/raw_metadrive_platform.png"
DB_DIR = Path(r"D:\MetaDrive\scenarionet_waymo_training_20s")
ROLE_SALT = "AC-TWIN-D2-V0.1|"

MM = 1 / 25.4
FIG_W, FIG_H = 180 * MM, 160 * MM
MIN_PT = 5.5

# Figure-set colours, plus the four scene colours the author-fixed panel used.
INK = "#0f172a"
MUTED = "#64748b"
FACTUAL = "#16a34a"       # factual ego future
COUNTERFACTUAL = "#ef4444"  # the one counterfactual action
OTHER = "#38bdf8"         # other vehicles' predicted future
LANE = "#94a3b8"
OTHER_MAP = "#cbd5e1"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.0,
    "svg.fonttype": "none",   # keep in-figure text editable, as in Figs. 1-4
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
})

# Layout, in figure fractions. The full figure remains within a 180 mm
# two-column width and groups the lower six axes as two logical panels.
MARGIN = 0.025


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_matplotlib_panel_alignment(fig, panel_axes, *, tolerance_pt=.25):
    """Fail closed if the two three-panel case rows drift out of alignment."""
    if len(panel_axes) != 2 or any(len(row) != 3 for row in panel_axes):
        raise RuntimeError("Fig. 5 alignment gate expects a 2 x 3 case grid")
    width_pt, height_pt = (float(value * 72.0) for value in fig.get_size_inches())
    boxes = []
    for row_i, row in enumerate(panel_axes):
        box_row = []
        for col_i, axis in enumerate(row):
            position = axis.get_position(original=False)
            box_row.append([
                float(position.x0 * width_pt), float(position.y0 * height_pt),
                float(position.x1 * width_pt), float(position.y1 * height_pt),
            ])
        boxes.append(box_row)
    for row in boxes:
        widths = [box[2] - box[0] for box in row]
        if max(widths) - min(widths) > tolerance_pt:
            raise RuntimeError("case-row panel widths are not aligned")
        gutters = [row[i + 1][0] - row[i][2] for i in range(2)]
        if max(gutters) - min(gutters) > tolerance_pt:
            raise RuntimeError("case-row horizontal gutters are not aligned")
    for col_i in range(3):
        if abs(boxes[0][col_i][0] - boxes[1][col_i][0]) > tolerance_pt:
            raise RuntimeError("case-grid columns are not aligned")
        if abs(boxes[0][col_i][2] - boxes[1][col_i][2]) > tolerance_pt:
            raise RuntimeError("case-grid column widths are not aligned")
    manifest = {
        "schema_version": 1,
        "backend": "python-matplotlib",
        "figure": {"width_pt": width_pt, "height_pt": height_pt},
        "panels": [
            {"id": f"{'c' if row_i == 0 else 'd'}{col_i + 1}",
             "bbox_pt": boxes[row_i][col_i]}
            for row_i in range(2) for col_i in range(3)
        ],
        "row_groups": [
            {"id": "success-row", "panels": ["c1", "c2", "c3"]},
            {"id": "failure-row", "panels": ["d1", "d2", "d3"]},
        ],
        "column_groups": [
            {"id": f"case-column-{col_i + 1}",
             "panels": [f"c{col_i + 1}", f"d{col_i + 1}"]}
            for col_i in range(3)
        ],
        "exemptions": [],
    }
    (OUT / "Fig5_layout_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def top_header_layout():
    """Use one figure-level baseline for both top panel labels and titles."""
    y = .935
    return {
        "a_axes_left": .035,
        "b_axes_left": .595,
        "a_label": (.035, y),
        "a_title": (.070, y),
        "b_label": (.595, y),
        "b_title": (.630, y),
    }


def load_scenario(sid: str):
    from metadrive.scenario.utils import read_scenario_data

    with (DB_DIR / "dataset_summary.pkl").open("rb") as stream:
        summary = pickle.load(stream)
    for file_name in summary:
        candidate = str(summary[file_name].get(
            "scenario_id",
            file_name.replace("sd_waymo_v1.2_", "").replace(".pkl", "")))
        if candidate == sid:
            return read_scenario_data(str(DB_DIR / file_name), centralize=True)
    raise KeyError(f"scenario {sid} not found in the frozen ScenarioNet index")


def select_platform(pilot, pilot_meta, pool):
    """Recompute the frozen, effect-blind platform selection."""
    from ac_twin.stage4d_paper_visualization import platform_legibility

    index_by_id = {row["scenario_id"]: i for i, row in enumerate(pilot_meta)}
    rows = []
    for candidate in pool["pool"]:
        i = index_by_id[candidate["scenario_id"]]
        assert str(pilot_meta[i]["t_star"]) == candidate["context_id"]
        rows.append({**candidate,
                     **platform_legibility(pilot["object_history"][i]),
                     "pilot_index": i})
    rows.sort(key=lambda row: (-row["score"], row["hash"]))
    return rows[0], rows


def rank_visualization_candidates(rows):
    """Rank the frozen K=20 pool for a readable, non-inferential paper panel.

    The hierarchy is fixed before rendering: retain the richest scene first,
    then prefer a visibly separated factual/counterfactual ego branch, then the
    original geometry-only legibility score, with the frozen candidate hash as
    the deterministic final tie-break.  This ranking never enters a Gate.
    """
    return sorted(
        rows,
        key=lambda row: (
            -int(row["visible_entities"]),
            -float(row["max_branch_separation_m"]),
            -float(row["score"]),
            str(row["hash"]),
        ),
    )


def select_visible_branch_platform(pilot, pilot_meta, pool):
    """Mechanically select the best-of-K platform illustration."""
    from ac_twin.stage4d_paper_visualization import platform_legibility

    index_by_key = {
        (row["scenario_id"], str(row["t_star"])): i
        for i, row in enumerate(pilot_meta)
    }
    rows = []
    for candidate in pool["pool"]:
        key = (candidate["scenario_id"], candidate["context_id"])
        if key not in index_by_key:
            raise RuntimeError(f"frozen visualization candidate missing from pilot: {key}")
        i = index_by_key[key]
        factual = pilot["object_factual_future"][i]
        intervention = pilot["object_counterfactual_futures"][i, 0]
        valid = (factual[:, 0, 0] > .5) & (intervention[:, 0, 0] > .5)
        if not valid.any():
            max_sep = float("-inf")
        else:
            max_sep = float(np.linalg.norm(
                factual[valid, 0, 2:4] - intervention[valid, 0, 2:4], axis=1
            ).max())
        rows.append({
            **candidate,
            **platform_legibility(pilot["object_history"][i]),
            "pilot_index": i,
            "max_branch_separation_m": max_sep,
        })
    ranked = rank_visualization_candidates(rows)
    if not ranked:
        raise RuntimeError("frozen visualization candidate pool is empty")
    return ranked[0], ranked


def scenario_lookup():
    with (DB_DIR / "dataset_summary.pkl").open("rb") as stream:
        summary = pickle.load(stream)
    result = {}
    for index, file_name in enumerate(summary):
        sid = str(summary[file_name].get(
            "scenario_id",
            file_name.replace("sd_waymo_v1.2_", "").replace(".pkl", ""),
        ))
        result[sid] = (index, file_name)
    return result


def load_scenario_from_lookup(sid, lookup):
    from metadrive.scenario.utils import read_scenario_data

    index, file_name = lookup[sid]
    return index, read_scenario_data(str(DB_DIR / file_name), centralize=True)


def local_to_world(frame, features):
    from ac_twin.stage4d_paper_visualization import frenet_to_world

    return frenet_to_world(
        frame.reference, features[..., 2], features[..., 3], origin_s=frame.origin_s
    )


def to_plot(frame, features):
    from ac_twin.stage4d_paper_visualization import world_to_ego_plot

    return world_to_ego_plot(
        local_to_world(frame, features), frame.reference, origin_s=frame.origin_s
    )


def plot_limits(frame, *arrays):
    points = []
    for features in arrays:
        xy = to_plot(frame, features)
        points.append(xy[features[..., 0] > .5])
    all_xy = np.concatenate(points, axis=0)
    low, high = np.min(all_xy, axis=0) - 10.0, np.max(all_xy, axis=0) + 10.0
    return (float(low[0]), float(high[0])), (float(low[1]), float(high[1]))


def draw_map(ax, scenario, frame, xlim, ylim, *, label_axes=True):
    from ac_twin.stage4d_paper_visualization import world_to_ego_plot

    ax.set_facecolor("#f8fafc")
    for item in scenario.get("map_features", {}).values():
        if not isinstance(item, dict):
            continue
        poly = np.asarray(item.get("polyline", []), dtype=np.float64)
        if poly.ndim != 2 or len(poly) < 2 or poly.shape[1] < 2:
            continue
        poly = world_to_ego_plot(poly[:, :2], frame.reference, origin_s=frame.origin_s)
        if (poly[:, 0].max() < xlim[0] or poly[:, 0].min() > xlim[1]
                or poly[:, 1].max() < ylim[0] or poly[:, 1].min() > ylim[1]):
            continue
        is_lane = "LANE" in str(item.get("type", ""))
        ax.plot(poly[:, 0], poly[:, 1], color=LANE if is_lane else OTHER_MAP,
                lw=0.62 if is_lane else 0.42, alpha=.70, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    if label_axes:
        ax.set_xlabel("Forward (m; ego-heading frame)", fontsize=MIN_PT, labelpad=1.2)
        ax.set_ylabel("Left (m)", fontsize=MIN_PT, labelpad=1.2)
    ax.tick_params(labelsize=MIN_PT, length=1.8, pad=1.2)


def draw_tracks(ax, frame, history, future, *, ego_color, mark_peak=None,
                counterfactual_style=False):
    past, fut = to_plot(frame, history), to_plot(frame, future)
    for slot in range(5):
        pmask = history[:, slot, 0] > .5
        fmask = future[:, slot, 0] > .5
        if pmask.any():
            ax.plot(past[pmask, slot, 0], past[pmask, slot, 1], color=MUTED,
                    lw=1.30 if slot == 0 else .65, alpha=.86, zorder=4)
        if fmask.any():
            color = ego_color if slot == 0 else OTHER
            kwargs = dict(color=color, lw=1.75 if slot == 0 else .72,
                          alpha=1 if slot == 0 else .65, zorder=6 if slot == 0 else 5)
            if slot == 0 and counterfactual_style:
                kwargs.update(ls="--", dashes=(4, 2))
            ax.plot(fut[fmask, slot, 0], fut[fmask, slot, 1], **kwargs)
            ax.scatter(fut[fmask, slot, 0][-1], fut[fmask, slot, 1][-1],
                       s=18 if slot == 0 else 7, color=color, zorder=7)
        if pmask.any():
            ax.scatter(past[pmask, slot, 0][-1], past[pmask, slot, 1][-1],
                       s=38 if slot == 0 else 17,
                       color=INK if slot == 0 else "#0284c7", zorder=8)
            ax.annotate("EGO" if slot == 0 else f"V{slot}",
                        past[pmask, slot][-1], xytext=(2.5, 2.5),
                        textcoords="offset points", fontsize=MIN_PT,
                        weight="bold", color=INK, zorder=9)
    if mark_peak is not None and future[mark_peak, 0, 0] > .5:
        ax.scatter(*fut[mark_peak, 0], marker="*", s=78,
                   facecolor="#facc15", edgecolor=INK, linewidth=.55, zorder=10)


def render_metadrive_screenshot(sid, pilot_meta, scenario, scenario_index, dest):
    """Replay one frozen pilot prefix and capture MetaDrive's own camera."""
    import stage4d_metadrive_adapter as adapter
    from metadrive.constants import DEFAULT_AGENT

    cls = adapter.make_env_class()
    lookup = scenario_lookup()
    cfg = dict(
        data_directory=str(DB_DIR), start_scenario_index=0,
        num_scenarios=len(lookup), sequential_seed=False,
        agent_policy=cls.ACTWIN_EGO_POLICY, reactive_traffic=True,
        filter_overlapping_car=False, no_static_vehicles=False, no_light=False,
        store_map=True, store_data=True, use_render=True,
        window_size=(1600, 900), show_interface=False, manual_control=False,
        horizon=None, decision_repeat=5,
        physics_world_step_size=adapter.NATIVE_DT / 5.0,
        show_policy_mark=False, show_fps=True, out_of_route_done=False,
        crash_vehicle_done=False, crash_object_done=False,
        crash_human_done=False, relax_out_of_road_done=True,
        max_lateral_dist=1e6, truncate_as_terminate=False,
    )
    context = adapter.prepare_group_context(scenario, sid, ROLE_SALT)
    if (context.t_star != int(pilot_meta["t_star"])
            or context.eval_track_ids != pilot_meta["eval_track_ids"]):
        raise RuntimeError("MetaDrive replay context does not match frozen pilot")
    env = cls(cfg)
    try:
        env.reset(seed=int(scenario_index))
        ego_obj_id = env.engine.agent_manager.agent_to_object(DEFAULT_AGENT)
        policy = env.engine.get_policy(ego_obj_id)
        policy.set_group_context(dict(
            t_star=context.t_star,
            sdc_track=env.engine.data_manager.current_scenario["tracks"][str(
                env.engine.data_manager.current_scenario["metadata"]["sdc_id"])],
            ego_lane_id=context.ego_lane_id,
            adjacency=context.adjacency,
            target_lanes=context.target_lanes,
            plan=np.asarray(context.factual_plan, dtype=np.int64),
            contract=context.contract,
        ))
        for _ in range(2 * context.t_star):
            env.step([0.0, 0.0])
        env.render()
        with tempfile.TemporaryDirectory(prefix="ac_twin_fig5_", dir=str(DB_DIR.parent)) as tmp:
            temp_png = Path(tmp) / "capture.png"
            from panda3d.core import Filename, PNMImage
            shot = PNMImage()
            if not env.engine.win.getScreenshot(shot):
                raise RuntimeError("Panda3D did not return a screenshot")
            if not shot.write(Filename.from_os_specific(str(temp_png))):
                raise RuntimeError("Panda3D could not write the screenshot")
            shutil.copyfile(temp_png, dest)
    finally:
        env.close()
    if not dest.is_file() or dest.stat().st_size < 20_000:
        raise RuntimeError("MetaDrive screenshot is missing or unexpectedly small")


def load_frozen_case_rows(lookup):
    """Load the two create-once qualitative cases without reselection."""
    from ac_twin import stage4d_external_features as features
    from ac_twin.stage4d_paper_visualization import peak_step

    receipt = read_json(RECEIPT)
    frozen_cases = receipt["confirmation_cases"]["chosen"]
    groups = np.load(D2F / "roles/confirm/groups.npz", allow_pickle=False)
    quantities = np.load(D2F / "confirm/confirm_group_quantities.npz", allow_pickle=False)
    metadata = read_json(D2F / "roles/confirm/groups_meta.json")
    actions = json.loads(str(groups["action_plans_json"]))
    rows = []
    for frozen in frozen_cases:
        index = int(frozen["group_index"])
        meta = metadata[index]
        if (meta["scenario_id"] != frozen["scenario_id"]
                or str(meta["t_star"]) != frozen["context_id"]):
            raise RuntimeError("frozen qualitative case alignment failure")
        _, scenario = load_scenario_from_lookup(meta["scenario_id"], lookup)
        frame = features.build_group_frame(scenario, int(meta["t_star"]), meta["ego_lane_id"])
        a_hat, a_star = int(quantities["a_hat"][index]), int(quantities["a_star"][index])
        if actions[a_hat] != frozen["action_selected"] or actions[a_star] != frozen["action_oracle"]:
            raise RuntimeError("frozen qualitative action alignment failure")
        selected = groups["object_counterfactual_futures"][index, a_hat]
        oracle = groups["object_counterfactual_futures"][index, a_star]
        peak = peak_step(local_to_world(frame, selected)[:, 0],
                         local_to_world(frame, oracle)[:, 0])
        if int(peak + 1) != int(frozen["t_peak_1_based"]):
            raise RuntimeError("frozen qualitative peak-step alignment failure")
        rows.append(dict(
            frozen=frozen, scenario=scenario, frame=frame,
            history=groups["object_history"][index],
            factual=groups["object_factual_future"][index],
            selected=selected, oracle=oracle, actions=actions,
            a_hat=a_hat, a_star=a_star, peak=peak,
        ))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    from ac_twin.stage4d_external_features import build_group_frame
    from ac_twin.stage4d_paper_visualization import (
        frenet_to_world, world_to_ego_plot)

    # ---- 1. frozen selection, asserted against the create-once receipt ------
    receipt = read_json(RECEIPT)
    pilot = np.load(PILOT_DIR / "groups.npz", allow_pickle=False)
    pilot_meta = read_json(PILOT_DIR / "groups_meta.json")
    pool = read_json(POOL_PATH)
    assert pool["k_platform"] == 20 and len(pool["pool"]) == 20
    platform, _ranked = select_platform(pilot, pilot_meta, pool)
    frozen = receipt["platform"]
    assert platform["scenario_id"] == frozen["selected_scenario_id"], "scene drift"
    assert platform["context_id"] == frozen["selected_context_id"], "context drift"
    assert frozen["intervention_branch"] == [0, 1]
    pilot_i = platform["pilot_index"]
    meta = pilot_meta[pilot_i]
    print("frozen platform scene %s  t*=%s  pilot_index=%d  sha256(3D)=%s"
          % (platform["scenario_id"], platform["context_id"], pilot_i,
             sha256(RAW_3D)[:16]))

    # ---- 2. frozen geometry ------------------------------------------------
    scenario = load_scenario(platform["scenario_id"])
    frame = build_group_frame(scenario, int(meta["t_star"]), meta["ego_lane_id"])
    history = pilot["object_history"][pilot_i]
    factual = pilot["object_factual_future"][pilot_i]
    intervention = pilot["object_counterfactual_futures"][pilot_i, 0]

    def to_plot(features):
        return world_to_ego_plot(
            frenet_to_world(frame.reference, features[..., 2], features[..., 3],
                            origin_s=frame.origin_s),
            frame.reference, origin_s=frame.origin_s)

    pts = []
    for features in (history, factual, intervention):
        xy = to_plot(features)
        pts.append(xy[features[..., 0] > .5])
    all_xy = np.concatenate(pts, axis=0)
    xlim = (float(all_xy[:, 0].min()) - 12, float(all_xy[:, 0].max()) + 12)
    ylim = (float(all_xy[:, 1].min()) - 12, float(all_xy[:, 1].max()) + 12)
    max_sep = float(np.max(np.linalg.norm(factual[:, 0, 2:4]
                                          - intervention[:, 0, 2:4], axis=1)))

    # ---- 3. figure ---------------------------------------------------------
    rgb = np.asarray(Image.open(RAW_3D).convert("RGB"))
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    ax_a = fig.add_axes([MARGIN, AX_Y, A_W, AX_H])
    ax_b = fig.add_axes([MARGIN + A_W + GAP, AX_Y, B_W, AX_H])

    ax_a.imshow(rgb, aspect="auto")
    ax_a.axis("off")

    ax_b.set_facecolor("#f8fafc")
    for item in scenario.get("map_features", {}).values():
        if not isinstance(item, dict):
            continue
        poly = np.asarray(item.get("polyline", []), dtype=np.float64)
        if poly.ndim != 2 or len(poly) < 2 or poly.shape[1] < 2:
            continue
        poly = world_to_ego_plot(poly[:, :2], frame.reference,
                                 origin_s=frame.origin_s)
        if (poly[:, 0].max() < xlim[0] or poly[:, 0].min() > xlim[1]
                or poly[:, 1].max() < ylim[0] or poly[:, 1].min() > ylim[1]):
            continue
        is_lane = "LANE" in str(item.get("type", ""))
        ax_b.plot(poly[:, 0], poly[:, 1], color=LANE if is_lane else OTHER_MAP,
                  lw=0.9 if is_lane else 0.6, alpha=.7, zorder=1)

    past = to_plot(history)
    future = to_plot(factual)
    for slot in range(5):
        pmask, fmask = history[:, slot, 0] > .5, factual[:, slot, 0] > .5
        if pmask.any():
            ax_b.plot(past[pmask, slot, 0], past[pmask, slot, 1], color=MUTED,
                      lw=1.8 if slot == 0 else 0.9, alpha=.85, zorder=4)
        if fmask.any():
            color = FACTUAL if slot == 0 else OTHER
            ax_b.plot(future[fmask, slot, 0], future[fmask, slot, 1], color=color,
                      lw=2.4 if slot == 0 else 1.0, alpha=1 if slot == 0 else .65,
                      zorder=5)
            ax_b.scatter(future[fmask, slot, 0][-1], future[fmask, slot, 1][-1],
                         s=26 if slot == 0 else 9, color=color, zorder=6)
        if pmask.any():
            ax_b.scatter(past[pmask, slot, 0][-1], past[pmask, slot, 1][-1],
                         s=58 if slot == 0 else 26,
                         color=INK if slot == 0 else "#0284c7", zorder=7)
            ax_b.annotate("EGO" if slot == 0 else f"V{slot}",
                          past[pmask, slot][-1], xytext=(3.5, 3.5),
                          textcoords="offset points", fontsize=MIN_PT,
                          weight="bold", color=INK, zorder=8)

    branch = to_plot(intervention)
    mask = intervention[:, 0, 0] > .5
    ax_b.plot(branch[mask, 0, 0], branch[mask, 0, 1], color=COUNTERFACTUAL,
              lw=1.8, ls="--", label="One counterfactual action", zorder=6)

    ax_b.set_xlim(*xlim)
    ax_b.set_ylim(*ylim)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.grid(False)
    ax_b.set_xlabel("Forward (m; ego-heading frame)", fontsize=8.0, labelpad=2.0)
    ax_b.set_ylabel("Left (m)", fontsize=8.0, labelpad=2.0)
    ax_b.tick_params(labelsize=MIN_PT, length=2.2, pad=1.6)
    ax_b.legend(loc="lower left", fontsize=MIN_PT, framealpha=.9,
                handlelength=1.6, borderpad=0.3)

    HEAD_Y = 0.936
    fig.text(MARGIN, HEAD_Y, "(a)", fontsize=9.6, fontweight="bold",
             color="#2B2B2B", ha="left", va="center")
    fig.text(MARGIN + 0.5 * A_W, HEAD_Y,
             "MetaDrive / Waymo at intervention time",
             fontsize=8.8, fontweight="bold", color="#2B2B2B", ha="center",
             va="center")
    fig.text(MARGIN + A_W + GAP, HEAD_Y, "(b)", fontsize=9.6,
             fontweight="bold", color="#2B2B2B", ha="left", va="center")
    fig.text(MARGIN + A_W + GAP + 0.5 * B_W, HEAD_Y,
             "same scenario: map + frozen tracks",
             fontsize=8.8, fontweight="bold", color="#2B2B2B", ha="center",
             va="center")

    stem = "Fig5_platform_panel"
    written, blocked = [], []
    for ext in ("svg", "png", "pdf"):
        target = OUT / f"{stem}.{ext}"
        try:
            if ext == "png":
                fig.savefig(target, dpi=600, facecolor="white")
            elif ext == "pdf":
                fig.savefig(target, dpi=600, facecolor="white",
                            metadata={"Title": "MetaDrive / Waymo platform panel"})
            else:
                fig.savefig(target, facecolor="white")
            written.append(ext)
        except PermissionError:
            blocked.append(ext)
    plt.close(fig)

    print("wrote %s/%s  (%.0f x %.0f mm)" % (stem, "/".join(written),
                                             FIG_W / MM, FIG_H / MM))
    print("  max Map-Frenet ego-branch separation = %.6f m" % max_sep)
    print("  vehicles drawn: %d"
          % int(sum(1 for s in range(5) if (history[:, s, 0] > .5).any())))
    if blocked:
        print("WARNING: %s held open by another program; close it and re-run"
              % ", ".join(blocked))
    return 0


def action_label(plan) -> str:
    names = {0: "LEFT", 1: "IDLE", 2: "RIGHT", 3: "FASTER", 4: "SLOWER"}
    return f"{names[int(plan[0])]} x {int(plan[1])} step(s)"


def render_full_figure() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-screenshot", action="store_true",
        help="replay the frozen MetaDrive candidate even if a cached capture exists",
    )
    args = parser.parse_args()

    from ac_twin import stage4d_external_features as features

    print("[phase 1/4] verify frozen sources and select the platform illustration")
    confirm_report = read_json(D2F / "confirm/confirm_report.json")
    if confirm_report.get("status") != "D2F_META_DRIVE_EXTERNAL_CONFIRMED":
        raise RuntimeError("Stage 4D-D2F terminal status is not confirmed")
    if not (D2F / "confirm/confirmation_used.lock").is_file():
        raise RuntimeError("the frozen D2F confirmation lock is missing")

    pilot = np.load(PILOT_DIR / "groups.npz", allow_pickle=False)
    pilot_meta = read_json(PILOT_DIR / "groups_meta.json")
    pool = read_json(POOL_PATH)
    if int(pool.get("k_platform", -1)) != 20 or len(pool.get("pool", [])) != 20:
        raise RuntimeError("the frozen platform pool is not K=20")
    platform, ranked = select_visible_branch_platform(pilot, pilot_meta, pool)
    expected = ("1cac180d139c96de", "13")
    selected_key = (platform["scenario_id"], str(platform["context_id"]))
    if selected_key != expected:
        raise RuntimeError(f"platform selection drift: {selected_key} != {expected}")
    if float(platform["max_branch_separation_m"]) <= 1.0:
        raise RuntimeError("selected platform branch is not visibly separated")

    lookup = scenario_lookup()
    scenario_index, scenario = load_scenario_from_lookup(platform["scenario_id"], lookup)
    pilot_i = int(platform["pilot_index"])
    meta = pilot_meta[pilot_i]
    frame = features.build_group_frame(scenario, int(meta["t_star"]), meta["ego_lane_id"])
    history = pilot["object_history"][pilot_i]
    factual = pilot["object_factual_future"][pilot_i]
    intervention = pilot["object_counterfactual_futures"][pilot_i, 0]
    xlim, ylim = plot_limits(frame, history, factual, intervention)

    raw_capture = OUT / "Fig5_selected_platform_raw.png"
    print("[phase 2/4] render the matching MetaDrive camera view")
    if args.refresh_screenshot or not raw_capture.is_file():
        render_metadrive_screenshot(
            platform["scenario_id"], meta, scenario, scenario_index, raw_capture,
        )
    rgb = np.asarray(Image.open(raw_capture).convert("RGB"))

    print("[phase 3/4] compose platform panels and the six frozen case panels")
    case_rows = load_frozen_case_rows(lookup)
    if len(case_rows) != 2:
        raise RuntimeError("the frozen qualitative receipt must contain two cases")

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

    # Top row: native MetaDrive view and its matching geometric reconstruction.
    ax_a = fig.add_axes([.035, .650, .505, .270])
    ax_b = fig.add_axes([.595, .650, .365, .270])
    ax_a.imshow(rgb, aspect="auto")
    ax_a.axis("off")
    headers = top_header_layout()
    fig.text(*headers["a_label"], "(a)", fontsize=8.2, fontweight="bold",
             ha="left", va="center")
    fig.text(*headers["a_title"], "MetaDrive / Waymo scene at intervention time",
             fontsize=7.2, fontweight="bold", ha="left", va="center")

    draw_map(ax_b, scenario, frame, xlim, ylim, label_axes=True)
    draw_tracks(ax_b, frame, history, factual, ego_color=FACTUAL)
    branch_xy = to_plot(frame, intervention)
    branch_mask = intervention[:, 0, 0] > .5
    if not branch_mask.any():
        raise RuntimeError("selected counterfactual ego branch is empty")
    red_line, = ax_b.plot(
        branch_xy[branch_mask, 0, 0], branch_xy[branch_mask, 0, 1],
        color=COUNTERFACTUAL, lw=2.35, ls="--", dashes=(1.5, 1.0),
        label="Counterfactual: LEFT x 1", zorder=12,
    )
    ax_b.scatter(
        branch_xy[branch_mask, 0, 0][-1], branch_xy[branch_mask, 0, 1][-1],
        s=28, facecolor=COUNTERFACTUAL, edgecolor="white", linewidth=.45,
        zorder=13,
    )
    ax_b.legend(loc="lower left", fontsize=MIN_PT, framealpha=.92,
                handlelength=2.2, borderpad=.3)
    fig.text(*headers["b_label"], "(b)", fontsize=8.2, fontweight="bold",
             ha="left", va="center")
    fig.text(*headers["b_title"],
             "Same scene: factual vs. counterfactual tracks",
             fontsize=7.2, fontweight="bold", ha="left", va="center")

    # The lower panels preserve the create-once success/failure selections.
    case_axes = [[], []]
    row_y = [.365, .075]
    row_header_y = [.575, .285]
    panel_letters = ["(c)", "(d)"]
    for row_i, row in enumerate(case_rows):
        frozen = row["frozen"]
        if row_i == 0:
            heading = (
                "Correctly flagged high-regret case (best of 205 eligible)  |  "
                f"regret={float(frozen['regret']):.3f}  |  "
                f"audit-risk percentile={100*float(frozen['direct_risk_percentile']):.1f}%  |  "
                f"baseline={100*float(frozen['baseline_risk_percentile']):.1f}%"
            )
        else:
            heading = (
                "Worst false-negative (mechanical selection)  |  "
                f"regret={float(frozen['regret']):.3f}  |  "
                f"audit-risk percentile={100*float(frozen['direct_risk_percentile']):.1f}%  |  "
                f"baseline={100*float(frozen['baseline_risk_percentile']):.1f}%"
            )
        fig.text(.020, row_header_y[row_i], panel_letters[row_i],
                 fontsize=8.2, fontweight="bold", va="center")
        fig.text(.050, row_header_y[row_i], heading,
                 fontsize=6.4, fontweight="bold", va="center")

        variants = [
            ("Factual context", row["factual"], MUTED, None),
            ("Model-selected action\n" + action_label(frozen["action_selected"]),
             row["selected"], COUNTERFACTUAL, row["peak"]),
            ("Oracle action\n" + action_label(frozen["action_oracle"]),
             row["oracle"], FACTUAL, row["peak"]),
        ]
        case_xlim, case_ylim = plot_limits(
            row["frame"], row["history"], row["factual"],
            row["selected"], row["oracle"],
        )
        for col_i, (title, future, color, peak) in enumerate(variants):
            ax = fig.add_axes([.055 + col_i * .315, row_y[row_i], .275, .145])
            case_axes[row_i].append(ax)
            draw_map(ax, row["scenario"], row["frame"], case_xlim, case_ylim,
                     label_axes=False)
            draw_tracks(ax, row["frame"], row["history"], future,
                        ego_color=color, mark_peak=peak,
                        counterfactual_style=(col_i == 1))
            ax.set_title(title, fontsize=6.2, fontweight="bold", pad=2)
            if row_i == 1:
                ax.set_xlabel("Forward (m)", fontsize=MIN_PT, labelpad=.8)
            if col_i == 0:
                ax.set_ylabel("Left (m)", fontsize=MIN_PT, labelpad=.8)
            else:
                ax.set_ylabel("")

    fig.text(
        .5, .018,
        "Platform pair: best-rendered of 20 frozen candidates, selected only for visual legibility; "
        "not used for inference or model selection.\n"
        "Lower rows retain the preregistered qualitative success and mechanical false-negative; "
        "stars mark peak model/oracle ego-branch separation.",
        ha="center", va="bottom", fontsize=5.5, color="#334155",
    )

    # Render before export so visibility can be checked in display coordinates.
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, case_axes)
    red_xy_px = ax_b.transData.transform(np.column_stack([
        branch_xy[branch_mask, 0, 0], branch_xy[branch_mask, 0, 1],
    ]))
    red_span_px = np.ptp(red_xy_px, axis=0)
    if float(np.max(red_span_px)) < 8.0:
        raise RuntimeError("panel (b) counterfactual branch remains visually collapsed")
    if red_line.get_zorder() <= 9:
        raise RuntimeError("panel (b) red counterfactual branch is not top-layer visible")

    print("[phase 4/4] export PNG, SVG, PDF, and provenance receipt")
    stem = "Fig5_metadrive_external_validation"
    output_paths = {}
    for ext in ("svg", "png", "pdf"):
        target = OUT / f"{stem}.{ext}"
        save_kwargs = dict(facecolor="white")
        if ext == "png":
            save_kwargs["dpi"] = 600
        elif ext == "pdf":
            save_kwargs["dpi"] = 600
            save_kwargs["metadata"] = {
                "Title": "AC-Twin external validation and qualitative cases"
            }
        fig.savefig(target, **save_kwargs)
        output_paths[ext] = target
    plt.close(fig)

    source_paths = {
        "platform_pool": POOL_PATH,
        "pilot_groups": PILOT_DIR / "groups.npz",
        "pilot_metadata": PILOT_DIR / "groups_meta.json",
        "frozen_case_receipt": RECEIPT,
        "confirm_groups": D2F / "roles/confirm/groups.npz",
        "confirm_quantities": D2F / "confirm/confirm_group_quantities.npz",
    }
    render_receipt = {
        "status": "FIG5_RENDER_COMPLETE",
        "purpose": "paper visualization only; no experimental or inferential use",
        "selection_rule": [
            "visible_entities descending",
            "max_branch_separation_m descending",
            "original_legibility_score descending",
            "candidate_hash ascending",
        ],
        "platform": {
            "pool_size": 20,
            "scenario_id": platform["scenario_id"],
            "context_id": str(platform["context_id"]),
            "pilot_index": pilot_i,
            "candidate_hash": platform["hash"],
            "visible_entities": int(platform["visible_entities"]),
            "max_branch_separation_m": float(platform["max_branch_separation_m"]),
            "legibility_score": float(platform["score"]),
            "fps_overlay_retained": True,
            "red_branch_display_span_px": [float(v) for v in red_span_px],
        },
        "ranked_candidates": [
            {
                "scenario_id": row["scenario_id"],
                "context_id": str(row["context_id"]),
                "candidate_hash": row["hash"],
                "visible_entities": int(row["visible_entities"]),
                "max_branch_separation_m": float(row["max_branch_separation_m"]),
                "legibility_score": float(row["score"]),
            }
            for row in ranked
        ],
        "frozen_cases": [row["frozen"] for row in case_rows],
        "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
        "output_sha256": {
            "raw_metadrive_capture": sha256(raw_capture),
            **{ext: sha256(path) for ext, path in output_paths.items()},
        },
        "figure_size_mm": [180, 160],
    }
    (OUT / "Fig5_render_receipt.json").write_text(
        json.dumps(render_receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "SUCCESS: %s (180 x 160 mm); scene=%s; separation=%.3f m; red span=%.1f px"
        % (stem, platform["scenario_id"],
           float(platform["max_branch_separation_m"]), float(np.max(red_span_px)))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(render_full_figure())
