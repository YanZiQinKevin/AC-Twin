"""Export the frozen D2F qualitative case figure as six separate PNG panels.

Run with the existing Stage 4D simulation entry. This is visualization only:
no model inference, training, resampling, or Gate evaluation occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

import render_stage4d_d2f_paper_figures as render  # noqa: E402
from ac_twin import stage4d_external_features as features  # noqa: E402
from ac_twin.stage4d_paper_visualization import peak_step  # noqa: E402

D2F = ROOT / "experiments/stage4_reliability/stage4d_d2f"
SOURCE_RECEIPT = D2F / "paper_visuals_layout_v4/render_receipt.json"
OUT_DIR = D2F / "paper_visuals_subpanels"


def panel_specs(chosen: list[dict]) -> list[dict]:
    """Fixed order and names; no ranking or case replacement is permitted."""
    if len(chosen) != 2:
        raise ValueError("expected the frozen success and failure cases")
    specs = []
    for label, case in zip(("success", "failure"), chosen):
        for number, branch in enumerate(("factual", "selected", "oracle"), start=1):
            specs.append({
                "filename": f"{label}_{number:02d}_{branch}.png",
                "case": label,
                "group_index": int(case["group_index"]),
                "branch": branch,
            })
    return specs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_data(index, frozen, groups, quantities, metadata, lookup):
    meta = metadata[index]
    if meta["scenario_id"] != frozen["scenario_id"] or str(meta["t_star"]) != frozen["context_id"]:
        raise RuntimeError("frozen selected case does not match confirmation data")
    a_hat = int(quantities["a_hat"][index])
    a_star = int(quantities["a_star"][index])
    actions = json.loads(str(groups["action_plans_json"]))
    if actions[a_hat] != frozen["action_selected"] or actions[a_star] != frozen["action_oracle"]:
        raise RuntimeError("frozen selected/oracle action mismatch")
    _, scenario = render.load_scenario(meta["scenario_id"], lookup)
    frame = features.build_group_frame(scenario, int(meta["t_star"]), meta["ego_lane_id"])
    history = groups["object_history"][index]
    factual = groups["object_factual_future"][index]
    selected = groups["object_counterfactual_futures"][index, a_hat]
    oracle = groups["object_counterfactual_futures"][index, a_star]
    selected_xy = render.local_to_world(frame, selected)[:, 0]
    oracle_xy = render.local_to_world(frame, oracle)[:, 0]
    peak = peak_step(selected_xy, oracle_xy)
    if peak + 1 != frozen["t_peak_1_based"]:
        raise RuntimeError("frozen peak-separation step mismatch")
    return {
        "scenario": scenario, "frame": frame, "history": history,
        "branches": {"factual": factual, "selected": selected, "oracle": oracle},
        "limits": render._limits(frame, history, factual, selected, oracle),
        "peak": peak,
        "actions": {"factual": "observed factual plan",
                    "selected": actions[a_hat], "oracle": actions[a_star]},
    }


def _export_panel(spec, data, dest):
    fig, ax = plt.subplots(figsize=(6.0, 2.75))
    fig.subplots_adjust(left=.12, right=.98, bottom=.23, top=.96)
    xlim, ylim = data["limits"]
    render.draw_map(ax, data["scenario"], data["frame"], xlim, ylim)
    branch = spec["branch"]
    render.draw_tracks(
        ax, data["frame"], data["history"], data["branches"][branch],
        ego_color={"factual": "#64748b", "selected": "#dc2626", "oracle": "#16a34a"}[branch],
        mark_peak=data["peak"] if branch != "factual" else None,
    )
    fig.savefig(dest, dpi=600, facecolor="white")
    plt.close(fig)


def main() -> int:
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-revision", type=int,
                        help="write a new, shorter standalone layout without replacing earlier panels")
    args = parser.parse_args()
    if args.layout_revision is not None:
        if args.layout_revision < 2:
            raise ValueError("layout revision must be >=2")
        OUT_DIR = D2F / f"paper_visuals_subpanels_v{args.layout_revision}"
    render._validate_terminal()
    if not SOURCE_RECEIPT.is_file():
        raise FileNotFoundError(SOURCE_RECEIPT)
    receipt = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    frozen = receipt["confirmation_cases"]["chosen"]
    specs = panel_specs(frozen)
    manifest_path = OUT_DIR / "panels_manifest.json"
    if manifest_path.exists() or any((OUT_DIR / spec["filename"]).exists() for spec in specs):
        raise FileExistsError("standalone panels already exported; original images are preserved")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    groups_path = D2F / "roles/confirm/groups.npz"
    quantities_path = D2F / "confirm/confirm_group_quantities.npz"
    groups = np.load(groups_path, allow_pickle=False)
    quantities = np.load(quantities_path, allow_pickle=False)
    metadata = json.loads((D2F / "roles/confirm/groups_meta.json").read_text(encoding="utf-8"))
    lookup = render._scenario_lookup()
    case_data = {}
    for label, frozen_case in zip(("success", "failure"), frozen):
        case_data[label] = _case_data(frozen_case["group_index"], frozen_case,
                                      groups, quantities, metadata, lookup)
    print("[standalone-panels] 0/6 | rendering frozen success/failure scenes", flush=True)
    records = []
    for number, spec in enumerate(specs, start=1):
        path = OUT_DIR / spec["filename"]
        _export_panel(spec, case_data[spec["case"]], path)
        if not path.is_file() or path.stat().st_size < 20_000:
            raise RuntimeError(f"panel missing or unexpectedly small: {path}")
        record = {**spec, "sha256": _sha256(path), "bytes": path.stat().st_size,
                  "scenario_id": frozen[0 if spec["case"] == "success" else 1]["scenario_id"],
                  "action": case_data[spec["case"]]["actions"][spec["branch"]],
                  "t_peak_1_based": int(case_data[spec["case"]]["peak"] + 1),
                  "pixel_size": [3600, 1650]}
        records.append(record)
        print(f"[standalone-panels] {number}/6 | {path.name}", flush=True)
    manifest = {
        "purpose": "six unassembled, independently reusable panels from the frozen D2F qualitative cases",
        "source_selection_receipt_sha256": _sha256(SOURCE_RECEIPT),
        "source_groups_sha256": _sha256(groups_path),
        "source_quantities_sha256": _sha256(quantities_path),
        "script_sha256": _sha256(Path(__file__)),
        "k_success": int(receipt["confirmation_cases"]["k_success"]),
        "selection_boundary": "same frozen cases; no new ranking, model inference, or statistical adjudication",
        "panels": records,
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"[standalone-panels] SUCCESS | 6 individual PNGs and {manifest_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
