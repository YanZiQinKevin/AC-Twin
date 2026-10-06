"""Focused contracts for the Fig. 5 visualization-only platform selection."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper/figures/fig5/build_fig5.py"
SPEC = importlib.util.spec_from_file_location("build_fig5", SCRIPT)
assert SPEC and SPEC.loader
FIG5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIG5)


def test_rank_visualization_candidates_prefers_visibility_then_branch_separation() -> None:
    rows = [
        {"visible_entities": 4, "max_branch_separation_m": 30.0, "score": 900.0, "hash": "a"},
        {"visible_entities": 5, "max_branch_separation_m": 2.0, "score": 600.0, "hash": "c"},
        {"visible_entities": 5, "max_branch_separation_m": 3.0, "score": 500.0, "hash": "b"},
        {"visible_entities": 5, "max_branch_separation_m": 3.0, "score": 500.0, "hash": "a"},
    ]
    ranked = FIG5.rank_visualization_candidates(rows)
    assert [row["hash"] for row in ranked] == ["a", "b", "c", "a"]


def test_frozen_pool_selects_visible_five_vehicle_scene_with_red_branch_support() -> None:
    pilot_dir = ROOT / "experiments/stage4_reliability/stage4d_external/roles/pilot"
    pilot = np.load(pilot_dir / "groups.npz", allow_pickle=False)
    metadata = json.loads((pilot_dir / "groups_meta.json").read_text(encoding="utf-8"))
    pool = json.loads((ROOT / "experiments/stage4_reliability/stage4d_external/platform/viz_platform_pool.json").read_text(encoding="utf-8"))

    selected, ranked = FIG5.select_visible_branch_platform(pilot, metadata, pool)

    assert len(ranked) == 20
    assert selected["scenario_id"] == "1cac180d139c96de"
    assert selected["context_id"] == "13"
    assert selected["visible_entities"] == 5
    assert selected["max_branch_separation_m"] > 3.0


def test_top_panel_labels_and_titles_share_one_baseline() -> None:
    layout = FIG5.top_header_layout()
    assert layout["a_label"][1] == layout["a_title"][1]
    assert layout["b_label"][1] == layout["b_title"][1]
    assert layout["a_label"][1] == layout["b_label"][1]
    assert layout["a_label"][0] == layout["a_axes_left"]
    assert layout["b_label"][0] == layout["b_axes_left"]
