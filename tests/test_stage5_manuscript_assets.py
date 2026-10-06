"""Paper-facing Stage 5 table and reference-ledger tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "paper" / "tables" / "table_stage5_sota"
    / "build_stage5_manuscript_assets.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("build_stage5_manuscript_assets", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_builds_one_nine_row_comparison_table_without_placeholder() -> None:
    module = _load_module()
    rows = module.build_comparison_rows()
    assert len(rows) == 9
    assert rows[-1]["method"] == "AC-Twin audit"
    assert all("frozen" not in str(row.get("inference_ms_per_group", "")).lower()
               for row in rows)
    assert sum(row["category"] == "published" for row in rows) == 5
    assert sum(row["category"] == "internal" for row in rows) == 3
    learned = [row for row in rows if row["param_count_per_seed"] is not None]
    assert all(row["approx_forward_flops_per_group"]
               == 2 * row["param_count_per_seed"] for row in learned)
    assert all(row["approx_forward_flops_per_group"] is None
               for row in rows if row["category"] == "internal")


def test_reference_ledger_starts_with_all_registered_stage5_works() -> None:
    module = _load_module()
    rows = module.build_reference_rows()
    assert len(rows) == 16
    keys = [row["ref_key"] for row in rows]
    assert len(keys) == len(set(keys))
    assert all(row["manuscript_no"] == "" for row in rows)
