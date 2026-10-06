"""The qualitative figure is exported as six independently reusable panels."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from export_stage4d_d2f_case_panels import panel_specs  # noqa: E402


def test_panel_specs_are_six_unique_panels_in_stable_order() -> None:
    chosen = [{"kind": "Success, best-of-K", "group_index": 550},
              {"kind": "Failure, worst false-negative", "group_index": 1821}]
    specs = panel_specs(chosen)
    assert [s["filename"] for s in specs] == [
        "success_01_factual.png", "success_02_selected.png", "success_03_oracle.png",
        "failure_01_factual.png", "failure_02_selected.png", "failure_03_oracle.png",
    ]
    assert [(s["group_index"], s["branch"]) for s in specs] == [
        (550, "factual"), (550, "selected"), (550, "oracle"),
        (1821, "factual"), (1821, "selected"), (1821, "oracle"),
    ]
