from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "code" / "run_stage3b_plus.py"
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from run_stage3b_plus import (  # noqa: E402
    evaluate_learned_dev_screen,
    screen_oracle_headroom,
    select_shape_lambda,
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DRIVER), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=180,
    )


def test_preflight_and_real_smoke_resume(tmp_path: Path):
    preflight = _run("preflight")
    assert preflight.returncode == 0, preflight.stdout + preflight.stderr
    assert "PREFLIGHT_PASS" in preflight.stdout

    output = tmp_path / "smoke"
    interrupted = _run(
        "smoke", "--output-dir", str(output), "--episodes", "4", "--epochs", "1",
        "--interrupt-after-checkpoint",
    )
    assert interrupted.returncode == 75, interrupted.stdout + interrupted.stderr
    assert (output / "checkpoints" / "uniform_suffix.pt").is_file()

    resumed = _run(
        "smoke", "--output-dir", str(output), "--episodes", "4", "--epochs", "1", "--resume"
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    report = json.loads((output / "smoke_report.json").read_text(encoding="utf-8"))
    assert report["state"] == "ENGINEERING_SMOKE_ONLY"
    assert report["resumed"] is True
    assert report["checkpoint_roundtrip"] is True
    assert report["temporary_gate"]["crossfit_receipt_seen"] is True


def test_oracle_headroom_screen_is_mechanical_and_prefers_smaller_tie():
    passing = {
        "state_full": {
            "validation_ade": 0.9,
            "a_rel_improvements": [0.06, 0.06, 0.04],
            "b_rel_improvements": [0.05, 0.08, 0.04],
        },
        "state_two_channel": {
            "validation_ade": 0.9,
            "a_rel_improvements": [0.06, 0.07, 0.08],
            "b_rel_improvements": [0.06, 0.07, 0.08],
        },
    }
    decision = screen_oracle_headroom(passing)
    assert decision["passed"] is True
    assert decision["selected_architecture"] == "state_full"

    failing = {
        "state_full": {"validation_ade": 1.0, "a_rel_improvements": [0.04] * 3, "b_rel_improvements": [0.06] * 3},
        "state_two_channel": {"validation_ade": 0.8, "a_rel_improvements": [0.06] * 3, "b_rel_improvements": [-0.1, 0.06, 0.06]},
    }
    assert screen_oracle_headroom(failing)["status"] == "NO_ROUTE_ONLY_ORACLE_HEADROOM"


def test_shape_lambda_selection_and_learned_screen_are_frozen_mechanical_rules():
    assert select_shape_lambda({0.0: 0.3, 0.05: 0.2}) == 0.05
    assert select_shape_lambda({0.0: 0.2, 0.05: 0.2}) == 0.0
    passing = {
        "a_uniform": {"point": 0.08, "ci80_lower": 0.02},
        "a_direct": {"point": 0.06, "ci80_lower": 0.01},
        "b_uniform": {"point": 0.07, "ci80_lower": 0.03},
        "b_direct": {"point": 0.05, "ci80_lower": 0.001},
    }
    assert evaluate_learned_dev_screen(passing)["passed"] is True
    failing = dict(passing)
    failing["b_direct"] = {"point": 0.049, "ci80_lower": 0.01}
    assert evaluate_learned_dev_screen(failing)["status"] == "LEARNED_ROUTER_DEV_INSUFFICIENT"
