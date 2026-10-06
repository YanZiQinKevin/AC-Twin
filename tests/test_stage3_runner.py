from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage3_protocol as s3p  # noqa: E402

import run_stage3  # noqa: E402


DEVELOPMENT = ROOT / "data/object_centric_stage/merge_2000/object_twinworld_2000.npz"


@pytest.mark.skipif(not DEVELOPMENT.is_file(), reason="development npz not present")
def test_smoke_outputs_engineering_smoke_only(tmp_path, capsys) -> None:
    rc = run_stage3.main([
        "smoke", "--episodes", "12", "--epochs", "1",
        "--output-dir", str(tmp_path),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ENGINEERING_SMOKE_ONLY" in out
    assert "SUCCESS" in out

    report_path = tmp_path / "smoke_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # The smoke status can only ever be ENGINEERING_SMOKE_ONLY.
    assert report["status"] == "ENGINEERING_SMOKE_ONLY"
    # Output signature is recomputable from the on-disk JSON.
    assert s3p.output_signature_from_disk(report_path) == report["output_signature"]
    # Joint lock verifies.
    assert s3p.verify_joint_lock(tmp_path)["lock_sha256"]


def test_validate_config_rejects_sealed(tmp_path, capsys) -> None:
    config = {
        "plan_id": s3p.PLAN_ID,
        "development_source": "data/object_centric_stage/stage2_merge_1200/object_twinworld_stage2_merge_1200.npz",
        "response_bins": {"zero_tolerance": 0.0001, "t1": 0.24843344965718578, "t2": 1.898992796585611},
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed"):
        run_stage3.main(["validate-config", "--config", str(path)])
