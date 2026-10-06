"""Contracts for the frozen Stage 4D-D2 environment entry point."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4d_external_protocol as protocol  # noqa: E402
PLAN = ROOT / "plans" / "2026-09-21_AC-Twin_Stage4D-D2复杂平台外部复现实施计划_v0.2.md"
LAUNCHER = ROOT / "code" / "run_stage4d_env.ps1"

SIM_PYTHON = r"C:\ProgramData\anaconda3\envs\ac_twin_d2_sim_py311\python.exe"
CONVERT_PYTHON = "/root/miniconda3/envs/ac_twin_waymo_convert_py39/bin/python"
WSL_DISTRO = "Ubuntu-22.04"


def test_plan_freezes_exact_environment_entrypoint() -> None:
    text = PLAN.read_text(encoding="utf-8")
    assert SIM_PYTHON in text
    assert CONVERT_PYTHON in text
    assert WSL_DISTRO in text
    assert "code\\run_stage4d_env.ps1" in text
    assert "禁止使用未限定解释器的 `python`、`pip` 或 `conda run`" in text


def test_launcher_show_reports_frozen_interpreters() -> None:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LAUNCHER),
            "show",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    payload = json.loads(completed.stdout)
    assert payload["sim_python"] == SIM_PYTHON
    assert payload["conversion_python"] == CONVERT_PYTHON
    assert payload["wsl_distribution"] == WSL_DISTRO
    assert Path(payload["project_root"]).resolve() == ROOT.resolve()


def test_environment_lock_uses_exact_configured_interpreters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def capture(cmd: list[str]) -> str:
        calls.append(list(cmd))
        return ""

    monkeypatch.setattr(protocol, "_run_capture", capture)
    config = {
        "sim_env_python_exe": SIM_PYTHON,
        "conversion_wsl_distribution": WSL_DISTRO,
        "conversion_env_python_exe": CONVERT_PYTHON,
    }

    protocol._pip_freeze_sim_env(config)
    protocol._pip_freeze_conversion_env(config)

    assert calls == [
        [SIM_PYTHON, "-m", "pip", "freeze"],
        [
            "wsl.exe",
            "-d",
            WSL_DISTRO,
            "--",
            CONVERT_PYTHON,
            "-m",
            "pip",
            "freeze",
        ],
    ]


def test_sim_entry_preloads_torch_before_metadrive(tmp_path: Path) -> None:
    probe = tmp_path / "probe_import_order.py"
    probe.write_text(
        "import json, sys\n"
        "torch_was_preloaded = 'torch' in sys.modules\n"
        "import metadrive\n"
        "import torch\n"
        "print(json.dumps({'torch_was_preloaded': torch_was_preloaded, "
        "'cuda': torch.cuda.is_available()}))\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LAUNCHER),
            "sim",
            str(probe),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload == {"torch_was_preloaded": True, "cuda": True}
