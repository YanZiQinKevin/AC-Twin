"""Pytest shim for the frozen Stage 4D-D2 environment entry point.

``run_stage4d_env.ps1 sim`` cannot forward a leading ``-m`` (PowerShell binds it as
a script parameter), so test runs go through this shim:

    powershell.exe -NoProfile -ExecutionPolicy Bypass -File code\\run_stage4d_env.ps1 `
        sim code/stage4d_run_pytest.py tests/test_stage4d_external_split.py -q

Warnings-as-errors are controlled via the PYTHONWARNINGS environment variable.
"""
from __future__ import annotations

import sys

import pytest


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("usage: stage4d_run_pytest.py <pytest args>")
        return 2
    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
