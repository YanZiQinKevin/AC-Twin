"""Frozen bootstrap for all Stage 4D-D2 Windows simulation processes.

On this Windows stack MetaDrive/Panda3D must not be imported before PyTorch,
otherwise PyTorch's ``c10.dll`` can fail to initialize.  This bootstrap makes
the safe import order part of the environment contract instead of relying on
the import order of each downstream script.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch  # noqa: F401  # Must be first third-party import in this process.


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("stage4d_sim_entry.py requires a script path or -m module")

    if args[0] == "-m":
        if len(args) < 2:
            raise SystemExit("-m requires a module name")
        module = args[1]
        sys.argv = [module, *args[2:]]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return 0

    script = Path(args[0])
    if not script.is_file():
        raise SystemExit(f"Stage 4D simulation script not found: {script}")
    sys.argv = [str(script), *args[1:]]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

