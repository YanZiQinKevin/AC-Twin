"""Stage 4D-D2R plan-5 power pre-check driver (read-only; NO collection).

Thin entry for ``ac_twin.stage4d_d2r_power.run_power`` so the plan-5 Go/No-Go can run under the
frozen Stage 4D environment BEFORE the full ``run_stage4d_d2r.py`` phased runner exists. Exit 0
when a screen size is selected (the plan continues); exit 5 on the preregistered terminal
``D2R_SCREEN_UNDERPOWERED`` (no collection is created).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from ac_twin import stage4d_d2r_power as PW  # noqa: E402


def main() -> int:
    last = {"at": 0.0}
    t0 = time.perf_counter()

    def progress(line: str) -> None:
        now = time.perf_counter()
        if now - last["at"] >= 5.0:
            last["at"] = now
            print(f"[stage4d-d2r-power] (+{now-t0:.0f}s) {line}", flush=True)

    report = PW.run_power(progress=progress)
    print(f"[stage4d-d2r-power] SUMMARY status={report['status']} "
          f"selected_n={report['selected_n']} underpowered={report['underpowered']} "
          f"total_elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return 5 if report["underpowered"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
