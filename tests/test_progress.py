from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.progress import ProgressReporter, format_duration  # noqa: E402


def test_format_duration_uses_compact_human_units() -> None:
    assert format_duration(None) == "--:--"
    assert format_duration(65) == "00:01:05"
    assert format_duration(3661) == "01:01:01"


def test_reporter_renders_percentage_rate_elapsed_and_eta(monkeypatch, capsys) -> None:
    clock = iter([110.0])
    monkeypatch.setattr("ac_twin.progress.time.perf_counter", lambda: next(clock))
    reporter = ProgressReporter(total=100, every=10, prefix="collect", started_at=100.0)

    line = reporter.update(20, attempted=30, details="rejected=10", force=True)

    assert line is not None
    assert "20.0%" in line
    assert "20/100" in line
    assert "attempts=30" in line
    assert "elapsed=00:00:10" in line
    assert "ETA=00:00:40" in line
    assert "rejected=10" in line
    assert capsys.readouterr().out.strip() == line


def test_reporter_suppresses_non_interval_updates(capsys) -> None:
    reporter = ProgressReporter(total=100, every=10, prefix="collect")

    assert reporter.update(1) is None
    assert capsys.readouterr().out == ""


def test_reporter_always_emits_completion(capsys) -> None:
    reporter = ProgressReporter(total=3, every=10, prefix="train")

    line = reporter.update(3)

    assert line is not None
    assert "100.0%" in line
    assert "3/3" in line
    assert "ETA=00:00:00" in line


def test_heartbeat_fires_on_wall_clock_not_completion(monkeypatch, capsys) -> None:
    """The 30s heartbeat fires on elapsed time even with no completion change."""
    clock = {"t": 1000.0}
    monkeypatch.setattr("ac_twin.progress.time.perf_counter", lambda: clock["t"])
    reporter = ProgressReporter(
        total=100, every=50, prefix="train", started_at=1000.0, heartbeat_seconds=30.0
    )

    # Count-interval not reached (1 % 50 != 0); before 30s the heartbeat is silent.
    assert reporter.update(1) is None
    assert reporter.heartbeat(1) is None  # only 0s since start

    # Cross the 30s mark with no completion change -> heartbeat fires.
    clock["t"] += 31.0
    first = reporter.heartbeat(1)
    assert first is not None
    assert "1/100" in first

    # Only 10s elapsed since the last emit -> heartbeat suppressed.
    clock["t"] += 10.0
    assert reporter.heartbeat(1) is None

    # 20s more (>= 30s total) -> heartbeat fires again.
    clock["t"] += 20.0
    second = reporter.heartbeat(1)
    assert second is not None


def test_heartbeat_disabled_by_default(capsys) -> None:
    reporter = ProgressReporter(total=100, every=10, prefix="train")

    assert reporter.heartbeat(5) is None
    assert capsys.readouterr().out == ""


def test_heartbeat_resets_after_count_emit(monkeypatch, capsys) -> None:
    """A regular count emit also resets the heartbeat clock (no double print)."""
    clock = {"t": 0.0}
    monkeypatch.setattr("ac_twin.progress.time.perf_counter", lambda: clock["t"])
    reporter = ProgressReporter(
        total=100, every=10, prefix="train", started_at=0.0, heartbeat_seconds=30.0
    )

    assert reporter.update(10) is not None  # count emit at t=0
    clock["t"] = 20.0
    assert reporter.heartbeat(12) is None  # only 20s since the count emit
    clock["t"] = 31.0
    assert reporter.heartbeat(12) is not None
