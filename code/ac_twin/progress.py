"""Dependency-free textual progress reporting for long AC-Twin jobs."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


def format_duration(seconds: float | None) -> str:
    """Format a duration as HH:MM:SS, or a placeholder when unavailable."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass
class ProgressReporter:
    """Print progress at fixed completion intervals with a rough ETA.

    In addition to the completion-count trigger (``every``), an optional
    wall-clock heartbeat (``heartbeat_seconds``) forces an update at least once
    per interval even when no completion boundary is crossed, so long-running
    jobs never stay silent longer than the heartbeat period.
    """

    total: int
    every: int
    prefix: str
    started_at: float = field(default_factory=time.perf_counter)
    heartbeat_seconds: float | None = None
    _last_emit_at: float = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError("total must be positive")
        if self.every <= 0:
            raise ValueError("every must be positive")
        if self.heartbeat_seconds is not None and self.heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        # None here means "no emit yet"; the first heartbeat check uses started_at.
        self._last_emit_at = self.started_at

    def update(
        self,
        completed: int,
        *,
        attempted: int | None = None,
        details: str = "",
        force: bool = False,
    ) -> str | None:
        completed = max(0, min(int(completed), self.total))
        if not force and completed != self.total and completed % self.every != 0:
            return None
        return self._emit(completed, attempted=attempted, details=details)

    def heartbeat(
        self,
        completed: int,
        *,
        attempted: int | None = None,
        details: str = "",
    ) -> str | None:
        """Emit only if at least ``heartbeat_seconds`` elapsed since the last emit.

        Returns ``None`` (prints nothing) when the heartbeat period has not
        elapsed or no heartbeat is configured. Uses ``time.perf_counter`` so
        tests can drive a fake clock via monkeypatching.
        """
        if self.heartbeat_seconds is None:
            return None
        now = time.perf_counter()
        if now - self._last_emit_at < self.heartbeat_seconds:
            return None
        completed = max(0, min(int(completed), self.total))
        return self._emit(completed, attempted=attempted, details=details)

    def _emit(
        self,
        completed: int,
        *,
        attempted: int | None,
        details: str,
    ) -> str:
        now = time.perf_counter()
        self._last_emit_at = now
        elapsed = max(0.0, now - self.started_at)
        rate = completed / elapsed if completed > 0 and elapsed > 0 else 0.0
        eta = (self.total - completed) / rate if rate > 0 else None
        fields = [
            f"[{self.prefix}] {100.0 * completed / self.total:5.1f}%",
            f"{completed}/{self.total}",
        ]
        if attempted is not None:
            fields.append(f"attempts={int(attempted)}")
        fields.extend(
            [
                f"elapsed={format_duration(elapsed)}",
                f"ETA={format_duration(eta)}",
            ]
        )
        if details:
            fields.append(details)
        line = " | ".join(fields)
        print(line, flush=True)
        return line
