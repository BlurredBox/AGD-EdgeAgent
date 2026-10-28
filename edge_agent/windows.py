"""Cadence windows cut on epoch-aligned wall-clock boundaries.

A window is bucket `n = floor(ts / cadence)`; `n` is also the absolute index
the trajectory operators use, so a gap in the stream is a gap in the index and
`delta`/`ewma_slope` are None across it (never a rate across a hole).
Mirrors `ReplayHarness._windows`: boundary-aligned, not row-count-aligned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from edge_agent.records import Record


@dataclass
class ClosedWindow:
    model_id: str
    index: int                       # absolute bucket number
    start: datetime
    end: datetime
    records: list[Record] = field(default_factory=list)
    late_dropped: int = 0            # records that arrived after this window closed


class WindowCutter:
    """One per model. Records may arrive slightly out of order inside an open
    window; a record for an already-closed window is dropped and counted."""

    def __init__(self, model_id: str, cadence_seconds: int) -> None:
        self.model_id = model_id
        self.cadence = cadence_seconds
        self._open: dict[int, list[Record]] = {}
        self._closed_upto = -1            # highest bucket already emitted
        self.late_dropped = 0

    def bucket(self, ts: datetime) -> int:
        return int(ts.timestamp() // self.cadence)

    def bounds(self, n: int) -> tuple[datetime, datetime]:
        s = datetime.fromtimestamp(n * self.cadence, timezone.utc)
        return s, s + timedelta(seconds=self.cadence)

    def add(self, rec: Record) -> list[ClosedWindow]:
        """Add a record; return every window the record's time closes."""
        n = self.bucket(rec.ts)
        if n <= self._closed_upto:
            self.late_dropped += 1
            return []
        self._open.setdefault(n, []).append(rec)
        return self._close(lambda b: b < n)

    def tick(self, now: datetime, lateness_seconds: float = 0.0) -> list[ClosedWindow]:
        """Close windows by the wall clock (a quiet feed must not hold the last window forever)."""
        cutoff = self.bucket(now - timedelta(seconds=lateness_seconds))
        return self._close(lambda b: b < cutoff)

    def flush(self) -> list[ClosedWindow]:
        return self._close(lambda b: True)

    def _close(self, pred) -> list[ClosedWindow]:
        out = []
        for b in sorted(k for k in self._open if pred(k)):
            recs = sorted(self._open.pop(b), key=lambda r: r.ts)
            s, e = self.bounds(b)
            out.append(ClosedWindow(self.model_id, b, s, e, recs))
            self._closed_upto = max(self._closed_upto, b)
        return out
