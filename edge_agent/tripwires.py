"""Local tripwires: feed down, schema break, hard constraints.

They fire HERE, immediately, with no round trip, and keep working when the
backend link is down. Alerts go to a local JSONL file, the `edge_agent.alerts`
logger and any registered callbacks (the partner's pager/UI). They may name
feature names -- they are local -- and they are never sent to the backend.
"""
from __future__ import annotations

import json
import os
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("edge_agent.alerts")

FEED_DOWN, FEED_RECOVERED = "feed_down", "feed_recovered"
SCHEMA_BREAK, HARD_CONSTRAINT = "schema_break", "hard_constraint"
LINK_DOWN, LINK_RECOVERED = "link_down", "link_recovered"
REJECTED = "window_rejected"
LABELS_MISSING, LABELS_RECOVERED = "labels_missing", "labels_recovered"


@dataclass
class LocalAlert:
    kind: str
    model_id: str
    message: str
    at: str
    details: dict[str, Any] = field(default_factory=dict)


class LocalAlerter:
    MAX_FILE_BYTES = 2_000_000
    KEEP_LINES = 1000

    def __init__(self, path: str | Path, cooldown_seconds: float = 300.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.path = Path(path)
        self.cooldown = cooldown_seconds
        self._clock = clock
        self._last: dict[tuple[str, str, str], float] = {}
        self._callbacks: list[Callable[[LocalAlert], None]] = []
        self.history: list[LocalAlert] = []

    def subscribe(self, cb: Callable[[LocalAlert], None]) -> None:
        self._callbacks.append(cb)

    def raise_alert(self, kind: str, model_id: str, message: str, *,
                    key: str = "", **details: Any) -> LocalAlert | None:
        """Fire unless the same (kind, model, key) fired inside the cooldown."""
        now = self._clock()
        k = (kind, model_id, key)
        last = self._last.get(k)
        if last is not None and now - last < self.cooldown:
            return None
        self._last[k] = now
        alert = LocalAlert(kind, model_id, message,
                           datetime.now(timezone.utc).isoformat(), details)
        self.history.append(alert)
        log.warning("%s model=%s %s", kind, model_id, message)
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with os.fdopen(os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), "a") as fh:
                fh.write(json.dumps(asdict(alert), sort_keys=True, default=str) + "\n")
            self._trim()
        except OSError:
            log.exception("could not persist local alert")
        for cb in self._callbacks:
            try:
                cb(alert)
            except Exception:                  # noqa: BLE001 -- a bad sink must not stop the agent
                log.exception("local alert callback failed")
        return alert

    def _trim(self) -> None:
        """Keep the alert file bounded (audit F24): past MAX_FILE_BYTES only the newest lines stay."""
        try:
            if self.path.stat().st_size <= self.MAX_FILE_BYTES:
                return
            lines = self.path.read_text().splitlines()[-self.KEEP_LINES:]
            tmp = self.path.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as fh:
                fh.write("\n".join(lines) + "\n")
            os.replace(tmp, self.path)
        except OSError:
            log.exception("could not trim the local alert file")

    def clear(self, kind: str, model_id: str) -> None:
        """Re-arm a condition (e.g. after recovery) so its next breach alerts at once."""
        for k in [k for k in self._last if k[0] == kind and k[1] == model_id]:
            del self._last[k]


# Known limit: the required columns are learned from the first LEARN_ROWS rows after a start; a column that
# disappears inside them is optional until the next start (the window quality gate still compares against
# the frozen reference schema). Upgrade path: seed the required set from ReferenceStats once it is frozen.
LEARN_ROWS = 10


class RecordGuard:
    """Per-record checks, run at ingest so a breach alerts at once rather than
    at window close. Returns the reasons a record is unusable (empty = fine)."""

    def __init__(self, model_id: str, alerter: LocalAlerter, *,
                 feature_bounds: dict[str, tuple[float, float]],
                 output_bounds: tuple[float, float] | None) -> None:
        self.model_id, self.alerter = model_id, alerter
        self.bounds, self.output_bounds = feature_bounds, output_bounds
        self.expected_schema: set[str] | None = None  # required: present in every learning row
        self.known_columns: set[str] = set()          # present in any learning row
        self.expected_types: dict[str, str] = {}      # name -> "num" | "cat", from its first non-null value
        self._learn_left = LEARN_ROWS

    def learn(self, features: dict[str, Any]) -> None:
        """One learning row: a key absent from any of them is optional, never a reason to drop."""
        cols = set(features)
        self.expected_schema = cols if self.expected_schema is None else self.expected_schema & cols
        self.known_columns |= cols
        self._learn_left -= 1

    def _learn_types(self, features: dict[str, Any]) -> None:
        for k, v in features.items():
            if k not in self.expected_types and v is not None and not _isnan(v):
                self.expected_types[k] = _kind(v)

    def check(self, features: dict[str, Any], output: float) -> list[str]:
        bad: list[str] = []
        learning = self._learn_left > 0
        if learning:
            self.learn(features)
        cols = set(features)
        missing = set() if learning else self.expected_schema - cols
        extra = set() if learning else cols - self.known_columns
        if missing:
            bad.append("schema_missing_columns")
            self.alerter.raise_alert(
                SCHEMA_BREAK, self.model_id,
                f"feature columns missing: {sorted(missing)[:8]}", key="missing",
                missing=sorted(missing))
        if extra:
            self.alerter.raise_alert(
                SCHEMA_BREAK, self.model_id,
                f"unexpected new feature columns: {sorted(extra)[:8]}", key="extra",
                extra=sorted(extra))
        flipped = [c for c in cols
                   if features[c] is not None and not _isnan(features[c])
                   and self.expected_types.get(c) not in (None, _kind(features[c]))]
        self._learn_types(features)
        if flipped:
            bad.append("dtype_change")
            self.alerter.raise_alert(
                SCHEMA_BREAK, self.model_id,
                f"feature type changed (numeric<->categorical): {sorted(flipped)[:8]}",
                key="dtype", columns=sorted(flipped))
        for name, (lo, hi) in self.bounds.items():
            v = features.get(name)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) \
                    and not lo <= v <= hi:
                bad.append("feature_out_of_bounds")
                self.alerter.raise_alert(
                    HARD_CONSTRAINT, self.model_id,
                    f"feature {name!r} = {v} outside [{lo}, {hi}]", key=f"feat:{name}")
        if not math.isfinite(output):
            bad.append("output_not_finite")
            self.alerter.raise_alert(HARD_CONSTRAINT, self.model_id,
                                     "model output is NaN/inf", key="out_nonfinite")
        elif self.output_bounds and not self.output_bounds[0] <= output <= self.output_bounds[1]:
            bad.append("output_out_of_bounds")
            self.alerter.raise_alert(
                HARD_CONSTRAINT, self.model_id,
                f"model output {output} outside {list(self.output_bounds)}", key="out_bounds")
        return bad


def _isnan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _kind(v: Any) -> str:
    return "num" if isinstance(v, (int, float)) and not isinstance(v, bool) else "cat"
