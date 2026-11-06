"""Commit-after-durable checkpointing for the table and folder adapters.

The tail adapter saves its offset when it hands records over. These adapters
do better: a source position is written to disk only after the agent reports,
through `commit(before)`, that every record up to it sits in a closed window
that is already in the durable spool. A crash therefore replays from the last
committed position: records are never lost, and the replay can at worst
re-emit a window the server has already accepted, which the server dedupes by
window id (the spool file name is the window end, so an unsent duplicate
overwrites itself).

Positions are staged per batch, not per row: memory is O(batches).
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from edge_agent.adapters import SourceAdapter

# A record dated further ahead than this cannot hold the checkpoint back: its window
# would stay open for years and stall the source. Such a record is garbage anyway.
FUTURE_CLAMP_SECONDS = 86_400.0
SOURCES_DIR = "sources"


def state_path(state_dir: Path, name: str) -> Path:
    return Path(state_dir) / SOURCES_DIR / f"{name}.json"


def write_json_atomic(path: Path, payload: Any) -> None:
    write_text_atomic(path, json.dumps(payload, sort_keys=True))


def write_text_atomic(path: Path, text: str) -> None:
    """Temp file, fsync, replace, fsync the directory: after a crash the file is the old or the new text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def fsync_dir(directory: Path) -> None:
    """Make a rename durable over a power loss (POSIX). Windows cannot open a directory; NTFS needs nothing here."""
    if os.name == "nt":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class CheckpointedAdapter(SourceAdapter):
    """Subclasses implement `_apply(delta)` (advance the committed state) and
    `_state()` (the JSON-able committed state); the base class does the staging."""

    def __init__(self, state_dir: Path, name: str, *, max_uncommitted_rows: int,
                 clock=time.time) -> None:
        self.name = name
        self.state_file = state_path(state_dir, name)
        self.max_uncommitted_rows = max_uncommitted_rows
        self._clock = clock
        self._staged: deque[tuple[float, Any, int]] = deque()   # (max_ts, delta, rows)
        self._staged_rows = 0
        self.rows_delivered = 0
        self.rows_committed = 0
        self.bad_rows = 0
        self.last_error: str | None = None
        self.last_error_at: float | None = None

    # -- staging ------------------------------------------------------------------------
    def _stage(self, max_ts: float | None, delta: Any, rows: int) -> None:
        """Hold `delta` until every row in it is durable. `max_ts` is the newest record
        time in the batch (None: nothing in it waits on a window)."""
        if max_ts is None:
            ts = float("-inf")
        else:
            ts = min(max_ts, self._clock() + FUTURE_CLAMP_SECONDS)
        self._staged.append((ts, delta, rows))
        self._staged_rows += rows
        self.rows_delivered += rows

    @property
    def blocked(self) -> bool:
        """Too many delivered rows are still uncommitted: stop reading until they commit."""
        return self._staged_rows >= self.max_uncommitted_rows

    def commit(self, before: datetime | None = None) -> None:
        limit = None if before is None else before.timestamp()
        moved = False
        while self._staged and (limit is None or self._staged[0][0] < limit):
            _, delta, rows = self._staged.popleft()
            self._staged_rows -= rows
            self.rows_committed += rows
            self._apply(delta)
            moved = True
        if moved:
            write_json_atomic(self.state_file, self._state())

    def _apply(self, delta: Any) -> None:            # pragma: no cover - interface
        raise NotImplementedError

    def _state(self) -> Any:                         # pragma: no cover - interface
        raise NotImplementedError

    def _fail(self, exc: BaseException) -> None:
        """Never raise out of poll: remember the class (not the text) for `--status`."""
        self.last_error = type(exc).__name__
        self.last_error_at = self._clock()

    def _ok(self) -> None:
        self.last_error = None
