"""Pluggable local record sources.

An adapter turns whatever the partner has (a log file, an in-process call)
into `Record`s. The agent only ever calls `poll()`; adapters must not block.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Iterable

from edge_agent.records import Record


class Sample:
    """A bounded look at a source for `--check`: nothing is staged, persisted or sent."""
    def __init__(self, records: list[Record], bad: int) -> None:
        self.records, self.bad = records, bad


class SourceAdapter(ABC):
    @abstractmethod
    def poll(self, max_records: int = 10_000) -> list[Record]:
        """Return the records that arrived since the last call (maybe none)."""

    def commit(self, before: "datetime | None" = None) -> None:
        """The agent has durably consumed every delivered record with `ts < before`
        (all of them when `before` is None). Adapters that checkpoint on commit
        override this; the tail/in-process adapters checkpoint on poll and ignore it."""

    def describe(self) -> dict:
        """Counts-only description for `--status` (no row values, ids or file names)."""
        return {}


class InProcessAdapter(SourceAdapter):
    """In-process hook: the partner's serving code calls `observe()` per prediction."""

    def __init__(self, maxsize: int = 100_000) -> None:
        self._q: "queue.Queue[Record]" = queue.Queue(maxsize=maxsize)
        self.dropped = 0

    def observe(self, ts: Any, model_id: str, instrument: str,
                features: dict[str, Any], output: Any, *, prediction_id: str | None = None,
                outcome: int | None = None, outcome_observed_at: Any = None) -> bool:
        """Never blocks the partner's hot path; a full buffer drops and counts."""
        try:
            self._q.put_nowait(Record.make(ts, model_id, instrument, features, output,
                                          prediction_id=prediction_id, outcome=outcome,
                                          outcome_observed_at=outcome_observed_at))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def poll(self, max_records: int = 10_000) -> list[Record]:
        out: list[Record] = []
        while len(out) < max_records:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out


# ponytail: the head hash catches copytruncate (same inode); a new file whose first HEAD_BYTES equal the old one's
# is not told apart. Rows left unread in a renamed-away file are not chased.
HEAD_BYTES = 256


class JsonlTailAdapter(SourceAdapter):
    """Tails a JSON-lines file; the byte offset is persisted so a restart neither
    replays nor skips lines. Rotation restarts at 0 (audit F13): a new inode, a changed
    head (copytruncate), a shrunk file, or a saved offset that is not at a line end.

    Default line shape: {"ts", "model_id", "instrument", "features": {...}, "output"}.
    `parse` can map any other shape to those five keys. Unparseable lines are
    counted (`bad_lines`), never raised: a corrupt log line must not stop the agent.
    """

    def __init__(self, path: str | Path, offset_file: str | Path, *,
                 parse: Callable[[dict], dict] | None = None) -> None:
        self.path, self.offset_file = Path(path), Path(offset_file)
        self.parse = parse
        self.bad_lines = 0
        self.rotations = 0
        # "<inode>:<sha256 of the first min(offset, HEAD_BYTES) bytes>", beside the offset file (which stays a bare int)
        self._ident_file = self.offset_file.with_name(self.offset_file.name + ".id")
        self._ident: str | None = None
        self._offset = self._load_offset()

    def _load_offset(self) -> int:
        try:
            self._ident = self._ident_file.read_text().strip() or None
        except OSError:
            self._ident = None             # no identity saved yet (older agent): the line-end check still applies
        try:
            return int(self.offset_file.read_text().strip() or 0)
        except (OSError, ValueError):
            return 0

    def _save_offset(self) -> None:
        self.offset_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.offset_file.with_suffix(".tmp")
        tmp.write_text(str(self._offset))
        os.replace(tmp, self.offset_file)
        if self._ident:                    # after the offset: a crash between the two re-reads from 0, never skips
            tmp.write_text(self._ident)
            os.replace(tmp, self._ident_file)

    def _identity(self, fh, ino: int) -> str:
        fh.seek(0)
        return f"{ino}:{hashlib.sha256(fh.read(min(self._offset, HEAD_BYTES))).hexdigest()}"

    def _rotated(self, fh, st: os.stat_result) -> bool:
        """The saved offset does not belong to this file, so resuming there would skip rows or start mid-line."""
        if self._offset == 0:
            return False
        if st.st_size < self._offset or (self._ident is not None and self._ident != self._identity(fh, st.st_ino)):
            return True
        fh.seek(self._offset - 1)
        return fh.read(1) != b"\n"

    def poll(self, max_records: int = 10_000) -> list[Record]:
        try:
            fh = open(self.path, "rb")
        except OSError:
            return []                      # file absent: the feed-down tripwire reports it
        out: list[Record] = []
        with fh:
            st = os.fstat(fh.fileno())     # the inode of the file actually read, not of a path renamed meanwhile
            if self._rotated(fh, st):
                self._offset = 0
                self.rotations += 1
            fh.seek(self._offset)
            while len(out) < max_records:
                line = fh.readline()
                if not line or not line.endswith(b"\n"):
                    break                  # EOF or a half-written line: wait for the rest
                self._offset += len(line)
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    if self.parse:
                        d = self.parse(d)
                    out.append(Record.make(d["ts"], d["model_id"], d["instrument"],
                                           d["features"], d["output"],
                                           prediction_id=d.get("prediction_id"), outcome=d.get("outcome"),
                                           outcome_observed_at=d.get("outcome_observed_at")))
                except Exception:          # noqa: BLE001 -- counted, never raised
                    self.bad_lines += 1
            self._ident = self._identity(fh, st.st_ino)
        self._save_offset()
        return out

    def describe(self) -> dict:
        return {"type": "jsonl", "offset": self._offset, "bad_lines": self.bad_lines, "rotations": self.rotations}

    def sample(self, limit: int = 1000) -> Sample:
        """Parse up to `limit` non-blank lines from the start of the file; no offset is written."""
        recs: list[Record] = []
        rows = bad = 0
        with open(self.path, encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                rows += 1
                try:
                    d = json.loads(line)
                    if self.parse:
                        d = self.parse(d)
                    recs.append(Record.make(d["ts"], d["model_id"], d["instrument"], d["features"],
                                            d["output"], prediction_id=d.get("prediction_id"),
                                            outcome=d.get("outcome"),
                                            outcome_observed_at=d.get("outcome_observed_at")))
                except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
                    bad += 1
                if rows >= limit:
                    break
        return Sample(recs, bad)


def drain(adapters: Iterable[SourceAdapter]) -> list[Record]:
    recs: list[Record] = []
    for a in adapters:
        recs.extend(a.poll())
    return recs


class JsonlOutcomeAdapter(JsonlTailAdapter):
    """Tail local outcome rows; timestamps mean when the result became known.

    Rows contain model_id, prediction_id, outcome (0/1) and
    outcome_observed_at. The resulting records belong only to the outcome
    source input and must never be fed into prediction statistics.
    """

    def __init__(self, path: str | Path, offset_file: str | Path) -> None:
        def parse(row: dict) -> dict:
            return {"ts": row["outcome_observed_at"], "model_id": row["model_id"],
                    "instrument": "", "features": {}, "output": 0,
                    "prediction_id": row["prediction_id"], "outcome": row["outcome"],
                    "outcome_observed_at": row["outcome_observed_at"]}
        super().__init__(path, offset_file, parse=parse)
