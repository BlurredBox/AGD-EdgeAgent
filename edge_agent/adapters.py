"""Pluggable local record sources.

An adapter turns whatever the partner has (a log file, an in-process call)
into `Record`s. The agent only ever calls `poll()`; adapters must not block.
"""
from __future__ import annotations

import json
import os
import queue
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Iterable

from edge_agent.records import Record


class SourceAdapter(ABC):
    @abstractmethod
    def poll(self, max_records: int = 10_000) -> list[Record]:
        """Return the records that arrived since the last call (maybe none)."""


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


class JsonlTailAdapter(SourceAdapter):
    """Tails a JSON-lines file; the byte offset is persisted so a restart neither
    replays nor skips lines. Rotation/truncation (file shrank) restarts at 0.

    Default line shape: {"ts", "model_id", "instrument", "features": {...}, "output"}.
    `parse` can map any other shape to those five keys. Unparseable lines are
    counted (`bad_lines`), never raised: a corrupt log line must not stop the agent.
    """

    def __init__(self, path: str | Path, offset_file: str | Path, *,
                 parse: Callable[[dict], dict] | None = None) -> None:
        self.path, self.offset_file = Path(path), Path(offset_file)
        self.parse = parse
        self.bad_lines = 0
        self._offset = self._load_offset()

    def _load_offset(self) -> int:
        try:
            return int(self.offset_file.read_text().strip() or 0)
        except (OSError, ValueError):
            return 0

    def _save_offset(self) -> None:
        self.offset_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.offset_file.with_suffix(".tmp")
        tmp.write_text(str(self._offset))
        os.replace(tmp, self.offset_file)

    def poll(self, max_records: int = 10_000) -> list[Record]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []                      # file absent: the feed-down tripwire reports it
        if size < self._offset:
            self._offset = 0
        out: list[Record] = []
        with open(self.path, "rb") as fh:
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
        self._save_offset()
        return out


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
