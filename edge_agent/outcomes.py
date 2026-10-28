"""Bounded, local-only joins of predictions and binary realised outcomes.

Labels arriving after emission are retained as metadata, never attached to a
later window. Full replay also requires the original local prediction source.
No identifier or individual prediction is sent to the backend.
"""
from __future__ import annotations

import os
import sqlite3
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import structlog

from edge_agent.records import Record

log = structlog.get_logger(__name__)


class OutcomeJoin:
    """Persist bounded prediction metadata and report late/unknown outcomes honestly."""

    def __init__(self, path: Path, *, max_records: int, max_age_seconds: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        os.chmod(path, 0o600)
        self.db = sqlite3.connect(path)
        self.max_records, self.max_age_seconds = max_records, max_age_seconds
        self._seen: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._closed = False
        self.db.execute("CREATE TABLE IF NOT EXISTS predictions (model TEXT, id TEXT, "
                        "ts REAL NOT NULL, output REAL NOT NULL, label INTEGER, known_at REAL, "
                        "emitted INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(model,id))")
        self.db.execute("CREATE INDEX IF NOT EXISTS prediction_age ON predictions(ts)")
        self._retained = self.db.execute("SELECT count(*) FROM predictions").fetchone()[0]
        self.counts = {"matched": 0, "late": 0, "unknown": 0, "conflicts": 0, "evicted": 0,
                       "duplicate_predictions": 0}

    def register(self, rec: Record, now: datetime) -> bool:
        """Remember a prediction; an existing id may only repeat identical metadata."""
        if rec.prediction_id is None:
            return True
        key = (rec.model_id, rec.prediction_id)
        row = self.db.execute("SELECT ts,output,label,emitted FROM predictions WHERE model=? AND id=?",
                              (rec.model_id, rec.prediction_id)).fetchone()
        if row is not None:
            if row[0] != rec.ts.timestamp() or row[1] != rec.output or (
                    rec.outcome is not None and row[2] is not None and rec.outcome != row[2]):
                self.counts["conflicts"] += 1
                log.warning("outcome_prediction_conflict")
                raise ValueError("prediction id already belongs to a different prediction or outcome")
            if rec.outcome is not None and row[2] is None:
                self.record(rec.model_id, rec.prediction_id, rec.outcome, rec.outcome_observed_at)
            if row[3] or key in self._seen:
                self.counts["duplicate_predictions"] += 1
                return False
            # A restart must allow redelivery of predictions whose window was
            # never emitted; only this process's in-flight ids are deduplicated.
            self._remember(key)
            return True
        with self.db:
            self.db.execute("INSERT INTO predictions(model,id,ts,output,label,known_at) VALUES(?,?,?,?,?,?)",
                            (rec.model_id, rec.prediction_id, rec.ts.timestamp(), rec.output, rec.outcome,
                             rec.outcome_observed_at.timestamp() if rec.outcome_observed_at else None))
            removed = self.db.execute("DELETE FROM predictions WHERE ts < ?",
                                      (now.timestamp() - self.max_age_seconds,)).rowcount
            self._retained += 1 - removed
            overflow = max(0, self._retained - self.max_records)
            if overflow:
                pruned = self.db.execute("DELETE FROM predictions WHERE rowid IN (SELECT rowid FROM "
                                         "predictions ORDER BY ts, rowid LIMIT ?)", (overflow,)).rowcount
                removed += pruned
                self._retained -= pruned
            self.counts["evicted"] += removed
        self._remember(key)
        return True

    def _remember(self, key: tuple[str, str]) -> None:
        self._seen[key] = None
        self._seen.move_to_end(key)
        while len(self._seen) > self.max_records:
            self._seen.popitem(last=False)

    def record(self, model_id: str, prediction_id: str, outcome: int,
               observed_at: datetime) -> str:
        """Join a label to an existing prediction; return matched/late/unknown/duplicate."""
        if type(outcome) is not int or outcome not in (0, 1):
            raise ValueError("outcome must be the integer 0 or 1")
        if not isinstance(prediction_id, str) or not 1 <= len(prediction_id) <= 256:
            raise ValueError("prediction_id must be a nonempty string of at most 256 characters")
        if not isinstance(observed_at, datetime) or observed_at.tzinfo is None:
            raise ValueError("outcome observed_at must have a timezone")
        row = self.db.execute("SELECT ts,label,emitted FROM predictions WHERE model=? AND id=?",
                              (model_id, prediction_id)).fetchone()
        if row is None or observed_at.timestamp() - row[0] > self.max_age_seconds:
            self.counts["unknown"] += 1
            log.warning("outcome_prediction_unknown_or_expired")
            return "unknown"
        if observed_at.timestamp() < row[0]:
            raise ValueError("outcome cannot be observed before its prediction")
        if row[1] is not None:
            if row[1] != outcome:
                self.counts["conflicts"] += 1
                log.warning("outcome_label_conflict")
                raise ValueError("conflicting outcome for prediction")
            return "duplicate"
        with self.db:
            self.db.execute("UPDATE predictions SET label=?,known_at=? WHERE model=? AND id=?",
                            (outcome, observed_at.timestamp(), model_id, prediction_id))
        disposition = "late" if row[2] else "matched"
        self.counts[disposition] += 1
        if disposition == "late":
            log.info("outcome_requires_historical_replay")
        return disposition

    def resolve(self, rec: Record, window_end: datetime) -> Record:
        """Attach only a label for this exact prediction before its window is emitted."""
        if rec.outcome is not None:
            if rec.outcome_observed_at and rec.outcome_observed_at <= window_end:
                return rec
            return replace(rec, outcome=None, outcome_observed_at=None)
        if rec.prediction_id is None:
            return rec
        row = self.db.execute("SELECT ts,output,label,emitted,known_at FROM predictions WHERE model=? AND id=?",
                              (rec.model_id, rec.prediction_id)).fetchone()
        if (row and row[0] == rec.ts.timestamp() and row[1] == rec.output and not row[3]
                and row[4] is not None and row[4] <= window_end.timestamp()):
            return replace(rec, outcome=row[2],
                           outcome_observed_at=datetime.fromtimestamp(row[4], timezone.utc))
        return rec

    def mark_emitted(self, model: str, start: datetime, end: datetime) -> None:
        """Seal a durably queued interval; later labels never mutate its scoring."""
        with self.db:
            self.db.execute("UPDATE predictions SET emitted=1 WHERE model=? AND ts>=? AND ts<?",
                            (model, start.timestamp(), end.timestamp()))

    def expire(self, now: datetime) -> None:
        """Remove expired metadata even while the prediction stream is idle."""
        with self.db:
            removed = self.db.execute("DELETE FROM predictions WHERE ts < ?",
                                      (now.timestamp() - self.max_age_seconds,)).rowcount
        self._retained -= removed
        self.counts["evicted"] += removed

    def status(self) -> dict:
        """Return aggregate join health without prediction ids, outputs or labels."""
        return {**self.counts, "retained": self._retained,
                "late_outcome_policy": "historical_replay_requires_original_local_source",
                "unknown_outcome_policy": "counted_not_buffered; retain local source for replay",
                "counter_scope": "current_process"}

    def close(self) -> None:
        """Release the local database connection."""
        if not self._closed:
            self.db.close()
            self._closed = True
