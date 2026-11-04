"""Early PROXY labels (G52), joined locally and sent as per-window aggregates.

A proxy outcome (chargeback within hours, first-payment default, intraday P&L) arrives before the full
label. It is configured as `proxy_source` (same column mapping as an `outcome_source`) plus the proxy's
own name and delay, and it is kept in its OWN sqlite file: never in the outcome join, so a proxy label
can never be resolved into a window as a full label and never reaches `outcome_labels` or the
calibration. Only counts per window leave the host, after the proxy's delay has passed:

    "proxy_source": {"type": "postgres", ..., "proxy_name": "chargeback_6h", "delay_seconds": 21600,
                     "columns": {"prediction_id": "pid", "outcome": "chargeback_flag",
                                 "outcome_observed_at": "chargeback_seen_at", "model_id": "model"}}

Shadow-only: the server stores it beside the alert as context and as an input the learned lines may read
later; it pages nothing.
"""
from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from edge_agent.mapping import ConfigError

PROXY_PATH = "/api/v1/ingest/signal/proxy-labels"
MAX_ROWS_PER_POST = 100
_NAME = re.compile(r"^[A-Za-z0-9_-]{1,48}$")


def split_proxy_specs(raw: Any) -> list[tuple[dict[str, Any], str, float]]:
    """Each block with `proxy_name` and `delay_seconds` removed from it: (source spec, name, delay)."""
    items = raw if isinstance(raw, list) else [raw]
    if not items or not all(isinstance(i, dict) for i in items):
        raise ConfigError("proxy_source must be an object (or a nonempty list of objects)")
    out, names = [], set()
    for i, spec in enumerate(items):
        where = f"proxy_source[{i}]" if len(items) > 1 else "proxy_source"
        spec = dict(spec)
        name, delay = spec.pop("proxy_name", None), spec.pop("delay_seconds", None)
        if not isinstance(name, str) or not _NAME.match(name):
            raise ConfigError(f"{where}.proxy_name must be 1-48 letters, digits, '_' or '-'")
        if name in names:
            raise ConfigError(f"{where}.proxy_name '{name}' is used twice")
        names.add(name)
        if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not 0 <= delay <= 30 * 86400:
            raise ConfigError(f"{where}.delay_seconds must be a number of seconds from 0 to 30 days: "
                              "how long after a window closes its proxy labels are counted")
        spec.setdefault("name", f"proxy_{name}"[:40])
        out.append((spec, name, float(delay)))
    return out


class ProxyTracker:
    def __init__(self, path: Path, *, max_age_seconds: float = 40 * 86400) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.close(os.open(path, os.O_CREAT | os.O_RDWR, 0o600))
        os.chmod(path, 0o600)
        self.db = sqlite3.connect(path)
        self.max_age = max_age_seconds
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS preds (model TEXT, id TEXT, ts REAL NOT NULL, PRIMARY KEY (model, id));
            CREATE TABLE IF NOT EXISTS labels (name TEXT, model TEXT, id TEXT, label INTEGER NOT NULL,
                known_at REAL NOT NULL, PRIMARY KEY (name, model, id));
            CREATE TABLE IF NOT EXISTS wins (model TEXT, start REAL, end REAL, PRIMARY KEY (model, end));
            CREATE TABLE IF NOT EXISTS sent (name TEXT, model TEXT, end REAL, PRIMARY KEY (name, model, end));
            CREATE INDEX IF NOT EXISTS preds_ts ON preds(ts);
        """)
        self.counts = {"labels": 0, "labels_unknown": 0, "labels_conflict": 0, "rows_sent": 0}

    def add_prediction(self, model: str, pid: str | None, ts: datetime) -> None:
        if pid is None:
            return
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO preds VALUES (?,?,?)", (model, pid, ts.timestamp()))

    def add_window(self, model: str, start: datetime, end: datetime) -> None:
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO wins VALUES (?,?,?)", (model, start.timestamp(), end.timestamp()))

    def record(self, name: str, model: str, pid: str, label: int, known_at: datetime) -> str:
        """Join one proxy label to a remembered prediction. A conflicting repeat is refused (ValueError)."""
        if type(label) is not int or label not in (0, 1):
            raise ValueError("proxy outcome must be the integer 0 or 1")
        if known_at.tzinfo is None:
            raise ValueError("proxy observed_at must have a timezone")
        if self.db.execute("SELECT 1 FROM preds WHERE model=? AND id=?", (model, pid)).fetchone() is None:
            self.counts["labels_unknown"] += 1
            return "unknown"
        row = self.db.execute("SELECT label FROM labels WHERE name=? AND model=? AND id=?",
                              (name, model, pid)).fetchone()
        if row is not None:
            if row[0] != label:
                self.counts["labels_conflict"] += 1
                raise ValueError("conflicting proxy outcome for prediction")
            return "duplicate"
        with self.db:
            self.db.execute("INSERT INTO labels VALUES (?,?,?,?,?)", (name, model, pid, label, known_at.timestamp()))
        self.counts["labels"] += 1
        return "matched"

    def mature(self, delays: Mapping[str, float], install_id: str, now: datetime) -> list[dict[str, Any]]:
        """Rows for windows whose proxy delay has passed and that have at least one proxy label,
        each sent once per (proxy, model, window). Counts are over labels known by `now`."""
        rows: list[dict[str, Any]] = []
        for name, delay in delays.items():
            due = self.db.execute(
                "SELECT w.model, w.start, w.end FROM wins w WHERE w.end + ? <= ? AND NOT EXISTS "
                "(SELECT 1 FROM sent s WHERE s.name=? AND s.model=w.model AND s.end=w.end) ORDER BY w.end LIMIT ?",
                (delay, now.timestamp(), name, MAX_ROWS_PER_POST)).fetchall()
            for model, start, end in due:
                agg = self.db.execute(
                    "SELECT count(*), sum(l.label), avg(l.known_at - p.ts) FROM labels l JOIN preds p "
                    "ON p.model=l.model AND p.id=l.id WHERE l.name=? AND l.model=? AND p.ts >= ? AND p.ts < ? "
                    "AND l.known_at <= ?", (name, model, start, end, now.timestamp())).fetchone()
                if not agg[0]:
                    continue                      # nothing labelled yet: stays due, nothing is invented
                rows.append({"monitored_model_id": model, "proxy_name": name,
                             "window_start": datetime.fromtimestamp(start, now.tzinfo).isoformat(),
                             "window_end": datetime.fromtimestamp(end, now.tzinfo).isoformat(),
                             "n_labelled": float(agg[0]), "n_errors": float(agg[1] or 0),
                             "delay_s": max(0.0, float(agg[2] or 0.0))})
        return rows[:MAX_ROWS_PER_POST]

    def mark_sent(self, rows: list[dict[str, Any]]) -> None:
        with self.db:
            for r in rows:
                self.db.execute("INSERT OR IGNORE INTO sent VALUES (?,?,?)", (
                    r["proxy_name"], r["monitored_model_id"], datetime.fromisoformat(r["window_end"]).timestamp()))
            self.counts["rows_sent"] += len(rows)

    def expire(self, now: datetime) -> None:
        cut = now.timestamp() - self.max_age
        with self.db:
            for table, col in (("preds", "ts"), ("labels", "known_at"), ("wins", "end"), ("sent", "end")):
                self.db.execute(f"DELETE FROM {table} WHERE {col} < ?", (cut,))

    def status(self) -> dict[str, int]:
        return dict(self.counts)

    def close(self) -> None:
        self.db.close()
