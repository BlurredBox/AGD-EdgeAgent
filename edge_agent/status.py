"""Read-only report of an edge agent's local state directory.

Nothing here opens the network, takes a lock or writes a byte. It reports
counts, ages and dispositions only: no prediction id, label, feature name,
output value or payload is ever read into the report.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

STATUS_FILE = "status.json"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _offset(path: Path) -> int | None:
    try:
        return int(path.read_text().strip() or 0)
    except (OSError, ValueError):
        return None


def _spool(root: Path, now: float) -> dict[str, Any]:
    models: dict[str, Any] = {}
    total = 0
    if root.is_dir():
        for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "rejected"):
            files = sorted(d.glob("*.json"))
            total += len(files)
            oldest = min((f.stat().st_mtime for f in files), default=None)
            models[d.name] = {"queued": len(files),
                              "oldest_age_seconds": None if oldest is None else max(0.0, now - oldest)}
    rejected = root / "rejected"
    return {"queued": total, "models": models,
            "rejected": len(list(rejected.glob("*.json"))) if rejected.is_dir() else 0}


def _outcome_rows(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0)
        try:
            return db.execute("SELECT count(*) FROM predictions").fetchone()[0]
        finally:
            db.close()
    except sqlite3.Error:
        return None


def _alerts(path: Path) -> dict[str, Any]:
    kinds: dict[str, int] = {}
    last = None
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    a = json.loads(line)
                except ValueError:
                    continue
                kinds[str(a.get("kind"))] = kinds.get(str(a.get("kind")), 0) + 1
                last = a.get("at")
    except OSError:
        pass
    return {"by_kind": kinds, "last_at": last}


def read_status(state_dir: str | Path, *, now: float | None = None) -> dict[str, Any]:
    """Summarise `state_dir` as written by a running or stopped agent."""
    root, now = Path(state_dir), time.time() if now is None else now
    if not root.is_dir():
        return {"state_dir_exists": False}
    snap = _read_json(root / STATUS_FILE)
    written = (root / STATUS_FILE).stat().st_mtime if (root / STATUS_FILE).is_file() else None
    return {
        "state_dir_exists": True,
        "agent_snapshot": snap,
        "agent_snapshot_age_seconds": None if written is None else max(0.0, now - written),
        "spool": _spool(root / "spool", now),
        "offsets": {"source": _offset(root / "source.offset"), "outcome": _offset(root / "outcome.offset")},
        "outcome_join_rows_retained": _outcome_rows(root / "outcomes.sqlite"),
        "local_alerts": _alerts(root / "alerts.jsonl"),
        "note": "read-only local report; counts and dispositions only",
    }
