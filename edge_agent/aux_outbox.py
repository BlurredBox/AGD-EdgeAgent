"""Durable outbox for the small side-channel posts (upstream events, proxy-label aggregates).

Same rules as the window spool: written (fsync) BEFORE any send; oldest first; a transport-class failure
keeps everything queued and backs off; a deterministic rejection (other 4xx) can never succeed, so the
item moves to `rejected/` and is counted instead of blocking the ones behind it. Items older than
`max_age_seconds` are dropped and counted. The windows' own spool and ordering are not touched.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from blurredbox_agd.ingest import IngestError

_RETRYABLE_4XX = {408, 425, 429}


class AuxOutbox:
    def __init__(self, root: str | Path, send: Callable[[str, dict[str, Any]], Any], *,
                 clock: Callable[[], datetime] | None = None, max_age_seconds: float = 7 * 86400,
                 max_items: int = 5000) -> None:
        self.root = Path(root)
        self.rejected = self.root / "rejected"
        self.rejected.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        self._send = send
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.max_age, self.max_items = max_age_seconds, max_items
        self._seq = 0
        self._fails = 0
        self._retry_at = 0.0
        self.counts = {"queued": 0, "sent": 0, "rejected": 0, "expired": 0, "overflow": 0, "transport_failures": 0}

    def pending(self) -> list[Path]:
        return sorted(self.root.glob("*.json"))

    def put(self, path: str, body: dict[str, Any]) -> bool:
        if len(self.pending()) >= self.max_items:
            self.counts["overflow"] += 1             # bounded disk: drop the new item, count it
            return False
        self._seq += 1
        now = self._clock()
        name = self.root / f"{int(now.timestamp() * 1000):015d}-{self._seq:06d}.json"
        tmp = name.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump({"path": path, "body": body, "queued_at": now.isoformat()}, fh, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, name)
        self.counts["queued"] += 1
        return True

    def drain(self, *, force: bool = False) -> int:
        now = self._clock()
        if not force and now.timestamp() < self._retry_at:
            return 0
        sent = 0
        for item in self.pending():
            try:
                entry = json.loads(item.read_text())
                age = (now - datetime.fromisoformat(entry["queued_at"])).total_seconds()
                path, body = entry["path"], entry["body"]
            except (OSError, ValueError, KeyError):
                os.replace(item, self.rejected / item.name)
                self.counts["rejected"] += 1
                continue
            if age > self.max_age:
                item.unlink(missing_ok=True)
                self.counts["expired"] += 1
                continue
            try:
                self._send(path, body)
            except IngestError as exc:
                errs = exc.errors
                if bool(errs) and all(c is not None and 400 <= c < 500 and c not in _RETRYABLE_4XX
                                      for _, c, _ in errs):
                    os.replace(item, self.rejected / item.name)
                    self.counts["rejected"] += 1
                    continue
                self.counts["transport_failures"] += 1
                self._fails += 1
                self._retry_at = now.timestamp() + min(300.0, 5.0 * 2 ** min(self._fails, 6))
                return sent
            item.unlink(missing_ok=True)
            sent += 1
            self.counts["sent"] += 1
            self._fails, self._retry_at = 0, 0.0
        return sent

    def status(self) -> dict[str, int]:
        return {**self.counts, "pending": len(self.pending())}
