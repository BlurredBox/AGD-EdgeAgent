"""Signed delivery with a durable offline queue.

* Every `z_t` is written to an on-disk spool BEFORE any send attempt, so a
  crash or outage loses nothing (~33 days of offline queue).
* Delivery uses the SDK's `IngestTransport` (HMAC-signed via `_signing.
  sign_request`, fresh nonce per attempt, multi-endpoint). The spool is
  drained oldest-first per model so the server's per-(tenant, model) ordering
  rule holds on replay.
* Transport-class failures (no endpoint reachable, 5xx, 408/425/429) keep the
  window queued and back off. A deterministic rejection (other 4xx) can never
  succeed on retry, so it is moved to `rejected/` and alerted locally instead
  of blocking everything behind it.
* BACKLOG: a window delivered more than `backlog_after_seconds` after it
  closed, or after a failed attempt, is a backlog replay. The wire schema is
  closed and has no backlog field, so the flag is carried locally -- spool
  entry, `Sender.stats`, a local `link_recovered` alert -- and the server sees the window's original
  `window_timestamp`, which makes the lateness observable without a new field.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from blurredbox_agd.ingest import SIGNAL_PATH, IngestError, IngestTransport

from edge_agent.config import EdgeConfig
from edge_agent.tripwires import LINK_DOWN, LINK_RECOVERED, REJECTED, LocalAlerter

log = logging.getLogger("edge_agent.transport")

_RETRYABLE_4XX = {408, 425, 429}


class DurableSpool:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.rejected = self.root / "rejected"
        self.root.mkdir(parents=True, exist_ok=True)
        self.rejected.mkdir(parents=True, exist_ok=True)

    def put(self, body: dict[str, Any], window_end: datetime) -> Path:
        model = body["monitored_model_id"]
        d = self.root / _safe(model)
        d.mkdir(parents=True, exist_ok=True)
        name = f"{int(window_end.timestamp() * 1000):015d}.json"
        path = d / name
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump({"body": body, "end": window_end.isoformat(), "attempts": 0}, fh,
                      sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return path

    def pending(self) -> list[Path]:
        """Oldest-first within each model."""
        out: list[Path] = []
        for d in sorted(p for p in self.root.iterdir() if p.is_dir() and p != self.rejected):
            out += sorted(d.glob("*.json"))
        return out

    def read(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text())

    def bump(self, path: Path, entry: dict[str, Any]) -> None:
        entry["attempts"] = entry.get("attempts", 0) + 1
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry, sort_keys=True))
        os.replace(tmp, path)

    def reject(self, path: Path) -> None:
        os.replace(path, self.rejected / f"{path.parent.name}-{path.name}")

    def remove(self, path: Path) -> None:
        path.unlink(missing_ok=True)


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:96]


@dataclass
class SendStats:
    sent: int = 0
    backlog_sent: int = 0
    rejected: int = 0
    expired: int = 0
    transport_failures: int = 0
    last_backlog_flush: int = 0
    reasons: list[str] = field(default_factory=list)


class Sender:
    def __init__(self, cfg: EdgeConfig, spool: DurableSpool, alerter: LocalAlerter, *,
                 post: Callable | None = None, clock: Callable[[], datetime] | None = None) -> None:
        self.cfg, self.spool, self.alerter = cfg, spool, alerter
        self.transport = IngestTransport(
            cfg.endpoints, secret=cfg.secret, tenant_id=cfg.tenant_id,
            install_id=cfg.install_id, monitored_model_id="-",
            mode="failover", timeout=cfg.timeout_seconds,
            schema_version=cfg.schema_version, post=post,
            client_cert=cfg.client_cert, ca_file=cfg.ca_file)
        self.stats = SendStats()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._fails = 0
        self._retry_at = 0.0
        self._link_down = False
        self._backlog_in_run = 0

    def enqueue(self, body: dict[str, Any], window_end: datetime) -> None:
        self.spool.put(body, window_end)

    def pending_count(self) -> int:
        return len(self.spool.pending())

    def drain(self, *, force: bool = False) -> int:
        """Send what is queued, oldest first. Returns how many were accepted."""
        now = self._clock()
        if not force and now.timestamp() < self._retry_at:
            return 0
        sent = 0
        self._backlog_in_run = 0
        for path in self.spool.pending():
            try:
                entry = self.spool.read(path)
            except (OSError, ValueError):
                self.spool.reject(path)
                self.stats.rejected += 1
                continue
            end = datetime.fromisoformat(entry["end"])
            age = (now - end).total_seconds()
            if age > self.cfg.spool_max_age_seconds:
                self.spool.remove(path)
                self.stats.expired += 1
                self.alerter.raise_alert(REJECTED, entry["body"]["monitored_model_id"],
                                         "queued window expired past the offline queue limit",
                                         key="expired")
                continue
            backlog = entry.get("attempts", 0) > 0 or age > self.cfg.backlog_after_seconds
            try:
                self.transport.send(SIGNAL_PATH, entry["body"])
            except IngestError as exc:
                if self._deterministic(exc):
                    self.spool.reject(path)
                    self.stats.rejected += 1
                    self.stats.reasons.append(str(exc)[:200])
                    self.alerter.raise_alert(
                        REJECTED, entry["body"]["monitored_model_id"],
                        "backend refused a window deterministically; moved to rejected/",
                        key=entry["body"]["window_id"], error=str(exc)[:300])
                    continue
                self.spool.bump(path, entry)
                self._link_failed(entry["body"]["monitored_model_id"])
                return sent                      # keep order: nothing behind this may overtake it
            sent += 1
            self.stats.sent += 1
            if backlog:
                self.stats.backlog_sent += 1
                self._backlog_in_run += 1
            self.spool.remove(path)
        if sent or not self.spool.pending():
            self._link_ok()
        return sent

    @staticmethod
    def _deterministic(exc: IngestError) -> bool:
        errs = exc.errors
        return bool(errs) and all(
            code is not None and 400 <= code < 500 and code not in _RETRYABLE_4XX
            for _, code, _ in errs)

    def _link_failed(self, model: str) -> None:
        self.stats.transport_failures += 1
        self._fails += 1
        self._retry_at = self._clock().timestamp() + min(300.0, 5.0 * 2 ** min(self._fails, 6))
        if not self._link_down:
            self._link_down = True
            self.alerter.raise_alert(
                LINK_DOWN, "*", "backend unreachable; windows are being queued locally",
                key="link", queued=self.pending_count())

    def _link_ok(self) -> None:
        self._fails = 0
        self._retry_at = 0.0
        if self._link_down:
            self._link_down = False
            self.stats.last_backlog_flush = self._backlog_in_run
            self.alerter.clear(LINK_DOWN, "*")
            self.alerter.raise_alert(
                LINK_RECOVERED, "*",
                f"backend reachable again; replayed {self._backlog_in_run} backlog window(s)",
                key="link_up", backlog_replayed=self._backlog_in_run)

    def close(self) -> None:
        self.transport.close()
