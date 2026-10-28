"""The edge agent: ingest -> tripwires -> windows -> z_t -> wire -> spool -> send."""
from __future__ import annotations

import json
import math
import os
import signal
import threading
import time
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from edge_agent.adapters import SourceAdapter
from edge_agent.config import EdgeConfig
from edge_agent.featureids import FeatureIdMap
from edge_agent.prices import FredDailyAdapter, PriceAdapter, StaticPriceAdapter
from edge_agent.records import Record
from edge_agent.outcomes import OutcomeJoin
from edge_agent.status import STATUS_FILE
from corpus import onboarding
from edge_agent.signals import ModelSignals, WindowInput
from edge_agent.transport import DurableSpool, Sender
from edge_agent.tripwires import (FEED_DOWN, FEED_RECOVERED, REJECTED, SCHEMA_BREAK,
                                  LocalAlert, LocalAlerter, RecordGuard)
from edge_agent.windows import ClosedWindow, WindowCutter
from edge_agent.wire import WireViolation, build_envelope, build_z_t

STATUS_INTERVAL_SECONDS = 5.0
collection_log = structlog.get_logger("edge_agent.collection")

# A record with these problems cannot enter the statistics (the harness would
# raise or compute on garbage). Bound breaches are NOT dropped: an out-of-range
# input is real drift the statistics must see; it only raises the local alert.
_DROP_REASONS = {"schema_missing_columns", "dtype_change", "output_not_finite",
                 "output_out_of_range"}


@dataclass
class _ModelRuntime:
    model_id: str
    cutter: WindowCutter
    guard: RecordGuard
    signals: ModelSignals
    last_seen: datetime | None = None
    feed_down: bool = False
    invalid: dict[int, dict[str, int]] = field(default_factory=dict)   # bucket -> reason -> n
    late_seen: int = 0
    emitted: int = 0
    skipped: int = 0
    failed: int = 0
    last_window_end: str | None = None
    collection: dict[str, Any] = field(default_factory=dict)


class EdgeAgent:
    def __init__(self, cfg: EdgeConfig, sources: list[SourceAdapter] | None = None,
                 prices: PriceAdapter | None = None, *,
                 post: Callable | None = None,
                 clock: Callable[[], datetime] | None = None,
                 ops_provider: Callable[[str, ClosedWindow], dict[str, Any]] | None = None,
                 required_spec: onboarding.RequiredSpec | None = None,
                 outcome_sources: list[SourceAdapter] | None = None) -> None:
        self.cfg = cfg
        self.sources = list(sources or [])
        self.outcome_sources = list(outcome_sources or [])
        self.outcome_rows_rejected = 0
        self._status_at: datetime | None = None
        self.prediction_rows_rejected = 0
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.ops_provider = ops_provider            # optional host-ops (`ops_*`) hook
        sd = cfg.state_dir
        sd.mkdir(parents=True, exist_ok=True)
        if prices is None:
            prices = (FredDailyAdapter(cfg.instrument_series, sd / "prices",
                                       max_staleness_days=cfg.price_max_staleness_days)
                      if cfg.instrument_series else StaticPriceAdapter({}))
        self.prices = prices
        self.alerter = LocalAlerter(sd / "alerts.jsonl", cfg.alert_cooldown_seconds)
        self.ids = FeatureIdMap(sd / "feature_ids.json")
        self.sender = Sender(cfg, DurableSpool(sd / "spool"), self.alerter,
                             post=post, clock=self._clock)
        self.models: dict[str, _ModelRuntime] = {}
        for local_id, wire_id in cfg.models.items():
            self.models[local_id] = _ModelRuntime(
                local_id,
                WindowCutter(local_id, cfg.cadence_seconds),
                RecordGuard(local_id, self.alerter, feature_bounds=cfg.feature_bounds,
                            output_bounds=cfg.output_bounds),
                ModelSignals(cfg, local_id, wire_id, self.prices, self.alerter,
                             sd / "models" / f"{_fname(wire_id)}.state"))
        self.started = self._clock()
        self.unknown_model_records = 0
        self._instruments: set[str] = set()
        self.fields_dropped_by_wire: dict[str, int] = {}
        self.outcomes = OutcomeJoin(sd / "outcomes.sqlite", max_records=cfg.outcome_join_max_records,
                                    max_age_seconds=cfg.outcome_join_max_age_seconds)
        self.required_spec = required_spec or onboarding.default_required()
        self.collection_contract = onboarding.collection_contract(cfg.output_kind, self.required_spec)

    # -- alerts -------------------------------------------------------------------
    def on_alert(self, cb: Callable[[LocalAlert], None]) -> None:
        self.alerter.subscribe(cb)

    # -- ingest -------------------------------------------------------------------
    def ingest(self, rec: Record) -> None:
        m = self.models.get(rec.model_id)
        if m is None:
            self.unknown_model_records += 1
            self.alerter.raise_alert(SCHEMA_BREAK, rec.model_id,
                                     "record for a model that is not configured; ignored",
                                     key="unknown_model")
            return
        now = self._clock()
        if rec.outcome is not None and (rec.outcome_observed_at is None or rec.outcome_observed_at > now):
            raise ValueError("inline outcome must have become known by the ingest clock")
        self._instruments.add(rec.instrument)
        m.last_seen = now
        if m.feed_down:
            m.feed_down = False
            self.alerter.clear(FEED_DOWN, m.model_id)
            self.alerter.raise_alert(FEED_RECOVERED, m.model_id, "model feed is flowing again",
                                     key="recovered")
        bad = m.guard.check(rec.features, rec.output)
        if (self.cfg.output_kind == "probability" and math.isfinite(rec.output)
                and not 0 <= rec.output <= 1):
            bad.append("output_out_of_range")
        drop = [b for b in bad if b in _DROP_REASONS]
        if drop:
            reasons = m.invalid.setdefault(m.cutter.bucket(rec.ts), {})
            for b in drop:
                reasons[b] = reasons.get(b, 0) + 1
            return
        if not self.outcomes.register(rec, now):
            return
        for win in m.cutter.add(rec):
            self._window(m, win)

    def ingest_many(self, recs: list[Record]) -> None:
        for r in recs:
            self.ingest(r)

    def record_outcome(self, model_id: str, prediction_id: str, outcome: int,
                       observed_at: datetime) -> str:
        """Join a local binary outcome; late labels require local historical replay.

        The timestamp is when the outcome became known, not when a batch was
        exported. Neither ids nor individual labels are transmitted.
        """
        if model_id not in self.models:
            raise ValueError("outcome model is not configured")
        if observed_at.tzinfo is None:
            raise ValueError("outcome observed_at must have a timezone")
        if observed_at > self._clock():
            raise ValueError("outcome observed_at cannot be in the future")
        return self.outcomes.record(model_id, prediction_id, outcome, observed_at)

    def poll(self) -> int:
        """Merge bounded source batches causally; predictions precede labels at a tie."""
        events = []
        n = 0
        for source in self.sources:
            recs = source.poll()
            n += len(recs)
            events.extend((rec.ts, 0, rec) for rec in recs)
        for source in self.outcome_sources:
            for rec in source.poll():
                events.append((rec.outcome_observed_at, 1, rec))
        for _, kind, rec in sorted(events, key=lambda event: (event[0], event[1])):
            if kind == 0:
                try:
                    self.ingest(rec)
                except ValueError:
                    self.prediction_rows_rejected += 1
                    collection_log.warning("prediction_source_row_rejected")
            else:
                try:
                    self.record_outcome(rec.model_id, rec.prediction_id, rec.outcome, rec.outcome_observed_at)
                except ValueError:
                    self.outcome_rows_rejected += 1
                    collection_log.warning("outcome_source_row_rejected")
        return n

    # -- clock-driven work --------------------------------------------------------
    def tick(self, now: datetime | None = None) -> None:
        """Close windows by the wall clock, check the feed, refresh prices, drain the spool."""
        now = now or self._clock()
        self.outcomes.expire(now)
        if self._instruments:
            self.prices.refresh(sorted(self._instruments))
        for m in self.models.values():
            for win in m.cutter.tick(now, self.cfg.lateness_seconds):
                self._window(m, win)
            ref = m.last_seen or self.started
            if (now - ref).total_seconds() > self.cfg.feed_down_after_seconds and not m.feed_down:
                m.feed_down = True
                self.alerter.raise_alert(
                    FEED_DOWN, m.model_id,
                    f"no model records for {(now - ref).total_seconds():.0f}s "
                    f"(limit {self.cfg.feed_down_after_seconds:.0f}s)", key="down")
        self.sender.drain()
        self.write_status(now)

    def write_status(self, now: datetime | None = None, *, force: bool = False) -> None:
        """Publish a counts-only snapshot for `python -m edge_agent --status` (atomic, 0600)."""
        now = now or self._clock()
        if (not force and self._status_at is not None
                and (now - self._status_at).total_seconds() < STATUS_INTERVAL_SECONDS):
            return
        self._status_at = now
        path = self.cfg.state_dir / STATUS_FILE
        tmp = path.with_suffix(".tmp")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as fh:
                json.dump({**self.status(), "written_at": now.isoformat()}, fh, sort_keys=True)
            os.replace(tmp, path)
        except OSError:
            collection_log.warning("status_snapshot_write_failed")   # never stops collection

    def run_once(self) -> None:
        self.poll()
        self.tick()

    def run_forever(self, interval_seconds: float = 1.0) -> None:       # pragma: no cover
        # SIGTERM/SIGINT only set a flag: the loop exits between iterations, so a stop never
        # interrupts a window half-built, and `close()` then flushes and sends open windows.
        stop = threading.Event()
        previous = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous[sig] = signal.signal(sig, lambda *_: stop.set())
        try:
            while not stop.is_set():
                self.run_once()
                stop.wait(interval_seconds)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            self.close()

    def flush(self) -> None:
        """Close every open window now (shutdown / end of a replay)."""
        for m in self.models.values():
            for win in m.cutter.flush():
                self._window(m, win)
        self.sender.drain(force=True)

    def close(self) -> None:
        self.flush()
        self.write_status(force=True)
        self.sender.close()
        self.outcomes.close()

    # -- one closed window ----------------------------------------------------------
    def _window(self, m: _ModelRuntime, win: ClosedWindow) -> None:
        known_before = min(win.end, self._clock())
        win.records = [self.outcomes.resolve(r, known_before) for r in win.records]
        reasons = m.invalid.pop(win.index, {})
        for stale in [b for b in m.invalid if b < win.index]:
            m.invalid.pop(stale, None)            # a bucket that never closed: do not leak counters
        win.late_dropped = m.cutter.late_dropped - m.late_seen
        m.late_seen = m.cutter.late_dropped
        dropped = sum(reasons.values())
        wi = WindowInput(win, received=len(win.records) + dropped, dropped=dropped,
                         dropped_reasons=reasons)
        try:
            z = m.signals.process(wi, self.ops_provider)
        except Exception as exc:                  # noqa: BLE001 -- one bad window must not stop the agent
            m.failed += 1
            m.collection = {"disposition": "computation_failed_requires_local_source_replay",
                            "outcome_metrics_available": False}
            collection_log.error("collection_computation_failed", error_type=type(exc).__name__)
            self.alerter.raise_alert(REJECTED, m.model_id,
                                     f"z_t computation failed ({type(exc).__name__}); local source replay required",
                                     key="compute")
            return
        m.last_window_end = win.end.isoformat()
        labelled = sum(r.outcome is not None for r in win.records)
        m.collection = {"rows": len(win.records), "labelled_rows": labelled,
                        "label_coverage": labelled / len(win.records) if win.records else 0.0,
                        "outcome_metrics_available": z is not None and all(
                            type(z.get(k)) in (int, float) and math.isfinite(z[k])
                            for k in ("brier", "nll", "ece", "mce"))}
        if z is None:
            m.signals.save()
            m.collection["disposition"] = "skipped_insufficient_usable_rows"
            m.skipped += 1
            return
        wire_z, dropped_fields = build_z_t(z, self.ids)
        spec = self.required_spec
        missing = [need.name for need in spec.needs
                   if onboarding.field_series_status(need, [wire_z])["status"] != "ok"]
        m.collection["paths"] = {
            path: {"input_ready": not set(names).intersection(missing),
                   "fields_unavailable": sorted(set(names).intersection(missing))}
            for path, names in spec.by_path.items()}
        for f in dropped_fields:
            self.fields_dropped_by_wire[f] = self.fields_dropped_by_wire.get(f, 0) + 1
        try:
            body = build_envelope(
                schema_version=self.cfg.schema_version, tenant_id=self.cfg.tenant_id,
                install_id=self.cfg.install_id,
                monitored_model_id=self.cfg.models[m.model_id],
                window_end=win.end, z_t=wire_z)
        except WireViolation as exc:
            m.failed += 1
            m.collection["disposition"] = "wire_rejected_requires_local_source_replay"
            self.alerter.raise_alert(REJECTED, m.model_id, f"payload failed the wire audit: {exc}",
                                     key="audit")
            return
        try:
            self.sender.enqueue(body, win.end)
        except OSError:
            m.failed += 1
            m.collection["disposition"] = "spool_failed_requires_local_source_replay"
            collection_log.error("collection_spool_write_failed")
            raise
        # Do not suppress prediction redelivery until a durable copy exists.
        # Checkpointing is not atomic with the spool: source retention is still
        # required for recovery from a crash between these operations.
        m.collection["disposition"] = "durably_queued_checkpoint_pending"
        try:
            m.signals.save()
        except OSError:
            m.failed += 1
            collection_log.error("collection_checkpoint_failed_payload_queued")
            raise
        self.outcomes.mark_emitted(m.model_id, win.start, win.end)
        m.collection["disposition"] = "durably_queued"
        m.emitted += 1

    # -- introspection ------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "models": {
                k: {"phase": "live" if m.signals.live else "cold_start",
                    "reference_windows": m.signals.n_ref_windows,
                    "emitted": m.emitted, "skipped": m.skipped, "failed": m.failed,
                    "feed_down": m.feed_down, "last_window_end": m.last_window_end,
                    "collection": m.collection}
                for k, m in self.models.items()},
            "queued": self.sender.pending_count(),
            "sent": self.sender.stats.sent,
            "backlog_sent": self.sender.stats.backlog_sent,
            "rejected": self.sender.stats.rejected,
            "alerts": len(self.alerter.history),
            "outcomes": self.outcomes.status(),
            "outcome_rows_rejected": self.outcome_rows_rejected,
            "prediction_rows_rejected": self.prediction_rows_rejected,
            "outcome_sources": [{"bad_lines": getattr(s, "bad_lines", 0)} for s in self.outcome_sources],
            "collection_contract": self.collection_contract,
            "sources": [{"bad_lines": getattr(s, "bad_lines", 0),
                         "dropped": getattr(s, "dropped", 0)} for s in self.sources],
        }


def _fname(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:96]
