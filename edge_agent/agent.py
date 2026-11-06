"""The edge agent: ingest -> tripwires -> windows -> z_t -> wire -> spool -> send."""
from __future__ import annotations

import dataclasses
import json
import math
import os
import signal
import threading
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from edge_agent.adapters import SourceAdapter
from edge_agent.checkpoint import write_json_atomic
from edge_agent.config import EdgeConfig
from edge_agent.featureids import FeatureIdMap
from edge_agent.prices import FredDailyAdapter, PriceAdapter, StaticPriceAdapter
from edge_agent.records import Record, record_from_json, record_to_json
from edge_agent.outcomes import OutcomeJoin
from edge_agent.aux_outbox import AuxOutbox
from edge_agent.proxy import PROXY_PATH, ProxyTracker
from edge_agent.upstream import MAX_EVENTS_PER_POST, UPSTREAM_PATH
from edge_agent.status import STATUS_FILE
from corpus import onboarding
from edge_agent.signals import ModelSignals, WindowInput
from edge_agent.transport import DurableSpool, Sender
from edge_agent.tripwires import (FEED_DOWN, FEED_RECOVERED, LABELS_MISSING, LABELS_RECOVERED, REJECTED,
                                  SCHEMA_BREAK, LocalAlert, LocalAlerter, RecordGuard)
from edge_agent.windows import ClosedWindow, WindowCutter
from edge_agent.wire import WireViolation, build_envelope, build_z_t

STATUS_INTERVAL_SECONDS = 5.0
# Windows still open when the agent stops are kept here (with their rows) and resumed after the restart, so a stop
# never closes a window before its end (the server refuses a future close: 422 invalid_window_time). Local only.
OPEN_WINDOWS_FILE = "open_windows.json"
OPEN_WINDOWS_VERSION = 1
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
    labels_seen: bool = False          # a closed window has carried at least one label (arms `labels_missing`)
    unlabelled_run: int = 0            # consecutive closed windows with label coverage 0 since then
    labels_missing: bool = False
    invalid: dict[int, dict[str, int]] = field(default_factory=dict)   # bucket -> reason -> n
    late_seen: int = 0
    future_dropped: int = 0            # rows stamped beyond cfg.future_tolerance_seconds ahead of the clock (F6)
    withheld: int = 0                  # windows sent with statistics withheld by the local quality gate
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
                 outcome_sources: list[SourceAdapter] | None = None,
                 proxy_sources: list[SourceAdapter] | None = None,
                 upstream_sources: list[Any] | None = None) -> None:
        self.cfg = cfg
        self.sources = list(sources or [])
        # kitrun6 C-03: a folder source reads a file only `min_age_seconds` after its last write, and the file holds rows
        # written before that. Closing a window by the clock at its end dropped those rows as late (journal:
        # window_rejected ... ingest_degraded; windows skipped). A source that declares such a delay holds the clock close
        # by that delay plus one window; a busy feed still closes windows at once, when rows of a later window arrive.
        delays = [float(getattr(s, "arrival_delay_seconds", 0.0) or 0.0) for s in self.sources]
        held = max(delays, default=0.0)
        self.close_lateness_seconds = max(cfg.lateness_seconds, held + cfg.cadence_seconds if held > 0 else 0.0)
        self.outcome_sources = list(outcome_sources or [])
        self.proxy_sources = list(proxy_sources or [])          # G52: tagged with .proxy_name / .proxy_delay_s
        self.upstream_sources = list(upstream_sources or [])    # G52: deploy / job / freshness status, read-only
        self.proxy_rows_rejected = 0
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
        self.proxy = (ProxyTracker(sd / "proxy.sqlite", max_age_seconds=cfg.outcome_join_max_age_seconds * 4)
                      if self.proxy_sources else None)
        self.outbox = (AuxOutbox(sd / "aux", self.sender.transport.send, clock=self._clock)
                       if self.proxy_sources or self.upstream_sources else None)
        self._emitted_file = sd / "emitted.json"
        self._emitted_end: dict[str, str] = _load_json(self._emitted_file)
        self.replayed_windows_skipped = 0
        self.required_spec = required_spec or onboarding.default_required()
        self.collection_contract = onboarding.collection_contract(cfg.output_kind, self.required_spec)
        self._open_file = sd / OPEN_WINDOWS_FILE
        self._restored_upto: dict[str, int] = {}
        self.open_rows_restored = self._restore_open_windows()

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
        if rec.ts > now + timedelta(seconds=self.cfg.future_tolerance_seconds):
            # Audit F6: windowing it would close the current window early and leave a future one open.
            m.future_dropped += 1
            self.alerter.raise_alert(SCHEMA_BREAK, rec.model_id,
                                     "record timestamped in the future (clock or time zone); dropped",
                                     key="future_ts")
            return
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
        if rec.arrived_at is None and self.cfg.arrival_stamp == "ingest":
            rec = dataclasses.replace(rec, arrived_at=now)      # G53: local arrival stamp, never sent
        if not self.outcomes.register(rec, now):
            return
        if self.proxy is not None:
            self.proxy.add_prediction(rec.model_id, rec.prediction_id, rec.ts)
        for win in m.cutter.add(rec, self.cfg.lateness_seconds):
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
        for source in self.outcome_sources:       # a label is in the join (sqlite) once record_outcome returns
            source.commit(None)
        self._poll_proxy()
        self._poll_upstream()
        return n

    def _poll_proxy(self) -> None:
        """Proxy labels join in their OWN store (never `self.outcomes`): they cannot be resolved into a
        window as a full label."""
        for source in self.proxy_sources:
            for rec in source.poll():
                try:
                    if rec.outcome_observed_at > self._clock():
                        raise ValueError("proxy outcome cannot be observed in the future")
                    self.proxy.record(source.proxy_name, rec.model_id, rec.prediction_id, rec.outcome,
                                      rec.outcome_observed_at)
                except ValueError:
                    self.proxy_rows_rejected += 1
                    collection_log.warning("proxy_source_row_rejected")
            source.commit(None)

    def _poll_upstream(self) -> None:
        for source in self.upstream_sources:
            events = source.poll()
            for i in range(0, len(events), MAX_EVENTS_PER_POST):
                if not self.outbox.put(UPSTREAM_PATH, {"source": "edge", "events": events[i:i + MAX_EVENTS_PER_POST]}):
                    return                         # outbox full: do not commit, the source re-reads
            source.commit()

    def _commit_prediction_sources(self, *, everything: bool = False) -> None:
        """Tell checkpointing sources what is now safe: every record whose window is closed
        (hence in the durable spool). Records still in an open window are replayed after a crash."""
        before = None
        if not everything:
            starts = [s for s in (m.cutter.earliest_open_start() for m in self.models.values()) if s]
            before = min(starts) if starts else None
        for source in self.sources:
            try:
                source.commit(before)
            except OSError:
                collection_log.error("source_checkpoint_write_failed")   # replay covers it; never stop

    # -- clock-driven work --------------------------------------------------------
    def tick(self, now: datetime | None = None) -> None:
        """Close windows by the wall clock, check the feed, refresh prices, drain the spool."""
        now = now or self._clock()
        self.outcomes.expire(now)
        if self.proxy is not None:
            self.proxy.expire(now)
            delays = {s.proxy_name: s.proxy_delay_s for s in self.proxy_sources}
            rows = self.proxy.mature(delays, self.cfg.install_id, now)
            if rows and self.outbox.put(PROXY_PATH, {"install_id": self.cfg.install_id, "rows": rows}):
                self.proxy.mark_sent(rows)
        if self.outbox is not None:
            self.outbox.drain()
        if self._instruments:
            self.prices.refresh(sorted(self._instruments))
        for m in self.models.values():
            for win in m.cutter.tick(now, self.close_lateness_seconds):
                self._window(m, win)
            ref = m.last_seen or self.started
            if (now - ref).total_seconds() > self.cfg.feed_down_after_seconds and not m.feed_down:
                m.feed_down = True
                self.alerter.raise_alert(
                    FEED_DOWN, m.model_id,
                    f"no model records for {(now - ref).total_seconds():.0f}s "
                    f"(limit {self.cfg.feed_down_after_seconds:.0f}s)", key="down")
        self.sender.drain()
        self._commit_prediction_sources()
        self._forget_restored_once_closed()
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
            for sig in (signal.SIGTERM, signal.SIGINT, *([signal.SIGBREAK] if hasattr(signal, "SIGBREAK") else [])):
                previous[sig] = signal.signal(sig, lambda *_: stop.set())
        try:
            while not stop.is_set():
                self.run_once()
                stop.wait(interval_seconds)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            self.close(keep_open=True)          # a stop keeps open windows; they close at their end after restart

    def flush(self) -> None:
        """Close every open window now (shutdown / end of a replay)."""
        for m in self.models.values():
            for win in m.cutter.flush():
                self._window(m, win)
        self.sender.drain(force=True)

    def close(self, *, keep_open: bool = False) -> None:
        """Shut down. `keep_open` (the running service's stop): windows still open are written to the state with
        their rows and resumed by the next start, which sends each at its real end; nothing is closed early.
        Without it (end of a replay, tests): every open window is closed and queued now."""
        if keep_open:
            try:
                self.save_open_windows()
            except (TypeError, ValueError, OSError) as exc:
                # A row that cannot be kept as JSON, or a state write that fails: fall back to closing now
                # (the old behaviour) rather than losing the rows.
                collection_log.error("open_windows_keep_failed", error_type=type(exc).__name__)
                self.flush()
            else:
                # The kept rows are in open_windows.json now: commit the source offsets BEFORE the drain. The drain
                # can run past TimeoutStopSec / stop_grace_period; a SIGKILL during it must not leave the rows both
                # in the kept file and uncommitted in the source, or the restart counts them twice (pre-tag review).
                self._commit_prediction_sources(everything=True)
                self.sender.drain(force=True)
        else:
            self.flush()
            self._open_file.unlink(missing_ok=True)
        self._commit_prediction_sources(everything=True)  # open rows are in the state file or closed (no-op if done)
        for source in self.outcome_sources:
            source.commit(None)
        if self.outbox is not None:
            self.outbox.drain(force=True)
        self.write_status(force=True)
        self.sender.close()
        self.outcomes.close()
        if self.proxy is not None:
            self.proxy.close()
        for source in [*self.sources, *self.outcome_sources, *self.proxy_sources]:
            getattr(source, "close", lambda: None)()

    # -- open windows kept over a stop ------------------------------------------------
    def save_open_windows(self) -> int:
        """Write every open window (rows and per-window drop counts) to the state; returns the rows kept."""
        models, rows = {}, 0
        for k, m in self.models.items():
            wins = m.cutter.open_windows()
            if wins:
                models[k] = {"windows": [{"bucket": b, "records": [record_to_json(r) for r in recs],
                                          "invalid": m.invalid.get(b, {})} for b, recs in wins.items()]}
                rows += sum(len(v) for v in wins.values())
        if not models:
            self._open_file.unlink(missing_ok=True)
            return 0
        write_json_atomic(self._open_file, {"version": OPEN_WINDOWS_VERSION,
                                            "cadence_seconds": self.cfg.cadence_seconds, "models": models})
        collection_log.info("open_windows_kept", rows=rows)
        return rows

    def _restore_open_windows(self) -> int:
        """Resume the windows the last stop kept. A window already queued (at or before the emitted watermark)
        is not restored again, so a restart after a crash cannot count rows twice."""
        try:
            saved = json.loads(self._open_file.read_text())
        except FileNotFoundError:
            return 0
        except (OSError, ValueError):
            collection_log.error("open_windows_unreadable")
            return 0
        if not isinstance(saved, dict) or saved.get("version") != OPEN_WINDOWS_VERSION:
            collection_log.error("open_windows_unknown_version")
            return 0
        same_cadence = saved.get("cadence_seconds") == self.cfg.cadence_seconds
        n = 0
        for k, entry in (saved.get("models") or {}).items():
            m = self.models.get(k)
            if m is None:
                continue
            done = self._emitted_end.get(k)
            for w in entry.get("windows", []):
                recs = []
                for d in w.get("records", []):
                    try:
                        recs.append(record_from_json(d))
                    except (KeyError, TypeError, ValueError):
                        continue
                recs = [r for r in recs if done is None or m.cutter.bounds(m.cutter.bucket(r.ts))[1].isoformat() > done]
                n += m.cutter.restore(recs)
                if recs:
                    self._restored_upto[k] = max(self._restored_upto.get(k, -1), *(m.cutter.bucket(r.ts) for r in recs))
                if same_cadence and w.get("invalid") and recs:
                    reasons = m.invalid.setdefault(int(w["bucket"]), {})
                    for reason, c in w["invalid"].items():
                        reasons[reason] = reasons.get(reason, 0) + int(c)
        if n:
            collection_log.info("open_windows_restored", rows=n)
        return n

    def _forget_restored_once_closed(self) -> None:
        """Once every restored window has closed (and is queued), the kept file has done its job."""
        if not self._restored_upto:
            return
        for k, upto in list(self._restored_upto.items()):
            if not any(b <= upto for b in self.models[k].cutter.open_buckets()):
                del self._restored_upto[k]
        if not self._restored_upto:
            self._open_file.unlink(missing_ok=True)

    # -- one closed window ----------------------------------------------------------
    def _window(self, m: _ModelRuntime, win: ClosedWindow) -> None:
        done = self._emitted_end.get(m.model_id)
        if done is not None and win.end.isoformat() < done:
            # Source replay after a crash re-reads rows of windows that were already queued
            # (the checkpoint trails the spool). Re-computing them would count them twice in the
            # baseline; the window of the same end is still re-sent below so the spool stays whole.
            self.replayed_windows_skipped += 1
            collection_log.info("replayed_window_skipped")
            return
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
        m.signals.last_gate_failures = []
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
        self._label_tripwire(m, labelled, len(win.records))
        if z is None:
            m.signals.save()
            m.collection["disposition"] = "skipped_insufficient_usable_rows"
            m.skipped += 1
            return
        if m.signals.last_gate_failures:
            m.withheld += 1
            m.collection["statistics_withheld"] = list(m.signals.last_gate_failures)
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
        if self.proxy is not None:
            self.proxy.add_window(m.model_id, win.start, win.end)
        m.collection["disposition"] = "durably_queued"
        m.emitted += 1
        self._emitted_end[m.model_id] = win.end.isoformat()
        try:
            write_json_atomic(self._emitted_file, self._emitted_end)
        except OSError:
            collection_log.error("emitted_watermark_write_failed")     # replay protection only

    def _label_tripwire(self, m: _ModelRuntime, labelled: int, rows: int) -> None:
        """Decision 2 (2026-10-29): `labels_missing` once a model whose windows carried labels closes
        `labels_missing_after_windows` consecutive windows with label coverage 0; `labels_recovered` when a
        labelled window returns. Local only, like `feed_down`: nothing here is sent to the backend.

        A model that has never had a labelled window is not armed (many models never see labels at window
        close, and that is not an outage), and an empty window says nothing about labels (`feed_down`
        covers it)."""
        limit = self.cfg.labels_missing_after_windows
        if not limit or rows == 0:
            return
        if labelled:
            m.labels_seen, m.unlabelled_run = True, 0
            if m.labels_missing:
                m.labels_missing = False
                self.alerter.clear(LABELS_MISSING, m.model_id)
                self.alerter.raise_alert(LABELS_RECOVERED, m.model_id,
                                         "windows carry labels again", key="recovered")
            return
        if not m.labels_seen:
            return
        m.unlabelled_run += 1
        if m.unlabelled_run >= limit and not m.labels_missing:
            m.labels_missing = True
            self.alerter.raise_alert(
                LABELS_MISSING, m.model_id,
                f"label coverage 0 for {m.unlabelled_run} consecutive windows (limit {limit})",
                key="missing", windows=m.unlabelled_run)

    # -- introspection ------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "models": {
                k: {"phase": "live" if m.signals.live else "cold_start",
                    "reference_windows": m.signals.n_ref_windows,
                    "emitted": m.emitted, "skipped": m.skipped, "failed": m.failed,
                    "withheld": m.withheld, "late_rows_dropped": m.cutter.late_dropped,
                    "future_rows_dropped": m.future_dropped,
                    "feed_down": m.feed_down, "labels_missing": m.labels_missing,
                    "last_window_end": m.last_window_end,
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
            "outcome_sources": [{"bad_lines": getattr(s, "bad_lines", 0), **s.describe()}
                                for s in self.outcome_sources],
            "proxy": ({**self.proxy.status(), "rows_rejected": self.proxy_rows_rejected,
                       "sources": [{"proxy_name": s.proxy_name, "delay_seconds": s.proxy_delay_s, **s.describe()}
                                   for s in self.proxy_sources]} if self.proxy is not None else None),
            "upstream": ({"sources": [s.describe() for s in self.upstream_sources],
                          "outbox": self.outbox.status()} if self.upstream_sources else None),
            "replayed_windows_skipped": self.replayed_windows_skipped,
            "open_rows_restored": self.open_rows_restored,
            "collection_contract": self.collection_contract,
            "sources": [{"bad_lines": getattr(s, "bad_lines", 0),
                         "dropped": getattr(s, "dropped", 0), **s.describe()} for s in self.sources],
        }


def _load_json(path) -> dict[str, str]:
    try:
        d = json.loads(path.read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _fname(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:96]
