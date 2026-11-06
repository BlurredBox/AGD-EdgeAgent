"""Per-model `z_t` computation. Every formula is IMPORTED from `corpus/`.

What is reused, and how
-----------------------
* Groups A/B/E/F plus the multivariate, anomaly and novelty fields: the
  harness's own `ReplayHarness._compute_z_t`, called on a shim that carries
  only the four attributes it reads (`reference`, `numeric`, `categorical`,
  `seed`). Calling it -- instead of re-deriving the orchestration -- is what
  keeps the edge and the corpus from drifting apart. Its labelled-calibration
  block (ece/mce/brier/nll) runs on dummy labels inside `_statistics`, so
  those four are DROPPED there. `process` adds the shared calibration
  formulas back only for complete, causally known real binary outcomes.
* Quality gate: `corpus.quality.check_window`.
* Realised performance, market regime, output behaviour:
  `corpus.trading_signals.compute_trading_signals`, per instrument.
* Trajectory operators: `corpus.trajectory._SlotState` driven by the same
  per-window loop as `trajectory.expand`, so state is incremental (`expand`
  takes a whole series). `tests/edge_agent/test_trajectory_incremental.py`
  asserts the two agree exactly.
"""
from __future__ import annotations

import logging
import math
import os
import pickle
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from corpus import signal_vector as sv
from corpus import trading_signals as tr
from corpus import lead_signals as lead
from corpus import trajectory as traj
from corpus.adapters import Row
from corpus.harness import ReferenceWindow, ReplayHarness
from corpus.quality import check_window

from edge_agent.checkpoint import fsync_dir
from edge_agent.config import EdgeConfig
from edge_agent.prices import PriceAdapter
from edge_agent.reference import (ReferenceStats, build_reference, is_missing,
                                  split_columns)
from edge_agent.tripwires import REJECTED, SCHEMA_BREAK, LocalAlerter
from edge_agent.windows import ClosedWindow

log = logging.getLogger("edge_agent.signals")

# Fields the harness computes against labels the edge does not have.
_LABELLED = ("ece", "mce", "brier", "nll")
# Meaningful only when the model output is a probability.
_PROBABILITY_ONLY = ("positive_decision_rate", "predictive_entropy")
# Per-model state file layout (audit F7). Files written before the version key are version 1.
_STATE_VERSION = 1
_STATE_NULLABLE = {"reference": ReferenceWindow, "stats": ReferenceStats, "ref_output_sd": float}


@dataclass
class WindowInput:
    """A closed window after the per-record guard: only usable records remain."""
    window: ClosedWindow
    received: int                 # records seen for this window (usable + dropped)
    dropped: int                  # records the guard refused
    dropped_reasons: dict[str, int] = field(default_factory=dict)


class ModelSignals:
    def __init__(self, cfg: EdgeConfig, model_id: str, wire_model_id: str,
                 prices: PriceAdapter, alerter: LocalAlerter, state_path: Path) -> None:
        self.cfg, self.model_id, self.wire_model_id = cfg, model_id, wire_model_id
        self.prices, self.alerter, self.state_path = prices, alerter, state_path
        self.n_ref_windows = 0
        self.last_gate_failures: list[str] = []      # the last window's quality-gate failures (agent counts them)
        self.ref_rows: list[tuple[datetime, dict[str, Any]]] = []
        self.ref_rows_seen = 0
        self.ref_window_sizes: list[int] = []
        self.ref_outputs: list[float] = []
        self.ref_volumes: list[float] = []
        self.reference: ReferenceWindow | None = None
        self.stats: ReferenceStats | None = None
        self.numeric: list[str] = []
        self.categorical: list[str] = []
        self.rolling: dict[str, tr.RollingState] = {}
        self.slot_state = {s.name: traj._SlotState() for s in _LIVE_SLOTS}
        # G53 slice shift: per-slice (n, sum of outputs) over the reference windows, and the reference
        # output sd. Slice keys are partner values: local state only, never on the wire.
        self.ref_slice_acc: dict[str, list[float]] = {}
        self.ref_output_sd: float | None = None
        self._rng = random.Random(cfg.seed)  # noqa: S311 -- seeded subsampling, not security
        self._load()

    # -- persistence (local only; raw reference rows live on the partner's disk) ----
    _PERSIST = ("n_ref_windows", "ref_rows", "ref_rows_seen", "ref_window_sizes",
                "ref_outputs", "ref_volumes", "reference", "stats", "numeric",
                "categorical", "rolling", "slot_state", "ref_slice_acc", "ref_output_sd")

    def _load(self) -> None:
        try:
            with open(self.state_path, "rb") as fh:
                saved = pickle.load(fh)  # noqa: S301 -- our own 0600 file on the partner's disk
        except FileNotFoundError:
            return                                # first start
        except Exception as exc:                  # noqa: BLE001 -- truncated, corrupt, or an incompatible class
            self._cold_start(type(exc).__name__)
            return
        if not self._valid(saved):
            self._cold_start("incompatible_state")
            return
        for k in self._PERSIST:
            if k in saved:
                setattr(self, k, saved[k])

    def _valid(self, saved: Any) -> bool:
        if not isinstance(saved, dict) or saved.get("version", _STATE_VERSION) != _STATE_VERSION:
            return False
        for k in self._PERSIST:
            if k in saved and not (k in _STATE_NULLABLE and saved[k] is None) and not isinstance(
                    saved[k], _STATE_NULLABLE.get(k) or type(getattr(self, k))):
                return False
        return True

    def _cold_start(self, why: str) -> None:
        """Keep the unreadable file aside (never delete it) and say so: the reference rebuilds from scratch."""
        aside: Path | None = self.state_path.with_name(f"{self.state_path.name}.corrupt-{int(time.time())}")
        try:
            os.replace(self.state_path, aside)
        except OSError:
            aside = None
        kept = aside.name if aside else "nothing (it could not be moved)"
        log.error("model state unreadable (%s), kept as %s: cold start", why, kept)
        self.alerter.raise_alert(
            REJECTED, self.model_id, f"model state unreadable ({why}); kept as {kept}; cold start: the "
            f"reference rebuilds over {self.cfg.reference_windows} windows", key="state")

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            pickle.dump({"version": _STATE_VERSION, **{k: getattr(self, k) for k in self._PERSIST}}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.state_path)
        fsync_dir(self.state_path.parent)

    @property
    def live(self) -> bool:
        return self.reference is not None

    # -- one window -------------------------------------------------------------
    def process(self, wi: WindowInput, ops_provider: Callable[[str, ClosedWindow], dict] | None
                ) -> dict[str, Any] | None:
        """Return this window's z_t, or None when it is below the emission bar."""
        win, cfg = wi.window, self.cfg
        recs = win.records
        if len(recs) < cfg.min_rows_per_window:
            return None                       # Phase 1's minimum emission bar: no vector at all
        received = list(recs)                 # before the completeness filter: null rates need them

        if self.live:        # rows with a missing numeric value cannot enter the detectors
            usable = [r for r in recs if self._complete(r)]
            wi_missing = len(recs) - len(usable)
            if len(usable) < cfg.min_rows_per_window:
                return None
            recs = usable
        else:
            wi_missing = 0
        dropped = wi.dropped + win.late_dropped + wi_missing
        total = dropped + len(recs)
        dropped_fraction = dropped / total if total else 0.0
        z: dict[str, Any]

        gate = self._quality_gate(win, recs, dropped_fraction)
        if gate:
            z = {"schema_version": cfg.schema_version,
                 "warmup_state": "live" if self.live else "cold_start"}
        else:
            z = self._statistics(win, recs)
            z.update(self._trading(win, recs))
            # Whole-window calibration must not be estimated from a labelled
            # subset (selection bias), nor from labels known only in the future.
            if (cfg.output_kind == "probability" and recs
                    and all(r.outcome is not None and r.outcome_observed_at is not None
                            and r.outcome_observed_at <= win.end and 0 <= r.output <= 1 for r in recs)):
                calibrated = self._subsample(recs)  # same rows as the other harness statistics
                probs, labels = [r.output for r in calibrated], [r.outcome for r in calibrated]
                pairs = list(zip(probs, labels, strict=True))
                z.update({"brier": sv.brier(probs, labels), "nll": sv.nll(probs, labels),
                          "ece": sv.ece(pairs), "mce": sv.mce(pairs)})
                accuracy = sum((p >= 0.5) == bool(y) for p, y in pairs) / len(pairs)
                z.update({"realised_metric": accuracy, "ci_width": sv.ci_width(accuracy, len(pairs))})
        z["ingest_dropped_fraction"] = float(dropped_fraction)
        z["ingest_incomplete_flag"] = bool(dropped > 0 or gate)
        if self.stats and self.stats.expected_rows:
            z["row_count_vs_expected"] = sv.row_count_ratio(len(recs), self.stats.expected_rows)
        if ops_provider is not None:
            z.update(ops_provider(self.model_id, win) or {})   # optional host-ops hook
        z.update(self._lead_indicators(received))

        z.update(self._trajectory(win.index, z))

        if not self.live:
            self._accumulate_reference(win, recs)
        return z

    # -- G53 leading indicators (counts, rates, lags; corpus/lead_signals.py) ----------
    def _slice_key(self, r) -> str | None:
        v = r.features.get(self.cfg.slice_feature)
        return None if lead.is_null(v) else str(v)

    def _lead_indicators(self, recs) -> dict[str, Any]:
        cfg, st = self.cfg, self.stats
        cols = st.expected_schema if st and st.expected_schema else {c for r in recs for c in r.features}
        out: dict[str, Any] = dict(lead.null_summary([r.features for r in recs], sorted(cols),
                                                     cfg.null_rate_ceiling))
        out.update(lead.arrival_lags(
            (r.ts.timestamp(), None if r.arrived_at is None else r.arrived_at.timestamp()) for r in recs))
        if cfg.slice_feature and self.live:
            acc: dict[str, list[float]] = {}
            for r in recs:
                k = self._slice_key(r)
                if k is not None and k in self.ref_slice_acc and math.isfinite(r.output):
                    a = acc.setdefault(k, [0, 0.0])
                    a[0] += 1
                    a[1] += r.output
            window = {k: (int(n), sm / n) for k, (n, sm) in acc.items()}
            ref = {k: (int(n), sm / n) for k, (n, sm) in self.ref_slice_acc.items() if n}
            out.update(lead.slice_shift(window, ref, self.ref_output_sd, cfg.slice_min_rows,
                                        cfg.slice_shift_limit))
        return out

    # -- quality gate -----------------------------------------------------------
    def _quality_gate(self, win: ClosedWindow, recs, dropped_fraction: float) -> bool:
        rows = [r.features for r in recs]
        st = self.stats
        rep = check_window(
            rows,
            expected_schema=st.expected_schema if st else None,
            expected_dtypes=st.expected_dtypes if st else None,
            declared_bounds=self.cfg.feature_bounds or None,
            reference_categories=st.reference_categories if st else None,
            reference_missingness=st.reference_missingness if st else None,
            expected_row_count=st.expected_rows if st else None,
            timestamps=[r.ts for r in recs],
        )
        failures = list(rep.structural_failures)
        if dropped_fraction > self.cfg.max_invalid_row_fraction:
            failures.append("ingest_degraded")
        self.last_gate_failures = failures
        if failures:
            self.alerter.raise_alert(
                REJECTED, self.model_id,
                f"window {win.end.isoformat()} statistics withheld: {failures}",
                key=",".join(failures), failures=failures,
                dropped_fraction=dropped_fraction)
        return bool(failures)

    # -- harness groups ---------------------------------------------------------
    def _complete(self, r) -> bool:
        return all(not is_missing(r.features.get(c)) and math.isfinite(float(r.features[c]))
                   for c in self.numeric)

    def _subsample(self, recs):
        cap = self.cfg.max_rows_per_window
        if len(recs) <= cap:
            return list(recs)
        step = len(recs) / cap
        return [recs[int(i * step)] for i in range(cap)]

    def _statistics(self, win: ClosedWindow, recs) -> dict[str, Any]:
        cfg = self.cfg
        live = self.live
        used = self._subsample(recs)
        bucket = [Row(r.ts, r.features, 0) for r in used]       # y_true is a dummy: see _LABELLED
        scores = [r.output for r in used]
        shim = _harness_shim(cfg, self.reference, self.numeric, self.categorical)
        z = shim._compute_z_t(bucket, scores, 0, reference_established=live)
        for k in _LABELLED:
            z.pop(k, None)
            z.pop(f"{k}_valid", None)
        if cfg.output_kind == "signed":
            for k in _PROBABILITY_ONLY:
                z.pop(k, None)
                z.pop(f"{k}_valid", None)
        z["schema_version"] = cfg.schema_version
        z["warmup_state"] = "live" if live else "cold_start"
        return z

    # -- trading signals (realised performance, regime, behaviour) -------------
    def _bars(self, recs):
        out = {}
        for r in recs:
            out[id(r)] = self.prices.price_at(r.instrument, r.ts)
        return out

    def _trading(self, win: ClosedWindow, recs) -> dict[str, Any]:
        st = self.stats
        by_inst: dict[str, list] = {}
        for r in recs:
            by_inst.setdefault(r.instrument, []).append(r)
        bars = self._bars(recs)
        peers: dict[str, list[float]] = {}
        per: dict[str, dict[str, float | None]] = {}
        built: dict[str, list[dict]] = {}
        for inst, rs in by_inst.items():
            rows = []
            for r in rs:
                b = bars[id(r)]
                rows.append({"pred": r.output,
                             "price": None if b is None else b.price,
                             "volume": None if b is None else b.volume,
                             "high": None if b is None else b.high,
                             "low": None if b is None else b.low,
                             "bid": None if b is None else b.bid,
                             "ask": None if b is None else b.ask})
            built[inst] = rows
            ps = [x["price"] for x in rows]
            if all(p is not None for p in ps):
                peers[inst] = ps
            if not self.live:
                self.ref_volumes += [x["volume"] for x in rows
                                     if x["volume"] is not None and math.isfinite(x["volume"])]
        pw_hits = pw_n = 0
        for inst, rows in built.items():
            roll = self.rolling.setdefault(inst, tr.RollingState())
            prev_last = roll.hits[-1] if roll.hits else None
            sig = tr.compute_trading_signals(
                rows, peer_prices=peers if len(peers) >= 2 else None,
                ref_volume=st.ref_volume if st else None,
                ref_pred_rms=st.ref_pred_rms if st else None, rolling=roll)
            # compute_trading_signals returns all-None without prices, but the
            # three output-behaviour fields need only the model's own output.
            preds = [x["pred"] for x in rows]
            if sig["tr_turnover"] is None:
                sig["tr_turnover"] = tr.turnover(preds)
                sig["tr_signal_autocorr"] = tr.signal_autocorr(preds)
                sig["tr_output_scale_drift"] = tr.output_scale_drift(
                    preds, st.ref_pred_rms if st else None)
            per[inst] = sig
            if roll.hits and roll.hits[-1] is not prev_last:
                pw_hits += roll.hits[-1][0]
                pw_n += roll.hits[-1][1]
        out: dict[str, Any] = {}
        for f in tr.TRADING_SIGNAL_FIELDS:
            vals = [s[f] for s in per.values() if s[f] is not None and math.isfinite(s[f])]
            out[f] = float(np.mean(vals)) if vals else None
        if pw_n > 0:                                   # pair-weighted, not mean of means
            out["tr_hit_rate"] = float(pw_hits / pw_n)
            # Group D: the directional hit rate is the trading analogue of the
            # realised accuracy, with the corpus's own CI width.
            out["realised_metric"] = out["tr_hit_rate"]
            out["ci_width"] = sv.ci_width(out["tr_hit_rate"], pw_n)
        return out

    # -- trajectory operators ---------------------------------------------------
    def _trajectory(self, idx: int, z: dict[str, Any]) -> dict[str, float | None]:
        row: dict[str, float | None] = {}
        for s in _LIVE_SLOTS:
            ops = traj.FAMILY_OPERATORS[s.family]
            vals = self.slot_state[s.name].update(
                idx, traj._read(z, s), ops, traj.THRESHOLDS.get(s.name))
            for op in ops:
                if op == "time_above_threshold" and s.name not in traj.THRESHOLDS:
                    continue
                row[f"{s.name}__{op}"] = vals[op]
        return row

    # -- reference lifecycle ----------------------------------------------------
    def _accumulate_reference(self, win: ClosedWindow, recs) -> None:
        cap = self.cfg.reference_max_rows
        for r in recs:                      # seeded reservoir: bounded memory, reproducible
            self.ref_rows_seen += 1
            item = (r.ts, dict(r.features))
            if len(self.ref_rows) < cap:
                self.ref_rows.append(item)
            else:
                j = self._rng.randrange(self.ref_rows_seen)
                if j < cap:
                    self.ref_rows[j] = item
        self.ref_outputs += [r.output for r in recs if math.isfinite(r.output)]
        if self.cfg.slice_feature:
            for r in recs:
                k = self._slice_key(r)
                if k is None or not math.isfinite(r.output):
                    continue
                if k not in self.ref_slice_acc and len(self.ref_slice_acc) >= self.cfg.slice_max:
                    continue                       # bounded: further slice values are not scored
                a = self.ref_slice_acc.setdefault(k, [0, 0.0])
                a[0] += 1
                a[1] += r.output
        self.ref_outputs = self.ref_outputs[-200_000:]
        self.ref_window_sizes.append(len(recs))
        self.n_ref_windows += 1
        if self.n_ref_windows >= self.cfg.reference_windows:
            self.freeze_reference()

    def freeze_reference(self) -> bool:
        """Fit the reference ONCE and freeze it, with ref_volume and ref_pred_rms."""
        rows = self.ref_rows
        try:
            numeric, categorical = split_columns([f for _, f in rows])
            ref = build_reference(rows, f"edge:{self.wire_model_id}", self.wire_model_id,
                                  numeric, categorical)
        except Exception as exc:            # noqa: BLE001 -- stay in cold start, try again later
            self.alerter.raise_alert(
                SCHEMA_BREAK, self.model_id,
                f"reference could not be frozen yet ({type(exc).__name__}: {exc}); "
                "staying in cold_start", key="reference_build")
            return False
        cols = {c for _, f in rows for c in f}
        miss = {c: sum(1 for _, f in rows if is_missing(f.get(c))) / len(rows) for c in cols}
        none = type(None)
        dtypes = {c: ((int, float, none) if c in numeric else (str, bool, int, float, none))
                  for c in cols}
        cats = {c: frozenset(str(f[c]) for _, f in rows if f.get(c) is not None)
                for c in categorical}
        sizes = sorted(self.ref_window_sizes)
        outs = np.asarray(self.ref_outputs, dtype=float)
        rms = float(np.sqrt(np.mean(outs ** 2))) if len(outs) else None
        self.ref_output_sd = float(np.std(outs)) if len(outs) > 1 else None
        vol = float(np.mean(self.ref_volumes)) if self.ref_volumes else None
        self.stats = ReferenceStats(
            expected_schema=cols, expected_dtypes=dtypes, reference_categories=cats,
            reference_missingness=miss,
            expected_rows=sizes[len(sizes) // 2] if sizes else None,
            ref_volume=vol, ref_pred_rms=rms if rms and rms > 0 else None)
        self.numeric, self.categorical, self.reference = numeric, categorical, ref
        self.ref_rows = []                  # raw rows no longer needed
        return True


_LIVE_SLOTS = [s for s in traj.SLOTS if s.source is not None and s.name not in traj.EVT_EXCLUDED]


def _harness_shim(cfg: EdgeConfig, reference, numeric, categorical) -> ReplayHarness:
    """A ReplayHarness without a Split: `_compute_z_t` reads only these four attributes."""
    h = ReplayHarness.__new__(ReplayHarness)
    h.reference, h.numeric, h.categorical, h.seed = reference, numeric, categorical, cfg.seed
    return h
