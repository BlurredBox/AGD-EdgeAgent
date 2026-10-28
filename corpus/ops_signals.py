"""
corpus/ops_signals.py

Host-operations scalars of the closed z_t schema, and their pure formulas.

These are NOT a new family. Eleven of the fields below are the contract's Group F
names already enumerated in `corpus/trajectory.py` SLOTS and
the field table; this module promotes them from "known to the
contract" to "accepted on the wire" (`corpus.serving._BASE_SCALARS`) and adds
exactly two new scalars, `feed_staleness_s` and `host_anomaly_score`.

EXACT FIELD LIST (value field, unit/domain). Each also has `<name>_valid`
(bool), following the `inference_latency_p50_valid` pattern: valid=false and
value=None when the source is unavailable. NEVER 0 for an absent source.

    inference_latency_p50      ms          >= 0     (already on the wire)
    inference_latency_p95      ms          >= 0
    inference_latency_p99      ms          >= 0
    cpu_utilization            fraction    [0, 1]   mean over the window
    gpu_utilization            fraction    [0, 1]   mean; None without a GPU
    memory_headroom            fraction    [0, 1]   1 - used/limit, floored at 0
    request_rate               requests/s  >= 0
    timeout_rate               fraction    [0, 1]   timed-out / requests
    exception_rate             per request >= 0     unhandled exceptions / requests
    retry_rate                 per request >= 0     retried requests / requests
    queue_depth                count       >= 0     mean pending requests
    feed_staleness_s           seconds     >= 0     NEW. Seconds since the last
                               market-data tick for the model's instruments
                               (max over the declared instruments). The edge
                               feed-freshness tripwire input.
    host_anomaly_score         unitless    finite   NEW. The host detector's
                               IsolationForest + flicker output; None while warming up.

UNIT NOTE. The original Group F field table gives cpu/gpu_utilization as percent
[0,100] and memory_headroom as bytes under a 2GB ceiling. This module
(2026-10-14) fixes the WIRE units as fractions and a
headroom fraction. The original table text is therefore stale for those three
fields until amended; an edge agent must follow THIS list.

WHAT THESE ARE FOR. No corpus carries any of them (the replay harness has no
real host telemetry), so no forecaster was ever fitted on them and
`ForecasterBundle.predict` reports them in `fields_ignored`. They feed the
detector bank and edge tripwires ONLY. They support no forecast-skill claim.

WHERE THIS RUNS. Edge agent, per window, over scalars the host already holds.
No row content, feature name or customer identifier is read or returned (non-content rule).

FORMULAS (pure; no clock reads, `now` is passed in -- no lookahead)
  percentile   linear interpolation at rank p*(n-1); identical to the shared
               conformance reference (`group_f_latency_pct`).
  staleness    max over instruments of max(0, now - last_tick_ts); None if no
               instrument is declared or any declared one has never ticked
               (unknown member => unknown max). A tick stamped after `now`
               is fresh (0), not negative.
  rates        count / n_requests, None if n_requests <= 0.
  request_rate n_requests / window_seconds, None if window_seconds <= 0.
  headroom     max(0, 1 - used/limit), None if limit <= 0.
  cpu/gpu/queue mean of the window's samples; None if no sample.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

__all__ = [
    "OPS_VALUE_FIELDS", "OPS_SIGNAL_FIELDS", "OPS_BOUNDS", "OPS_UNITS",
    "latency_percentile", "latency_percentiles", "feed_staleness", "rate",
    "request_rate", "memory_headroom", "mean_sample", "compute_ops_signals",
]

OPS_VALUE_FIELDS: tuple[str, ...] = (
    "inference_latency_p50", "inference_latency_p95", "inference_latency_p99",
    "cpu_utilization", "gpu_utilization", "memory_headroom",
    "request_rate", "timeout_rate", "exception_rate", "retry_rate",
    "queue_depth", "feed_staleness_s", "host_anomaly_score",
)

# Every name this module contributes to the wire: values plus _valid flags.
OPS_SIGNAL_FIELDS: tuple[str, ...] = tuple(
    n for f in OPS_VALUE_FIELDS for n in (f, f"{f}_valid"))

OPS_UNITS: dict[str, str] = {
    "inference_latency_p50": "ms", "inference_latency_p95": "ms",
    "inference_latency_p99": "ms", "cpu_utilization": "fraction",
    "gpu_utilization": "fraction", "memory_headroom": "fraction",
    "request_rate": "requests/s", "timeout_rate": "fraction",
    "exception_rate": "per request", "retry_rate": "per request",
    "queue_depth": "count", "feed_staleness_s": "s",
    "host_anomaly_score": "unitless",
}

# Inclusive (low, high) wire bounds; None = unbounded. Enforced at ingest.
OPS_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "inference_latency_p50": (0.0, None), "inference_latency_p95": (0.0, None),
    "inference_latency_p99": (0.0, None), "cpu_utilization": (0.0, 1.0),
    "gpu_utilization": (0.0, 1.0), "memory_headroom": (0.0, 1.0),
    "request_rate": (0.0, None), "timeout_rate": (0.0, 1.0),
    "exception_rate": (0.0, None), "retry_rate": (0.0, None),
    "queue_depth": (0.0, None), "feed_staleness_s": (0.0, None),
    "host_anomaly_score": (None, None),
}


def _clean(xs: Sequence[float | None] | None) -> list[float]:
    return [float(x) for x in (xs or ())
            if x is not None and not isinstance(x, bool) and math.isfinite(x)]


def latency_percentile(latencies_ms, p: float) -> float | None:
    s = sorted(_clean(latencies_ms))
    if not s:
        return None
    rank = p * (len(s) - 1)
    lo, hi = int(math.floor(rank)), int(math.ceil(rank))
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (rank - lo)


def latency_percentiles(latencies_ms) -> dict[str, float | None]:
    return {"p50": latency_percentile(latencies_ms, 0.50),
            "p95": latency_percentile(latencies_ms, 0.95),
            "p99": latency_percentile(latencies_ms, 0.99)}


def feed_staleness(now: float | None,
                   last_tick_ts: Mapping[str, float | None] | None) -> float | None:
    """Seconds. Keys are instrument identifiers used only as dict keys and
    never returned."""
    if now is None or not last_tick_ts:
        return None
    worst = 0.0
    for ts in last_tick_ts.values():
        if ts is None or isinstance(ts, bool) or not math.isfinite(ts):
            return None
        worst = max(worst, now - ts)
    return max(0.0, worst)


def rate(count: float | None, n_requests: float | None) -> float | None:
    if count is None or n_requests is None or n_requests <= 0:
        return None
    return count / n_requests


def request_rate(n_requests: float | None, window_seconds: float | None) -> float | None:
    return rate(n_requests, window_seconds)


def memory_headroom(used: float | None, limit: float | None) -> float | None:
    if used is None or limit is None or limit <= 0:
        return None
    return max(0.0, 1.0 - used / limit)


def mean_sample(samples) -> float | None:
    v = _clean(samples)
    return sum(v) / len(v) if v else None


def compute_ops_signals(*, latencies_ms=None, cpu_samples=None, gpu_samples=None,
                        mem_used=None, mem_limit=None, n_requests=None,
                        window_seconds=None, n_timed_out=None, n_exceptions=None,
                        n_retried=None, queue_samples=None, now=None,
                        last_tick_ts=None, host_anomaly_score=None
                        ) -> dict[str, float | bool | None]:
    """All `OPS_SIGNAL_FIELDS` for one window. Uncomputable => value None and
    `_valid` False (never a fabricated 0)."""
    q = latency_percentiles(latencies_ms)
    score = (float(host_anomaly_score)
             if host_anomaly_score is not None
             and not isinstance(host_anomaly_score, bool)
             and math.isfinite(host_anomaly_score) else None)
    values = {
        "inference_latency_p50": q["p50"], "inference_latency_p95": q["p95"],
        "inference_latency_p99": q["p99"],
        "cpu_utilization": mean_sample(cpu_samples),
        "gpu_utilization": mean_sample(gpu_samples),
        "memory_headroom": memory_headroom(mem_used, mem_limit),
        "request_rate": request_rate(n_requests, window_seconds),
        "timeout_rate": rate(n_timed_out, n_requests),
        "exception_rate": rate(n_exceptions, n_requests),
        "retry_rate": rate(n_retried, n_requests),
        "queue_depth": mean_sample(queue_samples),
        "feed_staleness_s": feed_staleness(now, last_tick_ts),
        "host_anomaly_score": score,
    }
    out: dict[str, float | bool | None] = {}
    for name, v in values.items():
        out[name] = v
        out[f"{name}_valid"] = v is not None
    return out
