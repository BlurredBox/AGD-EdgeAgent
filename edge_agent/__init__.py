"""
edge_agent -- the partner-side producer of `z_t` (gap G1).

Runs on the PARTNER's infrastructure. It observes the monitored model's
inputs and outputs locally, computes the signal vector by importing `corpus/`,
and POSTs only the closed-schema `z_t` scalars to `/api/v1/ingest/signal`.
What leaves is per-window summary statistics: score quantiles (p1..p99 of the
model's outputs) and per-feature drift statistics under hashed feature ids.
Individual records, raw feature values, feature names, logs and prices are not
sent: see `wire.py` for the single place a payload is assembled and audited.

The only other outbound call is optional and off by default: with
`instrument_series` configured, the agent fetches public daily closes from
fred.stlouisfed.org (one request per series, which tells that third party which
series you follow; size, time and redirect limits apply, see `prices.py`).

Platform: Linux is the supported platform (systemd service or container). The
Windows service wrapper in `packaging/edge_agent/installer/windows/` is untested
and not offered; on Windows the 0600 state-file modes are not enforced.

Data flow
---------
    SourceAdapter -> Record -> EdgeAgent.ingest
        per-record tripwires (schema break, hard constraints)   [local alert]
        -> WindowCutter (epoch-aligned cadence windows)
        -> SignalComputer (corpus: harness/quality/trading/trajectory)
        -> wire.build_envelope (flatten_z_t, id re-keying, allow-list audit)
        -> DurableSpool -> Sender (SDK IngestTransport, HMAC-signed)
"""
import os

# Math-library threads capped at 2 before numpy/sklearn load (G60): the domain classifier gives the same output and
# runs 15-60x faster under a CPU limit (docs/measurements/edge-classifier-cost-2026-10-29.md). Here, in the package
# init, it precedes every edge entry point (`python -m edge_agent`, `blurredbox-edge-replay`, the scripts that import
# the package). Defaults only: a value already in the environment (agent.env, the shell) wins.
THREAD_CAP_DEFAULTS = {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
for _name, _value in THREAD_CAP_DEFAULTS.items():
    os.environ.setdefault(_name, _value)

from edge_agent.agent import EdgeAgent  # noqa: E402
from edge_agent.config import EdgeConfig
from edge_agent.records import Record

__all__ = ["EdgeAgent", "EdgeConfig", "Record"]
