"""
edge_agent -- the partner-side producer of `z_t` (gap G1).

Runs on the PARTNER's infrastructure. It observes the monitored model's
inputs and outputs locally, pulls PUBLIC market prices itself, computes the
signal vector by importing `corpus/`, and POSTs only the closed-schema `z_t`
scalars to `/v1/ingest/signal`. Raw feature values, predictions, prices, logs
and feature names never leave: see `wire.py` for the single place a payload is
assembled and audited.

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
