"""
edge_agent -- the partner-side producer of `z_t`.

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
from edge_agent.agent import EdgeAgent
from edge_agent.config import EdgeConfig
from edge_agent.records import Record

__all__ = ["EdgeAgent", "EdgeConfig", "Record"]
