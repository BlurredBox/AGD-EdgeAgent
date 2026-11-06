"""Audit F19 remainder: spool 33-day expiry and client_cert config validation."""
from __future__ import annotations

import pytest

from edge_agent import EdgeAgent
from tests.edge_agent.synth import T0, Collector, make_cfg, make_stream, utc


def test_a_window_queued_past_the_offline_limit_is_dropped_not_sent(tmp_path):
    cfg = make_cfg(tmp_path)
    assert cfg.spool_max_age_seconds == 33 * 86400
    sink, clock = Collector(), [utc(T0)]
    agent = EdgeAgent(cfg, post=sink, clock=lambda: clock[0])
    for r in make_stream(2, rows=40):
        clock[0] = r.ts
        agent.ingest(r)
    clock[0] = utc(T0 + 200)
    real = agent.sender.drain
    agent.sender.drain = lambda **k: 0               # link down: windows stay queued
    agent.flush()
    agent.sender.drain = real
    assert agent.sender.pending_count() >= 1
    sent_before = len(sink.bodies())
    clock[0] = utc(T0 + 34 * 86400)
    agent.sender.drain(force=True)
    assert agent.sender.stats.expired >= 1
    assert agent.sender.pending_count() == 0
    assert len(sink.bodies()) == sent_before
    agent.close()


@pytest.mark.parametrize("bad", [("only-one",), ("a", ""), ("a", 5), "ab"])
def test_malformed_client_cert_is_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="client_cert"):
        make_cfg(tmp_path, client_cert=bad)
