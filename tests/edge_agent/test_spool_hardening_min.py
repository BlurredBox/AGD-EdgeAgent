"""Spool: bump fsyncs, legacy-name dedup, status skips quarantine, malformed entries quarantined."""
from __future__ import annotations

import json
import os

import pytest

from edge_agent import EdgeAgent
from edge_agent.status import read_status
from edge_agent.transport import DurableSpool
from tests.edge_agent.synth import T0, Collector, make_cfg, make_stream, utc

def _agent(tmp_path, sink):
    clock = [utc(T0)]
    cfg = make_cfg(tmp_path)
    agent = EdgeAgent(cfg, post=sink, clock=lambda: clock[0])
    return cfg, agent, clock


def _queue_one(agent, clock):
    for r in make_stream(2, rows=40):
        clock[0] = r.ts
        agent.ingest(r)
    clock[0] = utc(T0 + 200)
    real = agent.sender.drain
    agent.sender.drain = lambda **k: 0
    agent.flush()
    agent.sender.drain = real
    assert agent.sender.pending_count() >= 1


def _body(model, wid="w1"):
    return {"monitored_model_id": model, "window_id": wid}


@pytest.mark.parametrize("patch", [{"end": "2026-01-01T00:00:00"}, {"attempts": "x"}, {"attempts": True}])
def test_naive_end_or_bad_attempts_quarantine_without_crash(tmp_path, patch):
    sink = Collector()
    _, agent, clock = _agent(tmp_path, sink)
    _queue_one(agent, clock)
    for p in agent.sender.spool.pending():
        e = json.loads(p.read_text())
        e.update(patch)
        p.write_text(json.dumps(e))
    agent.sender.drain(force=True)
    assert agent.sender.stats.quarantined >= 1 and agent.sender.pending_count() == 0


def test_bump_fsyncs(tmp_path, monkeypatch):
    sp = DurableSpool(tmp_path / "s")
    p = sp.put(_body("m"), utc(T0))
    calls = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    sp.bump(p, sp.read(p))
    assert calls and sp.read(p)["attempts"] == 1


def test_put_sees_legacy_ts_only_name(tmp_path):
    sp = DurableSpool(tmp_path / "s")
    p = sp.put(_body("m", "w9"), utc(T0))
    legacy = p.with_name(f"{int(utc(T0).timestamp() * 1000):015d}.json")
    p.rename(legacy)
    sp.put(_body("m", "w9"), utc(T0))
    assert len(sp.pending()) == 1


def test_status_does_not_count_quarantine_as_queued(tmp_path):
    sp = DurableSpool(tmp_path / "state" / "spool")
    sp.put(_body("m"), utc(T0))
    assert sp.quarantine_file(sp.pending()[0]).exists()
    assert read_status(tmp_path / "state")["spool"]["queued"] == 0
