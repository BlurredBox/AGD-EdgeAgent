"""F10: a closed window that fails computation or the wire audit is quarantined locally, not dropped."""
from __future__ import annotations

import json
import stat

import edge_agent.agent as agent_mod
from edge_agent import EdgeAgent, Record
from edge_agent.wire import WireViolation
from tests.edge_agent.synth import Collector, LOCAL_MODEL, T0, make_cfg, utc


def _agent(tmp_path):
    a = EdgeAgent(make_cfg(tmp_path), post=Collector(), clock=lambda: utc(T0 + 60))
    a.ingest(Record.make(utc(T0), LOCAL_MODEL, "i", {"x": 1}, 0.2, prediction_id="one"))
    return a


def _files(a):
    return sorted((a.cfg.state_dir / "quarantine").rglob("*.json"))


def _boom(*_a, **_k):
    raise RuntimeError("x")


def test_computation_failure_quarantines_rows_and_agent_continues(tmp_path, monkeypatch):
    a = _agent(tmp_path)
    m = a.models[LOCAL_MODEL]
    monkeypatch.setattr(m.signals, "process", _boom)
    a.flush()
    (f,) = _files(a)
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    body = json.loads(f.read_text())
    assert body["reason"] == "computation_failed" and body["error_type"] == "RuntimeError"
    assert len(body["records"]) == 1 and body["records"][0]["prediction_id"] == "one"
    assert m.collection["disposition"] == "computation_failed_rows_quarantined"
    assert str(f) in a.alerter.history[-1].message
    monkeypatch.undo()
    a.ingest(Record.make(utc(T0 + 60), LOCAL_MODEL, "i", {"x": 1}, 0.2, prediction_id="two"))
    a.flush()                                   # next window is processed, no crash
    assert m.collection["disposition"] != "computation_failed_rows_quarantined"
    a.close()


def test_wire_violation_quarantines_rows(tmp_path, monkeypatch):
    a = _agent(tmp_path)
    monkeypatch.setattr(agent_mod, "build_envelope", lambda **_k: (_ for _ in ()).throw(WireViolation("bad")))
    monkeypatch.setattr(a.models[LOCAL_MODEL].signals, "process", lambda *_a: {"n": 1})
    a.flush()
    (f,) = _files(a)
    assert json.loads(f.read_text())["reason"] == "wire_rejected"
    assert a.models[LOCAL_MODEL].collection["disposition"] == "wire_rejected_rows_quarantined"
    assert str(f) in a.alerter.history[-1].message
    a.close()


def test_quarantine_write_failure_is_reported_not_fatal(tmp_path, monkeypatch):
    a = _agent(tmp_path)
    monkeypatch.setattr(a.models[LOCAL_MODEL].signals, "process", _boom)

    def no_disk(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(agent_mod, "write_json_atomic", no_disk)
    a.flush()
    assert a.models[LOCAL_MODEL].collection["disposition"] == "computation_failed_rows_lost"
    assert "LOST" in a.alerter.history[-1].message
    a.close()
