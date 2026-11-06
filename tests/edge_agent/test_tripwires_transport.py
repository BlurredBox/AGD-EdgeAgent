"""Local tripwires fire with the backend link DOWN; the offline queue buffers and
replays in order with the backlog flagged; deterministic refusals do not block."""
from __future__ import annotations

from datetime import timedelta

from edge_agent import EdgeAgent, Record
from tests.edge_agent.synth import (CADENCE, LOCAL_MODEL, T0, Collector, make_cfg, make_stream,
                                    utc)


def _agent(tmp_path, sink, clock, **kw):
    # Cold-start only (reference_windows large): cheap, no detectors needed.
    cfg = make_cfg(tmp_path, reference_windows=200, min_rows_per_window=5, **kw)
    return EdgeAgent(cfg, post=sink, clock=lambda: clock[0])


def _feed(agent, clock, recs):
    for r in recs:
        clock[0] = r.ts
        agent.ingest(r)


def kinds(agent):
    return [a.kind for a in agent.alerter.history]


def test_tripwires_fire_locally_while_the_backend_link_is_down(tmp_path):
    sink = Collector()
    sink.raises = ConnectionError("link down")
    clock = [utc(T0)]
    agent = _agent(tmp_path, sink, clock, feature_bounds={"beta_px_zz": (-1.0, 1.0)},
                   output_bounds=(-0.05, 0.05))
    recs = make_stream(2, rows=20)
    _feed(agent, clock, recs[:20])

    # schema break: a column disappears
    broken = {k: v for k, v in recs[20].features.items() if k != "gamma_vol_zz"}
    agent.ingest(Record.make(recs[20].ts, LOCAL_MODEL, "INSTR_SENTINEL_A", broken, 0.0))
    # hard constraints: feature out of declared bounds, output not finite / out of range
    agent.ingest(Record.make(recs[21].ts, LOCAL_MODEL, "INSTR_SENTINEL_A",
                             {**recs[21].features, "beta_px_zz": 50.0}, 0.0))
    agent.ingest(Record.make(recs[22].ts, LOCAL_MODEL, "INSTR_SENTINEL_A",
                             recs[22].features, float("nan")))
    agent.ingest(Record.make(recs[23].ts, LOCAL_MODEL, "INSTR_SENTINEL_A",
                             recs[23].features, 9.0))
    k = kinds(agent)
    assert "schema_break" in k and k.count("hard_constraint") >= 3
    assert sink.requests == []                           # nothing needed the backend

    # feed down: no records for > 3 x cadence
    agent.tick(clock[0] + timedelta(seconds=10 * CADENCE))
    assert "feed_down" in kinds(agent)
    # alerts are persisted locally and name features (they never leave)
    text = (tmp_path / "state" / "alerts.jsonl").read_text()
    assert "gamma_vol_zz" in text and "beta_px_zz" in text


def test_feed_recovers_and_unknown_model_is_ignored(tmp_path):
    sink = Collector()
    clock = [utc(T0)]
    agent = _agent(tmp_path, sink, clock)
    agent.tick(utc(T0 + 10 * CADENCE))
    assert "feed_down" in kinds(agent)
    r = make_stream(1, rows=10)[0]
    clock[0] = utc(T0 + 11 * CADENCE)
    agent.ingest(r)
    assert "feed_recovered" in kinds(agent)
    agent.ingest(Record.make(r.ts, "NOT_CONFIGURED", "X", {}, 0.0))
    assert agent.unknown_model_records == 1


def test_outage_buffers_then_replays_in_order_with_backlog_flagged(tmp_path):
    sink = Collector()
    clock = [utc(T0)]
    agent = _agent(tmp_path, sink, clock)
    recs = make_stream(6, rows=10)
    sink.raises = ConnectionError("outage")
    for w in range(6):
        _feed(agent, clock, [r for r in recs if int((r.ts.timestamp() - T0) // CADENCE) == w])
    clock[0] = utc(T0 + 7 * CADENCE)
    agent.flush()
    assert sink.requests == [] and agent.status()["queued"] == 6      # buffered, not lost
    assert "link_down" in kinds(agent)

    # link restored: replay oldest-first
    sink.raises = None
    clock[0] = utc(T0 + 20 * CADENCE)
    agent.sender.drain(force=True)
    assert agent.status()["queued"] == 0
    ends = [b["window_timestamp"] for b in sink.bodies()]
    assert ends == sorted(ends) and len(ends) == 6
    assert len({b["window_id"] for b in sink.bodies()}) == 6          # stable ids: replays dedupe
    assert agent.sender.stats.backlog_sent == 6
    assert "link_recovered" in kinds(agent)
    rec = [a for a in agent.alerter.history if a.kind == "link_recovered"][0]
    assert rec.details["backlog_replayed"] == 6


def test_resend_after_restart_uses_the_same_window_id(tmp_path):
    sink = Collector()
    clock = [utc(T0)]
    sink.raises = ConnectionError("outage")
    agent = _agent(tmp_path, sink, clock)
    _feed(agent, clock, make_stream(1, rows=10))
    clock[0] = utc(T0 + 3 * CADENCE)
    agent.flush()
    assert agent.status()["queued"] == 1
    sink2 = Collector()
    agent2 = EdgeAgent(agent.cfg, post=sink2, clock=lambda: utc(T0 + 4 * CADENCE))
    agent2.sender.drain(force=True)                      # a fresh process drains the old spool
    assert len(sink2.requests) == 1 and agent2.status()["queued"] == 0


def test_deterministic_refusal_does_not_block_the_queue(tmp_path):
    sink = Collector()
    clock = [utc(T0)]
    agent = _agent(tmp_path, sink, clock)
    sink.status = 422
    _feed(agent, clock, make_stream(2, rows=10))
    clock[0] = utc(T0 + 5 * CADENCE)
    agent.flush()
    assert agent.sender.stats.rejected == 2 and agent.status()["queued"] == 0
    assert len(list((tmp_path / "state" / "spool" / "rejected").glob("*.json"))) == 2
    assert "window_rejected" in kinds(agent)


def test_rejected_window_keeps_the_http_answer(tmp_path, caplog):
    """rc5 upgrade rehearsal 2026-10-30: 4 windows went to rejected/ with no record of the
    answer. The moved file and the log line now carry the status and reason."""
    import json
    import logging

    sink = Collector()
    clock = [utc(T0)]
    agent = _agent(tmp_path, sink, clock)
    sink.status = 409
    _feed(agent, clock, make_stream(1, rows=10))
    clock[0] = utc(T0 + 3 * CADENCE)
    with caplog.at_level(logging.WARNING, logger="edge_agent.transport"):
        agent.flush()
    files = list((tmp_path / "state" / "spool" / "rejected").glob("*.json"))
    assert len(files) == 1
    entry = json.loads(files[0].read_text())
    assert entry["body"]["window_id"]                      # the window itself is kept for replay
    (answer,) = entry["rejection"]["answers"]
    assert answer["status"] == 409 and answer["endpoint"] and answer["reason"]
    assert entry["rejection"]["at"]
    line = next(r.getMessage() for r in caplog.records if "window_rejected" in r.getMessage())
    assert "http=409" in line and entry["body"]["window_id"] in line
    alert = next(a for a in agent.alerter.history if a.kind == "window_rejected")
    assert alert.details["http_status"] == [409]


def test_5xx_is_retried_not_dropped(tmp_path):
    sink = Collector()
    clock = [utc(T0)]
    agent = _agent(tmp_path, sink, clock)
    sink.status = 503
    _feed(agent, clock, make_stream(1, rows=10))
    clock[0] = utc(T0 + 3 * CADENCE)
    agent.flush()
    assert agent.status()["queued"] == 1 and agent.sender.stats.rejected == 0


def test_below_minimum_rows_emits_nothing(tmp_path):
    sink = Collector()
    clock = [utc(T0)]
    agent = _agent(tmp_path, sink, clock)
    _feed(agent, clock, make_stream(1, rows=3))
    clock[0] = utc(T0 + 3 * CADENCE)
    agent.flush()
    assert sink.requests == [] and agent.status()["queued"] == 0


def test_ops_hook_is_filtered_to_the_wire_contract(tmp_path):
    from corpus.serving import wire_fields
    sink = Collector()
    clock = [utc(T0)]
    cfg = make_cfg(tmp_path, reference_windows=200, min_rows_per_window=5)
    agent = EdgeAgent(cfg, post=sink, clock=lambda: clock[0],
                      ops_provider=lambda m, w: {"ops_not_defined_yet": 1.0,
                                                 "unauthorized_deploy_flag": True})
    _feed(agent, clock, make_stream(1, rows=10))
    clock[0] = utc(T0 + 3 * CADENCE)
    agent.flush()
    z = sink.bodies()[0]["z_t"]
    assert set(z) <= wire_fields()
    assert "ops_not_defined_yet" not in z and "unauthorized_deploy_flag" not in z
    assert agent.fields_dropped_by_wire["ops_not_defined_yet"] == 1


# --- transport and spool audit (edge F4 F8 F9; audit 21 53 54) --------------------------------

class _Reply(Collector):
    """Answers `detail`/status per call; raises for models in `down`."""
    detail: object = {}
    down: tuple = ()

    def __call__(self, url, path, body, headers):
        if any(m.encode() in body for m in self.down):
            raise ConnectionError("down")
        self.requests.append((url, path, body, dict(headers)))
        return self.status, ({"accepted": True} if self.status < 300 else {"detail": self.detail})


def _put(agent, model, wid, end_s):
    agent.sender.enqueue({"monitored_model_id": model, "window_id": wid}, utc(T0 + end_s))


def test_auth_failure_keeps_window_and_alerts(tmp_path):
    for status in (401, 403, 404):
        sink, clock = _Reply(), [utc(T0)]
        agent = _agent(tmp_path / str(status), sink, clock)
        sink.status = status
        _put(agent, "mA", "w1", 0)
        agent.sender.drain(force=True)
        assert agent.status()["queued"] == 1 and agent.sender.stats.rejected == 0
        assert "backend_auth_failed" in kinds(agent)


def test_invalid_window_time_422_is_retried_with_alert(tmp_path):
    sink, clock = _Reply(), [utc(T0)]
    agent = _agent(tmp_path, sink, clock)
    sink.status, sink.detail = 422, "invalid_window_time"
    _put(agent, "mA", "w1", 0)
    agent.sender.drain(force=True)
    assert agent.status()["queued"] == 1 and "backend_auth_failed" in kinds(agent)


def test_one_models_failure_does_not_block_another(tmp_path):
    sink, clock = _Reply(), [utc(T0)]
    sink.down = ("mA",)
    agent = _agent(tmp_path, sink, clock)
    _put(agent, "mA", "w1", 0)
    _put(agent, "mB", "w2", 0)
    assert agent.sender.drain(force=True) == 1
    assert agent.status()["queued"] == 1
    assert b"mB" in sink.requests[0][2]


def test_incomplete_spool_file_is_quarantined_not_crashing(tmp_path):
    sink, clock = _Reply(), [utc(T0)]
    agent = _agent(tmp_path, sink, clock)
    bad = agent.sender.spool.root / "mA" / "000000000000001.json"
    bad.parent.mkdir()
    bad.write_text('{"body": {}}')
    _put(agent, "mB", "w2", 0)
    assert agent.sender.drain(force=True) == 1
    assert agent.sender.stats.quarantined == 1 and not bad.exists()
    assert len(list((agent.sender.spool.root / "quarantine").glob("*.json"))) == 1
    assert agent.sender.drain(force=True) == 0               # does not loop on it


def test_same_millisecond_windows_do_not_overwrite(tmp_path):
    sink, clock = _Reply(), [utc(T0)]
    agent = _agent(tmp_path, sink, clock)
    _put(agent, "mA", "w1", 0)
    _put(agent, "mA", "w2", 0)
    _put(agent, "mA", "w1", 0)                               # exact re-enqueue stays one file
    assert agent.status()["queued"] == 2


def test_healthy_model_does_not_reset_a_failing_models_backoff_or_signal_recovery(tmp_path):
    sink, clock = _Reply(), [utc(T0)]
    sink.down = ("mA",)
    agent = _agent(tmp_path, sink, clock)
    agent.alerter.cooldown = 0
    _put(agent, "mA", "wA", 0)
    for i in range(4):
        _put(agent, "mB", f"wB{i}", i + 1)
        clock[0] = utc(T0 + 1000 * (i + 1))
        agent.sender.drain(force=True)
        assert agent.sender._fails == {"mA": i + 1} and agent.sender._link_down
    assert "link_recovered" not in kinds(agent)
    assert agent.sender._retry_at["mA"] > clock[0].timestamp()
    sink.down = ()
    agent.sender.drain(force=True)
    assert "link_recovered" in kinds(agent) and agent.sender._fails == {}
