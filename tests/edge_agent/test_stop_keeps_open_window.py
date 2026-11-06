"""A stop must not close the open window early (lane edgeimage, 2026-10-30).

Before: SIGTERM ran `close()`, which flushed every open window. The open window's end lies in the future, the server
refuses that (422 invalid_window_time), the window went to `spool/rejected/` and its rows were lost, once per stop or
restart, under systemd and in the container alike. Now the running service keeps open windows in the state
(`open_windows.json`, with their rows) and the next start closes each at its real end, once.
"""
from __future__ import annotations

import dataclasses
import json
from datetime import timedelta

import pytest

from edge_agent import EdgeAgent
from edge_agent.adapters import JsonlTailAdapter
from edge_agent.agent import OPEN_WINDOWS_FILE
from edge_agent.records import record_from_json, record_to_json
from tests.edge_agent.synth import CADENCE, LOCAL_MODEL, T0, Collector, make_cfg, make_stream, utc


def _cfg(tmp_path):
    return make_cfg(tmp_path, reference_windows=200, min_rows_per_window=5)


def _ends(sink: Collector) -> list[str]:
    return [b["window_timestamp"] for b in sink.bodies()]


def _feed(agent, clock, recs):
    for r in recs:
        clock[0] = r.ts
        agent.ingest(r)


def test_stop_keeps_the_open_window_and_the_restart_sends_it_once_at_its_end(tmp_path):
    cfg = _cfg(tmp_path)
    recs = make_stream(3, rows=10)                 # windows 0, 1, 2 with 10 rows each
    sink, clock = Collector(), [utc(T0)]
    agent = EdgeAgent(cfg, post=sink, clock=lambda: clock[0])
    _feed(agent, clock, recs[:15])                 # window 0 closes; window 1 holds 5 rows, still open
    agent.tick(clock[0])
    agent.close(keep_open=True)                    # the service's stop (SIGTERM)
    assert len(sink.requests) == 1                 # only window 0: nothing was closed early
    kept = json.loads((cfg.state_dir / OPEN_WINDOWS_FILE).read_text())
    assert [len(w["records"]) for w in kept["models"][LOCAL_MODEL]["windows"]] == [5]
    assert not list((cfg.state_dir / "spool" / "rejected").glob("*.json"))

    sink2 = Collector()
    agent2 = EdgeAgent(cfg, post=sink2, clock=lambda: clock[0])
    assert agent2.open_rows_restored == 5 and agent2.status()["open_rows_restored"] == 5
    _feed(agent2, clock, recs[15:25])              # the rest of window 1, then window 2 starts and closes 1
    agent2.tick(clock[0])                          # drains the spool (window 2 is still open)
    w1 = utc(T0 + 2 * CADENCE).isoformat()
    assert _ends(sink2) == [w1]
    assert agent2.models[LOCAL_MODEL].collection["rows"] == 10       # the 5 kept rows and the 5 new ones
    clock[0] = utc(T0 + 2 * CADENCE + 1)
    agent2.tick(clock[0])
    assert not (cfg.state_dir / OPEN_WINDOWS_FILE).exists()           # restored windows closed: file gone
    agent2.close()
    assert _ends(sink) + _ends(sink2) == [utc(T0 + (w + 1) * CADENCE).isoformat() for w in range(3)]


def test_a_long_stop_closes_the_kept_window_at_the_first_tick_with_its_real_end(tmp_path):
    cfg = _cfg(tmp_path)
    recs = make_stream(1, rows=10)
    sink, clock = Collector(), [utc(T0)]
    agent = EdgeAgent(cfg, post=sink, clock=lambda: clock[0])
    _feed(agent, clock, recs[:7])
    agent.close(keep_open=True)
    assert sink.requests == []
    clock[0] = utc(T0 + 10 * CADENCE)              # down for ten windows
    sink2 = Collector()
    agent2 = EdgeAgent(cfg, post=sink2, clock=lambda: clock[0])
    agent2.tick(clock[0])
    assert _ends(sink2) == [utc(T0 + CADENCE).isoformat()]
    assert agent2.models[LOCAL_MODEL].collection["rows"] == 7


def test_a_window_already_queued_is_not_restored_twice(tmp_path):
    """A crash after the kept window was queued but before the file was removed must not count rows twice."""
    cfg = _cfg(tmp_path)
    recs = make_stream(1, rows=10)
    clock = [utc(T0)]
    agent = EdgeAgent(cfg, post=Collector(), clock=lambda: clock[0])
    _feed(agent, clock, recs[:6])
    agent.close(keep_open=True)
    saved = (cfg.state_dir / OPEN_WINDOWS_FILE).read_text()
    agent2 = EdgeAgent(cfg, post=Collector(), clock=lambda: clock[0])
    clock[0] = utc(T0 + CADENCE + 1 + cfg.lateness_seconds)
    agent2.tick(clock[0])                          # window 0 is queued; the emitted watermark moves
    (cfg.state_dir / OPEN_WINDOWS_FILE).write_text(saved)   # as if the process died before removing the file
    sink3 = Collector()
    agent3 = EdgeAgent(cfg, post=sink3, clock=lambda: clock[0])
    assert agent3.open_rows_restored == 0
    agent3.tick(utc(T0 + 3 * CADENCE))
    assert sink3.requests == []


def test_close_without_keep_open_still_flushes_for_replays(tmp_path):
    cfg = _cfg(tmp_path)
    sink, clock = Collector(), [utc(T0)]
    agent = EdgeAgent(cfg, post=sink, clock=lambda: clock[0])
    _feed(agent, clock, make_stream(1, rows=8))
    agent.close()
    assert _ends(sink) == [utc(T0 + CADENCE).isoformat()]
    assert not (cfg.state_dir / OPEN_WINDOWS_FILE).exists()


def test_jsonl_offset_and_kept_rows_together_lose_nothing_and_repeat_nothing(tmp_path):
    """The tail adapter saves its offset as it reads; the kept file holds the rows past the last closed window."""
    cfg = _cfg(tmp_path)
    log = tmp_path / "model.jsonl"
    recs = make_stream(3, rows=10)
    with log.open("w") as fh:
        for r in recs[:14]:
            fh.write(json.dumps(record_to_json(r)) + "\n")
    sink, clock = Collector(), [utc(T0 + CADENCE + 30)]
    src = JsonlTailAdapter(log, cfg.state_dir / "source.offset")
    agent = EdgeAgent(cfg, [src], post=sink, clock=lambda: clock[0])
    agent.run_once()
    agent.close(keep_open=True)
    assert _ends(sink) == [utc(T0 + CADENCE).isoformat()]
    with log.open("a") as fh:                      # the model writes on while the agent is stopped
        for r in recs[14:]:
            fh.write(json.dumps(record_to_json(r)) + "\n")
    clock[0] = utc(T0 + 3 * CADENCE + 1 + cfg.lateness_seconds)
    sink2 = Collector()
    agent2 = EdgeAgent(cfg, [JsonlTailAdapter(log, cfg.state_dir / "source.offset")], post=sink2,
                       clock=lambda: clock[0])
    agent2.run_once()
    assert _ends(sink2) == [utc(T0 + (w + 1) * CADENCE).isoformat() for w in (1, 2)]
    assert agent2.status()["models"][LOCAL_MODEL]["emitted"] == 2
    agent2.close()


def test_record_json_round_trip_keeps_every_field(tmp_path):
    r = make_stream(1, rows=1)[0]
    assert record_from_json(json.loads(json.dumps(record_to_json(r)))) == r
    late = r.__class__.make(r.ts, r.model_id, r.instrument, r.features, r.output, prediction_id="p1", outcome=1,
                            outcome_observed_at=r.ts + timedelta(seconds=5), arrived_at=r.ts + timedelta(seconds=1))
    assert record_from_json(json.loads(json.dumps(record_to_json(late)))) == late


class _Killed(BaseException):
    """SIGKILL from systemd (TimeoutStopSec) or the container runtime (stop_grace_period): nothing after it runs."""


def test_a_kill_during_the_stop_drain_does_not_count_the_kept_rows_twice(tmp_path):
    """Pre-tag review (edgefix2, 2026-10-30): `close(keep_open=True)` saved the open window, then drained the spool,
    and only then committed the source offsets. A drain that ran past the stop timeout was killed after the save and
    before the commit; the restart restored the kept rows AND re-read them from the source, doubling the window.
    Driven through the real folder source and the real close path; the kill lands inside the drain."""
    from edge_agent.sources import build_sources
    from tests.edge_agent.test_source_folder import spec, write_csv

    cfg = _cfg(tmp_path)
    folder = tmp_path / "in"
    folder.mkdir()
    recs = make_stream(2, rows=40)
    write_csv(folder / "a.csv", recs[:60])        # window 0 whole, 20 rows of window 1
    clock = [utc(T0 + CADENCE + 30)]
    agent = EdgeAgent(cfg, build_sources(spec(folder), cfg.state_dir, "prediction"), post=Collector(),
                      clock=lambda: clock[0])
    agent.run_once()                              # window 0 closes and is queued; window 1 is open with 20 rows
    assert agent.models[LOCAL_MODEL].collection["rows"] == 40

    def killed(*_a, **_k):
        raise _Killed()
    agent.sender.drain = killed                   # the stop's drain runs past the timeout and is killed
    try:
        agent.close(keep_open=True)
    except _Killed:
        pass
    kept = json.loads((cfg.state_dir / OPEN_WINDOWS_FILE).read_text())
    assert [len(w["records"]) for w in kept["models"][LOCAL_MODEL]["windows"]] == [20]

    clock[0] = utc(T0 + 2 * CADENCE + 1 + cfg.lateness_seconds)
    agent2 = EdgeAgent(cfg, build_sources(spec(folder), cfg.state_dir, "prediction"), post=Collector(),
                       clock=lambda: clock[0])
    assert agent2.open_rows_restored == 20
    agent2.run_once()                             # the source must not hand the 20 kept rows over again
    assert agent2.models[LOCAL_MODEL].last_window_end == utc(T0 + 2 * CADENCE).isoformat()
    assert agent2.models[LOCAL_MODEL].collection["rows"] == 20
    agent2.close()


def test_a_future_row_is_dropped_and_never_closes_the_open_window(tmp_path):
    """Audit F6 (edge #6): one row dated a year ahead closed the open window early, the rest of its rows were then
    dropped as late, and the future window stayed open until a shutdown sent it."""
    cfg = _cfg(tmp_path)
    recs = make_stream(1, rows=10)
    sink, clock = Collector(), [utc(T0)]
    agent = EdgeAgent(cfg, post=sink, clock=lambda: clock[0])
    _feed(agent, clock, recs[:5])
    agent.ingest(dataclasses.replace(recs[0], ts=clock[0] + timedelta(days=365)))
    _feed(agent, clock, recs[5:])
    m = agent.status()["models"][LOCAL_MODEL]
    assert m["future_rows_dropped"] == 1 and m["late_rows_dropped"] == 0
    assert agent.models[LOCAL_MODEL].cutter.open_buckets() == [T0 // CADENCE]
    agent.close()
    assert _ends(sink) == [utc(T0 + CADENCE).isoformat()]
    assert agent.models[LOCAL_MODEL].collection["rows"] == 10


@pytest.mark.parametrize("lateness, kept", [(None, True), (0, False)])
def test_a_row_two_seconds_late_is_kept_under_the_default_lateness(tmp_path, lateness, kept):
    """Audit F14 (edge #14): with lateness 0 a row stamped 1 s before the boundary that arrived 2 s late was dropped.
    The default grace keeps it, whether the window would close by the clock or by a row of the next window;
    an explicit 0 is still honoured."""
    cfg = _cfg(tmp_path) if lateness is None else make_cfg(tmp_path, reference_windows=200, min_rows_per_window=5,
                                                            lateness_seconds=lateness)
    recs = make_stream(2, rows=10)
    sink, clock = Collector(), [utc(T0)]
    agent = EdgeAgent(cfg, post=sink, clock=lambda: clock[0])
    _feed(agent, clock, recs[:9])                          # window 0 without its last row
    clock[0] = utc(T0 + CADENCE + 1)
    agent.tick(clock[0])                                   # the clock has passed the boundary
    agent.ingest(dataclasses.replace(recs[10], ts=utc(T0 + CADENCE + 1)))     # a row of window 1
    agent.ingest(dataclasses.replace(recs[9], ts=utc(T0 + CADENCE - 1)))      # 2 s late
    clock[0] = utc(T0 + CADENCE + cfg.lateness_seconds + 1)
    agent.tick(clock[0])
    m = agent.status()["models"][LOCAL_MODEL]
    assert m["late_rows_dropped"] == (0 if kept else 1)
    assert agent.models[LOCAL_MODEL].collection["rows"] == (10 if kept else 9)
    agent.close()
