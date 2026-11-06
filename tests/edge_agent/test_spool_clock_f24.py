"""Audit F24: retry backoff runs on the monotonic clock; spool expiry holds on an unconfirmed wall clock."""
from __future__ import annotations

import time

from edge_agent.transport import DurableSpool, Sender
from edge_agent.tripwires import REJECTED, LocalAlerter
from tests.edge_agent.synth import T0, Collector, make_cfg, utc

YEAR, DAY = 365 * 86400, 86400


def _sender(tmp_path, sink, wall, mono=None):
    return Sender(make_cfg(tmp_path), DurableSpool(tmp_path / "spool"), LocalAlerter(tmp_path / "alerts.jsonl"),
                  post=sink, clock=lambda: utc(wall[0]), monotonic=(lambda: mono[0]) if mono else time.monotonic)


def _put(sender, wid, end_s):
    sender.enqueue({"monitored_model_id": "mA", "window_id": wid}, utc(end_s))


def _held(sender):
    return [a for a in sender.alerter.history if a.kind == REJECTED and "kept" in a.message]


def test_backward_wall_step_does_not_freeze_retries(tmp_path):
    sink, wall, mono = Collector(), [T0], [100.0]
    s = _sender(tmp_path, sink, wall, mono)
    _put(s, "w1", T0)
    sink.raises = ConnectionError("down")
    assert s.drain() == 0
    sink.raises, wall[0], mono[0] = None, T0 - YEAR, 111.0      # wall steps back a year; 11 s really passed
    assert s.drain() == 1 and s.pending_count() == 0


def test_clock_years_ahead_at_start_keeps_the_queue_and_alerts(tmp_path):
    sink, wall = Collector(), [T0]
    first = _sender(tmp_path, sink, wall)
    _put(first, "w1", T0)
    _put(first, "w2", T0 + 60)
    wall[0] = T0 + 5 * YEAR                                      # RTC boots years ahead
    s = _sender(tmp_path, sink, wall)
    assert s.drain(force=True) == 0
    _put(s, "w3", wall[0])                                       # fresh window on the wrong clock
    sink.status = 401                                            # the backend refuses the skewed signature
    assert s.drain(force=True) == 0
    assert s.pending_count() == 3 and s.stats.expired == 0 and _held(s)


def test_old_queue_with_a_sane_clock_expires_after_a_confirmed_send(tmp_path):
    sink, wall = Collector(), [T0]
    first = _sender(tmp_path, sink, wall)
    _put(first, "w1", T0)
    _put(first, "w2", T0 + 60)
    wall[0] = T0 + 34 * DAY                                      # offline past the 33-day limit, clock correct
    s = _sender(tmp_path, sink, wall)
    assert s.drain(force=True) == 0 and s.pending_count() == 2 and _held(s)
    _put(s, "w3", wall[0])
    assert s.drain(force=True) == 1                              # accepted: the clock is confirmed
    s.drain(force=True)
    assert s.stats.expired == 2 and s.pending_count() == 0
    assert [b["window_id"] for b in sink.bodies()] == ["w3"]
