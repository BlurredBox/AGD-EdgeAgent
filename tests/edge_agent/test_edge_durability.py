"""Audit 2026-11-04 F5, F7, F10: a crash, a corrupt state file or an unexpected error loses no row and counts none twice.

Row accounting uses the per-model reference state (`ref_window_sizes`: one entry per window that entered the
baseline, with its row count), which is persisted with each queued window. A lost row shows as a short window, a
double count as an extra entry.
"""
from __future__ import annotations

import errno
import json
import os
import pickle
import sqlite3

import pytest

from edge_agent import EdgeAgent
from edge_agent import checkpoint
from edge_agent.adapters import JsonlTailAdapter
from edge_agent.agent import OPEN_WINDOWS_FILE
from edge_agent.records import record_to_json
from edge_agent.tripwires import REJECTED
from tests.edge_agent.synth import CADENCE, LOCAL_MODEL, T0, Collector, make_cfg, make_stream, utc


class _Killed(BaseException):
    """SIGKILL, OOM kill or power loss: nothing after it runs."""


def _cfg(tmp_path):
    return make_cfg(tmp_path, reference_windows=200, min_rows_per_window=5)


def _append(log, recs):
    with log.open("a") as fh:
        for r in recs:
            fh.write(json.dumps(record_to_json(r)) + "\n")


def _agent(cfg, log, sink, clock):
    return EdgeAgent(cfg, [JsonlTailAdapter(log, cfg.state_dir / "source.offset")], post=sink,
                     clock=lambda: clock[0])


def _ends(*sinks):
    return [b["window_timestamp"] for s in sinks for b in s.bodies()]


def _sizes(agent):
    return agent.models[LOCAL_MODEL].signals.ref_window_sizes


# -- F5: the source offset moves only once its rows are durable -----------------------------------------
@pytest.mark.parametrize("kill", ["after_poll", "before_offset_write", "after_offset_write", "after_tick"])
def test_f5_a_kill_at_any_step_loses_no_tailed_row_and_counts_none_twice(tmp_path, monkeypatch, kill):
    cfg, log = _cfg(tmp_path), tmp_path / "model.jsonl"
    recs = make_stream(3, rows=10)                 # windows 0, 1, 2 with 10 rows each
    sink, clock = Collector(), [utc(T0 + 30)]
    a = _agent(cfg, log, sink, clock)
    _append(log, recs[:10])
    a.run_once()                                   # window 0 read, still open
    clock[0] = utc(T0 + CADENCE + 30)
    _append(log, recs[10:15])                      # window 0 closes and is queued; window 1 holds 5 rows

    real = checkpoint.write_text_atomic

    def killed_write(path, text):
        if path.name != "source.offset":
            return real(path, text)
        if kill == "after_offset_write":
            real(path, text)
        raise _Killed()
    if kill.endswith("offset_write"):
        monkeypatch.setattr(checkpoint, "write_text_atomic", killed_write)
    try:
        a.poll() if kill == "after_poll" else a.run_once()
    except _Killed:
        pass
    monkeypatch.setattr(checkpoint, "write_text_atomic", real)
    if kill == "after_offset_write":
        assert int((cfg.state_dir / "source.offset").read_text()) > 0   # window 0's rows were committed
    del a                                          # no close(), no flush

    _append(log, recs[15:])
    clock[0] = utc(T0 + 3 * CADENCE + 1 + cfg.lateness_seconds)
    sink2 = Collector()
    b = _agent(cfg, log, sink2, clock)
    b.run_once()
    assert _sizes(b) == [10, 10, 10]               # every row once: none lost, none counted twice
    assert sorted(_ends(sink, sink2)) == [utc(T0 + (w + 1) * CADENCE).isoformat() for w in range(3)]
    b.close()


def test_f5_offset_write_is_fsynced_file_then_directory(tmp_path, monkeypatch):
    log = tmp_path / "m.jsonl"
    _append(log, make_stream(1, rows=3))
    src = JsonlTailAdapter(log, tmp_path / "st" / "source.offset")
    src.poll()
    assert not (tmp_path / "st" / "source.offset").exists()        # read is not committed
    calls = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (calls.append("replace"), real_replace(a, b))[1])
    src.commit(None)
    assert calls[:3] == ["fsync", "replace", "fsync"] or os.name == "nt"           # the offset file itself
    assert int((tmp_path / "st" / "source.offset").read_text()) == log.stat().st_size


# -- F7: corrupt or incompatible per-model state ------------------------------------------------------
def _state_file(cfg):
    return cfg.state_dir / "models" / "mm-wire-0001.state"


@pytest.mark.parametrize("content", [
    pickle.dumps(None),                            # unpickles to a non-dict: was a TypeError crash
    b"",                                           # empty after a power loss: was a silent cold start
    pickle.dumps({"n_ref_windows": 3})[:9],        # truncated
    b"cno_such_module_zz\nThing\n.",               # a class that no longer exists after an upgrade
    pickle.dumps({"n_ref_windows": "three"}),      # wrong type
    pickle.dumps({"version": 99, "n_ref_windows": 3}),
], ids=["none", "empty", "truncated", "missing_class", "wrong_type", "future_version"])
def test_f7_unreadable_state_is_kept_aside_alerted_and_cold_started(tmp_path, content):
    cfg = _cfg(tmp_path)
    _state_file(cfg).parent.mkdir(parents=True)
    _state_file(cfg).write_bytes(content)
    agent = EdgeAgent(cfg, post=Collector())
    assert agent.models[LOCAL_MODEL].signals.n_ref_windows == 0
    aside = list(_state_file(cfg).parent.glob("mm-wire-0001.state.corrupt-*"))
    assert len(aside) == 1 and aside[0].read_bytes() == content     # kept, not deleted
    assert any(a.kind == REJECTED and a.model_id == LOCAL_MODEL and "cold start" in a.message
               for a in agent.alerter.history)


def test_f7_state_without_a_version_key_still_loads(tmp_path):
    cfg = _cfg(tmp_path)
    _state_file(cfg).parent.mkdir(parents=True)
    _state_file(cfg).write_bytes(pickle.dumps({"n_ref_windows": 3, "ref_window_sizes": [1, 2, 3]}))
    agent = EdgeAgent(cfg, post=Collector())
    assert agent.models[LOCAL_MODEL].signals.n_ref_windows == 3
    assert not agent.alerter.history and not list(_state_file(cfg).parent.glob("*.corrupt-*"))


def test_f7_save_fsyncs_before_replace_and_the_directory_after(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    signals = EdgeAgent(cfg, post=Collector()).models[LOCAL_MODEL].signals
    signals.n_ref_windows = 5
    calls = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (calls.append("replace"), real_replace(a, b))[1])
    signals.save()
    assert calls == ["fsync", "replace", "fsync"] or os.name == "nt"
    assert pickle.loads(_state_file(cfg).read_bytes())["version"] == 1
    signals.n_ref_windows = 0
    signals._load()
    assert signals.n_ref_windows == 5 and not signals.alerter.history


# -- F10: an unexpected error mid-poll or mid-tick holds the source; the restart replays ---------------
@pytest.mark.parametrize("fail", ["ingest_sqlite", "spool_disk_full"])
def test_f10_unexpected_error_holds_the_offset_and_the_restart_replays_every_row(tmp_path, fail):
    cfg, log = _cfg(tmp_path), tmp_path / "model.jsonl"
    recs = make_stream(2, rows=10)
    _append(log, recs)
    sink, clock = Collector(), [utc(T0 + 2 * CADENCE + 30)]
    a = _agent(cfg, log, sink, clock)
    if fail == "ingest_sqlite":
        real, n = a.outcomes.register, [0]

        def register(rec, now):
            n[0] += 1
            if n[0] == 13:                         # inside window 1, after window 0 was queued
                raise sqlite3.OperationalError("database is locked")
            return real(rec, now)
        a.outcomes.register = register
        err = sqlite3.OperationalError
    else:
        def enqueue(body, end):
            raise OSError(errno.ENOSPC, "No space left on device")
        a.sender.enqueue = enqueue
        err = OSError
    with pytest.raises(err):
        a.run_once()
    a.close(keep_open=True)                        # what run_forever's `finally` does on the way out
    assert not (cfg.state_dir / "source.offset").exists()          # no position past unprocessed rows
    assert not (cfg.state_dir / OPEN_WINDOWS_FILE).exists()         # nor kept rows the replay re-reads
    assert any(x.kind == REJECTED and "held" in x.message for x in a.alerter.history)

    sink2 = Collector()
    b = _agent(cfg, log, sink2, clock)
    b.run_once()
    assert _sizes(b) == [10, 10]
    assert sorted(_ends(sink, sink2)) == [utc(T0 + (w + 1) * CADENCE).isoformat() for w in range(2)]
    b.close()
