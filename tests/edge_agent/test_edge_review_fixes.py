"""Review fixes for the edge quality PR: durability order, lock errors, salt identity, file modes, input checks."""
from __future__ import annotations

import pytest

from edge_agent import EdgeAgent, Record
from tests.edge_agent.synth import LOCAL_MODEL, T0, Collector, make_cfg, utc


# -- review fixes ---------------------------------------------------------------------------------
def _joined(cfg) -> int:
    import sqlite3
    con = sqlite3.connect(cfg.state_dir / "outcomes.sqlite")
    try:
        return con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    finally:
        con.close()


@pytest.mark.parametrize("keep_open", [True, False])
def test_close_commits_the_join_before_the_source_offsets_advance(tmp_path, keep_open):
    cfg = make_cfg(tmp_path)
    agent = EdgeAgent(cfg, post=Collector(), clock=lambda: utc(T0 + 1))
    for i in range(5):
        agent.ingest(Record.make(utc(T0 + i * 0.1), LOCAL_MODEL, "i", {"x": 1.0}, 0.5, prediction_id=f"p{i}"))

    def killed(**_):                                     # SIGKILL as the offsets are about to move
        raise RuntimeError("killed")
    agent._commit_prediction_sources = killed
    with pytest.raises(RuntimeError):
        agent.close(keep_open=keep_open)
    assert _joined(cfg) == 5


def test_emitted_seal_is_durable_before_the_window_is_sent(tmp_path):
    cfg = make_cfg(tmp_path, min_rows_per_window=5)
    clock = [utc(T0)]
    agent = EdgeAgent(cfg, post=Collector(), clock=lambda: clock[0])
    for i in range(40):
        clock[0] = utc(T0 + i)
        agent.ingest(Record.make(utc(T0 + i), LOCAL_MODEL, "i", {"x": float(i)}, 0.5, prediction_id=f"p{i}"))
    import sqlite3
    seen = []
    real_drain = agent.sender.drain

    def drain(**kw):
        con = sqlite3.connect(cfg.state_dir / "outcomes.sqlite")
        seen.append(con.execute("SELECT count(*) FROM predictions WHERE emitted=1").fetchone()[0])
        con.close()
        return real_drain(**kw)
    agent.sender.drain = drain
    agent.tick(utc(T0 + 130))
    assert seen and seen[0] > 0
    agent.close()


@pytest.mark.parametrize("code", ["EACCES", "EIO"])
def test_unreadable_feature_id_file_stops_instead_of_minting_a_new_salt(tmp_path, monkeypatch, code):
    import errno
    from pathlib import Path
    from edge_agent.featureids import FeatureIdMap
    path = tmp_path / "feature_ids.json"
    first = FeatureIdMap(path).id_for("price")
    real = Path.read_text

    def broken(self, *a, **k):
        if self.name.startswith("feature_ids"):
            raise OSError(getattr(errno, code), code)
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", broken)
    with pytest.raises(OSError):
        FeatureIdMap(path)
    monkeypatch.undo()
    assert FeatureIdMap(path).id_for("price") == first          # untouched, same salt


def test_only_a_held_lock_means_another_agent_other_errors_are_lock_errors(tmp_path, monkeypatch):
    import errno
    import fcntl
    from edge_agent.statelock import StateDirBusy, StateLockError, acquire
    held = acquire(tmp_path)
    with pytest.raises(StateDirBusy):
        acquire(tmp_path)                                   # a real second holder
    held.close()

    def no_locks(*a):
        raise OSError(errno.ENOLCK, "No locks available")
    monkeypatch.setattr(fcntl, "flock", no_locks)
    with pytest.raises(StateLockError) as err:
        acquire(tmp_path)
    assert not isinstance(err.value, StateDirBusy) and "errno" in str(err.value)


def test_series_id_with_a_trailing_newline_is_refused(tmp_path):
    from edge_agent.prices import FredDailyAdapter
    with pytest.raises(ValueError):
        FredDailyAdapter({"I": "DGS10\n"}, tmp_path)


def test_a_cache_write_failure_sets_last_error_and_the_worker_survives(tmp_path):
    from edge_agent.prices import FredDailyAdapter
    blocker = tmp_path / "prices"
    blocker.write_text("a file where the cache directory should be")
    ad = FredDailyAdapter({"I": "SER1", "J": "SER2"}, blocker, fetch=lambda url: "DATE,X\n2026-01-02,5.0\n",
                          background=True)
    ad.refresh(["I", "J"])
    ad.join(10)
    assert ad.last_error and "Error" in ad.last_error
    assert ad._data["J"][1] == [5.0]                       # the second series was still processed


def test_trimmed_alert_file_stays_owner_only(tmp_path):
    import os
    import stat
    from edge_agent.tripwires import LocalAlerter
    path = tmp_path / "alerts.jsonl"
    path.touch(mode=0o600)
    alerter = LocalAlerter(path, cooldown_seconds=0.0)
    alerter.MAX_FILE_BYTES, alerter.KEEP_LINES = 5_000, 10
    old = os.umask(0o022)
    try:
        for i in range(200):
            alerter.raise_alert("x", "m", "message " * 5, key=str(i))
    finally:
        os.umask(old)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_empty_secret_is_a_clean_error_not_a_traceback(tmp_path):
    import json
    import subprocess
    import sys
    conf = tmp_path / "agent.json"
    conf.write_text(json.dumps({"tenant_id": "t", "install_id": "i", "models": {"a": "mm-1"}, "secret": "",
                                "endpoints": ["https://x.invalid"], "state_dir": str(tmp_path / "state"),
                                "source": {"type": "jsonl", "path": str(tmp_path / "in.jsonl")}}))
    run = subprocess.run([sys.executable, "-m", "edge_agent", "--config", str(conf)], capture_output=True,
                         text=True, timeout=120)
    assert run.returncode == 2 and "Traceback" not in run.stderr and "secret" in run.stderr
