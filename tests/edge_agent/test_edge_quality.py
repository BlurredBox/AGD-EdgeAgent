"""Audit 2026-11-04 edge quality items F15, F16, F20, F22, F23, F24, F25 (each drives the real path)."""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import pickle
import subprocess
import sys
import threading
import time

import pytest

from edge_agent import EdgeAgent, Record
from edge_agent.__main__ import load_edge_config
from edge_agent.featureids import FeatureIdMap
from edge_agent.housekeeping import REJECTED_MAX_FILES, trim_spool
from edge_agent.prices import FredDailyAdapter, _http_get_text
from edge_agent.tripwires import LocalAlerter
from tests.edge_agent.synth import LOCAL_MODEL, T0, Collector, make_cfg, utc


# -- F22 ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [
    {"max_rows_per_window": 0}, {"min_rows_per_window": 0}, {"reference_max_rows": 0},
    {"timeout_seconds": -1.0}, {"timeout_seconds": 0.0}, {"spool_max_age_seconds": 0},
    {"spool_max_age_seconds": -5}, {"lateness_seconds": -1.0}, {"models": {"a": ""}},
    {"models": {"a": 5}}, {"alert_cooldown_seconds": float("nan")},
])
def test_config_rejects_values_that_break_every_window(tmp_path, kw):
    with pytest.raises(ValueError):
        make_cfg(tmp_path, **kw)


def test_live_run_rejects_an_empty_secret_but_check_does_not_need_one():
    raw = {"tenant_id": "t", "install_id": "i", "models": {"a": "b"}, "secret": "",
           "endpoints": ["https://x.invalid"], "state_dir": "/nonexistent-state"}
    with pytest.raises(ValueError, match="secret"):
        load_edge_config(dict(raw), need_secret=True)
    assert load_edge_config(dict(raw), need_secret=False).secret == ""


def test_changed_cadence_is_alerted(tmp_path):
    EdgeAgent(make_cfg(tmp_path), post=Collector()).close()
    agent = EdgeAgent(make_cfg(tmp_path, cadence_seconds=120), post=Collector())
    assert any("cadence_seconds changed" in a.message for a in agent.alerter.history)
    agent.close()


# -- F25 ------------------------------------------------------------------------------------------
def test_feature_ids_survive_a_corrupt_file_through_the_backup(tmp_path):
    path = tmp_path / "feature_ids.json"
    before = FeatureIdMap(path).id_for("price")
    path.write_text('{"salt": "00')                      # half-written
    again = FeatureIdMap(path)
    assert again.id_for("price") == before


# -- F15 ------------------------------------------------------------------------------------------
def test_prediction_rows_commit_once_per_batch_not_per_row(tmp_path):
    agent = EdgeAgent(make_cfg(tmp_path), post=Collector(), clock=lambda: utc(T0 + 1))
    commits = []
    agent.outcomes.db.set_trace_callback(lambda sql: commits.append(sql) if sql.strip().upper() == "COMMIT" else None)
    for i in range(200):
        agent.ingest(Record.make(utc(T0 + i * 0.1), LOCAL_MODEL, "i", {"x": 1.0}, 0.5, prediction_id=f"p{i}"))
    agent.tick(utc(T0 + 1))
    assert len(commits) <= 2, len(commits)
    assert agent.outcomes.db.execute("SELECT count(*) FROM predictions").fetchone()[0] == 200
    agent.close()


def test_registered_rows_are_durable_after_the_tick(tmp_path):
    cfg = make_cfg(tmp_path)
    agent = EdgeAgent(cfg, post=Collector(), clock=lambda: utc(T0 + 1))
    agent.ingest(Record.make(utc(T0), LOCAL_MODEL, "i", {"x": 1.0}, 0.5, prediction_id="p1"))
    agent.tick(utc(T0 + 1))
    import sqlite3
    other = sqlite3.connect(cfg.state_dir / "outcomes.sqlite")
    assert other.execute("SELECT count(*) FROM predictions").fetchone()[0] == 1
    other.close()
    agent.close()


# -- F16 ------------------------------------------------------------------------------------------
def test_price_refresh_does_not_block_the_caller(tmp_path):
    release = threading.Event()

    def slow(url: str) -> str:
        release.wait(10)
        return "DATE,X\n2026-01-02,5.0\n"

    ad = FredDailyAdapter({"I": "SER1"}, tmp_path, fetch=slow, background=True)
    t0 = time.monotonic()
    ad.refresh(["I"])
    assert time.monotonic() - t0 < 1.0
    release.set()
    ad.join(5)
    assert ad._data["I"][1] == [5.0]


def test_hostile_series_id_is_refused(tmp_path):
    with pytest.raises(ValueError):
        FredDailyAdapter({"I": "../x"}, tmp_path)


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):                                    # noqa: N802
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/big")
            self.end_headers()
        else:
            body = b"x" * 3_000_000
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except OSError:
                pass

    def log_message(self, *a):
        pass


@pytest.fixture()
def local_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_fetch_has_a_size_limit_and_does_not_follow_redirects(local_server):
    with pytest.raises(ValueError, match="size limit"):
        _http_get_text(local_server + "/big")
    import httpx
    with pytest.raises(httpx.HTTPStatusError):
        _http_get_text(local_server + "/redirect")


# -- F23 ------------------------------------------------------------------------------------------
def test_second_agent_on_the_same_state_dir_exits(tmp_path):
    state = tmp_path / "state"
    src = tmp_path / "in.jsonl"
    src.write_text("")
    conf = tmp_path / "agent.json"
    conf.write_text(json.dumps({"tenant_id": "t", "install_id": "i", "models": {"a": "mm-1"}, "secret": "s" * 20,
                                "endpoints": ["https://x.invalid"], "state_dir": str(state),
                                "source": {"type": "jsonl", "path": str(src)}}))
    conf.chmod(0o600)          # an inline secret in a group/world-readable config is refused (edge F11)
    cmd = [sys.executable, "-m", "edge_agent", "--config", str(conf)]
    first = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(100):
            if (state / "agent.lock").exists():
                break
            time.sleep(0.2)
        second = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert second.returncode != 0 and "already running" in second.stderr
        assert first.poll() is None
    finally:
        first.terminate()
        first.wait(30)
    third = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)      # the lock died with the process
    try:
        time.sleep(3)
        assert third.poll() is None
    finally:
        third.terminate()
        third.wait(30)


# -- F24 ------------------------------------------------------------------------------------------
def test_rejected_dir_and_stale_tmp_files_are_trimmed(tmp_path):
    rej = tmp_path / "rejected"
    rej.mkdir()
    now = time.time()
    for i in range(REJECTED_MAX_FILES + 5):
        p = rej / f"m-{i:05d}.json"
        p.write_text("{}")
        os.utime(p, (now - 10_000 + i, now - 10_000 + i))
    (rej / "old.json").write_text("{}")
    os.utime(rej / "old.json", (now - 200 * 86400,) * 2)
    q = tmp_path / "model"
    q.mkdir()
    (q / "001.json").write_text("{}")
    (q / "002.tmp").write_text("{}")
    os.utime(q / "002.tmp", (now - 7200,) * 2)
    removed = trim_spool(tmp_path, now=now)
    assert removed == {"rejected": 6, "tmp": 1}
    assert (q / "001.json").exists() and not (q / "002.tmp").exists()


def test_alert_file_is_bounded(tmp_path):
    path = tmp_path / "alerts.jsonl"
    alerter = LocalAlerter(path, cooldown_seconds=0.0)
    alerter.MAX_FILE_BYTES, alerter.KEEP_LINES = 20_000, 50
    for i in range(500):
        alerter.raise_alert("x", "m", "message " * 5, key=str(i))
    assert path.stat().st_size < 30_000 and len(path.read_text().splitlines()) <= 200


# -- F20 ------------------------------------------------------------------------------------------
def test_onboarding_cli_needs_the_pinned_digest(tmp_path, capsys):
    from corpus import onboarding
    (tmp_path / "forecaster.pkl").write_bytes(pickle.dumps({"a": 1}))
    good = hashlib.sha256((tmp_path / "forecaster.pkl").read_bytes()).hexdigest()
    assert onboarding.main(["template", "--bundle", str(tmp_path)]) == 1
    assert onboarding.main(["template", "--bundle", str(tmp_path), "--bundle-sha256", "0" * 64]) == 1
    assert "digest" in capsys.readouterr().err
    with pytest.raises(ValueError, match="ForecasterBundle"):      # right digest, wrong kind of object
        onboarding.load_pinned_bundle(tmp_path, good)

