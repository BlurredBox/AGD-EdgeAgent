"""`edge_agent replay-rejected`: which rejected windows are resent, how, and what is left behind.

Unit half (the SDK's signed transport against a `post` stub). The real api and Postgres half is
`tests/test_edge_replay_rejected_live.py`.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

from edge_agent import EdgeAgent
from edge_agent.replay_rejected import collect, main as replay_main
from edge_agent.transport import REPLAYED_DIR
from tests.edge_agent.synth import CADENCE, T0, WIRE_MODEL, Collector, make_cfg, make_stream, utc

ROOT = Path(__file__).resolve().parents[2]


class Answers(Collector):
    """Answers each request with the next status of a script (then 202)."""

    def __init__(self, *statuses: int, detail: str = "Authentication failed. Provide a valid Bearer token."):
        super().__init__()
        self.script = list(statuses)
        self.detail = detail
        self.replayed = False

    def __call__(self, url, path, body, headers):
        self.requests.append((url, path, body, dict(headers)))
        code = self.script.pop(0) if self.script else 202
        if code >= 300:
            return code, {"detail": self.detail}
        return code, {"accepted": True, "replayed_response": self.replayed}


def _rejected_run(tmp_path, statuses, n=None):
    """A real agent sends n windows; the backend answers `statuses` in turn; refused ones land in rejected/."""
    n = n or len(statuses)
    sink = Answers(*statuses)
    clock = [utc(T0)]
    cfg = make_cfg(tmp_path, reference_windows=200, min_rows_per_window=5)
    agent = EdgeAgent(cfg, post=sink, clock=lambda: clock[0])
    for r in make_stream(n, rows=10):
        clock[0] = r.ts
        agent.ingest(r)
    clock[0] = utc(T0 + (n + 2) * CADENCE)
    agent.flush()
    return cfg, sink, agent


def _rejected(cfg) -> list[Path]:
    return sorted((cfg.state_dir / "spool" / "rejected").glob("*.json"))


def _config_file(tmp_path, cfg) -> Path:
    p = tmp_path / "agent.json"
    p.write_text(json.dumps({"tenant_id": cfg.tenant_id, "install_id": cfg.install_id, "models": cfg.models,
                             "secret_env": "EDGE_TEST_SECRET", "endpoints": cfg.endpoints,
                             "state_dir": str(cfg.state_dir), "cadence_seconds": cfg.cadence_seconds,
                             "source": {"type": "jsonl", "path": str(tmp_path / "x.jsonl")}}))
    return p


def _run(argv, *, post=None, now=None):
    out = io.StringIO()
    rc = replay_main(argv, post=post, clock=(lambda: now) if now else None, out=out)
    return rc, out.getvalue()


def test_dry_run_lists_status_and_eligibility_and_changes_nothing(tmp_path, monkeypatch):
    cfg, _sink, _agent = _rejected_run(tmp_path, [401, 403, 413, 422])
    files = _rejected(cfg)
    assert len(files) == 4
    old = json.loads(files[0].read_text())                 # simulate a pre-0.1.2 file: no recorded answer
    old.pop("rejection")
    files[0].write_text(json.dumps(old))
    before = {f.name: f.read_bytes() for f in files}
    monkeypatch.delenv("EDGE_TEST_SECRET", raising=False)   # a dry run needs no secret
    rc, text = _run(["--config", str(_config_file(tmp_path, cfg)), "--json"], now=utc(T0 + 10 * CADENCE))
    assert rc == 0
    rows = json.loads(text)["windows"]
    assert [r["status"] for r in rows] == ["unknown", [403], [413], [422]]
    assert not any(r["eligible"] for r in rows)
    assert "--include-unknown" in rows[0]["why"] and "--force-status" in rows[1]["why"]
    assert all(r["model"] == WIRE_MODEL and r["window_end"] and r["window_id"] for r in rows)
    assert {f.name: f.read_bytes() for f in _rejected(cfg)} == before
    assert not (cfg.state_dir / "spool" / REPLAYED_DIR).exists()


def test_apply_resends_401s_byte_for_byte_signed_and_moves_them_with_a_receipt(tmp_path, monkeypatch):
    cfg, sink, _agent = _rejected_run(tmp_path, [401, 401, 422])
    first_bodies = [b for _, _, b, _ in sink.requests]
    monkeypatch.setenv("EDGE_TEST_SECRET", cfg.secret)
    post = Answers()
    rc, text = _run(["--config", str(_config_file(tmp_path, cfg)), "--apply"], post=post,
                    now=utc(T0 + 10 * CADENCE))
    assert rc == 0, text
    assert [b for _, _, b, _ in post.requests] == first_bodies[:2]           # same bytes, oldest first
    assert all(h["X-AGD-Signature"] and h["X-AGD-Nonce"] for *_, h in post.requests)
    assert not {h["X-AGD-Nonce"] for *_, h in post.requests} & {h["X-AGD-Nonce"] for *_, h in sink.requests}
    left = _rejected(cfg)
    assert len(left) == 1 and json.loads(left[0].read_text())["rejection"]["answers"][0]["status"] == 422
    done = cfg.state_dir / "spool" / REPLAYED_DIR
    moved = sorted(done.glob("*.json"))
    assert len(moved) == 2
    receipt = json.loads(moved[0].read_text())["replay_receipt"]
    assert receipt["result"] == "accepted" and receipt["status_before"] == [401]
    lines = [json.loads(x) for x in (done / "receipts.jsonl").read_text().splitlines()]
    assert [x["result"] for x in lines] == ["accepted", "accepted"]
    assert "body" not in lines[0] and "z_t" not in json.dumps(lines)          # receipts carry no payload

    # the live spool never treats replayed/ as a queue
    agent2 = EdgeAgent(cfg, post=Collector(), clock=lambda: utc(T0 + 11 * CADENCE))
    assert agent2.sender.spool.pending() == [] and agent2.status()["queued"] == 0

    # a second apply finds nothing eligible and sends nothing
    again = Answers()
    rc, text = _run(["--config", str(_config_file(tmp_path, cfg)), "--apply"], post=again,
                    now=utc(T0 + 10 * CADENCE))
    assert rc == 0 and again.requests == [] and '"replayed": 0' in text


def test_unknown_needs_its_flag_and_a_forced_status_is_resent(tmp_path, monkeypatch):
    cfg, _sink, _agent = _rejected_run(tmp_path, [401, 403])
    f = _rejected(cfg)[0]
    e = json.loads(f.read_text())
    e.pop("rejection")
    f.write_text(json.dumps(e))
    monkeypatch.setenv("EDGE_TEST_SECRET", cfg.secret)
    conf = str(_config_file(tmp_path, cfg))
    now = utc(T0 + 10 * CADENCE)
    post = Answers()
    assert _run(["--config", conf, "--apply"], post=post, now=now)[0] == 0 and post.requests == []
    assert _run(["--config", conf, "--apply", "--include-unknown"], post=post, now=now)[0] == 0
    assert len(post.requests) == 1
    rc, text = _run(["--config", conf, "--apply", "--force-status", "403", "--json"], post=post, now=now)
    assert rc == 0 and len(post.requests) == 2
    assert "forced (--force-status 403)" in text and _rejected(cfg) == []


def test_a_refusal_stays_in_rejected_with_the_attempt_and_is_judged_by_it_next_time(tmp_path, monkeypatch):
    cfg, _sink, _agent = _rejected_run(tmp_path, [401, 401])
    monkeypatch.setenv("EDGE_TEST_SECRET", cfg.secret)
    conf = str(_config_file(tmp_path, cfg))
    now = utc(T0 + 10 * CADENCE)
    post = Answers(422, 202, detail='{"reason": "out_of_order_window"}')
    rc, text = _run(["--config", conf, "--apply"], post=post, now=now)
    assert rc == 1 and "out_of_order_window" in text
    [left] = _rejected(cfg)
    e = json.loads(left.read_text())
    assert e["rejection"]["answers"][0]["status"] == 401
    assert e["replay_attempts"][0]["answers"][0]["status"] == 422
    [c] = [c for c in collect(cfg, now=now)]
    assert c.source == "replay_attempt" and not c.eligible and c.statuses == [422]


def test_a_link_failure_stops_the_run_and_keeps_order(tmp_path, monkeypatch):
    cfg, _sink, _agent = _rejected_run(tmp_path, [401, 401, 401])
    monkeypatch.setenv("EDGE_TEST_SECRET", cfg.secret)
    post = Answers(503)
    rc, text = _run(["--config", str(_config_file(tmp_path, cfg)), "--apply"], post=post,
                    now=utc(T0 + 10 * CADENCE))
    assert rc == 1 and len(post.requests) == 1 and "link failed" in text
    assert len(_rejected(cfg)) == 3


def test_held_back_while_an_older_window_of_the_model_is_queued(tmp_path, monkeypatch):
    cfg, _sink, agent = _rejected_run(tmp_path, [401])
    [f] = _rejected(cfg)
    e = json.loads(f.read_text())
    from datetime import datetime, timedelta
    older = datetime.fromisoformat(e["end"]) - timedelta(seconds=CADENCE)
    agent.sender.spool.put({**e["body"], "window_id": "older"}, older)   # still queued in the live spool
    monkeypatch.setenv("EDGE_TEST_SECRET", cfg.secret)
    post = Answers()
    rc, text = _run(["--config", str(_config_file(tmp_path, cfg)), "--apply"], post=post,
                    now=utc(T0 + 10 * CADENCE))
    assert rc == 0 and post.requests == [] and "held_back" in text and len(_rejected(cfg)) == 1


def test_filters_and_the_offline_queue_limit(tmp_path, monkeypatch):
    cfg, _sink, _agent = _rejected_run(tmp_path, [401, 401, 401])
    conf = str(_config_file(tmp_path, cfg))
    ends = [json.loads(f.read_text())["end"] for f in _rejected(cfg)]
    now = utc(T0 + 10 * CADENCE)

    def rows(*extra):
        return json.loads(_run(["--config", conf, "--json", *extra], now=now)[1])["windows"]

    assert len(rows()) == 3
    assert [r["window_end"] for r in rows("--since", ends[1])] == ends[1:]
    assert [r["window_end"] for r in rows("--until", ends[1])] == ends[:2]
    assert rows("--model", "someone-else") == []
    assert len(rows("--model", "LOCAL_MODEL_SENTINEL")) == 3                 # local name maps to the wire id
    late = utc(T0 + 34 * 86400)
    stale = json.loads(_run(["--config", conf, "--json"], now=late)[1])["windows"]
    assert not any(r["eligible"] for r in stale) and "offline-queue limit" in stale[0]["why"]


def _clean_env() -> dict:
    """A bare environment for `python -m edge_agent`, plus the loader path the interpreter itself may
    need: setup-python's CPython is a shared-library build that does not start without
    LD_LIBRARY_PATH (CI run 41081991532: libpython3.12.so.1.0 not found)."""
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{ROOT}:{ROOT / 'sdk'}"}
    if os.environ.get("LD_LIBRARY_PATH"):
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    return env


def test_python_dash_m_dispatches_the_subcommand(tmp_path):
    cfg, _sink, _agent = _rejected_run(tmp_path, [401])
    r = subprocess.run([sys.executable, "-m", "edge_agent", "replay-rejected", "--config",
                        str(_config_file(tmp_path, cfg)), "--json"], capture_output=True, text=True,
                       cwd=ROOT, env=_clean_env())
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["summary"]["applied"] is False and out["windows"][0]["status"] == [401]


def _wrapper(tmp_path, monkeypatch):
    for k in ("BBE_INSTALL_DIR", "BBE_CONF_DIR", "BBE_DATA_DIR"):
        monkeypatch.setenv(k, str(tmp_path / k.lower()))
    path = ROOT / "packaging/edge_agent/installer/blurredbox-edge"
    spec = importlib.util.spec_from_loader("bbe_cli_rr", importlib.machinery.SourceFileLoader("bbe_cli_rr", str(path)))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls: list = []
    monkeypatch.setattr(mod, "as_service", lambda cmd, env, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    return mod, calls


def test_operator_wrapper_refuses_apply_while_the_service_runs(tmp_path, monkeypatch, capsys):
    mod, calls = _wrapper(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "service_active", lambda: True)
    assert mod.main(["replay-rejected", "--apply"]) == 2 and calls == []
    assert "stop it first" in capsys.readouterr().err
    assert mod.main(["replay-rejected"]) == 0                                 # a dry run is fine while running
    assert mod.main(["replay-rejected", "--apply", "--while-running", "--include-unknown"]) == 0
    assert calls[0][-3:] == ["replay-rejected", "--config", str(mod.CONFIG)]
    assert "--while-running" not in calls[1] and calls[1][-2:] == ["--apply", "--include-unknown"]
    monkeypatch.setattr(mod, "service_active", lambda: False)
    assert mod.main(["replay-rejected", "--apply"]) == 0 and calls[2][-1] == "--apply"
