"""`edge_agent incident ...` / `edge_agent erase` (edge_agent/partner_api.py) and their `blurredbox-edge` wrappers.

The request is checked with the server's formula (src/api/security/signature.py: HMAC-SHA256 over
`timestamp:nonce:canonical_body`); the live round trip through the real routes is in
tests/test_edge_partner_api_live.py.
"""
from __future__ import annotations

import hashlib
import hmac
import importlib.machinery
import importlib.util
import io
import json
from pathlib import Path

import orjson
import pytest

from edge_agent.partner_api import main

SECRET = "p" * 40
KEY = "agd_live_testkey"
INST = Path(__file__).resolve().parents[2] / "packaging/edge_agent/installer"


def _conf(tmp_path, models=None, endpoints=None) -> Path:
    p = tmp_path / "agent.json"
    p.write_text(json.dumps({"tenant_id": "t-1", "install_id": "i-1",
                             "models": models or {"risk_v3": "bb-model-1"}, "secret_env": "EDGE_TEST_SECRET",
                             "endpoints": endpoints or ["https://a.example.invalid"],
                             "state_dir": str(tmp_path / "state"),
                             "source": {"type": "jsonl", "path": str(tmp_path / "unused.jsonl")}}))
    return p


def _server_check(body: bytes, headers) -> None:
    """What verify_hmac_signature computes, byte for byte."""
    canon = orjson.dumps(orjson.loads(body), option=orjson.OPT_SORT_KEYS) if body else b""
    msg = f"{headers['X-AGD-Timestamp']}:{headers['X-AGD-Nonce']}:{canon.decode()}".encode()
    assert hmac.compare_digest(hmac.new(SECRET.encode(), msg, hashlib.sha256).hexdigest(), headers["X-AGD-Signature"])
    assert headers["Authorization"] == f"Bearer {KEY}"


class _Server:
    def __init__(self, answers=None, down=()):
        self.calls, self.answers, self.down = [], answers or {}, set(down)

    def __call__(self, method, url, path, body, headers):
        if url in self.down:
            raise ConnectionError("refused")
        _server_check(body, headers)
        self.calls.append((method, url, path, json.loads(body) if body else None, headers["X-AGD-Nonce"]))
        return self.answers.get((method, path), (200, {}))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("EDGE_TEST_SECRET", SECRET)
    monkeypatch.setenv("BLURREDBOX_API_KEY", KEY)


def _run(argv, server, inp=None):
    out = io.StringIO()
    return main(argv, request=server, out=out, inp=inp or io.StringIO()), out.getvalue()


def test_mark_signs_the_body_and_maps_the_local_model(tmp_path):
    srv = _Server({("POST", "/api/v1/incidents"): (201, {"id": "1f0e"})})
    rc, out = _run(["incident", "mark", "--config", str(_conf(tmp_path)), "--start", "2026-11-03T14:05",
                    "--end", "2026-11-03T15:00Z", "--model", "risk_v3", "--class", "data_feed",
                    "--note", "vendor feed stale"], srv)
    assert rc == 0 and "marked: incident 1f0e for bb-model-1" in out
    (lm, _, lpath, _, _), (method, url, path, body, _) = srv.calls       # the existing marks are listed first
    assert (lm, lpath) == ("GET", "/api/v1/incidents")
    assert (method, url, path) == ("POST", "https://a.example.invalid", "/api/v1/incidents")
    assert body == {"monitored_model_id": "bb-model-1", "failed_from": "2026-11-03T14:05:00+00:00",
                    "failed_to": "2026-11-03T15:00:00+00:00", "failure_class": "data_feed",
                    "notes": "vendor feed stale"}
    assert SECRET not in out and KEY not in out


MARK = ["--start", "2026-11-03T14:05", "--end", "2026-11-03T15:00Z", "--class", "data_feed"]


def test_mark_that_exists_is_not_created_again(tmp_path):
    existing = {"id": "e1", "monitored_model_id": "bb-model-1", "failed_from": "2026-11-03T14:05:00Z",
                "failed_to": "2026-11-03T15:00:00+00:00", "failure_class": "data_feed", "notes": None}
    srv = _Server({("GET", "/api/v1/incidents"): (200, {"incidents": [existing]})})
    rc, out = _run(["incident", "mark", "--config", str(_conf(tmp_path)), *MARK], srv)
    assert rc == 0 and "already marked" in out and "e1" in out
    assert [c[0] for c in srv.calls] == ["GET"]
    # another class, or another end, is a different mark
    rc, out = _run(["incident", "mark", "--config", str(_conf(tmp_path)), "--start", "2026-11-03T14:05",
                    "--end", "2026-11-03T15:00Z"], srv)
    assert [c[0] for c in srv.calls] == ["GET", "GET", "POST"]


def test_lost_answer_is_checked_before_the_next_endpoint(tmp_path):
    conf = str(_conf(tmp_path, endpoints=["https://a.invalid", "https://b.invalid"]))
    stored: list[dict] = []

    def server(method, url, path, body, headers):
        _server_check(body, headers)
        if method == "GET":
            return 200, {"incidents": stored}
        stored.append({"id": f"m{len(stored)}", **json.loads(body)})
        if url == "https://a.invalid":
            raise TimeoutError("read timed out")          # stored, answer lost
        return 201, {"id": stored[-1]["id"]}
    out = io.StringIO()
    rc = main(["incident", "mark", "--config", conf, *MARK], request=server, out=out, inp=io.StringIO())
    assert rc == 0 and len(stored) == 1 and "already marked" in out.getvalue() and "m0" in out.getvalue()


def test_lost_answer_and_unreachable_list_stops_with_outcome_unknown_and_no_second_post(tmp_path):
    """Pre-tag review L4: the POST to the first endpoint got no answer and the marks cannot be listed to
    check. POSTing to the next endpoint could create a second mark, so the command stops and says so."""
    conf = str(_conf(tmp_path, endpoints=["https://a.invalid", "https://b.invalid"]))
    calls: list[tuple[str, str]] = []

    def server(method, url, path, body, headers):
        calls.append((method, url))
        if method == "GET":
            if any(m == "POST" for m, _ in calls):
                raise ConnectionError("refused")              # the list is unreachable after the lost POST
            return 200, {"incidents": []}
        raise TimeoutError("read timed out")                  # maybe stored, answer lost
    out = io.StringIO()
    rc = main(["incident", "mark", "--config", conf, *MARK], request=server, out=out, inp=io.StringIO())
    assert rc == 1
    assert "outcome unknown" in out.getvalue() and "incident list" in out.getvalue()
    assert [c for c in calls if c[0] == "POST"] == [("POST", "https://a.invalid")]


def test_mark_needs_a_model_when_several_are_configured_and_rejects_reversed_times(tmp_path):
    conf = str(_conf(tmp_path, models={"a": "bb-a", "b": "bb-b"}))
    srv = _Server()
    with pytest.raises(SystemExit) as e:
        _run(["incident", "mark", "--config", conf, "--start", "2026-11-03T14:00", "--end", "2026-11-03T15:00"], srv)
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        _run(["incident", "mark", "--config", conf, "--model", "bb-b",
              "--start", "2026-11-03T15:00", "--end", "2026-11-03T14:00"], srv)
    assert e.value.code == 2 and srv.calls == []


def test_unreachable_endpoint_falls_through_with_a_fresh_nonce_and_a_refusal_is_final(tmp_path):
    conf = str(_conf(tmp_path, endpoints=["https://down.invalid", "https://up.invalid", "https://third.invalid"]))
    srv = _Server({("GET", "/api/v1/incidents"): (403, {"detail": "Invalid Message Signature"})},
                  down={"https://down.invalid"})
    rc, out = _run(["incident", "list", "--config", conf], srv)
    assert rc == 1 and "refused: HTTP 403" in out
    assert [c[1] for c in srv.calls] == ["https://up.invalid"]        # not repeated on the third


def test_erase_asks_and_refuses_without_confirmation(tmp_path):
    conf = str(_conf(tmp_path))
    srv = _Server({("DELETE", "/api/v1/telemetry-data"): (200, {"deleted": {"runtime_windows": 3}})})
    rc, out = _run(["erase", "--config", conf], srv)                       # no terminal, no --yes
    assert rc == 2 and srv.calls == []

    class Tty(io.StringIO):
        def isatty(self):
            return True
    rc, out = _run(["erase", "--config", conf], srv, inp=Tty("no\n"))
    assert rc == 2 and srv.calls == []
    rc, out = _run(["erase", "--config", conf], srv, inp=Tty("ERASE\n"))
    assert rc == 0 and [c[:3] for c in srv.calls] == [("DELETE", "https://a.example.invalid", "/api/v1/telemetry-data")]
    rc, out = _run(["erase", "--config", conf, "--yes"], srv)
    assert rc == 0 and "consent is now off" in out and len(srv.calls) == 2


def test_delete_takes_only_an_incident_id(tmp_path):
    with pytest.raises(SystemExit) as e:
        _run(["incident", "delete", "../telemetry-data", "--config", str(_conf(tmp_path))], _Server())
    assert e.value.code == 2


def test_missing_secret_names_the_variable_only(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("EDGE_TEST_SECRET")
    with pytest.raises(SystemExit) as e:
        _run(["incident", "list", "--config", str(_conf(tmp_path))], _Server())
    assert e.value.code == 2 and "EDGE_TEST_SECRET" in capsys.readouterr().err


def _cli():
    loader = importlib.machinery.SourceFileLoader("bbe_cli_partner", str(INST / "blurredbox-edge"))
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("bbe_cli_partner", loader))
    loader.exec_module(mod)
    return mod


def test_blurredbox_edge_runs_incident_and_erase_as_the_service_user_with_agent_env(tmp_path, monkeypatch):
    cli = _cli()
    env = tmp_path / "agent.env"
    env.write_text("AGD_HMAC_SECRET=x\nBLURREDBOX_API_KEY=y\n")
    monkeypatch.setattr(cli, "ENVFILE", env)
    seen = []

    class R:
        returncode = 0
    monkeypatch.setattr(cli, "as_service", lambda cmd, e, **kw: seen.append((cmd, e)) or R())
    assert cli.main(["incident", "mark", "--start", "2026-11-03T14:00", "--end", "2026-11-03T15:00"]) == 0
    assert cli.main(["erase", "--yes"]) == 0
    (c1, e1), (c2, e2) = seen
    assert c1[1:] == ["-I", "-m", "edge_agent", "incident", "mark", "--start", "2026-11-03T14:00",
                      "--end", "2026-11-03T15:00", "--config", str(cli.CONFIG)]
    assert c2[1:] == ["-I", "-m", "edge_agent", "erase", "--yes", "--config", str(cli.CONFIG)]
    assert e1 == e2 == {"AGD_HMAC_SECRET": "x", "BLURREDBOX_API_KEY": "y"}       # environment, not argv
    assert not any("x" == a or "y" == a for a in c1 + c2)
