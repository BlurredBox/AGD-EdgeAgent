"""Audit 2026-11-04 F11/F12 and AGD #475: modes, secret hygiene, https-only, no redirects."""
import http.server
import os
import stat
import threading
import urllib.error

import pytest

from edge_agent.config import EdgeConfig
from edge_agent.transport import DurableSpool
from edge_agent.tripwires import LocalAlerter
from edge_agent.upstream import AirflowSource


def _cfg(tmp_path, **kw):
    base = dict(tenant_id="t", install_id="i", models={"m": "mid"}, secret="TOPSECRETVALUE",
                endpoints=["https://e.invalid"], state_dir=tmp_path / "s")
    base.update(kw)
    return EdgeConfig(**base)


def test_secret_not_in_repr(tmp_path):
    c = _cfg(tmp_path)
    assert "TOPSECRETVALUE" not in repr(c) and "TOPSECRETVALUE" not in str(c)


@pytest.mark.parametrize("ep", ["http://e.invalid", "http://192.0.2.1", "http://example.com", "ftp://x", "e.invalid"])
def test_non_https_endpoint_refused(tmp_path, ep):
    with pytest.raises(ValueError, match="https"):
        _cfg(tmp_path, endpoints=[ep])


def test_spool_dirs_0700(tmp_path):
    old = os.umask(0)
    try:
        sp = DurableSpool(tmp_path / "x" / "spool")
    finally:
        os.umask(old)
    assert stat.S_IMODE(sp.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(sp.rejected.stat().st_mode) == 0o700


def test_alerts_file_0600(tmp_path):
    f = tmp_path / "d" / "alerts.jsonl"
    old = os.umask(0)
    try:
        LocalAlerter(f).raise_alert("k", "m", "msg")
    finally:
        os.umask(old)
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert stat.S_IMODE(f.parent.stat().st_mode) == 0o700


def test_upstream_does_not_follow_redirect():
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            if self.path == "/a":
                self.send_response(302)
                self.send_header("Location", "/b")
                self.end_headers()
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        obj = object.__new__(AirflowSource)
        obj.timeout, obj._ca = 2, None
        with pytest.raises(urllib.error.HTTPError):
            obj._http_get(f"http://127.0.0.1:{srv.server_address[1]}/a", {"Authorization": "x"})
        assert seen == ["/a"]
    finally:
        srv.shutdown()


def _raw(**kw):
    return dict(tenant_id="t", install_id="i", models={"m": "mid"}, endpoints=["https://e.invalid"],
                state_dir="/tmp/x", **kw)


@pytest.mark.parametrize("ep", ["http://127.0.0.1:9", "http://localhost:9", "http://[::1]:9"])
def test_loopback_http_is_accepted(tmp_path, ep):
    _cfg(tmp_path, endpoints=[ep])


@pytest.mark.parametrize("ep", ["http://127.0.0.1.evil.com", "http://localhost.evil.com"])
def test_lookalike_hosts_are_refused(tmp_path, ep):
    with pytest.raises(ValueError, match="https"):
        _cfg(tmp_path, endpoints=[ep])


def test_inline_secret_in_a_readable_config_is_refused(tmp_path):
    from edge_agent.__main__ import load_edge_config
    f = tmp_path / "agent.json"
    f.write_text("{}")
    f.chmod(0o644)
    with pytest.raises(ValueError, match="chmod 600") as e:
        load_edge_config(_raw(secret="TOPSECRETVALUE"), need_secret=True, path=str(f))
    assert str(f) in str(e.value) and "TOPSECRETVALUE" not in str(e.value)
    f.chmod(0o600)
    assert load_edge_config(_raw(secret="TOPSECRETVALUE"), need_secret=True, path=str(f)).secret == "TOPSECRETVALUE"


def test_unset_secret_env_is_a_clear_error(monkeypatch):
    from edge_agent.__main__ import load_edge_config
    monkeypatch.delenv("BB_NO_SUCH_SECRET", raising=False)
    with pytest.raises(ValueError, match="BB_NO_SUCH_SECRET"):
        load_edge_config(_raw(secret_env="BB_NO_SUCH_SECRET"), need_secret=True)
