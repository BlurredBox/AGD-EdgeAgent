"""Hunt B / H3: the agent refuses a non-loopback http:// endpoint (F12), but the pairing tool
(scripts/partner_pair.py, shipped as edge/partner_pair.py and /opt/blurredbox-edge/lib/partner_pair.py) and the
container's `pair` command take the same endpoints[0] with no scheme check, and send the bearer API key and the
pairing code in clear. Local fake server on 127.0.0.2 only (loopback, but outside PR #588's literal allow-list, so it
stands for a remote http host).
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
API_KEY = "hunt-b-fake-api-key-not-real"
SECRET = "hunt-b-fake-hmac-secret-not-real"


@pytest.fixture()
def plain_http():
    seen = []

    class H(BaseHTTPRequestHandler):
        def _answer(self):
            n = int(self.headers.get("Content-Length") or 0)
            seen.append((self.command, self.path, dict(self.headers), self.rfile.read(n)))
            body = (json.dumps({"config": {"cadence_minutes": 5}, "desired_generation": 1})
                    if "edge-configuration" in self.path else "{}").encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        do_GET = do_POST = _answer

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.2", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.2:{srv.server_address[1]}", seen
    srv.shutdown()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_agent_config_refuses_this_url(plain_http, tmp_path):
    """The PR #588 check itself (fails on origin/dev, where http:// is accepted: that is F12)."""
    from edge_agent.config import EdgeConfig
    url, _ = plain_http
    with pytest.raises(ValueError, match="https"):
        EdgeConfig(tenant_id="t", install_id="i", models={"m": "bb"}, secret="s", endpoints=[url],
                   state_dir=tmp_path)


@pytest.mark.skipif(not (REPO / "scripts/partner_pair.py").exists(), reason="partner_pair.py ships in the kit, not here")
def test_partner_pair_refuses_http_before_any_request(plain_http, monkeypatch, capsys):
    url, seen = plain_http
    pp = _load("partner_pair_hunt", REPO / "scripts/partner_pair.py")
    monkeypatch.setenv("AGD_HMAC_SECRET", SECRET)
    monkeypatch.setenv("BLURREDBOX_API_KEY", API_KEY)
    args = argparse.Namespace(base_url=url, install_id="install-1", model=["bb-model-1:PAIRCODE123"], ca_file=None,
                              edge_config_out=None)
    assert asyncio.run(pp.amain(args)) == 2
    assert seen == []                                   # nothing was sent
    err = capsys.readouterr().err
    assert "https" in err and API_KEY not in err and "PAIRCODE123" not in err


def test_container_pair_refuses_http_endpoint(plain_http, tmp_path, monkeypatch):
    url, _ = plain_http
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"endpoints": [url], "install_id": "install-1", "models": {"m": "bb-model-1"},
                               "secret_env": "AGD_HMAC_SECRET", "state_dir": str(tmp_path / "state")}))
    env = tmp_path / "agent.env"
    env.write_text(f"AGD_HMAC_SECRET={SECRET}\nBLURREDBOX_API_KEY={API_KEY}\n")
    env.chmod(0o600)
    monkeypatch.setenv("BBE_CONFIG", str(cfg))
    ep = _load("bbe_entrypoint_hunt", REPO / "packaging/edge_agent/container/entrypoint.py")
    with pytest.raises(ep.SetupError, match="https"):
        ep.prepare(["pair", "--model", "bb-model-1:PAIRCODE123"], {"BBE_ENV_FILE": str(env)})
