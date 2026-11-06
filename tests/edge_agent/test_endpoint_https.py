"""Audit F12: endpoints must be https (or literal-loopback http); the API key and signature must not travel in clear."""
import pytest

from edge_agent.config import EdgeConfig


def _cfg(tmp_path, ep):
    return EdgeConfig(tenant_id="t", install_id="i", models={"m": "mid"}, secret="s", endpoints=[ep],
                      state_dir=tmp_path)


@pytest.mark.parametrize("ep", ["http://e.invalid", "http://127.0.0.2:1", "ftp://x", "e.invalid"])
def test_non_https_endpoint_refused(tmp_path, ep):
    with pytest.raises(ValueError, match="https"):
        _cfg(tmp_path, ep)


@pytest.mark.parametrize("ep", ["https://e.invalid", "http://127.0.0.1:1", "http://localhost:8", "http://[::1]:9"])
def test_https_and_loopback_accepted(tmp_path, ep):
    _cfg(tmp_path, ep)
