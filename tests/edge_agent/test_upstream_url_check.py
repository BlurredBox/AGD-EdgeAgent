import pytest

from edge_agent.mapping import ConfigError
from edge_agent.upstream import AirflowSource


def _mk(tmp_path, url):
    return AirflowSource({"type": "airflow", "base_url": url, "dag_ids": ["d"], "token_env": "T"}, tmp_path, name="x",
                         env={"T": "t"})


@pytest.mark.parametrize("url", ["https://x", "HTTPS://x", "HTTP://127.0.0.1", "http://localhost:8080",
                                 "http://[::1]:8080"])
def test_accepted(tmp_path, url):
    _mk(tmp_path, url)


@pytest.mark.parametrize("url", ["http://localhost.evil.com", "http://127.0.0.1.nip.io", "http://airflow.internal",
                                 "ftp://x", "http://127.0.0.1@evil.com"])
def test_rejected(tmp_path, url):
    with pytest.raises(ConfigError):
        _mk(tmp_path, url)
