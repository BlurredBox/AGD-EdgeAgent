"""Kafka adapter: configuration, watermark and staging logic. No broker needed.

The behaviour against a real broker (TLS, SASL, rebalance, crash, outage) is in
test_source_kafka_live.py.
"""
from __future__ import annotations

import json
from collections import deque

import pytest

pytest.importorskip("confluent_kafka")

from edge_agent.mapping import ConfigError                              # noqa: E402
from edge_agent.sources import build_sources                            # noqa: E402

ENV = {"K_USER": "edge", "K_PASS": "pw-sentinel-9f2c", "K_KEYPW": "kp-sentinel-1a7e",
       "K_CID": "cid-sentinel", "K_SECRET": "cs-sentinel-77"}
COLUMNS = {"ts": "scored_at", "model_id": "model", "instrument": "ticker", "output": "p",
           "features": ["f1", "f2"]}


def spec(**kw):
    base = {"type": "kafka", "bootstrap_servers": "broker.invalid:9093", "topics": ["preds"],
            "start_at": "now", "tls": {}, "columns": COLUMNS,
            "sasl": {"mechanism": "SCRAM-SHA-512", "username_env": "K_USER", "password_env": "K_PASS"}}
    base.update(kw)
    return {k: v for k, v in base.items() if v is not None}


def build(tmp_path, **kw):
    return build_sources(spec(**kw), tmp_path / "st", "prediction", env=ENV)[0]


@pytest.mark.parametrize("patch,needle", [
    ({"password": "x"}, "credentials must not be in the config"),
    ({"sasl_password": "x"}, "credentials must not be in the config"),
    ({"sasl": {"mechanism": "PLAIN", "username_env": "K_USER", "password": "x"}}, "credentials must not be"),
    ({"sasl": {"mechanism": "PLAIN", "username": "edge", "password_env": "K_PASS"}}, "use username_env"),
    ({"tls": {"key_password": "x"}}, "credentials must not be"),
    ({"schema_registry": {"url": "https://r", "password": "x"}, "value_format": "avro"}, "credentials must not be"),
    ({"sasl": {"mechanism": "OAUTHBEARER", "oidc": {"token_endpoint_url": "https://i/t", "client_secret": "x"}}},
     "credentials must not be"),
    ({"tls": False}, "unencrypted"),
    ({"tls": False, "allow_insecure_tls": True}, "SASL over a plaintext connection"),
    ({"tls": "yes"}, "tls must be an object"),
    ({"tls": {"cert_file": "/nonexistent.pem", "key_file": "/nonexistent.key"}}, "file not found"),
    ({"sasl": {"mechanism": "GSSAPI"}}, "mechanism must be one of"),
    ({"sasl": {"mechanism": "PLAIN", "username_env": "K_USER"}}, "password_env is required"),
    ({"sasl": {"mechanism": "PLAIN", "username_env": "NOPE", "password_env": "K_PASS"}}, "NOPE is not set"),
    ({"sasl": {"mechanism": "OAUTHBEARER"}}, "sasl.oidc is required"),
    ({"sasl": {"mechanism": "OAUTHBEARER", "oidc": {"token_endpoint_url": "http://i/t",
      "client_id_env": "K_CID", "client_secret_env": "K_SECRET"}}}, "must be an https"),
    ({"topics": None}, "topics is required"),
    ({"topics": ["^pred.*"]}, "exact topic names"),
    ({"topics": ["a", "a"]}, "exact topic names"),
    ({"bootstrap_servers": None}, "bootstrap_servers is required"),
    ({"bootstrap_servers": "nohostport"}, "bootstrap_servers is required"),
    ({"start_at": None}, "start_at is required"),
    ({"start_at": "yesterday"}, "start_at is required"),
    ({"value_format": "xml"}, "value_format must be"),
    ({"value_format": "avro"}, "exactly one of"),
    ({"schema_registry": {"url": "https://r"}}, "apply to value_format 'avro' only"),
    ({"value_format": "avro", "schema_registry": {"url": "http://r"}}, "plain http"),
    ({"max_uncommitted_rows": 5}, "max_uncommitted_rows"),
    ({"batch_size": 0}, "batch_size"),
    ({"surprise": 1}, "unknown key"),
    ({"columns": {k: v for k, v in COLUMNS.items() if k != "features"}}, "features is required"),
])
def test_config_errors_name_the_key(tmp_path, patch, needle):
    with pytest.raises(ConfigError, match=needle):
        build(tmp_path, **patch)


def test_tls_is_on_by_default_and_verifies_the_server(tmp_path):
    a = build(tmp_path, sasl=None)
    assert a._security["security.protocol"] == "SSL"
    assert a._security["enable.ssl.certificate.verification"] is True
    assert a._security["ssl.endpoint.identification.algorithm"] == "https"
    b = build(tmp_path)
    assert b._security["security.protocol"] == "SASL_SSL" and b._security["sasl.mechanism"] == "SCRAM-SHA-512"
    plain = build(tmp_path, tls=False, sasl=None, allow_insecure_tls=True)
    assert plain._security["security.protocol"] == "PLAINTEXT"


def test_secrets_come_from_the_environment_and_are_not_kept_in_the_adapter(tmp_path):
    a = build(tmp_path, tls={"key_file": __file__, "cert_file": __file__, "key_password_env": "K_KEYPW"})
    conf = a._conf(callbacks=False)
    assert conf["sasl.username"] == "edge" and conf["sasl.password"] == ENV["K_PASS"]
    assert conf["ssl.key.password"] == ENV["K_KEYPW"]
    held = json.dumps(a.describe()) + repr({k: v for k, v in vars(a).items() if k != "_env"})
    for secret in (ENV["K_PASS"], ENV["K_KEYPW"]):
        assert secret not in held


def test_oauthbearer_maps_to_the_oidc_client_credentials_flow(tmp_path):
    oidc = {"token_endpoint_url": "https://idp.example/oauth/token", "client_id_env": "K_CID",
            "client_secret_env": "K_SECRET", "scope": "kafka"}
    conf = build(tmp_path, sasl={"mechanism": "OAUTHBEARER", "oidc": oidc})._conf(callbacks=False)
    assert conf["sasl.mechanism"] == "OAUTHBEARER" and conf["sasl.oauthbearer.method"] == "oidc"
    assert conf["sasl.oauthbearer.client.id"] == "cid-sentinel"
    assert conf["sasl.oauthbearer.client.secret"] == "cs-sentinel-77"
    assert conf["sasl.oauthbearer.token.endpoint.url"] == oidc["token_endpoint_url"]


def test_auto_commit_is_off_and_the_group_defaults_per_source(tmp_path):
    a = build(tmp_path, name="credit")
    conf = a._conf(callbacks=False)
    assert conf["enable.auto.commit"] is False and conf["enable.auto.offset.store"] is False
    assert a.group_id == "blurredbox-edge-credit"
    assert build(tmp_path, group_id="mine").group_id == "mine"


def test_outcome_source_needs_outcome_columns(tmp_path):
    out = {"model_id": "model", "prediction_id": "pid", "outcome": "label", "outcome_observed_at": "known"}
    (a,) = build_sources(spec(columns=out), tmp_path / "st", "outcome", env=ENV)
    assert a.kind == "outcome"
    with pytest.raises(ConfigError, match="columns.prediction_id is required"):
        build_sources(spec(columns={k: v for k, v in out.items() if k != "prediction_id"}),
                      tmp_path / "st2", "outcome", env=ENV)


def test_building_touches_neither_network_nor_state(tmp_path):
    a = build(tmp_path)
    assert a._consumer is None and not (tmp_path / "st").exists()


# -- watermark / release / staging logic with a stub consumer ------------------------------------------
class StubConsumer:
    def __init__(self, hw):
        self.hw = hw

    def get_watermark_offsets(self, tp, timeout=None, cached=False):
        return 0, self.hw[(tp.topic, tp.partition)]


def adapter_with(tmp_path, parts, **kw):
    """parts: {partition: (hw, fetch_pos, [(ts, offset)...], seen_max)}"""
    a = build(tmp_path, max_uncommitted_rows=kw.pop("cap", 1_000_000), **kw)
    hw = {}
    for p, (h, pos, entries, seen) in parts.items():
        key = ("preds", p)
        hw[key] = h
        a._assigned.add(key)
        a._fetch_pos[key] = pos
        a._last_msg[key] = a._mono()
        a._buf[key] = deque((ts, off, object()) for ts, off in entries)
        a._buffered += len(entries)
        if seen is not None:
            a._seen_max[key] = seen
    return a, StubConsumer(hw)


def test_watermark_is_the_minimum_over_partitions_that_still_have_a_backlog(tmp_path):
    a, c = adapter_with(tmp_path, {
        0: (100, 100, [(500.0, 90), (510.0, 91)], 510.0),        # caught up: no constraint
        1: (200, 150, [(300.0, 140), (320.0, 141)], 320.0),      # backlog, read up to t=320
        2: (50, 10, [(100.0, 9)], 100.0)})                       # backlog, read only up to t=100
    assert a._watermark(c) == 100.0
    out = a._release(c, 100)
    assert len(out) == 1                                          # only t=100 may go out
    a._seen_max[("preds", 2)] = 400.0                             # partition 2 advanced
    a._buf[("preds", 2)].extend([(400.0, 10, object())])
    a._buffered += 1
    assert a._watermark(c) == 320.0
    assert len(a._release(c, 100)) == 2                           # t=300 and t=320 (p1); t=400 and p0's 500 wait


def test_release_merges_partitions_in_time_order_and_stages_per_partition_offsets(tmp_path):
    a, c = adapter_with(tmp_path, {
        0: (3, 3, [(10.0, 0), (30.0, 1), (50.0, 2)], 50.0),
        1: (3, 3, [(20.0, 0), (40.0, 1), (60.0, 2)], 60.0)})
    for key, q in a._buf.items():
        a._buf[key] = deque((ts, off, ts) for ts, off, _ in q)
    out = a._release(c, 4)
    assert out == [10.0, 20.0, 30.0, 40.0]
    (staged,) = a._staged
    assert staged[1] == {"preds/0": 2, "preds/1": 2} and staged[2] == 4 and staged[0] == 40.0
    assert a._buffered == 2


def test_offsets_move_only_when_commit_says_the_window_is_durable(tmp_path):
    from datetime import datetime, timezone
    a, c = adapter_with(tmp_path, {0: (4, 4, [(10.0, 0), (20.0, 1), (130.0, 2), (140.0, 3)], 140.0)})
    a._release(c, 2)                                              # offsets 0,1
    a._release(c, 2)                                              # offsets 2,3
    a.commit(datetime.fromtimestamp(60, timezone.utc))           # windows before t=60 are durable
    assert a._committed == {("preds", 0): 2}
    a.commit(None)
    assert a._committed == {("preds", 0): 4}
    assert json.loads(a.state_file.read_text())["committed"] == {"preds/0": 4}


def test_the_cap_counts_staged_and_buffered_rows(tmp_path):
    a, c = adapter_with(tmp_path, {0: (2000, 1500, [(float(i), i) for i in range(1500)], 1500.0)},
                        cap=1000)
    assert a.blocked
    a._staged_rows = 1000
    assert a._release(c, 10_000) == []                            # nothing leaves while 1000 are uncommitted
    assert a.describe()["blocked"] is True


def test_a_partition_that_stays_silent_does_not_hold_the_others_back_forever(tmp_path):
    now = [1000.0]
    a, c = adapter_with(tmp_path, {0: (10, 10, [(5.0, 9)], 5.0), 1: (10, 3, [], None)})
    a._mono = lambda: now[0]
    a._last_msg = {("preds", 0): 1000.0, ("preds", 1): 1000.0}
    assert a._watermark(c) == float("-inf") and a._release(c, 10) == []
    now[0] += a.idle_seconds + 1                                  # no message from p1 for a long time
    assert a._watermark(c) == float("inf") and len(a._release(c, 10)) == 1


# -- --check on a caught-up group (pre-tag review, edgefix2 2026-10-30) ------------------------------------------------
class _CheckConsumer:
    def __init__(self, low, high, committed):
        self.low, self.high, self.c = low, high, committed

    def committed(self, tps, timeout=None):
        from confluent_kafka import TopicPartition
        return [TopicPartition(t.topic, t.partition, self.c.get((t.topic, t.partition), -1001)) for t in tps]

    def get_watermark_offsets(self, tp, timeout=None, cached=False):
        return self.low[(tp.topic, tp.partition)], self.high[(tp.topic, tp.partition)]


def test_sample_positions_of_a_caught_up_group_look_at_the_latest_messages(tmp_path):
    import confluent_kafka as ck
    a = build(tmp_path, start_at="beginning")
    keys = [("preds", 0), ("preds", 1)]
    low, high = {keys[0]: 0, keys[1]: 95}, {keys[0]: 500, keys[1]: 100}
    start, end = a._sample_positions(_CheckConsumer(low, high, dict(high)), ck, keys, 20)
    assert start == {keys[0]: 490, keys[1]: 95} and end == high      # newest 10 per partition, within retention
    behind = {keys[0]: 480, keys[1]: 100}                              # a backlog: read from the position as before
    assert a._sample_positions(_CheckConsumer(low, high, behind), ck, keys, 20)[0] == behind
    fresh = a._sample_positions(_CheckConsumer(low, {k: low[k] for k in keys}, {}), ck, keys, 20)
    assert fresh[0] == fresh[1] == low                                 # fresh group on an empty topic: nothing


def test_sample_positions_of_a_fresh_group_with_nothing_after_its_iso_start_look_at_the_latest(tmp_path):
    """kitrun6 R-02 (the PostgreSQL finding, checked for Kafka): "now" already samples the latest messages; an ISO
    start_at with no message after it yet did not, so the check said NO_ROWS_YET on a busy topic."""
    import confluent_kafka as ck

    class _Fresh(_CheckConsumer):
        def offsets_for_times(self, tps, timeout=None):
            return [ck.TopicPartition(t.topic, t.partition, -1) for t in tps]   # nothing at or after that time

    keys = [("preds", 0), ("preds", 1)]
    low, high = {keys[0]: 0, keys[1]: 0}, {keys[0]: 500, keys[1]: 100}
    for start_at in ("now", "2099-01-01T00:00:00Z"):
        a = build(tmp_path / start_at.replace(":", ""), start_at=start_at)
        start, end = a._sample_positions(_Fresh(low, high, {}), ck, keys, 20)
        assert start == {keys[0]: 490, keys[1]: 90} and end == high, start_at
