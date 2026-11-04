"""Kafka adapter against a real single-node broker (KRaft) with TLS, SASL and ACLs.

Start the broker with tests/edge_agent/kafka_test_broker.sh and export EDGE_KAFKA_TEST_DIR; skipped
without it. Every test uses its own topic and consumer group; nothing else on the broker is touched.
EDGE_KAFKA_TEST_CONTAINER (the container name) additionally enables the broker-outage test.
"""
from __future__ import annotations

import io
import json
import os
import secrets
import socket
import struct
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ck = pytest.importorskip("confluent_kafka")
from confluent_kafka import ConsumerGroupTopicPartitions, Producer          # noqa: E402
from confluent_kafka.admin import (AclBinding, AclOperation, AclPermissionType, AdminClient,   # noqa: E402
                                   NewTopic, ResourcePatternType, ResourceType)

from edge_agent import EdgeAgent                                    # noqa: E402
from edge_agent import __main__ as cli                              # noqa: E402
from edge_agent.sources import build_sources                        # noqa: E402
from edge_agent.status import read_status                           # noqa: E402
from tests.edge_agent.synth import (CADENCE, FEATURES, LOCAL_MODEL, T0, Collector,  # noqa: E402
                                    make_cfg, make_stream, utc)

DIR = os.getenv("EDGE_KAFKA_TEST_DIR")
pytestmark = pytest.mark.skipif(not DIR, reason="needs EDGE_KAFKA_TEST_DIR (see kafka_test_broker.sh)")
CONTAINER = os.getenv("EDGE_KAFKA_TEST_CONTAINER")
ROWS = 40
WAIT = 45.0


def _kenv() -> dict[str, str]:
    return dict(l.strip().split("=", 1) for l in Path(DIR, "env").read_text().splitlines() if "=" in l)


KENV = _kenv() if DIR else {}
CA = f"{DIR}/ca.pem"
ADAPTER_ENV = {"EDGE_KAFKA_USER": KENV.get("EDGE_KAFKA_USER", ""), "EDGE_KAFKA_PASSWORD": KENV.get("EDGE_KAFKA_PASSWORD", ""),
               "EDGE_KAFKA_KEYPW": KENV.get("EDGE_KAFKA_CLIENT_KEY_PASSWORD", "")}
SECRETS = [v for k, v in KENV.items() if "PASSWORD" in k]


def admin_conf(**extra):
    return {"bootstrap.servers": KENV["EDGE_KAFKA_BOOTSTRAP"], "security.protocol": "SASL_SSL",
            "sasl.mechanism": "SCRAM-SHA-512", "sasl.username": KENV["EDGE_KAFKA_ADMIN_USER"],
            "sasl.password": KENV["EDGE_KAFKA_ADMIN_PASSWORD"], "ssl.ca.location": CA, **extra}


@pytest.fixture(scope="module")
def admin():
    a = AdminClient(admin_conf())
    for f in a.create_acls([AclBinding(ResourceType.GROUP, "g-", ResourcePatternType.PREFIXED, "User:edge", "*",
                                       AclOperation.READ, AclPermissionType.ALLOW)]).values():
        f.result(30)
    created: list[str] = []
    a._created = created                                          # type: ignore[attr-defined]
    yield a
    if created:
        for f in a.delete_topics(created).values():
            try:
                f.result(30)
            except Exception:                                       # noqa: BLE001
                pass


@pytest.fixture(scope="module")
def make_topic(admin):
    def make(partitions: int = 1, *, acl: bool = True) -> str:
        name = f"t{secrets.token_hex(5)}"
        admin.create_topics([NewTopic(name, num_partitions=partitions, replication_factor=1)])[name].result(30)
        admin._created.append(name)
        if acl:                      # the documented read-only grant: READ on the topic (DESCRIBE is implied)
            admin.create_acls([AclBinding(ResourceType.TOPIC, name, ResourcePatternType.LITERAL, "User:edge",
                                          "*", AclOperation.READ, AclPermissionType.ALLOW)])
            time.sleep(0.5)
        deadline = time.time() + 20
        while time.time() < deadline and len(admin.list_topics(name, timeout=10).topics[name].partitions) < partitions:
            time.sleep(0.2)
        return name
    return make


@pytest.fixture(scope="module")
def producer():
    return Producer(admin_conf(**{"linger.ms": 5}))


def payload(r, **extra) -> dict:
    d = {"scored_at": r.ts.timestamp(), "model": r.model_id, "ticker": r.instrument, "p": r.output,
         "f1": r.features[FEATURES[0]], "f2": r.features[FEATURES[1]], "f3": r.features[FEATURES[2]],
         "regime": r.features["regime_zz"]}
    d.update(extra)
    return d


def send(producer, topic, items):
    """items: iterable of (partition, value, timestamp_ms|None); a dict is sent as JSON."""
    for part, value, ts in items:
        if isinstance(value, dict):
            value = json.dumps(value).encode()
        kw = {} if ts is None else {"timestamp": int(ts)}
        while True:
            try:
                producer.produce(topic, value=value, partition=part, **kw)
                break
            except BufferError:
                producer.poll(0.2)
    assert producer.flush(60) == 0


def stream_to(producer, topic, recs, partitions=1, **extra):
    send(producer, topic, [(i % partitions, payload(r, **extra), r.ts.timestamp() * 1000) for i, r in enumerate(recs)])


def windows(first, n):
    return make_stream(n, rows=ROWS, first_window=first)


def spec(topic, group=None, **kw):
    s = {"type": "kafka", "bootstrap_servers": KENV["EDGE_KAFKA_BOOTSTRAP"], "topics": [topic],
         "group_id": group or f"g-{secrets.token_hex(4)}", "tls": {"ca_file": CA},
         "sasl": {"mechanism": "SCRAM-SHA-512", "username_env": "EDGE_KAFKA_USER", "password_env": "EDGE_KAFKA_PASSWORD"},
         "start_at": "beginning", "session_timeout_ms": 6000, "poll_timeout_seconds": 0.1, "batch_size": 100, "commit_chunk_rows": 10,
         "columns": {"ts": "scored_at", "model_id": "model", "instrument": "ticker", "output": "p",
                     "features": ["f1", "f2", "f3", "regime"]}}
    s.update(kw)
    return {k: v for k, v in s.items() if v is not None}


def adapter(tmp_path, topic, *, state="st", kind="prediction", **kw):
    return build_sources(spec(topic, **kw), tmp_path / state, kind, env=ADAPTER_ENV)[0]


def drain(adapters, want: int, timeout: float = WAIT, commit: bool = False):
    """Poll until `want` records arrived (or the time is up); returns the records in arrival order."""
    adapters = adapters if isinstance(adapters, list) else [adapters]
    got, end = [], time.time() + timeout
    while len(got) < want and time.time() < end:
        for a in adapters:
            chunk = a.poll()
            got += chunk
            if commit:
                a.commit(None)
    return got


def settle(adapters, seconds: float):
    """Keep polling (and discard) so that assignment / fetch / error callbacks run."""
    adapters = adapters if isinstance(adapters, list) else [adapters]
    end, got = time.time() + seconds, []
    while time.time() < end:
        for a in adapters:
            got += a.poll()
    return got


def group_offsets(admin, group) -> dict[tuple[str, int], int]:
    res = admin.list_consumer_group_offsets([ConsumerGroupTopicPartitions(group)])[group].result(30)
    return {(t.topic, t.partition): t.offset for t in res.topic_partitions if t.offset >= 0}


def wait_offsets(admin, group, pred, timeout=20.0):
    end = time.time() + timeout
    while True:
        o = group_offsets(admin, group)
        if pred(o) or time.time() > end:
            return o
        time.sleep(0.3)


def pump(agent, until, timeout=WAIT):
    end = time.time() + timeout
    while time.time() < end:
        agent.run_once()
        if until():
            return True
    return False


def ids(sink):
    return [b["window_id"] for b in sink.bodies()]


def crash(agent):
    """Stand-in for kill -9: the process vanishes without committing anything further."""
    for s in agent.sources:
        if s._consumer is not None:
            s._consumer.close()
            s._consumer = None


# -- reading, ordering, watermark -------------------------------------------------------------------------
def test_records_are_released_in_time_order_across_partitions_and_no_window_is_late(make_topic, producer, tmp_path):
    topic = make_topic(3)
    recs = windows(0, 6)
    items = [(i % 3, payload(r), r.ts.timestamp() * 1000) for i, r in enumerate(recs)]
    items.sort(key=lambda x: (x[0], x[2]))                      # all of partition 0 first, then 1, then 2
    send(producer, topic, items)
    cfg = make_cfg(tmp_path)
    a = build_sources(spec(topic, batch_size=25), cfg.state_dir, "prediction", env=ADAPTER_ENV)[0]
    got, polls = [], 0
    end = time.time() + WAIT
    while len(got) < len(recs) and time.time() < end:
        chunk = a.poll(max_records=30)
        polls += bool(chunk)
        assert len(chunk) <= 30
        got += chunk
    assert len(got) == len(recs) and polls > 4
    stamps = [r.ts for r in got]
    assert stamps == sorted(stamps), "a record was released before an earlier one from another partition"
    a.close()
    sink = Collector()
    clock = [utc(T0 + 6 * CADENCE + 30)]
    agent = EdgeAgent(cfg, [adapter(tmp_path, topic, state="st2")], post=sink, clock=lambda: clock[0])
    assert pump(agent, lambda: agent.sources[0].rows_delivered >= len(recs))
    agent.close()
    assert len(set(ids(sink))) == 6 and all(m.cutter.late_dropped == 0 for m in agent.models.values())


def test_several_topics_in_one_source(make_topic, producer, tmp_path):
    t1, t2 = make_topic(2), make_topic(1)
    stream_to(producer, t1, windows(0, 2), 2)
    stream_to(producer, t2, windows(2, 1))
    a = build_sources(spec(t1, topics=[t1, t2]), tmp_path / "st", "prediction", env=ADAPTER_ENV)[0]
    got = drain(a, 3 * ROWS)
    assert len(got) == 3 * ROWS and [r.ts for r in got] == sorted(r.ts for r in got)
    assert a.describe()["partitions_assigned"] == 3
    a.close()


def test_features_json_column_and_kafka_timestamp_as_time(make_topic, producer, tmp_path):
    topic = make_topic()
    ms = (T0 + 5) * 1000
    send(producer, topic, [(0, {"model": "m", "ticker": "t", "p": 0.25, "feats": {"a": 1.5, "b": "x"}}, ms)])
    s = spec(topic)
    s["columns"] = {"ts": "__kafka_timestamp", "model_id": "model", "instrument": "ticker", "output": "p",
                    "features": {"json_column": "feats"}}
    a = build_sources(s, tmp_path / "st", "prediction", env=ADAPTER_ENV)[0]
    (rec,) = drain(a, 1)
    assert rec.features == {"a": 1.5, "b": "x"} and rec.ts.timestamp() == T0 + 5
    a.close()


def test_poison_messages_are_counted_skipped_and_their_offsets_still_commit(admin, make_topic, producer, tmp_path):
    topic, group = make_topic(), f"g-{secrets.token_hex(4)}"
    good = windows(0, 1)[:5]
    send(producer, topic, [(0, b"{not json", None), (0, None, None), (0, b"[1,2]", None),
                           (0, {"scored_at": "garbage"}, None)] +
         [(0, payload(r), r.ts.timestamp() * 1000) for r in good])
    a = adapter(tmp_path, topic, group=group)
    assert len(drain(a, 5)) == 5
    settle(a, 0.5)
    assert a.bad_rows == 4
    a.commit(None)
    assert wait_offsets(admin, group, lambda o: o.get((topic, 0)) == 9) == {(topic, 0): 9}
    a.close()


# -- commit semantics ---------------------------------------------------------------------------------------
def test_nothing_is_committed_until_the_agent_says_the_windows_are_durable(admin, make_topic, producer, tmp_path):
    topic, group = make_topic(2), f"g-{secrets.token_hex(4)}"
    stream_to(producer, topic, windows(0, 3), 2)
    cfg = make_cfg(tmp_path)
    clock = [utc(T0 + 2 * CADENCE + 30)]
    sink = Collector()
    agent = EdgeAgent(cfg, [build_sources(spec(topic, group=group), cfg.state_dir, "prediction", env=ADAPTER_ENV)[0]],
                      post=sink, clock=lambda: clock[0])
    src = agent.sources[0]
    end = time.time() + WAIT
    while src.rows_delivered < 3 * ROWS and time.time() < end:
        agent.poll()                                             # read only: no tick, so no commit
    assert src.rows_delivered == 3 * ROWS
    time.sleep(1.0)
    assert group_offsets(admin, group) == {}, "offsets were committed before any window was durable"
    agent.tick()                                                 # windows 0 and 1 close and are spooled
    assert len(ids(sink)) == 2
    off = wait_offsets(admin, group, lambda o: sum(o.values()) > 0)
    first_open = {0: 40, 1: 40}                                  # window 2 starts at record 80 -> offset 40 per partition
    assert 0 < sum(off.values()) and all(off[(topic, p)] <= first_open[p] for p in (0, 1))
    agent.close()
    assert wait_offsets(admin, group, lambda o: sum(o.values()) == 3 * ROWS) == {(topic, 0): 60, (topic, 1): 60}


def test_crash_restart_loses_nothing_and_never_emits_a_window_twice(make_topic, producer, tmp_path):
    topic, group = make_topic(2), f"g-{secrets.token_hex(4)}"
    stream = windows(0, 6)
    stream_to(producer, topic, stream[: 3 * ROWS], 2)
    cfg = make_cfg(tmp_path)
    clock = [utc(T0 + 2 * CADENCE + 30)]

    def new(sink):
        return EdgeAgent(cfg, [build_sources(spec(topic, group=group), cfg.state_dir, "prediction",
                                             env=ADAPTER_ENV)[0]], post=sink, clock=lambda: clock[0])
    sink_a = Collector()
    a = new(sink_a)
    assert pump(a, lambda: len(ids(sink_a)) == 2 and a.sources[0].rows_committed > 0)
    crash(a)
    stream_to(producer, topic, stream[3 * ROWS:], 2)
    clock[0] = utc(T0 + 6 * CADENCE + 30)
    sink_b = Collector()
    b = new(sink_b)
    assert pump(b, lambda: b.sources[0].rows_delivered >= 3 * ROWS)
    b.close()
    got = ids(sink_a) + ids(sink_b)
    assert len(got) == 6 and len(set(got)) == 6


def test_crash_before_the_checkpoint_replays_with_at_most_one_duplicate(make_topic, producer, tmp_path):
    topic, group = make_topic(2), f"g-{secrets.token_hex(4)}"
    stream_to(producer, topic, windows(0, 6), 2)
    cfg = make_cfg(tmp_path)
    clock = [utc(T0 + 5 * CADENCE + 30)]

    def new(sink):
        return EdgeAgent(cfg, [build_sources(spec(topic, group=group), cfg.state_dir, "prediction",
                                             env=ADAPTER_ENV)[0]], post=sink, clock=lambda: clock[0])
    sink_a = Collector()
    a = new(sink_a)
    a.sources[0].commit = lambda before=None: None               # the checkpoint never happens
    assert pump(a, lambda: len(ids(sink_a)) == 5)
    crash(a)
    clock[0] = utc(T0 + 6 * CADENCE + 30)
    sink_b = Collector()
    b = new(sink_b)
    assert pump(b, lambda: b.sources[0].rows_delivered >= 6 * ROWS)
    b.close()
    both = ids(sink_a) + ids(sink_b)
    assert len(set(both)) == 6 and len(both) - len(set(both)) <= 1
    assert b.replayed_windows_skipped == 4


# -- rebalance ----------------------------------------------------------------------------------------------
def test_rebalance_between_two_agents_in_one_group_loses_nothing(admin, make_topic, producer, tmp_path):
    topic, group = make_topic(4), f"g-{secrets.token_hex(4)}"
    recs = windows(0, 10)                                        # 400 records, 100 per partition
    s = lambda: spec(topic, group=group, batch_size=50)          # noqa: E731
    s_cols = s()["columns"] | {"prediction_id": "pid"}
    stream = [(i % 4, payload(r, pid=f"id{i}"), r.ts.timestamp() * 1000) for i, r in enumerate(recs)]
    A = build_sources({**s(), "columns": s_cols}, tmp_path / "a", "prediction", env=ADAPTER_ENV)[0]
    B = build_sources({**s(), "columns": s_cols}, tmp_path / "b", "prediction", env=ADAPTER_ENV)[0]
    send(producer, topic, stream[:160])
    seen: dict[str, int] = {}

    def take(recs_):
        for r in recs_:
            seen[r.prediction_id] = seen.get(r.prediction_id, 0) + 1
    take(drain(A, 160, commit=True))                             # A alone: everything handed over and durable
    assert len(seen) == 160 and A.describe()["partitions_assigned"] == 4
    send(producer, topic, stream[160:300])
    take(drain(A, 60))                                           # A reads a part but has NOT committed it
    uncommitted_at_join = A._staged_rows
    assert uncommitted_at_join > 0
    end = time.time() + WAIT                                     # B joins: the group rebalances
    while time.time() < end and not (A.describe()["partitions_assigned"] == 2 and B.describe()["partitions_assigned"] == 2):
        take(A.poll()); take(B.poll())                           # noqa: E702
        A.commit(None), B.commit(None)                           # noqa: E702
    assert A.describe()["partitions_assigned"] == 2 and B.describe()["partitions_assigned"] == 2
    assert A.rebalances >= 1
    send(producer, topic, stream[300:])
    end = time.time() + WAIT
    while time.time() < end and len(seen) < 400:
        take(A.poll()); take(B.poll())                           # noqa: E702
        A.commit(None), B.commit(None)                           # noqa: E702
    assert len(seen) == 400, "records were lost in the rebalance"
    dups = sum(n - 1 for n in seen.values())
    assert dups <= uncommitted_at_join, f"{dups} records read twice; only {uncommitted_at_join} were uncommitted"
    B.close()                                                    # B leaves: A takes everything back
    send(producer, topic, [(i % 4, payload(r, pid=f"late{i}"), r.ts.timestamp() * 1000)
                           for i, r in enumerate(windows(10, 1))])
    end = time.time() + WAIT
    while time.time() < end and len([k for k in seen if k.startswith("late")]) < ROWS:
        take(A.poll())
        A.commit(None)
    assert len([k for k in seen if k.startswith("late")]) == ROWS and A.describe()["partitions_assigned"] == 4
    assert all(n == 1 for k, n in seen.items() if k.startswith("late"))
    A.close()
    assert sum(wait_offsets(admin, group, lambda o: sum(o.values()) >= 440).values()) == 440


# -- TLS / SASL -----------------------------------------------------------------------------------------------
def test_tls_is_verified_by_default_a_server_not_in_the_trust_file_is_refused(make_topic, producer, tmp_path):
    topic = make_topic()
    stream_to(producer, topic, windows(0, 1))
    wrong = adapter(tmp_path, topic, tls={"ca_file": f"{DIR}/client.pem"})      # some other certificate
    assert settle(wrong, 6) == [] and wrong.last_error in ("_SSL", "_TRANSPORT", "_ALL_BROKERS_DOWN")
    wrong.close()
    right = adapter(tmp_path, topic, state="st2")
    got = drain(right, ROWS)
    assert len(got) == ROWS and right.last_error is None, right.describe()
    right.close()


@pytest.mark.parametrize("mech", ["PLAIN", "SCRAM-SHA-512"])
def test_sasl_mechanisms_read_with_credentials_from_the_environment(make_topic, producer, tmp_path, mech):
    topic = make_topic()
    stream_to(producer, topic, windows(0, 1))
    a = adapter(tmp_path, topic, sasl={"mechanism": mech, "username_env": "EDGE_KAFKA_USER",
                                       "password_env": "EDGE_KAFKA_PASSWORD"})
    assert len(drain(a, ROWS)) == ROWS
    a.close()


def test_wrong_password_is_reported_never_raised_and_secrets_never_leak(make_topic, producer, tmp_path, capfd, caplog):
    topic = make_topic()
    stream_to(producer, topic, windows(0, 1))
    bad_env = {**ADAPTER_ENV, "EDGE_KAFKA_PASSWORD": "wrong-pass-sentinel-31337"}
    cfg = make_cfg(tmp_path)
    a = build_sources(spec(topic), cfg.state_dir, "prediction", env=bad_env)[0]
    agent = EdgeAgent(cfg, [a], post=Collector(), clock=lambda: utc(T0 + 3 * CADENCE))
    end = time.time() + 20
    while time.time() < end and a.last_error is None:
        agent.run_once()
    assert a.last_error is not None and a.rows_delivered == 0       # _ALL_BROKERS_DOWN: every broker refused the login
    agent.close()
    good = build_sources(spec(topic), tmp_path / "good", "prediction", env=ADAPTER_ENV)[0]
    assert len(drain(good, ROWS)) == ROWS
    good.close()
    out = capfd.readouterr()
    blob = out.out + out.err + caplog.text + json.dumps(read_status(cfg.state_dir), default=str) + json.dumps(a.describe())
    for p in cfg.state_dir.rglob("*"):
        if p.is_file():
            blob += p.read_text(errors="ignore")
    for secret in SECRETS + ["wrong-pass-sentinel-31337"]:
        assert secret not in blob


def test_mutual_tls_with_a_client_certificate_and_an_encrypted_key(make_topic, producer, tmp_path):
    topic = make_topic(acl=False)                                 # CN=edge-client is a super user
    stream_to(producer, topic, windows(0, 1))
    base = dict(bootstrap_servers=KENV["EDGE_KAFKA_TLS_BOOTSTRAP"], sasl=None)
    a = adapter(tmp_path, topic, tls={"ca_file": CA, "cert_file": f"{DIR}/client.pem", "key_file": f"{DIR}/client.key"},
                **base)
    assert len(drain(a, ROWS)) == ROWS
    a.close()
    b = adapter(tmp_path, topic, state="st2",
                tls={"ca_file": CA, "cert_file": f"{DIR}/client.pem", "key_file": f"{DIR}/client-enc.key",
                     "key_password_env": "EDGE_KAFKA_KEYPW"}, **base)
    assert len(drain(b, ROWS)) == ROWS
    b.close()
    nocert = adapter(tmp_path, topic, state="st3", tls={"ca_file": CA}, **base)
    assert settle(nocert, 5) == [] and nocert.last_error is not None      # the listener requires a client certificate
    nocert.close()


def test_read_only_acl_is_enough_and_a_missing_grant_is_reported(make_topic, producer, tmp_path):
    topic = make_topic(acl=False)
    stream_to(producer, topic, windows(0, 1))
    a = adapter(tmp_path, topic)
    assert settle(a, 6) == [] and a.last_error is not None and "AUTHORIZATION" in a.last_error
    a.close()


# -- start position ----------------------------------------------------------------------------------------------
def test_start_at_now_skips_history_and_is_remembered_across_a_restart(make_topic, producer, tmp_path):
    topic, group = make_topic(), f"g-{secrets.token_hex(4)}"
    stream_to(producer, topic, windows(0, 1))                    # history
    a = adapter(tmp_path, topic, group=group, start_at="now")
    assert settle(a, 4) == []
    stream_to(producer, topic, windows(1, 1))
    assert len(drain(a, ROWS)) == ROWS
    a.close()                                                    # nothing was committed
    b = adapter(tmp_path, topic, group=group, start_at="now")
    got = drain(b, ROWS)
    assert len(got) == ROWS and len(settle(b, 2)) == 0           # the start was not re-resolved: no history, no more
    b.close()


def test_start_at_iso_time_uses_the_message_timestamps(make_topic, producer, tmp_path):
    topic = make_topic()
    recs = windows(0, 1)
    stream_to(producer, topic, recs)
    cutoff = T0 + 30
    expected = sum(r.ts.timestamp() >= cutoff for r in recs)
    iso = utc(cutoff).isoformat()
    a = adapter(tmp_path, topic, start_at=iso)
    got = drain(a, expected)
    assert len(got) == expected and len(settle(a, 1.5)) == 0
    a.close()


# -- back-pressure ---------------------------------------------------------------------------------------------------
def test_memory_is_bounded_by_max_uncommitted_rows_and_resumes_after_commit(make_topic, producer, tmp_path):
    topic = make_topic(2)
    n = 3000
    send(producer, topic, [(i % 2, {"scored_at": T0 + i * 0.01, "model": "m", "ticker": "t", "p": 0.5,
                                    "f1": 1.0, "f2": 2.0, "f3": 3.0, "regime": "c"}, None) for i in range(n)])
    a = adapter(tmp_path, topic, max_uncommitted_rows=1000, batch_size=100)
    got = drain(a, n, timeout=8)                                 # never committed: it must stop on its own
    assert 0 < len(got) <= 1000 and a.blocked
    assert a.describe()["rows_uncommitted"] + a.describe()["rows_buffered"] <= 1100     # the cap plus 10%: the lagging partition is still read
    assert settle(a, 1.5) == []
    total = len(got)
    end = time.time() + WAIT
    while total < n and time.time() < end:
        a.commit(None)
        total += len(a.poll())
    assert total == n
    a.close()


# -- outage -------------------------------------------------------------------------------------------------------
def test_unreachable_broker_is_reported_and_the_agent_keeps_running(tmp_path):
    a = build_sources(spec("nope", bootstrap_servers="127.0.0.1:1"), tmp_path / "st", "prediction", env=ADAPTER_ENV)[0]
    end = time.time() + 15
    while time.time() < end and a.last_error is None:
        assert a.poll() == []
    assert a.last_error in ("_TRANSPORT", "_ALL_BROKERS_DOWN", "_RESOLVE")
    assert a.describe()["last_error"] == a.last_error
    a.close()


# -- Avro ------------------------------------------------------------------------------------------------------------
AVRO = {"type": "record", "name": "Pred", "fields": [
    {"name": "scored_at", "type": "double"}, {"name": "model", "type": "string"}, {"name": "ticker", "type": "string"},
    {"name": "p", "type": "double"}, {"name": "f1", "type": "double"}, {"name": "f2", "type": "double"},
    {"name": "f3", "type": "double"}, {"name": "regime", "type": "string"}]}


class Registry:
    """The few endpoints of a schema registry the adapter uses; it can be taken down and brought back."""

    def __init__(self):
        outer = self
        self.schemas = {7: json.dumps(AVRO)}
        self.up = True

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if not outer.up:
                    self.send_response(503)
                    self.end_headers()
                    return
                sid = int(self.path.rsplit("/", 1)[1])
                if sid not in outer.schemas:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps({"schema": outer.schemas[sid]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/vnd.schemaregistry.v1+json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


def avro_bytes(r, sid=7, wire=True) -> bytes:
    import fastavro
    buf = io.BytesIO()
    if wire:
        buf.write(b"\x00" + struct.pack(">I", sid))
    fastavro.schemaless_writer(buf, fastavro.parse_schema(AVRO), payload(r))
    return buf.getvalue()


def test_avro_through_a_schema_registry_with_unknown_ids_counted_and_an_outage_without_loss(make_topic, producer, tmp_path):
    pytest.importorskip("fastavro")
    reg, topic = Registry(), make_topic()
    try:
        recs = windows(0, 1)
        send(producer, topic, [(0, avro_bytes(r), None) for r in recs[:20]] + [(0, avro_bytes(recs[0], sid=99), None)])
        a = adapter(tmp_path, topic, value_format="avro", schema_registry={"url": reg.url}, allow_insecure_tls=True)
        assert len(drain(a, 20)) == 20 and settle(a, 0.5) == [] and a.bad_rows == 1
        reg.up = False
        reg.schemas[8] = json.dumps(AVRO)
        send(producer, topic, [(0, avro_bytes(r, sid=8), None) for r in recs[20:]])
        assert settle(a, 3) == [] and a.last_error is not None and a.last_error.startswith("schema_registry_")
        reg.up = True
        rest = drain(a, 20)
        assert len(rest) == 20 and [r.ts for r in rest] == [r.ts for r in recs[20:]]       # nothing skipped, nothing twice
        a.close()
    finally:
        reg.close()


def test_avro_without_a_registry_uses_the_reader_schema_file(make_topic, producer, tmp_path):
    pytest.importorskip("fastavro")
    topic = make_topic()
    (tmp_path / "pred.avsc").write_text(json.dumps(AVRO))
    send(producer, topic, [(0, avro_bytes(r, wire=False), None) for r in windows(0, 1)])
    a = adapter(tmp_path, topic, value_format="avro", avro_schema_file=str(tmp_path / "pred.avsc"))
    assert len(drain(a, ROWS)) == ROWS
    a.close()


# -- outcomes ------------------------------------------------------------------------------------------------------------
def test_outcome_topic_joins_labels_to_predictions(make_topic, producer, tmp_path):
    t_pred, t_out = make_topic(2), make_topic()
    recs = [type(r)(r.ts, r.model_id, r.instrument, r.features, 0.8, f"id{i}") for i, r in enumerate(windows(0, 1))]
    send(producer, t_pred, [(i % 2, payload(r, pid=r.prediction_id), r.ts.timestamp() * 1000) for i, r in enumerate(recs)])
    pred = spec(t_pred)
    pred["columns"] = {**pred["columns"], "prediction_id": "pid"}
    outs = spec(t_out)
    outs["columns"] = {"model_id": "model", "prediction_id": "pid", "outcome": "label", "outcome_observed_at": "known_at"}
    cfg = make_cfg(tmp_path, output_kind="probability")
    taken: set[str] = set()
    sources = build_sources(pred, cfg.state_dir, "prediction", taken=taken, env=ADAPTER_ENV)
    labels = build_sources(outs, cfg.state_dir, "outcome", taken=taken, env=ADAPTER_ENV)
    sink = Collector()
    agent = EdgeAgent(cfg, sources, post=sink, outcome_sources=labels, clock=lambda: utc(T0 + 59))
    assert pump(agent, lambda: sources[0].rows_delivered == ROWS)          # labels arrive after their predictions
    labelled = [(0, {"model": r.model_id, "pid": r.prediction_id, "label": 1, "known_at": T0 + 58.9}, None)
                for r in recs]
    send(producer, t_out, labelled + [(0, {"model": LOCAL_MODEL, "pid": "pending", "label": None, "known_at": T0}, None)])
    ok = pump(agent, lambda: agent.outcomes.status()["matched"] == ROWS)
    assert ok, (agent.outcomes.status(), labels[0].describe(), agent.outcome_rows_rejected)
    agent.close()
    assert labels[0].bad_rows == 0 and labels[0].kind == "outcome"


# -- --check / --status -----------------------------------------------------------------------------------------------------
def agent_json(tmp_path, source):
    cfg = {"tenant_id": "t", "install_id": "i", "models": {LOCAL_MODEL: "mm-wire-0001"},
           "secret_env": "EDGE_TEST_SECRET", "endpoints": ["https://x.invalid"],
           "state_dir": str(tmp_path / "state"), "cadence_seconds": CADENCE, "output_kind": "signed",
           "source": source}
    p = tmp_path / "agent.json"
    p.write_text(json.dumps(cfg))
    return p


def run_check(tmp_path, monkeypatch, capsys, source):
    monkeypatch.setenv("EDGE_KAFKA_USER", ADAPTER_ENV["EDGE_KAFKA_USER"])
    monkeypatch.setenv("EDGE_KAFKA_PASSWORD", ADAPTER_ENV["EDGE_KAFKA_PASSWORD"])
    monkeypatch.setattr("sys.argv", ["edge_agent", "--config", str(agent_json(tmp_path, source)), "--check"])
    with pytest.raises(SystemExit) as done:
        cli.main()
    return done.value.code, json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("start", ["beginning", "now"])
def test_check_samples_without_joining_the_group_committing_or_writing_state(
        admin, make_topic, producer, tmp_path, monkeypatch, capsys, start):
    topic, group = make_topic(2), f"g-{secrets.token_hex(4)}"
    stream_to(producer, topic, windows(0, 2), 2)
    code, out = run_check(tmp_path, monkeypatch, capsys, spec(topic, group=group, start_at=start))
    assert out["source_check"]["valid_shape"] == 2 * ROWS and out["source_check"]["invalid_shape"] == 0
    assert not (tmp_path / "state").exists()
    assert group_offsets(admin, group) == {}
    assert group not in {g.group_id for g in admin.list_consumer_groups().result(30).valid}
    blob = json.dumps(out)
    for private in SECRETS + [KENV["EDGE_KAFKA_USER"], FEATURES[0], "INSTR_SENTINEL"]:
        assert private not in blob


def test_check_is_bounded_by_the_limit(make_topic, producer, tmp_path):
    topic = make_topic(2)
    stream_to(producer, topic, windows(0, 4), 2)
    a = adapter(tmp_path, topic)
    s = a.sample(50)
    assert 50 <= len(s.records) + s.bad < 50 + 500            # one consume batch of slack at most
    assert a.sample(10).records and not (tmp_path / "st").exists()


@pytest.mark.parametrize("patch,needle", [
    ({"topics": ["no_such_topic_zz"]}, "topic cannot be read"),
    ({"bootstrap_servers": "127.0.0.1:1"}, "cannot reach the Kafka brokers"),
    ({"tls": {"ca_file": "/etc/hostname"}}, "rejected the configuration"),
])
def test_check_turns_source_errors_into_a_clear_verdict(make_topic, tmp_path, monkeypatch, capsys, patch, needle):
    topic = make_topic()
    s = spec(topic, check_timeout_seconds=5)
    s.update(patch)
    code, out = run_check(tmp_path, monkeypatch, capsys, s)
    assert code == 2 and out["verdict"] == "NOT_READY" and needle in out["config_error"]
    assert not any(x in json.dumps(out) for x in SECRETS)


def test_check_reports_a_missing_grant(make_topic, tmp_path, monkeypatch, capsys):
    topic = make_topic(acl=False)
    code, out = run_check(tmp_path, monkeypatch, capsys, spec(topic))
    assert code == 2 and out["verdict"] == "NOT_READY" and "topic cannot be read" in out["config_error"]


def test_status_shows_committed_partitions_and_lag_without_topic_names_or_values(make_topic, producer, tmp_path):
    topic = make_topic(2)
    stream_to(producer, topic, windows(0, 3), 2)
    cfg = make_cfg(tmp_path)
    clock = [utc(T0 + 3 * CADENCE + 30)]
    agent = EdgeAgent(cfg, [build_sources(spec(topic, name="warehouse"), cfg.state_dir, "prediction",
                                          env=ADAPTER_ENV)[0]], post=Collector(), clock=lambda: clock[0])
    assert pump(agent, lambda: agent.sources[0].rows_committed == 3 * ROWS)
    settle(agent.sources[0], 2)
    agent.write_status(force=True)
    report = read_status(cfg.state_dir)
    (src,) = report["sources"]
    assert src["name"] == "warehouse" and src["type"] == "kafka" and src["partitions_committed"] == 2
    snap = report["agent_snapshot"]["sources"][0]
    assert snap["rows_committed"] == 3 * ROWS and snap["caught_up"] is True and snap["lag_seconds"] == 0.0
    assert snap["lag_messages"] == 0 and snap["partitions_assigned"] == 2 and snap["last_error"] is None
    blob = json.dumps(report, default=str)
    assert topic not in blob and not any(x in blob for x in SECRETS)
    agent.close()


# -- outage of the broker itself: keep last, it restarts the shared test broker -------------------------------------
@pytest.mark.skipif(not CONTAINER, reason="needs EDGE_KAFKA_TEST_CONTAINER")
def test_broker_outage_mid_run_resumes_without_loss_or_repeat(make_topic, producer, tmp_path):
    topic, group = make_topic(), f"g-{secrets.token_hex(4)}"
    stream_to(producer, topic, windows(0, 1))
    a = adapter(tmp_path, topic, group=group, session_timeout_ms=30_000)
    assert len(drain(a, ROWS, commit=True)) == ROWS
    subprocess.run(["podman", "stop", "-t", "5", CONTAINER], check=True, capture_output=True)
    try:
        end = time.time() + 20
        while time.time() < end and a.last_error is None:
            assert a.poll() == []
            a.commit(None)
        assert a.last_error is not None and a.describe()["last_error"] == a.last_error
    finally:
        subprocess.run(["podman", "start", CONTAINER], check=True, capture_output=True)
    end = time.time() + 90
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", int(KENV["EDGE_KAFKA_BOOTSTRAP"].split(":")[1])), 1).close()
            break
        except OSError:
            time.sleep(1)
    time.sleep(8)
    p2 = Producer(admin_conf())
    stream_to(p2, topic, windows(1, 1))
    got = drain(a, ROWS, timeout=90, commit=True)
    assert len(got) == ROWS and [r.ts.timestamp() for r in got] == sorted(r.ts.timestamp() for r in got)
    assert min(r.ts.timestamp() for r in got) >= T0 + CADENCE      # only the new window: nothing re-emitted
    end = time.time() + 20
    while time.time() < end and a.last_error is not None:
        a.poll()
    assert a.last_error is None
    a.close()


# -- a caught-up group is healthy, not "no rows yet" (pre-tag review, edgefix2 2026-10-30) ------------------------------
def test_check_of_a_caught_up_group_samples_the_latest_messages_and_a_fresh_empty_topic_has_no_rows(
        admin, make_topic, producer, tmp_path):
    import dataclasses
    topic, group = make_topic(2), f"g-{secrets.token_hex(4)}"
    stream_to(producer, topic, windows(0, 2), 2)
    cfg = make_cfg(tmp_path)
    clock = [utc(T0 + 2 * CADENCE + 30)]
    agent = EdgeAgent(cfg, [build_sources(spec(topic, group=group), cfg.state_dir, "prediction", env=ADAPTER_ENV)[0]],
                      post=Collector(), clock=lambda: clock[0])
    assert pump(agent, lambda: agent.sources[0].rows_delivered >= 2 * ROWS)
    agent.tick()
    agent.close(keep_open=True)                                  # the service's stop: everything committed
    assert wait_offsets(admin, group, lambda o: sum(o.values()) == 2 * ROWS) == {(topic, 0): ROWS, (topic, 1): ROWS}
    src = adapter(tmp_path, topic, state="state", group=group)
    s = src.sample(20)
    assert 20 <= len(s.records) + s.bad and s.bad == 0           # the newest messages, not "no rows yet"
    assert group_offsets(admin, group) == {(topic, 0): ROWS, (topic, 1): ROWS}   # the check committed nothing

    empty = make_topic(1)
    fcfg = dataclasses.replace(make_cfg(tmp_path / "fresh"), output_kind="probability")
    fresh = build_sources(spec(empty), fcfg.state_dir, "prediction", env=ADAPTER_ENV)
    assert cli.check_source(fcfg, adapters=fresh)["verdict"] == "NO_ROWS_YET"   # run 1 of a fresh install
