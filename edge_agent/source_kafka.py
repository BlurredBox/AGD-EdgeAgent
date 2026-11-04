"""Read-only Kafka input (predictions or outcomes): one consumer group per edge agent.

    "source": {"type": "kafka", "bootstrap_servers": "broker1:9093,broker2:9093",
               "topics": ["model-predictions"], "group_id": "blurredbox-edge-credit",
               "tls": {"ca_file": "/opt/blurredbox/certs/kafka-ca.pem"},
               "sasl": {"mechanism": "SCRAM-SHA-512",
                        "username_env": "EDGE_KAFKA_USER", "password_env": "EDGE_KAFKA_PASSWORD"},
               "value_format": "json", "start_at": "now",
               "columns": {"ts": "scored_at", "model_id": "model", "instrument": "ticker",
                           "output": "p", "features": ["x1", "x2"]}}

Guarantees
  * Offsets are committed to the broker only from `commit(before)`, i.e. after every
    record up to them sits in a closed window that is already in the durable spool.
    `enable.auto.commit` is off; nothing else ever commits.
  * Records are released to the agent in time order across partitions: a record is held
    back until every partition that still has a backlog has been read at least as far in
    time (watermark = minimum over the partitions that are not caught up). Inside a
    partition the order is the offset order.
  * A rebalance drops only what was read but not yet handed over; what was handed over
    stays staged and is committed later if the partition is still ours.
  * TLS is on unless "allow_insecure_tls" is true. SASL user names, passwords, client
    secrets and key passwords are read from environment variables named in the config;
    a literal one in the config is refused. Nothing secret is logged or reported.
  * A broker outage never stops the agent: `last_error` is reported, reading resumes
    from the last committed offsets.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import structlog

from edge_agent.adapters import Sample
from edge_agent.checkpoint import CheckpointedAdapter, read_json, write_json_atomic
from edge_agent.mapping import (ColumnMapping, ConfigError, check_keys, epoch_of, int_in,
                                num_in)

log = structlog.get_logger("edge_agent.source")

ALLOWED = {"type", "name", "bootstrap_servers", "topics", "group_id", "tls", "sasl", "value_format",
           "schema_registry", "avro_schema_file", "columns", "constants", "start_at",
           "allow_insecure_tls", "max_uncommitted_rows", "batch_size", "poll_timeout_seconds",
           "idle_partition_seconds", "session_timeout_ms", "check_timeout_seconds", "commit_chunk_rows"}
_TLS_KEYS = {"ca_file", "cert_file", "key_file", "key_password_env"}
_SASL_KEYS = {"mechanism", "username_env", "password_env", "oidc"}
_OIDC_KEYS = {"token_endpoint_url", "client_id_env", "client_secret_env", "scope"}
_REGISTRY_KEYS = {"url", "ca_file", "username_env", "password_env"}
_MECHANISMS = ("PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512", "OAUTHBEARER")
_SECRET_WORDS = ("password", "secret", "jaas", "credential")
_SECRET_KEYS = ("token", "access_token", "bearer_token", "api_key", "apikey")
_TRANSPORT_ERRORS = ("_ALL_BROKERS_DOWN", "_TRANSPORT", "_RESOLVE", "_SSL", "_AUTHENTICATION")
_AVRO_MAGIC = 0
Key = tuple[str, int]


def _lib():
    try:
        import confluent_kafka
    except ImportError as exc:       # optional extra
        raise ConfigError("the kafka source needs the optional dependency confluent-kafka: "
                          "pip install -r requirements-edge-kafka.txt") from exc
    return confluent_kafka


def _refuse_literal_secrets(obj: Any, where: str) -> None:
    """A key that holds a secret must be spelled `*_env` (the name of an environment variable)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            low = str(k).lower()
            if (low in _SECRET_KEYS or any(w in low for w in _SECRET_WORDS)) and not low.endswith("_env"):
                raise ConfigError(f"{where}.{k}: credentials must not be in the config file; "
                                  f"use {k}_env (the name of an environment variable)")
            if low in ("username", "user", "sasl_username"):
                raise ConfigError(f"{where}.{k}: use username_env (the name of an environment variable)")
            _refuse_literal_secrets(v, f"{where}.{k}")


def _env_name(spec: dict, key: str, where: str, required: bool = True) -> str | None:
    v = spec.get(key)
    if v is None and not required:
        return None
    if not isinstance(v, str) or not v:
        raise ConfigError(f"{where}.{key} is required: the name of an environment variable")
    return v


def _file(spec: dict, key: str, where: str) -> str | None:
    v = spec.get(key)
    if v is None:
        return None
    if not isinstance(v, str) or not v:
        raise ConfigError(f"{where}.{key} must be a file path")
    if not Path(v).is_file():
        raise ConfigError(f"{where}.{key}: file not found or not a regular file")
    return v


class _Decoder:
    """Message value -> flat dict. JSON, or Avro (Confluent wire format via a schema registry,
    or plain Avro binary with a reader schema file)."""

    class Transient(Exception):
        """The schema registry could not be reached: try this message again later."""

    def __init__(self, fmt: str, registry: dict | None, schema_file: str | None,
                 env: Mapping[str, str]) -> None:
        self.fmt, self._registry, self._env = fmt, registry, env
        self._schemas: dict[int, Any] = {}
        self._reader = None
        self._http = None
        if fmt == "avro":
            self._fastavro = self._import_fastavro()
            if schema_file:
                self._reader = self._fastavro.parse_schema(json.loads(Path(schema_file).read_text()))

    @staticmethod
    def _import_fastavro():
        try:
            import fastavro
        except ImportError as exc:
            raise ConfigError("the kafka source with value_format 'avro' needs the optional dependency "
                              "fastavro: pip install -r requirements-edge-kafka.txt") from exc
        return fastavro

    def _client(self):
        if self._http is None:
            import httpx
            auth = None
            if self._registry.get("username_env"):
                auth = (self._env[self._registry["username_env"]], self._env[self._registry["password_env"]])
            self._http = httpx.Client(base_url=self._registry["url"].rstrip("/"), timeout=5.0, auth=auth,
                                      verify=self._registry.get("ca_file") or True,
                                      headers={"Accept": "application/vnd.schemaregistry.v1+json"})
        return self._http

    def _schema(self, sid: int):
        got = self._schemas.get(sid)
        if got is not None:
            return got
        import httpx
        try:
            r = self._client().get(f"/schemas/ids/{sid}")
        except httpx.HTTPError as exc:
            raise self.Transient(type(exc).__name__) from None
        if r.status_code in (404, 400):
            raise ValueError("unknown schema id")
        if r.status_code >= 500 or r.status_code in (401, 403, 408, 429):
            raise self.Transient(f"HTTP{r.status_code}")
        r.raise_for_status()
        parsed = self._fastavro.parse_schema(json.loads(r.json()["schema"]))
        if len(self._schemas) > 1000:
            self._schemas.clear()
        self._schemas[sid] = parsed
        return parsed

    def decode(self, value: bytes | None) -> dict:
        if value is None:
            raise ValueError("tombstone")
        if self.fmt == "json":
            obj = json.loads(value)
        else:
            import io
            if self._reader is not None:
                obj = self._fastavro.schemaless_reader(io.BytesIO(value), self._reader)
            else:
                if len(value) < 6 or value[0] != _AVRO_MAGIC:
                    raise ValueError("not in the schema registry wire format")
                sid = int.from_bytes(value[1:5], "big")
                obj = self._fastavro.schemaless_reader(io.BytesIO(value[5:]), self._schema(sid))
        if not isinstance(obj, dict):
            raise TypeError("message value must be an object")
        return obj


class KafkaAdapter(CheckpointedAdapter):
    def __init__(self, spec: Mapping[str, Any], state_dir: str | Path, *, kind: str = "prediction",
                 name: str | None = None, env: Mapping[str, str] | None = None,
                 clock=time.time, monotonic=time.monotonic) -> None:
        _lib()                              # optional extra: say so before any other complaint
        where = "source" if kind == "prediction" else "outcome_source"
        spec = dict(spec)
        _refuse_literal_secrets(spec, where)
        check_keys(spec, ALLOWED, where)
        self.kind, self._env = kind, os.environ if env is None else env
        name = name or spec.get("name") or ("predictions" if kind == "prediction" else "outcomes")
        super().__init__(Path(state_dir), name, clock=clock,
                         max_uncommitted_rows=int_in(spec, "max_uncommitted_rows", 1_000_000,
                                                     1_000, 1_000_000_000, where))
        self._mono = monotonic
        self.mapping = ColumnMapping(spec.get("columns"), spec.get("constants"), kind=kind,
                                     allow_pattern_features=True)
        self._ts_from_kafka = "__kafka_timestamp" in self.mapping.cols.values()
        self.topics = self._topics(spec, where)
        self.group_id = spec.get("group_id", f"blurredbox-edge-{name}")
        if not isinstance(self.group_id, str) or not self.group_id or len(self.group_id) > 200:
            raise ConfigError(f"{where}.group_id must be a nonempty string")
        self.batch_size = int_in(spec, "batch_size", 2_000, 1, 100_000, where)
        self.poll_timeout = num_in(spec, "poll_timeout_seconds", 0.2, 0.0, 5.0, where)
        self.chunk_rows = int_in(spec, "commit_chunk_rows", 100, 1, 100_000, where)
        self.idle_seconds = num_in(spec, "idle_partition_seconds", 30.0, 1.0, 3600.0, where)
        self.session_timeout_ms = int_in(spec, "session_timeout_ms", 30_000, 6_000, 300_000, where)
        self.check_timeout = num_in(spec, "check_timeout_seconds", 15.0, 1.0, 300.0, where)
        self._start = self._parse_start(spec.get("start_at"), where)
        self._servers = self._bootstrap(spec, where)
        self._security = self._security_conf(spec, where)       # holds secrets: never logged or reported
        self.decoder = self._decoder(spec, where)
        # -- runtime state ---------------------------------------------------------------
        saved = read_json(self.state_file) or {}
        self._started = bool(saved.get("started"))
        self._start_offsets: dict[str, int] = dict(saved.get("start") or {})
        self._committed: dict[Key, int] = {self._parse_key(k): int(v)
                                           for k, v in (saved.get("committed") or {}).items()}
        self._broker_committed: dict[Key, int] = {}
        self._inflight: dict[Key, int] = {}
        self._consumer = None
        self._assigned: set[Key] = set()
        self._buf: dict[Key, deque] = {}
        self._buffered = 0
        self._seen_max: dict[Key, float] = {}
        self._fetch_pos: dict[Key, int] = {}
        self._next_delivered: dict[Key, int] = {}
        self._last_msg: dict[Key, float] = {}
        self._paused: set[Key] = set()
        self._delivered_ts: float | None = None
        self.rebalances = 0
        self.caught_up = False
        self.lag_messages: int | None = None
        self._credentials()                 # fail at start-up, not at the first poll

    # -- config -----------------------------------------------------------------------------
    @staticmethod
    def _parse_key(text: str) -> Key:
        topic, _, part = text.rpartition("/")
        return topic, int(part)

    @staticmethod
    def _key_text(key: Key) -> str:
        return f"{key[0]}/{key[1]}"

    def _topics(self, spec: dict, where: str) -> list[str]:
        t = spec.get("topics")
        if not isinstance(t, list) or not t or not all(isinstance(x, str) and x for x in t):
            raise ConfigError(f"{where}.topics is required: a nonempty list of topic names")
        if len(set(t)) != len(t) or any(x.startswith("^") or "*" in x or "/" in x for x in t):
            raise ConfigError(f"{where}.topics must be exact topic names without repeats "
                              "(patterns are not supported)")
        return list(t)

    def _bootstrap(self, spec: dict, where: str) -> str:
        b = spec.get("bootstrap_servers")
        if isinstance(b, list):
            b = ",".join(b)
        if not isinstance(b, str) or not b.strip() or any(
                ":" not in h.strip() for h in b.split(",")):
            raise ConfigError(f"{where}.bootstrap_servers is required: \"host:port[,host:port...]\"")
        return b

    def _parse_start(self, v: Any, where: str) -> Any:
        if v in ("beginning", "now"):
            return v
        if isinstance(v, str):
            try:
                d = datetime.fromisoformat(v.replace("Z", "+00:00"))
                return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        raise ConfigError(f"{where}.start_at is required: \"beginning\" (read the whole retained history), "
                          "\"now\" (only messages from the first start on) or an ISO-8601 time")

    def _security_conf(self, spec: dict, where: str) -> dict[str, Any]:
        insecure = spec.get("allow_insecure_tls") is True
        tls = spec.get("tls", {})
        sasl = spec.get("sasl")
        conf: dict[str, Any] = {}
        if tls is False:
            if not insecure:
                raise ConfigError(f"{where}.tls is false: Kafka traffic would be unencrypted; set "
                                  "\"allow_insecure_tls\": true to accept that (not recommended)")
            plain = True
        elif isinstance(tls, dict):
            check_keys(tls, _TLS_KEYS, f"{where}.tls")
            plain = False
            ca = _file(tls, "ca_file", f"{where}.tls")
            cert, key = _file(tls, "cert_file", f"{where}.tls"), _file(tls, "key_file", f"{where}.tls")
            if bool(cert) != bool(key):
                raise ConfigError(f"{where}.tls.cert_file and key_file go together (client certificate)")
            if ca:
                conf["ssl.ca.location"] = ca
            else:
                log.info("kafka_tls_uses_default_root_store")
            if cert:
                conf["ssl.certificate.location"], conf["ssl.key.location"] = cert, key
            self._key_pw_env = _env_name(tls, "key_password_env", f"{where}.tls", required=False)
            conf["enable.ssl.certificate.verification"] = True
            conf["ssl.endpoint.identification.algorithm"] = "https"
        else:
            raise ConfigError(f"{where}.tls must be an object (ca_file, cert_file, key_file) or false")
        if not isinstance(tls, dict):
            self._key_pw_env = None
        self._sasl = None
        if sasl is None:
            conf["security.protocol"] = "PLAINTEXT" if plain else "SSL"
        else:
            if not isinstance(sasl, dict):
                raise ConfigError(f"{where}.sasl must be an object")
            check_keys(sasl, _SASL_KEYS, f"{where}.sasl")
            if plain:
                raise ConfigError(f"{where}: SASL over a plaintext connection would send credentials in "
                                  "the clear; TLS cannot be turned off together with SASL")
            mech = sasl.get("mechanism")
            if mech not in _MECHANISMS:
                raise ConfigError(f"{where}.sasl.mechanism must be one of {list(_MECHANISMS)}")
            conf["security.protocol"], conf["sasl.mechanism"] = "SASL_SSL", mech
            if mech == "OAUTHBEARER":
                oidc = sasl.get("oidc")
                if not isinstance(oidc, dict):
                    raise ConfigError(f"{where}.sasl.oidc is required for OAUTHBEARER "
                                      "(token_endpoint_url, client_id_env, client_secret_env)")
                check_keys(oidc, _OIDC_KEYS, f"{where}.sasl.oidc")
                url = oidc.get("token_endpoint_url")
                if not isinstance(url, str) or not url.startswith("https://"):
                    raise ConfigError(f"{where}.sasl.oidc.token_endpoint_url must be an https:// URL")
                self._sasl = {"mech": mech, "id": _env_name(oidc, "client_id_env", f"{where}.sasl.oidc"),
                              "secret": _env_name(oidc, "client_secret_env", f"{where}.sasl.oidc")}
                conf.update({"sasl.oauthbearer.method": "oidc", "sasl.oauthbearer.token.endpoint.url": url})
                if oidc.get("scope"):
                    conf["sasl.oauthbearer.scope"] = str(oidc["scope"])
            else:
                self._sasl = {"mech": mech, "id": _env_name(sasl, "username_env", f"{where}.sasl"),
                              "secret": _env_name(sasl, "password_env", f"{where}.sasl")}
        return conf

    def _decoder(self, spec: dict, where: str) -> _Decoder:
        fmt = spec.get("value_format", "json")
        if fmt not in ("json", "avro"):
            raise ConfigError(f"{where}.value_format must be 'json' or 'avro'")
        reg, schema_file = spec.get("schema_registry"), spec.get("avro_schema_file")
        if fmt == "json" and (reg or schema_file):
            raise ConfigError(f"{where}: schema_registry / avro_schema_file apply to value_format 'avro' only")
        if fmt == "avro":
            if bool(reg) == bool(schema_file):
                raise ConfigError(f"{where}: value_format 'avro' needs exactly one of schema_registry "
                                  "(Confluent wire format) or avro_schema_file (plain Avro binary)")
            if schema_file:
                schema_file = _file(spec, "avro_schema_file", where)
        if reg:
            if not isinstance(reg, dict):
                raise ConfigError(f"{where}.schema_registry must be an object")
            check_keys(reg, _REGISTRY_KEYS, f"{where}.schema_registry")
            url = reg.get("url")
            if not isinstance(url, str) or not url.startswith(("https://", "http://")):
                raise ConfigError(f"{where}.schema_registry.url is required")
            if url.startswith("http://") and spec.get("allow_insecure_tls") is not True:
                raise ConfigError(f"{where}.schema_registry.url is plain http; use https or set "
                                  "\"allow_insecure_tls\": true")
            _file(reg, "ca_file", f"{where}.schema_registry")
            if ("username_env" in reg) != ("password_env" in reg):
                raise ConfigError(f"{where}.schema_registry.username_env and password_env go together")
            for k in ("username_env", "password_env"):
                if k in reg and not self._env.get(_env_name(reg, k, f"{where}.schema_registry")):
                    raise ConfigError(f"environment variable {reg[k]} is not set")
        return _Decoder(fmt, reg, schema_file, self._env)

    def _credentials(self) -> dict[str, str]:
        """Resolve every secret from the environment (raises ConfigError naming the variable)."""
        out: dict[str, str] = {}

        def need(var: str) -> str:
            v = self._env.get(var)
            if not v:
                raise ConfigError(f"environment variable {var} is not set")
            return v
        if self._sasl:
            if self._sasl["mech"] == "OAUTHBEARER":
                out["sasl.oauthbearer.client.id"] = need(self._sasl["id"])
                out["sasl.oauthbearer.client.secret"] = need(self._sasl["secret"])
            else:
                out["sasl.username"], out["sasl.password"] = need(self._sasl["id"]), need(self._sasl["secret"])
        if self._key_pw_env:
            out["ssl.key.password"] = need(self._key_pw_env)
        return out

    # -- consumer -----------------------------------------------------------------------------
    def _conf(self, *, callbacks: bool) -> dict[str, Any]:
        conf = {"bootstrap.servers": self._servers, "group.id": self.group_id,
                "client.id": "blurredbox-edge-agent", "enable.auto.commit": False,
                "enable.auto.offset.store": False, "auto.offset.reset": "earliest",
                "partition.assignment.strategy": "range", "session.timeout.ms": self.session_timeout_ms,
                "queued.max.messages.kbytes": 16_384, "queued.min.messages": 1_000,
                "fetch.message.max.bytes": 1_048_576, "log.connection.close": False,
                **self._security, **self._credentials()}
        if callbacks:
            conf.update({"error_cb": self._on_error, "stats_cb": self._on_stats,
                         "statistics.interval.ms": 5_000, "on_commit": self._on_commit})
        return conf

    def _ensure_consumer(self):
        if self._consumer is None:
            ck = _lib()
            try:
                self._consumer = ck.Consumer(self._conf(callbacks=True))
            except (ck.KafkaException, ValueError, TypeError) as exc:
                raise ConfigError(f"the Kafka client rejected the configuration ({type(exc).__name__}); "
                                  "check the tls file paths and the sasl block") from None
            self._consumer.subscribe(self.topics, on_assign=self._on_assign,
                                     on_revoke=self._on_revoke, on_lost=self._on_revoke)
        return self._consumer

    def close(self) -> None:
        c, self._consumer = self._consumer, None
        if c is not None:
            try:
                self._flush_broker(c, sync=True)
                c.close()
            except Exception:                # noqa: BLE001
                pass
        self._assigned.clear()
        self._buf.clear()
        self._buffered = 0

    # -- librdkafka callbacks (run inside consume()) ---------------------------------------------
    def _on_error(self, err) -> None:
        self.last_error, self.last_error_at = err.name(), self._clock()

    def _on_stats(self, payload: str) -> None:
        try:
            brokers = json.loads(payload).get("brokers", {})
        except ValueError:
            return
        if self.last_error in _TRANSPORT_ERRORS and any(
                b.get("state") == "UP" for b in brokers.values() if b.get("nodeid", -1) >= 0):
            self.last_error = None

    def _on_commit(self, err, partitions) -> None:
        for tp in partitions:
            key = (tp.topic, tp.partition)
            self._inflight.pop(key, None)
            if err is None and tp.offset >= 0:
                self._broker_committed[key] = max(self._broker_committed.get(key, -1), tp.offset)
        if err is not None:
            self.last_error, self.last_error_at = err.name(), self._clock()

    def _on_revoke(self, consumer, partitions) -> None:
        """Drop what was read but never handed over; it is re-read by whoever owns it next.
        What was handed over stays staged (it is in windows) and is committed if we get it back."""
        self.rebalances += 1
        self._commit_before_losing(consumer, partitions)
        self._assigned.clear()
        self._buf.clear()
        self._buffered = 0
        self._paused.clear()
        self._seen_max.clear()
        self._fetch_pos.clear()
        self._last_msg.clear()
        self._inflight.clear()
        self.caught_up = False

    def _commit_before_losing(self, consumer, partitions) -> None:
        """Positions that are already durable here but not yet acknowledged by the broker are sent
        synchronously before the partitions move, so the next owner starts exactly there."""
        ck = _lib()
        keys = {(t.topic, t.partition) for t in partitions}
        pending = [ck.TopicPartition(t, p, o) for (t, p), o in self._committed.items()
                   if (t, p) in keys and o > self._broker_committed.get((t, p), -1)]
        if not pending:
            return
        try:
            consumer.commit(offsets=pending, asynchronous=False)
            for tp in pending:
                self._broker_committed[(tp.topic, tp.partition)] = tp.offset
        except ck.KafkaException:
            pass                     # the next owner re-reads from the last acknowledged position

    def _on_assign(self, consumer, partitions) -> None:
        ck = _lib()
        committed = {(t.topic, t.partition): t.offset
                     for t in consumer.committed(partitions, timeout=15)}
        chosen, fresh = [], {}
        for tp in partitions:
            key = (tp.topic, tp.partition)
            off = committed.get(key, ck.OFFSET_INVALID)
            if off >= 0:
                self._broker_committed[key] = off
            else:
                off = self._start_offsets.get(self._key_text(key), -1)
                if off < 0:
                    off = self._resolve_start(consumer, tp)
                    fresh[self._key_text(key)] = off
            off = max(off, self._next_delivered.get(key, 0))      # never re-read what we already handed over
            chosen.append(ck.TopicPartition(tp.topic, tp.partition, off))
            self._fetch_pos[key] = off
            self._buf[key] = deque()
            self._assigned.add(key)
            self._last_msg[key] = self._mono()
        if fresh or not self._started:
            self._start_offsets.update(fresh)
            self._started = True
            write_json_atomic(self.state_file, self._state())
        consumer.assign(chosen)

    def _resolve_start(self, consumer, tp) -> int:
        """First read of a partition with no committed offset and no remembered start."""
        ck = _lib()
        lo, hi = consumer.get_watermark_offsets(tp, timeout=15, cached=False)
        if self._started or self._start == "beginning":
            return lo               # a partition that appears later, or a rebalance before the first commit
        if self._start == "now":
            return hi
        probe = ck.TopicPartition(tp.topic, tp.partition, int(self._start.timestamp() * 1000))
        got = consumer.offsets_for_times([probe], timeout=15)[0].offset
        return hi if got < 0 else max(lo, got)

    # -- reading -----------------------------------------------------------------------------------
    @property
    def blocked(self) -> bool:
        return self._staged_rows + self._buffered >= self.max_uncommitted_rows

    def _hw_lag(self, consumer) -> None:
        """Refresh caught-up / lag from the cached high watermarks (no network)."""
        ck = _lib()
        lag, behind = 0, False
        for key in self._assigned:
            try:
                _, hi = consumer.get_watermark_offsets(ck.TopicPartition(*key), cached=True)
            except ck.KafkaException:
                hi = -1
            if hi >= 0:
                lag += max(0, hi - self._fetch_pos.get(key, hi))
            if self._lagging(key, hi):
                behind = True
        self.lag_messages = lag if self._assigned else None
        self.caught_up = bool(self._assigned) and not behind and not any(self._buf.values())

    def _lagging(self, key: Key, hi: int) -> bool:
        if hi >= 0 and self._fetch_pos.get(key, 0) >= hi:
            return False
        # no message for a long time although the broker reports more: control records or an
        # unreachable leader; do not let one such partition hold every other partition back
        return self._mono() - self._last_msg.get(key, 0.0) < self.idle_seconds

    def _watermark(self, consumer) -> float:
        ck = _lib()
        wm = float("inf")
        for key in self._assigned:
            try:
                _, hi = consumer.get_watermark_offsets(ck.TopicPartition(*key), cached=True)
            except ck.KafkaException:
                hi = -1
            if self._lagging(key, hi):
                wm = min(wm, self._seen_max.get(key, float("-inf")))
        return wm

    def _throttle(self, consumer) -> None:
        """Keep calling consume() (group liveness) but stop fetching while too much is held."""
        ck = _lib()
        if self.blocked:
            keep = None
            slack = max(self.batch_size, self.max_uncommitted_rows // 10)
            if (self._staged_rows < self.max_uncommitted_rows and self._assigned
                    and self._staged_rows + self._buffered < self.max_uncommitted_rows + slack):
                # the partition furthest behind in time must still be read, or the watermark never moves
                keep = min(self._assigned, key=lambda k: self._seen_max.get(k, float("-inf")))
            todo = [k for k in self._assigned if k != keep and k not in self._paused]
            if todo:
                consumer.pause([ck.TopicPartition(*k) for k in todo])
                self._paused.update(todo)
            if keep in self._paused:
                consumer.resume([ck.TopicPartition(*keep)])
                self._paused.discard(keep)
        elif self._paused:
            consumer.resume([ck.TopicPartition(*k) for k in self._paused if k in self._assigned])
            self._paused.clear()

    def _ingest(self, msg) -> None:
        key = (msg.topic(), msg.partition())
        if key not in self._assigned:
            return
        self._last_msg[key] = self._mono()
        self._fetch_pos[key] = msg.offset() + 1
        row = None
        try:
            row = self.decoder.decode(msg.value())
            if self._ts_from_kafka:
                ms = msg.timestamp()[1]
                row["__kafka_timestamp"] = ms / 1000.0 if ms and ms > 0 else None
        except _Decoder.Transient:
            raise
        except Exception:                     # noqa: BLE001 -- a poison message must not stop the agent
            row = None
        rec, ts = None, None
        if row is not None:
            ts = epoch_of(row.get(self.mapping.time_column))
            try:
                rec = self.mapping.to_record(row)
            except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
                row = None
        if ts is None:
            kts = msg.timestamp()[1]
            ts = self._seen_max.get(key) if self._seen_max.get(key) is not None else (
                kts / 1000.0 if kts and kts > 0 else 0.0)
        if row is None:
            self.bad_rows += 1
        if ts > self._seen_max.get(key, float("-inf")):
            self._seen_max[key] = ts
        self._buf[key].append((ts, msg.offset(), rec))
        self._buffered += 1

    def _on_message_error(self, err) -> None:
        self.last_error, self.last_error_at = err.name(), self._clock()

    def _room(self) -> int:
        return max(1, min(self.batch_size, self.max_uncommitted_rows - self._staged_rows - self._buffered))

    def _fill(self, consumer) -> None:
        self._throttle(consumer)
        timeout = self.poll_timeout
        for _ in range(100):                  # at most ~100 batches per poll
            want = self._room()
            msgs = consumer.consume(num_messages=want, timeout=timeout)
            timeout = 0.0
            for i, msg in enumerate(msgs):
                if msg.error():
                    self._on_message_error(msg.error())
                    continue
                try:
                    self._ingest(msg)
                except _Decoder.Transient as exc:
                    self.last_error, self.last_error_at = f"schema_registry_{exc}", self._clock()
                    self._rewind(consumer, msgs[i:])
                    return
            if len(msgs) < want or self.blocked:
                return

    def _rewind(self, consumer, rest) -> None:
        """Schema registry down: re-read from the first unprocessed message of each partition."""
        ck = _lib()
        first: dict[Key, int] = {}
        for m in rest:
            if not m.error():
                first.setdefault((m.topic(), m.partition()), m.offset())
        for key, off in first.items():
            if key in self._assigned:
                consumer.seek(ck.TopicPartition(key[0], key[1], off))
                self._fetch_pos[key] = off

    def _release(self, consumer, max_records: int) -> list:
        """Hand over records in time order. They are staged in chunks of `commit_chunk_rows`, so a
        large backlog read in one poll can be committed piece by piece as its windows close."""
        max_records = min(max_records, self.max_uncommitted_rows - self._staged_rows)
        if max_records <= 0:
            return []
        wm = self._watermark(consumer)
        out, delta, rows, max_ts = [], {}, 0, None
        total = 0
        while total < max_records:
            best = None
            for key, q in self._buf.items():
                if q and q[0][0] <= wm and (best is None or q[0][0] < self._buf[best][0][0]):
                    best = key
            if best is None:
                break
            ts, off, rec = self._buf[best].popleft()
            self._buffered -= 1
            delta[self._key_text(best)] = off + 1
            self._next_delivered[best] = off + 1
            rows += 1
            total += 1
            max_ts = ts if max_ts is None else max(max_ts, ts)
            if rec is not None:
                out.append(rec)
            if rows >= self.chunk_rows:
                self._stage_chunk(delta, rows, max_ts)
                delta, rows, max_ts = {}, 0, None
        if rows:
            self._stage_chunk(delta, rows, max_ts)
        return out

    def _stage_chunk(self, delta: dict, rows: int, max_ts: float) -> None:
        self._stage(max_ts, delta, rows)
        self._delivered_ts = max(self._delivered_ts or max_ts, max_ts)

    # -- SourceAdapter ---------------------------------------------------------------------------------
    def poll(self, max_records: int = 10_000) -> list:
        out: list = []
        try:
            consumer = self._ensure_consumer()
            self._fill(consumer)
            out = self._release(consumer, max_records)
            self._hw_lag(consumer)
            self._flush_broker(consumer)
        except ConfigError:
            raise
        except Exception as exc:               # noqa: BLE001 -- a broker outage must not stop the agent
            self._fail(exc)
            self.caught_up = False
            log.warning("kafka_source_unavailable", source=self.name, error_type=type(exc).__name__)
        return out

    def commit(self, before: datetime | None = None) -> None:
        super().commit(before)
        if self._consumer is not None:
            try:
                self._flush_broker(self._consumer)
            except Exception as exc:           # noqa: BLE001
                self._fail(exc)

    def _apply(self, delta: Any) -> None:
        for text, off in delta.items():
            key = self._parse_key(text)
            if off > self._committed.get(key, 0):
                self._committed[key] = off

    def _state(self) -> dict[str, Any]:
        return {"v": 1, "type": "kafka", "started": self._started, "start": self._start_offsets,
                "committed": {self._key_text(k): v for k, v in sorted(self._committed.items())}}

    def _flush_broker(self, consumer, *, sync: bool = False) -> None:
        """Send the committed positions of the partitions we own to the broker (async; a failure
        leaves them pending and they are sent again on the next call)."""
        ck = _lib()
        pending = [ck.TopicPartition(t, p, o) for (t, p), o in self._committed.items()
                   if (t, p) in self._assigned and o > self._broker_committed.get((t, p), -1)
                   and (sync or self._inflight.get((t, p)) != o)]
        if not pending:
            return
        try:
            consumer.commit(offsets=pending, asynchronous=not sync)
        except ck.KafkaException as exc:
            err = exc.args[0]
            self.last_error, self.last_error_at = err.name() if hasattr(err, "name") else "KafkaException", \
                self._clock()
            return
        if not sync:
            for tp in pending:
                self._inflight[(tp.topic, tp.partition)] = tp.offset
        else:
            for tp in pending:
                self._broker_committed[(tp.topic, tp.partition)] = tp.offset

    # -- --check ----------------------------------------------------------------------------------------
    def sample(self, limit: int = 1000) -> Sample:
        """Read up to `limit` messages from the group's committed position (or the configured
        start; with start_at "now" the most recent ones), join no group, commit nothing, write
        nothing, then disconnect. Raises ConfigError with a clear message.

        When the group has committed and is caught up on every partition (a quiet night, after a stop) there is
        nothing after the position; the most recent messages are read instead, so a healthy quiet topic is not
        reported as having no rows. A fresh group keeps its configured start."""
        ck = _lib()
        try:
            consumer = ck.Consumer(self._conf(callbacks=False))
        except (ck.KafkaException, ValueError, TypeError):
            raise ConfigError("the Kafka client rejected the configuration; check the tls file paths "
                              "and the sasl block") from None
        try:
            tps = self._sample_partitions(consumer, ck)
            if not tps:
                return Sample([], 0)
            start, end = self._sample_positions(consumer, ck, tps, limit)
            consumer.assign([ck.TopicPartition(t, p, o) for (t, p), o in start.items()])
            wanted = {k for k in start if start[k] < end[k]}
            recs, bad = [], 0
            deadline = time.monotonic() + self.check_timeout
            while wanted and len(recs) + bad < limit and time.monotonic() < deadline:
                for msg in consumer.consume(num_messages=500, timeout=1.0):
                    if msg.error():
                        code = msg.error().name()
                        if "AUTHORIZATION" in code:
                            raise ConfigError(f"the broker refused access ({code}): the account needs READ "
                                              "on the topics and on the consumer group")
                        continue
                    key = (msg.topic(), msg.partition())
                    if msg.offset() + 1 >= end[key]:
                        wanted.discard(key)
                    if msg.offset() >= end[key]:
                        continue
                    r, ok = self._sample_one(msg)
                    if not ok:
                        bad += 1
                    elif r is not None:
                        recs.append(r)
                    if len(recs) + bad >= limit:
                        break
            return Sample(recs, bad)
        finally:
            try:
                consumer.close()
            except Exception:                  # noqa: BLE001
                pass

    def _sample_partitions(self, consumer, ck) -> list[Key]:
        out: list[Key] = []
        for topic in self.topics:
            try:
                md = consumer.list_topics(topic, timeout=15)
            except ck.KafkaException as exc:
                code = exc.args[0].name() if exc.args and hasattr(exc.args[0], "name") else type(exc).__name__
                raise ConfigError(f"cannot reach the Kafka brokers ({code}); check bootstrap_servers, TLS "
                                  "and the SASL credentials in the environment") from None
            t = md.topics.get(topic)
            if t is None or t.error is not None:
                code = t.error.name() if t is not None and t.error is not None else "UNKNOWN"
                raise ConfigError(f"topic cannot be read ({code}): it does not exist, or the account has no "
                                  "DESCRIBE/READ on it")
            out += [(topic, p) for p in sorted(t.partitions)]
        return out

    def _sample_positions(self, consumer, ck, keys: list[Key], limit: int):
        start, end, low = {}, {}, {}
        tps = [ck.TopicPartition(t, p) for t, p in keys]
        committed = {(t.topic, t.partition): t.offset for t in consumer.committed(tps, timeout=15)}
        per_part = max(1, -(-limit // len(keys)))
        for key in keys:
            lo, hi = consumer.get_watermark_offsets(ck.TopicPartition(*key), timeout=15, cached=False)
            end[key], low[key] = hi, lo
            off = committed.get(key, -1)
            if off < 0:
                if self._start == "beginning":
                    off = lo
                elif self._start == "now":
                    off = max(lo, hi - per_part)             # nothing is "after now": look at the latest
                else:
                    got = consumer.offsets_for_times(
                        [ck.TopicPartition(key[0], key[1], int(self._start.timestamp() * 1000))],
                        timeout=15)[0].offset
                    off = hi if got < 0 else max(lo, got)
            start[key] = max(lo, min(off, hi))
        # Caught up everywhere, or (kitrun6 R-02, as for PostgreSQL) a fresh group whose ISO start_at has nothing after
        # it yet: look at the latest messages. The check only tests their shape; the stream still starts at start_at.
        if all(start[k] >= end[k] for k in keys):
            for key in keys:
                start[key] = max(low[key], end[key] - per_part)
        return start, end

    def _sample_one(self, msg):
        try:
            row = self.decoder.decode(msg.value())
            if self._ts_from_kafka:
                ms = msg.timestamp()[1]
                row["__kafka_timestamp"] = ms / 1000.0 if ms and ms > 0 else None
            return self.mapping.to_record(row), True
        except _Decoder.Transient:
            raise ConfigError("the schema registry cannot be reached; check schema_registry.url and "
                              "its credentials") from None
        except Exception:                      # noqa: BLE001
            return None, False

    def describe(self) -> dict[str, Any]:
        now = self._clock()
        return {"type": "kafka", "name": self.name, "kind": self.kind,
                "partitions_assigned": len(self._assigned),
                "rows_delivered": self.rows_delivered, "rows_committed": self.rows_committed,
                "rows_uncommitted": self._staged_rows, "rows_buffered": self._buffered,
                "bad_rows": self.bad_rows, "caught_up": self.caught_up,
                "lag_messages": self.lag_messages,
                "lag_seconds": 0.0 if self.caught_up else
                (None if self._delivered_ts is None else max(0.0, now - self._delivered_ts)),
                "rebalances": self.rebalances, "blocked": self.blocked, "last_error": self.last_error}
