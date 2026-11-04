"""Read-only PostgreSQL table/view input (predictions or outcomes).

    "source": {"type": "postgres", "host": "db.internal", "dbname": "ml", "user": "edge_ro",
               "password_env": "EDGE_PG_PASSWORD", "sslmode": "verify-full",
               "sslrootcert": "/etc/ssl/db-ca.pem",
               "relation": "ml.predictions", "id_column": "id",
               "start_at": "now",
               "columns": {"ts": "scored_at", "model_id": "model", "instrument": "ticker",
                           "output": "p", "features": ["x1", "x2"]}}

Guarantees: it only ever issues one SELECT shape, inside a READ ONLY transaction
(the session is also read-only server side); it has a statement timeout; every page is
at most `batch_size` rows; rows are read in keyset order on (time column, id_column),
so the cost per page does not grow with table size. The password / DSN is read from
the environment named in the config and is never stored or logged.

The time column plus `id_column` must order new rows after old ones: a row inserted
later with an older (time, id) than the committed position is not seen. If the writer
can lag, set `settle_seconds` so only rows older than that are read.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import structlog

from edge_agent.adapters import Sample
from edge_agent.checkpoint import CheckpointedAdapter, read_json, write_json_atomic
from edge_agent.mapping import (ColumnMapping, ConfigError, check_keys, epoch_of, int_in,
                                num_in)
from edge_agent.records import Record

log = structlog.get_logger("edge_agent.source")

ALLOWED = {"type", "name", "host", "port", "dbname", "user", "password_env", "dsn_env", "sslmode",
           "sslrootcert", "sslcert", "sslkey", "allow_insecure_tls", "relation", "id_column",
           "columns", "constants", "batch_size", "statement_timeout_ms", "connect_timeout_seconds",
           "settle_seconds", "start_at", "time_type", "max_uncommitted_rows"}
_SECRET_KEYS = ("password", "dsn", "url", "uri", "connection_string")
_STRONG_TLS = ("verify-full", "verify-ca")
_SSLMODES = _STRONG_TLS + ("require", "prefer", "allow", "disable")


def _driver():
    try:
        import psycopg2
        import psycopg2.errors
        import psycopg2.extras
        import psycopg2.sql
    except ImportError as exc:       # optional extra
        raise ConfigError("the postgres source needs the optional dependency psycopg2: "
                          "pip install -r requirements-edge-postgres.txt") from exc
    return psycopg2


def _enc(v: Any) -> Any:
    if isinstance(v, datetime):
        return {"t": "dt", "v": v.isoformat()}
    if isinstance(v, bool):
        raise TypeError("unsupported key type")
    if isinstance(v, int):
        return {"t": "int", "v": v}
    if isinstance(v, float):
        return {"t": "float", "v": v}
    if isinstance(v, Decimal):
        return {"t": "dec", "v": str(v)}
    if isinstance(v, str):
        return {"t": "str", "v": v}
    return {"t": "str", "v": str(v)}


def _dec(o: Any) -> Any:
    t, v = o["t"], o["v"]
    return {"dt": datetime.fromisoformat, "int": int, "float": float, "dec": Decimal,
            "str": str}[t](v)


class PostgresAdapter(CheckpointedAdapter):
    def __init__(self, spec: Mapping[str, Any], state_dir: str | Path, *, kind: str = "prediction",
                 name: str | None = None, env: Mapping[str, str] | None = None,
                 clock=time.time) -> None:
        _driver()                          # optional extra: say so before any other complaint
        where = "source" if kind == "prediction" else "outcome_source"
        spec = dict(spec)
        for k in _SECRET_KEYS:
            if k in spec:
                raise ConfigError(f"{where}.{k}: credentials must not be in the config file; "
                                  "use password_env or dsn_env (the name of an environment variable)")
        check_keys(spec, ALLOWED, where)
        self.kind, self._env = kind, os.environ if env is None else env
        name = name or spec.get("name") or ("predictions" if kind == "prediction" else "outcomes")
        super().__init__(Path(state_dir), name, clock=clock,
                         max_uncommitted_rows=int_in(spec, "max_uncommitted_rows", 5_000_000,
                                                     1_000, 1_000_000_000, where))
        self.mapping = ColumnMapping(spec.get("columns"), spec.get("constants"), kind=kind,
                                     allow_pattern_features=False)
        self._parse_relation(spec, where)
        self.batch_size = int_in(spec, "batch_size", 5_000, 1, 100_000, where)
        self.statement_timeout_ms = int_in(spec, "statement_timeout_ms", 15_000, 100, 600_000, where)
        self.connect_timeout = int_in(spec, "connect_timeout_seconds", 10, 1, 120, where)
        self.settle_seconds = num_in(spec, "settle_seconds", 0.0, 0.0, 30 * 86400.0, where)
        self.time_type = spec.get("time_type", "timestamp")
        if self.time_type not in ("timestamp", "epoch_seconds"):
            raise ConfigError(f"{where}.time_type must be 'timestamp' or 'epoch_seconds'")
        self._conn_args = self._connection(spec, where)
        self._start = self._parse_start(spec.get("start_at"), where)
        self._credentials()                # fail at start-up, not at the first poll
        saved = read_json(self.state_file) or {}
        pos = saved.get("position")
        self._committed: tuple[Any, Any] | None = (
            (_dec(pos["ts"]), None if pos["id"] is None else _dec(pos["id"])) if pos else None)
        self._cursor = self._committed
        self._conn = None
        self.caught_up = False
        self._delivered_ts: float | None = None

    # -- config ---------------------------------------------------------------------------
    def _parse_relation(self, spec: dict, where: str) -> None:
        rel = spec.get("relation")
        if not isinstance(rel, str) or not rel.strip():
            raise ConfigError(f"{where}.relation is required, e.g. \"schema.table_or_view\"")
        parts = rel.split(".")
        if len(parts) > 2 or any(not p or '"' in p or "\0" in p for p in parts):
            raise ConfigError(f"{where}.relation must be \"table\" or \"schema.table\"")
        self.relation = tuple(parts)
        idc = spec.get("id_column")
        if not isinstance(idc, str) or not idc or '"' in idc:
            raise ConfigError(f"{where}.id_column is required: a unique column that breaks time ties "
                              "(keyset pagination is on (time column, id_column))")
        self.id_column = idc

    def _parse_start(self, v: Any, where: str) -> Any:
        if v in ("beginning", "now"):
            return v
        if isinstance(v, str):
            try:
                d = datetime.fromisoformat(v.replace("Z", "+00:00"))
                return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        raise ConfigError(f"{where}.start_at is required: \"beginning\" (read the whole history), "
                          "\"now\" (only rows from the first start on) or an ISO-8601 time")

    def _connection(self, spec: dict, where: str) -> dict[str, Any]:
        sslmode = spec.get("sslmode", "verify-full")
        if sslmode not in _SSLMODES:
            raise ConfigError(f"{where}.sslmode must be one of {list(_SSLMODES)}")
        if sslmode not in _STRONG_TLS and spec.get("allow_insecure_tls") is not True:
            raise ConfigError(f"{where}.sslmode '{sslmode}' does not verify the server; use verify-full "
                              "(default) or verify-ca, or set \"allow_insecure_tls\": true to accept that")
        args: dict[str, Any] = {"sslmode": sslmode}
        dsn_env, pw_env = spec.get("dsn_env"), spec.get("password_env")
        if dsn_env is not None:
            if not isinstance(dsn_env, str) or not dsn_env:
                raise ConfigError(f"{where}.dsn_env must be an environment variable name")
            if any(k in spec for k in ("host", "port", "dbname", "user", "password_env")):
                raise ConfigError(f"{where}: dsn_env replaces host/port/dbname/user/password_env; give one or the other")
            self._dsn_env, self._pw_env = dsn_env, None
        else:
            for k in ("host", "dbname", "user"):
                if not isinstance(spec.get(k), str) or not spec[k]:
                    raise ConfigError(f"{where}.{k} is required (or use dsn_env)")
            if not isinstance(pw_env, str) or not pw_env:
                raise ConfigError(f"{where}.password_env is required: the name of the environment "
                                  "variable holding the database password")
            args.update(host=spec["host"], dbname=spec["dbname"], user=spec["user"],
                        port=int_in(spec, "port", 5432, 1, 65535, where))
            self._dsn_env, self._pw_env = None, pw_env
        for k in ("sslrootcert", "sslcert", "sslkey"):
            if k in spec:
                if not isinstance(spec[k], str) or not spec[k]:
                    raise ConfigError(f"{where}.{k} must be a file path")
                args[k] = spec[k]
        if sslmode in _STRONG_TLS and "sslrootcert" not in spec:
            log.info("postgres_tls_uses_default_root_store")
        return args

    def _credentials(self) -> dict[str, Any]:
        name = self._dsn_env or self._pw_env
        value = self._env.get(name)
        if not value:
            raise ConfigError(f"environment variable {name} is not set (it must hold the database "
                              f"{'DSN' if self._dsn_env else 'password'})")
        return {"dsn": value} if self._dsn_env else {"password": value}

    # -- connection ------------------------------------------------------------------------
    def _connect(self):
        pg = _driver()
        cred = self._credentials()
        options = (f"-c statement_timeout={self.statement_timeout_ms} "
                   "-c default_transaction_read_only=on "
                   f"-c idle_in_transaction_session_timeout={max(60_000, 4 * self.statement_timeout_ms)} "
                   "-c lock_timeout=2000")
        kw = {**self._conn_args, "connect_timeout": self.connect_timeout, "options": options,
              "application_name": "blurredbox-edge-agent"}
        dsn = cred.pop("dsn", None)
        kw.update(cred)
        conn = pg.connect(dsn, **kw) if dsn else pg.connect(**kw)
        conn.set_session(readonly=True, autocommit=False)
        with conn.cursor() as cur:
            cur.execute("SHOW transaction_read_only")
            if cur.fetchone()[0] != "on":
                conn.close()
                raise ConfigError("the database session is not read-only; refusing to continue")
        conn.rollback()
        return conn

    def _get_conn(self):
        if self._conn is not None and not self._conn.closed:
            return self._conn
        self._conn = self._connect()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:                 # noqa: BLE001
                pass
            self._conn = None

    # -- queries ---------------------------------------------------------------------------
    def _now(self, conn) -> Any:
        with conn.cursor() as cur:
            if self.time_type == "epoch_seconds":
                cur.execute("SELECT extract(epoch from now())")
            else:
                cur.execute("SELECT now()")
            v = cur.fetchone()[0]
        conn.rollback()
        return float(v) if self.time_type == "epoch_seconds" else v

    def _resolve_start(self, conn) -> tuple[Any, Any] | None:
        s = self._start
        if s == "beginning":
            return None
        if s == "now":
            return (self._now(conn), None)
        return (s.timestamp() if self.time_type == "epoch_seconds" else s, None)

    def _page(self, conn, cursor: tuple[Any, Any] | None, limit: int, *,
              latest: bool = False) -> list[dict[str, Any]]:
        pg = _driver()
        sql = pg.sql
        ts = sql.Identifier(self.mapping.time_column)
        idc = sql.Identifier(self.id_column)
        cols = list(dict.fromkeys(self.mapping.fixed_columns() + [self.id_column]))
        where, params = [sql.SQL("{} IS NOT NULL").format(ts)], []     # a row with no time cannot be placed
        if cursor is not None:
            if cursor[1] is None:
                where.append(sql.SQL("{} >= %s").format(ts))
                params.append(cursor[0])
            else:
                where.append(sql.SQL("({}, {}) > (%s, %s)").format(ts, idc))
                params += [cursor[0], cursor[1]]
        if self.settle_seconds:
            bound = ("now() - make_interval(secs => %s)" if self.time_type == "timestamp"
                     else "extract(epoch from now()) - %s")
            where.append(sql.SQL("{} <= " + bound).format(ts))
            params.append(self.settle_seconds)
        order = sql.SQL("ORDER BY {ts} DESC, {id} DESC" if latest else "ORDER BY {ts}, {id}").format(ts=ts, id=idc)
        query = sql.SQL("SELECT {cols} FROM {rel}{where} {order} LIMIT %s").format(
            cols=sql.SQL(", ").join(map(sql.Identifier, cols)),
            rel=sql.SQL(".").join(map(sql.Identifier, self.relation)),
            where=sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where), order=order)
        try:
            with conn.cursor(cursor_factory=pg.extras.RealDictCursor) as cur:
                cur.execute(query, params + [limit])
                rows = cur.fetchall()
        finally:
            conn.rollback()                    # end the read-only transaction: never idle in one
        return rows

    def _rows_to_batch(self, rows: list[dict[str, Any]]):
        recs, bad, newest = [], 0, None
        for row in rows:
            t = epoch_of(row.get(self.mapping.time_column))
            if t is not None and (newest is None or t > newest):
                newest = t
            try:
                rec = self.mapping.to_record(row)
            except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
                bad += 1
                continue
            if rec is not None:
                recs.append(rec)
        return recs, bad, newest

    # -- SourceAdapter ----------------------------------------------------------------------
    def poll(self, max_records: int = 10_000) -> list[Record]:
        out: list[Record] = []
        if self.blocked:
            return out
        try:
            conn = self._get_conn()
            if self._cursor is None and self._start != "beginning" and not self.state_file.exists():
                self._cursor = self._committed = self._resolve_start(conn)
                self._persist_start()
            while len(out) < max_records and not self.blocked:
                limit = min(self.batch_size, max_records - len(out),
                            self.max_uncommitted_rows - self._staged_rows)
                rows = self._page(conn, self._cursor, limit)
                if rows:
                    recs, bad, newest = self._rows_to_batch(rows)
                    last = rows[-1]
                    self._cursor = (last[self.mapping.time_column], last[self.id_column])
                    self.bad_rows += bad
                    self._stage(newest, self._cursor, len(rows))
                    if newest is not None:
                        self._delivered_ts = max(self._delivered_ts or newest, newest)
                    out.extend(recs)
                self.caught_up = len(rows) < limit
                if self.caught_up:
                    break
            self._ok()
        except ConfigError:
            raise
        except Exception as exc:               # noqa: BLE001 -- a DB outage must not stop the agent
            self._fail(exc)
            self.close()
            self.caught_up = False
            log.warning("postgres_source_unavailable", source=self.name, error_type=type(exc).__name__)
        return out

    def _persist_start(self) -> None:
        write_json_atomic(self.state_file, self._state())

    def _apply(self, delta: Any) -> None:
        self._committed = delta

    def _state(self) -> dict[str, Any]:
        c = self._committed
        return {"v": 1, "type": "postgres", "position": None if c is None else
                {"ts": _enc(c[0]), "id": None if c[1] is None else _enc(c[1])}}

    def sample(self, limit: int = 1000) -> Sample:
        """Read up to `limit` rows from the committed position (or the configured start),
        write nothing, then disconnect. Raises ConfigError with a clear message.

        Once the agent has committed a position and is caught up there is nothing after it (a quiet night, after a
        stop); the check then reads the newest `limit` rows instead, ignoring the position, so a healthy quiet source
        is not reported as having no rows. A fresh install (nothing committed) reads from its start; when nothing is
        there yet (kitrun6 R-02: `"start_at": "now"` is the current time at check time, so a busy table had no row
        after it, and the first check always said NO_ROWS_YET) it also reads the newest rows. The check only tests that
        the rows fit the mapping; streaming still starts where `start_at` says (nothing is written here)."""
        pg = _driver()
        try:
            conn = self._connect()
        except pg.OperationalError as exc:
            raise ConfigError(f"cannot connect to the database ({type(exc).__name__}); check host, port, "
                              "credentials in the environment, and TLS settings") from None
        try:
            cursor = self._committed if self._committed is not None else self._resolve_start(conn)
            rows = []
            try:
                while len(rows) < limit:
                    want = min(limit - len(rows), self.batch_size)
                    page = self._page(conn, cursor, want)
                    rows += page
                    if len(page) < want:
                        break
                    cursor = (page[-1][self.mapping.time_column], page[-1][self.id_column])
                if not rows:
                    rows = self._page(conn, None, limit, latest=True)
            except (pg.errors.UndefinedColumn, pg.errors.UndefinedTable, pg.errors.InsufficientPrivilege) as exc:
                raise ConfigError(f"the relation or a mapped column cannot be read: "
                                  f"{type(exc).__name__}: {str(exc).splitlines()[0]}") from None
            except pg.Error as exc:
                raise ConfigError(f"query failed: {type(exc).__name__}: {str(exc).splitlines()[0]}") from None
            recs, bad, _ = self._rows_to_batch(rows)
            return Sample(recs, bad)
        finally:
            conn.close()

    def describe(self) -> dict[str, Any]:
        now = self._clock()
        committed_ts = epoch_of(self._committed[0]) if self._committed else None
        return {"type": "postgres", "name": self.name, "kind": self.kind,
                "committed_ts": None if committed_ts is None else
                datetime.fromtimestamp(committed_ts, timezone.utc).isoformat(),
                "rows_delivered": self.rows_delivered, "rows_committed": self.rows_committed,
                "rows_uncommitted": self._staged_rows, "bad_rows": self.bad_rows,
                "caught_up": self.caught_up,
                "lag_seconds": 0.0 if self.caught_up else
                (None if self._delivered_ts is None else max(0.0, now - self._delivered_ts)),
                "blocked": self.blocked, "last_error": self.last_error}
