"""PostgreSQL adapter against a real server (own throwaway database + read-only role).

Needs TEST_ADMIN_DATABASE_URL, a superuser DSN of the isolated test server (:55434 on
the lab host); skipped without it. Nothing here touches any other database.
"""
from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from edge_agent import EdgeAgent                                    # noqa: E402
from edge_agent import __main__ as cli                              # noqa: E402
from edge_agent.mapping import ConfigError                          # noqa: E402
from edge_agent.sources import build_sources                        # noqa: E402
from edge_agent.status import read_status                           # noqa: E402
from tests.edge_agent.synth import (CADENCE, FEATURES, LOCAL_MODEL, T0, Collector,  # noqa: E402
                                    make_cfg, make_stream, utc)

ADMIN = os.getenv("TEST_ADMIN_DATABASE_URL")
pytestmark = pytest.mark.skipif(not ADMIN, reason="needs TEST_ADMIN_DATABASE_URL (superuser DSN)")
ROWS = 40
PW_ENV = "EDGE_TEST_PG_PASSWORD"

DDL = """
CREATE TABLE preds (id bigserial PRIMARY KEY, scored_at timestamptz NOT NULL, model text NOT NULL,
                    ticker text NOT NULL, p double precision, f1 double precision, f2 double precision,
                    f3 double precision, regime text, feats jsonb, pid text);
CREATE INDEX ON preds (scored_at, id);
CREATE TABLE outs (id bigserial PRIMARY KEY, model text NOT NULL, pid text NOT NULL, label int,
                   known_at timestamptz NOT NULL);
CREATE VIEW slow_preds AS SELECT preds.*, (pg_sleep(1.5) IS NOT NULL) AS slow FROM preds;
"""


@pytest.fixture(scope="module")
def db():
    params = psycopg2.extensions.parse_dsn(ADMIN)
    admin = psycopg2.connect(**params)
    admin.autocommit = True
    name, role, pw = f"edge_ad_{secrets.token_hex(4)}", f"edge_ro_{secrets.token_hex(4)}", secrets.token_hex(12)
    cur = admin.cursor()
    cur.execute(f'CREATE DATABASE "{name}"')
    cur.execute(f"CREATE ROLE \"{role}\" LOGIN PASSWORD %s", (pw,))
    own = psycopg2.connect(**{**params, "dbname": name})
    own.autocommit = True
    oc = own.cursor()
    oc.execute(DDL)
    oc.execute(f'GRANT CONNECT ON DATABASE "{name}" TO "{role}"')
    oc.execute(f'GRANT SELECT ON ALL TABLES IN SCHEMA public TO "{role}"')
    info = {"host": params.get("host", "127.0.0.1"), "port": int(params.get("port", 5432)),
            "dbname": name, "user": role, "password": pw, "own": own}
    yield info
    own.close()
    cur.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
    cur.execute(f'DROP ROLE "{role}"')
    admin.close()


@pytest.fixture()
def tables(db):
    c = db["own"].cursor()
    c.execute("TRUNCATE preds, outs RESTART IDENTITY")
    return c


def insert(cur, recs, *, pid=None):
    for i, r in enumerate(recs):
        cur.execute("INSERT INTO preds(scored_at, model, ticker, p, f1, f2, f3, regime, pid) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (r.ts, r.model_id, r.instrument, r.output, *[r.features[n] for n in FEATURES],
                     r.features["regime_zz"], None if pid is None else f"{pid}{i}"))


def spec(db, **kw):
    return {"type": "postgres", "host": db["host"], "port": db["port"], "dbname": db["dbname"],
            "user": db["user"], "password_env": PW_ENV, "sslmode": "disable", "allow_insecure_tls": True,
            "relation": "public.preds", "id_column": "id", "start_at": "beginning", "batch_size": ROWS,
            "columns": {"ts": "scored_at", "model_id": "model", "instrument": "ticker", "output": "p",
                        "features": ["f1", "f2", "f3", "regime"]}, **kw}


@pytest.fixture(autouse=True)
def pw(db, monkeypatch):
    monkeypatch.setenv(PW_ENV, db["password"])


def windows(first, n):
    return make_stream(n, rows=ROWS, first_window=first)


def ids(sink):
    return [b["window_id"] for b in sink.bodies()]


# -- configuration ------------------------------------------------------------------------------
@pytest.mark.parametrize("patch,needle", [
    ({"password": "x"}, "credentials must not be in the config"),
    ({"dsn": "postgresql://u:p@h/d"}, "credentials must not be in the config"),
    ({"password_env": None}, "password_env is required"),
    ({"sslmode": "require", "allow_insecure_tls": None}, "does not verify the server"),
    ({"sslmode": "bogus"}, "sslmode must be one of"),
    ({"relation": None}, "relation is required"),
    ({"relation": 'a."b'}, "relation must be"),
    ({"id_column": None}, "id_column is required"),
    ({"start_at": None}, "start_at is required"),
    ({"start_at": "yesterday"}, "start_at is required"),
    ({"batch_size": 0}, "batch_size"),
    ({"batch_size": 10**7}, "batch_size"),
    ({"statement_timeout_ms": 5}, "statement_timeout_ms"),
    ({"port": 99999}, "port"),
    ({"time_type": "ms"}, "time_type"),
    ({"surprise": 1}, "unknown key"),
    ({"columns": {"ts": "scored_at", "output": "p", "features": {"rest": True}, "model_id": "m",
                  "instrument": "t"}}, "only available for file sources"),
])
def test_config_errors_name_the_key(db, tmp_path, patch, needle):
    s = {k: v for k, v in {**spec(db), **patch}.items() if v is not None}
    with pytest.raises(ConfigError, match=needle):
        build_sources(s, tmp_path / "st", "prediction")


def test_default_tls_is_verify_full_and_a_server_that_cannot_be_verified_is_refused(db, tmp_path):
    s = {k: v for k, v in spec(db).items() if k not in ("sslmode", "allow_insecure_tls")}
    a = build_sources(s, tmp_path / "st", "prediction")[0]           # accepted: verify-full is the default
    assert a._conn_args["sslmode"] == "verify-full"
    assert a.poll() == [] and a.last_error == "OperationalError"      # the test server has no verifiable TLS
    strong = {**s, "sslmode": "verify-ca", "sslrootcert": "/etc/ssl/db-ca.pem",
              "sslcert": "/etc/ssl/c.pem", "sslkey": "/etc/ssl/c.key"}
    assert build_sources(strong, tmp_path / "st2", "prediction")[0]._conn_args["sslrootcert"] == "/etc/ssl/db-ca.pem"


def test_missing_password_variable_fails_at_start_up(db, tmp_path, monkeypatch):
    monkeypatch.delenv(PW_ENV)
    with pytest.raises(ConfigError, match=f"{PW_ENV} is not set"):
        build_sources(spec(db), tmp_path / "st", "prediction")


# -- reading ---------------------------------------------------------------------------------------
def test_keyset_pagination_reads_every_row_once_in_order_including_timestamp_ties(db, tables, tmp_path):
    recs = windows(0, 3)
    for r in recs[:10]:                                     # a block of identical timestamps
        tables.execute("INSERT INTO preds(scored_at, model, ticker, p, f1, f2, f3, regime) "
                       "VALUES (%s,'m','t',0.5,1,2,3,'c')", (recs[0].ts,))
    insert(tables, recs)
    a = build_sources(spec(db, batch_size=7), tmp_path / "st", "prediction")[0]
    got, seen = [], 0
    while chunk := a.poll(max_records=25):
        assert len(chunk) <= 25
        got += chunk
        seen += 1
    assert len(got) == 10 + len(recs) and seen > 4
    assert [r.ts for r in got] == sorted(r.ts for r in got)
    assert a.describe()["caught_up"] is True and a.describe()["lag_seconds"] == 0.0
    tables.execute("SELECT count(*) FROM preds")
    assert tables.fetchone()[0] == 10 + len(recs)           # nothing written


def test_session_is_read_only_and_cannot_write(db, tables, tmp_path):
    insert(tables, windows(0, 1))
    a = build_sources(spec(db), tmp_path / "st", "prediction")[0]
    assert len(a.poll()) == ROWS
    conn = a._get_conn()
    with conn.cursor() as cur:
        cur.execute("SHOW transaction_read_only")
        assert cur.fetchone()[0] == "on"
        with pytest.raises(psycopg2.Error):
            cur.execute("DELETE FROM preds")
    conn.rollback()
    with conn.cursor() as cur:                              # even a role that could write is refused
        cur.execute("SELECT 1")
        with pytest.raises(psycopg2.errors.ReadOnlySqlTransaction):
            cur.execute("CREATE TABLE x(a int)")
    conn.rollback()
    tables.execute("SELECT count(*) FROM preds")
    assert tables.fetchone()[0] == ROWS
    plain = psycopg2.connect(host=db["host"], port=db["port"], dbname=db["dbname"], user=db["user"],
                             password=db["password"])          # the role itself holds only SELECT
    with plain.cursor() as cur, pytest.raises(psycopg2.errors.InsufficientPrivilege):
        cur.execute("INSERT INTO preds(scored_at, model, ticker) VALUES (now(),'a','b')")
    plain.close()


def test_statement_timeout_bounds_a_slow_query_and_the_agent_keeps_running(db, tables, tmp_path):
    insert(tables, windows(0, 1))
    a = build_sources(spec(db, relation="public.slow_preds", statement_timeout_ms=200),
                      tmp_path / "st", "prediction")[0]
    assert a.poll() == [] and a.last_error == "QueryCanceled"        # bounded, reported, not raised
    assert a.describe()["last_error"] == "QueryCanceled"


def test_unreachable_database_is_reported_not_raised(db, tmp_path):
    a = build_sources(spec(db, port=1, connect_timeout_seconds=1), tmp_path / "st", "prediction")[0]
    assert a.poll() == [] and a.last_error == "OperationalError"


def test_credentials_never_reach_state_status_or_describe(db, tables, tmp_path):
    insert(tables, windows(0, 1))
    cfg = make_cfg(tmp_path)
    a = build_sources(spec(db), cfg.state_dir, "prediction")[0]
    agent = EdgeAgent(cfg, [a], post=Collector(), clock=lambda: utc(T0 + 3 * CADENCE))
    agent.run_once()
    agent.close()
    blob = json.dumps(read_status(cfg.state_dir)) + json.dumps(a.describe())
    for p in cfg.state_dir.rglob("*"):
        if p.is_file() and p.suffix in (".json", ".offset"):
            blob += p.read_text()
    assert db["password"] not in blob and db["user"] not in blob


def test_start_at_now_skips_history_and_is_sticky_across_restart(db, tables, tmp_path):
    insert(tables, windows(0, 1))                           # history
    s = spec(db, start_at="now")
    a = build_sources(s, tmp_path / "st", "prediction")[0]
    assert a.poll() == []
    now = datetime.now(timezone.utc)
    tables.execute("INSERT INTO preds(scored_at, model, ticker, p, f1, f2, f3, regime) "
                   "VALUES (%s,'m','t',0.5,1,2,3,'c')", (now,))
    assert len(a.poll()) == 1
    b = build_sources(s, tmp_path / "st", "prediction")[0]  # restart: the start is not re-resolved
    assert len(b.poll()) == 1                               # the uncommitted row is replayed, history is not


def test_settle_seconds_holds_back_rows_that_are_too_new(db, tables, tmp_path):
    now = datetime.now(timezone.utc)
    tables.execute("INSERT INTO preds(scored_at, model, ticker, p, f1, f2, f3, regime) "
                   "VALUES (%s,'m','t',0.5,1,2,3,'c')", (now,))
    a = build_sources(spec(db, settle_seconds=3600), tmp_path / "st", "prediction")[0]
    assert a.poll() == []


def test_json_column_features_and_bad_rows_are_counted(db, tables, tmp_path):
    ts = utc(T0)
    tables.execute("INSERT INTO preds(scored_at, model, ticker, p, feats) VALUES (%s,'m','t',0.2,%s)",
                   (ts, json.dumps({"a": 1.5, "b": "x"})))
    tables.execute("INSERT INTO preds(scored_at, model, ticker, p, feats) VALUES (%s,'m','t',0.2,'[1]')", (ts,))
    s = spec(db)
    s["columns"] = {**s["columns"], "features": {"json_column": "feats"}}
    a = build_sources(s, tmp_path / "st", "prediction")[0]
    (rec,) = a.poll()
    assert rec.features == {"a": 1.5, "b": "x"} and a.bad_rows == 1


# -- outcomes -----------------------------------------------------------------------------------------
def test_outcome_table_joins_labels_and_unlabelled_rows_are_skipped(db, tables, tmp_path):
    recs = [r for r in windows(0, 2)]
    recs = [type(r)(r.ts, r.model_id, r.instrument, r.features, 0.8, f"id{i}") for i, r in enumerate(recs)]
    for r in recs:
        tables.execute("INSERT INTO preds(scored_at, model, ticker, p, f1, f2, f3, regime, pid) "
                       "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                       (r.ts, r.model_id, r.instrument, r.output, *[r.features[n] for n in FEATURES],
                        r.features["regime_zz"], r.prediction_id))
        tables.execute("INSERT INTO outs(model, pid, label, known_at) VALUES (%s,%s,%s,%s)",
                       (r.model_id, r.prediction_id, 1, r.ts))
    tables.execute("INSERT INTO outs(model, pid, label, known_at) VALUES (%s,'pending',NULL,%s)",
                   (LOCAL_MODEL, utc(T0)))
    pred = spec(db)
    pred["columns"] = {**pred["columns"], "prediction_id": "pid"}
    outs = spec(db, relation="public.outs")
    outs["columns"] = {"model_id": "model", "prediction_id": "pid", "outcome": "label",
                       "outcome_observed_at": "known_at"}
    cfg = make_cfg(tmp_path, output_kind="probability")
    taken: set[str] = set()
    sources = build_sources(pred, cfg.state_dir, "prediction", taken=taken)
    labels = build_sources(outs, cfg.state_dir, "outcome", taken=taken)
    sink = Collector()
    agent = EdgeAgent(cfg, sources, post=sink, outcome_sources=labels,
                      clock=lambda: utc(T0 + 2 * CADENCE + 30))
    assert agent.poll() == 2 * ROWS
    assert agent.outcomes.status()["matched"] == 2 * ROWS
    assert labels[0].bad_rows == 0
    agent.close()
    assert all(b["z_t"]["brier"] == pytest.approx(0.04) for b in sink.bodies())


# -- crash / restart through the real agent --------------------------------------------------------
def new_agent(cfg, db, sink, clock, **kw):
    return EdgeAgent(cfg, build_sources(spec(db, **kw), cfg.state_dir, "prediction"),
                     post=sink, clock=lambda: clock[0])


def test_crash_restart_loses_nothing_and_never_emits_a_window_twice(db, tables, tmp_path):
    stream = windows(0, 6)
    insert(tables, stream[: 3 * ROWS])
    cfg = make_cfg(tmp_path)
    clock = [utc(T0 + 2 * CADENCE + 30)]
    sink_a = Collector()
    a = new_agent(cfg, db, sink_a, clock)
    a.run_once()
    assert len(ids(sink_a)) == 2
    del a                                                   # kill -9
    insert(tables, stream[3 * ROWS:])
    clock[0] = utc(T0 + 6 * CADENCE + 30)
    sink_b = Collector()
    b = new_agent(cfg, db, sink_b, clock)
    for _ in range(3):
        b.run_once()
    b.close()
    got = ids(sink_a) + ids(sink_b)
    assert len(got) == 6 and len(set(got)) == 6


def test_crash_before_the_checkpoint_replays_with_at_most_one_duplicate(db, tables, tmp_path):
    insert(tables, windows(0, 6))
    cfg = make_cfg(tmp_path)
    clock = [utc(T0 + 5 * CADENCE + 30)]
    sink_a = Collector()
    a = new_agent(cfg, db, sink_a, clock)
    a.sources[0].commit = lambda before=None: None
    a.run_once()
    assert len(ids(sink_a)) == 5
    del a
    clock[0] = utc(T0 + 6 * CADENCE + 30)
    sink_b = Collector()
    b = new_agent(cfg, db, sink_b, clock)
    for _ in range(3):
        b.run_once()
    b.close()
    both = ids(sink_a) + ids(sink_b)
    assert len(set(both)) == 6 and len(both) - len(set(both)) <= 1
    assert b.replayed_windows_skipped == 5             # up to and including the last emitted window (audit F5)


# -- --check / --status ----------------------------------------------------------------------------
def agent_json(tmp_path, source):
    cfg = {"tenant_id": "t", "install_id": "i", "models": {LOCAL_MODEL: "mm-wire-0001"},
           "secret_env": "EDGE_TEST_SECRET", "endpoints": ["https://x.invalid"],
           "state_dir": str(tmp_path / "state"), "cadence_seconds": CADENCE, "output_kind": "signed",
           "source": source}
    p = tmp_path / "agent.json"
    p.write_text(json.dumps(cfg))
    return p


def test_check_samples_the_table_and_writes_no_offset(db, tables, tmp_path, monkeypatch, capsys):
    insert(tables, windows(0, 2))
    monkeypatch.setattr("sys.argv", ["edge_agent", "--config", str(agent_json(tmp_path, spec(db))), "--check"])
    with pytest.raises(SystemExit):
        cli.main()
    out = json.loads(capsys.readouterr().out)
    assert out["source_check"]["valid_shape"] == 2 * ROWS and out["source_check"]["invalid_shape"] == 0
    assert not (tmp_path / "state").exists()
    tables.execute("SELECT count(*) FROM preds")
    assert tables.fetchone()[0] == 2 * ROWS
    for private in (db["password"], db["user"], FEATURES[0], "INSTR_SENTINEL"):
        assert private not in json.dumps(out)


@pytest.mark.parametrize("patch,needle", [
    ({"relation": "public.nope"}, "UndefinedTable"),
    ({"columns": {"ts": "scored_at", "model_id": "model", "instrument": "ticker", "output": "p",
                  "features": ["f1", "no_such_col"]}}, "UndefinedColumn"),
    ({"port": 1, "connect_timeout_seconds": 1}, "cannot connect"),
])
def test_check_turns_source_errors_into_a_clear_verdict(db, tmp_path, monkeypatch, capsys, patch, needle):
    monkeypatch.setattr("sys.argv", ["edge_agent", "--config",
                                     str(agent_json(tmp_path, spec(db, **patch))), "--check"])
    with pytest.raises(SystemExit) as done:
        cli.main()
    out = json.loads(capsys.readouterr().out)
    assert done.value.code == 2 and out["verdict"] == "NOT_READY" and needle in out["config_error"]
    assert db["password"] not in json.dumps(out)


def test_status_shows_committed_position_and_lag_without_ids(db, tables, tmp_path):
    insert(tables, windows(0, 3))
    cfg = make_cfg(tmp_path)
    clock = [utc(T0 + 3 * CADENCE + 30)]
    agent = new_agent(cfg, db, Collector(), clock, name="warehouse")
    agent.run_once()
    agent.write_status(force=True)
    report = read_status(cfg.state_dir)
    (src,) = report["sources"]
    assert src["name"] == "warehouse" and src["type"] == "postgres" and src["committed_ts"]
    snap = report["agent_snapshot"]["sources"][0]
    assert snap["rows_committed"] == 3 * ROWS and snap["caught_up"] is True and snap["lag_seconds"] == 0.0
    assert snap["committed_ts"] == datetime.fromisoformat(src["committed_ts"]).astimezone(timezone.utc).isoformat()
    agent.close()


# -- a caught-up table is healthy, not "no rows yet" (pre-tag review, edgefix2 2026-10-30) ----------------------
def _caught_up(db, tables, tmp_path):
    insert(tables, windows(0, 2))
    cfg = make_cfg(tmp_path)
    clock = [utc(T0 + 2 * CADENCE + 30)]
    a = new_agent(cfg, db, Collector(), clock)
    a.run_once()
    a.close(keep_open=True)
    (src,) = build_sources(spec(db), cfg.state_dir, "prediction")
    assert src.poll() == []                                # committed at the end of the table
    return cfg


def test_sample_of_a_caught_up_table_reads_the_newest_rows(db, tables, tmp_path):
    cfg = _caught_up(db, tables, tmp_path)
    (src,) = build_sources(spec(db), cfg.state_dir, "prediction")
    s = src.sample(25)
    assert len(s.records) == 25 and s.bad == 0
    newest = sorted(r.ts for r in windows(0, 2))[-25:]
    assert sorted(r.ts for r in s.records) == newest


def test_fresh_empty_table_is_no_rows_yet_and_a_purged_one_after_windows_is_caught_up(db, tables, tmp_path):
    import dataclasses
    fresh = tmp_path / "fresh"
    fcfg = dataclasses.replace(make_cfg(fresh), output_kind="probability")
    rep = cli.check_source(fcfg, adapters=build_sources(spec(db, start_at="now"), fcfg.state_dir, "prediction"))
    assert rep["verdict"] == "NO_ROWS_YET"                 # run 1 of a fresh install
    cfg = _caught_up(db, tables, tmp_path / "old")
    tables.execute("TRUNCATE preds")                       # retention job emptied the table
    pcfg = dataclasses.replace(cfg, output_kind="probability")
    rep = cli.check_source(pcfg, adapters=build_sources(spec(db), cfg.state_dir, "prediction"))
    assert rep["verdict"] == "CAUGHT_UP"


def test_container_check_reads_password_env_from_agent_env(db, tables, tmp_path, monkeypatch, capsys):
    """#323 review: the image's `check` passed no agent.env, so a `password_env` source always failed the check and
    the HEALTHCHECK stayed unhealthy. The password now comes from agent.env, which holds no HMAC secret or API key."""
    import importlib.util
    from pathlib import Path
    ep = Path(__file__).resolve().parents[2] / "packaging/edge_agent/container/entrypoint.py"
    sp = importlib.util.spec_from_file_location("bbe_entry_pg", ep)
    entry = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(entry)
    insert(tables, windows(0, 1))
    conf = agent_json(tmp_path, spec(db))
    env = tmp_path / "agent.env"
    env.write_text(f"{PW_ENV}={db['password']}\n")
    env.chmod(0o600)
    monkeypatch.delenv(PW_ENV)
    monkeypatch.setattr(entry, "CONFIG", conf)
    monkeypatch.setattr(entry, "ENV_FILES", (tmp_path / "none.env", env))
    code = entry.main(["--check"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert "config_error" not in out, out
    assert out["source_check"]["valid_shape"] == ROWS
    assert code == 2 and out["verdict"] == "NOT_READY"     # signed output: the contract, not the password
    assert db["password"] not in json.dumps(out)


def test_check_with_start_at_now_samples_a_busy_table_and_streaming_still_starts_now(db, tables, tmp_path):
    """kitrun6 R-02: `"start_at": "now"` (the documented example) read from the check's own "now", so a table written
    at 2 rows/s always answered NO_ROWS_YET and the install never started. A writer keeps inserting during the check."""
    import dataclasses
    import threading
    for i in range(50):                                     # rows already in the table, written earlier today
        tables.execute("INSERT INTO preds(scored_at, model, ticker, p, f1, f2, f3, regime) "
                       "VALUES (now() - make_interval(secs => %s), %s, 'INSTR_A', 0.5, 1, 2, 3, 'c')", (600 - i, LOCAL_MODEL))
    stop = threading.Event()
    own = psycopg2.connect(**{**psycopg2.extensions.parse_dsn(ADMIN), "dbname": db["dbname"]})
    own.autocommit = True

    def writer():
        c = own.cursor()
        while not stop.is_set():
            c.execute("INSERT INTO preds(scored_at, model, ticker, p, f1, f2, f3, regime) "
                      "VALUES (now(), %s, 'INSTR_A', 0.5, 1, 2, 3, 'c')", (LOCAL_MODEL,))
            stop.wait(0.05)
    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        cfg = dataclasses.replace(make_cfg(tmp_path), output_kind="probability")
        rep = cli.check_source(cfg, adapters=build_sources(spec(db, start_at="now"), cfg.state_dir, "prediction"))
        assert rep["verdict"] == "SOURCE_COMPATIBLE", rep
        assert rep["source_check"]["rows_checked"] > 0 and rep["source_check"]["invalid_shape"] == 0
        assert not cfg.state_dir.exists() or not any(cfg.state_dir.rglob("*.json"))   # the check wrote no position
    finally:
        stop.set()
        t.join(5)
        own.close()
    # streaming still starts at the agent's first start, not at the rows the check looked at
    tables.execute("SELECT count(*) FROM preds")
    before = tables.fetchone()[0]
    a = build_sources(spec(db, start_at="now"), cfg.state_dir, "prediction")[0]
    assert a.poll() == [] and before > 50
