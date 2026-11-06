"""CSV / Parquet folder adapter, the shared column mapping, and the commit-after-durable
contract, driven through the real EdgeAgent and real files."""
from __future__ import annotations

import csv
import json
import os
import time

import pytest

from edge_agent import EdgeAgent
from edge_agent import __main__ as cli
from edge_agent.mapping import ColumnMapping, ConfigError
from edge_agent.source_folder import FolderAdapter
from edge_agent.sources import build_sources
from edge_agent.status import read_status
from tests.edge_agent.synth import (CADENCE, LOCAL_MODEL, T0, Collector, FEATURES, make_cfg,
                                    make_stream, utc)

COLUMNS = {"ts": "scored_at", "model_id": "model", "instrument": "ticker", "output": "p",
           "features": {"prefix": "f_"}}
ROWS = 40                                   # per window: above min_rows_per_window=30


def spec(folder, **kw):
    return {"type": "folder", "path": str(folder), "columns": COLUMNS, "min_age_seconds": 0,
            "scan_interval_seconds": 0, "batch_size": ROWS, **kw}


def header():
    return ["scored_at", "model", "ticker", "p", *[f"f_{n}" for n in FEATURES], "f_regime_zz"]


def csv_rows(recs):
    for r in recs:
        yield [f"{r.ts.timestamp():.3f}", r.model_id, r.instrument, repr(r.output),
               *[repr(r.features[n]) for n in FEATURES], r.features["regime_zz"]]


def write_csv(path, recs, *, with_header=True, mode="w"):
    with open(path, mode, newline="") as fh:
        w = csv.writer(fh)
        if with_header:
            w.writerow(header())
        w.writerows(csv_rows(recs))


def windows(first, n):
    return make_stream(n, rows=ROWS, first_window=first)


def ids(sink):
    return [b["window_id"] for b in sink.bodies()]


def new_agent(cfg, folder, sink, clock, **kw):
    adapters = build_sources(spec(folder, **kw), cfg.state_dir, "prediction")
    return EdgeAgent(cfg, adapters, post=sink, clock=lambda: clock[0])


# -- configuration ------------------------------------------------------------------------------
@pytest.mark.parametrize("patch,needle", [
    ({"path": None}, "source.path is required"),
    ({"columns": {**COLUMNS, "ts": None}}, "columns.ts"),
    ({"columns": {k: v for k, v in COLUMNS.items() if k != "output"}}, "columns.output is required"),
    ({"columns": {**COLUMNS, "bogus": "x"}}, "unknown field"),
    ({"columns": {**COLUMNS, "features": []}}, "at least one column"),
    ({"columns": {**COLUMNS, "features": {"nope": 1}}}, "columns.features"),
    ({"batch_size": 0}, "batch_size"),
    ({"batch_size": 10**9}, "batch_size"),
    ({"patterns": []}, "patterns"),
    ({"patterns": ["a/b.csv"]}, "patterns"),
    ({"min_age_seconds": -1}, "min_age_seconds"),
    ({"format": "xlsx"}, "format"),
    ({"surprise": 1}, "unknown key"),
    ({"constants": {"model_id": "m"}, "columns": {**COLUMNS, "model_id": "model"}}, "both as a column"),
    ({"type": "rabbitmq"}, "type must be one of"),
    ({"type": "ftp"}, "type must be one of"),
])
def test_config_errors_name_the_key(tmp_path, patch, needle):
    s = {**spec(tmp_path), **patch}
    s = {k: v for k, v in s.items() if v is not None}
    with pytest.raises(ConfigError, match=needle):
        build_sources(s, tmp_path / "state", "prediction")


def test_duplicate_names_and_outcome_columns_are_rejected(tmp_path):
    two = [spec(tmp_path, name="a"), spec(tmp_path, name="a")]
    with pytest.raises(ConfigError, match="used twice"):
        build_sources(two, tmp_path / "s", "prediction")
    with pytest.raises(ConfigError, match="unknown field"):
        build_sources(spec(tmp_path), tmp_path / "s", "outcome")


def test_mapping_converts_csv_text_and_rejects_bad_rows():
    m = ColumnMapping({**COLUMNS, "prediction_id": "pid"}, None, null_values=("", "NA"))
    row = {"scored_at": "1700000040", "model": "m", "ticker": "AAA", "p": "0.25", "f_x": "3.5",
           "f_c": "CAT", "f_n": "NA", "pid": "p1"}
    rec = m.to_record(row)
    assert rec.features == {"x": 3.5, "c": "CAT", "n": None} and rec.output == 0.25
    assert rec.ts.timestamp() == 1700000040 and rec.prediction_id == "p1"
    with pytest.raises(ValueError):
        m.to_record({**row, "scored_at": "not-a-time"})


# -- reading, partial files, rotation, restart -------------------------------------------------------
def test_reads_complete_files_ignores_partial_and_waits_for_young_ones(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    write_csv(d / "a.csv", windows(0, 1))
    write_csv(d / "b.csv.tmp", windows(1, 1))             # writer's temp name
    write_csv(d / ".c.csv", windows(2, 1))                 # hidden
    write_csv(d / "young.csv", windows(3, 1))
    os.utime(d / "a.csv", (time.time() - 100,) * 2)
    a = FolderAdapter(spec(d, min_age_seconds=30), tmp_path / "st")
    recs = a.poll()
    assert len(recs) == ROWS and {r.ts.timestamp() // CADENCE for r in recs} == {T0 // CADENCE}
    assert a.poll() == []
    os.rename(d / "b.csv.tmp", d / "b.csv")                # writer finishes ...
    assert a.poll() == []                                  # ... but the file is still young
    os.utime(d / "b.csv", (time.time() - 100,) * 2)
    assert len(a.poll()) == ROWS
    assert a.describe()["files_waiting"] == 2              # young.csv and the hidden file


def test_half_written_last_line_is_waited_for_and_appends_resume(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    f = d / "a.csv"
    write_csv(f, windows(0, 1))
    with open(f, "a") as fh:                               # a row without its line ending yet
        fh.write("1700000100.0,LOCAL_MODEL_SENTINEL,I,0.1,1,2,3,CAT")
    a = FolderAdapter(spec(d), tmp_path / "st")
    assert len(a.poll()) == ROWS
    with open(f, "a") as fh:
        fh.write("\n1700000101.0,LOCAL_MODEL_SENTINEL,I,0.1,1,2,3,CAT\n")
    assert len(a.poll()) == 2                              # the completed row and the next one, once each
    assert a.poll() == []


def test_uncommitted_rows_replay_after_restart_and_committed_rows_do_not(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    write_csv(d / "a.csv", windows(0, 2))
    a = FolderAdapter(spec(d), tmp_path / "st")
    assert len(a.poll()) == 2 * ROWS
    assert len(FolderAdapter(spec(d), tmp_path / "st").poll()) == 2 * ROWS      # crash before commit: replayed
    a.commit(None)
    b = FolderAdapter(spec(d), tmp_path / "st")
    assert b.poll() == []                                                         # committed: never again
    assert b.describe()["files_done"] == 1
    assert (tmp_path / "st" / "sources" / "predictions.json").stat().st_mode & 0o777 == 0o600


def test_commit_holds_back_batches_that_still_have_rows_in_an_open_window(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    write_csv(d / "a.csv", windows(0, 3))                  # three batches of one window each
    a = FolderAdapter(spec(d), tmp_path / "st")
    assert len(a.poll()) == 3 * ROWS
    a.commit(utc(T0 + 2 * CADENCE))                        # windows 0 and 1 are in the spool, 2 is open
    again = FolderAdapter(spec(d), tmp_path / "st")
    assert [int(r.ts.timestamp() - T0) // CADENCE for r in again.poll()] == [2] * ROWS


def test_rotation_rename_truncate_and_recycled_inode(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    write_csv(d / "app.csv", windows(0, 1))
    a = FolderAdapter(spec(d), tmp_path / "st")
    assert len(a.poll()) == ROWS
    a.commit(None)
    os.rename(d / "app.csv", d / "app.csv.1")              # rotation by rename: same inode, same file
    write_csv(d / "app.csv", windows(1, 1))                # the new live file
    a = FolderAdapter(spec(d, patterns=["app.csv*"]), tmp_path / "st")
    got = a.poll()
    assert len(got) == ROWS and got[0].ts.timestamp() >= T0 + CADENCE      # only the new file's rows
    a.commit(None)
    # delete + recreate under the same name: the OS may hand out the same inode; the content differs
    os.remove(d / "app.csv")
    write_csv(d / "app.csv", windows(5, 1))
    got = a.poll()
    assert len(got) == ROWS and got[0].ts.timestamp() >= T0 + 5 * CADENCE
    a.commit(None)
    write_csv(d / "app.csv", windows(7, 1))                # truncated and rewritten in place
    got = a.poll()
    assert len(got) == ROWS and got[0].ts.timestamp() >= T0 + 7 * CADENCE


def test_removed_files_are_forgotten(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    write_csv(d / "a.csv", windows(0, 1))
    a = FolderAdapter(spec(d), tmp_path / "st")
    a.poll()
    a.commit(None)
    assert a.describe()["files_tracked"] == 1
    os.remove(d / "a.csv")
    a.poll()
    a.commit(None)
    assert FolderAdapter(spec(d), tmp_path / "st").describe()["files_tracked"] == 0


def test_memory_is_bounded_by_the_batch(tmp_path, monkeypatch):
    d = tmp_path / "in"
    d.mkdir()
    write_csv(d / "big.csv", windows(0, 10))               # 400 rows
    a = FolderAdapter(spec(d, batch_size=25), tmp_path / "st")
    seen = []
    original = a._csv_page
    monkeypatch.setattr(a, "_csv_page", lambda *args: seen.append(args[3]) or original(*args))
    assert len(a.poll(max_records=60)) == 60
    assert max(seen) <= 25 and a._staged_rows == 60
    total = 60
    while chunk := a.poll(max_records=60):
        assert len(chunk) <= 60
        total += len(chunk)
    assert total == 400


def test_backpressure_stops_reading_until_the_checkpoint_catches_up(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    write_csv(d / "a.csv", windows(0, 30))                 # 1200 rows
    a = FolderAdapter(spec(d, max_uncommitted_rows=1000), tmp_path / "st")
    n = len(a.poll(max_records=5000))
    assert n == 1000 and a.blocked and a.poll() == []
    a.commit(None)
    assert len(a.poll()) == 1200 - n


def test_bad_rows_and_missing_columns_are_counted_not_raised(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    write_csv(d / "a.csv", windows(0, 1))
    with open(d / "a.csv", "a") as fh:
        fh.write("not-a-time,m,I,0.1,1,2,3,C\nshort,row\n")
    (d / "wrong.csv").write_text("x,y\n1,2\n")
    a = FolderAdapter(spec(d), tmp_path / "st")
    assert len(a.poll()) == ROWS and a.bad_rows == 2
    assert a.describe()["files_rejected"] == 1


def _three_files(d):
    """a.csv and c.csv clean; b.csv has one 200,000-character field (over the csv module's limit) in row 1."""
    d.mkdir()
    now = time.time()
    for i, (name, p) in enumerate((("a", 0.1), ("b", 0.2), ("c", 0.3))):
        path = d / f"{name}.csv"
        with open(path, "w") as fh:
            fh.write("scored_at,p,x1,note\n")
            for j in range(5):
                fh.write(f"{T0 + 10 * i + j},{p},{j}.5,{'z' * 200_000 if name == 'b' and j == 0 else 'ok'}\n")
        os.utime(path, (now - 300 + 100 * i, now - 300 + 100 * i))
    return {"type": "folder", "path": str(d), "min_age_seconds": 0, "scan_interval_seconds": 0,
            "columns": {"ts": "scored_at", "output": "p", "features": ["x1"]},
            "constants": {"model_id": LOCAL_MODEL, "instrument": "all"}}


def test_an_oversized_csv_field_is_a_bad_row_and_the_scan_goes_on(tmp_path):
    a = FolderAdapter(_three_files(tmp_path / "in"), tmp_path / "st")
    got = a.poll()
    assert sorted(r.output for r in got) == [0.1] * 5 + [0.2] * 4 + [0.3] * 5
    assert a.bad_rows == 1 and a.last_error is None


def test_a_scan_that_fails_part_way_marks_no_file_read(tmp_path, monkeypatch):
    a = FolderAdapter(_three_files(tmp_path / "in"), tmp_path / "st")
    real = FolderAdapter._csv_page

    def fail_on_b(self, path, *rest):
        if path.name == "b.csv":
            raise RuntimeError("unexpected")
        return real(self, path, *rest)

    monkeypatch.setattr(FolderAdapter, "_csv_page", fail_on_b)
    assert a.poll() == [] and a.last_error
    monkeypatch.setattr(FolderAdapter, "_csv_page", real)
    assert sum(r.output == 0.1 for r in a.poll()) == 5        # the failed scan did not mark a.csv read


# -- Parquet ---------------------------------------------------------------------------------------
def write_parquet(path, recs, row_group_size=ROWS):
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = {"scored_at": [r.ts for r in recs], "model": [r.model_id for r in recs],
            "ticker": [r.instrument for r in recs], "p": [r.output for r in recs]}
    for n in FEATURES:
        cols[f"f_{n}"] = [r.features[n] for r in recs]
    cols["f_regime_zz"] = [r.features["regime_zz"] for r in recs]
    pq.write_table(pa.table(cols), path, row_group_size=row_group_size)


def test_parquet_resumes_mid_file_and_skips_a_file_without_footer(tmp_path):
    pytest.importorskip("pyarrow")
    d = tmp_path / "in"
    d.mkdir()
    recs = windows(0, 4)
    write_parquet(d / "a.parquet", recs)
    whole = (d / "a.parquet").read_bytes()
    (d / "b.parquet").write_bytes(whole[: len(whole) // 2])          # still being copied: no footer
    a = FolderAdapter(spec(d, batch_size=50), tmp_path / "st")
    first = a.poll(max_records=70)                                    # stops in the middle of a row group
    assert len(first) == 70
    a.commit(None)
    b = FolderAdapter(spec(d, batch_size=50), tmp_path / "st")
    rest = b.poll()
    assert len(rest) == 4 * ROWS - 70
    assert [r.ts for r in first + rest] == sorted(r.ts for r in recs)   # no row lost or repeated
    assert b.describe()["files_waiting"] == 1                         # the footerless copy
    (d / "b.parquet").write_bytes(whole)                              # copy finishes
    assert len(b.poll()) == 4 * ROWS                                  # a different file: read in full once
    b.commit(None)
    assert b.poll() == []


# -- through the real agent ---------------------------------------------------------------------------
def test_agent_crash_restart_loses_nothing_and_never_emits_a_window_twice(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    stream = windows(0, 6)
    write_csv(d / "part1.csv", stream[: 3 * ROWS])
    cfg = make_cfg(tmp_path, min_rows_per_window=30)
    clock = [utc(T0 + 2 * CADENCE + 30)]                  # mid window 2: windows 0 and 1 close, 2 stays open
    sink_a = Collector()
    a = new_agent(cfg, d, sink_a, clock)
    a.run_once()
    assert len(ids(sink_a)) == 2
    del a                                                 # kill -9: no close(), no flush
    write_csv(d / "part2.csv", stream[3 * ROWS:])
    clock[0] = utc(T0 + 6 * CADENCE + 30)
    sink_b = Collector()
    b = new_agent(cfg, d, sink_b, clock)
    for _ in range(3):
        b.run_once()
    b.close()
    got = ids(sink_a) + ids(sink_b)
    assert len(got) == 6 and len(set(got)) == 6, "a window was lost or emitted twice"
    assert b.replayed_windows_skipped == 0


def test_crash_between_spool_and_checkpoint_replays_without_loss_and_at_most_one_duplicate(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    write_csv(d / "part1.csv", windows(0, 6))
    cfg = make_cfg(tmp_path, min_rows_per_window=30)
    clock = [utc(T0 + 5 * CADENCE + 30)]                  # windows 0-4 close; 5 is open
    sink_a = Collector()
    a = new_agent(cfg, d, sink_a, clock)
    a.sources[0].commit = lambda before=None: None        # the process dies before any checkpoint lands
    a.run_once()
    assert len(ids(sink_a)) == 5
    assert not (cfg.state_dir / "sources" / "predictions.json").exists()
    del a
    clock[0] = utc(T0 + 6 * CADENCE + 30)
    sink_b = Collector()
    b = new_agent(cfg, d, sink_b, clock)
    for _ in range(3):
        b.run_once()
    b.close()
    both = ids(sink_a) + ids(sink_b)
    assert len(set(both)) == 6                            # nothing lost
    assert len(both) - len(set(both)) <= 1                # only the last emitted window can repeat
    assert b.replayed_windows_skipped == 5                # windows 0-4: up to the last emitted one (audit F5)
    # a duplicate carries the same window id, so the server (and the spool file name) dedupes it
    assert len({p.name for p in (cfg.state_dir / "spool").rglob("*.json")}) <= 6


def test_outcome_folder_joins_labels_and_commits_immediately(tmp_path):
    d, o = tmp_path / "in", tmp_path / "out"
    d.mkdir()
    o.mkdir()
    recs = windows(0, 2)
    with open(d / "p.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header() + ["pid"])
        w.writerows([*row[:3], "0.8", *row[4:], f"id{i}"] for i, row in enumerate(csv_rows(recs)))
    with open(o / "o.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "pid", "label", "known_at"])
        w.writerows([LOCAL_MODEL, f"id{i}", 1, f"{r.ts.timestamp() + 1:.3f}"] for i, r in enumerate(recs))
    pred = {**spec(d), "columns": {**COLUMNS, "prediction_id": "pid"}}
    outs = {"type": "folder", "path": str(o), "min_age_seconds": 0, "scan_interval_seconds": 0,
            "columns": {"model_id": "model", "prediction_id": "pid", "outcome": "label",
                        "outcome_observed_at": "known_at"}}
    cfg = make_cfg(tmp_path, output_kind="probability")
    taken: set[str] = set()
    sources = build_sources(pred, cfg.state_dir, "prediction", taken=taken)
    labels = build_sources(outs, cfg.state_dir, "outcome", taken=taken)
    sink = Collector()
    agent = EdgeAgent(cfg, sources, post=sink, outcome_sources=labels,
                      clock=lambda: utc(T0 + 2 * CADENCE + 30))
    assert agent.poll() == 2 * ROWS
    assert agent.outcomes.status()["matched"] > 0
    assert FolderAdapter(outs, cfg.state_dir, kind="outcome").poll() == []     # label file was checkpointed
    agent.close()


# -- --check and --status -----------------------------------------------------------------------------
def agent_json(tmp_path, source, **extra):
    cfg = {"tenant_id": "t", "install_id": "i", "models": {LOCAL_MODEL: "mm-wire-0001"},
           "secret_env": "EDGE_TEST_SECRET", "endpoints": ["https://x.invalid"],
           "state_dir": str(tmp_path / "state"), "cadence_seconds": CADENCE, "output_kind": "signed",
           "source": source, **extra}
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(cfg))
    return path


def test_check_reads_a_sample_writes_nothing_and_reports_counts_only(tmp_path, monkeypatch, capsys):
    d = tmp_path / "in"
    d.mkdir()
    write_csv(d / "a.csv", windows(0, 2))
    before = {p: p.stat().st_mtime_ns for p in d.rglob("*")}
    monkeypatch.setattr("sys.argv", ["edge_agent", "--config", str(agent_json(tmp_path, spec(d))), "--check"])
    with pytest.raises(SystemExit) as done:               # signed output: not probability-compatible
        cli.main()
    report = json.loads(capsys.readouterr().out)
    assert report["source_check"]["valid_shape"] == 2 * ROWS and report["source_check"]["invalid_shape"] == 0
    assert done.value.code == 2 and report["verdict"] == "NOT_READY"
    assert not (tmp_path / "state").exists()              # no offsets, no state of any kind
    assert {p: p.stat().st_mtime_ns for p in d.rglob("*")} == before
    text = json.dumps(report)
    for private in (FEATURES[0], "INSTR_SENTINEL", LOCAL_MODEL):
        assert private not in text


def test_check_reports_a_config_error_as_a_verdict(tmp_path, monkeypatch, capsys):
    bad = {**spec(tmp_path), "batch_size": 0}
    monkeypatch.setattr("sys.argv", ["edge_agent", "--config", str(agent_json(tmp_path, bad)), "--check"])
    with pytest.raises(SystemExit) as done:
        cli.main()
    out = json.loads(capsys.readouterr().out)
    assert done.value.code == 2 and out["verdict"] == "NOT_READY" and "batch_size" in out["config_error"]


def test_check_also_samples_the_outcome_source(tmp_path, monkeypatch, capsys):
    d, o = tmp_path / "in", tmp_path / "out"
    d.mkdir()
    o.mkdir()
    write_csv(d / "a.csv", windows(0, 1))
    (o / "o.csv").write_text("model,pid,label,known_at\nLOCAL_MODEL_SENTINEL,p1,1,1700000100\nx,y,maybe,1\n")
    outs = {"type": "folder", "path": str(o), "min_age_seconds": 0,
            "columns": {"model_id": "model", "prediction_id": "pid", "outcome": "label",
                        "outcome_observed_at": "known_at"}}
    cfgfile = agent_json(tmp_path, spec(d), outcome_source=outs, output_kind="probability")
    monkeypatch.setattr("sys.argv", ["edge_agent", "--config", str(cfgfile), "--check"])
    with pytest.raises(SystemExit):
        cli.main()
    out = json.loads(capsys.readouterr().out)
    assert out["outcome_source_check"]["valid_shape"] == 1 and out["outcome_source_check"]["invalid_shape"] == 1
    assert out["verdict"] == "NOT_READY"


def test_status_shows_per_source_position_and_lag_without_names_or_values(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    write_csv(d / "secret-customer-export.csv", windows(0, 3))
    cfg = make_cfg(tmp_path)
    sink = Collector()
    clock = [utc(T0 + 3 * CADENCE + 30)]
    agent = new_agent(cfg, d, sink, clock, name="exports")
    agent.run_once()
    agent.write_status(force=True)
    report = read_status(cfg.state_dir)
    (src,) = report["sources"]
    assert src["name"] == "exports" and src["type"] == "folder"
    assert src["files_tracked"] == 1 and src["files_done"] == 1
    snap = report["agent_snapshot"]["sources"][0]
    assert snap["rows_committed"] == 3 * ROWS and snap["rows_uncommitted"] == 0 and snap["blocked"] is False
    text = json.dumps(report)
    assert "secret-customer-export" not in text and FEATURES[0] not in text and "INSTR_SENTINEL" not in text
    agent.close()


# -- a caught-up source is healthy, not "no rows yet" (pre-tag review, edgefix2 2026-10-30) ---------------------
def _caught_up(tmp_path):
    """The agent read every file, closed and queued every window, committed, and stopped."""
    d = tmp_path / "in"
    d.mkdir()
    write_csv(d / "a.csv", windows(0, 2))
    cfg = make_cfg(tmp_path)
    clock = [utc(T0 + 2 * CADENCE + 30)]
    agent = new_agent(cfg, d, Collector(), clock)
    agent.run_once()
    agent.close(keep_open=True)
    assert FolderAdapter(spec(d), cfg.state_dir).poll() == []      # nothing left past the committed position
    return cfg, d


def test_sample_of_a_caught_up_folder_reads_the_files_still_there(tmp_path):
    cfg, d = _caught_up(tmp_path)
    s = FolderAdapter(spec(d), cfg.state_dir).sample(1000)
    assert len(s.records) == 2 * ROWS and s.bad == 0


def test_check_verdict_fresh_empty_is_no_rows_yet_caught_up_and_cleaned_is_caught_up(tmp_path):
    import dataclasses
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / "in").mkdir()
    fcfg = dataclasses.replace(make_cfg(fresh), output_kind="probability")
    rep = cli.check_source(fcfg, adapters=[FolderAdapter(spec(fresh / "in"), fcfg.state_dir)])
    assert rep["verdict"] == "NO_ROWS_YET" and rep["message"].startswith("no rows yet")   # run 1 of a fresh install

    cfg, d = _caught_up(tmp_path)
    pcfg = dataclasses.replace(cfg, output_kind="probability")
    (d / "a.csv").unlink()                         # the partner cleans processed files up overnight
    rep = cli.check_source(pcfg, adapters=[FolderAdapter(spec(d), cfg.state_dir)])
    assert rep["verdict"] == "CAUGHT_UP" and rep["source_check"]["rows_checked"] == 0
    assert rep["message"].startswith("caught up")


def test_check_cli_exits_zero_for_a_caught_up_source(tmp_path, monkeypatch, capsys):
    cfg, d = _caught_up(tmp_path)
    (d / "a.csv").unlink()
    path = agent_json(tmp_path, spec(d), output_kind="probability")
    raw = json.loads(path.read_text())
    raw["state_dir"] = str(cfg.state_dir)
    path.write_text(json.dumps(raw))
    monkeypatch.setattr("sys.argv", ["edge_agent", "--config", str(path), "--check"])
    cli.main()                                     # no SystemExit: exit 0, the image HEALTHCHECK stays healthy
    assert json.loads(capsys.readouterr().out)["verdict"] == "CAUGHT_UP"


# -- kitrun6 C-03 (lane edgefix6): a folder's rows are readable only min_age after the write ----------------------------
def test_clock_does_not_close_a_window_before_its_folder_rows_are_readable(tmp_path):
    """The journal said window_rejected ... ingest_degraded and windows were skipped: the clock closed each window at its
    end, and the rows of the file written at that time became readable min_age_seconds (+ a scan) later, as late rows.
    A source with an arrival delay holds the clock close by that delay plus one window; the status counts withheld
    windows and late rows."""
    from edge_agent import EdgeAgent
    from tests.edge_agent.synth import CADENCE, Collector, T0, make_cfg, make_stream, utc

    class Delayed:                                  # stands in for the folder source: same delay attribute
        arrival_delay_seconds = 35.0
        name, kind = "predictions", "prediction"

        def __init__(self, rows):
            self.rows, self.ready = rows, []

        def poll(self, max_records=10_000):
            out, self.ready = self.ready, []
            return out

        def commit(self, before=None):
            pass

        def describe(self):
            return {"type": "folder"}

    cfg = make_cfg(tmp_path)
    stream = make_stream(1, rows=60)                # one window's rows, [T0, T0 + CADENCE)
    src = Delayed(stream)
    clock = [utc(T0 + CADENCE + 1)]
    agent = EdgeAgent(cfg, [src], post=Collector(), clock=lambda: clock[0])
    assert agent.close_lateness_seconds == 35.0 + CADENCE
    agent.run_once()                                # the window's end has passed; its file is not readable yet
    clock[0] = utc(T0 + CADENCE + 40)
    src.ready = list(stream)                        # min_age + scan later the rows are read
    agent.run_once()
    clock[0] = utc(T0 + 2 * CADENCE + 36 + 1)
    agent.run_once()                                # now the clock closes it, with every row in it
    st = agent.status()["models"]
    (m,) = st.values()
    assert m["late_rows_dropped"] == 0 and m["collection"].get("rows") == 60, m
    assert "withheld" in m
    agent.close()

    plain = EdgeAgent(make_cfg(tmp_path / "b"), [], post=Collector(), clock=lambda: clock[0])
    assert plain.close_lateness_seconds == plain.cfg.lateness_seconds   # a log tail or a database: the plain grace
    plain.close()


def test_folder_source_declares_its_arrival_delay(tmp_path):
    a = FolderAdapter({"type": "folder", "path": str(tmp_path), "min_age_seconds": 30, "scan_interval_seconds": 5,
                       "columns": {"ts": "ts", "model_id": "m", "instrument": "i", "output": "p", "features": ["x"]}},
                      tmp_path / "st", kind="prediction")
    assert a.arrival_delay_seconds == 35.0
