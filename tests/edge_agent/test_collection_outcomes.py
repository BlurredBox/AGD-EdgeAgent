"""Local-only collection contract, causal outcome joins and real producer coverage."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta

import pytest

from corpus import onboarding, signal_vector as sv
from edge_agent import EdgeAgent, Record
from edge_agent.__main__ import check_source
from edge_agent.adapters import InProcessAdapter, JsonlTailAdapter, JsonlOutcomeAdapter
from edge_agent.outcomes import OutcomeJoin
from tests.edge_agent.synth import Collector, LOCAL_MODEL, T0, make_cfg, make_stream, utc


def test_signed_producer_declares_both_pinned_paths_incompatible():
    report = onboarding.collection_contract("signed")
    for path in onboarding.PATHS:
        assert not report["paths"][path]["output_compatible"]
        assert "positive_decision_rate" in report["paths"][path]["incompatible_fields"]
    assert all(row["output_compatible"] for row in onboarding.collection_contract("probability")["fields"])


def test_source_preflight_is_local_readonly_and_does_not_disclose_rows(tmp_path):
    source = tmp_path / "private.jsonl"
    raw = {"ts": utc(T0).isoformat(), "model_id": LOCAL_MODEL,
           "instrument": "PRIVATE_INSTRUMENT", "features": {"PRIVATE_COLUMN": 438.237}, "output": 0.8}
    source.write_text(json.dumps(raw) + "\nnot-json\n")
    cfg = make_cfg(tmp_path, output_kind="probability")
    before = source.read_bytes()
    report = check_source(cfg, str(source))
    assert report["source_check"]["valid_shape"] == 1
    assert report["source_check"]["invalid_shape"] == 1
    assert not report["scoring_verified"]
    assert not cfg.state_dir.exists()
    assert source.read_bytes() == before
    for private in ("PRIVATE_INSTRUMENT", "PRIVATE_COLUMN", "438.237", LOCAL_MODEL):
        assert private not in json.dumps(report)


def test_preflight_rejects_empty_source_and_uses_operator_required_contract(tmp_path):
    source = tmp_path / "sample.jsonl"
    source.write_text("")
    cfg = make_cfg(tmp_path)
    assert check_source(cfg, str(source))["verdict"] == "NOT_READY"
    source.write_text(json.dumps({"ts": utc(T0).isoformat(), "model_id": LOCAL_MODEL,
                                 "instrument": "i", "features": {"x": 1}, "output": -0.2}) + "\n")
    assert check_source(cfg, str(source))["verdict"] == "NOT_READY"
    spec = onboarding.derive_required({1: ["score_mean"]}, "operator-pinned-artifact")
    report = check_source(cfg, str(source), required=spec)
    assert report["verdict"] == "SOURCE_COMPATIBLE"
    assert report["contract"]["required_source"] == "operator-pinned-artifact"


def test_adapters_preserve_local_prediction_ids_and_outcome_known_time(tmp_path):
    at = utc(T0)
    hook = InProcessAdapter()
    assert hook.observe(at, LOCAL_MODEL, "private", {"x": 1}, 0.8,
                        prediction_id="private-id", outcome=1, outcome_observed_at=at)
    assert hook.poll()[0].outcome == 1
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"ts": at.isoformat(), "model_id": LOCAL_MODEL,
                                 "instrument": "private", "features": {"x": 1}, "output": 0.8,
                                 "prediction_id": "private-id", "outcome": 1,
                                 "outcome_observed_at": at.isoformat()}) + "\n")
    rec = JsonlTailAdapter(source, tmp_path / "offset").poll()[0]
    assert rec.prediction_id == "private-id" and rec.outcome_observed_at == at
    with pytest.raises(ValueError, match="outcome_observed_at"):
        Record.make(at, LOCAL_MODEL, "i", {}, 0.5, outcome=1)


def test_outcome_join_is_model_scoped_persistent_bounded_and_conflict_safe(tmp_path):
    path = tmp_path / "join.sqlite"
    store = OutcomeJoin(path, max_records=2, max_age_seconds=3600)
    rec = Record.make(utc(T0), "a", "i", {}, 0.8, prediction_id="same")
    store.register(rec, utc(T0))
    assert store.record("b", "same", 1, utc(T0 + 1)) == "unknown"
    assert store.record("a", "same", 1, utc(T0 + 1)) == "matched"
    assert store.record("a", "same", 1, utc(T0 + 1)) == "duplicate"
    with pytest.raises(ValueError, match="conflicting"):
        store.record("a", "same", 0, utc(T0 + 2))
    store.close()
    store = OutcomeJoin(path, max_records=2, max_age_seconds=3600)
    assert store.register(rec, utc(T0 + 2))   # restart must accept an un-emitted prediction
    assert store.resolve(rec, utc(T0 + 2)).outcome == 1
    assert store.resolve(rec, utc(T0)).outcome is None
    for i in range(3):
        store.register(Record.make(utc(T0 + i + 3), "a", "i", {}, 0.5,
                                   prediction_id=f"new{i}"), utc(T0 + 6))
    assert store.status()["retained"] == 2
    assert path.stat().st_mode & 0o777 == 0o600
    store.close()


def test_late_label_is_not_used_by_any_new_window(tmp_path):
    store = OutcomeJoin(tmp_path / "join.sqlite", max_records=10, max_age_seconds=3600)
    rec = Record.make(utc(T0), "a", "i", {}, 0.8, prediction_id="private-id")
    store.register(rec, utc(T0))
    store.mark_emitted("a", utc(T0), utc(T0 + 60))
    assert store.record("a", "private-id", 1, utc(T0 + 70)) == "late"
    assert store.resolve(rec, utc(T0 + 120)).outcome is None
    assert store.status()["late"] == 1
    assert "private-id" not in json.dumps(store.status())
    store.close()


def _probability_run(tmp_path, *, missing_last=False):
    cfg = make_cfg(tmp_path, output_kind="probability")
    sink = Collector()
    clock = [utc(T0)]
    agent = EdgeAgent(cfg, post=sink, clock=lambda: clock[0])
    recs = [replace(r, output=0.2 if i % 2 else 0.8, prediction_id=f"PRIVATE_PRED_{i}")
            for i, r in enumerate(make_stream(1))]
    for i, rec in enumerate(recs):
        clock[0] = rec.ts
        agent.ingest(rec)
        if not (missing_last and i == len(recs) - 1):
            assert agent.record_outcome(rec.model_id, rec.prediction_id, i % 2, rec.ts) == "matched"
    clock[0] = utc(T0 + 60)
    agent.flush()
    return agent, sink, recs


def test_real_producer_emits_calibration_only_for_complete_original_window(tmp_path):
    agent, sink, recs = _probability_run(tmp_path)
    z = sink.bodies()[0]["z_t"]
    labels = [i % 2 for i in range(len(recs))]
    probs = [r.output for r in recs]
    assert z["brier"] == pytest.approx(sv.brier(probs, labels))
    assert z["nll"] == pytest.approx(sv.nll(probs, labels))
    assert z["ci_width"] == pytest.approx(sv.ci_width(0, len(recs)))
    health = agent.status()["models"][LOCAL_MODEL]["collection"]
    assert health["label_coverage"] == 1 and health["outcome_metrics_available"]
    assert "PRIVATE_PRED_" not in json.dumps(sink.bodies())
    agent.close()


def test_two_jsonl_sources_merge_labels_before_original_window_closes(tmp_path):
    recs = [replace(r, output=0.8, prediction_id=f"private{i}")
            for i, r in enumerate(make_stream(2))]
    predictions, labels = tmp_path / "predictions.jsonl", tmp_path / "outcomes.jsonl"
    predictions.write_text("\n".join(json.dumps({"ts": r.ts.isoformat(), "model_id": r.model_id,
                                                "instrument": r.instrument, "features": r.features,
                                                "output": r.output, "prediction_id": r.prediction_id})
                                      for r in recs) + "\n")
    labels.write_text("\n".join(json.dumps({"model_id": r.model_id, "prediction_id": r.prediction_id,
                                          "outcome": 1, "outcome_observed_at": r.ts.isoformat()})
                                for r in recs) + "\n")
    sink = Collector()
    agent = EdgeAgent(make_cfg(tmp_path, output_kind="probability"),
                      sources=[JsonlTailAdapter(predictions, tmp_path / "prediction.offset")],
                      outcome_sources=[JsonlOutcomeAdapter(labels, tmp_path / "outcome.offset")],
                      post=sink, clock=lambda: utc(T0 + 120))
    assert agent.poll() == len(recs)
    agent.flush()
    assert len(sink.bodies()) == 2
    assert all(b["z_t"]["brier"] == pytest.approx(0.04) for b in sink.bodies())
    assert agent.outcomes.status()["late"] == 0
    agent.close()


def test_incomplete_labels_do_not_masquerade_as_window_calibration(tmp_path):
    agent, sink, _ = _probability_run(tmp_path, missing_last=True)
    z = sink.bodies()[0]["z_t"]
    assert "brier" not in z and "nll" not in z
    health = agent.status()["models"][LOCAL_MODEL]["collection"]
    assert 0 < health["label_coverage"] < 1 and not health["outcome_metrics_available"]
    assert not health["paths"][onboarding.PATH_LIVE]["input_ready"]
    agent.close()


def test_future_labels_and_duplicate_prediction_ids_cannot_inflate_window(tmp_path):
    clock = [utc(T0)]
    agent = EdgeAgent(make_cfg(tmp_path, output_kind="probability"), clock=lambda: clock[0])
    rec = Record.make(clock[0], LOCAL_MODEL, "i", {"x": 1}, 0.8, prediction_id="one")
    agent.ingest(rec)
    agent.ingest(rec)
    assert agent.outcomes.status()["duplicate_predictions"] == 1
    assert len(next(iter(agent.models[LOCAL_MODEL].cutter._open.values()))) == 1
    with pytest.raises(ValueError, match="future"):
        agent.record_outcome(LOCAL_MODEL, "one", 1, clock[0] + timedelta(seconds=1))
    with pytest.raises(ValueError, match="ingest clock"):
        agent.ingest(replace(rec, prediction_id="two", outcome=1,
                             outcome_observed_at=clock[0] + timedelta(seconds=1)))
    agent.close()


@pytest.mark.parametrize("label", [True, 0.5, -1, 2, "1"])
def test_direct_record_construction_cannot_bypass_binary_label_validation(label):
    with pytest.raises(ValueError, match="integer 0 or 1"):
        Record(utc(T0), LOCAL_MODEL, "i", {}, 0.8, "one", label, utc(T0 + 1))


def test_direct_record_construction_rejects_naive_and_pre_prediction_label_times():
    with pytest.raises(ValueError, match="timezone"):
        Record(utc(T0).replace(tzinfo=None), LOCAL_MODEL, "i", {}, 0.8)
    with pytest.raises(ValueError, match="timezone"):
        Record(utc(T0), LOCAL_MODEL, "i", {}, 0.8, "one", 1, utc(T0).replace(tzinfo=None))
    with pytest.raises(ValueError, match="before its prediction"):
        Record(utc(T0), LOCAL_MODEL, "i", {}, 0.8, "one", 1, utc(T0 - 1))
    with pytest.raises(ValueError, match="requires an outcome"):
        Record.make(utc(T0), LOCAL_MODEL, "i", {}, 0.8, outcome_observed_at=utc(T0))


def test_idle_expiry_removes_outcome_metadata_without_new_predictions(tmp_path):
    store = OutcomeJoin(tmp_path / "join.sqlite", max_records=10, max_age_seconds=10)
    rec = Record.make(utc(T0), "a", "i", {}, 0.8, prediction_id="one")
    store.register(rec, utc(T0))
    store.expire(utc(T0 + 11))
    assert store.status()["retained"] == 0 and store.status()["evicted"] == 1
    assert store.record("a", "one", 1, utc(T0 + 11)) == "unknown"
    store.close()


def test_spool_failure_does_not_seal_unsent_predictions_or_checkpoint_signals(tmp_path, monkeypatch):
    cfg, sink = make_cfg(tmp_path), Collector()
    clock = [utc(T0)]
    agent = EdgeAgent(cfg, post=sink, clock=lambda: clock[0])
    recs = [replace(r, prediction_id=f"private{i}") for i, r in enumerate(make_stream(1))]
    for rec in recs:
        clock[0] = rec.ts
        agent.ingest(rec)
    clock[0] = utc(T0 + 60)

    def fail_put(body, end):
        raise OSError("disk full")

    monkeypatch.setattr(agent.sender, "enqueue", fail_put)
    with pytest.raises(OSError, match="disk full"):
        agent.flush()
    model = agent.models[LOCAL_MODEL]
    assert model.failed == 1 and model.emitted == 0
    assert model.collection["disposition"] == "spool_failed_requires_local_source_replay"
    assert not model.signals.state_path.exists()
    assert agent.outcomes.db.execute("SELECT count(*) FROM predictions WHERE emitted<>0").fetchone()[0] == 0
    agent.close()
    recovered = OutcomeJoin(cfg.state_dir / "outcomes.sqlite", max_records=1000, max_age_seconds=3600)
    assert recovered.register(recs[0], utc(T0 + 61))
    recovered.close()


def test_prediction_sealing_happens_after_a_durable_spool_copy(tmp_path, monkeypatch):
    agent = EdgeAgent(make_cfg(tmp_path), post=Collector(), clock=lambda: utc(T0 + 60))
    for i, rec in enumerate(make_stream(1)):
        agent.ingest(replace(rec, prediction_id=f"private{i}"))
    original = agent.outcomes.mark_emitted
    seen = []

    def checked(model, start, end):
        seen.append(agent.sender.pending_count())
        assert agent.sender.pending_count() == 1
        original(model, start, end)

    monkeypatch.setattr(agent.outcomes, "mark_emitted", checked)
    agent.flush()
    assert seen == [1]
    assert agent.outcomes.db.execute("SELECT count(*) FROM predictions WHERE emitted=1").fetchone()[0] == 220
    agent.close()


def test_skipped_small_windows_are_not_falsely_sealed_as_emitted(tmp_path):
    agent = EdgeAgent(make_cfg(tmp_path), post=Collector(), clock=lambda: utc(T0 + 60))
    agent.ingest(Record.make(utc(T0), LOCAL_MODEL, "i", {"x": 1}, 0.2, prediction_id="one"))
    agent.flush()
    assert agent.models[LOCAL_MODEL].collection["disposition"] == "skipped_insufficient_usable_rows"
    assert agent.outcomes.db.execute("SELECT emitted FROM predictions").fetchone()[0] == 0
    agent.close()


def test_computation_failure_is_visible_unsealed_and_does_not_log_raw_exception(tmp_path, monkeypatch, capsys):
    agent = EdgeAgent(make_cfg(tmp_path), post=Collector(), clock=lambda: utc(T0 + 60))
    agent.ingest(Record.make(utc(T0), LOCAL_MODEL, "i", {"x": 1}, 0.2, prediction_id="one"))

    def fail_process(*args):
        raise ValueError("PRIVATE_FEATURE_VALUE_FROM_EXCEPTION")

    monkeypatch.setattr(agent.models[LOCAL_MODEL].signals, "process", fail_process)
    agent.flush()
    model = agent.models[LOCAL_MODEL]
    assert model.failed == 1 and model.emitted == 0
    assert model.collection["disposition"] == "computation_failed_requires_local_source_replay"
    assert agent.outcomes.db.execute("SELECT emitted FROM predictions").fetchone()[0] == 0
    captured = capsys.readouterr()
    text = captured.out + captured.err + (agent.cfg.state_dir / "alerts.jsonl").read_text()
    assert "PRIVATE_FEATURE_VALUE_FROM_EXCEPTION" not in text
    agent.close()


def test_out_of_range_probability_is_dropped_under_its_own_reason(tmp_path):
    cfg = make_cfg(tmp_path, output_kind="probability")
    agent = EdgeAgent(cfg, post=Collector(), clock=lambda: utc(T0))
    rec = replace(make_stream(1)[0], output=1.5)
    agent.ingest(rec)
    reasons = agent.models[rec.model_id].invalid[agent.models[rec.model_id].cutter.bucket(rec.ts)]
    assert reasons == {"output_out_of_range": 1}
    assert agent.outcomes.status()["retained"] == 0


def test_status_report_is_read_only_counts_and_discloses_nothing(tmp_path, capsys, monkeypatch):
    from edge_agent import __main__ as cli
    from edge_agent.status import read_status

    agent, sink, recs = _probability_run(tmp_path)
    agent.close()
    state = agent.cfg.state_dir
    before = {p: p.stat().st_mtime_ns for p in state.rglob("*") if p.is_file()}
    report = read_status(state)
    assert report["state_dir_exists"] and report["agent_snapshot"]["models"]
    assert report["spool"]["queued"] + agent.sender.stats.sent >= 1
    assert report["outcome_join_rows_retained"] is not None
    text = json.dumps(report)
    assert "PRIVATE_PRED_" not in text
    assert {p: p.stat().st_mtime_ns for p in state.rglob("*") if p.is_file()} == before
    cfg_file = tmp_path / "agent.json"
    cfg_file.write_text(json.dumps({"state_dir": str(state)}))
    monkeypatch.setattr("sys.argv", ["edge_agent", "--config", str(cfg_file), "--status"])
    with pytest.raises(SystemExit) as done:
        cli.main()
    assert done.value.code == 0
    assert json.loads(capsys.readouterr().out)["state_dir_exists"] is True
    assert read_status(tmp_path / "absent") == {"state_dir_exists": False}


def _write_jsonl(path, recs, mode="w"):
    with open(path, mode) as fh:
        for r in recs:
            fh.write(json.dumps({"ts": r.ts.isoformat(), "model_id": r.model_id, "instrument": r.instrument,
                                 "features": r.features, "output": r.output}) + "\n")


def test_sigterm_on_the_real_loop_flushes_the_open_window_and_a_restart_neither_loses_nor_repeats(tmp_path):
    import os
    import signal
    import threading

    log = tmp_path / "model.jsonl"
    stream = make_stream(3)                       # windows 0, 1, 2
    _write_jsonl(log, stream[:220 + 110])         # window 0 whole, window 1 half full: it is still open
    clock = [utc(T0 + 60 + 40)]                   # inside window 1: the clock closes nothing
    cfg = make_cfg(tmp_path, reference_windows=1)

    def new_agent(sink):
        return EdgeAgent(cfg, [JsonlTailAdapter(log, cfg.state_dir / "source.offset")],
                         post=sink, clock=lambda: clock[0])

    sink = Collector()
    agent = new_agent(sink)
    threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    agent.run_forever(interval_seconds=0.05)      # SIGTERM arrives mid-loop; the loop returns
    assert len(sink.bodies()) == 2, "the half-filled window was not flushed and sent"
    assert agent.sender.pending_count() == 0      # sent, nothing stranded in the spool
    offset = int((cfg.state_dir / "source.offset").read_text())
    assert offset == log.stat().st_size           # every line read was accounted for
    assert (cfg.state_dir / "status.json").is_file()

    # Restart on the same state: no row is re-read, no window is re-emitted.
    sink2 = Collector()
    again = new_agent(sink2)
    again.run_once()
    again.close()
    assert sink2.bodies() == []
    # More rows arrive later: only the new window is emitted, nothing already sent repeats.
    _write_jsonl(log, stream[330:], mode="a")
    clock[0] = utc(T0 + 180 + 30)
    sink3 = Collector()
    third = new_agent(sink3)
    third.run_once()
    third.close()
    assert len(sink3.bodies()) == 2               # the rest of window 1 (a new partial) and window 2
