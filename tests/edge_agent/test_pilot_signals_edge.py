"""G52 edge half: upstream status sources (file, Airflow REST) and early proxy labels.

Proxy labels are joined in their own store and leave as per-window counts after the proxy's delay; they
never touch the outcome join, the window's z_t or the calibration. Server half: tests/test_pilot_signals_live.py.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta

import pytest

from blurredbox_agd.ingest import IngestError
from edge_agent import EdgeAgent
from edge_agent.adapters import JsonlTailAdapter
from edge_agent.aux_outbox import AuxOutbox
from edge_agent.mapping import ConfigError
from edge_agent.proxy import PROXY_PATH, split_proxy_specs
from edge_agent.sources import build_proxy_sources
from edge_agent.upstream import (KIND_STATUSES, UPSTREAM_PATH, AirflowSource, FileStatusSource, build_upstream_sources,
                                 clean_event)
from tests.edge_agent.synth import Collector, LOCAL_MODEL, T0, make_cfg, make_stream, utc


def _ev(i="e1", **kw):
    return {"event_id": i, "kind": "deploy", "status": "success", "subject": "scorer@2.0",
            "occurred_at": utc(T0).isoformat(), **kw}


def test_allowlists_match_the_server():
    from src.api import pilot_signals as ps
    assert KIND_STATUSES == ps.KIND_STATUSES
    assert UPSTREAM_PATH.endswith("/upstream-events") and PROXY_PATH.endswith("/proxy-labels")


def test_clean_event_applies_the_server_rules():
    now = utc(T0 + 60)
    assert clean_event(_ev(), now)["subject"] == "scorer@2.0"
    for bad in (_ev(kind="login"), _ev(status="running"), _ev(subject="two words"), _ev(i="a b"),
                {**_ev(), "message": "hi"}, _ev(occurred_at="2026-01-01T00:00:00"),
                _ev(occurred_at=utc(T0 + 3600).isoformat()), _ev(occurred_at=utc(T0 - 40 * 86400).isoformat()),
                _ev(freshness_lag_s=3.0), _ev(kind="job_run", status="late", freshness_lag_s=-1), "x", None):
        assert clean_event(bad, now) is None, bad


def test_file_source_reads_valid_lines_drops_bad_ones_and_resumes_after_commit(tmp_path):
    f = tmp_path / "ev.jsonl"
    f.write_text(json.dumps(_ev("a")) + "\nnot json\n" + json.dumps(_ev("b", kind="job_run", status="failure")) + "\n"
                 + json.dumps({**_ev("c"), "note": "free text"}) + "\n" + '{"event_id": "half')
    src = FileStatusSource({"type": "file", "path": str(f)}, tmp_path / "st", name="u", clock=lambda: utc(T0 + 60))
    got = src.poll()
    assert [e["event_id"] for e in got] == ["a", "b"] and src.dropped == 2   # the half-written line waits
    src.commit()
    again = FileStatusSource({"type": "file", "path": str(f)}, tmp_path / "st", name="u", clock=lambda: utc(T0 + 60))
    assert again.poll() == []                                                  # resumed past what was committed
    with open(f, "a") as fh:
        fh.write('-line"}\n' + json.dumps(_ev("d")) + "\n")
    assert [e["event_id"] for e in again.poll()] == ["d"]
    f.write_text(json.dumps(_ev("z")) + "\n")                                  # rotated / truncated: start over
    assert [e["event_id"] for e in again.poll()] == ["z"]


def test_airflow_source_is_read_only_maps_states_and_keeps_secrets_in_env(tmp_path):
    now = utc(T0 + 3600)
    calls = []

    def fetch(url, headers):
        calls.append((url, headers))
        return {"dag_runs": [
            {"dag_run_id": "r1", "state": "success", "start_date": utc(T0).isoformat(), "end_date": utc(T0 + 60).isoformat()},
            {"dag_run_id": "r2", "state": "failed", "start_date": utc(T0).isoformat(), "end_date": utc(T0 + 90).isoformat()},
            {"dag_run_id": "r3", "state": "running", "start_date": utc(T0).isoformat(), "end_date": None},
            {"dag_run_id": "r4", "state": "queued", "start_date": None, "end_date": None}, "junk"]}
    spec = {"type": "airflow", "base_url": "https://airflow.internal:8080", "dag_ids": ["features_daily"],
            "token_env": "EDGE_AIRFLOW_TOKEN", "late_after_seconds": 1800}
    src = AirflowSource(spec, tmp_path, name="af", env={"EDGE_AIRFLOW_TOKEN": "tok-123"}, clock=lambda: now, fetch=fetch)
    evs = src.poll()
    assert sorted(e["status"] for e in evs) == ["failure", "late", "success"]
    assert all(e["kind"] == "job_run" and e["subject"] == "features_daily" for e in evs)
    assert calls[0][0].endswith("/api/v1/dags/features_daily/dagRuns?order_by=-start_date&limit=10")
    assert calls[0][1]["Authorization"] == "Bearer tok-123"
    src.commit()
    assert "tok-123" not in json.dumps([e for e in evs]) and "tok-123" not in (tmp_path / "sources" / "af.upstream.json").read_text()
    src._last_poll = 0.0
    assert src.poll() == []                                                    # already reported
    bad = AirflowSource(spec, tmp_path, name="af2", env={"EDGE_AIRFLOW_TOKEN": "t"}, clock=lambda: now,
                        fetch=lambda u, h: (_ for _ in ()).throw(OSError("down")))
    assert bad.poll() == [] and bad.errors == 1
    with pytest.raises(ConfigError, match="token_env"):
        AirflowSource({**spec, "token": "inline"}, tmp_path, name="x", env={})
    with pytest.raises(ConfigError, match="https"):
        AirflowSource({**spec, "base_url": "http://airflow.internal"}, tmp_path, name="x", env={})
    with pytest.raises(ConfigError):
        AirflowSource({k: v for k, v in spec.items() if k != "token_env"}, tmp_path, name="x", env={})
    with pytest.raises(ConfigError, match="unknown"):
        build_upstream_sources({"type": "file", "path": "/x", "write": True}, tmp_path)
    with pytest.raises(ConfigError, match="type"):
        build_upstream_sources({"type": "ftp"}, tmp_path)


def test_agent_sends_upstream_events_through_the_outbox_and_survives_an_outage(tmp_path):
    f = tmp_path / "ev.jsonl"
    f.write_text(json.dumps(_ev("a")) + "\n" + json.dumps(_ev("b", kind="feature_freshness", status="stale",
                                                               freshness_lag_s=7200)) + "\n")
    sink, clock = Collector(), [utc(T0 + 60)]
    src = build_upstream_sources({"type": "file", "path": str(f)}, make_cfg(tmp_path).state_dir,
                                 clock=lambda: clock[0])
    agent = EdgeAgent(make_cfg(tmp_path), post=sink, clock=lambda: clock[0], upstream_sources=src)
    sink.raises = OSError("link down")
    agent.run_once()
    assert agent.outbox.status()["pending"] == 1 and sink.requests == []
    sink.raises = None
    clock[0] = utc(T0 + 4000)
    agent.tick()
    posts = [(p, json.loads(b)) for _, p, b, _ in sink.requests if p == UPSTREAM_PATH]
    assert len(posts) == 1 and posts[0][1]["source"] == "edge"
    assert [e["event_id"] for e in posts[0][1]["events"]] == ["a", "b"]
    assert agent.outbox.status()["pending"] == 0
    agent.tick()
    assert len([1 for _, p, _, _ in sink.requests if p == UPSTREAM_PATH]) == 1   # sent once
    assert agent.status()["upstream"]["sources"][0]["events_read"] == 2
    agent.close()


def test_outbox_moves_a_deterministic_rejection_aside_and_keeps_order_on_transport_failure(tmp_path):
    outcomes = [IngestError("x", [("u", 422, "bad")]), IngestError("x", [("u", None, "down")]), None, None]
    sent = []

    def send(path, body):
        o = outcomes.pop(0)
        if o:
            raise o
        sent.append(body)
    box = AuxOutbox(tmp_path / "aux", send, clock=lambda: utc(T0))
    for n in range(3):
        box.put("/p", {"n": n})
    assert box.drain() == 0                                  # n=0 rejected (4xx), n=1 transport failure -> stop
    assert box.status()["rejected"] == 1 and box.status()["pending"] == 2
    assert box.drain() == 0                                  # backing off
    assert box.drain(force=True) == 2 and [b["n"] for b in sent] == [1, 2]


def _proxy_world(tmp_path, delay=100.0):
    recs = [replace(r, output=0.8, prediction_id=f"p{i}") for i, r in enumerate(make_stream(2))]
    predictions, proxy = tmp_path / "predictions.jsonl", tmp_path / "chargebacks.jsonl"
    predictions.write_text("\n".join(json.dumps({"ts": r.ts.isoformat(), "model_id": r.model_id,
                                                "instrument": r.instrument, "features": r.features,
                                                "output": r.output, "prediction_id": r.prediction_id})
                                     for r in recs) + "\n")
    # every 10th prediction was charged back, known 30 s after the prediction
    proxy.write_text("\n".join(json.dumps({"model_id": r.model_id, "prediction_id": r.prediction_id,
                                           "outcome": 1 if i % 10 == 0 else 0,
                                           "outcome_observed_at": (r.ts + timedelta(seconds=30)).isoformat()})
                               for i, r in enumerate(recs)) + "\n")
    cfg = make_cfg(tmp_path, output_kind="probability")
    spec = {"type": "jsonl", "path": str(proxy), "proxy_name": "chargeback_6h", "delay_seconds": delay}
    return recs, predictions, spec, cfg


@pytest.mark.slow
def test_proxy_labels_leave_as_per_window_counts_after_their_delay_and_never_touch_full_labels(tmp_path):
    recs, predictions, spec, cfg = _proxy_world(tmp_path)
    sink, clock = Collector(), [utc(T0 + 400)]
    proxies = build_proxy_sources(spec, cfg.state_dir)
    agent = EdgeAgent(cfg, sources=[JsonlTailAdapter(predictions, tmp_path / "prediction.offset")],
                      proxy_sources=proxies, post=sink, clock=lambda: clock[0])
    assert agent.poll() == len(recs)
    agent.flush()
    # the full-label join and the windows know nothing of the proxy
    assert agent.outcomes.status()["matched"] == 0 and agent.outcomes.status()["late"] == 0
    for body in sink.bodies():
        assert "brier" not in body["z_t"] and "outcome_labels" not in body and "chargeback" not in json.dumps(body)
    assert agent.status()["models"][LOCAL_MODEL]["collection"]["labelled_rows"] == 0
    # before the delay: nothing sent; after: one row per window, counts only
    clock[0] = utc(T0 + 60 + 50)
    agent.tick()
    assert [p for _, p, _, _ in sink.requests if p == PROXY_PATH] == []
    clock[0] = utc(T0 + 400)
    agent.tick()
    posts = [json.loads(b) for _, p, b, _ in sink.requests if p == PROXY_PATH]
    assert len(posts) == 1 and posts[0]["install_id"] == "install-1"
    rows = sorted(posts[0]["rows"], key=lambda r: r["window_end"])
    assert len(rows) == 2 and all(r["proxy_name"] == "chargeback_6h" for r in rows)
    per_window = [recs[:220], recs[220:]]
    for row, w in zip(rows, per_window):
        assert row["n_labelled"] == len(w)
        assert row["n_errors"] == sum(1 for r in w if int(r.prediction_id[1:]) % 10 == 0)
        assert row["delay_s"] == pytest.approx(30.0)
    wire = json.dumps(posts)
    for private in ("INSTR_SENTINEL", "secret_alpha_feature_zz", '"p0"', "prediction_id"):
        assert private not in wire
    agent.tick()
    assert len([1 for _, p, _, _ in sink.requests if p == PROXY_PATH]) == 1      # once per window
    assert agent.status()["proxy"]["rows_sent"] == 2
    agent.close()


@pytest.mark.slow
def test_proxy_config_is_validated_and_a_future_proxy_label_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="proxy_name"):
        split_proxy_specs({"type": "jsonl", "path": "/x", "delay_seconds": 5})
    with pytest.raises(ConfigError, match="delay_seconds"):
        split_proxy_specs({"type": "jsonl", "path": "/x", "proxy_name": "p"})
    with pytest.raises(ConfigError, match="used twice"):
        split_proxy_specs([{"proxy_name": "p", "delay_seconds": 1}, {"proxy_name": "p", "delay_seconds": 1}])
    recs, predictions, spec, cfg = _proxy_world(tmp_path)
    clock = [utc(T0 + 5)]                                    # labels are stamped after this clock
    agent = EdgeAgent(cfg, sources=[JsonlTailAdapter(predictions, tmp_path / "prediction.offset")],
                      proxy_sources=build_proxy_sources(spec, cfg.state_dir), post=Collector(), clock=lambda: clock[0])
    agent.poll()
    assert agent.proxy_rows_rejected > 0
    agent.close()
