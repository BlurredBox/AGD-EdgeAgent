"""Audit F19: paths that lose data and were not driven: corrupt state files, source rotation, status, pinned bundle."""
from __future__ import annotations

import hashlib
import json
import pickle

import pytest

from edge_agent import EdgeAgent
from edge_agent.__main__ import _trusted_bundle
from edge_agent.adapters import JsonlTailAdapter
from edge_agent.status import read_status
from tests.edge_agent.synth import LOCAL_MODEL, T0, Collector, make_cfg, make_stream, utc


def _line(i: int) -> str:
    return json.dumps({"ts": utc(T0 + i).isoformat(), "model_id": LOCAL_MODEL, "instrument": "i",
                       "features": {"x": float(i)}, "output": 0.1}) + "\n"


def test_jsonl_rotation_resumes_at_the_start_of_the_new_file(tmp_path):
    src = tmp_path / "pred.jsonl"
    src.write_text(_line(1) + _line(2) + _line(3))
    ad = JsonlTailAdapter(src, tmp_path / "source.offset")
    assert len(ad.poll()) == 3
    ad.commit(None)
    src.write_text(_line(4))                       # rotated: new, shorter file
    got = ad.poll()
    assert [r.features["x"] for r in got] == [4.0]


@pytest.mark.parametrize("name", ["open_windows.json", "emitted.json", "feature_ids.json", "status.json"])
def test_agent_starts_over_a_corrupt_state_file(tmp_path, name):
    cfg = make_cfg(tmp_path)
    cfg.state_dir.mkdir(parents=True)
    (cfg.state_dir / name).write_text("{not json")
    sink, clock = Collector(), [utc(T0)]
    agent = EdgeAgent(cfg, post=sink, clock=lambda: clock[0])
    for r in make_stream(2, rows=40):
        clock[0] = r.ts
        agent.ingest(r)
    clock[0] = utc(T0 + 200)
    agent.tick(clock[0])
    agent.close()
    assert agent.status()["models"][LOCAL_MODEL]["emitted"] >= 1


def test_corrupt_spool_file_is_moved_aside_and_the_rest_is_sent(tmp_path):
    cfg = make_cfg(tmp_path)
    sink, clock = Collector(), [utc(T0)]
    agent = EdgeAgent(cfg, post=sink, clock=lambda: clock[0])
    for r in make_stream(2, rows=40):
        clock[0] = r.ts
        agent.ingest(r)
    clock[0] = utc(T0 + 200)
    agent.flush()
    spool = cfg.state_dir / "spool"
    queued = [p for p in agent.sender.spool.pending()]
    assert queued or sink.bodies()
    d = spool / "mm-wire-0001"
    d.mkdir(exist_ok=True)
    (d / "000000000000001.json").write_text("{broken")
    agent.sender.drain(force=True)
    # edge F9 (#589): a corrupt spool file is quarantined, not treated as a backend rejection
    assert (spool / "quarantine" / "mm-wire-0001-000000000000001.json").exists()
    assert not (d / "000000000000001.json").exists()
    agent.close()


def test_status_report_after_a_run_has_counts_and_no_row_values(tmp_path):
    cfg = make_cfg(tmp_path)
    clock = [utc(T0)]
    agent = EdgeAgent(cfg, post=Collector(), clock=lambda: clock[0])
    for r in make_stream(2, rows=40):
        clock[0] = r.ts
        agent.ingest(r)
    clock[0] = utc(T0 + 200)
    agent.tick(clock[0])
    agent.close()
    report = read_status(cfg.state_dir)
    text = json.dumps(report, default=str)
    assert report["state_dir_exists"] and report["agent_snapshot"] is not None
    for secret in ("secret_alpha_feature_zz", "CATEGORY_SENTINEL_A"):
        assert secret not in text
    assert read_status(tmp_path / "missing") == {"state_dir_exists": False}


def test_trusted_bundle_checks_the_digest_of_the_bytes_it_loads(tmp_path):
    (tmp_path / "forecaster.pkl").write_bytes(pickle.dumps({"not": "a bundle"}))
    good = hashlib.sha256((tmp_path / "forecaster.pkl").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="SHA-256"):
        _trusted_bundle(str(tmp_path), None)
    with pytest.raises(ValueError, match="mismatch"):
        _trusted_bundle(str(tmp_path), "0" * 64)
    with pytest.raises(ValueError, match="ForecasterBundle"):
        _trusted_bundle(str(tmp_path), good)
