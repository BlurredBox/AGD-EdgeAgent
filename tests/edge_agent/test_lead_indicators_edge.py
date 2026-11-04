"""G53 leading indicators on the edge: per-feature null summary, arrival lag with a per-record arrival
stamp, optional slice shift. Counts and lags only: no feature name, slice value or raw value on the wire,
and the real backend validators accept every payload."""
from __future__ import annotations

import dataclasses
import json
import math

import pytest

from corpus import lead_signals as lead
from corpus.serving import wire_fields
from edge_agent import EdgeAgent
from edge_agent.mapping import ColumnMapping
from edge_agent.records import Record
from tests.edge_agent.synth import (CADENCE, CATEGORY, CATS, FEATURES, INSTRUMENTS, LOCAL_MODEL, T0,
                                    Collector, make_cfg, make_prices, make_stream, utc)

pytestmark = pytest.mark.timeout(300)
N_REF, N_LIVE = 6, 4


# -- pure formulas --------------------------------------------------------------------------
def test_null_summary_counts_absent_none_nan_and_blank():
    rows = [{"a": 1.0, "b": None}, {"a": float("nan"), "b": 2.0}, {"b": ""}, {"a": 3.0, "b": 4.0}]
    out = lead.null_summary(rows, ["a", "b"], ceiling=0.3)
    assert out == {"feature_null_rate_max": 0.5, "feature_null_over_ceiling_count": 2}
    assert lead.null_summary([], ["a"], 0.2) == {} and lead.null_summary(rows, [], 0.2) == {}


def test_arrival_lags_quantiles_clip_negative_and_skip_unstamped():
    out = lead.arrival_lags([(0.0, 10.0), (0.0, 20.0), (0.0, None), (5.0, 0.0), (0.0, 30.0)])
    assert out["arrival_lag_max_s"] == 30.0 and out["arrival_lag_p50_s"] == 15.0
    assert out["arrival_lag_p95_s"] == pytest.approx(28.5)
    assert lead.arrival_lags([(0.0, None)]) == {}


def test_slice_shift_suppresses_small_slices_and_returns_no_keys():
    ref = {"S1": (100, 0.0), "S2": (100, 1.0), "S3": (5, 0.0)}
    win = {"S1": (40, 0.5), "S2": (10, 9.0), "S3": (40, 9.0)}
    out = lead.slice_shift(win, ref, ref_sd=0.25, min_rows=30, limit=1.0)
    assert out == {"slice_count_scored": 1, "slice_count_over_limit": 1, "slice_shift_max": 2.0}
    assert not any(isinstance(k, str) and k.startswith("S") for k in json.dumps(out))
    assert lead.slice_shift(win, ref, ref_sd=0.0, min_rows=30, limit=1.0) == {}


def test_fields_are_in_the_wire_contract_with_bounds():
    assert set(lead.LEAD_SIGNAL_FIELDS) <= wire_fields()
    assert set(lead.LEAD_BOUNDS) == set(lead.LEAD_SIGNAL_FIELDS)


def test_backend_validator_types_and_bounds():
    from src.api.signal_ingest import SignalRejected, _validate_z_t
    ok = {"feature_null_rate_max": 0.1, "feature_null_over_ceiling_count": 2, "arrival_lag_p95_s": 12.5,
          "slice_shift_max": 0.3, "slice_count_scored": 3, "slice_count_over_limit": 0}
    assert _validate_z_t(ok, edge_ahead_minor=False)[0] == ok
    for bad in ({"feature_null_rate_max": 1.5}, {"feature_null_over_ceiling_count": 1.5},
                {"arrival_lag_p50_s": -1.0}, {"slice_count_scored": True}):
        with pytest.raises(SignalRejected):
            _validate_z_t(bad, edge_ahead_minor=False)


# -- arrival stamp ----------------------------------------------------------------------------
def test_record_arrival_stamp_is_optional_aware_and_mapped():
    r = Record.make(T0, "m", "i", {"x": 1.0}, 0.1, arrived_at=T0 + 30)
    assert r.arrived_at == utc(T0 + 30)
    assert Record.make(T0, "m", "i", {"x": 1.0}, 0.1).arrived_at is None
    m = ColumnMapping({"ts": "t", "output": "y", "features": ["x"], "arrived_at": "arr"},
                      {"model_id": "m", "instrument": "all"})
    rec = m.to_record({"t": str(T0), "y": "0.5", "x": "1", "arr": str(T0 + 90)})
    assert (rec.arrived_at - rec.ts).total_seconds() == 90
    assert m.to_record({"t": str(T0), "y": "0.5", "x": "1", "arr": None}).arrived_at is None


def _run(tmp_path, *, lag_s: float = 0.0, null_from: int | None = None, **cfg_kw):
    cfg = make_cfg(tmp_path, **cfg_kw)
    sink = Collector()
    clock = [utc(T0)]
    agent = EdgeAgent(cfg, prices=make_prices(N_REF + N_LIVE), post=sink, clock=lambda: clock[0])
    recs = make_stream(N_REF + N_LIVE, drift_from=N_REF + 2)
    for k, r in enumerate(recs):
        if null_from is not None and r.ts >= utc(T0 + null_from * CADENCE) and k % 2:
            r = dataclasses.replace(r, features={**r.features, FEATURES[1]: None})
        clock[0] = utc(r.ts.timestamp() + lag_s)
        agent.ingest(r)
    clock[0] = utc(T0 + (N_REF + N_LIVE) * CADENCE + lag_s + 1)
    agent.flush()
    return agent, sink, recs, cfg


def test_agent_stamps_arrival_at_ingest_and_emits_lag(tmp_path):
    agent, sink, recs, cfg = _run(tmp_path, lag_s=42.0)
    zs = [b["z_t"] for b in sink.bodies()]
    assert zs and all(z["arrival_lag_p50_s"] == pytest.approx(42.0) for z in zs)
    assert all(z["arrival_lag_max_s"] == pytest.approx(42.0) for z in zs)


def test_source_only_mode_emits_no_lag_without_an_arrival_column(tmp_path):
    agent, sink, recs, cfg = _run(tmp_path, lag_s=42.0, arrival_stamp="source_only")
    assert sink.bodies() and all("arrival_lag_p50_s" not in b["z_t"] for b in sink.bodies())


def test_null_summary_rises_with_a_null_creep(tmp_path):
    agent, sink, recs, cfg = _run(tmp_path, null_from=N_REF + 1)
    zs = [b["z_t"] for b in sink.bodies()]
    before = [z["feature_null_rate_max"] for z in zs[:N_REF]]
    after = [z["feature_null_rate_max"] for z in zs[N_REF + 1:]]
    assert max(before) == 0.0 and min(after) == pytest.approx(0.5, abs=0.01)
    assert all(z["feature_null_over_ceiling_count"] == 1 for z in zs[N_REF + 1:])


def test_slice_shift_counts_only_on_the_wire_and_backend_accepts(tmp_path):
    agent, sink, recs, cfg = _run(tmp_path, slice_feature=CATEGORY, slice_min_rows=30)
    bodies = sink.bodies()
    live = [b["z_t"] for b in bodies if b["z_t"].get("warmup_state") == "live"]
    assert live and all(z["slice_count_scored"] == len(CATS) for z in live)
    assert all(math.isfinite(z["slice_shift_max"]) and z["slice_shift_max"] >= 0 for z in live)
    assert max(z["slice_count_over_limit"] for z in live) >= 1            # the drift from window 8
    blob = "\n".join(body.decode() + json.dumps(h) for _, _, body, h in sink.requests)
    for s in (*CATS, CATEGORY, *FEATURES, LOCAL_MODEL, *INSTRUMENTS):
        assert s not in blob, f"{s!r} leaked onto the wire"
    from src.api.signal_ingest import SignalEnvelope, _validate_z_t
    for _, _, body, _ in sink.requests:
        env = SignalEnvelope(**json.loads(body))
        accepted, unknown, refused = _validate_z_t(env.z_t, edge_ahead_minor=False)
        assert unknown == [] and refused == [] and accepted == env.z_t


def test_slice_state_survives_a_restart(tmp_path):
    agent, sink, recs, cfg = _run(tmp_path, slice_feature=CATEGORY)
    m = next(iter(agent.models.values())).signals
    assert set(m.ref_slice_acc) == set(CATS) and m.ref_output_sd and m.ref_output_sd > 0
    from edge_agent.signals import ModelSignals
    again = ModelSignals(cfg, m.model_id, m.wire_model_id, m.prices, m.alerter, m.state_path)
    assert again.ref_slice_acc == m.ref_slice_acc and again.ref_output_sd == m.ref_output_sd


def test_config_refuses_bad_lead_settings(tmp_path):
    for kw in ({"arrival_stamp": "sometimes"}, {"null_rate_ceiling": 1.0}, {"slice_min_rows": 0},
               {"slice_feature": " "}, {"slice_shift_limit": 0.0}):
        with pytest.raises(ValueError):
            make_cfg(tmp_path, **kw)
