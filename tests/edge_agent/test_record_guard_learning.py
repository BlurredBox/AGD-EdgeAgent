"""The ingest guard learns the feature schema from more than the first row: a null or an optional key in
row one must not make every later row unusable, while a real type flip or a column that disappears is
still caught."""
from __future__ import annotations

from edge_agent import EdgeAgent, Record
from edge_agent.signals import ModelSignals
from edge_agent.tripwires import LEARN_ROWS, LocalAlerter, RecordGuard
from tests.edge_agent.synth import (CADENCE, FEATURES, LOCAL_MODEL, T0, Collector, make_cfg,
                                    make_prices, make_stream, utc)


def _run_with_first_row(tmp_path, monkeypatch, first_features):
    """Feed three windows whose first record is replaced; return (received, usable, reasons) per window."""
    seen = []
    orig = ModelSignals.process

    def spy(self, wi, ops):
        seen.append((wi.received, len(wi.window.records), dict(wi.dropped_reasons)))
        return orig(self, wi, ops)

    monkeypatch.setattr(ModelSignals, "process", spy)
    clock = [utc(T0)]
    agent = EdgeAgent(make_cfg(tmp_path), prices=make_prices(3), post=Collector(), clock=lambda: clock[0])
    recs = make_stream(3)
    r0 = recs[0]
    recs[0] = Record.make(r0.ts, LOCAL_MODEL, r0.instrument, first_features(dict(r0.features)), r0.output)
    for r in recs:
        clock[0] = r.ts
        agent.ingest(r)
    clock[0] = utc(T0 + 3 * CADENCE)
    agent.flush()
    return seen


def test_a_null_numeric_feature_in_the_first_row_does_not_drop_later_rows(tmp_path, monkeypatch):
    seen = _run_with_first_row(tmp_path, monkeypatch, lambda f: {**f, FEATURES[0]: None})
    assert len(seen) == 3
    assert all(usable == received for received, usable, _ in seen), seen


def test_an_optional_key_only_in_the_first_row_does_not_drop_later_rows(tmp_path, monkeypatch):
    seen = _run_with_first_row(tmp_path, monkeypatch, lambda f: {**f, "promo_code": "X1"})
    assert len(seen) == 3
    assert all(usable == received for received, usable, _ in seen), seen


def _guard():
    return RecordGuard(LOCAL_MODEL, LocalAlerter("/dev/null"), feature_bounds={}, output_bounds=None)


def test_a_number_to_string_flip_after_numeric_rows_is_still_a_dtype_change():
    g = _guard()
    assert g.check({"x": None, "y": "a"}, 0.0) == []         # type of x learned from its first value
    for i in range(50):
        assert g.check({"x": float(i), "y": "a"}, 0.0) == []
    assert "dtype_change" in g.check({"x": "oops", "y": "a"}, 0.0)


def test_a_column_present_in_every_learning_row_is_required_afterwards():
    g = _guard()
    for i in range(LEARN_ROWS):
        g.check({"x": float(i), "y": "a", **({"opt": 1} if i == 0 else {})}, 0.0)
    assert g.check({"x": 1.0}, 0.0) == ["schema_missing_columns"]      # y: seen in every row
    assert g.check({"x": 1.0, "y": "a"}, 0.0) == []                     # opt: seen once, optional
