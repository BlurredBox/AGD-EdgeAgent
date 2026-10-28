"""The edge's relaxed reference build must equal the corpus's `ReferenceWindow.build`
wherever `build` accepts the input, and the harness must produce the same vector
from either; plus the incremental trajectory pass must equal `trajectory.expand`."""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from corpus.adapters import Row
from corpus.harness import ReferenceWindow
from corpus import trajectory as traj
from edge_agent.config import EdgeConfig
from edge_agent.reference import build_reference
from edge_agent.signals import ModelSignals, _LIVE_SLOTS, _harness_shim


def _rows(n=2100, hours=30):
    rng = np.random.default_rng(3)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    out = []
    for i in range(n):
        out.append(Row(t0 + timedelta(hours=hours * i / n),
                       {"a": float(rng.normal()), "b": float(rng.normal(2, 3)),
                        "c": float(rng.uniform()), "k": str(rng.choice(["x", "y", "z"]))}, 0))
    return out


@pytest.mark.timeout(300)
def test_edge_reference_equals_corpus_build():
    rows = _rows()
    ref_c = ReferenceWindow.build(rows, "ds", "g", ["a", "b", "c"], ["k"])
    ref_e = build_reference([(r.ts, r.features) for r in rows], "ds", "g", ["a", "b", "c"], ["k"])
    assert ref_e.reference_window_id == ref_c.reference_window_id
    assert ref_e.n_rows == ref_c.n_rows and ref_e.bin_edges == ref_c.bin_edges
    assert ref_e.ref_bin_counts == ref_c.ref_bin_counts and ref_e.cat_counts == ref_c.cat_counts
    assert ref_e.cat_values == ref_c.cat_values
    assert np.allclose(ref_e.multivariate.precision, ref_c.multivariate.precision)
    assert ref_e.multivariate.mahalanobis_median == ref_c.multivariate.mahalanobis_median

    # And the harness yields the same vector from either reference.
    cfg = EdgeConfig(tenant_id="t", install_id="i", models={"m": "w"}, secret="x" * 16,
                     endpoints=["http://x.invalid"], state_dir="/nonexistent", seed=0)
    cur = _rows(400, 1)
    for r in cur:
        object.__setattr__(r, "features", {**r.features, "a": r.features["a"] + 0.5})
    scores = [0.01 * r.features["a"] for r in cur]
    zc = _harness_shim(cfg, ref_c, ["a", "b", "c"], ["k"])._compute_z_t(cur, scores, 400)
    ze = _harness_shim(cfg, ref_e, ["a", "b", "c"], ["k"])._compute_z_t(cur, scores, 400)
    assert zc.keys() == ze.keys()
    for k in zc:
        if k == "reference_window_id":
            continue
        assert zc[k] == ze[k], k


def test_edge_reference_builds_where_corpus_refuses():
    rows = _rows(300, 2)
    try:
        ReferenceWindow.build(rows, "ds", "g", ["a", "b", "c"], ["k"])
        raise AssertionError("corpus build should refuse a 2h / 300-row reference")
    except ValueError:
        pass
    ref = build_reference([(r.ts, r.features) for r in rows], "ds", "g", ["a", "b", "c"], ["k"])
    assert ref.n_rows == 300 and ref.multivariate is not None


def test_incremental_trajectory_equals_expand():
    rnd = random.Random(5)
    n = 60
    series = []
    for i in range(n):
        if i in (10, 11, 40):                         # skipped windows: gaps in the index
            series.append({})
            continue
        series.append({"score_mean": rnd.gauss(0, 1), "score_std": rnd.random(),
                       "psi_aggregate": rnd.random() if i % 7 else None,
                       "psi": {"x": rnd.random(), "y": rnd.random()},
                       "score_quantiles": {"p50": rnd.random(), "p95": rnd.random()},
                       "row_count_vs_expected": 1 + rnd.gauss(0, .1)})
    expected = traj.expand(series)

    ms = ModelSignals.__new__(ModelSignals)
    ms.slot_state = {s.name: traj._SlotState() for s in _LIVE_SLOTS}
    for idx, z in enumerate(series):                  # absolute index, as the agent uses
        got = ms._trajectory(idx, z)
        assert got.keys() == expected[idx].keys()
        for k, v in got.items():
            e = expected[idx][k]
            assert (v is None and e is None) or math.isclose(v, e, rel_tol=0, abs_tol=0), (idx, k)
