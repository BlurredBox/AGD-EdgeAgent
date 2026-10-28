"""Freezing the reference window edge-side.

`corpus.harness.ReferenceWindow.build` is the reference implementation, but it
refuses a reference under 2000 rows or 24 h -- a replay-corpus bar. The edge
freezes after `reference_windows` (200) windows, which at a 5-minute cadence is
~16.7 h, and a daily-cadence model never reaches 2000 rows. So the edge cannot
call `build`. This module repeats `build`'s steps -- every statistic still
comes from the corpus primitives (`signal_vector`, `multivariate`, `anomaly`,
`sketches`); only the two size/span guards are dropped. `tests/edge_agent/
test_reference_conformance.py` asserts the result equals `ReferenceWindow.build`
on a reference that passes the guards, so a change to `build` fails a test
rather than silently forking the edge.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from corpus import anomaly as an
from corpus import multivariate as mv
from corpus import signal_vector as sv
from corpus.harness import SCHEMA_VERSION, ReferenceWindow
from corpus.sketches import HyperLogLog, TDigest

MIN_REFERENCE_ROWS = 20          # below this the multivariate fit is meaningless


def split_columns(rows: Sequence[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Numeric = every present value is a finite-or-missing int/float (bool excluded)."""
    cols = sorted({c for r in rows for c in r})
    numeric, categorical = [], []
    for c in cols:
        vals = [r.get(c) for r in rows if r.get(c) is not None]
        if vals and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical


def is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def build_reference(rows: Sequence[tuple[datetime, dict[str, Any]]], dataset_id: str,
                    group_id: str, numeric: Sequence[str], categorical: Sequence[str]
                    ) -> ReferenceWindow:
    rows = [(ts, f) for ts, f in rows
            if all(not is_missing(f.get(c)) and math.isfinite(float(f[c])) for c in numeric)
            and all(f.get(c) is not None for c in categorical)]
    if len(rows) < MIN_REFERENCE_ROWS:
        raise ValueError(f"only {len(rows)} complete reference rows")
    if not numeric:
        raise ValueError("no numeric feature in the reference")
    ts_all = [ts for ts, _ in rows]
    span_h = (max(ts_all) - min(ts_all)).total_seconds() / 3600

    samples = {f: [float(r[f]) for _, r in rows] for f in numeric}
    digests = {f: TDigest(compression=100).extend(samples[f]) for f in numeric}
    cats = {f: HyperLogLog(precision=14).extend(str(r[f]) for _, r in rows)
            for f in categorical}
    payload = json.dumps({
        "dataset": dataset_id, "group": group_id,
        "start": min(ts_all).isoformat(), "end": max(ts_all).isoformat(),
        "schema": SCHEMA_VERSION,
    }, sort_keys=True)
    rwid = hashlib.sha256(payload.encode()).hexdigest()[:16]
    sorted_samples = {f: sorted(v) for f, v in samples.items()}
    edges = {f: sv.bin_edges_from_reference(v, 10) for f, v in sorted_samples.items()}
    ref_counts = {f: sv.bin_counts(sorted_samples[f], edges[f], 10) for f in sorted_samples}
    cat_counts = {f: h.count() for f, h in cats.items()}
    cat_values = {f: frozenset(str(r[f]) for _, r in rows) for f in categorical}

    X = np.array([[float(r[f]) for f in numeric] for _, r in rows])
    multivariate = mv.MultivariateReference.fit(X)
    iforest = (an.IsolationForestDetector(seed=0).fit(X)
               if len(X) >= an.IF_SUBSAMPLE else None)
    hst = an.HalfSpaceTrees(n_features=len(numeric), seed=0)
    hst.set_range(X.min(axis=0), X.max(axis=0))
    for row in X:
        hst.update(row)
    hst.roll_window()

    return ReferenceWindow(dataset_id, group_id, samples, cats, digests,
                           len(rows), span_h, rwid,
                           sorted_samples, edges, ref_counts, cat_counts, cat_values,
                           multivariate, iforest, hst)


@dataclass
class ReferenceStats:
    """Frozen with the reference; the quality gate and trading fields need them."""
    expected_schema: set[str]
    expected_dtypes: dict[str, Any]
    reference_categories: dict[str, frozenset]
    reference_missingness: dict[str, float]
    expected_rows: int | None          # median rows per reference window
    ref_volume: float | None           # mean public volume over the reference
    ref_pred_rms: float | None         # rms of the model's output over the reference
