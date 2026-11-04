"""
corpus/lead_signals.py

Leading-indicator scalars of the closed z_t schema (G53, `leadsignals`), and their pure formulas.

The edge agent computes them per window on the partner's host; they are COUNTS, RATES and LAGS
only. No feature name, feature value, slice value or identifier is read back out of these
functions: feature and slice keys are used as dict keys and never returned.

EXACT FIELD LIST (name, unit/domain). Absent (not sent) when the source is not available, never 0.

    feature_null_rate_max            fraction [0, 1]  max over the window's feature columns of the share of
                                                      received rows whose value is missing (None, NaN, "")
    feature_null_over_ceiling_count  count    >= 0    number of feature columns whose null rate is above
                                                      the edge's configured ceiling (`null_rate_ceiling`)
    arrival_lag_p50_s                seconds  >= 0    window quantiles of (arrival time - event time) over
    arrival_lag_p95_s                seconds  >= 0    the rows that carry an arrival stamp; negative lags
    arrival_lag_max_s                seconds  >= 0    (clock skew) count as 0
    slice_count_scored               count    >= 0    slices with >= `slice_min_rows` rows in this window
                                                      and in the reference
    slice_shift_max                  sd units >= 0    max over scored slices of |window mean output - reference
                                                      mean output of the slice| / reference output sd
    slice_count_over_limit           count    >= 0    scored slices whose shift is above `slice_shift_limit`

Only aggregates; a slice with fewer than `slice_min_rows` rows is not scored (k-suppression, the same idea as
`MIN_CATEGORY_COUNT`), so a small slice never contributes a per-slice statistic.

WHAT THESE ARE FOR. Shadow-only forecasting of when an indicator will cross a declared limit
(`src/api/leadsignals_shadow.py`), and the partner history replay of
`docs/measurements/leadsignals-preregistration-2026-10-29.md` section 9. No forecaster was fitted on them;
`ForecasterBundle.predict` reports them in `fields_ignored`. They page nothing.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = ["LEAD_SIGNAL_FIELDS", "LEAD_BOUNDS", "LEAD_COUNT_FIELDS", "is_null", "null_summary",
           "arrival_lags", "slice_shift"]

LEAD_SIGNAL_FIELDS: tuple[str, ...] = (
    "feature_null_rate_max", "feature_null_over_ceiling_count",
    "arrival_lag_p50_s", "arrival_lag_p95_s", "arrival_lag_max_s",
    "slice_count_scored", "slice_shift_max", "slice_count_over_limit",
)
LEAD_COUNT_FIELDS = frozenset({"feature_null_over_ceiling_count", "slice_count_scored",
                               "slice_count_over_limit"})

# Inclusive (low, high) wire bounds; None = unbounded. Enforced at ingest.
LEAD_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "feature_null_rate_max": (0.0, 1.0), "feature_null_over_ceiling_count": (0.0, 100_000.0),
    "arrival_lag_p50_s": (0.0, 1e8), "arrival_lag_p95_s": (0.0, 1e8), "arrival_lag_max_s": (0.0, 1e8),
    "slice_count_scored": (0.0, 100_000.0), "slice_shift_max": (0.0, 1e6),
    "slice_count_over_limit": (0.0, 100_000.0),
}


def is_null(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return isinstance(v, str) and not v.strip()


def null_summary(rows: Sequence[Mapping[str, Any]], columns: Iterable[str],
                 ceiling: float) -> dict[str, float | int]:
    """Max per-column null rate and the count of columns above `ceiling`. A column absent from a
    row counts as null in that row. Empty window or no column: {} (not computable, not 0)."""
    cols = list(dict.fromkeys(columns))
    n = len(rows)
    if n == 0 or not cols:
        return {}
    rates = [sum(1 for r in rows if is_null(r.get(c))) / n for c in cols]
    return {"feature_null_rate_max": float(max(rates)),
            "feature_null_over_ceiling_count": int(sum(1 for x in rates if x > ceiling))}


def _pct(s: list[float], p: float) -> float:
    rank = p * (len(s) - 1)
    lo, hi = int(math.floor(rank)), int(math.ceil(rank))
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (rank - lo)


def arrival_lags(pairs: Iterable[tuple[float, float | None]]) -> dict[str, float]:
    """pairs: (event epoch seconds, arrival epoch seconds or None). Quantiles over stamped rows only."""
    lags = sorted(max(0.0, a - e) for e, a in pairs
                  if a is not None and math.isfinite(a) and math.isfinite(e))
    if not lags:
        return {}
    return {"arrival_lag_p50_s": float(_pct(lags, 0.50)), "arrival_lag_p95_s": float(_pct(lags, 0.95)),
            "arrival_lag_max_s": float(lags[-1])}


def slice_shift(window: Mapping[str, tuple[int, float]], reference: Mapping[str, tuple[int, float]],
                ref_sd: float | None, min_rows: int, limit: float) -> dict[str, float | int]:
    """window / reference: slice key -> (n, mean output). Keys never leave this function."""
    if not ref_sd or not math.isfinite(ref_sd) or ref_sd <= 0:
        return {}
    shifts = []
    for k, (n, m) in window.items():
        ref = reference.get(k)
        if n < min_rows or ref is None or ref[0] < min_rows or not math.isfinite(m):
            continue
        shifts.append(abs(m - ref[1]) / ref_sd)
    out: dict[str, float | int] = {"slice_count_scored": len(shifts),
                                   "slice_count_over_limit": int(sum(1 for s in shifts if s > limit))}
    if shifts:
        out["slice_shift_max"] = float(max(shifts))
    return out
