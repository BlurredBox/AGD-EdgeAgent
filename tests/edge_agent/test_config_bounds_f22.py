"""Audit F22: EdgeConfig refuses the remaining unbounded and malformed values."""
from __future__ import annotations

import pytest

from edge_agent.config import EdgeConfig

NAN, INF = float("nan"), float("inf")
BASE = dict(tenant_id="t", install_id="i", models={"m": "w"}, secret="s" * 16,
            endpoints=["https://e.invalid"], state_dir="/tmp/x")

BAD = [
    *[(f, v) for f in ("cadence_seconds", "reference_windows", "slice_min_rows", "slice_max")
      for v in (NAN, INF, 1.5, True)],
    ("lateness_seconds", INF), ("alert_cooldown_seconds", INF),
    *[(f, v) for f in ("feed_down_after_seconds", "backlog_after_seconds")
      for v in (0, -1, NAN, INF, "5")],
    *[("max_invalid_row_fraction", v) for v in (-1, 1.5, NAN, "0.2")],
    ("price_max_staleness_days", -1), ("price_max_staleness_days", 1.5), ("price_max_staleness_days", True),
    ("feature_bounds", {"f": (2.0, 1.0)}), ("feature_bounds", {"f": (NAN, 1.0)}),
    ("output_bounds", (1.0, 0.0)), ("output_bounds", (0.0, NAN)),
    ("tenant_id", ""), ("tenant_id", 5), ("install_id", ""), ("install_id", None),
    ("endpoints", []), ("state_dir", ""), ("schema_version", ""),
    ("instrument_series", {"AAPL": 5}), ("instrument_series", {"AAPL": ""}),
]


@pytest.mark.parametrize("field,value", BAD, ids=[f"{f}={v!r}" for f, v in BAD])
def test_refused(field, value):
    with pytest.raises(ValueError):
        EdgeConfig(**{**BASE, field: value})


def test_sane_values_still_accepted():
    EdgeConfig(**BASE, max_invalid_row_fraction=1, feed_down_after_seconds=10, backlog_after_seconds=None,
               feature_bounds={"f": (0, 1)}, output_bounds=(-INF, INF), price_max_staleness_days=0)
