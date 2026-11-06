"""A history replay reads to the end of the staged history even across a long run of unparsable rows."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
import yaml

from edge_agent import history_replay as hr

pytestmark = pytest.mark.timeout(300)
T0 = datetime(2026, 3, 2, tzinfo=timezone.utc)
BAD = 25_000                                  # more than two full polls (10,000 rows each) of bad rows


def _row(t: datetime, fj: str) -> dict:
    iso = t.isoformat().replace("+00:00", "Z")
    return {"t": iso, "arr": iso, "y": "0.4", "fj": fj, "bf": "false"}


def test_replay_does_not_stop_at_a_run_of_unparsable_rows(tmp_path):
    day1 = [_row(T0 + timedelta(minutes=30 * i), json.dumps({"x": i})) for i in range(48)]
    bad = [_row(T0 + timedelta(hours=24), f"{{\"x\": {i}") for i in range(BAD)]
    day2 = [_row(T0 + timedelta(hours=25, minutes=30 * i), json.dumps({"x": i})) for i in range(48)]
    pd.DataFrame(day1 + bad + day2).to_csv(tmp_path / "p.csv", index=False)
    (tmp_path / "m.yaml").write_text(yaml.safe_dump({
        "cadence_seconds": 3600, "edge": {"min_rows_per_window": 2},
        "predictions": {"path": "p.csv", "columns": {"ts": "t", "arrived_at": "arr", "output": "y",
                                                     "features": {"json_column": "fj"}},
                        "constants": {"model_id": "m1", "instrument": "all"}, "backfill": "bf"}}))
    spec = hr.load_spec(tmp_path / "m.yaml")
    staged = hr.stage_predictions(spec, hr.load_frames(spec), tmp_path / "stage")
    sent = []

    def post(url, path, body, headers):
        sent.append(json.loads(body))
        return 202, {"accepted": True}

    st = hr.replay(spec, staged, tenant_id="00000000-0000-0000-0000-000000000001", install_id="i",
                   models={"m1": "wire-model"}, secret="s", endpoints=["https://x"],
                   work_dir=tmp_path / "w", post=post)
    last = max(datetime.fromisoformat(b["window_timestamp"]) for b in sent)
    assert last >= T0 + timedelta(hours=48)               # day 2 was replayed
    assert st["source_bad_rows"] == BAD                    # and the skipped rows are reported
