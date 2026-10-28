from __future__ import annotations

import json
from datetime import datetime, timezone

from edge_agent.adapters import InProcessAdapter, JsonlTailAdapter
from edge_agent.featureids import FeatureIdMap
from edge_agent.prices import FredDailyAdapter, IntradayStubAdapter

CSV = "observation_date,SP500\n2024-03-01,5000.5\n2024-03-04,5010.0\n2024-03-05,.\n2024-03-06,5020.25\n"


def at(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_fred_uses_only_closes_known_before_the_day_and_skips_missing(tmp_path):
    a = FredDailyAdapter({"X": "SP500"}, tmp_path, fetch=lambda url: CSV)
    a.refresh(["X"])
    assert a.price_at("X", at("2024-03-04T10:00:00")).price == 5000.5   # 03-04's close not known yet
    assert a.price_at("X", at("2024-03-06T10:00:00")).price == 5010.0   # 03-05 is "." -> skipped
    assert a.price_at("X", at("2024-03-07T00:00:01")).price == 5020.25
    assert a.price_at("X", at("2024-02-01T00:00:00")) is None            # before any data
    assert a.price_at("X", at("2024-04-30T00:00:00")) is None            # stale: not carried forward
    assert a.price_at("UNKNOWN", at("2024-03-07T00:00:00")) is None


def test_fred_failure_keeps_serving_the_disk_cache(tmp_path):
    a = FredDailyAdapter({"X": "SP500"}, tmp_path, fetch=lambda url: CSV)
    a.refresh(["X"])

    def boom(url):
        raise OSError("offline")
    b = FredDailyAdapter({"X": "SP500"}, tmp_path, fetch=boom, refresh_every_seconds=0)
    b.refresh(["X"])                                                    # must not raise
    assert b.last_error and "offline" in b.last_error
    assert b.price_at("X", at("2024-03-07T00:00:01")).price == 5020.25


def test_fred_url_is_the_public_keyless_csv_endpoint(tmp_path):
    seen = []
    FredDailyAdapter({"X": "DEXUSEU"}, tmp_path, fetch=lambda u: seen.append(u) or CSV).refresh(["X"])
    assert seen == ["https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU"]


def test_intraday_stub_gives_no_price():
    assert IntradayStubAdapter().price_at("X", at("2024-03-07T00:00:00")) is None


def test_feature_ids_stable_across_restart_salted_and_not_name_derived(tmp_path):
    p = tmp_path / "ids.json"
    a = FeatureIdMap(p)
    i1 = a.id_for("my secret column")
    assert i1.startswith("f_") and "secret" not in i1 and i1 == a.id_for("my secret column")
    b = FeatureIdMap(p)
    assert b.id_for("my secret column") == i1 and b.name_for(i1) == "my secret column"
    other = FeatureIdMap(tmp_path / "other.json")
    assert other.id_for("my secret column") != i1                       # per-install salt


def test_jsonl_tail_resumes_from_offset_and_waits_for_partial_lines(tmp_path):
    log = tmp_path / "m.jsonl"
    def line(n):
        return json.dumps({"ts": 1700000000 + n, "model_id": "m", "instrument": "i",
                           "features": {"a": n}, "output": 0.1}) + "\n"
    log.write_text(line(1) + line(2) + line(3)[:20])                    # last line half-written
    t = JsonlTailAdapter(log, tmp_path / "off")
    assert [r.features["a"] for r in t.poll()] == [1, 2]
    with open(log, "a") as fh:
        fh.write(line(3)[20:] + "not json\n" + line(4))
    t2 = JsonlTailAdapter(log, tmp_path / "off")                        # restart
    assert [r.features["a"] for r in t2.poll()] == [3, 4] and t2.bad_lines == 1


def test_in_process_adapter_never_blocks_and_counts_drops():
    a = InProcessAdapter(maxsize=2)
    assert a.observe(1700000000, "m", "i", {"a": 1}, 0.1)
    assert a.observe(1700000001, "m", "i", {"a": 2}, 0.1)
    assert not a.observe(1700000002, "m", "i", {"a": 3}, 0.1) and a.dropped == 1
    assert len(a.poll()) == 2
