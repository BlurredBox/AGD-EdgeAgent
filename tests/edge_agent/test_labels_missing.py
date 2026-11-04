"""Decision 2 (2026-10-29): the edge tripwire `labels_missing`, local like `feed_down`.

A model whose windows have carried labels raises `labels_missing` once `labels_missing_after_windows` consecutive
closed windows have label coverage 0, and `labels_recovered` when a labelled window returns. Nothing is sent to the
backend. The first test drives the real EdgeAgent through the demo seed's own label-lag episode (`retail-credit-pd-v4`).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from demo_seed_lib import catalog  # noqa: E402
from demo_seed_lib.generator import state_at, window_rows  # noqa: E402
from edge_agent.agent import EdgeAgent  # noqa: E402
from edge_agent.config import EdgeConfig  # noqa: E402
from edge_agent.records import Record  # noqa: E402
from edge_agent.tripwires import LABELS_MISSING, LABELS_RECOVERED  # noqa: E402
from tests.edge_agent.synth import Collector  # noqa: E402

SPEC = catalog.BY_NAME["retail-credit-pd-v4"]
SEED = 11
TENANT = "00000000-0000-4000-8000-0000000e9152"


def _utc(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, timezone.utc)


def _agent(tmp_path, sim, **over) -> tuple[EdgeAgent, Collector]:
    sink = Collector()
    cfg = EdgeConfig(tenant_id=TENANT, install_id="demoseed-retail", models={SPEC.name: "mid-retail"},
                     secret="labels-test-not-a-secret", endpoints=["http://stub.invalid"], state_dir=tmp_path / "edge",
                     cadence_seconds=SPEC.demo_cadence_seconds, output_kind="probability", timeout_seconds=5.0,
                     **over)
    return EdgeAgent(cfg, [], clock=lambda: _utc(sim[0]), post=sink), sink


def _drive_seed(tmp_path, *, warmup: int = 20, **over):
    """The seed's own windows for retail-credit-pd-v4 (labels stop at T0+2:00, return at +14:00)."""
    c = SPEC.demo_cadence_seconds
    sim = [0.0]
    agent, sink = _agent(tmp_path, sim, **over)
    b_end = int(1_800_000_000 // c)
    t0 = b_end * c
    seen: list[tuple[str, float]] = []
    for bucket in range(b_end - warmup, b_end + int(catalog.SCRIPT_SECONDS // c) + 2):
        sim[0] = (bucket + 1) * c + 0.25
        st = state_at(SPEC.name, None if bucket < b_end else bucket * c - t0)
        for r in window_rows(SPEC, bucket, c, SEED, st):
            agent.ingest(Record.make(r["ts"], r["model_id"], r["instrument"], r["features"], r["output"],
                                     outcome=r.get("outcome"), outcome_observed_at=r.get("outcome_observed_at")))
        agent.tick(_utc(sim[0]))
        for a in agent.alerter.history[len(seen):]:
            seen.append((a.kind, sim[0] - t0))
    status = agent.status()
    agent.close()
    return seen, sink, c, status


def test_the_seed_label_lag_raises_labels_missing_once_after_k_windows_and_recovers(tmp_path):
    ep = next(e for e in catalog.EPISODES if e.key == "label-lag")
    alerts, sink, c, _ = _drive_seed(tmp_path)
    kinds = [k for k, _ in alerts]
    assert kinds.count(LABELS_MISSING) == 1 and kinds.count(LABELS_RECOVERED) == 1, alerts
    down = next(t for k, t in alerts if k == LABELS_MISSING)
    up = next(t for k, t in alerts if k == LABELS_RECOVERED)
    k = EdgeConfig.__dataclass_fields__["labels_missing_after_windows"].default
    # the first unlabelled window closes at start_s + c; the K-th one K - 1 windows later (+ a tick)
    assert ep.start_s + k * c <= down <= ep.start_s + (k + 1) * c + 1, (down, k, c)
    assert down < ep.end_s and ep.end_s <= up <= ep.end_s + 2 * c + 1, (down, up)
    # local only: no window body names it, and the wire envelope is unchanged
    assert sink.bodies() and not any("labels_missing" in str(b) for b in sink.bodies())


def test_a_model_that_never_had_labels_never_raises_it(tmp_path):
    sim = [1_800_000_000.0]
    agent, _ = _agent(tmp_path, sim, labels_missing_after_windows=2)
    c = SPEC.demo_cadence_seconds
    b0 = int(sim[0] // c)
    for bucket in range(b0, b0 + 12):
        sim[0] = (bucket + 1) * c + 0.25
        for r in window_rows(SPEC, bucket, c, SEED, state_at(SPEC.name, None)):
            agent.ingest(Record.make(r["ts"], r["model_id"], r["instrument"], r["features"], r["output"]))
        agent.tick(_utc(sim[0]))
    assert not [a for a in agent.alerter.history if a.kind in (LABELS_MISSING, LABELS_RECOVERED)]
    assert agent.status()["models"][SPEC.name]["labels_missing"] is False
    agent.close()


def test_k_is_configurable_and_zero_turns_it_off(tmp_path):
    for k, expect in ((2, 1), (0, 0)):
        alerts, *_ = _drive_seed(tmp_path / f"k{k}", labels_missing_after_windows=k)
        assert [a for a, _ in alerts].count(LABELS_MISSING) == expect, (k, alerts)


def test_status_reports_the_open_condition(tmp_path):
    sim = [1_800_000_000.0]
    agent, _ = _agent(tmp_path, sim, labels_missing_after_windows=2)
    c = SPEC.demo_cadence_seconds
    b0 = int(sim[0] // c)
    for i, bucket in enumerate(range(b0, b0 + 5)):
        sim[0] = (bucket + 1) * c + 0.25
        for r in window_rows(SPEC, bucket, c, SEED, state_at(SPEC.name, None)):
            labelled = i == 0
            agent.ingest(Record.make(r["ts"], r["model_id"], r["instrument"], r["features"], r["output"],
                                     outcome=(r.get("outcome", 0) if labelled else None),
                                     outcome_observed_at=(r["ts"] if labelled else None)))
        agent.tick(_utc(sim[0]))
    assert agent.status()["models"][SPEC.name]["labels_missing"] is True
    assert [a.kind for a in agent.alerter.history].count(LABELS_MISSING) == 1
    agent.close()


@pytest.mark.parametrize("bad", [-1, 1.5, "6", True])
def test_bad_k_is_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="labels_missing_after_windows"):
        EdgeConfig(tenant_id=TENANT, install_id="x", models={"m": "mid"}, secret="s", endpoints=["http://e.invalid"],
                   state_dir=tmp_path, labels_missing_after_windows=bad)

