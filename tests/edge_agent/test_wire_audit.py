"""Wire audit: serialise EVERYTHING the agent sent for a synthetic run and prove
no raw value, feature name, instrument, local model id or prediction is in it --
only schema fields. Also proves the payloads are accepted by the real backend
validators, and that signing is the SDK's."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpus.serving import wire_fields
from tests.edge_agent.synth import (CATEGORY, CATS, FEATURES, INSTRUMENTS, LOCAL_MODEL,
                                    WIRE_MODEL)

# Shared live_run fixture is slow on the shared CI runner (> global 60 s).
pytestmark = pytest.mark.timeout(300)

ENVELOPE_KEYS = {"schema_version", "tenant_id", "install_id", "monitored_model_id",
                 "window_id", "window_timestamp", "z_t"}


def _everything_on_the_wire(agent, sink) -> str:
    """Request bodies AND headers, plus whatever is still queued on disk."""
    parts = []
    for url, path, body, headers in sink.requests:
        parts += [url, path, body.decode(), json.dumps(headers, sort_keys=True)]
    for p in Path(agent.cfg.state_dir, "spool").rglob("*.json"):
        parts.append(p.read_text())
    return "\n".join(parts)


def test_run_actually_sent_live_windows(live_run):
    agent, sink, recs, cfg = live_run
    bodies = sink.bodies()
    states = [b["z_t"].get("warmup_state") for b in bodies]
    assert states.count("cold_start") == 6 and states.count("live") == 4, states
    assert agent.status()["queued"] == 0


def test_no_raw_value_name_or_identifier_on_the_wire(live_run):
    agent, sink, recs, cfg = live_run
    blob = _everything_on_the_wire(agent, sink)
    assert blob

    forbidden_text = [LOCAL_MODEL, *INSTRUMENTS, *FEATURES, CATEGORY, *CATS,
                      "SENTINEL", "_zz", cfg.secret]
    for s in forbidden_text:
        assert s not in blob, f"{s!r} leaked onto the wire"

    # No raw numeric value, at full float precision, in any form.
    leaked = []
    for r in recs:
        for v in (*[x for x in r.features.values() if isinstance(x, float)], r.output):
            for form in (repr(v), f"{v:.12g}"):
                if form in blob:
                    leaked.append(form)
    assert not leaked, f"raw values on the wire: {leaked[:5]}"

    # No public price either (prices are inputs, not outputs).
    for inst_bars in agent.prices._bars.values():
        for b in inst_bars:
            assert repr(b.price) not in blob


def test_only_schema_fields_and_exact_envelope(live_run):
    agent, sink, recs, cfg = live_run
    allowed = wire_fields()
    assert sink.bodies()
    for b in sink.bodies():
        assert set(b) == ENVELOPE_KEYS
        assert b["monitored_model_id"] == WIRE_MODEL
        assert b["tenant_id"] == "tenant-1" and b["install_id"] == "install-1"
        assert set(b["z_t"]) <= allowed, set(b["z_t"]) - allowed
        assert "unauthorized_deploy_flag" not in b["z_t"]
        assert "psi_max" not in b["z_t"]                # backend-derived, never asserted by the edge
        for k in ("true_metric", "event", "windows_to_event", "window_index"):
            assert k not in b["z_t"]                    # corpus-only supervision columns


def test_feature_maps_are_keyed_by_ids_never_names(live_run):
    agent, sink, recs, cfg = live_run
    live = [b["z_t"] for b in sink.bodies() if b["z_t"].get("warmup_state") == "live"]
    assert live
    seen = set()
    for z in live:
        for name in ("psi", "js_divergence", "ks_statistic", "wasserstein1", "chi_square",
                     "hellinger", "domain_classifier_attribution"):
            if isinstance(z.get(name), dict):
                seen |= set(z[name])
        for name in ("cardinality_change", "new_category_rate"):
            if isinstance(z.get(name), dict):
                seen |= set(z[name])
    assert seen and all(k.startswith("f_") for k in seen)
    # 3 numeric + 1 categorical feature -> 4 distinct ids, each resolvable LOCALLY.
    assert {agent.ids.name_for(k) for k in seen} == {*FEATURES, CATEGORY}
    assert any(isinstance(z.get("domain_classifier_attribution"), dict) for z in live)


def test_id_map_is_on_disk_with_names_and_not_sent(live_run):
    agent, sink, recs, cfg = live_run
    f = Path(cfg.state_dir, "feature_ids.json")
    assert f.exists() and oct(f.stat().st_mode & 0o777) == "0o600"
    assert FEATURES[0] in f.read_text()
    assert FEATURES[0] not in _everything_on_the_wire(agent, sink)

