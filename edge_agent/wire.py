"""The single place a payload is assembled -- and audited.

What may leave
--------------
The envelope: `schema_version`, `tenant_id`, `install_id`, `monitored_model_id`
(all BlurredBox-generated), `window_id` (sha256 of generated ids + window end),
`window_timestamp` (window close) and `z_t`. Inside `z_t`: only names in
`corpus.serving.wire_fields()`, with scalar types the backend's
`_validate_z_t` accepts. Per-feature maps are keyed by stable feature ids
(`featureids.py`), never by names. `score_quantiles` keys (`p1`..`p99`) are
fixed labels, not feature names, and pass unchanged.

`corpus.serving.flatten_z_t` is the projection that defines the scalar field
names the backend and the forecaster share; the scalars here are exactly its
output, restricted to the wire contract (its derived `psi_max` is computed by
the backend, so it is not sent).
"""
from __future__ import annotations

import hashlib
import math
from datetime import datetime
from typing import Any

from corpus.serving import (BACKEND_AUTHORITATIVE_FIELDS, MAP_VALUED_FIELDS,
                            PSI_AGGREGATE_COLUMN, flatten_z_t, wire_fields)

from edge_agent.featureids import FeatureIdMap

# Maps keyed by the partner's feature names -> re-keyed by id before sending.
FEATURE_KEYED_MAPS = tuple(m for m in MAP_VALUED_FIELDS if m != "score_quantiles") + (
    "domain_classifier_attribution",)
_FIXED_KEY_MAPS = ("score_quantiles",)
_BOOL_FIELDS = {"estimator_valid", "aux_model_valid", "ingest_incomplete_flag",
                "unauthorized_deploy_flag", "shed_event_flag", "rebaseline_event_flag"}
_TEXT_FIELDS = {"reference_window_id", "schema_version", "warmup_state",
                "domain_classifier_family", "mahalanobis_method", "novelty_score_method"}


class WireViolation(ValueError):
    """The payload would not be an acceptable, allow-listed, raw-data-free `z_t`. Never sent."""


def _num(v: Any) -> Any:
    """numpy/py number -> JSON-safe python number; non-finite -> None."""
    if isinstance(v, bool):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if not math.isfinite(f):
        return None
    return int(v) if isinstance(v, int) else f


def build_z_t(z_t: dict[str, Any], ids: FeatureIdMap) -> tuple[dict[str, Any], list[str]]:
    """Project a computed `z_t` onto the wire. Returns (wire z_t, dropped field names)."""
    allowed = wire_fields()
    flat = flatten_z_t(z_t)                     # the one train/serve projection
    wire: dict[str, Any] = {}
    dropped: list[str] = []
    for k, v in flat.items():
        if k == PSI_AGGREGATE_COLUMN:
            continue                            # backend-derived from `psi`
        if k in BACKEND_AUTHORITATIVE_FIELDS or k not in allowed:
            dropped.append(k)
            continue
        wire[k] = _num(v) if not isinstance(v, str) else v
    for name in FEATURE_KEYED_MAPS:             # flatten drops maps; re-add them id-keyed
        m = z_t.get(name)
        if isinstance(m, dict):
            wire[name] = {ids.id_for(str(k)): _num(v) for k, v in m.items()}
    for name in _FIXED_KEY_MAPS:
        m = z_t.get(name)
        if isinstance(m, dict):
            wire[name] = {str(k): _num(v) for k, v in m.items()}
    return wire, dropped


def audit_z_t(z_t: dict[str, Any], *, forbidden: set[str] | None = None) -> None:
    """Raise WireViolation unless `z_t` is exactly schema fields with schema types."""
    allowed = wire_fields()
    for k, v in z_t.items():
        if k not in allowed or k in BACKEND_AUTHORITATIVE_FIELDS:
            raise WireViolation(f"field {k!r} is not in the wire contract")
        if v is None:
            continue
        if k == "detector_alarms_fired":
            ok = type(v) is int and 0 <= v <= 31
        elif k in _BOOL_FIELDS or k.endswith("_valid"):
            ok = type(v) is bool
        elif k in _TEXT_FIELDS:
            ok = isinstance(v, str) and len(v) <= 128 and v.isascii()
        elif k in FEATURE_KEYED_MAPS or k in _FIXED_KEY_MAPS:
            ok = isinstance(v, dict) and len(v) <= 500 and all(
                isinstance(a, str) and len(a) <= 128 and a.isascii()
                and (b is None or (type(b) in (int, float) and math.isfinite(b)))
                for a, b in v.items())
            if ok and k in FEATURE_KEYED_MAPS:
                ok = all(a.startswith("f_") for a in v)     # ids only, never names
        else:
            ok = type(v) in (int, float) and math.isfinite(v)
        if not ok:
            raise WireViolation(f"invalid value type/bound for {k!r}")
        if forbidden:                           # belt and braces for the audit test
            for s in _strings(v):
                if s in forbidden:
                    raise WireViolation(f"forbidden string in {k!r}")
    if len(z_t) > 500:
        raise WireViolation("signal vector exceeds the 500-field width budget")


def _strings(v: Any):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for a, b in v.items():
            yield a
            yield from _strings(b)


def window_id(wire_model_id: str, window_end: datetime) -> str:
    """Deterministic: a re-send is a replay on the server, never a second alert."""
    raw = f"{wire_model_id}|{window_end.isoformat()}".encode()
    return hashlib.sha256(raw).hexdigest()[:32]


def build_envelope(*, schema_version: str, tenant_id: str, install_id: str,
                   monitored_model_id: str, window_end: datetime,
                   z_t: dict[str, Any]) -> dict[str, Any]:
    audit_z_t(z_t)
    return {
        "schema_version": schema_version,
        "tenant_id": tenant_id,
        "install_id": install_id,
        "monitored_model_id": monitored_model_id,
        "window_id": window_id(monitored_model_id, window_end),
        "window_timestamp": window_end.isoformat(),
        "z_t": z_t,
    }
