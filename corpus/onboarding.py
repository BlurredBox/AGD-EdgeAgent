"""
corpus/onboarding.py

Onboarding validator: tells an integrator BEFORE go-live whether the
telemetry they will send can produce a forecast, and says which fields to fix
when it cannot.

Why this exists. A deployment that sends "timestamp, model, daily AUC" supplies none
of the signal-vector fields the forecaster bundle selects, and nothing raises:
`ForecasterBundle.predict` NaN-fills (or, on the deployed path, refuses the
horizons that need an absent column) and the product shows nothing. This module
makes that failure loud and names the fields.

Everything here is LOCAL and READ-ONLY. It never opens a socket; the telemetry
file is read, analysed in memory and summarised as counts and field NAMES. The
same functions (`analyse_model`) are what a server-side diagnostic
runs over stored windows, so the two cannot
disagree about what "missing" or "all-NaN" means.

Nothing is remembered from documents: the required-field list is DERIVED from a
forecaster bundle's selected feature columns through `corpus.serving.
base_field_of` (the same map the per-model refusal contract uses), the accepted
field names are `corpus.serving.wire_fields()`, the warm-up numbers are read
from the runtime's own constants (with baked fallbacks that a test pins to the
code), and the value domains are the contract's (`corpus.ops_signals.OPS_BOUNDS`
plus the mathematical bounds of the formulas in `corpus.signal_vector`).

    python -m corpus.onboarding validate telemetry.csv [more files ...]
    python -m corpus.onboarding validate envelopes.jsonl --json
    python -m corpus.onboarding template            # the 17-field column guide

Accepted inputs: CSV, Parquet, JSONL.
  * JSONL: one signal envelope per line (`{"monitored_model_id", "window_timestamp",
    "z_t": {...}, "external_events": [...], "outcome_labels": [...]}`), i.e. exactly
    the body `POST /v1/ingest/signal` takes.
  * CSV / Parquet: one row per window. A timestamp column, an optional model column,
    then one column per field by its WIRE name. Map-valued fields (`wasserstein1`,
    `cardinality_change`, ...) are a JSON object in the cell or `field.key` columns;
    `score_quantiles` may be `score_quantile_p5` / `_p95` / `_p99` columns.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from corpus.ops_signals import OPS_BOUNDS
from corpus.serving import (
    IDENTIFIERS,
    MAP_VALUED_FIELDS,
    TRAJECTORY_SUFFIXES,
    base_field_of,
    wire_fields,
)

__all__ = [
    "READY", "NOT_READY", "OUTCOME_FIELDS", "FieldNeed", "RequiredSpec",
    "SANDBOX_BUNDLE_FEATURES", "SANDBOX_BUNDLE_SHA256", "default_required",
    "SANDBOX_PAIRED_FEATURES", "SANDBOX_PAIRED_SHA256", "SANDBOX_LEARNED_MANIFEST_SHA256",
    "PATH_LIVE", "PATH_LEARNED", "PATHS", "merge_required",
    "required_from_bundle", "derive_required", "warmup_requirements",
    "analyse_cadence", "WindowRow", "analyse_model", "validate_files",
    "load_windows", "render_text", "RUNTIME_CADENCE_MAX_MINUTES", "FIELD_BOUNDS",
    "field_series_status", "tail_null_run", "collection_contract",
]

READY = "READY"
NOT_READY = "NOT READY"


def collection_contract(output_kind: str, required: "RequiredSpec | None" = None) -> dict[str, Any]:
    """Describe required local sources and output compatibility for pinned artifact fields.

    Compatibility is a prerequisite, never a statement that a scoring path is
    calibrated or that its required fields have actually been observed.
    """
    if output_kind not in {"signed", "probability"}:
        raise ValueError("output_kind must be signed or probability")
    spec = required or default_required()
    rows = []
    for need in spec.needs:
        name = need.wire_field
        probability_only = name in {"positive_decision_rate", "predictive_entropy", "brier", "nll", "ece", "mce"}
        if need.needs_outcomes:
            source = "realised_outcomes"
        elif name.startswith("score_") or name in {"predictive_entropy", "positive_decision_rate"}:
            source = "model_outputs"
        elif name == "row_count_vs_expected":
            source = "prediction_stream_and_reference"
        else:
            source = "input_features_and_reference"
        rows.append({"field": need.name, "paths": list(need.paths), "source": source,
                     "needs_outcomes": need.needs_outcomes,
                     "output_compatible": not probability_only or output_kind == "probability"})
    return {"required_source": spec.source, "output_kind": output_kind, "fields": rows,
            "paths": {path: {"output_compatible": all(r["output_compatible"] for r in rows if path in r["paths"]),
                             "incompatible_fields": [r["field"] for r in rows
                                                     if path in r["paths"] and not r["output_compatible"]]}
                      for path in spec.by_path},
            "raw_data_boundary": "local_only"}

# The server runtime limits cadence to (0, 60] minutes:
# a channel cannot be configured for a cadence above an hour.
RUNTIME_CADENCE_MAX_MINUTES = 60.0
# A file's cadence is regular enough to convert windows to wall-clock when this
# share of its intervals equal the median (same bar as the server-side replay).
CADENCE_REGULARITY_BAR = 0.90
GAP_FACTOR = 3.0          # an interval this many times the median is a gap

# Fields that need REALISED OUTCOMES (labels), which a trading/ops deployment often
# cannot supply.
OUTCOME_FIELDS = frozenset({"brier", "nll", "ece", "mce", "realised_metric", "ci_width"})

# ── What the bundle needs ───────────────────────────────────────────────────

# The selected feature columns, per horizon, of the reference
# forecaster artifact (the hosted
# sandbox's artifact), read with `ForecasterBundle.features_by_horizon`.
# An upstream validator test re-derives this from the pickle when it is
# present and fails on any drift, so this is a snapshot with a tripwire, not a
# memory. `--bundle DIR` (or the server's live bundle) replaces it.
SANDBOX_BUNDLE_SHA256 = "21abffc554f68c49eabb2994ff120b1c76ec3e53c7e4770a0831eac5c66f9cb1"
SANDBOX_BUNDLE_FEATURES: dict[int, tuple[str, ...]] = {
    1: ("ci_width", "brier", "brier__ewma_slope", "score_quantile_p5__second_derivative",
        "positive_decision_rate", "nll__ewma_abs_change", "score_kurtosis"),
    3: ("ci_width", "brier__ewma_slope", "hstrees_score", "row_count_vs_expected__ewma_slope",
        "predictive_entropy", "score_quantile_p95__ewma_abs_change", "score_std"),
    6: ("brier__ewma_slope", "positive_decision_rate__ewma",
        "covariance_delta_frobenius__ewma_abs_change", "cardinality_change__ewma_abs_change",
        "row_count_vs_expected__ewma", "ci_width__ewma", "ci_width"),
    12: ("wasserstein1__ewma_abs_change", "score_std__ewma", "ci_width__ewma_abs_change",
         "brier__ewma_slope", "row_count_vs_expected__ewma", "cardinality_change__ewma_abs_change",
         "row_count_vs_expected__ewma_slope"),
    24: ("cardinality_change__ewma", "score_std__ewma", "row_count_vs_expected__ewma",
         "score_mean__ewma_abs_change", "row_count_vs_expected__ewma_slope",
         "score_quantile_p99__ewma", "brier__ewma_slope"),
    48: ("score_std__ewma", "cardinality_change__ewma", "nll__ewma",
         "cardinality_change__ewma_abs_change", "score_quantile_p95__ewma",
         "domain_classifier_auc__ewma", "covariance_delta_frobenius__ewma"),
}

# The learned detector (manifest sha256 below) does not read the live
# forecaster's hazards: it reads the six hazards of its OWN paired forecaster, and the
# deployed path refuses a horizon whose selected column is absent, so one absent field =
# no hazard at that horizon = `hazards_missing` = the detector never goes live. Its
# required fields are therefore the paired bundle's selected columns, NOT the live
# bundle's. Snapshot of its paired `forecaster.pkl`; the same tripwire
# test re-derives it from the pickle.
PATH_LIVE = "live_forecast"
PATH_LEARNED = "learned_detector"
PATHS = (PATH_LIVE, PATH_LEARNED)
SANDBOX_PAIRED_SHA256 = "068abc90b513ce10e4f2754818ba4a23e256646f5419ca5366a1a0f4a71324c0"
SANDBOX_LEARNED_MANIFEST_SHA256 = "91ef044d7756b4651ecc488435adadf57db07c740dc9449bb9a6c599c06e20af"
SANDBOX_PAIRED_FEATURES: dict[int, tuple[str, ...]] = {
    1: ("score_quantile_p95__ewma_slope", "score_std", "positive_decision_rate__ewma_slope",
        "score_quantile_p50__ewma_slope", "domain_classifier_auc__ewma", "score_mean__ewma_slope",
        "score_quantile_p50__delta", "prediction_entropy__ewma", "score_mean__delta",
        "row_count_vs_expected__ewma", "psi_max", "mahalanobis_distance",
        "cardinality_change__ewma_abs_change"),
    3: ("score_std__ewma", "positive_decision_rate__ewma_slope", "covariance_delta_frobenius__ewma",
        "score_std", "row_count_vs_expected__ewma", "psi_max", "score_quantile_p50__ewma_slope",
        "ks_statistic__ewma", "cardinality_change__ewma_abs_change",
        "score_quantile_p95__ewma_abs_change", "predictive_entropy",
        "chi_square__ewma_abs_change", "row_count_vs_expected"),
    6: ("positive_decision_rate", "covariance_delta", "cardinality_change__ewma_abs_change",
        "row_count_vs_expected__ewma", "score_quantile_p50__ewma", "score_std__ewma",
        "row_count_vs_expected__ewma_abs_change", "epistemic_variance__ewma",
        "positive_decision_rate__ewma_slope", "score_mean__ewma_slope",
        "covariance_delta_frobenius__ewma", "score_quantile_p50__ewma_slope",
        "positive_decision_rate__ewma"),
    12: ("row_count_vs_expected__ewma", "cardinality_change__ewma_abs_change",
         "epistemic_variance__ewma", "row_count_vs_expected__ewma_abs_change",
         "row_count_vs_expected__ewma_slope", "score_std__ewma", "positive_decision_rate",
         "positive_decision_rate__ewma_slope", "score_quantile_p50__ewma_abs_change",
         "ks_statistic__ewma", "psi__ewma", "pca_reconstruction_error__second_derivative",
         "wasserstein1__ewma"),
    24: ("row_count_vs_expected__ewma", "row_count_vs_expected__ewma_abs_change",
         "row_count_vs_expected__ewma_slope", "cardinality_change__ewma",
         "cardinality_change__ewma_abs_change", "score_quantile_p50__ewma_abs_change",
         "wasserstein1__ewma", "covariance_delta_frobenius__ewma", "score_std__ewma",
         "score_quantile_p99__ewma", "score_quantile_p50__ewma", "hellinger__ewma",
         "positive_decision_rate__ewma_slope"),
    48: ("score_std__ewma", "row_count_vs_expected__ewma_abs_change",
         "row_count_vs_expected__ewma_slope", "cardinality_change__ewma", "row_count_vs_expected__ewma",
         "wasserstein1__ewma", "covariance_delta_frobenius__ewma", "psi__ewma",
         "chi_square__ewma_abs_change", "cardinality_change__ewma_abs_change",
         "score_quantile_p50__ewma_abs_change", "ks_statistic__ewma", "top_eigenvalue_ratio__ewma"),
}


@dataclass(frozen=True)
class FieldNeed:
    """One thing a deployment must send: a wire field, or one key of `score_quantiles`."""
    wire_field: str
    subkey: str | None
    horizons: tuple[int, ...]
    needs_outcomes: bool
    # Which serving paths read it (PATHS). A field one path needs and the other does not
    # still blocks the path that needs it.
    paths: tuple[str, ...] = (PATH_LIVE,)

    @property
    def name(self) -> str:
        return f"{self.wire_field}.{self.subkey}" if self.subkey else self.wire_field


@dataclass(frozen=True)
class RequiredSpec:
    needs: tuple[FieldNeed, ...]
    by_horizon: dict[int, tuple[str, ...]]      # horizon -> FieldNeed.name it selects (live path)
    source: str
    # path -> horizon -> FieldNeed.names, and path -> all names. `by_horizon` stays the LIVE
    # path's (horizons collide across paths, so the union keeps one map per path).
    by_path_horizon: dict[str, dict[int, tuple[str, ...]]] = field(default_factory=dict)
    by_path: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(sorted(self.by_horizon))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(n.name for n in self.needs)


def _need_of_column(col: str) -> tuple[str, str | None]:
    """A bundle feature column -> (wire field, subkey). `corpus.serving.base_field_of`
    resolves the wire field; a `corpus.trajectory` subkey slot (score_quantile_p5)
    also pins WHICH key of the map the column reads."""
    from corpus.trajectory import SLOTS
    base = col
    for sfx in TRAJECTORY_SUFFIXES:
        if base.endswith(sfx):
            base = base[: -len(sfx)]
            break
    for slot in SLOTS:
        if slot.name == base and slot.kind == "subkey":
            return base_field_of(col), slot.subkey
    return base_field_of(col), None


def derive_required(columns_by_horizon: Mapping[int, Sequence[str]], source: str,
                    path: str = PATH_LIVE) -> RequiredSpec:
    by_h: dict[int, list[str]] = {}
    acc: dict[tuple[str, str | None], set[int]] = {}
    for h, cols in columns_by_horizon.items():
        for c in cols:
            wf, sk = _need_of_column(c)
            acc.setdefault((wf, sk), set()).add(int(h))
            name = f"{wf}.{sk}" if sk else wf
            if name not in by_h.setdefault(int(h), []):
                by_h[int(h)].append(name)
    needs = tuple(FieldNeed(wf, sk, tuple(sorted(hs)), wf in OUTCOME_FIELDS, (path,))
                  for (wf, sk), hs in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")))
    bh = {h: tuple(v) for h, v in sorted(by_h.items())}
    return RequiredSpec(needs, bh, source, {path: bh}, {path: tuple(n.name for n in needs)})


def merge_required(live: RequiredSpec, learned: RequiredSpec) -> RequiredSpec:
    """The union of the two serving paths' field sets, each field tagged with the paths that
    read it. `by_horizon` stays the live path's; both paths' horizon maps are in `by_path_horizon`."""
    acc: dict[tuple[str, str | None], tuple[set[int], set[str]]] = {}
    for spec in (live, learned):
        for n in spec.needs:
            hs, ps = acc.setdefault((n.wire_field, n.subkey), (set(), set()))
            hs.update(n.horizons)
            ps.update(n.paths)
    needs = tuple(FieldNeed(wf, sk, tuple(sorted(hs)), wf in OUTCOME_FIELDS,
                            tuple(p for p in PATHS if p in ps))
                  for (wf, sk), (hs, ps) in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")))
    return RequiredSpec(needs, dict(live.by_horizon), f"{live.source} + {learned.source}",
                        {**live.by_path_horizon, **learned.by_path_horizon},
                        {**live.by_path, **learned.by_path})


def default_required(path: str = "all") -> RequiredSpec:
    """The snapshot of the shipped artifacts. `path`: `all` (default; the union, every field
    either serving path needs), `live_forecast` or `learned_detector`."""
    live = derive_required(SANDBOX_BUNDLE_FEATURES,
                           f"sandbox forecaster snapshot sha256={SANDBOX_BUNDLE_SHA256[:12]}", PATH_LIVE)
    if path == PATH_LIVE:
        return live
    learned = derive_required(SANDBOX_PAIRED_FEATURES,
                              f"paired forecaster snapshot sha256={SANDBOX_PAIRED_SHA256[:12]}", PATH_LEARNED)
    if path == PATH_LEARNED:
        return learned
    if path != "all":
        raise ValueError(f"path must be all, {PATH_LIVE} or {PATH_LEARNED}")
    return merge_required(live, learned)


def required_from_bundle(bundle_or_dir: Any, paired: Any = None, path: str = PATH_LIVE) -> RequiredSpec:
    """Derive the list from a `ForecasterBundle` (or a directory holding
    `forecaster.pkl`). With `paired` (the learned detector's own forecaster), the union
    of both paths. Unpickling executes code: only point this at an artifact you
    built or were handed by BlurredBox."""
    if paired is not None:
        return merge_required(required_from_bundle(bundle_or_dir),
                              required_from_bundle(paired, path=PATH_LEARNED))
    if isinstance(bundle_or_dir, (str, Path)):
        p = Path(bundle_or_dir)
        bundle = pickle.loads((p / "forecaster.pkl").read_bytes())  # trusted artifact
        src = f"bundle {p.name}"
    else:
        bundle, src = bundle_or_dir, ("learned-detector forecaster bundle" if path == PATH_LEARNED
                                      else "live forecaster bundle")
    return derive_required({h: list(bundle.features_by_horizon[h]) for h in bundle.horizons}, src, path)


# ── Warm-up, from the runtime's own constants ───────────────────────────────

# Baked fallbacks, pinned to the code by an upstream validator test. Used only
# where the runtime module is not importable (an edge machine with corpus/ alone).
_FALLBACK = {"alpha": 0.0076, "n_horizons": 6, "floor_min": 200, "bank_reference": 200,
             "bank_scale": 100, "bank_calibration": 1500, "learned_ref": 60, "learned_min_live": 60,
             "learned_score_from": 100, "learned_tau_warmup": 96, "perf_ref": 60, "perf_post": 20,
             "ext_samples": 288}


def _runtime_constants() -> dict[str, Any]:
    c = dict(_FALLBACK)
    try:
        from src.core.forecasters import forecaster_service as fs
        c["alpha"], c["n_horizons"] = fs.DEFAULT_ALPHA, fs.N_HORIZONS
    except Exception:
        pass
    try:
        from src.policy.forecast import learned_detector as ld
        c.update(learned_ref=ld.REF_WINDOWS, learned_min_live=ld.MIN_LIVE, learned_score_from=ld.SCORE_FROM)
    except Exception:
        pass
    try:
        from src.policy.forecast.tau_calibration import CalibrationConfig
        c["learned_tau_warmup"] = CalibrationConfig().warmup_windows
    except Exception:
        pass
    try:
        from src.policy.forecast.perf_monitor import PerfConfig
        c.update(perf_ref=PerfConfig().ref_windows, perf_post=PerfConfig().min_post_labels)
    except Exception:
        pass
    try:
        from src.api import signal_runtime as sr
        c["bank_calibration"] = sr.DEFAULT_BANK_CALIBRATION
    except Exception:
        pass
    return c


def warmup_requirements(alpha: float | None = None, n_horizons: int | None = None) -> list[dict[str, Any]]:
    """The windows each runtime component needs before it can act, with where the number
    comes from. `windows` is a floor on windows of ONE model with usable fields."""
    c = _runtime_constants()
    a = c["alpha"] if alpha is None else alpha
    h = c["n_horizons"] if n_horizons is None else n_horizons
    floor = max(c["floor_min"], math.ceil(h / a) - 1)
    return [
        {"component": "forecaster_calibration", "windows": floor, "required": True,
         "source": "forecaster_service.stage_2_to_3_floor: M = max(200, ceil(H/alpha) - 1), "
                   f"H={h}, alpha={a:g}",
         "meaning": "below M the conformal threshold tau_h does not exist: the window is 'uncalibrated' "
                    "and no calibrated tier can fire"},
        {"component": "persistence_charts", "windows": 2 * floor, "required": False,
         "source": "pipeline._tracker: needs M finite margins AFTER tau exists, so 2M "
                   "(M with overlap_warmup)",
         "meaning": "CUSUM/EWMA persistence needs the margins before it can escalate"},
        {"component": "detector_bank", "windows": c["bank_reference"] + c["bank_scale"] + c["bank_calibration"],
         "required": False,
         "source": "signal_runtime bank config (moving): reference "
                   f"{c['bank_reference']} + scale {c['bank_scale']} + calibration {c['bank_calibration']} "
                   "nominal windows (more if the per-family conformal floor binds)",
         "meaning": "DETECTED alerts need the bank live"},
        {"component": "learned_detector", "windows": c["learned_score_from"], "required": False,
         "source": f"learned_detector: REF_WINDOWS={c['learned_ref']}, MIN_LIVE={c['learned_min_live']}, "
                   f"first scored hazard row SCORE_FROM={c['learned_score_from']}",
         "meaning": "scores start once the reference closes; the artifact tau is used until tau "
                    f"calibration (+{c['learned_tau_warmup']} scored windows if "
                    "AGD_SIGNAL_LEARNED_CALIBRATION=true)"},
        {"component": "learned_detector_tau_calibration",
         "windows": c["learned_score_from"] + c["learned_tau_warmup"], "required": False,
         "source": "tau_calibration.CalibrationConfig.warmup_windows (only when the flag is on)",
         "meaning": "per-deployment alert threshold"},
        {"component": "performance_monitor", "windows": c["perf_ref"], "required": False,
         "source": f"perf_monitor.PerfConfig: ref_windows={c['perf_ref']} labelled reference windows, then "
                   f"min_post_labels={c['perf_post']} labels before any alert",
         "meaning": "inert until outcome_labels arrive (AGD_SIGNAL_PERF_MONITOR)"},
    ]


# ── Value domains ───────────────────────────────────────────────────────────

# Mathematical bounds of the formulas in corpus.signal_vector, plus the contract's
# OPS_BOUNDS. A value outside them is a unit/mapping error (percent for a fraction,
# a negative std), not an unusual model.
FIELD_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "brier": (0.0, 2.0), "nll": (0.0, None), "ci_width": (0.0, 2.0),
    "ece": (0.0, 1.0), "mce": (0.0, 1.0), "positive_decision_rate": (0.0, 1.0),
    "score_std": (0.0, None), "row_count_vs_expected": (0.0, None),
    "predictive_entropy": (0.0, None), "wasserstein1": (0.0, None),
    "covariance_delta": (0.0, None), "score_kurtosis": (-2.0, None),
    **OPS_BOUNDS,
}
QUANTILE_ORDER = ("p1", "p5", "p50", "p95", "p99")

_MODEL_KEYS = ("monitored_model_id", "model_id", "model", "model_name", "modelid")
_TS_KEYS = ("window_timestamp", "window_end", "timestamp", "ts", "time", "datetime", "date")
_ENVELOPE_ONLY = {"tenant_id", "install_id", "window_id", "alert_ttl_seconds", "sdk_written_us",
                  "external_events", "outcome_labels", "z_t"}
# Common telemetry columns that are not signal fields, with what they are NOT.
_HINTS = {
    "auc": "a model-quality metric is not a z_t field: the 17 fields are computed from per-record "
           "predictions (and features); see 'template'",
    "daily_auc": "an aggregate AUC is not a z_t field; send per-window score statistics instead",
    "accuracy": "accuracy is not a z_t field (realised_metric needs per-window outcomes)",
    "prediction": "per-record predictions are reduced to score_mean/score_std/score_quantiles on the "
                  "partner's machine; they are not sent",
    "score": "per-record scores are reduced to score_mean/score_std/score_quantiles; they are not sent",
    "domain_classifier_auc": "the wire name is `domain_classifier_drift`",
    "covariance_delta_frobenius": "the wire name is `covariance_delta`",
    "prediction_entropy": "the wire name is `predictive_entropy`",
}


# ── Reading files ───────────────────────────────────────────────────────────

@dataclass
class WindowRow:
    ts: datetime | None
    z_t: dict[str, Any]
    external_events: list[Any] | None = None
    outcome_labels: list[Any] | None = None
    schema_version: str | None = None
    ts_naive: bool = False


_NULLS = {"", "nan", "null", "none", "na", "n/a", "NaN", "NULL", "None", "NA"}


def _cell(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (bool, int, float, dict, list)):
        return v
    s = str(v).strip()
    if s in _NULLS or s.lower() in _NULLS:
        return None
    if s[0] in "{[":
        try:
            return json.loads(s)
        except ValueError:
            return s
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return float(s)
    except ValueError:
        return s


def _parse_ts(v: Any) -> tuple[datetime | None, bool]:
    """-> (UTC datetime or None, was_naive)."""
    if v is None:
        return None, False
    if isinstance(v, datetime):
        dt = v
    elif isinstance(v, (int, float)) and not isinstance(v, bool):
        x = float(v)
        if x > 1e17:
            x /= 1e9       # ns
        elif x > 1e14:
            x /= 1e6       # us
        elif x > 1e11:
            x /= 1e3       # ms
        try:
            return datetime.fromtimestamp(x, timezone.utc), False
        except (OverflowError, OSError, ValueError):
            return None, False
    elif hasattr(v, "to_pydatetime"):          # pandas Timestamp
        dt = v.to_pydatetime()
    else:
        s = str(v).strip()
        if s.endswith(("Z", "z")):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            try:
                dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S%z")
            except ValueError:
                return None, False
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc), True
    return dt.astimezone(timezone.utc), False


def _pick(row: Mapping[str, Any], keys: Sequence[str]) -> tuple[str | None, Any]:
    low = {str(k).lower(): k for k in row}
    for k in keys:
        if k in low:
            return low[k], row[low[k]]
    return None, None


def _flat_to_window(row: Mapping[str, Any]) -> tuple[str, WindowRow, set[str]]:
    """One flat CSV/Parquet row -> (model, window, extra column names)."""
    mk, model = _pick(row, _MODEL_KEYS)
    tk, tsv = _pick(row, _TS_KEYS)
    ts, naive = _parse_ts(_cell(tsv) if not hasattr(tsv, "to_pydatetime") else tsv)
    z: dict[str, Any] = {}
    nested = _cell(row.get("z_t")) if "z_t" in row else None
    if isinstance(nested, dict):
        z.update(nested)
    quant: dict[str, Any] = {}
    maps: dict[str, dict[str, Any]] = {}
    extra: set[str] = set()
    allowed = wire_fields()
    for k, raw in row.items():
        if k in (mk, tk) or k in _ENVELOPE_ONLY or k in IDENTIFIERS:
            continue
        v = _cell(raw)
        if k.startswith("score_quantile_") and k[len("score_quantile_"):] in QUANTILE_ORDER:
            quant[k[len("score_quantile_"):]] = v
        elif "." in k and k.split(".", 1)[0] in MAP_VALUED_FIELDS:
            f, key = k.split(".", 1)
            maps.setdefault(f, {})[key] = v
        elif k in allowed:
            z[k] = v
        else:
            extra.add(k)
    if quant and "score_quantiles" not in z:
        z["score_quantiles"] = quant
    for f, m in maps.items():
        z.setdefault(f, m)
    ev = _cell(row.get("external_events")) if "external_events" in row else None
    lb = _cell(row.get("outcome_labels")) if "outcome_labels" in row else None
    sv = z.get("schema_version")
    return (str(model) if model is not None and _cell(model) is not None else "all",
            WindowRow(ts, z, ev if isinstance(ev, list) else None,
                      lb if isinstance(lb, list) else None,
                      sv if isinstance(sv, str) else None, naive), extra)


def _envelope_to_window(obj: Mapping[str, Any]) -> tuple[str, WindowRow]:
    z = obj.get("z_t") if isinstance(obj.get("z_t"), dict) else {}
    ts, naive = _parse_ts(obj.get("window_timestamp") or _pick(obj, _TS_KEYS)[1])
    ev, lb = obj.get("external_events"), obj.get("outcome_labels")
    return (str(obj.get("monitored_model_id") or _pick(obj, _MODEL_KEYS)[1] or "all"),
            WindowRow(ts, dict(z), ev if isinstance(ev, list) else None,
                      lb if isinstance(lb, list) else None,
                      obj.get("schema_version") if isinstance(obj.get("schema_version"), str) else None,
                      naive))


def load_windows(path: str | Path) -> tuple[dict[str, list[WindowRow]], dict[str, set[str]], dict[str, Any]]:
    """Read one file -> ({model: [WindowRow...]}, {model: extra columns}, file info).

    The format is the suffix: .csv / .parquet / .jsonl (.ndjson, .json lines)."""
    p = Path(path)
    suf = p.suffix.lower()
    models: dict[str, list[WindowRow]] = {}
    extras: dict[str, set[str]] = {}
    info: dict[str, Any] = {"path": str(p), "format": suf.lstrip("."), "rows": 0, "envelope_rows": 0,
                            "bad_lines": 0, "columns": []}
    if suf in (".jsonl", ".ndjson"):
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    info["bad_lines"] += 1
                    continue
                if not isinstance(obj, dict):
                    info["bad_lines"] += 1
                    continue
                info["rows"] += 1
                if "z_t" in obj:
                    info["envelope_rows"] += 1
                    m, w = _envelope_to_window(obj)
                    extras.setdefault(m, set())
                else:
                    m, w, ex = _flat_to_window(obj)
                    extras.setdefault(m, set()).update(ex)
                models.setdefault(m, []).append(w)
        return models, extras, info
    if suf == ".csv":
        with p.open(newline="", encoding="utf-8-sig") as fh:
            rd = csv.DictReader(fh)
            info["columns"] = list(rd.fieldnames or [])
            rows: Iterable[Mapping[str, Any]] = rd
            for r in rows:
                info["rows"] += 1
                m, w, ex = _flat_to_window(r)
                models.setdefault(m, []).append(w)
                extras.setdefault(m, set()).update(ex)
        return models, extras, info
    if suf in (".parquet", ".pq"):
        import pandas as pd
        df = pd.read_parquet(p)
        info["columns"] = [str(c) for c in df.columns]
        for r in df.to_dict(orient="records"):
            info["rows"] += 1
            m, w, ex = _flat_to_window(r)
            models.setdefault(m, []).append(w)
            extras.setdefault(m, set()).update(ex)
        return models, extras, info
    raise ValueError(f"{p.name}: unsupported format {suf!r}; use .csv, .parquet or .jsonl")


# ── Per-field analysis ──────────────────────────────────────────────────────

def _finite(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def tail_null_run(finite_flags: Sequence[bool]) -> int:
    """Length of the run of non-finite windows at the END of the series."""
    n = 0
    for ok in reversed(finite_flags):
        if ok:
            break
        n += 1
    return n


def field_series_status(need: FieldNeed | tuple[str, str | None], z_rows: Sequence[Mapping[str, Any]],
                        *, constant_min_n: int = 20) -> dict[str, Any]:
    """Analyse one field over one model's windows.

    status (worst first): `missing` (no window carries the key), `all_nan` (carried, never a
    finite value), `wrong_shape` (map-valued sent as a scalar or the reverse),
    `out_of_range` (>=5% of finite values outside the field's domain), `constant`, `ok`.
    """
    wf, sk = (need.wire_field, need.subkey) if isinstance(need, FieldNeed) else need
    n = len(z_rows)
    present = shape_bad = 0
    vals: list[float] = []
    flags: list[bool] = []
    is_map = wf in MAP_VALUED_FIELDS
    n_inf = 0
    for z in z_rows:
        raw = z.get(wf) if isinstance(z, Mapping) else None
        if wf in z:
            present += 1
        v: float | None = None
        if raw is None:
            pass
        elif is_map:
            if not isinstance(raw, dict):
                shape_bad += 1
            elif sk is not None:
                v = _finite(raw.get(sk))
            else:
                fin = [x for x in (_finite(y) for y in raw.values()) if x is not None]
                v = float(np.mean(fin)) if fin else None   # a map counts as present when any key is finite
        else:
            if isinstance(raw, (dict, list, str)):
                shape_bad += 1
            else:
                v = _finite(raw)
                if v is None and isinstance(raw, (int, float, np.floating)) and not isinstance(raw, bool):
                    n_inf += 1          # NaN/inf literal: ingest rejects it, null is how to say "unknown"
        flags.append(v is not None)
        if v is not None:
            vals.append(v)
    lo, hi = FIELD_BOUNDS.get(wf, (None, None))
    out = sum(1 for x in vals if (lo is not None and x < lo) or (hi is not None and x > hi)) if lo is not None or hi is not None else 0
    out_frac = out / len(vals) if vals else 0.0
    arr = np.asarray(vals, dtype=float)
    constant = bool(len(arr) >= constant_min_n and float(np.ptp(arr)) == 0.0)
    if present == 0:
        status = "missing"
    elif shape_bad and not vals:
        status = "wrong_shape"
    elif not vals:
        status = "all_nan"
    elif out_frac >= 0.05:
        status = "out_of_range"
    elif constant:
        status = "constant"
    else:
        status = "ok"
    return {
        "field": f"{wf}.{sk}" if sk else wf, "status": status, "windows": n,
        "windows_with_key": present, "windows_finite": len(vals),
        "windows_null": n - len(vals), "tail_null_run": tail_null_run(flags),
        "windows_wrong_shape": shape_bad, "windows_out_of_range": out,
        "windows_nonfinite_literal": n_inf,
        "min": float(arr.min()) if len(arr) else None, "max": float(arr.max()) if len(arr) else None,
        "domain": [lo, hi] if (lo is not None or hi is not None) else None,
    }


_FIX = {
    "missing": "add the field `{f}` to every window (see `template`)",
    "all_nan": "`{f}` is present but never has a finite value: it is being computed from nothing "
               "(empty join, wrong column, or a window below the minimum row count). Fix the computation",
    "wrong_shape": "`{f}` has the wrong shape: " ,
    "out_of_range": "`{f}` has values outside its domain {d}: check units (fraction vs percent) and sign",
    "constant": "`{f}` never changes across {n} windows: check it is recomputed per window, not copied",
}


def _fix_text(r: dict[str, Any], wf: str) -> str:
    s = r["status"]
    f = r["field"]
    if s == "wrong_shape":
        if wf in MAP_VALUED_FIELDS:
            return (f"`{f}` is per-feature map-valued on the wire: send an object "
                    '{"<feature>": value, ...} (CSV: a JSON cell or `' + f + '.<feature>` columns), not a scalar')
        return f"`{f}` is a scalar on the wire: send a number or null, not an object/string"
    return _FIX[s].format(f=f, d=r.get("domain"), n=r["windows"])


# ── Cadence ─────────────────────────────────────────────────────────────────

def analyse_cadence(timestamps: Sequence[datetime | None], *, declared_minutes: float | None = None
                    ) -> dict[str, Any]:
    ts = [t for t in timestamps if t is not None]
    out: dict[str, Any] = {"timestamps": len(ts), "unparseable": len(timestamps) - len(ts),
                           "declared_minutes": declared_minutes}
    if len(ts) < 2:
        out.update(measured_minutes=None, regular=False, reason="fewer than two timestamps")
        return out
    epoch = np.array([t.timestamp() for t in ts], dtype=float)
    out_of_order = int(np.sum(np.diff(epoch) < 0))
    s = np.sort(epoch)
    d = np.diff(s)
    dups = int(np.sum(d == 0))
    pos = d[d > 0]
    out.update(first=datetime.fromtimestamp(float(s[0]), timezone.utc).isoformat(),
               last=datetime.fromtimestamp(float(s[-1]), timezone.utc).isoformat(),
               span_hours=round(float(s[-1] - s[0]) / 3600.0, 3),
               out_of_order=out_of_order, duplicates=dups)
    if not len(pos):
        out.update(measured_minutes=None, regular=False, reason="every timestamp is identical")
        return out
    med = float(np.median(pos))
    tol = max(1.0, 0.05 * med)
    regularity = float(np.mean(np.abs(pos - med) <= tol))
    gaps = pos[pos > GAP_FACTOR * med]
    missing = int(np.sum(np.round(pos[pos > 1.5 * med] / med) - 1)) if med > 0 else 0
    out.update(
        median_seconds=med, measured_minutes=round(med / 60.0, 4),
        regularity=round(regularity, 4), regular=regularity >= CADENCE_REGULARITY_BAR,
        p95_seconds=float(np.percentile(pos, 95)), max_gap_seconds=float(pos.max()),
        gaps=int(len(gaps)), missing_windows_est=missing,
        reason=(f"{regularity:.1%} of intervals match the median {med / 60.0:g} min"))
    return out


# ── Optional channels ───────────────────────────────────────────────────────

def _check_optional(rows: Sequence[WindowRow]) -> dict[str, Any]:
    ev_rows = [r for r in rows if r.external_events]
    lb_rows = [r for r in rows if r.outcome_labels]
    ev: dict[str, Any] = {"windows_with": len(ev_rows), "events": 0, "malformed": 0, "kinds": {},
                          "enabled_by": "AGD_SIGNAL_EXTERNAL_EVENTS=true"}
    try:
        from src.policy.forecast.external_signals import ExternalEvent
    except Exception:
        ExternalEvent = None  # type: ignore[assignment]
    for r in ev_rows:
        for e in r.external_events or []:
            ev["events"] += 1
            if ExternalEvent is None:
                continue
            try:
                m = ExternalEvent.model_validate(e)
                ev["kinds"][m.kind] = ev["kinds"].get(m.kind, 0) + 1
            except Exception:
                ev["malformed"] += 1
    lb: dict[str, Any] = {"windows_with": len(lb_rows), "labels": 0, "malformed": 0,
                          "enabled_by": "AGD_SIGNAL_PERF_MONITOR"}
    for r in lb_rows:
        for e in r.outcome_labels or []:
            lb["labels"] += 1
            ok = (isinstance(e, dict) and type(e.get("window_index")) is int and e["window_index"] >= 0
                  and _finite(e.get("n_labelled")) is not None and _finite(e.get("n_errors")) is not None
                  and 0 <= e["n_errors"] <= e["n_labelled"] and set(e) <= {"window_index", "n_labelled", "n_errors"})
            if not ok:
                lb["malformed"] += 1
    return {"external_events": ev, "outcome_labels": lb}


# ── One model ───────────────────────────────────────────────────────────────

def _fmt_duration(windows: float, minutes: float | None) -> str | None:
    if minutes is None:
        return None
    h = windows * minutes / 60.0
    return f"{h:.1f} h" if h < 48 else f"{h / 24:.1f} days"


def analyse_model(model_id: str, rows: Sequence[WindowRow], *, required: RequiredSpec | None = None,
                  alpha: float | None = None, declared_cadence_minutes: float | None = None,
                  allow_short_history: bool = False, extra_columns: Iterable[str] = (),
                  sort: bool = True) -> dict[str, Any]:
    """The full per-model report. Pure: same function for a file and for stored windows."""
    spec = required or default_required()
    rows = sorted(rows, key=lambda r: (r.ts is None, r.ts)) if sort else list(rows)
    z_rows = [r.z_t for r in rows]
    n = len(rows)
    blockers: list[dict[str, str]] = []
    warnings: list[str] = []
    fixes: list[str] = []

    def block(kind: str, msg: str, fix: str | None = None) -> None:
        blockers.append({"kind": kind, "message": msg})
        if fix:
            fixes.append(fix)

    # fields
    fields = []
    for need in spec.needs:
        r = field_series_status(need, z_rows)
        r.update(required_by_horizons=list(need.horizons), needs_outcomes=need.needs_outcomes,
                 required_by_paths=list(need.paths))
        fields.append(r)
    by_name = {f["field"]: f for f in fields}

    # horizons
    horizons = {}
    for h, names in spec.by_horizon.items():
        blocked_by = [nm for nm in names if by_name[nm]["status"] in ("missing", "all_nan", "wrong_shape")]
        horizons[h] = {"scorable": not blocked_by, "blocked_by": blocked_by}
    scorable = [h for h, v in horizons.items() if v["scorable"]]
    # Per serving path. The learned detector needs ALL of its paired forecaster's horizons
    # (one missing hazard = `hazards_missing`, never live); the live path needs one.
    paths: dict[str, Any] = {}
    for pth, hmap in spec.by_path_horizon.items():
        ph = {}
        for h, names in hmap.items():
            bad = [nm for nm in names if by_name[nm]["status"] in ("missing", "all_nan", "wrong_shape")]
            ph[str(h)] = {"scorable": not bad, "blocked_by": bad}
        ok = all(v["scorable"] for v in ph.values()) if pth == PATH_LEARNED else any(v["scorable"] for v in ph.values())
        paths[pth] = {"verdict": READY if ok else NOT_READY, "horizons": ph,
                      "fields_not_ok": sorted({nm for v in ph.values() for nm in v["blocked_by"]})}

    absent = [f for f in fields if f["status"] == "missing"]
    allnan = [f for f in fields if f["status"] == "all_nan"]
    wrong = [f for f in fields if f["status"] == "wrong_shape"]
    if n == 0:
        block("rows", "no rows for this model", "send at least one window per model")
    if len(absent) == len(fields) and n:
        block("fields", f"NONE of the {len(fields)} required fields is present: this telemetry cannot "
                        "produce a forecast. The product would show nothing",
              "compute the required fields per window from your predictions (and features): see `template`")
    elif absent:
        block("fields", "required field(s) missing: " + ", ".join(
            f["field"] + (f" [{'+'.join(f['required_by_paths'])}]" if len(spec.by_path) > 1 else "")
            for f in absent))
    if absent and len(absent) != len(fields):
        fixes.append("add to every window: " + ", ".join(f"`{f['field']}`" for f in absent) + " (see `template`)")
    if allnan:
        block("fields", "required field(s) present but all-NaN/null: " + ", ".join(f["field"] for f in allnan))
        fixes += [_fix_text(f, f["field"].split(".")[0]) for f in allnan]
    if wrong:
        block("shape", "required field(s) have the wrong shape (ingest would reject): "
              + ", ".join(f["field"] for f in wrong))
        fixes += [_fix_text(f, f["field"].split(".")[0]) for f in wrong]
    for f in fields:
        if f["status"] == "out_of_range":
            warnings.append(f"{f['field']}: {f['windows_out_of_range']} of {f['windows_finite']} values outside "
                            f"{f['domain']} (units?)")
            fixes.append(_fix_text(f, f["field"].split(".")[0]))
        elif f["status"] == "constant":
            warnings.append(f"{f['field']}: constant over {f['windows']} windows")
        elif f["status"] == "ok" and f["tail_null_run"] >= 5:
            warnings.append(f"{f['field']}: null for the last {f['tail_null_run']} windows")
        if f["windows_nonfinite_literal"]:
            block("shape", f"{f['field']}: {f['windows_nonfinite_literal']} NaN/inf literal(s); ingest rejects "
                           "non-finite numbers (schema_invalid)",
                  f"send null (not NaN/inf) for `{f['field']}` when it cannot be computed")
        if f["windows_wrong_shape"] and f["status"] != "wrong_shape":
            block("shape", f"{f['field']}: {f['windows_wrong_shape']} window(s) with the wrong shape "
                           "(ingest rejects them)", _fix_text({**f, "status": "wrong_shape"}, f["field"].split(".")[0]))
    if absent or allnan or wrong:
        out_blocked = [f["field"] for f in absent + allnan + wrong if f["needs_outcomes"]]
        if out_blocked:
            fixes.append("fields needing realised outcomes (" + ", ".join(sorted(out_blocked)) + "): compute them "
                         "per window from outcome-bearing records; if outcomes arrive late or never, the bundle "
                         "that selects them cannot be served for this model: ask BlurredBox for a label-free bundle")
    if scorable and len(scorable) < len(horizons):
        warnings.append(f"only horizons {sorted(scorable)} of {sorted(horizons)} could be scored")

    # contract-level issues in the stored/sent z_t
    allowed = wire_fields()
    unknown: dict[str, int] = {}
    for z in z_rows:
        for k in z:
            if k not in allowed:
                unknown[k] = unknown.get(k, 0) + 1
    if unknown:
        block("unknown_fields", "z_t carries fields outside the signal contract (ingest rejects with "
                                f"schema_invalid): {sorted(unknown)[:8]}",
              "remove them from z_t; the schema is closed (no metadata/extra)")
    bad_sv = sorted({r.schema_version for r in rows if r.schema_version and not _version_ok(r.schema_version)})
    if bad_sv:
        block("schema_version", f"schema_version {bad_sv} is not accepted by this backend",
              "send schema_version 1.0-1.3")
    extra = sorted(set(extra_columns))
    extra_hints = {}
    if extra:
        for c in extra:
            hint = _HINTS.get(c.lower())
            if hint is None:
                m = difflib.get_close_matches(c, sorted(allowed), n=1, cutoff=0.75)
                hint = f"did you mean `{m[0]}`?" if m else "not part of the signal contract; it will not be used"
            extra_hints[c] = hint

    # timestamps / cadence
    cad = analyse_cadence([r.ts for r in rows], declared_minutes=declared_cadence_minutes)
    if n and cad["unparseable"]:
        block("timestamps", f"{cad['unparseable']} of {n} windows have no parseable timestamp",
              "send `window_timestamp` as an ISO-8601 time WITH timezone (ingest requires it)")
    if sum(1 for r in rows if r.ts_naive):
        warnings.append(f"{sum(1 for r in rows if r.ts_naive)} timestamp(s) carry no timezone; read as UTC. "
                        "Ingest requires a timezone-aware window_timestamp")
        fixes.append("send window_timestamp with an explicit offset (e.g. 2026-10-20T14:05:00Z)")
    cad_min = cad.get("measured_minutes") if cad.get("regular") else None
    if cad.get("duplicates"):
        block("timestamps", f"{cad['duplicates']} duplicate window timestamp(s): the runtime keeps one window "
                            "per (model, window_end)", "de-duplicate: one row per model per window close")
    if cad.get("out_of_order"):
        warnings.append(f"{cad['out_of_order']} row(s) out of time order (sorted for this analysis); the "
                        "runtime rejects windows that do not follow the last accepted one (out_of_order_window)")
        fixes.append("send windows in time order per model")
    if cad.get("measured_minutes") is not None and not cad["regular"]:
        warnings.append("irregular sampling: " + cad["reason"] + "; windows cannot be converted to wall-clock")
    if cad.get("measured_minutes") is not None and cad["regular"] and cad["measured_minutes"] > RUNTIME_CADENCE_MAX_MINUTES:
        block("cadence", f"measured cadence {cad['measured_minutes']:g} min exceeds the runtime limit of "
                         f"{RUNTIME_CADENCE_MAX_MINUTES:g} min (cadence_minutes <= 60)",
              "emit windows at most hourly (the default is every 5 minutes), computed from record-level data")
    if cad.get("gaps"):
        warnings.append(f"{cad['gaps']} gap(s) longer than {GAP_FACTOR:g}x the median interval "
                        f"(~{cad['missing_windows_est']} windows missing; largest "
                        f"{cad['max_gap_seconds'] / 60:.0f} min)")
    use_min = declared_cadence_minutes or cad_min

    # warm-up
    wr = warmup_requirements(alpha=alpha)
    complete = _complete_windows(z_rows, spec)
    warm = []
    for w in wr:
        warm.append({**w, "met": n >= w["windows"], "have": n, "have_complete": complete,
                     "wall_clock": _fmt_duration(w["windows"], use_min),
                     "short_by": max(0, w["windows"] - n)})
    cal = warm[0]
    if n and not cal["met"]:
        msg = (f"{n} windows is below the forecaster's calibration floor of {cal['windows']} "
               f"({cal['wall_clock'] or 'wall-clock unknown: cadence not measured'}): the model stays "
               "'uncalibrated' and cannot raise a calibrated alert")
        if allow_short_history:
            warnings.append(msg)
        else:
            block("history", msg, f"supply at least {cal['windows']} windows per model "
                                  "(or keep streaming until reached; rerun with --allow-short-history for a live check)")
    elif n and complete < cal["windows"]:
        warnings.append(f"only {complete} of {n} windows have every required field finite "
                        f"(floor {cal['windows']})")

    optional = _check_optional(rows)
    if optional["external_events"]["malformed"]:
        block("external_events", f"{optional['external_events']['malformed']} malformed external_events item(s): "
                                 "ingest rejects the window (schema_invalid)", "fix the event shape: see the external_events schema")
    if optional["outcome_labels"]["malformed"]:
        block("outcome_labels", f"{optional['outcome_labels']['malformed']} malformed outcome_labels item(s)",
              "each label is {window_index>=0, n_labelled>=n_errors>=0}")
    info = []
    if not optional["outcome_labels"]["windows_with"]:
        info.append("no outcome_labels: the performance monitor stays inert (optional)")
    if not optional["external_events"]["windows_with"]:
        info.append("no external_events: deploy/pipeline-health context is not available (optional)")

    verdict = READY if not blockers else NOT_READY
    return {
        "model": model_id, "verdict": verdict, "windows": n, "windows_complete": complete,
        "blockers": blockers, "warnings": warnings, "info": info,
        "fix_instructions": list(dict.fromkeys(fixes)),
        "fields": fields, "horizons": {str(h): v for h, v in horizons.items()},
        "horizons_scorable": sorted(scorable), "paths": paths,
        "cadence": cad, "warmup": warm, "optional": optional,
        "unknown_fields": unknown, "extra_columns": extra_hints,
        "required_source": spec.source,
    }


def _version_ok(raw: str) -> bool:
    try:
        major, minor = raw.split(".")
        return major.isdigit() and minor.isdigit() and 0 <= int(major) <= 1
    except ValueError:
        return False


def _row_finite(z: Mapping[str, Any], need: FieldNeed) -> bool:
    raw = z.get(need.wire_field)
    if need.wire_field in MAP_VALUED_FIELDS:
        if not isinstance(raw, dict):
            return False
        if need.subkey is not None:
            return _finite(raw.get(need.subkey)) is not None
        return any(_finite(y) is not None for y in raw.values())
    return _finite(raw) is not None


def _complete_windows(z_rows: Sequence[Mapping[str, Any]], spec: RequiredSpec) -> int:
    """Windows where every required field that does NOT need outcomes is finite (a label lag
    must not zero the count; label fields are judged through `fields`/`horizons`)."""
    needs = [x for x in spec.needs if not x.needs_outcomes]
    return sum(1 for z in z_rows if all(_row_finite(z, x) for x in needs))


# ── Files ───────────────────────────────────────────────────────────────────

def validate_files(paths: Sequence[str | Path], *, bundle: Any = None, alpha: float | None = None,
                   declared_cadence_minutes: float | None = None, allow_short_history: bool = False,
                   required: RequiredSpec | None = None, paired_bundle: Any = None) -> dict[str, Any]:
    """Validate telemetry file(s); one verdict per model, and an overall one. Default: the
    union of the live forecast's and the learned detector's fields (`paired_bundle` names the
    detector's own forecaster when `bundle` is given)."""
    spec = required or (required_from_bundle(bundle, paired_bundle) if bundle is not None
                        else default_required())
    merged: dict[str, list[WindowRow]] = {}
    extras: dict[str, set[str]] = {}
    files = []
    for p in paths:
        m, ex, info = load_windows(p)
        files.append(info)
        for k, v in m.items():
            merged.setdefault(k, []).extend(v)
        for k, v in ex.items():
            extras.setdefault(k, set()).update(v)
    models = [analyse_model(k, v, required=spec, alpha=alpha,
                            declared_cadence_minutes=declared_cadence_minutes,
                            allow_short_history=allow_short_history, extra_columns=extras.get(k, ()))
              for k, v in sorted(merged.items())]
    if any("bad_lines" in f and f["bad_lines"] for f in files):
        pass
    overall = READY if models and all(m["verdict"] == READY for m in models) else NOT_READY
    return {"verdict": overall, "models": models, "files": files, "required_source": spec.source,
            "required_fields": list(spec.names), "required_by_path": {k: list(v) for k, v in spec.by_path.items()},
            "outcome_fields": sorted(
                n.name for n in spec.needs if n.needs_outcomes),
            "horizons": {str(h): list(v) for h, v in spec.by_horizon.items()},
            "no_models": not models}


# ── Rendering ───────────────────────────────────────────────────────────────

def render_text(rep: dict[str, Any]) -> str:
    L: list[str] = []
    L.append(f"ONBOARDING VALIDATION: {rep['verdict']}")
    L.append(f"required fields derived from: {rep['required_source']}")
    for f in rep["files"]:
        L.append(f"file {f['path']}: {f['rows']} rows ({f['format']})"
                 + (f", {f['bad_lines']} unreadable line(s)" if f.get("bad_lines") else ""))
    if rep["no_models"]:
        L.append("NOT READY: no readable rows in the input")
    for m in rep["models"]:
        L.append("")
        L.append(f"model {m['model']}: {m['verdict']}   ({m['windows']} windows, "
                 f"{m['windows_complete']} complete)")
        c = m["cadence"]
        if c.get("measured_minutes") is not None:
            L.append(f"  cadence: {c['measured_minutes']:g} min median, "
                     f"{'regular' if c['regular'] else 'IRREGULAR'} ({c['reason']}); span {c.get('span_hours')} h; "
                     f"{c.get('gaps', 0)} gap(s), {c.get('duplicates', 0)} duplicate(s)")
        else:
            L.append(f"  cadence: not measurable ({c.get('reason', 'no timestamps')})")
        L.append("  required fields:")
        for f in m["fields"]:
            tag = "needs outcomes" if f["needs_outcomes"] else ""
            L.append(f"    {f['status'].upper():<12} {f['field']:<30} finite {f['windows_finite']}/{f['windows']}"
                     + (f"  [{tag}]" if tag else ""))
        L.append("  horizons: " + ", ".join(
            f"h{h}={'ok' if v['scorable'] else 'REFUSED'}" for h, v in m["horizons"].items()))
        for pth, pv in m.get("paths", {}).items():
            L.append(f"  path {pth}: {pv['verdict']}"
                     + (f"  (blocked by: {', '.join(pv['fields_not_ok'])})" if pv["fields_not_ok"] else ""))
        L.append("  warm-up (windows needed vs available):")
        for w in m["warmup"]:
            L.append(f"    {'met   ' if w['met'] else 'SHORT '} {w['component']:<34} {w['windows']:>6}"
                     f"  ({w['wall_clock'] or 'wall-clock n/a'})"
                     + ("  [required]" if w["required"] else ""))
        o = m["optional"]
        L.append(f"  optional: external_events in {o['external_events']['windows_with']} windows "
                 f"({o['external_events']['events']} events); outcome_labels in "
                 f"{o['outcome_labels']['windows_with']} windows ({o['outcome_labels']['labels']} labels)")
        if m["extra_columns"]:
            L.append("  columns not in the signal contract (ignored): " + ", ".join(
                f"{k} ({v})" for k, v in m["extra_columns"].items()))
        for b in m["blockers"]:
            L.append(f"  BLOCKER [{b['kind']}]: {b['message']}")
        for w in m["warnings"]:
            L.append(f"  warning: {w}")
        for i in m["info"]:
            L.append(f"  note: {i}")
        if m["fix_instructions"]:
            L.append("  HOW TO FIX:")
            for i, f in enumerate(m["fix_instructions"], 1):
                L.append(f"    {i}. {f}")
    return "\n".join(L) + "\n"


TEMPLATE_NOTES = {
    "score_mean": "mean of the window's per-record scores (corpus.signal_vector.score_moments)",
    "score_std": "population std of the window's scores (score_moments)",
    "score_kurtosis": "sample-corrected EXCESS kurtosis of the scores (score_moments)",
    "score_quantiles": "dict with keys p5, p95, p99 (also p1, p50): percentile_linear over the scores",
    "positive_decision_rate": "fraction of records decided positive at YOUR decision threshold",
    "predictive_entropy": "mean binary predictive entropy (bits) over the window's records",
    "row_count_vs_expected": "records in the window / records expected per window",
    "wasserstein1": "per-FEATURE map {feature: earth-mover distance vs the reference window}",
    "cardinality_change": "per-FEATURE map {feature: categorical cardinality change vs reference} (needs features)",
    "covariance_delta": "Frobenius norm of (reference - current) feature covariance (corpus.multivariate.covariance_delta)",
    "domain_classifier_drift": "reference-vs-current classifier drift score on features "
                               "(corpus.multivariate.domain_classifier(...).drift_score)",
    "hstrees_score": "mean Half-Space-Trees anomaly score of the window's feature vectors (corpus.anomaly.HalfSpaceTrees)",
    "psi": "population stability index per FEATURE {feature: psi vs reference} (corpus.signal_vector)",
    "chi_square": "per-FEATURE map {feature: chi-square statistic vs reference} (categorical features)",
    "hellinger": "per-FEATURE map {feature: Hellinger distance vs reference}",
    "ks_statistic": "per-FEATURE map {feature: two-sample KS statistic vs reference}",
    "mahalanobis_distance": "mean Mahalanobis distance of the window's feature vectors from the reference (corpus.multivariate)",
    "novelty_score": "mean novelty score of the window's feature vectors (corpus.anomaly)",
    "pca_reconstruction_error": "mean PCA reconstruction error vs the reference subspace (corpus.multivariate)",
    "top_eigenvalue_ratio": "top-eigenvalue ratio of the window covariance vs reference (corpus.multivariate)",
    "brier": "Brier score over outcome-bearing records (needs realised outcomes)",
    "nll": "negative log-likelihood over outcome-bearing records (needs realised outcomes)",
    "ci_width": "95% CI width of the realised accuracy at the window's labelled n (needs realised outcomes)",
}


def template(required: RequiredSpec | None = None) -> dict[str, Any]:
    spec = required or default_required()
    cols = []
    for n in spec.needs:
        cols.append({"field": n.name, "wire_field": n.wire_field, "needs_outcomes": n.needs_outcomes,
                     "map_valued": n.wire_field in MAP_VALUED_FIELDS,
                     "horizons": list(n.horizons), "paths": list(n.paths), "how": TEMPLATE_NOTES.get(n.wire_field, "")})
    example = {"schema_version": "1.3", "tenant_id": "<yours>", "install_id": "<yours>",
               "monitored_model_id": "model-a", "window_timestamp": "2026-10-20T14:05:00Z",
               "z_t": {"score_mean": 0.41, "score_std": 0.19, "score_kurtosis": -0.3,
                       "score_quantiles": {"p5": 0.1, "p50": 0.4, "p95": 0.8, "p99": 0.93},
                       "positive_decision_rate": 0.22, "predictive_entropy": 0.62,
                       "row_count_vs_expected": 0.98,
                       "wasserstein1": {"feature_a": 0.03}, "cardinality_change": {"feature_a": 0.0},
                       "covariance_delta": 0.12, "domain_classifier_drift": 0.04, "hstrees_score": 0.51,
                       "psi": {"feature_a": 0.02}, "chi_square": {"feature_a": 0.01},
                       "hellinger": {"feature_a": 0.03}, "ks_statistic": {"feature_a": 0.04},
                       "mahalanobis_distance": 1.4, "novelty_score": 0.2,
                       "pca_reconstruction_error": 0.15, "top_eigenvalue_ratio": 0.31,
                       "brier": 0.17, "nll": 0.5, "ci_width": 0.06}}
    return {"source": spec.source, "columns": cols,
            "csv_header": ["window_timestamp", "monitored_model_id"] + [
                n.wire_field if not n.subkey else f"score_quantile_{n.subkey}" for n in spec.needs
                if n.wire_field not in MAP_VALUED_FIELDS] + [
                n.wire_field for n in spec.needs if n.wire_field in MAP_VALUED_FIELDS],
            "example_envelope": example}


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m corpus.onboarding",
                                 description="Local onboarding validator. Reads files, prints a verdict; "
                                             "nothing leaves this machine.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="validate telemetry file(s)")
    v.add_argument("files", nargs="+")
    v.add_argument("--json", action="store_true", help="machine-readable report on stdout")
    v.add_argument("--bundle", help="DIR with forecaster.pkl to derive the required fields from "
                                    "(unpickles: trusted artifacts only). Default: the built-in reference bundle")
    v.add_argument("--paired-bundle", help="DIR with the learned detector's own forecaster.pkl; with "
                                           "--bundle, the learned-detector path is checked too")
    v.add_argument("--alpha", type=float, help="false-alarm level (default: the runtime's DEFAULT_ALPHA)")
    v.add_argument("--cadence-minutes", type=float, help="declared cadence (default: measured)")
    v.add_argument("--allow-short-history", action="store_true",
                   help="treat 'fewer windows than the calibration floor' as a warning")
    t = sub.add_parser("template", help="print the required-field guide and an example envelope")
    t.add_argument("--bundle")
    t.add_argument("--paired-bundle")
    a = ap.parse_args(argv)
    if a.cmd == "template":
        spec = required_from_bundle(a.bundle, a.paired_bundle) if a.bundle else default_required()
        print(json.dumps(template(spec), indent=2))
        return 0
    try:
        rep = validate_files(a.files, bundle=a.bundle, paired_bundle=a.paired_bundle, alpha=a.alpha,
                             declared_cadence_minutes=a.cadence_minutes,
                             allow_short_history=a.allow_short_history)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(json.dumps(rep, indent=2, default=str) + "\n" if a.json else render_text(rep))
    return 0 if rep["verdict"] == READY else 2


if __name__ == "__main__":
    raise SystemExit(main())
