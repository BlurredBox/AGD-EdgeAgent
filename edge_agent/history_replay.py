"""Partner history replay, partner side (G53): `validate` and `send`, packaged in the edge-agent wheel.

    blurredbox-edge-replay validate --mapping replay.yaml --out DIR
    blurredbox-edge-replay send     --mapping replay.yaml --state-file replay-state.json --out DIR
                                    --base-url https://<sandbox> [--client-cert C K] [--ca-file F]

(also `blurredbox-edge replay ...` on a host set up with the edge installer, and
`python -m edge_agent.history_replay ...`).

`validate` checks a history export against its mapping file (schema, time order, duplicates,
arrival-vs-event clock, backfill flag, incident times inside the data, operational feed) and writes
DIR/validation.json: counts only, no cell value. Local files only; sends nothing. Exit 2 on a blocking error.

`send` validates, stages a sorted copy under DIR/stage, and drives the REAL edge agent (`EdgeAgent`, the
folder source, window cutter, z_t computation, wire audit, durable spool and signed transport; arrival
stamps from the source only) over the history at accelerated speed, one fresh edge state per model under
DIR/work. What leaves the host is what a live edge sends: one `/api/v1/ingest/signal` envelope per closed
window, assembled and audited by `edge_agent/wire.py` (counts, lags and window statistics; feature names
replaced by ids). The incident marks (`POST /api/v1/incidents`: model id, time range, failure class) are
sent ONLY with `--post-incident-marks`; by default the incident history stays with the files.

This is the one implementation: `scripts/partner_history_replay.py` (onboard / report / run, on our side)
imports it. Runbook: docs/runbooks/PARTNER_HISTORY_REPLAY.md.

The mapping file (YAML): `cadence_seconds`, optional `edge` (EdgeConfig overrides, allow-listed),
`leadsignals_limits` (checked in full on our side at onboarding; here only its shape), and `predictions` /
`incidents` / `ops_feed` blocks, each with a `path` (CSV or Parquet, relative to the mapping file) and
`columns` (+ optional `constants`). The `predictions` block is the edge agent's own `columns` /
`constants` format (edge_agent/mapping.py) plus an optional `backfill` column.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
import yaml

from edge_agent.mapping import ColumnMapping, ConfigError
from edge_agent.upstream import KIND_STATUSES

K_LABEL_LAG = 3                     # net lead = gross - k (signed protocol's label-lag allowance)
LATE_LABEL_HOURS = 24.0             # an incident reported later than this after it started is a "late label"
ARRIVAL_SKEW_TOLERANCE_S = 60.0     # arrival earlier than event by more than this = clock problem
BACKFILL_NOT_ASSESSABLE = 0.5       # a model with a larger backfilled share: freshness "not assessable"
MIN_WEEKS_SCORABLE = 8              # prereg section 9: models with fewer weeks are not scored
EDGE_OVERRIDES = {"min_rows_per_window", "reference_windows", "reference_max_rows", "max_rows_per_window",
                  "lateness_seconds", "output_kind", "null_rate_ceiling", "slice_feature", "slice_min_rows",
                  "slice_shift_limit", "slice_max", "seed"}
INCIDENT_FIELDS = ("incident_id", "model_id", "failed_from", "failed_to", "reported_at", "failure_class")
OPS_FIELDS = ("event_id", "occurred_at", "arrived_at", "backfill", "kind", "status", "subject", "model_id")
_TRUE = {"true", "t", "1", "yes", "y"}
_FALSE = {"false", "f", "0", "no", "n", ""}
_MODEL_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


# ---------------------------------------------------------------------------------------------------
# mapping
# ---------------------------------------------------------------------------------------------------
@dataclass
class Block:
    path: Path
    columns: dict[str, Any]
    constants: dict[str, str] = field(default_factory=dict)
    backfill: str | None = None


@dataclass
class ReplaySpec:
    base: Path
    cadence_seconds: int
    edge: dict[str, Any]
    limits: dict[str, dict[str, float]]
    predictions: Block
    incidents: Block | None
    ops_feed: Block | None


def load_spec(path: str | Path, *, normalize_limits: Callable[[Any], dict[str, dict[str, float]]] | None = None
              ) -> ReplaySpec:
    """Read and check the mapping file. Raises ConfigError naming the key. `normalize_limits`: the
    server's full check of `leadsignals_limits` (scripts/partner_history_replay.py passes
    src.api.leadsignals_shadow.normalize_limits); the edge package checks only the shape."""
    normalize_limits = normalize_limits or _limits_shape
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError("mapping file must be a YAML object")
    extra = sorted(set(raw) - {"cadence_seconds", "edge", "leadsignals_limits", "predictions", "incidents",
                               "ops_feed"})
    if extra:
        raise ConfigError(f"unknown top-level key(s) {extra}")
    cad = raw.get("cadence_seconds")
    if type(cad) is not int or not 60 <= cad <= 86400:
        raise ConfigError("cadence_seconds must be an integer between 60 and 86400")
    edge = raw.get("edge") or {}
    if not isinstance(edge, dict) or set(edge) - EDGE_OVERRIDES:
        raise ConfigError(f"edge: only {sorted(EDGE_OVERRIDES)} may be overridden")
    limits: dict[str, dict[str, float]] = {}
    if raw.get("leadsignals_limits") is not None:
        try:
            limits = normalize_limits(raw["leadsignals_limits"])
        except ConfigError:
            raise
        except ValueError as exc:                    # the server's InvalidLimits is a ValueError
            raise ConfigError(f"leadsignals_limits: {exc}") from None
    base = path.resolve().parent
    preds = _block(raw.get("predictions"), "predictions", base, required=True)
    ColumnMapping(preds.columns, preds.constants or None)      # the edge agent's own checks
    inc = _block(raw.get("incidents"), "incidents", base)
    if inc is not None:
        bad = sorted(set(inc.columns) - set(INCIDENT_FIELDS))
        if bad:
            raise ConfigError(f"incidents.columns: unknown field(s) {bad}; allowed {list(INCIDENT_FIELDS)}")
        for need in ("failed_from", "failed_to"):
            if need not in inc.columns:
                raise ConfigError(f"incidents.columns.{need} is required")
        if "model_id" not in inc.columns and "model_id" not in inc.constants:
            raise ConfigError("incidents: map columns.model_id or set constants.model_id")
    ops = _block(raw.get("ops_feed"), "ops_feed", base)
    if ops is not None:
        bad = sorted(set(ops.columns) - set(OPS_FIELDS))
        if bad:
            raise ConfigError(f"ops_feed.columns: unknown field(s) {bad}; allowed {list(OPS_FIELDS)}")
        for need in ("occurred_at", "kind", "status"):
            if need not in ops.columns:
                raise ConfigError(f"ops_feed.columns.{need} is required")
    return ReplaySpec(base, cad, dict(edge), limits, preds, inc, ops)


def _block(raw: Any, where: str, base: Path, *, required: bool = False) -> Block | None:
    if raw is None:
        if required:
            raise ConfigError(f"{where} is required")
        return None
    if not isinstance(raw, dict) or set(raw) - {"path", "columns", "constants", "backfill"}:
        raise ConfigError(f"{where} must be an object with path, columns and optional constants / backfill")
    p = raw.get("path")
    if not isinstance(p, str) or not p:
        raise ConfigError(f"{where}.path is required")
    cols = raw.get("columns")
    if not isinstance(cols, dict) or not cols:
        raise ConfigError(f"{where}.columns is required")
    consts = raw.get("constants") or {}
    if not isinstance(consts, dict) or not all(isinstance(v, str) and v for v in consts.values()):
        raise ConfigError(f"{where}.constants must map fields to nonempty strings")
    bf = raw.get("backfill")
    if bf is not None and (not isinstance(bf, str) or not bf):
        raise ConfigError(f"{where}.backfill must be a column name")
    pp = Path(p)
    return Block(pp if pp.is_absolute() else base / pp, dict(cols), dict(consts), bf)


# ---------------------------------------------------------------------------------------------------
# loading and parsing
# ---------------------------------------------------------------------------------------------------
def read_table(path: Path) -> pd.DataFrame:
    """CSV (all columns as text, as the edge's folder source reads them) or Parquet."""
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def to_utc(s: pd.Series) -> pd.Series:
    """Event / arrival times as aware UTC; epoch seconds or ISO-8601; naive is read as UTC (as the edge
    does). Unparseable or empty -> NaT."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, utc=True)
    txt = s.astype("string").str.strip()
    num = pd.to_numeric(txt, errors="coerce")
    out = pd.to_datetime(txt.where(num.isna()), utc=True, errors="coerce", format="ISO8601")
    epoch = pd.to_datetime(num, unit="s", utc=True, errors="coerce")
    return out.where(num.isna(), epoch)


def to_flag(s: pd.Series) -> tuple[pd.Series, int]:
    """Backfill flag -> bool; returns (flags, count of values that are not a recognisable boolean)."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool), 0
    txt = s.astype("string").fillna("").str.strip().str.lower()
    good = txt.isin(_TRUE | _FALSE)
    return txt.isin(_TRUE), int((~good).sum())


def _col(df: pd.DataFrame, block: Block, name: str) -> pd.Series | None:
    if name in block.columns:
        return df[block.columns[name]]
    if name in block.constants:
        return pd.Series([block.constants[name]] * len(df), index=df.index, dtype="string")
    return None


def _feature_cols(df: pd.DataFrame, block: Block) -> list[str]:
    f = block.columns.get("features")
    if isinstance(f, list):
        return list(f)
    if isinstance(f, dict) and "prefix" in f:
        return [c for c in df.columns if c.startswith(f["prefix"]) and len(c) > len(f["prefix"])]
    if isinstance(f, dict) and "json_column" in f:
        return [f["json_column"]]
    used = {v for k, v in block.columns.items() if k != "features"} | ({block.backfill} if block.backfill else set())
    return [c for c in df.columns if c not in used]


@dataclass
class Frames:
    predictions: pd.DataFrame          # + _ts, _arr, _bf, _model (parsed)
    incidents: pd.DataFrame | None     # + _model, _from, _to, _reported, _class, _id
    ops: pd.DataFrame | None           # + _at, _arr, _bf, _model, _kind, _status, _subject, _id
    errors: list[str] = field(default_factory=list)


def load_frames(spec: ReplaySpec, *, predictions: bool = True) -> Frames:
    """predictions=False: only the incident history and the ops feed (the report on the receiving side,
    where the prediction log never is)."""
    errors: list[str] = []
    if not predictions:
        p = pd.DataFrame({"_ts": pd.Series([], dtype="datetime64[ns, UTC]"), "_model": pd.Series([], dtype="string")})
        return Frames(p, *_load_side(spec, errors), errors)
    p = read_table(spec.predictions.path)
    needed = [v for k, v in spec.predictions.columns.items() if k != "features"]
    if isinstance(spec.predictions.columns.get("features"), list):
        needed += spec.predictions.columns["features"]
    if spec.predictions.backfill:
        needed.append(spec.predictions.backfill)
    missing = sorted(set(needed) - set(p.columns))
    if missing:
        errors.append(f"predictions: mapped column(s) missing from the file: {missing}")
        return Frames(p, None, None, errors)
    p = p.copy()
    p["_ts"] = to_utc(p[spec.predictions.columns["ts"]])
    p["_arr"] = (to_utc(p[spec.predictions.columns["arrived_at"]]) if "arrived_at" in spec.predictions.columns
                 else pd.Series(pd.NaT, index=p.index, dtype="datetime64[ns, UTC]"))
    if spec.predictions.backfill:
        p["_bf"], p.attrs["bad_backfill"] = to_flag(p[spec.predictions.backfill])
    else:
        p["_bf"], p.attrs["bad_backfill"] = False, 0
    p["_model"] = _col(p, spec.predictions, "model_id").astype("string")
    return Frames(p, *_load_side(spec, errors), errors)


def _load_side(spec: ReplaySpec, errors: list[str]) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """The incident history and the operational feed, parsed."""
    inc = None
    if spec.incidents is not None:
        inc = read_table(spec.incidents.path).copy()
        miss = sorted(set(spec.incidents.columns.values()) - set(inc.columns))
        if miss:
            errors.append(f"incidents: mapped column(s) missing from the file: {miss}")
            inc = None
        else:
            inc["_model"] = _col(inc, spec.incidents, "model_id").astype("string")
            inc["_from"] = to_utc(inc[spec.incidents.columns["failed_from"]])
            inc["_to"] = to_utc(inc[spec.incidents.columns["failed_to"]])
            r = _col(inc, spec.incidents, "reported_at")
            inc["_reported"] = to_utc(r) if r is not None else pd.NaT
            c = _col(inc, spec.incidents, "failure_class")
            inc["_class"] = c.astype("string") if c is not None else None
            i = _col(inc, spec.incidents, "incident_id")
            inc["_id"] = i.astype("string") if i is not None else [f"incident-{k + 1}" for k in range(len(inc))]

    ops = None
    if spec.ops_feed is not None:
        ops = read_table(spec.ops_feed.path).copy()
        miss = sorted(set(spec.ops_feed.columns.values()) - set(ops.columns))
        if miss:
            errors.append(f"ops_feed: mapped column(s) missing from the file: {miss}")
            ops = None
        else:
            cols = spec.ops_feed.columns
            ops["_at"] = to_utc(ops[cols["occurred_at"]])
            ops["_arr"] = to_utc(ops[cols["arrived_at"]]) if "arrived_at" in cols else pd.NaT
            if "backfill" in cols:
                ops["_bf"], ops.attrs["bad_backfill"] = to_flag(ops[cols["backfill"]])
            else:
                ops["_bf"], ops.attrs["bad_backfill"] = False, 0
            for k in ("kind", "status", "subject", "model_id", "event_id"):
                s = _col(ops, spec.ops_feed, k)
                ops["_" + ("model" if k == "model_id" else "id" if k == "event_id" else k)] = (
                    s.astype("string") if s is not None else None)
    return inc, ops


# ---------------------------------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------------------------------
def _q(s: pd.Series) -> dict[str, float] | None:
    s = s.dropna()
    if s.empty:
        return None
    return {"p50": float(s.quantile(0.5)), "p95": float(s.quantile(0.95)), "max": float(s.max())}


def validate(spec: ReplaySpec, fr: Frames) -> dict[str, Any]:
    """Counts only (no cell value is ever printed). `ok` is False when a blocking error was found."""
    errors, warnings = list(fr.errors), []
    out: dict[str, Any] = {"label": "validation of a history export; counts only", "errors": errors,
                           "warnings": warnings, "cadence_seconds": spec.cadence_seconds}
    p = fr.predictions
    if errors and "_ts" not in p:
        out["ok"] = False
        return out
    n = len(p)
    pv: dict[str, Any] = {"rows": n}
    out["predictions"] = pv
    if n == 0:
        errors.append("predictions: the file has no rows")
        out["ok"] = False
        return out
    bad_ts = int(p["_ts"].isna().sum())
    pv["rows_unparseable_time"] = bad_ts
    if bad_ts == n:
        errors.append("predictions: no row has a readable event time (ts)")
    elif bad_ts:
        warnings.append(f"predictions: {bad_ts} row(s) with an unreadable event time are dropped")
    bad_model = int(p["_model"].isna().sum() + (p["_model"].fillna("") == "").sum())
    pv["rows_without_model"] = bad_model
    if bad_model:
        warnings.append(f"predictions: {bad_model} row(s) without a model id are dropped")
    out_order = int((p["_ts"].diff().dt.total_seconds() < 0).sum())
    pv["rows_out_of_time_order_in_file"] = out_order
    if out_order:
        warnings.append(f"predictions: {out_order} row(s) earlier than the row before them in the file; "
                        "the replay sorts by event time")
    mapped = [v for k, v in spec.predictions.columns.items() if k != "features"] + _feature_cols(p, spec.predictions)
    dup_rows = int(p.duplicated(subset=list(dict.fromkeys(mapped))).sum())
    pv["exact_duplicate_rows"] = dup_rows
    if dup_rows:
        warnings.append(f"predictions: {dup_rows} exact duplicate row(s) are dropped before replay")
    if "prediction_id" in spec.predictions.columns:
        ids = p[[spec.predictions.columns["prediction_id"]]].assign(_m=p["_model"])
        dup_ids = int(ids.duplicated().sum()) - dup_rows
        pv["duplicate_prediction_ids_not_exact_rows"] = max(0, dup_ids)
        if dup_ids > 0:
            warnings.append(f"predictions: {dup_ids} prediction id(s) repeated with different content "
                            "(re-scored or re-sent rows?); kept, each counts in its window")
    # arrival vs event clock
    has_arr = "arrived_at" in spec.predictions.columns
    pv["arrival_column"] = has_arr
    if has_arr:
        lag = (p["_arr"] - p["_ts"]).dt.total_seconds()
        stamped = lag.notna()
        pv["rows_with_arrival"] = int(stamped.sum())
        neg = int((lag < -ARRIVAL_SKEW_TOLERANCE_S).sum())
        pv["rows_arrival_before_event"] = neg
        if neg:
            warnings.append(f"predictions: {neg} row(s) arrive more than {ARRIVAL_SKEW_TOLERANCE_S:.0f} s before "
                            "their event time (clock skew or a wrong arrival column); their lag counts as 0")
        pv["arrival_lag_s_not_backfilled"] = _q(lag[~p["_bf"]])
    else:
        warnings.append("predictions: no arrival column; arrival-lag indicators are not available (replay "
                        "never invents an arrival time)")
    # backfill flag
    pv["backfill_column"] = bool(spec.predictions.backfill)
    bad_bf = int(p.attrs.get("bad_backfill", 0))
    pv["backfill_values_unrecognised"] = bad_bf
    if bad_bf:
        errors.append(f"predictions: {bad_bf} backfill flag value(s) are not true/false/1/0/yes/no")
    if has_arr and not spec.predictions.backfill:
        warnings.append("predictions: arrival column without a backfill flag; backfilled rows would count "
                        "as late arrivals, so arrival lag is reported 'not assessable' (prereg 9(b))")
    # per model
    models: dict[str, Any] = {}
    cad = spec.cadence_seconds
    good = p[p["_ts"].notna() & p["_model"].notna()]
    for m, g in good.groupby("_model"):
        ts = g["_ts"]
        span_days = (ts.max() - ts.min()).total_seconds() / 86400
        buckets = ((ts - pd.Timestamp(0, tz="UTC")).dt.total_seconds() // cad).astype("int64")
        per = buckets.value_counts()
        n_buckets = int(buckets.max() - buckets.min() + 1)
        min_rows = int(spec.edge.get("min_rows_per_window", 30))
        bf_share = float(g["_bf"].mean()) if spec.predictions.backfill else None
        assess = has_arr and bool(spec.predictions.backfill) and (bf_share or 0) <= BACKFILL_NOT_ASSESSABLE
        models[str(m)] = {
            "rows": int(len(g)), "first": ts.min().isoformat(), "last": ts.max().isoformat(),
            "span_days": round(span_days, 2), "windows_in_span": n_buckets,
            "windows_with_rows": int(len(per)), "windows_without_rows": n_buckets - int(len(per)),
            "windows_below_min_rows": int((per < min_rows).sum()), "rows_per_window_median": float(per.median()),
            "backfilled_share": None if bf_share is None else round(bf_share, 4),
            "arrival_lag_assessable": bool(assess),
            "scorable_span": span_days >= 7 * MIN_WEEKS_SCORABLE}
        if span_days < 7 * MIN_WEEKS_SCORABLE:
            warnings.append(f"model {_safe(m)}: {span_days:.1f} days of history; fewer than "
                            f"{MIN_WEEKS_SCORABLE} weeks is not scored under prereg section 9 (replay still runs)")
        if (per < min_rows).sum():
            warnings.append(f"model {_safe(m)}: {int((per < min_rows).sum())} window(s) below "
                            f"min_rows_per_window={min_rows} send nothing (edge-health gaps)")
        if not _MODEL_RE.match(str(m)):
            errors.append("predictions: a model id has characters outside [A-Za-z0-9._:-] or is too long")
    pv["models"] = models
    out["incidents"] = _validate_incidents(fr, models, warnings, errors)
    out["ops_feed"] = _validate_ops(fr, models, warnings, errors)
    out["ok"] = not errors
    return out


def _safe(m: Any) -> str:
    s = str(m)
    return s if _MODEL_RE.match(s) else "<model>"


def _validate_incidents(fr: Frames, models: dict[str, Any], warnings: list[str], errors: list[str]) -> Any:
    inc = fr.incidents
    if inc is None:
        warnings.append("no incident history mapped: the report has no incidents to compare against")
        return None
    rows = []
    for _, r in inc.iterrows():
        problems = []
        m = r["_model"]
        if pd.isna(r["_from"]) or pd.isna(r["_to"]):
            problems.append("unreadable failed_from / failed_to")
        elif r["_to"] < r["_from"]:
            problems.append("failed_to before failed_from")
        if pd.isna(m) or str(m) not in models:
            problems.append("model not in the prediction log")
        elif not pd.isna(r["_from"]):
            first, last = pd.Timestamp(models[str(m)]["first"]), pd.Timestamp(models[str(m)]["last"])
            if not first <= r["_from"] <= last:
                problems.append("failed_from outside the model's prediction history")
        if r["_class"] is not None and not pd.isna(r["_class"]) and len(str(r["_class"])) > 64:
            problems.append("failure_class longer than 64 characters")
        delay_h = None
        if not pd.isna(r["_reported"]) and not pd.isna(r["_from"]):
            delay_h = (r["_reported"] - r["_from"]).total_seconds() / 3600
            if delay_h < 0:
                warnings.append(f"incident {_safe(r['_id'])}: reported before it started")
        rows.append({"incident_id": _safe(r["_id"]), "model": _safe(m), "usable": not problems,
                     "problems": problems, "label_delay_hours": None if delay_h is None else round(delay_h, 2),
                     "late_label": delay_h is not None and delay_h > LATE_LABEL_HOURS})
        for pr in problems:
            warnings.append(f"incident {_safe(r['_id'])}: {pr}; excluded from the report")
    dups = int(inc.duplicated(subset=["_model", "_from"]).sum())
    if dups:
        warnings.append(f"incidents: {dups} duplicate (model, failed_from) pair(s)")
    late = [x["incident_id"] for x in rows if x["late_label"]]
    if late:
        warnings.append(f"incidents: {len(late)} late label(s) (reported more than {LATE_LABEL_HOURS:.0f} h after "
                        "the start); listed in the report, B stays the marked start")
    usable = sum(x["usable"] for x in rows)
    if rows and not usable:
        errors.append("incidents: no incident is usable (all outside the data or malformed)")
    return {"rows": len(rows), "usable": usable, "duplicates": dups, "late_labels": late, "incidents": rows}


def _validate_ops(fr: Frames, models: dict[str, Any], warnings: list[str], errors: list[str]) -> Any:
    ops = fr.ops
    if ops is None:
        return None
    n = len(ops)
    bad_t = int(ops["_at"].isna().sum())
    kind_ok = ops.apply(lambda r: (not pd.isna(r["_kind"]) and r["_kind"] in KIND_STATUSES
                                   and r["_status"] in KIND_STATUSES[r["_kind"]]), axis=1) if n else pd.Series([], dtype=bool)
    bad_kind = int((~kind_ok).sum()) if n else 0
    out_order = int((ops["_at"].diff().dt.total_seconds() < 0).sum())
    lag = (ops["_arr"] - ops["_at"]).dt.total_seconds() if "_arr" in ops and ops["_arr"].notna().any() else None
    neg = int((lag < -ARRIVAL_SKEW_TOLERANCE_S).sum()) if lag is not None else 0
    dup = int(ops["_id"].dropna().duplicated().sum()) if ops["_id"] is not None else 0
    bad_bf = int(ops.attrs.get("bad_backfill", 0))
    if bad_t:
        warnings.append(f"ops_feed: {bad_t} event(s) with an unreadable time are ignored")
    if bad_kind:
        warnings.append(f"ops_feed: {bad_kind} event(s) with a kind/status outside {sorted(KIND_STATUSES)} are ignored")
    if neg:
        warnings.append(f"ops_feed: {neg} event(s) received before they occurred")
    if dup:
        warnings.append(f"ops_feed: {dup} repeated event id(s)")
    if bad_bf:
        errors.append(f"ops_feed: {bad_bf} backfill flag value(s) are not true/false/1/0/yes/no")
    firsts = [pd.Timestamp(m["first"]) for m in models.values()]
    lasts = [pd.Timestamp(m["last"]) for m in models.values()]
    outside = int(((ops["_at"] < min(firsts)) | (ops["_at"] > max(lasts))).sum()) if firsts else 0
    return {"events": n, "unreadable_time": bad_t, "unknown_kind_or_status": bad_kind,
            "out_of_time_order_in_file": out_order, "received_before_occurred": neg, "duplicate_ids": dup,
            "backfilled": int(ops["_bf"].sum()), "outside_prediction_span": outside,
            "arrival_lag_s_not_backfilled": None if lag is None else _q(lag[~ops["_bf"]])}


# ---------------------------------------------------------------------------------------------------
# staging (on the replay host; never sent)
# ---------------------------------------------------------------------------------------------------
def stage_predictions(spec: ReplaySpec, fr: Frames, stage_dir: str | Path) -> dict[str, Any]:
    """Sorted by event time, exact duplicates and unreadable rows dropped, the arrival stamp BLANKED on
    backfilled rows (a re-sent row's arrival time is not when the data arrived), the backfill column
    removed (it must not become a feature). Event and arrival times are written as ISO-8601 UTC."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    for old in stage_dir.glob("*.csv"):
        old.unlink()
    p = fr.predictions
    mapped = [v for k, v in spec.predictions.columns.items() if k != "features"] + _feature_cols(p, spec.predictions)
    keep = p[p["_ts"].notna() & p["_model"].notna() & (p["_model"].fillna("") != "")]
    keep = keep[~keep.duplicated(subset=list(dict.fromkeys(mapped)))]
    keep = keep.sort_values("_ts", kind="stable")
    cols = spec.predictions.columns
    out = keep.drop(columns=[c for c in keep.columns if c.startswith("_")]).copy()
    if spec.predictions.backfill and spec.predictions.backfill in out.columns:
        out = out.drop(columns=[spec.predictions.backfill])
    out[cols["ts"]] = keep["_ts"].map(lambda t: t.isoformat())
    if "arrived_at" in cols:
        arr = keep["_arr"].where(~keep["_bf"].astype(bool))
        out[cols["arrived_at"]] = arr.map(lambda t: "" if pd.isna(t) else t.isoformat())
    path = stage_dir / "history.csv"
    tmp = stage_dir / ".history.csv.tmp"          # the folder source ignores dot-files and *.tmp
    out.to_csv(tmp, index=False, na_rep="")
    os.replace(tmp, path)
    return {"path": str(path), "rows": int(len(out)), "dropped": int(len(p) - len(out)),
            "arrival_blanked_backfill": int(keep["_bf"].astype(bool).sum()) if "arrived_at" in cols else 0,
            "models": sorted(str(m) for m in keep["_model"].unique()),
            "first": keep["_ts"].min().isoformat(), "last": keep["_ts"].max().isoformat()}


# ---------------------------------------------------------------------------------------------------
# replay through the real edge agent
# ---------------------------------------------------------------------------------------------------
class _ReplayClock:
    """The agent's clock during a replay: the event time of the newest row handed to it. Windows close
    as the history moves past them, exactly as a live agent closes them by the wall clock."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


class _Tracking:
    """Wraps the real folder source; advances the replay clock to the newest event time read."""

    def __init__(self, inner: Any, clock: _ReplayClock) -> None:
        self._inner, self._clock = inner, clock

    def poll(self, *a: Any, **k: Any):
        recs = self._inner.poll(*a, **k)
        if recs:
            self._clock.now = max(self._clock.now, max(r.ts for r in recs))
        return recs

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


REPLAY_DROPS = ("prediction_id", "outcome", "outcome_observed_at")


def replay_columns(spec: ReplaySpec) -> dict[str, Any]:
    """The prediction mapping the edge source gets. The replay joins no outcomes (the incident history is
    the label), so prediction ids and inline outcomes are not read: an id makes the edge's outcome join
    write every row to its local sqlite store (tens of rows per second, far too slow for months of
    history), and a historical label "known" after the replay clock would be refused row by row.
    Validation still checks the ids for duplicates."""
    return {k: v for k, v in spec.predictions.columns.items() if k not in REPLAY_DROPS}


def edge_config(spec: ReplaySpec, *, tenant_id: str, install_id: str, models: Mapping[str, str],
                secret: str, endpoints: list[str], state_dir: Path) -> Any:
    from edge_agent.config import EdgeConfig
    return EdgeConfig(tenant_id=tenant_id, install_id=install_id, models=dict(models), secret=secret,
                      endpoints=list(endpoints), state_dir=state_dir, cadence_seconds=spec.cadence_seconds,
                      arrival_stamp="source_only",               # never a replay-time arrival stamp
                      spool_max_age_seconds=20 * 366 * 86400.0,   # history is old by definition
                      **spec.edge)


def replay(spec: ReplaySpec, staged: Mapping[str, Any], *, tenant_id: str, install_id: str,
           models: Mapping[str, str], secret: str, endpoints: list[str], work_dir: str | Path,
           post: Callable | None = None, speed: float | None = None,
           progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Drive the real EdgeAgent (real folder source, windows, z_t, wire audit, spool, signed transport)
    over the staged history. `speed`: simulated seconds per wall second (None = as fast as possible).
    `post` replaces the HTTP call (tests: an in-process app)."""
    from edge_agent.agent import EdgeAgent
    from edge_agent.prices import StaticPriceAdapter
    from edge_agent.sources import build_sources

    work = Path(work_dir)
    state = work / "edge-state"
    if state.exists() and any(state.iterdir()):
        raise RuntimeError(f"{state} is not empty: a replay starts from a fresh edge state")
    state.mkdir(parents=True, exist_ok=True)
    unknown = sorted(set(staged["models"]) - set(models))
    if unknown:
        raise ConfigError(f"{len(unknown)} model(s) in the history have no monitored model id")
    cfg = edge_config(spec, tenant_id=tenant_id, install_id=install_id, models=models, secret=secret,
                      endpoints=endpoints, state_dir=state)
    src_spec = {"type": "folder", "path": str(Path(staged["path"]).parent), "patterns": ["history.csv"],
                "format": "csv", "min_age_seconds": 0, "scan_interval_seconds": 0, "batch_size": 5000,
                "columns": replay_columns(spec), "unterminated_line_after_seconds": 1}
    if spec.predictions.constants:
        src_spec["constants"] = spec.predictions.constants
    clock = _ReplayClock(datetime.fromisoformat(staged["first"]) - timedelta(seconds=1))
    (adapter,) = build_sources(src_spec, state, "prediction")
    agent = EdgeAgent(cfg, [_Tracking(adapter, clock)], prices=StaticPriceAdapter({}), post=post, clock=clock)
    sim0, wall0, last_note = clock.now, time.monotonic(), time.monotonic()
    try:
        while True:
            n = agent.poll()
            agent.tick(clock.now)
            if speed:
                ahead = (clock.now - sim0).total_seconds() / speed - (time.monotonic() - wall0)
                if ahead > 0:
                    time.sleep(min(ahead, 60.0))
            if progress and time.monotonic() - last_note > 30:
                last_note = time.monotonic()
                progress(f"replayed to {clock.now.isoformat()} sent={agent.sender.stats.sent}")
            if n == 0:
                break
        agent.flush()
        for _ in range(50):                       # retryable failures stay queued: drain them in order
            if not agent.sender.pending_count():
                break
            agent.sender.drain(force=True)
            time.sleep(0.2 if post is None else 0)
        status = agent.status()
    finally:
        agent.close()
    wall = time.monotonic() - wall0
    sim = (clock.now - sim0).total_seconds()
    return {"wall_seconds": round(wall, 2), "simulated_seconds": round(sim, 1),
            "speedup": round(sim / wall, 1) if wall > 0 else None,
            "sent": status["sent"], "rejected": status["rejected"], "queued_unsent": status["queued"],
            "backlog_sent": status["backlog_sent"], "rejected_reasons": agent.sender.stats.reasons[:5],
            "models": {k: {x: v[x] for x in ("phase", "reference_windows", "emitted", "skipped", "failed")}
                       for k, v in status["models"].items()},
            "prediction_rows_rejected": status["prediction_rows_rejected"],
            "source_bad_rows": sum(s.get("bad_rows", 0) or 0 for s in status["sources"])}


def incident_marks(spec: ReplaySpec, fr: Frames, validation: Mapping[str, Any],
                   models: Mapping[str, str]) -> list[dict[str, Any]]:
    """The usable incidents as `POST /api/v1/incidents` bodies (prereg 9: loaded as incident marks)."""
    if fr.incidents is None or not validation.get("incidents"):
        return []
    usable = {(x["incident_id"], x["model"]) for x in validation["incidents"]["incidents"] if x["usable"]}
    out = []
    for _, r in fr.incidents.iterrows():
        if (_safe(r["_id"]), _safe(r["_model"])) not in usable or str(r["_model"]) not in models:
            continue
        body = {"monitored_model_id": models[str(r["_model"])], "failed_from": r["_from"].isoformat(),
                "failed_to": r["_to"].isoformat()}
        if r["_class"] is not None and not pd.isna(r["_class"]) and str(r["_class"]):
            body["failure_class"] = str(r["_class"])[:64]
        out.append(body)
    return out



# ---------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------
def _write_json(path: Path, obj: Any, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
    path.chmod(mode)


def cmd_validate(a: argparse.Namespace, *, load: Callable[..., ReplaySpec] = load_spec) -> int:
    spec = load(a.mapping)
    fr = load_frames(spec)
    v = validate(spec, fr)
    _write_json(Path(a.out) / "validation.json", v)
    print(json.dumps({"ok": v["ok"], "errors": len(v["errors"]), "warnings": len(v["warnings"])}))
    for e in v["errors"]:
        print("ERROR", e)
    for w in v["warnings"]:
        print("WARN ", w)
    return 0 if v["ok"] else 2


def _http_post(timeout: float = 60.0, cert: tuple[str, str] | None = None, ca_file: str | None = None) -> Callable:
    """Like the SDK's default post, but honours 429 Retry-After (a lab stack rate-limits ingest)."""
    import httpx
    clients: dict[str, Any] = {}
    kw: dict[str, Any] = {}
    if cert or ca_file:
        import ssl
        ctx = ssl.create_default_context(cafile=ca_file)
        if cert:
            ctx.load_cert_chain(*cert)
        kw["verify"] = ctx

    def post(url: str, path: str, body: bytes, headers: Mapping[str, str]):
        c = clients.get(url) or clients.setdefault(url, httpx.Client(base_url=url, timeout=timeout, **kw))
        for _ in range(10):
            r = c.post(path, content=body, headers=dict(headers))
            if r.status_code != 429:
                break
            time.sleep(min(60.0, float(r.headers.get("Retry-After") or 5) + 0.5))
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {}
    return post


def _tenant(a: argparse.Namespace) -> dict[str, Any]:
    """The tenant block of a state file: `--state-file F`, or `<--state-dir>/state.json` (written by
    `onboard` on our side, mode 0600): {"tenants": {name: {tenant_id, api_key, hmac_secret, models:
    {local id: {model_id, install_id}}}}}. The partner's copy (`replay-state.json`) carries no secrets: the
    API key and HMAC secret are given as BLURREDBOX_API_KEY / AGD_HMAC_SECRET. `--tenant` may be omitted
    when the file holds one tenant."""
    path = Path(a.state_file) if getattr(a, "state_file", None) else Path(a.state_dir) / "state.json"
    try:
        tenants = json.loads(path.read_text())["tenants"]
    except (OSError, ValueError, KeyError, TypeError):
        raise ConfigError(f"cannot read the tenants from the state file {path}") from None
    name = getattr(a, "tenant", None)
    if name is None:
        if len(tenants) != 1:
            raise ConfigError(f"{path.name} holds {len(tenants)} tenants; name one with --tenant")
        name = next(iter(tenants))
    if name not in tenants:
        raise ConfigError(f"tenant {name!r} is not in {path.name}")
    a.tenant = name
    ten = dict(tenants[name])
    ten["api_key"] = ten.get("api_key") or os.environ.get("BLURREDBOX_API_KEY", "")
    ten["hmac_secret"] = ten.get("hmac_secret") or os.environ.get("AGD_HMAC_SECRET", "")
    return ten


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _base_url(a: argparse.Namespace) -> str:
    """The ingest base URL. Security review rc5: plain http:// only to this machine (the default), never to a remote
    address, where the API key and the signed windows would cross the network in clear text."""
    from urllib.parse import urlparse
    url = a.base_url or f"http://127.0.0.1:{a.port}"
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "https" and host:
        return url
    if parsed.scheme == "http" and (host in _LOOPBACK_HOSTS or host.startswith("127.")):
        return url
    raise ValueError(f"refused base URL {url!r}: use https:// (plain http:// only to localhost / 127.0.0.1)")


def post_incident_marks(a: argparse.Namespace, spec: ReplaySpec, fr: Frames, validation: Mapping[str, Any],
                        ten: Mapping[str, Any]) -> int:
    """POST the usable incidents as incident marks (model id, time range, failure class), signed."""
    from blurredbox_agd.ingest import IngestTransport
    cert = tuple(a.client_cert) if getattr(a, "client_cert", None) else None
    models = {local: m["model_id"] for local, m in ten["models"].items()}
    os.environ["BLURREDBOX_API_KEY"] = ten["api_key"]           # read by the SDK transport; never printed
    tr = IngestTransport([_base_url(a)], secret=ten["hmac_secret"], tenant_id=ten["tenant_id"],
                         install_id="-", monitored_model_id="-", mode="failover",
                         post=_http_post(cert=cert, ca_file=getattr(a, "ca_file", None)))
    marks = incident_marks(spec, fr, validation, models)
    for body in marks:
        tr.send("/api/v1/incidents", body)
    return len(marks)


def cmd_send(a: argparse.Namespace, *, load: Callable[..., ReplaySpec] = load_spec) -> int:
    """Validate, stage and replay each model through the edge agent; with `post_incident_marks`, then post
    the incident marks. This is the part that runs where the prediction log is: nothing but the window
    envelopes of edge_agent/wire.py (and, if asked, the incident marks) is sent."""
    try:
        _base_url(a)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    spec = load(a.mapping)
    fr = load_frames(spec)
    out = Path(a.out)
    v = validate(spec, fr)
    _write_json(out / "validation.json", v)
    if not v["ok"]:
        print("validation failed:", *v["errors"], sep="\n  ", file=sys.stderr)
        return 2
    ten = _tenant(a)
    if not ten["api_key"] or not ten["hmac_secret"]:
        print("no API key / HMAC secret (state file or BLURREDBOX_API_KEY / AGD_HMAC_SECRET)", file=sys.stderr)
        return 2
    models = {local: m["model_id"] for local, m in ten["models"].items()}
    missing = sorted(set(v["predictions"]["models"]) - set(models))
    if missing:
        print(f"{len(missing)} model(s) in the export are not onboarded for tenant {a.tenant}", file=sys.stderr)
        return 2
    os.environ["BLURREDBOX_API_KEY"] = ten["api_key"]           # read by the SDK transport; never printed
    cert = tuple(a.client_cert) if a.client_cert else None
    stats: dict[str, Any] = {"models": {}}
    # one EdgeAgent per model: each monitored model is paired with its own install id
    for local, m in ten["models"].items():
        if local not in v["predictions"]["models"]:
            continue
        sub = out / "work" / re.sub(r"[^A-Za-z0-9_-]", "_", m["model_id"])
        st = replay(spec, _only_model(spec, fr, out / "stage" / sub.name, local), tenant_id=ten["tenant_id"],
                    install_id=m["install_id"], models={local: m["model_id"]}, secret=ten["hmac_secret"],
                    endpoints=[_base_url(a)], work_dir=sub, post=_http_post(cert=cert, ca_file=a.ca_file),
                    speed=a.speed, progress=print)
        stats["models"][local] = st
        print(f"{m['model_id']}: sent={st['sent']} rejected={st['rejected']} wall={st['wall_seconds']}s "
              f"speedup={st['speedup']}x", flush=True)
    stats["incident_marks_posted"] = (post_incident_marks(a, spec, fr, v, ten)
                                      if getattr(a, "post_incident_marks", False) else 0)
    _write_json(out / "send.json", stats)
    return 0 if all(s["rejected"] == 0 and s["queued_unsent"] == 0 for s in stats["models"].values()) else 1


def _only_model(spec: ReplaySpec, fr: Frames, stage_dir: Path, local: str) -> dict[str, Any]:
    sub = Frames(fr.predictions[fr.predictions["_model"] == local], fr.incidents, fr.ops)
    return stage_predictions(spec, sub, stage_dir)


def _limits_shape(raw: Any) -> dict[str, dict[str, float]]:
    """Partner-side check of `leadsignals_limits`: `{indicator: {"above"|"below": number}}`. The full
    check (the indicator names the server knows) runs on our side when the tenant is set up."""
    if not isinstance(raw, dict) or not raw:
        raise ConfigError("leadsignals_limits must be a nonempty object keyed by indicator")
    out: dict[str, dict[str, float]] = {}
    for ind, spec in raw.items():
        if not isinstance(ind, str) or not isinstance(spec, dict) or not spec or set(spec) - {"above", "below"}:
            raise ConfigError(f"leadsignals_limits.{ind}: give 'above' and/or 'below' with a number")
        for d, v in spec.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ConfigError(f"leadsignals_limits.{ind}.{d} must be a finite number")
        out[ind] = {d: float(v) for d, v in spec.items()}
    return out


def main(argv: list[str] | None = None, *, prog: str = "blurredbox-edge-replay") -> int:
    """The partner-side CLI: validate and send. Nothing here reads a database or a BlurredBox server
    other than the ingest base URL given to `send`."""
    ap = argparse.ArgumentParser(prog=prog, description="BlurredBox partner history replay, partner side: "
                                 "validate a history export and send its window statistics.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("validate", help="check the export; writes OUT/validation.json (counts only); sends nothing")
    p.add_argument("--mapping", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("send", help="replay the history through the edge agent and send the window statistics")
    p.add_argument("--mapping", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--state-file", required=True,
                   help="replay-state.json from BlurredBox: tenant id and model ids (no secrets); the API key "
                        "and HMAC secret come from BLURREDBOX_API_KEY / AGD_HMAC_SECRET")
    p.add_argument("--tenant", default=None, help="tenant name in the state file (default: the only one)")
    p.add_argument("--base-url", required=True, help="the ingest base URL BlurredBox gives you (https://...)")
    p.add_argument("--client-cert", nargs=2, metavar=("CERT", "KEY"))
    p.add_argument("--ca-file")
    p.add_argument("--speed", type=float, default=None,
                   help="simulated seconds per wall second (default: as fast as the server accepts)")
    p.add_argument("--post-incident-marks", action="store_true",
                   help="also POST the usable incidents (model id, time range, failure class) as incident marks")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "validate":
            return cmd_validate(a)
        a.state_dir = None
        return cmd_send(a)
    except ConfigError as exc:
        print(f"mapping error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
