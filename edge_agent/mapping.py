"""Column mapping shared by the table/file input adapters.

A partner's table or file rarely has the agent's field names. `ColumnMapping`
validates a small config block once, at start-up, with errors that name the
offending key, and turns one source row (a mapping of column name -> value)
into a `Record`.

    "columns": {"ts": "scored_at", "model_id": "model", "instrument": "ticker",
                "output": "p_default", "features": ["age", "income", "region"],
                "prediction_id": "pid", "outcome": "defaulted",
                "outcome_observed_at": "defaulted_known_at"},
    "constants": {"model_id": "credit-scorer"}

`features` is a list of column names, `{"json_column": "name"}` (one JSON
object column) or, for files only, `{"prefix": "f_"}` / `{"rest": true}`.
`constants` supplies `model_id` / `instrument` when the source has no such
column. Nothing here reads a row value into a message: errors name columns.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Mapping

from edge_agent.records import Record


class ConfigError(ValueError):
    """The source configuration is wrong; the message says which key and why."""


PREDICTION_FIELDS = ("ts", "model_id", "instrument", "output", "features",
                     "prediction_id", "outcome", "outcome_observed_at",
                     "arrived_at")   # optional (G53): when the row arrived; history replay arrival column
OUTCOME_FIELDS = ("model_id", "prediction_id", "outcome", "outcome_observed_at")
_CONSTANT_FIELDS = ("model_id", "instrument")
_FEATURE_FORMS = ("json_column", "prefix", "rest")


def _name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ConfigError(f"{where} must be a nonempty column name (at most 128 characters)")
    return value


class ColumnMapping:
    def __init__(self, columns: Any, constants: Any = None, *, kind: str = "prediction",
                 allow_pattern_features: bool = True, null_values: tuple[str, ...] = ()) -> None:
        if kind not in ("prediction", "outcome"):
            raise ConfigError("kind must be 'prediction' or 'outcome'")
        self.kind = kind
        self.null_values = frozenset(null_values)
        if not isinstance(columns, dict):
            raise ConfigError("'columns' is required and must be an object mapping fields to column names")
        allowed = OUTCOME_FIELDS if kind == "outcome" else PREDICTION_FIELDS
        extra = sorted(set(columns) - set(allowed))
        if extra:
            raise ConfigError(f"columns: unknown field(s) {extra}; allowed for {kind} sources: {list(allowed)}")
        constants = {} if constants is None else constants
        if not isinstance(constants, dict):
            raise ConfigError("'constants' must be an object")
        bad = sorted(set(constants) - set(_CONSTANT_FIELDS))
        if bad:
            raise ConfigError(f"constants: only {list(_CONSTANT_FIELDS)} may be constants, got {bad}")
        for k, v in constants.items():
            if not isinstance(v, str) or not v:
                raise ConfigError(f"constants.{k} must be a nonempty string")
        self.constants: dict[str, str] = dict(constants)
        self.cols: dict[str, str] = {}
        self.features: Any = None
        for field, col in columns.items():
            if field == "features":
                self.features = self._features(col, allow_pattern_features)
            else:
                self.cols[field] = _name(col, f"columns.{field}")
        for field in _CONSTANT_FIELDS:
            if field in self.cols and field in self.constants:
                raise ConfigError(f"'{field}' is given both as a column and as a constant; choose one")
        self._require(kind)

    def _features(self, spec: Any, allow_pattern: bool) -> Any:
        if isinstance(spec, list):
            if not spec:
                raise ConfigError("columns.features must list at least one column")
            names = [_name(c, "columns.features[]") for c in spec]
            if len(set(names)) != len(names):
                raise ConfigError("columns.features lists a column twice")
            return ("list", names)
        if isinstance(spec, dict) and len(spec) == 1:
            (form, arg), = spec.items()
            if form == "json_column":
                return ("json", _name(arg, "columns.features.json_column"))
            if form in ("prefix", "rest") and not allow_pattern:
                raise ConfigError(f"columns.features.{form} is only available for file sources; "
                                  "a table source needs an explicit column list or json_column")
            if form == "prefix":
                return ("prefix", _name(arg, "columns.features.prefix"))
            if form == "rest":
                if arg is not True:
                    raise ConfigError("columns.features.rest must be true")
                return ("rest", None)
        raise ConfigError("columns.features must be a list of column names or one of "
                          f"{{{', '.join(repr(f) for f in _FEATURE_FORMS)}: ...}}")

    def _require(self, kind: str) -> None:
        if kind == "prediction":
            need = ["ts", "output"]
            if self.features is None:
                raise ConfigError("columns.features is required for a prediction source")
        else:
            need = ["prediction_id", "outcome", "outcome_observed_at"]
            if self.features is not None:
                raise ConfigError("columns.features does not apply to an outcome source")
        for field in need:
            if field not in self.cols:
                raise ConfigError(f"columns.{field} is required for a {kind} source")
        if "model_id" not in self.cols and "model_id" not in self.constants:
            raise ConfigError("model_id is required: map columns.model_id or set constants.model_id")
        if kind == "prediction" and "instrument" not in self.cols and "instrument" not in self.constants:
            raise ConfigError("instrument is required: map columns.instrument or set constants.instrument "
                              "(a fixed label such as \"all\" is fine)")
        if kind == "prediction":
            have_o, have_t = "outcome" in self.cols, "outcome_observed_at" in self.cols
            if have_o != have_t:
                raise ConfigError("columns.outcome and columns.outcome_observed_at go together")

    # -- what to read ---------------------------------------------------------------------
    @property
    def time_column(self) -> str:
        """The column rows are ordered by: prediction time, or when an outcome became known."""
        return self.cols["outcome_observed_at" if self.kind == "outcome" else "ts"]

    def fixed_columns(self) -> list[str]:
        """Every explicitly named column, in a stable order, without duplicates."""
        names = list(self.cols.values())
        if self.features and self.features[0] == "list":
            names += self.features[1]
        elif self.features and self.features[0] == "json":
            names.append(self.features[1])
        return list(dict.fromkeys(names))

    # -- row -> Record ----------------------------------------------------------------------
    def to_record(self, row: Mapping[str, Any]) -> Record | None:
        """Build the Record, or None for an outcome row that has no label yet.

        Raises ValueError/KeyError/TypeError for a malformed row (the adapter
        counts it as bad); the message never contains a cell value.
        """
        get = lambda f: self._cell(row[self.cols[f]])           # noqa: E731
        model = self.constants.get("model_id") or _text(get("model_id"), "model_id")
        if self.kind == "outcome":
            label = get("outcome")
            if label is None:
                return None
            known = _time(get("outcome_observed_at"))
            pid = get("prediction_id")
            return Record.make(known, model, "", {}, 0, prediction_id=_text(pid, "prediction_id"),
                               outcome=_label(label), outcome_observed_at=known)
        instrument = self.constants.get("instrument") or _text(get("instrument"), "instrument")
        outcome = observed = pid = None
        if "outcome" in self.cols:
            outcome = get("outcome")
            observed = _time(get("outcome_observed_at"))
            if outcome is None:
                outcome = observed = None
            else:
                outcome = _label(outcome)
        if "prediction_id" in self.cols:
            pid = get("prediction_id")
            pid = None if pid is None else _text(pid, "prediction_id")
        arrived = None
        if "arrived_at" in self.cols:
            arrived = get("arrived_at")
            arrived = None if arrived is None else _time(arrived)
        return Record.make(_time(get("ts")), model, instrument, self._feature_values(row), get("output"),
                           prediction_id=pid, outcome=outcome, outcome_observed_at=observed,
                           arrived_at=arrived)

    def _cell(self, value: Any) -> Any:
        if isinstance(value, str):
            if value in self.null_values:
                return None
            return value
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
        return value

    def _feature_values(self, row: Mapping[str, Any]) -> dict[str, Any]:
        form, arg = self.features
        if form == "list":
            return {c: self._feature(row[c]) for c in arg}
        if form == "json":
            raw = row[arg]
            if isinstance(raw, (str, bytes)):
                raw = json.loads(raw)
            if not isinstance(raw, dict):
                raise TypeError("features column must hold a JSON object")
            return dict(raw)
        used = set(self.cols.values())
        if form == "prefix":
            return {c[len(arg):]: self._feature(v) for c, v in row.items()
                    if c.startswith(arg) and len(c) > len(arg)}
        return {c: self._feature(v) for c, v in row.items() if c not in used}

    def _feature(self, value: Any) -> Any:
        if isinstance(value, float):
            return value                       # NaN stays NaN: the tripwires judge it, not the adapter
        value = self._cell(value)
        if isinstance(value, str):
            try:
                return float(value) if value.strip() and _numeric(value) else value
            except ValueError:
                return value
        if isinstance(value, datetime):
            return value.isoformat()
        return value


def _numeric(text: str) -> bool:
    t = text.strip().lower()
    return bool(t) and (t[0] in "+-.0123456789" or t in ("nan", "inf", "-inf", "infinity"))


def _text(value: Any, field: str) -> str:
    if value is None or (isinstance(value, str) and not value):
        raise ValueError(f"{field} is empty")
    return value if isinstance(value, str) else str(value)


def _label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        value = {"0": 0, "1": 1, "true": 1, "false": 0, "t": 1, "f": 0}.get(value.strip().lower(), value)
    if isinstance(value, float) and value in (0.0, 1.0):
        value = int(value)
    if type(value) is not int or value not in (0, 1):
        raise ValueError("outcome must be 0 or 1")
    return value


def _time(value: Any) -> Any:
    """A time cell as Record.make wants it: CSV hands numbers over as text."""
    if isinstance(value, str):
        t = value.strip()
        try:
            return float(t) if t and t.lstrip("+-").replace(".", "", 1).isdigit() else value
        except ValueError:
            return value
    return value


def epoch_of(value: Any) -> float | None:
    """Best-effort POSIX seconds of a raw time cell (None if it cannot be read)."""
    value = _time(value)
    try:
        if isinstance(value, datetime):
            d = value
        elif isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
            return float(value)
        elif isinstance(value, str):
            d = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        else:
            return None
        return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()
    except (ValueError, OverflowError, OSError):
        return None


# -- small validators shared by the source configs --------------------------------------------
def check_keys(spec: dict, allowed: set[str], where: str) -> None:
    extra = sorted(set(spec) - allowed)
    if extra:
        raise ConfigError(f"{where}: unknown key(s) {extra}; allowed: {sorted(allowed)}")


def int_in(spec: dict, key: str, default: int, lo: int, hi: int, where: str) -> int:
    v = spec.get(key, default)
    if type(v) is not int or not lo <= v <= hi:
        raise ConfigError(f"{where}.{key} must be an integer between {lo} and {hi}")
    return v


def num_in(spec: dict, key: str, default: float, lo: float, hi: float, where: str) -> float:
    v = spec.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
        raise ConfigError(f"{where}.{key} must be a number between {lo} and {hi}")
    return float(v)
