"""The one input record type: what the agent observes about the model."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Record:
    """One (timestamp, model, instrument, feature row, model output) observation.

    `ts` is timezone-aware UTC. Everything here is RAW partner data and stays
    on the local machine; nothing on a Record is ever put on the wire.
    """
    ts: datetime
    model_id: str
    instrument: str
    features: dict[str, Any]
    output: float
    prediction_id: str | None = None
    outcome: int | None = None
    outcome_observed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ts, datetime) or self.ts.tzinfo is None:
            raise ValueError("prediction timestamp must have a timezone")
        if self.prediction_id is not None and (not isinstance(self.prediction_id, str)
                                              or not 1 <= len(self.prediction_id) <= 256):
            raise ValueError("prediction_id must be a nonempty string of at most 256 characters")
        if self.outcome is not None:
            if type(self.outcome) is not int or self.outcome not in (0, 1):
                raise ValueError("outcome must be the integer 0 or 1")
            if (not isinstance(self.outcome_observed_at, datetime)
                    or self.outcome_observed_at.tzinfo is None):
                raise ValueError("outcome_observed_at must have a timezone")
            if self.outcome_observed_at < self.ts:
                raise ValueError("outcome cannot be observed before its prediction")
        elif self.outcome_observed_at is not None:
            raise ValueError("outcome_observed_at requires an outcome")

    @staticmethod
    def make(ts: datetime | float | int | str, model_id: str, instrument: str,
             features: dict[str, Any], output: Any, *,
             prediction_id: str | None = None, outcome: int | None = None,
             outcome_observed_at: datetime | float | int | str | None = None) -> "Record":
        """Normalize a local prediction; optional labels are binary realised outcomes."""
        if prediction_id is not None and (not isinstance(prediction_id, str)
                                          or not 1 <= len(prediction_id) <= 256):
            raise ValueError("prediction_id must be a nonempty string of at most 256 characters")
        if outcome is not None and (type(outcome) is not int or outcome not in (0, 1)):
            raise ValueError("outcome must be the integer 0 or 1")
        if outcome is None and outcome_observed_at is not None:
            raise ValueError("outcome_observed_at requires an outcome")
        if isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(float(ts), timezone.utc)
        elif isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        known = None
        if outcome is not None:
            if outcome_observed_at is None:
                raise ValueError("outcome_observed_at is required with an outcome")
            if isinstance(outcome_observed_at, datetime):
                known = outcome_observed_at
            elif isinstance(outcome_observed_at, (int, float)):
                known = datetime.fromtimestamp(outcome_observed_at, timezone.utc)
            else:
                known = datetime.fromisoformat(outcome_observed_at.replace("Z", "+00:00"))
            if known.tzinfo is None:
                raise ValueError("outcome_observed_at must have a timezone")
            known = known.astimezone(timezone.utc)
            if known < ts:
                raise ValueError("outcome cannot be observed before its prediction")
        try:
            out = float(output)
        except (TypeError, ValueError):
            out = math.nan
        return Record(ts.astimezone(timezone.utc), str(model_id), str(instrument),
                      dict(features), out, prediction_id, outcome, known)
