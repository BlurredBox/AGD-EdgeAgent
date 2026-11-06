"""Edge agent configuration. Everything here is local; none of it is sent
except the three BlurredBox-generated identifiers and the schema version."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path


@dataclass
class EdgeConfig:
    # -- identity (BlurredBox-generated; the only non-signal fields on the wire)
    tenant_id: str
    install_id: str
    # partner's LOCAL model id -> BlurredBox `monitored_model_id`. The local id
    # never leaves; only the generated one does.
    models: dict[str, str]
    secret: str | bytes
    endpoints: list[str]
    state_dir: Path

    # -- windows (Phase 1 Task 8 default cadence: 5 minutes)
    cadence_seconds: int = 300
    min_rows_per_window: int = 30
    reference_windows: int = 200
    reference_max_rows: int = 20_000        # seeded reservoir cap on memory
    max_rows_per_window: int = 5_000        # strided cap for the heavy detectors
    outcome_join_max_records: int = 100_000
    outcome_join_max_age_seconds: float = 7 * 86400
    # Grace before a window is closed, by the clock or by a row of a later window (audit F14). 0 closed a window the
    # instant its end passed, so a row whose log write or flush trailed by a second was dropped as late. 5 s covers
    # buffered log writes and the 1 s poll with margin, and holds each window open under 2% of the 300 s cadence.
    # An explicit 0 is honoured.
    lateness_seconds: float = 5.0
    # A row stamped further than this ahead of the agent's clock is counted and dropped, never windowed (audit F6):
    # one such row would close the current window early and leave a future window open. 60 s is the clock-skew
    # allowance AGD #453 gives the server for window_end.
    future_tolerance_seconds: float = 60.0

    # -- model output semantics
    # "probability": the output is a score in [0, 1]; entropy / positive-decision
    #   rate are meaningful. "signed": a return forecast / position; those two
    #   probability-only fields are NOT emitted (they would be meaningless).
    output_kind: str = "signed"

    # -- local tripwires
    feed_down_after_seconds: float | None = None   # default 3 x cadence
    # Decision 2 (2026-10-29): a model whose windows carried labels raises a local `labels_missing` once this
    # many consecutive closed windows have label coverage 0, and `labels_recovered` when a labelled window
    # returns. 0 turns it off. Default 6: one cadence-relative episode threshold like `feed_down` (30 min at
    # the 5-minute default cadence, 6 min at 1 minute); it fires inside the demo seed's 12-window label lag;
    # and a model whose windows are each labelled with probability p fires by chance about once per
    # (1 - p) ** -6 windows, i.e. under once a week at a 5-minute cadence (2,016 windows) for p >= 0.72.
    labels_missing_after_windows: int = 6
    feature_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    output_bounds: tuple[float, float] | None = None
    max_invalid_row_fraction: float = 0.2
    alert_cooldown_seconds: float = 300.0

    # -- leading indicators (G53; counts, rates and lags only, see corpus/lead_signals.py)
    null_rate_ceiling: float = 0.2          # a feature column above this null share is counted
    # "ingest": a record without its own arrival stamp is stamped when the agent receives it.
    # "source_only": only a mapped arrival column counts (history replay: a replay-time stamp
    # would be meaningless, so no arrival lag is emitted without the column).
    arrival_stamp: str = "ingest"
    slice_feature: str | None = None        # LOCAL feature name whose value defines a slice; never sent
    slice_min_rows: int = 30                # rows per slice per window (and in the reference) to score it
    slice_shift_limit: float = 1.0          # shift in reference output sd units counted as "over"
    slice_max: int = 50                     # distinct slices tracked; further values are not scored

    # -- public prices
    instrument_series: dict[str, str] = field(default_factory=dict)  # instrument -> FRED id
    price_max_staleness_days: int = 5

    # -- transport
    schema_version: str = "1.4"           # 1.4: G53 leading-indicator scalars (corpus/lead_signals.py)
    spool_max_age_seconds: float = 33 * 86400       # D-50: ~33-day offline queue
    backlog_after_seconds: float | None = None      # default 2 x cadence
    timeout_seconds: float = 5.0
    client_cert: tuple[str, str] | None = None
    ca_file: str | None = None

    seed: int = 0

    def __post_init__(self) -> None:
        self.state_dir = Path(self.state_dir)
        if self.client_cert is not None:
            if (not isinstance(self.client_cert, (tuple, list)) or len(self.client_cert) != 2
                    or not all(isinstance(path, str) and path.strip() for path in self.client_cert)):
                raise ValueError("client_cert must contain certificate and private-key file paths")
            self.client_cert = tuple(self.client_cert)
        if self.ca_file is not None and (not isinstance(self.ca_file, str) or not self.ca_file.strip()):
            raise ValueError("ca_file must be a nonempty CA certificate file path")
        if not (isinstance(self.future_tolerance_seconds, (int, float)) and math.isfinite(self.future_tolerance_seconds)
                and self.future_tolerance_seconds >= 0):
            raise ValueError("future_tolerance_seconds must be a number >= 0")
        if self.output_kind not in ("signed", "probability"):
            raise ValueError("output_kind must be 'signed' or 'probability'")
        if self.cadence_seconds < 1 or self.reference_windows < 1:
            raise ValueError("cadence_seconds and reference_windows must be >= 1")
        if not self.models:
            raise ValueError("at least one model must be configured")
        if (type(self.outcome_join_max_records) is not int or self.outcome_join_max_records < 1
                or not math.isfinite(self.outcome_join_max_age_seconds) or self.outcome_join_max_age_seconds <= 0):
            raise ValueError("outcome join capacity and retention must be positive")
        if self.arrival_stamp not in ("ingest", "source_only"):
            raise ValueError("arrival_stamp must be 'ingest' or 'source_only'")
        if not (0 <= self.null_rate_ceiling < 1) or self.slice_min_rows < 1 or self.slice_max < 1 \
                or not (math.isfinite(self.slice_shift_limit) and self.slice_shift_limit > 0):
            raise ValueError("null_rate_ceiling in [0, 1), slice_min_rows and slice_max >= 1, "
                             "slice_shift_limit > 0")
        if self.slice_feature is not None and (not isinstance(self.slice_feature, str)
                                               or not self.slice_feature.strip()):
            raise ValueError("slice_feature must be a nonempty feature name")
        if type(self.labels_missing_after_windows) is not int or self.labels_missing_after_windows < 0:
            raise ValueError("labels_missing_after_windows must be an integer >= 0 (0 turns it off)")
        if self.feed_down_after_seconds is None:
            self.feed_down_after_seconds = 3.0 * self.cadence_seconds
        if self.backlog_after_seconds is None:
            self.backlog_after_seconds = 2.0 * self.cadence_seconds
