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

    # -- windows (default cadence: 5 minutes)
    cadence_seconds: int = 300
    min_rows_per_window: int = 30
    reference_windows: int = 200
    reference_max_rows: int = 20_000        # seeded reservoir cap on memory
    max_rows_per_window: int = 5_000        # strided cap for the heavy detectors
    outcome_join_max_records: int = 100_000
    outcome_join_max_age_seconds: float = 7 * 86400
    lateness_seconds: float = 0.0           # grace before a window is closed by the clock

    # -- model output semantics
    # "probability": the output is a score in [0, 1]; entropy / positive-decision
    #   rate are meaningful. "signed": a return forecast / position; those two
    #   probability-only fields are NOT emitted (they would be meaningless).
    output_kind: str = "signed"

    # -- local tripwires
    feed_down_after_seconds: float | None = None   # default 3 x cadence
    feature_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    output_bounds: tuple[float, float] | None = None
    max_invalid_row_fraction: float = 0.2
    alert_cooldown_seconds: float = 300.0

    # -- public prices
    instrument_series: dict[str, str] = field(default_factory=dict)  # instrument -> FRED id
    price_max_staleness_days: int = 5

    # -- transport
    schema_version: str = "1.2"
    spool_max_age_seconds: float = 33 * 86400       # ~33-day offline queue
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
        if self.output_kind not in ("signed", "probability"):
            raise ValueError("output_kind must be 'signed' or 'probability'")
        if self.cadence_seconds < 1 or self.reference_windows < 1:
            raise ValueError("cadence_seconds and reference_windows must be >= 1")
        if not self.models:
            raise ValueError("at least one model must be configured")
        if (type(self.outcome_join_max_records) is not int or self.outcome_join_max_records < 1
                or not math.isfinite(self.outcome_join_max_age_seconds) or self.outcome_join_max_age_seconds <= 0):
            raise ValueError("outcome join capacity and retention must be positive")
        if self.feed_down_after_seconds is None:
            self.feed_down_after_seconds = 3.0 * self.cadence_seconds
        if self.backlog_after_seconds is None:
            self.backlog_after_seconds = 2.0 * self.cadence_seconds
