"""Synthetic partner stream with unmistakable sentinels, for the edge agent tests."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from edge_agent import EdgeConfig, Record
from edge_agent.prices import PriceBar, StaticPriceAdapter

T0 = 1_700_000_040          # divisible by 60: windows start on a boundary
CADENCE = 60
LOCAL_MODEL = "LOCAL_MODEL_SENTINEL"
WIRE_MODEL = "mm-wire-0001"
INSTRUMENTS = ("INSTR_SENTINEL_A", "INSTR_SENTINEL_B")
FEATURES = ("secret_alpha_feature_zz", "beta_px_zz", "gamma_vol_zz")
CATEGORY = "regime_zz"
CATS = ("CATEGORY_SENTINEL_A", "CATEGORY_SENTINEL_B", "CATEGORY_SENTINEL_C")


def utc(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, timezone.utc)


def make_cfg(tmp_path, **kw) -> EdgeConfig:
    base = dict(tenant_id="tenant-1", install_id="install-1",
                models={LOCAL_MODEL: WIRE_MODEL}, secret="s3cret-s3cret-s3cret",
                endpoints=["https://ingest.example.invalid"], state_dir=tmp_path / "state",
                cadence_seconds=CADENCE, min_rows_per_window=30, reference_windows=6)
    base.update(kw)
    return EdgeConfig(**base)


def make_stream(n_windows: int, rows: int = 220, *, first_window: int = 0,
                drift_from: int | None = None, seed: int = 7) -> list[Record]:
    rng = np.random.default_rng(seed + first_window)
    recs: list[Record] = []
    for w in range(first_window, first_window + n_windows):
        shift = 3.0 if drift_from is not None and w >= drift_from else 0.0
        for i in range(rows):
            ts = T0 + w * CADENCE + i * CADENCE / rows
            f = {FEATURES[0]: float(rng.normal(shift, 1.0)),
                 FEATURES[1]: float(rng.normal(0.0, 2.0)),
                 FEATURES[2]: float(abs(rng.normal(5.0, 1.0))),
                 CATEGORY: str(rng.choice(CATS))}
            out = 0.002 * f[FEATURES[0]] + float(rng.normal(0, 0.001))
            recs.append(Record.make(ts, LOCAL_MODEL, INSTRUMENTS[i % 2], f, out))
    return recs


def make_prices(n_windows: int, seed: int = 11) -> StaticPriceAdapter:
    rng = np.random.default_rng(seed)
    series = {}
    for k, inst in enumerate(INSTRUMENTS):
        p, bars = 100.0 + 17.0 * k, []
        for s in range(0, (n_windows + 2) * CADENCE, 5):
            p *= float(np.exp(rng.normal(0, 0.0007)))
            bars.append((utc(T0 + s), PriceBar(price=p, volume=float(rng.uniform(900, 1100)))))
        series[inst] = bars
    return StaticPriceAdapter(series)


class Collector:
    """A `post` stub: records every signed request the SDK would put on the wire."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, bytes, dict]] = []
        self.status = 202
        self.raises: Exception | None = None

    def __call__(self, url, path, body, headers):
        if self.raises is not None:
            raise self.raises
        self.requests.append((url, path, body, dict(headers)))
        return self.status, ({"accepted": True} if self.status < 300 else {"detail": {}})

    def bodies(self) -> list[dict]:
        return [json.loads(b) for _, _, b, _ in self.requests]
