"""Public-price adapters. No paid source, no partner data requested.

The agent pulls PUBLIC prices itself. Prices are only ever consumed locally
(realised performance, market regime); they are inputs to scalars, never sent.
"""
from __future__ import annotations

import bisect
import csv
import io
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class PriceBar:
    price: float
    volume: float | None = None
    high: float | None = None
    low: float | None = None
    bid: float | None = None
    ask: float | None = None


class PriceAdapter(ABC):
    @abstractmethod
    def price_at(self, instrument: str, ts: datetime) -> PriceBar | None:
        """The latest public bar known at `ts` (no lookahead), or None.

        None means "no honest price": the trading fields that need it become
        None, never a carried-forward or fabricated number.
        """

    def refresh(self, instruments: list[str]) -> None:
        """Best-effort fetch of new public data; never raises."""


class StaticPriceAdapter(PriceAdapter):
    """In-memory series, for tests and for partners who mirror public data themselves."""

    def __init__(self, series: dict[str, list[tuple[datetime, PriceBar]]],
                 max_staleness: timedelta | None = None) -> None:
        self._ts = {k: [t for t, _ in sorted(v, key=lambda x: x[0])] for k, v in series.items()}
        self._bars = {k: [b for _, b in sorted(v, key=lambda x: x[0])] for k, v in series.items()}
        self.max_staleness = max_staleness

    def price_at(self, instrument: str, ts: datetime) -> PriceBar | None:
        keys = self._ts.get(instrument)
        if not keys:
            return None
        i = bisect.bisect_right(keys, ts) - 1
        if i < 0:
            return None
        if self.max_staleness is not None and ts - keys[i] > self.max_staleness:
            return None
        return self._bars[instrument][i]


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def _http_get_text(url: str, timeout: float = 15.0) -> str:
    import httpx
    r = httpx.get(url, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return r.text


class FredDailyAdapter(PriceAdapter):
    """Daily closes from FRED's public CSV endpoint (no key, no cost).

    A FRED observation dated D is the close of day D, published after it. So
    for a timestamp `ts`, the bar used is the latest observation dated STRICTLY
    BEFORE `ts`'s calendar day: the close of the current day is not known yet.
    Daily data therefore only supports cadences of a day or more; at a 5-minute
    cadence every bar in a window shares one price and the fields that need a
    price MOVE come out None (hit rate excludes zero moves). That is the honest
    outcome, and the reason intraday providers sit behind `IntradayStubAdapter`.

    The series is cached on disk. A failed refresh keeps serving the cache; a
    bar older than `max_staleness_days` is not used.
    """

    def __init__(self, instrument_series: dict[str, str], cache_dir: str | Path, *,
                 max_staleness_days: int = 5, refresh_every_seconds: float = 6 * 3600,
                 fetch: Callable[[str], str] | None = None) -> None:
        self.series = dict(instrument_series)
        self.cache_dir = Path(cache_dir)
        self.max_staleness_days = max_staleness_days
        self.refresh_every = refresh_every_seconds
        self._fetch = fetch or _http_get_text
        self._data: dict[str, tuple[list[date], list[float]]] = {}
        self._fetched_at: dict[str, float] = {}
        self.last_error: str | None = None
        for inst in self.series:
            self._load_cache(inst)

    @staticmethod
    def _parse(text: str) -> tuple[list[date], list[float]]:
        days: list[date] = []
        vals: list[float] = []
        rows = csv.reader(io.StringIO(text))
        next(rows, None)                       # header
        for row in rows:
            if len(row) < 2 or row[1] in ("", "."):
                continue                       # FRED marks holidays/missing with "."
            try:
                d, v = date.fromisoformat(row[0]), float(row[1])
            except ValueError:
                continue
            if v > 0:
                days.append(d)
                vals.append(v)
        return days, vals

    def _cache_path(self, inst: str) -> Path:
        return self.cache_dir / f"{self.series[inst]}.csv"

    def _load_cache(self, inst: str) -> None:
        try:
            text = self._cache_path(inst).read_text()
        except OSError:
            return
        self._data[inst] = self._parse(text)

    def refresh(self, instruments: list[str]) -> None:
        now = time.time()
        for inst in instruments:
            sid = self.series.get(inst)
            if sid is None or now - self._fetched_at.get(inst, 0.0) < self.refresh_every:
                continue
            self._fetched_at[inst] = now       # also rate-limits retries after a failure
            try:
                text = self._fetch(FRED_CSV_URL.format(series=sid))
                parsed = self._parse(text)
                if not parsed[0]:
                    raise ValueError("empty FRED series")
            except Exception as exc:           # noqa: BLE001 -- offline is a normal state
                self.last_error = f"{type(exc).__name__}: {exc}"
                continue
            self._data[inst] = parsed
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_path(inst).with_suffix(".tmp")
            tmp.write_text(text)
            os.replace(tmp, self._cache_path(inst))

    def price_at(self, instrument: str, ts: datetime) -> PriceBar | None:
        data = self._data.get(instrument)
        if not data or not data[0]:
            return None
        days, vals = data
        today = ts.astimezone(timezone.utc).date()
        i = bisect.bisect_left(days, today) - 1          # strictly before today
        if i < 0 or (today - days[i]).days > self.max_staleness_days:
            return None
        return PriceBar(price=vals[i])


class IntradayStubAdapter(PriceAdapter):
    """Placeholder for an intraday public-price provider.

    No free, licence-clean intraday source has been chosen (the brief allows no
    paid source). Wire one by subclassing `PriceAdapter`; until then this
    returns None so intraday price-dependent fields are absent, not invented.
    """

    def __init__(self, provider: str = "unconfigured") -> None:
        self.provider = provider

    def price_at(self, instrument: str, ts: datetime) -> PriceBar | None:
        return None
