"""
corpus/trading_signals.py

Layer 1 (edge) signals for trading-model deployments.

WHERE THIS RUNS. On the deployment side, inside the edge agent. It reads the
model's own outputs and PUBLIC prices of the instrument. Raw predictions,
prices and rows never leave: only the scalars returned by
`compute_trading_signals` do, as extra fields of `z_t` (registered in
`corpus.serving.wire_fields` through `TRADING_SIGNAL_FIELDS`).

WHAT IT IS. Pure functions over the rows of ONE window. A row is a mapping:

    pred    the model's output for this bar (return forecast, signed score,
            or position) -- required for the performance/behaviour families
    price   public close for the bar, ordered by time
    high, low, volume, bid, ask   optional public market fields

CAUSALITY. The value for window t is a function of the rows inside window t
and of the previous `k-1` windows' scalars (rolling fields only). Nothing from
a later window is read. The one place a "future" bar is used is the realised
move that follows a prediction: for bar i the move is `p_{i+1}/p_i - 1`, with
`i+1` inside the same window, so the last bar of a window has no realised
outcome yet and is dropped. This is what a live agent can know at window end.

FORMULAS (all instrument-agnostic: scale-free or in return units)
-----------------------------------------------------------------
Let p_i be closes, r_i = p_{i+1}/p_i - 1 the forward return, l_i = ln(p_i/p_{i-1})
the log return, s_i = pred_i, over the n-1 bars that have a realised move.

Realised performance
  tr_hit_rate            mean_i [ sign(s_i) == sign(r_i) ] over bars with
                         s_i != 0 and r_i != 0. Needs >= MIN_PAIRS pairs.
  tr_hit_rate_rolling    sum_k hits_k / sum_k n_k over the last ROLLING_K
                         windows, current included (pair-weighted).
  tr_realised_error      mean_i |s_i - r_i| / (mean_i |r_i| + EPS). Scale-free
                         MAE of the forecast against the realised move,
                         in units of the typical move.
  tr_signal_move_corr    Pearson corr(s_i, r_i).
  tr_signal_move_ic      Spearman rank corr(s_i, r_i) (information coefficient),
                         average ranks for ties.

Market regime (public data only)
  tr_realised_vol        sample std (ddof=1) of l_i, per bar (not annualised).
  tr_amihud_illiquidity  mean_i |l_i| / (p_i * volume_i) over bars with
                         volume_i > 0 (Amihud 2002).
  tr_log_volume_ratio    ln( mean(volume) / ref_volume ); ref_volume is a
                         frozen reference mean supplied by the caller.
  tr_spread_proxy        mean relative quoted spread (ask-bid)/mid when bid
                         and ask exist, else the Corwin-Schultz (2012)
                         high-low estimator, negatives floored at 0:
                           beta  = ln(H_t/L_t)^2 + ln(H_{t+1}/L_{t+1})^2
                           gamma = ln(max(H_t,H_{t+1}) / min(L_t,L_{t+1}))^2
                           alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt 2)
                                   - sqrt(gamma / (3 - 2 sqrt 2))
                           S     = 2 (e^alpha - 1) / (1 + e^alpha)
                         averaged over consecutive bar pairs.
  tr_cross_asset_corr    mean off-diagonal Pearson correlation of the log
                         returns of the peer instruments (>= 2 instruments).
  tr_cross_asset_corr_rolling   mean of that over the last ROLLING_K windows.

Output behaviour
  tr_turnover            mean_i |s_i - s_{i-1}| / (mean_i |s_i| + EPS).
  tr_signal_autocorr     lag-1 Pearson autocorrelation of s over the window.
  tr_output_scale_drift  ln( rms(s) / ref_pred_rms ); ref_pred_rms is the
                         frozen reference rms of the model's output.

Any field that cannot be computed honestly (too few rows, zero variance,
missing inputs) is `None`, never a fabricated 0.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "TRADING_SIGNAL_FIELDS", "MIN_PAIRS", "ROLLING_K", "RollingState",
    "forward_returns", "log_returns", "hit_rate", "realised_error",
    "pearson", "spearman", "realised_vol", "amihud_illiquidity",
    "log_volume_ratio", "spread_proxy", "corwin_schultz_spread",
    "cross_asset_corr", "turnover", "signal_autocorr", "output_scale_drift",
    "compute_trading_signals",
]

MIN_PAIRS = 5        # fewer (pred, move) pairs than this: the field is None
ROLLING_K = 12       # windows in the rolling hit rate / correlation
EPS = 1e-12

TRADING_SIGNAL_FIELDS: tuple[str, ...] = (
    "tr_hit_rate", "tr_hit_rate_rolling", "tr_realised_error",
    "tr_signal_move_corr", "tr_signal_move_ic",
    "tr_realised_vol", "tr_amihud_illiquidity", "tr_log_volume_ratio",
    "tr_spread_proxy", "tr_cross_asset_corr", "tr_cross_asset_corr_rolling",
    "tr_turnover", "tr_signal_autocorr", "tr_output_scale_drift",
)


def _arr(x: Sequence[float] | None) -> np.ndarray:
    if x is None:
        return np.zeros(0)
    return np.asarray([np.nan if v is None else v for v in x], dtype=float)


def forward_returns(prices: Sequence[float]) -> np.ndarray:
    """r_i = p_{i+1}/p_i - 1, length n-1. Non-positive prices give NaN."""
    p = _arr(prices)
    if len(p) < 2:
        return np.zeros(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = p[1:] / p[:-1] - 1.0
    r[(p[:-1] <= 0) | (p[1:] <= 0)] = np.nan
    return r


def log_returns(prices: Sequence[float]) -> np.ndarray:
    """l_i = ln(p_i / p_{i-1}), length n-1. Non-positive prices give NaN."""
    p = _arr(prices)
    if len(p) < 2:
        return np.zeros(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        l = np.log(p[1:] / p[:-1])
    l[(p[:-1] <= 0) | (p[1:] <= 0)] = np.nan
    return l


def _pairs(preds: Sequence[float], prices: Sequence[float]
           ) -> tuple[np.ndarray, np.ndarray]:
    """(s_i, r_i) for bars that have a realised move, finite pairs only."""
    s, r = _arr(preds), forward_returns(prices)
    n = min(len(s) - 1, len(r)) if len(s) else 0
    if n <= 0:
        return np.zeros(0), np.zeros(0)
    s, r = s[:n], r[:n]
    ok = np.isfinite(s) & np.isfinite(r)
    return s[ok], r[ok]


def hit_rate(s: np.ndarray, r: np.ndarray) -> tuple[float | None, int]:
    """(hit rate, pair count). Zero predictions and zero moves carry no
    directional claim and are excluded."""
    m = (s != 0) & (r != 0)
    n = int(m.sum())
    if n < MIN_PAIRS:
        return None, n
    return float(np.mean(np.sign(s[m]) == np.sign(r[m]))), n


def realised_error(s: np.ndarray, r: np.ndarray) -> float | None:
    if len(s) < MIN_PAIRS:
        return None
    return float(np.mean(np.abs(s - r)) / (np.mean(np.abs(r)) + EPS))


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < MIN_PAIRS or len(a) != len(b):
        return None
    sa, sb = np.std(a), np.std(b)
    if sa < EPS or sb < EPS:
        return None
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < MIN_PAIRS or len(a) != len(b):
        return None
    return pearson(_rank(a), _rank(b))


def realised_vol(prices: Sequence[float]) -> float | None:
    l = log_returns(prices)
    l = l[np.isfinite(l)]
    if len(l) < MIN_PAIRS:
        return None
    return float(np.std(l, ddof=1))


def amihud_illiquidity(prices: Sequence[float],
                       volumes: Sequence[float] | None) -> float | None:
    if volumes is None:
        return None
    p, v = _arr(prices), _arr(volumes)
    l = log_returns(p)
    n = min(len(l), len(v) - 1, len(p) - 1)
    if n < MIN_PAIRS:
        return None
    # bar i (i>=1) carries return l[i-1] and its own dollar volume p[i]*v[i]
    lr, pi, vi = l[:n], p[1:n + 1], v[1:n + 1]
    ok = np.isfinite(lr) & np.isfinite(pi) & np.isfinite(vi) & (vi > 0) & (pi > 0)
    if int(ok.sum()) < MIN_PAIRS:
        return None
    return float(np.mean(np.abs(lr[ok]) / (pi[ok] * vi[ok])))


def log_volume_ratio(volumes: Sequence[float] | None,
                     ref_volume: float | None) -> float | None:
    if volumes is None or ref_volume is None or ref_volume <= 0:
        return None
    v = _arr(volumes)
    v = v[np.isfinite(v)]
    if len(v) < MIN_PAIRS or v.mean() <= 0:
        return None
    return float(math.log(v.mean() / ref_volume))


_CS_K = 3.0 - 2.0 * math.sqrt(2.0)


def corwin_schultz_spread(high: Sequence[float], low: Sequence[float]
                          ) -> float | None:
    h, l = _arr(high), _arr(low)
    n = min(len(h), len(l))
    if n < 2:
        return None
    h, l = h[:n], l[:n]
    out = []
    for t in range(n - 1):
        H0, L0, H1, L1 = h[t], l[t], h[t + 1], l[t + 1]
        if not all(np.isfinite([H0, L0, H1, L1])) or min(L0, L1) <= 0 \
                or H0 < L0 or H1 < L1:
            continue
        beta = math.log(H0 / L0) ** 2 + math.log(H1 / L1) ** 2
        gamma = math.log(max(H0, H1) / min(L0, L1)) ** 2
        alpha = ((math.sqrt(2 * beta) - math.sqrt(beta)) / _CS_K
                 - math.sqrt(gamma / _CS_K))
        s = 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))
        out.append(max(0.0, s))
    if len(out) < MIN_PAIRS - 1:
        return None
    return float(np.mean(out))


def spread_proxy(bid: Sequence[float] | None, ask: Sequence[float] | None,
                 high: Sequence[float] | None, low: Sequence[float] | None
                 ) -> float | None:
    if bid is not None and ask is not None:
        b, a = _arr(bid), _arr(ask)
        n = min(len(b), len(a))
        b, a = b[:n], a[:n]
        ok = np.isfinite(b) & np.isfinite(a) & (a >= b) & (b > 0)
        if int(ok.sum()) >= MIN_PAIRS:
            mid = (a[ok] + b[ok]) / 2.0
            return float(np.mean((a[ok] - b[ok]) / mid))
    if high is not None and low is not None:
        return corwin_schultz_spread(high, low)
    return None


def cross_asset_corr(peer_prices: Mapping[str, Sequence[float]] | None
                     ) -> float | None:
    if not peer_prices or len(peer_prices) < 2:
        return None
    rets = [log_returns(p) for p in peer_prices.values()]
    n = min(len(x) for x in rets)
    if n < MIN_PAIRS:
        return None
    M = np.vstack([x[-n:] for x in rets])           # aligned on the latest bars
    M = M[:, np.all(np.isfinite(M), axis=0)]
    if M.shape[1] < MIN_PAIRS:
        return None
    sd = M.std(axis=1)
    if np.any(sd < EPS):
        return None
    C = np.corrcoef(M)
    iu = np.triu_indices(len(C), k=1)
    return float(np.mean(C[iu]))


def turnover(preds: Sequence[float]) -> float | None:
    s = _arr(preds)
    s = s[np.isfinite(s)]
    if len(s) < MIN_PAIRS:
        return None
    return float(np.mean(np.abs(np.diff(s))) / (np.mean(np.abs(s)) + EPS))


def signal_autocorr(preds: Sequence[float], lag: int = 1) -> float | None:
    s = _arr(preds)
    s = s[np.isfinite(s)]
    if len(s) < MIN_PAIRS + lag:
        return None
    return pearson(s[:-lag], s[lag:])


def output_scale_drift(preds: Sequence[float], ref_rms: float | None
                       ) -> float | None:
    if ref_rms is None or ref_rms <= 0:
        return None
    s = _arr(preds)
    s = s[np.isfinite(s)]
    if len(s) < MIN_PAIRS:
        return None
    rms = float(np.sqrt(np.mean(s ** 2)))
    return float(math.log(rms / ref_rms)) if rms > 0 else None


@dataclass
class RollingState:
    """Per-model memory for the rolling fields: scalars of past windows only.

    `push` is called once per window AFTER that window's scalars are known, so
    the rolling value for window t uses windows t-k+1..t and never later ones.
    """
    k: int = ROLLING_K
    hits: deque = field(default_factory=deque)      # (hits, n) pairs
    corrs: deque = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.hits = deque(self.hits, maxlen=self.k)
        self.corrs = deque(self.corrs, maxlen=self.k)


def compute_trading_signals(
        rows: Sequence[Mapping[str, Any]], *,
        peer_prices: Mapping[str, Sequence[float]] | None = None,
        ref_volume: float | None = None,
        ref_pred_rms: float | None = None,
        rolling: RollingState | None = None,
) -> dict[str, float | None]:
    """All `TRADING_SIGNAL_FIELDS` for one window of rows.

    `rolling` (optional) is mutated: this window's hit count and cross-asset
    correlation are appended after the window's own values are computed.
    """
    def col(name: str) -> list[float] | None:
        vals = [r.get(name) for r in rows]
        return vals if any(v is not None for v in vals) else None

    preds, prices = col("pred"), col("price")
    out: dict[str, float | None] = {f: None for f in TRADING_SIGNAL_FIELDS}
    if prices is None:
        return out

    n_hit = 0
    if preds is not None:
        s, r = _pairs(preds, prices)
        hr, n_hit = hit_rate(s, r)
        out["tr_hit_rate"] = hr
        out["tr_realised_error"] = realised_error(s, r)
        out["tr_signal_move_corr"] = pearson(s, r)
        out["tr_signal_move_ic"] = spearman(s, r)
        out["tr_turnover"] = turnover(preds)
        out["tr_signal_autocorr"] = signal_autocorr(preds)
        out["tr_output_scale_drift"] = output_scale_drift(preds, ref_pred_rms)

    vol = col("volume")
    out["tr_realised_vol"] = realised_vol(prices)
    out["tr_amihud_illiquidity"] = amihud_illiquidity(prices, vol)
    out["tr_log_volume_ratio"] = log_volume_ratio(vol, ref_volume)
    out["tr_spread_proxy"] = spread_proxy(col("bid"), col("ask"),
                                          col("high"), col("low"))
    cac = cross_asset_corr(peer_prices)
    out["tr_cross_asset_corr"] = cac

    if rolling is not None:
        if out["tr_hit_rate"] is not None:
            rolling.hits.append((out["tr_hit_rate"] * n_hit, n_hit))
        if cac is not None:
            rolling.corrs.append(cac)
        tot = sum(n for _, n in rolling.hits)
        if tot > 0:
            out["tr_hit_rate_rolling"] = float(
                sum(h for h, _ in rolling.hits) / tot)
        if rolling.corrs:
            out["tr_cross_asset_corr_rolling"] = float(np.mean(rolling.corrs))
    return out
