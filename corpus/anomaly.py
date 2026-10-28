"""
corpus/anomaly.py

Three anomaly detectors: a batch tabular baseline, a genuinely
streaming one, and an extreme-value threshold method.

HS-Trees is implemented here rather than pulled from a library because its
whole justification is a MEMORY GUARANTEE: 25 depth-8 trees is 12,775 mass
counters allocated once, O(1) per record, no allocation, no eviction. That
property is what made it beat RRCF and online KDE against a hard 2GB edge
ceiling, and it only holds if the implementation actually behaves that way.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["IsolationForestDetector", "HalfSpaceTrees", "EVTThreshold",
           "IF_SUBSAMPLE", "HST_TREES", "HST_DEPTH", "EVT_MIN_EXCEEDANCES"]

IF_SUBSAMPLE = 256      # standard; larger degrades contrast on clustered anomalies
IF_TREES = 100
HST_TREES = 25
HST_DEPTH = 8
EVT_FIT_QUANTILE = 0.98
EVT_MIN_EXCEEDANCES = 100
EVT_TARGET_FPR = 1e-4


# ---------------------------------------------------------------------------
# 1. Isolation Forest -- batch tabular baseline
# ---------------------------------------------------------------------------

class IsolationForestDetector:
    """Refit ONCE per reference window, never per emission window.

    It is a batch method; refitting per window would make the score compare
    against a moving baseline -- the same error PCA avoids by freezing its fit.
    """

    def __init__(self, seed: int = 0):
        self._model = None
        self.seed = seed

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        from sklearn.ensemble import IsolationForest
        X = np.asarray(X, dtype=float)
        if len(X) < IF_SUBSAMPLE:
            raise ValueError(
                f"Isolation Forest needs >= {IF_SUBSAMPLE} reference rows "
                f"(subsample size), got {len(X)}")
        self._model = IsolationForest(
            n_estimators=IF_TREES, max_samples=IF_SUBSAMPLE,
            random_state=self.seed).fit(X)
        return self

    def score(self, X: np.ndarray) -> dict[str, float]:
        """Returns the window MEAN and the FRACTION above 0.6.

        The mean alone hides a small number of extreme outliers -- which is
        exactly the case anomaly detection exists for, so both are reported.
        """
        if self._model is None:
            raise RuntimeError("IsolationForestDetector.score before fit")
        # sklearn's score_samples is negative-anomaly; map to [0,1] rising
        raw = -self._model.score_samples(np.asarray(X, dtype=float))
        s = 1.0 / (1.0 + np.exp(-4.0 * (raw - 0.5)))
        return {"mean": float(np.mean(s)),
                "fraction_above_0_6": float(np.mean(s > 0.6))}


# ---------------------------------------------------------------------------
# 2. Half-Space Trees -- the streaming detector
# ---------------------------------------------------------------------------

@dataclass
class _HSNode:
    __slots__ = ("split_dim", "split_val", "left", "right", "ref_mass", "cur_mass")
    split_dim: int
    split_val: float
    left: "_HSNode | None"
    right: "_HSNode | None"
    ref_mass: int
    cur_mass: int


class HalfSpaceTrees:
    """Streaming mass-based anomaly detector. O(1) update, fixed memory.

    Keeps COUNTS, not data -- which is what makes it streaming without a
    retained sample, and what makes its footprint constant by construction
    rather than bounded by policy.

    Two alternating mass profiles: at each window boundary the current profile
    becomes the reference and the current resets. A point landing in a
    low-mass region of the REFERENCE profile is anomalous.
    """

    def __init__(self, n_features: int, *, n_trees: int = HST_TREES,
                 depth: int = HST_DEPTH, seed: int = 0):
        self.n_features = n_features
        self.n_trees = n_trees
        self.depth = depth
        self._rng = np.random.default_rng(seed)
        self._roots = [self._build(0, np.zeros(n_features), np.ones(n_features))
                       for _ in range(n_trees)]
        self._ref_total = 0
        self._cur_total = 0
        self._lo = None
        self._hi = None

    # -- construction

    def _build(self, d: int, lo: np.ndarray, hi: np.ndarray) -> _HSNode:
        if d >= self.depth:
            return _HSNode(-1, 0.0, None, None, 0, 0)
        dim = int(self._rng.integers(self.n_features))
        val = float(self._rng.uniform(lo[dim], hi[dim]))
        llo, lhi = lo.copy(), hi.copy()
        rlo, rhi = lo.copy(), hi.copy()
        lhi[dim] = val
        rlo[dim] = val
        return _HSNode(dim, val, self._build(d + 1, llo, lhi),
                       self._build(d + 1, rlo, rhi), 0, 0)

    def set_range(self, lo: np.ndarray, hi: np.ndarray) -> None:
        """Feature ranges from the reference window. Rebuilds splits inside
        the observed range -- random splits over [0,1] on unscaled data would
        put every point in one leaf."""
        self._lo = np.asarray(lo, dtype=float)
        self._hi = np.asarray(hi, dtype=float)
        span = self._hi - self._lo
        span[span == 0] = 1.0
        self._hi = self._lo + span
        self._roots = [self._build(0, self._lo, self._hi)
                       for _ in range(self.n_trees)]

    # -- streaming update

    def update(self, x: np.ndarray) -> None:
        """One record. `depth` increments per tree, no allocation."""
        for root in self._roots:
            node = root
            while node.left is not None:
                node.cur_mass += 1
                node = node.left if x[node.split_dim] < node.split_val else node.right
            node.cur_mass += 1
        self._cur_total += 1

    def roll_window(self) -> None:
        """Current profile becomes reference; current resets. This is the only
        place memory is touched at a window boundary -- no reallocation."""
        for root in self._roots:
            self._roll(root)
        self._ref_total = self._cur_total
        self._cur_total = 0

    def _roll(self, node: _HSNode) -> None:
        node.ref_mass = node.cur_mass
        node.cur_mass = 0
        if node.left is not None:
            self._roll(node.left)
            self._roll(node.right)

    def score(self, x: np.ndarray) -> float:
        """Anomaly score in [0,1]. Low reference mass along the path = anomalous."""
        if self._ref_total == 0:
            raise RuntimeError("HalfSpaceTrees.score before the first roll_window")
        total = 0.0
        for root in self._roots:
            node, d = root, 0
            while node.left is not None:
                node = node.left if x[node.split_dim] < node.split_val else node.right
                d += 1
            # mass scaled by depth: a deep low-mass leaf is more anomalous
            total += node.ref_mass * (2 ** d)
        avg = total / self.n_trees
        expected = self._ref_total
        return float(1.0 / (1.0 + avg / max(expected, 1)))

    def memory_counters(self) -> int:
        """The number this detector was chosen for: fixed at construction."""
        return self.n_trees * (2 ** (self.depth + 1) - 1)


# ---------------------------------------------------------------------------
# 3. EVT -- Peaks-Over-Threshold with a GPD fit
# ---------------------------------------------------------------------------

class EVTThreshold:
    """Replaces hand-set thresholds on operational metric streams.

    The point is a MEASURABLE false-positive rate. Fitting a GPD on too few
    exceedances gives an unstable shape parameter, which gives a wrong
    threshold -- i.e. a differently-arbitrary hand-set constant, which is what
    this exists to remove. Hence the hard >= 100 exceedance requirement.
    """

    def __init__(self, target_fpr: float = EVT_TARGET_FPR):
        self.target_fpr = target_fpr
        self.fit_threshold: float | None = None
        self.shape: float | None = None
        self.scale: float | None = None
        self.alarm_threshold: float | None = None
        self.n_exceedances = 0

    def fit(self, sample: Sequence[float]) -> bool:
        """Returns False when there is not enough tail -- never a fabricated
        threshold."""
        x = np.sort(np.asarray(sample, dtype=float))
        if len(x) < 10:
            return False
        t = float(np.quantile(x, EVT_FIT_QUANTILE))
        exc = x[x > t] - t
        self.n_exceedances = len(exc)
        if self.n_exceedances < EVT_MIN_EXCEEDANCES:
            self.fit_threshold = t
            return False

        xi, sigma = _fit_gpd(exc)
        self.fit_threshold, self.shape, self.scale = t, xi, sigma

        n, nt = len(x), self.n_exceedances
        q = self.target_fpr
        if abs(xi) < 1e-8:
            z = t + sigma * math.log(nt / (q * n))
        else:
            z = t + (sigma / xi) * (((q * n / nt) ** -xi) - 1)
        self.alarm_threshold = float(z)
        return True

    def exceeds(self, value: float) -> bool | None:
        if self.alarm_threshold is None:
            return None
        return value > self.alarm_threshold

    def expected_false_alarms_per_window(self, records_per_window: int) -> float:
        """The number that makes the alert-fatigue claim checkable."""
        return self.target_fpr * records_per_window


def _fit_gpd(exceedances: np.ndarray) -> tuple[float, float]:
    """GPD shape and scale by method of moments, refined by grid MLE.

    Moments give a stable starting point; a coarse likelihood grid refines it
    without a dependency on an optimiser. Adequate for a threshold, and its
    stability matters more than the last decimal.
    """
    m, v = float(np.mean(exceedances)), float(np.var(exceedances))
    if v <= 0:
        return 0.0, max(m, 1e-9)
    xi0 = 0.5 * (1 - m * m / v)
    sigma0 = 0.5 * m * (m * m / v + 1)

    best, best_ll = (xi0, max(sigma0, 1e-9)), -np.inf
    for xi in np.linspace(max(xi0 - 0.4, -0.5), xi0 + 0.4, 25):
        for sc in np.linspace(max(sigma0 * 0.5, 1e-9), sigma0 * 1.5, 25):
            ll = _gpd_loglik(exceedances, xi, sc)
            if ll > best_ll:
                best_ll, best = ll, (float(xi), float(sc))
    return best


def _gpd_loglik(y: np.ndarray, xi: float, sigma: float) -> float:
    if sigma <= 0:
        return -np.inf
    if abs(xi) < 1e-8:
        return float(-len(y) * math.log(sigma) - np.sum(y) / sigma)
    z = 1 + xi * y / sigma
    if np.any(z <= 0):
        return -np.inf
    return float(-len(y) * math.log(sigma) - (1 + 1 / xi) * np.sum(np.log(z)))
