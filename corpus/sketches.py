"""
corpus/sketches.py

The two streaming sketches fixed for the wire contract:

  * t-digest, compression delta = 100, ~4 KB worst case  -- continuous fields
  * HyperLogLog, precision p = 14 (16,384 registers), 12 KB dense -- categoricals

These close the two fixtures the shared conformance runner leaves as stubs
(`sketch_tdigest_accuracy`, `sketch_hll_cardinality`).
This module closes them.

They matter beyond accuracy: the non-content guarantee argues that a
sketch transmits distributional *shape* and not recoverable individual values.
That argument is only true of a specific structure at a specific resolution,
so the structure has to exist and be tested rather than assumed.
"""
from __future__ import annotations

import hashlib
import math
from typing import Iterable

__all__ = ["TDigest", "HyperLogLog"]


class TDigest:
    """Merging t-digest.

    Centroid density scales with 1/(q(1-q)), so accuracy concentrates at the
    tails -- which is what the contract wants for p95/p99 drift on score
    distributions, where the interesting movement is in the tail rather than
    the middle.

    This is the simple merging variant: buffer, sort, merge with a size bound
    from the scale function. Adequate for the corpus's batch replay; a
    production edge agent under memory pressure would want the clustering
    variant.
    """

    def __init__(self, compression: float = 100.0):
        self.compression = float(compression)
        self._centroids: list[list[float]] = []   # [mean, weight]
        self._buffer: list[float] = []
        self._count = 0

    def push(self, x: float) -> None:
        self._buffer.append(float(x))
        self._count += 1
        if len(self._buffer) >= 1000:
            self._flush()

    def extend(self, xs: Iterable[float]) -> "TDigest":
        for x in xs:
            self.push(x)
        return self

    def _flush(self) -> None:
        if not self._buffer:
            return
        pts = [[x, 1.0] for x in self._buffer]
        self._buffer.clear()
        merged = sorted(self._centroids + pts, key=lambda c: c[0])
        total = sum(c[1] for c in merged)
        if total == 0:
            self._centroids = []
            return

        out: list[list[float]] = []
        weight_before_current = 0.0
        for mean, w in merged:
            if not out:
                out.append([mean, w])
                continue
            cm, cw = out[-1]
            # q0 is the quantile position where the CURRENT centroid starts.
            # The bound applies to that centroid's own weight, not to the
            # running total -- comparing against cumulative weight makes the
            # limit unreachable after the first few points and the digest
            # never merges at all.
            q0 = weight_before_current / total
            limit = self._q_limit(q0) * total
            if cw + w <= limit:
                nw = cw + w
                out[-1] = [cm + (mean - cm) * w / nw, nw]
            else:
                weight_before_current += cw
                out.append([mean, w])
        self._centroids = out

    def _q_limit(self, q0: float) -> float:
        """k-scale bound: max weight fraction for a centroid starting at q0.

        The 4*q(1-q) term is what concentrates resolution at the tails: near
        q=0.5 a centroid may absorb ~2% of total weight at delta=100, near the
        tails only ~1%. That asymmetry is the reason t-digest was chosen
        over KLL -- p95/p99 drift on score distributions is where the
        interesting movement is.
        """
        k = self.compression
        return 4 * (q0 * (1 - q0)) / k + 1.0 / k

    def quantile(self, q: float) -> float:
        """Interpolated quantile estimate, q in [0, 1].

        Interpolates between centroid *centres*, not boundaries. A centroid of
        weight w centred at cumulative position (prior_weight + w/2) is the
        best single estimate of that chunk of the distribution; interpolating
        on boundaries instead systematically pulls estimates toward the lower
        edge of each centroid and costs roughly an order of magnitude in
        accuracy at delta=100.
        """
        self._flush()
        cs = self._centroids
        if not cs:
            raise ValueError("quantile() on an empty digest")
        if len(cs) == 1:
            return cs[0][0]

        total = sum(c[1] for c in cs)
        target = q * total

        centres: list[tuple[float, float]] = []
        cum = 0.0
        for mean, w in cs:
            centres.append((cum + w / 2.0, mean))
            cum += w

        if target <= centres[0][0]:
            return cs[0][0]
        if target >= centres[-1][0]:
            return cs[-1][0]
        for (c0, m0), (c1, m1) in zip(centres, centres[1:]):
            if c0 <= target <= c1:
                frac = (target - c0) / (c1 - c0) if c1 > c0 else 0.0
                return m0 + (m1 - m0) * frac
        return cs[-1][0]

    def serialized_size_bytes(self) -> int:
        """Two float64 per centroid. The number bandwidth is sized from."""
        self._flush()
        return len(self._centroids) * 16

    def __len__(self) -> int:
        return self._count


class HyperLogLog:
    """HyperLogLog cardinality sketch with linear-counting small-range correction.

    The correction is not optional here. The contract uses this for
    `cardinality_change` and `new_category_rate`, and real categorical
    features routinely have tens of distinct values -- exactly the range where
    raw HLL is badly biased and linear counting is accurate.
    """

    def __init__(self, precision: int = 14):
        if not 4 <= precision <= 18:
            raise ValueError("precision must be in [4, 18]")
        self.p = precision
        self.m = 1 << precision
        self._registers = bytearray(self.m)

    @staticmethod
    def _hash64(value) -> int:
        data = str(value).encode("utf-8")
        return int.from_bytes(hashlib.sha1(data).digest()[:8], "big")

    def add(self, value) -> None:
        h = self._hash64(value)
        idx = h >> (64 - self.p)
        w = (h << self.p) & ((1 << 64) - 1)
        rank = 1
        if w == 0:
            rank = 64 - self.p + 1
        else:
            while (w >> 63) & 1 == 0:
                rank += 1
                w = (w << 1) & ((1 << 64) - 1)
        if rank > self._registers[idx]:
            self._registers[idx] = rank

    def extend(self, values: Iterable) -> "HyperLogLog":
        for v in values:
            self.add(v)
        return self

    def _alpha(self) -> float:
        m = self.m
        if m == 16:
            return 0.673
        if m == 32:
            return 0.697
        if m == 64:
            return 0.709
        return 0.7213 / (1 + 1.079 / m)

    def count(self) -> int:
        m = self.m
        raw = self._alpha() * m * m / sum(2.0 ** -r for r in self._registers)
        zeros = self._registers.count(0)
        if raw <= 2.5 * m and zeros > 0:
            # small range: linear counting is far more accurate than raw HLL
            return int(round(m * math.log(m / zeros)))
        return int(round(raw))

    def serialized_size_bytes(self) -> int:
        """Dense encoding at 6 bits/register -- 12 KB at p=14, the contract's number."""
        return math.ceil(self.m * 6 / 8)

    def standard_error(self) -> float:
        return 1.04 / math.sqrt(self.m)
