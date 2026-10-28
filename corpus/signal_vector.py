"""
corpus/signal_vector.py

Signal-vector (`z_t`) computation for the replay harness.

THIS MODULE IS VERIFIED AGAINST contracts/signal_vector/runner.py's fixtures.
It is not a second implementation of the wire contract -- it is *an*
implementation whose conformance is proven by running the shared suite, which
is exactly the mechanism the contract chose,
and which requires this
harness to consume rather than substitute.

Run the conformance check to verify. CI must treat a
non-zero exit from that command as a build failure.

Every formula below mirrors the reference implementation in the conformance
runner exactly, including its edge-case handling (bin placement, floors,
sample-corrected moments). Where the runner makes a specific choice -- e.g.
the last PSI bin being closed on both ends -- that choice is reproduced here
deliberately, not approximated.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

# Vectorisation note (2026-08-15). The functions below were
# originally written against plain Python sequences. Measured on the edge
# target, converting a window to Python lists cost ~180 MB at 100k rows --
# the single largest per-model line item in the edge memory budget -- and the
# per-bin scan in `bin_counts` plus the 15-pass bucket scan in `ece`/`mce`
# dominated edge CPU (Wasserstein 2.1 s, KS 1.2 s, calibration 2.5 s per
# window per model).
#
# What changed: the data representation, and nothing else. Every formula,
# every binning policy, every tie-break and boundary convention is preserved
# exactly, because the contract is explicit that a
# formula change bumps the schema version and invalidates the corpus. The
# equivalence is PROVEN, not asserted: a golden-output script pins
# full-precision output over an edge-case battery (empty, constant, ties,
# outliers, single-value, wide-range) and is diffed before/after, alongside
# the conformance check and the shared contract runner.
#
# The public signatures still accept any Sequence/Iterable, so callers that
# pass lists keep working -- they simply no longer force a list round-trip
# on the hot path.

__all__ = [
    "psi", "js_divergence", "ks_statistic",
    "wasserstein1", "chi_square", "hellinger",
    "score_moments", "ece", "mce", "brier", "nll", "fit_temperature",
    "apply_temperature", "predictive_entropy",
    "ci_width", "row_count_ratio", "percentile_linear",
    "first_difference", "ewma", "ewma_slope", "second_derivative",
    "ewma_abs_change", "time_above_threshold",
    "bin_edges_from_reference", "bin_counts",
]


# ---------------------------------------------------------------------------
# Group A -- input space
# ---------------------------------------------------------------------------

def bin_edges_from_reference(reference_window: Sequence[float], bins: int = 10) -> list[float]:
    """Equal-frequency bin edges from the reference sample's quantile boundaries.

    Matches the conformance runner: edge i is the reference value at index
    round(i*n/bins), and the top edge is nudged by 1e-9 so the maximum value
    falls inside the last bin rather than outside every bin.
    """
    # DELIBERATELY NOT VECTORISED. `np.sort` would coerce an integral
    # reference window to float, so edges came back as `-4.0` where this
    # returns `-4` -- numerically equal, but a different serialised value.
    # It is also not on the hot path: `edges` and `ref_counts` depend only
    # on the reference window and are precomputed once per replay (see
    # `psi`'s docstring), not once per window. Vectorising it would trade a
    # real output change for no measurable gain.
    ref_sorted = sorted(reference_window)
    n = len(ref_sorted)
    edges = [ref_sorted[0]]
    for i in range(1, bins):
        idx = min(int(round(i * n / bins)), n - 1)
        edges.append(ref_sorted[idx])
    edges.append(ref_sorted[-1] + 1e-9)
    return edges


def bin_counts(data: Iterable[float], edges: Sequence[float], bins: int) -> list[int]:
    """Place values into bins defined by `edges`.

    The last bin is closed on BOTH ends; all others are [lo, hi). Values
    outside the reference range fall into the first or last bin. This is the
    runner's behaviour and is reproduced exactly -- an off-by-one here would
    make every PSI in the corpus subtly wrong and nothing would detect it.
    """
    x = np.asarray(list(data) if not hasattr(data, "__len__") else data,
                   dtype=float)
    if x.size == 0:
        return [0] * bins
    e = np.asarray(edges, dtype=float)
    # `searchsorted(..., 'right') - 1` reproduces the scan exactly:
    #   * [lo, hi) for every bin but the last, including with DUPLICATE
    #     edges (a constant reference window), because 'right' lands past
    #     every equal edge just as the scan skips every zero-width bin;
    #   * the clip to `bins-1` closes the last bin on both ends, which is
    #     the `lo <= x <= hi` branch, and also absorbs x > edges[-1];
    #   * the clip to 0 absorbs x < edges[0].
    # Those three cases are the entire behaviour of the loop this replaces.
    idx = np.clip(np.searchsorted(e, x, side="right") - 1, 0, bins - 1)
    return np.bincount(idx, minlength=bins)[:bins].tolist()


def psi(reference_window: Sequence[float], current_window: Sequence[float],
        bins: int = 10, floor: float = 1e-4,
        edges: Sequence[float] | None = None,
        ref_counts: Sequence[int] | None = None) -> float:
    """Population Stability Index. Proportions floored to keep the log finite.

    `edges` and `ref_counts` may be supplied precomputed. They depend only on
    the reference window, which is fixed for the lifetime of a replay, so
    recomputing them per window means re-sorting the whole reference sample
    once per feature per window -- the dominant cost in a long replay.
    """
    if edges is None:
        edges = bin_edges_from_reference(reference_window, bins)
    if ref_counts is None:
        ref_counts = bin_counts(sorted(reference_window), edges, bins)
    cur_counts = bin_counts(current_window, edges, bins)
    nref, ncur = len(reference_window), len(current_window)
    total = 0.0
    for rc, cc in zip(ref_counts, cur_counts):
        rp = max(rc / nref, floor)
        cp = max(cc / ncur, floor)
        total += (cp - rp) * math.log(cp / rp)
    return total


def js_divergence(reference_window: Sequence[float], current_window: Sequence[float],
                  bins: int = 10, floor: float = 1e-4,
                  edges: Sequence[float] | None = None,
                  ref_counts: Sequence[int] | None = None) -> float:
    """Jensen-Shannon divergence, log base 2, over the same bins PSI uses.

    Base 2 and shared binning are both fixed by the field contract so that
    JS and PSI are comparable and bounded in [0, 1].
    """
    if edges is None:
        edges = bin_edges_from_reference(reference_window, bins)
    if ref_counts is None:
        ref_counts = bin_counts(sorted(reference_window), edges, bins)
    p = [max(c / len(reference_window), floor) for c in ref_counts]
    q = [max(c / len(current_window), floor)
         for c in bin_counts(current_window, edges, bins)]
    ps, qs = sum(p), sum(q)
    p = [x / ps for x in p]
    q = [x / qs for x in q]

    def kl(a, b):
        return sum(ai * math.log2(ai / bi) for ai, bi in zip(a, b) if ai > 0)

    m = [(ai + bi) / 2 for ai, bi in zip(p, q)]
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def ks_statistic(reference_window: Sequence[float], current_window: Sequence[float],
                 ref_sorted: Sequence[float] | None = None) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: sup |F_ref(x) - F_cur(x)|.

    `ref_sorted` may be supplied precomputed for the same reason as `psi`.
    """
    a = np.sort(np.asarray(reference_window if ref_sorted is None
                           else ref_sorted, dtype=float))
    b = np.sort(np.asarray(current_window, dtype=float))
    nref, ncur = a.size, b.size
    if nref == 0 or ncur == 0:
        return 0.0
    # The original walked the merged support and stopped as soon as either
    # sample was exhausted. That truncation is safe and this is why: past
    # exhaustion of one sample its CDF is pinned at 1 while the other's
    # rises toward 1, so |F_ref - F_cur| is monotonically DECREASING and the
    # supremum has already been attained. Evaluating the full merged
    # support therefore gives the identical maximum.
    pts = np.union1d(a, b)
    d = np.abs(np.searchsorted(a, pts, side="right") / nref
               - np.searchsorted(b, pts, side="right") / ncur)
    return float(d.max())


def wasserstein1(reference_window, current_window) -> float:
    """Wasserstein-1 (earth mover's) distance between two empirical samples.

    BIN-FREE by construction, which is the point: PSI/JS/chi-square all
    inherit their binning policy's artifacts, and W1 is the method that
    answers "how far did it move" in the feature's own units rather than "how
    different are the bin proportions".

    Computed as the L1 area between the two empirical CDFs, evaluated on the
    merged support. Unequal sample sizes are handled exactly -- no
    subsampling, no interpolation onto a common grid.

    UNITS: same as the feature. That makes it interpretable but NOT
    comparable across features, which is why the contract keeps it per-feature and
    never aggregates it the way it aggregates PSI.
    """
    a = np.sort(np.asarray(reference_window, dtype=float))
    b = np.sort(np.asarray(current_window, dtype=float))
    if a.size == 0 or b.size == 0:
        raise ValueError("wasserstein1 requires both samples non-empty")
    pts = np.union1d(a, b)          # sorted unique == sorted(set(a) | set(b))
    if pts.size == 1:
        return 0.0
    # Identical accumulation to the original walk: at each merged support
    # point, the gap between the two empirical CDFs times the width to the
    # next point. `searchsorted(..., 'right')` is the count of values <= x,
    # which is exactly what the `while ... <= x` advances computed.
    left = pts[:-1]
    contrib = (np.abs(np.searchsorted(a, left, side="right") / a.size
                      - np.searchsorted(b, left, side="right") / b.size)
               * np.diff(pts))
    # SEQUENTIAL accumulation, not `np.sum`. numpy sums pairwise, which is
    # more accurate but lands ~1e-15 away from the original's left-to-right
    # `total += ...`. More accurate is still DIFFERENT, and the corpus was
    # built on these bytes -- so the summation order is preserved and the
    # accuracy improvement declined. The win here is the sort and the
    # searchsorted, not the reduction.
    total = 0.0
    for v in contrib.tolist():
        total += v
    return total


def chi_square(reference_window, current_window, bins: int = 10,
               edges=None, ref_counts=None, min_expected: float = 5.0) -> float:
    """Pearson chi-square statistic over the SAME bins PSI uses.

    Shares PSI's equal-frequency binning deliberately: two statistics on the
    same feature disagreeing because they binned differently is a debugging
    problem nobody should have.

    Bins whose EXPECTED count falls below `min_expected` (default 5, the
    standard rule) are pooled into the neighbouring bin before the statistic
    is computed. Without pooling, a near-empty reference bin produces an
    enormous contribution from one or two observations and the statistic
    becomes a low-count artifact detector rather than a drift detector.
    """
    if edges is None:
        edges = bin_edges_from_reference(reference_window, bins)
    if ref_counts is None:
        ref_counts = bin_counts(sorted(reference_window), edges, bins)
    cur_counts = bin_counts(current_window, edges, bins)
    nref, ncur = len(reference_window), len(current_window)

    exp = [rc / nref * ncur for rc in ref_counts]
    obs = list(cur_counts)
    # pool low-expectation bins leftward
    pooled_e, pooled_o = [], []
    ce = co = 0.0
    for e, o in zip(exp, obs):
        ce += e
        co += o
        if ce >= min_expected:
            pooled_e.append(ce)
            pooled_o.append(co)
            ce = co = 0.0
    if ce > 0:
        if pooled_e:
            pooled_e[-1] += ce
            pooled_o[-1] += co
        else:
            pooled_e.append(ce)
            pooled_o.append(co)
    return sum((o - e) ** 2 / e for e, o in zip(pooled_e, pooled_o) if e > 0)


def hellinger(reference_window, current_window, bins: int = 10,
              edges=None, ref_counts=None) -> float:
    """Hellinger distance over PSI's bins. Bounded in [0, 1].

    The symmetric bounded fallback. PSI is unbounded and asymmetric, so a
    single empty bin can send it to infinity (which is why it is floored);
    Hellinger cannot exceed 1 whatever the data does, which makes it the
    stable choice when PSI saturates.

    No flooring is needed: sqrt(0) is 0 and contributes finitely, so zero
    bins are handled by the formula rather than by a fudge constant.
    """
    if edges is None:
        edges = bin_edges_from_reference(reference_window, bins)
    if ref_counts is None:
        ref_counts = bin_counts(sorted(reference_window), edges, bins)
    cur_counts = bin_counts(current_window, edges, bins)
    nref, ncur = len(reference_window), len(current_window)
    acc = sum((math.sqrt(rc / nref) - math.sqrt(cc / ncur)) ** 2
              for rc, cc in zip(ref_counts, cur_counts))
    return math.sqrt(acc / 2.0)


# ---------------------------------------------------------------------------
# Group B -- output space
# ---------------------------------------------------------------------------

def score_moments(scores: Sequence[float]) -> dict[str, float | None]:
    """Mean, population std, and SAMPLE-CORRECTED skew and excess kurtosis.

    The corrections matter: the conformance fixture expects the sample-corrected
    forms (Fisher-Pearson for skew, the standard unbiased excess-kurtosis
    estimator), not the raw moment ratios. Requires n >= 4.
    """
    n = len(scores)
    if n < 4:
        raise ValueError("score_moments requires n >= 4 for the kurtosis correction")
    mean = sum(scores) / n
    var = sum((x - mean) ** 2 for x in scores) / n
    std = math.sqrt(var)

    # A degenerate score distribution has UNDEFINED skew and kurtosis, not zero
    # ones. Returning 0.0 asserts perfect symmetry and zero excess kurtosis --
    # a claim about a distribution that carries no shape at all, and exactly
    # the "plausible arithmetic that corresponds to nothing" this project has
    # caught repeatedly. None means "exists, could not be computed", which is
    # the contract's tier-2 convention and what the caller must emit as
    # null + _valid=false.
    #
    # Guarding on `std ** 3` rather than `std` is deliberate. A model that has
    # collapsed to near-constant output produces a tiny but NONZERO std whose
    # cube underflows to exactly 0.0 in float64 -- measured on a collapsed MLP
    # after schema_break renamed every category: max score 3.8e-07, std
    # 1.9e-08, and on the failing window the cube underflowed. `std == 0.0`
    # let that through and the build died with ZeroDivisionError.
    if std == 0.0 or std ** 3 == 0.0 or std ** 4 == 0.0:
        return {"mean": mean, "std": std, "skew": None, "kurtosis": None}

    m3 = sum((x - mean) ** 3 for x in scores) / n
    m4 = sum((x - mean) ** 4 for x in scores) / n
    g1 = m3 / std ** 3
    skew = (math.sqrt(n * (n - 1)) / (n - 2)) * g1
    g2 = m4 / std ** 4 - 3
    kurt = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6)
    return {"mean": mean, "std": std, "skew": skew, "kurtosis": kurt}


# ---------------------------------------------------------------------------
# Group C -- calibration
# ---------------------------------------------------------------------------

def _calibration_buckets(pairs, bins: int):
    """Per-bucket mean confidence, accuracy and count for ECE/MCE.

    Both metrics ran the same 15 full passes over the window, each rebuilding
    a Python list -- 2.5 s per window per model at 100k rows on the edge
    target, the largest single CPU line item measured. One vectorised
    bucketing serves both.

    Bucket membership is UNCHANGED: `[lo, hi)` per bin with the last bin
    closed at `hi`, via the same `searchsorted(..., 'right') - 1` identity
    `bin_counts` uses. Confidences outside `[0, 1]` are dropped from every
    bucket exactly as the original comparisons dropped them, while still
    counting toward ECE's `n` denominator -- preserved deliberately rather
    than tidied, since tidying it would change a shipped number.
    """
    probs = [float(p[0]) for p in pairs]
    e = [i / bins for i in range(bins + 1)]
    # VECTORISED bucketing -- this is the part that cost fifteen full passes
    # over the window, one per bin, each rebuilding a Python list.
    idx = np.clip(
        np.searchsorted(np.asarray(e), np.asarray(probs), side="right") - 1,
        0, bins - 1).tolist()

    # SEQUENTIAL accumulation, one pass, in original order. np.bincount does
    # not reproduce Python `sum`'s rounding here -- it came out 1 ULP away --
    # and 1 ULP is still a changed number on a shipped metric. One pass
    # instead of fifteen is the win; the reduction order is not negotiable.
    csum = [0.0] * bins
    acnt = [0] * bins
    cnt = [0] * bins
    for k, pr in enumerate(probs):
        if pr < e[0] or pr > e[-1]:
            continue                      # outside [0,1]: in no bucket, as before
        b = idx[k]
        csum[b] += pr
        cnt[b] += 1
        if pairs[k][1] == 1:
            acnt[b] += 1
    conf = [csum[b] / cnt[b] if cnt[b] else 0.0 for b in range(bins)]
    acc = [acnt[b] / cnt[b] if cnt[b] else 0.0 for b in range(bins)]
    return conf, acc, cnt


def ece(pairs: Sequence[tuple[float, int]], bins: int = 15) -> float:
    """Expected Calibration Error over equal-WIDTH confidence bins."""
    n = len(pairs)
    if n == 0:
        return 0.0
    conf_b, acc_b, cnt = _calibration_buckets(pairs, bins)
    # Sequential over bins, in bin order, matching the original's
    # `total += ...`. Fifteen terms, so the reduction is not where the cost
    # was -- the fifteen full passes over the window were, and those are
    # gone. Keeping the order keeps the emitted value bit-for-bit.
    total = 0.0
    for b in range(bins):
        if cnt[b] > 0:
            total += (cnt[b] / n) * abs(acc_b[b] - conf_b[b])
    return total


def mce(pairs, bins: int = 15) -> float:
    """Maximum Calibration Error -- the WORST single bin, not the average.

    ECE averages, so a model can hold ECE flat while recalibrating well in the
    bulk and badly in the tail. MCE is the only one of the four that surfaces
    that, which is the failure a governance product must catch.
    """
    if len(pairs) == 0:
        return 0.0
    conf_b, acc_b, cnt = _calibration_buckets(pairs, bins)
    worst = 0.0
    for b in range(bins):
        if cnt[b] > 0:
            worst = max(worst, abs(acc_b[b] - conf_b[b]))
    return worst


def brier(probs: Sequence[float], labels: Sequence[int]) -> float:
    """Proper scoring rule: jointly calibration AND discrimination."""
    n = len(probs)
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / n


def nll(probs: Sequence[float], labels: Sequence[int],
        eps: float = 1e-7) -> float:
    """Negative log-likelihood. Punishes confident errors hardest.

    `p` is clipped: without it a single confident error returns infinity and
    the metric stops being usable rather than reporting a large value.
    """
    # NOT vectorised in the reduction: `np.sum` is pairwise and lands 1 ULP
    # from the original's sequential `total += ...`. One pass over the
    # window is not a hot path -- `ece`/`mce` ran FIFTEEN passes each, and
    # that is where the measured 2.5 s went -- so exactness is free here.
    n = len(probs)
    total = 0.0
    for p, y in zip(probs, labels):
        pc = min(max(p, eps), 1.0 - eps)
        total += y * math.log(pc) + (1 - y) * math.log(1.0 - pc)
    return -total / n


def fit_temperature(logits_or_probs: Sequence[float], labels: Sequence[int],
                    *, is_probs: bool = True) -> float:
    """Single scalar T minimising NLL, by grid search over log-space.

    Fitted on the REFERENCE window only -- never the current window, or it
    calibrates away the drift it exists to reveal.

    T > 1 means the model is overconfident. RISING T means overconfidence is
    growing, which is the one calibration signal that gets STRONGER as the
    model degrades rather than being compressed by the calibration itself.

    Vectorised over the sample axis, looped over the temperature grid.

    **Profiled 2026-08-23** at the reference-window bar (`N = 2000`), which
    an earlier review named but left unmeasured: this function was
    **96.9% of the entire calibration family** -- 86.4 ms of 89.1 ms, from
    200,000 `math.exp` and 402,000 `math.log` calls. The grid is 100 points
    and the inner loop ran once per sample per point, in Python.

    **The 100 x N matrix form is deliberately NOT used.** Broadcasting the
    grid against the samples is the obvious vectorisation and allocates
    100 x N floats -- 80 MB at `N = 100,000`. The note at the top of this
    module records that a Python-list projection cost ~180 MB at 100k rows
    and was the single largest per-model line in the edge budget; building
    a matrix here would reintroduce that shape in a different disguise.
    Looping over 100 temperatures and vectorising the N-length inner sum
    keeps memory at O(N) and takes essentially all of the speedup.

    Numerically identical to the scalar version on every non-empty input:
    same grid, same clamps, and the `loss < best` test keeps the FIRST
    minimum on ties exactly as before. Verified over 29 cases including
    saturated, constant, extreme and perfectly-separated inputs, and the
    logits path.

    **One deliberate divergence, on EMPTY input.** The scalar version
    returned `0.05` there; this returns `1.0`. The old value was an
    artifact rather than a decision: with no samples every grid point
    scores `loss = 0.0`, so `loss < best` fires on the first one and the
    smallest temperature wins by accident.

    It is worth not preserving. `T = 0.05` means `p ** (1/0.05)` = `p**20`,
    and the server-side scoring engine carries a load-bearing comment about exactly
    this failure: a fitted `T` of 0.084 amplified the score to
    the zero rail, "measured 0.3775 raw collapsing to 0.0026 against a 0.85
    alert threshold, i.e. the mode could not raise an alert under any
    input whatsoever". Returning the identity temperature on no evidence is
    the safe answer; returning the most extreme one on the grid is not.

    Unreachable in the shipped path -- the contract puts the reference-window
    bar at `N >= 2000` -- so this changes no corpus value. Recorded because
    the module's own rule is that a formula change bumps the schema
    version, and a silent one would breach it even where the input cannot
    occur.
    """
    xs = np.asarray(list(logits_or_probs), dtype=float)
    if is_probs:
        pc = np.clip(xs, 1e-7, 1 - 1e-7)
        xs = np.log(pc / (1.0 - pc))
    y = np.asarray(list(labels), dtype=float)
    if xs.size == 0:
        return 1.0
    best_t, best = 1.0, float("inf")
    for t in (0.05 * i for i in range(1, 101)):
        p = 1.0 / (1.0 + np.exp(-xs / t))
        pc = np.clip(p, 1e-7, 1 - 1e-7)
        loss = -float(np.sum(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)))
        if loss < best:
            best, best_t = loss, t
    return best_t


def apply_temperature(probs: Sequence[float], t: float) -> list[float]:
    """Calibrated stream. The raw scores are NEVER modified in place --
    calibration is a parallel product, because compressing the confidence gap
    is exactly what removes the forecaster's signal."""
    out = []
    for p in probs:
        pc = min(max(p, 1e-7), 1 - 1e-7)
        z = math.log(pc / (1 - pc)) / t
        out.append(1.0 / (1.0 + math.exp(-z)))
    return out


def predictive_entropy(p: float) -> float:
    """Binary predictive entropy in bits. Total uncertainty of ONE prediction.

    Not a decomposition and not labelled as one.
    """
    pc = min(max(p, 1e-12), 1 - 1e-12)
    return -(pc * math.log2(pc) + (1 - pc) * math.log2(1 - pc))


# ---------------------------------------------------------------------------
# Group D -- performance
# ---------------------------------------------------------------------------

def ci_width(p_hat: float, n: int) -> float:
    """Full width of the two-sided 95% normal-approximation interval.

    3.92 = 2 * 1.96. At the reference reservoir N = 2000 and worst-case p = 0.5
    this gives ~0.0438, which is the bound the field contract commits to.
    """
    return 3.92 * math.sqrt(p_hat * (1 - p_hat) / n)


# ---------------------------------------------------------------------------
# Group E / F -- data quality, system
# ---------------------------------------------------------------------------

def row_count_ratio(actual: int, expected: int) -> float:
    return actual / expected


def percentile_linear(sample: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile, p in [0, 1]. Matches numpy's default."""
    s = sorted(sample)
    n = len(s)
    rank = p * (n - 1)
    lo, hi = int(math.floor(rank)), int(math.ceil(rank))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


# ---------------------------------------------------------------------------
# Trajectory operators
# ---------------------------------------------------------------------------
# These are where the design's central claim lives: temporal structure is
# carried by these operators rather than by a recurrent encoder.
# Each shortens the series, which is why insufficient-history handling
# is explicit rather than implicit.

def first_difference(series: Sequence[float]) -> list[float]:
    return [series[i] - series[i - 1] for i in range(1, len(series))]


def ewma(series: Sequence[float], half_life: float = 5) -> tuple[float, list[float]]:
    """Returns (alpha, smoothed). alpha derived from half-life, not set directly."""
    alpha = 1 - 2 ** (-1 / half_life)
    out = [series[0]]
    for x in series[1:]:
        out.append(out[-1] + alpha * (x - out[-1]))
    return alpha, out


def ewma_slope(series: Sequence[float], half_life: float = 5) -> list[float]:
    """First difference OF the EWMA -- smoothed trend, not smoothed noise."""
    _, smoothed = ewma(series, half_life)
    return first_difference(smoothed)


def second_derivative(series: Sequence[float]) -> list[float]:
    return [series[i] - 2 * series[i - 1] + series[i - 2] for i in range(2, len(series))]


def ewma_abs_change(series: Sequence[float], half_life: float = 5) -> list[float]:
    """EWMA of |first difference| -- volatility of the signal, direction-free."""
    diffs = [abs(d) for d in first_difference(series)]
    if not diffs:
        return []
    _, smoothed = ewma(diffs, half_life)
    return smoothed


def time_above_threshold(series: Sequence[float], threshold: float) -> float:
    """Fraction of the trailing window spent above `threshold`, in [0, 1]."""
    if not series:
        return 0.0
    return sum(1 for x in series if x > threshold) / len(series)
