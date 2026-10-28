"""
corpus/quality.py

Deterministic data-quality rules, plus the novelty score.

These are RULES, not estimators -- they catch pipeline failures, not
distributional ones, and the tolerance for a false negative on a broken
pipeline is far lower than on a drift statistic.

*** THEY GATE THE STATISTICAL FIELDS ***

A PSI computed over a window with silent dtype corruption is not a wrong
number, it is a meaningless one that LOOKS LIKE DRIFT. So a structural failure
(schema, dtype, train/serve parity) invalidates the statistical fields for that
window rather than letting them be computed on corrupt input.

Without that gate, the first symptom of a broken pipeline is a drift alert, and
attribution points at the wrong thing -- the specific harm the attribution rule exists to
prevent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

__all__ = ["QualityReport", "check_window", "novelty_score",
           "MIN_CATEGORY_COUNT", "STRUCTURAL_CHECKS"]

# The non-content guarantee: a category appearing fewer than this
# many times can reveal a single record. Suppression costs real early
# schema-break sensitivity -- accepted because the non-content guarantee outranks it.
MIN_CATEGORY_COUNT = 5

STRUCTURAL_CHECKS = ("schema_valid", "dtype_conformant", "train_serve_parity")


@dataclass
class QualityReport:
    fields: dict[str, Any] = field(default_factory=dict)
    structural_failures: list[str] = field(default_factory=list)

    @property
    def gates_statistics(self) -> bool:
        """True when the statistical fields must NOT be computed this window."""
        return bool(self.structural_failures)


def check_window(rows: Sequence[dict], *,
                 expected_schema: set[str] | None = None,
                 expected_dtypes: dict[str, type] | None = None,
                 declared_bounds: dict[str, tuple[float, float]] | None = None,
                 reference_categories: dict[str, frozenset] | None = None,
                 reference_missingness: dict[str, float] | None = None,
                 expected_row_count: int | None = None,
                 unique_key: str | None = None,
                 timestamps: Sequence[datetime] | None = None,
                 train_fingerprint: dict[str, float] | None = None,
                 ) -> QualityReport:
    """Runs all ten checks. Absent inputs mean a check does not apply -- an
    absent check is never a failure (see train/serve parity)."""
    rep = QualityReport()
    n = len(rows)
    if n == 0:
        return rep
    cols = set(rows[0].keys())

    # 1. schema validity -- STRUCTURAL
    if expected_schema is not None:
        missing = expected_schema - cols
        unexpected = cols - expected_schema
        rep.fields["schema_hash_change"] = bool(missing or unexpected)
        # a MISSING column is fatal; an UNEXPECTED one is a warning -- a new
        # column nobody monitors breaks nothing, though it usually signals an
        # upstream change worth surfacing
        if missing:
            rep.structural_failures.append("schema_valid")
        rep.fields["schema_missing_columns"] = sorted(missing)

    # 2. dtype conformance -- STRUCTURAL
    if expected_dtypes:
        bad = []
        for c, t in expected_dtypes.items():
            if c in cols and not isinstance(rows[0][c], t):
                bad.append(c)
        rep.fields["dtype_change_flag"] = bool(bad)
        if bad:
            # numeric -> string is the dangerous direction: it parses as
            # categorical, gets cardinality statistics instead of quantiles,
            # and every numeric drift field for it becomes meaningless
            rep.structural_failures.append("dtype_conformant")

    # 3. range / plausibility
    if declared_bounds:
        viol = sum(1 for r in rows for c, (lo, hi) in declared_bounds.items()
                   if c in r and isinstance(r[c], (int, float))
                   and not (lo <= r[c] <= hi))
        rep.fields["out_of_range_fraction"] = viol / max(1, n * len(declared_bounds))

    # 4. missingness -- relative to reference, not absolute
    miss = {c: sum(1 for r in rows if r.get(c) is None) / n for c in cols}
    rep.fields["missingness_rate"] = miss
    if reference_missingness:
        # Rule: "> 2x reference rate, or > 0.5 absolute". An earlier
        # version added a `ref + 0.04` floor to damp spikes at tiny reference
        # rates -- which blocked exactly the case the spec uses as its example
        # (1% -> 5% is a broken upstream join even though 5% sounds small).
        # The floor was not in the spec and is removed.
        spike = any(miss.get(c, 0) > 2 * ref or miss.get(c, 0) > 0.5
                    for c, ref in reference_missingness.items())
        rep.fields["null_spike_indicator"] = spike

    # 5. duplicates
    if unique_key and unique_key in cols:
        keys = [r[unique_key] for r in rows]
        rep.fields["duplicate_rate"] = 1.0 - len(set(keys)) / n
        rep.fields["duplicate_key_source"] = "declared"
    else:
        hashes = [hash(tuple(sorted((k, str(v)) for k, v in r.items()))) for r in rows]
        rep.fields["duplicate_rate"] = 1.0 - len(set(hashes)) / n
        rep.fields["duplicate_key_source"] = "full_row_hash"

    # 6-7. freshness and ordering -- different failures, easily conflated
    if timestamps and len(timestamps) > 1:
        ts = list(timestamps)
        backwards = sum(1 for a, b in zip(ts, ts[1:]) if b < a)
        rep.fields["event_vs_ingest_skew"] = backwards / (len(ts) - 1)
        span = (max(ts) - min(ts)).total_seconds()
        rep.fields["upstream_freshness_lag"] = span

    # 8. row-count deviation
    if expected_row_count:
        rep.fields["row_count_vs_expected"] = n / expected_row_count

    # 9. unknown categories -- with k<5 suppression of PER-CATEGORY signal
    if reference_categories:
        rates, below = {}, {}
        for c, known in reference_categories.items():
            vals = [str(r[c]) for r in rows if c in r and r[c] is not None]
            if not vals:
                rates[c] = None
                below[c] = None
                continue
            counts: dict[str, int] = {}
            for v in vals:
                if v not in known:
                    counts[v] = counts.get(v, 0) + 1
            # The non-content guarantee's "minimum-count suppression":
            # below MIN_CATEGORY_COUNT occurrences the category emits no
            # PER-CATEGORY signal, but its volume is "folded into an aggregate
            # below-threshold new-category count contribution to the window's
            # new_category_rate numerator -- the rate itself still reflects
            # that volume happened".
            #
            # This used to drop those occurrences from the numerator entirely,
            # which is a different guarantee from the one the contract makes:
            # it hides that new-category mass occurred at all, and it made this
            # module disagree with `harness.py`'s rate on the same window --
            # the corpus-versus-wire divergence that is the design's
            # most likely silent failure. What the threshold protects is the
            # ability to point at WHICH category, and no per-category value is
            # emitted here at any count.
            rare = sum(k for k in counts.values() if k < MIN_CATEGORY_COUNT)
            rates[c] = sum(counts.values()) / len(vals)
            below[c] = rare
        rep.fields["new_category_rate"] = rates
        # The aggregate the contract names: how much of the numerator carries
        # no per-category attribution.
        rep.fields["new_category_below_threshold"] = below

    # 10. train/serve parity -- STRUCTURAL, but only when a fingerprint exists.
    # A missing fingerprint must NEVER invalidate a window, or the most
    # valuable check becomes the one that breaks monitoring for every customer
    # who cannot supply it.
    if train_fingerprint:
        deltas = []
        for c, ref_mean in train_fingerprint.items():
            vals = [r[c] for r in rows
                    if c in r and isinstance(r[c], (int, float))]
            if vals:
                deltas.append(abs(sum(vals) / len(vals) - ref_mean))
        if deltas:
            skew = sum(deltas) / len(deltas)
            rep.fields["train_serve_skew_delta"] = skew
            if skew > 1.0:
                rep.structural_failures.append("train_serve_parity")

    return rep


def novelty_score(mahalanobis: float, reference_median: float,
                  pca_ratio: float) -> float:
    """The distance proxy -- the default estimator, zero marginal cost.

    `max` of the two normalised components rather than a mean: they catch
    different novelty. Mahalanobis sees distance ALONG known correlation
    structure; PCA reconstruction sees DEPARTURE FROM it. A mean lets one mask
    the other.

    NOT an epistemic variance. A high value says the input is unusual, not
    that the model is unsure about it.
    """
    m = mahalanobis / reference_median if reference_median > 0 else 0.0
    return float(max(m, pca_ratio))
