"""
corpus/trajectory.py

The six trajectory operators, materialised into corpus rows.

The operators themselves live in corpus/signal_vector.py and are verified
against the shared conformance fixtures. Until this module existed they were
called by nothing except those fixtures -- true of the module, false of the
corpus, which is the same defect class previously recorded for the detectors.
`corpus/harness.py` emits per-window scalars; the forecaster's feature selection runs over
the OPERATOR-DERIVED fields, so without this pass the selection pool does not
exist in any training row.

What this pass is:

  * a POST-pass over an ordered window sequence, not a per-window computation
    -- an operator is a statement about a field's trajectory, so it needs the
    series;
  * strictly causal. The value at window `t` depends only on windows <= t.
    `test_no_lookahead` appends and rewrites later windows and asserts every
    earlier value is byte-identical;
  * scoped per family, per the operator contract. The default
    of "all six on everything" is rejected there and rejected here.

What it deliberately does NOT do:

  * tune an alert threshold. Operator 6 (time-above-threshold) needs a
    contract-fixed threshold per field. `THRESHOLDS` below now holds the
    ones with a natural convention, each cited in `THRESHOLD_SOURCES`.
    Slots without one stay `threshold_undeclared` in `coverage()`
    rather than being filled with a plausible number;
  * bridge a gap in the window sequence. A first difference across a
    ten-window hole is not a per-window rate, so it is `None`, not a number.

NOTE: the threshold registry exists now, but the figures below are the pre-registry
measurement and are kept as history; the threshold registry holds today's
numbers.

Read `coverage()`'s numbers against the right base. The contract's total is **493**, of
which **120 is group G** -- absent from this corpus by construction (by
design), so no G slot is registered here and `spec_width` computes **373**,
the corpus-eligible remainder. Measured against that base, the shortfall is
**219**, and `threshold_undeclared` is the SMALL part of it: 37 slots the
harness does not emit at all, 4 struck by `EVT_EXCLUDED`, and 7 slots whose
operator 6 has no threshold. The missing-source slots, not operator 6, are
what keeps the pool narrow.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

# NOTE: corpus.signal_vector is deliberately NOT imported here. The operators
# are computed incrementally below so the pass stays single-traversal and can
# enforce adjacency per window; signal_vector's series-at-a-time versions
# remain the reference, and the upstream operator tests assert this
# module agrees with them exactly. Reference and implementation are compared,
# not shared -- which is what makes the comparison worth anything.

__all__ = [
    "EVT_EXCLUDED",
    "FAMILY_OPERATORS",
    "HALF_LIFE_WINDOWS",
    "OPERATORS",
    "SLOTS",
    "THRESHOLDS",
    "THRESHOLD_SOURCES",
    "FAMILY_EXCLUDED_CONVENTIONS",
    "Coverage",
    "Slot",
    "coverage",
    "expand",
    "merge_into_z",
]

# EWMA half-life is 5 emission windows.
HALF_LIFE_WINDOWS = 5

OPERATORS = (
    "delta",                # 1 -- first difference
    "ewma",                 # 2
    "ewma_slope",           # 3 -- first difference OF the EWMA
    "second_derivative",    # 4
    "ewma_abs_change",      # 5 -- EWMA of |delta|, direction-free volatility
    "time_above_threshold",  # 6
)

# The operator contract's family table, verbatim. Second derivative is excluded
# from E and F (step-function metrics -- curvature is noise); time-above-
# threshold is excluded from A, B and E (no contract-fixed threshold exists,
# and encoding a tuning knob here would promote a tenant config value into a
# schema constant).
FAMILY_OPERATORS: dict[str, tuple[str, ...]] = {
    "A": ("delta", "ewma", "ewma_slope", "second_derivative", "ewma_abs_change"),
    "B": ("delta", "ewma", "ewma_slope", "second_derivative", "ewma_abs_change"),
    "C": OPERATORS,
    "D": OPERATORS,
    "E": ("delta", "ewma", "ewma_slope", "ewma_abs_change"),
    "F": ("delta", "ewma", "ewma_slope", "ewma_abs_change", "time_above_threshold"),
    "G": OPERATORS,
}


@dataclass(frozen=True)
class Slot:
    """One operator-eligible field, per the contract's per-family enumeration.

    `source` is the key `corpus/harness.py` emits, which is not always the
    name the contract uses -- the divergences are recorded here rather than silently
    renamed, because a slot that is present under another name and a slot that
    is genuinely absent are different facts for the forecaster's pool size.

    `kind`:
      scalar   -- read the value directly
      map_mean -- the contract budgets map-valued statistics at ONE representative
                  aggregate slot each, "its stated `_aggregate` field where
                  one exists ... or, where none is named, the statistic's
                  single per-window summary value". No such summary is
                  emitted for five of the six drift maps, so the mean over
                  non-null keys is used -- the same construction harness.py
                  already uses for `psi_aggregate`.
      subkey   -- one named key inside a fixed-shape dict (score_quantiles).
    """
    name: str
    family: str
    source: str | None
    kind: str = "scalar"
    subkey: str | None = None


def _s(name, fam, source=None, kind="scalar", subkey=None) -> Slot:
    return Slot(name, fam, source, kind, subkey)


# The operator contract's slot list. Every slot the contract names is listed,
# including the ones the harness does not yet emit (`source=None`) -- coverage()
# reports those rather than this module pretending the pool is full width.
SLOTS: tuple[Slot, ...] = (
    # -- Group A, input space: 17 operator-eligible fixed slots --------------
    _s("psi", "A", "psi", "map_mean"),
    _s("ks_statistic", "A", "ks_statistic", "map_mean"),
    _s("js_divergence", "A", "js_divergence", "map_mean"),
    _s("wasserstein1", "A", "wasserstein1", "map_mean"),
    _s("chi_square", "A", "chi_square", "map_mean"),
    _s("hellinger", "A", "hellinger", "map_mean"),
    _s("missingness_rate", "A", None, "map_mean"),
    _s("cardinality_change", "A", "cardinality_change", "map_mean"),
    _s("new_category_rate", "A", "new_category_rate", "map_mean"),
    _s("out_of_training_range_fraction", "A", None, "map_mean"),
    _s("train_serve_skew_delta", "A", None, "map_mean"),
    # The contract names this `covariance_delta_frobenius`; harness emits the same
    # Frobenius quantity as `covariance_delta` (corpus/multivariate.py).
    _s("covariance_delta_frobenius", "A", "covariance_delta"),
    _s("top_eigenvalue_ratio", "A", "top_eigenvalue_ratio"),
    _s("pca_reconstruction_error", "A", "pca_reconstruction_error"),
    # The contract names this `domain_classifier_auc`. The harness emits
    # `domain_classifier_drift`, which is the same detector's scalar output on
    # a different scale. Mapped, not invented; flagged in coverage().
    _s("domain_classifier_auc", "A", "domain_classifier_drift"),
    _s("upstream_freshness_lag", "A", None),
    _s("event_ingest_time_skew", "A", None),
    # -- Group B, output space: 19 fields ------------------------------------
    _s("prediction_psi", "B", None),
    _s("prediction_jsd", "B", None),
    _s("score_mean", "B", "score_mean"),
    _s("score_std", "B", "score_std"),
    _s("score_skew", "B", "score_skew"),
    _s("score_kurtosis", "B", "score_kurtosis"),
    _s("score_quantile_p1", "B", "score_quantiles", "subkey", "p1"),
    _s("score_quantile_p5", "B", "score_quantiles", "subkey", "p5"),
    _s("score_quantile_p50", "B", "score_quantiles", "subkey", "p50"),
    _s("score_quantile_p95", "B", "score_quantiles", "subkey", "p95"),
    _s("score_quantile_p99", "B", "score_quantiles", "subkey", "p99"),
    _s("positive_decision_rate", "B", "positive_decision_rate"),
    _s("prediction_entropy", "B", "predictive_entropy"),
    _s("max_softmax_mean", "B", None),
    _s("max_softmax_p50", "B", None),
    _s("max_softmax_p95", "B", None),
    _s("prediction_churn", "B", None),
    _s("ensemble_disagreement_rate", "B", None),
    _s("abstain_band_occupancy", "B", None),
    # -- Group C, calibration and uncertainty: 11 fields ---------------------
    _s("ece", "C", "ece"),
    _s("mce", "C", "mce"),
    _s("brier", "C", "brier"),
    _s("nll", "C", "nll"),
    _s("refit_temperature", "C", None),
    _s("refit_temperature_drift", "C", None),
    # ONE novelty field ships with a provenance companion; the contract's
    # `epistemic_variance` slot is what it fills.
    _s("epistemic_variance", "C", "novelty_score"),
    _s("aleatoric_variance", "C", None),
    _s("conformal_set_size_mean", "C", None),
    _s("nonconformity_shift", "C", None),
    _s("confidence_gap", "C", None),
    # -- Group D, performance: 8 operator-eligible fixed slots ---------------
    _s("estimated_metric", "D", None, "map_mean"),
    _s("ci_width", "D", "ci_width"),
    _s("realised_metric", "D", "realised_metric"),
    _s("l2_trust_score", "D", None, "map_mean"),
    _s("per_segment_spread", "D", None, "map_mean"),
    _s("effective_sample_size_ratio", "D", None),
    _s("error_autocorrelation", "D", None),
    _s("residual_heteroskedasticity", "D", None),
    # -- Group E, data quality and pipeline: 6 fields ------------------------
    _s("row_count_vs_expected", "E", "row_count_vs_expected"),
    _s("duplicate_rate", "E", None),
    _s("late_arrival_percentage", "E", None),
    _s("feature_store_staleness", "E", None),
    _s("join_failure_rate", "E", None),
    _s("null_propagation_depth", "E", None),
    # -- Group F, system and operational: 11 fields --------------------------
    # Emitted as cannot-compute by construction: replay has no real latency,
    # and a design review makes EVT-derived fields an operational alarm
    # rather than a forecaster input while that stays true.
    _s("inference_latency_p50", "F", "inference_latency_p50"),
    _s("inference_latency_p95", "F", None),
    _s("inference_latency_p99", "F", None),
    _s("timeout_rate", "F", None),
    _s("request_rate", "F", None),
    _s("queue_depth", "F", None),
    _s("cpu_utilization", "F", None),
    _s("gpu_utilization", "F", None),
    _s("memory_headroom", "F", None),
    _s("exception_rate", "F", None),
    _s("retry_rate", "F", None),
    # -- Group G: ABSENT from the corpus by construction (by design:
    # transmitted-but-not-consumed). No slots are registered, so the corpus
    # cannot accidentally grow a group G column.
)

# A binding constraint, carried into the pool the forecaster selects over:
# EVT's output is an operational alarm, never a forecaster input, while the
# corpus contains no real latency data. The feature-construction rule
# strikes EVT's own outputs AND the four group F streams EVT is fit on, before
# selection runs -- so they must not be materialised as operator columns here
# either. Leaving them in would put fabricated-or-absent values in front of
# RFE, which could seat one of the 28.
#
# This is a CORPUS-SUPPORT exclusion, not a judgement that the fields are
# uninformative. Re-entry is a corpus property -- the corpus carrying real
# latency -- never a forecaster setting to flip.
EVT_EXCLUDED = frozenset({
    "inference_latency_p50",
    "inference_latency_p95",
    "inference_latency_p99",
    "request_rate",
})

# Operator 6's per-slot alert threshold (added 2026-10-14).
#
# Each value is FIXED FROM CONVENTION and cited in THRESHOLD_SOURCES. None was
# tuned on any corpus, forecaster or detector result, and none may be: a
# threshold changed after looking at a result is a tuning knob promoted into a
# schema constant, which is what the operator contract forbids.
# Changing one is a contract change (new corpus build, invalidates built
# corpora), not a config edit. Per-tenant SLOs belong in tenant config.
#
# Only slots whose family carries operator 6 (C, D, F, G) can be listed here
# and have an effect. A threshold for a slot with `source=None` is declared
# ahead of its source: coverage() still reports the slot as missing, and the
# column materialises the day a source is wired, with no further contract edit.
# Units are the WIRE units of corpus/ops_signals.py (utilisation and rates as
# fractions, latency in ms).
THRESHOLDS: dict[str, float] = {
    # C. Calibration error above 10 percentage points: the usual
    # "visibly miscalibrated" reading of a reliability diagram.
    "ece": 0.10,
    # F. Latency: Nielsen (1993) response-time limits 0.1 s / 1 s / 10 s
    # (instantaneous / uninterrupted flow / attention lost), one per percentile.
    "inference_latency_p50": 100.0,
    "inference_latency_p95": 1000.0,
    "inference_latency_p99": 10000.0,
    # F. A 1% failure budget is the 99% availability SLO convention.
    "timeout_rate": 0.01,
    "exception_rate": 0.01,
    # F. Sustained CPU above 80% is the common saturation alarm (Brendan
    # Gregg's USE method; typical autoscaler targets sit at 60-80%).
    "cpu_utilization": 0.80,
}

THRESHOLD_SOURCES: dict[str, str] = {
    "ece": "Reliability-diagram convention: ECE > 0.10 is visibly miscalibrated.",
    "inference_latency_p50": "Nielsen 1993, 0.1 s instantaneous-response limit.",
    "inference_latency_p95": "Nielsen 1993, 1 s uninterrupted-flow limit.",
    "inference_latency_p99": "Nielsen 1993, 10 s attention limit.",
    "timeout_rate": "99% availability SLO convention (1% error budget).",
    "exception_rate": "99% availability SLO convention (1% error budget).",
    "cpu_utilization": "USE-method saturation convention, 80% sustained.",
}

# Slots that have a natural threshold but whose FAMILY excludes operator 6
# (the operator contract: A, B and E). Recorded so the decision
# is visible, NOT used by expand()/coverage(): enabling operator 6 for A/E
# changes the spec width (373) and the closed wire schema, so it is a
# separate contract amendment.
#   psi                    A  0.10 / 0.25 industry PSI bands (<0.1 stable,
#                             0.1-0.25 moderate, >0.25 major); the static baseline
#                             uses 0.20 per feature.
#   row_count_vs_expected  E  two-sided band [0.5, 2.0]
#                             (data-quality check #8);
#                             "above" is the wrong shape for it.
#   feature_store_staleness / upstream_freshness_lag: contract rule is
#                             relative (3x reference median lag), no fixed value.
#   feed_staleness_s       (new wire scalar, no slot): no convention for a
#                             fixed number; the edge tripwire owns it.
FAMILY_EXCLUDED_CONVENTIONS: dict[str, str] = {
    "psi": "0.10 / 0.25 PSI bands",
    "row_count_vs_expected": "band [0.5, 2.0]",
    "upstream_freshness_lag": "3x reference median lag (relative)",
}


@dataclass
class Coverage:
    """Measured, not asserted: how much of the operator pool a given corpus
    build actually materialises.

    `spec_width` is the CORPUS-ELIGIBLE spec -- the contract's 493 minus its 120-wide
    group G, which this corpus never carries -- so it comes to 373. Comparing
    `materialised_width` against 493 would charge this build for a group it is
    specified not to have.
    """
    spec_width: int = 0
    materialised_width: int = 0
    slots_total: int = 0
    slots_materialised: int = 0
    missing_slots: list[str] = field(default_factory=list)
    threshold_undeclared: list[str] = field(default_factory=list)
    renamed: dict[str, str] = field(default_factory=dict)
    evt_excluded: list[str] = field(default_factory=list)

    @property
    def shortfall(self) -> int:
        return self.spec_width - self.materialised_width


def coverage(slots: Sequence[Slot] = SLOTS) -> Coverage:
    """Width arithmetic against the contract, computed rather than quoted."""
    cov = Coverage()
    for s in slots:
        ops = FAMILY_OPERATORS[s.family]
        cov.slots_total += 1
        cov.spec_width += len(ops)
        if s.name in EVT_EXCLUDED:
            # Counted against spec_width -- the contract still transmits these
            # -- but never materialised. Reported separately from missing_slots
            # because the cause and the remedy are different: missing is "the
            # harness does not compute it", excluded is "the corpus cannot
            # support it yet".
            cov.evt_excluded.append(s.name)
            continue
        if s.source is None:
            cov.missing_slots.append(s.name)
            continue
        cov.slots_materialised += 1
        for op in ops:
            if op == "time_above_threshold" and s.name not in THRESHOLDS:
                if s.name not in cov.threshold_undeclared:
                    cov.threshold_undeclared.append(s.name)
                continue
            cov.materialised_width += 1
        if s.source != s.name:
            cov.renamed[s.name] = s.source
    return cov


# -- value extraction --------------------------------------------------------

def _mean_valid(m) -> float | None:
    """Representative aggregate for a map-valued statistic: mean over non-null
    keys. Mirrors `psi_aggregate` in harness.py rather than inventing a second
    convention for the same job."""
    if not isinstance(m, dict):
        return None
    vals = [float(v) for v in m.values() if v is not None]
    return sum(vals) / len(vals) if vals else None


def _read(z: dict, slot: Slot) -> float | None:
    if slot.source is None or not z:
        return None
    raw = z.get(slot.source)
    if raw is None:
        return None
    if slot.kind == "map_mean":
        return _mean_valid(raw)
    if slot.kind == "subkey":
        if not isinstance(raw, dict):
            return None
        v = raw.get(slot.subkey)
        return None if v is None else float(v)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


# -- the pass ----------------------------------------------------------------

class _SlotState:
    """Incremental, causal operator state for one slot.

    Adjacency matters. `delta`, `ewma_slope`, `second_derivative` and
    `ewma_abs_change` all describe a per-window rate, so they are emitted only
    when the windows they span are consecutive AND valid. A difference taken
    across a hole in the stream is not a rate, and `None` is the honest value
    for it -- the same reasoning applied to ramp-up.

    `ewma` and `time_above_threshold` are defined over the valid history rather
    than over an adjacent pair, so they survive a gap and carry their state
    forward.
    """

    __slots__ = (
        "alpha",
        "ewma",
        "ewma_abs",
        "n_above",
        "n_seen",
        "prev",
        "prev2",
        "prev2_idx",
        "prev_idx",
        "smoothed_prev",
        "smoothed_prev_idx",
    )

    def __init__(self) -> None:
        self.alpha = 1 - 2 ** (-1 / HALF_LIFE_WINDOWS)
        self.ewma: float | None = None
        self.prev: float | None = None
        self.prev2: float | None = None
        self.prev_idx: int | None = None
        self.prev2_idx: int | None = None
        self.ewma_abs: float | None = None
        self.n_seen = 0
        self.n_above = 0
        self.smoothed_prev: float | None = None
        self.smoothed_prev_idx: int | None = None

    def update(self, idx: int, x: float | None, ops: Sequence[str],
               threshold: float | None) -> dict[str, float | None]:
        out: dict[str, float | None] = {op: None for op in ops}
        if x is None:
            return out

        adjacent = self.prev_idx == idx - 1
        adjacent2 = adjacent and self.prev2_idx == idx - 2

        if "delta" in ops and adjacent:
            out["delta"] = x - self.prev            # type: ignore[operator]

        if "second_derivative" in ops and adjacent2:
            out["second_derivative"] = x - 2 * self.prev + self.prev2  # type: ignore[operator]

        # EWMA: seeded on the first valid value, then decayed. Defined over
        # history, so a gap does not reset it.
        self.ewma = x if self.ewma is None else self.ewma + self.alpha * (x - self.ewma)
        if "ewma" in ops:
            out["ewma"] = self.ewma

        if "ewma_slope" in ops and self.smoothed_prev_idx == idx - 1:
            out["ewma_slope"] = self.ewma - self.smoothed_prev  # type: ignore[operator]
        self.smoothed_prev, self.smoothed_prev_idx = self.ewma, idx

        # EWMA of |delta| inherits delta's adjacency requirement: it can only
        # take a new input when a real per-window difference exists.
        if "ewma_abs_change" in ops:
            if adjacent:
                d = abs(x - self.prev)              # type: ignore[operator]
                self.ewma_abs = d if self.ewma_abs is None else \
                    self.ewma_abs + self.alpha * (d - self.ewma_abs)
            out["ewma_abs_change"] = self.ewma_abs

        if "time_above_threshold" in ops and threshold is not None:
            self.n_seen += 1
            if x > threshold:
                self.n_above += 1
            out["time_above_threshold"] = self.n_above / self.n_seen

        self.prev2, self.prev2_idx = self.prev, self.prev_idx
        self.prev, self.prev_idx = x, idx
        return out


def expand(z_series: Iterable[dict], slots: Sequence[Slot] = SLOTS
           ) -> list[dict[str, float | None]]:
    """Compute operator columns for an ordered sequence of `z_t` dicts.

    Returns one dict per input window, keyed `<slot>__<operator>`. A window the
    harness skipped (empty `z_t`) still gets a row -- all `None` -- so indices
    stay aligned with the WindowResult list and a skip does not silently become
    adjacency for the windows either side of it.
    """
    live = [s for s in slots
            if s.source is not None and s.name not in EVT_EXCLUDED]
    state = {s.name: _SlotState() for s in live}
    rows: list[dict[str, float | None]] = []

    for idx, z in enumerate(z_series):
        row: dict[str, float | None] = {}
        for s in live:
            ops = FAMILY_OPERATORS[s.family]
            vals = state[s.name].update(idx, _read(z, s), ops,
                                        THRESHOLDS.get(s.name))
            for op in ops:
                if op == "time_above_threshold" and s.name not in THRESHOLDS:
                    continue        # not emitted at all; see THRESHOLDS
                row[f"{s.name}__{op}"] = vals[op]
        rows.append(row)
    return rows


def merge_into_z(results, slots: Sequence[Slot] = SLOTS) -> list:
    """Attach operator columns to a `list[WindowResult]` in place.

    Opt-in. The base `z_t` shape is what the shared conformance suite compares
    against, so this never runs implicitly inside the harness -- a caller that
    wants a training corpus asks for it, and a caller checking wire conformance
    does not.
    """
    rows = expand((r.z_t for r in results), slots)
    for r, row in zip(results, rows):
        if r.z_t:
            r.z_t.update(row)
    return results
