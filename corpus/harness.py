"""
corpus/harness.py

The replay harness.

Given a dataset and a trained base model, walks a time-ordered stream and at
each window computes:

  * the signal vector `z_t`, via corpus/signal_vector.py, which is
    verified against the SHARED conformance fixtures (contracts/signal_vector/);
  * the TRUE performance metric, which is available here and nowhere else --
    in production there is no window-level ground truth, which is the entire
    reason the performance estimator estimates it;
  * a provenance record, written at generation time because it cannot be
    reconstructed afterwards.

No lookahead anywhere. Windows are cut on wall-clock boundaries at the contract's
cadence, and the true metric is computed over EXACTLY the rows that produced
the vector -- misalignment between the two would be a silent label-noise
source that no downstream check would catch.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

from corpus import anomaly as an
from corpus import multivariate as mv
from corpus import signal_vector as sv
from corpus.adapters import Row, Split
from corpus.quality import novelty_score
from corpus.sketches import HyperLogLog, TDigest

__all__ = ["DEFAULT_CADENCE_S", "ReferenceWindow", "ReplayHarness", "WindowResult"]

# Default emission cadence, 5 minutes.
DEFAULT_CADENCE_S = 300
# 1.3: Group F host-ops scalars (inference_latency_p95/p99, cpu/gpu_utilization,
# memory_headroom, request/timeout/exception/retry rate, queue_depth,
# feed_staleness_s, host_anomaly_score, each with _valid) admitted on the wire.
# Additive optional fields => MINOR (schema versioning rules). The harness
# does not emit them; the stamp tracks the contract it conforms to.
SCHEMA_VERSION = "1.3"


@dataclass
class ReferenceWindow:
    """The drift baseline: per-feature quantile sketches plus the raw reference
    samples the PSI/JS/KS binning needs.

    `reference_window_id` follows the contract's format so a consumer can join a
    stored `z_t` back to the baseline it was computed against -- necessary
    because a re-baseline changes the meaning of every drift field without
    changing the schema.
    """
    dataset_id: str
    group_id: str
    samples: dict[str, list[float]]
    categoricals: dict[str, HyperLogLog]
    digests: dict[str, TDigest]
    n_rows: int
    span_hours: float
    reference_window_id: str
    # Cached once: these depend only on the reference window, which is fixed
    # for the whole replay. Recomputing them per window is the dominant cost.
    sorted_samples: dict[str, list[float]] = field(default_factory=dict)
    bin_edges: dict[str, list[float]] = field(default_factory=dict)
    ref_bin_counts: dict[str, list[int]] = field(default_factory=dict)
    cat_counts: dict[str, int] = field(default_factory=dict)
    # Exact value set for new_category_rate. Bounded by real categorical
    # cardinality; if a feature ever has unbounded cardinality this must
    # become a sketch, and the contract's k=1 suppression already applies.
    cat_values: dict[str, frozenset] = field(default_factory=dict)
    # Multivariate / anomaly / novelty state, fitted once per reference window. Refitting
    # per emission window would make every statistic compare against a moving
    # target -- the error the univariate path avoids by caching bin edges.
    multivariate: mv.MultivariateReference | None = None
    iforest: an.IsolationForestDetector | None = None
    hstrees: an.HalfSpaceTrees | None = None

    @classmethod
    def build(cls, rows: Sequence[Row], dataset_id: str, group_id: str,
              numeric: Sequence[str], categorical: Sequence[str]) -> ReferenceWindow:
        if len(rows) < 2000:
            raise ValueError(f"reference window needs N>=2000, got {len(rows)}")
        span_h = (max(r.ts for r in rows) - min(r.ts for r in rows)).total_seconds() / 3600
        if span_h < 24:
            raise ValueError(f"reference window needs >=24h, got {span_h:.1f}h")

        samples = {f: [float(r.features[f]) for r in rows] for f in numeric}
        digests = {f: TDigest(compression=100).extend(samples[f]) for f in numeric}
        cats = {f: HyperLogLog(precision=14).extend(str(r.features[f]) for r in rows)
                for f in categorical}

        payload = json.dumps({
            "dataset": dataset_id, "group": group_id,
            "start": min(r.ts for r in rows).isoformat(),
            "end": max(r.ts for r in rows).isoformat(),
            "schema": SCHEMA_VERSION,
        }, sort_keys=True)
        rwid = hashlib.sha256(payload.encode()).hexdigest()[:16]
        sorted_samples = {f: sorted(v) for f, v in samples.items()}
        edges = {f: sv.bin_edges_from_reference(v, 10)
                 for f, v in sorted_samples.items()}
        ref_counts = {f: sv.bin_counts(sorted_samples[f], edges[f], 10)
                      for f in sorted_samples}
        cat_counts = {f: h.count() for f, h in cats.items()}
        cat_values = {f: frozenset(str(r.features[f]) for r in rows)
                      for f in categorical}

        # --- detector state, all fitted once here
        import numpy as _np
        X = _np.array([[float(r.features[f]) for f in numeric] for r in rows])
        multivariate = mv.MultivariateReference.fit(X)
        iforest = (an.IsolationForestDetector(seed=0).fit(X)
                   if len(X) >= an.IF_SUBSAMPLE else None)
        hst = an.HalfSpaceTrees(n_features=len(numeric), seed=0)
        hst.set_range(X.min(axis=0), X.max(axis=0))
        for row in X:
            hst.update(row)
        hst.roll_window()

        return cls(dataset_id, group_id, samples, cats, digests,
                   len(rows), span_h, rwid,
                   sorted_samples, edges, ref_counts, cat_counts, cat_values,
                   multivariate, iforest, hst)


@dataclass
class WindowResult:
    """One row of the corpus."""
    window_start: str
    window_end: str
    n_rows: int
    z_t: dict
    true_metric: float | None
    true_metric_name: str
    provenance: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)


def _accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


class ReplayHarness:
    """Walks a replay stream and emits one WindowResult per cadence window."""

    def __init__(self, split: Split, predict_fn: Callable[[Sequence[Row]], list[float]],
                 *, dataset_id: str, group_id: str,
                 numeric: Sequence[str], categorical: Sequence[str],
                 cadence_seconds: int = DEFAULT_CADENCE_S,
                 min_rows_per_window: int = 30,
                 seed: int = 0):
        self.split = split
        self.predict_fn = predict_fn
        self.dataset_id = dataset_id
        self.group_id = group_id
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.cadence = timedelta(seconds=cadence_seconds)
        self.min_rows = min_rows_per_window
        self.seed = seed
        self.reference = ReferenceWindow.build(
            split.reference, dataset_id, group_id, numeric, categorical)

    # -- windowing ---------------------------------------------------------

    def _windows(self, rows: Sequence[Row]) -> Iterator[tuple[datetime, datetime, list[Row]]]:
        """Cut on fixed wall-clock boundaries, not on row counts.

        Boundary-aligned windows mean a gap in the stream produces a window
        with too few rows -- which is information (Group E's row-count ratio)
        rather than something to paper over by rebalancing.
        """
        if not rows:
            return
        start = min(r.ts for r in rows)
        end = max(r.ts for r in rows)
        cur = start
        idx = 0
        ordered = sorted(rows, key=lambda r: r.ts)
        while cur <= end:
            nxt = cur + self.cadence
            bucket = []
            while idx < len(ordered) and ordered[idx].ts < nxt:
                bucket.append(ordered[idx])
                idx += 1
            yield cur, nxt, bucket
            cur = nxt

    # -- z_t ---------------------------------------------------------------

    def _compute_z_t(self, bucket: Sequence[Row], scores: Sequence[float],
                     expected_rows: int, *,
                     reference_established: bool = True) -> dict:
        """Groups A, B, C, E, F. Group D is added by the caller when a true
        metric exists; group G is ABSENT by construction (by design:
        transmitted-not-consumed, so the corpus never contains it).

        `reference_established=False` emits the REFERENCE RAMP-UP form: the
        narrow vector a newly-onboarded model produces before it has a baseline
        to compare against. Every reference-dependent field is OMITTED, not
        nulled -- the contract is explicit that the two ramp-ups must stay
        wire-distinguishable:

            absent      = this comparison does not exist yet
            null + flag = it exists and could not be computed this window

        Collapsing them would make a brand-new model indistinguishable from an
        established one having a quiet hour, and those need different responses
        from both the forecaster and the operator. The forecaster's cold-start Stage 1
        trains on exactly this narrow form, so the corpus has to contain it.
        """
        z: dict = {"schema_version": SCHEMA_VERSION}
        if reference_established:
            # No reference means no id to join back to. Emitting one during
            # ramp-up would point at a baseline that does not exist yet.
            z["reference_window_id"] = self.reference.reference_window_id

        if reference_established:
            # A -- input space. ALL SIX univariate methods, always-on: the
            # vector's shape must not depend on load or on who is watching, or the
            # corpus and production emit different shapes for the same conditions.
            psi_map, js_map, ks_map = {}, {}, {}
            w1_map, chi2_map, hell_map = {}, {}, {}
            for f in self.numeric:
                cur = [float(r.features[f]) for r in bucket]
                ref = self.reference.samples[f]
                if len(cur) < 2:
                    psi_map[f] = js_map[f] = ks_map[f] = None
                    w1_map[f] = chi2_map[f] = hell_map[f] = None
                    continue
                edges = self.reference.bin_edges[f]
                rc = self.reference.ref_bin_counts[f]
                ref_sorted = self.reference.sorted_samples[f]
                # Four share one binning pass; the reference-side structures are
                # cached on the reference window, so the per-window cost is in the
                # CURRENT window's size, not the reference's.
                psi_map[f] = sv.psi(ref, cur, edges=edges, ref_counts=rc)
                js_map[f] = sv.js_divergence(ref, cur, edges=edges, ref_counts=rc)
                chi2_map[f] = sv.chi_square(ref, cur, edges=edges, ref_counts=rc)
                hell_map[f] = sv.hellinger(ref, cur, edges=edges, ref_counts=rc)
                # KS and W1 are bin-free; both consume the cached sorted reference.
                ks_map[f] = sv.ks_statistic(ref, cur, ref_sorted=ref_sorted)
                w1_map[f] = sv.wasserstein1(ref_sorted, cur)
            z["psi"] = psi_map
            z["js_divergence"] = js_map
            z["ks_statistic"] = ks_map
            z["wasserstein1"] = w1_map
            z["chi_square"] = chi2_map
            z["hellinger"] = hell_map
            valid_psi = [v for v in psi_map.values() if v is not None]
            z["psi_aggregate"] = sum(valid_psi) / len(valid_psi) if valid_psi else None
            z["psi_aggregate_valid"] = bool(valid_psi)

            # A -- categorical cardinality via HLL, plus new-category rate.
            # cardinality_change measures how many distinct values there are;
            # new_category_rate measures whether they are the SAME values. A
            # category rename keeps the count identical and replaces every value,
            # so only the second field sees it -- which is precisely the
            # integration-regression case the contract wants covered.
            card, new_rate = {}, {}
            for f in self.categorical:
                vals = [str(r.features[f]) for r in bucket]
                hll = HyperLogLog(precision=14).extend(vals)
                ref_n = self.reference.cat_counts[f]
                card[f] = (hll.count() - ref_n) / ref_n if ref_n else None
                seen = self.reference.cat_values.get(f, frozenset())
                new_rate[f] = (sum(1 for v in vals if v not in seen) / len(vals)
                               if vals and seen else None)
            z["cardinality_change"] = card
            z["new_category_rate"] = new_rate

        # B -- output space
        if len(scores) >= 4:
            m = sv.score_moments(scores)
            z.update({"score_mean": m["mean"], "score_std": m["std"],
                      "score_skew": m["skew"], "score_kurtosis": m["kurtosis"]})
            # A collapsed model -- every row scored near-identically -- has no
            # distribution shape to report. score_moments returns None there
            # rather than asserting zero skew, and the contract's tier-2 convention
            # requires the companion flag alongside the null. Measured on an
            # MLP after schema_break renamed every category: max score 3.8e-07,
            # std 1.9e-08. That is a real production failure mode, not an edge
            # case, and the corpus should carry it as one.
            for k in ("skew", "kurtosis"):
                if m[k] is None:
                    z[f"score_{k}_valid"] = False
            z["score_quantiles"] = {
                f"p{int(p*100)}": sv.percentile_linear(scores, p)
                for p in (0.01, 0.05, 0.50, 0.95, 0.99)
            }
            z["positive_decision_rate"] = sum(1 for s in scores if s >= 0.5) / len(scores)
        else:
            # The contract's two-tier cannot-compute convention: null + explicit flag
            for k in ("score_mean", "score_std", "score_skew", "score_kurtosis",
                      "positive_decision_rate"):
                z[k] = None
                z[f"{k}_valid"] = False

        # C -- calibration
        if len(bucket) >= 4:
            # Confidence is confidence in the PREDICTED class, i.e.
            # max(s, 1-s) -- not the raw positive-class score. Pairing a raw
            # score against a correctness flag mixes two different quantities:
            # a confident negative prediction (s=0.02) would be recorded as
            # 2% confident while being correct 98% of the time, which makes
            # ECE nearly insensitive to label-side degradation. The
            # conformance fixture's pair_format is explicit that the first
            # element is confidence and the second a correctness flag.
            pairs = [(max(s, 1.0 - s), 1 if (s >= 0.5) == (r.y_true == 1) else 0)
                     for s, r in zip(scores, bucket)]
            z["ece"] = sv.ece(pairs)
            z["mce"] = sv.mce(pairs)
            z["brier"] = sv.brier(list(scores), [int(r.y_true) for r in bucket])
            z["nll"] = sv.nll(list(scores), [int(r.y_true) for r in bucket])
        else:
            z["ece"], z["ece_valid"] = None, False

        if reference_established:
            # --- multivariate. Four of five return one scalar for
            # the whole vector; only the domain classifier attributes per feature,
            # which is why the attribution rule makes it non-optional.
            import numpy as _np
            Xw = _np.array([[float(r.features[f]) for f in self.numeric]
                            for r in bucket])
            mref = self.reference.multivariate
            z["pca_reconstruction_error"] = mv.pca_reconstruction_error(mref, Xw)
            z["mahalanobis_distance"] = mv.mahalanobis_mean(mref, Xw)
            z["mahalanobis_method"] = "ledoit_wolf"      # provenance
            z["covariance_delta"] = mv.covariance_delta(mref, Xw)
            z["top_eigenvalue_ratio"] = mv.top_eigenvalue_ratio(mref, Xw)
            z["mmd"] = mv.mmd_rbf(mref, Xw, seed=self.seed)

            dc = mv.domain_classifier(mref, Xw, seed=self.seed)
            if dc is None:
                # Below the 200-row minimum. Multivariate attribution is
                # absent, so the alert layer must not raise an unattributable
                # multivariate alert on this window (attribution is mandatory).
                z["domain_classifier_drift"] = None
                z["domain_classifier_drift_valid"] = False
                z["domain_classifier_attribution"] = None
            else:
                z["domain_classifier_drift"] = dc.drift_score
                z["domain_classifier_attribution"] = {
                    self.numeric[i]: v for i, v in dc.attribution.items()
                    if i < len(self.numeric)}
            z["domain_classifier_family"] = "hist_gbm"   # provenance

            # --- anomaly
            if self.reference.iforest is not None:
                ifs = self.reference.iforest.score(Xw)
                z["if_anomaly_mean"] = ifs["mean"]
                z["if_anomaly_fraction_high"] = ifs["fraction_above_0_6"]
            else:
                z["if_anomaly_mean"] = None
                z["if_anomaly_mean_valid"] = False
            hst = self.reference.hstrees
            z["hstrees_score"] = float(_np.mean([hst.score(r) for r in Xw]))

            # --- novelty. Zero marginal cost -- reuses the two above.
            z["novelty_score"] = novelty_score(
                z["mahalanobis_distance"], mref.mahalanobis_median,
                z["pca_reconstruction_error"])
            z["novelty_score_method"] = "distance_proxy"  # provenance, 3 methods
        z["predictive_entropy"] = float(
            sum(sv.predictive_entropy(s) for s in scores) / len(scores)
        ) if scores else None

        # E -- data quality
        z["row_count_vs_expected"] = sv.row_count_ratio(len(bucket), expected_rows) \
            if expected_rows else None

        # F -- system/operational. Real latency is not available in replay;
        # emitted as cannot-compute rather than fabricated, so the corpus never
        # teaches the forecaster a synthetic latency distribution.
        z["inference_latency_p50"] = None
        z["inference_latency_p50_valid"] = False

        return z

    # -- run ---------------------------------------------------------------

    def run(self, *, injection: Callable[[list[Row], int], list[Row]] | None = None,
            metric_name: str = "accuracy",
            rampup_windows: int = 0) -> list[WindowResult]:
        """Walk the replay stream, one WindowResult per cadence window.

        `rampup_windows` emits that many REFERENCE RAMP-UP windows first: the
        structurally narrow vector a newly-onboarded model produces before it
        has a baseline. Reference-dependent fields are absent rather than null
        (per the contract), and `warmup_state` records the stage.

        The forecaster's cold-start Stage 1 fits its GBT on exactly this narrow form,
        so a corpus built with `rampup_windows=0` gives that stage nothing to
        train on -- which is
        the reason this parameter exists.
        """
        results: list[WindowResult] = []
        expected = max(1, int(self.cadence.total_seconds() /
                              max(1, self._median_interarrival())))

        for w_idx, (start, end, bucket) in enumerate(self._windows(self.split.replay)):
            if injection is not None:
                bucket = injection(list(bucket), w_idx)
            if len(bucket) < self.min_rows:
                # Below the minimum emission bar -- no vector is emitted
                # at all. Recording the skip keeps the gap visible.
                results.append(WindowResult(
                    window_start=start.isoformat(), window_end=end.isoformat(),
                    n_rows=len(bucket), z_t={}, true_metric=None,
                    true_metric_name=metric_name,
                    provenance={"skipped": "below_min_rows",
                                "min_rows": self.min_rows}))
                continue

            scores = self.predict_fn(bucket)
            established = w_idx >= rampup_windows
            z = self._compute_z_t(bucket, scores, expected,
                                  reference_established=established)
            # The cold-start stage, carried on every row so a consumer never
            # has to infer it from which fields happen to be present.
            z["warmup_state"] = "live" if established else "cold_start"

            # TRUE performance -- over exactly the rows that produced z_t.
            y_true = [int(r.y_true) for r in bucket]
            y_pred = [1 if s >= 0.5 else 0 for s in scores]
            true_metric = _accuracy(y_true, y_pred)

            # D -- performance group carries the realised metric and its CI
            z["realised_metric"] = true_metric
            z["ci_width"] = sv.ci_width(true_metric, len(bucket))

            results.append(WindowResult(
                window_start=start.isoformat(), window_end=end.isoformat(),
                n_rows=len(bucket), z_t=z,
                true_metric=true_metric, true_metric_name=metric_name,
                provenance={
                    "dataset": self.dataset_id,
                    "group_id": self.group_id,
                    "reference_window_id": self.reference.reference_window_id,
                    "schema_version": SCHEMA_VERSION,
                    "seed": self.seed,
                    "window_index": w_idx,
                },
            ))
        return results

    def _median_interarrival(self) -> float:
        ts = sorted(r.ts for r in self.split.replay[:1000])
        if len(ts) < 2:
            return 1.0
        gaps = [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]
        gaps.sort()
        return gaps[len(gaps) // 2] or 1.0
