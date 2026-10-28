"""
corpus/forecaster.py

The failure forecaster -- stage 1 (gradient-boosted trees).

This is the thing every other component exists to make possible: a model that,
given a window of the signal vector, says how likely the monitored model
is to breach its service objective within each horizon.

Four design decisions are implemented here:

  features       single-timestep input; the candidate pool is the operator
                 columns the corpus materialises PLUS the raw levels, so
                 the detector-vs-predictor rule can actually fire;
                 selection is GBT-importance RFE fitted INSIDE each training
                 fold, never once over the whole corpus.
  ladder         gradient-boosted trees ship FIRST, standalone, fit
                 per-horizon-bucket so a single model family produces
                 hazard-per-horizon rather than one boolean. The survival head
                 is stage 2 and is gated on offline evaluation.
  loss           the optimisation target is **partial AUROC at low FPR**
                 (FPR in [0, 0.1]), not full AUROC -- alert fatigue kills the
                 product before accuracy does. Training uses class-balanced
                 pair sampling concentrated on the low-FPR band
                 (`pauc_pair_weights`); see its docstring for what that is
                 and, importantly, what it is not.
  horizons       the output object is hazard per horizon, a survival curve,
                 expected time to breach with a confidence interval, an
                 attribution payload, and `warmup_state`.

**What this deliberately does not do.** It does not judge whether the model is
good enough -- that is the evaluation protocol's job. It reports partial
AUROC because the loss decision names it as the target, not as a claim of adequacy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CHANCE_PARTIAL_AUROC",
    "FPR_MAX",
    "HORIZONS",
    "TARGET_FIELD",
    "Forecast",
    "ForecasterBundle",
    "classify_feature",
    "drop_post_event_windows",
    "exclude_families",
    "family_of",
    "pauc_pair_weights",
    "CONTROL_ARM",
    "fit_forecaster",
    "load_episodes",
    "partial_auroc",
]

# The horizon set, in WINDOWS. The corpus is nominal-time (synthetic
# timestamps), so horizons are expressed in windows rather than hours -- a window is
# the unit the harness actually emits and the only one that means the same
# thing across HMDA's 5-minute and turbofan's 15-minute cadence.
#
# **The ceiling is the pre-onset runway, not the episode length.** Injections
# fire at `ONSET_WINDOW = 60` (corpus build), and `windows_to_event` is only
# defined for windows at or before the onset, so no label exists for a horizon
# beyond 60 windows however long the episode runs. 48 is the largest horizon
# that leaves usable runway in front of it.
#
# Against the committed horizon sets, at the 5-minute default cadence:
#
#   trading   15 min / 1h / 4h  ->  3 / 12 / 48 windows   ALL deliverable
#   credit    24h / 7d / 30d    ->  288 / 2016 / 8640     NONE deliverable
#
# The credit family was re-scoped rather than left as an unbacked promise
# (2026-08-14). Reaching 24h needs a corpus whose runway is ~5x
# longer, which is corpus-build work.
HORIZONS = (1, 3, 6, 12, 24, 48)

# The uninjected arm's injection label. Canonical HERE because three modules
# need it and none of them may import each other: the corpus builder
# WRITES it, the trainer must HOLD IT OUT of fitting, and
# the evaluator scores against it for sim-to-real. It previously
# existed as two independent literals in the builder and the evaluator, which is
# how the trainer ended up fitting on the arm reserved to validate it.
CONTROL_ARM = "none"


def load_episodes(corpus: "Path") -> dict:
    """Episode provenance, keyed by episode_id.

    Only USABLE episodes are here -- the corpus builder writes `episodes.jsonl`
    from the usable set while writing window rows for every job that ran, so
    `SPONTANEOUS_PRE_ONSET` episodes (the base model was already breaching
    before injection) have rows and no provenance. The caller filters on this
    dict, which is what keeps them out of both fitting and scoring.

    Canonical HERE, alongside CONTROL_ARM and for the same reason: the trainer
    and the evaluator must apply the SAME usable-episode filter or they
    disagree about G, and therefore about k_max. They did disagree -- the
    evaluator filtered and the trainer did not, so the trainer fit 9 features
    where the evaluator's own basis allowed 7.
    """
    out = {}
    path = corpus / "episodes.jsonl"
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        # ONLY a failed episode has an onset. `time_to_event` is overloaded:
        # windows-to-onset when `event = 1`, windows-to-stream-end when the
        # episode was censored (per the label definition). Reading
        # the second as an onset invents a breach and hands the reactive
        # baseline a ~19-hour lead on an episode where nothing failed.
        e["onset_index"] = (e.get("time_to_event")
                            if int(e.get("event", 0)) == 1 else None)
        out[e["episode_id"]] = e
    return out


# Loss: the band the target is computed over, and the same band the
# promotion gate uses.
FPR_MAX = 0.1

# The value a RANDOM ranker scores on `partial_auroc`. It is not 0.5.
#
# This is recorded as a constant because an earlier version of this module
# claimed in prose that normalising by `fpr_max` "keeps 0.5 meaning no better
# than chance". It does not. A random ranker's ROC is the diagonal, so its
# area over `fpr in [0, fpr_max]` is `fpr_max^2 / 2`, and dividing by
# `fpr_max` leaves `fpr_max / 2` = 0.05. Verified empirically: random scores
# read 0.051, a perfect ranking reads 1.000, an inverted one reads 0.000.
#
# The claim mattered. Held-out folds scoring 0.2-0.5 look like failure against
# a 0.5 chance line and like 4-10x chance against the real one, and the
# promotion gate reads exactly these numbers.
CHANCE_PARTIAL_AUROC = FPR_MAX / 2


# The field the LABEL is computed from. the labeller derives the
# breach threshold from the `true_metric` series, and `harness.py` copies that
# same series into the signal vector as `realised_metric`, where
# `trajectory.py` gives it all six operator derivatives.
#
# It is kept in the candidate pool -- at window t its value is known at time t,
# so it predicts rather than leaks -- but it must be COUNTED SEPARATELY.
# The feature-construction binding rule asks whether trajectory
# operators or raw levels dominate, in order to answer "is this a predictor or
# just a detector". Scoring `realised_metric__ewma` on the operator side lets
# a model that does nothing but watch the target metric's own trajectory
# report itself as a predictor, which is the exact conclusion the rule exists
# to refuse.
TARGET_FIELD = "realised_metric"

# These fields cannot exist until a ground-truth outcome arrives.  That is
# effectively immediate for the machine telemetry corpus, but 30--180 days for
# credit and absent for trading.  Treating them as ordinary levels/operators
# lets an offline model silently depend on evidence its deployment cannot see.
LABEL_DEPENDENT_FIELDS = frozenset({"brier", "nll", "ece", "mce", "ci_width"})


def classify_feature(name: str) -> str:
    """Classify deployability and trajectory role for the detector-vs-predictor rule."""
    if name == TARGET_FIELD or name.startswith(TARGET_FIELD + "__"):
        return "target_derived"
    if base_field(name) in LABEL_DEPENDENT_FIELDS:
        return "label_dependent"
    return "operator" if "__" in name else "level"


def partial_auroc(y_true, y_score, fpr_max: float = FPR_MAX) -> float:
    """AUROC restricted to `fpr <= fpr_max`, normalised so 1.0 is perfect.

    Full AUROC rewards ranking improvements in a region the product never
    operates in: nobody ships an alerting system at 60% false-positive rate.
    The loss therefore optimises and reports the low-FPR region only.

    **Read this against `CHANCE_PARTIAL_AUROC`, not against 0.5.** Normalising
    by `fpr_max` puts a perfect ranking at 1.0 and a random one at
    `fpr_max / 2` -- 0.05 at the default band, not 0.5. This is the opposite
    of the intuition full AUROC trains, which is why the constant exists and
    why every caller compares against it rather than hard-coding a number.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")

    # TIES ARE AVERAGED, and the ROC is built on DISTINCT SCORE THRESHOLDS
    # rather than on rows. Ranking rows inside a block of identical scores is
    # ranking that the model did not do: a GBT emits many exactly-equal
    # probabilities, so walking the block row-by-row draws a staircase whose
    # shape is the corpus's write order, not the model's skill.
    #
    # Measured before this was fixed: an uninformative model emitting five
    # distinct probabilities over 2,000 rows scored anywhere in 0.029-0.060
    # on the SAME data, purely by permuting row order -- straddling
    # CHANCE_PARTIAL_AUROC in both directions, which is exactly what
    # the promotion gate counts folds against. The evaluator's `pr_auc`
    # already averaged ties for this reason; this is the same rule on the
    # metric the gate actually reads.
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    s_sorted = y_score[order]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    # One vertex per distinct score: the last row of each tie block. The
    # trapezoid across a block is the average over its orderings.
    ends = np.r_[np.flatnonzero(s_sorted[1:] != s_sorted[:-1]), len(y) - 1]
    tpr = np.r_[0.0, tp[ends] / n_pos]
    fpr = np.r_[0.0, fp[ends] / n_neg]

    # Truncate AT the cap, interpolating the segment that crosses it, instead
    # of dropping the whole vertex. Dropping it was tolerable when vertices
    # were single rows; with tie blocks a discarded vertex can be a large part
    # of the band, which would bias the metric down by an amount that depends
    # on where the ties fall.
    cut = np.searchsorted(fpr, fpr_max, side="right")
    f = fpr[:cut]
    t = tpr[:cut]
    if cut < len(fpr) and fpr[cut] > f[-1]:
        frac = (fpr_max - f[-1]) / (fpr[cut] - f[-1])
        f = np.r_[f, fpr_max]
        t = np.r_[t, t[-1] + frac * (tpr[cut] - t[-1])]
    if len(f) < 2:
        return float("nan")
    return float(np.trapezoid(t, f) / fpr_max)


def monotone_cumulative_risks(risks: np.ndarray) -> np.ndarray:
    """Project horizon-major cumulative risks onto a non-decreasing curve.

    Stage 1 fits horizons independently, so sampling noise can produce
    ``risk(h=12) < risk(h=6)`` even though the events are nested. The smallest
    conservative correction is the running maximum: it never lowers a stated
    risk and changes a horizon only when the cumulative-risk contract would
    otherwise be impossible.
    """
    values = np.asarray(risks, dtype=float)
    if values.ndim != 2:
        raise ValueError("cumulative risks must be horizon-major 2-D data")
    return np.maximum.accumulate(values, axis=0)


@dataclass
class Forecast:
    """One window's cumulative failure-risk forecast.

    ``hazard`` is retained as a wire and pickle compatibility alias. Stage 1
    trains ``P(event within h)`` independently at each horizon; those values
    are cumulative risks, not conditional bucket hazards. Consequently this
    object does not manufacture a survival curve, expected breach time, or
    interval from them. Only a censor-aware conditional-hazard head may emit
    those quantities.

    The last three fields are the missing-telemetry wire shape. They were
    specified in the design and existed in no code path
    until 2026-08-14, which an independent review caught: the document
    described a degradation report the forecaster never emitted.
    """
    hazard: dict[int, float]
    survival_curve: list[tuple[int, float]]
    expected_windows_to_breach: float | None
    confidence_interval: tuple[float, float] | None
    attribution_payload: list[tuple[str, float]]
    warmup_state: str
    output_semantics: str = "cumulative_risk_within_horizon"
    # Missing telemetry (1): which SELECTED fields were absent this window.
    # Absence degrades the forecast, it never fails it.
    fields_missing: list[str] = field(default_factory=list)
    # Missing telemetry (2): fields present on the wire that this model does
    # not consume. Reported so a customer who enables a wider domain profile
    # is not silently wrong about what the forecaster is using.
    fields_ignored: list[str] = field(default_factory=list)
    # Missing telemetry (3): the performance estimator declined to estimate, and a forecast was
    # issued anyway with a widened interval. Orthogonal to `warmup_state` --
    # a window can be `live` AND have a declined estimate, and the console
    # must be able to check both independently.
    estimate_declined_forecast_issued: bool = False
    # The per-model refusal contract (
    # "refused, not imputed"). Populated only when `predict` is called
    # with `refuse_absent=True`, which the serving path does. A horizon one of
    # whose selected columns is ABSENT from the window is not scored at all --
    # its key is missing from `hazard` -- and is listed here with the absent
    # columns, so the reason travels with the output. `fields_used` is the
    # union of the selected columns of the horizons that WERE scored: the
    # label of what this forecast actually rests on.
    horizons_refused: dict[int, list[str]] = field(default_factory=dict)
    fields_used: list[str] = field(default_factory=list)

    @property
    def risk_within_horizon(self) -> dict[int, float]:
        """The mathematically correct name for the legacy ``hazard`` field."""
        return self.hazard

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_semantics": self.output_semantics,
            "risk_within_horizon": {str(k): v for k, v in self.hazard.items()},
            "hazard": {str(k): v for k, v in self.hazard.items()},
            "hazard_compatibility": "deprecated_alias_of_risk_within_horizon",
            "survival_curve": [[h, s] for h, s in self.survival_curve],
            "expected_windows_to_breach": self.expected_windows_to_breach,
            "confidence_interval": list(self.confidence_interval)
            if self.confidence_interval else None,
            "attribution_payload": [[f, w] for f, w in self.attribution_payload],
            "warmup_state": self.warmup_state,
            "fields_missing": list(self.fields_missing),
            "fields_ignored": list(self.fields_ignored),
            "estimate_declined_forecast_issued":
                self.estimate_declined_forecast_issued,
            "horizons_refused": {str(h): list(c) for h, c
                                 in self.horizons_refused.items()},
            "fields_used": list(self.fields_used),
        }


@dataclass
class ForecasterBundle:
    """One fitted forecaster: a per-horizon-bucket model set plus its features."""
    models: dict[int, Any]
    # Per-horizon feature lists. NOT one shared list: selection is fitted per
    # horizon, so each model is fitted on its own columns. Storing a single
    # list meant evaluation fed four of five models the wrong columns, and the
    # held-out scores came back BELOW chance -- which read like a
    # generalisation failure and was actually a column mismatch.
    features_by_horizon: dict[int, list[str]]
    features: list[str]
    horizons: tuple[int, ...]
    metrics: dict[str, Any] = field(default_factory=dict)
    # Importances captured at fit time. HistGradientBoostingClassifier exposes
    # no feature_importances_ at all, so reading one off the model would make
    # _attribute return an empty payload on every window -- silently violating
    # the rule that makes attribution mandatory on EVERY alert rather than on the
    # ones where it happens to be available.
    importances_by_horizon: dict[int, list[float]] = field(default_factory=dict)
    importances: list[float] = field(default_factory=list)

    def predict(self, X: np.ndarray, *, warmup_state: str = "live",
                feature_names: list[str] | None = None,
                estimator_valid: bool = True,
                aux_model_valid: bool = True,
                refuse_absent: bool = False,
                 strict_attribution: bool = True,
                 attribute: bool = True) -> list[Forecast]:
        """Score a matrix, returning one horizon output object per row.

        `X`'s columns are named by `feature_names` (defaulting to
        `self.features`). Each horizon's matrix is assembled from those names,
        never positionally.

        `estimator_valid` and `aux_model_valid` are the performance estimator's two refusal
        surfaces, read INDEPENDENTLY as the missing-telemetry rule
        requires -- the trigger is an OR of two separately-read fields, not
        one flag, and this method does not re-derive a decline condition of
        its own.

        `refuse_absent=True` is the per-model refusal contract
        (a
        required field the model does not have is "refused, not imputed").
        A horizon with ANY selected column absent from `feature_names` is
        not scored: it is left out of `Forecast.hazard` and named, with its
        absent columns, in `Forecast.horizons_refused`. Nothing is NaN-filled
        for an absent column. A column that is PRESENT with a null value is
        a tier-2 null -- the estimator was fitted with
        exactly those NaNs and routes them natively -- so it is not refused.
        The window is never rejected: every horizon that can be scored is,
        which keeps the missing-telemetry never-fail-closed rule.

        `strict_attribution=False` (the serving path): a window with no non-zero
        signal contribution is returned with `attribution_payload == []` instead of
        raising, so one unattributable window cannot abort a stream. The CALLER then
        owns the attribution rule: it must not page a forecast whose payload is empty
        (`SignalPipeline.unattributable`). Default True keeps training/eval strict.

        `attribute=False` skips the attribution payload (left empty). Only for a caller that
        reads the hazards alone and never raises an alert from this forecast: the learned
        detector's paired forecaster. Every alerting forecaster keeps the default (attribution is mandatory).
        """
        names = feature_names or self.features
        idx = {n: i for i, n in enumerate(names)}

        # Two rules meet here and they pull in opposite directions.
        #
        # Missing-telemetry rule: a missing selected field must DEGRADE the
        # forecast, never fail it -- "it never fails closed on missing fields".
        # An earlier version of this method raised on any absent column, which
        # honoured the wrong-column fix below but broke that rule outright.
        #
        # The wrong-column fix: this once read `Xi = X[:, take] if len(take) ==
        # len(cols) else X`, silently scoring a horizon on whatever matrix was
        # passed. Every horizon selects exactly k_max columns, so the
        # substitute usually has the right WIDTH and sklearn accepts it -- the
        # model reads another horizon's features and returns a confident,
        # meaningless hazard. An earlier change fixed that in the evaluation path,
        # where it had driven held-out scores below chance; it survived here,
        # in the SERVING path, where no held-out score exists to catch it.
        #
        # Both are satisfied by building each horizon's matrix BY NAME and
        # filling absent columns with NaN, which the estimator routes natively
        # (the missing-telemetry rule's stated mechanism). Nothing is ever
        # substituted positionally, so a wrong-column hazard remains
        # impossible, and a genuinely absent field degrades instead of raising.
        selected = {c for cols in self.features_by_horizon.values() for c in cols}
        fields_missing = sorted(c for c in selected if c not in idx)
        fields_ignored = sorted(n for n in names if n not in selected)
        # An OR of two independently-read fields. Not one flag, and not
        # re-derived here from anything else.
        declined = (not estimator_valid) or (not aux_model_valid)

        hz = {}
        refused: dict[int, list[str]] = {}
        for h in self.horizons:
            cols = self.features_by_horizon.get(h, names)
            present = [c for c in cols if c in idx]
            if refuse_absent and len(present) < len(cols):
                refused[h] = sorted(c for c in cols if c not in idx)
                continue
            if not present:
                # Every field this horizon was fitted on is gone. Degrading is
                # one thing; inventing a hazard from an all-NaN row is another.
                raise ValueError(
                    f"horizon {h} was fitted on {cols} and none of them are "
                    f"present. A forecast from no inputs is not a degraded "
                    f"forecast, it is a fabricated one.")
            Xi = np.full((X.shape[0], len(cols)), np.nan, dtype=float)
            for j, c in enumerate(cols):
                if c in idx:
                    Xi[:, j] = X[:, idx[c]]
            hz[h] = self.models[h].predict_proba(Xi)[:, 1]

        # The running maximum is taken over the SCORED horizons only, in
        # horizon order. A refused horizon has no risk to project, and
        # projecting across its gap is still the nested-event bound between
        # the horizons either side of it.
        scored = [h for h in self.horizons if h in hz]
        if scored:
            projected = monotone_cumulative_risks(
                np.vstack([hz[h] for h in scored])
            )
            hz = {h: projected[i] for i, h in enumerate(scored)}
        used = sorted({c for h in scored
                       for c in self.features_by_horizon.get(h, names)
                       if c in idx})

        out: list[Forecast] = []
        for i in range(X.shape[0]):
            hazard = {h: float(hz[h][i]) for h in scored}
            # Attribution is over the horizons that were scored. With every
            # horizon refused there is no hazard, so no alert can be raised
            # from this forecast and there is nothing to attribute.
            attribution = []
            if scored and attribute:
                try:
                    attribution = self._attribute(X[i], names, horizons=scored)
                except ValueError:
                    if strict_attribution or not (self.importances_by_horizon
                                                  or self.importances):
                        raise
                    # no signal contributes: unattributable, reported as empty
            out.append(Forecast(hazard=hazard, survival_curve=[],
                                expected_windows_to_breach=None,
                                confidence_interval=None,
                                attribution_payload=attribution,
                                warmup_state=warmup_state,
                                fields_missing=fields_missing,
                                fields_ignored=fields_ignored,
                                estimate_declined_forecast_issued=declined,
                                horizons_refused=dict(refused),
                                fields_used=list(used)))
        return out

    def _attribute(self, row: np.ndarray, names: list[str], top: int = 5,
                   horizons: list[int] | None = None):
        """Top contributing signals. Mandatory on every alert.

        Importance x value, which is a contribution proxy rather than a
        Shapley value: it is cheap enough to run on every window, and the attribution rule
        requires attribution on EVERY alert rather than on the ones that
        happen to be affordable. The alert layer owns presentation; this only
        produces the ranking.

        Attribution is computed PER HORIZON and merged, because each horizon
        is fitted on its own columns with its own importances. An earlier
        version used `self.importances`, which is overwritten on every loop
        iteration in `fit_forecaster` and therefore holds only the LAST
        horizon's values -- so on a five-horizon bundle, four horizons' alerts
        were attributed with a different model's importances.

        Scope, stated so it is not overclaimed: the attribution rule names four elements --
        first detector, top contributing signals, the concentrating
        segment/feature, and correlated deploy events. This produces the
        SECOND. The rest is routed through
        the alert layer; this method alone does not discharge the rule.
        """
        idx = {n: i for i, n in enumerate(names)}
        per_h = self.importances_by_horizon or (
            {h: self.importances for h in self.horizons} if self.importances
            else {})
        if not per_h:
            raise ValueError(
                "forecaster has no importances, so it cannot attribute. Attribution "
                "is mandatory on every alert -- refusing rather "
                "than emitting an alert with an empty payload.")

        merged: dict[str, float] = {}
        for h, imps in per_h.items():
            if horizons is not None and h not in horizons:
                continue
            cols = self.features_by_horizon.get(h, names)
            for c, w in zip(cols, imps, strict=False):
                if c in idx:
                    v = float(np.nan_to_num(row[idx[c]]))
                    # Max across horizons rather than sum: a signal driving
                    # the 1-window hazard hard should not be diluted by
                    # horizons it says nothing about.
                    merged[c] = max(merged.get(c, 0.0), abs(w * v))

        ranked = sorted(merged.items(), key=lambda kv: -kv[1])[:top]
        out = [(c, v) for c, v in ranked if v > 0.0]
        if not out:
            # The `> 0` filter used to be able to empty the payload silently,
            # so a hazard of 0.99 could ship with attribution_payload = [] --
            # exactly the outcome the guard above exists to prevent, reached
            # by a different route. Measured on a row of zeros, which is not
            # exotic: delta, EWMA-slope and second-derivative columns are
            # legitimately 0.0, and permutation importance legitimately
            # returns 0.0 for uninformative features that k_max retains
            # anyway when the pool is thin.
            raise ValueError(
                "no signal has a non-zero contribution for this window, so "
                "the alert cannot be attributed. Attribution is "
                "mandatory on every alert -- refusing rather than emitting "
                "an alert with an empty payload.")
        return out


def pauc_pair_weights(y, scores=None, *, fpr_max: float = FPR_MAX,
                      pairs_per_positive: int = 4, seed: int = 7,
                      band_share: float = 0.8):
    """The loss design's class-balanced pair sampling, restricted to low FPR.

    Returns per-sample weights standing in for a sampled set of
    (positive, negative) comparison pairs.

    **What this is, stated precisely, because the document it implements was
    previously described as "confirmed as implemented" when nothing existed.**
    The loss design specifies "a pairwise ranking surrogate restricted to the
    low-FPR region, with class-balanced pair sampling".
    `HistGradientBoostingClassifier` accepts `sample_weight` but **no custom
    objective**, so a literal differentiable pairwise loss is not reachable
    without changing estimator — LightGBM's lambdarank would mean a new
    dependency and a lockfile regeneration, which this project treats as
    expensive for good reason.

    So the pair structure is expressed through SAMPLING rather than through
    the loss: pairs are drawn, and each sample's weight is how often it was
    drawn into one. That targets the same quantity — the ordering of
    positives against the negatives that matter — without pretending to be a
    gradient-level ranking objective. Anything stronger requires the
    estimator swap, which is left as future work rather than smuggled in.

    Two properties the loss design asks for, both real here:

    `class-balanced` — at a base rate near 0.3% (measured on a corpus build),
    uniformly drawn pairs are almost all negative-negative and carry no
    ranking signal at all. Every pair here is positive-versus-negative by
    construction, so no draw is wasted.

    `restricted to the low-FPR region` — once `scores` are supplied, most
    negatives are drawn from those already ranking inside the top `fpr_max`
    of negatives. Those are the only negatives partial AUROC depends on: a
    negative ranked below every positive cannot affect the area over
    `FPR ∈ [0, fpr_max]` however far down it sits. A full-range sampler
    spends its budget on comparisons the product never operates in.
    """
    y = np.asarray(y, dtype=int)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return np.ones(len(y), dtype=float)

    rng = np.random.default_rng(seed)
    n_pairs = max(1, pairs_per_positive * len(pos))

    take_pos = rng.choice(pos, size=n_pairs, replace=True)

    if scores is None:
        take_neg = rng.choice(neg, size=n_pairs, replace=True)
    else:
        s = np.asarray(scores, dtype=float)
        cut = np.quantile(s[neg], 1.0 - fpr_max)
        band = neg[s[neg] >= cut]
        if len(band) == 0:
            take_neg = rng.choice(neg, size=n_pairs, replace=True)
        else:
            # Not exclusively the band: sampling only hard negatives lets the
            # model forget why the easy ones were easy, and the band is
            # re-derived each round from the current scores anyway.
            n_band = int(round(n_pairs * band_share))
            take_neg = np.concatenate([
                rng.choice(band, size=n_band, replace=True),
                rng.choice(neg, size=n_pairs - n_band, replace=True)])

    w = np.zeros(len(y), dtype=float)
    np.add.at(w, take_pos, 1.0)
    np.add.at(w, take_neg, 1.0)
    # Never zero: a sample drawn into no pair should be de-emphasised, not
    # deleted, or the tree sees a corpus with holes in it.
    w[w == 0.0] = 1e-3
    return w * (len(y) / w.sum())


def actionability_selection_weights(y, benign_disturbance):
    """Balance failures against benign disturbances for feature selection only.

    The mask comes from training-episode outcomes. It never enters the model's
    feature matrix or inference path. Reserve 40% of selection weight for
    nonbreaching post-injection windows when both negative strata exist.
    """
    y = np.asarray(y, dtype=int)
    benign = np.asarray(benign_disturbance, dtype=bool)
    if benign.shape != y.shape:
        raise ValueError("benign_disturbance must align with training labels")
    if np.any(benign & (y != 0)):
        raise ValueError("benign_disturbance may mark only negative labels")
    positive = y == 1
    other_negative = (y == 0) & ~benign
    if not positive.any() or not (benign | other_negative).any():
        raise ValueError("selection requires positive and negative labels")

    weights = np.zeros(len(y), dtype=float)
    weights[positive] = 0.5 / positive.sum()
    benign_share = (0.4 if other_negative.any() else 0.5) if benign.any() else 0.0
    if benign.any():
        weights[benign] = benign_share / benign.sum()
    if other_negative.any():
        weights[other_negative] = (0.5 - benign_share) / other_negative.sum()
    return weights * len(y)


def drop_post_event_windows(df):
    """Remove windows that fall AFTER their episode's breach.

    Found by an independent review, 2026-08-14, and confirmed by repro.

    The corpus builder writes `windows_to_event = onset - w` while `w <= onset`
    and `None` afterwards, keeping `event = 1` on every row of a failed
    episode. `_horizon_labels` maps `None` to infinity, so every post-breach
    window becomes a NEGATIVE -- while carrying the most degraded telemetry in
    the corpus. The model was being taught that severe degradation predicts no
    breach.

    Discrete-time survival removes a subject from the risk set once its event
    occurs; it does not keep scoring it as a survivor. That is what this does.

    **Censored episodes are untouched, and the distinction is the whole point.**
    A censored episode also has `windows_to_event = None` everywhere, but its
    `event` is 0 and its rows are legitimate negatives -- observed to the end
    of the window without breaching. Filtering on the null alone would delete
    exactly the right-censoring signal that put run-to-failure data in the
    corpus in the first place.
    """
    if "event" not in df.columns or "windows_to_event" not in df.columns:
        return df
    post = (df["event"] == 1) & df["windows_to_event"].isna()
    return df.loc[~post]


def family_of(group_id: str) -> str:
    """The base-model FAMILY of a `dataset:family:variant` group id."""
    parts = str(group_id).split(":")
    return parts[1] if len(parts) > 2 else "unknown"


def exclude_families(df, families):
    """Drop whole base-model families from a window frame, at LOAD time.

    Added 2026-09-22 for a corpus requalification. `k_max = G / 10` on the
    fitting corpus, and `G` counts `dataset:family:variant` groups, so a
    family that must not be counted has to leave the frame BEFORE `G` is
    measured -- every caller derives `g = df["group_id"].nunique()` and then
    `k_max_for(g)`, so filtering here recomputes both correctly and no caller
    passes a `k_max` by hand. That is the whole reason this is a load-time
    filter and not a post-hoc mask.

    **This exists for one narrowly legitimate move: withdrawing a family that
    FAILED its own pre-registered independence check.** Check 3 of
    the family-separability check judges each added family against the
    most-agreeing pair among the five families every published `G` already
    counted, and a family above that ceiling raises `k_max` while adding no
    generalisation evidence -- which is the loophole the `k_max` rule exists to close. A
    family that fails is not a valid group and removal is the prescribed
    remedy.

    **It is not a knob.** Removing families until the check passes is tuning,
    and it would void `k_max` exactly as surely as keeping an illegitimate
    family would; the pre-registered kill criteria bar the
    corpus-version treadmill it leads to. Every caller defaults to OFF, so
    every existing path stays byte-reproducible, and every caller records the
    exclusion in its report so a filtered run cannot be mistaken for the full
    protocol.

    An unknown name raises rather than filtering nothing: a typo that silently
    left the family in would publish a `k_max` the check never cleared.
    """
    wanted = {f.strip() for f in families if f.strip()}
    if not wanted:
        return df, []
    present = {family_of(g) for g in df["group_id"].unique()}
    unknown = sorted(wanted - present)
    if unknown:
        raise ValueError(
            f"--exclude-families names {unknown}, which is not a family in "
            f"this corpus. Present: {sorted(present)}. Refusing rather than "
            f"filtering nothing -- a typo here would publish a k_max the "
            f"independence check never cleared.")
    keep = ~df["group_id"].map(family_of).isin(wanted)
    return df.loc[keep], sorted(wanted)


def _horizon_labels(windows_to_event, event, horizon: int) -> np.ndarray:
    """Per-window binary target: does this window breach within `horizon`?

    Censored windows (no event) are label 0 -- correctly. The episode was
    observed to that point without breaching, so within this horizon it did
    not breach. That is an observation, not an assumption, which is why
    right-censoring is usable here rather than something to drop.
    """
    wte = np.asarray([np.inf if w is None else float(w) for w in windows_to_event])
    ev = np.asarray(event, dtype=int)
    return ((ev == 1) & (wte <= horizon)).astype(int)


def horizon_labels(frame, horizon: int) -> np.ndarray:
    """Return 0/1 targets and ``-1`` where the horizon was not observed.

    A censored episode is a valid negative only while at least ``horizon``
    future windows remain in its observed stream. Its tail is unknown, not a
    negative. Event episodes are known through their recorded event: a finite
    ``windows_to_event`` greater than the horizon is a genuine 0 and a value
    inside the horizon is a 1.

    ``window_index`` is used rather than row position so missing emitted
    windows do not compress time. The helper intentionally keeps ``-1`` in
    the returned array; every fitting or scoring caller must choose its known
    rows explicitly instead of being able to mistake unknown for false.
    """
    y = _horizon_labels(frame["windows_to_event"], frame["event"], horizon)
    required = {"episode_id", "window_index"}
    if not required.issubset(frame.columns):
        raise ValueError(
            "censor-aware horizon labels require episode_id and window_index"
        )

    import pandas as pd

    wte = pd.to_numeric(frame["windows_to_event"], errors="coerce").to_numpy()
    event = frame["event"].to_numpy(dtype=int)
    last = frame.groupby("episode_id")["window_index"].transform("max")
    remaining = (last - frame["window_index"]).to_numpy(dtype=float)
    event_time_known = (event == 1) & np.isfinite(wte)
    observed = event_time_known | (remaining >= float(horizon))
    return np.where(observed, y, -1).astype(np.int8)


def base_field(name: str) -> str:
    """The measurement a column is derived from: the part before the first `__`.

    `row_count_vs_expected__ewma` and `row_count_vs_expected__ewma_slope` are
    two operators over ONE measurement. A raw level carries no `__` and is its
    own base field.
    """
    return name.split("__", 1)[0]


def fit_forecaster(df, *, k_max: int, horizons=HORIZONS,
                   seed: int = 7, min_positives: int = 20,
                   include_target_fields: bool = False,
                   include_label_dependent_fields: bool = True,
                   max_per_base_field: int | None = None,
                   benign_disturbance=None) -> ForecasterBundle:
    """Fit forecaster stage 1 on a window-level corpus.

    `df` is the Parquet corpus: one row per window, operator columns plus
    survival fields. `k_max` comes from the MEASURED group count, never
    from a target -- selecting more features than the group count supports is
    the loophole `k_max` exists to close.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    # The candidate pool: operator-derived columns AND raw levels.
    #
    # Levels were excluded until 2026-08-14, and that quietly voided the one
    # question the feature-construction rule exists to answer. Its
    # binding rule says that if selection is dominated by raw levels with few
    # trajectory operators surviving, that is an evaluation finding about the
    # product -- "this is a detector, not a predictor". With levels absent
    # from the pool, operators won 100% of slots BY CONSTRUCTION, and
    # reporting that as evidence would have been a tautology dressed as a
    # result. The rule can only fire if levels can actually compete.
    #
    # Identifiers and label fields are excluded, obviously. `true_metric` is
    # excluded as well -- not as a label, but because the harness already
    # copies it into the vector as `realised_metric`, so keeping both would
    # enter the same quantity twice and split its importance.
    #
    # `realised_metric` itself STAYS. It is this window's achieved accuracy,
    # and the label concerns whether a breach arrives in the NEXT h windows,
    # so it is a legitimate predictor rather than leakage: an operator who
    # knows their model is at 0.71 today is exactly who wants the forecast.
    excluded = {"dataset", "group_id", "injection", "episode_id",
                "window_index", "event", "windows_to_event", "true_metric",
                "warmup_state"}

    # Non-content guard. An independent review observed that identifiers stayed out of
    # the pool only by two accidents: the names above happen to be listed, and
    # the numeric-dtype filter happens to drop strings. A NUMERIC tenant or
    # install id added to the corpus would enter the candidate pool silently
    # and become a selectable feature -- which is a customer identifier
    # crossing the boundary, the exact thing the non-content rule forbids.
    #
    # The signal-vector contract owns the boundary and should never emit such a column. This is
    # the second line, and it refuses loudly rather than filtering quietly:
    # a corpus carrying one is a corpus with a provenance problem, and
    # dropping the column would hide it.
    banned = [c for c in df.columns
              if c not in excluded
              and any(t in c.lower() for t in ("tenant", "customer", "account",
                                               "client", "install", "user_id"))]
    if banned:
        raise ValueError(
            f"candidate columns look like customer identifiers: {banned}. "
            f"No customer identifier may cross the boundary into training. "
            f"Refusing rather than dropping them -- their presence means the "
            f"corpus build emitted something it should have withheld.")
    # Dtype detection has to be robust, not `np.issubdtype`.
    #
    # Widening the pool to raw levels exposed this: `np.issubdtype` raises
    # TypeError on pandas extension dtypes (`StringDtype`), and it never saw
    # one while the pool was `"__" in c`, because operator columns are always
    # numeric. One corpus build also carries 25 columns of `object` dtype -- operator
    # columns that are all-null for every dataset in a given Parquet file, so
    # pyarrow could not infer a numeric type for them.
    #
    # An all-null column is coerced and kept only if coercion finds real
    # values; a column that is null everywhere carries no information for a
    # tree model and is dropped rather than fed in as a constant.
    import pandas as pd

    numeric: list[str] = []
    for c in df.columns:
        if c in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        elif df[c].dtype == object:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().any():
                df = df.assign(**{c: coerced})
                numeric.append(c)
    if not include_target_fields:
        # DEFAULT since 2026-08-14, on measured evidence rather than caution.
        #
        # `realised_metric` is the series the breach label is thresholded
        # from. At window t its value is known at time t, so it predicts
        # rather than leaks -- but production usually cannot supply it on
        # time. Performance estimation exists *because* labels lag, and
        # the field contract defines this field as null+flag until labels
        # land. A model leaning on it assumes a fidelity the deployment will
        # not have, and would degrade silently rather than loudly.
        #
        # The ablation the independent review asked for was run on a corpus build and
        # settled it. Excluding these fields is free or better:
        #
        #   cross-config   0.690 with  ->  0.722 without
        #   cross-domain   0.322 with  ->  0.322 without
        #
        # And when they WERE available, selection chose a target-derived
        # column for exactly 1 of 42 slots. The model was never using them,
        # so keeping them bought nothing and carried the deployment risk.
        # `include_target_fields=True` remains available to reproduce the
        # comparison, not as a shipping configuration.
        numeric = [c for c in numeric
                   if classify_feature(c) != "target_derived"]
    if not include_label_dependent_fields:
        # A deployable label-free arm must not learn from proper scoring rules
        # or uncertainty intervals that require outcomes at the current window.
        # Keep this independent from `include_target_fields`: realised_metric is
        # the target series itself, while these are transformations of labels.
        numeric = [c for c in numeric
                   if classify_feature(c) != "label_dependent"]
    # Drop columns that cannot split. A column with fewer than two distinct
    # non-null values carries no information for a tree, and sklearn's binner
    # raises outright on one ("window shape cannot be larger than input array
    # shape") rather than ignoring it.
    #
    # These were invisible while NaN was being zero-imputed: an all-null
    # column became an all-zero column, which is still one distinct value, but
    # the widened pool plus that build's 25 object-dtype all-null operator columns
    # is what first drove a fold to have nothing else to bin. Dropping them is
    # not a workaround -- a constant feature is genuinely not a candidate.
    keep = df[numeric].nunique(dropna=True) >= 2
    numeric = [c for c in numeric if bool(keep.get(c, False))]
    if not numeric:
        raise ValueError(
            "every candidate column is constant or empty in this slice -- "
            "there is nothing to select over.")

    operators = [c for c in numeric if "__" in c]
    if not operators:
        raise ValueError(
            "no operator columns in the corpus. The build must run "
            "trajectory.merge_into_z -- without it the forecaster has "
            "nothing to select over.")

    # NaN is passed through to the estimator, NOT imputed.
    #
    # The missing-telemetry rule requires "native missing-value routing (NaN-
    # aware split decisions) rather than a learned or heuristic imputer", and
    # argues the point at length: a second, separately-validated imputation
    # mechanism is a liability. This line used to read
    # `np.nan_to_num(X_all, nan=0.0, ...)`, which deleted the native routing
    # HistGradientBoostingClassifier already has and substituted exactly the
    # rejected option.
    #
    # Zero is not a neutral filler here. For delta, EWMA-slope and second-
    # derivative columns 0.0 is the value meaning NO CHANGE, so after
    # imputation a missing field was indistinguishable from a perfectly stable
    # one -- the precise confusion the design rejects.
    #
    # Infinities are still cleaned: they are an arithmetic artefact of ratio
    # operators on a zero denominator, not a statement about missingness.
    X_all = np.array(df[numeric].to_numpy(dtype=float), copy=True)
    X_all[np.isinf(X_all)] = np.nan
    benign = None
    if benign_disturbance is not None:
        benign = np.asarray(benign_disturbance, dtype=bool)
        if benign.shape != (len(df),):
            raise ValueError("benign_disturbance must align with training rows")

    kinds = [classify_feature(c) for c in numeric]
    models: dict[int, Any] = {}
    metrics: dict[str, Any] = {
        "horizons": {}, "n_windows": len(df),
        "pool_size": len(numeric), "k_max": k_max,
        # Reported so the operator-vs-level question has a denominator. A
        # survival count means nothing without knowing how many of each were
        # available to be chosen. Three-way, not two -- see TARGET_FIELD.
        "pool_operators": kinds.count("operator"),
        "pool_levels": kinds.count("level"),
        "pool_target_derived": kinds.count("target_derived"),
        "pool_label_dependent": kinds.count("label_dependent"),
    }
    selected: list[str] = []
    selected_importances: list[float] = []
    # Per horizon, because selection is fitted per horizon. Keeping only the
    # last horizon's list is what fed four of five models the wrong columns.
    feats_by_h: dict[int, list[str]] = {}
    imps_by_h: dict[int, list[float]] = {}

    for h in horizons:
        y_all = horizon_labels(df, h)
        known = y_all >= 0
        y = y_all[known]
        X_h = X_all[known]
        benign_h = None if benign is None else benign[known]
        pos = int(y.sum())
        if pos < min_positives or pos == len(y):
            metrics["horizons"][h] = {
                "skipped": True,
                "positives": pos,
                "known_windows": int(known.sum()),
                "unknown_windows": int((~known).sum()),
            }
            continue

        # Selection is fitted on THIS horizon's training data only. Doing it
        # once over the whole corpus would make k_max bound nothing, which is
        # the exact failure the width-ratio rule makes
        # a hard requirement rather than guidance.
        # Experimental actionability arm changes feature SELECTION only.
        # Final pair-sampled fit and probability target remain unchanged.
        selection_weights = (None if benign_h is None else
                             actionability_selection_weights(y, benign_h))
        pre = HistGradientBoostingClassifier(random_state=seed, max_iter=60)
        pre.fit(X_h, y, sample_weight=selection_weights)
        try:
            from sklearn.inspection import permutation_importance
            imp = permutation_importance(pre, X_h, y, n_repeats=2,
                                         random_state=seed, n_jobs=1,
                                         sample_weight=selection_weights,
                                         ).importances_mean
        except Exception:                                   # noqa: BLE001
            if selection_weights is not None:
                raise
            # nanvar, not var: NaN now reaches the estimator by design, and
            # plain var would return NaN for every column carrying a missing
            # value, ranking them all equal and arbitrary.
            imp = np.nanvar(X_h, axis=0)
        order = np.argsort(-imp)
        if max_per_base_field is None:
            keep = order[:k_max]
        else:
            # Redundancy control, pre-registered in
            # a selection experiment. Walk the
            # SAME importance ranking and skip a candidate once its base field
            # has taken its quota. Not a field exclusion list: the rule names no
            # field, applies at every horizon and fold, and the higher-importance
            # member of a family always keeps the slot.
            #
            # Default is None so the qualified artifact
            # stays byte-reproducible. Changing that default is a
            # decision about the product, not a tidy-up.
            taken: dict[str, int] = {}
            picked: list[int] = []
            for i in order:
                bf = base_field(numeric[i])
                if taken.get(bf, 0) >= max_per_base_field:
                    continue
                taken[bf] = taken.get(bf, 0) + 1
                picked.append(int(i))
                if len(picked) == k_max:
                    break
            # A pool with fewer distinct base fields than k_max cannot fill the
            # slots under this rule. Fall back to the unconstrained ranking
            # rather than fitting a narrower model than k_max declares, and say
            # so -- silently returning fewer columns would make `k_max` mean two
            # different things in one report.
            if len(picked) < k_max:
                metrics.setdefault("dedup_underfilled", []).append(
                    {"horizon": int(h), "distinct_base_fields": len(taken),
                     "k_max": k_max})
                keep = order[:k_max]
            else:
                keep = np.array(picked)
        cols = [numeric[i] for i in keep]
        selected = cols
        selected_importances = [float(imp[i]) for i in keep]
        feats_by_h[h] = cols
        imps_by_h[h] = selected_importances

        # Loss pair sampling, in two rounds.
        #
        # Round 1 draws class-balanced positive-versus-negative pairs, which
        # is the part that does not need scores. Round 2 re-draws them
        # concentrated on the negatives currently ranking inside the low-FPR
        # band -- the only negatives partial AUROC depends on -- which cannot
        # be identified until something has scored them. Hence two fits, not
        # a tunable number of rounds: one to find the band, one to train
        # against it.
        Xk = X_h[:, keep]
        warm = HistGradientBoostingClassifier(random_state=seed, max_iter=60)
        warm.fit(Xk, y, sample_weight=pauc_pair_weights(y, seed=seed))

        m = HistGradientBoostingClassifier(random_state=seed)
        m.fit(Xk, y, sample_weight=pauc_pair_weights(
            y, warm.predict_proba(Xk)[:, 1], seed=seed))
        models[h] = m

        p = m.predict_proba(Xk)[:, 1]
        k = [classify_feature(c) for c in cols]
        metrics["horizons"][h] = {
            "skipped": False,
            "positives": pos,
            "known_windows": int(known.sum()),
            "unknown_windows": int((~known).sum()),
            "base_rate": float(y.mean()),
            "partial_auroc_low_fpr": partial_auroc(y, p),
            "features": cols,
            # The detector-vs-predictor rule is evaluated per horizon, not just overall:
            # operators may win at short horizons and lose at long ones, and
            # that difference IS the detector-vs-predictor answer.
            "operators_selected": k.count("operator"),
            "levels_selected": k.count("level"),
            "target_derived_selected": k.count("target_derived"),
            "label_dependent_selected": k.count("label_dependent"),
        }

    if not models:
        raise ValueError(
            "no horizon had enough positives to fit. The corpus is dominated "
            "by censored episodes -- report that rather than lowering the bar.")

    return ForecasterBundle(models=models,
                            features_by_horizon=feats_by_h,
                            features=selected,
                            horizons=tuple(models), metrics=metrics,
                            importances_by_horizon=imps_by_h,
                            importances=selected_importances)
