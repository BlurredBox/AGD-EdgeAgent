"""
corpus/serving.py

The one projection from a signal vector to forecaster features.

`ReplayHarness._compute_z_t` produces a NESTED dict: per-feature maps for the
six input-space methods, scalars for everything else. A fitted forecaster
consumes a FLAT row. Something has to project one onto the other, and until
this module existed that something lived inside `scripts/build_corpus.py::
_write_windows` -- in a build script, reachable only by running a corpus
build.

That placement is the bug this file exists to make impossible. Serving has to
apply the identical projection or the model is fed columns that do not mean
what its fitted columns meant, and nothing raises: the row has plausible
names and plausible values, so the estimator returns a confident, wrong
hazard. It is the same failure shape as the two defects found on 2026-08-20
-- `k_max` read from two sources, and a usable-episode filter one script had
and another did not -- and both of those were caught by luck rather than by
construction.

So the rule here is the same one applied to `CONTROL_ARM` and
`load_episodes` in `corpus/forecaster.py`: when a definition must agree
across the training path and the serving path, it gets exactly one home and
both paths import it. `scripts/build_corpus.py` calls `flatten_z_t`;
`src/core/forecasters/forecaster_service.py` calls `flatten_z_t`. Training-
serving skew for this projection is now a code change, not an oversight.
"""
from __future__ import annotations

from typing import Any

from corpus.lead_signals import LEAD_SIGNAL_FIELDS
from corpus.ops_signals import OPS_SIGNAL_FIELDS
from corpus.trading_signals import TRADING_SIGNAL_FIELDS

__all__ = ["flatten_z_t", "PSI_AGGREGATE_COLUMN", "MAP_VALUED_FIELDS",
           "wire_fields", "allowed_corpus_columns", "IDENTIFIERS",
           "CORPUS_ONLY_COLUMNS", "TRAJECTORY_SUFFIXES",
           "BACKEND_AUTHORITATIVE_FIELDS"]

# The per-feature map that survives flattening, and the column it becomes.
#
# `baselines-v1` §1's static-drift baseline is the industry's PER-FEATURE PSI
# rule at 0.20. `psi_aggregate` is a MEAN, so one shifted feature among fifty
# cannot move it -- which is exactly the case a credit reviewer acts on.
# Scoring that baseline on the aggregate would have understated it while
# claiming the industry's number, so the max is carried as its own column.
PSI_AGGREGATE_COLUMN = "psi_max"

# Named for the conformance check, not consumed by the projection itself --
# `flatten_z_t` drops map-valued fields by type, so it stays correct when
# Phase 1 adds a seventh method. This list is what a reader should expect to
# see dropped, and a test asserts the two agree.
MAP_VALUED_FIELDS = (
    "psi", "js_divergence", "ks_statistic", "wasserstein1", "chi_square",
    "hellinger", "cardinality_change", "new_category_rate", "score_quantiles",
)


def flatten_z_t(z_t: dict[str, Any]) -> dict[str, Any]:
    """Project one signal vector onto the flat feature row a model is fitted on.

    Scalars pass through unchanged. Map-valued fields are DROPPED, with one
    exception: `psi` contributes `psi_max`.

    Maps are dropped rather than expanded because their keys are the
    customer's feature names. Expanding them would make the row's width and
    column identity depend on whose model is being monitored, so a forecaster
    fitted on one customer could not score another -- and under **I-1** those
    names are not supposed to reach a shared model at all. `fc-v1` Task 4 §4
    accordingly budgets map statistics at one representative aggregate each,
    and the operator columns already carry those aggregates.

    Absence is preserved as absence. A field the harness OMITTED during
    reference ramp-up stays omitted here rather than becoming `None`, because
    `ForecasterBundle.predict` builds each horizon's matrix by name and fills
    genuinely absent columns with NaN, which the estimator routes natively
    (`missingtelemetry-v1` §1). Substituting a null here would convert
    "this comparison does not exist yet" into "it exists and could not be
    computed", and Phase 3 Task 7 §0 requires those two stay distinguishable.
    """
    row: dict[str, Any] = {}
    for k, v in z_t.items():
        if isinstance(v, dict):
            if k == "psi":
                vals = [x for x in v.values() if x is not None]
                row[PSI_AGGREGATE_COLUMN] = max(vals) if vals else None
            continue
        row[k] = v
    return row


# ── The wire contract: what an edge agent may send ──────────────────────────
#
# `ingestion-surface-v1` (`PLAN/answers/phase-09-ingestion-surface.md` §2)
# requires a CLOSED schema -- "the endpoint takes a `z_t` and rejects anything
# else", with no `metadata`, `extra` or JSON blob. That makes I-1 a property
# of the payload's SHAPE rather than something a reviewer has to inspect for.
#
# A closed schema needs an explicit accepted-field set, and transcribing 200
# names into the API module would drift from the harness the first time
# Phase 1 adds a field. So the set is DERIVED here -- base fields crossed with
# the trajectory operators -- and `tests/test_serving_bridge.py` asserts it
# against a real corpus parquet, so a divergence fails a test instead of
# quietly widening or narrowing what the backend accepts.

# `phase-01-trajectory-operators.md`'s five operators, as column suffixes.
TRAJECTORY_SUFFIXES = (
    "__delta", "__ewma", "__ewma_abs_change", "__ewma_slope",
    "__second_derivative",
)

def base_field_of(column: str) -> str:
    """The WIRE field a flat feature column is computed from.

    A bundle selects flat columns -- `brier__ewma_slope`, `psi_max`,
    `score_quantile_p50__ewma`, `covariance_delta_frobenius__delta` -- and a
    partner supplies wire fields: `brier`, `psi`, `score_quantiles`,
    `covariance_delta`. The per-model refusal contract names the field the
    partner would have to send, so it needs this map. The operator suffix is
    stripped, `psi_max` is `psi`'s backend-side reduction, and a
    `corpus.trajectory` slot name resolves to its `source` -- the key the
    harness emits, which is not always the slot's own name. Anything else is
    already a wire scalar and is returned unchanged.
    """
    base = column
    for sfx in TRAJECTORY_SUFFIXES:
        if base.endswith(sfx):
            base = base[: -len(sfx)]
            break
    if base == PSI_AGGREGATE_COLUMN:
        return "psi"
    from corpus.trajectory import SLOTS
    for slot in SLOTS:
        if slot.name == base and slot.source:
            return slot.source
    return base


# Fields the CORPUS adds and an edge agent must never be able to send.
#
# This is the security-relevant half of the closed schema, and the reason the
# derivation could not simply be "every column in the parquet". `true_metric`,
# `event` and `windows_to_event` are the label factory's SUPERVISION -- the
# answer, not the question. An ingestion surface that accepted them would let
# a caller post the ground truth alongside the observation, and every metric
# computed downstream would be measuring the caller rather than the model.
# The rest are replay bookkeeping that has no meaning outside a build.
CORPUS_ONLY_COLUMNS = frozenset({
    "dataset", "episode_id", "event", "group_id", "injection", "n_rows",
    "true_metric", "window_index", "windows_to_event",
})

# Scalars a window carries directly. `realised_metric` IS here and is not a
# label: it is group D, the LAGGED metric a customer reports back once labels
# arrive (`phase-07-lagged-label-ingestion.md`), which is transmitted by
# design. `true_metric` is the corpus's oracle and is not.
_BASE_SCALARS = frozenset({
    "brier", "ci_width", "covariance_delta", "domain_classifier_attribution",
    "domain_classifier_drift", "domain_classifier_drift_valid",
    "domain_classifier_family", "ece", "hstrees_score",
    "if_anomaly_fraction_high", "if_anomaly_mean", "inference_latency_p50",
    "inference_latency_p50_valid", "mahalanobis_distance",
    "mahalanobis_method", "mce", "mmd", "nll", "novelty_score",
    "novelty_score_method", "pca_reconstruction_error",
    "positive_decision_rate", "predictive_entropy", "psi_aggregate",
    "psi_aggregate_valid", "realised_metric", "reference_window_id",
    "row_count_vs_expected", "schema_version", "score_kurtosis",
    "score_kurtosis_valid", "score_mean", "score_skew", "score_skew_valid",
    "score_std", "top_eigenvalue_ratio", "warmup_state",
    # D-49's three Group F additions (`phase-01-fields.md` Group F). Added
    # here 2026-09-03: the schema table carried them from D-49 onward but this
    # registry never learned them, so they parsed as unknown rather than as
    # known scalars -- which is what the Phase 16 D-49 live exercise observed
    # in `fields_ignored_unknown`.
    "ingest_incomplete_flag", "ingest_dropped_fraction",
    "unauthorized_deploy_flag",
    # v1.2: explicit detector state, estimator refusal, and reference lifecycle.
    # Absent alarm state is unavailable, never an observed zero alarms.
    "detector_alarms_fired", "estimator_valid", "aux_model_valid",
    "shed_event_flag", "rebaseline_event_flag",
}) | frozenset(TRADING_SIGNAL_FIELDS) | frozenset(OPS_SIGNAL_FIELDS) | frozenset(LEAD_SIGNAL_FIELDS)
# ^ v4 edge scalars (realised performance, market regime, output behaviour):
#   `corpus/trading_signals.py`. Aggregate statistics only; raw rows stay local.
# ^ host-ops scalars (Group F promotions + feed_staleness_s, host_anomaly_score):
#   `corpus/ops_signals.py`. No corpus carries them, so a forecaster ignores
#   them (fields_ignored); detector bank / edge tripwires only.
# ^ leading-indicator scalars (G53, schema 1.4): `corpus/lead_signals.py`. Counts, rates and lags
#   only; read by the shadow-only leadsignals forecaster, never by a paging path.

# Fields the BACKEND computes and an edge agent may not assert.
#
# `unauthorized_deploy_flag` is accepted as a KNOWN name (it is a real Group F
# field, so a payload carrying it is not malformed) but its edge-supplied
# VALUE is discarded before the vector reaches the policy layer.
#
# The reason is specific and not stylistic. `phase-01-fields.md` Group F says
# the field is "Computed backend-side where the per-model config already lives
# (D-13's channel), not at the edge", and `hardconstraint-v1`
# (`phase-06-hard-constraint-bypass.md` §2) consumes it as:
#
#     metric: unauthorized_deploy_flag, operator: GTE, threshold: 1.0,
#     action: ROLLBACK, rollback: IMMEDIATE
#
# at `PolicyLayer.SAFETY` priority 10,000. Hard constraints are never
# budget-gated (§4) and do not route through the manual-confirm path. So an
# edge that could assert this boolean could force an IMMEDIATE rollback of the
# customer's production model with no human in the loop. Accepting the name
# while refusing the value is what keeps D-49's parse correct without opening
# that path.
BACKEND_AUTHORITATIVE_FIELDS = frozenset({"unauthorized_deploy_flag"})

# The operator columns are NOT derived from a flat cross-product here, and an
# earlier draft that did so was wrong by exactly one field.
# `phase-01-trajectory-operators.md` §2 excludes `second_derivative` from
# families E and F (step-function metrics, where curvature is noise) and
# `time_above_threshold` from A, B and E (no contract-fixed threshold exists,
# and encoding a tuning knob would promote a tenant config value into a schema
# constant). `row_count_vs_expected` is family E, so
# `row_count_vs_expected__second_derivative` does not exist -- and a closed
# schema that accepted it would accept a field nothing can produce.
#
# `corpus.trajectory` already holds that table and the per-slot family
# assignments, both public. Deriving from them makes this class of drift
# structurally impossible rather than merely test-caught.
def _operator_columns() -> frozenset[str]:
    from corpus.trajectory import FAMILY_OPERATORS, SLOTS
    return frozenset(f"{slot.name}__{op}"
                     for slot in SLOTS
                     for op in FAMILY_OPERATORS[slot.family])


# BlurredBox-generated identifiers -- the only non-signal fields that cross.
# I-1 permits exactly these: nothing here derives from a customer record.
IDENTIFIERS = ("tenant_id", "install_id", "monitored_model_id")


def wire_fields() -> frozenset[str]:
    """Every field name an edge agent may put in a `z_t`.

    Map-valued fields are included: the WIRE carries `psi` as a per-feature
    map (Phase 1's contract), and `flatten_z_t` reduces it to `psi_max` only
    once it is inside the backend. `psi_max` is therefore a derived column and
    is deliberately NOT accepted on the wire -- a caller sending it directly
    would be supplying a statistic the backend is supposed to compute.
    """
    return frozenset(_BASE_SCALARS | set(MAP_VALUED_FIELDS)
                     | _operator_columns())


def allowed_corpus_columns() -> frozenset[str]:
    """Every column a built corpus is permitted to contain.

    A SUPERSET of what any real build produces, and deliberately so. The
    relationship is worth stating because it is not obvious and it matters:

        wire_fields()            422   Phase 1's full contract -- what an edge
                                       agent may send in production
        a real corpus parquet    201   what the replay harness can actually
                                       compute from a recorded dataset

    The 221-column gap is fields no replay can produce -- `cpu_utilization`,
    `queue_depth`, `retry_rate`, `gpu_utilization` and the rest of the serving-
    infrastructure groups. They have no source in a historical dataset, so the
    corpus has never contained them and **no forecaster has ever been fitted
    on them**.

    This is exactly the skew `phase-07-compute-inventory.md` §156-159 names:
    *"the forecaster trains on one distribution of `z_t` and serves against
    another."* It is SAFE -- `ForecasterBundle.predict` selects each horizon's
    columns by name and reports the remainder in `Forecast.fields_ignored`,
    which `missingtelemetry-v1` §2 requires precisely so "a customer who
    enables a wider domain profile is not silently wrong about what the
    forecaster is using". It is not, however, free: a customer sending the
    full contract is sending ~54% of it into `fields_ignored`.

    So the drift test asserts a SUBSET relation, not equality. The dangerous
    direction is a corpus emitting a column the wire contract does not know
    about -- that means training saw something serving cannot supply. The
    benign direction, the wire knowing fields no corpus can produce, is the
    normal state and must not fail a test.
    """
    return frozenset(wire_fields() | CORPUS_ONLY_COLUMNS
                     | {PSI_AGGREGATE_COLUMN})
