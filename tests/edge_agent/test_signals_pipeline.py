"""z_t content: corpus code reused (not re-derived), reference frozen after N windows."""
from __future__ import annotations

import math

import pytest

from corpus import trading_signals as tr
from corpus.serving import flatten_z_t, wire_fields
from tests.edge_agent.synth import FEATURES


def _live(sink):
    return [b["z_t"] for b in sink.bodies() if b["z_t"].get("warmup_state") == "live"]


def _cold(sink):
    return [b["z_t"] for b in sink.bodies() if b["z_t"].get("warmup_state") == "cold_start"]


def test_reference_freezes_after_n_windows(live_run):
    agent, sink, recs, cfg = live_run
    sig = agent.models["LOCAL_MODEL_SENTINEL"].signals
    assert sig.live and sig.n_ref_windows == 6
    assert sig.stats.ref_pred_rms and sig.stats.ref_pred_rms > 0
    assert sig.stats.ref_volume and 900 < sig.stats.ref_volume < 1100
    assert sig.stats.expected_rows == 220
    assert sorted(sig.numeric) == sorted(FEATURES)
    # Frozen: every live window carries the same reference id.
    ids = {z["reference_window_id"] for z in _live(sink)}
    assert len(ids) == 1 and not any("reference_window_id" in z for z in _cold(sink))


def test_cold_start_vector_is_the_narrow_ramp_up_form(live_run):
    agent, sink, recs, cfg = live_run
    for z in _cold(sink):
        # reference-dependent fields are ABSENT, not null (missingtelemetry-v1)
        for k in ("psi", "mmd", "hstrees_score", "domain_classifier_attribution",
                  "mahalanobis_distance", "if_anomaly_mean"):
            assert k not in z
        assert z["score_mean"] is not None


def test_live_vector_has_the_harness_fields_and_drift_is_visible(live_run):
    agent, sink, recs, cfg = live_run
    live = _live(sink)
    for z in live:
        for k in ("psi", "js_divergence", "ks_statistic", "wasserstein1", "chi_square",
                  "hellinger", "psi_aggregate", "mmd", "mahalanobis_distance",
                  "hstrees_score", "novelty_score", "domain_classifier_drift",
                  "score_quantiles", "row_count_vs_expected"):
            assert k in z, k
        assert z["domain_classifier_attribution"] is not None      # >=200 rows: attribution exists
        assert z["row_count_vs_expected"] == pytest.approx(1.0)
    calm, drifted = live[0]["psi_aggregate"], live[-1]["psi_aggregate"]
    assert drifted > calm * 5 and drifted > 0.2


def test_no_label_dependent_or_probability_only_fields_for_a_signed_output(live_run):
    agent, sink, recs, cfg = live_run
    for b in sink.bodies():
        for k in ("ece", "mce", "brier", "nll", "ece_valid",
                  "predictive_entropy", "positive_decision_rate"):
            assert k not in b["z_t"], k


def test_trading_signals_realised_performance_and_regime(live_run):
    agent, sink, recs, cfg = live_run
    live = _live(sink)
    for z in live:
        for k in tr.TRADING_SIGNAL_FIELDS:
            assert k in z
    z = live[-1]
    assert z["tr_realised_vol"] is not None and z["tr_realised_vol"] > 0
    assert z["tr_log_volume_ratio"] is not None and abs(z["tr_log_volume_ratio"]) < 0.5
    assert z["tr_turnover"] is not None and z["tr_signal_autocorr"] is not None
    assert z["tr_output_scale_drift"] is not None
    assert z["tr_hit_rate"] is not None
    assert z["realised_metric"] == pytest.approx(z["tr_hit_rate"])
    assert z["ci_width"] > 0
    assert z["tr_hit_rate_rolling"] is not None


def test_trajectory_operator_columns_are_present_and_causal(live_run):
    agent, sink, recs, cfg = live_run
    bodies = sink.bodies()
    first, last = bodies[0]["z_t"], bodies[-1]["z_t"]
    assert any(k.endswith("__ewma") for k in last)
    assert first["score_mean__delta"] is None           # first window: nothing to difference
    assert last["score_mean__delta"] is not None        # adjacent windows
    assert set(last) <= wire_fields()


def test_wire_scalars_equal_corpus_flatten_projection(live_run):
    agent, sink, recs, cfg = live_run
    z = _live(sink)[-1]
    flat = flatten_z_t(z)
    for k, v in z.items():
        if not isinstance(v, dict):
            assert flat[k] == v
    # flatten's derived column is computed from the wire map
    assert flat["psi_max"] == max(x for x in z["psi"].values() if x is not None)


def test_every_number_is_finite(live_run):
    agent, sink, recs, cfg = live_run
    for b in sink.bodies():
        for k, v in b["z_t"].items():
            vals = v.values() if isinstance(v, dict) else [v]
            for x in vals:
                if isinstance(x, float):
                    assert math.isfinite(x), k


def test_state_survives_a_restart_without_refreezing(live_run, tmp_path):
    from edge_agent import EdgeAgent
    agent, sink, recs, cfg = live_run
    again = EdgeAgent(cfg, prices=agent.prices)
    m = again.models["LOCAL_MODEL_SENTINEL"].signals
    assert m.live and m.reference.reference_window_id == agent.models[
        "LOCAL_MODEL_SENTINEL"].signals.reference.reference_window_id
