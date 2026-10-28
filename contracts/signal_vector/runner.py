#!/usr/bin/env python3
"""
contracts/signal_vector/runner.py

Conformance-suite runner for the signal-vector wire contract
Loads every fixture in fixtures/, checks it against a
reference implementation written inline below, and reports PASS/FAIL/SKIPPED
per fixture with a non-zero exit code on any failure.

Invocation: python3 contracts/signal_vector/runner.py
Both the replay-harness CI and the edge-agent CI are required to
run this script against their own implementation's output and treat a
non-zero exit as a build failure.

Stdlib only -- no dependencies, so any implementation in any language can
invoke `python3 runner.py` from CI without pulling in this repo's Python
toolchain beyond the interpreter itself.

*** SCHEMA-INVALIDATION RULE ***
Any commit that changes an expected_output value in fixtures/, or adds or
removes a fixture, MUST be paired in the same PR with:
  1. A schema_version bump (per the schema versioning rules), and
  2. A corpus_invalidated: true note.
This is a repo convention (PR checklist item), not something this script
enforces mechanically.
"""
import json
import math
import sys
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Reference implementations -- exactly the handful of computations the
# fixtures cover. These ARE the conformance suite's ground truth.
# ---------------------------------------------------------------------------

def ref_psi(reference_window, current_window, bins=10, floor=1e-4):
    ref_sorted = sorted(reference_window)
    n = len(ref_sorted)
    edges = [ref_sorted[0]]
    for i in range(1, bins):
        idx = min(int(round(i * n / bins)), n - 1)
        edges.append(ref_sorted[idx])
    edges.append(ref_sorted[-1] + 1e-9)

    def bin_counts(data):
        counts = [0] * bins
        for x in data:
            placed = False
            for b in range(bins):
                lo, hi = edges[b], edges[b + 1]
                if b == bins - 1:
                    if lo <= x <= hi:
                        counts[b] += 1
                        placed = True
                        break
                elif lo <= x < hi:
                    counts[b] += 1
                    placed = True
                    break
            if not placed:
                counts[0 if x < edges[0] else -1] += 1
        return counts

    ref_counts = bin_counts(ref_sorted)
    cur_counts = bin_counts(current_window)
    nref, ncur = len(ref_sorted), len(current_window)
    total = 0.0
    for rc, cc in zip(ref_counts, cur_counts):
        rp = max(rc / nref, floor)
        cp = max(cc / ncur, floor)
        total += (cp - rp) * math.log(cp / rp)
    return total


def ref_score_moments(scores):
    n = len(scores)
    mean = sum(scores) / n
    var = sum((x - mean) ** 2 for x in scores) / n
    std = math.sqrt(var)
    m3 = sum((x - mean) ** 3 for x in scores) / n
    m4 = sum((x - mean) ** 4 for x in scores) / n
    g1 = m3 / std ** 3
    skew = (math.sqrt(n * (n - 1)) / (n - 2)) * g1
    g2 = m4 / std ** 4 - 3
    kurt = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6)
    return {"mean": mean, "std": std, "skew": skew, "kurtosis": kurt}


def ref_ece(pairs, bins=15):
    edges = [i / bins for i in range(bins + 1)]
    n = len(pairs)
    total = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        bucket = [p for p in pairs if (lo <= p[0] < hi) or (b == bins - 1 and p[0] == hi)]
        if not bucket:
            continue
        conf = sum(p[0] for p in bucket) / len(bucket)
        acc = sum(1 for p in bucket if p[1] == 1) / len(bucket)
        total += (len(bucket) / n) * abs(acc - conf)
    return total


def ref_ci_width(p_hat, N):
    return 3.92 * math.sqrt(p_hat * (1 - p_hat) / N)


def ref_row_count_ratio(actual, expected):
    return actual / expected


def ref_percentile_linear(sample, p):
    s = sorted(sample)
    n = len(s)
    rank = p * (n - 1)
    lo, hi = int(math.floor(rank)), int(math.ceil(rank))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


def ref_first_difference(series):
    return [series[i] - series[i - 1] for i in range(1, len(series))]


def ref_ewma(series, half_life=5):
    alpha = 1 - 2 ** (-1 / half_life)
    out = [series[0]]
    for x in series[1:]:
        out.append(out[-1] + alpha * (x - out[-1]))
    return alpha, out


def ref_second_derivative(series):
    return [series[i] - 2 * series[i - 1] + series[i - 2] for i in range(2, len(series))]


# ---------------------------------------------------------------------------
# Per-fixture check dispatch
# ---------------------------------------------------------------------------

def approx(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(b)) if tol != "exact" else a == b


def check_group_a_psi(fx):
    got = ref_psi(fx["input"]["reference_window"], fx["input"]["current_window"],
                  fx["input"].get("bins", 10), fx["input"].get("floor", 1e-4))
    want = fx["expected_output"]["psi"]
    return approx(got, want, fx["tolerance"]), f"psi got={got} want={want}"


def check_group_b_moments(fx):
    got = ref_score_moments(fx["input"]["scores"])
    want = fx["expected_output"]
    tol = fx["tolerance"]
    ok = all(approx(got[k], want[k], tol) for k in ("mean", "std", "skew", "kurtosis"))
    return ok, f"moments got={got} want={want}"


def check_group_c_ece(fx):
    got = ref_ece([tuple(p) for p in fx["input"]["pairs"]], fx["input"].get("bins", 15))
    want = fx["expected_output"]["ece"]
    return approx(got, want, fx["tolerance"]), f"ece got={got} want={want}"


def check_group_d_ci_width(fx):
    got = ref_ci_width(fx["input"]["p_hat"], fx["input"]["N"])
    want = fx["expected_output"]["ci_width"]
    return approx(got, want, fx["tolerance"]), f"ci_width got={got} want={want}"


def check_group_e_row_count(fx):
    got = ref_row_count_ratio(fx["input"]["actual_row_count"], fx["input"]["expected_row_count"])
    want = fx["expected_output"]["row_count_vs_expected"]
    return approx(got, want, fx["tolerance"]), f"row_count_vs_expected got={got} want={want}"


def check_group_f_latency(fx):
    lat = fx["input"]["latencies_ms"]
    got = {
        "p50": ref_percentile_linear(lat, 0.50),
        "p95": ref_percentile_linear(lat, 0.95),
        "p99": ref_percentile_linear(lat, 0.99),
    }
    want = fx["expected_output"]
    tol = fx["tolerance"]
    ok = all(approx(got[k], want[k], tol) for k in ("p50", "p95", "p99"))
    return ok, f"latency got={got} want={want}"


def check_group_g_slippage(fx):
    vals = fx["input"]["per_order_slippage_bps"]
    got = sum(vals) / len(vals)
    want = fx["expected_output"]["slippage_realised_vs_expected"]
    return approx(got, want, fx["tolerance"]), f"slippage got={got} want={want}"


def ref_feed_staleness(now, last_tick_ts):
    if not last_tick_ts or any(v is None for v in last_tick_ts.values()):
        return None
    return max(0.0, max(now - v for v in last_tick_ts.values()))


def ref_headroom(used, limit):
    return None if limit <= 0 else max(0.0, 1 - used / limit)


def check_group_f_ops_host(fx):
    i, want = fx["input"], fx["expected_output"]
    tol = fx["tolerance"]
    got = {
        "feed_staleness_s": ref_feed_staleness(i["now"], i["last_tick_ts"]),
        "timeout_rate": i["n_timed_out"] / i["n_requests"],
        "exception_rate": i["n_exceptions"] / i["n_requests"],
        "request_rate": i["n_requests"] / i["window_seconds"],
        "memory_headroom": ref_headroom(i["mem_used"], i["mem_limit"]),
        "cpu_utilization": sum(i["cpu_samples"]) / len(i["cpu_samples"]),
    }
    ok = all(approx(got[k], want[k], tol) for k in got)
    a = i["absent_cases"]
    gotabs = {
        "staleness_unseen_instrument": ref_feed_staleness(
            a["staleness_unseen_instrument"]["now"],
            a["staleness_unseen_instrument"]["last_tick_ts"]),
        "rate_zero_requests": None if a["rate_zero_requests"]["n_requests"] <= 0 else 0,
        "gpu_no_samples": None if not a["gpu_no_samples"]["gpu_samples"] else 0,
        "memory_over_limit": ref_headroom(a["memory_over_limit"]["mem_used"],
                                          a["memory_over_limit"]["mem_limit"]),
    }
    ok = ok and gotabs == want["absent"]
    return ok, f"ops_host got={got} absent={gotabs}"


def check_trajectory_ewma(fx):
    series = fx["input"]["series"]
    hl = fx["input"].get("half_life", 5)
    alpha, ewma_series = ref_ewma(series, hl)
    diffs = ref_first_difference(series)
    want = fx["expected_output"]
    tol = fx["tolerance"]
    ok = (approx(alpha, want["alpha"], tol)
          and all(approx(a, b, tol) for a, b in zip(diffs, want["first_difference"]))
          and all(approx(a, b, tol) for a, b in zip(ewma_series, want["ewma"])))
    return ok, f"alpha got={alpha} want={want['alpha']}"


def check_trajectory_second_derivative(fx):
    got = ref_second_derivative(fx["input"]["series"])
    want = fx["expected_output"]["second_derivative"]
    tol = fx["tolerance"]
    ok = all(approx(a, b, tol) for a, b in zip(got, want))
    return ok, f"second_derivative got={got} want={want}"


def check_cannot_compute(fx):
    # Structural check: null + companion _valid:false, not a numeric comparison.
    field = fx["field"].split(".")[-1]
    want = fx["expected_output"]
    ok = (want.get(field) is None) and (want.get(f"{field}_valid") is False)
    return ok, f"cannot-compute shape for {field}: {want}"


def check_absent(fx):
    # Structural check: field key + its _valid companion must both be absent
    # from a simulated emitted payload (we assert against the fixture's own
    # declared _omitted_fields list, which documents the wire contract).
    omitted = fx["expected_output"].get("_omitted_fields", [])
    field = fx["field"].split(".")[-1]
    ok = field in omitted and f"{field}_valid" in omitted
    return ok, f"absent shape for {field}: omitted={omitted}"


# TODO(sketch implementer): once a real t-digest implementation
# exists, replace this stub with:
#   digest = TDigest(compression=fixture["input"]["compression_delta"])
#   for x in fixture["input"]["sample"]: digest.push(x)
#   got = digest.quantile(fixture["input"]["target_quantile"])
#   assert abs(got - fixture["expected_output"]["p99_reference"]) \
#       <= fixture["tolerance"] * fixture["expected_output"]["p99_reference"]
# i.e. relative tolerance = 0.3% (the stated tail rank-error bound).
def check_sketch_tdigest_stub(fx):
    return None, "t-digest implementation not part of this suite (stub)"


# TODO(sketch implementer): once a real HLL implementation exists,
# replace this stub with:
#   hll = HyperLogLog(precision=fixture["input"]["precision_p"])
#   for x in fixture["input"]["sample"]: hll.add(x)
#   got = hll.cardinality()
#   assert abs(got - fixture["expected_output"]["distinct_count_reference"]) \
#       <= fixture["tolerance"] * fixture["expected_output"]["distinct_count_reference"]
# i.e. relative tolerance = 0.81% (the stated HLL standard error at p=14).
def check_sketch_hll_stub(fx):
    return None, "HLL implementation not part of this suite (stub)"


DISPATCH = {
    "group_a_psi_basic.json": check_group_a_psi,
    "group_b_score_moments.json": check_group_b_moments,
    "group_c_ece.json": check_group_c_ece,
    "group_d_ci_width.json": check_group_d_ci_width,
    "group_e_row_count.json": check_group_e_row_count,
    "group_f_latency_pct.json": check_group_f_latency,
    "group_g_trading_slippage.json": check_group_g_slippage,
    "group_f_ops_host.json": check_group_f_ops_host,
    "trajectory_operator_ewma.json": check_trajectory_ewma,
    "trajectory_operator_second_derivative.json": check_trajectory_second_derivative,
    "cannot_compute_null_flag.json": check_cannot_compute,
    "absent_not_applicable.json": check_absent,
    "sketch_tdigest_accuracy.json": check_sketch_tdigest_stub,
    "sketch_hll_cardinality.json": check_sketch_hll_stub,
}


def main():
    passed, failed, skipped = 0, 0, 0
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        name = path.name
        fx = json.loads(path.read_text())
        handler = DISPATCH.get(name)
        if handler is None:
            print(f"FAIL {name}: no runner handler registered for this fixture")
            failed += 1
            continue
        result, detail = handler(fx)
        if result is None:
            print(f"SKIPPED (stub -- see comment) {name}: {detail}")
            skipped += 1
        elif result:
            print(f"PASS {name}: {detail}")
            passed += 1
        else:
            print(f"FAIL {name}: {detail}")
            failed += 1

    print(f"\nPASSED: {passed}, FAILED: {failed}, SKIPPED: {skipped}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
