from __future__ import annotations

import pytest

# The public edge repository carries this directory without the server code. A test that needs `src` or
# `demo_seed_lib` skips there instead of failing; and with the SDK (`blurredbox_agd`) absent nothing in the
# edge agent can be imported, so the whole directory is skipped at collection.
SERVER_ONLY_PACKAGES = ("src", "demo_seed_lib", "scripts")

try:
    import blurredbox_agd  # noqa: F401
    SDK_PRESENT = True
except ImportError:
    SDK_PRESENT = False


def pytest_ignore_collect(collection_path, config):
    """Without the SDK only test_sdk_present.py is collected (one visible skip, exit code 0)."""
    if (not SDK_PRESENT and collection_path.suffix == ".py" and collection_path.name.startswith("test_")
            and collection_path.name != "test_sdk_present.py"):
        return True
    return None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except ImportError as exc:
        if (exc.name or "").split(".")[0] in SERVER_ONLY_PACKAGES:
            pytest.skip(f"needs server code that is not in this tree: {exc.name}")
        raise


N_REF, N_LIVE = 6, 4


@pytest.fixture(scope="module")
def live_run(tmp_path_factory):
    """One full synthetic run: 6 reference windows then 4 live windows (drift from #8)."""
    from edge_agent import EdgeAgent
    from tests.edge_agent.synth import (CADENCE, T0, Collector, make_cfg, make_prices,
                                        make_stream, utc)
    tmp = tmp_path_factory.mktemp("edge")
    cfg = make_cfg(tmp)
    sink = Collector()
    clock = [utc(T0)]
    agent = EdgeAgent(cfg, prices=make_prices(N_REF + N_LIVE), post=sink,
                      clock=lambda: clock[0])
    recs = make_stream(N_REF + N_LIVE, drift_from=N_REF + 2)
    for r in recs:
        clock[0] = r.ts
        agent.ingest(r)
    clock[0] = utc(T0 + (N_REF + N_LIVE) * CADENCE)
    agent.flush()
    return agent, sink, recs, cfg
