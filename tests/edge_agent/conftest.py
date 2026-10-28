from __future__ import annotations

import pytest

from edge_agent import EdgeAgent
from tests.edge_agent.synth import (CADENCE, T0, Collector, make_cfg, make_prices,
                                    make_stream, utc)

N_REF, N_LIVE = 6, 4


@pytest.fixture(scope="module")
def live_run(tmp_path_factory):
    """One full synthetic run: 6 reference windows then 4 live windows (drift from #8)."""
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
