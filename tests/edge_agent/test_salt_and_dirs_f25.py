"""Edge F25: a damaged feature-id salt stops the agent; models whose spool folders collide are rejected."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from edge_agent.__main__ import load_edge_config
from edge_agent.featureids import FeatureIdMap, FeatureIdSaltError


def test_first_start_mints_a_salt(tmp_path):
    path = tmp_path / "feature_ids.json"
    FeatureIdMap(path).id_for("price")
    assert path.exists() and path.with_suffix(".bak").exists()


def test_corrupt_main_with_good_backup_recovers(tmp_path):
    path = tmp_path / "feature_ids.json"
    before = FeatureIdMap(path).id_for("price")
    path.write_text('{"salt": "00')
    assert FeatureIdMap(path).id_for("price") == before


def test_both_copies_corrupt_refuses_and_keeps_both(tmp_path):
    path = tmp_path / "feature_ids.json"
    bak = path.with_suffix(".bak")
    path.write_text("garbage")
    bak.write_text("garbage too")
    with pytest.raises(FeatureIdSaltError, match="salt is damaged") as e:
        FeatureIdMap(path)
    assert str(bak) in str(e.value) and "delete" in str(e.value)
    assert len(list(tmp_path.glob("feature_ids.corrupt-*"))) == 1 and bak.read_text() == "garbage too"
    assert not path.exists()                    # no new salt written


def test_main_exits_cleanly_on_a_damaged_salt(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "feature_ids.json").write_text("garbage")
    (state / "feature_ids.bak").write_text("garbage")
    src = tmp_path / "in.jsonl"
    src.write_text("")
    conf = tmp_path / "agent.json"
    conf.write_text(json.dumps({"tenant_id": "t", "install_id": "i", "models": {"a": "mm-1"}, "secret": "s" * 20,
                                "endpoints": ["https://x.invalid"], "state_dir": str(state),
                                "source": {"type": "jsonl", "path": str(src)}}))
    conf.chmod(0o600)
    r = subprocess.run([sys.executable, "-m", "edge_agent", "--config", str(conf)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "salt is damaged" in r.stderr and "Traceback" not in r.stderr


def _cfg(models):
    return load_edge_config({"tenant_id": "t", "install_id": "i", "models": models, "secret": "s" * 20,
                             "endpoints": ["https://x.invalid"], "state_dir": "/tmp/x"}, need_secret=True)


@pytest.mark.parametrize("pair", [("a.b", "a_b"), ("a/b", "a_b"), ("x" * 96 + "1", "x" * 96 + "2")])
def test_colliding_model_ids_are_rejected(pair):
    with pytest.raises(ValueError, match="share the folder name"):
        _cfg({"m1": pair[0], "m2": pair[1]})


def test_distinct_model_ids_pass():
    assert _cfg({"m1": "a-b", "m2": "a_b", "m3": "ab"}).models["m3"] == "ab"
