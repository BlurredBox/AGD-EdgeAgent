import hashlib
import pickle

import pytest

from corpus.onboarding import required_from_bundle


class _Marker:
    def __init__(self, f):
        self.f = f

    def __reduce__(self):
        return (open, (str(self.f), "w"))


@pytest.mark.parametrize("digest", [None, "0" * 64])
def test_path_without_matching_digest_never_unpickles(tmp_path, digest):
    marker = tmp_path / "pwned"
    (tmp_path / "forecaster.pkl").write_bytes(pickle.dumps(_Marker(marker)))
    with pytest.raises(ValueError):
        required_from_bundle(tmp_path, sha256=digest)
    assert not marker.exists()


def test_pinned_non_bundle_is_rejected_after_digest(tmp_path):
    raw = pickle.dumps({"x": 1})
    (tmp_path / "forecaster.pkl").write_bytes(raw)
    with pytest.raises(ValueError):
        required_from_bundle(tmp_path, sha256=hashlib.sha256(raw).hexdigest())
