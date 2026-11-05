"""Without the BlurredBox SDK the rest of this directory is not collected (see conftest.py); this shows why."""
import pytest


def test_sdk_is_installed():
    pytest.importorskip("blurredbox_agd", reason="the BlurredBox SDK (blurredbox_agd) is not installed")
