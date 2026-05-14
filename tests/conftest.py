"""Test configuration — ensures backend reset between tests."""

import pytest
from neurowave.core.backend import set_backend


@pytest.fixture(autouse=True)
def reset_backend():
    """Ensure each test starts with numpy backend."""
    set_backend('numpy')
    yield
    set_backend('numpy')
