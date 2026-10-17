"""Fixtures shared by the test modules (tests/helpers.py and tests/cases.py hold the shared functions)."""

import pytest

from helpers import GPU, gpu_engines


@pytest.fixture(params=gpu_engines())
def gpu(request):
    """Each available GPU engine in turn ("metal", "cuda"; the other skipped): ``gpu.simulate`` is its simulate
    function, ``gpu.name`` its name. The GPU engines share their driver, so one test runs on both."""
    return GPU(request.param)
