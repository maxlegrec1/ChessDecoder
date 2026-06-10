"""Shared fixtures and auto-skip logic for the test suite."""

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(pytest.mark.skip(reason="No CUDA GPU available"))
        if "cpp" in item.keywords:
            try:
                import _v2_inference_cpp  # noqa: F401
            except ImportError:
                item.add_marker(pytest.mark.skip(reason="V2 C++ engine not built"))


SAMPLE_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "8/5pk1/6p1/8/8/2B5/5PPP/6K1 w - - 0 40",
    "r1bq1rk1/pp2nppp/2n1p3/3pP3/3P4/2N2N2/PP2BPPP/R1BQ1RK1 b - - 1 10",
    "8/P7/8/8/8/8/8/4K2k w - - 0 1",
]


@pytest.fixture(scope="session")
def sample_fens():
    return SAMPLE_FENS
