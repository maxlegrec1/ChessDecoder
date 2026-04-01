"""Shared fixtures and auto-skip logic for the test suite."""

import pytest
import torch


# ---------------------------------------------------------------------------
# Auto-skip markers
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(pytest.mark.skip(reason="No CUDA GPU available"))
        if "cpp" in item.keywords:
            try:
                import _decoder_inference_cpp  # noqa: F401
            except ImportError:
                item.add_marker(pytest.mark.skip(reason="C++ engine not built"))


# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------

SAMPLE_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",          # starting position
    "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # open game
    "8/5pk1/6p1/8/8/2B5/5PPP/6K1 w - - 0 40",                              # endgame, no castling
    "r1bq1rk1/pp2nppp/2n1p3/3pP3/3P4/2N2N2/PP2BPPP/R1BQ1RK1 b - - 1 10", # black to move
    "8/P7/8/8/8/8/8/4K2k w - - 0 1",                                        # promotion
]


@pytest.fixture(scope="session")
def sample_fens():
    return SAMPLE_FENS


# ---------------------------------------------------------------------------
# Tiny model (GPU)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_model():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")
    from src.models.vocab import vocab_size
    from src.models.model import ChessDecoder
    model = ChessDecoder(
        vocab_size=vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        max_seq_len=512,
        d_ff=128,
        n_buckets=100,
        value_hidden_size=32,
        num_fourier_freq=16,
    ).to("cuda").eval()
    return model


# ---------------------------------------------------------------------------
# C++ engines
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def single_engine():
    cpp = pytest.importorskip("_decoder_inference_cpp")
    engine = cpp.ThinkingInferenceEngine(
        "exports/backbone.pt", "exports/weights",
        "exports/vocab.json", "exports/config.json",
    )
    return engine


@pytest.fixture(scope="session")
def batched_engine():
    cpp = pytest.importorskip("_decoder_inference_cpp")
    engine = cpp.BatchedInferenceEngine(
        "exports/backbone.pt", "exports/weights",
        "exports/vocab.json", "exports/config.json",
        8,
    )
    return engine
