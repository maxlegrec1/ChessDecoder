"""2-D-simplex value bucket helpers (CPU)."""
import torch

from chessdecoder.models.value_buckets import (
    CELL_WDL, N_CELLS, mean_wdl, project_targets,
)


def test_cell_wdl_lives_on_simplex():
    assert CELL_WDL.shape == (N_CELLS, 3)
    s = CELL_WDL.sum(-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    assert (CELL_WDL >= 0).all()


def test_project_targets_returns_distribution():
    q = torch.tensor([0.0, -0.5, 0.8])
    d = torch.tensor([0.3, 0.5, 0.1])
    out = project_targets(q, d)
    assert out.shape == (3, N_CELLS)
    assert torch.allclose(out.sum(-1), torch.ones(3), atol=1e-5)
    assert (out >= 0).all()


def test_mean_wdl_matches_input_under_uniform_logits():
    """Uniform logits -> mean WDL is the mean of CELL_WDL (by construction)."""
    logits = torch.zeros(N_CELLS)
    p = mean_wdl(logits, CELL_WDL)
    assert p.shape == (3,)
    assert torch.allclose(p.sum(), torch.tensor(1.0), atol=1e-5)


def test_mean_wdl_is_a_distribution_for_random_logits():
    logits = torch.randn(5, N_CELLS)
    p = mean_wdl(logits, CELL_WDL)
    assert p.shape == (5, 3)
    assert torch.allclose(p.sum(-1), torch.ones(5), atol=1e-5)
    assert (p >= 0).all()
