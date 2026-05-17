"""2-D simplex value discretization (markdowns/12)."""
import torch

from chessdecoder.models.v2.value_buckets import (
    Q_CENTERS, D_CENTERS, CELL_WDL, CELL_QD, N_CELLS, project_targets)


def test_grid_geometry():
    # odd nQ -> an exact center at Q=0; Q concentrated near 0
    assert (Q_CENTERS.abs() < 1e-6).any()
    sp = torch.diff(Q_CENTERS)
    assert sp[len(sp) // 2] < sp[0]                       # finer near 0
    # every cell inside the simplex triangle |Q| <= 1 - D
    q, d = CELL_QD[:, 0], CELL_QD[:, 1]
    assert (q.abs() <= 1.0 - d + 1e-6).all()
    # cell WDL rows are valid distributions
    assert torch.allclose(CELL_WDL.sum(-1), torch.ones(N_CELLS), atol=1e-5)
    assert (CELL_WDL >= 0).all()


def test_projection_unbiased_and_local():
    # random valid (Q,D) targets -> soft categorical
    torch.manual_seed(0)
    d = torch.rand(500) * 0.9
    q = (torch.rand(500) * 2 - 1) * (1 - d)              # inside the triangle
    t = project_targets(q, d)
    assert t.shape == (500, N_CELLS)
    assert torch.allclose(t.sum(-1), torch.ones(500), atol=1e-5)   # distributions
    assert (t >= 0).all()
    # at most 4 cells active (factorized 2x2 barycentric)
    assert (t > 1e-6).sum(-1).max() <= 4
    # expectation recovers the target (unbiased away from the boundary)
    rec = t @ CELL_WDL                                   # [N,3] mean (W,D,L)
    q_rec, d_rec = rec[:, 0] - rec[:, 2], rec[:, 1]
    assert (q_rec - q).abs().mean() < 0.03
    assert (d_rec - d).abs().mean() < 0.03


def test_projection_bimodal_capable():
    # the whole point: two distinct hypotheses can be represented as mass on
    # two far-apart cells (single Dirichlet/softmax cannot do this).
    t = project_targets(torch.tensor([0.9, -0.9]), torch.tensor([0.0, 0.0]))
    mix = 0.5 * t[0] + 0.5 * t[1]
    active = (mix > 1e-6).nonzero().flatten()
    qs = CELL_QD[active, 0]
    assert qs.max() > 0.5 and qs.min() < -0.5            # mass at both wings
