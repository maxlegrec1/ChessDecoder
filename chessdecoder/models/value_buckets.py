"""2-D simplex value discretization (markdowns/12).

WDL lives on the 2-simplex; in (Q=W-L, D) coords the valid region is the
triangle |Q| <= 1 - D.  We tile it with a product grid:

  * Q centers  : Gaussian-CDF quantiles -> concentrated near Q=0 (sigma),
                 odd nQ so a center sits exactly at 0; coarse toward +-1.
  * D centers  : uniform in (0, 1).
  * keep only cells whose center is inside the simplex (|Q| <= 1 - D).

A continuous target (q, d) is turned into a *soft* categorical by a
factorized C51-style barycentric projection: 1-D barycentric weights on Q
and on D, outer-product over the (<=4) surrounding cells, invalid cells
dropped + renormalized.  This keeps the per-axis expectation unbiased
(E[Q]=q, E[D]=d) so the value the model acts on stays correct.

Single source of truth: the model uses CELL_WDL (cell centers as W,D,L) for
its output dim + decoding; the loader uses ``project_targets``.
"""
import math

import torch

NQ = 51          # Q levels (odd -> exact center at Q=0)
ND = 13          # D levels (uniform)
SIGMA = 0.5      # Q concentration near 0 (smaller = tighter)


def _q_centers(nq=NQ, sigma=SIGMA):
    t = torch.linspace(0.5 / nq, 1 - 0.5 / nq, nq)
    c = (sigma * math.sqrt(2) * torch.erfinv(2 * t - 1)).clamp(-1.0, 1.0)
    return torch.unique(torch.round(c * 1e6) / 1e6)            # sorted, dedup


def _d_centers(nd=ND):
    return torch.linspace(0.5 / nd, 1 - 0.5 / nd, nd)


def _edges(c):
    m = (c[:-1] + c[1:]) / 2
    return torch.cat([(2 * c[0] - m[0]).view(1), m, (2 * c[-1] - m[-1]).view(1)])


def _build():
    Qc, Dc = _q_centers(), _d_centers()
    nQ, nD = len(Qc), len(Dc)
    # (qi,dj) -> cell id  (-1 = outside simplex)
    cid = torch.full((nQ, nD), -1, dtype=torch.long)
    wdl, qd = [], []
    k = 0
    for i in range(nQ):
        for j in range(nD):
            q, d = Qc[i].item(), Dc[j].item()
            if abs(q) <= 1.0 - d + 1e-9:
                w = (1.0 - d + q) / 2.0
                l = (1.0 - d - q) / 2.0
                v = torch.tensor([max(w, 0.0), d, max(l, 0.0)])
                wdl.append(v / v.sum())
                qd.append([q, d])
                cid[i, j] = k
                k += 1
    return (Qc, Dc, cid,
            torch.stack(wdl),                  # [n_cells,3] (W,D,L)
            torch.tensor(qd))                  # [n_cells,2] (Q,D)


Q_CENTERS, D_CENTERS, _CID, CELL_WDL, CELL_QD = _build()
N_CELLS = CELL_WDL.shape[0]


def mean_wdl(logits: torch.Tensor, cell_wdl: torch.Tensor) -> torch.Tensor:
    """[..., N_CELLS] cell logits -> [..., 3] mean WDL = E_p[(W,D,L)]."""
    p = torch.softmax(logits.float(), dim=-1)
    return p @ cell_wdl.to(p.dtype)


def _bary(centers, v):
    """1-D barycentric: v [N] -> (lo_idx, hi_idx, w_lo, w_hi) onto sorted
    ``centers`` (value clamped into range)."""
    v = v.clamp(centers[0], centers[-1])
    hi = torch.searchsorted(centers, v).clamp(1, len(centers) - 1)
    lo = hi - 1
    span = (centers[hi] - centers[lo]).clamp(min=1e-8)
    w_hi = ((v - centers[lo]) / span).clamp(0.0, 1.0)
    return lo, hi, 1.0 - w_hi, w_hi


def project_targets(q, d):
    """q,d: [N] floats -> soft categorical target [N, N_CELLS] (rows sum to
    1; per-axis expectation unbiased away from the simplex boundary)."""
    q = q.flatten().float()
    d = d.flatten().float()
    N = q.shape[0]
    qlo, qhi, wql, wqh = _bary(Q_CENTERS.to(q.device), q)
    dlo, dhi, wdl_, wdh = _bary(D_CENTERS.to(d.device), d)
    cidm = _CID.to(q.device)
    out = torch.zeros(N, N_CELLS, device=q.device)
    corners = [(qlo, dlo, wql * wdl_), (qlo, dhi, wql * wdh),
               (qhi, dlo, wqh * wdl_), (qhi, dhi, wqh * wdh)]
    for qi, dj, w in corners:
        c = cidm[qi, dj]                                   # [N], -1 if invalid
        valid = c >= 0
        idx = torch.nonzero(valid, as_tuple=True)[0]
        if idx.numel():
            out[idx, c[idx]] += w[idx]
    s = out.sum(-1, keepdim=True)
    bad = (s.squeeze(-1) <= 1e-8)
    if bad.any():                                          # snap to nearest cell
        qd = CELL_QD.to(q.device)
        dist = ((q[bad, None] - qd[None, :, 0]) ** 2
                + (d[bad, None] - qd[None, :, 1]) ** 2)
        out[bad] = 0.0
        out[bad, dist.argmin(-1)] = 1.0
        s = out.sum(-1, keepdim=True)
    return out / s
