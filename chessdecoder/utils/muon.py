"""Muon optimizer (Newton-Schulz orthogonalized momentum on 2-D hidden
matrices; AdamW for embeddings / output heads / norms / 1-D params).
"""
import torch
import torch.optim as optim


def _newtonschulz5(G, steps=5, eps=1e-7):
    """Newton-Schulz orthogonalization. Works on a 2-D matrix or a BATCH of
    matrices (3-D, e.g. MoE expert weights ``[E, out, in]``) — the matmuls
    operate on the last two dims so each expert is orthogonalized independently.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)   # per-matrix norm
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-2, -1)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-2, -1)
    return X.to(G.dtype)


class MuonWithAdam(torch.optim.Optimizer):
    def __init__(self, muon_params, adam_params, lr=1e-3, weight_decay=0.0,
                 momentum=0.95, betas=(0.9, 0.95), eps=1e-8):
        groups = [
            dict(params=list(muon_params), kind="muon", lr=lr,
                 weight_decay=weight_decay, momentum=momentum),
            dict(params=list(adam_params), kind="adam", lr=lr,
                 weight_decay=weight_decay, betas=betas, eps=eps),
        ]
        super().__init__(groups, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            lr, wd = g["lr"], g["weight_decay"]
            if g["kind"] == "muon":
                mom = g["momentum"]
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "buf" not in st:
                        # plain momentum buffer (p may be a ScaledGroupedMMTensor
                        # for fp8 MoE experts; keep all the orthogonalization math
                        # on the plain grad tensor and only touch p in-place).
                        st["buf"] = torch.zeros_like(p.grad)
                    buf = st["buf"]
                    buf.mul_(mom).add_(p.grad)
                    upd = p.grad.add(buf, alpha=mom)            # Nesterov
                    o = _newtonschulz5(upd)                     # 2-D or batched 3-D
                    # scale by sqrt(out/in) using the matrix dims (last two axes),
                    # so 3-D expert weights [E, out, in] use out/in, not E.
                    scale = max(1.0, p.shape[-2] / p.shape[-1]) ** 0.5
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.add_(o, alpha=-lr * scale)
            else:  # AdamW
                b1, b2 = g["betas"]; eps = g["eps"]
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "step" not in st:
                        st["step"] = 0
                        st["m"] = torch.zeros_like(p)
                        st["v"] = torch.zeros_like(p)
                    st["step"] += 1
                    m, v = st["m"], st["v"]
                    m.mul_(b1).add_(p.grad, alpha=1 - b1)
                    v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    bc1 = 1 - b1 ** st["step"]
                    bc2 = 1 - b2 ** st["step"]
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.addcdiv_(m / bc1, (v / bc2).sqrt_().add_(eps), value=-lr)
        return loss


def build_optimizer(model, name, lr, wd):
    """name: 'adamw' | 'muon'. Muon routes hidden 2-D matrices through
    Newton-Schulz; embeddings (tok / pos), output heads (policy / wdl), norms
    and 1-D params go to the AdamW arm."""
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "muon":
        muon_p, adam_p = [], []
        # bias_module owns the relpos / geometric attention-bias tables.
        # Those are 2-D lookup tables (``[num_heads, n_buckets]``), not linear
        # projections — Newton-Schulz orthogonalization treats each row as a
        # direction in bucket-space and demolishes the bucket semantics.
        # Route the whole submodule through AdamW.
        # Also exclude smolgen's lookup-table-shaped params explicitly: the
        # per-layer ``sp_bias`` (kind-pair table, same shape semantics as
        # relpos2d's) and ``pos_enc_weight`` (one row per board-square
        # pair — orthogonalizing those rows in gen-size space wrecks the
        # spatial decoding the einsum relies on).
        # ``router`` (MoE gate, a small [E, d] matrix) stays on AdamW — Newton-
        # Schulz orthogonalizes the routing directions and destabilizes routing.
        excluded = ("tok_embedding", "pos_embedding",
                    "policy_head", "wdl_head", "bias_module",
                    "sp_bias", "pos_enc_weight", "router")
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # 2-D Linear matrices AND 3-D MoE expert weights [E, out, in] go to
            # Muon (each expert orthogonalized independently); embeddings / heads
            # / norms / biases / router go to AdamW.
            is_matrix = p.ndim in (2, 3) and not any(k in n for k in excluded)
            (muon_p if is_matrix else adam_p).append(p)
        return MuonWithAdam(muon_p, adam_p, lr=lr, weight_decay=wd)
    raise ValueError(f"unknown optimizer {name!r}")
