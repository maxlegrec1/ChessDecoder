"""Top-k Mixture-of-Experts SwiGLU feed-forward (FP8-friendly grouped GEMM).

Drop-in replacement for the dense ``FeedForward`` in ``EncoderLayer``. Experts
are stored as 3-D weight tensors ``[E, in, out]`` and run through
``torch._grouped_mm`` on expert-sorted tokens (the torchtitan pattern), so the
whole expert compute is two grouped GEMMs (gate/up) + SwiGLU + one grouped GEMM
(down). torchao's ``MoETrainingConfig`` swaps those 3-D params for a tensor
subclass that makes ``_grouped_mm`` run in rowwise float8 — see
``chessdecoder/utils/fp8.py``.

Same-FLOP comparison: an MoE with ``num_experts`` experts of width
``expert_d_ff`` routed ``top_k`` ways has the SAME active FFN FLOPs/token as a
dense SwiGLU of width ``top_k * expert_d_ff`` (just ``num_experts/top_k`` times
the params, sparsely accessed). Pick ``expert_d_ff = dense_d_ff / top_k``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFeedForward(nn.Module):
    def __init__(self, embed_dim: int, expert_d_ff: int, num_experts: int = 8,
                 top_k: int = 2, aux_loss_weight: float = 1e-2,
                 capacity_factor: float = None, router_noise: float = 0.0,
                 z_loss_weight: float = 0.0, bias_balance: bool = False,
                 bias_update_rate: float = 1e-3):
        super().__init__()
        self.bias_balance = bias_balance
        self.bias_update_rate = bias_update_rate
        self.embed_dim = embed_dim
        self.expert_d_ff = expert_d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        # router z-loss (ST-MoE): penalize logsumexp(logits)^2 to keep router
        # logits small. Without it the router logits blow up (we measured max~30+,
        # top-1 prob ->1.0, early layers collapse to 2-3 of 8 experts). 0 = off.
        self.z_loss_weight = z_loss_weight
        # capacity_factor: None -> dynamic token padding (no drops, EAGER only,
        #   data-dependent shapes). float -> fixed per-expert capacity
        #   ceil(cap * tokens*top_k/E) rounded to 16: STATIC shapes -> compiles
        #   under fp8, but overflow tokens are dropped.
        self.capacity_factor = capacity_factor
        # router_noise: stddev of Gaussian noise added to router logits in
        # training (Shazeer noisy-gating) for expert exploration. 0 = off.
        self.router_noise = router_noise
        # router stays small (bf16); E < 256 so it's not FP8-eligible anyway.
        self.router = nn.Linear(embed_dim, num_experts, bias=False)
        # Expert weights stored [E, out, in] (nn.Linear convention). In forward
        # we pass ``w.transpose(-2, -1)`` to torch._grouped_mm — that transpose
        # makes the GEMM's B operand column-major, which the float8 grouped GEMM
        # (torchao ScaledGroupedMMTensor) requires. SwiGLU: down(silu(gate)*up).
        self.gate_w = nn.Parameter(torch.empty(num_experts, expert_d_ff, embed_dim))
        self.up_w = nn.Parameter(torch.empty(num_experts, expert_d_ff, embed_dim))
        self.down_w = nn.Parameter(torch.empty(num_experts, embed_dim, expert_d_ff))
        self._reset_parameters()
        # set by forward, read+cleared by the training loop each step.
        self.aux_loss = None   # load-balancing aux
        self.z_loss = None     # router z-loss (tracked separately for logging)
        # DeepSeek-V3 auxiliary-loss-free load balancing: a per-expert bias added
        # to the routing score for SELECTION ONLY (gate weights use the raw probs),
        # nudged each step OUTSIDE gradient descent toward balanced load. Lets us
        # balance experts with NO gradient term polluting the policy/value loss.
        # expert_load stashes this step's per-expert token count for that update.
        self.register_buffer("expert_bias", torch.zeros(num_experts))
        self.expert_load = None

    def _reset_parameters(self):
        # match nn.Linear(bias=False) init (uniform on 1/sqrt(fan_in)) per expert.
        # fan_in is the input dim = last axis of the [E, out, in] weight.
        for w in (self.gate_w, self.up_w, self.down_w):
            bound = (1.0 / w.shape[-1]) ** 0.5
            nn.init.uniform_(w, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.reshape(-1, self.embed_dim)                       # [T, d]
        T = x.shape[0]

        # ---- route ----
        logits = self.router(x.float())                        # [T, E] (router in fp32)
        if self.training and self.router_noise > 0:
            logits = logits + torch.randn_like(logits) * self.router_noise
        probs = logits.softmax(dim=-1)
        if self.bias_balance:
            # DeepSeek: bias added to the pre-softmax affinity affects WHICH experts
            # are picked; the gate weight still comes from the raw softmax probs.
            topk_i = (logits + self.expert_bias).topk(self.top_k, dim=-1).indices
            topk_w = probs.gather(-1, topk_i)
        else:
            topk_w, topk_i = probs.topk(self.top_k, dim=-1)    # [T, k]
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)     # renormalize the k weights

        # load-balancing aux loss (Switch-Transformer): E * sum_i f_i * P_i,
        # f_i = fraction of tokens routed to expert i (top-1 of the k), P_i =
        # mean router prob for expert i. Encourages uniform expert usage.
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(topk_i, self.num_experts).float()   # [T,k,E]
                f_i = one_hot.sum(dim=(0, 1)) / (T * self.top_k)        # [E]
            P_i = probs.mean(dim=0)                                     # [E]
            self.aux_loss = self.aux_loss_weight * self.num_experts * (f_i * P_i).sum()
            # z-loss: penalize large logsumexp so router logits stay small and
            # routing stays soft (keeps experts alive). Tracked separately so it
            # can be logged on its own; the training loop adds BOTH to the total.
            if self.z_loss_weight > 0:
                z = torch.logsumexp(logits, dim=-1)                    # [T]
                self.z_loss = self.z_loss_weight * (z * z).mean()
            else:
                self.z_loss = None
        else:
            self.aux_loss = None
            self.z_loss = None

        # ---- expert-sort the (token, expert) assignments ----
        M = T * self.top_k
        flat_expert = topk_i.reshape(-1)                       # [M]
        flat_weight = topk_w.reshape(-1).to(x.dtype)           # [M]
        flat_token = torch.arange(T, device=x.device).repeat_interleave(self.top_k)
        order = torch.argsort(flat_expert)
        s_expert = flat_expert[order]
        s_token = flat_token[order]
        s_weight = flat_weight[order]
        s_x = x[s_token]                                       # [M, d], expert-sorted
        # token count per expert. bincount is a dynamic-shape op and graph-breaks
        # under torch.compile; one_hot().sum() is static-shape [E] and compiles.
        counts = F.one_hot(flat_expert, self.num_experts).sum(0)
        if self.training and self.bias_balance:
            self.expert_load = counts.detach().float()         # for the bias update
        orig_start = counts.cumsum(0) - counts                 # [E] start of each expert
        within = torch.arange(M, device=x.device) - orig_start[s_expert]  # pos in expert

        E, d = self.num_experts, self.embed_dim
        # ---- no-drop dispatch with a STATIC buffer ----
        # Pad each expert group up to a multiple of 16 (the fp8 grouped-GEMM
        # requires every group size % 16 == 0). The buffer is sized to the static
        # upper bound M + 16*E (each expert pads by < 16 rows, and M = tokens *
        # top_k is static for a fixed batch), so torch.compile sees a fixed shape
        # — no data-dependent .item(). Rows beyond offs[-1] are ignored by
        # _grouped_mm; padding rows within a group are zeros (produce zero output,
        # gathered by no real token). No tokens are dropped, which is what kept
        # the eager path stable — fixed-capacity dropping was the source of NaNs.
        pad_counts = ((counts + 15) // 16) * 16                # [E]
        offs = pad_counts.cumsum(0).to(torch.int32)            # [E] end-offsets
        pad_start = pad_counts.cumsum(0) - pad_counts          # [E]
        dest = pad_start[s_expert] + within                    # [M] padded rows
        px = x.new_zeros(M + 16 * E, d)                        # static upper bound
        px[dest] = s_x

        # ---- grouped expert SwiGLU (transpose -> column-major B for fp8) ----
        gate = torch._grouped_mm(px, self.gate_w.transpose(-2, -1), offs=offs)
        up = torch._grouped_mm(px, self.up_w.transpose(-2, -1), offs=offs)
        h = F.silu(gate) * up
        out_p = torch._grouped_mm(h, self.down_w.transpose(-2, -1), offs=offs)
        out = out_p[dest] * s_weight.unsqueeze(-1)

        # ---- combine back to tokens ----
        y = torch.zeros_like(x)
        y.index_add_(0, s_token, out)
        return y.reshape(orig_shape)

    @torch.no_grad()
    def update_bias(self):
        """DeepSeek loss-free balancing: nudge the per-expert selection bias toward
        balanced load, OUTSIDE gradient descent. Call once per optimizer step.
        Bias up for under-loaded experts, down for over-loaded; recentre to bound it."""
        if not self.bias_balance or self.expert_load is None:
            return
        load = self.expert_load
        err = load.mean() - load                               # >0 = under-loaded
        self.expert_bias += self.bias_update_rate * torch.sign(err)
        self.expert_bias -= self.expert_bias.mean()

    def _capacity(self, T: int) -> int:
        """Per-expert capacity (static for a fixed token count T), rounded up to
        a multiple of 16 for the fp8 grouped GEMM."""
        import math
        cap = math.ceil(self.capacity_factor * T * self.top_k / self.num_experts)
        return ((cap + 15) // 16) * 16
