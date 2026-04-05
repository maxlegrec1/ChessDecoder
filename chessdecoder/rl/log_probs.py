"""Log-probability computation for GRPO policy gradient.

Only runs the prefix pass (not causal). Board tokens are deterministic
(temperature=0) so their log-probs don't contribute to the policy ratio.
Only move tokens (thinking_policy_head + policy_head) matter.

Returns per-token log-probs so that GRPO can compute per-token ratios
and KL penalties as in the original DeepSeek-Math formulation.
"""

import torch
import torch.nn.functional as F
from torch.amp import autocast

from chessdecoder.models.model import ChessDecoder


def _prefix_forward(
    model: ChessDecoder,
    batch: dict,
    use_amp: bool = True,
) -> torch.Tensor:
    """Run prefix pass and return hidden states [B, S, E].

    Handles Fourier value injection (discretize WL/D to bucket centers).
    """
    input_ids = batch["input_ids"]
    block_id = batch["block_id"]
    wl_positions = batch["wl_positions"]
    d_positions = batch["d_positions"]
    wl_values = batch["wl_values"]
    d_values = batch["d_values"]

    # Discretize values to nearest bucket center for Fourier injection
    wl_fourier = torch.zeros_like(wl_values)
    d_fourier = torch.zeros_like(d_values)

    if wl_positions.any():
        wl_vals = wl_values[wl_positions]
        wl_fourier[wl_positions] = model.discretize_to_bucket(wl_vals, model.wl_bucket_centers)
    if d_positions.any():
        d_vals = d_values[d_positions]
        d_fourier[d_positions] = model.discretize_to_bucket(d_vals, model.d_bucket_centers)

    with autocast("cuda", enabled=use_amp):
        h = model(
            input_ids,
            mask_type="prefix",
            block_id=block_id,
            wl_values=wl_fourier,
            d_values=d_fourier,
            wl_positions=wl_positions,
            d_positions=d_positions,
        )
    return h


def _gather_per_token_log_probs(
    model: ChessDecoder,
    h: torch.Tensor,
    batch: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract per-token log-probs at all move positions.

    Args:
        model: the ChessDecoder (for head access).
        h: hidden states [B, S, E] from prefix pass.
        batch: must contain thinking_move_mask, final_move_mask, move_token_ids.

    Returns:
        token_log_probs: [B, S] log-probs at move positions (0.0 elsewhere).
        move_mask: [B, S] bool — True at positions with valid move log-probs.
    """
    B, S, _E = h.shape
    thinking_mask = batch["thinking_move_mask"]   # [B, S]
    final_mask = batch["final_move_mask"]          # [B, S]
    move_ids = batch["move_token_ids"]             # [B, S] move sub-vocab indices

    token_log_probs = torch.zeros(B, S, device=h.device, dtype=h.dtype)
    move_mask = torch.zeros(B, S, device=h.device, dtype=torch.bool)

    # Thinking move positions → thinking_policy_head
    if thinking_mask.any():
        think_h = h[thinking_mask]                                       # [N_think, E]
        think_logits = model.thinking_policy_head(think_h)               # [N_think, 1924]
        think_log_p = F.log_softmax(think_logits, dim=-1)               # [N_think, 1924]
        think_targets = move_ids[thinking_mask]                          # [N_think]
        think_token_lp = think_log_p.gather(1, think_targets.unsqueeze(1)).squeeze(1)
        token_log_probs[thinking_mask] = think_token_lp
        move_mask[thinking_mask] = True

    # Final move positions → policy_head
    if final_mask.any():
        final_h = h[final_mask]                                          # [N_final, E]
        final_logits = model.policy_head(final_h)                        # [N_final, 1924]
        final_log_p = F.log_softmax(final_logits, dim=-1)               # [N_final, 1924]
        final_targets = move_ids[final_mask]                             # [N_final]
        final_token_lp = final_log_p.gather(1, final_targets.unsqueeze(1)).squeeze(1)
        token_log_probs[final_mask] = final_token_lp
        move_mask[final_mask] = True

    return token_log_probs, move_mask


@torch.no_grad()
def compute_ref_log_probs(
    ref_model: ChessDecoder,
    batch: dict,
    use_amp: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute reference model per-token log-probs (frozen, no gradients).

    Returns:
        token_log_probs: [B, S] per-token log-probs.
        move_mask: [B, S] bool.
    """
    h = _prefix_forward(ref_model, batch, use_amp)
    lp, mask = _gather_per_token_log_probs(ref_model, h, batch)
    return lp.float(), mask


def compute_current_log_probs(
    model: ChessDecoder,
    batch: dict,
    use_amp: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute current policy per-token log-probs (with gradients).

    Returns:
        token_log_probs: [B, S] per-token log-probs.
        move_mask: [B, S] bool.
    """
    h = _prefix_forward(model, batch, use_amp)
    return _gather_per_token_log_probs(model, h, batch)


@torch.no_grad()
def compute_policy_entropy(
    model: ChessDecoder,
    batch: dict,
    use_amp: bool = True,
) -> tuple[float, float]:
    """Compute mean entropy of thinking and final policy distributions.

    Returns:
        (thinking_entropy, final_entropy) as floats.
    """
    h = _prefix_forward(model, batch, use_amp)

    thinking_mask = batch["thinking_move_mask"]
    final_mask = batch["final_move_mask"]

    thinking_ent = 0.0
    if thinking_mask.any():
        logits = model.thinking_policy_head(h[thinking_mask])
        probs = F.softmax(logits, dim=-1)
        ent = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)
        thinking_ent = ent.mean().item()

    final_ent = 0.0
    if final_mask.any():
        logits = model.policy_head(h[final_mask])
        probs = F.softmax(logits, dim=-1)
        ent = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)
        final_ent = ent.mean().item()

    return thinking_ent, final_ent
