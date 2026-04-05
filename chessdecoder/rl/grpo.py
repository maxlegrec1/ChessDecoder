"""Core GRPO algorithm with improvements from recent literature.

Implements per-token GRPO with:
- Per-token importance sampling ratios
- Schulman KL approximation: (r-1) - log(r), always >= 0
- Loss normalization: divide by total tokens across all G sequences in a group
  (avoids length bias between generations)
- Clip-Higher: asymmetric clipping [1-ε_low, 1+ε_high] to allow reinforcement
  of rare but insightful reasoning steps and prevent entropy collapse
- Non-diverse group elimination: groups where all rewards are identical
  contribute zero advantage and are filtered from training batches
"""

import torch


def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute group-relative advantages and a mask of diverse groups.

    Args:
        rewards: [B, G] reward for each completion in each group.

    Returns:
        advantages: [B, G] un-normalized advantages (r_i - mean).
                    Zero for constant-reward groups.
        diverse_mask: [B] bool — True for groups with non-identical rewards.
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    diverse = std.squeeze(1) > eps  # [B]
    advantages = torch.where(
        diverse.unsqueeze(1),
        rewards - mean,
        torch.zeros_like(rewards),
    )
    return advantages, diverse


def normalize_advantages_minibatch(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages within a mini-batch (mean=0, std=1).

    Args:
        advantages: [N] per-sequence advantages.

    Returns:
        normalized: [N] advantages with zero mean and unit variance.
    """
    mean = advantages.mean()
    std = advantages.std()
    if std < eps:
        return torch.zeros_like(advantages)
    return (advantages - mean) / (std + eps)


def grpo_loss(
    token_log_probs: torch.Tensor,
    old_token_log_probs: torch.Tensor,
    ref_token_log_probs: torch.Tensor,
    move_mask: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon_low: float,
    clip_epsilon_high: float,
    kl_coeff: float,
    group_size: int,
) -> tuple[torch.Tensor, dict]:
    """GRPO per-token clipped policy loss with improvements.

    Args:
        token_log_probs:     [N, S] current policy log P(a_t|s) at each position.
        old_token_log_probs: [N, S] log-probs at rollout time (detached).
        ref_token_log_probs: [N, S] reference policy log-probs (detached).
        move_mask:           [N, S] bool — True at valid move token positions.
        advantages:          [N] per-sequence advantages (already filtered to
                             diverse groups and mini-batch normalized).
        clip_epsilon_low:    Lower clipping bound (standard ε).
        clip_epsilon_high:   Upper clipping bound (≥ ε_low, allows more exploration).
        kl_coeff:            KL penalty weight (beta).
        group_size:          G — number of sequences per group (for loss normalization).

    Returns:
        loss: scalar (to be minimized).
        info: dict with diagnostic metrics.
    """
    N = token_log_probs.shape[0]

    # Per-token importance sampling ratio (current vs old policy)
    ratio = torch.exp(token_log_probs - old_token_log_probs)  # [N, S]

    # Per-token clipped surrogate objective (asymmetric: Clip-Higher)
    adv = advantages.unsqueeze(1)  # [N, 1]
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon_low, 1.0 + clip_epsilon_high) * adv
    clipped_obj = torch.min(surr1, surr2)  # [N, S]

    # Per-token KL penalty (Schulman approximation: always >= 0)
    ref_ratio = torch.exp(token_log_probs - ref_token_log_probs)  # [N, S]
    per_token_kl = (ref_ratio - 1) - torch.log(ref_ratio + 1e-8)  # [N, S], >= 0

    # Per-token loss: negative clipped objective + KL penalty
    per_token_loss = -clipped_obj + kl_coeff * per_token_kl  # [N, S]

    # Loss normalization: sum token losses across all sequences, divide by
    # total move tokens in the group.  This avoids length bias — longer
    # sequences don't dominate the gradient just because they have more tokens.
    # With N sequences in a mini-batch (may span multiple groups), we normalize
    # by total tokens across the mini-batch (equivalent to group normalization
    # when mini-batch = one group, and consistent otherwise).
    masked_loss = per_token_loss * move_mask  # [N, S]
    total_tokens = move_mask.sum().clamp(min=1)
    loss = masked_loss.sum() / total_tokens

    # Diagnostics (detached)
    with torch.no_grad():
        masked_ratio = ratio[move_mask]
        clipped_low = (masked_ratio < 1.0 - clip_epsilon_low).float().mean().item()
        clipped_high = (masked_ratio > 1.0 + clip_epsilon_high).float().mean().item()
        clip_fraction = clipped_low + clipped_high
        mean_kl = per_token_kl[move_mask].mean().item() if move_mask.any() else 0.0

    return loss, {
        "policy_loss": (-clipped_obj * move_mask).sum().item() / total_tokens.item(),
        "kl_loss": (kl_coeff * per_token_kl * move_mask).sum().item() / total_tokens.item(),
        "kl": mean_kl,
        "clip_fraction": clip_fraction,
        "clip_fraction_low": clipped_low,
        "clip_fraction_high": clipped_high,
    }
