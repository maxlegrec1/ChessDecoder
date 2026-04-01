"""Core GRPO algorithm: group-relative advantages, per-token clipped loss, KL penalty.

Implements the DeepSeek-Math GRPO formulation with:
- Per-token importance sampling ratios (not per-sequence)
- Schulman KL approximation: (r-1) - log(r), always >= 0
- Per-sequence averaging of token losses, then cross-sequence mean
"""

import torch


def compute_group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute group-relative advantages.

    Args:
        rewards: [B, G] reward for each completion in each group.

    Returns:
        advantages: [B, G] normalized advantages. Zero for constant-reward groups.
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    # Groups where all rewards are identical get zero advantage
    return torch.where(std > eps, (rewards - mean) / (std + eps), torch.zeros_like(rewards))


def grpo_loss(
    token_log_probs: torch.Tensor,
    old_token_log_probs: torch.Tensor,
    ref_token_log_probs: torch.Tensor,
    move_mask: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
    kl_coeff: float,
) -> tuple[torch.Tensor, dict]:
    """GRPO per-token clipped policy loss with KL penalty.

    Args:
        token_log_probs:     [N, S] current policy log P(a_t|s) at each position.
        old_token_log_probs: [N, S] log-probs at rollout time (detached).
        ref_token_log_probs: [N, S] reference policy log-probs (detached).
        move_mask:           [N, S] bool — True at valid move token positions.
        advantages:          [N] per-sequence group-relative advantages (detached).
        clip_epsilon:        PPO clip range.
        kl_coeff:            KL penalty weight (beta).

    Returns:
        loss: scalar (to be minimized).
        info: dict with diagnostic metrics.
    """
    # Per-token importance sampling ratio (current vs old policy)
    ratio = torch.exp(token_log_probs - old_token_log_probs)  # [N, S]

    # Per-token clipped surrogate objective
    # advantages [N] → [N, 1] for broadcasting across S
    adv = advantages.unsqueeze(1)  # [N, 1]
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
    clipped_obj = torch.min(surr1, surr2)  # [N, S]

    # Per-token KL penalty (Schulman approximation: always >= 0)
    # KL(pi_theta || pi_ref) ≈ (r - 1) - log(r) where r = pi_theta / pi_ref
    ref_ratio = torch.exp(token_log_probs - ref_token_log_probs)  # [N, S]
    per_token_kl = (ref_ratio - 1) - torch.log(ref_ratio + 1e-8)  # [N, S], >= 0

    # Per-token loss: negative clipped objective + KL penalty
    per_token_loss = -clipped_obj + kl_coeff * per_token_kl  # [N, S]

    # Mask to valid move positions and average per-sequence, then across sequences
    per_token_loss = per_token_loss * move_mask  # [N, S]
    tokens_per_seq = move_mask.sum(dim=1).clamp(min=1)  # [N]
    per_seq_loss = per_token_loss.sum(dim=1) / tokens_per_seq  # [N]
    loss = per_seq_loss.mean()

    # Diagnostics (detached)
    with torch.no_grad():
        masked_ratio = ratio[move_mask]
        clip_fraction = ((masked_ratio - 1.0).abs() > clip_epsilon).float().mean().item()
        mean_kl = per_token_kl[move_mask].mean().item()

    return loss, {
        "policy_loss": (-clipped_obj * move_mask).sum().item() / tokens_per_seq.sum().item(),
        "kl_loss": (kl_coeff * per_token_kl * move_mask).sum().item() / tokens_per_seq.sum().item(),
        "kl": mean_kl,
        "clip_fraction": clip_fraction,
    }
