"""Core GRPO algorithm: group-relative advantages, clipped policy loss, KL penalty."""

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
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
    kl_coeff: float,
) -> tuple[torch.Tensor, dict]:
    """GRPO clipped policy loss with KL penalty.

    Args:
        log_probs:     [N] current policy log P(action|state).
        old_log_probs: [N] log-probs at rollout time (detached).
        ref_log_probs: [N] reference policy log-probs (detached).
        advantages:    [N] group-relative advantages (detached).
        clip_epsilon:  PPO clip range.
        kl_coeff:      KL penalty weight (beta).

    Returns:
        loss: scalar (to be minimized).
        info: dict with diagnostic metrics.
    """
    # Importance sampling ratio
    ratio = torch.exp(log_probs - old_log_probs)

    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Approximate KL divergence (DeepSeek-Math / GRPO formulation)
    approx_kl = (log_probs - ref_log_probs).mean()
    kl_loss = kl_coeff * approx_kl

    loss = policy_loss + kl_loss

    with torch.no_grad():
        clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean().item()

    return loss, {
        "policy_loss": policy_loss.item(),
        "kl_loss": kl_loss.item(),
        "kl": approx_kl.item(),
        "clip_fraction": clip_fraction,
    }
