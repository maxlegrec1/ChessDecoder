"""Shared training utilities used by pretrain, finetune, and RL training loops.

This module contains helpers that were previously duplicated across
``src/train/train.py``, ``src/finetune/train.py`` and ``src/rl/train.py``.
Behavior is intentionally preserved byte-for-byte — this file only centralizes
the definitions.
"""

import yaml
import torch
import torch.nn.functional as F

from src.utils.distributed import print_rank0


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def soft_bucket_loss(logits, target_values, bucket_centers, valid_mask):
    """Soft CE loss: distribute target probability across two nearest buckets via linear interpolation."""
    N = target_values.shape[0]
    n_buckets = bucket_centers.shape[0]

    if N == 0 or valid_mask.sum() == 0:
        # Return zero without requires_grad to avoid memory leak from retaining computation graph
        return torch.tensor(0.0, device=logits.device)

    # Find lower bucket: last bucket where center <= target
    diffs = target_values.unsqueeze(-1) - bucket_centers  # (N, B)
    lower_idx = (diffs >= 0).long().sum(dim=-1) - 1        # (N,)
    lower_idx = lower_idx.clamp(0, n_buckets - 2)
    upper_idx = lower_idx + 1

    lower_centers = bucket_centers[lower_idx]
    upper_centers = bucket_centers[upper_idx]
    span = (upper_centers - lower_centers).clamp(min=1e-8)
    upper_weight = (target_values - lower_centers) / span
    upper_weight = upper_weight.clamp(0.0, 1.0)

    soft_labels = torch.zeros(N, n_buckets, device=logits.device)
    soft_labels.scatter_(1, lower_idx.unsqueeze(1), (1 - upper_weight).unsqueeze(1))
    soft_labels.scatter_(1, upper_idx.unsqueeze(1), upper_weight.unsqueeze(1))

    loss = -(soft_labels * F.log_softmax(logits, dim=-1)).sum(dim=-1)  # (N,)
    return (loss * valid_mask.float()).sum() / (valid_mask.sum() + 1e-8)


def prepare_fourier_inputs(model, wl_targets, d_targets, wl_positions, d_positions):
    """Build Fourier input tensors for WL/D signals, shared by both causal and prefix passes.

    Zero-initialized copies of the target tensors are populated at the valid
    WL/D positions with the discretized bucket-center values.
    """
    wl_fourier_input = torch.zeros_like(wl_targets)
    d_fourier_input = torch.zeros_like(d_targets)

    if wl_positions.any():
        wl_vals_at_pos = wl_targets[wl_positions]
        wl_disc = model.discretize_to_bucket(wl_vals_at_pos, model.wl_bucket_centers)
        wl_fourier_input[wl_positions] = wl_disc

    if d_positions.any():
        d_vals_at_pos = d_targets[d_positions]
        d_disc = model.discretize_to_bucket(d_vals_at_pos, model.d_bucket_centers)
        d_fourier_input[d_positions] = d_disc

    return wl_fourier_input, d_fourier_input


def save_training_checkpoint(path, *, model, optimizer, scaler, step, extra_state=None):
    """Save a training checkpoint to ``path``.

    Shared by pretrain, finetune, and RL. The core dict is always
    ``{step, model_state_dict, optimizer_state_dict, scaler_state_dict}``;
    any caller-supplied ``extra_state`` is merged on top (e.g. ``epoch`` +
    ``epoch_step`` + ``config`` for pretrain/finetune, ``position_stream`` +
    partial ``config`` for RL).

    The model is unwrapped via ``.module`` if present, so DDP-wrapped models
    are handled transparently.
    """
    raw_model = model.module if hasattr(model, "module") else model
    state = {
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    if extra_state:
        state.update(extra_state)
    torch.save(state, path)
    print_rank0(f"Saved checkpoint: {path}")
