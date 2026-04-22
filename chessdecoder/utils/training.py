"""Shared training utilities used by pretrain, finetune, and RL training loops.

This module contains helpers that were previously duplicated across
``chessdecoder/train/train.py``, ``chessdecoder/finetune/train.py`` and ``chessdecoder/rl/train.py``.
Behavior is intentionally preserved byte-for-byte — this file only centralizes
the definitions.
"""

import os

import yaml
import wandb
import torch
import torch.nn.functional as F

from chessdecoder.utils.distributed import print_rank0


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


def compute_board_block_metrics(board_correct, board_mask, block_id):
    """Per-block board-prediction accuracy, restricted to real 68-token board blocks.

    The loader assigns a unique ``block_id`` to each orphan position (for prefix
    attention masking), which makes them indistinguishable from real blocks in a
    raw scatter. We identify real blocks as those containing square positions
    (intra-block indices 1-64) and drop everything else.

    Returns ``board_total_acc`` (all non-structural tokens in the block correct),
    ``board_square_acc`` (all 64 squares correct), and per-token
    ``board_castling_acc`` / ``board_stm_acc``.
    """
    device = board_correct.device
    zeros = {"board_total_acc": 0.0, "board_square_acc": 0.0,
             "board_castling_acc": 0.0, "board_stm_acc": 0.0}

    B, _ = block_id.shape
    max_bid = block_id.max() + 1
    uid = (block_id + (torch.arange(B, device=device) * max_bid).unsqueeze(1)).view(-1)
    bp = board_mask.view(-1).nonzero(as_tuple=True)[0]
    if bp.numel() == 0:
        return zeros

    bp_bc = board_correct.view(-1).float()[bp]
    _, inv = uid[bp].unique(return_inverse=True)
    n = int(inv.max().item()) + 1
    ones = torch.ones_like(bp_bc)

    # Intra-block offset relative to the earliest masked position in the block.
    block_starts = torch.full((n,), float('inf'), device=device)
    block_starts.scatter_reduce_(0, inv, bp.float(), reduce="amin")
    intra = (bp.float() - block_starts[inv]).long()

    # Real blocks contain square predictions (intra 1-64). Orphan single-token
    # blocks only ever sit at intra 0, so they're excluded.
    sq = (intra >= 1) & (intra <= 64)
    if not sq.any():
        return zeros

    def _per_block_sum(values, groups):
        return torch.zeros(n, device=device).scatter_add_(0, groups, values)

    sq_correct = _per_block_sum(bp_bc[sq], inv[sq])
    sq_total   = _per_block_sum(ones[sq],  inv[sq])
    real = sq_total > 0
    n_real = real.sum().clamp(min=1)

    all_correct = _per_block_sum(bp_bc, inv)
    all_total   = _per_block_sum(ones,  inv)

    castle = intra == 65
    stm    = intra == 66

    return {
        "board_total_acc":    (((all_correct == all_total) & real).float().sum() / n_real).item(),
        "board_square_acc":   (((sq_correct  == sq_total)  & real).float().sum() / n_real).item(),
        "board_castling_acc": bp_bc[castle].mean().item() if castle.any() else 0.0,
        "board_stm_acc":      bp_bc[stm].mean().item()    if stm.any()    else 0.0,
    }


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


def load_pretrained_checkpoint(model, checkpoint_path, device):
    """Load pretrained weights into ``model`` and clone policy_head → thinking_policy_head.

    Pretraining does not train ``thinking_policy_head``, so at the start of
    finetuning (and RL, which starts from a finetuned checkpoint) we
    initialize it from the trained ``policy_head``.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    state_dict["thinking_policy_head.weight"] = state_dict["policy_head.weight"].clone()
    state_dict["thinking_policy_head.bias"] = state_dict["policy_head.bias"].clone()
    model.load_state_dict(state_dict)


def init_wandb_with_resume(*, project, run_name, config, checkpoint_dir):
    """Initialize a wandb run, resuming via ``wandb_run_id.txt`` if present.

    On a fresh run, starts a new wandb run and persists its ID to
    ``<checkpoint_dir>/wandb_run_id.txt`` so a future resume picks it up.
    On resume, reads the ID and starts wandb with ``resume="must"``.

    Caller is responsible for gating on ``is_main_process()``.
    """
    wandb_id_path = os.path.join(checkpoint_dir, "wandb_run_id.txt")
    wandb_run_id = None
    if os.path.exists(wandb_id_path):
        wandb_run_id = open(wandb_id_path).read().strip()
        print_rank0(f"Resuming wandb run: {wandb_run_id}")

    wandb.init(
        project=project,
        name=run_name,
        config=config,
        id=wandb_run_id,
        resume="must" if wandb_run_id else None,
    )

    if not wandb_run_id:
        with open(wandb_id_path, "w") as f:
            f.write(wandb.run.id)
        print_rank0(f"Saved wandb run ID to {wandb_id_path}")
