"""Shared DDP utilities for single-node multi-GPU training.

Uses manual gradient averaging instead of DDP's automatic sync because the
training loop does two forward passes (causal + prefix) with separate head
calls before a single backward. DDP's assumption of forward→backward pairing
doesn't fit this pattern.

All functions are no-ops when torch.distributed is not initialized,
so training scripts remain backwards-compatible with plain `python` launch.
"""

import os

import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed process group from torchrun env vars.

    Returns (rank, local_rank, world_size).
    Falls back to (0, 0, 1) when launched without torchrun.
    """
    if "RANK" not in os.environ:
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed():
    """Destroy distributed process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Return True on rank 0 or when not distributed."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_device(local_rank):
    """Return the CUDA device for this local rank."""
    return torch.device(f"cuda:{local_rank}")


def average_gradients(model):
    """All-reduce and average gradients across ranks.

    No-op when not distributed.
    """
    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)


def barrier():
    """Synchronization barrier if distributed is initialized."""
    if dist.is_initialized():
        dist.barrier()


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)
