"""Shared training utilities used by the V2 pretraining loop."""

import os

import yaml
import wandb
import torch

from chessdecoder.utils.distributed import print_rank0


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_training_checkpoint(path, *, model, optimizer, scaler, step, extra_state=None):
    """Save a training checkpoint. DDP-wrapped models are unwrapped via ``.module``;
    the ``_orig_mod.`` prefix that ``torch.compile`` inserts (FP8 path wraps encoder
    + decoder) is stripped so plain consumers can load_state_dict directly."""
    raw_model = model.module if hasattr(model, "module") else model
    sd = {k.replace("_orig_mod.", ""): v for k, v in raw_model.state_dict().items()}
    state = {
        "step": step,
        "model_state_dict": sd,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    if extra_state:
        state.update(extra_state)
    torch.save(state, path)
    print_rank0(f"Saved checkpoint: {path}")


def init_wandb_with_resume(*, project, run_name, config, checkpoint_dir):
    """Init wandb, resuming via ``<checkpoint_dir>/wandb_run_id.txt`` if present.
    Caller is responsible for gating on ``is_main_process()``."""
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
