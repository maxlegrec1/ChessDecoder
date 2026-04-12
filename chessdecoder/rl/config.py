"""GRPO reinforcement learning configuration."""

from dataclasses import dataclass, field

import yaml


@dataclass
class GRPOConfig:
    # --- Project ---
    project_name: str = "chess-decoder-grpo"
    run_name: str = "grpo-v1"

    # --- GRPO hyperparams ---
    group_size: int = 10
    clip_epsilon_low: float = 0.2
    clip_epsilon_high: float = 0.28
    kl_coeff: float = 0.05
    ppo_epochs: int = 1
    max_kl: float = 0.05

    # --- Rollout ---
    rollout_batch_size: int = 640
    inference_batch_size: int = 64
    think_temperature: float = 1.5
    policy_temperature: float = 1.5
    board_temperature: float = 0.0

    # --- Training ---
    learning_rate: float = 1e-6
    weight_decay: float = 0.1
    warmup_steps: int = 20
    grad_accum_steps: int = 16
    max_grad_norm: float = 1.0
    use_amp: bool = True
    mini_batch_size: int = 4

    # --- Rewards ---
    reward_move_quality_weight: float = 1.0
    reward_format_weight: float = 0.5
    reward_coherence_weight: float = 0.3
    # When true, format and coherence become a hard gate on move_quality
    # rather than additive rewards: total = move_quality_weight * move_quality
    # iff format == 1.0 AND coherence == 1.0, else 0. The format/coherence
    # weights are ignored in this mode.
    format_coherence_as_gate: bool = False

    # --- Data ---
    variation_parquet_dir: str = "./parquets_variations/"
    pretrain_parquet_dir: str = "/home/maxime/parquet_files_decoder/"
    # num_train_positions removed — PositionStream reads fresh data from files
    num_eval_positions: int = 200
    eval_seed: int = 42

    # --- Checkpointing ---
    finetuned_checkpoint: str = ""
    resume_from: str = ""
    checkpoint_dir: str = "checkpoints/"
    save_every: int = 50
    eval_every: int = 50
    log_every: int = 1
    num_outer_steps: int = 500

    # --- Model (same structure as finetune config) ---
    model: dict = field(default_factory=lambda: {
        "embed_dim": 1024,
        "num_heads": 16,
        "num_layers": 12,
        "max_seq_len": 4096,
        "d_ff": 1536,
        "n_buckets": 100,
        "wl_sigma": 0.4,
        "value_hidden_size": 256,
        "num_fourier_freq": 128,
    })

    @classmethod
    def from_yaml(cls, path: str) -> "GRPOConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        kwargs = {}
        # Flatten nested sections into dataclass fields
        for section_key, section in raw.items():
            if section_key == "model":
                kwargs["model"] = section
            elif isinstance(section, dict):
                kwargs.update(section)
            else:
                kwargs[section_key] = section

        # Only pass fields that exist in the dataclass
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in kwargs.items() if k in valid})
