"""
Shared export utilities: model loading, head weights, vocabulary, config.

Used by export_torchscript.py.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.model import ChessDecoder
from src.models.vocab import (
    vocab, vocab_size, token_to_idx,
    board_vocab, board_vocab_size, board_idx_to_full_idx,
    move_vocab, move_vocab_size, move_idx_to_full_idx,
    board_token_to_idx, move_token_to_idx,
)


def load_model(checkpoint_path, device="cpu"):
    """Load a ChessDecoder from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = ChessDecoder(
        vocab_size=vocab_size,
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        max_seq_len=config["model"]["max_seq_len"],
        d_ff=config["model"].get("d_ff"),
        n_buckets=config["model"].get("n_buckets", 100),
        value_hidden_size=config["model"].get("value_hidden_size", 256),
        num_fourier_freq=config["model"].get("num_fourier_freq", 128),
        wl_sigma=config["model"].get("wl_sigma", 0.4),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def save_tensor_fp16(tensor, path):
    """Save tensor as raw FP16 binary."""
    data = tensor.detach().float().cpu().numpy().astype(np.float16)
    data.tofile(str(path))


def save_tensor_fp32(tensor, path):
    """Save tensor as raw FP32 binary."""
    data = tensor.detach().float().cpu().numpy()
    data.tofile(str(path))


def export_head_weights(model, weights_dir):
    """Export all head weights as raw binary files."""
    weights_dir.mkdir(parents=True, exist_ok=True)

    # board_head: Linear(embed_dim -> board_vocab_size)
    save_tensor_fp16(model.board_head.weight, weights_dir / "board_head_weight.bin")
    save_tensor_fp16(model.board_head.bias, weights_dir / "board_head_bias.bin")

    # policy_head: Linear(embed_dim -> move_vocab_size)
    save_tensor_fp16(model.policy_head.weight, weights_dir / "policy_head_weight.bin")
    save_tensor_fp16(model.policy_head.bias, weights_dir / "policy_head_bias.bin")

    # thinking_policy_head: Linear(embed_dim -> move_vocab_size)
    save_tensor_fp16(model.thinking_policy_head.weight, weights_dir / "thinking_policy_head_weight.bin")
    save_tensor_fp16(model.thinking_policy_head.bias, weights_dir / "thinking_policy_head_bias.bin")

    # wl_head: Linear(E -> 256) -> Mish -> Linear(256 -> 100)
    wl_mlp = model.wl_head.mlp
    save_tensor_fp16(wl_mlp[0].weight, weights_dir / "wl_head_w1_weight.bin")
    save_tensor_fp16(wl_mlp[0].bias, weights_dir / "wl_head_w1_bias.bin")
    save_tensor_fp16(wl_mlp[2].weight, weights_dir / "wl_head_w2_weight.bin")
    save_tensor_fp16(wl_mlp[2].bias, weights_dir / "wl_head_w2_bias.bin")

    # d_head: same structure
    d_mlp = model.d_head.mlp
    save_tensor_fp16(d_mlp[0].weight, weights_dir / "d_head_w1_weight.bin")
    save_tensor_fp16(d_mlp[0].bias, weights_dir / "d_head_w1_bias.bin")
    save_tensor_fp16(d_mlp[2].weight, weights_dir / "d_head_w2_weight.bin")
    save_tensor_fp16(d_mlp[2].bias, weights_dir / "d_head_w2_bias.bin")

    # Bucket centers (FP32 for precision)
    save_tensor_fp32(model.wl_bucket_centers, weights_dir / "wl_bucket_centers.bin")
    save_tensor_fp32(model.d_bucket_centers, weights_dir / "d_bucket_centers.bin")

    print(f"  Head weights saved to {weights_dir}/")


def export_vocab(output_dir):
    """Export vocabulary as JSON for C++ engine."""
    vocab_data = {
        "vocab": vocab,
        "token_to_idx": token_to_idx,
        "board_vocab": board_vocab,
        "board_idx_to_full_idx": board_idx_to_full_idx,
        "move_vocab": move_vocab,
        "move_idx_to_full_idx": move_idx_to_full_idx,
        "board_token_to_idx": board_token_to_idx,
        "move_token_to_idx": move_token_to_idx,
    }
    path = output_dir / "vocab.json"
    with open(path, "w") as f:
        json.dump(vocab_data, f, indent=2)
    print(f"  Vocabulary saved to {path}")


def export_config(config, model, output_dir):
    """Export model config for C++ engine."""
    mc = config["model"]
    cfg = {
        "embed_dim": mc["embed_dim"],
        "num_heads": mc["num_heads"],
        "num_layers": mc["num_layers"],
        "max_seq_len": mc["max_seq_len"],
        "d_ff": mc.get("d_ff", 4 * mc["embed_dim"]),
        "vocab_size": vocab_size,
        "head_dim": mc["embed_dim"] // mc["num_heads"],
        "board_vocab_size": board_vocab_size,
        "move_vocab_size": move_vocab_size,
        "n_buckets": mc.get("n_buckets", 100),
        "value_hidden_size": mc.get("value_hidden_size", 256),
        "num_fourier_freq": mc.get("num_fourier_freq", 128),
    }
    path = output_dir / "config.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Config saved to {path}")
