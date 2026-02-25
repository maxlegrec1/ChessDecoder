"""
Export ChessDecoder backbone to ONNX (+ optional TRT build).

NOTE: The active inference path uses TorchScript (export_torchscript.py).
This ONNX export is kept for reference but is not required by the C++ engine.

Usage:
    uv run python src/export/export_onnx.py --checkpoint checkpoint.pt --output-dir export/
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.vocab import token_to_idx
from src.dataloader.data import fen_to_position_tokens
from src.export.backbone_causal import from_chess_decoder as causal_from_decoder
from src.export.backbone_prefix import from_chess_decoder as prefix_from_decoder
from src.export.common import (
    load_model, export_head_weights, export_vocab, export_config,
)


def verify_backbone(name, backbone, model, device):
    """Verify exported backbone matches original model on test inputs."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    tokens = fen_to_position_tokens(fen)
    input_ids = torch.tensor([[token_to_idx[t] for t in tokens]], dtype=torch.long, device=device)
    S = input_ids.shape[1]
    input_pos = torch.arange(S, device=device).unsqueeze(0)

    override_values = torch.zeros(1, S, device=device)
    override_mask = torch.zeros(1, S, dtype=torch.bool, device=device)

    if name == "causal":
        h_ref = model(input_ids, input_pos=input_pos, mask_type="causal")

        num_layers = len(model.layers)
        num_heads = model.layers[0].attn.num_heads
        head_dim = model.layers[0].attn.head_dim

        mask = torch.where(
            torch.tril(torch.ones(S, S, device=device)).bool(),
            torch.tensor(0.0, device=device),
            torch.tensor(-1e9, device=device),
        ).unsqueeze(0).unsqueeze(0)

        past_keys = torch.zeros(num_layers, 1, num_heads, 0, head_dim, device=device)
        past_values = torch.zeros(num_layers, 1, num_heads, 0, head_dim, device=device)

        h_export, _, _ = backbone(input_ids, input_pos, mask, past_keys, past_values,
                                  override_values, override_mask)
    else:
        block_id = torch.zeros(1, S, dtype=torch.long, device=device)
        h_ref = model(input_ids, input_pos=input_pos, mask_type="prefix", block_id=block_id)

        causal_mask = torch.tril(torch.ones(S, S, device=device)).bool()
        same_block = block_id.unsqueeze(-1) == block_id.unsqueeze(-2)
        full_mask = causal_mask.unsqueeze(0) | same_block
        attn_mask = torch.where(full_mask, torch.tensor(0.0, device=device),
                                torch.tensor(-1e9, device=device)).unsqueeze(1)

        h_export = backbone(input_ids, input_pos, attn_mask, override_values, override_mask)

    max_err = (h_ref - h_export).abs().max().item()
    mean_err = (h_ref - h_export).abs().mean().item()
    print(f"  {name} verification: max_err={max_err:.6f}, mean_err={mean_err:.6f}")

    if max_err > 1e-3:
        print(f"  WARNING: max error exceeds 1e-3 threshold!")
        return False
    return True


def export_onnx_causal(backbone, output_dir, config):
    """Export causal backbone to ONNX."""
    mc = config["model"]
    num_layers = mc["num_layers"]
    num_heads = mc["num_heads"]
    embed_dim = mc["embed_dim"]
    head_dim = embed_dim // num_heads

    backbone.eval()

    S = 68
    input_ids = torch.zeros(1, S, dtype=torch.long)
    input_pos = torch.arange(S).unsqueeze(0)
    attention_mask = torch.zeros(1, 1, S, S)
    past_keys = torch.zeros(num_layers, 1, num_heads, 0, head_dim)
    past_values = torch.zeros(num_layers, 1, num_heads, 0, head_dim)
    override_values = torch.zeros(1, S)
    override_mask = torch.zeros(1, S, dtype=torch.bool)

    path = output_dir / "backbone_causal.onnx"

    torch.onnx.export(
        backbone,
        (input_ids, input_pos, attention_mask, past_keys, past_values,
         override_values, override_mask),
        str(path),
        input_names=["input_ids", "input_pos", "attention_mask",
                      "past_keys", "past_values",
                      "override_values", "override_mask"],
        output_names=["hidden_states", "present_keys", "present_values"],
        dynamic_axes={
            "input_ids": {1: "seq_len"}, "input_pos": {1: "seq_len"},
            "attention_mask": {2: "seq_len", 3: "total_len"},
            "past_keys": {3: "past_len"}, "past_values": {3: "past_len"},
            "override_values": {1: "seq_len"}, "override_mask": {1: "seq_len"},
            "hidden_states": {1: "seq_len"},
            "present_keys": {3: "total_len"}, "present_values": {3: "total_len"},
        },
        opset_version=18,
        do_constant_folding=True,
    )
    print(f"  Causal backbone exported to {path}")


def export_onnx_prefix(backbone, output_dir, config):
    """Export prefix backbone to ONNX."""
    backbone.eval()

    S = 68
    input_ids = torch.zeros(1, S, dtype=torch.long)
    input_pos = torch.arange(S).unsqueeze(0)
    attention_mask = torch.zeros(1, 1, S, S)
    override_values = torch.zeros(1, S)
    override_mask = torch.zeros(1, S, dtype=torch.bool)

    path = output_dir / "backbone_prefix.onnx"

    torch.onnx.export(
        backbone,
        (input_ids, input_pos, attention_mask, override_values, override_mask),
        str(path),
        input_names=["input_ids", "input_pos", "attention_mask",
                      "override_values", "override_mask"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_ids": {1: "seq_len"}, "input_pos": {1: "seq_len"},
            "attention_mask": {2: "seq_len", 3: "seq_len"},
            "override_values": {1: "seq_len"}, "override_mask": {1: "seq_len"},
            "hidden_states": {1: "seq_len"},
        },
        opset_version=18,
        do_constant_folding=True,
    )
    print(f"  Prefix backbone exported to {path}")


def main():
    parser = argparse.ArgumentParser(description="Export ChessDecoder to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="export")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("Loading checkpoint...")
    model, config = load_model(args.checkpoint, device)
    model.to(device)

    print("Exporting vocabulary...")
    export_vocab(output_dir)

    print("Exporting config...")
    export_config(config, model, output_dir)

    print("Exporting head weights...")
    export_head_weights(model, output_dir / "weights")

    print("Creating causal backbone...")
    causal_backbone = causal_from_decoder(model).to(device)

    print("Creating prefix backbone...")
    prefix_backbone = prefix_from_decoder(model).to(device)

    if not args.skip_verify:
        print("Verifying backbones...")
        ok_causal = verify_backbone("causal", causal_backbone, model, device)
        ok_prefix = verify_backbone("prefix", prefix_backbone, model, device)
        if not ok_causal or not ok_prefix:
            print("VERIFICATION FAILED — aborting export")
            sys.exit(1)
        print("  All verifications passed!")

    print("Exporting causal ONNX...")
    export_onnx_causal(causal_backbone.cpu(), output_dir, config)

    print("Exporting prefix ONNX...")
    export_onnx_prefix(prefix_backbone.cpu(), output_dir, config)

    print("\nDone!")


if __name__ == "__main__":
    main()
