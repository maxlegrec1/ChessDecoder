"""
Export ChessDecoder backbone as TorchScript for libtorch C++ inference.

Usage:
    uv run python src/export/export_torchscript.py --checkpoint checkpoint_step_32000.pt --output-dir export/

Outputs:
    export/backbone.pt          - TorchScript traced backbone (FP16)
    export/weights/             - Head + fourier weights as raw FP16/FP32 .bin files
    export/vocab.json           - Full vocabulary mapping
    export/config.json          - Model config for C++ engine
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.export.backbone_causal import from_chess_decoder as causal_from_decoder
from src.export.common import load_model, export_head_weights, export_vocab, export_config


def export_torchscript(backbone, output_dir, config):
    """Trace causal backbone to TorchScript and save."""
    mc = config["model"]
    num_layers = mc["num_layers"]
    num_heads = mc["num_heads"]
    embed_dim = mc["embed_dim"]
    head_dim = embed_dim // num_heads

    backbone.eval().half().cuda()

    # Dummy inputs matching inference shapes.
    # override_values is FP16 to match Python reference:
    #   wl_val = torch.zeros(1, S, dtype=torch.float16, device=device)
    S = 68
    input_ids = torch.zeros(1, S, dtype=torch.long, device="cuda")
    input_pos = torch.arange(S, device="cuda").unsqueeze(0)
    attention_mask = torch.zeros(1, 1, S, S, dtype=torch.float32, device="cuda")
    past_keys = torch.zeros(num_layers, 1, num_heads, 0, head_dim,
                            dtype=torch.float16, device="cuda")
    past_values = torch.zeros(num_layers, 1, num_heads, 0, head_dim,
                              dtype=torch.float16, device="cuda")
    override_values = torch.zeros(1, S, dtype=torch.float16, device="cuda")
    override_mask = torch.zeros(1, S, dtype=torch.bool, device="cuda")

    print("  Tracing with torch.jit.trace...")
    with torch.no_grad():
        traced = torch.jit.trace(
            backbone,
            (input_ids, input_pos, attention_mask,
             past_keys, past_values,
             override_values, override_mask),
        )

    path = output_dir / "backbone.pt"
    traced.save(str(path))
    print(f"  TorchScript saved to {path}")

    # Verify: load back and check exact match
    print("  Verifying round-trip...")
    loaded = torch.jit.load(str(path), map_location="cuda")
    loaded.eval()

    with torch.no_grad():
        h1, k1, v1 = backbone(input_ids, input_pos, attention_mask,
                               past_keys, past_values,
                               override_values, override_mask)
        h2, k2, v2 = loaded(input_ids, input_pos, attention_mask,
                             past_keys, past_values,
                             override_values, override_mask)

    max_err_h = (h1 - h2).abs().max().item()
    max_err_k = (k1 - k2).abs().max().item()
    max_err_v = (v1 - v2).abs().max().item()
    print(f"  hidden max_err={max_err_h:.6f}, keys max_err={max_err_k:.6f}, "
          f"values max_err={max_err_v:.6f}")
    assert max_err_h == 0.0 and max_err_k == 0.0 and max_err_v == 0.0, \
        "TorchScript round-trip verification failed!"
    print("  Verification passed (exact match)")

    # Also verify with non-empty KV cache (incremental mode)
    print("  Verifying incremental mode...")
    with torch.no_grad():
        # First: prefill 68 tokens
        causal_mask = torch.where(
            torch.tril(torch.ones(S, S, device="cuda")).bool(),
            torch.tensor(0.0, device="cuda"),
            torch.tensor(-1e9, device="cuda"),
        ).unsqueeze(0).unsqueeze(0)

        h_pre, pk, pv = loaded(input_ids, input_pos, causal_mask,
                                past_keys, past_values,
                                override_values, override_mask)

        # Then: 1-token incremental
        inc_ids = torch.zeros(1, 1, dtype=torch.long, device="cuda")
        inc_pos = torch.tensor([[S]], dtype=torch.long, device="cuda")
        inc_mask = torch.zeros(1, 1, 1, S + 1, dtype=torch.float32, device="cuda")
        inc_ov = torch.zeros(1, 1, dtype=torch.float16, device="cuda")
        inc_om = torch.zeros(1, 1, dtype=torch.bool, device="cuda")

        h_inc1, _, _ = backbone(inc_ids, inc_pos, inc_mask, pk, pv, inc_ov, inc_om)
        h_inc2, _, _ = loaded(inc_ids, inc_pos, inc_mask, pk, pv, inc_ov, inc_om)

    inc_err = (h_inc1 - h_inc2).abs().max().item()
    print(f"  incremental max_err={inc_err:.6f}")
    assert inc_err == 0.0, "Incremental verification failed!"
    print("  Incremental verification passed (exact match)")

    return traced


def main():
    parser = argparse.ArgumentParser(description="Export ChessDecoder as TorchScript")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--output-dir", default="export",
                        help="Output directory")
    parser.add_argument("--skip-weights", action="store_true",
                        help="Skip re-exporting head weights/vocab/config")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint...")
    model, config = load_model(args.checkpoint, "cpu")

    if not args.skip_weights:
        print("Exporting vocabulary...")
        export_vocab(output_dir)

        print("Exporting config...")
        export_config(config, model, output_dir)

        print("Exporting head weights...")
        export_head_weights(model, output_dir / "weights")

    print("Creating causal backbone...")
    causal_backbone = causal_from_decoder(model)

    print("Exporting TorchScript...")
    export_torchscript(causal_backbone, output_dir, config)

    print("\nDone! Files:")
    print(f"  {output_dir}/backbone.pt")
    if not args.skip_weights:
        print(f"  {output_dir}/weights/")
        print(f"  {output_dir}/vocab.json")
        print(f"  {output_dir}/config.json")


if __name__ == "__main__":
    main()
