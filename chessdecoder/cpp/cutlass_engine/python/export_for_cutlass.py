"""Export a ChessDecoder checkpoint into the layout cutlass_engine expects.

Layout (under <export_dir>/weights/):
  backbone/
    layer_<i>_sa_norm.bin       FP16 [E]                  scale of pre-attn RMSNorm
    layer_<i>_mlp_norm.bin      FP16 [E]                  scale of pre-mlp RMSNorm
    layer_<i>_qkv_w.bin         FP16 [3*E, E]             concat(q.weight,k.weight,v.weight) along dim 0
    layer_<i>_out_w.bin         FP16 [E, E]               attn.output_proj.weight
    layer_<i>_gate_up_w.bin     FP16 [2*d_ff, E]          concat(mlp.w1.weight, mlp.w3.weight) along dim 0
    layer_<i>_down_w.bin        FP16 [E, d_ff]            mlp.w2.weight
    tok_embedding.bin           FP16 [V, E]
    final_norm.bin              FP16 [E]
    fourier_freq.bin            FP16 [F]                  fourier_encoder.frequencies (squeezed)
    fourier_proj_w.bin          FP16 [E, 2*F]             fourier_encoder.proj.weight
    fourier_proj_b.bin          FP16 [E]                  fourier_encoder.proj.bias

Plus the existing head/bucket files at <export_dir>/weights/ (top-level).
Plus config.json + vocab.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from chessdecoder.export.common import (
    save_tensor_fp16,
    save_tensor_fp32,
    export_head_weights,
    export_vocab,
    export_config,
)


def export_for_cutlass(model, config: dict, export_dir: Path) -> None:
    """Dump everything cutlass_engine needs from `model` into `export_dir`.

    `model` is a ChessDecoder (eager); `config` is the full nested config dict
    used by the existing exporter (with a "model" key).
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = export_dir / "weights"
    backbone_dir = weights_dir / "backbone"
    backbone_dir.mkdir(parents=True, exist_ok=True)

    # 1. Heads + bucket centers + fourier (existing exporter handles heads+buckets).
    export_head_weights(model, weights_dir)

    # 2. Vocab + config.
    export_vocab(export_dir)
    export_config(config, model, export_dir)

    # 3. Backbone per-layer slabs.
    num_layers = len(model.layers)
    for i in range(num_layers):
        L = model.layers[i]
        # RMSNorm scales.
        save_tensor_fp16(L.sa_norm.scale,  backbone_dir / f"layer_{i}_sa_norm.bin")
        save_tensor_fp16(L.mlp_norm.scale, backbone_dir / f"layer_{i}_mlp_norm.bin")

        # Fused QKV: stack [q, k, v] along out_dim. PyTorch convention is
        # [out, in], so concat along dim 0 yields [3*E, E].
        attn = L.attn
        qkv = torch.cat([attn.q_proj.weight,
                         attn.k_proj.weight,
                         attn.v_proj.weight], dim=0)
        save_tensor_fp16(qkv, backbone_dir / f"layer_{i}_qkv_w.bin")

        # Output proj.
        save_tensor_fp16(attn.output_proj.weight, backbone_dir / f"layer_{i}_out_w.bin")

        # Fused gate+up: torchtune's FeedForward names them w1 (gate) / w3 (up) / w2 (down).
        gate_up = torch.cat([L.mlp.w1.weight, L.mlp.w3.weight], dim=0)
        save_tensor_fp16(gate_up, backbone_dir / f"layer_{i}_gate_up_w.bin")

        # Down proj.
        save_tensor_fp16(L.mlp.w2.weight, backbone_dir / f"layer_{i}_down_w.bin")

    # 4. Embedding + final norm.
    save_tensor_fp16(model.tok_embedding.weight, backbone_dir / "tok_embedding.bin")
    save_tensor_fp16(model.norm.scale,           backbone_dir / "final_norm.bin")

    # 5. Fourier encoder. Note: model.fourier_encoder.frequencies is [1, F]
    # in the original model.  We squeeze to [F] for the kernel.
    save_tensor_fp16(model.fourier_encoder.frequencies.squeeze(0),
                     backbone_dir / "fourier_freq.bin")
    save_tensor_fp16(model.fourier_encoder.proj.weight,
                     backbone_dir / "fourier_proj_w.bin")
    save_tensor_fp16(model.fourier_encoder.proj.bias,
                     backbone_dir / "fourier_proj_b.bin")
