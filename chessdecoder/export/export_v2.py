"""V2 TorchScript export (Phase G, step 1 of 2).

The plan (markdowns/11 §12.3 G) splits V2 export into three modules. Two of
them — the **BoardEncoder** (68 ids -> k latents) and the **TransitionHead**
(latents+move -> next board) — are fixed-shape and batchable, so they
``torch.jit.trace`` cleanly and are exported here with an eager-vs-scripted
parity gate (the contract the future C++ engine consumes).

The **CausalLatentDecoder** is the hard part: like V1, an exportable
KV-cache rebuild is required (cf. `export/backbone_causal.py::BackboneCausal`)
because tracing bakes in the sequence length and there is no incremental
cache. That rebuild is the bulk of the remaining Phase-G work and is
intentionally **not** stubbed here — V2's advantage is it becomes a *textbook*
causal cache (no V1 block-boundary invalidation), but it is still a
from-scratch module + numerical-parity port and belongs in the dedicated
Phase-G session. This file gives the C++ side its two easy modules now and a
single source of truth for the export contract.
"""
import json
import os

import torch


def export_v2_modules(model, out_dir: str, example_batch: int = 1):
    """Trace BoardEncoder + TransitionHead to TorchScript and write them with
    a config.json describing the contract. Returns the parity max-abs-err."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    k, E = model.num_latents, model.embed_dim

    enc_ex = torch.zeros(example_batch, 68, dtype=torch.long)
    with torch.no_grad():
        enc_eager = model.board_encoder(enc_ex)
    enc_ts = torch.jit.trace(model.board_encoder, (enc_ex,),
                             check_trace=False, strict=False)

    lat_ex = torch.randn(example_batch, k, E)
    mv_ex = torch.randn(example_batch, E)
    with torch.no_grad():
        tr_eager = model.transition_head(lat_ex, mv_ex)
    tr_ts = torch.jit.trace(model.transition_head, (lat_ex, mv_ex),
                            check_trace=False, strict=False)

    # parity gate
    with torch.no_grad():
        e_err = (enc_ts(enc_ex) - enc_eager).abs().max().item()
        tr_out = tr_ts(lat_ex, mv_ex)
        t_err = max((tr_out[key] - tr_eager[key]).abs().max().item()
                    for key in ("square", "stm", "castling"))
    max_err = max(e_err, t_err)

    torch.jit.save(enc_ts, os.path.join(out_dir, "board_encoder.ts"))
    torch.jit.save(tr_ts, os.path.join(out_dir, "transition_head.ts"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"num_latents": k, "embed_dim": E,
                   "board_tokens": 68,
                   "modules": {"board_encoder": "[B,68]->[B,k,E]",
                               "transition_head": "([B,k,E],[B,E])->"
                               "{square[B,64,13],stm[B,2],castling[B,16]}"},
                   "decoder": "NOT EXPORTED — needs KV-cache rebuild "
                              "(see export_v2.py docstring / Phase G)",
                   "parity_max_abs_err": max_err}, f, indent=2)
    return max_err
