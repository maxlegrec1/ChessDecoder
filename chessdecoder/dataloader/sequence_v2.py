"""V2 general mixed-sequence assembler (Phase D core; also used by Phase E/RL).

Phase B's `assemble_decoder_inputs` handles the *regular* pretraining layout
(`[z|move|wl|d]` per ply, closed-form indices). Thinking traces are
**irregular** — variable #variations × variable PV depth, interleaved control
tokens — so we need an explicit ordered *segment plan* and a generic splice:

    plan = [Seg("board", board_ids=<68 ids>),     # -> k latent positions
            Seg("token", token_id=<start_think>), # -> 1 embedding position
            Seg("token", token_id=<root_move>),
            Seg("wl", value=<wl>), Seg("d", value=<d>),
            Seg("board", ...), ...]

`build_mixed_sequence` walks the plan, encodes every board once (batched),
and emits `inputs_embeds [1,S,E]` plus a `pos` map recording the flattened
decoder index of every segment (so the finetune/RL loop can place
policy / thinking_policy / wl / d / transition supervision). This is exactly
the segment-id splice described in markdowns/11 §12.0: board blocks become k
encoder latents, every other token is a single embedding, all in one causal
stream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from chessdecoder.models.vocab import token_to_idx


@dataclass
class Seg:
    kind: str                       # "board" | "token" | "wl" | "d"
    token_id: Optional[int] = None  # for kind=="token"
    value: Optional[float] = None   # for kind=="wl"/"d"
    board_ids: Optional[List[int]] = None  # for kind=="board" (68 full-vocab ids)
    tag: Optional[str] = None       # free label for the caller's bookkeeping
    meta: dict = field(default_factory=dict)


def build_mixed_sequence(model, plan: List[Seg], device=None):
    """plan: ordered list of Seg. Returns (inputs_embeds [1,S,E], pos) where
    pos maps:
      - per-segment span: pos["spans"][i] = (start, end) flattened indices
      - pos["last"][i]   = last decoder index of segment i (where the model's
        prediction *after* that segment is read; for a board this is the last
        latent, for a token it is that token)
      - pos["board_latents"][i] = [1,k,E] latents of board-segment i (for the
        transition head, which conditions on a board's own latents + a move)
    The encoder runs once over all board segments (batched, parallel)."""
    device = device or next(model.parameters()).device
    k = model.num_latents
    E = model.embed_dim

    board_segs = [i for i, s in enumerate(plan) if s.kind == "board"]
    if board_segs:
        bids = torch.tensor([plan[i].board_ids for i in board_segs],
                            dtype=torch.long, device=device)        # [Nb,68]
        lat = model.encode_boards(bids)                              # [Nb,k,E]
    else:
        lat = torch.empty(0, k, E, device=device)

    chunks, spans, last, board_latents = [], [], [], {}
    cur = 0
    bptr = 0
    for i, s in enumerate(plan):
        if s.kind == "board":
            z = lat[bptr:bptr + 1]                                  # [1,k,E]
            board_latents[i] = z
            bptr += 1
            chunks.append(z[0])                                      # [k,E]
            n = k
        elif s.kind == "token":
            e = model.tok_embedding(
                torch.tensor([s.token_id], device=device))           # [1,E]
            chunks.append(e)
            n = 1
        elif s.kind in ("wl", "d"):
            e = model.embed_value(
                torch.tensor([float(s.value)], device=device))       # [1,E]
            chunks.append(e.to(chunks[0].dtype) if chunks else e)
            n = 1
        else:
            raise ValueError(f"bad seg kind {s.kind}")
        spans.append((cur, cur + n))
        last.append(cur + n - 1)
        cur += n

    inputs_embeds = torch.cat(chunks, dim=0).unsqueeze(0)            # [1,S,E]
    return inputs_embeds, {"spans": spans, "last": last,
                           "board_latents": board_latents, "S": cur}


# --- thinking-trace plan builder (V2 analogue of finetune/data.py) ----------

def variation_plan_from_token_ids(ids, block_boundaries, value_positions):
    """Bridge: turn finetune/data.py's flat (ids, block_boundaries,
    value_data) output into a V2 segment plan, so the existing, well-tested
    variation parser is reused verbatim and only the *representation* changes
    (68-token board block -> one Seg("board"); wl/d placeholder ids -> Seg
    with the continuous value). `value_positions` maps a flat wl/d position
    index -> its float value.

    Every position covered by a block_boundary becomes a single board Seg;
    a wl_value/d_value id at a recorded value position becomes a wl/d Seg;
    everything else is a token Seg. Preserves order exactly."""
    wl_id, d_id = token_to_idx["wl_value"], token_to_idx["d_value"]
    # position -> board span index
    in_board = {}
    for bidx, (a, b) in enumerate(block_boundaries):
        for p in range(a, b):
            in_board[p] = bidx
    plan: List[Seg] = []
    p = 0
    n = len(ids)
    while p < n:
        if p in in_board:
            bidx = in_board[p]
            a, b = block_boundaries[bidx]
            plan.append(Seg("board", board_ids=list(ids[a:b]),
                            tag="board", meta={"flat_start": a}))
            p = b
            continue
        tid = ids[p]
        if tid == wl_id and p in value_positions:
            plan.append(Seg("wl", value=value_positions[p], tag="wl",
                            meta={"flat": p}))
        elif tid == d_id and p in value_positions:
            plan.append(Seg("d", value=value_positions[p], tag="d",
                            meta={"flat": p}))
        else:
            plan.append(Seg("token", token_id=tid, tag="token",
                            meta={"flat": p}))
        p += 1
    return plan
