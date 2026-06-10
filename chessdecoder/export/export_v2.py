"""V2 TorchScript export (Phase G, step 1 of 2).

The plan (markdowns/11 §12.3 G) splits V2 export into three modules. Two of
them — the **BoardEncoder** (68 ids -> k latents) and the **TransitionHead**
(latents+move -> next board) — are fixed-shape and batchable, so they
``torch.jit.trace`` cleanly and are exported here with an eager-vs-scripted
parity gate (the contract the future C++ engine consumes).

For the **first-board-only MCTS** path (the C++ V2 engine's evaluator), we
*also* export a bundled ``BoardForward`` module: 68 token ids in, policy
logits + WDL out, fixed seq_len = num_latents + 1. This is the full 1-ply
inference path (encoder + WDLHead + Fourier value injection + decoder over
[z|value] + policy_head at the value slot) collapsed into a single
trace-friendly nn.Module. The full autoregressive decoder export (KV-cache
rebuild) is still Phase-G step 2 and not needed for MCTS-without-thinking.
"""
import json
import os

import torch
import torch.nn as nn

from chessdecoder.models.vocab import (
    token_to_idx, idx_to_token, full_idx_to_move_idx, move_idx_to_full_idx,
    piece_tokens, castling_tokens, vocab_size, move_vocab_size,
)


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


# ---------------------------------------------------------------------------
# BoardForward: the bundled "first-board MCTS evaluator" — the V2 C++ engine
# consumes this single .ts module.
# ---------------------------------------------------------------------------

class BoardForward(nn.Module):
    """Bundle V2's 1-ply inference into one trace-friendly forward.

    Mirrors ``ChessDecoderV2.predict_move`` exactly (sans legal-move masking
    and argmax — those happen on the C++ MCTS side):

        board_ids [B,68]
          → z = encode_boards(board_ids)           [B, k, E]
          → wdl = WDLHead.mean_wdl(WDLHead(z))     [B, 3]
          → value_emb = Fourier(W-L) + Fourier(D)  [B, E]   (model.embed_wdl)
          → seq = concat(z, value_emb.unsqueeze(1)) along dim=1   [B, k+1, E]
          → h = decoder(seq)                       [B, k+1, E]
          → policy = policy_head(h[:, k, :])       [B, 1924]
        returns (policy_logits, wdl_mean)

    Fixed shapes: seq_len = k + 1 (k=16 → 17 positions). The C++ MCTS
    evaluates one node at a time but batches across the leaf-collection
    frontier, so we accept any B.
    """

    def __init__(self, model):
        super().__init__()
        self.encoder = model.board_encoder
        self.wdl_head = model.wdl_head
        self.decoder = model.decoder
        self.policy_head = model.policy_head
        self.fourier_encoder = model.fourier_encoder
        self.num_latents = model.num_latents

    def forward(self, board_ids: torch.Tensor):
        z = self.encoder(board_ids)                              # [B,k,E]
        wdl_logits = self.wdl_head(z)                            # [B,N_CELLS]
        wdl = self.wdl_head.mean_wdl(wdl_logits)                 # [B,3]
        # embed_wdl inlined (so the whole forward is one scriptable graph)
        value_emb = (self.fourier_encoder(wdl[:, 0] - wdl[:, 2])
                     + self.fourier_encoder(wdl[:, 1]))          # [B,E]
        seq = torch.cat([z, value_emb.unsqueeze(1).to(z.dtype)], dim=1)
        h = self.decoder(seq)                                    # [B,k+1,E]
        policy_logits = self.policy_head(h[:, self.num_latents, :])  # [B,1924]
        return policy_logits, wdl


# Squares 1..64 in fen_to_position_tokens map to a1..h8 (rank-major as
# emitted by the dataloader). The board sub-vocab is: "empty" + 6 white +
# 6 black piece tokens (13 classes per square).
_SQUARE_TOKENS = ["empty"] + list(piece_tokens)  # 13 tokens
_STM_TOKENS = ["white_to_move", "black_to_move"]
_STRUCTURAL_TOKENS = ["start_pos", "end_pos"]


def export_v2_board_forward(model, out_dir: str, example_batch: int = 1):
    """Trace BoardForward + dump vocab.json for the C++ V2 MCTS engine.

    Outputs in ``out_dir``:
      - board_forward.ts        — TorchScript of BoardForward (above)
      - vocab.json              — token_to_idx (full vocab subset the cpp
                                  side needs to tokenize FENs) +
                                  move_sub_to_uci (1924 strings, ordered)
      - board_forward_config.json — shape + parity contract

    Returns max-abs-err between eager and scripted on the example batch.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    bf = BoardForward(model).eval()
    ex = torch.zeros(example_batch, 68, dtype=torch.long,
                     device=next(model.parameters()).device)
    with torch.no_grad():
        pol_eager, wdl_eager = bf(ex)
    bf_ts = torch.jit.trace(bf, (ex,), check_trace=False, strict=False)

    with torch.no_grad():
        pol_ts, wdl_ts = bf_ts(ex)
    max_err = max((pol_ts - pol_eager).abs().max().item(),
                  (wdl_ts - wdl_eager).abs().max().item())

    torch.jit.save(bf_ts, os.path.join(out_dir, "board_forward.ts"))

    # ---- vocab.json --------------------------------------------------------
    # Board tokens the cpp side needs to tokenize a FEN. Pick a curated subset
    # (the others are never produced as 'input ids by fen_to_position_tokens):
    #   pieces + "empty" + structural + castling + stm.
    needed = (_SQUARE_TOKENS + _STRUCTURAL_TOKENS
              + list(castling_tokens) + _STM_TOKENS)
    board_token_to_idx = {t: int(token_to_idx[t]) for t in needed}

    # Move sub-vocab: sub_id (0..1923) -> UCI string.
    # move_idx_to_full_idx[sub] is a full-vocab id; idx_to_token resolves to
    # the UCI string the model trained on.
    move_sub_to_uci = [idx_to_token[move_idx_to_full_idx[i]]
                       for i in range(move_vocab_size)]

    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump({
            "board_token_to_idx": board_token_to_idx,
            "move_sub_to_uci": move_sub_to_uci,
            "board_seq_len": 68,
            "move_vocab_size": move_vocab_size,
            "full_vocab_size": vocab_size,
        }, f, indent=2)

    # vocab.txt: a line-oriented twin of vocab.json the C++ side reads (avoids
    # vendoring a JSON parser for one schema we control).
    #
    #   BOARD <n_board>
    #   <token> <id>            (n_board lines)
    #   MOVES <n_move>
    #   <uci>                   (n_move lines, ordered by sub-vocab id)
    with open(os.path.join(out_dir, "vocab.txt"), "w") as f:
        f.write(f"BOARD {len(board_token_to_idx)}\n")
        for tok, idx in board_token_to_idx.items():
            f.write(f"{tok} {idx}\n")
        f.write(f"MOVES {len(move_sub_to_uci)}\n")
        for uci in move_sub_to_uci:
            f.write(f"{uci}\n")

    with open(os.path.join(out_dir, "board_forward_config.json"), "w") as f:
        json.dump({
            "num_latents": model.num_latents,
            "embed_dim": model.embed_dim,
            "decoder_seq_len": model.num_latents + 1,   # [z|value]
            "policy_logits_size": move_vocab_size,
            "wdl_size": 3,
            "input_dtype": "int64",
            "input_shape": "[B, 68]",
            "output_shapes": {
                "policy_logits": f"[B, {move_vocab_size}]",
                "wdl_mean": "[B, 3]",
            },
            "parity_max_abs_err": max_err,
        }, f, indent=2)
    return max_err
