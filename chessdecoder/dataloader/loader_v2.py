"""V2 dataloader / sequence builder (Phase B).

V1 emits one flat token stream with 68-token board blocks. V2's decoder runs
over a *mixed* stream where every board is `k` encoder latents and move/wl/d
are single tokens. The encoder is part of the model, so this loader cannot
emit embeddings — it emits, per game:

  - board_ids   [P,68]  : the board token ids at every ply (encoder input)
  - move_full   [P]     : full-vocab id of the move played from each board
  - policy_tgt  [P]     : best-move move-sub-vocab id  (+ policy_valid [P])
  - wl,d        [P]     : value targets               (+ wdl_valid   [P])
  - trans_*     [P]     : next-ply board class targets (+ trans_valid [P])
  - ply_mask    [P]     : real-ply mask (P padded to a fixed P_max)

The decoder layout is regular and closed-form (no thinking trace yet, that is
Phase D): ply i occupies V2_PLY_LEN = k + 3 contiguous positions
``[ z_i (k latents) | move | wl | d ]`` so

    policy_pos[i] = i*L + (k-1)      # last latent of z_i -> predicts move_i
    move_pos[i]   = i*L + k
    wl_pos[i]     = i*L + k + 1
    d_pos[i]      = i*L + k + 2

``assemble_decoder_inputs`` materializes the mixed embedding stream from the
model's per-ply latents + token embeddings (WL/D get Fourier injection,
exactly V1's mechanism — decoder-side only since WL/D never touch the
encoder). Transition target for ply i is board i+1 (last ply has none).
"""
import glob
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader

from chessdecoder.dataloader.data import fen_to_position_tokens
from chessdecoder.models.vocab import (token_to_idx, full_idx_to_move_idx)
from chessdecoder.models.v2.model_v2 import board_tokens_to_transition_targets
from chessdecoder.models.v2.value_buckets import N_CELLS, project_targets

IGNORE_INDEX = -100


def game_to_v2_arrays(game_df, max_plies):
    """game_df: rows with fen, played_move, best_move, played_q, played_d
    (same schema V1's game_to_token_ids consumes). Returns fixed-shape
    [max_plies, ...] tensors + ply_mask. Plies beyond max_plies are dropped."""
    rows = [r for r in game_df.itertuples(index=False)
            if getattr(r, 'played_move', None)]
    # V1-style random start point (no fixed window): begin the sequence at a
    # uniformly random ply so the model sees mid/endgame histories, then crop
    # the tail to max_plies so the sequence never exceeds the context window.
    if len(rows) > 1:
        start = random.randint(0, len(rows) - 1)
        rows = rows[start:]
    rows = rows[:max_plies]
    P = len(rows)

    board_ids = torch.zeros(max_plies, 68, dtype=torch.long)
    move_full = torch.zeros(max_plies, dtype=torch.long)
    policy_tgt = torch.full((max_plies,), IGNORE_INDEX, dtype=torch.long)
    policy_valid = torch.zeros(max_plies, dtype=torch.bool)
    # Per-position WDL target (markdowns/12), from the position's ORIGINAL
    # eval orig_q/orig_d (a position quantity, unlike played_q/d). We keep
    # both the exact mean WDL [3] (for Fourier value-token injection) and the
    # 2-D-simplex soft categorical [N_CELLS] (the loss target).
    wdl_mean = torch.zeros(max_plies, 3, dtype=torch.float32)
    wdl_tgt = torch.zeros(max_plies, N_CELLS, dtype=torch.float32)
    wdl_valid = torch.zeros(max_plies, dtype=torch.bool)
    ply_mask = torch.zeros(max_plies, dtype=torch.bool)
    q_buf, d_buf, q_idx = [], [], []

    for i, row in enumerate(rows):
        toks = fen_to_position_tokens(row.fen)
        board_ids[i] = torch.tensor([token_to_idx[t] for t in toks])
        move_full[i] = token_to_idx[row.played_move]
        bm = getattr(row, 'best_move', None)
        if bm is not None and bm in token_to_idx:
            policy_tgt[i] = full_idx_to_move_idx[token_to_idx[bm]]
            policy_valid[i] = True
        q, dd = getattr(row, 'orig_q', None), getattr(row, 'orig_d', None)
        if q is not None and pd.notna(q) and dd is not None and pd.notna(dd):
            q, dd = float(q), float(dd)
            w = (1.0 - dd + q) / 2.0
            l = (1.0 - dd - q) / 2.0
            v = torch.tensor([w, dd, l], dtype=torch.float32).clamp_(0.0, 1.0)
            wdl_mean[i] = v / v.sum().clamp_(min=1e-8)
            q_buf.append(max(-1.0, min(1.0, w - l)))
            d_buf.append(dd)
            q_idx.append(i)
            wdl_valid[i] = True
        ply_mask[i] = True

    if q_idx:                                 # vectorized C51-style projection
        cat = project_targets(torch.tensor(q_buf), torch.tensor(d_buf))
        wdl_tgt[torch.tensor(q_idx)] = cat

    # Transition target for ply i is the board at ply i+1.
    tsq = torch.full((max_plies, 64), IGNORE_INDEX, dtype=torch.long)
    tstm = torch.full((max_plies,), IGNORE_INDEX, dtype=torch.long)
    tcas = torch.full((max_plies,), IGNORE_INDEX, dtype=torch.long)
    trans_valid = torch.zeros(max_plies, dtype=torch.bool)
    if P >= 2:
        sq, stm, cas = board_tokens_to_transition_targets(board_ids[1:P])
        tsq[:P - 1] = sq
        tstm[:P - 1] = stm
        tcas[:P - 1] = cas
        trans_valid[:P - 1] = True

    return {
        "board_ids": board_ids, "move_full": move_full,
        "policy_tgt": policy_tgt, "policy_valid": policy_valid,
        "wdl_tgt": wdl_tgt, "wdl_mean": wdl_mean, "wdl_valid": wdl_valid,
        "trans_sq": tsq, "trans_stm": tstm, "trans_cas": tcas,
        "trans_valid": trans_valid, "ply_mask": ply_mask,
    }


def assemble_decoder_inputs(latents, move_emb, value_emb):
    """Build the reshaped decoder stream (markdowns/12).

    Per-ply layout ``[ z (k latents) | value | move ]``, L = k + 2:
      latents   [B,P,k,E]
      value_emb [B,P,E]   model.embed_wdl(target/predicted WDL) — Fourier(Q)+Fourier(D)
      move_emb  [B,P,E]   tok_embedding(played_move)

    Returns inputs_embeds [B, P*L, E] + closed-form positions. The move (and
    policy supervision) is read at the **value slot** so move choice is
    conditioned on board + its evaluation (eval-aware)."""
    B, P, k, E = latents.shape
    L = k + 2
    dev = latents.device
    seq = torch.zeros(B, P, L, E, device=dev, dtype=latents.dtype)
    seq[:, :, :k, :] = latents
    seq[:, :, k, :] = value_emb.to(seq.dtype)
    seq[:, :, k + 1, :] = move_emb
    inputs_embeds = seq.reshape(B, P * L, E)

    idx = torch.arange(P, device=dev) * L
    pos = {"policy_pos": idx + k, "value_pos": idx + k,
           "move_pos": idx + k + 1, "ply_len": L}
    return inputs_embeds, pos


class ChessV2IterableDataset(IterableDataset):
    """Mirrors V1 ChessIterableDataset sharding/shuffling/resumption, but
    yields the per-game V2 arrays (fixed P_max). max_plies caps game length;
    decoder seq len = max_plies * (k + 3)."""

    def __init__(self, parquet_dir, max_plies=128, shuffle_files=True,
                 shuffle_games=True, seed=42, rank=0, world_size=1):
        self.parquet_dir = parquet_dir
        self.max_plies = max_plies
        self.shuffle_files = shuffle_files
        self.shuffle_games = shuffle_games
        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.world_size = world_size
        self.files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        if not self.files:
            print(f"Warning: No parquet files found in {parquet_dir}")

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi is not None else 0
        es = self.seed + self.epoch * 100003 + self.rank * 7919 + wid * 997
        random.seed(es)
        np.random.seed(es % (2**32))

        rank_files = self.files[self.rank::self.world_size]
        if wi is None:
            files = list(rank_files)
        else:
            per = int(math.ceil(len(rank_files) / float(wi.num_workers)))
            files = rank_files[wid * per: min((wid + 1) * per, len(rank_files))]
        if self.shuffle_files:
            random.shuffle(files)

        for fp in files:
            try:
                df = pd.read_parquet(fp)
                grouped = df.groupby('game_id', sort=False)
                gids = list(grouped.groups.keys())
                if self.shuffle_games:
                    np.random.shuffle(gids)
                for gid in gids:
                    gdf = grouped.get_group(gid).sort_values('ply')
                    yield game_to_v2_arrays(gdf, self.max_plies)
            except Exception as e:
                print(f"Error reading file {fp}: {e}")
                continue


def get_v2_dataloader(parquet_dir, batch_size=16, num_workers=0,
                      max_plies=128, seed=42, rank=0, world_size=1):
    ds = ChessV2IterableDataset(parquet_dir, max_plies=max_plies, seed=seed,
                                rank=rank, world_size=world_size)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers), ds
