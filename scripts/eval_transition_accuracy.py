"""Audit the V2 TransitionHead's next-board accuracy on real LC0 positions.

Streams (fen, played_move) rows from one parquet, buckets each move by type
(regular / capture / castling / promotion / en passant), runs the transition
head batched, and reports per-category accuracy.

Metric definitions:
  square_acc       = mean over 64 squares of (predicted == true)  -- dominated
                     by ~60/64 "copy from input" squares per move
  changed_acc      = accuracy restricted to squares whose token actually
                     CHANGED (2 for a normal move, 4 for castle, 2-4 for
                     promotion, 3 for en-passant capture) -- the honest signal
  castling_acc     = castling-rights field (1 class out of 16) correct
  stm_acc          = side-to-move toggle correct
  board_total_acc  = all 66 transition cells correct simultaneously

Usage:  CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_transition_accuracy.py
"""
import sys
import os
import random
from collections import defaultdict

import torch
import chess
import pandas as pd

from chessdecoder.models.vocab import vocab_size, token_to_idx, move_token_to_idx
from chessdecoder.models.v2.model_v2 import (
    ChessDecoderV2, board_tokens_to_transition_targets)
from chessdecoder.dataloader.data import fen_to_position_tokens

CKPT = (sys.argv[1] if len(sys.argv) > 1 else
        "/home/maxime/ChessDecoder/checkpoints/v2_pretrain_muon1e3/"
        "v2-pretrain-muon1e3_20260518_075615/checkpoint_436000.pt")
PARQUET_DIR = "/home/maxime/parquet_files_decoder"
N_PER_CAT = {"regular": 1000, "capture": 500, "castling": 200,
             "promotion": 200, "en_passant": 100}
DEV = "cuda"


def classify(board: chess.Board, mv: chess.Move) -> str:
    if board.is_castling(mv):
        return "castling"
    if mv.promotion is not None:
        return "promotion"
    if board.is_en_passant(mv):
        return "en_passant"
    if board.is_capture(mv):
        return "capture"
    return "regular"


CASTLE_MOVES = {"e1g1", "e1c1", "e1h1", "e1a1",
                "e8g8", "e8c8", "e8h8", "e8a8"}


def _find_ep_move(fen: str) -> str | None:
    """Given a FEN whose EP-target square is set, return UCI of a legal EP
    capture move (preferring one that the move-vocab knows about), or None."""
    parts = fen.split()
    if len(parts) < 4 or parts[3] == "-":
        return None
    ep_sq = parts[3]
    stm = parts[1]
    if stm == "w" and ep_sq[1] == "6":
        src_rank = "5"
    elif stm == "b" and ep_sq[1] == "3":
        src_rank = "4"
    else:
        return None
    f_idx = ord(ep_sq[0]) - ord("a")
    try:
        b = chess.Board(fen)
    except Exception:
        return None
    for df in (-1, 1):
        nf = f_idx + df
        if not (0 <= nf <= 7):
            continue
        cand = f"{chr(ord('a') + nf)}{src_rank}{ep_sq}"
        try:
            m = chess.Move.from_uci(cand)
        except Exception:
            continue
        if m in b.legal_moves and b.is_en_passant(m):
            return cand
    return None


def _verify(fen: str, mv_str: str) -> tuple[chess.Board, chess.Move] | None:
    try:
        b = chess.Board(fen)
        m = chess.Move.from_uci(mv_str)
    except Exception:
        return None
    if m not in b.legal_moves:
        return None
    return b, m


def collect(target_counts: dict) -> dict:
    """Pre-filter rows by *string shape* (vectorized pandas) so we only run
    chess.Board on candidate rows — 100x faster than parsing every row."""
    buckets = defaultdict(list)
    files = sorted(f for f in os.listdir(PARQUET_DIR) if f.endswith(".parquet"))
    random.seed(0)
    random.shuffle(files)

    # Shape predicates (move *might* be of that type; verified with chess.Board)
    promo_re = r"^[a-h][27][a-h][18][qrbn]$"
    # EP can only happen from rank 5 (white) or rank 4 (black), diagonal capture.
    ep_re_w = r"^[a-h]5[a-h]6$"   # white pawn from rank 5 captures to 6
    ep_re_b = r"^[a-h]4[a-h]3$"   # black pawn from rank 4 captures to 3

    for fp in files:
        df = pd.read_parquet(os.path.join(PARQUET_DIR, fp),
                             columns=["fen", "played_move"])
        if "played_move" not in df.columns:
            continue
        mv = df["played_move"].astype("string").fillna("")
        in_vocab = mv.isin(set(move_token_to_idx.keys()))
        df = df[in_vocab].reset_index(drop=True)
        mv = df["played_move"].astype("string")
        fen = df["fen"]

        # --- promotions: very tight shape filter, every row needs verify ---
        if len(buckets["promotion"]) < target_counts["promotion"]:
            cand = df[mv.str.match(promo_re)]
            for f, m in zip(cand["fen"].tolist(), cand["played_move"].tolist()):
                if len(buckets["promotion"]) >= target_counts["promotion"]:
                    break
                v = _verify(f, m)
                if v and v[1].promotion is not None:
                    buckets["promotion"].append((f, m))

        # --- castling: 8 specific move strings; verify is_castling on the board
        if len(buckets["castling"]) < target_counts["castling"]:
            cand = df[mv.isin(CASTLE_MOVES)]
            for f, m in zip(cand["fen"].tolist(), cand["played_move"].tolist()):
                if len(buckets["castling"]) >= target_counts["castling"]:
                    break
                v = _verify(f, m)
                if v and v[0].is_castling(v[1]):
                    buckets["castling"].append((f, m))

        # EP captures are *absent from LC0 parquets entirely*: the FEN's EP
        # field is always '-' (verified: 1.26M rows, 0 EP-set FENs). The
        # model's tokenizer also drops the EP field, so the model has never
        # seen any signal that EP is legal. We synthesize EP positions in a
        # separate phase below by playing random games — see `_synth_ep`.

        # --- regular + capture: random subsample of remaining rows, verify ---
        need_reg = target_counts["regular"] - len(buckets["regular"])
        need_cap = target_counts["capture"] - len(buckets["capture"])
        if need_reg > 0 or need_cap > 0:
            # Sample 4x the still-needed count to get enough capture/regular
            sample_n = min(len(df), (need_reg + need_cap) * 4 + 200)
            sample = df.sample(n=sample_n, random_state=len(buckets["regular"]))
            for f, m in zip(sample["fen"].tolist(),
                            sample["played_move"].tolist()):
                if need_reg <= 0 and need_cap <= 0:
                    break
                v = _verify(f, m)
                if not v:
                    continue
                b, mv_obj = v
                if b.is_castling(mv_obj) or mv_obj.promotion is not None \
                   or b.is_en_passant(mv_obj):
                    continue  # belongs to a specialty bucket
                if b.is_capture(mv_obj):
                    if need_cap > 0:
                        buckets["capture"].append((f, m)); need_cap -= 1
                else:
                    if need_reg > 0:
                        buckets["regular"].append((f, m)); need_reg -= 1

        done = all(len(buckets[c]) >= target_counts[c]
                   for c in target_counts if c != "en_passant")
        print(f"  after {fp}: " + "  ".join(
            f"{c}={len(buckets[c])}/{target_counts[c]}"
            for c in target_counts if c != "en_passant"), flush=True)
        if done:
            break

    if target_counts.get("en_passant", 0) > 0:
        print(f"  synthesizing {target_counts['en_passant']} EP positions "
              "via random play (LC0 parquets contain none)...", flush=True)
        buckets["en_passant"] = _synth_ep(target_counts["en_passant"])
        print(f"  en_passant: {len(buckets['en_passant'])}/"
              f"{target_counts['en_passant']} synthesized", flush=True)
    return buckets


def _synth_ep(n_target: int, max_games: int = 5000) -> list:
    """Play random legal games and collect (fen, ep_move) at every position
    where en passant is legal. The FEN preserves the EP-target square, but
    note: `fen_to_position_tokens` drops it — so the model sees a position
    that LOOKS like any other capture-able pawn structure."""
    random.seed(0)
    samples = []
    for _ in range(max_games):
        if len(samples) >= n_target:
            break
        b = chess.Board()
        for _ in range(120):
            if b.is_game_over():
                break
            legal = list(b.legal_moves)
            if not legal:
                break
            # Find EP moves at the current position; if any, sample it.
            ep_moves = [m for m in legal if b.is_en_passant(m)]
            if ep_moves and len(samples) < n_target:
                mv = ep_moves[0]
                if mv.uci() in move_token_to_idx:
                    samples.append((b.fen(), mv.uci()))
            b.push(random.choice(legal))
    return samples


def board_to_ids(fen: str) -> list[int]:
    return [token_to_idx[t] for t in fen_to_position_tokens(fen)]


@torch.no_grad()
def eval_bucket(model, samples, batch=128):
    """Return dict of metrics aggregated across `samples` list of (fen, mv)."""
    sq_correct = torch.zeros((), dtype=torch.long, device=DEV)
    sq_total = 0
    changed_correct = torch.zeros((), dtype=torch.long, device=DEV)
    changed_total = 0
    stm_correct = 0
    cas_correct = 0
    total_pos_correct = 0
    n = 0
    failures = []  # store first few failures for inspection

    for i in range(0, len(samples), batch):
        chunk = samples[i:i + batch]
        pre_ids = torch.tensor([board_to_ids(f) for f, _ in chunk],
                               dtype=torch.long, device=DEV)
        post_ids_list = []
        for f, mv_str in chunk:
            b = chess.Board(f)
            b.push(chess.Move.from_uci(mv_str))
            post_ids_list.append(board_to_ids(b.fen()))
        post_ids = torch.tensor(post_ids_list, dtype=torch.long, device=DEV)
        mv_full = torch.tensor([token_to_idx[mv] for _, mv in chunk],
                               dtype=torch.long, device=DEV)

        z = model.encode_boards(pre_ids)
        move_emb = model.tok_embedding(mv_full)
        out = model.transition_head(z, move_emb)
        pred_ids = model.decode_transition(out)

        # Per-square comparison.
        pre_sq = pre_ids[:, 1:65]
        true_sq = post_ids[:, 1:65]
        pred_sq = pred_ids[:, 1:65]
        sq_eq = (pred_sq == true_sq)
        sq_correct += sq_eq.sum()
        sq_total += sq_eq.numel()

        # Changed squares = squares whose true value differs from input.
        changed = (true_sq != pre_sq)
        if changed.any():
            changed_correct += (sq_eq & changed).sum()
            changed_total += changed.sum().item()

        stm_eq = (pred_ids[:, 67] == post_ids[:, 67])
        cas_eq = (pred_ids[:, 66] == post_ids[:, 66])
        stm_correct += stm_eq.sum().item()
        cas_correct += cas_eq.sum().item()

        total_eq = sq_eq.all(dim=1) & stm_eq & cas_eq
        total_pos_correct += total_eq.sum().item()
        n += len(chunk)

        # Capture up to 3 failures per bucket for inspection.
        if len(failures) < 3:
            bad_idx = (~total_eq).nonzero(as_tuple=True)[0].tolist()
            for j in bad_idx[: max(0, 3 - len(failures))]:
                wrong = (pred_sq[j] != true_sq[j]).nonzero(as_tuple=True)[0].tolist()
                failures.append({
                    "fen": chunk[j][0], "move": chunk[j][1],
                    "wrong_squares": wrong,
                    "stm_wrong": not bool(stm_eq[j].item()),
                    "castling_wrong": not bool(cas_eq[j].item()),
                })

    return {
        "n": n,
        "square_acc": (sq_correct.item() / sq_total) if sq_total else 0.0,
        "changed_acc": (changed_correct.item() / changed_total) if changed_total else float("nan"),
        "stm_acc": stm_correct / n if n else 0.0,
        "castling_acc": cas_correct / n if n else 0.0,
        "board_total_acc": total_pos_correct / n if n else 0.0,
        "failures": failures,
    }


def main():
    print(f"Loading {CKPT}", flush=True)
    ck = torch.load(CKPT, map_location=DEV, weights_only=False)
    mc, dc = ck["config"]["model"], ck["config"]["data"]
    model = ChessDecoderV2(
        vocab_size=vocab_size, embed_dim=mc["embed_dim"],
        num_heads=mc["num_heads"],
        num_encoder_layers=mc["num_encoder_layers"],
        num_decoder_layers=mc["num_decoder_layers"],
        num_latents=mc["num_latents"],
        decoder_max_seq_len=dc["max_plies"] * (mc["num_latents"] + 3),
        d_ff=mc["d_ff"]).to(DEV).eval()
    model.load_state_dict({k.replace("_orig_mod.", ""): v
                           for k, v in ck["model_state_dict"].items()})

    print(f"Collecting positions: {N_PER_CAT}", flush=True)
    buckets = collect(N_PER_CAT)
    for cat, samples in buckets.items():
        print(f"  {cat:11s}: {len(samples):5d} collected", flush=True)

    print(f"\n{'category':12s}  {'N':>5s}  {'square':>8s}  {'changed':>8s}  "
          f"{'stm':>6s}  {'castle':>7s}  {'total':>7s}")
    print("-" * 64)
    for cat in ["regular", "capture", "castling", "promotion", "en_passant"]:
        samples = buckets.get(cat, [])
        if not samples:
            print(f"{cat:12s}  {'-':>5s}  (no samples)")
            continue
        m = eval_bucket(model, samples)
        print(f"{cat:12s}  {m['n']:5d}  {m['square_acc']:8.4f}  "
              f"{m['changed_acc']:8.4f}  {m['stm_acc']:6.4f}  "
              f"{m['castling_acc']:7.4f}  {m['board_total_acc']:7.4f}")
        for f in m["failures"][:2]:
            print(f"    failure  move={f['move']:6s}  "
                  f"wrong_squares={f['wrong_squares']}  "
                  f"stm_wrong={f['stm_wrong']}  cas_wrong={f['castling_wrong']}")
            print(f"             fen={f['fen']}")


if __name__ == "__main__":
    main()
