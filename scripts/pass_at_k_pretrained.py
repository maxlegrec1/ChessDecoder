"""
Evaluate pass@k for the pretrained (non-thinking) model.
Single forward pass per FEN: encode board -> policy head -> sample move.

Usage:
    uv run python scripts/pass_at_k_pretrained.py \
        --checkpoint checkpoint_616000.pt \
        --num-fens 100 --k 10 --temperature 0.5

Baseline (argmax):
    uv run python scripts/pass_at_k_pretrained.py \
        --checkpoint checkpoint_616000.pt \
        --num-fens 100 --k 1
"""

import argparse
import os
import random
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

from chessdecoder.models.model import ChessDecoder
from chessdecoder.models.vocab import vocab_size


def load_fen_bestmove_pairs(n, seed, data_dir):
    """Load (fen, best_move) pairs from pretraining parquet files."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
    rng = random.Random(seed)
    fname = rng.choice(files)
    table = pq.read_table(os.path.join(data_dir, fname), columns=["fen", "best_move"])
    total = len(table)
    indices = rng.sample(range(total), min(n * 3, total))
    seen = set()
    pairs = []
    for i in indices:
        fen = table.column("fen")[i].as_py()
        if fen not in seen:
            seen.add(fen)
            pairs.append((fen, table.column("best_move")[i].as_py()))
        if len(pairs) >= n:
            break
    return pairs[:n]


def evaluate_pass_at_k(model, pairs, k, temperature):
    """Returns pass@k accuracy."""
    correct = 0
    for fen, best_move in pairs:
        moves = set()
        for _ in range(k):
            moves.add(model.predict_move(fen, temperature=temperature))
        if best_move in moves:
            correct += 1
    return correct / len(pairs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate pass@k for pretrained (non-thinking) model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="/home/maxime/parquet_files_decoder/")
    parser.add_argument("--num-fens", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = ChessDecoder(
        vocab_size=vocab_size, embed_dim=1024, num_heads=16,
        num_layers=12, max_seq_len=256, d_ff=1536,
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    model.to(args.device)

    pairs = load_fen_bestmove_pairs(args.num_fens, args.seed, args.data_dir)
    print(f"Loaded {len(pairs)} FEN/best_move pairs")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Temperature: {args.temperature}, k={args.k}")
    print(f"Evaluating pass@{args.k} ...")

    t0 = time.time()
    score = evaluate_pass_at_k(model, pairs, args.k, args.temperature)
    elapsed = time.time() - t0

    print(f"\npass@{args.k}: {score:.1%} ({int(score * len(pairs))}/{len(pairs)})")
    print(f"Time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
