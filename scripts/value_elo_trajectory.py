"""Sweep VALUE-mode ELO vs SF 2500 across a list of checkpoints, save CSV +
PNG plot.

Mirrors the latched-policy VALUE logic from `eval_v2_vs_sf2000.py`: try the
argmax-policy move; if the value head deems the resulting position a
near-certain win (≥90%) we LATCH to pure policy for the rest of that game;
otherwise pick the legal move whose child position minimizes opponent Q.

Reuses one model object across checkpoints (state_dict swap; no rebuild) so
the per-ckpt cost is only the eval games. The `_orig_mod.` torch.compile
prefix is stripped on load.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/value_elo_trajectory.py \\
        [stride_steps=10000] [n_games=50] [elo=2500] [out_dir=value_traj_out/]
"""
from __future__ import annotations

import csv
import glob
import os
import re
import sys
import time

import chess
import torch

from chessdecoder.dataloader.data import fen_to_position_tokens
from chessdecoder.eval.elo_eval import model_vs_stockfish
from chessdecoder.eval.engine import PytorchModelAdapter
from chessdecoder.models.v2.model_v2 import ChessDecoderV2
from chessdecoder.models.vocab import token_to_idx, vocab_size

# --- args -------------------------------------------------------------------
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
N_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 50
SF_ELO = int(sys.argv[3]) if len(sys.argv) > 3 else 2500
OUT_DIR = sys.argv[4] if len(sys.argv) > 4 else "value_traj_out"
CKPT_DIR = ("checkpoints/v2_pretrain_muon1e3/"
            "v2-pretrain-muon1e3_20260518_075615")
DEV = "cuda"

os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, f"value_elo_traj_sf{SF_ELO}_n{N_GAMES}.csv")
PNG_PATH = os.path.join(OUT_DIR, f"value_elo_traj_sf{SF_ELO}_n{N_GAMES}.png")


def list_ckpts() -> list[tuple[int, str]]:
    """All step-numbered ckpts (sorted by step) matching ``checkpoint_<step>.pt``."""
    paths = glob.glob(os.path.join(CKPT_DIR, "checkpoint_*.pt"))
    out: list[tuple[int, str]] = []
    for p in paths:
        m = re.match(r"^checkpoint_(\d+)\.pt$", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def pick_stride(all_ckpts: list[tuple[int, str]],
                stride: int) -> list[tuple[int, str]]:
    """Every ``stride`` steps, plus the latest ckpt (in case it's off-grid)."""
    if not all_ckpts:
        return []
    picked = [(s, p) for s, p in all_ckpts if s % stride == 0]
    if not picked or picked[-1][0] != all_ckpts[-1][0]:
        picked.append(all_ckpts[-1])
    return picked


# --- build the model ONCE; swap state_dict per ckpt -------------------------
def build_model(cfg) -> ChessDecoderV2:
    mc = cfg["model"]; dc = cfg["data"]
    return ChessDecoderV2(
        vocab_size=vocab_size, embed_dim=mc["embed_dim"], num_heads=mc["num_heads"],
        num_encoder_layers=mc["num_encoder_layers"],
        num_decoder_layers=mc["num_decoder_layers"],
        num_latents=mc["num_latents"],
        decoder_max_seq_len=dc["max_plies"] * (mc["num_latents"] + 2),
        d_ff=mc["d_ff"]).to(DEV).eval()


def load_into(model: ChessDecoderV2, ckpt_path: str) -> None:
    ck = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    sd = {k.replace("_orig_mod.", ""): v
          for k, v in ck["model_state_dict"].items()}
    model.load_state_dict(sd)


# --- per-game latched policy-first VALUE move selector ----------------------
# Mirrors eval_v2_vs_sf2000.py; we keep the same closure here so the trajectory
# sweep uses an identical strategy across checkpoints.
WIN_THRESH = 0.90
_state = {"latched": False, "prev_fullmove": 10**9}


def reset_latch() -> None:
    _state["latched"] = False
    _state["prev_fullmove"] = 10**9


@torch.no_grad()
def value_best_move(model: ChessDecoderV2, fen: str) -> str:
    board = chess.Board(fen)
    fm = board.fullmove_number
    if fm <= 1 or fm < _state["prev_fullmove"]:   # new game -> reset latch
        _state["latched"] = False
    _state["prev_fullmove"] = fm

    legal = list(board.legal_moves)
    if not legal:
        return ""

    if _state["latched"]:
        return model.predict_move(fen, temperature=0.0, force_legal=True)

    # policy-first gate: argmax policy; latch if value says it's a near-win.
    pmv = model.predict_move(fen, temperature=0.0, force_legal=True)
    try:
        pm = chess.Move.from_uci(pmv)
    except ValueError:
        pm = None
    if pm in legal:
        board.push(pm)
        if board.is_checkmate():
            board.pop(); _state["latched"] = True; return pmv
        if not (board.is_stalemate() or board.is_insufficient_material()
                or board.can_claim_draw()):
            toks = fen_to_position_tokens(board.fen())
            wdl = model.predict_wdl(torch.tensor(
                [[token_to_idx[t] for t in toks]],
                dtype=torch.long, device=DEV))[0]
            our_win = wdl[2].item()              # opponent loss = our win
            board.pop()
            if our_win >= WIN_THRESH:
                _state["latched"] = True
                return pmv
        else:
            board.pop()

    cand, child_ids, immediate = [], [], None
    for mv in legal:
        board.push(mv)
        if board.is_checkmate():
            board.pop(); immediate = mv.uci(); break
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.can_claim_draw()):
            cand.append((mv, None))
        else:
            toks = fen_to_position_tokens(board.fen())
            cand.append((mv, len(child_ids)))
            child_ids.append([token_to_idx[t] for t in toks])
        board.pop()
    if immediate:
        return immediate
    if child_ids:
        wdl = model.predict_wdl(
            torch.tensor(child_ids, dtype=torch.long, device=DEV))
        cq = (wdl[:, 0] - wdl[:, 2]).tolist()
    best, best_q = None, 1e9
    for mv, idx in cand:
        qv = 0.0 if idx is None else cq[idx]
        if qv < best_q:
            best_q, best = qv, mv
    return (best or legal[0]).uci()


# --- evaluate one ckpt ------------------------------------------------------
def eval_ckpt(model: ChessDecoderV2, step: int, ckpt_path: str) -> dict:
    load_into(model, ckpt_path)
    reset_latch()
    adapter = PytorchModelAdapter(lambda fen, _t: value_best_move(model, fen))
    t0 = time.time()
    wr, elo = model_vs_stockfish(
        model=adapter, model1_name=f"v2-VALUE-step{step}",
        num_games=N_GAMES, temperature=0.0, elo=SF_ELO,
        pgn_dir=os.path.join(OUT_DIR, "pgns"))
    dt = time.time() - t0
    print(f"[VALUE step={step:>7d}] wr={wr:.3f} elo={elo:.1f} "
          f"({dt:.0f}s)", flush=True)
    return {"step": step, "winrate": wr, "elo": elo, "wall_s": dt,
            "ckpt": os.path.basename(ckpt_path)}


# --- plot -------------------------------------------------------------------
def make_plot(rows: list[dict]) -> None:
    # Headless plot — no display needed.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] / 1000 for r in rows]
    elos = [r["elo"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, elos, "o-", color="#1f77b4", lw=2, ms=6)
    ax.axhline(SF_ELO, color="grey", ls="--", lw=1,
               label=f"SF {SF_ELO} baseline")
    ax.set_xlabel("training step (×1000)")
    ax.set_ylabel(f"estimated ELO vs SF {SF_ELO}")
    ax.set_title(f"V2 VALUE-mode ELO trajectory (N={N_GAMES} games / ckpt, "
                 f"temp 0, latched ≥90%)")
    # ±1σ band on a Bernoulli proportion test for visual context.
    import math
    sigma_elo = 800 / math.log(10) / math.sqrt(N_GAMES)  # rough ELO-σ approx
    ax.fill_between(steps, [e - sigma_elo for e in elos],
                    [e + sigma_elo for e in elos], color="#1f77b4", alpha=0.15,
                    label=f"±1σ from N={N_GAMES}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=130)
    print(f"plot -> {PNG_PATH}", flush=True)


# --- main -------------------------------------------------------------------
def main() -> None:
    all_ckpts = list_ckpts()
    picked = pick_stride(all_ckpts, STRIDE)
    print(f"checkpoints found: {len(all_ckpts)} (range "
          f"{all_ckpts[0][0]}..{all_ckpts[-1][0]})", flush=True)
    print(f"evaluating {len(picked)} ckpts (stride={STRIDE}): "
          f"{[s for s, _ in picked]}", flush=True)

    # read config from the first ckpt to build the model once
    first_ck = torch.load(picked[0][1], map_location="cpu", weights_only=False)
    model = build_model(first_ck["config"])
    del first_ck

    rows: list[dict] = []
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "winrate", "elo",
                                          "wall_s", "ckpt"])
        w.writeheader()
        for step, p in picked:
            r = eval_ckpt(model, step, p)
            rows.append(r)
            w.writerow(r); f.flush()

    make_plot(rows)
    print(f"csv -> {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
