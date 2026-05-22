"""Measure the current V2 checkpoint vs Stockfish UCI_Elo 2000, temp 0.

  policy : play argmax of policy_head (model.predict_move, temp 0)
  value  : play the move whose CHILD position has the best value for us, i.e.
           argmin over legal moves of the child's Q (=W-L from the child's
           side-to-move POV = the opponent), so minimising opponent score
           == maximising our value. Terminal children handled explicitly
           (checkmate = take it; stalemate/insufficient = draw value 0).

Reads a checkpoint read-only (does not touch the running training).
"""
import sys

import chess
import torch

from chessdecoder.models.vocab import vocab_size
from chessdecoder.models.v2.model_v2 import ChessDecoderV2
from chessdecoder.dataloader.data import fen_to_position_tokens
from chessdecoder.models.vocab import token_to_idx
from chessdecoder.eval.engine import PytorchModelAdapter
from chessdecoder.eval.elo_eval import model_vs_stockfish

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "checkpoints/v2_pretrain_muon1e3/v2-pretrain-muon1e3_20260518_075615/checkpoint_60000.pt"
NGAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 30
ELO = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
DEV = "cuda"

ck = torch.load(CKPT, map_location=DEV, weights_only=False)
cfg = ck["config"]; mc = cfg["model"]
model = ChessDecoderV2(
    vocab_size=vocab_size, embed_dim=mc["embed_dim"], num_heads=mc["num_heads"],
    num_encoder_layers=mc["num_encoder_layers"],
    num_decoder_layers=mc["num_decoder_layers"], num_latents=mc["num_latents"],
    decoder_max_seq_len=cfg["data"]["max_plies"] * (mc["num_latents"] + 2),
    d_ff=mc["d_ff"]).to(DEV).eval()
# strip the ``_orig_mod.`` prefix torch.compile inserts when training the V2
# FP8 path (board_encoder + decoder are wrapped). Older non-FP8 checkpoints
# pass through unchanged.
model.load_state_dict({k.replace("_orig_mod.", ""): v
                       for k, v in ck["model_state_dict"].items()})
print(f"loaded {CKPT} (step {ck.get('step')})  games={NGAMES} vs SF Elo {ELO} temp0",
      flush=True)


WIN_THRESH = 0.90   # once the policy move's resulting position is >=90% a
                    # win for us (per the value head), LATCH to pure policy
                    # for the rest of that game (no more value search).

# per-game latch state (the adapter only sees a FEN, so detect a new game
# from the fullmove counter resetting / going backwards)
_latched = False
_prev_fullmove = 10**9


@torch.no_grad()
def value_best_move(fen: str) -> str:
    global _latched, _prev_fullmove
    board = chess.Board(fen)
    fm = board.fullmove_number
    if fm <= 1 or fm < _prev_fullmove:           # new game -> reset latch
        _latched = False
    _prev_fullmove = fm

    legal = list(board.legal_moves)
    if not legal:
        return ""

    # once latched, pure argmax policy for the whole remaining game
    if _latched:
        return model.predict_move(fen, temperature=0.0, force_legal=True)

    # --- policy-first gate: argmax policy move; if the value head deems the
    # resulting position a near-certain win, LATCH and play it ---
    pmv = model.predict_move(fen, temperature=0.0, force_legal=True)
    try:
        pm = chess.Move.from_uci(pmv)
    except ValueError:
        pm = None
    if pm in legal:
        board.push(pm)
        if board.is_checkmate():                # mate -> take it (latch too)
            board.pop(); _latched = True; return pmv
        if not (board.is_stalemate() or board.is_insufficient_material()
                or board.can_claim_draw()):
            toks = fen_to_position_tokens(board.fen())
            wdl = model.predict_wdl(torch.tensor(
                [[token_to_idx[t] for t in toks]], dtype=torch.long,
                device=DEV))[0]                  # child WDL, opponent POV
            our_win = wdl[2].item()              # opponent's loss = our win
            board.pop()
            if our_win >= WIN_THRESH:
                _latched = True                  # permanent for this game
                return pmv
        else:
            board.pop()

    cand, child_ids, immediate = [], [], None
    for mv in legal:
        board.push(mv)
        if board.is_checkmate():           # we just delivered mate -> take it
            board.pop(); immediate = mv.uci(); break
        if board.is_stalemate() or board.is_insufficient_material() \
           or board.can_claim_draw():
            cand.append((mv, None))        # drawn child -> Q 0
        else:
            toks = fen_to_position_tokens(board.fen())
            cand.append((mv, len(child_ids)))
            child_ids.append([token_to_idx[t] for t in toks])
        board.pop()
    if immediate:
        return immediate
    q = {}
    if child_ids:
        wdl = model.predict_wdl(torch.tensor(child_ids, dtype=torch.long,
                                             device=DEV))            # [n,3]
        cq = (wdl[:, 0] - wdl[:, 2]).tolist()                        # child Q
    # pick the move minimising the child's (opponent) Q
    best, best_q = None, 1e9
    for mv, idx in cand:
        qv = 0.0 if idx is None else cq[idx]
        if qv < best_q:
            best_q, best = qv, mv
    return (best or legal[0]).uci()


policy_ad = PytorchModelAdapter(
    lambda fen, temp: model.predict_move(fen, temperature=0.0, force_legal=True))
value_ad = PytorchModelAdapter(lambda fen, temp: value_best_move(fen))

for name, ad in [("POLICY", policy_ad), ("VALUE(best-child)", value_ad)]:
    print(f"\n===== {name} vs Stockfish {ELO} (temp 0, {NGAMES} games) =====",
          flush=True)
    wr, elo = model_vs_stockfish(model=ad, model1_name=f"v2-{name}",
                                 num_games=NGAMES, temperature=0.0, elo=ELO,
                                 pgn_dir="pgns")
    print(f">>> {name}: winrate={wr:.3f}  estimated_elo={elo}", flush=True)
