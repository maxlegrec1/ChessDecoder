"""Agent token vocabulary: patch-encoded boards + moves + bins + controls.

Board = 19 self-contained tokens (markdowns/01-search-agent-plan.md §2.1):

  16 patch tokens   one per 2x2 board patch, fixed order (patch 0 = a1/b1/a2/b2,
                    row-major over 4x4 patch grid). Each patch value packs its
                    4 squares' 13-way classes: c0*13^3 + c1*13^2 + c2*13 + c3,
                    squares ordered (r,f), (r,f+1), (r+1,f), (r+1,f+1).
  1 castling token  4-bit mask, bit0=K bit1=Q bit2=k bit3=q (0 = no rights)
  1 stm token       0 = white to move, 1 = black
  1 ep token        0 = none, 1+sq = en-passant target square (explicit — the
                    oracle ignores ep, but the agent's language carries it)

Square classes: 0 = empty; 1..6 = white PAWN..KING (python-chess piece_type
order); 7..12 = black PAWN..KING.

Vocab regions (single flat id space, offsets computed below):
  PATCH 28,561 | MOVE 1,924 (oracle move sub-vocab, same index order) |
  QBIN 128 | DBIN 32 | CASTLE 16 | STM 2 | EP 65 | NUM 257 (budgets +
  stream indices) | CTRL 12
"""
from __future__ import annotations

import chess

from chessdecoder.models.vocab import move_vocab as _MOVE_VOCAB

# ---------------------------------------------------------------------------
# Region layout
# ---------------------------------------------------------------------------
N_PATCH = 13 ** 4                 # 28561
N_MOVE = len(_MOVE_VOCAB)         # 1924
N_QBIN = 128
N_DBIN = 32
N_CASTLE = 16
N_STM = 2
N_EP = 65
N_NUM = 257                       # 0..256

CTRL_TOKENS = ["<root>", "<oracle>", "<probe>", "<answer>", "<invalid>",
               "<line>", "<best>", "<recall>", "<pad>",
               "<legal>", "<played>", "<lastmove>", "<horizon>", "<target>",
               "<next>", "<swing>", "<bestw>", "<bestb>", "<reach>",
               "<opening>", "<fill>", "<mask>", "<spare0>", "<spare1>"]
N_CTRL = len(CTRL_TOKENS)

PATCH_BASE = 0
MOVE_BASE = PATCH_BASE + N_PATCH
QBIN_BASE = MOVE_BASE + N_MOVE
DBIN_BASE = QBIN_BASE + N_QBIN
CASTLE_BASE = DBIN_BASE + N_DBIN
STM_BASE = CASTLE_BASE + N_CASTLE
EP_BASE = STM_BASE + N_STM
NUM_BASE = EP_BASE + N_EP
CTRL_BASE = NUM_BASE + N_NUM
VOCAB_SIZE = CTRL_BASE + N_CTRL   # 30,997

CTRL = {t: CTRL_BASE + i for i, t in enumerate(CTRL_TOKENS)}
ROOT, ORACLE, PROBE, ANSWER = CTRL["<root>"], CTRL["<oracle>"], CTRL["<probe>"], CTRL["<answer>"]
INVALID, LINE, BEST, RECALL, PAD = (CTRL["<invalid>"], CTRL["<line>"],
                                    CTRL["<best>"], CTRL["<recall>"], CTRL["<pad>"])
LEGAL, PLAYED, LASTMOVE, HORIZON = (CTRL["<legal>"], CTRL["<played>"],
                                    CTRL["<lastmove>"], CTRL["<horizon>"])
TARGET, NEXT, SWING = CTRL["<target>"], CTRL["<next>"], CTRL["<swing>"]
BESTW, BESTB, REACH = CTRL["<bestw>"], CTRL["<bestb>"], CTRL["<reach>"]
OPENING, FILL, MASK = CTRL["<opening>"], CTRL["<fill>"], CTRL["<mask>"]


def uci_to_token(uci: str) -> int | None:
    """Corpus/python-chess uci -> MOVE-region id. Knight promotions follow
    the lc0 convention (bare move string, no 'n' suffix)."""
    t = MOVE_TO_ID.get(uci)
    if t is None and uci.endswith("n"):
        t = MOVE_TO_ID.get(uci[:-1])
    return t

BOARD_LEN = 19                    # 16 patches + castle + stm + ep
REPLY_LEN = 7                     # <oracle> q d m1 m2 m3 m4  (budget appended
                                  # by the harness at episode time, not in
                                  # pretrain streams)

# uci string <-> move-region token id (same index order as the oracle's
# 1924-way policy head, so oracle outputs map 1:1).
MOVE_TO_ID = {u: MOVE_BASE + i for i, u in enumerate(_MOVE_VOCAB)}
ID_TO_MOVE = {MOVE_BASE + i: u for i, u in enumerate(_MOVE_VOCAB)}

# ---------------------------------------------------------------------------
# Square classes
# ---------------------------------------------------------------------------

def _piece_class(piece: chess.Piece | None) -> int:
    if piece is None:
        return 0
    return piece.piece_type + (0 if piece.color == chess.WHITE else 6)


_CLASS_TO_PIECE = {0: None}
for pt in range(1, 7):
    _CLASS_TO_PIECE[pt] = chess.Piece(pt, chess.WHITE)
    _CLASS_TO_PIECE[pt + 6] = chess.Piece(pt, chess.BLACK)

# Per-patch square indices, fixed order. Patch p: rows (p//4)*2..+1,
# cols (p%4)*2..+1; squares (r,f),(r,f+1),(r+1,f),(r+1,f+1); sq = r*8+f.
PATCH_SQUARES: list[tuple[int, int, int, int]] = []
for p in range(16):
    r, f = (p // 4) * 2, (p % 4) * 2
    PATCH_SQUARES.append((r * 8 + f, r * 8 + f + 1,
                          (r + 1) * 8 + f, (r + 1) * 8 + f + 1))


def _castle_bits(board: chess.Board) -> int:
    bits = 0
    if board.castling_rights & chess.BB_H1: bits |= 1
    if board.castling_rights & chess.BB_A1: bits |= 2
    if board.castling_rights & chess.BB_H8: bits |= 4
    if board.castling_rights & chess.BB_A8: bits |= 8
    return bits


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------

def encode_board(board: chess.Board) -> list[int]:
    """chess.Board -> 19 token ids."""
    ids = []
    for sqs in PATCH_SQUARES:
        v = 0
        for sq in sqs:
            v = v * 13 + _piece_class(board.piece_at(sq))
        ids.append(PATCH_BASE + v)
    ids.append(CASTLE_BASE + _castle_bits(board))
    ids.append(STM_BASE + (0 if board.turn == chess.WHITE else 1))
    ep = board.ep_square
    ids.append(EP_BASE + (0 if ep is None else 1 + ep))
    return ids


def decode_board(ids: list[int]) -> chess.Board | None:
    """19 token ids -> chess.Board (no validity check), or None if any token
    falls outside its expected region."""
    if len(ids) != BOARD_LEN:
        return None
    board = chess.Board.empty()
    for p, tok in enumerate(ids[:16]):
        v = tok - PATCH_BASE
        if not (0 <= v < N_PATCH):
            return None
        sqs = PATCH_SQUARES[p]
        for i in (3, 2, 1, 0):              # peel base-13 digits, last first
            cls = v % 13
            v //= 13
            piece = _CLASS_TO_PIECE[cls]
            if piece is not None:
                board.set_piece_at(sqs[i], piece)
    cas = ids[16] - CASTLE_BASE
    stm = ids[17] - STM_BASE
    ep = ids[18] - EP_BASE
    if not (0 <= cas < N_CASTLE and 0 <= stm < N_STM and 0 <= ep < N_EP):
        return None
    rights = 0
    if cas & 1: rights |= chess.BB_H1
    if cas & 2: rights |= chess.BB_A1
    if cas & 4: rights |= chess.BB_H8
    if cas & 8: rights |= chess.BB_A8
    board.castling_rights = rights
    board.turn = chess.WHITE if stm == 0 else chess.BLACK
    board.ep_square = None if ep == 0 else ep - 1
    return board


def board_is_valid(board: chess.Board) -> bool:
    """Semantic validity the harness enforces on probes."""
    return board.is_valid()


# ---------------------------------------------------------------------------
# Value bins
# ---------------------------------------------------------------------------

def q_to_bin(q: float) -> int:
    """q = W - L in [-1, 1] -> 0..127."""
    b = int((q + 1.0) * 0.5 * N_QBIN)
    return min(max(b, 0), N_QBIN - 1)


def d_to_bin(d: float) -> int:
    b = int(d * N_DBIN)
    return min(max(b, 0), N_DBIN - 1)


def qbin_token(q: float) -> int:
    return QBIN_BASE + q_to_bin(q)


def dbin_token(d: float) -> int:
    return DBIN_BASE + d_to_bin(d)


def num_token(n: int) -> int:
    assert 0 <= n < N_NUM
    return NUM_BASE + n


# ---------------------------------------------------------------------------
# Moves (dual castling form, mirroring eval_vs_stockfish.pick_move)
# ---------------------------------------------------------------------------

def move_keys(board: chess.Board, mv: chess.Move) -> list[str]:
    """All vocab spellings of mv: python-chess uci, plus the lc0
    king-takes-rook form for castling."""
    keys = [mv.uci()]
    if board.is_castling(mv):
        frm = chess.square_name(mv.from_square)
        rook_file = "h" if board.is_kingside_castling(mv) else "a"
        keys.append(frm + rook_file + frm[1])
    return keys


def region_of(tok: int) -> str:
    """Debug helper: which vocab region a token id belongs to."""
    for name, base, n in (("patch", PATCH_BASE, N_PATCH), ("move", MOVE_BASE, N_MOVE),
                          ("qbin", QBIN_BASE, N_QBIN), ("dbin", DBIN_BASE, N_DBIN),
                          ("castle", CASTLE_BASE, N_CASTLE), ("stm", STM_BASE, N_STM),
                          ("ep", EP_BASE, N_EP), ("num", NUM_BASE, N_NUM),
                          ("ctrl", CTRL_BASE, N_CTRL)):
        if base <= tok < base + n:
            return name
    return "OOB"
