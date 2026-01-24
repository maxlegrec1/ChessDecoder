# Vocabulary and Tokenization System

## Overview

The vocabulary system (`src/models/vocab.py`) defines how chess positions and moves are converted to token indices for the transformer model.

**Total Vocabulary Size**: ~4,500 tokens

---

## Vocabulary Components

### 1. Policy Index (~1,900 tokens)

All possible chess moves in UCI notation. These are the first tokens in the vocabulary (indices 0 to ~1,899).

```python
policy_index = [
    # Standard moves from each square
    "a1b1", "a1c1", "a1d1", ...,  # Rook-like moves from a1
    "a1a2", "a1a3", ...,          # Vertical moves
    "b1a3", "b1c3", ...,          # Knight moves from b1
    ...

    # Pawn promotions
    "a2a1q", "a2a1r", "a2a1b", "a2a1n",  # Underpromotions
    "a7a8q", "a7a8r", "a7a8b", "a7a8n",  # Standard promotions
    ...
]
```

**Move Generation Logic** (from `vocab.py`):
- For each square, generate all possible destination squares
- Knight moves: L-shaped patterns
- Sliding pieces: Ray directions (horizontal, vertical, diagonal)
- Pawn promotions: All 4 piece types (Q, R, B, N)

**Castling Representation**:
The model uses rook destination squares internally:
| Standard UCI | Model's Internal |
|--------------|------------------|
| `e1g1` (O-O) | `e1h1` |
| `e1c1` (O-O-O) | `e1a1` |
| `e8g8` (O-O) | `e8h8` |
| `e8c8` (O-O-O) | `e8a8` |

This is converted back during inference in `model.py:168-175`.

---

### 2. Piece Tokens (12 tokens)

```python
piece_tokens = [
    "white_king", "white_queen", "white_rook",
    "white_bishop", "white_knight", "white_pawn",
    "black_king", "black_queen", "black_rook",
    "black_bishop", "black_knight", "black_pawn"
]
```

Used to represent pieces on the 64 board squares.

---

### 3. Special Tokens (8 tokens)

```python
special_tokens = [
    "start_pos",      # Marks beginning of board representation
    "end_pos",        # Marks end of 64-square section
    "white_to_move",  # Side to move indicator
    "black_to_move",  # Side to move indicator
    "empty",          # Empty square
    "pad",            # Padding token
    "bos",            # Beginning of sequence (unused currently)
    "eos"             # End of sequence (unused currently)
]
```

---

### 4. Castling Rights Tokens (16 tokens)

All combinations of castling availability:

```python
castling_tokens = [
    "K",      # White kingside only
    "Q",      # White queenside only
    "KQ",     # White both
    "k",      # Black kingside only
    "q",      # Black queenside only
    "Kk",     # White kingside + Black kingside
    "Kq",     # White kingside + Black queenside
    "Qk",     # etc...
    "Qq",
    "KQk",
    "KQq",
    "Kkq",
    "Qkq",
    "KQkq",   # All castling available (starting position)
    "kq",     # Black both
    "no_castling_rights"  # No castling available
]
```

---

## Key Mappings

```python
# Token string → vocabulary index
token_to_idx: Dict[str, int]
# Example: token_to_idx["e2e4"] = 847

# Vocabulary index → token string
idx_to_token: Dict[int, str]
# Example: idx_to_token[847] = "e2e4"

# UCI move → policy index (within policy tokens only)
policy_to_idx: Dict[str, int]
# Example: policy_to_idx["e2e4"] = 847

# Constants
vocab_size: int  # Total vocabulary size (~4,500)
POSITION_TOKEN_LENGTH = 68  # Fixed tokens per board position
```

---

## FEN to Token Conversion

The function `fen_to_position_tokens()` in `data.py` converts a FEN string to 68 tokens:

### Input FEN Example
```
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
```

### Conversion Process

```python
def fen_to_position_tokens(fen: str) -> List[str]:
    board = chess.Board(fen)
    tokens = []

    # 1. Start marker
    tokens.append("start_pos")

    # 2. 64 square tokens (a1, b1, c1, ..., h8 order)
    for square in chess.SQUARES:  # 0-63, a1=0, h8=63
        piece = board.piece_at(square)
        if piece is None:
            tokens.append("empty")
        else:
            color = "white" if piece.color else "black"
            piece_name = chess.piece_name(piece.piece_type)
            tokens.append(f"{color}_{piece_name}")

    # 3. End marker
    tokens.append("end_pos")

    # 4. Castling rights
    castling = board.castling_xfen()
    if castling == "-":
        tokens.append("no_castling_rights")
    else:
        tokens.append(castling)

    # 5. Side to move
    if board.turn == chess.WHITE:
        tokens.append("white_to_move")
    else:
        tokens.append("black_to_move")

    return tokens  # Always exactly 68 tokens
```

### Output Token Sequence (68 tokens)

```
Index  Token
─────  ─────────────────
0      start_pos
1      white_rook        (a1)
2      white_knight      (b1)
3      white_bishop      (c1)
4      white_queen       (d1)
5      white_king        (e1)
6      white_bishop      (f1)
7      white_knight      (g1)
8      white_rook        (h1)
9      white_pawn        (a2)
10     white_pawn        (b2)
11     white_pawn        (c2)
12     white_pawn        (d2)
13     empty             (e2) ← pawn moved to e4
14     white_pawn        (f2)
15     white_pawn        (g2)
16     white_pawn        (h2)
17     empty             (a3)
...
28     white_pawn        (e4) ← pawn is here now
...
48     black_pawn        (a7)
...
56     black_rook        (a8)
57     black_knight      (b8)
58     black_bishop      (c8)
59     black_queen       (d8)
60     black_king        (e8)
61     black_bishop      (f8)
62     black_knight      (g8)
63     black_rook        (h8)
64     end_pos
65     KQkq              (castling rights)
66     black_to_move     (side to move)
```

**Note**: Square ordering follows `chess.SQUARES` which is a1, b1, c1, ..., h1, a2, b2, ..., h8 (file-major order within each rank).

---

## Square Index Mapping

```
Chess square → Index in token sequence (after start_pos)

    a   b   c   d   e   f   g   h
  ┌───┬───┬───┬───┬───┬───┬───┬───┐
8 │57 │58 │59 │60 │61 │62 │63 │64 │  (indices 57-64 in tokens)
  ├───┼───┼───┼───┼───┼───┼───┼───┤
7 │49 │50 │51 │52 │53 │54 │55 │56 │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
6 │41 │42 │43 │44 │45 │46 │47 │48 │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
5 │33 │34 │35 │36 │37 │38 │39 │40 │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
4 │25 │26 │27 │28 │29 │30 │31 │32 │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
3 │17 │18 │19 │20 │21 │22 │23 │24 │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
2 │ 9 │10 │11 │12 │13 │14 │15 │16 │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
1 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │  (indices 1-8 in tokens)
  └───┴───┴───┴───┴───┴───┴───┴───┘

Token index = chess.square + 1  (offset by start_pos)
```

---

## Important Implementation Details

### Move Token Detection

In `model.py`, moves are identified by their token index:

```python
is_move = (x < self.num_policy_tokens)  # Policy tokens are first in vocab
```

This works because all move tokens have indices 0 to ~1,899, while board tokens (pieces, special tokens) have higher indices.

### Vocabulary Construction Order

```python
# Order in vocabulary:
# 1. Policy tokens (moves)     indices 0 to ~1,899
# 2. Piece tokens              indices ~1,900 to ~1,911
# 3. Special tokens            indices ~1,912 to ~1,919
# 4. Castling tokens           indices ~1,920 to ~1,935
```

This ordering is critical because `is_move = (x < num_policy_tokens)` relies on moves being the lowest indices.

---

## En Passant Note

The current implementation does **not** encode the en passant square in the token sequence. The FEN's en passant field is parsed by `chess.Board()` but not included as a separate token.

This means:
- The model cannot distinguish positions that differ only in en passant availability
- En passant moves are still valid (handled by `chess.Board.legal_moves`)
- Could be a future improvement to add an en passant token
