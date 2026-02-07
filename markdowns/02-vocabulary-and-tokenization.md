# Vocabulary and Tokenization System

## Overview

The vocabulary system (`src/models/vocab.py`) defines how chess positions and moves are converted to token indices for the transformer model. It also defines **sub-vocabularies** that allow each prediction head to output logits over only its relevant token subset.

**Total Vocabulary Size**: 1968 tokens
**Board Sub-Vocabulary**: 41 tokens (for `board_head`)
**Move Sub-Vocabulary**: 1924 tokens (for `policy_head` / `thinking_policy_head`)

---

## Full Vocabulary Components

### 1. Policy Index (1924 tokens) -- indices 0-1923

All possible chess moves in UCI notation. These are the first tokens in the vocabulary.

```python
policy_index = [
    "a1b1", "a1c1", ...,           # Standard moves from each square
    "a7a8q", "a7a8r", "a7a8b",     # Promotions (queen, rook, bishop -- no knight suffix, default)
    "a2a1q", "a2a1r", "a2a1b",     # Underpromotions
    ...
]
```

**Move generation**: For each of the 64 squares, all reachable destination squares are enumerated (rook-like, bishop-like, knight-like moves). Pawn promotions include queen, rook, and bishop suffixes (knight promotion is the base move without suffix).

**Castling representation**: The model uses king-captures-rook notation internally:

| Standard UCI | Model's Internal |
|--------------|------------------|
| `e1g1` (O-O white) | `e1h1` |
| `e1c1` (O-O-O white) | `e1a1` |
| `e8g8` (O-O black) | `e8h8` |
| `e8c8` (O-O-O black) | `e8a8` |

Conversion happens automatically in `predict_move()` and related methods.

### 2. Piece Tokens (12 tokens) -- indices 1924-1935

```python
piece_tokens = [
    "white_king", "white_queen", "white_rook",
    "white_bishop", "white_knight", "white_pawn",
    "black_king", "black_queen", "black_rook",
    "black_bishop", "black_knight", "black_pawn"
]
```

Used to represent pieces on the 64 board squares.

### 3. Special Tokens (12 tokens) -- indices 1936-1947

```python
special_tokens = [
    "start_pos",       # Marks beginning of board representation
    "end_pos",         # Marks end of 64-square section
    "white_to_move",   # Side to move indicator
    "black_to_move",   # Side to move indicator
    "empty",           # Empty square
    "pad",             # Padding token
    "bos",             # Beginning of sequence (unused currently)
    "eos",             # End of sequence (unused currently)
    "wl_value",        # WL value placeholder (Fourier features injected here)
    "d_value",         # D value placeholder (Fourier features injected here)
    "start_think",     # Marks beginning of thinking block (finetuning)
    "end_think"        # Marks end of thinking block (finetuning)
]
```

### 4. Castling Rights Tokens (16 tokens) -- indices 1948-1963

All combinations of castling availability, generated from `itertools.combinations("KQkq", r)` for `r` in 1..4, plus `"no_castling_rights"`:

```python
castling_tokens = [
    "K", "Q", "k", "q",                           # Single rights
    "KQ", "Kk", "Kq", "Qk", "Qq", "kq",          # Pairs
    "KQk", "KQq", "Kkq", "Qkq",                   # Triples
    "KQkq",                                         # All rights
    "no_castling_rights"                             # None
]
```

### 5. Signal Tokens (4 tokens) -- indices 1964-1967

```python
["end_var", "continue_var", "new_variation", "generic_move"]
```

| Token | Purpose |
|-------|---------|
| `end_var` | Marks end of a variation line |
| `continue_var` | Board head target at PV continuation positions (not in input sequence) |
| `new_variation` | Board head target when starting a new variation after `end_var` (not in input sequence) |
| `generic_move` | Board head target at STM positions signaling "a move comes next" (not in input sequence) |

Note: `continue_var`, `new_variation`, and `generic_move` are **target-only tokens** -- they appear in `board_target_ids` but never in the input sequence. `end_var` appears in both input and target sequences.

---

## Sub-Vocabularies

### Board Sub-Vocabulary (41 tokens)

All tokens that `board_head` can predict. These are the tokens that can appear as targets in `board_target_ids`:

```python
board_vocab = (
    piece_tokens                                              # 12 pieces
    + ["start_pos", "end_pos", "white_to_move",
       "black_to_move", "empty", "wl_value", "d_value"]      # 7 special
    + castling_tokens                                          # 16 castling
    + ["end_var", "continue_var", "new_variation",
       "generic_move", "end_think", "start_think"]             # 6 signal
)
board_vocab_size = 41
```

**Why these tokens?** The board head predicts the natural shifted next-token for the sequence. Looking at every position in the sequence:
- After a board square token, the next token is another piece/empty/end_pos -> **board tokens**
- After end_pos, the next token is a castling rights token -> **castling tokens**
- After castling, the next token is side_to_move -> **stm tokens**
- After STM, the next token is a move (overridden to `generic_move`) -> **generic_move**
- After a move, the next token is `wl_value` -> **wl_value**
- After `wl_value`, the next token is `d_value` -> **d_value**
- After `d_value`, the next token is `start_pos` (of next board) -> **start_pos**
- After end of PV board STM, the target is `continue_var` (override) or `end_var` (natural) -> **continue_var, end_var**
- After `end_var`, the target is `new_variation` (override) or `end_think` (natural) -> **new_variation, end_think**
- After `start_think`, the target is `generic_move` (override) -> **generic_move**

### Move Sub-Vocabulary (1924 tokens)

All UCI policy moves. These are the tokens that can appear as targets in `move_target_ids`:

```python
move_vocab = policy_index  # Same as the policy_index list
move_vocab_size = 1924
```

---

## Mapping Dictionaries

```python
# Full vocabulary mappings
token_to_idx: Dict[str, int]      # "e2e4" -> 847
idx_to_token: Dict[int, str]      # 847 -> "e2e4"
vocab_size: int                    # 1968

# Board sub-vocabulary mappings
board_token_to_idx: Dict[str, int]      # "white_king" -> 0
board_idx_to_full_idx: List[int]        # [1924, 1925, ...] (board sub-idx -> full vocab idx)
full_idx_to_board_idx: Dict[int, int]   # {1924: 0, 1925: 1, ...} (full vocab idx -> board sub-idx)

# Move sub-vocabulary mappings
move_token_to_idx: Dict[str, int]       # "e2e4" -> 847 (same as policy_to_idx)
move_idx_to_full_idx: List[int]         # [0, 1, 2, ...] (move sub-idx -> full vocab idx)
full_idx_to_move_idx: Dict[int, int]    # {0: 0, 1: 1, ...} (full vocab idx -> move sub-idx)

# Policy-specific lookup
policy_to_idx: Dict[str, int]           # "e2e4" -> 847 (O(1) lookup)
```

---

## FEN to Token Conversion

The function `fen_to_position_tokens()` in `src/dataloader/data.py` converts a FEN string to exactly 68 tokens:

### Process

```python
def fen_to_position_tokens(fen: str) -> List[str]:
    board = chess.Board(fen)
    tokens = ["start_pos"]

    # 64 squares in chess.SQUARES order (a1, b1, c1, ..., h8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = "white" if piece.color == chess.WHITE else "black"
            tokens.append(f"{color}_{chess.piece_name(piece.piece_type)}")
        else:
            tokens.append("empty")

    tokens.append("end_pos")

    # Castling rights
    rights = ""
    if board.has_kingside_castling_rights(chess.WHITE): rights += "K"
    if board.has_queenside_castling_rights(chess.WHITE): rights += "Q"
    if board.has_kingside_castling_rights(chess.BLACK): rights += "k"
    if board.has_queenside_castling_rights(chess.BLACK): rights += "q"
    tokens.append(rights if rights else "no_castling_rights")

    # Side to move
    tokens.append("white_to_move" if board.turn == chess.WHITE else "black_to_move")

    return tokens  # Always exactly 68 tokens
```

### Output Structure (68 tokens)

```
Index  Token             Description
-----  ----------------  -----------
0      start_pos         Board start marker
1      <piece/empty>     Square a1
2      <piece/empty>     Square b1
...
64     <piece/empty>     Square h8
65     end_pos           Board end marker
66     <castling>        Castling rights (e.g., "KQkq")
67     <stm>             Side to move ("white_to_move" or "black_to_move")
```

### Square Index Mapping

```
chess.SQUARES order: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
In token sequence: token_index = chess.square + 1 (offset by start_pos)

    a   b   c   d   e   f   g   h
  +---+---+---+---+---+---+---+---+
8 |57 |58 |59 |60 |61 |62 |63 |64 |
  +---+---+---+---+---+---+---+---+
7 |49 |50 |51 |52 |53 |54 |55 |56 |
  +---+---+---+---+---+---+---+---+
6 |41 |42 |43 |44 |45 |46 |47 |48 |
  +---+---+---+---+---+---+---+---+
5 |33 |34 |35 |36 |37 |38 |39 |40 |
  +---+---+---+---+---+---+---+---+
4 |25 |26 |27 |28 |29 |30 |31 |32 |
  +---+---+---+---+---+---+---+---+
3 |17 |18 |19 |20 |21 |22 |23 |24 |
  +---+---+---+---+---+---+---+---+
2 | 9 |10 |11 |12 |13 |14 |15 |16 |
  +---+---+---+---+---+---+---+---+
1 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
  +---+---+---+---+---+---+---+---+
```

---

## IGNORE_INDEX Convention

Both `board_target_ids` and `move_target_ids` use `IGNORE_INDEX = -100` (PyTorch's `CrossEntropyLoss` default) for padding and positions that should not contribute to loss. This replaces the old approach of using `pad_id` as the target for positions to ignore.

---

## En Passant Note

The current implementation does **not** encode the en passant square in the token sequence. The FEN's en passant field is parsed by `chess.Board()` but not included as a separate token. This means positions that differ only in en passant availability are indistinguishable to the model.
