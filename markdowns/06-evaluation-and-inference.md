# Evaluation and Inference

## Overview

ChessDecoder supports several inference modes: single-position move prediction, move + value prediction, multi-position context prediction, and autoregressive thinking inference with variation generation.

---

## Single Position: `predict_move()`

Predicts the best move for a given FEN position.

```python
model = ChessDecoder(vocab_size=1968, embed_dim=1024, num_heads=16, num_layers=12, max_seq_len=256)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

move = model.predict_move(
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    temperature=0.0,       # 0.0 = greedy (argmax)
    force_legal=True       # Filter to legal moves
)
# Returns: "e2e4" (standard UCI)
```

### Process

1. Convert FEN to 68 tokens via `fen_to_position_tokens()`
2. Forward pass with `mask_type="prefix"` and single block (all 68 tokens bidirectional)
3. `policy_head` at position 67 (STM token) -> 1924 move sub-vocab logits
4. If `force_legal`: mask illegal moves to `-inf` using `move_token_to_idx` for O(1) lookup
5. Sample or argmax in move sub-vocab space
6. Map sub-vocab index -> full vocab index via `move_idx_to_full_idx`
7. Convert pseudo-castling back to standard UCI (e.g., `e1h1` -> `e1g1`)

### Legal Move Filtering

```python
board = chess.Board(fen)
for move in board.legal_moves:
    uci = move.uci()
    if board.is_castling(move):
        uci = castling_map[uci]  # Standard -> pseudo-castling
    if uci in move_token_to_idx:
        vocab_legal_moves.append(move_token_to_idx[uci])

legal_mask = torch.full_like(logits, float('-inf'))
legal_mask[vocab_legal_moves] = 0
logits = logits + legal_mask
```

---

## Move + Value: `predict_move_and_value()`

Predicts the best move and WDL evaluation for a position.

```python
move, wdl = model.predict_move_and_value(
    fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    temperature=0.0, force_legal=True
)
# move: "e7e5"
# wdl: {"win": 0.30, "draw": 0.42, "loss": 0.28}
```

### Process

1. First pass: predict move (same as `predict_move()`)
2. Extend sequence with: `[move_token] [wl_value] [d_value]`
3. Second pass (prefix): predict WL from hidden state at move token position
4. Inject WL Fourier features at `wl_value` position
5. Third pass (prefix): predict D from hidden state at `wl_value` position
6. Reconstruct: `W = (1 - D + WL) / 2`, `L = (1 - D - WL) / 2`

### Value Prediction Details

WL and D are predicted via bucket distribution:

```python
wl_logits = model.wl_head(h_at_move)           # [1, 100]
wl_idx = torch.argmax(wl_logits, dim=-1)        # Bucket index
wl_value = model.wl_bucket_centers[wl_idx]       # Scalar in [-1, 1]

# Then inject WL as Fourier features for D prediction
d_logits = model.d_head(h_at_wl)                # [1, 100]
d_value = model.d_bucket_centers[d_idx]           # Scalar in [0, 1]
```

---

## Multi-Position Context: `predict_move_n()`

Predicts the best move given an initial position and a history of moves played.

```python
move, wl_d_cache = model.predict_move_n(
    initial_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    history=[
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "e2e4"),
        ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "e7e5"),
    ],
    temperature=0.0,
    force_legal=True,
    cached_wl_d=None  # Or previous wl_d_cache for incremental inference
)
```

### Context Sequence

```
[board_0_68] [move_1] [wl] [d] [board_1_68] [move_2] [wl] [d] ... [board_N_68]
```

Total tokens: `68 + N * 71`. Maximum history length: `(max_seq_len - 68) // 71`.

### Autoregressive Value Prediction

WL/D values must be predicted sequentially because each D prediction depends on the WL Fourier features:

```
For each position i (uncached):
  Pass 1: Inject WL_1..WL_{i-1}, D_1..D_{i-1} -> predict WL_i from move_i position
  Pass 2: Inject WL_1..WL_i, D_1..D_{i-1}     -> predict D_i from wl_value_i position
```

This requires `2 * (N - num_cached) + 1` forward passes for N history positions.

### Caching

The `cached_wl_d` parameter accepts previously computed (WL, D) tuples, allowing incremental inference when moves are added one at a time (e.g., during game play).

---

## Thinking Inference: `scripts/think.py`

Autoregressive generation of thinking sequences with variation exploration.

```bash
python scripts/think.py --checkpoint checkpoints/model.pt \
    --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
    --temperature 0.0 --device cuda
```

### State Machine

The inference follows a state machine that mirrors the training sequence format:

```
States: MOVE -> WL_D -> BOARD -> AFTER_BOARD -> AFTER_END_VAR -> FINAL

1. Emit root board (deterministic from FEN)
2. Emit start_think token
3. State machine loop:
   MOVE:           thinking_policy_head -> sample move
   WL_D:           wl_head/d_head -> predict + emit wl_value, d_value tokens
   BOARD:          board_head (causal) -> generate 68 board tokens
   AFTER_BOARD:    board_head -> decide: continue_var (-> MOVE) or end_var (-> AFTER_END_VAR)
   AFTER_END_VAR:  board_head -> decide: new_variation (-> MOVE) or end_think (-> FINAL)
   FINAL:          policy_head -> sample final move + wl/d
```

### Sub-Vocabulary Logit Mapping

All heads output sub-vocabulary logits that must be mapped back to full vocab:

```python
# Board head: 41 logits -> board sub-vocab index -> full vocab index
board_sub_idx = sample_token(logits, temperature)
full_idx = board_idx_to_full_idx[board_sub_idx]
token = idx_to_token[full_idx]

# Move head: 1924 logits -> move sub-vocab index -> full vocab index
move_sub_idx = sample_token(logits, temperature)
full_idx = move_idx_to_full_idx[move_sub_idx]
token = idx_to_token[full_idx]
```

### Decision Points

At `AFTER_BOARD`, the board_head decides whether to continue the current variation or end it:
- If `board_sub_idx == board_token_to_idx["end_var"]` -> emit `end_var`, transition to AFTER_END_VAR
- Otherwise -> transition to MOVE (continue PV), don't emit `continue_var` token to context

At `AFTER_END_VAR`, the board_head decides whether to start a new variation or end thinking:
- If `board_sub_idx == board_token_to_idx["end_think"]` -> emit `end_think`, transition to FINAL
- Otherwise -> transition to MOVE (new variation), don't emit `new_variation` token to context

---

## Temperature Guide

| Temperature | Behavior |
|-------------|----------|
| 0.0 | Greedy (deterministic argmax) |
| 0.1 | Near-deterministic, slight variation |
| 0.5 | Moderate diversity |
| 1.0 | Sample proportional to model probabilities |
| > 1.0 | Increased randomness |

Recommended: `temperature=0.0` for evaluation, `temperature=0.1` for game play.

---

## Loading a Model for Inference

```python
import torch
from src.models.model import ChessDecoder
from src.models.vocab import vocab_size

# Load checkpoint
checkpoint = torch.load("checkpoints/model.pt", map_location="cuda", weights_only=False)
config = checkpoint["config"]

model = ChessDecoder(
    vocab_size=vocab_size,
    embed_dim=config["model"]["embed_dim"],
    num_heads=config["model"]["num_heads"],
    num_layers=config["model"]["num_layers"],
    max_seq_len=config["model"]["max_seq_len"],
    d_ff=config["model"].get("d_ff"),
    n_buckets=config["model"].get("n_buckets", 100),
    value_hidden_size=config["model"].get("value_hidden_size", 256),
    num_fourier_freq=config["model"].get("num_fourier_freq", 128),
    wl_sigma=config["model"].get("wl_sigma", 0.4),
).to("cuda")

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Use any inference method
move = model.predict_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
```
