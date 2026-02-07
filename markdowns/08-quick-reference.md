# Quick Reference

## File Locations

| Purpose | File |
|---------|------|
| Decoder model | `src/models/model.py` |
| Vocabulary + sub-vocabs | `src/models/vocab.py` |
| FEN tokenization | `src/dataloader/data.py` |
| Pretraining dataset | `src/dataloader/loader.py` |
| Finetuning dataset | `finetune/loader.py` |
| Variation sequence generation | `finetune/data.py` |
| Pretraining loop | `src/train/train.py` |
| Finetuning loop | `finetune/train.py` |
| Pretraining config | `src/train/config.yaml` |
| Finetuning config | `finetune/config.yaml` |
| Thinking inference | `scripts/think.py` |

---

## Key Constants

```python
# From vocab.py
POSITION_TOKEN_LENGTH = 68     # Tokens per board position
vocab_size = 1968              # Total vocabulary size
board_vocab_size = 41          # Board sub-vocabulary (board_head output)
move_vocab_size = 1924         # Move sub-vocabulary (policy_head output)
IGNORE_INDEX = -100            # PyTorch CE default for ignored targets

# From config.yaml (pretrain)
embed_dim = 1024
num_heads = 16
num_layers = 12
d_ff = 1536
max_seq_len = 256              # Pretrain
max_seq_len = 1024             # Finetune
n_buckets = 100                # WL/D bucket count
value_hidden_size = 256        # Value head hidden dim
num_fourier_freq = 128         # Fourier encoder frequencies
wl_sigma = 0.4                # Gaussian CDF quantile spread
```

---

## Token Sequence Structure

### Single Position (68 tokens)
```
[start_pos] [64 squares a1->h8] [end_pos] [castling] [side_to_move]
     0            1-64             65         66           67
```

### Pretraining Game Sequence
```
[BOARD_0: 68] [MOVE_0] [WL] [D] [BOARD_1: 68] [MOVE_1] [WL] [D] ...
```

Per move: 68 (board) + 1 (move) + 1 (wl) + 1 (d) = 71 tokens.

With `skip_board_prob=0.2`, some non-first boards are omitted:
```
[BOARD_0: 68] [MOVE_0] [WL] [D] [MOVE_1] [WL] [D] [BOARD_2: 68] ...
```

### Finetuning Thinking Sequence
```
[BOARD_0: 68] [start_think]
  [root_move_1] [wl] [d] [BOARD: 68] [pv_move_1] [wl] [d] [BOARD: 68] ... [end_var]
  [root_move_2] [wl] [d] [BOARD: 68] ... [end_var]
[end_think] [final_move] [wl] [d]
```

---

## Sub-Vocabularies

### Board Sub-Vocabulary (41 tokens) -- `board_head`

```python
board_vocab = (
    piece_tokens           # 12: white_king, ..., black_pawn
    + special_subset       #  7: start_pos, end_pos, white_to_move, black_to_move, empty, wl_value, d_value
    + castling_tokens      # 16: K, Q, k, q, KQ, ..., KQkq, no_castling_rights
    + signal_tokens        #  6: end_var, continue_var, new_variation, generic_move, end_think, start_think
)
```

### Move Sub-Vocabulary (1924 tokens) -- `policy_head` / `thinking_policy_head`

All UCI moves from `policy_index`.

### Target-Only Tokens

These tokens appear in `board_target_ids` but **never** in the input sequence:
- `generic_move` -- board_head target at STM positions (signals "a move comes next")
- `continue_var` -- board_head target at PV continuation positions
- `new_variation` -- board_head target at `end_var` when a new variation follows

---

## Important Mappings

```python
from src.models.vocab import (
    # Full vocabulary
    token_to_idx,              # "e2e4" -> 847
    idx_to_token,              # 847 -> "e2e4"
    vocab_size,                # 1968

    # Board sub-vocabulary
    board_token_to_idx,        # "white_king" -> 0
    board_idx_to_full_idx,     # [1924, 1925, ...] (board sub-idx -> full vocab idx)
    full_idx_to_board_idx,     # {1924: 0, 1925: 1, ...} (full vocab idx -> board sub-idx)
    board_vocab_size,          # 41

    # Move sub-vocabulary
    move_token_to_idx,         # "e2e4" -> 847 (same as policy_to_idx)
    move_idx_to_full_idx,      # [0, 1, 2, ...] (move sub-idx -> full vocab idx)
    full_idx_to_move_idx,      # {0: 0, 1: 1, ...} (full vocab idx -> move sub-idx)
    move_vocab_size,           # 1924

    # Policy-specific
    policy_to_idx,             # "e2e4" -> 847 (O(1) lookup)
    policy_index,              # List of all 1924 move tokens
)
```

---

## Training Commands

### Pretraining
```bash
cd /mnt/2tb_2/decoder
python -m src.train.train
```

### Finetuning
```bash
cd /mnt/2tb_2/decoder
python -m finetune.train
```

### Thinking Inference
```bash
cd /mnt/2tb_2/decoder
python scripts/think.py --checkpoint checkpoints/model.pt \
    --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
    --temperature 0.0 --device cuda
```

---

## Load Model for Inference

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

# Single move prediction
move = model.predict_move(
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    temperature=0.0,
    force_legal=True
)
# Returns: "e2e4" (standard UCI)

# Move + value prediction
move, wdl = model.predict_move_and_value(
    fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    temperature=0.0,
    force_legal=True
)
# move: "e7e5"
# wdl: {"win": 0.30, "draw": 0.42, "loss": 0.28}
```

---

## Castling Notation

| Standard UCI | Internal (Model) |
|--------------|------------------|
| e1g1 (O-O white) | e1h1 |
| e1c1 (O-O-O white) | e1a1 |
| e8g8 (O-O black) | e8h8 |
| e8c8 (O-O-O black) | e8a8 |

Conversion happens automatically in `predict_move()`, `predict_move_and_value()`, `predict_move_n()`.

---

## Loss Weights

### Pretraining (`src/train/config.yaml`)

| Loss | Weight | Head | Description |
|------|--------|------|-------------|
| Move | 5.0 | `policy_head` | CE over move sub-vocab at STM positions |
| Board | 1.0 | `board_head` | CE over board sub-vocab (unified, all board predictions) |
| WL | 1.0 | `wl_head` | Soft bucket CE for WL prediction (100 Gaussian buckets) |
| D | 1.0 | `d_head` | Soft bucket CE for D prediction (100 uniform buckets) |

### Finetuning (`finetune/config.yaml`)

| Loss | Weight | Head | Description |
|------|--------|------|-------------|
| Final Move | 5.0 | `policy_head` | CE over move sub-vocab at `end_think` |
| Thinking Move | 2.0 | `thinking_policy_head` | CE over move sub-vocab at thinking positions |
| Board | 1.0 | `board_head` | CE over board sub-vocab (unified) |
| WL | 1.0 | `wl_head` | Soft bucket CE for WL |
| D | 1.0 | `d_head` | Soft bucket CE for D |

---

## Mask Types

| Mask | Use Case | Attention Pattern |
|------|----------|-------------------|
| `causal` | Board generation (Pass 1) | Standard lower-triangular |
| `prefix` | Move/value prediction (Pass 2) | Causal + bidirectional within board blocks |

**Prefix mask formula**: `mask = causal_mask | same_block_mask`

---

## Target Tensors

| Tensor | Index Space | Size | Content |
|--------|------------|------|---------|
| `board_target_ids` | Board sub-vocab (0-40) | `[B, S]` | Shifted next-token mapped to board sub-vocab, with overrides at STM (`generic_move`), PV continuation (`continue_var`), new variation (`new_variation`) |
| `move_target_ids` | Move sub-vocab (0-1923) | `[B, S]` | Best move at STM positions (pretrain), thinking moves + final move (finetune) |
| Both use `IGNORE_INDEX = -100` for positions that don't contribute to loss |

---

## Batch Output Format

### Pretraining

```python
{
    "input_ids":        [B, S],  # Full vocab token IDs (padded with pad_id)
    "board_target_ids": [B, S],  # Board sub-vocab indices (-100 = ignore)
    "move_target_ids":  [B, S],  # Move sub-vocab indices (-100 = ignore)
    "move_mask":        [B, S],  # True at STM positions
    "wl_positions":     [B, S],  # True at wl_value positions
    "d_positions":      [B, S],  # True at d_value positions
    "wl_targets":       [B, S],  # WL values (float, at relevant positions)
    "d_targets":        [B, S],  # D values (float, at relevant positions)
    "wdl_valid":        [B, S],  # True if WDL data is valid
    "block_id":         [B, S],  # Block IDs for prefix masking
}
```

### Finetuning (additional fields)

```python
{
    ...,  # All pretraining fields
    "thinking_move_mask":  [B, S],  # True at thinking move prediction positions
    "first_is_not_best":   [B],     # True if first variation != final move
    "continue_var_mask":   [B, S],  # True at PV continuation positions (metrics only)
    "new_variation_mask":  [B, S],  # True at new variation positions (metrics only)
    "variation_epoch":     [B],     # Variation data recycling counter
}
```

---

## Data Format (Parquet)

### Pretraining

```
game_id     : str/int  - Unique game identifier
ply         : int      - Move number (0-indexed)
fen         : str      - Position in FEN notation
played_move : str      - Move played (UCI)
best_move   : str      - Stockfish recommendation (UCI)
win         : float    - Win probability
draw        : float    - Draw probability
loss        : float    - Loss probability
```

### Finetuning (Variations)

```
fen         : str      - Root position FEN
variations  : JSON str - List of MCTS variation dicts
mcts_action : str      - Best move from MCTS search
win         : float    - Root position win probability
draw        : float    - Root position draw probability
loss        : float    - Root position loss probability
```

---

## Square Index Mapping

```
chess.SQUARES order: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63

In token sequence (after start_pos):
Token index = chess.square + 1

   a  b  c  d  e  f  g  h
8 |57|58|59|60|61|62|63|64|
7 |49|50|51|52|53|54|55|56|
6 |41|42|43|44|45|46|47|48|
5 |33|34|35|36|37|38|39|40|
4 |25|26|27|28|29|30|31|32|
3 |17|18|19|20|21|22|23|24|
2 | 9|10|11|12|13|14|15|16|
1 | 1| 2| 3| 4| 5| 6| 7| 8|
```

---

## Common Debugging

### Check if token is a move
```python
from src.models.vocab import move_vocab_size
is_move = token_idx < move_vocab_size  # First 1924 tokens are moves
```

### Get board tokens from sequence
```python
starts = (input_ids == token_to_idx["start_pos"]).nonzero()
# Each board is 68 tokens from start_pos
```

### Verify FEN tokenization
```python
from src.dataloader.data import fen_to_position_tokens
tokens = fen_to_position_tokens(fen)
assert len(tokens) == 68
```

### Decode board_target_ids back to token names
```python
from src.models.vocab import board_idx_to_full_idx, idx_to_token

for i, board_idx in enumerate(board_target_ids):
    if board_idx == -100:
        continue
    full_idx = board_idx_to_full_idx[board_idx]
    token_name = idx_to_token[full_idx]
    print(f"Position {i}: target = {token_name}")
```

### Decode move_target_ids back to token names
```python
from src.models.vocab import move_idx_to_full_idx, idx_to_token

for i, move_idx in enumerate(move_target_ids):
    if move_idx == -100:
        continue
    full_idx = move_idx_to_full_idx[move_idx]
    token_name = idx_to_token[full_idx]
    print(f"Position {i}: target = {token_name}")
```

---

## Hyperparameters Summary

| Parameter | Pretrain | Finetune |
|-----------|----------|----------|
| Batch size | 64 | 16 |
| Learning rate | 1e-4 | 3e-5 |
| Weight decay | 0.1 | 0.1 |
| Gradient accumulation | 4 | 8 |
| Effective batch size | 256 | 128 |
| Max sequence length | 256 | 1024 |
| Skip board prob | 0.2 | 0.0 |
| Mixed precision | true | true |
| Warmup steps | None | 500 |
| Gradient clip max_norm | 10.0 | 10.0 |

---

## Model Dimensions

| Parameter | Value |
|-----------|-------|
| `vocab_size` | 1968 |
| `embed_dim` | 1024 |
| `num_heads` | 16 |
| `head_dim` | 64 |
| `num_layers` | 12 |
| `d_ff` | 1536 |
| `n_buckets` | 100 |
| `value_hidden_size` | 256 |
| `num_fourier_freq` | 128 |
| Total parameters | ~116M |
