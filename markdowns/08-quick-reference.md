# Quick Reference

## File Locations

| Purpose | File |
|---------|------|
| Decoder model | `src/models/model.py` |
| Encoder model | `src/models/encoder.py` |
| Vocabulary | `src/models/vocab.py` |
| FEN tokenization | `src/dataloader/data.py` |
| Decoder dataset | `src/dataloader/loader.py` |
| Encoder dataset | `src/dataloader/encoder_loader.py` |
| Decoder training | `src/train/train.py` |
| Encoder training | `src/train/train_encoder.py` |
| Decoder config | `src/train/config.yaml` |
| Encoder config | `src/train/config_encoder.yaml` |
| Evaluation | `src/eval/play_move.py` |

---

## Key Constants

```python
# From vocab.py
POSITION_TOKEN_LENGTH = 68  # Tokens per board position
vocab_size = ~4,500         # Total vocabulary size
num_policy_tokens = ~1,900  # Number of move tokens

# From config.yaml
embed_dim = 768
num_heads = 12
num_layers = 12
max_seq_len = 256  # Decoder
max_seq_len = 68   # Encoder
```

---

## Token Sequence Structure

### Single Position (68 tokens)
```
[start_pos] [64 squares a1→h8] [end_pos] [castling] [side_to_move]
     0            1-64            65         66           67
```

### Game Sequence (Decoder)
```
[BOARD_0: 68 tokens] [MOVE_0] [BOARD_1: 68 tokens] [MOVE_1] ...
```

---

## Important Mappings

```python
from src.models.vocab import (
    token_to_idx,    # "e2e4" → 847
    idx_to_token,    # 847 → "e2e4"
    policy_to_idx,   # UCI move → policy index
    policy_index,    # List of all move tokens
    vocab_size,      # Total vocabulary size
)
```

---

## Training Commands

### Decoder Training
```bash
cd /mnt/2tb_2/decoder
python -m src.train.train
```

### Encoder Training
```bash
cd /mnt/2tb_2/decoder
python -m src.train.train_encoder
```

---

## Load Model for Inference

```python
import torch
from src.models.model import ChessDecoder
from src.models.vocab import vocab_size

# Load model
model = ChessDecoder(
    vocab_size=vocab_size,
    embed_dim=768,
    num_heads=12,
    num_layers=12,
    max_seq_len=256
)
model.load_state_dict(torch.load("checkpoints/checkpoint_epoch_10.pt"))
model.to("cuda")
model.eval()

# Predict move
move = model.predict_move(
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    temperature=0.1,
    force_legal=True
)
print(move)  # e.g., "e2e4"
```

---

## Castling Notation

| Standard UCI | Internal (Model) |
|--------------|------------------|
| e1g1 | e1h1 |
| e1c1 | e1a1 |
| e8g8 | e8h8 |
| e8c8 | e8a8 |

Conversion happens automatically in `predict_move()`.

---

## Loss Weights (Default)

| Loss | Weight | Description |
|------|--------|-------------|
| Move | 5.0 | CE at move positions (prefix logits) |
| Board | 1.0 | CE at board positions (causal logits) |
| WDL | 1.0 | MSE for win/draw/loss (prefix logits) |

---

## Mask Types

| Mask | Use Case | Attention Pattern |
|------|----------|-------------------|
| `causal` | Board generation | Each token sees only past |
| `prefix` | Move prediction | Bidirectional within board blocks |

---

## Data Format (Parquet)

Required columns:
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

---

## Square Index Mapping

```
chess.SQUARES order: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63

In token sequence (after start_pos):
Token index = chess.square + 1

   a  b  c  d  e  f  g  h
8 │57│58│59│60│61│62│63│64│
7 │49│50│51│52│53│54│55│56│
6 │41│42│43│44│45│46│47│48│
5 │33│34│35│36│37│38│39│40│
4 │25│26│27│28│29│30│31│32│
3 │17│18│19│20│21│22│23│24│
2 │ 9│10│11│12│13│14│15│16│
1 │ 1│ 2│ 3│ 4│ 5│ 6│ 7│ 8│
```

---

## Common Debugging

### Check if token is a move
```python
is_move = token_idx < num_policy_tokens
```

### Get board tokens from sequence
```python
# Find start_pos indices
starts = (input_ids == token_to_idx["start_pos"]).nonzero()
# Each board is 68 tokens from start_pos
```

### Verify FEN tokenization
```python
from src.dataloader.data import fen_to_position_tokens
tokens = fen_to_position_tokens(fen)
assert len(tokens) == 68
print(tokens)
```

---

## Hyperparameters Summary

| Parameter | Decoder | Encoder |
|-----------|---------|---------|
| Batch size | 16 | 256 |
| Learning rate | 5e-5 | 1e-4 |
| Weight decay | 0.1 | 0.1 |
| Gradient accumulation | 4 | 1 |
| Max sequence length | 256 | 68 |
| Skip board prob | 0.2 | N/A |
