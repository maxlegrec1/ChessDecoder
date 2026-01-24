# Data Pipeline

## Overview

The data pipeline transforms Parquet files containing Stockfish-analyzed chess games into training batches for the transformer models.

---

## Data Source Format

### Parquet Schema

Training data is stored in Parquet files with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | string/int | Unique identifier for each game |
| `ply` | int | Move number within the game (0-indexed) |
| `fen` | string | Chess position in FEN notation |
| `played_move` | string | Move actually played (UCI notation) |
| `best_move` | string | Stockfish's recommended move (UCI notation) |
| `win` | float | Win probability (0.0 - 1.0) |
| `draw` | float | Draw probability (0.0 - 1.0) |
| `loss` | float | Loss probability (0.0 - 1.0) |

### Example Row

```
game_id: "game_12345"
ply: 5
fen: "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
played_move: "d7d6"
best_move: "b8c6"
win: 0.28
draw: 0.45
loss: 0.27
```

---

## Decoder Data Pipeline

### File: `src/dataloader/loader.py`

The `ChessIterableDataset` class implements a streaming dataset for efficient memory usage.

### Sequence Construction

#### Step 1: Game to Token Sequence

Function: `game_to_token_ids()` in `data.py`

```python
def game_to_token_ids(game_df, skip_board_prob=0.0):
    """
    Converts a game dataframe to a token sequence.

    Returns:
        ids: List[int] - Token indices
        wdl_data: List[Tuple] - (move_idx, best_move, [w,d,l], is_valid)
    """
```

**Process**:

```
For each position in game (sorted by ply):
    1. If random() > skip_board_prob:
           Append 68 board tokens (from FEN)
    2. Append played_move token
    3. Record (move_index, best_move, [win, draw, loss], is_valid)
```

**Example Output**:

```
Game: 1. e4 e5 2. Nf3 Nc6

Token Sequence (no skipping):
┌──────────────────────────────────────────────────────────────────────┐
│ [BOARD_0: 68 tokens] [e2e4] [BOARD_1: 68 tokens] [e7e5] ...         │
│  ↑ starting position  ↑ move  ↑ after e4          ↑ move            │
└──────────────────────────────────────────────────────────────────────┘

Indices:
0-67:   Board tokens (starting position)
68:     Move token (e2e4)
69-136: Board tokens (after e4)
137:    Move token (e7e5)
...

wdl_data:
[
    (68, "e2e4", [0.32, 0.41, 0.27], True),   # Move at index 68
    (137, "d7d5", [0.30, 0.42, 0.28], True),  # Move at index 137, best_move differs!
    ...
]
```

#### Step 2: Random Slicing

The dataloader randomly selects a starting point within the game:

```python
# Valid start positions: game start OR right after any move
valid_starts = [0] + [wdl[0] + 1 for wdl in wdl_data[:-1]]

# Randomly choose where to start
start_idx = random.choice(valid_starts)
ids = ids[start_idx:]
```

**Why**: Ensures the model learns to play from any position, not just openings.

**Example**:
```
Full sequence: [BOARD_0][MOVE_0][BOARD_1][MOVE_1][BOARD_2][MOVE_2]...
Valid starts:     ↑         ↑              ↑              ↑
                  0       69 (after e2e4)  138           207

If start_idx=69 chosen:
    Sliced: [BOARD_1][MOVE_1][BOARD_2][MOVE_2]...
```

#### Step 3: Padding/Truncation

```python
max_seq_len = 256  # From config

if len(ids) > max_seq_len:
    ids = ids[:max_seq_len]  # Truncate
else:
    ids = ids + [pad_idx] * (max_seq_len - len(ids))  # Pad
```

#### Step 4: Target Construction

```python
# Default: next-token prediction (shifted by 1)
target_ids = [0] * max_seq_len
target_ids[:len(ids)-1] = ids[1:len(ids)]

# Override move positions with BEST move (not played move)
for move_idx, best_move, wdl, is_valid in wdl_data:
    adjusted_idx = move_idx - start_idx
    if 0 <= adjusted_idx < max_seq_len:
        target_idx = adjusted_idx - 1  # Predict move from position BEFORE it
        target_ids[target_idx] = token_to_idx[best_move]
        wdl_targets[target_idx] = wdl
        wdl_mask[target_idx] = True
```

**Critical Insight**: The target is `best_move` from Stockfish, NOT `played_move`. This teaches optimal play rather than imitating potentially suboptimal human moves.

### Batch Output Format

```python
{
    "input_ids": torch.LongTensor,    # Shape: [batch_size, max_seq_len]
    "target_ids": torch.LongTensor,   # Shape: [batch_size, max_seq_len]
    "wdl_targets": torch.FloatTensor, # Shape: [batch_size, max_seq_len, 3]
    "wdl_mask": torch.BoolTensor      # Shape: [batch_size, max_seq_len]
}
```

### Multi-Worker Support

```python
def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        # Distribute files across workers
        files = self.parquet_files[worker_info.id::worker_info.num_workers]
    else:
        files = self.parquet_files
```

Each worker processes a disjoint subset of files.

---

## Encoder Data Pipeline

### File: `src/dataloader/encoder_loader.py`

The `ChessEncoderDataset` is simpler - one position per sample.

### Sample Construction

```python
def __getitem__(self, idx):
    row = self.data.iloc[idx]

    # Convert FEN to 68 tokens
    tokens = fen_to_position_tokens(row["fen"])
    input_ids = [token_to_idx[t] for t in tokens]

    # Target is the best move's policy index
    best_move = row["best_move"]
    target = policy_to_idx[best_move]

    # Attention mask (all 1s, no padding needed)
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": torch.LongTensor(input_ids),
        "attention_mask": torch.LongTensor(attention_mask),
        "target": torch.LongTensor([target])
    }
```

### Filtering Invalid Moves

```python
# Skip positions where best_move isn't in vocabulary
valid_mask = self.data["best_move"].isin(policy_to_idx.keys())
self.data = self.data[valid_mask]
```

---

## Skip Board Probability

### Config Parameter: `skip_board_prob`

With probability `skip_board_prob`, board tokens are omitted:

```python
if random.random() > skip_board_prob:
    ids.extend(board_tokens)  # Include board
# else: skip board tokens entirely
ids.append(move_token)  # Always include move
```

### Effect on Sequences

```
skip_board_prob = 0.0 (default):
[BOARD_0][MOVE_0][BOARD_1][MOVE_1][BOARD_2][MOVE_2]

skip_board_prob = 0.2:
[BOARD_0][MOVE_0][MOVE_1][BOARD_2][MOVE_2]  # BOARD_1 skipped (20% chance)
            ↑        ↑
            Consecutive moves, no board between
```

### Purpose

Forces the model to:
1. Infer board state from move history
2. Not rely entirely on explicit board tokens
3. Improve generalization

### Current Setting

```yaml
# config.yaml
data:
  skip_board_prob: 0.2  # Skip 20% of boards
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PARQUET FILES                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │
│  │ games_1.pq  │  │ games_2.pq  │  │ games_3.pq  │  ...                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                      │
└─────────┼────────────────┼────────────────┼─────────────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      ChessIterableDataset                                │
│                                                                          │
│  1. Load parquet file                                                    │
│  2. Group rows by game_id                                                │
│  3. Shuffle games within file                                            │
│  4. For each game:                                                       │
│     ├── game_to_token_ids() → token sequence + wdl_data                 │
│     ├── Random slice (start from random position)                        │
│     ├── Pad/truncate to max_seq_len                                     │
│     └── Construct targets (best_move, not played_move)                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         DataLoader                                       │
│                                                                          │
│  Collate batches:                                                        │
│  {                                                                       │
│      "input_ids":    [B, 256]     # Token indices                       │
│      "target_ids":   [B, 256]     # Next-token targets                  │
│      "wdl_targets":  [B, 256, 3]  # Win/Draw/Loss probabilities         │
│      "wdl_mask":     [B, 256]     # True at move positions              │
│  }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Training Loop                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Decoder (`config.yaml`)

```yaml
data:
  parquet_dir: "parquets/"
  batch_size: 16
  max_seq_len: 256
  num_workers: 4
  skip_board_prob: 0.2
```

### Encoder (`config_encoder.yaml`)

```yaml
data:
  parquet_dir: "parquets/"
  batch_size: 256
  max_seq_len: 68  # Fixed for single position
  num_workers: 4
```

---

## Memory Considerations

### IterableDataset Advantages

- **Streaming**: Files loaded one at a time
- **No full dataset in memory**: Only current batch + buffer
- **Shuffling**: Within-file shuffling (games), not global

### Limitations

- **No random access**: Cannot index specific positions
- **Epoch boundaries**: Approximate (file-based)
- **Shuffling quality**: Less random than fully-loaded dataset

### Recommendations for Large Datasets

1. Split data into many small-medium parquet files
2. Use multiple workers (`num_workers=4+`)
3. Shuffle file order between epochs
4. Consider gradient accumulation for larger effective batch size
