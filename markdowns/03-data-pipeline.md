# Data Pipeline

## Overview

The data pipeline transforms Parquet files containing Stockfish-analyzed chess games into training batches. There are two pipelines: one for pretraining (normal game sequences) and one for finetuning (mixed normal + thinking variation sequences).

---

## Data Source Format

### Pretraining Parquet Schema

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | string/int | Unique identifier for each game |
| `ply` | int | Move number within the game (0-indexed) |
| `fen` | string | Chess position in FEN notation |
| `played_move` | string | Move actually played (UCI notation) |
| `best_move` | string | Stockfish's recommended move (UCI notation) |
| `win` | float | Win probability (0.0-1.0) from Stockfish |
| `draw` | float | Draw probability (0.0-1.0) |
| `loss` | float | Loss probability (0.0-1.0) |

### Variation Parquet Schema (Finetuning)

| Column | Type | Description |
|--------|------|-------------|
| `fen` | string | Root position FEN |
| `variations` | JSON string | List of MCTS variation dicts |
| `mcts_action` | string | Best move from MCTS search |
| `win`, `draw`, `loss` | float | Root position WDL values |

Each variation dict contains: `root_move`, `visit_count`, `visit_fraction`, `prior`, and `nodes` (list of `{fen, move, wdl, visit_count}` dicts representing the PV line).

---

## Pretraining Data Pipeline

### File: `src/dataloader/loader.py` -- `ChessIterableDataset`

### Step 1: Game to Token Sequence

Function `game_to_token_ids()` in `src/dataloader/data.py`:

```
For each position in game (sorted by ply):
    1. If random() > skip_board_prob:
           Append 68 board tokens (from FEN)
           Record block_boundary = (block_start, block_end)
    2. Append played_move token
    3. Append wl_value placeholder token
    4. Append d_value placeholder token
    5. Record wdl_data: (move_idx, best_move, [w,d,l], is_valid)
    6. Record value_data: (wl_pos, d_pos, wl=win-loss, d=draw, is_valid)
```

**Returns**: `ids`, `wdl_data`, `block_boundaries`, `value_data`

### Step 2: Random Slicing

```python
# Valid start positions: game start OR right after any d_value token
valid_starts = [0] + [vd[1] + 1 for vd in value_data[:-1]]
start_idx = random.choice(valid_starts)
ids = ids[start_idx:]
```

This ensures the model learns to play from any position. All associated metadata (wdl_data, value_data, block_boundaries) is adjusted to the new start index.

### Step 3: Truncation

```python
if len(ids) > max_seq_len:
    ids = ids[:max_seq_len]
    # Trim wdl_data, value_data, block_boundaries to fit
    # Ensure positions are fully included (wl_pos and d_pos both within range)
```

### Step 4: Target Construction

Two separate target tensors are built:

**`board_target_ids`** (board sub-vocab indices):
```python
IGNORE_INDEX = -100
board_target_ids = torch.full((max_seq_len,), IGNORE_INDEX, dtype=torch.long)

# Shifted next-token prediction, mapped to board sub-vocab
for i in range(seq_len - 1):
    full_idx = input_ids[i + 1].item()
    board_target_ids[i] = full_idx_to_board_idx.get(full_idx, IGNORE_INDEX)

# Override STM positions with generic_move
for move_idx, best_move, wdl, is_valid in wdl_data:
    stm_pos = move_idx - 1
    board_target_ids[stm_pos] = board_token_to_idx["generic_move"]
```

Move tokens naturally map to `IGNORE_INDEX` via `full_idx_to_board_idx.get()` since move tokens are not in the board sub-vocab. These are exactly the STM positions that get overridden with `generic_move`.

**`move_target_ids`** (move sub-vocab indices):
```python
move_target_ids = torch.full((max_seq_len,), IGNORE_INDEX, dtype=torch.long)

for move_idx, best_move, wdl, is_valid in wdl_data:
    stm_pos = move_idx - 1
    move_target_ids[stm_pos] = full_idx_to_move_idx[token_to_idx[best_move]]
    move_mask[stm_pos] = True
```

**Value targets**:
```python
for wl_pos, d_pos, wl, d, is_valid in value_data:
    wl_positions[wl_pos] = True; wl_targets[wl_pos] = wl
    d_positions[d_pos] = True;   d_targets[d_pos] = d
    wdl_valid[wl_pos] = is_valid; wdl_valid[d_pos] = is_valid
    # Also store WL/D at the STM position for convenience
    stm_pos = wl_pos - 2
    wl_targets[stm_pos] = wl; d_targets[stm_pos] = d
    wdl_valid[stm_pos] = is_valid
```

### Step 5: Block ID Construction

```python
max_block_num = len(block_boundaries)
block_id = torch.arange(max_seq_len, dtype=torch.long) + max_block_num  # Orphan IDs
for block_num, (b_start, b_end) in enumerate(block_boundaries):
    block_id[b_start:b_end] = block_num  # Board blocks share an ID
```

### Batch Output Format

```python
{
    "input_ids":        torch.LongTensor,   # [B, max_seq_len]
    "board_target_ids": torch.LongTensor,   # [B, max_seq_len] (board sub-vocab indices, -100 = ignore)
    "move_target_ids":  torch.LongTensor,   # [B, max_seq_len] (move sub-vocab indices, -100 = ignore)
    "move_mask":        torch.BoolTensor,   # [B, max_seq_len] (True at STM positions)
    "wl_positions":     torch.BoolTensor,   # [B, max_seq_len] (True at wl_value positions)
    "d_positions":      torch.BoolTensor,   # [B, max_seq_len] (True at d_value positions)
    "wl_targets":       torch.FloatTensor,  # [B, max_seq_len] (WL values at relevant positions)
    "d_targets":        torch.FloatTensor,  # [B, max_seq_len] (D values at relevant positions)
    "wdl_valid":        torch.BoolTensor,   # [B, max_seq_len] (True if WDL data is valid)
    "block_id":         torch.LongTensor,   # [B, max_seq_len] (block IDs for prefix masking)
}
```

---

## Finetuning Data Pipeline

### File: `finetune/loader.py` -- `FinetuneIterableDataset`

The finetuning dataloader mixes normal pretraining data with thinking variation data using a configurable `variation_ratio` (default 0.2 = 20% thinking, 80% normal).

### Mixing Strategy

```python
while True:
    if random.random() < variation_ratio:
        sample = next(variation_iter)   # Thinking variation sample
    else:
        sample = next(pretrain_iter)    # Normal pretraining sample
    yield sample
```

The variation iterator restarts when exhausted (variation data is typically smaller than pretraining data). A `variation_epoch` counter tracks how many times the variation data has been recycled.

### Pretrain Tensors (`_build_pretrain_tensors`)

Identical to the pretraining loader, with additional fields set to zero/False for compatibility: `thinking_move_mask`, `continue_var_mask`, `new_variation_mask`, `first_is_not_best`.

### Variation Tensors (`_build_variation_tensors`)

Uses `variation_to_token_ids()` from `finetune/data.py` (see doc 09) to produce the token sequence. Then builds:

**`board_target_ids`** -- same shifted-next-token approach, plus overrides:

```python
for predict_from_pos, move_token in thinking_move_data:
    tok_id = ids[predict_from_pos]
    if tok_id == start_think_id:
        board_target_ids[predict_from_pos] = board_token_to_idx["generic_move"]
    elif tok_id == end_var_id:
        board_target_ids[predict_from_pos] = board_token_to_idx["new_variation"]
    else:  # Board STM position (PV continuation)
        board_target_ids[predict_from_pos] = board_token_to_idx["continue_var"]
```

**`move_target_ids`** -- thinking moves via `thinking_move_mask`, final move via `move_mask`:

```python
# Thinking moves (predicted by thinking_policy_head)
for predict_from_pos, move_token in thinking_move_data:
    move_target_ids[predict_from_pos] = full_idx_to_move_idx[token_to_idx[move_token]]
    thinking_move_mask[predict_from_pos] = True

# Final move (predicted by policy_head)
move_target_ids[end_think_pos] = full_idx_to_move_idx[token_to_idx[final_move_token]]
move_mask[end_think_pos] = True
board_target_ids[end_think_pos] = board_token_to_idx["generic_move"]  # Override
```

### Finetuning Batch Output Format

Same as pretraining, plus:

```python
{
    ...,  # All pretraining fields
    "thinking_move_mask":  torch.BoolTensor,  # [B, max_seq_len] (True at thinking move positions)
    "first_is_not_best":   torch.BoolTensor,  # [B] (True if first variation != final move)
    "continue_var_mask":   torch.BoolTensor,  # [B, max_seq_len] (metrics only)
    "new_variation_mask":  torch.BoolTensor,  # [B, max_seq_len] (metrics only)
    "variation_epoch":     torch.LongTensor,  # [B] (variation data recycling counter)
}
```

---

## Skip Board Probability

With probability `skip_board_prob`, non-first board tokens are omitted:

```
skip_board_prob = 0.0: [BOARD_0][MOVE_0][WL][D][BOARD_1][MOVE_1][WL][D]...
skip_board_prob = 0.2: [BOARD_0][MOVE_0][WL][D][MOVE_1][WL][D][BOARD_2]...  (BOARD_1 skipped)
```

Forces the model to infer board state from move history when boards are missing. Current default: `skip_board_prob = 0.2`.

---

## Multi-Worker Support

Both `ChessIterableDataset` and `FinetuneIterableDataset` implement file-level sharding across DataLoader workers:

```python
worker_info = torch.utils.data.get_worker_info()
if worker_info is not None:
    per_worker = ceil(len(files) / num_workers)
    files_to_read = files[worker_id * per_worker : (worker_id + 1) * per_worker]
```

Each worker processes a disjoint subset of parquet files.

---

## Data Flow Diagram

```
PARQUET FILES
  |
  v
+----------------------------------+
| ChessIterableDataset             |
|                                  |
|  1. Load parquet file            |
|  2. Group rows by game_id        |
|  3. Shuffle games within file    |
|  4. For each game:               |
|     - game_to_token_ids()        |
|     - Random slice               |
|     - Truncate to max_seq_len    |
|     - Build target tensors:      |
|       board_target_ids (shifted) |
|       move_target_ids (best_move)|
|       wl/d targets               |
|     - Build block_id tensor      |
+----------------------------------+
  |
  v
+----------------------------------+
| DataLoader (collates batches)    |
|                                  |
| Batch:                           |
|   input_ids:        [B, S]       |
|   board_target_ids: [B, S]       |
|   move_target_ids:  [B, S]       |
|   move_mask:        [B, S]       |
|   wl/d positions:   [B, S]       |
|   wl/d targets:     [B, S]       |
|   block_id:         [B, S]       |
+----------------------------------+
  |
  v
Training Loop (src/train/train.py)
```
