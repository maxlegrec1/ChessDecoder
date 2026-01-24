# Decoder Model Training

## Overview

The decoder model uses a sophisticated dual-pass training strategy that combines causal and bidirectional attention to learn both board generation and move prediction.

---

## The Dual-Pass Strategy

### Why Two Passes?

The model needs to learn two different tasks:

1. **Board Generation**: Predict next board tokens (needs causal masking to prevent "seeing the future")
2. **Move Prediction**: Predict the best move (benefits from seeing the FULL board context)

Using a single mask would compromise one of these objectives.

### Pass 1: Causal Masking

```python
policy_logits_causal, _ = model(input_ids, mask_type="causal")
```

**Attention Pattern**:
```
Token:   B1  B2  B3  ...  B68  M1  B1'  B2'  ...
         ↓   ↓   ↓        ↓    ↓   ↓    ↓
B1  →    ✓   ✗   ✗   ✗    ✗    ✗   ✗    ✗
B2  →    ✓   ✓   ✗   ✗    ✗    ✗   ✗    ✗
B3  →    ✓   ✓   ✓   ✗    ✗    ✗   ✗    ✗
...
B68 →    ✓   ✓   ✓   ✓    ✗    ✗   ✗    ✗
M1  →    ✓   ✓   ✓   ✓    ✓    ✗   ✗    ✗
B1' →    ✓   ✓   ✓   ✓    ✓    ✓   ✗    ✗

✓ = can attend, ✗ = cannot attend
```

Each token can only see tokens that came before it. This is the standard autoregressive pattern.

### Pass 2: Prefix Masking

```python
policy_logits_prefix, value_logits_prefix = model(input_ids, mask_type="prefix")
```

**Attention Pattern**:
```
Token:   B1  B2  B3  ...  B68  M1  B1'  B2'  ...
         ↓   ↓   ↓        ↓    ↓   ↓    ↓
B1  →    ✓   ✓   ✓   ✓    ✗    ✗   ✗    ✗    ← Bidirectional within board
B2  →    ✓   ✓   ✓   ✓    ✗    ✗   ✗    ✗
B3  →    ✓   ✓   ✓   ✓    ✗    ✗   ✗    ✗
...
B68 →    ✓   ✓   ✓   ✓    ✗    ✗   ✗    ✗
M1  →    ✓   ✓   ✓   ✓    ✓    ✗   ✗    ✗    ← Move sees full board
B1' →    ✓   ✓   ✓   ✓    ✓    ✓   ✓    ✗    ← Next board starts new block

         └─── Board 1 ───┘     └── Board 2 ──
```

Within each board block (between `start_pos` and the next move), tokens can attend to each other bidirectionally. This allows the model to use full board context when predicting moves.

### Mask Construction Code

From `model.py:81-101`:

```python
if mask_type == "prefix":
    # Start with causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len))

    # Find start_pos and move tokens
    is_start = (x == self.start_pos_id)
    is_move = (x < self.num_policy_tokens)

    # For each board block, enable bidirectional attention
    for b in range(bsz):
        starts = is_start[b].nonzero().flatten()
        moves = is_move[b].nonzero().flatten()

        for s in starts:
            # Find the next move after this start_pos
            moves_after = moves[moves > s]
            if len(moves_after) > 0:
                m_idx = moves_after[0].item()
                # Enable bidirectional attention from start to move
                mask[b, s:m_idx, s:m_idx] = True
            else:
                # No move found, enable to end
                mask[b, s:, s:] = True
```

---

## Loss Functions

### Three Loss Components

```python
total_loss = (
    move_weight * move_loss +      # 5.0 default
    board_weight * board_loss +    # 1.0 default
    wdl_weight * wdl_loss          # 1.0 default
)
```

### 1. Move Loss (Primary Objective)

**Purpose**: Learn to predict the best move given a position.

```python
# Only computed at move positions, using PREFIX logits
move_mask = wdl_mask & mask  # wdl_mask marks move positions
move_loss = (ce_loss_prefix * move_mask.float()).sum() / (move_mask.sum() + 1e-8)
```

**Details**:
- Uses **Cross-Entropy** loss
- Target is `best_move` (from Stockfish), not `played_move`
- Uses prefix logits (full board context)
- Weight: 5.0 (highest priority)

### 2. Board Loss (Auxiliary Task)

**Purpose**: Learn to generate coherent board representations.

```python
# Exclude move positions and pre-first-move tokens
board_mask = (~wdl_mask) & mask & (~pre_first_move_mask)
board_loss = (ce_loss_causal * board_mask.float()).sum() / (board_mask.sum() + 1e-8)
```

**Details**:
- Uses **Cross-Entropy** loss
- Target is next board token
- Uses causal logits (prevents cheating)
- Excludes padding and tokens before first move
- Weight: 1.0

### 3. WDL Loss (Value Learning)

**Purpose**: Learn to evaluate position quality.

```python
# MSE between predicted and target win/draw/loss probabilities
wdl_loss_raw = mse_loss_fn(value_logits_prefix, wdl_targets)
wdl_mask_expanded = wdl_mask.unsqueeze(-1).expand_as(wdl_loss_raw)
wdl_loss = (wdl_loss_raw * wdl_mask_expanded).sum() / (wdl_mask_expanded.sum() + 1e-8)
```

**Details**:
- Uses **MSE** loss
- Target is `[win, draw, loss]` probabilities from Stockfish
- Only computed at move positions
- Uses prefix logits (full board context)
- Weight: 1.0

---

## Masking Details

### Mask Types in Training

```python
# All positions mask (excludes padding)
mask = target_ids != token_to_idx["pad"]

# Move positions (where wdl_data exists)
wdl_mask = batch["wdl_mask"]  # True at positions just before moves

# Pre-first-move mask (to exclude initial context from board loss)
first_move_idx = wdl_mask.int().argmax(dim=1)
pre_first_move_mask = indices < first_move_idx.unsqueeze(1)

# Board positions = not move, not padding, not pre-first-move
board_mask = (~wdl_mask) & mask & (~pre_first_move_mask)
```

### Why Exclude Pre-First-Move Tokens?

When the sequence starts mid-game (due to random slicing), the first board tokens have no preceding move to predict. Including them in board loss would confuse the model.

```
Sliced sequence: [BOARD_3][MOVE_3][BOARD_4][MOVE_4]...
                 └──excluded──┘    └─included in board loss─┘
```

---

## Training Loop Flow

```
For each batch:
│
├── 1. Load batch to device
│   input_ids, target_ids, wdl_targets, wdl_mask = batch
│
├── 2. Forward Pass 1: Causal
│   policy_logits_causal, _ = model(input_ids, mask_type="causal")
│
├── 3. Forward Pass 2: Prefix
│   policy_logits_prefix, value_logits_prefix = model(input_ids, mask_type="prefix")
│
├── 4. Compute Losses
│   ├── move_loss  = CE(policy_prefix[move_positions], best_moves)
│   ├── board_loss = CE(policy_causal[board_positions], next_tokens)
│   └── wdl_loss   = MSE(value_prefix[move_positions], wdl_targets)
│
├── 5. Combine Losses
│   total_loss = 5.0 * move_loss + 1.0 * board_loss + 1.0 * wdl_loss
│
├── 6. Backward Pass
│   loss = total_loss / gradient_accumulation_steps
│   loss.backward()
│
├── 7. Optimizer Step (every N accumulation steps)
│   clip_grad_norm_(model.parameters(), max_norm=10.0)
│   optimizer.step()
│   optimizer.zero_grad()
│
└── 8. Logging
    wandb.log({losses, accuracies, step})
```

---

## Configuration

### Full Config (`config.yaml`)

```yaml
project_name: "chess-decoder"
run_name: "decoder-v1"

data:
  parquet_dir: "parquets/"
  batch_size: 16
  max_seq_len: 256
  num_workers: 4
  skip_board_prob: 0.2

model:
  embed_dim: 768
  num_heads: 12
  num_layers: 12
  max_seq_len: 256

training:
  device: "cuda"
  num_epochs: 10
  learning_rate: 5.0e-5
  weight_decay: 0.1
  gradient_accumulation_steps: 4
  log_every_n_steps: 100

loss:
  move_weight: 5.0
  board_weight: 1.0
  wdl_weight: 1.0
```

### Effective Batch Size

```
Effective batch = batch_size × gradient_accumulation_steps
                = 16 × 4
                = 64 samples per optimizer step
```

---

## Metrics

### Tracked During Training

| Metric | Description |
|--------|-------------|
| `train/total_loss` | Combined weighted loss |
| `train/move_loss` | CE loss for move prediction |
| `train/board_loss` | CE loss for board generation |
| `train/wdl_loss` | MSE loss for position evaluation |
| `train/policy_acc` | Overall accuracy (moves + boards) |
| `train/move_acc` | Accuracy at move positions only |
| `train/board_acc` | Accuracy at board positions only |

### Accuracy Computation

```python
# Move accuracy (using prefix logits)
preds_prefix = torch.argmax(policy_logits_prefix, dim=-1)
move_correct = (preds_prefix == target_ids) & wdl_mask
move_acc = move_correct.sum() / wdl_mask.sum()

# Board accuracy (using causal logits)
preds_causal = torch.argmax(policy_logits_causal, dim=-1)
board_correct = (preds_causal == target_ids) & board_mask
board_acc = board_correct.sum() / board_mask.sum()
```

---

## Checkpointing

```python
# Save every epoch
torch.save(model.state_dict(), f"checkpoints/checkpoint_epoch_{epoch+1}.pt")
```

### Loading a Checkpoint

```python
model = ChessDecoder(vocab_size=vocab_size, ...)
model.load_state_dict(torch.load("checkpoints/checkpoint_epoch_5.pt"))
model.eval()
```

---

## Potential Improvements

### 1. Learning Rate Scheduling

Current: Fixed learning rate
Suggested: Cosine annealing or linear warmup + decay

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs * steps_per_epoch
)
```

### 2. Mixed Precision Training

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    policy_logits, value_logits = model(input_ids)
    loss = compute_loss(...)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Validation Loop

Add periodic evaluation on held-out data to detect overfitting.

### 4. Early Stopping

Stop training when validation loss stops improving.

### 5. Distributed Training

For multi-GPU setups, use `DistributedDataParallel`:

```python
model = torch.nn.parallel.DistributedDataParallel(model)
```
