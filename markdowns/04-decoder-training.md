# Pretraining

## Overview

The pretraining loop (`src/train/train.py`) uses the two-pass architecture to train the model on normal game sequences. Each step performs a causal pass (for board generation) and a prefix pass (for move and value prediction).

---

## Two-Pass Training Strategy

### Fourier Input Preparation (shared by both passes)

Ground-truth WL/D values are discretized to nearest bucket centers and used as Fourier feature inputs. These are computed once and injected into **both** passes, so the model sees real evaluation context in both causal and prefix attention.

```python
# Discretize ground-truth values to bucket centers
wl_fourier_input[wl_positions] = discretize_to_bucket(wl_targets[wl_positions])
d_fourier_input[d_positions] = discretize_to_bucket(d_targets[d_positions])
```

### Pass 1: Causal Masking (Board Generation)

```python
h_causal = model(input_ids, mask_type="causal",
                 wl_values=wl_fourier_input, d_values=d_fourier_input,
                 wl_positions=wl_positions, d_positions=d_positions)
board_logits = model.board_head(h_causal)  # [B, S, 41]
```

Standard autoregressive attention with Fourier-encoded WL/D values at placeholder positions. The `board_head` outputs 41 logits (board sub-vocabulary) at every position. Loss is computed only at positions within `board_mask`.

### Pass 2: Prefix Masking (Move + Value Prediction)

```python
h_prefix = model(input_ids, mask_type="prefix", block_id=block_id,
                 wl_values=wl_fourier_input, d_values=d_fourier_input,
                 wl_positions=wl_positions, d_positions=d_positions)
move_logits = model.policy_head(h_prefix)      # [B, S, 1924]
wl_logits = model.wl_head(h_at_move)           # [N, 100]
d_logits = model.d_head(h_at_wl)               # [M, 100]
```

Bidirectional within board blocks. Same Fourier-encoded WL/D values are injected at placeholder positions.

---

## Loss Functions

### 1. Board Loss (Unified Autoregressive)

```python
IGNORE_INDEX = -100
board_ce_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='none')

board_mask = (board_target_ids != IGNORE_INDEX) & (~pre_first_move_mask)

ce_board = board_ce_fn(board_logits.view(-1, board_vocab_size), board_target_ids.view(-1))
ce_board = ce_board.view(board_target_ids.shape)
board_loss = (ce_board * board_mask.float()).sum() / (board_mask.sum() + 1e-8)
```

**Board mask construction**:
1. Start from all non-IGNORE positions: `board_target_ids != -100`
2. Exclude positions before the first move (no useful causal context from the initial board):
   ```python
   first_move_idx = move_mask.int().argmax(dim=1)
   pre_first_move_mask = indices < first_move_idx.unsqueeze(1)
   board_mask = board_mask & (~pre_first_move_mask)
   ```

This single board_loss covers **all** board head predictions: board tokens, structural tokens (`wl_value`, `d_value`, `start_pos`), and signal tokens (`generic_move`, `continue_var`, `end_var`, `new_variation`, `end_think`). No separate loss terms needed.

**Weight**: 1.0

### 2. Move Loss

```python
move_ce_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='none')

ce_move = move_ce_fn(move_logits.view(-1, move_vocab_size), move_target_ids.view(-1))
ce_move = ce_move.view(move_target_ids.shape)
move_loss = (ce_move * move_mask.float()).sum() / (move_mask.sum() + 1e-8)
```

Cross-entropy over the move sub-vocabulary (1924 tokens) at STM positions. Target is `best_move` from Stockfish.

**Weight**: 5.0

### 3. WL Loss (Soft Bucket Cross-Entropy)

```python
wl_loss = soft_bucket_loss(wl_logits, wl_gt_flat, model.wl_bucket_centers, wl_valid_flat)
```

WL values are predicted as distributions over 100 buckets in [-1, 1]. The bucket centers are concentrated near 0 via Gaussian CDF quantiles (controlled by `wl_sigma`). The target probability mass is distributed across the two nearest bucket centers via linear interpolation.

```python
def soft_bucket_loss(logits, target_values, bucket_centers, valid_mask):
    # Find two nearest buckets for each target
    lower_idx = (diffs >= 0).long().sum(dim=-1) - 1
    upper_idx = lower_idx + 1
    # Linearly interpolate probability between them
    upper_weight = (target_values - lower_centers) / span
    soft_labels = scatter(lower_weight) + scatter(upper_weight)
    loss = -(soft_labels * F.log_softmax(logits, dim=-1)).sum(dim=-1)
```

WL prediction happens at the **move token position** (STM + 1), where the model has seen the move but can use the preceding board context.

**Weight**: 1.0

### 4. D Loss (Soft Bucket Cross-Entropy)

Same soft bucket loss but with 100 uniform buckets in [0, 1]. D prediction happens at the **wl_value placeholder position** (STM + 2) during the prefix pass, where the Fourier-encoded WL value has been injected.

**Weight**: 1.0

### Total Loss

```python
total_loss = (
    move_weight * move_loss +      # 5.0
    board_weight * board_loss +    # 1.0
    wl_weight * wl_loss +          # 1.0
    d_weight * d_loss              # 1.0
)
```

---

## Fourier Value Injection (Training)

During training, ground-truth WL/D values are used (teacher forcing). Values are first discretized to nearest bucket centers, then encoded via the `FourierEncoder`:

```python
if wl_positions.any():
    wl_vals_at_pos = wl_targets[wl_positions]
    wl_disc = model.discretize_to_bucket(wl_vals_at_pos, model.wl_bucket_centers)
    wl_fourier_input[wl_positions] = wl_disc

# In model.forward():
if wl_positions is not None and wl_positions.any():
    h[wl_positions] = self.fourier_encoder(wl_values[wl_positions]).to(h.dtype)
```

The Fourier encoder produces `embed_dim`-sized vectors from scalar values, which replace the token embeddings at `wl_value`/`d_value` positions.

---

## Metrics

### Per-Step Metrics Logged to WandB

| Metric | Description |
|--------|-------------|
| `train/total_loss` | Combined weighted loss |
| `train/move_loss` | CE loss for move prediction (move sub-vocab) |
| `train/board_loss` | CE loss for board generation (board sub-vocab) |
| `train/wl_loss` | Soft bucket CE for WL prediction |
| `train/d_loss` | Soft bucket CE for D prediction |
| `train/move_acc` | Accuracy at move positions (move sub-vocab argmax) |
| `train/board_acc` | Per-token accuracy at board positions (board sub-vocab argmax) |
| `train/board_total_acc` | Per-block accuracy (all tokens in block correct) |
| `train/board_square_acc` | Per-block accuracy on 64 square tokens only |
| `train/board_castling_acc` | Accuracy on castling token prediction |
| `train/board_stm_acc` | Accuracy on side-to-move prediction |
| `train/wl_mae` | WL mean absolute error (expected value) |
| `train/wl_mse` | WL mean squared error (argmax bucket center) |
| `train/d_mae` | D mean absolute error |
| `train/d_mse` | D mean squared error |
| `train/move_acc_nth/{n}` | Move accuracy for the nth move in a sequence (0-indexed, up to 20) |

---

## Checkpoint Migration

When loading old checkpoints with full-vocab heads (1967 tokens), the `migrate_state_dict()` function extracts sub-vocab rows:

```python
# board_head: extract 41 rows from old full-vocab weights
for i, full_idx in enumerate(board_idx_to_full_idx):
    if full_idx < old_vocab_sz:
        new_w[i] = old_w[full_idx]

# policy_head / thinking_policy_head: extract 1924 rows
for i, full_idx in enumerate(move_idx_to_full_idx):
    if full_idx < old_vocab_sz:
        new_w[i] = old_w[full_idx]

# tok_embedding: zero-pad if vocabulary expanded
if t.shape[0] < vocab_size:
    state_dict["tok_embedding.weight"] = torch.cat([t, zeros], dim=0)

# Clone policy_head -> thinking_policy_head if missing
if "thinking_policy_head.weight" not in state_dict:
    state_dict["thinking_policy_head.weight"] = state_dict["policy_head.weight"].clone()
```

---

## Configuration

### `src/train/config.yaml`

```yaml
project_name: "chess-decoder"
run_name: "run-1"

data:
  parquet_dir: "/home/maxime/parquet_files_decoder/"
  batch_size: 64
  num_workers: 8
  max_seq_len: 256
  skip_board_prob: 0.2

model:
  embed_dim: 1024
  num_heads: 16
  num_layers: 12
  max_seq_len: 256
  d_ff: 1536
  n_buckets: 100
  wl_sigma: 0.4
  value_hidden_size: 256
  num_fourier_freq: 128

training:
  learning_rate: 1.0e-4
  weight_decay: 0.1
  gradient_accumulation_steps: 4
  num_epochs: 100
  log_every_n_steps: 4
  save_every_n_steps: 1000
  device: "cuda"
  use_amp: true
  checkpoint_dir: "checkpoints/"
  max_track_nth_moves: 20

loss:
  move_weight: 5.0
  board_weight: 1.0
  wl_weight: 1.0
  d_weight: 1.0
```

**Effective batch size**: `batch_size * gradient_accumulation_steps = 64 * 4 = 256`

---

## Training Loop Summary

```
For each batch:
  1. Load batch to device (input_ids, board_target_ids, move_target_ids, masks, block_id)
  2. Pass 1 (causal): h_causal -> board_head -> board_logits [B, S, 41]
  3. Compute board_mask, board_loss
  4. Prepare Fourier inputs (discretize ground-truth WL/D to bucket centers)
  5. Pass 2 (prefix): h_prefix -> policy_head, wl_head, d_head
  6. Compute move_loss, wl_loss, d_loss
  7. total_loss = 5.0*move + 1.0*board + 1.0*wl + 1.0*d
  8. Backward + gradient accumulation
  9. Clip gradients (max_norm=10.0), optimizer step
  10. Log metrics to WandB
```
