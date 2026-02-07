# Known Issues and Potential Improvements

## Current Limitations

### 1. Missing En Passant Encoding

**Issue**: The en passant square from FEN is not encoded in the token sequence.

**Impact**:
- Model cannot distinguish positions that differ only in en passant availability
- May incorrectly evaluate positions where en passant is critical

**Current Behavior**:
```python
# In data.py - fen_to_position_tokens()
# The FEN's en passant field (e.g., "e3" in "... KQkq e3 0 1") is parsed
# by chess.Board() but NOT included as a token
```

**Fix**: Add an en passant token (65 possibilities: 16 squares for white, 16 for black, or none).

```python
# Suggested addition to token sequence:
# Position: start_pos + 64 squares + end_pos + castling + en_passant + side_to_move
# Total: 69 tokens instead of 68

en_passant_tokens = [
    "ep_none",
    "ep_a3", "ep_b3", ..., "ep_h3",  # White can capture
    "ep_a6", "ep_b6", ..., "ep_h6",  # Black can capture
]
```

---

### 2. No Halfmove/Fullmove Clock

**Issue**: The 50-move rule counter and move number are not encoded.

**Impact**:
- Model cannot distinguish positions approaching 50-move draw
- Cannot learn time-sensitive endgame strategies

**Severity**: Low for most games, but matters in long endgames.

---

### 3. No Learning Rate Scheduling (Pretraining)

**Issue**: Pretraining uses a fixed learning rate with no warmup or decay.

**Impact**: May converge slower or to worse optima than with proper scheduling.

**Note**: Finetuning already has linear warmup via `LambdaLR` (default 500 warmup steps).

---

### 4. No Validation Loop

**Issue**: No held-out data evaluation during training.

**Impact**:
- Cannot detect overfitting
- No early stopping possible
- Training curves may be misleading

**Recommendation**: Add validation every N steps:
```python
if step % val_every == 0:
    model.eval()
    val_loss = evaluate(model, val_dataloader)
    model.train()
    wandb.log({"val/loss": val_loss})
```

---

### 5. Single GPU Only

**Issue**: No distributed training support.

**Impact**: Training speed limited by single GPU.

**Recommendation**: Add DDP support:
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
model = DDP(model, device_ids=[local_rank])
```

---

## Implemented Improvements

### 1. Mixed Precision Training

**Status**: Implemented in both `src/train/train.py` and `finetune/train.py`

**Benefit**: ~2x speedup, ~50% memory reduction.

**Usage**: Set `use_amp: true` in config (enabled by default).

**Implementation Details**:
- Uses `torch.autocast` with float16 for forward passes
- Uses `torch.amp.GradScaler` for loss scaling
- Properly handles gradient clipping with `scaler.unscale_()`
- Compatible with gradient accumulation

### 2. Vectorized Prefix Mask via block_id

**Status**: Implemented

**Benefit**: Eliminates Python loop overhead in prefix mask construction.

**Measured Performance** (RTX 3090, batch=16, seq=1024):
| Masking Method | fp32 | fp16 |
|----------------|------|------|
| Legacy (Python loops) | 993 ms | 891 ms |
| Vectorized (block_id) | 330 ms | 133 ms |
| **Speedup** | **3.0x** | **6.7x** |

**How it works**:

1. **Dataloader** computes `block_id` tensor during data loading:
   - Each position gets a block ID indicating which board block it belongs to
   - Positions in the same block can attend to each other bidirectionally
   - Orphan positions (move tokens, structural tokens) get unique IDs

2. **Model** uses vectorized tensor operations:
   ```python
   same_block = block_id.unsqueeze(-1) == block_id.unsqueeze(-2)  # [B, S, S]
   mask = causal_mask.unsqueeze(0) | same_block
   ```

### 3. Sub-Vocabulary Heads

**Status**: Implemented

**Benefit**: Reduces head output dimensions dramatically, improving efficiency and training signal quality.

| Head | Old Output Size | New Output Size | Reduction |
|------|----------------|-----------------|-----------|
| `board_head` | 1968 (full vocab) | 41 (board sub-vocab) | 48x smaller |
| `policy_head` | 1968 (full vocab) | 1924 (move sub-vocab) | 1.02x smaller |
| `thinking_policy_head` | 1968 (full vocab) | 1924 (move sub-vocab) | 1.02x smaller |

**Key changes**:
- `board_head` outputs 41 logits over board sub-vocabulary only
- `policy_head` / `thinking_policy_head` output 1924 logits over move sub-vocabulary only
- Separate target tensors: `board_target_ids` (board sub-vocab indices) + `move_target_ids` (move sub-vocab indices)
- Both use `IGNORE_INDEX = -100` instead of pad token for targets
- Checkpoint migration extracts sub-vocab rows from old full-vocab heads

### 4. Unified Board Loss

**Status**: Implemented

**Benefit**: Eliminates gradient imbalance between continuation and termination decisions.

**Before**: 4 separate board-related losses (`board_loss`, `start_pos_loss`, `continue_var_loss`, `new_variation_loss`), creating a ~104x gradient imbalance favoring end_var over continue_var.

**After**: Single `board_loss` over the board sub-vocabulary. All decisions (continue_var, end_var, new_variation, end_think, generic_move) are equally weighted within the unified loss.

**Simplified board mask**:
```python
# Before (complex exclusion logic):
board_mask = valid & ~move_mask & ~thinking_move_mask & ~wl_positions & ~d_positions & ~pre_first_move

# After (simple):
board_mask = (board_target_ids != IGNORE_INDEX) & (~pre_first_move_mask)
```

### 5. Soft Bucket Value Losses

**Status**: Implemented

**Benefit**: Better calibrated value predictions via distribution over buckets instead of single-point regression.

- WL: 100 Gaussian CDF quantile buckets in [-1, 1] (concentrated near 0 via `wl_sigma=0.4`)
- D: 100 uniform buckets in [0, 1]
- Soft cross-entropy: target probability distributed across two nearest bucket centers via linear interpolation

### 6. Fourier Value Injection

**Status**: Implemented

**Benefit**: Scalar WL/D values encoded as rich `embed_dim`-sized vectors via learned Fourier features.

- `FourierEncoder` with 128 learned frequencies
- Fourier embeddings **replace** token embeddings at `wl_value`/`d_value` positions
- During training: ground-truth values discretized to nearest bucket center, then Fourier-encoded
- During inference: predicted values injected sequentially (WL first, then D)

---

## Potential Improvements

### High Priority

#### 1. Gradient Checkpointing

**Benefit**: Trade compute for memory, enabling larger batch sizes.

```python
from torch.utils.checkpoint import checkpoint

for layer in self.layers:
    h = checkpoint(layer, h, mask, input_pos, use_reentrant=False)
```

---

### Medium Priority

#### 2. Data Augmentation

**Idea**: Mirror boards horizontally (flip a-h to h-a).

**Benefit**: 2x effective data, helps model learn symmetry.

```python
def mirror_position(tokens):
    # Swap files: a<->h, b<->g, c<->f, d<->e
    # Mirror move: e2e4 -> d2d4 (after adjusting)
```

#### 3. Curriculum Learning

**Idea**: Start with simpler positions, gradually increase complexity.

```python
# Stages:
# 1. Endgames (fewer pieces)
# 2. Middlegames
# 3. Full games from opening
```

#### 4. Contrastive Learning for Value Head

**Idea**: Learn relative position values, not just absolute WDL.

```python
# Given positions A and B where A is better:
# loss = max(0, value(B) - value(A) + margin)
```

---

### Lower Priority

#### 5. Move History Encoding

**Idea**: Add explicit move history tokens for better context.

**Current**: Board tokens implicitly encode history.
**Improved**: Add last N moves as explicit tokens.

#### 6. Multi-Task Learning with Puzzle Data

**Idea**: Train on tactical puzzles alongside games.

**Benefit**: Better tactical awareness, especially for rare positions.

#### 7. MCTS Integration

**Idea**: Use model as policy/value network in Monte Carlo Tree Search.

**Benefit**: Stronger play at inference time (like AlphaZero).

---

## Performance Optimization Checklist

| Optimization | Status | Measured Speedup |
|--------------|--------|------------------|
| Mixed precision (FP16) | Implemented | ~2.5x |
| Vectorized prefix masking | Implemented | 3x (fp32), 6.7x (fp16) |
| Sub-vocabulary heads | Implemented | Reduced head parameters |
| Unified board loss | Implemented | Simpler, balanced gradients |
| Soft bucket value losses | Implemented | Better calibration |
| Fourier value injection | Implemented | Rich scalar encoding |
| Gradient checkpointing | Not implemented | Memory: 2-4x expected |
| Multi-GPU (DDP) | Not implemented | Linear with GPUs |
| Optimized data loading | IterableDataset | Baseline |

---

## Recommended Next Steps

### Immediate (Quick Wins)

1. Add validation loop to detect overfitting
2. Implement learning rate scheduling for pretraining

### Short-term

3. Add en passant encoding
4. Implement gradient checkpointing for larger batches
5. Add unit tests for data pipeline

### Medium-term

6. Multi-GPU training support
7. Implement MCTS for stronger play

### Long-term (Research)

8. Curriculum learning experiments
9. Self-play training (RL fine-tuning)
10. Distillation from stronger models
