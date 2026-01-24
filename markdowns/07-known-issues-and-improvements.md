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

### 3. No Learning Rate Scheduling

**Issue**: Training uses a fixed learning rate.

**Impact**: May converge slower or to worse optima than with proper scheduling.

**Recommendation**:
```python
# Add to train.py
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1000, T_mult=2
)
# Or linear warmup + cosine decay
```

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

**Status**: ✅ Implemented in `train.py`

**Benefit**: ~2× speedup, ~50% memory reduction.

**Usage**: Set `use_amp: true` in `config.yaml` (enabled by default).

```yaml
# config.yaml
training:
  use_amp: true  # Set to false to disable
```

**Implementation Details**:
- Uses `torch.autocast` with float16 for forward passes
- Uses `torch.amp.GradScaler` for loss scaling
- Properly handles gradient clipping with `scaler.unscale_()`
- Compatible with gradient accumulation

### 2. Vectorized Prefix Mask via block_id

**Status**: ✅ Implemented

**Benefit**: Eliminates Python loop overhead in prefix mask construction.

**Measured Performance** (RTX 3090, batch=16, seq=1024):
| Masking Method | fp32 | fp16 |
|----------------|------|------|
| Legacy (Python loops) | 993 ms | 891 ms |
| Vectorized (block_id) | 330 ms | 133 ms |
| **Speedup** | **3.0×** | **6.7×** |

**How it works**:

1. **Dataloader** (`loader.py`) computes `block_id` tensor during data loading:
   - Each position gets a block ID indicating which board block it belongs to
   - Positions in the same block can attend to each other bidirectionally
   - Orphan positions (padding, move tokens) get unique IDs

2. **Model** (`model.py`) uses vectorized tensor operations:
   ```python
   # Instead of nested Python loops:
   same_block = block_id.unsqueeze(-1) == block_id.unsqueeze(-2)  # [B, S, S]
   mask = causal_mask.unsqueeze(0) | same_block
   ```

**Files Modified**:
- `src/dataloader/data.py`: Returns `block_boundaries` from `game_to_token_ids()`
- `src/dataloader/loader.py`: Builds and yields `block_id` tensor
- `src/models/model.py`: Uses vectorized mask construction with `block_id`
- `src/train/train.py`: Passes `block_id` to model

---

## Potential Improvements

### High Priority

#### 1. Gradient Checkpointing

**Benefit**: Trade compute for memory.

```python
from torch.utils.checkpoint import checkpoint

for layer in self.layers:
    h = checkpoint(layer, h, mask, input_pos, use_reentrant=False)
```

---

### Medium Priority

#### 2. Data Augmentation

**Idea**: Mirror boards horizontally (flip a-h to h-a).

**Benefit**: 2× effective data, helps model learn symmetry.

```python
def mirror_position(tokens):
    # Swap files: a↔h, b↔g, c↔f, d↔e
    # Mirror move: e2e4 → d2d4 (after adjusting)
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

## Code Quality Improvements

### 1. Type Hints

Add type annotations throughout:
```python
def fen_to_position_tokens(fen: str) -> List[str]:
    ...

def game_to_token_ids(
    game_df: pd.DataFrame,
    skip_board_prob: float = 0.0
) -> Tuple[List[int], List[Tuple[int, str, List[float], bool]], List[Tuple[int, int]]]:
    ...
```

### 2. Configuration Validation

Add schema validation for config files:
```python
from pydantic import BaseModel, validator

class TrainingConfig(BaseModel):
    learning_rate: float
    batch_size: int
    num_epochs: int

    @validator('learning_rate')
    def lr_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('learning_rate must be positive')
        return v
```

### 3. Logging Improvements

Replace print statements with proper logging:
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Starting epoch {epoch}")
logger.debug(f"Batch loss: {loss.item()}")
```

### 4. Unit Tests

Add tests for critical functions:
```python
def test_fen_to_tokens():
    tokens = fen_to_position_tokens("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert len(tokens) == 68
    assert tokens[0] == "start_pos"
    assert tokens[67] == "white_to_move"

def test_vocab_consistency():
    for token, idx in token_to_idx.items():
        assert idx_to_token[idx] == token
```

---

## Performance Optimization Checklist

| Optimization | Status | Measured Speedup |
|--------------|--------|------------------|
| Mixed precision (FP16) | ✅ Implemented | ~2.5× |
| Vectorized prefix masking | ✅ Implemented | 3× (fp32), 6.7× (fp16) |
| Gradient checkpointing | ❌ Not implemented | Memory: 2-4× expected |
| Multi-GPU (DDP) | ❌ Not implemented | Linear with GPUs |
| Optimized data loading | ✅ IterableDataset | Baseline |

---

## Recommended Next Steps

### Immediate (Quick Wins)

1. Add validation loop to detect overfitting
2. Implement learning rate scheduling

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
