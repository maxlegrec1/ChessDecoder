# C++ Inference Engine

C++/libtorch inference engine for the ChessDecoder thinking model. Replicates the full Python thinking inference pipeline with CUDA graph acceleration.

**Current performance**: ~672 tok/s FP16 avg (875 tok/s @ 712 tokens, 992 tok/s @ 427 tokens). ~8.4x over Python's ~80 tok/s.

## Architecture

```
                   TorchScript backbone (FP16)
                   ┌──────────────────────────┐
                   │  12-layer transformer     │
  input_ids ──────►│  with RoPE + KV cache     │──────► hidden [1, 1, E]
  past_k/v ──────► │  override values (Fourier)│──────► present_k/v
                   └──────────────────────────┘
                              │
            ┌─────────────────┼──────────────────┐
            ▼                 ▼                   ▼
       board_head        policy_heads         value_heads
     (41 classes)      (1924 moves)         (WL + D MLPs)
       CPU/GPU            GPU                   GPU
```

### Components

| File | Role |
|---|---|
| `torch_backbone.hpp/cpp` | TorchScript backbone wrapper, dual KV cache, CUDA graph capture/replay |
| `decoder_engine.hpp/cpp` | Full thinking state machine (MOVE → WL_D → BOARD → AFTER_BOARD → ...) |
| `heads.hpp/cpp` | Board/policy/value head weights loaded from raw `.bin` files |
| `vocab.hpp/cpp` | Token vocabulary, FEN→token encoding, legal move masking (uses `bulletchess`) |
| `bindings.cpp` | pybind11 Python interface |
| `setup.py` | Build configuration linking against libtorch |

### Dual KV Cache

Two independent KV caches operate in parallel:

1. **Causal KV cache** — for autoregressive board token generation (67 tokens per board). Uses CUDA graph with padded fixed-size buffers (`max_seq_len` wide).

2. **Prefix KV cache** — for move/value prediction. Supports:
   - Incremental 1-token forward (CUDA graph) for orphan tokens (moves, WL, D)
   - Block forward (dynamic) for board blocks (bidirectional intra-block attention)

After each board generation phase, the causal cache results are synced to the prefix cache for the subsequent prefix block forward.

### CUDA Graphs

Multiple CUDA graphs are captured at initialization:

1. **Causal graph** — 1-token incremental with `max_seq_len`-wide padded KV buffer
2. **Prefix graph** — 1-token incremental with `max_seq_len`-wide padded KV buffer
3. **Tiered board gen graphs** — 5 additional causal graphs at sizes 128, 256, 512, 1024, 2048. Board generation selects the smallest tier that fits `board_start_pos + 67`, reducing KV bandwidth by up to 20x at early positions. Falls back to the full-size graph for positions > 2048. All tier graphs share input tensors with the main causal graph. (~186 MB additional memory)
4. **Board gen** uses the selected tier/causal graph with GPU-side head eval (matmul + argmax + LUT) to avoid any CPU synchronization during the 67-step board generation loop

Key detail: after `cat(past_k, new_k)`, the new token's KV lands at position `max_len` (not `cache_len`). A scatter copy moves it to `cache_len` after each replay.

## Using a New Checkpoint

### Step 1: Export

```bash
uv run python src/export/export_torchscript.py \
    --checkpoint path/to/your_checkpoint.pt \
    --output-dir export
```

This produces:
```
export/
├── backbone.pt          # TorchScript traced backbone (FP16)
├── config.json          # Model dimensions (embed_dim, num_layers, etc.)
├── vocab.json           # Full 1968-token vocabulary
└── weights/
    ├── board_head_weight.bin    # [41, E] FP16
    ├── board_head_bias.bin      # [41] FP16
    ├── policy_head_weight.bin   # [1924, E] FP16
    ├── policy_head_bias.bin     # [1924] FP16
    ├── thinking_policy_head_weight.bin
    ├── thinking_policy_head_bias.bin
    ├── wl_w1_weight.bin         # [H, E] FP16
    ├── wl_w1_bias.bin           # [H] FP16
    ├── wl_w2_weight.bin         # [B, H] FP16
    ├── wl_w2_bias.bin           # [B] FP16
    ├── wl_bucket_centers.bin    # [B] FP32 (precision-sensitive)
    ├── d_w1_weight.bin
    ├── d_w1_bias.bin
    ├── d_w2_weight.bin
    ├── d_w2_bias.bin
    └── d_bucket_centers.bin
```

The export script:
- Extracts the causal backbone from the full `ChessDecoder` model
- Traces it with `torch.jit.trace` in FP16
- Verifies exact round-trip match (prefill + incremental)
- Saves head weights as raw binary (read by C++ at load time)
- Writes `config.json` with model hyperparameters and `vocab.json`

### Step 2: Build the C++ Extension

```bash
uv pip install -e src/cpp/decoder --force-reinstall --no-deps --no-build-isolation
```

This compiles the pybind11 extension against your installed PyTorch's libtorch. Requires:
- PyTorch with CUDA support
- pybind11
- C++17 compiler
- CUDA toolkit at `/usr/local/cuda`

### Step 3: Verify Correctness

Quick check (5 FENs, exact token match):
```bash
uv run python scripts/verify_quick.py \
    --checkpoint path/to/your_checkpoint.pt \
    --export-dir export
```

Full check (100 FENs, move + COT match rates):
```bash
uv run python scripts/verify_cpp_vs_python.py \
    --checkpoint path/to/your_checkpoint.pt \
    --export-dir export \
    --num-fens 100
```

### Step 4: Use in Python

```python
import _decoder_inference_cpp as cpp

engine = cpp.ThinkingInferenceEngine(
    "export/backbone.pt",
    "export/weights",
    "export/vocab.json",
    "export/config.json",
)

move = engine.predict_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0.0)
print(move)  # e.g. "e2e4"

# Inspect thinking trace
token_ids = list(engine.last_token_ids())
token_names = [engine.idx_to_token(t) for t in token_ids]

# Throughput stats (accumulated across calls)
print(f"{engine.total_tokens / engine.total_time:.0f} tok/s")
```

### Step 5: Profile

```bash
uv run python scripts/profile_engine.py
```

Outputs per-phase breakdown (ms/tok):
- `board_gen` — CUDA graph board generation loop (67 steps per board)
- `board_prefill` / `board_catchup` — causal cache warm-up
- `prefix_init` — initial prefix KV cache population
- `prefix_block` — prefix block forward for board tokens
- `prefix_incr` — prefix incremental for orphan tokens
- `causal_incr` — causal incremental for AFTER_BOARD/AFTER_END_VAR
- `head_eval` — policy/value head GPU evaluation

### Step 6: Run Elo Evaluation

```bash
uv run python scripts/test_evaluate_thinking.py
```

Uses the C++ engine as a drop-in replacement for the Python `ThinkingModelWrapper` in Stockfish games.

## Changing Model Architecture

If the model architecture changes (different `embed_dim`, `num_layers`, `num_heads`, `head_dim`, vocabulary size, head structure), you need to:

1. Update `src/export/backbone_causal.py` if the backbone forward signature changes (this is the Python wrapper, not the exported file)
2. Update `src/export/common.py` `export_head_weights()` if heads change
3. Re-export with `export_torchscript.py` (produces new `config.json` with dimensions)
4. The C++ engine reads dimensions from `config.json` at runtime — no recompilation needed for dimension changes
5. If head *structure* changes (e.g., adding a new head), update `heads.hpp/cpp` and `decoder_engine.cpp`
6. Rebuild: `uv pip install -e src/cpp/decoder --force-reinstall --no-deps --no-build-isolation`

## Current Limits

### Performance

- **Board generation is ~77% of total time** (~1.15 ms/tok). Tiered CUDA graphs reduce KV bandwidth at early positions, but later boards still use larger tiers.
- **No quantization**: runs in FP16. INT8 quantization would roughly halve memory bandwidth requirements.
- **Single-batch only**: no batched inference. Each `predict_move` call processes one FEN.
- **CPU↔GPU syncs**: causal catch-up before each board (non-first) does a `syncGraphToCausalCache` → dynamic forward → `syncCausalCacheToGraph` round-trip to maintain numerical exactness.
- **Prefix block forward is dynamic** (not CUDA-graphed) because board block length varies. This is ~5% of total time.

### Correctness

- **FP16 precision**: different GEMM kernel dispatch across buffer sizes (tiered 128-2048 vs Python's dynamic tensors) can cause rare board token divergences that cascade into different thinking traces. Moves match 98% and COT exact match is 85% across 100 FENs.
- **Causal catch-up**: subsequent boards use a non-graph forward to avoid accumulating FP16 precision drift. This costs ~0.1 ms/tok but ensures board token accuracy.

### Deployment

- **Hardcoded CUDA path**: `setup.py` assumes `/usr/local/cuda`. Change `CUDA_ROOT` for non-standard installations.
- **No Windows support**: Linux-only (tested on Ubuntu 22.04+).
- **PyTorch version coupling**: the built `.so` links against a specific PyTorch version's libtorch. Upgrading PyTorch requires rebuilding.
- **No graceful error recovery**: if the model generates garbage (e.g., wrong tokens for a board), the engine continues to max_seq_len and falls back to the first root move or a no-thinking fallback.

## Potential Improvements

### Near-term (high impact, moderate effort)

1. ~~**Tiered CUDA graphs for board generation**~~ — **DONE**. 5 tiers (128-2048) reduce board_gen from 1.60 to 1.15 ms/tok. Overall ~520 → ~672 tok/s (+29%).

2. **INT8 weight-only quantization** — Quantize the backbone weights to INT8 while keeping activations in FP16 (W8A16). Halves weight memory bandwidth (~288MB → ~144MB). Combined with tiered graphs, could reach **~1000 tok/s**.

3. **Fused causal catch-up** — Instead of `syncGraph→dynamicForward→syncGraph` for non-first boards, run the catch-up tokens through a captured CUDA graph with variable-length KV (would need multiple graph captures or a different approach). Saves ~0.1 ms/tok.

### Medium-term (moderate impact, significant effort)

4. **Batched inference** — Process multiple FENs in parallel. Would require batched KV cache management and batched state machine logic. Useful for Elo evaluation (multiple games in parallel) or serving.

5. **FlashAttention / SDPA integration** — Replace the cuBLAS-based Q*K matmul with a fused attention kernel. Note: for S=1 queries, SDPA's `memory_efficient_attention` was tested and was **2x slower** due to higher per-call overhead. FlashAttention-2's `flash_attn_with_kvcache` might be better for single-query incremental since it's designed for this case.

6. **Speculative decoding for board tokens** — Board sequences are highly predictable (68 tokens, mostly deterministic). A small draft model could propose multiple tokens at once, verified in parallel by the main model. Would reduce the number of backbone forward calls per board.

7. **Prefix block forward via CUDA graph** — Board blocks are always 68 tokens. Could capture a dedicated CUDA graph for the prefix block forward, avoiding dynamic tensor allocation.

### Long-term (high impact, high effort)

8. **Custom CUDA kernels** — Replace PyTorch's generic `torch::mm` / `torch::cat` / `index_put` with fused kernels (e.g., fused KV scatter, fused head eval). The current board gen loop has ~5 kernel launches per step that could be fused into 1-2.

9. **TensorRT backbone** — TRT would optimize the full backbone graph (layer fusion, kernel auto-tuning) for potentially 2-3x over TorchScript. Main blocker: TRT's KV cache handling is non-trivial.

10. **Continuous batching / tree search** — Instead of generating one full thinking trace per FEN, explore multiple variation branches in parallel and prune early. Would require fundamental changes to the state machine.
