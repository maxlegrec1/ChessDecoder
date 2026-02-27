# C++ Thinking Inference Engine — Full Rewrite Plan

## Context

The current C++ inference engine (`src/cpp/decoder/`) achieves 97% move match and 81% COT match vs Python. After extensive debugging, three intertwined bugs were identified:

1. **CUDA graph state pollution**: `syncGraphToPrefixCache()` (torch_backbone.cpp:338-339) creates **views** into graph buffers (`pg_buf_k_`), not clones. When the prefix graph replays, it overwrites `pg_buf_k_`, corrupting the "dynamic" prefix cache that the view points to. This causes index-out-of-bounds crashes on the 2nd FEN.

2. **CPU vs GPU head divergence**: WL/D heads were evaluated on CPU with scalar FP32 `gemv`, while Python runs FP16 GEMM on GPU. Different accumulation order causes different argmax results for close logits, picking different bucket centers. Since WL/D values get Fourier-encoded and injected as embeddings, a single bucket difference cascades into full sequence divergence.

3. **Debug code pollution**: Extensive `dbg_sync` barriers, debug prints, and `cudaDeviceSynchronize` calls scattered throughout from debugging sessions.

Rather than patching these bugs incrementally, we will **rewrite the engine from scratch** with a clean design that guarantees 100% numerical match with Python.

## Framework Decision: TorchScript (libtorch)

**Use TorchScript**, not TensorRT. Reasons:
- TorchScript backbone dispatches the **exact same CUDA kernels** as PyTorch -> guaranteed numerical match
- Already have a working export pipeline (`src/export/export_torchscript.py`)
- Head weights are small enough that `torch::mm` on GPU matches Python exactly
- TRT would require extensive validation of every quantization/fusion decision
- Current TorchScript approach already achieves ~448 tok/s (5.6x Python)

## File Structure

Keep the existing 5-file structure. Delete `trt_backbone.cpp/hpp` (unused).

| File | Role |
|------|------|
| `vocab.hpp/cpp` | Vocabulary, FEN tokenization, legal move generation -- **no changes needed** |
| `heads.hpp/cpp` | CPU head eval + weight loading + data accessors -- **no changes needed** |
| `torch_backbone.hpp/cpp` | libtorch backbone, KV caches, CUDA graphs -- **fix syncGraphToPrefixCache bug** |
| `decoder_engine.hpp/cpp` | State machine, GPU heads, orchestration -- **full rewrite** |
| `bindings.cpp` | pybind11 bindings -- **no changes needed** |
| `setup.py` | Build config -- **no changes needed** |

## Critical Bug Fix: `torch_backbone.cpp`

### syncGraphToPrefixCache -- view -> clone

```cpp
// BEFORE (BUG: creates views into graph buffer -- graph replay corrupts them)
prefix_past_keys_ = pg_buf_k_.index({Slice(), Slice(), Slice(), Slice(0, len)});
prefix_past_values_ = pg_buf_v_.index({Slice(), Slice(), Slice(), Slice(0, len)});

// AFTER (FIX: clone creates independent memory)
prefix_past_keys_ = pg_buf_k_.index({Slice(), Slice(), Slice(), Slice(0, len)}).clone();
prefix_past_values_ = pg_buf_v_.index({Slice(), Slice(), Slice(), Slice(0, len)}).clone();
```

## Rewrite: `decoder_engine.hpp`

Clean header with:
- Same public API as current (predictMove, lastTokenIds, etc.)
- GPU tensor members for all heads (keep current pattern -- already correct)
- No `saved_prefix_hidden_` CPU vector -- only keep `saved_prefix_hidden_gpu_`
- Keep State enum

## Rewrite: `decoder_engine.cpp`

### Constructor (keep as-is, it's correct)

The current constructor correctly:
1. Reads config JSON
2. Creates vocab, heads, backbone
3. Uploads all head weights to GPU as transposed FP16 CUDA tensors
4. Creates board LUT on GPU

### State Machine -- Exact Python Match

The state transitions mirror `test_evaluate_thinking.py:ThinkingModelWrapper.predict_move`:

1. **MOVE**: `evalThinkingPolicyHeadGpu()` -> append move token -> `prefixIncrGraph` -> go to WL_D
2. **WL_D**: `predictWlGpu()` -> append wl_value -> `prefixIncrGraph(ov=wl, of=1)` -> `predictDGpu()` -> append d_value -> `prefixIncrGraph(ov=d, of=1)` -> go to BOARD
3. **BOARD**: causal prefill/catch-up -> 68 board tokens via GPU loop -> `syncGraphToPrefixCache()` -> `prefixBlockForward` -> AFTER_BOARD
4. **AFTER_BOARD**: causal graph -> board_head logits -> if `end_var`: AFTER_END_VAR; else: MOVE
5. **AFTER_END_VAR**: causal graph -> board_head logits -> if `end_think`: FINAL; else: MOVE
6. **FINAL**: `evalPolicyHeadGpu()` with legal move masking -> return move

### What to remove
- All `dbg_sync` calls and debug prints
- The `c10::cuda::getCurrentCUDAStream().synchronize()` at top of predictMove
- The `saved_prefix_hidden_` CPU vector

## Implementation Phases

### Phase 1: Fix root cause + cleanup
- Edit `torch_backbone.cpp:338-339`: view -> clone
- Clean all debug code from decoder_engine.cpp
- Remove CPU hidden state vector

### Phase 2: Build and verify correctness
- Build: `uv pip install -e src/cpp/decoder --force-reinstall --no-deps --no-build-isolation`
- Quick test (5 FENs): `scripts/verify_quick.py`
- Full test (100 FENs): `scripts/verify_cpp_vs_python.py`

### Phase 3: Optimize to 1000 tok/s

Current: ~448 tok/s. Target: 1000 tok/s (2.2x improvement needed).

#### Optimization strategies:

1. **Eliminate remaining CPU<->GPU transfers**
   - `prefixInit` and `prefixBlockForward` output to CPU (`float* hidden_out`) then we upload to GPU
   - Add GPU-output variants: `prefixInitGpu()` / `prefixBlockForwardGpu()` that return `torch::Tensor` directly
   - Saves 2 transfers per variation (one for prefixInit, one per prefixBlockForward)

2. **Fuse causal catch-up tokens**
   - Between boards, 3 tokens (move, wl, d) are processed one-by-one via `causalIncrementalGraph`
   - Batch these into a single 3-token forward pass using dynamic cache path (skipping graph for 3 tokens)
   - Or: keep graph but avoid per-token CPU sync overhead

3. **Avoid `item<int>()` / `item<float>()` sync points**
   - `argmax().item<int>()` forces a CUDA sync each time
   - For board generation: already batched via `causalBoardStep` (GPU-only loop)
   - For WL/D prediction: chain the argmax -> bucket_centers lookup entirely on GPU, only sync once
   - For AFTER_BOARD/AFTER_END_VAR: keep argmax on GPU, use GPU scalar comparison

4. **Reduce prefixBlockForward overhead**
   - 68-token block forward is the bottleneck per variation
   - Profile whether attention mask construction is significant
   - Consider: pre-allocate mask buffer, reuse across calls

5. **CUDA stream pipelining**
   - Overlap causal board generation with prefix cache sync
   - Use separate streams for head eval vs backbone forward

6. **Memory pre-allocation**
   - Pre-allocate all temporary vectors (input_ids, input_pos, override_values, etc.)
   - Avoid per-call `std::vector` allocation in hot paths

7. **Profile-guided optimization**
   - Use `nsys profile` to identify actual bottlenecks
   - Focus optimization effort on the top 2-3 hotspots

## Verification

### Success criteria
1. **100% COT exact match** on all 100 FENs (token_ids identical)
2. **100% move match** on all 100 FENs
3. **Exact WL/D match** (same bucket center, since both use FP16 GEMM + argmax)
4. **No crashes** when running multiple FENs sequentially
5. **Speed**: 1000+ tok/s after optimization phase

### Commands
```bash
# Build
uv pip install -e src/cpp/decoder --force-reinstall --no-deps --no-build-isolation

# Quick test (5 FENs)
uv run python scripts/verify_quick.py --checkpoint checkpoint_step_32000.pt --export-dir export

# Full test (100 FENs)
uv run python scripts/verify_cpp_vs_python.py --checkpoint checkpoint_step_32000.pt --export-dir export --num-fens 100
```
