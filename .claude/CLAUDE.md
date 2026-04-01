# Chess Decoder Project

## What This Is

A decoder-only chess transformer (~116M params, 12 layers, 1024 embed, 16 heads) that autoregressively generates thinking traces (variations) before outputting a final move. Two-pass architecture: causal for board generation, prefix (bidirectional within board blocks) for move/value prediction.

## Directory Layout

```
src/models/model.py          # ChessDecoder model (5 heads: board, policy, thinking_policy, wl, d)
src/models/vocab.py          # Vocabulary (1968 tokens), sub-vocab mappings
src/dataloader/              # FEN-to-token conversion, reconstitute_games, policy_index
src/train/                   # Pretraining loop + config.yaml
src/finetune/                # Thinking variation finetuning + config.yaml
src/rl/                      # GRPO reinforcement learning + config.yaml
src/eval/                    # ELO evaluation against Stockfish
src/export/                  # TorchScript export for C++ engine
src/cpp/decoder/             # C++ inference engine (single + batched, pybind11)
src/cpp/                     # MCTS/TensorRT engine (optional, needs TRT)
scripts/                     # Evaluation, inference, data generation scripts
tests/                       # Pytest suite (77 tests)
markdowns/                   # Technical documentation (01-10)
exports/                     # Exported TorchScript models (gitignored)
exports/base/                # Default export (backbone.pt, weights/, vocab.json, config.json)
checkpoints/                 # Model checkpoints (gitignored)
bin/                         # External binaries like Stockfish (gitignored)
trt/                         # TensorRT engine files for MCTS (gitignored)
```

## Build & Run

- **Always use `uv run`** for all Python commands
- **Build everything**: `uv sync` (builds C++ decoder engine via pybind11)
- **Build with MCTS**: `uv sync --extra mcts` (also builds TensorRT-based MCTS engine)
- **Rebuild C++ decoder after changes**: `uv pip install -e src/cpp/decoder/ --no-build-isolation`

## Two C++ Extensions

| Module | Import | Needs | Purpose |
|--------|--------|-------|---------|
| `_decoder_inference_cpp` | Always built | libtorch (via PyTorch) | Decoder inference (single + batched) |
| `_inference_cpp` | Optional (`--extra mcts`) | TensorRT + CUDA | MCTS tree search with Leela model |

## Tests

```bash
uv run pytest tests/ -m "not gpu and not cpp" -v   # CPU-only (~2s)
uv run pytest tests/ -v                              # All tests (~90s, needs GPU + exports/base/)
```

Markers: `@pytest.mark.gpu` (CUDA), `@pytest.mark.cpp` (exported model in exports/base/).

## Data

- **Pretrain parquets**: downloaded via `./scripts/download_and_convert_pretraining_data.sh`, path configured in each module's `config.yaml` under `parquet_dir` or `pretrain_parquet_dir`
- **Variation parquets**: generated via `scripts/generate_variations.py` (needs MCTS), configured as `variation_parquet_dir`
- Set `PARQUET_DIR` env var to control download location (default: `parquets/`)

## Key Architecture Details

- **68 tokens per board**: start_pos + 64 squares + end_pos + castling + side_to_move
- **Sub-vocabs**: board_head outputs 41 tokens, policy heads output 1924 move tokens
- **Fourier injection**: WL/D continuous values encoded via learned Fourier features, replace placeholder token embeddings before transformer layers
- **Thinking trace format**: `[board_68] start_think [root_move wl d board_68 pv_move wl d board_68 ... end_var] ... end_think final_move wl d`
- **State machine** for autoregressive generation: MOVE → WL_D → BOARD → AFTER_BOARD → AFTER_END_VAR → FINAL

## Batched C++ Engine

The batched engine (`src/cpp/decoder/batched_engine.cpp`) uses per-element valid mask buffers (not KV zeroing) to handle sequences of different lengths. Pre-allocated `[B, 1, 1, max_seq_len]` FP32 mask buffers are updated incrementally via `masked_fill_`. This is critical for correctness — KV zeroing causes attention weight dilution (exp(0)=1 steals probability mass).

## Stockfish

Binary at `bin/stockfish`. Eval scripts auto-discover it before falling back to PATH. Download from https://stockfishchess.org/download/.
