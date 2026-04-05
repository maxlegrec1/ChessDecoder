# ChessDecoder

A decoder-only transformer (~116M parameters) that learns to play chess by generating board representations autoregressively, predicting moves, and evaluating positions. The model uses a two-pass forward architecture — a causal pass for board token generation and a prefix (bidirectional-within-block) pass for move and value prediction — and can be finetuned with thinking variations that teach it to reason through candidate moves before selecting a final move.

## Quick Start

```bash
# Clone, install, download data, run inference — all in one go
git clone --recursive https://github.com/maxlegrec1/ChessDecoder.git
cd ChessDecoder
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv (skip if already installed)
uv sync                                             # install deps + build C++ engine

# Download a few pretraining parquets
./scripts/download_and_convert_pretraining_data.sh 2

# Run thinking inference on the starting position (requires a checkpoint)
uv run python chessdecoder/inference/think.py \
    --checkpoint checkpoints/your_model.pt \
    --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

## Key Ideas

- **Fixed-length board encoding.** Every chess position is exactly 68 tokens: `start_pos` + 64 squares + `end_pos` + castling + side-to-move.
- **Sub-vocabulary heads.** The board head outputs 41 tokens; the policy heads output 1924 UCI move tokens. No head ever predicts the full 1968-token vocabulary.
- **Two-pass forward.** Pass 1 (causal) generates board tokens autoregressively. Pass 2 (prefix) allows bidirectional attention within each board block to predict moves and values.
- **Fourier value injection.** Win/loss and draw scalars are encoded via learned Fourier features and injected as embeddings at placeholder positions.
- **Thinking variations.** After pretraining on board-move-value triples, the model is finetuned on MCTS-derived variation sequences with Plackett-Luce ordering, teaching it to explore candidate lines before committing to a move.

## Project Structure

```
ChessDecoder/
├── chessdecoder/            # Installable Python package (import chessdecoder...)
│   ├── models/              # Transformer model, vocabulary
│   ├── dataloader/          # Pretraining data pipeline (Parquet → token sequences)
│   ├── train/               # Pretraining loop + config
│   ├── finetune/            # Finetuning with thinking variations + config
│   ├── rl/                  # GRPO reinforcement learning + config
│   ├── eval/                # ELO evaluation against Stockfish
│   ├── inference/           # Thinking inference (autoregressive generation)
│   ├── export/              # TorchScript export for C++ engine
│   ├── cpp/decoder/         # C++ decoder inference engine (pybind11)
│   └── cpp/mcts/            # C++ MCTS/TensorRT engine (optional)
├── scripts/             # Evaluation and utility scripts
├── tests/               # Pytest test suite (77 tests)
├── markdowns/           # Technical documentation (10 guides)
├── exports/             # Exported TorchScript models (gitignored)
├── checkpoints/         # Model checkpoints (gitignored)
├── bin/                 # External binaries — Stockfish (gitignored)
└── trt/                 # TensorRT engines for MCTS (gitignored)
```

## Installation

### 1. Clone and install

```bash
git clone --recursive https://github.com/maxlegrec1/ChessDecoder.git
cd ChessDecoder
uv sync
```

This creates a virtualenv, installs all Python dependencies, and builds the C++ decoder inference engine against libtorch and CUDA.

> **Need [uv](https://github.com/astral-sh/uv)?** `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 2. (Optional) Build the MCTS engine

Only needed if you want to generate finetuning variation data via MCTS:

```bash
uv sync --extra mcts
```

Requires TensorRT at `/usr/local/TensorRT-10.14.1.48` and CUDA at `/usr/local/cuda`. Edit `chessdecoder/cpp/mcts/setup.py` if your paths differ.

### 3. (Optional) Install Stockfish

Only needed for ELO evaluation. Download from <https://stockfishchess.org/download/>:

```bash
mkdir -p bin
mv stockfish-ubuntu-x86-64-avx2 bin/stockfish
chmod +x bin/stockfish
```

The eval scripts look for `bin/stockfish` automatically, falling back to `PATH`.

### 4. Verify

```bash
uv run python -c "from chessdecoder.models.model import ChessDecoder; print('OK')"
uv run python -c "import _decoder_inference_cpp; print('C++ decoder engine OK')"
```

## Data Preparation

### Pretraining Data

Pretraining data comes from [Leela Chess Zero](https://lczero.org/) V6 training archives:

```bash
# Download 5 tar files, convert to Parquet (default: parquets/)
./scripts/download_and_convert_pretraining_data.sh 5

# Or specify a custom output directory:
PARQUET_DIR=/path/to/data ./scripts/download_and_convert_pretraining_data.sh 10
```

Then set `parquet_dir` in `chessdecoder/train/config.yaml` to point at your parquets directory.

**Manual conversion** of a single tar file:

```bash
uv run python chessdecoder/dataloader/reconstitute_games.py lc0_tars/training.2411.tar
```

### Finetuning Data (requires MCTS)

Generate MCTS variation data from pretraining parquets:

```bash
./scripts/generate_finetuning_data.sh trt/model_dynamic_leela.trt parquets parquets_variations
```

Then set `variation_parquet_dir` in `chessdecoder/finetune/config.yaml`.

## Training

```bash
uv run python chessdecoder/train/train.py        # Pretraining  (chessdecoder/train/config.yaml)
uv run python chessdecoder/finetune/train.py     # Finetuning   (chessdecoder/finetune/config.yaml)
uv run python chessdecoder/rl/train.py           # RL with GRPO (chessdecoder/rl/config.yaml)
```

## Inference

```bash
uv run python chessdecoder/inference/think.py \
    --checkpoint checkpoints/model.pt \
    --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
    --temperature 0.0
```

## Testing

```bash
uv run pytest tests/ -m "not gpu and not cpp" -v   # CPU-only (~2s)
uv run pytest tests/ -v                              # All tests (~90s, needs GPU + exports/base/)
```

## Evaluation

```bash
# Thinking model vs Stockfish
uv run python scripts/eval_elo_thinking.py --export-dir exports/base --num-games 200 --elo 2000

# Root policy (no thinking) vs Stockfish
uv run python scripts/eval_elo_root.py --export-dir exports/base --num-games 200 --elo 2000

# Pass@k accuracy
uv run python scripts/pass_at_k.py --export-dir exports/base --num-fens 100 --k 10
```

## Model at a Glance

| Property | Value |
|---|---|
| Architecture | Decoder-only Transformer (RoPE, SwiGLU) |
| Parameters | ~116M |
| Layers | 12 |
| Embedding dim | 1024 |
| Attention heads | 16 |
| Vocabulary | 1968 tokens (41 board + 1924 move + special) |
| Context window | 2048 tokens |
| Value prediction | 100-bucket soft distributions for WL and D |

### Output Heads

| Head | Output | Purpose |
|---|---|---|
| `board_head` | 41 logits | Next board / structural / signal token |
| `policy_head` | 1924 logits | Final or normal move prediction |
| `thinking_policy_head` | 1924 logits | Variation move prediction (finetuning) |
| `wl_head` | 100 buckets | Win/loss value in [-1, 1] |
| `d_head` | 100 buckets | Draw probability in [0, 1] |

## Documentation

| # | Document | Description |
|---|---|---|
| 01 | [Architecture Overview](markdowns/01-architecture-overview.md) | Two-pass forward, sequence formats, attention masks |
| 02 | [Vocabulary & Tokenization](markdowns/02-vocabulary-and-tokenization.md) | 1968-token vocabulary, sub-vocabularies, FEN conversion |
| 03 | [Data Pipeline](markdowns/03-data-pipeline.md) | Data loading, Parquet format, batch construction |
| 04 | [Decoder Training](markdowns/04-decoder-training.md) | Pretraining loop, loss functions and weights |
| 05 | [Model Architecture](markdowns/05-model-architecture.md) | Layer-by-layer components, Fourier encoder |
| 06 | [Evaluation & Inference](markdowns/06-evaluation-and-inference.md) | Inference modes (single, multi-position, thinking) |
| 07 | [Known Issues](markdowns/07-known-issues-and-improvements.md) | Limitations and planned improvements |
| 08 | [Quick Reference](markdowns/08-quick-reference.md) | Key constants, token layouts, commands |
| 09 | [Finetuning Variations](markdowns/09-finetuning-thinking-variations.md) | Thinking sequences, Plackett-Luce, dual policy heads |
| 10 | [RL GRPO Training](markdowns/10-rl-grpo-training.md) | Rollouts, rewards, policy optimization |
