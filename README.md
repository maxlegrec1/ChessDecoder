# ChessDecoder

A decoder-only transformer (~116M parameters) that learns to play chess by generating board representations autoregressively, predicting moves, and evaluating positions. The model uses a two-pass forward architecture — a causal pass for board token generation and a prefix (bidirectional-within-block) pass for move and value prediction — and can be finetuned with thinking variations that teach it to reason through candidate moves before selecting a final move.

## Key Ideas

- **Fixed-length board encoding.** Every chess position is exactly 68 tokens: `start_pos` + 64 squares + `end_pos` + castling + side-to-move.
- **Sub-vocabulary heads.** The board head outputs 41 tokens; the policy heads output 1924 UCI move tokens. No head ever predicts the full 1968-token vocabulary.
- **Two-pass forward.** Pass 1 (causal) generates board tokens autoregressively. Pass 2 (prefix) allows bidirectional attention within each board block to predict moves and values.
- **Fourier value injection.** Win/loss and draw scalars are encoded via learned Fourier features and injected as embeddings at placeholder positions.
- **Thinking variations.** After pretraining on board-move-value triples, the model is finetuned on MCTS-derived variation sequences with Plackett-Luce ordering, teaching it to explore candidate lines before committing to a move.

## Project Structure

```
ChessDecoder/
├── src/
│   ├── models/          # Transformer model, vocabulary, encoder
│   ├── dataloader/      # Pretraining data pipeline (Parquet → token sequences)
│   ├── train/           # Pretraining loop and config
│   ├── finetune/        # Finetuning with thinking variations
│   ├── rl/              # GRPO reinforcement learning
│   ├── eval/            # ELO evaluation against engines
│   ├── export/          # TorchScript export for C++ engine
│   ├── mcts/            # Monte Carlo Tree Search (C++ via pybind11)
│   └── cpp/             # C++ inference engines (decoder + MCTS)
├── scripts/             # Inference, evaluation, and utility scripts
├── tests/               # Pytest test suite
├── markdowns/           # In-depth technical documentation
├── exports/             # Exported TorchScript models and head weights
├── checkpoints/         # Saved model weights
├── bin/                 # External binaries (Stockfish)
└── trt/                 # TensorRT engine files (for MCTS)
```

## Prerequisites

- **Python >= 3.13**
- **[uv](https://github.com/astral-sh/uv)** — fast Python package manager (required)
- **CUDA toolkit** — NVIDIA GPU with CUDA support
- **[TensorRT](https://developer.nvidia.com/tensorrt)** — only needed for MCTS-based variation generation (optional)
- **[Stockfish](https://stockfishchess.org/)** — for ELO evaluation (optional)

## Installation

### 1. Clone the repository and submodules

```bash
git clone https://github.com/maxlegrec1/ChessDecoder.git
cd ChessDecoder
git submodule init
git submodule update
```

This pulls the [chess-library](https://github.com/disservin/chess-library) C++ dependency used by the inference engine.

### 2. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Install dependencies and build the C++ decoder engine

```bash
uv sync
```

This single command:
- Creates a virtual environment with Python 3.13
- Installs all Python dependencies (PyTorch, python-chess, PyArrow, etc.)
- Builds the C++ pybind11 decoder inference engine (`decoder-inference-cpp`) against libtorch and CUDA

To also build the MCTS/TensorRT engine (needed for variation generation):

```bash
uv sync --extra mcts
```

> **Note:** The MCTS build expects TensorRT at `/usr/local/TensorRT-10.14.1.48` and CUDA at `/usr/local/cuda`. If your paths differ, edit `src/cpp/setup.py`.

### 4. Verify the installation

```bash
# Check that the Python package works
uv run python -c "from src.models.model import ChessDecoder; print('Model OK')"

# Check that the C++ decoder engine loaded
uv run python -c "import _decoder_inference_cpp; print('C++ decoder engine OK')"

# (Optional) Check MCTS engine
uv run python -c "import _inference_cpp; print('C++ MCTS engine OK')"
```

## Installing Stockfish

Stockfish is needed for ELO evaluation. Download the binary for your platform from <https://stockfishchess.org/download/> and place it at `bin/stockfish`:

```bash
mkdir -p bin
# Download and extract Stockfish, then:
mv stockfish-ubuntu-x86-64-avx2 bin/stockfish
chmod +x bin/stockfish
```

The eval scripts automatically look for `bin/stockfish` before falling back to `PATH`.

## Data Preparation

### Pretraining Data

Pretraining data comes from [Leela Chess Zero](https://lczero.org/) V6 training archives. The download script fetches tar files from https://storage.lczero.org/files/training_data/test91/ and converts them to Parquet.

```bash
# Download 5 tar files and convert to parquets/
./scripts/download_and_convert_pretraining_data.sh 5

# Or specify a custom output directory via env var:
PARQUET_DIR=/path/to/data ./scripts/download_and_convert_pretraining_data.sh 10
```

Then point `src/train/config.yaml` → `data.parquet_dir` at your parquets directory.

### Finetuning Data

Finetuning data is generated by running MCTS variations on the pretraining parquets. This requires a TensorRT engine file from a trained model and the MCTS extension (`uv sync --extra mcts`).

```bash
# Generate finetuning parquets with MCTS variations
./scripts/generate_finetuning_data.sh trt/model_dynamic_leela.trt parquets parquets_variations
```

Then point `src/finetune/config.yaml` → `data.variation_parquet_dir` at the output directory.

## Training

```bash
# Pretraining on game sequences
uv run python -m src.train.train           # uses src/train/config.yaml

# Finetuning with thinking variations (requires a pretrained checkpoint)
uv run python -m src.finetune.train        # uses src/finetune/config.yaml

# RL training with GRPO (requires a finetuned checkpoint)
uv run python -m src.rl.train              # uses src/rl/config.yaml
```

Training configuration (batch size, learning rate, data paths, etc.) lives in the respective `config.yaml` files.

## Inference

```bash
# Run thinking inference (generate variations, then choose a move)
uv run python scripts/think.py \
    --checkpoint checkpoints/model.pt \
    --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
    --temperature 0.0 --device cuda
```

## Testing

```bash
# Run all CPU tests (~2s)
uv run pytest tests/ -m "not gpu and not cpp" -v

# Run all tests including GPU and C++ engine (~90s)
uv run pytest tests/ -v
```

## Evaluation

```bash
uv run python src/eval/elo_eval.py       # single-game ELO evaluation
uv run python src/eval/elo_eval_n.py     # multi-game ELO evaluation
```

These scripts play the model against Stockfish and compute an ELO estimate. Make sure Stockfish is installed (see above).

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

Detailed technical documentation is organized into topic-specific guides in the [`markdowns/`](markdowns/) directory:

| # | Document | Description |
|---|---|---|
| 01 | [Architecture Overview](markdowns/01-architecture-overview.md) | Two-pass forward design, sequence formats, attention masks, and output heads |
| 02 | [Vocabulary & Tokenization](markdowns/02-vocabulary-and-tokenization.md) | The 1968-token vocabulary, sub-vocabularies, and FEN-to-token conversion |
| 03 | [Data Pipeline](markdowns/03-data-pipeline.md) | Pretraining and finetuning data loading, Parquet format, batch construction |
| 04 | [Decoder Training](markdowns/04-decoder-training.md) | Pretraining loop, two-pass training strategy, loss functions and weights |
| 05 | [Model Architecture](markdowns/05-model-architecture.md) | Layer-by-layer model components, parameter counts, Fourier encoder details |
| 06 | [Evaluation & Inference](markdowns/06-evaluation-and-inference.md) | Single-position, multi-position, and thinking inference modes |
| 07 | [Known Issues & Improvements](markdowns/07-known-issues-and-improvements.md) | Current limitations and planned improvements |
| 08 | [Quick Reference](markdowns/08-quick-reference.md) | File locations, key constants, token layouts, and training commands |
| 09 | [Finetuning Thinking Variations](markdowns/09-finetuning-thinking-variations.md) | Thinking sequences, Plackett-Luce ordering, dual policy heads |
| 10 | [RL GRPO Training](markdowns/10-rl-grpo-training.md) | Reinforcement learning with GRPO: rollouts, rewards, policy optimization |

## Dependencies

Core libraries: **PyTorch**, **TorchTune**, **python-chess**, **PyArrow**, **NumPy**, **Weights & Biases**. See [`pyproject.toml`](pyproject.toml) for the full list and version constraints.
