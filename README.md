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
│   ├── eval/            # ELO evaluation against engines
│   ├── mcts/            # Monte Carlo Tree Search (C++ via pybind11)
│   └── cpp/             # C++ TensorRT inference engine and MCTS
├── scripts/             # Inference, data-generation, and utility scripts
├── markdowns/           # In-depth technical documentation
├── tests/               # Test suite (CI/CD)
└── checkpoints/         # Saved model weights
```

## Prerequisites

- **Python >= 3.13**
- **[uv](https://github.com/astral-sh/uv)** — fast Python package manager (required)
- **CUDA toolkit** — NVIDIA GPU with CUDA support
- **[TensorRT 10.14.1](https://developer.nvidia.com/tensorrt)** — installed at `/usr/local/TensorRT-10.14.1.48` (required for the C++ inference engine)
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

### 3. Install Python dependencies and build the C++ extension

```bash
uv sync
```

This single command:
- Creates a virtual environment with Python 3.13
- Installs all Python dependencies (PyTorch, python-chess, PyArrow, etc.)
- Builds the C++ pybind11 inference extension (`chessrl-inference-cpp`) against TensorRT and CUDA

> **Note:** The C++ build expects TensorRT at `/usr/local/TensorRT-10.14.1.48` and CUDA at `/usr/local/cuda`. If your paths differ, edit `src/cpp/setup.py` accordingly.

### 4. Verify the installation

```bash
# Check that the Python package works
uv run python -c "from src.models.model import ChessDecoder; print('Model OK')"

# Check that the C++ extension loaded
uv run python -c "import _inference_cpp; print('C++ extension OK')"
```

## Installing Stockfish

Stockfish is needed for ELO evaluation. Install it for your platform:

**Ubuntu / Debian:**

```bash
sudo apt update && sudo apt install stockfish
```

**macOS (Homebrew):**

```bash
brew install stockfish
```

**From source (latest):**

```bash
git clone https://github.com/official-stockfish/Stockfish.git
cd Stockfish/src
make -j$(nproc) build ARCH=x86-64-modern
sudo cp stockfish /usr/local/bin/
```

Verify with:

```bash
stockfish <<< "uci" | head -1
# Should print: Stockfish <version> by ...
```

## Data Preparation

### Pretraining Data

Pretraining data comes from [Leela Chess Zero](https://lczero.org/) V6 training archives.
The tar files can be downloaded from https://storage.lczero.org/files/training_data/test91/.

`reconstitute_games.py` converts these binary archives into Parquet files with columns like `fen`, `played_move`, `best_move`, `game_id`, `ply`, `win`, `draw`, `loss`, etc.

**Quick start — download and convert automatically:**

```bash
# Download 5 tar files and convert them to parquets in parquets/
./scripts/download_and_convert_pretraining_data.sh 5 parquets
```

**Manual step-by-step:**

```bash
# 1. Download a tar file
mkdir -p lc0_tars
curl -O --output-dir lc0_tars \
    https://storage.lczero.org/files/training_data/test91/training.2411.tar

# 2. Convert to Parquet
uv run python reconstitute_games.py lc0_tars/training.2411.tar

# Output: lc0_tars/training.2411.parquet
# Move it to your data directory:
mkdir -p parquets && mv lc0_tars/training.2411.parquet parquets/
```

Then point `src/train/config.yaml` → `data.parquet_dir` at your parquets directory.

### Finetuning Data

Finetuning data is generated by running MCTS variations on the pretraining parquets using `scripts/generate_variations.py`. This requires a TensorRT engine file from a trained model.

**Quick start:**

```bash
# Generate finetuning parquets with MCTS variations
./scripts/generate_finetuning_data.sh model_dynamic_leela.trt parquets parquets_variations
```

**Manual:**

```bash
uv run python scripts/generate_variations.py \
    --parquet-dir parquets \
    --output-dir parquets_variations \
    --simulations 600 \
    --max-variations 5 \
    --max-variation-depth 20 \
    --engine-path model_dynamic_leela.trt \
    --parallel-trees 128
```

Then point `src/finetune/config.yaml` → `data.variation_parquet_dir` at the output directory.

## Training

```bash
# Pretraining on game sequences
uv run python -m src.train.train           # uses src/train/config.yaml

# Finetuning with thinking variations (requires a pretrained checkpoint)
uv run python -m src.finetune.train        # uses src/finetune/config.yaml
```

Training configuration (batch size, learning rate, data paths, etc.) lives in the respective `config.yaml` files.

## Inference

```bash
# Predict a move from a FEN position
uv run python scripts/play_move.py

# Run thinking inference (generate variations, then choose a move)
uv run python scripts/think.py \
    --checkpoint checkpoints/model.pt \
    --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
    --temperature 0.0 --device cuda
```

## Evaluation

```bash
uv run python src/eval/elo_eval.py       # single-game ELO evaluation
uv run python src/eval/elo_eval_n.py     # multi-game ELO evaluation
```

These scripts play the model against Stockfish and compute an ELO estimate. Make sure Stockfish is installed and on your `PATH`.

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

## Dependencies

Core libraries: **PyTorch**, **TorchTune**, **python-chess**, **PyArrow**, **NumPy**, **Weights & Biases**. See [`pyproject.toml`](pyproject.toml) for the full list and version constraints.
