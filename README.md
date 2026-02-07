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
│   ├── eval/            # ELO evaluation against engines
│   └── mcts/            # Monte Carlo Tree Search utilities
├── finetune/            # Finetuning with thinking variations
│   ├── train.py         # Finetuning loop (mixed pretrain + variation batches)
│   ├── loader.py        # Finetuning dataset
│   ├── data.py          # Variation sequence generation
│   └── config.yaml      # Finetuning hyperparameters
├── scripts/             # Inference and data-generation scripts
│   ├── think.py         # Thinking inference (autoregressive variation generation)
│   ├── play_move.py     # Single-move prediction from a FEN
│   └── generate_variations.py
├── markdowns/           # In-depth technical documentation
├── cpp/                 # C++ utilities for engine integration
├── tests/               # Test suite
└── checkpoints/         # Saved model weights
```

## Getting Started

### Prerequisites

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- CUDA-capable GPU (for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/maxlegrec1/ChessDecoder.git
cd ChessDecoder

# Install dependencies with uv
uv sync
```

### Training

```bash
# Pretraining on game sequences
python -m src.train.train           # uses src/train/config.yaml

# Finetuning with thinking variations (requires a pretrained checkpoint)
python -m finetune.train            # uses finetune/config.yaml
```

Training configuration (batch size, learning rate, data paths, etc.) lives in the respective `config.yaml` files.

### Inference

```bash
# Predict a move from a FEN position
python scripts/play_move.py

# Run thinking inference (generate variations, then choose a move)
python scripts/think.py \
    --checkpoint checkpoints/model.pt \
    --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
    --temperature 0.0 --device cuda
```

### Evaluation

```bash
python src/eval/elo_eval.py       # single-game ELO evaluation
python src/eval/elo_eval_n.py     # multi-game ELO evaluation
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
