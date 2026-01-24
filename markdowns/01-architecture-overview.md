# Chess Decoder - Architecture Overview

## Project Purpose

This project implements a chess-playing transformer AI that learns to:
1. **Predict optimal moves** given a board position
2. **Evaluate positions** (win/draw/loss probabilities)
3. **Generate board states** autoregressively

The system uses supervised learning on Stockfish-analyzed games rather than reinforcement learning.

---

## Project Structure

```
decoder/
├── src/
│   ├── models/                    # Neural network architectures
│   │   ├── model.py               # ChessDecoder (causal transformer)
│   │   ├── encoder.py             # ChessEncoder (bidirectional transformer)
│   │   └── vocab.py               # Tokenization & vocabulary system
│   ├── dataloader/                # Data loading pipeline
│   │   ├── data.py                # FEN → token conversion
│   │   ├── loader.py              # ChessIterableDataset (decoder)
│   │   └── encoder_loader.py      # ChessEncoderDataset (encoder)
│   ├── train/                     # Training infrastructure
│   │   ├── train.py               # Decoder training loop
│   │   ├── train_encoder.py       # Encoder training loop
│   │   ├── config.yaml            # Decoder hyperparameters
│   │   ├── config_encoder.yaml    # Encoder hyperparameters
│   │   └── checkpoints/           # Saved model weights
│   └── eval/                      # Evaluation & inference
│       ├── play_move.py           # Model vs Stockfish games
│       └── elo_eval.py            # ELO estimation utilities
├── scripts/                       # Utility scripts
│   ├── extract_fen_moves.py       # Extract (FEN, move) pairs from parquets
│   └── estimate_move_tokens.py    # Compute sequence statistics
├── parquets/                      # Training data (Parquet format)
├── checkpoints/                   # Additional checkpoint storage
├── pgns/                          # Generated game records
└── wandb/                         # Weights & Biases logs
```

---

## Two Model Architectures

### 1. ChessDecoder (`model.py`)

A **causal decoder** transformer that processes game sequences autoregressively.

| Property | Value |
|----------|-------|
| Architecture | Causal Decoder with dual masking |
| Input | Interleaved board + move sequences |
| Max Sequence Length | 256 tokens |
| Output Heads | Policy (vocab_size) + Value (3) |
| Use Case | Full game simulation, move + value prediction |

**Key Innovation**: Uses both causal and prefix masking to prevent "cheating" during training while still allowing full board context for move selection.

### 2. ChessEncoder (`encoder.py`)

A **bidirectional encoder** transformer that processes single positions.

| Property | Value |
|----------|-------|
| Architecture | Bidirectional Encoder |
| Input | Single board position (68 tokens) |
| Max Sequence Length | 68 tokens (fixed) |
| Output Heads | Policy only |
| Use Case | Position-only move prediction |

**Simpler approach**: No game history, just evaluates one position at a time.

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Framework | PyTorch |
| Transformer Modules | TorchTune |
| Chess Logic | python-chess, BulletChess |
| Experiment Tracking | Weights & Biases |
| Evaluation Baseline | Stockfish |
| Data Format | Apache Parquet |

---

## Key Design Decisions

### 1. Fixed Board Representation (68 tokens)

Every chess position is encoded as exactly 68 tokens:
- Enables consistent tensor shapes
- Simplifies masking logic
- Position always starts with `start_pos` and ends with side-to-move

### 2. Moves as Vocabulary Tokens

All ~1,900 legal chess moves are individual tokens (UCI format).
- Direct policy prediction (no separate move decoder)
- Enables move masking for legal move filtering

### 3. Supervised Learning from Stockfish

- Targets are `best_move` from engine analysis, not human moves
- WDL probabilities provide position evaluation signal
- More consistent training signal than human games

### 4. Dual Masking Strategy (Decoder)

- **Causal mask**: For board token prediction (prevents looking ahead)
- **Prefix mask**: For move prediction (allows full board context)
- Prevents the model from "seeing" the answer before predicting

### 5. RoPE Instead of Absolute Positions

- Rotary Positional Embeddings handle position information
- Better extrapolation to different sequence lengths
- No learned position embeddings needed

---

## Data Requirements

Training data must be Parquet files with columns:
- `game_id`: Unique game identifier
- `ply`: Move number within game
- `fen`: Chess position (FEN notation)
- `played_move`: Move actually played (UCI)
- `best_move`: Stockfish's recommended move (UCI)
- `win`, `draw`, `loss`: Probability estimates from Stockfish

---

## Model Comparison

| Aspect | Decoder | Encoder |
|--------|---------|---------|
| Attention Type | Causal + Prefix | Bidirectional |
| Sequence Input | Game history | Single position |
| Batch Size | 16 | 256 |
| Learning Rate | 5e-5 | 1e-4 |
| Loss Functions | Move + Board + WDL | Move only |
| Parameters | ~85M (12L, 768D, 12H) | ~85M (same) |
| Training Complexity | Higher (dual pass) | Lower (single pass) |

---

## File Quick Reference

| File | Lines | Purpose |
|------|-------|---------|
| `src/models/model.py` | 178 | ChessDecoder architecture |
| `src/models/encoder.py` | 206 | ChessEncoder architecture |
| `src/models/vocab.py` | 245 | Vocabulary & tokenization |
| `src/dataloader/data.py` | 84 | FEN tokenization functions |
| `src/dataloader/loader.py` | 120 | Decoder dataset & dataloader |
| `src/dataloader/encoder_loader.py` | 120 | Encoder dataset & dataloader |
| `src/train/train.py` | 187 | Decoder training loop |
| `src/train/train_encoder.py` | 197 | Encoder training loop |
| `src/train/config.yaml` | 30 | Decoder configuration |
| `src/train/config_encoder.yaml` | 27 | Encoder configuration |
| `src/eval/play_move.py` | 216 | Model vs Stockfish evaluation |
| `src/eval/elo_eval.py` | ~50 | ELO calculation utilities |
