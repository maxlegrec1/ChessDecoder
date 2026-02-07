# Architecture Overview

## Summary

ChessDecoder is a decoder-only transformer model for chess that generates board representations autoregressively, predicts moves, and evaluates positions. The model uses a **two-pass forward architecture**: a causal pass for board token generation and a prefix (bidirectional-within-block) pass for move and value prediction.

The model is first **pretrained** on standard game sequences (board-move-value triples), then **finetuned** with thinking variation sequences that teach the model to explore candidate moves before selecting a final move.

## Core Design Principles

1. **Fixed-length board representation**: Every chess position is encoded as exactly 68 tokens (`start_pos` + 64 squares + `end_pos` + castling + side_to_move).
2. **Sub-vocabulary heads**: Each prediction head outputs logits over only its relevant sub-vocabulary, not the full vocab. The `board_head` predicts 41 board tokens, the policy heads predict 1924 UCI move tokens.
3. **Two-pass architecture**: Board generation uses causal (autoregressive) attention. Move and value prediction use prefix attention (bidirectional within board blocks, causal across blocks).
4. **Fourier value injection**: WL/D evaluation values are encoded via learned Fourier features and injected as embeddings at placeholder positions, providing value context for subsequent predictions.
5. **Soft bucket losses**: Value heads predict distributions over 100 buckets rather than single scalars, using soft cross-entropy loss with linear interpolation between nearest buckets.

## Sequence Format

### Pretraining Sequence

A game is represented as a sequence of position-move-value triples:

```
[board_68] [move] [wl_value] [d_value] [board_68] [move] [wl_value] [d_value] ...
```

Each position contributes 71 tokens: 68 board tokens + 1 move + 1 WL placeholder + 1 D placeholder.

### Finetuning Variation Sequence

Thinking sequences add exploration before the final move:

```
[root_board_68] [start_think]
  [root_move_1] [wl] [d] [board_68] [pv_move_1] [wl] [d] [board_68] ... [end_var]
  [root_move_2] [wl] [d] [board_68] ... [end_var]
[end_think] [final_move] [wl_value] [d_value]
```

## Model Heads

| Head | Type | Output Size | Attention | Purpose |
|------|------|-------------|-----------|---------|
| `board_head` | `nn.Linear` | 41 (board sub-vocab) | Causal (pass 1) | Predict next board/structural/signal token |
| `policy_head` | `nn.Linear` | 1924 (move sub-vocab) | Prefix (pass 2) | Predict final/normal moves |
| `thinking_policy_head` | `nn.Linear` | 1924 (move sub-vocab) | Prefix (pass 2) | Predict variation moves (root + PV) |
| `wl_head` | `ValueHead` MLP | 100 buckets | Prefix (pass 2) | Predict WL value in [-1, 1] |
| `d_head` | `ValueHead` MLP | 100 buckets | Prefix (pass 2) | Predict D (draw) value in [0, 1] |

## Two-Pass Forward

### Pass 1: Causal (Board Generation)

Standard autoregressive attention (lower-triangular mask). Every position can only see positions before it. The `board_head` is applied to these hidden states to predict:

- **Board tokens**: pieces, empty squares, `start_pos`, `end_pos`, castling, side_to_move
- **Structural tokens**: `wl_value`, `d_value`, `start_pos` (the d_value -> start_pos transition)
- **Signal tokens**: `generic_move` (at STM positions), `continue_var`, `end_var`, `new_variation`, `end_think`, `start_think`

### Pass 2: Prefix (Move + Value Prediction)

Uses a combined causal + same-block mask: positions within the same board block can attend to each other bidirectionally, while cross-block attention remains causal. This gives the move/value heads full bidirectional context within each board position.

The prefix pass receives:
- **Fourier-encoded WL/D values**: At `wl_value`/`d_value` placeholder positions, the token embeddings are replaced with learned Fourier features encoding the actual evaluation values.
- **Block IDs**: Each board's 68 tokens share a block ID. Non-board tokens (moves, wl/d placeholders, structural tokens) get unique "orphan" IDs so they only attend causally.

From the prefix hidden states:
- `policy_head` predicts the final/normal move at STM positions
- `thinking_policy_head` predicts variation moves at `start_think`/`end_var`/STM positions
- `wl_head` predicts WL at the move token position (STM + 1)
- `d_head` predicts D at the `wl_value` placeholder position (STM + 2)

## Attention Mask Visualization

For a pretraining sequence with two boards:

```
Causal mask (pass 1):
  Standard lower-triangular. Position i sees positions 0..i.

Prefix mask (pass 2):
  Block 0: root_board (pos 0-67)
  Orphan: move (pos 68), wl (pos 69), d (pos 70)
  Block 1: board2 (pos 71-138)
  Orphan: move (pos 139), wl (pos 140), d (pos 141)

  For pos 67 (STM of board1, block 0):
    Can see: pos 0-67 (causal) UNION pos 0-67 (block 0)
    = pos 0-67 (full bidirectional within board1)

  For pos 138 (STM of board2, block 1):
    Can see: pos 0-138 (causal) UNION pos 71-138 (block 1)
    = pos 0-138 (block 1 is subset of causal range)

  For pos 71 (start_pos of board2, block 1):
    Can see: pos 0-71 (causal) UNION pos 71-138 (block 1)
    = pos 0-138 (bidirectional within block!)
```

## Target Tensors

The model uses **separate target tensors** for each head type:

- **`board_target_ids`**: Board sub-vocab indices (0-40). Natural shifted next-token for most positions, with overrides at STM positions (`generic_move`), continuation positions (`continue_var`), and new variation positions (`new_variation`). Padded with `IGNORE_INDEX = -100`.
- **`move_target_ids`**: Move sub-vocab indices (0-1923). Set only at positions where a move head should predict. Padded with `IGNORE_INDEX = -100`.
- **`move_mask`**: Boolean mask for positions where `policy_head` should predict (STM positions in pretraining, `end_think` in finetuning).
- **`thinking_move_mask`**: Boolean mask for positions where `thinking_policy_head` should predict (`start_think`, `end_var`, PV board STM positions). Only used during finetuning.

## Technology Stack

| Component | Technology |
|-----------|------------|
| Framework | PyTorch |
| Transformer Modules | TorchTune (Llama-style) |
| Chess Logic | python-chess |
| Experiment Tracking | Weights & Biases |
| Evaluation Baseline | Stockfish |
| Data Format | Apache Parquet |

## File Structure

| File | Purpose |
|------|---------|
| `src/models/vocab.py` | Vocabulary definitions (1968 tokens), sub-vocabulary mappings |
| `src/models/model.py` | ChessDecoder model with sub-vocab heads |
| `src/dataloader/data.py` | FEN-to-token conversion, game sequence generation |
| `src/dataloader/loader.py` | Pretraining IterableDataset and DataLoader |
| `src/train/train.py` | Pretraining training loop |
| `src/train/config.yaml` | Pretraining configuration |
| `finetune/data.py` | Variation sequence generation (Plackett-Luce ordering) |
| `finetune/loader.py` | Finetuning IterableDataset (mixed pretrain + variation) |
| `finetune/train.py` | Finetuning training loop |
| `finetune/config.yaml` | Finetuning configuration |
| `scripts/think.py` | Thinking inference (autoregressive variation generation) |
