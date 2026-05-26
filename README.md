# ChessEncoder — encoder-only chess training base

Single classical transformer encoder over the 68 board tokens. The same stack
predicts policy and WDL from the CLS (``start_pos``) token — no decoder, no
Perceiver latents, no transition head. This is the base point for A/B
architecture experiments (different LR/decay schedules, attention variants,
LC0-style 64-token boards with stm/castling folded into per-square embeddings,
history planes, LC0's cross-attention policy head, etc.).

## Layout

```
chessdecoder/
├── dataloader/
│   ├── data.py           # FEN -> 68 position tokens
│   └── loader.py         # IterableDataset: group by game_id, sample N random plies per game
├── models/
│   ├── vocab.py          # full vocab + move sub-vocab (1924 UCI moves)
│   ├── layers.py         # EncoderLayer (pre-RMSNorm, bidir attn, SwiGLU)
│   ├── value_buckets.py  # 2-D-simplex WDL categorical (project_targets / mean_wdl)
│   └── model.py          # ChessEncoder (tok+pos emb -> N layers -> policy/wdl heads off CLS)
├── train/
│   ├── config.yaml
│   └── train.py
└── utils/
    ├── distributed.py    # DDP shims (no-op on single GPU)
    ├── fp8.py            # torchao Float8Linear conversion + compile(model.encoder)
    ├── muon.py           # Muon + AdamW (embeddings / heads / norms on AdamW arm)
    └── training.py       # load_config / save_checkpoint / wandb resume
```

## Forward shape

```
board_ids [N, 68]
   layout:  [ start_pos | a1..h8 (64 squares) | end_pos | castling | stm ]
      │
      ▼  tok_emb + pos_emb (learned absolute)
      ▼  EncoderLayer × num_layers (RMSNorm-attn-SwiGLU)
      ▼  final RMSNorm   →  h [N, 68, E]
      │
      ├──► policy_head: Linear(E, 1924)  ←  h[:, 0, :]
      └──► wdl_head:    Linear(E, N_CELLS) ← h[:, 0, :]
```

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # if needed
uv sync
```

## Train

```bash
CUDA_VISIBLE_DEVICES=0 uv run python chessdecoder/train/train.py [chessdecoder/train/config.yaml]
```

Knobs in `config.yaml`:

- `model.{embed_dim, num_heads, num_layers, d_ff, seq_len}` — the architecture A/B surface.
- `data.{batch_size, positions_per_game}` — effective positions/step = `B × N`.
- `training.optimizer` — `muon` (Muon on hidden matrices + AdamW on embeddings/heads/norms) or `adamw`.
- `training.{use_fp8, fp8_recipe, fp8_compile}` — torchao Float8Linear swap of every Linear with both dims `% 16 == 0` and `>= 256`, plus `torch.compile(model.encoder)`.
- `training.resume_from` — checkpoint dir; latest `checkpoint_*.pt` is picked.

## Metrics (wandb)

`total_loss / move_loss / wdl_loss`, `move_acc`, `wdl_acc / q_mae / d_mae / wdl_entropy`,
`policy_logit_max / value_logit_max`, **`pos_per_s` (positions/second = `B × N / step_time`)**.

## Tests

```bash
uv run pytest tests/ -m "not gpu" -v
```
