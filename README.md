# ChessDecoder V2 — training-only base

This branch is a **clean training base** for A/B architecture experiments. It contains only
the V2 pretraining loop (encoder-latents + causal decoder + transition head), its dataloader,
the V2 model, and the small set of utilities the loop needs (DDP, FP8, Muon, wandb).

RL, finetuning, MCTS, C++ inference engines, TorchScript export, evaluation and inference
scripts have been **removed** from this branch on purpose, so the next architecture
iteration can be reasoned about and benchmarked against a known baseline.

## Layout

```
chessdecoder/
├── dataloader/
│   ├── data.py           # FEN -> 68 position tokens
│   └── loader_v2.py      # IterableDataset: group rows by game_id, build per-game arrays
├── models/
│   ├── vocab.py          # full vocab + move sub-vocab (1924 UCI moves)
│   └── v2/
│       ├── encoder_mode.py    # PerceiverPool / encoder benchmark
│       ├── layers.py          # TransformerEncoderLayer, FourierEncoder
│       ├── model_v2.py        # ChessDecoderV2 (encoder + causal decoder + heads)
│       └── value_buckets.py   # 2-D-simplex WDL categorical
├── train/
│   ├── config_v2.yaml    # training config
│   └── train_v2.py       # the training entrypoint
└── utils/
    ├── distributed.py    # DDP helpers (no-op on single-GPU)
    ├── fp8.py            # torchao Float8Linear conversion + compile
    ├── muon.py           # Muon optimizer for hidden 2-D weights, AdamW elsewhere
    └── training.py       # load_config / save_checkpoint / wandb resume
```

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv (skip if already installed)
uv sync
```

## Train

```bash
uv run python chessdecoder/train/train_v2.py [chessdecoder/train/config_v2.yaml]
```

Resume / FP8 / Muon are toggled in `config_v2.yaml`:

- `training.resume_from` — checkpoint dir to resume from (latest `checkpoint_*.pt` is picked).
- `training.use_fp8` + `training.fp8_recipe` + `training.fp8_compile` — torchao FP8 Linear swap.
- `training.optimizer` — `muon` (Muon for hidden 2-D weights + AdamW for the rest) or `adamw`.

## Tests

```bash
uv run pytest tests/ -m "not gpu" -v
```
