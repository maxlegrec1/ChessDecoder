# Search Agent — learning to probe a frozen chess oracle

Can RL teach an agent **what to evaluate**? A frozen 30M oracle maps any
position to (policy, value); the agent probes it on boards *of its own
choosing* — any board, written patch-by-patch, not just children of visited
nodes — then must output a move stronger than the oracle's own greedy policy.

Full plan + claim ladder (L0-L4): [markdowns/01-search-agent-plan.md](markdowns/01-search-agent-plan.md)

## Measured baselines

| | Elo vs SF UCI_Elo 2400 (0.1s/move) |
|---|---|
| **L0 — oracle greedy** (30M, ckpt 246k) | **2486, 95% CI [2458, 2515]** (400 games) |

## Layout

```
chessdecoder/
├── agent/                 # the search agent
│   ├── patch_vocab.py     # board = 16x 2x2-patch tokens + castle/stm/ep (19 tok)
│   ├── grammar.py         # episode state machine + per-slot token masks
│   ├── model.py           # 110M decoder-only LM over the ~31k agent vocab
│   ├── oracle.py          # frozen oracle wrapper (batched, legal top-4, memo)
│   ├── tasks.py           # Stage-A tasks: copy/apply/line/aggregate/distill/distance
│   ├── pretrain.py        # Stage-A loop (muon, bf16+compile, per-task wandb metrics)
│   └── config_stageA.yaml
├── dataloader/            # FEN -> 68 tokens + npz shard cache (oracle training)
├── models/                # ChessEncoder (the oracle architecture)
├── train/                 # oracle training loop + config_30M_oracle.yaml
└── utils/                 # muon, fp8, distributed shims, training helpers
scripts/
├── gen_t5_labels.py       # oracle-labeled corpus for distillation tasks
├── eval_vs_stockfish.py   # Elo evaluation (single SF process, sequential)
├── brawl.py               # model-vs-model head-to-head
└── bench_fp8_4090.py      # fp16/bf16/fp8 precision benchmark
experiments/               # encoder-architecture probes (oracle lineage docs)
tests/                     # pytest (agent vocab round-trip, vocab, models)
```

## Workflow

```bash
uv sync                                                          # env
uv run python chessdecoder/train/train.py \
    chessdecoder/train/config_30M_oracle.yaml                    # oracle (done -> ckpt 246k)
uv run python scripts/gen_t5_labels.py 5000000                   # label corpus
uv run python -m chessdecoder.agent.pretrain \
    chessdecoder/agent/config_stageA.yaml                        # Stage A (wandb: search-agent)
```

Stage B (scripted-teacher SFT) and the GRPO loop are specified in the plan
doc and land next.
