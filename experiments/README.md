# BT4-shape loss/step study

**Goal:** beat the BT4-shape dense baseline on **loss-per-step** at fixed effective
batch 2048. Metric is sample efficiency, NOT throughput.

**Hard constraints (never change):** `num_layers=15`, `embed_dim=1024`,
effective batch `2048`. Everything else is fair game.

**Fixed protocol for fair comparison:**
- same data + val split (`val_pct=2`, game-hash leakage-free), `seed=42`
- `log_every_n_steps=5`, `positions_per_game=1`
- precision: fp8 + compile (same for every run, so it's not a confound)
- micro-batch 1024 × grad_accum 2 = 2048 effective
- compare at **matched steps**: train total/move/wdl loss + held-out val move_acc
- one run at a time (single H100)

**Allowed levers:** FFN capacity via MoE; per-arm LR (Muon vs AdamW) multiplier;
weight decay; router aux-loss weight; gradient-flow fixes; init; LR schedule.

---

## Experiment log

| # | name | change vs baseline | result (vs baseline @ matched step) | verdict |
|---|------|--------------------|--------------------------------------|---------|
| 0 | `bt4_baseline` | — (15L/1024/32h/SwiGLU 1536, Muon lr 3e-3 shared) | reference (below) | ✅ done, 20k |
| 1 | `lrmult_0.1` | AdamW arm lr 3e-3 → 3e-4 (adamw_lr_mult=0.1) | worse early (Δacc −0.05 @5k), ties @20k (Δloss −0.19, Δacc −0.001, val slightly worse) | ❌ neutral/neg |
| 2 | `muon_6e3` | Muon lr 3e-3 → 6e-3, AdamW held at 3e-3 (lr=6e-3, adamw_lr_mult=0.5) | faster @1k (Δloss −0.66) but wash by 20k (move_acc 0.562 vs 0.556, val tied) | ❌ within noise |
| 3 | `bt4_moe` | FFN → MoE 8 experts top-2, expert_d_ff 768 (iso active FLOPs, ~4× FFN params) | running | — |

**Conclusion on LR levers:** baseline LR (Muon 3e-3, AdamW 3e-3) is already near-optimal;
per-arm decoupling and Muon-LR bumps move loss/step < noise floor. Moving to capacity (MoE).

### Baseline (bt4_baseline) reference — train loss/acc, fp8+compile, batch 2048
| step | loss | pol | wdl | move_acc | val move_acc |
|---|---|---|---|---|---|
| 1000 | 13.878 | 2.075 | 3.504 | 0.370 | — |
| 5000 | 10.557 | 1.529 | 2.914 | 0.496 | — |
| 10000 | 9.632 | 1.374 | 2.761 | 0.552 | — |
| 15000 | 9.456 | 1.355 | 2.679 | 0.549 | — |
| 20000 | 9.151 | 1.299 | 2.656 | 0.556 | 0.556 |

Train↔val gap ≈ 0 at 20k (no overfitting — epoch-bug fix holds). ~4.5 steps/s.

## Notes / observations
- Baseline params: 138.3M (BT4 reports 191.3M; gap = BT4's smolgen + mish).
- Known suboptimality at start: Muon and AdamW **share one LR** (3e-3). AdamW
  arm (embeddings/heads/norms/router) likely wants ~10x lower. First experiment.
