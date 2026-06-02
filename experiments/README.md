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
| 3 | `bt4_moe` | FFN → MoE 8 experts top-2, expert_d_ff 768 (iso active FLOPs, 351M/138M active) | extended to 40k. Clean read (val move_acc 30-40k: dense 0.561 vs moe 0.558; pol-loss 1.237 vs 1.258): **dense slightly ahead**. 20k "crossover" was single-batch noise. Data-limited (~6% epoch) → extra sparse params don't bind yet | ❌ iso-FLOP no win @40k |
| 4 | `bt4_moe_aux1e3` | iso-FLOP MoE, aux_loss_weight 1e-2 → 1e-3 (less balancing pressure → more specialization) | running | — |

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

## ⚠️ RESUME DATA-REPLAY BUG (found 06-01, fixed)
The "strange" curves on the resumed ext runs were a real bug. `loader.__iter__`
seeded the per-epoch RNG with `max(int(self.epoch), self._epoch_iter)`. With
persistent_workers the worker freezes `self.epoch` at spawn (== start_epoch on
resume) and `_epoch_iter` resets to 0, so the `max` emits the SAME seed for
`start_epoch+1` epochs → it **REPLAYS the resume epoch's plies**. Resuming our
ext runs at epoch 1 replayed epoch 1 during steps 22.6k–34k: train acc spiked
+5% (re-memorizing) while **val stayed flat**, then loss JUMPED at 34k when fresh
data (ep2) finally arrived. Three confirmations: ep-sequence sim (1,1,2),
jump only at the post-replay boundary, train↔val gap opening +0.05 in the
replay window and collapsing at 34k.
**Fix:** seed `_epoch_iter` from spawn-time epoch (`getattr(...,int(self.epoch)-1)+1`),
drop the `max`; every epoch now unique. Regression test:
tests/test_loader_epoch_resume.py. Also `wandb.log(..., step=step)`.
**Implication:** resumed-run curves past the resume epoch are contaminated; trust
the FRESH 0–20k runs (no resume) — there dense ≈ MoE (iso-FLOP, no win). Re-run
extensions fresh (no resume) for clean long curves.

## SCALING re-verification (06-01) — old flat scaling was the epoch bug
User flagged that the 8M-120M sweep barely scaled. Root cause: that sweep
(commit 8fcfc01) ran BEFORE the persistent_workers epoch fix (c44a669), with
positions_per_game=1 — so every size trained on the SAME frozen subset → all
converged to identical val regardless of capacity. Not physics, the bug.
- Overfit probe (experiments/overfit_probe.py): 138M fits a fixed 512-pos batch
  to 100% acc / 0.023 loss in BOTH bf16 and fp8 → model healthy, fp8 NOT a cap.
- Fresh runs w/ FIXED loader, matched steps @20k: clear monotonic scaling:
  | size | pol-loss | move_acc |
  |------|----------|----------|
  | 15M  | 1.408    | 0.532    |
  | 35M  | (running)| (running)|
  | 138M | 1.299    | 0.556    |
  138M beats 15M at every step (Δpol up to -1.1 early). Scaling restored.
**Implication: ALL pre-c44a669 sweeps (scaling, attention, input-format, lr) are
contaminated and should be re-run with the fixed loader before drawing conclusions.**

## MoE vs dense FINAL verdict (06-01, fixed loader, fresh, smoothed val @14-20k)
| run | active | val move_acc |
|-----|--------|--------------|
| 256M wide-dense (d_ff4096) | 256M | 0.5473 |
| 138M baseline (d_ff1536)   | 138M | 0.5439 |
| 351M MoE iso-FLOP (e768)   | 138M | 0.5345 |
| 823M MoE big (e2048)       | 256M | 0.5095 |
- Widening dense FFN (1536->4096) improves TRAIN (pol 1.299->1.222) but NOT val
  (0.544->0.547, within noise) -> FFN width buys fitting, not generalization.
- MoE is WORSE than dense on val at both sizes; bigger MoE much worse (anti-scaling).
  Routers are balanced (aux ~0.017/layer, near floor) -> NOT collapse; the cause is
  EXPERT UNDERTRAINING (each token updates 2/8 experts -> each expert sees ~1/4 the
  data; bigger experts more undertrained -> anti-scaling). MoE is the wrong lever in
  this data-limited / 20k-step regime. Confound: big MoE ran micro-512 (memory) vs
  micro-1024, but small<big MoE anti-scaling holds regardless.
- CONCLUSION: no FFN-capacity lever (dense width or MoE) beats the baseline on val.
  We're val/data-limited; generalization is capped by embed/depth (fixed) + data.
  Next val lever must add signal-per-step (aux supervision) or break the data limit
  (more data / longer training), not capacity.

## Big-batch test (06-01) — does feeding experts more tokens/update help MoE?
Hypothesis (user): MoE trails dense because experts are undertrained; a much
bigger batch (more tokens/expert/update) should fix it. Ran baseline + iso-FLOP
MoE at effective batch 8192 (micro1024 x gradaccum8). LR notes: 3e-3 under-updates
(4x fewer opt-steps), 1e-2 DIVERGES (the AdamW arm, not Muon, blows up), 6e-3 with
adamw_lr_mult=1.0 converges poorly (AdamW too hot on heads). Fix: Muon 6e-3 +
adamw_lr_mult=0.5 (AdamW 3e-3) -> clean & fast (pol 3.13 vs 3.97 @99 updates).
Result @4000 opt-updates: dense pol 1.247 / val 0.5585; MoE pol 1.263 / val 0.5529.
**SAME verdict as batch 2048: tied train, dense slightly ahead on val. Gap stable,
not closing -> undertraining was NOT the cause.** MoE's only edge is extra params,
which don't help in a data-limited regime (cf. wide-dense: more FFN -> better train,
flat val), and it pays a routing/coherence tax (2-of-8 routed 768 experts < one
co-trained 1536 FFN). MoE needs a capacity-limited regime to win; we're not in one.
Also reconfirmed: AdamW arm must be LR-decoupled at high Muon LR (adamw_lr_mult<1).

## ROUTER COLLAPSE + z-loss fix (06-01) — revives the MoE thread
experiments/router_analysis.py on a trained MoE router shows COLLAPSE: logits
blow up (logsumexp 9-19, max|logit| ~30), routing goes hard (top-1 prob ->1.0,
~0 entropy -> top-2 degenerates to top-1), and EARLY LAYERS leave 4-6 of 8
experts DEAD. Confirmed on a clean fixed-loader BT4 MoE (not just the pre-fix
checkpoint) -> intrinsic MoE dynamics, a real reason MoE underperformed.
**Fix: router z-loss (model.moe_z_loss_weight=1e-3).** A/B at 7500 steps (only
z-loss differs): logsumexp 9-19 -> ~0; top-1 prob L0 1.00 -> 0.76; entropy L0
0.00 -> 0.67; dead experts L0/L1/L2 5/4/6 -> 1/0/0 (total dead: many -> 1).
Router fixed, all experts used. Early perf (7500): z-loss MoE pol 1.518 <
dense 1.548 < ... wait dense 1.548, no-z 1.546, z-loss 1.518 -> z-loss MoE has
LOWEST pol-loss (first time MoE edges dense on anything). move_acc dense 0.504
still > z-loss 0.496. Running full 32k head-to-head (bt4_moe_zloss_full) vs
dense big-batch + no-z-loss MoE to see if it holds. Next knob if needed: lower
router LR (router is on AdamW arm).

## z-loss 32k head-to-head (big-batch 8192, Muon 6e-3 / AdamW 3e-3)
| run | pol@16k | pol@32k | acc@32k | val(>=24k) |
|-----|---------|---------|---------|-----------|
| dense              | 1.342 | 1.254 | 0.586 | 0.5585 |
| MoE no-z-loss      | 1.376 | 1.299 | 0.572 | 0.5529 |
| MoE z-loss 1e-3    | 1.328 | 1.259 | 0.584 | 0.5593 |
z-loss (router fix) moved iso-FLOP MoE from clearly-behind to TIED with dense:
val 0.5529 -> 0.5593 (= dense 0.5585, within noise). LED dense at 16k (pol 1.328
vs 1.342). Not yet a decisive win -> data-limit caps the generalization gain.
Next pushes: lower router LR; tune z-loss weight; train longer. train/moe_z_loss
now logged separately (commit 9833e76) for future runs.

## Gradient-flow analysis (experiments/grad_flow.py, on 20k checkpoints)
Both dense and MoE are **clean** — signal reaches all 15 layers, no vanishing/
exploding, no dead modules. Dense: attn grad-norm ~uniform across depth, FFN rises
toward output (normal). MoE: attn declines smoothly 0.25→0.13 with depth, router
has the smallest grad (1.3e-2, expected for [E]-dim output). **Not the bottleneck** —
the MoE's slow start is inherent cold-start, not a gradient bug. (Probe uses small
batch + policy-CE only; within-run depth profile is the signal, not absolute scale.)

## Notes / observations
- Baseline params: 138.3M (BT4 reports 191.3M; gap = BT4's smolgen + mish).
- Known suboptimality at start: Muon and AdamW **share one LR** (3e-3). AdamW
  arm (embeddings/heads/norms/router) likely wants ~10x lower. First experiment.
