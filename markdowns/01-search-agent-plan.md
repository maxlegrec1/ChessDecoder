# Search Agent ("Oracle Prober") — Plan

Branch: `search-agent` (from `v2-encoder-train-clean`).

## 0. Thesis

Given a **frozen oracle** — a small encoder that maps any position to
(policy, value) — train an **agent** that probes the oracle on positions *of
its own choosing* (any board whatsoever, not just children of visited nodes)
and then outputs a final move that is **stronger than the oracle's own greedy
policy**. The thing to prove: **RL teaches the agent what to evaluate.**

### Claim ladder (each rung independently measurable)

| | Claim | Control |
|---|---|---|
| L0 | — | oracle greedy policy (the bar) |
| L1 | agent + K probes > oracle greedy | agent with K=0 (forced immediate answer) isolates "agent is just a better policy" from "probing added value" |
| L2 | agent + K probes ≥ PUCT-MCTS + K sims (same oracle, same budget) | hardcoded search baseline |
| L3 | RL'd agent > SFT-only agent at equal K | **the headline** — keep the SFT checkpoint |
| L4 | adaptive budget beats fixed budget at equal mean compute | later; needs per-probe cost λ |

Process is deliberately unrewarded (no PUCT-similarity bonus, no "good probe"
signal). If search-like probing emerges, it emerged from move-quality pressure
alone — that is the experiment.

## 0.1 Measured L0 baseline (2026-06-10)

Oracle = `ChessEncoder` 30M dense (D=512, H=8, L=10, d_ff=1152, geom +
cross_attn, lc0_64), trained ~246k micro-steps (~21 epochs over the 1.37B-
position LC0 corpus, bf16+compile, muon).
Checkpoint: `checkpoints/oracle_30M/oracle-30M_20260609_235451/checkpoint_246000.pt`
(val move_acc 0.593, wdl_acc 0.925, q_mae 0.086).

Protocol: greedy legal-masked policy, alternating colors, single Stockfish
process (Threads=1, 0.1s/move), sequential games, `scripts/eval_vs_stockfish.py`.

| SF UCI_Elo | games | W/D/L | score |
|---|---|---|---|
| 1800 | 30 | 23/5/2 | 85.0% |
| 2100 | 30 | 16/10/4 | 70.0% |
| **2400 (main)** | **400** | **182/133/85** | **62.1%** |

**L0 = 62.1% ± 3.8% (95%, empirical SE) vs SF 2400 ⇒ Elo 2486, 95% CI
[2458, 2515]** on this instrument (same binary + time control as all other
session baselines: V2-179M raw ≈ 2595, BT4 raw ≈ 2917). UCI_Elo's curve is
non-linear (implied Elo rises with anchor: 2100/2250/2486 at the three
anchors), so cross-anchor comparisons must reuse the 2400 anchor.

## 1. Components

1. **Oracle** (frozen): a small `ChessEncoder` from this branch's framework —
   position → (policy over 1924 moves, WDL). Trained once (Milestone 0),
   then never updated. Queried only by the harness.
2. **Agent**: small decoder-only transformer (~50–120M, 12 layers / 768–1024
   dim), native context 8k+, single full-vocab output head, grammar-masked
   sampling. The entire search state is the context — no tree structure, no
   external navigation state.
3. **Harness**: enforces the grammar, reconstructs probed boards, validity-
   checks, calls the oracle, injects replies, tracks budget, assigns board IDs.
4. **Reference labeler** (offline): a strong external searcher (Stockfish, or
   big MCTS over the same oracle) producing `Q_ref(root, a)` for every root
   move of every training position. Computed once per position.

## 2. The language

### 2.1 Probe = 16 board patches (the compression scheme)

Naive 68-token boards explode the context (~76 tokens/probe round). Grouping
squares into *large* region tokens fails combinatorially (one rank of 8 =
13^8 ≈ 8×10⁸ contents — not a vocab), but **groups of 4 work**: 13⁴ = 28,561
contents is a normal vocab size.

> **A board is always 16 fixed-order 2×2-patch tokens + 3 meta tokens
> (castling 16-way, stm 2-way, ep 65-way) = 19 tokens, fully self-contained.**

- 2×2 geometry (4 patch-rows × 4 patch-cols) over 1×4 half-ranks because
  chess locality is 2-D — king shelter, pawn chains, piece clusters span
  ranks *and* files. (1×4 is the cheap ablation.)
- Fixed patch order ⇒ patch index is implicit by sequence position; no
  coordinate tags.
- The all-empty patch dominates frequency (like whitespace in text);
  permanently-illegal patches (e.g. two kings in one patch) get hard-masked
  rows.
- Explicit `ep` meta token — fixes V2's EP blindness in the new design.
  Promotion/castling need no special casing: the next board just *is*
  whatever its patches say.
- Every probe spells out the complete board: no references, no diff chains,
  no keyframes, no board IDs, no silent state-drift risk. (A ref+edit-op diff
  scheme was considered — ~1.7× more compact but adds an entire risk class
  of in-weights diff composition plus ID/keyframe machinery. Rejected:
  at our budgets both fit in context, so robustness wins.)

**Patch-vocab implementation (two variants):**
- **V-a (default): flat patch vocab** — 28,561 patch tokens, tied in/out
  embeddings (~22M params at dim 768; acceptable on an 80–120M agent). One
  softmax per position — purest GRPO log-probs.
- **V-b (param-saving): factorized patch** — one sequence position per patch;
  output = 4 parallel 13-way heads, input = sum of 4 square embeddings +
  patch-position. Within-patch independence is harmless for deterministic
  copy/apply targets (same argument as V2's parallel transition head).
  Log-prob = sum of 4 categoricals.

### 2.2 Episode shape

```
HARNESS primes:  <root> <19 board tokens> <oracle> [val] [m1 m2 m3 m4] [budget]
LOOP:
  agent:   <probe> <19 board tokens> <go>
  harness: [val] [m1..m4] [budget]                 ← valid probe
           <invalid> [budget]                      ← invalid (budget burned anyway)
TERMINAL (forced when budget = 0):
  agent:   <answer> e2e4                            ← masked to legal root moves
```

- Oracle replies: value as a **discrete bin token** (128 bins over Q, 32 over
  D — token-space-only infra; Fourier injection is the fallback ablation),
  policy as **rank-ordered top-4 move tokens** (order carries the signal, no
  probability scalars), plus remaining budget as a bin token.
- Budget counts **unique oracle evaluations** (memoized). Invalid probes burn
  budget — sloppy imagination costs.
- Token math: probe round ≈ 27 tokens (vs ~76 for raw 68-token boards).
  K=64 ≈ 1.8k; K=128 ≈ 3.5k; K=256 ≈ 7k — fits the 8k native context.

### 2.3 Grammar masking

Position types are deterministic given the state machine, so the harness masks
the single output head per state: after `<probe>` exactly 16 patch slots then
3 meta slots then `<go>` (each slot masked to its token class); at verb
positions only `<probe>`/`<answer>` (only `<answer>` when budget = 0); after
`<answer>` only legal root moves. Cross-patch semantic validity (exactly one
king per side, no pawns on back ranks, side-not-to-move not in check) is
checked by the harness on the decoded board — not per-token maskable by
design.

## 3. Agent architecture

- Decoder-only, pre-RMSNorm, SwiGLU, RoPE with max_len ≥ 8k from day one.
  ~50–120M params. (The oracle carries the chess; the agent carries the
  deliberation — don't overbuild.)
- One token-embedding table over the full agent vocab (ops + ids + moves +
  controls + value bins). One output head over the same vocab.
- No value head, no policy head, no auxiliary heads on the agent. Values come
  from the oracle; the final move is just a token. One categorical per emitted
  token ⇒ clean GRPO log-probs.

## 4. Pretraining (two stages, deliberately minimal)

RL from random init is silent (every episode = invalid spam). But
over-pretraining buries L3. So: heavy on *mechanics*, light on *strategy*.

### Stage A — board literacy + long-context mechanics (heavy, fully programmatic)

Base tasks (short):
1. **Copy** — board in context → re-emit its 19 tokens. (Format cold.)
2. **Apply** — board + move token → emit the resulting board's 19 tokens.
   This installs the world model in-weights; a move changes 1–3 patches, the
   rest is copying (only 16 tokens of it — trivially learnable).
   EP/castling/promotion edge cases explicitly covered. Train to ≥99% exact
   boards.
3. **Read** — board + injected oracle reply → predict the value bin / top
   move. (Replies are meaningful.)

Plus **rare-patch augmentation**: mix synthetic random-legal boards into
copy/apply data so the long tail of the 28.5k patch vocab gets trained —
real games concentrate on a small subset of patch values.

**The long-context recipe** (the Stage-A objective is natively short and
always starts at index 0 — these four mechanisms fix that):

1. **Random RoPE offsets.** Every sample trains at positions `[r, r+len)` for
   random `r` in the full context window. The skill becomes
   position-invariant. Directly kills the "always same beginning indexes"
   problem.
2. **Packing into episode-shaped streams.** Concatenate dozens of independent
   tasks into 4–8k-token sequences under full causal attention (loss on each
   task's target tokens). The model learns to execute a local task while
   embedded in a long stream of distractor boards — which is exactly what an
   episode is.
3. **Long-range retrieval tasks** (the real fix; same generators, new
   queries):
   - *Recall:* stream of many boards … "re-emit the j-th board" from 3k
     tokens back. (Retrieval over distance.)
   - *Apply-at-distance:* "apply move m to the j-th board" where it sits far
     upstream → emit the resulting board. (World model on distant context.)
   - *Best-of-stream:* many (board, value-bin) pairs → emit the index (or
     board) with the best value. This is literally the `<answer>`-time
     aggregation skill.
4. **Curriculum on K** (Stage B and RL below): the episode distribution
   itself grows 4 → 8 → 16 → 32 → 128 probes; long-context behavior is
   introduced by the actual task, not extrapolation tricks. We train from
   scratch at native 8k — no length-extrapolation hacks needed at all.

### Stage B — format SFT (small, scripted, undertrained on purpose)

A few thousand full episodes from a **scripted teacher**: probe the boards
after the oracle's top-4 root moves, probe the top reply to each, answer the
best backed-up move. Mediocre by construction — its only job is the loop
(probe → read → probe → answer). Stop at >95% valid probes + coherent
answers. This checkpoint is the **L3 control**; archive it.

## 5. RL

- **Algorithm:** GRPO. G = 8–16 episodes per root position; advantage =
  group-normalized reward (per-root grouping absorbs position difficulty — no
  value network). Loss on agent-emitted tokens only; injected tokens masked.
  Small KL anchor to the Stage-B policy (grammar protection, not behavior
  pinning).
- **Reward (outcome-only, external, precomputed):**
  `R = Q_ref(root, a_agent) − max_a Q_ref(root, a)` (regret ≤ 0). Dense,
  graded, unhackable (the agent's probes can't influence it). "Did you beat
  the oracle's greedy move" is the same reward up to a per-position constant,
  which group normalization erases — the thesis objective falls out for free.
- **Shaping:** small `−ε` per invalid probe. **No per-probe cost** at first
  (fixed K does the rationing); a λ-per-probe cost is the L4 knob, later.
- **Corpus filtering:** ≥50% of root positions must have
  `argmax_a Q_ref ≠ oracle-greedy` (search-sensitive positions). Elsewhere
  reward variance is ~0 and GRPO learns nothing.
- **Sampling:** temperature ~1.0 during RL rollouts; T=0 at eval.

## 6. Evaluation protocol

- Elo ladders vs Stockfish at fixed UCI_Elo + **direct head-to-head** with an
  opening book (Stockfish-mediated Elo saturates — h2h is the ground truth;
  lesson learned from the V2-MCTS vs BT4 measurements).
- Always at **matched probe budget** vs the PUCT baseline (budget = unique
  oracle calls for both).
- Fixed eval suites: tactics positions, search-sensitive set (held out from
  the RL filter), quiet set (sanity: probing shouldn't *hurt*).
- Per-stage gates: Stage-B agent must produce well-formed episodes; first RL
  stage must beat L1 at K=16 before scaling K.

### Diagnostics to instrument from day one

- Probe validity rate; probe *coherence* (is the probed board plausibly
  related to the root — reachable, or a sensible counterfactual — vs noise?
  measurable cheaply: distance-in-pieces from root, reachability check).
- Budget usage histograms; probe-depth distribution; fraction of probes that
  are children of root vs deeper vs unreachable counterfactuals (the
  emergent-behavior readout — if "probe children of plausible root moves"
  emerges from total freedom, tree search was *rediscovered*).
- Reward curve split by search-sensitive vs quiet positions.

## 7. Risks (named honestly)

1. **In-weights world-model accuracy.** The agent imagines boards entirely
   in its own weights; ~99%/board still means occasional wrong probes deep in
   imagined lines. Self-contained patch probes bound the blast radius (an
   error corrupts one probe, not a chain) — monitor apply-accuracy and probe
   validity; the long-tail patch augmentation is the lever if rare
   configurations underperform.
2. **GRPO variance over multi-token composite actions.** A probe is one
   ~10-token action; episodes are hundreds of decisions sharing one scalar
   advantage. Expect to need larger groups / more episodes per update than
   text-RL folklore. This is the most likely place the schedule slips.
3. **Exploration collapse** ("probe one line, answer"). Mitigations: regret
   reward (quiet positions contribute no gradient), search-sensitive corpus
   filter, entropy bonus if needed.
4. **Oracle OOD on weird-but-legal probes.** Garbage values on absurd
   positions are reward-channel noise the agent must learn to route around;
   budget for it, don't try to fix it.
5. **L2 may not be reached.** PUCT at equal K with a good oracle is strong.
   Tying it is already a result; L1 + L3 can both succeed while L2 fails.

## 8. Milestones

| # | Deliverable | Notes |
|---|---|---|
| 0 | **Oracle**: train a small `ChessEncoder` (policy+WDL) with this branch's `train/train.py`, freeze, export | framework already here; pick ~20–60M |
| 1 | **Harness + grammar**: patch tokenizer (board ↔ 19 tokens), validity, budget, grammar masks | pure python, no model |
| 2 | **Stage-A generators + pretrain**: copy/apply/read + the four long-context mechanisms | programmatic data, unlimited |
| 3 | **Stage-B scripted SFT** → archive as L3 control | deliberately undertrained |
| 4 | **Reference labeler**: `Q_ref` for the RL corpus; corpus filter | Stockfish or big-MCTS-over-oracle |
| 5 | **GRPO loop** + diagnostics | start K=16 |
| 6 | **Eval ladder** L1→L3 (then L2, L4) | h2h harness + opening book |

## 9. Open questions (deliberately deferred)

- Value-bin granularity (128 vs Fourier injection) — ablate at Stage A.
- Oracle top-k in replies: k=4 starting point; k=0 (value-only) is the
  ablation that tests whether the agent's own prior suffices.
- Patch geometry: 2×2 (default) vs 1×4 half-ranks — cheap ablation.
- Patch-vocab implementation: V-a flat 28.5k vocab (default) vs V-b
  factorized 4×13-way heads.
- Folding the 3 meta tokens into one 2,080-way token (19 → 17/board).
- Per-probe cost λ and adaptive budgets (L4).
