# 11 — V2 Architecture Plan: Encoder-Latents + Parallel Board Decode

Branch: `v2-encoder-latents-arch`
Decisions locked: **k = 16 latents/board**, **absolute 64-square transition
head**, **one unified model trained jointly on the same game data as V1**.

## 0. Core principle: it is ONE model, trained like V1

V2 is **not** a new multi-stage regime. It is the *same training setup as V1*
(`train/train.py`): one `nn.Module`, one optimizer, the same ordered game data
(`parquet_files_decoder`, which has `game_id`/`ply`), one combined loss. The
encoder is a **submodule**, never trained on its own — it receives gradients
from the decoder's policy/value loss (backprop through the latents) and from
the transition loss, exactly like a vision encoder inside a VLM.

Only **two internal swaps** vs V1:

1. Raw 68 board tokens in the causal stream → **k=16 encoder latents** per board.
2. Causal autoregressive `board_head` → **parallel absolute transition head**
   (predicts all 64 squares + stm + castling + ep in one pass).

```
  board_i (68 tok) ──► Board Encoder (bidir, full attn) ──► z_i  (16 latents)
                                                              │
  causal stream:  [z_0, m1,wl1,d1, z_1, m2,wl2,d2, z_2, ...]  │ (z_i are inputs)
                          │  Causal Decoder (one clean causal mask, KV cache)
                          │  heads: policy / thinking_policy / wl / d / ctrl
                          ▼
  (z_i, move_{i+1}) ──► Transition Head (parallel, 64 sq + stm/castle/ep)
                          ▼
                     board_{i+1}  ──► Board Encoder ──► z_{i+1}  (engine-free)
```

Game ordering is used everywhere: the transition target is the *next ply's*
board; policy/value are over the ordered game. The loader emits ordered plies
exactly as V1's does.

## 1. Definition of "transition"

The **transition head** is the world-model / dynamics function
`T(z_t, move) → board_{t+1}`. It is the direct replacement for V1's causal
`board_head`. Its sole purpose: make rollouts **engine-free** — the model
imagines the next position itself instead of asking a chess library to apply
the move.

- Input: current board latents `z_t` + embedding of the move taken.
- Output (absolute, per user decision): logits for all **64 square** classes
  (13-way: empty + 6 white + 6 black) **+** stm (2) **+** castling (16) **+**
  en-passant (65). One parallel forward, **no causal mask, no autoregression.**
- Training target: the *actual* next board from the game (teacher-forced
  early; scheduled-sampled later so the model sees its own predictions).
- Justification it works here: chess transitions are deterministic
  (`p(next_board | board, move)` is a single mode), so independent per-square
  heads can represent the target exactly — the usual non-autoregressive
  multimodality failure does not apply.

## 2. The "encoder equivalent" in V2

- **As a module:** the V2 board encoder *is* today's `ChessEncoder` backbone
  (bidirectional, RoPE, RMSNorm) with `policy_head` removed and a learned
  latent-pooling head added: 16 learned query tokens cross-attend to the 68
  encoded board tokens → `z ∈ R^{16×E}`. The model we are training now in
  `chessencoder-debug` is the **direct seed/initializer** of this module.
- **As a baseline:** the standalone bidirectional FEN→move predictor is the
  **degenerate special case** of V2 (causal decoder + thinking disabled,
  policy read off `z_t` with zero history). The encoder-vs-decoder data
  experiment stays valid as the V2 no-search lower bound.

## 3. Will pretraining be the same?

**Same data, same "one model jointly" setup as V1.** What changes is *internal
heads/masking*, not the training regime:

| Aspect | V1 (`train/train.py`) | V2 |
|---|---|---|
| Data | ordered game parquets (FEN, move, wl, d, game_id, ply) | **same** |
| Model | one model, joint loss | **same: one model, joint loss** |
| Board objective | causal next-board-token CE (`board_head`, causal mask) | **parallel absolute transition CE** (64 sq + stm + castle + ep) |
| Move/value | prefix-mask pass, `policy/wl/d` heads | same heads, over the **causal latent stream**, single clean mask |
| Forward passes | 2 (causal + prefix) | 1 encoder (batched over the game's boards, parallel) + 1 causal decoder + 1 transition |
| Seq length | 68/ply | 16/ply (~4× shorter) |
| Loss | `move·5 + board·1 + wl·1 + d·1` | `move·w_m + transition·w_t + wl·w + d·w + ctrl` (re-tune weights) |

Not a different pipeline — a re-architected forward/loss inside the same loop.
`skip_board_prob` becomes "occasionally drop a board block / rely on prior
latents" (latent dropout) rather than dropping 68-token blocks.

**ChessFENS note:** it has no `game_id` → cannot form ordered plies → it
**cannot drive V2 joint training**. It remains useful only for (a) the current
encoder-only baseline, or (b) an *optional* encoder warm-start. Not on the
critical path.

## 4. Optional (not the plan): staged stabilizers

Joint end-to-end is the default. Reach for these *only if* instability appears:
- Latent collapse: stop-grad / lower-LR on encoder for early steps, or warm
  the encoder from a `chessencoder-debug` checkpoint.
- Exposure bias / rollout drift in engine-free play: scheduled sampling
  (mix ground-truth and predicted boards), periodic keyframe full-board
  prediction, transition↔move consistency aux loss.
These are guardrails, not phases.

## 5. Implementation phases (code comes after plan sign-off)

### Phase 0 — Scaffolding
- `chessdecoder/models/v2/` (`board_encoder.py`, `transition.py`,
  `latent_decoder.py`, `model_v2.py`), `train/config_v2.yaml`.
- Config knobs: `num_latents: 16`, `transition: absolute`,
  `latent_pool: perceiver`, `scheduled_sampling`, `keyframe_every`,
  loss weights. Reuse V1 vocab / `policy_index` / value buckets unchanged
  (keeps `eval/` and `rl/` compatible).

### Phase 1 — Board encoder → 16 latents
- Fork `ChessEncoder` backbone, drop `policy_head`, add Perceiver pooling
  (16 query tokens, 1–2 cross-attn layers).
- Add 2-D (file/rank) positional encoding + optional (Δfile,Δrank) attention
  bias on the 64 board tokens (fixes the spatial-awareness gap).
- Init from a current `encoder-*` checkpoint.

### Phase 2 — Parallel absolute transition head
- 64 square-query tokens (carry 2-D pos) + stm/castle/ep queries,
  cross-attend to `[z_t, move_emb]`, shared linear → 13-way per square + aux.
- Loss: per-square CE (+ optional up-weight of changed squares to counter the
  ~60/64 copy-square imbalance, since we chose absolute not delta) + aux CE.
- Metric of record: **rollout accuracy under scheduled sampling over N plies**,
  not teacher-forced next-board accuracy (the latter saturates trivially).

### Phase 3 — Causal latent decoder
- Plain causal transformer over `[z_0(×16), m1,wl1,d1, z_1(×16), ...]` +
  control tokens. Reuse `policy_head`, `thinking_policy_head`, `wl_head`,
  `d_head`, `FourierEncoder`. Standard KV cache (delete V1 block-boundary
  cache logic).

### Phase 4 — Data pipeline
- Loader emits ordered plies (as V1); boards encoded by the encoder at train
  time (batched, parallel across the game's boards); transition target =
  next ply's board (exact, from data).

### Phase 5 — Training
- One loop, one joint loss, same data as V1. Teacher-forced boards first;
  enable scheduled sampling once teacher-forced metrics are healthy.

### Phase 6 — Inference (engine-free)
- Python reference loop: decoder→move→transition→encode→append. Validate vs
  V1 strength on `eval/elo_eval.py`. Then port to C++ (simpler: one causal
  cache; encoder+transition fixed-shape/batchable; FEN→latent cache).

### Phase 7 — Eval / back-compat / tests
- V1 untouched on `main`. V2 tests: latent shapes; transition exactness on
  random legal games; rollout drift over 40+ plies; cached-vs-recompute
  decoder equivalence; self-play legality.

## 6. Open questions / risks
- k=16 sufficient vs the k=1 bottleneck you observed (k is locked at 16 for the
  first build; revisit only if strength plateaus).
- Engine-free rollout drift over long thinking traces (mitigations in §4).
- Absolute-head copy-square imbalance (mitigate via changed-square loss
  weighting; revisit delta only if absolute underperforms).
- Joint latent collapse (unlikely — latents pinned by policy+value+transition;
  guardrails in §4 if observed).

## 7. Non-goals
- VQ-discretized latent world model (documented fallback, not implemented).
- Changing move vocab or value-bucket scheme.
- Disturbing the running `chessencoder-debug` baseline on `main`.

---

# 12. Full Codebase Conversion Plan (train / finetune / rl / eval / export / cpp)

## 12.0 Token routing — the core mental model (answers the open questions)

The decoder operates on a **heterogeneous causal sequence of vectors**. The
encoder is a *preprocessor* that replaces each 68-token board with `k=16`
latent vectors; every non-board token stays a single embedding. The decoder
does not care where a vector came from — encoder output or embedding table —
they're all just R^d vectors in one causal stream.

**What goes through the encoder (→ 16 latents):**
- The full **board block = 68 tokens**: `start_pos + 64 squares + end_pos +
  castling + side_to_move`. So **stm and castling rights ARE part of the board
  block and DO go through the encoder** (your assumption is correct — they are
  squares 66/67 of the 68, encoded bidirectionally with the position).

**What does NOT go through the encoder (stays a single token in the causal
decoder stream):**
- `move` (root move, pv move, final move)
- `WL`, `D` value tokens
- control tokens: `start_think`, `end_var`, `continue_var`, `new_variation`,
  `end_think`, final-move marker, etc.

So a ply contributes: `[16 board latents]  [move emb]  [WL]  [D]` to the
decoder sequence; a thinking trace interleaves control-token embeddings and
more `[move][WL][D]` / `[16 latents]` blocks. A per-position **segment id**
(board-block vs token) tells the sequence assembler where to splice the 16
encoder latents vs an embedding-table lookup. RoPE positions run over the
flattened mixed sequence.

## 12.1 WL & D Fourier injection — preserved, and cleaner

WL/D are **not** board tokens, so they never touch the encoder. Fourier
injection is now a purely decoder-side op (exactly V1's mechanism, simpler
because there is no encoder interaction): at the WL/D positions in the decoder
input sequence, replace the placeholder embedding with
`FourierEncoder(continuous_value)` before the causal layers. `wl_head` /
`d_head` read decoder hidden states at the move / WL placeholder positions as
in V1 (`predict_move_and_value` logic ports directly). **Conversion check:**
unit-test that injected Fourier values at WL/D positions reproduce V1's
behaviour bit-for-bit on a fixed example.

## 12.2 Flash attention — YES, this is a primary win

V2 is flash-friendly across all three components, *because* the split removes
V1's dense custom `causal | same_block` prefix mask (the thing that forced the
slow masked SDPA path):

| Component | Mask | Flash path |
|---|---|---|
| Board encoder (68 tok, bidirectional) | none (boards are always full-length, **zero padding**) | ✅ fused/flash — already validated bit-identical in the encoder-mode benchmark |
| Perceiver pool (16 q × 68 kv cross-attn) | none | ✅ fused |
| Causal latent decoder | **pure causal** (no prefix/block mask anymore) | ✅ flash via `is_causal=True` |

**Hard implementation requirement:** never materialize a dense mask. Encoder/
pool pass `mask=None`; decoder passes `is_causal=True` (NOT a materialized
`[B,S,S]` causal tensor — that silently kills the flash kernel, the exact V1
mistake). For batched **variable-length games**, do not pad+mask (loses flash);
use FlashAttention **varlen** (`cu_seqlens`) or fixed-length sequence packing
so the decoder keeps the flash path with mixed game lengths.

## 12.3 Conversion phases (model → train → finetune → rl → eval → export → cpp)

**Phase A — Full V2 model (`chessdecoder/models/v2/model_v2.py`)**
- Reuse the validated `ChessEncoderV2` encoder+perceiver as the BoardEncoder
  (drop its policy head).
- `CausalLatentDecoder`: causal transformer over the mixed sequence, `is_causal`
  flash path, RoPE over flattened positions.
- Heads: `policy_head`, `thinking_policy_head`, `wl_head`, `d_head`
  (ValueHead, reuse V1), `transition_head` (parallel absolute: 64×13 squares +
  stm + castling + ep, conditioned on a board's 16 latents + chosen move emb).
- `FourierEncoder` reused for WL/D injection in the decoder stream.
- Unit tests: latent shapes; Fourier-injection parity vs V1; transition head
  exact on random legal games; flash kernel actually selected (assert via
  `torch.backends.cuda` / profiler).

**Phase B — Dataloader / sequence builder (`dataloader/`)**
- New builder emits, per game: ordered plies as `(board_68, move, wl, d)` plus
  thinking-trace tokens; tags each position with segment id (board-block vs
  token) and the targets: transition (next board), policy, wl/d buckets,
  thinking moves, control tokens.
- Boards batched and encoded in parallel (no autoregression across boards) →
  big throughput win vs V1's 68-token causal board loss.

**Phase C — Pretraining (`train/train.py` + `config.yaml`)**
- Replace V1's two-pass (causal board + prefix move/value) with: encode boards
  → assemble mixed sequence → single causal decoder pass → heads.
- Loss = `w_policy·policy + w_trans·transition + w_wl·wl + w_d·d + w_ctrl·ctrl`
  (retune weights; V1 used move 5 / board 1 / wl 1 / d 1).
- **Config updates:** add `num_latents`, `num_encoder_layers`,
  `num_decoder_layers`, `transition: absolute`, drop V1 board-gen knobs;
  keep `n_buckets`, `wl_sigma`, `value_hidden_size`, `num_fourier_freq`.
  Provide `config_v2.yaml`; keep V1 `config.yaml` untouched on `main`.
- Teacher-forced boards first; enable scheduled sampling once stable.

**Phase D — Finetune thinking variations (`finetune/`)**
- Sequence builder extended for `start_think … [variations] … end_think`.
- Engine-free rollout via the parallel transition head with scheduled
  sampling + periodic keyframe full-board prediction; consistency loss
  (predicted Δ must match emitted move).

**Phase E — RL / GRPO (`rl/`)**
- `rollout.py` / `sequence.py`: generation loop = decoder→move→transition→
  encode→append latents (engine-free). Log-prob/reward computation re-pointed
  to the new policy-head positions in the mixed sequence.
- Reuse reward/metric code; only the sequence/log-prob plumbing changes.

**Phase F — Eval (`eval/`)**
- `ChessDecoderV2.predict_move` / `predict_move_and_value` / `predict_move_n`
  ported (move via policy head off latents; value via wl/d heads + Fourier
  re-injection; multi-ply via cached per-board latents). `PytorchModelAdapter`
  + `elo_eval` work unchanged (only need `predict_move`).

**Phase G — Export + C++ (LAST, hardest)**
- TorchScript export of three modules (encoder, decoder, transition).
- C++ engine rewrite: simpler than V1 — decoder is plain causal so KV cache is
  textbook (delete V1 block-boundary invalidation); encoder + transition are
  fixed-shape, batchable; add a `FEN→latents` cache (positions repeat across
  MCTS/variations). Batched engine mask buffers largely disappear.

## 12.4 Order & branching
All on `v2-encoder-latents-arch`. Order: A→B→C (get pretraining green +
matching/beating V1 on `eval/elo`), then D, E in parallel, F alongside C, G
last. V1 stays intact on `main` as the comparison baseline throughout. The
running HP sweep informs Phase C defaults (optimizer/LR/wd/clip).

## 12.5 Implementation status (2026-05-17)

| Phase | Status | Artifact | Tests |
|---|---|---|---|
| A — full V2 model | ✅ done | `chessdecoder/models/v2/model_v2.py` (BoardEncoder, CausalLatentDecoder, TransitionHead, ChessDecoderV2) | `tests/test_v2_model.py` (6) |
| B — dataloader / sequence builder | ✅ done | `chessdecoder/dataloader/loader_v2.py` (`game_to_v2_arrays`, `assemble_decoder_inputs`, `ChessV2IterableDataset`) | `tests/test_v2_loader.py` (4) |
| C — pretraining | ✅ done | `chessdecoder/train/train_v2.py` + `config_v2.yaml` | `tests/test_v2_train_step.py` (1, incl. encoder-grad-flow) |
| F — eval (single-position) | ✅ done | reuses `PytorchModelAdapter`; `ChessDecoderV2.predict_move` matches the contract | `tests/test_v2_eval_adapter.py` (1) |
| D — finetune thinking (sequence splice) | ✅ done | `dataloader/sequence_v2.py` (`Seg`, `build_mixed_sequence`, `variation_plan_from_token_ids` — reuses `finetune/data.py`'s variation parser verbatim, only the representation changes) | `tests/test_v2_sequence.py` (2) |
| D — engine-free rollout + scheduled sampling + finetune loop | ✅ done | `model_v2.decode_transition`/`rollout_next`/`scheduled_sample_latents`; `finetune/loader_v2.py` + `train_v2.py` + `config_v2.yaml` | `tests/test_v2_rollout.py` (3, incl. bit-exact inverse), `tests/test_v2_finetune.py` (2, incl. encoder grad-flow + ss path) |
| D — real-data finetune validation | ⏳ blocked | needs variation parquets (`scripts/generate_variations.py`, MCTS) — not present locally | — |
| E — RL / GRPO plumbing | ⏳ next | `rl/rollout.py`/`sequence.py` decoder→move→transition→encode loop; log-prob re-pointing | — |
| F — eval (multi-ply history) | ⏳ next | `predict_move_n` port (cached per-board latents) | — |
| G — export + C++ | ⏳ last | TorchScript 3 modules; textbook causal KV cache; FEN→latents cache | — |

**Notes / divergences from the plan, decided during implementation:**
- **No en-passant target.** `fen_to_position_tokens` does not tokenize ep
  (only start_pos + 64 squares + end_pos + castling + stm), so the absolute
  transition head predicts 64 squares (13-way) + stm (2) + castling (16) —
  an exact inverse of the model's own tokenization, no ep head.
- **Flash path is structural, not masked.** Encoder/pool pass `mask=None`;
  decoder layers are `is_causal=True` with `mask=None` — a regression test
  asserts no dense `[B,S,S]` mask is ever materialized.
- **Fourier WL/D injection** reuses V1's `FourierEncoder` verbatim; a parity
  test asserts bit-identical output vs V1 for shared weights.
- **Regular per-ply layout** (pretraining): `[z_i(k) | move | wl | d]`,
  `L = k+3` positions/ply, closed-form position indices — the complex
  segment-id splice is only needed for Phase D thinking traces.
- **Default V2 size** at `config_v2.yaml` (10 enc + 8 dec layers, k=16) is
  ~150M; `num_decoder_layers` is the knob to match V1's ~116M for the
  strict comparison. The HP sweep's winning optimizer/LR/wd/clip will set
  the Phase-C training defaults (currently AdamW / 3e-4 / wd 0.1 / clip 10
  pending the final sweep aggregation).
- V1 (`models/model.py`, `train/train.py`, `config.yaml`) untouched;
  full CPU test suite still 140 passed, 0 regressions.

# 13. HP sweep results — 2⁴⁻¹ resolution-IV factorial (8 runs, 10k steps)

Design: A=LR{3e-4,1e-3} · B=opt{adamw,muon} · C=wd{0,0.1} ·
D=grad_clip{1,10}, D=ABC generator. Single seed, eff. batch 2048, ChessFENS,
held-out val, 60-game Stockfish match (UCI_Elo, temp 0) at step 10k.

**Data-quality caveat (reported, not hidden):** the per-step `*.csv`
`val_*`/`train_top*` columns are unreliable — several runs were created with
the 6-col header but appended with 9-col rows after an orchestrator resume, so
a header-keyed parse mis-maps them. Analysis therefore uses only the two
schema-stable signals: the clean separate `*_sf.csv` **Stockfish ELO** and
**`train_loss`** (column 2, identical in both schemas).

**Final ranking by ELO:**

| LR | opt | wd | clip | ELO | train_loss |
|---|---|---|---|---|---|
| 1e-3 | muon | 0 | 1 | **1152** | 1.776 |
| 3e-4 | adamw | 0.1 | 10 | **1144** | 1.809 |
| 3e-4 | adamw | 0 | 1 | 1113 | 1.941 |
| 1e-3 | muon | 0.1 | 10 | 1097 | 1.812 |
| 3e-4 | muon | 0 | 10 | 903 | 2.334 |
| 3e-4 | muon | 0.1 | 1 | 837 | 2.432 |
| 1e-3 | adamw | 0.1 | 1 | 684 | 2.716 |
| 1e-3 | adamw | 0 | 10 | 0 (collapsed) | 2.939 |

**Main effects** (Δ = high−low; res-IV → each main effect aliased with a
3-factor interaction, so read as directional, not exact):

| Factor | ΔELO | Δtrain_loss |
|---|---|---|
| LR 1e-3 vs 3e-4 | −266 | +0.18 |
| optimizer muon vs adamw | +262 | −0.26 |
| weight decay 0.1 vs 0 | +149 | −0.06 |
| grad clip 10 vs 1 | −161 | +0.01 |

**Interpretation.** The dominant structure is an **LR × optimizer
interaction**, not clean main effects: AdamW needs the *low* LR (3e-4: 1144 /
1113; at 1e-3 it collapses to 684 / 0), while Muon needs the *high* LR (1e-3:
1152 / 1097; at 3e-4 only 903 / 837 — classic Muon behaviour, it wants a
larger step). The two regimes are statistically tied (top two within 8 ELO ≪
the ~±60 ELO SE of a 60-game match). Weight-decay 0.1 helps modestly and
consistently; grad-clip 10's apparent harm is mostly confounded with the
LR×opt aliasing (the clip=10 cell is loaded with the bad AdamW@1e-3 corner).

**Phase-C defaults (decided):** keep **AdamW / 3e-4 / wd 0.1 / clip 10** —
already set in `config_v2.yaml`, the more standard/robust corner and the
codebase's native optimizer. **Muon / 1e-3 / wd 0 / clip 1** is the
co-leading alternative; worth a 2-seed confirmation if a faster-converging
optimizer is later wanted (Muon also reached the best train_loss). No
single-corner blowups in the chosen regime.
