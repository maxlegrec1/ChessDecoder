# 12 — V2 sequence reshape + single WDL position-evaluator head

Branch: `v2-encoder-latents-arch`. Supersedes the value-head parts of plan 11.

## 0. Motivation

V1/early-V2 predicted WL (=Q) and D from the **causal decoder** at the move/wl
slots. Two defects:

1. **Value leak.** The decoder is causal over `… wl_{i-1} d_{i-1} … wl_i …`;
   the wl/d slots Fourier-inject the *ground-truth* values, so a value head
   reading the decoder could copy/smooth the highly autocorrelated previous
   true values instead of evaluating the board. Worse on V2 (full games).
2. **Action-value indirection.** The targets we had were *played-move* values
   (`played_q/d`), conceptually the value of the *resulting* position, forcing
   awkward n+1 reasoning.

## 1. New per-ply sequence shape

```
[ z_i (16 board latents) ] [ value_i ] [ move_i ]      L = num_latents + 2 = 18
        positions i·18 … i·18+15        +16        +17
```

- `z_i` = encoder latents of board `i`.
- `value_i` = a single token whose embedding is `Fourier(Q_i) + Fourier(D_i)`
  (re-injected so later moves/thinking are eval-aware). Teacher-forced with
  the target value at train (same regime as the board); predicted value at
  inference.
- `move_i` = `tok_embedding(played_move_i)` (unchanged).

Closed-form read positions: `policy_pos[i] = i·18+16` (the value slot — the
move is predicted **conditioned on board + its evaluation**, eval-aware),
`move_pos[i] = i·18+17`.

## 2. The WDL head — encoder-side pure position evaluator

`WDLHead(z_i) -> 3 logits (win, draw, loss)`. One learned query
cross-attends over the 16 board latents (`PerceiverPool`, same pattern as
`TransitionHead`) → `Linear(E,3)`. **Reads the encoder latents `z_i`
directly, never the decoder** → structurally leak-free, flash-friendly, and a
*uniform position evaluator* applied to every board (real, variation node,
final) identically.

- **Target:** `WDL = [W, D, L]` with `W=(1−D+Q)/2`, `L=(1−D−Q)/2`,
  reconstructed from the position's **`orig_q` / `orig_d`** (the network's
  original eval of that position — a position quantity, unlike `played_q/d`
  which is an action value; corr 0.97 with `root_q`, 0% out-of-range, exact
  `W+D+L=1`). Source column is one constant — switch to `root_q/d` trivially.
- **Loss:** soft cross-entropy `−Σ target · log_softmax(logits)` over the 3
  classes, masked by valid plies. Q and D are *derived* from one joint
  distribution (`Q=W−L`, `D=D`) so they inherently "see each other"; no
  separate heads, no coupling hack.

## 3. Drift monitoring

Engine-free thinking chains transition predictions; drift risk is real.
**Metric of record: total board accuracy** (`board_total_acc` = all 64
squares + stm + castling exactly right) — watched to ensure imagined boards
don't degrade. Mitigations if it drifts: scheduled sampling, periodic
keyframe full-board re-grounding, transition↔move consistency loss (plan 11
§4).

## 4. Thinking traces in this format (design — implement later)

A trace is a tree flattened; every board (real or transition-imagined) is the
same `[16 latents][value][move]` unit, control tokens interleaved:

```
[z_root][val_root] start_think
  [m_r1][z_A][val_A][pv1][z_B][val_B]… end_var
  [m_r2][z_C][val_C]… end_var  new_variation …
end_think [m_final] …
```

`z_A = encode(transition_head(z_root, m_r1))` — boards imagined by the
transition head, **no chess engine**. `val_X` always = `WDLHead(z_X)` (same
rule everywhere). A small **control head** off the decoder emits
`continue/end_var/new_variation/end_think`; no explicit variation stack —
causal attention to the in-stream `z_root` handles "return to root". Value
targets for variation nodes come from MCTS `node["wdl"]`
(`finetune/data.py`). Inference decode loop:
`encode→WDL→inject→thinking_policy→transition→encode→…`.

## 5. Scope of this change

Implement now (pretraining only): shape reshape, `WDLHead`, loader target +
layout, `train_v2` loss/metrics, config, tests. Thinking-trace reshape (§4)
+ control head are a later finetune-phase change; `sequence_v2.py` /
`finetune/*_v2` keep working and are converted when Phase D is revisited.
The transition head and policy head are unchanged.
