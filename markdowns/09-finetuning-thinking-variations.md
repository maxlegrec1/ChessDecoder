# Finetuning with Thinking Variations

## 1. Overview

Thinking variations teach the model to reason about candidate moves before committing to a final choice. During finetuning, each position is augmented with MCTS search tree data: the top candidate moves, their principal variation (PV) lines, and win/draw/loss evaluations at each node.

The model learns to:
- Enumerate candidate moves in a structured `start_think ... end_think` block
- Predict continuation moves along each PV line (via `thinking_policy_head`)
- Predict the final best move after reviewing all variations (via `policy_head`)
- Estimate WDL values at every board position in the tree
- Decide when to end a variation (`end_var`) vs continue it (`continue_var`)
- Decide when to end thinking (`end_think`) vs start a new variation (`new_variation`)

This is analogous to a human player "thinking out loud" -- considering several lines before selecting a move.

## 2. Thinking Sequence Format

Each finetuning sample follows this token layout:

```
[board_68 tokens]  start_think
  root_move_1  wl_value  d_value  [board_68]  pv_move_1  wl_value  d_value  [board_68]  pv_move_2 ... end_var
  root_move_2  wl_value  d_value  [board_68]  pv_move_1 ... end_var
  ...
end_think  final_move  wl_value  d_value
```

**Token roles:**
- `board_68`: 68-token board representation (`start_pos` + 64 squares + `end_pos` + castling + STM)
- `start_think` / `end_think`: delimiters for the thinking block
- `root_move_N`: candidate move at the root position
- `pv_move_N`: continuation move along the principal variation
- `wl_value` / `d_value`: placeholder tokens whose embeddings are replaced with Fourier-encoded scalar values
- `end_var`: marks the end of one variation line
- `final_move`: the chosen best move after deliberation

**Prediction assignments (which head predicts what, from which position):**

| What | Predicted From | Head |
|------|---------------|------|
| `root_move_1` | `start_think` | `thinking_policy_head` |
| `root_move_N` (N>1) | `end_var` of previous variation | `thinking_policy_head` |
| `pv_move_N` | STM token of preceding board | `thinking_policy_head` |
| `final_move` | `end_think` | `policy_head` |
| WL values | Move token position (STM + 1) | `wl_head` |
| D values | `wl_value` position | `d_head` |
| Board tokens | Previous token (causal) | `board_head` |
| Continuation decisions | Board STM / `end_var` / `end_think` | `board_head` |

## 3. Board Head Decisions (Sub-Vocabulary Targets)

The board_head predicts over the 41-token board sub-vocabulary. At key decision points, the target is overridden from the natural shifted next-token:

### Target Overrides

| Position | Input Token | Board Target | Meaning |
|----------|------------|--------------|---------|
| STM of root board | `white_to_move` / `black_to_move` | `generic_move` | "A move comes next" (natural target would be a move token, not in board sub-vocab) |
| `start_think` | `start_think` | `generic_move` | First variation move follows |
| PV board STM | `white_to_move` / `black_to_move` | `continue_var` | PV continues (override) |
| PV board STM (last) | `white_to_move` / `black_to_move` | `end_var` | Variation ends (natural shifted target) |
| `end_var` (more variations) | `end_var` | `new_variation` | New variation follows (override) |
| `end_var` (last variation) | `end_var` | `end_think` | Thinking ends (natural shifted target) |
| `end_think` | `end_think` | `generic_move` | Final move follows |

### Natural vs Override Targets

Most board targets are **natural** shifted next-token predictions mapped to board sub-vocab indices via `full_idx_to_board_idx`. Only three positions require overrides:

1. **STM positions** (move prediction points): shifted target would be a move token (not in board sub-vocab) -> override to `generic_move`
2. **PV continuation STM**: shifted target would be the PV move -> override to `continue_var`
3. **`end_var` with more variations**: shifted target would be the next root move -> override to `new_variation`

Structural tokens (`wl_value`, `d_value`, `start_pos`) are **natural** shifted targets -- no overrides needed. `end_var` and `end_think` are also natural when they appear as shifted next-tokens.

### Unified Board Loss

All these decisions are trained via a single cross-entropy loss over the board sub-vocabulary:

```python
IGNORE_INDEX = -100
board_mask = (board_target_ids != IGNORE_INDEX) & (~pre_first_move_mask)
ce_board = board_ce_fn(board_logits.view(-1, board_vocab_size), board_target_ids.view(-1))
board_loss = (ce_board * board_mask.float()).sum() / (board_mask.sum() + 1e-8)
```

This ensures `continue_var` and `end_var` receive equal gradient signal (both are just different board sub-vocab indices at the same positions), eliminating the previous gradient imbalance problem.

## 4. Plackett-Luce Variation Ordering

### Motivation

If variations are always presented in visit-count order (strongest first), the model can learn a shortcut: "the first move is always best." To prevent this, we stochastically reorder variations using the Plackett-Luce model, implemented via the Gumbel-max trick.

### Algorithm

Given `N` candidate moves with visit fractions `p_1, ..., p_N`:

1. Compute perceived utilities: `u_i = p_i / tau + g_i`, where `g_i ~ Gumbel(0, 1)`
2. Sort by descending `u_i` to obtain the presentation order
3. Record the ranking tuple (e.g., `(2, 1, 3)` means the second-strongest move is presented first)

The Gumbel noise ensures that stronger moves are *usually* first but not *always*, forcing the model to read the actual variation content rather than relying on position.

### Adaptive Temperature

The temperature `tau` controls how much noise is injected. We make it adaptive based on WDL variance:

```
tau = tau_base * (1 + tau_alpha * wdl_var)
```

where `wdl_var = (W + L) - (W - L)^2` (see Section 5).

**Intuition:** In uncertain positions (high WDL variance), the top moves are closer in strength, so we increase tau to shuffle more aggressively. In clear positions (low WDL variance), the best move is decisive, so we keep the ordering more stable.

**Default values:** `tau_base = 0.3`, `tau_alpha = 1.0`

## 5. WDL Variance Derivation

The root position has win/draw/loss probabilities `(W, D, L)` with `W + D + L = 1`. We model the outcome as a random variable `X` taking values `{+1, 0, -1}` with probabilities `{W, D, L}`.

```
E[X]   = W - L
E[X^2] = W + L          (since (+1)^2 = (-1)^2 = 1, 0^2 = 0)
Var[X]  = E[X^2] - (E[X])^2
        = (W + L) - (W - L)^2
```

**Numerical examples:**

| Position type | W | D | L | Var[X] | tau (base=0.3, alpha=1.0) |
|---|---|---|---|---|---|
| Decisive (white wins) | 0.90 | 0.08 | 0.02 | 0.148 | 0.344 |
| Balanced (drawish) | 0.15 | 0.70 | 0.15 | 0.300 | 0.390 |
| Sharp (unclear) | 0.45 | 0.10 | 0.45 | 0.900 | 0.570 |
| Equal but decisive | 0.50 | 0.00 | 0.50 | 1.000 | 0.600 |

## 6. Variable-Depth PV

Instead of always using `max_depth` nodes per variation, we sample a random depth uniformly from `[1, min(max_depth, available_nodes)]` for each variation independently.

This serves two purposes:
1. **Data augmentation**: The model sees the same positions with different amounts of lookahead, preventing overfitting to a fixed tree shape.
2. **Sequence length diversity**: Shorter PV lines produce shorter sequences, improving GPU utilization when batching.

The depth is sampled independently per variation per epoch, so the same position gets different truncations across training.

## 7. The `first_is_not_best` Metric

### Definition

`first_is_not_best` is a boolean flag that is `True` when the first variation's root move (after Plackett-Luce reordering) differs from the final best move (`mcts_action`).

In other words, the model is shown a "wrong" move first and must still identify the correct final move after reading all variations.

### Why It Matters

This metric isolates the model's "mind-changing" ability. When `first_is_not_best = True`:
- The model cannot succeed by simply copying the first variation's move
- It must genuinely process the variation content to determine the best move
- Accuracy on these samples (`final_move_acc_reordered`) measures true reasoning

### Training Metric

During training, we compute:
```
final_move_acc_reordered = correct_predictions[reorder_mask] / reorder_mask.sum()
```
where `reorder_mask = move_mask & first_is_not_best`.

This is logged to wandb as `train/final_move_acc_reordered` alongside `train/n_reordered_samples` (count of such samples in the batch).

## 8. Dual Policy Heads

The model uses two separate linear heads for move prediction:

| Head | Output Size | Predicts | From Token | Used During |
|---|---|---|---|---|
| `policy_head` | 1924 (move sub-vocab) | Final move | `end_think` or STM | Normal play + end of thinking |
| `thinking_policy_head` | 1924 (move sub-vocab) | Variation moves | `start_think`, `end_var`, board STM | Thinking block only |

Both heads output logits over the same 1924-token move sub-vocabulary. At initialization, `thinking_policy_head` is cloned from the pretrained `policy_head` weights via `migrate_state_dict()`.

**Rationale:** Variation moves serve a different purpose than the final move. In the thinking block, the model generates "what-if" continuations that are strong but not necessarily the chosen line. The final move integrates information from all variations. Separate heads let these roles specialize independently.

## 9. Two-Pass Training Architecture

Same as pretraining, finetuning uses two forward passes per batch:

Fourier inputs are prepared once (ground-truth WL/D discretized to nearest bucket centers) and injected into **both** passes.

### Pass 1: Causal (Board Generation)

```python
h_causal = model(input_ids, mask_type="causal",
                 wl_values=wl_fourier_input, d_values=d_fourier_input,
                 wl_positions=wl_positions, d_positions=d_positions)
board_logits = model.board_head(h_causal)  # [B, S, 41]
```

Standard autoregressive attention with Fourier-encoded WL/D values at placeholder positions. The board_head predicts all board-related tokens including structural tokens (`wl_value`, `d_value`, `start_pos`) and decision tokens (`generic_move`, `continue_var`, `end_var`, `new_variation`, `end_think`).

### Pass 2: Prefix (Move + Value Prediction)

```python
h_prefix = model(input_ids, mask_type="prefix", block_id=block_id,
                 wl_values=wl_fourier_input, d_values=d_fourier_input,
                 wl_positions=wl_positions, d_positions=d_positions)

final_logits = model.policy_head(h_prefix)              # [B, S, 1924]
think_logits = model.thinking_policy_head(h_prefix)      # [B, S, 1924]
wl_logits = model.wl_head(h_at_move)                     # [N, 100]
d_logits = model.d_head(h_at_wl)                         # [M, 100]
```

Bidirectional within board blocks (via `block_id`). Same Fourier-encoded WL/D values are injected at placeholder positions.

### Board Mask Construction

```python
IGNORE_INDEX = -100
board_mask = board_target_ids != IGNORE_INDEX

# Exclude positions before first move (no useful causal context for initial board)
any_move = move_mask | thinking_move_mask
first_move_idx = any_move.int().argmax(dim=1)
has_moves = any_move.any(dim=1)
first_move_idx[~has_moves] = any_move.size(1)
indices = torch.arange(any_move.size(1), device=device).unsqueeze(0)
pre_first_move_mask = indices < first_move_idx.unsqueeze(1)
board_mask = board_mask & (~pre_first_move_mask)
```

For thinking sequences, `first_move_idx` points to the `start_think` position (where the first thinking move is predicted), so the entire root board (positions 0-67) is excluded from board loss. The `start_think` position itself IS included.

## 10. Loss Functions

### Board Loss (Unified)

```python
board_ce_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='none')
ce_board = board_ce_fn(board_logits.view(-1, board_vocab_size), board_target_ids.view(-1))
ce_board = ce_board.view(board_target_ids.shape)
board_loss = (ce_board * board_mask.float()).sum() / (board_mask.sum() + 1e-8)
```

Single loss covering all board_head predictions. No separate losses for structural or signal tokens.

### Final Move Loss

```python
ce_final = move_ce_fn(final_logits.view(-1, move_vocab_size), move_target_ids.view(-1))
ce_final = ce_final.view(move_target_ids.shape)
final_move_loss = (ce_final * move_mask.float()).sum() / (move_mask.sum() + 1e-8)
```

Cross-entropy over the 1924-token move sub-vocabulary at `end_think` positions (where `move_mask` is True).

### Thinking Move Loss

```python
ce_think = move_ce_fn(think_logits.view(-1, move_vocab_size), move_target_ids.view(-1))
ce_think = ce_think.view(move_target_ids.shape)
thinking_move_loss = (ce_think * thinking_move_mask.float()).sum() / (thinking_move_mask.sum() + 1e-8)
```

Cross-entropy over the 1924-token move sub-vocabulary at `start_think`, `end_var`, and PV board STM positions (where `thinking_move_mask` is True).

### WL/D Losses

Same soft bucket cross-entropy as pretraining:

```python
wl_loss = soft_bucket_loss(wl_logits, wl_gt_flat, model.wl_bucket_centers, wl_valid_flat)
d_loss = soft_bucket_loss(d_logits, d_gt_flat, model.d_bucket_centers, d_valid_flat)
```

WL predicted at move token positions, D predicted at `wl_value` positions.

### Total Loss

```python
total_loss = (
    final_move_weight * final_move_loss +      # 5.0
    thinking_move_weight * thinking_move_loss + # 2.0
    board_weight * board_loss +                 # 1.0
    wl_weight * wl_loss +                       # 1.0
    d_weight * d_loss                           # 1.0
)
```

## 11. Annotated Example Sequence

### Root position with 2 variations, depth 2 and 1

```
Pos  Input Token         board_target_ids    move_target_ids   Masks
---  -----------         ----------------    ---------------   -----
0    start_pos           IGNORE (-100)       IGNORE            (excluded by pre_first_move)
1    white_rook          IGNORE (-100)       IGNORE            (excluded by pre_first_move)
...  (board tokens)      IGNORE (-100)       IGNORE            (excluded by pre_first_move)
66   KQkq                IGNORE (-100)       IGNORE            (excluded by pre_first_move)
67   white_to_move       IGNORE (-100)       IGNORE            (excluded by pre_first_move)
68   start_think         generic_move [O]    root_move_1       board_mask, thinking_move_mask
69   root_move_1         wl_value [N]        IGNORE            board_mask
70   wl_value            d_value [N]         IGNORE            board_mask, wl_positions
71   d_value             start_pos [N]       IGNORE            board_mask, d_positions
72   start_pos           square_a1 [N]       IGNORE            board_mask
...  (PV board1 tokens)  (shifted) [N]       IGNORE            board_mask
138  castling            stm_token [N]       IGNORE            board_mask
139  stm_token           continue_var [O]    pv_move_1         board_mask, thinking_move_mask
140  pv_move_1           wl_value [N]        IGNORE            board_mask
141  wl_value            d_value [N]         IGNORE            board_mask, wl_positions
142  d_value             start_pos [N]       IGNORE            board_mask, d_positions
143  start_pos           square_a1 [N]       IGNORE            board_mask
...  (PV board2 tokens)  (shifted) [N]       IGNORE            board_mask
209  castling            stm_token [N]       IGNORE            board_mask
210  stm_token           end_var [N]         IGNORE            board_mask
211  end_var             new_variation [O]   root_move_2       board_mask, thinking_move_mask
212  root_move_2         wl_value [N]        IGNORE            board_mask
213  wl_value            d_value [N]         IGNORE            board_mask, wl_positions
214  d_value             start_pos [N]       IGNORE            board_mask, d_positions
215  start_pos           square_a1 [N]       IGNORE            board_mask
...  (PV board3 tokens)  (shifted) [N]       IGNORE            board_mask
281  castling            stm_token [N]       IGNORE            board_mask
282  stm_token           end_var [N]         IGNORE            board_mask
283  end_var             end_think [N]       IGNORE            board_mask
284  end_think           generic_move [O]    final_move        board_mask, move_mask
285  final_move          wl_value [N]        IGNORE            board_mask
286  wl_value            d_value [N]         IGNORE            board_mask, wl_positions
287  d_value             IGNORE              IGNORE            (last position)
```

**Legend:** `[N]` = natural shifted target, `[O]` = override target.

### Key observations:

1. **Positions 0-67** (root board): excluded from board_mask by `pre_first_move_mask`
2. **Position 68** (`start_think`): first position in board_mask. Board target = `generic_move` (override). Move target = root_move_1 via thinking_policy_head.
3. **Position 139** (PV board1 STM): Board target = `continue_var` (override, signals PV continues). Move target = pv_move_1 via thinking_policy_head.
4. **Position 210** (PV board2 STM): Board target = `end_var` (natural shifted target, since `end_var` is the next input token).
5. **Position 211** (`end_var`): Board target = `new_variation` (override). Move target = root_move_2 via thinking_policy_head.
6. **Position 283** (`end_var`): Board target = `end_think` (natural shifted target).
7. **Position 284** (`end_think`): Board target = `generic_move` (override). Move target = final_move via policy_head.

## 12. Attention Masks on Variation Sequences

### Block Assignments

```
Block 0: root board (pos 0-67)
Orphan:  start_think (68), root_move_1 (69), wl (70), d (71)
Block 1: PV board1 (pos 72-139)
Orphan:  pv_move_1 (140), wl (141), d (142)
Block 2: PV board2 (pos 143-210)
Orphan:  end_var (211), root_move_2 (212), wl (213), d (214)
Block 3: PV board3 (pos 215-282)
Orphan:  end_var (283), end_think (284), final_move (285), wl (286), d (287)
```

### Causal Mask (Pass 1 -- board_head)

Standard lower-triangular. Each position sees all previous positions. Used for autoregressive board token generation and continuation/termination decisions.

### Prefix Mask (Pass 2 -- move/value heads)

```
Prefix mask = causal_mask | same_block_mask
```

**Example attention patterns:**

| Position | Input | Block | Can See |
|----------|-------|-------|---------|
| 139 (STM of board1) | stm_token | Block 1 | pos 0-139 (causal) + pos 72-139 (block 1 bidirectional) |
| 72 (start_pos of board1) | start_pos | Block 1 | pos 0-72 (causal) + pos 72-139 (block 1 bidirectional) = pos 0-139 |
| 140 (pv_move_1) | pv_move_1 | Orphan | pos 0-140 (causal only, no same-block peers) |
| 284 (end_think) | end_think | Orphan | pos 0-284 (causal only) |

This ensures that within each board block, all 68 tokens see each other bidirectionally (for rich board representation), while orphan tokens (moves, structural) only see their causal past.

## 13. Sequence Length Budget

A thinking variation sequence has length:

```
L = 68 (root board)
  + 1 (start_think)
  + sum over variations: [1 (root_move) + sum over nodes: [2 (wl+d) + 68 (board) + 1? (pv_move)] + 1 (end_var)]
  + 1 (end_think)
  + 1 (final_move)
  + 2 (final wl + d)
```

Per variation with `d` nodes: `1 + d * 71 + (d-1) + 1 = 72d + 1` tokens (approximately).

**Fit rates at max_seq_len=1024:**

| max_variations | max_depth | Approx tokens | Fits? |
|---|---|---|---|
| 3 | 5 | 68 + 3 + 3*(72*5+1) + 3 = 1155 | Marginal (with random depth, usually fits) |
| 3 | 4 | 68 + 3 + 3*(72*4+1) + 3 = 939 | Yes |
| 2 | 5 | 68 + 3 + 2*(72*5+1) + 3 = 796 | Yes |
| 3 | 3 | 68 + 3 + 3*(72*3+1) + 3 = 723 | Yes |

The dataloader automatically retries with reduced `max_variations` / `max_depth` if a sequence exceeds `max_seq_len`:

```python
if len(ids) > self.max_seq_len:
    for reduced_vars in range(self.max_variations - 1, 0, -1):
        for reduced_depth in range(self.max_depth, 0, -1):
            ids, ... = variation_to_token_ids(row, max_variations=reduced_vars, max_depth=reduced_depth, ...)
            if len(ids) <= self.max_seq_len:
                break
        if len(ids) <= self.max_seq_len:
            break
```

## 14. Checkpoint Migration

When loading a pretrained checkpoint for finetuning, `migrate_state_dict()` handles:

1. **Token embedding expansion**: Zero-pad rows if vocabulary grew (e.g., 1967 -> 1968)
2. **Board head migration**: Extract 41 rows from old full-vocab head using `board_idx_to_full_idx`. New tokens (e.g., `generic_move`) that didn't exist in the old vocab are zero-initialized.
3. **Policy head migration**: Extract 1924 rows from old full-vocab head using `move_idx_to_full_idx`.
4. **Thinking policy head cloning**: If `thinking_policy_head` not in checkpoint, clone from `policy_head`.

```python
def migrate_state_dict(state_dict):
    # 1. Expand tok_embedding
    if t.shape[0] < vocab_size:
        state_dict["tok_embedding.weight"] = torch.cat([t, zeros], dim=0)

    # 2. board_head: extract board sub-vocab rows
    if old_w.shape[0] > board_vocab_size:
        for i, full_idx in enumerate(board_idx_to_full_idx):
            if full_idx < old_vocab_sz:
                new_w[i] = old_w[full_idx]  # Known token
            # else: zero-initialized (new token)

    # 3. policy_head / thinking_policy_head: extract move sub-vocab rows
    for head in ["policy_head", "thinking_policy_head"]:
        if old_w.shape[0] > move_vocab_size:
            for i, full_idx in enumerate(move_idx_to_full_idx):
                if full_idx < old_vocab_sz:
                    new_w[i] = old_w[full_idx]

    # 4. Clone policy_head -> thinking_policy_head
    if "thinking_policy_head.weight" not in state_dict:
        state_dict["thinking_policy_head.weight"] = state_dict["policy_head.weight"].clone()
```

## 15. Data Pipeline Details

### File: `finetune/data.py` -- `variation_to_token_ids()`

Converts a parquet row with MCTS variation data into a finetuning token sequence.

**Input**: Row with `fen`, `variations` (JSON), `mcts_action`, `win`, `draw`, `loss`.

**Each variation dict contains**:
- `root_move`: candidate move (standard UCI)
- `visit_count`, `visit_fraction`, `prior`: MCTS statistics
- `nodes`: list of `{fen, move, wdl, visit_count}` dicts (the PV line)

**Processing steps**:
1. Sort variations by visit_count descending, cap at `max_variations`
2. Apply Plackett-Luce reordering (Gumbel-max trick)
3. For each variation:
   - Sample random depth from `[1, min(max_depth, available_nodes)]`
   - Emit root_move + nodes (wl + d + board + pv_move for each node)
   - Emit `end_var`
4. Emit `end_think` + final_move + wl + d
5. Convert all tokens to IDs

**Returns**: `ids`, `thinking_move_data`, `final_move_data`, `value_data`, `block_boundaries`, `ranking`, `first_is_not_best`

### File: `finetune/loader.py` -- `FinetuneIterableDataset`

Mixes normal pretraining data with thinking variation data.

**Mixing strategy**:
```python
while True:
    if random.random() < variation_ratio:  # default 0.2
        sample = next(variation_iter)       # 20% thinking
    else:
        sample = next(pretrain_iter)        # 80% normal
    yield sample
```

The variation iterator restarts when exhausted, with a `variation_epoch` counter tracking recycling.

**`_build_pretrain_tensors()`**: Identical to `ChessIterableDataset`, with additional zero/False fields for thinking-specific masks.

**`_build_variation_tensors()`**:
1. Shifted board targets via `full_idx_to_board_idx.get(full_idx, IGNORE_INDEX)`
2. Thinking move targets + board target overrides based on input token type
3. Final move target at `end_think` position
4. Value targets at `wl_value`/`d_value` positions
5. Block IDs for prefix masking

## 16. Metrics (Logged to WandB)

### Loss Metrics

| Metric | Description |
|--------|-------------|
| `train/total_loss` | Combined weighted loss |
| `train/final_move_loss` | CE loss for final move (policy_head, move sub-vocab) |
| `train/thinking_move_loss` | CE loss for thinking moves (thinking_policy_head, move sub-vocab) |
| `train/board_loss` | Unified CE loss for board generation (board_head, board sub-vocab) |
| `train/wl_loss` | Soft bucket CE for WL prediction |
| `train/d_loss` | Soft bucket CE for D prediction |

### Accuracy Metrics

| Metric | Description |
|--------|-------------|
| `train/final_move_acc` | Final move accuracy (policy_head at move_mask positions) |
| `train/thinking_move_acc` | Thinking move accuracy (thinking_policy_head at thinking_move_mask positions) |
| `train/board_acc` | Per-token board accuracy (board_head at board_mask positions) |
| `train/board_total_acc` | Per-block board accuracy (all tokens in block correct) |
| `train/board_square_acc` | Per-block accuracy on 64 square tokens only |
| `train/board_castling_acc` | Accuracy on castling token prediction |
| `train/board_stm_acc` | Accuracy on side-to-move prediction |
| `train/end_var_acc` | Accuracy at positions where target is `end_var` |
| `train/end_think_acc` | Accuracy at positions where target is `end_think` |
| `train/continue_var_acc` | Accuracy at PV continuation positions (target is `continue_var`) |
| `train/new_variation_acc` | Accuracy at new variation positions (target is `new_variation`) |
| `train/final_move_acc_reordered` | Final move accuracy on `first_is_not_best` samples |
| `train/wl_mae` | WL mean absolute error (expected value from softmax) |
| `train/d_mae` | D mean absolute error |

### Diagnostic Metrics

| Metric | Description |
|--------|-------------|
| `train/batch_thinking_samples` | Number of thinking samples in batch |
| `train/batch_normal_samples` | Number of normal (pretrain) samples in batch |
| `train/n_reordered_samples` | Number of `first_is_not_best` samples in batch |
| `train/variation_epoch` | How many times variation data has been recycled |
| `train/lr` | Current learning rate |
| `train/epoch` | Current epoch |
| `train/step` | Current training step |

## 17. Thinking Inference

See `scripts/think.py` and doc `06-evaluation-and-inference.md` for the inference state machine that mirrors this training format.

The inference loop follows the same state transitions:
```
MOVE -> WL_D -> BOARD -> AFTER_BOARD -> AFTER_END_VAR -> FINAL
```

At `AFTER_BOARD`, the board_head samples from its 41-token sub-vocabulary. The sampled index is mapped back to the full vocabulary via `board_idx_to_full_idx` to determine the token:
- If `end_var`: transition to AFTER_END_VAR
- Otherwise (`continue_var` or other): transition to MOVE (PV continues), don't emit the token

At `AFTER_END_VAR`:
- If `end_think`: transition to FINAL
- Otherwise (`new_variation` or other): transition to MOVE (new variation)

Move heads output 1924 logits, mapped back via `move_idx_to_full_idx`.

## 18. Configuration Reference

All parameters in `finetune/config.yaml`:

### Data Parameters

| Parameter | Default | Description |
|---|---|---|
| `pretrain_parquet_dir` | - | Directory containing normal game parquet files |
| `variation_parquet_dir` | - | Directory containing MCTS variation parquet files |
| `variation_ratio` | 0.2 | Fraction of samples that are thinking (vs normal pretraining) |
| `max_variations` | 3 | Maximum number of candidate moves per thinking block |
| `max_depth` | 5 | Maximum PV depth per variation (actual depth sampled from [1, max_depth]) |
| `tau_base` | 0.3 | Base temperature for Plackett-Luce variation ordering |
| `tau_alpha` | 1.0 | WDL variance multiplier for adaptive temperature |
| `max_seq_len` | 1024 | Maximum token sequence length |
| `batch_size` | 16 | Training batch size |
| `num_workers` | 4 | DataLoader worker processes |

### Model Parameters

| Parameter | Default | Description |
|---|---|---|
| `embed_dim` | 1024 | Hidden dimension |
| `num_heads` | 16 | Attention heads |
| `num_layers` | 12 | Transformer layers |
| `max_seq_len` | 1024 | Maximum sequence length |
| `d_ff` | 1536 | Feed-forward intermediate dimension |
| `n_buckets` | 100 | Number of value buckets |
| `wl_sigma` | 0.4 | WL bucket Gaussian CDF spread |
| `value_hidden_size` | 256 | Value head hidden dimension |
| `num_fourier_freq` | 128 | Fourier encoder frequencies |

### Training Parameters

| Parameter | Default | Description |
|---|---|---|
| `learning_rate` | 3e-5 | AdamW learning rate |
| `weight_decay` | 0.1 | AdamW weight decay |
| `gradient_accumulation_steps` | 8 | Steps before optimizer update |
| `num_epochs` | 5 | Total training epochs |
| `warmup_steps` | 500 | Linear LR warmup steps |
| `use_amp` | true | Mixed precision training |
| `pretrain_checkpoint` | - | Path to pretrained checkpoint |
| `save_every_n_steps` | 10000 | Checkpoint save frequency |

### Loss Weights

| Parameter | Default | Description |
|---|---|---|
| `final_move_weight` | 5.0 | Weight for final move cross-entropy loss |
| `thinking_move_weight` | 2.0 | Weight for thinking move cross-entropy loss |
| `board_weight` | 1.0 | Weight for unified board generation loss |
| `wl_weight` | 1.0 | Weight for WL soft bucket loss |
| `d_weight` | 1.0 | Weight for D soft bucket loss |
