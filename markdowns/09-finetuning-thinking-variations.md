# 09 - Finetuning with Thinking Variations

## 1. Overview

Thinking variations teach the model to reason about candidate moves before committing to a final choice. During finetuning, each position is augmented with MCTS search tree data: the top candidate moves, their principal variation (PV) lines, and win/draw/loss evaluations at each node.

The model learns to:
- Enumerate candidate moves in a structured `start_think ... end_think` block
- Predict continuation moves along each PV line (via `thinking_policy_head`)
- Predict the final best move after reviewing all variations (via `policy_head`)
- Estimate WDL values at every board position in the tree

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
- `board_68`: 68-token board representation (piece placement + side-to-move)
- `start_think` / `end_think`: delimiters for the thinking block
- `root_move_N`: candidate move at the root position
- `pv_move_N`: continuation move along the principal variation
- `wl_value` / `d_value`: placeholder tokens whose hidden states are used to regress WDL values
- `end_var`: marks the end of one variation line
- `final_move`: the chosen best move after deliberation

**Prediction heads:**
- `root_move_N` is predicted from the preceding `start_think` or `end_var` token -> `thinking_policy_head`
- `pv_move_N` is predicted from the side-to-move token of the preceding board -> `thinking_policy_head`
- `final_move` is predicted from `end_think` -> `policy_head` (the normal move head)

## 3. Plackett-Luce Variation Ordering

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

where `wdl_var = (W + L) - (W - L)^2` (see Section 4).

**Intuition:** In uncertain positions (high WDL variance), the top moves are closer in strength, so we increase tau to shuffle more aggressively. In clear positions (low WDL variance), the best move is decisive, so we keep the ordering more stable.

**Default values:** `tau_base = 0.3`, `tau_alpha = 1.0`

## 4. WDL Variance Derivation

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

## 5. Variable-Depth PV

Instead of always using `max_depth` nodes per variation, we sample a random depth uniformly from `[1, min(max_depth, available_nodes)]` for each variation independently.

This serves two purposes:
1. **Data augmentation**: The model sees the same positions with different amounts of lookahead, preventing overfitting to a fixed tree shape.
2. **Sequence length diversity**: Shorter PV lines produce shorter sequences, improving GPU utilization when batching.

The depth is sampled independently per variation per epoch, so the same position gets different truncations across training.

## 6. The `first_is_not_best` Metric

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

## 7. Dual Policy Heads

The model uses two separate linear heads for move prediction:

| Head | Predicts | From token | Used during |
|---|---|---|---|
| `policy_head` | Final move | `end_think` | Normal play + end of thinking |
| `thinking_policy_head` | Variation moves | `start_think`, `end_var`, board STM | Thinking block only |

Both heads share the same vocabulary and hidden dimension. At init, `thinking_policy_head` is cloned from the pretrained `policy_head` weights.

**Rationale:** Variation moves serve a different purpose than the final move. In the thinking block, the model generates "what-if" continuations that are strong but not necessarily the chosen line. The final move integrates information from all variations. Separate heads let these roles specialize independently.

## 8. Sequence Length Budget

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

The dataloader automatically retries with reduced `max_variations` / `max_depth` if a sequence exceeds `max_seq_len`.

## 9. Configuration Reference

All parameters are set in `finetune/config.yaml` under the `data:` section:

| Parameter | Default | Description |
|---|---|---|
| `variation_parquet_dir` | - | Directory containing MCTS variation parquet files |
| `pretrain_parquet_dir` | - | Directory containing normal game parquet files |
| `variation_ratio` | 0.2 | Fraction of batches that are thinking samples (vs normal pretraining) |
| `max_variations` | 3 | Maximum number of candidate moves per thinking block |
| `max_depth` | 5 | Maximum PV depth per variation (actual depth is sampled uniformly from [1, max_depth]) |
| `tau_base` | 0.3 | Base temperature for Plackett-Luce variation ordering |
| `tau_alpha` | 1.0 | WDL variance multiplier for adaptive temperature |
| `max_seq_len` | 1024 | Maximum token sequence length (longer sequences are retried or skipped) |
| `batch_size` | 16 | Training batch size |
| `num_workers` | 4 | DataLoader worker processes |

Loss weights (under `loss:`):

| Parameter | Default | Description |
|---|---|---|
| `final_move_weight` | 5.0 | Weight for final move cross-entropy loss |
| `thinking_move_weight` | 2.0 | Weight for thinking move cross-entropy loss |
| `board_weight` | 1.0 | Weight for board token generation loss |
| `wl_weight` | 1.0 | Weight for WL value regression loss |
| `d_weight` | 1.0 | Weight for D value regression loss |
