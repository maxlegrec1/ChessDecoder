# RL Training: GRPO for the Chess Decoder

This document describes the reinforcement learning pipeline that fine-tunes the Chess Decoder using Group Relative Policy Optimization (GRPO). The system generates thinking traces via the C++ inference engine, scores them with reward functions, and updates the model using a clipped policy gradient with KL regularization.

---

## 1. Overview

The training loop repeats the following cycle:

```
Sample B positions from pool
        |
Export model to TorchScript (subprocess)
        |
Generate B x G rollouts via C++ batched engine (subprocess)
        |
Score each rollout with composite reward
        |
Compute group-relative advantages
        |
Parse rollout tokens into model-ready tensors
        |
GRPO update (clipped surrogate + KL penalty)
        |
Checkpoint, log, evaluate
```

Each outer step produces `B x G` rollouts (default 64 x 10 = 640), computes rewards, and performs one round of mini-batch gradient updates.

---

## 2. Rollout Generation

Rollouts are generated in a **subprocess** to achieve complete GPU memory isolation. The C++ inference engine (libtorch) retains internal CUDA allocator references that survive Python-side cleanup, so subprocess termination is the only reliable way to free GPU memory before the training pass.

### Process

1. The PyTorch model is offloaded to CPU, freeing GPU memory entirely
2. A subprocess loads the exported TorchScript backbone + head weights
3. The `BatchedInferenceEngine` processes `B x G` FENs in chunks of `inference_batch_size`
4. Results are serialized to JSON and read by the parent process
5. The subprocess exits, releasing all GPU memory
6. The PyTorch model is moved back to GPU for training

### Configuration

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `rollout_batch_size` | 64 | Positions sampled per step |
| `group_size` | 10 | Rollouts per position |
| `inference_batch_size` | 64 | C++ engine batch size |
| `think_temperature` | 1.5 | Sampling temperature for thinking moves |
| `policy_temperature` | 1.5 | Sampling temperature for the final move |
| `board_temperature` | 0.0 | Deterministic board reconstruction |

### Output: RolloutResult

Each rollout produces:

```python
@dataclass
class RolloutResult:
    fen: str                              # starting position
    final_move: str                       # selected move (e.g. "e2e4")
    token_ids: list[int]                  # full thinking trace tokens
    wl_entries: list[tuple[int, float]]   # (position, win-loss value) pairs
    d_entries: list[tuple[int, float]]    # (position, draw value) pairs
    num_tokens: int                       # sequence length
```

The `token_ids` contain the full thinking trace: root board, `start_think`, variations (root moves, WL/D values, board reconstructions, PV moves, `end_var`), and `end_think`. The final move, WL, and D after `end_think` are stored separately.

---

## 3. Reward Functions

Three reward components are combined with configurable weights:

### Move Quality (weight: 1.0)

```
+1.0 if final_move == best_move (after castling normalization)
 0.0 otherwise
```

Binary signal: did the model find the Stockfish best move? This is the primary training objective.

### Format (weight: 0.5)

```
+1.0 if the thinking trace is well-formed
-0.5 if truncated (has start_think but no end_think)
 0.0 if structurally malformed
```

Validates:
- Presence of `start_think` and `end_think`
- At least one `end_var` between them
- All board blocks are exactly 68 tokens (complete `start_pos` ... `side_to_move` blocks)

This reward prevents the model from generating degenerate thinking traces.

### Coherence (weight: 0.3)

```
+1.0 if final_move was explored as a root move during thinking
 0.0 otherwise
```

Root moves are the first move token after `start_think` or `end_var` in each variation. This reward encourages the model to commit to moves it actually evaluated, rather than picking an unexplored move.

### Composite Score

```
total = 1.0 * move_quality + 0.5 * format + 0.3 * coherence
```

Maximum possible reward: 1.8. A rollout that finds the best move, has valid structure, and explored that move in variations gets the full score.

---

## 4. GRPO Algorithm

GRPO (Group Relative Policy Optimization) is a variant of PPO designed for language model RL. The key difference from standard PPO: advantages are computed **within groups** of completions for the same input, rather than from a learned value function.

### Group-Relative Advantages

For each position, `G` rollouts are generated with different random seeds (from temperature sampling). Advantages are normalized within each group:

```python
def compute_group_advantages(rewards):  # rewards: [B, G]
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + eps)
```

If all rollouts in a group receive the same reward, advantages are zero (no gradient signal). This naturally focuses training on positions where the model's behavior varies — positions where some rollouts succeed and others fail.

### Loss Function

The GRPO loss combines a clipped surrogate objective with a KL penalty:

```python
# Importance sampling ratio (current vs. rollout-time policy)
ratio = exp(log_prob_current - log_prob_old)

# Clipped surrogate (PPO-style)
surr1 = ratio * advantage
surr2 = clamp(ratio, 1-epsilon, 1+epsilon) * advantage
policy_loss = -min(surr1, surr2).mean()

# KL penalty against reference model (frozen finetuned checkpoint)
approx_kl = (log_prob_current - log_prob_ref).mean()
kl_loss = kl_coeff * approx_kl

loss = policy_loss + kl_loss
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `clip_epsilon` | 0.2 | Limits policy ratio to [0.8, 1.2] |
| `kl_coeff` | 0.05 | Weight of KL penalty |
| `max_kl` | 0.05 | Early stop if KL exceeds this |

### What Log-Probs Are Computed Over

Only move tokens contribute to the policy log-probability. Board tokens are generated deterministically (temperature=0) so they carry no policy gradient signal.

The per-sequence log-probability is the **sum** of log-probs at all move positions:
- **Thinking moves** (inside `start_think`...`end_think`): scored with `thinking_policy_head`
- **Final move** (after `end_think`): scored with `policy_head`

```python
seq_log_prob = sum(log P_think(move_i | context) for i in thinking_moves)
             + log P_policy(final_move | context)
```

This means the gradient flows through **all** move decisions in the thinking trace, not just the final move.

### Reference Model

The reference model is a frozen copy of the original finetuned checkpoint (not the latest RL checkpoint). This ensures the KL penalty measures divergence from the pre-RL policy, preventing the model from drifting too far from its initial capabilities.

### Early Stopping

If the approximate KL exceeds `max_kl` (0.05) during mini-batch iteration, the current outer step's training is terminated early. This prevents catastrophic policy updates.

---

## 5. Sequence Parsing

Rollout token sequences must be converted back into the tensor format expected by the model's prefix-pass forward. This is handled by `parse_rollout()`.

### State Machine

The parser walks the token sequence and identifies:
- **Board blocks**: 68-token chunks starting with `start_pos`, assigned sequential block IDs (0, 1, 2, ...)
- **Orphan tokens**: moves, WL/D values, structural tokens — each gets a unique block ID
- **Thinking moves**: move tokens inside the `start_think`...`end_think` region
- **Final move**: first move token after `end_think`

### Output Tensors

All tensors are padded to `max_seq_len`:

| Tensor | Dtype | Purpose |
|--------|-------|---------|
| `input_ids` | int64 | Token IDs (padded with PAD) |
| `block_id` | int64 | Board block grouping for prefix masking |
| `wl_positions` / `d_positions` | bool | Where value predictions occur |
| `wl_values` / `d_values` | float32 | Predicted values from C++ engine |
| `thinking_move_mask` | bool | Positions scored by `thinking_policy_head` |
| `final_move_mask` | bool | Position scored by `policy_head` |
| `move_token_ids` | int64 | Move sub-vocab indices (-1 for non-moves) |

### Prefix Forward Pass

During the GRPO update, only the **prefix pass** is run (not the causal pass). The prefix mask enables bidirectional attention within board blocks while maintaining causal ordering between blocks. WL/D values from the C++ engine are injected via Fourier encoding at their placeholder positions.

---

## 6. Training Loop Details

### Per-Step Lifecycle

```
1. Sample B=64 positions from training pool (50k positions)
2. Offload model + optimizer to CPU
3. Export model to TorchScript (subprocess)
4. Generate 640 rollouts via C++ batched engine (subprocess)
5. Reload model + optimizer to GPU
6. Compute rewards and group-relative advantages
7. Parse rollout sequences into batch tensors [640, max_seq_len]
8. Mini-batch GRPO update:
   - Iterate through 640 samples in mini-batches of 4
   - Accumulate gradients for 4 steps before optimizer.step()
   - Compute current, old, and reference log-probs
   - Clipped surrogate loss + KL penalty
   - Early stop if KL > 0.05
9. Log metrics (reward, KL, clip fraction, timing)
10. Checkpoint every 50 steps
11. Evaluate every 50 steps
```

### GPU Memory Management

The training loop alternates between two GPU-intensive phases that cannot coexist:

```
Phase A: C++ Engine (rollouts)     Phase B: PyTorch (training)
  - libtorch backbone                - PyTorch model
  - KV caches (B x max_seq_len)      - Optimizer states
  - Head weight buffers               - Gradient buffers
  - ~22 GB at B=64                    - ~12 GB with AMP
```

Offloading models to CPU between phases allows each phase to use nearly all available GPU memory.

### Optimizer

- **AdamW**: lr=1e-6, weight_decay=0.1
- **Warmup**: linear warmup for 20 steps, then constant LR
- **Gradient clipping**: max_norm=1.0
- **Mixed precision**: AMP with GradScaler
- **Gradient accumulation**: 4 mini-batches per optimizer step

### Position Sampling

Training positions are sampled from a pool of 50k positions loaded at startup from pretrain parquet files. Each step samples `rollout_batch_size` (64) random positions. Ground truth for each position includes the Stockfish `best_move`.

---

## 7. Evaluation

Every `eval_every` (50) steps, the model is evaluated on a fixed set of 200 positions using `evaluate_cpp_selfplay`. This measures:
- Move accuracy (does the model match Stockfish best move?)
- Thinking trace quality
- Policy entropy (thinking head and final head)

Two evaluation sets are used:
- **Variation positions**: from the variation training data
- **Pretrain positions**: from the pretraining data

---

## 8. Configuration Reference

Full configuration with defaults from `src/rl/config.yaml`:

```yaml
model:
  embed_dim: 1024
  num_heads: 16
  num_layers: 12
  max_seq_len: 4096
  d_ff: 1536
  n_buckets: 100
  wl_sigma: 0.4
  value_hidden_size: 256
  num_fourier_freq: 128

grpo:
  group_size: 10          # rollouts per position
  clip_epsilon: 0.2       # PPO clip range
  kl_coeff: 0.05          # KL penalty weight
  max_kl: 0.05            # early stop threshold

rollout:
  rollout_batch_size: 64  # positions per step
  inference_batch_size: 64
  think_temperature: 1.5
  policy_temperature: 1.5
  board_temperature: 0.0

training:
  learning_rate: 1.0e-6
  weight_decay: 0.1
  warmup_steps: 20
  grad_accum_steps: 4
  max_grad_norm: 1.0
  use_amp: true
  mini_batch_size: 3
  num_outer_steps: 500
  save_every: 50
  eval_every: 50

rewards:
  reward_move_quality_weight: 1.0
  reward_format_weight: 0.5
  reward_coherence_weight: 0.3

data:
  num_train_positions: 50000
  num_eval_positions: 200
```

---

## 9. File Reference

| File | Purpose |
|------|---------|
| `src/rl/train.py` | Main GRPO training loop |
| `src/rl/grpo.py` | `compute_group_advantages`, `grpo_loss` |
| `src/rl/log_probs.py` | Prefix forward, log-prob gathering, entropy |
| `src/rl/rollout.py` | Subprocess rollout generation, model export |
| `src/rl/rewards.py` | Reward functions and `CompositeReward` |
| `src/rl/sequence.py` | `parse_rollout`, `collate_rollouts` |
| `src/rl/config.py` | `GRPOConfig` dataclass, YAML loading |
| `src/rl/config.yaml` | Default configuration |
