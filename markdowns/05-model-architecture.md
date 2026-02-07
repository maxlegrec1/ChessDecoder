# Model Architecture

## ChessDecoder Architecture

### Overview Diagram

```
Input: Token IDs [B, S]
         |
         v
+-----------------------------+
|    Token Embedding          |  1968 -> 1024 dim
|    (full vocab_size)        |  (RoPE handles positions)
+-----------------------------+
         |
    [Override WL/D positions with Fourier features if prefix mode]
         |
         v
+-----------------------------+
|   Transformer Layer x12     |
|   +------------------------+|
|   | RMSNorm                ||
|   | Multi-Head Attention   ||  16 heads, 64 dim/head
|   | (with RoPE)            ||  Causal or Prefix mask
|   | + Residual             ||
|   |------------------------||
|   | RMSNorm                ||
|   | FeedForward (SwiGLU)   ||  1024 -> 1536 -> 1024
|   | + Residual             ||
|   +------------------------+|
+-----------------------------+
         |
         v
+-----------------------------+
|    RMSNorm                  |  Final normalization
+-----------------------------+
         |
         v
    h: [B, S, 1024]  (hidden states returned)
         |
    +----+--------+--------+---------+
    |             |        |         |
    v             v        v         v
+---------+ +---------+ +------+ +------+
|board_   | |policy_  | |wl_   | |d_    |
|head     | |head     | |head  | |head  |
|1024->41 | |1024->   | |MLP   | |MLP   |
|         | |1924     | |->100 | |->100 |
+---------+ +---------+ +------+ +------+
    |             |        |         |
    v             v        v         v
[board     [move      [WL       [D
 sub-vocab] sub-vocab] buckets]  buckets]
```

### Model Dimensions (Current Config)

| Parameter | Value |
|-----------|-------|
| `vocab_size` | 1968 |
| `embed_dim` | 1024 |
| `num_heads` | 16 |
| `head_dim` | 64 (1024 / 16) |
| `num_layers` | 12 |
| `d_ff` | 1536 |
| `max_seq_len` | 256 (pretrain) / 1024 (finetune) |
| `n_buckets` | 100 |
| `value_hidden_size` | 256 |
| `num_fourier_freq` | 128 |
| `wl_sigma` | 0.4 |

---

## Component Details

### Token Embedding

```python
self.tok_embedding = nn.Embedding(vocab_size, embed_dim)  # 1968 x 1024
```

Full vocabulary embedding. No learned positional embeddings -- RoPE handles position information in the attention layers. During prefix pass, embeddings at `wl_value`/`d_value` positions are **replaced** with Fourier features.

### Rotary Positional Embeddings (RoPE)

```python
rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
```

Applies rotation to Q/K vectors based on position. Shared across all layers. Provides relative position information without learned position embeddings.

### Multi-Head Attention

```python
attn = MultiHeadAttention(
    embed_dim=1024, num_heads=16, num_kv_heads=16,
    head_dim=64,
    q_proj=nn.Linear(1024, 1024, bias=False),
    k_proj=nn.Linear(1024, 1024, bias=False),
    v_proj=nn.Linear(1024, 1024, bias=False),
    output_proj=nn.Linear(1024, 1024, bias=False),
    pos_embeddings=rope, max_seq_len=max_seq_len,
    is_causal=True
)
```

Standard multi-head attention with RoPE. No grouped-query attention (num_kv_heads == num_heads). The `is_causal=True` flag is the default; the prefix mask overrides this when provided.

### FeedForward (SwiGLU)

```python
mlp = FeedForward(
    gate_proj=nn.Linear(1024, 1536, bias=False),
    down_proj=nn.Linear(1536, 1024, bias=False),
    up_proj=nn.Linear(1024, 1536, bias=False)
)
```

SwiGLU activation: `down_proj(silu(gate_proj(x)) * up_proj(x))`. Note: `d_ff=1536` is 1.5x expansion (not the typical 4x) to control parameter count.

### RMSNorm

```python
RMSNorm(dim=1024)
```

Root Mean Square normalization. Used as pre-norm in each layer (sa_norm, mlp_norm) and as final normalization.

---

## Output Heads

### board_head (Board Sub-Vocabulary)

```python
self.board_head = nn.Linear(embed_dim, board_vocab_size)  # 1024 -> 41
```

Applied to causal pass hidden states. Outputs 41 logits over the board sub-vocabulary. Predicts:
- Next board token (piece, empty, start_pos, end_pos, castling, STM)
- Structural tokens (wl_value, d_value)
- Signal tokens (generic_move, continue_var, end_var, new_variation, end_think, start_think)

### policy_head (Move Sub-Vocabulary)

```python
self.policy_head = nn.Linear(embed_dim, move_vocab_size)  # 1024 -> 1924
```

Applied to prefix pass hidden states. Outputs 1924 logits over the move sub-vocabulary. Used for:
- Final move prediction (from STM in pretraining, from `end_think` in finetuning)
- Normal move prediction in non-thinking sequences

### thinking_policy_head (Move Sub-Vocabulary)

```python
self.thinking_policy_head = nn.Linear(embed_dim, move_vocab_size)  # 1024 -> 1924
```

Same architecture as `policy_head`. Used for variation move prediction during thinking:
- Root moves (predicted from `start_think` or `end_var`)
- PV continuation moves (predicted from board STM in variations)

Initialized by cloning pretrained `policy_head` weights during finetuning.

### wl_head (Value Head)

```python
self.wl_head = ValueHead(embed_dim, n_buckets, value_hidden_size)
# ValueHead: Linear(1024, 256) -> Mish -> Linear(256, 100)
```

2-layer MLP with Mish activation. Predicts WL (win - loss) as distribution over 100 buckets in [-1, 1]. Bucket centers are concentrated near 0 via Gaussian CDF quantiles:

```python
def make_wl_buckets(n_buckets=100, sigma=0.4):
    t = torch.linspace(0.5/n_buckets, 1 - 0.5/n_buckets, n_buckets)
    centers = sigma * sqrt(2) * erfinv(2 * t - 1)
    return centers.clamp(-1.0, 1.0)
```

### d_head (Value Head)

```python
self.d_head = ValueHead(embed_dim, n_buckets, value_hidden_size)
```

Same architecture as `wl_head`. Predicts D (draw probability) as distribution over 100 uniform buckets in [0, 1]:

```python
def make_d_buckets(n_buckets=100):
    return torch.linspace(0.5/n_buckets, 1 - 0.5/n_buckets, n_buckets)
```

### FourierEncoder

```python
class FourierEncoder(nn.Module):
    def __init__(self, embed_dim, num_frequencies=128):
        self.frequencies = nn.Parameter(torch.randn(1, num_frequencies))  # Learned
        self.proj = nn.Linear(2 * num_frequencies, embed_dim)

    def forward(self, x):
        f = 2 * pi * x.unsqueeze(-1) @ self.frequencies  # (N,1) @ (1,F) -> (N,F)
        features = torch.cat([f.cos(), f.sin()], dim=-1)   # (N, 2F)
        return self.proj(features)                          # (N, embed_dim)
```

Encodes scalar values to `embed_dim`-sized vectors via learned Fourier features. Inspired by Moondream-3. The frequencies are **learned parameters**, not fixed.

---

## Forward Pass

```python
def forward(self, x, input_pos=None, mask_type="causal", block_id=None,
            wl_values=None, d_values=None, wl_positions=None, d_positions=None):

    # 1. Mask generation
    if mask_type == "causal":
        mask = None  # TorchTune handles causal internally
    else:  # "prefix"
        causal_mask = torch.tril(torch.ones(S, S))
        same_block = block_id.unsqueeze(-1) == block_id.unsqueeze(-2)  # [B, S, S]
        mask = causal_mask.unsqueeze(0) | same_block

    # 2. Token embedding
    h = self.tok_embedding(x)  # [B, S, E]

    # 3. Override embeddings at WL/D positions with Fourier encodings
    if wl_positions is not None and wl_positions.any():
        h[wl_positions] = self.fourier_encoder(wl_values[wl_positions])
    if d_positions is not None and d_positions.any():
        h[d_positions] = self.fourier_encoder(d_values[d_positions])

    # 4. Transformer layers
    for layer in self.layers:
        h = layer(h, mask=mask, input_pos=input_pos)

    h = self.norm(h)
    return h  # [B, S, E] -- heads applied externally
```

The model returns hidden states `h`. Heads are applied **outside** the forward method, allowing different heads to be applied to the same hidden states or to select specific positions.

---

## Inference Methods

### `predict_move(fen, temperature, force_legal)`

Single-position move prediction:
1. Tokenize FEN -> 68 tokens
2. Prefix forward pass with single block
3. `policy_head` at last position (STM) -> move sub-vocab logits
4. Filter to legal moves (if `force_legal=True`) using `move_token_to_idx`
5. Sample or argmax -> move sub-vocab index
6. Map back to full vocab via `move_idx_to_full_idx`
7. Convert castling notation

### `predict_move_and_value(fen, temperature, force_legal)`

Move prediction + WDL evaluation:
1. Same as `predict_move` for the move
2. Extend sequence with move + wl_value + d_value tokens
3. Predict WL from move position hidden state
4. Inject WL Fourier features, predict D from wl_value position hidden state
5. Reconstruct W, D, L from WL and D values

### `predict_move_n(initial_fen, history, temperature, force_legal, cached_wl_d)`

Multi-position context prediction:
1. Build sequence: `[board_0] [move_1] [wl] [d] [board_1] [move_2] [wl] [d] ... [board_N]`
2. For each history position (sequentially), predict WL then D (autoregressive value injection)
3. Cached values can skip already-predicted positions
4. Final prefix pass with all values injected -> `policy_head` at last position
5. Returns move + all (wl, d) pairs for caching

---

## Parameter Count (Approximate)

```
Token Embedding:     1968 x 1024 = 2.0M

Per Transformer Layer:
  Q, K, V projections: 3 x (1024 x 1024) = 3.1M
  Output projection:   1024 x 1024 = 1.0M
  Gate projection:     1024 x 1536 = 1.6M
  Up projection:       1024 x 1536 = 1.6M
  Down projection:     1536 x 1024 = 1.6M
  Norms:               2 x 1024 = 2K
  Total per layer:     ~9.0M

12 Layers:             12 x 9.0M = 108M

Final Norm:            1024

board_head:            1024 x 41 = 42K
policy_head:           1024 x 1924 = 2.0M
thinking_policy_head:  1024 x 1924 = 2.0M
wl_head:               1024*256 + 256*100 = 288K
d_head:                same = 288K
fourier_encoder:       128 + 256*1024 = 262K

Total:                 ~116M parameters
```

---

## TorchTune Components

| Component | TorchTune Class | Purpose |
|-----------|-----------------|---------|
| Self-Attention Layer | `TransformerSelfAttentionLayer` | Complete attention + FFN block |
| Attention | `MultiHeadAttention` | Multi-head attention with RoPE |
| FFN | `FeedForward` | SwiGLU feed-forward network |
| Normalization | `RMSNorm` | Root mean square normalization |
| Position Encoding | `RotaryPositionalEmbeddings` | RoPE implementation |
