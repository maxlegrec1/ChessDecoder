# Model Architecture Details

## ChessDecoder Architecture

### Overview

```
Input: Token IDs [batch_size, seq_len]
         │
         ▼
┌─────────────────────────────┐
│    Token Embedding          │  768-dim, no position embedding
│    (vocab_size → 768)       │  (RoPE handles positions)
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Transformer Layer ×12     │
│   ┌───────────────────────┐ │
│   │ RMSNorm               │ │
│   │ Multi-Head Attention  │ │  12 heads, 64 dim/head
│   │ (with RoPE)           │ │  Causal or Prefix mask
│   │ + Residual            │ │
│   ├───────────────────────┤ │
│   │ RMSNorm               │ │
│   │ FeedForward (SwiGLU)  │ │  768 → 3072 → 768
│   │ + Residual            │ │
│   └───────────────────────┘ │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│    RMSNorm                  │  Final normalization
└─────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌───────┐
│Policy │  │Value  │
│ Head  │  │ Head  │
│768→V  │  │768→3  │
└───────┘  └───────┘
    │         │
    ▼         ▼
[vocab_size] [win,draw,loss]
```

### Component Details

#### Token Embedding

```python
self.tok_embedding = nn.Embedding(vocab_size, embed_dim)
# vocab_size ≈ 4,500
# embed_dim = 768
```

No learned positional embeddings - RoPE handles position information within the attention mechanism.

#### Rotary Positional Embeddings (RoPE)

```python
rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
# head_dim = 64 (768 / 12 heads)
# max_seq_len = 256
```

RoPE applies rotation to query and key vectors based on position:
- Better length generalization than absolute positions
- Relative position information preserved
- Applied in each attention layer

#### Multi-Head Attention

```python
attn = MultiHeadAttention(
    embed_dim=768,
    num_heads=12,
    num_kv_heads=12,      # No grouped-query attention
    head_dim=64,          # 768 / 12
    q_proj=nn.Linear(768, 768, bias=False),
    k_proj=nn.Linear(768, 768, bias=False),
    v_proj=nn.Linear(768, 768, bias=False),
    output_proj=nn.Linear(768, 768, bias=False),
    pos_embeddings=rope,
    max_seq_len=256,
    is_causal=True        # Default causal, overridden by mask
)
```

#### FeedForward (SwiGLU)

```python
mlp = FeedForward(
    gate_proj=nn.Linear(768, 3072, bias=False),  # 4× expansion
    down_proj=nn.Linear(3072, 768, bias=False),
    up_proj=nn.Linear(768, 3072, bias=False)
)
```

SwiGLU activation: `down_proj(silu(gate_proj(x)) * up_proj(x))`

#### RMSNorm

```python
RMSNorm(dim=768)
```

Root Mean Square normalization - simpler and often more stable than LayerNorm.

#### Output Heads

```python
# Policy head: predicts next token (for both moves and board tokens)
self.policy_head = nn.Linear(768, vocab_size)  # 768 → ~4,500

# Value head: predicts win/draw/loss probabilities
self.value_head = nn.Linear(768, 3)  # 768 → 3
```

---

## ChessEncoder Architecture

### Overview

```
Input: Token IDs [batch_size, 68]
         │
         ▼
┌─────────────────────────────┐
│    Token Embedding          │
│    (vocab_size → 768)       │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Transformer Layer ×12     │
│   (Bidirectional attention) │
│   (Same structure as        │
│    decoder, but is_causal   │
│    = False)                 │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│    RMSNorm                  │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│    Pool from first token    │  h[:, 0, :]  (start_pos token)
│    (start_pos embedding)    │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│    Policy Head              │
│    (768 → num_policy)       │  Only move prediction
└─────────────────────────────┘
```

### Key Differences from Decoder

| Aspect | Decoder | Encoder |
|--------|---------|---------|
| Attention | Causal + Prefix | Bidirectional only |
| Input length | Variable (up to 256) | Fixed (68) |
| Pooling | Per-position output | First token only |
| Output heads | Policy + Value | Policy only |
| RoPE direction | Causal | Bidirectional |

---

## Parameter Count

### Approximate Calculation

```
Token Embedding:     vocab_size × embed_dim = 4,500 × 768 = 3.5M

Per Transformer Layer:
  Q,K,V projections: 3 × (768 × 768) = 1.8M
  Output projection: 768 × 768 = 0.6M
  Gate projection:   768 × 3072 = 2.4M
  Up projection:     768 × 3072 = 2.4M
  Down projection:   3072 × 768 = 2.4M
  Norms:             2 × 768 = 1.5K
  ─────────────────────────────────
  Total per layer:   ~9.6M

12 Layers:           12 × 9.6M = 115M

Final Norm:          768 = 768
Policy Head:         768 × 4,500 = 3.5M
Value Head:          768 × 3 = 2.3K

─────────────────────────────────────
Total:               ~122M parameters
```

---

## Forward Pass Details

### Decoder Forward

```python
def forward(self, x, input_pos=None, mask_type="causal"):
    bsz, seq_len = x.shape

    # 1. Generate position indices if not provided
    if input_pos is None:
        input_pos = torch.arange(seq_len, device=x.device).unsqueeze(0)

    # 2. Generate attention mask
    if mask_type == "causal":
        mask = None  # TorchTune handles causal internally
    else:  # "prefix"
        mask = self._build_prefix_mask(x, bsz, seq_len)

    # 3. Token embedding
    h = self.tok_embedding(x)  # [bsz, seq_len, 768]

    # 4. Transformer layers
    for layer in self.layers:
        h = layer(h, mask=mask, input_pos=input_pos)

    # 5. Final normalization
    h = self.norm(h)

    # 6. Output heads
    policy_logits = self.policy_head(h)  # [bsz, seq_len, vocab_size]
    value_logits = self.value_head(h)    # [bsz, seq_len, 3]

    return policy_logits, value_logits
```

### Prefix Mask Construction

```python
def _build_prefix_mask(self, x, bsz, seq_len):
    # Start with lower triangular (causal)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
    mask = mask.unsqueeze(0).repeat(bsz, 1, 1)

    # Identify board block boundaries
    is_start = (x == self.start_pos_id)      # start_pos tokens
    is_move = (x < self.num_policy_tokens)   # move tokens

    for b in range(bsz):
        starts = is_start[b].nonzero().flatten()
        moves = is_move[b].nonzero().flatten()

        for s in starts:
            s_idx = s.item()
            # Find next move after this start_pos
            moves_after = moves[moves > s_idx]
            if len(moves_after) > 0:
                m_idx = moves_after[0].item()
                # Enable bidirectional within [s_idx, m_idx)
                mask[b, s_idx:m_idx, s_idx:m_idx] = True
            else:
                # No move found, enable to end
                mask[b, s_idx:, s_idx:] = True

    return mask
```

---

## Inference: Move Prediction

### `predict_move()` Method

```python
@torch.no_grad()
def predict_move(self, fen: str, temperature: float = 1.0, force_legal: bool = True) -> str:
    self.eval()
    device = next(self.parameters()).device

    # 1. Convert FEN to tokens
    tokens = fen_to_position_tokens(fen)  # 68 tokens
    input_ids = torch.tensor([token_to_idx[t] for t in tokens]).unsqueeze(0).to(device)

    # 2. Forward pass with prefix mask (full board context)
    policy_logits, _ = self(input_ids, mask_type="prefix")

    # 3. Get logits for last position
    last_logits = policy_logits[0, -1, :]  # [vocab_size]

    # 4. Filter to legal moves if requested
    if force_legal:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        vocab_legal_moves = [token_to_idx[convert_move(m)] for m in legal_moves
                           if convert_move(m) in token_to_idx]

        # Mask illegal moves to -inf
        mask = torch.full_like(last_logits, float('-inf'))
        mask[vocab_legal_moves] = 0
        last_logits = last_logits + mask

    # 5. Sample from distribution
    if temperature == 0.0:
        idx = torch.argmax(last_logits).item()
    else:
        probs = torch.softmax(last_logits / temperature, dim=-1)
        idx = torch.multinomial(probs, 1).item()

    # 6. Convert index to move string
    move_str = idx_to_token[idx]

    # 7. Post-process castling notation
    move_str = convert_castling_back(move_str)

    return move_str
```

### Castling Conversion

The model uses rook destination for castling internally:

```python
# Input conversion (standard → internal)
"e1g1" → "e1h1"  # White kingside
"e1c1" → "e1a1"  # White queenside
"e8g8" → "e8h8"  # Black kingside
"e8c8" → "e8a8"  # Black queenside

# Output conversion (internal → standard)
"e1h1" → "e1g1"
"e1a1" → "e1c1"
"e8h8" → "e8g8"
"e8a8" → "e8c8"
```

---

## TorchTune Components Used

The model leverages TorchTune's pre-built modules:

| Component | TorchTune Class | Purpose |
|-----------|-----------------|---------|
| Self-Attention Layer | `TransformerSelfAttentionLayer` | Complete attention + FFN block |
| Attention | `MultiHeadAttention` | Multi-head attention with RoPE support |
| FFN | `FeedForward` | SwiGLU feed-forward network |
| Normalization | `RMSNorm` | Root mean square normalization |
| Position Encoding | `RotaryPositionalEmbeddings` | RoPE implementation |

### Why TorchTune?

- Well-optimized implementations
- Native RoPE support
- Consistent with modern LLM architectures (Llama-style)
- Flash Attention compatible

---

## Memory Footprint

### Per-Sample Memory (Training)

```
Input tokens:        256 × 4 bytes = 1 KB
Embeddings:          256 × 768 × 4 = 786 KB
Attention (per layer):
  Q, K, V:           3 × 256 × 768 × 4 = 2.4 MB
  Attention scores:  12 × 256 × 256 × 4 = 3 MB
  (can be reduced with Flash Attention)
```

### Batch Memory (batch_size=16)

```
Forward activations: ~16 × (embeddings + attention) × 12 layers
                   ≈ 16 × 50 MB × 12 = ~10 GB
Gradients:         ≈ same
Optimizer states:  2 × parameters ≈ 2 × 122M × 4 = ~1 GB
─────────────────────────────────────────────────────────
Total:             ~20-25 GB (estimate)
```

### Recommendations

1. Use gradient checkpointing for memory reduction
2. Consider Flash Attention for efficient attention computation
3. Use mixed precision (FP16/BF16) to halve memory
