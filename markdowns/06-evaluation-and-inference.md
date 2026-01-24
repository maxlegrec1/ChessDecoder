# Evaluation and Inference

## Overview

The evaluation system tests the trained model by playing games against Stockfish and estimating its ELO rating based on win rates.

---

## Play Move Module

### File: `src/eval/play_move.py`

### Main Function

```python
def main_model_vs_stockfish(
    model=None,
    device="cuda",
    num_games=200,
    temp=0.1,
    elo=2500
):
    """
    Play games between the model and Stockfish.

    Args:
        model: ChessDecoder instance (loaded)
        device: "cuda" or "cpu"
        num_games: Total games to play
        temp: Temperature for move sampling (0.0 = greedy)
        elo: Stockfish ELO setting

    Returns:
        win_rate: float (0.0 - 1.0)
        estimated_elo: float
    """
```

### Game Playing Loop

```python
for game_num in range(num_games):
    board = chess.Board()
    model_is_white = (game_num % 2 == 0)  # Alternate colors

    while not board.is_game_over():
        if board.turn == chess.WHITE and model_is_white:
            # Model's turn
            move = model.predict_move(
                board.fen(),
                temperature=temp,
                force_legal=True
            )
            board.push_uci(move)
        elif board.turn == chess.BLACK and not model_is_white:
            # Model's turn
            move = model.predict_move(...)
            board.push_uci(move)
        else:
            # Stockfish's turn
            result = stockfish.play(board, limit)
            board.push(result.move)

    # Record result
    results.append(get_result(board, model_is_white))
```

### Color Alternation

To ensure fair evaluation:
- Even games: Model plays White
- Odd games: Model plays Black

This accounts for White's first-move advantage.

### Result Recording

```python
def get_result(board, model_is_white):
    result = board.result()
    if result == "1-0":
        return 1.0 if model_is_white else 0.0
    elif result == "0-1":
        return 0.0 if model_is_white else 1.0
    else:  # Draw
        return 0.5
```

---

## ELO Estimation

### File: `src/eval/elo_eval.py`

### Formula

```python
def estimate_elo(win_rate: float, opponent_elo: int) -> float:
    """
    Estimate model ELO based on win rate against known opponent.

    ELO difference formula:
    expected_score = 1 / (1 + 10^((opponent_elo - player_elo) / 400))

    Solving for player_elo:
    player_elo = opponent_elo - 400 * log10((1 - win_rate) / win_rate)
    """
    if win_rate <= 0:
        return opponent_elo - 800  # Very weak
    if win_rate >= 1:
        return opponent_elo + 800  # Very strong

    return opponent_elo - 400 * math.log10((1 - win_rate) / win_rate)
```

### ELO Difference Table

| Win Rate | ELO Difference |
|----------|----------------|
| 50% | 0 |
| 60% | +72 |
| 70% | +149 |
| 80% | +241 |
| 90% | +382 |
| 95% | +511 |
| 99% | +798 |

---

## Temperature and Move Selection

### Temperature Effect

```python
# In model.predict_move()
if temperature == 0.0:
    idx = torch.argmax(last_logits).item()  # Greedy (deterministic)
else:
    probs = torch.softmax(last_logits / temperature, dim=-1)
    idx = torch.multinomial(probs, 1).item()  # Sample
```

| Temperature | Behavior |
|-------------|----------|
| 0.0 | Always pick highest probability move |
| 0.1 | Mostly top moves, occasional variation |
| 0.5 | More diverse move selection |
| 1.0 | Sample proportional to model's probabilities |
| > 1.0 | More random, flatter distribution |

### Recommended Settings

- **Evaluation**: `temp=0.1` (mostly deterministic, slight variation)
- **Analysis**: `temp=0.0` (see model's top choice)
- **Training games**: `temp=0.5-1.0` (explore different lines)

---

## Legal Move Filtering

### Process

```python
if force_legal:
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)

    # Convert to vocabulary indices
    vocab_legal_moves = []
    for move in legal_moves:
        uci = move.uci()
        # Handle castling conversion
        if board.is_castling(move):
            uci = convert_castling(uci)
        if uci in token_to_idx:
            vocab_legal_moves.append(token_to_idx[uci])

    # Mask illegal moves to -inf
    mask = torch.full_like(last_logits, float('-inf'))
    mask[vocab_legal_moves] = 0
    last_logits = last_logits + mask
```

### Why Filter?

1. Model might assign probability to illegal moves
2. Ensures valid games during evaluation
3. Prevents crashes from invalid UCI strings

---

## PGN Generation

Games are saved in PGN format for analysis:

```python
def save_pgn(game, filepath):
    """Save a single game to PGN file."""
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "Model Evaluation"
    pgn_game.headers["White"] = "Model" if model_is_white else "Stockfish"
    pgn_game.headers["Black"] = "Stockfish" if model_is_white else "Model"
    pgn_game.headers["Result"] = board.result()

    node = pgn_game
    for move in game.move_stack:
        node = node.add_variation(move)

    with open(filepath, "a") as f:
        print(pgn_game, file=f)
```

### Example PGN Output

```
[Event "Model Evaluation"]
[White "ChessDecoder"]
[Black "Stockfish 2500"]
[Result "0-1"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 ...
```

---

## Running Evaluation

### Basic Usage

```python
from src.models.model import ChessDecoder
from src.models.vocab import vocab_size
from src.eval.play_move import main_model_vs_stockfish

# Load model
model = ChessDecoder(vocab_size=vocab_size, embed_dim=768, num_heads=12, num_layers=12)
model.load_state_dict(torch.load("checkpoints/checkpoint_epoch_10.pt"))
model.to("cuda")
model.eval()

# Run evaluation
win_rate, estimated_elo = main_model_vs_stockfish(
    model=model,
    device="cuda",
    num_games=200,
    temp=0.1,
    elo=2500  # Stockfish strength
)

print(f"Win rate: {win_rate:.2%}")
print(f"Estimated ELO: {estimated_elo:.0f}")
```

### Evaluation Against Different Stockfish Levels

```python
elo_levels = [1500, 2000, 2500, 3000]
results = {}

for elo in elo_levels:
    win_rate, est_elo = main_model_vs_stockfish(
        model=model,
        num_games=100,
        elo=elo
    )
    results[elo] = {"win_rate": win_rate, "estimated_elo": est_elo}
```

---

## Inference for Analysis

### Single Position Analysis

```python
def analyze_position(model, fen, top_k=5, temperature=1.0):
    """Get top-k moves with probabilities."""
    model.eval()
    device = next(model.parameters()).device

    tokens = fen_to_position_tokens(fen)
    input_ids = torch.tensor([token_to_idx[t] for t in tokens]).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, value_logits = model(input_ids, mask_type="prefix")

    # Get move probabilities
    last_logits = policy_logits[0, -1, :]
    probs = torch.softmax(last_logits / temperature, dim=-1)

    # Filter to legal moves
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)

    move_probs = []
    for move in legal_moves:
        uci = convert_castling(move.uci())
        if uci in token_to_idx:
            idx = token_to_idx[uci]
            move_probs.append((move.uci(), probs[idx].item()))

    # Sort by probability
    move_probs.sort(key=lambda x: x[1], reverse=True)

    # Get WDL evaluation
    wdl = torch.softmax(value_logits[0, -1, :], dim=-1)
    win, draw, loss = wdl.tolist()

    return {
        "top_moves": move_probs[:top_k],
        "evaluation": {"win": win, "draw": draw, "loss": loss}
    }
```

### Example Output

```python
>>> analyze_position(model, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
{
    "top_moves": [
        ("e7e5", 0.42),
        ("c7c5", 0.18),
        ("e7e6", 0.12),
        ("d7d5", 0.09),
        ("c7c6", 0.07)
    ],
    "evaluation": {
        "win": 0.28,   # Black's win probability
        "draw": 0.45,
        "loss": 0.27   # Black's loss probability
    }
}
```

---

## Batch Inference

For analyzing multiple positions efficiently:

```python
def batch_predict(model, fens, temperature=0.0):
    """Predict best move for multiple positions."""
    model.eval()
    device = next(model.parameters()).device

    # Tokenize all positions
    all_tokens = []
    for fen in fens:
        tokens = fen_to_position_tokens(fen)
        ids = [token_to_idx[t] for t in tokens]
        all_tokens.append(ids)

    # Pad to same length
    max_len = max(len(t) for t in all_tokens)
    padded = [t + [token_to_idx["pad"]] * (max_len - len(t)) for t in all_tokens]
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)

    with torch.no_grad():
        policy_logits, _ = model(input_ids, mask_type="prefix")

    # Get predictions
    predictions = []
    for i, fen in enumerate(fens):
        last_logits = policy_logits[i, len(all_tokens[i]) - 1, :]

        # Apply legal move filtering
        board = chess.Board(fen)
        legal_mask = get_legal_mask(board, last_logits.device)
        masked_logits = last_logits + legal_mask

        if temperature == 0.0:
            idx = torch.argmax(masked_logits).item()
        else:
            probs = torch.softmax(masked_logits / temperature, dim=-1)
            idx = torch.multinomial(probs, 1).item()

        predictions.append(convert_castling_back(idx_to_token[idx]))

    return predictions
```

---

## Performance Benchmarks

### Expected Metrics to Track

| Metric | Description | Target |
|--------|-------------|--------|
| Move Accuracy | % of positions where model matches Stockfish | 40-60% |
| Top-3 Accuracy | % where Stockfish move in top 3 | 70-85% |
| Win Rate vs SF 1500 | Games won against weak Stockfish | > 90% |
| Win Rate vs SF 2500 | Games won against strong Stockfish | 30-50% |
| Inference Speed | Positions per second | 100+ |

### Typical Training Progression

| Epoch | Move Acc | Board Acc | WDL Loss | Est. ELO |
|-------|----------|-----------|----------|----------|
| 1 | 15% | 60% | 0.15 | ~1200 |
| 3 | 30% | 80% | 0.10 | ~1600 |
| 5 | 40% | 90% | 0.08 | ~1900 |
| 10 | 50% | 95% | 0.06 | ~2200 |

(Values are illustrative; actual results depend on data quality and quantity)
