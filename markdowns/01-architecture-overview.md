# Architecture Overview

## Summary

ChessDecoder is a decoder-only transformer model for chess that generates board representations autoregressively, predicts moves, and evaluates positions. The model uses a **two-pass forward architecture**: a causal pass for board token generation and a prefix (bidirectional-within-block) pass for move and value prediction.

The model is first **pretrained** on standard game sequences (board-move-value triples), then **finetuned** with thinking variation sequences that teach the model to explore candidate moves before selecting a final move.

## Core Design Principles

1. **Fixed-length board representation**: Every chess position is encoded as exactly 68 tokens (`start_pos` + 64 squares + `end_pos` + castling + side_to_move).
2. **Sub-vocabulary heads**: Each prediction head outputs logits over only its relevant sub-vocabulary, not the full vocab. The `board_head` predicts 41 board tokens, the policy heads predict 1924 UCI move tokens.
3. **Two-pass architecture**: Board generation uses causal (autoregressive) attention. Move and value prediction use prefix attention (bidirectional within board blocks, causal across blocks).
4. **Fourier value injection**: WL/D evaluation values are encoded via learned Fourier features and injected as embeddings at placeholder positions, providing value context for subsequent predictions.
5. **Soft bucket losses**: Value heads predict distributions over 100 buckets rather than single scalars, using soft cross-entropy loss with linear interpolation between nearest buckets.

## Model Heads

| Head | Type | Output Size | Attention | Purpose |
|------|------|-------------|-----------|---------|
| `board_head` | `nn.Linear` | 41 (board sub-vocab) | Causal (pass 1) | Predict next board/structural/signal token |
| `policy_head` | `nn.Linear` | 1924 (move sub-vocab) | Prefix (pass 2) | Predict final/normal moves |
| `thinking_policy_head` | `nn.Linear` | 1924 (move sub-vocab) | Prefix (pass 2) | Predict variation moves (root + PV) |
| `wl_head` | `ValueHead` MLP | 100 buckets | Prefix (pass 2) | Predict WL value in [-1, 1] |
| `d_head` | `ValueHead` MLP | 100 buckets | Prefix (pass 2) | Predict D (draw) value in [0, 1] |

---

## Pretraining: Worked Example

### The Game

Consider a 2-move snippet: starting position, white plays e2e4, then black plays e7e5.

### Input Sequence (142 tokens)

```
Pos   Token                 Description
----  --------------------  -----------
0     start_pos             Board 0 start
1     white_rook            a1
2     white_knight          b1
3     white_bishop          c1
4     white_queen           d1
5     white_king            e1
6     white_bishop          f1
7     white_knight          g1
8     white_rook            h1
9     white_pawn            a2
10    white_pawn            b2
11    white_pawn            c2
12    white_pawn            d2
13    white_pawn            e2
14    white_pawn            f2
15    white_pawn            g2
16    white_pawn            h2
17    empty                 a3
...   empty                 (a3-h6 all empty)
48    empty                 h6
49    black_pawn            a7
50    black_pawn            b7
51    black_pawn            c7
52    black_pawn            d7
53    black_pawn            e7
54    black_pawn            f7
55    black_pawn            g7
56    black_pawn            h7
57    black_rook            a8
58    black_knight          b8
59    black_bishop          c8
60    black_queen           d8
61    black_king            e8
62    black_bishop          f8
63    black_knight          g8
64    black_rook            h8
65    end_pos               Board 0 end
66    KQkq                  Castling rights
67    white_to_move         Side to move (STM)
68    e2e4                  White's move (played_move token)
69    wl_value              WL placeholder
70    d_value               D placeholder
71    start_pos             Board 1 start (position after e2e4)
72    white_rook            a1
...   (64 squares of the position after e2e4)
136   end_pos               Board 1 end
137   KQkq                  Castling rights
138   black_to_move         STM
139   e7e5                  Black's move
140   wl_value              WL placeholder
141   d_value               D placeholder
```

### Target Tensors

Both `board_target_ids` and `move_target_ids` use `IGNORE_INDEX = -100` for positions that should not contribute to loss. Index spaces are **different**: `board_target_ids` uses board sub-vocab indices (0-40), `move_target_ids` uses move sub-vocab indices (0-1923).

```
Pos   Input Token       board_target_ids              move_target_ids     Notes
----  ----------------  ----------------------------  ------------------  -----
0     start_pos         white_rook [N]                -100                board sub-vocab idx
1     white_rook        white_knight [N]              -100                shifted next-token
2     white_knight      white_bishop [N]              -100
...   (squares)         (shifted next square) [N]     -100
64    black_rook        end_pos [N]                   -100
65    end_pos           KQkq [N]                      -100
66    KQkq              white_to_move [N]             -100
67    white_to_move     generic_move [O]              best_move (e2e4)    *** STM OVERRIDE ***
68    e2e4              wl_value [N]                   -100               natural shifted
69    wl_value          d_value [N]                    -100               natural shifted
70    d_value           start_pos [N]                  -100               natural shifted
71    start_pos         white_rook [N]                 -100               Board 1 tokens
...   (Board 1 squares) (shifted) [N]                 -100
136   end_pos           KQkq [N]                       -100
137   KQkq              black_to_move [N]              -100
138   black_to_move     generic_move [O]               best_move (e7e5)   *** STM OVERRIDE ***
139   e7e5              wl_value [N]                    -100
140   wl_value          d_value [N]                     -100
141   d_value           -100 (last position)            -100
```

**Legend**: `[N]` = natural shifted next-token (mapped to board sub-vocab via `full_idx_to_board_idx`). `[O]` = override.

**Key points:**
- At pos 67 (STM), the natural shifted target would be `e2e4` (a move token), which is NOT in the board sub-vocab. So we **override** with `generic_move` (board sub-vocab signal meaning "a move comes next").
- At the same pos 67, `move_target_ids` is set to the board sub-vocab index of `best_move` (the Stockfish recommendation), and `move_mask[67] = True`.
- At pos 68 (the move token `e2e4`), the shifted target is `wl_value` which IS in the board sub-vocab -- no override needed.
- Positions 69, 70 (`wl_value`, `d_value`): natural shifted targets are `d_value` and `start_pos` respectively -- both in board sub-vocab.

### Masks

```
Pos   board_mask  move_mask  wl_pos  d_pos   wdl_valid
----  ----------  ---------  ------  ------  ---------
0-66  no          no         no      no      no          Pre-first-move (excluded)
67    YES         YES        no      no      yes         First STM -- board_mask starts here
68    YES         no         no      no      no          Move token (e2e4)
69    YES         no         YES     no      yes         wl_value placeholder
70    YES         no         no      YES     yes         d_value placeholder
71    YES         no         no      no      no          Board 1 start
...
137   YES         no         no      no      no
138   YES         YES        no      no      yes         Second STM
139   YES         no         no      no      no
140   YES         no         YES     no      yes
141   no          no         no      YES     yes         Last pos, board_target=-100
```

**board_mask construction**: `(board_target_ids != -100) & (~pre_first_move_mask)`.

The pre_first_move_mask excludes positions 0-66 (everything before the first move_mask position at 67). This is because the first board has no causal context from prior moves, so training on it would be just memorizing the starting position.

**wl_positions / d_positions**: Mark where Fourier features are injected and where value heads predict.

### Block IDs (for prefix mask)

```
Pos   block_id   Type
----  ---------  ----
0-67  0          Board block 0 (root board, all 68 tokens share block_id=0)
68    2          Orphan (move token, unique ID = max_blocks + offset)
69    3          Orphan (wl_value)
70    4          Orphan (d_value)
71-138 1         Board block 1 (second board)
139   5          Orphan (move token)
140   6          Orphan (wl_value)
141   7          Orphan (d_value)
```

Each board's 68 tokens share a block ID. Non-board tokens get unique IDs (constructed as `arange(max_seq_len) + max_block_num`).

---

### Pass 1: Causal Attention (board_head)

Standard lower-triangular mask. Each position can only attend to positions before it (and itself).

```
        pos: 0  1  2 ... 67 68 69 70 71 72 ... 138 139 140 141
             SP Wr Wn     STM mv wl d  SP Wr      STM mv  wl  d
pos  0  SP  [ 1  0  0     0  0  0  0  0  0       0   0   0   0 ]
     1  Wr  [ 1  1  0     0  0  0  0  0  0       0   0   0   0 ]
     2  Wn  [ 1  1  1     0  0  0  0  0  0       0   0   0   0 ]
    ...
    67 STM  [ 1  1  1     1  0  0  0  0  0       0   0   0   0 ]
    68  mv  [ 1  1  1     1  1  0  0  0  0       0   0   0   0 ]
    69  wl  [ 1  1  1     1  1  1  0  0  0       0   0   0   0 ]
    70   d  [ 1  1  1     1  1  1  1  0  0       0   0   0   0 ]
    71  SP  [ 1  1  1     1  1  1  1  1  0       0   0   0   0 ]
    72  Wr  [ 1  1  1     1  1  1  1  1  1       0   0   0   0 ]
   ...
   138 STM  [ 1  1  1     1  1  1  1  1  1       1   0   0   0 ]
   139  mv  [ 1  1  1     1  1  1  1  1  1       1   1   0   0 ]
   140  wl  [ 1  1  1     1  1  1  1  1  1       1   1   1   0 ]
   141   d  [ 1  1  1     1  1  1  1  1  1       1   1   1   1 ]
```

The `board_head` is applied to these hidden states. At pos 67, it sees the full first board (pos 0-67) and must predict `generic_move`. At pos 70 (d_value), it sees everything through pos 70 and must predict `start_pos` (the next board's first token).

**What board_head learns at each position type:**

| Position | Sees | Must Predict | Why |
|----------|------|-------------|-----|
| Square (e.g. pos 3) | Preceding squares | Next square | Board reconstruction |
| end_pos (pos 65) | Full 64 squares | Castling rights | Infer from piece positions |
| Castling (pos 66) | Board + castling | STM (side to move) | Infer from position |
| STM (pos 67) | Full board | `generic_move` | Signal that a move follows |
| Move (pos 68) | Board + move | `wl_value` | Always wl_value after a move |
| wl_value (pos 69) | Board + move + wl | `d_value` | Always d_value after wl |
| d_value (pos 70) | Board + move + wl + d | `start_pos` | Next board starts |

### Pass 2: Prefix Attention (policy_head, wl_head, d_head)

The prefix mask allows **bidirectional attention within board blocks** while maintaining causal attention across blocks.

```
prefix_mask = causal_mask | same_block_mask
```

Condensed view (grouping positions by type):

```
        pos: [Board0: 0-67]  68  69  70  [Board1: 71-138]  139 140 141
              block=0        orph orph orph block=1          orph orph orph

[Board0]    [ BIDIR          0   0   0    0                  0   0   0  ]
 pos 68 mv  [ 1..1           1   0   0    0                  0   0   0  ]
 pos 69 wl  [ 1..1           1   1   0    0                  0   0   0  ]
 pos 70  d  [ 1..1           1   1   1    0                  0   0   0  ]
[Board1]    [ 1..1           1   1   1    BIDIR              0   0   0  ]
 pos139 mv  [ 1..1           1   1   1    1..1               1   0   0  ]
 pos140 wl  [ 1..1           1   1   1    1..1               1   1   0  ]
 pos141  d  [ 1..1           1   1   1    1..1               1   1   1  ]
```

**BIDIR** means all positions within that block can attend to each other (full bidirectional). This is because `same_block_mask` is True for any pair of positions sharing the same `block_id`.

**Concrete examples:**

| Position | Block | Attends to | Effect |
|----------|-------|-----------|--------|
| pos 0 (start_pos, block 0) | 0 | pos 0-67 (block 0 bidirectional) | Sees the entire first board |
| pos 67 (STM, block 0) | 0 | pos 0-67 (block 0 bidirectional) | Full board for move prediction |
| pos 68 (move, orphan) | orphan | pos 0-68 (causal only) | Sees board + its own move |
| pos 69 (wl_value, orphan) | orphan | pos 0-69 (causal only) | Sees board + move + self |
| pos 71 (start_pos, block 1) | 1 | pos 0-71 (causal) + pos 71-138 (block 1) = **pos 0-138** | Entire block 1 + all prior |
| pos 138 (STM, block 1) | 1 | pos 0-138 (causal overlaps block 1) = **pos 0-138** | Full board 1 + all history |
| pos 139 (move, orphan) | orphan | pos 0-139 (causal only) | Everything up to this point |

**Key insight**: pos 71 (first token of Board 1) can see pos 138 (last token of Board 1) thanks to the same-block mask, even though pos 138 comes *after* pos 71 in the sequence. This is how the policy_head at pos 138 (STM) gets full bidirectional context within its board block.

**What each head reads from prefix hidden states:**

| Head | Reads from position | Block context | Predicts |
|------|--------------------|----|---------|
| `policy_head` | pos 67 (STM of board 0) | Full board 0 bidirectional | Best move for position 0 |
| `policy_head` | pos 138 (STM of board 1) | Full board 1 bidirectional + all prior context | Best move for position 1 |
| `wl_head` | pos 68 (move token after STM 0) | Board 0 + move (causal) | WL value for position 0 |
| `d_head` | pos 69 (wl_value after move 0) | Board 0 + move + Fourier(WL) | D value for position 0 |

**Fourier injection**: Before the prefix pass, the token embeddings at pos 69 and 70 (and 140, 141) are **replaced** with `FourierEncoder(wl_value)` and `FourierEncoder(d_value)` respectively, using ground-truth values during training.

---

## Finetuning: Worked Example

### The Position

Root position (starting position), with MCTS data producing 2 variations:
- Variation 1: root_move = e2e4, PV depth 1 (one continuation: e7e5)
- Variation 2: root_move = d2d4, PV depth 1 (one continuation: d7d5)
- Final best move (mcts_action): e2e4

### Input Sequence (288 tokens)

```
Pos    Token              Description
-----  -----------------  -----------
0-67   [root board 68]    Starting position (same as pretrain example above)
68     start_think        Begin thinking block

--- Variation 1: e2e4 ---
69     e2e4               Root move 1 (predicted from pos 68 = start_think)
70     wl_value           WL for position after e2e4
71     d_value            D for position after e2e4
72-139 [board 68]         Position after e2e4 (Board 1)
140    e7e5               PV move 1 (predicted from pos 139 = Board 1 STM)
141    wl_value           WL for position after e2e4 e7e5
142    d_value            D for position after e2e4 e7e5
143-210 [board 68]        Position after e2e4 e7e5 (Board 2)
211    end_var            End of variation 1

--- Variation 2: d2d4 ---
212    d2d4               Root move 2 (predicted from pos 211 = end_var)
213    wl_value           WL for position after d2d4
214    d_value            D for position after d2d4
215-282 [board 68]        Position after d2d4 (Board 3)
283    end_var            End of variation 2

--- Final decision ---
284    end_think          End thinking block
285    e2e4               Final move (predicted from pos 284 = end_think)
286    wl_value           WL for root position
287    d_value            D for root position
```

### Target Tensors

```
Pos    Input Token       board_target_ids              move_target_ids       Key Masks
-----  ----------------  ----------------------------  --------------------  ---------
0      start_pos         -100                          -100                  (pre-first-move)
...    (root board)      -100                          -100                  (pre-first-move)
67     white_to_move     -100                          -100                  (pre-first-move)
68     start_think       generic_move [O]              e2e4                  board_mask, thinking_move_mask
69     e2e4              wl_value [N]                   -100                 board_mask, (wl for val heads)
70     wl_value          d_value [N]                    -100                 board_mask, wl_positions
71     d_value           start_pos [N]                  -100                 board_mask, d_positions
72     start_pos         (square_a1) [N]                -100                 board_mask
...    (Board 1)         (shifted) [N]                  -100                 board_mask
138    KQkq              black_to_move [N]              -100                 board_mask
139    black_to_move     continue_var [O]               e7e5                 board_mask, thinking_move_mask
140    e7e5              wl_value [N]                    -100                 board_mask
141    wl_value          d_value [N]                     -100                board_mask, wl_positions
142    d_value           start_pos [N]                   -100                board_mask, d_positions
143    start_pos         (square_a1) [N]                 -100                board_mask
...    (Board 2)         (shifted) [N]                   -100                board_mask
209    KQkq              (stm) [N]                       -100                board_mask
210    (stm)             end_var [N]                     -100                board_mask
211    end_var           new_variation [O]               d2d4                board_mask, thinking_move_mask
212    d2d4              wl_value [N]                    -100                board_mask
213    wl_value          d_value [N]                     -100                board_mask, wl_positions
214    d_value           start_pos [N]                   -100                board_mask, d_positions
215    start_pos         (square_a1) [N]                 -100                board_mask
...    (Board 3)         (shifted) [N]                   -100                board_mask
281    KQkq              (stm) [N]                       -100                board_mask
282    (stm)             end_var [N]                     -100                board_mask
283    end_var           end_think [N]                   -100                board_mask
284    end_think         generic_move [O]                e2e4                board_mask, move_mask
285    e2e4              wl_value [N]                    -100                board_mask
286    wl_value          d_value [N]                     -100                board_mask, wl_positions
287    d_value           -100 (last)                     -100                -
```

### Understanding the Overrides

There are **4 types of overrides** (positions where `board_target_ids` differs from the natural shifted next-token):

| # | Position | Input Token | Natural Shifted Target | Override Target | Reason |
|---|----------|------------|----------------------|-----------------|--------|
| 1 | 68 | `start_think` | `e2e4` (move, not in board sub-vocab) | `generic_move` | Signals "a move follows" |
| 2 | 139 | `black_to_move` (Board 1 STM) | `e7e5` (move, not in board sub-vocab) | `continue_var` | Signals "PV continues" |
| 3 | 211 | `end_var` | `d2d4` (move, not in board sub-vocab) | `new_variation` | Signals "another variation" |
| 4 | 284 | `end_think` | `e2e4` (move, not in board sub-vocab) | `generic_move` | Signals "final move follows" |

Everything else is a **natural shifted target**: the next token in the sequence, mapped to board sub-vocab. For example:
- pos 69 (`e2e4`): shifted target is `wl_value` → mapped to board sub-vocab index for `wl_value` ✓
- pos 70 (`wl_value`): shifted target is `d_value` ✓
- pos 71 (`d_value`): shifted target is `start_pos` ✓
- pos 210 (Board 2 STM): shifted target is `end_var` → `end_var` IS in board sub-vocab, so it's natural ✓
- pos 283 (`end_var`): shifted target is `end_think` → `end_think` IS in board sub-vocab, so it's natural ✓

### Masks Summary

```
Mask                  Where True                             Count   Used By
--------------------  -------------------------------------  ------  -------
board_mask            pos 68-286 (minus -100 and last pos)   ~218    board_head loss
move_mask             pos 284 (end_think)                    1       policy_head loss
thinking_move_mask    pos 68, 139, 211                       3       thinking_policy_head loss
wl_positions          pos 70, 141, 213, 286                  4       Fourier injection + d_head
d_positions           pos 71, 142, 214, 287                  4       Fourier injection
continue_var_mask     pos 139                                1       metrics only
new_variation_mask    pos 211                                1       metrics only
```

**pre_first_move_mask**: Uses `any_move = move_mask | thinking_move_mask`. First True position is 68 (`start_think`, thinking_move_mask). So `pre_first_move_mask` = pos 0-67, excluding the entire root board from board loss.

### Block IDs

```
Pos      block_id   Type
-------  ---------  ----
0-67     0          Block 0 (root board)
68       5          Orphan (start_think)
69       6          Orphan (e2e4, root move 1)
70       7          Orphan (wl_value)
71       8          Orphan (d_value)
72-139   1          Block 1 (Board 1: position after e2e4)
140      9          Orphan (e7e5, PV move)
141      10         Orphan (wl_value)
142      11         Orphan (d_value)
143-210  2          Block 2 (Board 2: position after e2e4 e7e5)
211      12         Orphan (end_var)
212      13         Orphan (d2d4, root move 2)
213      14         Orphan (wl_value)
214      15         Orphan (d_value)
215-282  3          Block 3 (Board 3: position after d2d4)
283      16         Orphan (end_var)
284      17         Orphan (end_think)
285      18         Orphan (e2e4, final move)
286      19         Orphan (wl_value)
287      20         Orphan (d_value)
```

---

### Pass 1: Causal Attention Matrix (board_head)

Standard lower-triangular. Shown in block form:

```
              Block0  st  mv wl d  Block1  mv wl d  Block2  ev  mv wl d  Block3  ev  et  mv wl d
              0-67    68  69 70 71 72-139  140 .. .. 143-210 211 212.. .. 215-282 283 284 285.. ..

Block0 0-67  [ TRI    0   0  0  0  0       0        0       0   0        0       0   0   0      ]
st     68   [ 1..1   1   0  0  0  0       0        0       0   0        0       0   0   0      ]
mv     69   [ 1..1   1   1  0  0  0       0        0       0   0        0       0   0   0      ]
wl     70   [ 1..1   1   1  1  0  0       0        0       0   0        0       0   0   0      ]
d      71   [ 1..1   1   1  1  1  0       0        0       0   0        0       0   0   0      ]
Block1 72-139[ 1..1   1   1  1  1  TRI     0        0       0   0        0       0   0   0      ]
mv     140  [ 1..1   1   1  1  1  1..1    1        0       0   0        0       0   0   0      ]
wl     141  [ 1..1   1   1  1  1  1..1    1  1     0       0   0        0       0   0   0      ]
d      142  [ 1..1   1   1  1  1  1..1    1  1  1  0       0   0        0       0   0   0      ]
Block2 143-210[1..1   1   1  1  1  1..1    1  1  1  TRI     0   0        0       0   0   0      ]
ev     211  [ 1..1   1   1  1  1  1..1    1  1  1  1..1    1   0        0       0   0   0      ]
mv     212  [ 1..1   1   1  1  1  1..1    1  1  1  1..1    1   1        0       0   0   0      ]
...
Block3 215-282[.........................................................   TRI     0   0   0      ]
ev     283  [ ............................................................1..1    1   0   0      ]
et     284  [ ............................................................1..1    1   1   0      ]
mv     285  [ ............................................................1..1    1   1   1      ]
wl     286  [ ............................................................1..1    1   1   1  1   ]
d      287  [ ............................................................1..1    1   1   1  1  1]
```

**TRI** = lower-triangular within that block. `1..1` = all ones (full row of 1s for that block range).

**What the board_head decides at critical positions (causal context):**

| Pos | Input | Causal Context | board_target | Decision |
|-----|-------|---------------|--------------|----------|
| 68 | `start_think` | Root board + start_think | `generic_move` | "A thinking move follows" |
| 139 | Board 1 STM | Everything through Board 1 | `continue_var` | "PV continues" (vs `end_var`) |
| 210 | Board 2 STM | Everything through Board 2 | `end_var` | "End this variation" (natural target) |
| 211 | `end_var` | Everything through end_var | `new_variation` | "Start new variation" (vs `end_think`) |
| 282 | Board 3 STM | Everything through Board 3 | `end_var` | "End this variation" |
| 283 | `end_var` | Everything through end_var | `end_think` | "Done thinking" (natural target) |
| 284 | `end_think` | Everything through end_think | `generic_move` | "Final move follows" |

### Pass 2: Prefix Attention Matrix (move/value heads)

```
prefix_mask = causal_mask | same_block_mask
```

Shown in block form:

```
              Block0  st  mv wl d  Block1  mv wl d  Block2  ev  mv wl d  Block3  ev  et  mv wl d
              0-67    68  69 70 71 72-139  140 .. .. 143-210 211 212.. .. 215-282 283 284 285.. ..

Block0 0-67  [BIDIR   0   0  0  0  0       0        0       0   0        0       0   0   0      ]
st     68   [ 1..1   1   0  0  0  0       0        0       0   0        0       0   0   0      ]
mv     69   [ 1..1   1   1  0  0  0       0        0       0   0        0       0   0   0      ]
wl     70   [ 1..1   1   1  1  0  0       0        0       0   0        0       0   0   0      ]
d      71   [ 1..1   1   1  1  1  0       0        0       0   0        0       0   0   0      ]
Block1 72-139[ 1..1   1   1  1  1  BIDIR   0        0       0   0        0       0   0   0      ]
mv     140  [ 1..1   1   1  1  1  1..1    1        0       0   0        0       0   0   0      ]
wl     141  [ 1..1   1   1  1  1  1..1    1  1     0       0   0        0       0   0   0      ]
d      142  [ 1..1   1   1  1  1  1..1    1  1  1  0       0   0        0       0   0   0      ]
Block2 143-210[1..1   1   1  1  1  1..1    1  1  1  BIDIR   0   0        0       0   0   0      ]
ev     211  [ 1..1   1   1  1  1  1..1    1  1  1  1..1    1   0        0       0   0   0      ]
mv     212  [ 1..1   1   1  1  1  1..1    1  1  1  1..1    1   1        0       0   0   0      ]
...
Block3 215-282[.........................................................   BIDIR   0   0   0      ]
ev     283  [ ............................................................1..1    1   0   0      ]
et     284  [ ............................................................1..1    1   1   0      ]
mv     285  [ ............................................................1..1    1   1   1      ]
wl     286  [ ............................................................1..1    1   1   1  1   ]
d      287  [ ............................................................1..1    1   1   1  1  1]
```

**BIDIR** = full bidirectional attention within the block (all positions see all positions in the same block).

**The only difference from causal is the BIDIR blocks.** Orphan tokens behave identically to causal because they have no same-block peers.

**Concrete attention patterns for critical positions:**

| Pos | Input | Block | Attends To | Head That Reads |
|-----|-------|-------|-----------|----------------|
| 67 (root STM) | `white_to_move` | 0 | pos 0-67 (block 0, bidir) | -- (pre-first-move, excluded) |
| 68 (`start_think`) | `start_think` | orphan | pos 0-68 (causal) | `thinking_policy_head` → predicts e2e4 |
| 72 (Board 1 start_pos) | `start_pos` | 1 | pos 0-72 (causal) ∪ pos 72-139 (block 1) = **pos 0-139** | -- |
| 139 (Board 1 STM) | `black_to_move` | 1 | pos 0-139 (causal, superset of block 1) = **pos 0-139** | `thinking_policy_head` → predicts e7e5 |
| 211 (`end_var`) | `end_var` | orphan | pos 0-211 (causal) | `thinking_policy_head` → predicts d2d4 |
| 284 (`end_think`) | `end_think` | orphan | pos 0-284 (causal, sees ALL variations) | `policy_head` → predicts e2e4 (final) |

**Why bidirectional matters for Board 1 (pos 72-139):**
- Pos 72 (`start_pos`) needs to "see" pos 139 (STM = `black_to_move`) to know what side is to move
- Pos 139 (STM) needs to "see" pos 72-138 (all pieces) for full board context
- Without bidirectional: pos 72 can only see pos 0-72, missing the rest of its own board
- With bidirectional: pos 72 sees pos 0-139, getting complete context for all heads

---

## Two-Pass Forward Flow

```
                            input_ids [B, S]
                                  |
                    +-------------+-------------+
                    |                           |
              [Pass 1: Causal]           [Pass 2: Prefix]
              mask = lower-tri           mask = causal | same_block
              no Fourier injection       Fourier(WL,D) replaces embeddings
                    |                           |
              h_causal [B,S,E]           h_prefix [B,S,E]
                    |                           |
              board_head(h)              +------+------+------+
              → [B,S,41]                 |      |      |      |
                    |                 policy  think  wl_head d_head
              board_loss              head    _head   (MLP)  (MLP)
              (CE over board          →1924   →1924   →100   →100
               sub-vocab)               |      |      |      |
                                   move_loss think  wl_loss d_loss
                                         move_loss
```

## Loss Summary

### Pretraining

```
total_loss = 5.0 * move_loss + 1.0 * board_loss + 1.0 * wl_loss + 1.0 * d_loss
```

### Finetuning

```
total_loss = 5.0 * final_move_loss + 2.0 * thinking_move_loss + 1.0 * board_loss + 1.0 * wl_loss + 1.0 * d_loss
```

All losses use `IGNORE_INDEX = -100` and sub-vocabulary logits.

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Framework | PyTorch |
| Transformer Modules | TorchTune (Llama-style) |
| Chess Logic | python-chess |
| Experiment Tracking | Weights & Biases |
| Evaluation Baseline | Stockfish |
| Data Format | Apache Parquet |

## File Structure

| File | Purpose |
|------|---------|
| `src/models/vocab.py` | Vocabulary definitions (1968 tokens), sub-vocabulary mappings |
| `src/models/model.py` | ChessDecoder model with sub-vocab heads |
| `src/dataloader/data.py` | FEN-to-token conversion, game sequence generation |
| `src/dataloader/loader.py` | Pretraining IterableDataset and DataLoader |
| `src/train/train.py` | Pretraining training loop |
| `src/train/config.yaml` | Pretraining configuration |
| `finetune/data.py` | Variation sequence generation (Plackett-Luce ordering) |
| `finetune/loader.py` | Finetuning IterableDataset (mixed pretrain + variation) |
| `finetune/train.py` | Finetuning training loop |
| `finetune/config.yaml` | Finetuning configuration |
| `scripts/think.py` | Thinking inference (autoregressive variation generation) |
