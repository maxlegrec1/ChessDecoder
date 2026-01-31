"""
Convert variation parquet rows into token sequences for finetuning.

Each row has:
  - fen: root position FEN
  - variations: JSON list of variations from MCTS
  - mcts_action: best move from MCTS
  - win, draw, loss: WDL values for the position

Each variation has:
  - root_move: the candidate move (standard UCI, e.g. e1g1 for castling)
  - visit_count, visit_fraction, prior
  - nodes: list of {fen, move, wdl, visit_count} dicts (the PV line)

Output sequence format:
  [root_board_68] [start_think]
    [root_move_1] [wl] [d] [board_68] [pv_move_1] [wl] [d] [board_68] [pv_move_2] ... [end_var]
    [root_move_2] [wl] [d] [board_68] [pv_move_1] ... [end_var]
    ...
  [end_think] [final_move] [wl_value] [d_value]

Move prediction positions:
  - root_move_1 is predicted from start_think  -> thinking_policy_head
  - root_move_N is predicted from end_var (of previous variation) -> thinking_policy_head
  - pv_move is predicted from stm of preceding board -> thinking_policy_head
  - final_move is predicted from end_think -> policy_head (normal)
"""

import json
import chess
import pandas as pd
from src.dataloader.data import fen_to_position_tokens
from src.models.vocab import token_to_idx

# Standard UCI -> pseudo-castling (model vocabulary uses king-captures-rook)
_STANDARD_TO_PSEUDO_CASTLING = {
    "e1g1": "e1h1",
    "e1c1": "e1a1",
    "e8g8": "e8h8",
    "e8c8": "e8a8",
}


def _to_model_uci(move_uci: str) -> str:
    """Convert standard UCI castling to pseudo-castling used by model vocab."""
    return _STANDARD_TO_PSEUDO_CASTLING.get(move_uci, move_uci)


def variation_to_token_ids(row, max_variations=3, max_depth=5):
    """
    Convert a parquet row with variation data into a finetuning token sequence.

    Args:
        row: parquet row (dict-like) with fen, variations, mcts_action, win, draw, loss
        max_variations: max number of variations to include
        max_depth: max PV depth per variation (number of nodes)

    Returns:
        ids: list[int] - token IDs
        thinking_move_data: list[(pos, move_token_str)] - positions where thinking_policy_head predicts
        final_move_data: (pos, move_token_str) - position where policy_head predicts
        value_data: list[(wl_pos, d_pos, wl, d, is_valid)] - WL/D value target positions
        block_boundaries: list[(start, end)] - board block boundaries for prefix masking
    """
    fen = row["fen"]
    variations_json = row["variations"]
    mcts_action = row["mcts_action"]

    # Parse variations
    if isinstance(variations_json, str):
        variations = json.loads(variations_json)
    else:
        variations = variations_json

    # Sort by visit_count descending, cap at max_variations
    variations = sorted(variations, key=lambda v: v.get("visit_count", 0), reverse=True)
    variations = variations[:max_variations]

    # Final move WDL
    win = row["win"] if pd.notna(row["win"]) else 0.0
    draw = row["draw"] if pd.notna(row["draw"]) else 0.0
    loss = row["loss"] if pd.notna(row["loss"]) else 0.0
    is_valid_wdl = pd.notna(row["win"]) and pd.notna(row["draw"]) and pd.notna(row["loss"])
    final_wl = win - loss
    final_d = draw

    sequence = []
    block_boundaries = []
    thinking_move_data = []  # (position_idx, move_token_str) for thinking_policy_head
    value_data = []  # (wl_pos, d_pos, wl, d, is_valid)

    # 1. Root board (block 0)
    block_start = len(sequence)
    root_tokens = fen_to_position_tokens(fen)
    sequence.extend(root_tokens)
    block_boundaries.append((block_start, len(sequence)))

    # 2. start_think token
    sequence.append("start_think")

    # 3. For each variation
    for var_idx, var in enumerate(variations):
        root_move = var["root_move"]
        nodes = var.get("nodes", [])[:max_depth]

        if not nodes:
            continue

        # Root move token - predicted from previous token (start_think or end_var)
        root_move_model = _to_model_uci(root_move)
        if root_move_model not in token_to_idx:
            continue

        predict_from_pos = len(sequence) - 1  # start_think or end_var
        thinking_move_data.append((predict_from_pos, root_move_model))
        sequence.append(root_move_model)

        # Each node in the PV: wl, d, board, pv_move
        for node_idx, node in enumerate(nodes):
            node_fen = node["fen"]
            node_wdl = node.get("wdl", [0.0, 0.0, 0.0])
            # wdl is [win, draw, loss] from MCTS (from perspective of side-to-move at root)
            # For nodes at even depth (opponent's move), wdl is flipped
            # The MCTS already stores wdl from the perspective of the node's parent
            # We store wl = win - loss, d = draw
            node_win, node_draw, node_loss = node_wdl
            node_wl = node_win - node_loss
            node_d = node_draw

            # WL value token
            wl_pos = len(sequence)
            sequence.append("wl_value")
            # D value token
            d_pos = len(sequence)
            sequence.append("d_value")
            value_data.append((wl_pos, d_pos, node_wl, node_d, True))

            # Board tokens for this node's position
            block_start = len(sequence)
            board_tokens = fen_to_position_tokens(node_fen)
            sequence.extend(board_tokens)
            block_boundaries.append((block_start, len(sequence)))

            # PV continuation move (predicted from stm of this board)
            node_move = node.get("move")
            if node_move and node_idx < len(nodes) - 1:
                # This move leads to the next node
                pv_move_model = _to_model_uci(node_move)
                if pv_move_model in token_to_idx:
                    predict_from_pos = len(sequence) - 1  # stm token of board
                    thinking_move_data.append((predict_from_pos, pv_move_model))
                    sequence.append(pv_move_model)

        # end_var token
        sequence.append("end_var")

    # 4. end_think token
    sequence.append("end_think")

    # 5. Final move (predicted from end_think via policy_head)
    final_move_model = _to_model_uci(mcts_action)
    if final_move_model not in token_to_idx:
        # Fallback: use played_move or best_move
        played = row.get("played_move", row.get("best_move", ""))
        final_move_model = _to_model_uci(played) if played else None

    end_think_pos = len(sequence) - 1
    final_move_data = (end_think_pos, final_move_model) if final_move_model and final_move_model in token_to_idx else None
    if final_move_model and final_move_model in token_to_idx:
        sequence.append(final_move_model)

    # 6. Final WL and D value tokens
    wl_pos = len(sequence)
    sequence.append("wl_value")
    d_pos = len(sequence)
    sequence.append("d_value")
    value_data.append((wl_pos, d_pos, final_wl, final_d, is_valid_wdl))

    # Convert to token IDs
    ids = [token_to_idx[t] for t in sequence]

    return ids, thinking_move_data, final_move_data, value_data, block_boundaries
