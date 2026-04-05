"""Modular reward functions for GRPO reinforcement learning."""

from typing import Callable

from chessdecoder.models.vocab import token_to_idx, idx_to_token, move_vocab_size, POSITION_TOKEN_LENGTH
from chessdecoder.finetune.cpp_eval import _normalize_castling


_START_THINK = token_to_idx["start_think"]
_END_THINK = token_to_idx["end_think"]
_END_VAR = token_to_idx["end_var"]
_START_POS = token_to_idx["start_pos"]
_WL_VALUE = token_to_idx["wl_value"]
_D_VALUE = token_to_idx["d_value"]


def _is_move_token(tok_id: int) -> bool:
    """Move tokens occupy indices 0..move_vocab_size-1 (the policy_index portion of vocab)."""
    return 0 <= tok_id < move_vocab_size


# ---------------------------------------------------------------------------
# Individual reward functions
# ---------------------------------------------------------------------------

def move_quality_reward(
    final_move: str,
    token_ids: list[int],
    ground_truth: dict,
) -> float:
    """Reward for final move quality.

    +1.0 if final_move == best_move
     0.0 otherwise
    """
    move = _normalize_castling(final_move)
    best = _normalize_castling(ground_truth["best_move"])
    return 1.0 if move == best else 0.0


def format_reward(
    final_move: str,
    token_ids: list[int],
    ground_truth: dict,
) -> float:
    """Reward for well-formed thinking structure.

    +1.0 if valid: has start_think, end_think, every variation ends with end_var,
         board blocks are 68 tokens.
    -0.5 if truncated (no end_think).
     0.0 if structurally malformed.
    """
    if _START_THINK not in token_ids:
        return 0.0

    try:
        st_pos = token_ids.index(_START_THINK)
    except ValueError:
        return 0.0

    has_end_think = _END_THINK in token_ids
    if not has_end_think:
        return -0.5

    et_pos = token_ids.index(_END_THINK)

    # Check that between start_think and end_think, every variation ends with end_var
    # Walk through the thinking region and validate structure
    thinking_region = token_ids[st_pos + 1:et_pos]

    # Must have at least one end_var
    if _END_VAR not in thinking_region:
        return 0.0

    # Validate board blocks are exactly POSITION_TOKEN_LENGTH (68) tokens
    j = 0
    while j < len(thinking_region):
        if thinking_region[j] == _START_POS:
            block_end = j + POSITION_TOKEN_LENGTH
            if block_end > len(thinking_region):
                return 0.0  # Incomplete board block
            j = block_end
        else:
            j += 1

    return 1.0


def coherence_reward(
    final_move: str,
    token_ids: list[int],
    ground_truth: dict,
) -> float:
    """Membership reward: was the final move explored during thinking?

    +1.0 if final_move matches ANY root move in the thinking variations.
     0.0 otherwise.

    Root moves are the move tokens immediately after start_think or end_var.
    """
    if _START_THINK not in token_ids or _END_THINK not in token_ids:
        return 0.0

    st_pos = token_ids.index(_START_THINK)
    et_pos = token_ids.index(_END_THINK)

    # Collect root moves: first move token after start_think or end_var
    root_moves: set[str] = set()
    i = st_pos + 1
    while i < et_pos:
        tok = token_ids[i]
        if _is_move_token(tok):
            root_moves.add(_normalize_castling(idx_to_token[tok]))
            # Skip to end of this variation (next end_var)
            i += 1
            while i < et_pos and token_ids[i] != _END_VAR:
                i += 1
            i += 1  # skip end_var itself
        else:
            i += 1

    return 1.0 if _normalize_castling(final_move) in root_moves else 0.0


# ---------------------------------------------------------------------------
# Composite reward
# ---------------------------------------------------------------------------

# Signature: (final_move, token_ids, ground_truth) -> float
RewardFn = Callable[[str, list[int], dict], float]

_REWARD_REGISTRY: dict[str, RewardFn] = {
    "move_quality": move_quality_reward,
    "format": format_reward,
    "coherence": coherence_reward,
}


class CompositeReward:
    """Weighted combination of reward functions."""

    def __init__(self, weights: dict[str, float]):
        self.weights = weights
        for name in weights:
            if name not in _REWARD_REGISTRY:
                raise ValueError(f"Unknown reward function: {name}")

    def __call__(
        self,
        final_move: str,
        token_ids: list[int],
        ground_truth: dict,
    ) -> tuple[float, dict[str, float]]:
        """Compute weighted total reward and per-component breakdown."""
        components = {}
        total = 0.0
        for name, weight in self.weights.items():
            value = _REWARD_REGISTRY[name](final_move, token_ids, ground_truth)
            components[name] = value
            total += weight * value
        return total, components
