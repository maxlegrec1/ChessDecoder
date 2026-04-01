"""Vocabulary invariant tests (pure Python, no GPU)."""

from src.models.vocab import (
    vocab_size, board_vocab_size, move_vocab_size, POSITION_TOKEN_LENGTH,
    token_to_idx, idx_to_token,
    board_vocab, move_vocab, board_token_to_idx,
    board_idx_to_full_idx, move_idx_to_full_idx,
    full_idx_to_board_idx, full_idx_to_move_idx,
    policy_index,
)


def test_vocab_sizes():
    assert vocab_size >= 1967  # 1924 moves + pieces + special + castling + continuation
    assert board_vocab_size == 41
    assert move_vocab_size == 1924
    assert POSITION_TOKEN_LENGTH == 68


def test_token_to_idx_bijective():
    assert len(token_to_idx) == vocab_size
    assert len(idx_to_token) == vocab_size
    for tok, idx in token_to_idx.items():
        assert idx_to_token[idx] == tok


def test_board_vocab_subset_of_full():
    for tok in board_vocab:
        assert tok in token_to_idx, f"Board token '{tok}' not in full vocab"


def test_move_vocab_subset_of_full():
    for tok in move_vocab:
        assert tok in token_to_idx, f"Move token '{tok}' not in full vocab"


def test_board_idx_roundtrip():
    for i in range(board_vocab_size):
        full_idx = board_idx_to_full_idx[i]
        assert full_idx_to_board_idx[full_idx] == i


def test_move_idx_roundtrip():
    for i in range(move_vocab_size):
        full_idx = move_idx_to_full_idx[i]
        assert full_idx_to_move_idx[full_idx] == i


def test_special_tokens_exist():
    required = [
        "start_pos", "end_pos", "white_to_move", "black_to_move",
        "empty", "pad", "wl_value", "d_value",
        "start_think", "end_think", "end_var",
        "continue_var", "new_variation", "generic_move",
    ]
    for tok in required:
        assert tok in token_to_idx, f"Special token '{tok}' missing"


def test_board_move_vocabs_disjoint():
    board_full = set(board_idx_to_full_idx)
    move_full = set(move_idx_to_full_idx)
    assert board_full.isdisjoint(move_full)


def test_policy_index_length():
    assert len(policy_index) == 1924


def test_board_token_to_idx_consistent():
    for tok, idx in board_token_to_idx.items():
        assert board_vocab[idx] == tok
