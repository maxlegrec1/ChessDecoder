"""FEN conversion and game sequence tests (pure Python, no GPU)."""

import pandas as pd
import pytest

from src.dataloader.data import fen_to_position_tokens, game_to_token_ids
from src.models.vocab import (
    token_to_idx, vocab_size, castling_tokens, POSITION_TOKEN_LENGTH,
)
from tests.conftest import SAMPLE_FENS


@pytest.mark.parametrize("fen", SAMPLE_FENS)
def test_fen_to_position_tokens_length(fen):
    tokens = fen_to_position_tokens(fen)
    assert len(tokens) == POSITION_TOKEN_LENGTH


@pytest.mark.parametrize("fen", SAMPLE_FENS)
def test_fen_to_position_tokens_structure(fen):
    tokens = fen_to_position_tokens(fen)
    assert tokens[0] == "start_pos"
    assert tokens[65] == "end_pos"
    assert tokens[66] in castling_tokens or tokens[66] == "no_castling_rights"
    assert tokens[67] in ("white_to_move", "black_to_move")


@pytest.mark.parametrize("fen", SAMPLE_FENS)
def test_fen_all_tokens_in_vocab(fen):
    tokens = fen_to_position_tokens(fen)
    for tok in tokens:
        assert tok in token_to_idx, f"Token '{tok}' not in vocab"


def test_fen_starting_position_pieces():
    tokens = fen_to_position_tokens(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    # a1 = index 1 (rank 0, file 0)
    assert tokens[1] == "white_rook"
    # e1 = index 5
    assert tokens[5] == "white_king"
    # a2 = index 9 (rank 1, file 0)
    assert tokens[9] == "white_pawn"
    # e4 = rank 3, file 4 = index 1 + 3*8 + 4 = 29
    assert tokens[29] == "empty"
    # a8 = rank 7, file 0 = index 1 + 7*8 = 57
    assert tokens[57] == "black_rook"
    # Side to move
    assert tokens[67] == "white_to_move"
    assert tokens[66] == "KQkq"


def test_fen_side_to_move():
    w_tokens = fen_to_position_tokens(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    )
    assert w_tokens[67] == "black_to_move"


def test_fen_no_castling():
    tokens = fen_to_position_tokens("8/5pk1/6p1/8/8/2B5/5PPP/6K1 w - - 0 40")
    assert tokens[66] == "no_castling_rights"


def test_game_to_token_ids_basic():
    data = [
        {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
         "played_move": "e2e4", "best_move": "e2e4",
         "win": 0.3, "draw": 0.5, "loss": 0.2, "ply": 0},
        {"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
         "played_move": "e7e5", "best_move": "e7e5",
         "win": 0.4, "draw": 0.4, "loss": 0.2, "ply": 1},
    ]
    game_df = pd.DataFrame(data)
    ids, wdl_data, block_boundaries, value_data = game_to_token_ids(game_df)

    # All IDs valid
    for tid in ids:
        assert 0 <= tid < vocab_size

    # Block boundaries are 68-token board blocks
    for start, end in block_boundaries:
        assert ids[start] == token_to_idx["start_pos"]

    # Value positions: wl = move+1, d = move+2
    for wl_pos, d_pos, wl, d, valid in value_data:
        assert d_pos == wl_pos + 1
        assert ids[wl_pos] == token_to_idx["wl_value"]
        assert ids[d_pos] == token_to_idx["d_value"]
