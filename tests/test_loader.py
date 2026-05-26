"""Dataloader shape + sampling correctness (CPU)."""
import pandas as pd
import pytest
import torch

from chessdecoder.dataloader.loader import game_to_arrays, IGNORE_INDEX
from chessdecoder.models.value_buckets import N_CELLS


def _row(fen, played="e2e4", best="e2e4", q=0.1, d=0.5):
    return {"fen": fen, "played_move": played, "best_move": best,
            "orig_q": q, "orig_d": d, "ply": 0, "game_id": "g"}


def _gdf(n_rows=20):
    fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * n_rows
    return pd.DataFrame([_row(f) for f in fens])


@pytest.mark.parametrize("N", [1, 4, 8, 16])
def test_emits_exact_N_positions(N):
    s = game_to_arrays(_gdf(20), positions_per_game=N)
    assert s["board_ids"].shape == (N, 68)
    assert s["policy_tgt"].shape == (N,)
    assert s["wdl_tgt"].shape == (N, N_CELLS)
    assert s["wdl_mean"].shape == (N, 3)


def test_padding_when_game_too_short():
    # game has 3 rows, request 8 -> with-replacement sampling fills [8,...]
    s = game_to_arrays(_gdf(3), positions_per_game=8)
    assert s["board_ids"].shape == (8, 68)
    assert s["policy_valid"].all()
    assert s["wdl_valid"].all()


def test_skips_empty_games():
    df = pd.DataFrame([{"fen": "8/8/8/8/8/8/8/8 w - - 0 1", "played_move": None,
                        "best_move": None, "orig_q": None, "orig_d": None,
                        "ply": 0, "game_id": "g"}])
    assert game_to_arrays(df, positions_per_game=4) is None


def test_invalid_best_move_marks_policy_invalid():
    df = pd.DataFrame([_row("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                            played="e2e4", best="not-a-move")])
    s = game_to_arrays(df, positions_per_game=1)
    assert not s["policy_valid"][0]
    assert s["policy_tgt"][0] == IGNORE_INDEX


def test_wdl_target_rows_sum_to_one():
    s = game_to_arrays(_gdf(8), positions_per_game=8)
    assert torch.allclose(s["wdl_tgt"].sum(-1), torch.ones(8), atol=1e-5)
    assert torch.allclose(s["wdl_mean"].sum(-1), torch.ones(8), atol=1e-5)
