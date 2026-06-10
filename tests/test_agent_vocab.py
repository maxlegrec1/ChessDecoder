"""Round-trip + edge-case tests for the agent patch vocabulary."""
import chess
import pytest

from chessdecoder.agent import patch_vocab as pv

EDGE_FENS = [
    chess.STARTING_FEN,
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",                       # full castling
    "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 3 40",                        # partial rights
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",  # ep set
    "8/P7/8/8/8/8/p7/4K2k w - - 0 1",                              # promotions pending
    "8/5pk1/6p1/8/8/2B5/5PPP/6K1 w - - 0 40",                      # sparse endgame
    "8/8/8/8/8/8/8/K6k b - - 99 120",                              # bare kings
    "r1bq1rk1/pp2nppp/2n1p3/3pP3/3P4/2N2N2/PP2BPPP/R1BQ1RK1 b - - 1 10",
]


@pytest.mark.parametrize("fen", EDGE_FENS)
def test_roundtrip_edge(fen):
    b = chess.Board(fen)
    ids = pv.encode_board(b)
    assert len(ids) == pv.BOARD_LEN
    b2 = pv.decode_board(ids)
    assert b2 is not None
    assert b2.board_fen() == b.board_fen()
    assert b2.turn == b.turn
    assert b2.castling_rights == b.castling_rights
    assert b2.ep_square == b.ep_square


def test_roundtrip_random_games():
    import random
    random.seed(0)
    b = chess.Board()
    checked = 0
    for _ in range(300):
        if b.is_game_over():
            b = chess.Board()
        b.push(random.choice(list(b.legal_moves)))
        ids = pv.encode_board(b)
        b2 = pv.decode_board(ids)
        assert b2.board_fen() == b.board_fen()
        assert b2.castling_rights == b.castling_rights
        assert b2.ep_square == b.ep_square
        assert b2.turn == b.turn
        checked += 1
    assert checked == 300


def test_vocab_regions_disjoint():
    assert pv.VOCAB_SIZE == pv.CTRL_BASE + pv.N_CTRL
    ids = pv.encode_board(chess.Board())
    for i, tok in enumerate(ids[:16]):
        assert pv.region_of(tok) == "patch", i
    assert pv.region_of(ids[16]) == "castle"
    assert pv.region_of(ids[17]) == "stm"
    assert pv.region_of(ids[18]) == "ep"
    assert pv.region_of(pv.MOVE_TO_ID["e2e4"]) == "move"
    assert pv.region_of(pv.qbin_token(0.0)) == "qbin"
    assert pv.region_of(pv.PAD) == "ctrl"


def test_bins():
    assert pv.q_to_bin(-1.0) == 0
    assert pv.q_to_bin(1.0) == pv.N_QBIN - 1
    assert pv.q_to_bin(0.0) == pv.N_QBIN // 2
    assert pv.d_to_bin(0.0) == 0
    assert pv.d_to_bin(1.0) == pv.N_DBIN - 1


def test_decode_rejects_wrong_region():
    ids = pv.encode_board(chess.Board())
    bad = list(ids)
    bad[3] = pv.MOVE_TO_ID["e2e4"]          # move token in a patch slot
    assert pv.decode_board(bad) is None


def test_castling_move_keys():
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    mv = chess.Move.from_uci("e1g1")
    keys = pv.move_keys(b, mv)
    assert "e1g1" in keys and "e1h1" in keys
