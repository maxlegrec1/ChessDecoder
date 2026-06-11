"""CPU unit tests: rewarder scoring on hand-built episodes; buffer round-trip,
staleness drop, atomic weight publish."""
import numpy as np
import chess
import pytest
import torch

from chessdecoder.agent import patch_vocab as pv
from chessdecoder.agent.rl.episodes import Episode
from chessdecoder.agent.rl.reward import (RootRef, move_id_to_uci,
                                          score_episode)
from chessdecoder.agent.rl.buffer import (GroupBuffer, load_weights_if_newer,
                                          publish_weights)


def _ref():
    return RootRef(moves=["e2e4", "d2d4", "g1f3"],
                   q=np.array([0.30, 0.10, -0.05], dtype=np.float32),
                   oracle_greedy="d2d4", search_best="e2e4",
                   corpus_best="e2e4")


def _ep(uci, invalid=0):
    root = chess.Board()
    mid = None
    for mv in root.legal_moves:
        if mv.uci() == uci:
            mid = pv.MOVE_TO_ID[pv.move_keys(root, mv)[0]]
    e = Episode(root_fen=root.fen(), k_budget=4)
    e.final_move = mid
    e.probes_valid, e.probes_invalid = 2, invalid
    e.done = True
    return e, root


def test_score_beats_greedy():
    e, root = _ep("e2e4")
    s = score_episode(e, _ref(), root)
    assert s["reward"] == pytest.approx(0.0)        # picked the best move
    assert s["beat_greedy"] and s["match_search_best"] and s["match_corpus_best"]


def test_score_greedy_regret():
    e, root = _ep("d2d4")
    s = score_episode(e, _ref(), root)
    assert s["regret"] == pytest.approx(-0.20)
    assert not s["beat_greedy"] and s["match_greedy"]


def test_invalid_probe_penalty():
    e, root = _ep("e2e4", invalid=3)
    s = score_episode(e, _ref(), root, invalid_eps=0.01)
    assert s["reward"] == pytest.approx(-0.03)
    assert s["regret"] == pytest.approx(0.0)


def test_move_id_castling():
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    mv = chess.Move.from_uci("e1g1")
    for k in pv.move_keys(b, mv):
        if k in pv.MOVE_TO_ID:
            assert move_id_to_uci(b, pv.MOVE_TO_ID[k]) == "e1g1"


def test_buffer_roundtrip(tmp_path):
    buf = GroupBuffer(str(tmp_path))
    buf.write({"x": torch.arange(3), "v": 1}, version=1)
    buf.write({"x": torch.arange(4), "v": 2}, version=2)
    assert buf.depth() == 2
    groups, dropped = buf.consume(5, current_version=2, max_staleness=10)
    assert len(groups) == 2 and dropped == 0
    assert groups[0]["v"] == 1                       # oldest first
    assert buf.depth() == 0


def test_buffer_staleness_drop(tmp_path):
    buf = GroupBuffer(str(tmp_path))
    buf.write({"v": 1}, version=1)
    buf.write({"v": 9}, version=9)
    groups, dropped = buf.consume(5, current_version=10, max_staleness=3)
    assert dropped == 1 and len(groups) == 1 and groups[0]["v"] == 9


def test_weight_publish(tmp_path):
    d = str(tmp_path)
    sd = {"w": torch.randn(4, 4)}
    publish_weights(d, sd, version=3)
    assert load_weights_if_newer(d, have_version=3) is None
    out = load_weights_if_newer(d, have_version=2)
    assert out is not None
    sd2, v = out
    assert v == 3 and sd2["w"].dtype == torch.bfloat16
    assert torch.allclose(sd2["w"].float(), sd["w"], atol=0.01)
