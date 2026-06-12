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


def test_batch_positions_synthetic():
    """Trainer batch assembly on a synthetic 2-episode group: shapes, mask
    rows, advantage broadcast, zero-variance skip."""
    from chessdecoder.agent.rl.grpo import _batch_positions
    from chessdecoder.agent.rl.episodes import PREFIX_LEN

    root = chess.Board()
    # minimal episode: prefix + <answer> move  (k_budget=1, no probes)
    mv = next(iter(root.legal_moves))
    mid = pv.MOVE_TO_ID[pv.move_keys(root, mv)[0]]
    ids = ([pv.ROOT] + pv.encode_board(root)
           + [pv.ORACLE, pv.QBIN_BASE, pv.DBIN_BASE, mid, mid, mid, mid]
           + [pv.num_token(1)] + [pv.ANSWER, mid])
    assert len(ids) == PREFIX_LEN + 2
    g = dict(root_fen=root.fen(), version=1, temperature=1.0, k_budget=1,
             ids=torch.tensor([ids, ids], dtype=torch.int32),
             logprobs=torch.zeros(2, len(ids)),
             agent=torch.zeros(2, len(ids), dtype=torch.bool),
             rewards=torch.tensor([0.0, -0.5]),
             metrics=[], gen_seconds=0.0, tokens=0)
    zero_g = dict(g, rewards=torch.tensor([-0.1, -0.1]))
    out = _batch_positions([g, zero_g])
    ids_t, pos_b, pos_t, masks, adv, beh, n_eps, zero_var, fam, w = out
    # zero-var group now KEPT with adv=0 (entropy still applies): 4 episodes
    assert zero_var == 1 and n_eps == 4
    assert fam.tolist() == [1, 2] * 4
    assert pos_t.tolist() == [PREFIX_LEN, PREFIX_LEN + 1] * 4
    assert masks.shape[1] == pv.VOCAB_SIZE
    assert masks[0][pv.ANSWER] and masks[0][pv.PROBE]
    assert masks[1][mid] and not masks[1][pv.ANSWER]
    # advantages: g episodes nonzero opposite-signed, zero_g episodes zero
    assert adv[0] == adv[1] and adv[2] == adv[3]
    assert adv[0] > 0 > adv[2]
    assert adv[4] == adv[5] == adv[6] == adv[7] == 0.0
    # per-episode weights: 2 positions each -> 0.5
    assert torch.allclose(w, torch.full((8,), 0.5))


def test_corpus_bonus():
    """Sparse bonus tips near-ties toward the external judge; large Q_ref
    gaps still override it. Also castling-spelling normalization."""
    from chessdecoder.agent.rl.reward import _norm_corpus_best
    ref = _ref()                      # corpus_best = e2e4 (also search best)
    e, root = _ep("e2e4")
    s = score_episode(e, ref, root, corpus_bonus=0.1)
    assert s["reward"] == pytest.approx(0.1)        # regret 0 + bonus
    e, root = _ep("d2d4")
    s = score_episode(e, ref, root, corpus_bonus=0.1)
    assert s["reward"] == pytest.approx(-0.20)      # no bonus, regret only
    # exchange rate: corpus_best with regret -0.05 nets +0.05 > search_best 0
    ref2 = _ref()
    ref2.corpus_best = "d2d4"
    ref2.q = np.array([0.30, 0.25, -0.05], dtype=np.float32)
    e, root = _ep("d2d4")
    s = score_episode(e, ref2, root, corpus_bonus=0.1)
    assert s["reward"] == pytest.approx(0.05)
    assert _norm_corpus_best("e1h1", ["e1g1", "a2a3"]) == "e1g1"
    assert _norm_corpus_best("e7e5", ["e1g1", "a2a3"]) is None
