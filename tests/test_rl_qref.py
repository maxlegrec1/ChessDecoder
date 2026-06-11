"""PUCT sanity: sign conventions + tactics. A sign error anywhere makes
search WORSE than greedy, so mate-finding is the load-bearing test."""
import chess
import pytest

pytestmark = [pytest.mark.gpu]

MATE_IN_1 = [
    # (fen, mating move)
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8"),     # back rank
    ("r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1", "a8a1"),     # same, black
]


@pytest.fixture(scope="module")
def engine():
    from chessdecoder.agent.rl.oracle_engine import OracleEngine
    return OracleEngine()


def test_mate_in_1(engine):
    from chessdecoder.agent.rl.qref import search_batch
    roots = [chess.Board(f) for f, _ in MATE_IN_1]
    res = search_batch(engine, roots, sims=200)
    for (fen, mate), r in zip(MATE_IN_1, res):
        assert r.search_best == mate, f"{fen}: best {r.search_best} != {mate}"
        i = r.moves.index(mate)
        assert r.q[i] > 0.95, f"{fen}: mate Q {r.q[i]}"


def test_search_improves_visits(engine):
    """Visits concentrate (search is selective, not uniform)."""
    from chessdecoder.agent.rl.qref import search_batch
    b = chess.Board()
    r = search_batch(engine, [b], sims=200)[0]
    top = max(r.visits) / sum(r.visits)
    assert top > 0.15, f"flat visit distribution: {top}"


def test_q_monotone_with_sims(engine):
    """More sims must not flip an overwhelming-material position negative."""
    from chessdecoder.agent.rl.qref import search_batch
    fen = "6k1/8/8/8/8/8/PPP2PPP/RNBQKBNR w KQ - 0 1"   # white up a army
    b = chess.Board(fen)
    for sims in (50, 200):
        r = search_batch(engine, [b], sims=sims)[0]
        i = r.moves.index(r.search_best)
        assert r.q[i] > 0.3, f"sims={sims}: q {r.q[i]}"
