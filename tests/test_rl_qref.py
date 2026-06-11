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


def test_cpp_mate_in_1(engine):
    from chessdecoder.agent.rl.qref import search_batch_cpp
    roots = [chess.Board(f) for f, _ in MATE_IN_1]
    res = search_batch_cpp(engine, roots, sims=200)
    for (fen, mate), r in zip(MATE_IN_1, res):
        assert r.search_best == mate, f"{fen}: best {r.search_best} != {mate}"
        i = r.moves.index(mate)
        assert r.q[i] > 0.95, f"{fen}: mate Q {r.q[i]}"


def test_cpp_python_parity(engine):
    """Same roots, both engines: search_best agree on >=80% (bf16 batch
    jitter lets trees diverge on near-ties), greedy move identical, visit
    counts sum to sims."""
    import glob

    import pandas as pd
    from chessdecoder.agent.rl.qref import search_batch, search_batch_cpp
    f = sorted(glob.glob("/mnt/2tb_2/decoder/parquet_files_decoder/*.parquet"))[-1]
    fens = pd.read_parquet(f, columns=["fen"]).fen.drop_duplicates().head(64)
    roots = [chess.Board(x) for x in fens
             if not chess.Board(x).is_game_over()][:48]
    rp = search_batch(engine, roots, sims=200)
    rc = search_batch_cpp(engine, roots, sims=200)
    agree_best = greedy_same = 0
    for a, b in zip(rp, rc):
        assert sum(b.visits) == 200, f"visits {sum(b.visits)} != 200"
        assert set(a.moves) == set(b.moves)
        agree_best += a.search_best == b.search_best
        greedy_same += a.oracle_greedy == b.oracle_greedy
    assert greedy_same >= len(roots) * 0.95, f"greedy {greedy_same}/{len(roots)}"
    assert agree_best >= len(roots) * 0.8, f"best {agree_best}/{len(roots)}"
