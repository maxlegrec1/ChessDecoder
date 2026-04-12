"""Pure-Python tests for the tactics eval module (no GPU, no CSV download)."""

import csv

import pytest

from chessdecoder.eval.tactics import (
    Puzzle,
    PuzzleResult,
    aggregate,
    bucket_by_rating,
    bucket_by_theme,
    evaluate_puzzles,
    load_puzzles,
    wilson_ci,
)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


_SAMPLE_ROWS = [
    # PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays, Themes, GameUrl, OpeningTags
    # 2-ply mate-in-1
    ["P0001", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
     "h5f7 e8f7", "900", "80", "95", "1234", "mateIn1 opening short", "http://x", "Italian"],
    # 4-ply puzzle
    ["P0002", "rnb1kbnr/ppp2ppp/8/3q4/4p3/8/PPPPKPPP/RNBQ1BNR w kq - 0 4",
     "e2e3 d5e4 e3e2 e4e2", "1600", "60", "70", "500", "fork middlegame short", "http://y", ""],
    # Endgame
    ["P0003", "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
     "e2e4 e5d6 e3d4 d6e6 e4e5 e6f7", "2000", "40", "85", "300", "endgame pawnEndgame long", "http://z", ""],
]

_HEADER = ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation",
           "Popularity", "NbPlays", "Themes", "GameUrl", "OpeningTags"]


@pytest.fixture
def puzzle_csv(tmp_path):
    p = tmp_path / "puz.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_HEADER)
        w.writerows(_SAMPLE_ROWS)
    return str(p)


def test_load_all(puzzle_csv):
    puzzles = load_puzzles(puzzle_csv)
    assert len(puzzles) == 3
    ids = {p.puzzle_id for p in puzzles}
    assert ids == {"P0001", "P0002", "P0003"}


def test_load_rating_filter(puzzle_csv):
    low = load_puzzles(puzzle_csv, rating_range=(0, 1000))
    assert [p.puzzle_id for p in low] == ["P0001"]
    mid = load_puzzles(puzzle_csv, rating_range=(1000, 1800))
    assert [p.puzzle_id for p in mid] == ["P0002"]


def test_load_themes_any(puzzle_csv):
    mate = load_puzzles(puzzle_csv, themes_any={"mateIn1"})
    assert [p.puzzle_id for p in mate] == ["P0001"]
    multi = load_puzzles(puzzle_csv, themes_any={"fork", "endgame"})
    assert {p.puzzle_id for p in multi} == {"P0002", "P0003"}


def test_load_themes_all(puzzle_csv):
    both = load_puzzles(puzzle_csv, themes_all={"endgame", "pawnEndgame"})
    assert [p.puzzle_id for p in both] == ["P0003"]
    none = load_puzzles(puzzle_csv, themes_all={"endgame", "mateIn1"})
    assert none == []


def test_load_themes_none(puzzle_csv):
    no_mate = load_puzzles(puzzle_csv, themes_none={"mateIn1"})
    assert {p.puzzle_id for p in no_mate} == {"P0002", "P0003"}


def test_load_max_puzzles_is_deterministic_under_seed(puzzle_csv):
    a = load_puzzles(puzzle_csv, max_puzzles=2, seed=42)
    b = load_puzzles(puzzle_csv, max_puzzles=2, seed=42)
    c = load_puzzles(puzzle_csv, max_puzzles=2, seed=1)
    assert [p.puzzle_id for p in a] == [p.puzzle_id for p in b]
    # different seed may or may not pick different puzzles for 2/3, just require
    # it returned the right count
    assert len(c) == 2


def test_solver_moves_and_num_plies():
    p = Puzzle(
        puzzle_id="x", fen="", moves=["o0", "s1", "o1", "s2", "o2", "s3"],
        rating=0, rating_deviation=0, themes=frozenset(),
    )
    assert p.solver_moves == ["s1", "s2", "s3"]
    assert p.num_plies == 3


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mk_result(pid, rating, themes, ply_correct):
    p = Puzzle(pid, "", ["x"] * (2 * len(ply_correct) + 1),
               rating, 0, frozenset(themes))
    return PuzzleResult(puzzle=p, ply_correct=ply_correct,
                        model_moves=["" for _ in ply_correct])


def test_aggregate_basic():
    results = [
        _mk_result("a", 1000, [], [True]),          # first right, only ply
        _mk_result("b", 1000, [], [True, True]),    # both right
        _mk_result("c", 1000, [], [False, True]),   # first wrong, second right
        _mk_result("d", 1000, [], [True, False]),   # first right, second wrong
    ]
    agg = aggregate(results)
    assert agg.n_puzzles == 4
    # first_move_acc = 3/4
    assert abs(agg.first_move_acc - 0.75) < 1e-9
    # strict_solve = 2/4 (a and b)
    assert abs(agg.strict_solve_rate - 0.5) < 1e-9
    # per_ply_acc = 5/7
    assert abs(agg.ply_acc - 5 / 7) < 1e-9
    # CI bounds are valid probabilities and include the point estimate
    for lo, hi, p in [
        (*agg.first_move_ci, agg.first_move_acc),
        (*agg.strict_solve_ci, agg.strict_solve_rate),
        (*agg.ply_ci, agg.ply_acc),
    ]:
        assert 0.0 <= lo <= p <= hi <= 1.0


def test_wilson_ci_on_boundaries():
    lo, hi = wilson_ci(0, 0)
    assert (lo, hi) == (0.0, 0.0)
    lo, hi = wilson_ci(100, 100)
    assert hi == pytest.approx(1.0, abs=1e-3)
    assert lo > 0.95  # Wilson lower bound for 100/100


def test_bucket_by_rating():
    rs = [
        _mk_result("a", 500, [], [True]),
        _mk_result("b", 1500, [], [False]),
        _mk_result("c", 1501, [], [True]),
        _mk_result("d", 2999, [], [False]),
    ]
    buckets = bucket_by_rating(rs, [0, 1000, 1500, 2000, 3000])
    # Buckets in order: [0,1000), [1000,1500), [1500,2000), [2000,3000)
    assert len(buckets[0][1]) == 1 and buckets[0][1][0].puzzle.puzzle_id == "a"
    assert len(buckets[1][1]) == 0  # 1500 goes to next bucket (boundary is inclusive-left)
    assert {r.puzzle.puzzle_id for r in buckets[2][1]} == {"b", "c"}
    assert [r.puzzle.puzzle_id for r in buckets[3][1]] == ["d"]


def test_bucket_by_theme_filters_and_orders():
    rs = [
        _mk_result("a", 1000, ["fork"], [True]),
        _mk_result("b", 1000, ["pin", "endgame"], [True]),
        _mk_result("c", 1000, ["skewer"], [False]),
    ]
    buckets = bucket_by_theme(rs, ["fork", "pin"])
    assert [t for t, _ in buckets] == ["fork", "pin"]
    assert [r.puzzle.puzzle_id for r in buckets[0][1]] == ["a"]
    assert [r.puzzle.puzzle_id for r in buckets[1][1]] == ["b"]


# ---------------------------------------------------------------------------
# evaluate_puzzles with a stub engine (no GPU)
# ---------------------------------------------------------------------------


class _StubResult:
    def __init__(self, move): self.move = move


class _StubEngine:
    """Always returns the move dictated by a per-FEN lookup table."""
    def __init__(self, fen_to_move: dict[str, str]):
        self._map = fen_to_move

    def predict_moves(self, fens, _temp):
        return [_StubResult(self._map.get(f, "")) for f in fens]


def _make_mate_in_1_puzzle():
    """Opponent already moved into checkmate setup; solver delivers mate.

    Position: black to move has played, white to checkmate with Qxf7#.
    Starting FEN has white to move -- use the simple "scholar's mate" line.
    Use a position where the mate move is unambiguous.
    """
    # White to move; Qh5-f7 is mate (Scholar's-like setup).
    fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
    # `FEN` field in the Lichess CSV is the position BEFORE moves[0].
    # We want moves[0] to be an opponent move that is already legal in this
    # position and moves[1] to be the solver's mate reply.
    # Simpler: use a puzzle where move 0 is something legal from `fen` and
    # move 1 is a legal response after it.
    pre_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 4 4"
    # moves[0] = "c6d4" (black plays something)
    # moves[1] = "h5f7" (white mates) -- but white queen at h5 needs Qxf7 to be mate;
    # with knight on d4 this may not be mate. For a test we just need legality.
    return Puzzle(
        puzzle_id="T1",
        fen=pre_fen,
        moves=["c6d4", "h5f7"],  # opp setup then solver move
        rating=1000,
        rating_deviation=0,
        themes=frozenset({"mateIn1"}),
    )


def test_evaluate_puzzles_correct_move():
    p = _make_mate_in_1_puzzle()
    # After opp moves c6d4, solver plays h5f7. Board FEN after c6d4:
    import chess
    b = chess.Board(p.fen)
    b.push_uci("c6d4")
    solver_fen = b.fen()

    engine = _StubEngine({solver_fen: "h5f7"})
    results = evaluate_puzzles(engine, [p], batch_size=4, progress=False)
    assert len(results) == 1
    r = results[0]
    assert r.ply_correct == [True]
    assert r.first_move_correct
    assert r.fully_solved


def test_evaluate_puzzles_wrong_move_still_graded():
    p = _make_mate_in_1_puzzle()
    import chess
    b = chess.Board(p.fen)
    b.push_uci("c6d4")
    solver_fen = b.fen()

    # Model picks a different, legal but wrong move.
    engine = _StubEngine({solver_fen: "c4f7"})  # also legal Bxf7+ from Italian-like
    results = evaluate_puzzles(engine, [p], batch_size=4, progress=False)
    assert results[0].ply_correct == [False]
    assert not results[0].fully_solved


def test_evaluate_puzzles_illegal_engine_move():
    """Model outputs a completely invalid move; should count as incorrect."""
    p = _make_mate_in_1_puzzle()
    engine = _StubEngine({})  # returns "" for every FEN
    results = evaluate_puzzles(engine, [p], batch_size=4, progress=False)
    assert results[0].ply_correct == [False]
    assert results[0].model_moves == [""]
