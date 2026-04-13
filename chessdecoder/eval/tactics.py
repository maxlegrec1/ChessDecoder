"""Lichess-puzzle tactics accuracy eval for thinking chess models.

The Lichess puzzle CSV format (from database.lichess.org):

    PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays,
    Themes, GameUrl, OpeningTags

`FEN` is the position BEFORE the opponent plays the setup move that
creates the tactic. `Moves` is a space-separated UCI sequence starting
with that setup move. So:

    moves[0]       = opponent's setup move (applied, then the solver acts)
    moves[1,3,5..] = solver moves (what we're grading the model on)
    moves[2,4,6..] = opponent's forced responses

Evaluation plays the canonical line move-by-move, always advancing the
board with the EXPECTED solver move (not the model's guess) so that we
can continue to grade subsequent plies even after a mistake. Each ply
is recorded as correct / incorrect, and per-puzzle aggregations
(first_move, strict solve, per-ply rate) are derived from that trace.

Strict match is used: the model's move must equal the move recorded in
the CSV. Lichess puzzles occasionally accept alternative solver moves
of equal strength, but the CSV only encodes one canonical line, so a
small rate of "false negatives" is expected. Reported numbers should be
read as a consistent comparative proxy, not absolute puzzle strength.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import chess

from chessdecoder.eval.stats import wilson_ci

from chessdecoder.utils.uci import normalize_castling


# --------------------------------------------------------------------------
# Puzzle data class and CSV loading
# --------------------------------------------------------------------------


@dataclass
class Puzzle:
    puzzle_id: str
    fen: str               # position BEFORE moves[0]
    moves: list[str]       # [opp_setup, solver1, opp1, solver2, opp2, ...]
    rating: int
    rating_deviation: int
    themes: frozenset[str]
    popularity: int = 0
    nb_plays: int = 0

    @property
    def solver_moves(self) -> list[str]:
        """UCI moves the model is graded on."""
        return self.moves[1::2]

    @property
    def num_plies(self) -> int:
        return len(self.solver_moves)


def _matches_filters(
    rating: int,
    themes: frozenset[str],
    popularity: int | None,
    nb_plays: int | None,
    rating_range: tuple[int, int] | None,
    themes_any: set[str] | None,
    themes_all: set[str] | None,
    themes_none: set[str] | None,
    min_popularity: int | None,
    min_nb_plays: int | None,
) -> bool:
    if rating_range and not (rating_range[0] <= rating <= rating_range[1]):
        return False
    if themes_any and not (themes & themes_any):
        return False
    if themes_all and not themes_all.issubset(themes):
        return False
    if themes_none and (themes & themes_none):
        return False
    if min_popularity is not None and (popularity or 0) < min_popularity:
        return False
    if min_nb_plays is not None and (nb_plays or 0) < min_nb_plays:
        return False
    return True


def load_puzzles(
    csv_path: str | Path,
    *,
    rating_range: tuple[int, int] | None = None,
    themes_any: set[str] | None = None,
    themes_all: set[str] | None = None,
    themes_none: set[str] | None = None,
    min_popularity: int | None = None,
    min_nb_plays: int | None = None,
    max_puzzles: int | None = None,
    seed: int = 42,
) -> list[Puzzle]:
    """Stream the Lichess puzzle CSV, filter, then random-sample.

    Uses reservoir sampling when `max_puzzles` is set so memory is bounded
    regardless of how many puzzles match the filters.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/download_puzzles.sh first."
        )

    rng = random.Random(seed)
    reservoir: list[Puzzle] = []
    n_matches = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rating = int(row["Rating"])
                popularity = int(row.get("Popularity", 0) or 0)
                nb_plays = int(row.get("NbPlays", 0) or 0)
                rating_deviation = int(row.get("RatingDeviation", 0) or 0)
            except (ValueError, KeyError):
                continue

            themes = frozenset(t for t in (row.get("Themes") or "").split() if t)

            if not _matches_filters(
                rating, themes, popularity, nb_plays,
                rating_range, themes_any, themes_all, themes_none,
                min_popularity, min_nb_plays,
            ):
                continue

            moves = (row.get("Moves") or "").split()
            if len(moves) < 2:
                continue

            puzzle = Puzzle(
                puzzle_id=row["PuzzleId"],
                fen=row["FEN"],
                moves=moves,
                rating=rating,
                rating_deviation=rating_deviation,
                themes=themes,
                popularity=popularity,
                nb_plays=nb_plays,
            )

            if max_puzzles is None:
                reservoir.append(puzzle)
            else:
                if n_matches < max_puzzles:
                    reservoir.append(puzzle)
                else:
                    j = rng.randint(0, n_matches)
                    if j < max_puzzles:
                        reservoir[j] = puzzle
            n_matches += 1

    if max_puzzles is None:
        rng.shuffle(reservoir)

    return reservoir


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@dataclass
class PuzzleResult:
    puzzle: Puzzle
    ply_correct: list[bool]      # length = num_plies
    model_moves: list[str]       # what the model picked at each ply
    error: str = ""              # non-empty if the puzzle was aborted

    @property
    def first_move_correct(self) -> bool:
        return bool(self.ply_correct and self.ply_correct[0])

    @property
    def fully_solved(self) -> bool:
        return bool(self.ply_correct) and all(self.ply_correct)

    @property
    def plies_correct(self) -> int:
        return sum(self.ply_correct)


def evaluate_puzzles(
    engine,
    puzzles: list[Puzzle],
    *,
    batch_size: int = 32,
    progress: bool = True,
) -> list[PuzzleResult]:
    """Run the batched engine on puzzles and record per-ply correctness.

    The engine should have temperatures set to 0 by the caller.
    """
    # Build initial state for every puzzle
    states: list[dict] = []
    for p in puzzles:
        try:
            board = chess.Board(p.fen)
            board.push_uci(p.moves[0])  # opponent's setup move
        except Exception as e:
            states.append({"puzzle": p, "done": True, "error": f"setup: {e}",
                           "ply_correct": [], "model_moves": [],
                           "solver_idx": 0, "board": None})
            continue
        states.append({
            "puzzle": p,
            "board": board,
            "solver_idx": 0,
            "ply_correct": [],
            "model_moves": [],
            "done": False,
            "error": "",
        })

    total_plies = sum(p.num_plies for p in puzzles)
    plies_done = 0

    while True:
        active = [s for s in states if not s["done"]]
        if not active:
            break

        for chunk_start in range(0, len(active), batch_size):
            chunk = active[chunk_start:chunk_start + batch_size]
            fens = [s["board"].fen() for s in chunk]
            results = engine.predict_moves(fens, 0.0)

            for s, res in zip(chunk, results):
                p: Puzzle = s["puzzle"]
                expected = normalize_castling(p.solver_moves[s["solver_idx"]])
                model_move = normalize_castling(res.move) if res.move else ""

                correct = (model_move == expected) and (model_move != "")
                s["ply_correct"].append(correct)
                s["model_moves"].append(model_move)

                # Advance the board with the EXPECTED move so that we can
                # grade subsequent plies, even if the model diverged here.
                try:
                    s["board"].push_uci(expected)
                except Exception as e:
                    s["done"] = True
                    s["error"] = f"expected move illegal: {expected} ({e})"
                    continue

                s["solver_idx"] += 1
                plies_done += 1

                # Done?
                if s["solver_idx"] >= p.num_plies:
                    s["done"] = True
                    continue

                # Apply the opponent's forced reply (moves index 2*solver_idx)
                opp_idx = 2 * s["solver_idx"]
                if opp_idx < len(p.moves):
                    try:
                        s["board"].push_uci(p.moves[opp_idx])
                    except Exception as e:
                        s["done"] = True
                        s["error"] = f"opp move illegal: {p.moves[opp_idx]} ({e})"

        if progress:
            done = sum(1 for s in states if s["done"])
            pct = 100.0 * plies_done / total_plies if total_plies else 100.0
            print(f"  [tactics] {done}/{len(states)} puzzles, "
                  f"{plies_done}/{total_plies} plies ({pct:.0f}%)",
                  flush=True)

    return [
        PuzzleResult(
            puzzle=s["puzzle"],
            ply_correct=s["ply_correct"],
            model_moves=s["model_moves"],
            error=s["error"],
        )
        for s in states
    ]


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


@dataclass
class Aggregate:
    n_puzzles: int
    n_with_error: int
    first_move_acc: float
    first_move_ci: tuple[float, float]
    strict_solve_rate: float
    strict_solve_ci: tuple[float, float]
    ply_acc: float              # micro-average over plies
    ply_ci: tuple[float, float]
    avg_plies_per_puzzle: float
    n_plies: int

    def format(self, label: str = "") -> str:
        fm = self.first_move_acc * 100
        fm_lo, fm_hi = (x * 100 for x in self.first_move_ci)
        sr = self.strict_solve_rate * 100
        sr_lo, sr_hi = (x * 100 for x in self.strict_solve_ci)
        pa = self.ply_acc * 100
        pa_lo, pa_hi = (x * 100 for x in self.ply_ci)
        head = f"{label}  n={self.n_puzzles} ({self.n_plies} plies, avg={self.avg_plies_per_puzzle:.1f})"
        lines = [
            head,
            f"  first_move_acc : {fm:5.1f}%  [95% CI {fm_lo:4.1f}, {fm_hi:4.1f}]",
            f"  strict_solve   : {sr:5.1f}%  [95% CI {sr_lo:4.1f}, {sr_hi:4.1f}]",
            f"  per_ply_acc    : {pa:5.1f}%  [95% CI {pa_lo:4.1f}, {pa_hi:4.1f}]",
        ]
        if self.n_with_error:
            lines.append(f"  aborted ({self.n_with_error} puzzles — illegal main-line move)")
        return "\n".join(lines)


def aggregate(results: list[PuzzleResult]) -> Aggregate:
    n = len(results)
    valid = [r for r in results if r.ply_correct]  # ignore puzzles that errored before first ply
    n_err = n - len(valid)

    fm_correct = sum(1 for r in valid if r.first_move_correct)
    strict = sum(1 for r in valid if r.fully_solved)
    plies_correct = sum(r.plies_correct for r in valid)
    total_plies = sum(len(r.ply_correct) for r in valid)

    fm_lo, fm_hi = wilson_ci(fm_correct, len(valid))
    sr_lo, sr_hi = wilson_ci(strict, len(valid))
    pa_lo, pa_hi = wilson_ci(plies_correct, total_plies)

    return Aggregate(
        n_puzzles=len(valid),
        n_with_error=n_err,
        first_move_acc=fm_correct / len(valid) if valid else 0.0,
        first_move_ci=(fm_lo, fm_hi),
        strict_solve_rate=strict / len(valid) if valid else 0.0,
        strict_solve_ci=(sr_lo, sr_hi),
        ply_acc=plies_correct / total_plies if total_plies else 0.0,
        ply_ci=(pa_lo, pa_hi),
        avg_plies_per_puzzle=total_plies / len(valid) if valid else 0.0,
        n_plies=total_plies,
    )


def bucket_by_rating(
    results: list[PuzzleResult],
    edges: list[int],
) -> list[tuple[tuple[int, int], list[PuzzleResult]]]:
    """Group results into [edge_i, edge_{i+1}) buckets. `edges` must be sorted."""
    buckets = [((edges[i], edges[i + 1]), []) for i in range(len(edges) - 1)]
    for r in results:
        rating = r.puzzle.rating
        for (lo, hi), group in buckets:
            if lo <= rating < hi:
                group.append(r)
                break
    return buckets


def bucket_by_theme(
    results: list[PuzzleResult],
    themes: list[str],
) -> list[tuple[str, list[PuzzleResult]]]:
    """Group results by the first theme in `themes` that each puzzle has.
    A puzzle that has none of the requested themes is skipped. A puzzle that
    has multiple will be placed in only one bucket, by the order of `themes`.
    """
    return [
        (t, [r for r in results if t in r.puzzle.themes])
        for t in themes
    ]


def theme_counts(results: list[PuzzleResult]) -> Counter:
    """Count how many puzzles in `results` carry each theme (puzzles can be multi-tagged)."""
    c: Counter[str] = Counter()
    for r in results:
        for t in r.puzzle.themes:
            c[t] += 1
    return c
