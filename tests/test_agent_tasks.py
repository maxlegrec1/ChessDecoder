"""Correctness, mask-invariant and throughput tests for the task registry.

Self-contained: builds tiny fixture parquets (random-walk games + fake-but-
legal labels) in tmp dirs — no dependence on the real corpus or the oracle.
"""
import random
import time

import chess
import pandas as pd
import pytest

from chessdecoder.agent import patch_vocab as pv
from chessdecoder.agent.tasks import REGISTRY, TASK_NAMES
from chessdecoder.agent.tasks.helpers import apply_uci, qbin_centered, stm_of
from chessdecoder.agent.tasks.sources import Sources
from chessdecoder.agent.tasks.stream import AgentTaskDataset, DEFAULT_MIX


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("agent_fixtures")
    rng = random.Random(0)
    games_dir = root / "games"
    games_dir.mkdir()
    rows = []
    for gid in range(60):                       # random-walk games
        b = chess.Board()
        for ply in range(rng.randint(12, 40)):
            legal = list(b.legal_moves)
            if not legal or b.is_game_over():
                break
            mv = rng.choice(legal)
            rows.append((b.fen(), mv.uci(), gid))
            b.push(mv)
    pd.DataFrame(rows, columns=["fen", "played_move", "game_id"]
                 ).to_parquet(games_dir / "games_000.parquet")
    pd.DataFrame(rows, columns=["fen", "played_move", "game_id"]
                 ).to_parquet(games_dir / "games_001.parquet")

    lab = []                                     # fake labels: legal moves + random bins
    for fen, mv, _ in rng.sample(rows, 400):
        b = chess.Board(fen)
        legal = [pv.uci_to_token(m.uci()) for m in b.legal_moves]
        legal = [t for t in legal if t is not None]
        if len(legal) < 4:
            continue
        ms = rng.sample(legal, 4)
        lab.append((fen, rng.randrange(pv.N_QBIN), rng.randrange(pv.N_DBIN), *ms))
    pd.DataFrame(lab, columns=["fen", "q_bin", "d_bin", "m1", "m2", "m3", "m4"]
                 ).to_parquet(root / "labels_000.parquet")

    paired = []                                  # consecutive pairs + random bins
    for i in range(len(rows) - 1):
        if rows[i][2] != rows[i + 1][2]:
            continue
        paired.append((rows[i][0], rng.randrange(pv.N_QBIN), rng.randrange(pv.N_DBIN),
                       rows[i][1], rows[i + 1][0], rng.randrange(pv.N_QBIN),
                       rng.randrange(pv.N_DBIN)))
        if len(paired) >= 400:
            break
    pd.DataFrame(paired, columns=["parent_fen", "q_p", "d_p", "move",
                                  "child_fen", "q_c", "d_c"]
                 ).to_parquet(root / "paired_000.parquet")
    return root


@pytest.fixture(scope="session")
def sources(fixture_dir):
    rng = random.Random(1)
    return Sources({"games", "labels", "paired"}, str(fixture_dir / "games"),
                   str(fixture_dir / "labels_*.parquet"),
                   str(fixture_dir / "paired_*.parquet"), rng,
                   last_shard_only=False)


def _examples(name, sources, n=60, seed=2):
    rng = random.Random(seed)
    task = REGISTRY[name]()
    out = []
    for _ in range(n * 4):
        ex = task.make(sources, rng)
        if ex is not None:
            out.append(ex)
        if len(out) >= n:
            break
    assert out, f"{name} produced nothing"
    return out


# ---------------- invariants for every registered task ----------------------

@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_mask_invariants(name, sources):
    for ex in _examples(name, sources, 30):
        assert len(ex.ids) == len(ex.loss)
        assert any(ex.loss), "no loss positions"
        assert not ex.loss[0], "first token must be context"
        assert all(0 <= t < pv.VOCAB_SIZE for t in ex.ids)
        if ex.eid_span:
            s, l = ex.eid_span
            assert all(ex.loss[s:s + l]), "eid span must be loss positions"


# ---------------- per-task correctness --------------------------------------

def _decode_answer_board(ex):
    return pv.decode_board(ex.ids[-19:])


def test_t2_t3_t12_line_targets(sources):
    for name in ("t2_apply", "t3_line"):
        for ex in _examples(name, sources, 20):
            i = ex.ids.index(pv.LINE)
            j = ex.ids.index(pv.PROBE, i)
            board = pv.decode_board(ex.ids[1:20])
            for t in ex.ids[i + 1:j]:
                assert apply_uci(board, pv.ID_TO_MOVE[t])
            assert board.board_fen() == _decode_answer_board(ex).board_fen()
    for ex in _examples("t12_path", sources, 20):
        bA = pv.decode_board(ex.ids[1:20])
        bB = pv.decode_board(ex.ids[21:40])
        i = ex.ids.index(pv.LINE)
        for t in ex.ids[i + 1:]:
            assert apply_uci(bA, pv.ID_TO_MOVE[t])
        assert bA.board_fen() == bB.board_fen()


def test_t7_legal_targets(sources):
    for ex in _examples("t7_legal", sources, 30):
        board = pv.decode_board(ex.ids[1:20])
        for t in ex.ids[-3:]:
            assert apply_uci(board.copy(), pv.ID_TO_MOVE[t]), "illegal target"


def test_t11_horizon_and_t17_opening(sources):
    for ex in _examples("t11_horizon", sources, 20):
        h = ex.ids[-1] - pv.NUM_BASE
        assert 1 <= h < pv.N_NUM
    for ex in _examples("t17_opening", sources, 20):
        assert sum(ex.loss) == 4
        assert all(pv.MOVE_BASE <= t < pv.MOVE_BASE + pv.N_MOVE
                   for t in ex.ids[-4:])


def test_t18_fill_consistency(sources):
    for ex in _examples("t18_fill", sources, 30):
        masked = ex.ids[1:20]
        i_fill = ex.ids.index(pv.FILL)
        answers = ex.ids[i_fill + 1:]
        slots = [k for k, t in enumerate(masked[:16]) if t == pv.MASK]
        assert len(slots) == len(answers)
        for t in answers:
            assert pv.region_of(t) == "patch"


def test_t13_traj_successor_index(sources):
    for ex in _examples("t13_traj", sources, 15):
        # each <next> j a triple: block j's board, advanced one ply, equals block a's
        blocks = []
        k = 0
        while k < len(ex.ids):
            if ex.ids[k] == pv.PROBE:
                blocks.append(pv.decode_board(ex.ids[k + 1:k + 20]))
                k += 20
            else:
                break
        k_q = k
        while k_q < len(ex.ids):
            assert ex.ids[k_q] == pv.NEXT
            j = ex.ids[k_q + 1] - pv.NUM_BASE
            a = ex.ids[k_q + 2] - pv.NUM_BASE
            bj, ba = blocks[j], blocks[a]
            ok = any(bj.copy().push(m) or bj.copy().board_fen() != ba.board_fen() is False
                     for m in [])  # placeholder
            found = False
            for m in bj.legal_moves:
                t = bj.copy()
                t.push(m)
                if t.board_fen() == ba.board_fen():
                    found = True
                    break
            assert found, "successor block not one ply after source"
            k_q += 3


def test_t15_bestcolor_math(sources):
    for ex in _examples("t15_bestcolor", sources, 15):
        rows = []
        k = 0
        while ex.ids[k] == pv.PROBE:
            b = ex.ids[k + 1:k + 20]
            q = ex.ids[k + 21] - pv.QBIN_BASE
            rows.append((b, q))
            k += 23
        i_w = ex.ids.index(pv.BESTW)
        bw = ex.ids[i_w + 1] - pv.NUM_BASE
        wq = [qbin_centered(q) if stm_of(b) == 0 else -qbin_centered(q)
              for b, q in rows]
        assert bw == max(range(len(wq)), key=lambda i: wq[i])


def test_t14_swing_math(sources):
    for ex in _examples("t14_swing", sources, 15):
        qs = []
        k = 0
        while ex.ids[k] == pv.PROBE:
            qp = ex.ids[k + 21] - pv.QBIN_BASE
            qc = ex.ids[k + 46] - pv.QBIN_BASE
            qs.append(abs(-qbin_centered(qc) - qbin_centered(qp)))
            k += 48
        idx = ex.ids[ex.ids.index(pv.SWING) + 1] - pv.NUM_BASE
        assert idx == max(range(len(qs)), key=lambda i: qs[i])


# ---------------- throughput -------------------------------------------------

@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_task_throughput(name, sources):
    rng = random.Random(3)
    task = REGISTRY[name]()
    t0 = time.time()
    made = 0
    while made < 200 and time.time() - t0 < 5:
        if task.make(sources, rng) is not None:
            made += 1
    rate = made / max(time.time() - t0, 1e-9)
    assert rate > 300, f"{name}: {rate:.0f} ex/s (<300 floor)"


def test_combined_stream_throughput(fixture_dir):
    ds = AgentTaskDataset(str(fixture_dir / "games"),
                          str(fixture_dir / "labels_*.parquet"),
                          paired_glob=str(fixture_dir / "paired_*.parquet"),
                          seed=4)
    it = iter(ds)
    next(it)                                  # warm sources
    t0 = time.time()
    n, toks = 0, 0
    while time.time() - t0 < 6:
        ids, *_ = next(it)
        n += 1
        toks += ids.numel()
    rate = toks / (time.time() - t0)
    # the GPU consumes ~25k tok/s per worker; require 2x headroom
    assert rate > 50_000, f"packer: {rate/1000:.0f}k tok/s (<50k floor)"
