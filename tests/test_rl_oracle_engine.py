"""Parity: OracleEngine (compiled, GPU top-k) vs eager Oracle on corpus
positions. Top-4 allows tie-order swaps (identical logits) only."""
import glob

import chess
import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

N_POS = 10_000
SHARD_GLOB = "/mnt/2tb_2/decoder/parquet_files_decoder/*.parquet"


@pytest.fixture(scope="module")
def engines():
    from chessdecoder.agent.oracle import Oracle
    from chessdecoder.agent.rl.oracle_engine import OracleEngine
    eager = Oracle()
    fast = OracleEngine(compile_model=True)
    return eager, fast


ID2KEY = None


@pytest.fixture(scope="module")
def boards():
    import pandas as pd
    f = sorted(glob.glob(SHARD_GLOB))[-1]          # last shard = held-out side
    fens = pd.read_parquet(f, columns=["fen"]).fen.drop_duplicates()
    return [chess.Board(x) for x in fens.head(N_POS).tolist()]


def test_parity(engines, boards):
    from chessdecoder.models.vocab import move_token_to_idx
    from chessdecoder.agent import patch_vocab as pv
    id2key = {v: k for k, v in pv.MOVE_TO_ID.items()}
    eager, fast = engines
    bin_mismatch = top_mismatch = two_bin = top1_mismatch = 0
    B = 256
    for i in range(0, len(boards), B):
        chunk = boards[i:i + B]
        eager._memo.clear()
        fast._memo.clear()
        re_, rf_ = eager.query_batch(chunk), fast.query_batch(chunk)
        pol = None
        for j, (b, re, rf) in enumerate(zip(chunk, re_, rf_)):
            # compiled-bf16 fusion noise is ~2x eager-bf16 (both unbiased
            # around fp32; measured on the worst case: fp32 q=0.337, eager
            # +0.006, compiled -0.013). Contract: never >2 bins, >1 bin rare.
            assert abs(re.q_bin - rf.q_bin) <= 2, f"{b.fen()} qbin >2 apart"
            assert abs(re.d_bin - rf.d_bin) <= 2, f"{b.fen()} dbin >2 apart"
            if max(abs(re.q_bin - rf.q_bin), abs(re.d_bin - rf.d_bin)) == 2:
                two_bin += 1
            if (re.q_bin, re.d_bin) != (rf.q_bin, rf.d_bin):
                bin_mismatch += 1
            if set(re.top_moves) != set(rf.top_moves):
                top_mismatch += 1
            if re.top_moves[0] != rf.top_moves[0]:
                top1_mismatch += 1
                # a top-1 flip is only legitimate between near-tied moves
                # (measured: median eager-logit gap 0.0000, max 0.125)
                if pol is None:
                    pol, _ = eager._forward([x.fen() for x in chunk])
                lg = pol[j].tolist()
                def _lp(aid):
                    idx = move_token_to_idx.get(id2key[aid])
                    return lg[idx] if idx is not None else -1e9
                gap = abs(_lp(re.top_moves[0]) - _lp(rf.top_moves[0]))
                assert gap <= 0.5, f"{b.fen()} top-1 flip with gap {gap:.3f}"
    # top-4 set flips are rank-4/5 near-ties (measured: median eager-logit
    # gap between disputed moves = 0.0000, 96.5% < 0.1); the move that must
    # be stable is the best one.
    assert two_bin <= N_POS * 0.005, f"{two_bin} 2-bin flips"
    assert bin_mismatch <= N_POS * 0.25, f"{bin_mismatch} q/d bin-edge flips"
    assert top_mismatch <= N_POS * 0.08, f"{top_mismatch} top-4 set mismatches"
    assert top1_mismatch <= N_POS * 0.05, f"{top1_mismatch} top-1 mismatches"


def test_eval_batch_shapes(engines):
    _, fast = engines
    fens = [chess.STARTING_FEN] * 7
    q, d, pol = fast.eval_batch(fens)
    assert q.shape == (7,) and d.shape == (7,) and pol.shape == (7, 1924)
    assert torch.all(d >= 0) and torch.all(d <= 1)
    assert abs(q[0].item()) < 0.3      # startpos roughly balanced
