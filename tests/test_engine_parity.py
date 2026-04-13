"""Heavy integration tests: Python vs C++ single vs C++ batched engine parity.

These tests load the real 116M model and run thinking inference across all
three backends, verifying they produce similar results.  They require a GPU,
the exported model in exports/base/, and the finetuned checkpoint.

NOT run in CI.  Run locally with:
    uv run pytest tests/test_engine_parity.py -v

Markers: @pytest.mark.gpu, @pytest.mark.cpp
Expected runtime: ~5-10 minutes depending on GPU.
"""

import os
import random
import time

import chess
import pytest
import pyarrow.parquet as pq
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.cpp]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPORT_DIR = "exports/base"
CHECKPOINT = "checkpoints/finetune-thinking-v1_20260320_205453/checkpoint_step_282000.pt"
DATA_DIR = os.environ.get("PRETRAIN_PARQUET_DIR", "/home/maxime/parquet_files_decoder/")
NUM_FENS = 30
SEED = 123


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _load_fens(n, seed, data_dir):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
    rng = random.Random(seed)
    fname = rng.choice(files)
    table = pq.read_table(os.path.join(data_dir, fname), columns=["fen", "best_move"])
    indices = rng.sample(range(len(table)), min(n * 3, len(table)))
    seen = set()
    pairs = []
    for i in indices:
        fen = table.column("fen")[i].as_py()
        if fen not in seen:
            seen.add(fen)
            pairs.append((fen, table.column("best_move")[i].as_py()))
        if len(pairs) >= n:
            break
    return pairs[:n]


@pytest.fixture(scope="module")
def fen_pairs():
    if not os.path.isdir(DATA_DIR):
        pytest.skip(f"Data directory not found: {DATA_DIR}")
    return _load_fens(NUM_FENS, SEED, DATA_DIR)


@pytest.fixture(scope="module")
def python_engine():
    """Load the Python thinking engine (ThinkingModelWrapper)."""
    if not os.path.isfile(CHECKPOINT):
        pytest.skip(f"Checkpoint not found: {CHECKPOINT}")

    from chessdecoder.inference.think import load_model, sample_token
    from chessdecoder.utils.uci import normalize_castling
    from chessdecoder.models.vocab import (
        token_to_idx, idx_to_token, board_idx_to_full_idx,
        move_idx_to_full_idx, board_token_to_idx, move_token_to_idx,
    )
    from chessdecoder.dataloader.data import fen_to_position_tokens

    # Import ThinkingModelWrapper inline (it was removed from scripts, so
    # we reimplement a minimal version here for testing)
    model, max_seq_len = load_model(CHECKPOINT, "cuda")

    class _PythonThinkingEngine:
        """Minimal wrapper that runs Python thinking inference and captures results."""

        def __init__(self, model, device, max_seq_len):
            self.model = model
            self.device = device
            self.max_seq_len = max_seq_len
            self.last_token_ids = []
            self.last_wl_entries = []
            self.last_d_entries = []

        @torch.no_grad()
        def predict_move(self, fen, temperature=0.0):
            model = self.model
            device = self.device
            max_seq_len = self.max_seq_len

            token_ids = []
            block_ids = []
            wl_entries = []
            d_entries = []
            next_block = [0]
            orphan_ctr = [10000]

            _BOARD_END_VAR_IDX = board_token_to_idx["end_var"]
            _BOARD_END_THINK_IDX = board_token_to_idx["end_think"]

            def orphan():
                orphan_ctr[0] += 1
                return orphan_ctr[0]

            def append(tok_id, bid):
                token_ids.append(tok_id)
                block_ids.append(bid)

            def full():
                return len(token_ids) >= max_seq_len

            def prefix_forward():
                S = len(token_ids)
                inp = torch.tensor([token_ids], dtype=torch.long, device=device)
                blk = torch.tensor([block_ids], dtype=torch.long, device=device)
                wl_pos = torch.zeros(1, S, dtype=torch.bool, device=device)
                d_pos = torch.zeros(1, S, dtype=torch.bool, device=device)
                wl_val = torch.zeros(1, S, dtype=torch.float16, device=device)
                d_val = torch.zeros(1, S, dtype=torch.float16, device=device)
                for p, v in wl_entries:
                    wl_pos[0, p] = True; wl_val[0, p] = v
                for p, v in d_entries:
                    d_pos[0, p] = True; d_val[0, p] = v
                return model(inp, mask_type="prefix", block_id=blk,
                             wl_values=wl_val, d_values=d_val,
                             wl_positions=wl_pos, d_positions=d_pos)

            def causal_forward():
                S = len(token_ids)
                inp = torch.tensor([token_ids], dtype=torch.long, device=device)
                wl_pos = torch.zeros(1, S, dtype=torch.bool, device=device)
                d_pos = torch.zeros(1, S, dtype=torch.bool, device=device)
                wl_val = torch.zeros(1, S, dtype=torch.float16, device=device)
                d_val = torch.zeros(1, S, dtype=torch.float16, device=device)
                for p, v in wl_entries:
                    wl_pos[0, p] = True; wl_val[0, p] = v
                for p, v in d_entries:
                    d_pos[0, p] = True; d_val[0, p] = v
                return model(inp, mask_type="causal",
                             wl_values=wl_val, d_values=d_val,
                             wl_positions=wl_pos, d_positions=d_pos)

            def predict_wl(move_pos):
                h = prefix_forward()
                logits = model.wl_head(h[0, move_pos:move_pos+1, :])
                return model.wl_bucket_centers[torch.argmax(logits, dim=-1)].item()

            def predict_d(wl_pos):
                h = prefix_forward()
                logits = model.d_head(h[0, wl_pos:wl_pos+1, :])
                return model.d_bucket_centers[torch.argmax(logits, dim=-1)].item()

            def emit_wl_d(move_pos):
                wl = predict_wl(move_pos)
                wl_pos_idx = len(token_ids)
                append(token_to_idx["wl_value"], orphan())
                wl_entries.append((wl_pos_idx, wl))
                d = predict_d(wl_pos_idx)
                d_pos_idx = len(token_ids)
                append(token_to_idx["d_value"], orphan())
                d_entries.append((d_pos_idx, d))

            def emit_board():
                bid = next_block[0]; next_block[0] += 1
                for _ in range(68):
                    if full():
                        break
                    h = causal_forward()
                    logits = model.board_head(h)[0, -1, :]
                    board_sub_idx = torch.argmax(logits).item()
                    full_idx = board_idx_to_full_idx[board_sub_idx]
                    append(full_idx, bid)

            # Root board
            root_tokens = fen_to_position_tokens(fen)
            bid = next_block[0]; next_block[0] += 1
            for t in root_tokens:
                append(token_to_idx[t], bid)
            if full():
                self.last_token_ids = list(token_ids)
                self.last_wl_entries = list(wl_entries)
                self.last_d_entries = list(d_entries)
                return model.predict_move(fen, temperature=temperature, force_legal=True)

            append(token_to_idx["start_think"], orphan())

            state = "MOVE"
            first_root_move = None

            while not full():
                if state == "MOVE":
                    pos = len(token_ids) - 1
                    h = prefix_forward()
                    logits = model.thinking_policy_head(h)[0, pos, :]
                    move_sub_idx = sample_token(logits, temperature)
                    full_idx = move_idx_to_full_idx[move_sub_idx]
                    tok = idx_to_token[full_idx]
                    append(full_idx, orphan())
                    if first_root_move is None:
                        first_root_move = normalize_castling(tok)
                    state = "WL_D"
                elif state == "WL_D":
                    if full(): break
                    emit_wl_d(len(token_ids) - 1)
                    state = "BOARD"
                elif state == "BOARD":
                    if full(): break
                    emit_board()
                    state = "AFTER_BOARD"
                elif state == "AFTER_BOARD":
                    if full(): break
                    h = causal_forward()
                    logits = model.board_head(h)[0, -1, :]
                    board_sub_idx = sample_token(logits, temperature)
                    if board_sub_idx == _BOARD_END_VAR_IDX:
                        append(board_idx_to_full_idx[board_sub_idx], orphan())
                        state = "AFTER_END_VAR"
                    else:
                        state = "MOVE"
                elif state == "AFTER_END_VAR":
                    if full(): break
                    h = causal_forward()
                    logits = model.board_head(h)[0, -1, :]
                    board_sub_idx = sample_token(logits, temperature)
                    if board_sub_idx == _BOARD_END_THINK_IDX:
                        append(board_idx_to_full_idx[board_sub_idx], orphan())
                        state = "FINAL"
                    else:
                        state = "MOVE"
                elif state == "FINAL":
                    if full(): break
                    pos = len(token_ids) - 1
                    h = prefix_forward()
                    logits = model.policy_head(h)[0, pos, :]
                    move_sub_idx = torch.argmax(logits).item()
                    full_idx = move_idx_to_full_idx[move_sub_idx]
                    tok = idx_to_token[full_idx]
                    append(full_idx, orphan())
                    self.last_token_ids = list(token_ids)
                    self.last_wl_entries = list(wl_entries)
                    self.last_d_entries = list(d_entries)
                    return normalize_castling(tok)

            # Fallback
            self.last_token_ids = list(token_ids)
            self.last_wl_entries = list(wl_entries)
            self.last_d_entries = list(d_entries)
            if first_root_move:
                return first_root_move
            return model.predict_move(fen, temperature=temperature, force_legal=True)

    return _PythonThinkingEngine(model, "cuda", max_seq_len)


@pytest.fixture(scope="module")
def cpp_single():
    cpp = pytest.importorskip("_decoder_inference_cpp")
    return cpp.ThinkingSingleInferenceEngine(
        f"{EXPORT_DIR}/backbone.pt", f"{EXPORT_DIR}/weights",
        f"{EXPORT_DIR}/vocab.json", f"{EXPORT_DIR}/config.json",
    )


@pytest.fixture(scope="module")
def cpp_batched():
    cpp = pytest.importorskip("_decoder_inference_cpp")
    # Must be >= NUM_FENS (30) so test_cpp_batched_legal can submit all pairs
    # at once, and >= K (10) for pass@k tests.
    return cpp.ThinkingBatchedInferenceEngine(
        f"{EXPORT_DIR}/backbone.pt", f"{EXPORT_DIR}/weights",
        f"{EXPORT_DIR}/vocab.json", f"{EXPORT_DIR}/config.json",
        max(NUM_FENS, 32),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAllEnginesProduceLegalMoves:
    """All three engines must produce legal moves for every FEN."""

    def test_python_legal(self, python_engine, fen_pairs):
        for fen, _ in fen_pairs:
            board = chess.Board(fen)
            move = python_engine.predict_move(fen, temperature=0.0)
            assert chess.Move.from_uci(move) in board.legal_moves, \
                f"Python illegal: {move} for {fen}"

    def test_cpp_single_legal(self, cpp_single, fen_pairs):
        for fen, _ in fen_pairs:
            board = chess.Board(fen)
            move = cpp_single.predict_move(fen, 0.0)
            assert chess.Move.from_uci(move) in board.legal_moves, \
                f"C++ single illegal: {move} for {fen}"

    def test_cpp_batched_legal(self, cpp_batched, fen_pairs):
        fens = [fen for fen, _ in fen_pairs]
        results = cpp_batched.predict_moves(fens, 0.0)
        for (fen, _), r in zip(fen_pairs, results):
            board = chess.Board(fen)
            assert chess.Move.from_uci(r.move) in board.legal_moves, \
                f"C++ batched illegal: {r.move} for {fen}"


class TestCppSingleMatchesPython:
    """C++ single engine should closely match Python at temp=0.

    FP16 rounding in different GEMM kernels means not every FEN will agree,
    but the match rate should be high (>= 80%).
    """

    def test_move_match_rate(self, python_engine, cpp_single, fen_pairs):
        matches = 0
        for fen, _ in fen_pairs:
            py_move = python_engine.predict_move(fen, temperature=0.0)
            cpp_move = cpp_single.predict_move(fen, 0.0)
            if py_move == cpp_move:
                matches += 1
        rate = matches / len(fen_pairs)
        print(f"\nPython vs C++ single: {matches}/{len(fen_pairs)} = {rate:.0%}")
        assert rate >= 0.8, f"Match rate {rate:.0%} < 80%"


class TestBatchedInternalConsistency:
    """Batched engine with same FEN repeated must be deterministic at temp=0."""

    def test_same_fen_identical_results(self, cpp_batched):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        results = cpp_batched.predict_moves([fen] * 8, 0.0)
        moves = [r.move for r in results]
        assert len(set(moves)) == 1, f"Non-deterministic: {moves}"
        # Token sequences must also match
        for i in range(1, len(results)):
            assert results[i].token_ids == results[0].token_ids


class TestPassAtK:
    """Pass@k should be similar across all three engines.

    Runs k=10 rollouts at temp=1.5 for each FEN and checks if any
    match the best_move.  All engines should be in a similar range.
    """

    K = 10

    def _pass_at_k(self, engine_fn, fen_pairs, k):
        correct = 0
        for fen, best_move in fen_pairs:
            moves = set()
            for _ in range(k):
                moves.add(engine_fn(fen))
            if best_move in moves:
                correct += 1
        return correct / len(fen_pairs)

    def test_cpp_single_pass_at_k(self, cpp_single, fen_pairs):
        cpp_single.think_temperature = 1.5
        cpp_single.policy_temperature = 1.5
        rate = self._pass_at_k(
            lambda fen: cpp_single.predict_move(fen, 1.5),
            fen_pairs, self.K,
        )
        print(f"\nC++ single pass@{self.K}: {rate:.0%}")
        assert rate >= 0.5, f"pass@{self.K} = {rate:.0%} too low"

    def test_cpp_batched_pass_at_k(self, cpp_batched, fen_pairs):
        cpp_batched.think_temperature = 1.5
        cpp_batched.policy_temperature = 1.5

        correct = 0
        for fen, best_move in fen_pairs:
            results = cpp_batched.predict_moves([fen] * self.K, 1.5)
            moves = {r.move for r in results}
            if best_move in moves:
                correct += 1
        rate = correct / len(fen_pairs)
        print(f"\nC++ batched pass@{self.K}: {rate:.0%}")
        assert rate >= 0.5, f"pass@{self.K} = {rate:.0%} too low"

    def test_batched_vs_single_pass_at_k_similar(self, cpp_single, cpp_batched, fen_pairs):
        """The two C++ engines should have similar pass@k (within 15pp)."""
        cpp_single.think_temperature = 1.5
        cpp_single.policy_temperature = 1.5
        cpp_batched.think_temperature = 1.5
        cpp_batched.policy_temperature = 1.5

        single_correct = 0
        batched_correct = 0
        for fen, best_move in fen_pairs:
            # Single
            single_moves = set()
            for _ in range(self.K):
                single_moves.add(cpp_single.predict_move(fen, 1.5))
            if best_move in single_moves:
                single_correct += 1

            # Batched
            results = cpp_batched.predict_moves([fen] * self.K, 1.5)
            batched_moves = {r.move for r in results}
            if best_move in batched_moves:
                batched_correct += 1

        s_rate = single_correct / len(fen_pairs)
        b_rate = batched_correct / len(fen_pairs)
        gap = abs(s_rate - b_rate)
        print(f"\nSingle pass@{self.K}: {s_rate:.0%}, "
              f"Batched pass@{self.K}: {b_rate:.0%}, gap: {gap:.0%}")
        assert gap <= 0.15, f"Gap {gap:.0%} > 15pp between single and batched"


class TestTokenStructure:
    """All engines should produce well-formed thinking traces."""

    def _validate_structure(self, token_ids):
        from chessdecoder.models.vocab import token_to_idx, move_vocab_size, POSITION_TOKEN_LENGTH
        start_think = token_to_idx["start_think"]
        end_think = token_to_idx["end_think"]
        start_pos = token_to_idx["start_pos"]
        end_var = token_to_idx["end_var"]

        # Starts with 68-token board block
        assert token_ids[0] == start_pos
        assert start_think in token_ids
        assert end_think in token_ids

        # Every board block is exactly 68 tokens
        st = token_ids.index(start_think)
        et = token_ids.index(end_think)
        thinking = token_ids[st + 1:et]
        i = 0
        while i < len(thinking):
            if thinking[i] == start_pos:
                assert i + POSITION_TOKEN_LENGTH <= len(thinking), \
                    f"Incomplete board block at position {i}"
                i += POSITION_TOKEN_LENGTH
            else:
                i += 1

        # At least one end_var in thinking region
        assert end_var in thinking, "No end_var in thinking region"

    def test_cpp_single_structure(self, cpp_single, fen_pairs):
        for fen, _ in fen_pairs[:5]:
            cpp_single.predict_move(fen, 0.0)
            self._validate_structure(list(cpp_single.last_token_ids()))

    def test_cpp_batched_structure(self, cpp_batched, fen_pairs):
        fens = [fen for fen, _ in fen_pairs[:5]]
        results = cpp_batched.predict_moves(fens, 0.0)
        for r in results:
            self._validate_structure(r.token_ids)

    def test_python_structure(self, python_engine, fen_pairs):
        for fen, _ in fen_pairs[:5]:
            python_engine.predict_move(fen, temperature=0.0)
            self._validate_structure(python_engine.last_token_ids)


class TestThroughput:
    """Sanity check that engines aren't catastrophically slow."""

    def test_cpp_single_throughput(self, cpp_single, fen_pairs):
        cpp_single.total_tokens = 0
        cpp_single.total_time = 0.0
        for fen, _ in fen_pairs[:10]:
            cpp_single.predict_move(fen, 0.0)
        tok_per_s = cpp_single.total_tokens / max(cpp_single.total_time, 1e-6)
        print(f"\nC++ single: {tok_per_s:.0f} tok/s")
        assert tok_per_s >= 100, f"Too slow: {tok_per_s:.0f} tok/s"

    def test_cpp_batched_throughput(self, cpp_batched, fen_pairs):
        fens = [fen for fen, _ in fen_pairs[:8]]
        cpp_batched.total_tokens = 0
        cpp_batched.total_time = 0.0
        for _ in range(3):
            cpp_batched.predict_moves(fens, 0.0)
        tok_per_s = cpp_batched.total_tokens / max(cpp_batched.total_time, 1e-6)
        print(f"\nC++ batched B=8: {tok_per_s:.0f} tok/s")
        assert tok_per_s >= 100, f"Too slow: {tok_per_s:.0f} tok/s"
