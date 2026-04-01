"""
Quick diagnostic: compare variation counts between single and batched engine.
If the batched engine has a bug in AFTER_BOARD, variation counts will differ
even at temperature=0 (because the wrong hidden state changes the end_var decision).
"""
import sys
sys.path.insert(0, ".")

import _decoder_inference_cpp as cpp

EXPORT = "exports/base"

def make_single():
    e = cpp.ThinkingInferenceEngine(
        f"{EXPORT}/backbone.pt",
        f"{EXPORT}/weights",
        f"{EXPORT}/vocab.json",
        f"{EXPORT}/config.json",
    )
    e.board_temperature = 0.0
    e.think_temperature = 0.0
    e.policy_temperature = 0.0
    e.wl_temperature = 0.0
    e.d_temperature = 0.0
    return e

def make_batched(batch=1):
    e = cpp.BatchedInferenceEngine(
        f"{EXPORT}/backbone.pt",
        f"{EXPORT}/weights",
        f"{EXPORT}/vocab.json",
        f"{EXPORT}/config.json",
        batch,
    )
    e.board_temperature = 0.0
    e.think_temperature = 0.0
    e.policy_temperature = 0.0
    e.wl_temperature = 0.0
    e.d_temperature = 0.0
    return e

def count_variations(token_ids, vocab):
    """Count end_var and end_think tokens in a sequence."""
    end_var_id = vocab.end_var_idx()
    end_think_id = vocab.end_think_idx()
    n_vars = sum(1 for t in token_ids if t == end_var_id)
    n_think = sum(1 for t in token_ids if t == end_think_id)
    return n_vars, n_think

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
    "r2qkb1r/ppp2ppp/2np1n2/4p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2QK2R w KQkq - 0 7",
    "8/5pk1/6p1/7p/7P/6P1/5PK1/8 w - - 0 1",
]

def main():
    print("Loading engines...")
    single = make_single()
    batched = make_batched(batch=1)

    print(f"\n{'FEN':50s} {'S_move':8s} {'B_move':8s} {'S_seq':6s} {'B_seq':6s} {'S_vars':6s} {'B_vars':6s}")
    print("-" * 110)

    for fen in FENS:
        # Single
        s_move = single.predict_move(fen)
        s_toks = single.last_token_ids()
        s_len = len(s_toks)

        # Batched (B=1)
        results = batched.predict_moves([fen], 0.0)
        b_result = results[0]
        b_move = b_result.move
        b_toks = b_result.token_ids
        b_len = len(b_toks)

        print(f"{fen[:50]:50s} {s_move:8s} {b_move:8s} {s_len:6d} {b_len:6d}")

    print("\nDone.")

if __name__ == "__main__":
    main()
