"""
Elo evaluation for ChessDecoder using the C++/TensorRT thinking inference engine.

Usage:
    uv run python scripts/test_evaluate_thinking_trt.py \
        --export-dir export/ \
        --num-games 100 --elo 1500
"""

import argparse
import sys
import time

from src.eval.elo_eval import model_vs_stockfish


def main():
    parser = argparse.ArgumentParser(description="TRT elo evaluation for thinking ChessDecoder")
    parser.add_argument("--export-dir", default="export",
                        help="Directory with TRT engines, weights, vocab, config")
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--elo", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for final move selection")
    args = parser.parse_args()

    try:
        import _decoder_inference_cpp as cpp
    except ImportError:
        print("ERROR: _decoder_inference_cpp not built. Run:")
        print(f"  cd src/cpp/decoder && uv run python setup.py build_ext --inplace && cd ../../..")
        sys.exit(1)

    export_dir = args.export_dir.rstrip("/")

    print(f"Loading TRT engines from {export_dir}/...")
    engine = cpp.ThinkingInferenceEngine(
        f"{export_dir}/backbone_causal.trt",
        f"{export_dir}/backbone_prefix.trt",
        f"{export_dir}/weights",
        f"{export_dir}/vocab.json",
        f"{export_dir}/config.json",
    )

    # Quick smoke test
    print("Smoke test...")
    t0 = time.time()
    move = engine.predict_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    t1 = time.time()
    print(f"  Starting position -> {move} ({t1-t0:.3f}s)")
    print(f"  Tokens: {engine.total_tokens}, Time: {engine.total_time:.3f}s")
    if engine.total_time > 0:
        print(f"  Speed: {engine.total_tokens / engine.total_time:.0f} tok/s")

    # Reset stats
    engine.total_tokens = 0
    engine.total_time = 0.0

    print(f"\nRunning {args.num_games} games vs Stockfish {args.elo}...")
    model_vs_stockfish(
        model=engine,
        model1_name="thinking-trt",
        num_games=args.num_games,
        temperature=args.temperature,
        elo=args.elo,
    )

    if engine.total_time > 0:
        print(f"\nInference stats:")
        print(f"  Total tokens: {engine.total_tokens}")
        print(f"  Total time: {engine.total_time:.1f}s")
        print(f"  Speed: {engine.total_tokens / engine.total_time:.0f} tok/s")


if __name__ == "__main__":
    main()
