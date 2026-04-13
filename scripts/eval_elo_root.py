"""
Evaluate ELO of a model using root policy only (no thinking).

Usage:
    uv run python scripts/eval_elo_root.py \
        --export-dir export_eval_pretrained \
        --num-games 200 --elo 2000
"""

import argparse

import _decoder_inference_cpp as cpp
from chessdecoder.eval.elo_eval import model_vs_stockfish


class RootPolicyEngine:
    """Wraps ThinkingSingleInferenceEngine to only use root policy (no thinking)."""

    def __init__(self, engine):
        self._engine = engine

    def predict_move(self, fen, temperature=0.0):
        return self._engine.predict_move_root(fen, temperature)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--elo", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    engine = cpp.ThinkingSingleInferenceEngine(
        str(export_dir / "backbone.pt"),
        str(export_dir / "weights"),
        str(export_dir / "vocab.json"),
        str(export_dir / "config.json"),
    )

    model = RootPolicyEngine(engine)
    model_vs_stockfish(
        model,
        model1_name="root-policy",
        num_games=args.num_games,
        temperature=args.temperature,
        elo=args.elo,
    )


if __name__ == "__main__":
    main()
