"""CUTLASS-based selfplay evaluation for finetuning + RL.

Exports the model weights for the CUTLASS engine, runs in-process
inference (Phase 0 stable-VRAM means no GPU memory drift, no subprocess
isolation needed), returns accuracy metrics.

Replaces the legacy TorchScript+subprocess flow that depended on
`_decoder_inference_cpp.ThinkingSingleInferenceEngine` (Phase L).
"""

import shutil
import sys
import tempfile
from pathlib import Path

import torch

from chessdecoder.utils.uci import normalize_castling


# Add the cutlass engine helpers to the import path.
_CUTLASS_ENGINE_DIR = Path(__file__).resolve().parents[1] / "cpp" / "cutlass_engine"
sys.path.insert(0, str(_CUTLASS_ENGINE_DIR / "python"))
sys.path.insert(0, str(_CUTLASS_ENGINE_DIR / "src"))


def _run_inference_cutlass(tmp_dir: Path, var_positions: list, pt_positions: list,
                           batch_size: int = 32):
    """Run CUTLASS thinking inference in-process.

    Bundles the position lists into batches of `batch_size` for throughput.
    Returns the same metrics dict shape the libtorch path used.
    """
    from rl_adapter import CutlassRLEngine  # noqa: PLC0415

    engine = CutlassRLEngine(
        backbone_pt="",
        weights_dir=str(tmp_dir / "weights"),
        vocab_json=str(tmp_dir / "vocab.json"),
        config_json=str(tmp_dir / "config.json"),
        batch_size=batch_size,
    )
    for attr in ("board_temperature", "think_temperature",
                 "policy_temperature", "wl_temperature", "d_temperature"):
        setattr(engine, attr, 0.0)

    def _eval_batch(positions):
        if not positions:
            return [None] * 0
        moves: list[str] = []
        for start in range(0, len(positions), batch_size):
            chunk = positions[start:start + batch_size]
            chunk_fens = [p["fen"] for p in chunk]
            # Pad chunk to batch_size (cutlass engine has fixed B at construction).
            pad = batch_size - len(chunk_fens)
            padded = chunk_fens + [chunk_fens[-1]] * pad if pad > 0 else chunk_fens
            results = engine.predict_moves(padded, 0.0)
            for r in results[:len(chunk_fens)]:
                m = r.move
                moves.append(normalize_castling(m) if m else "")
            if (start + batch_size) % 200 == 0 or start + batch_size >= len(positions):
                done = min(start + batch_size, len(positions))
                print(f"  [cpp_eval] {done}/{len(positions)}", flush=True)
        return moves

    var_moves = _eval_batch(var_positions)
    pt_moves = _eval_batch(pt_positions)

    # Variation positions
    var_mcts_correct = 0
    var_best_correct = 0
    var_completed = 0
    for pos, move in zip(var_positions, var_moves):
        if move:
            var_completed += 1
            if move == pos["mcts_action"]:
                var_mcts_correct += 1
            if move == pos["best_move"]:
                var_best_correct += 1

    # Pretrain positions
    pt_best_correct = 0
    pt_best_total = 0
    for pos, move in zip(pt_positions, pt_moves):
        if move:
            pt_best_total += 1
            if move == pos["best_move"]:
                pt_best_correct += 1

    return {
        "var_mcts_acc": var_mcts_correct / (var_completed + 1e-8),
        "var_best_acc": var_best_correct / (var_completed + 1e-8),
        "pt_best_acc": pt_best_correct / (pt_best_total + 1e-8),
        "var_mcts_correct": var_mcts_correct,
        "var_best_correct": var_best_correct,
        "var_completed": var_completed,
        "pt_best_correct": pt_best_correct,
        "pt_best_total": pt_best_total,
    }


def evaluate(model, config, var_positions, pt_positions, step):
    """Export model weights, run CUTLASS selfplay in-process, return metrics
    dict or None on failure."""
    from export_for_cutlass import export_for_cutlass  # noqa: PLC0415

    tmp_dir = Path(tempfile.mkdtemp(prefix="cpp_eval_"))
    try:
        print(f"  [cpp_eval] Exporting model to {tmp_dir} ...", flush=True)
        raw_model = model.module if hasattr(model, "module") else model
        was_training = raw_model.training
        raw_model.eval()

        export_for_cutlass(raw_model, config, tmp_dir)

        if was_training:
            raw_model.train()
        torch.cuda.empty_cache()

        print(f"  [cpp_eval] Running CUTLASS inference (in-process) ...", flush=True)
        results = _run_inference_cutlass(tmp_dir, var_positions, pt_positions)

        print(f"  [cpp_eval] Step {step}: "
              f"var_mcts={results['var_mcts_correct']}/{results['var_completed']} "
              f"({results['var_mcts_acc']:.1%}), "
              f"var_best={results['var_best_correct']}/{results['var_completed']} "
              f"({results['var_best_acc']:.1%}), "
              f"pt_best={results['pt_best_correct']}/{results['pt_best_total']} "
              f"({results['pt_best_acc']:.1%})")

        return results

    except Exception as e:
        print(f"  [cpp_eval] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        torch.cuda.empty_cache()
