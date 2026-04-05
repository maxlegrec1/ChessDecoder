"""
C++ selfplay evaluation for finetuning.

Exports the model to TorchScript, runs the C++ inference engine in a
subprocess (to isolate GPU memory), and returns accuracy metrics.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from chessdecoder.utils.uci import normalize_castling


# ──────────────────────────────────────────────────────────
#  Subprocess worker (runs in isolated process)
# ──────────────────────────────────────────────────────────

def _run_inference(tmp_dir, var_positions_json, pt_positions_json, results_path):
    """Run C++ inference engine and write results to disk. Called in subprocess."""
    import _decoder_inference_cpp as cpp

    var_positions = json.loads(var_positions_json)
    pt_positions = json.loads(pt_positions_json)

    engine = cpp.ThinkingInferenceEngine(
        str(Path(tmp_dir) / "backbone.pt"),
        str(Path(tmp_dir) / "weights"),
        str(Path(tmp_dir) / "vocab.json"),
        str(Path(tmp_dir) / "config.json"),
    )

    # Variation positions
    var_mcts_correct = 0
    var_best_correct = 0
    var_completed = 0
    for i, pos in enumerate(var_positions):
        move = engine.predict_move(pos["fen"], 0.0)
        if move:
            move = normalize_castling(move)
            var_completed += 1
            if move == pos["mcts_action"]:
                var_mcts_correct += 1
            if move == pos["best_move"]:
                var_best_correct += 1
        if (i + 1) % 20 == 0:
            print(f"  [cpp_eval] Variation: {i+1}/{len(var_positions)}", flush=True)

    # Pretrain positions
    pt_best_correct = 0
    pt_best_total = 0
    for i, pos in enumerate(pt_positions):
        move = engine.predict_move(pos["fen"], 0.0)
        if move:
            pt_best_total += 1
            if normalize_castling(move) == pos["best_move"]:
                pt_best_correct += 1
        if (i + 1) % 20 == 0:
            print(f"  [cpp_eval] Pretrain: {i+1}/{len(pt_positions)}", flush=True)

    results = {
        "var_mcts_acc": var_mcts_correct / (var_completed + 1e-8),
        "var_best_acc": var_best_correct / (var_completed + 1e-8),
        "pt_best_acc": pt_best_correct / (pt_best_total + 1e-8),
        "var_mcts_correct": var_mcts_correct,
        "var_best_correct": var_best_correct,
        "var_completed": var_completed,
        "pt_best_correct": pt_best_correct,
        "pt_best_total": pt_best_total,
    }

    with open(results_path, "w") as f:
        json.dump(results, f)


# ──────────────────────────────────────────────────────────
#  Orchestrator (called from training loop)
# ──────────────────────────────────────────────────────────

def evaluate(model, config, var_positions, pt_positions, step):
    """Export model, run C++ selfplay in subprocess, return metrics dict or None."""
    from chessdecoder.export.common import export_head_weights, export_vocab, export_config
    from chessdecoder.export.backbone_causal import from_chess_decoder
    from chessdecoder.export.export_torchscript import export_torchscript

    tmp_dir = Path(tempfile.mkdtemp(prefix="cpp_eval_"))
    try:
        # Export model
        print(f"  [cpp_eval] Exporting model to {tmp_dir} ...", flush=True)
        raw_model = model.module if hasattr(model, "module") else model
        was_training = raw_model.training
        raw_model.eval()

        export_vocab(tmp_dir)
        export_config(config, raw_model, tmp_dir)
        export_head_weights(raw_model, tmp_dir / "weights")

        causal_backbone = from_chess_decoder(raw_model)
        export_torchscript(causal_backbone, tmp_dir, config)
        del causal_backbone
        torch.cuda.empty_cache()

        if was_training:
            raw_model.train()

        # Write positions to temp files
        var_json_path = tmp_dir / "var_positions.json"
        pt_json_path = tmp_dir / "pt_positions.json"
        results_path = tmp_dir / "results.json"
        with open(var_json_path, "w") as f:
            json.dump(var_positions, f)
        with open(pt_json_path, "w") as f:
            json.dump(pt_positions, f)

        # Run in subprocess to isolate GPU memory
        print(f"  [cpp_eval] Running inference in subprocess ...", flush=True)
        proc = subprocess.run(
            [sys.executable, "-m", "chessdecoder.finetune.cpp_eval_worker",
             str(tmp_dir), str(var_json_path), str(pt_json_path), str(results_path)],
            capture_output=False,
            timeout=600,
        )

        if proc.returncode != 0:
            print(f"  [cpp_eval] Subprocess exited with code {proc.returncode}")
            return None

        with open(results_path) as f:
            results = json.load(f)

        print(f"  [cpp_eval] Step {step}: "
              f"var_mcts={results['var_mcts_correct']}/{results['var_completed']} ({results['var_mcts_acc']:.1%}), "
              f"var_best={results['var_best_correct']}/{results['var_completed']} ({results['var_best_acc']:.1%}), "
              f"pt_best={results['pt_best_correct']}/{results['pt_best_total']} ({results['pt_best_acc']:.1%})")

        return results

    except Exception as e:
        print(f"  [cpp_eval] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
