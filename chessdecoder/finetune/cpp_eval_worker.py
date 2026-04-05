"""Subprocess entry point for C++ selfplay inference. Isolates GPU memory."""

import sys
from pathlib import Path

from chessdecoder.finetune.cpp_eval import _run_inference

if __name__ == "__main__":
    tmp_dir, var_json_path, pt_json_path, results_path = sys.argv[1:5]
    _run_inference(
        tmp_dir,
        Path(var_json_path).read_text(),
        Path(pt_json_path).read_text(),
        results_path,
    )
