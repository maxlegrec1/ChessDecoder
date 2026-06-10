"""V2 inference + MCTS C++ extension.

Single-board policy/value forward via libtorch (TorchScript export of
BoardForward) + a PUCT MCTS tree on top. Reuses chessdecoder/cpp/chess-library
(header-only) for board state. No TensorRT.
"""
import os
from pathlib import Path

import torch
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).resolve().parent
CPP_ROOT = ROOT.parent  # chessdecoder/cpp/

TORCH_DIR = Path(os.path.dirname(torch.__file__))
TORCH_LIB = TORCH_DIR / "lib"
TORCH_INCLUDE = TORCH_DIR / "include"
TORCH_INCLUDE_CSRC = TORCH_INCLUDE / "torch" / "csrc" / "api" / "include"
CUDA_ROOT = Path("/usr/local/cuda")

CXX11_ABI = "0"
try:
    CXX11_ABI = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"
except AttributeError:
    pass

ext_modules = [
    Pybind11Extension(
        "_v2_inference_cpp",
        sources=[
            str(ROOT / "bindings.cpp"),
            str(ROOT / "vocab_v2.cpp"),
            str(ROOT / "board_forward.cpp"),
            str(ROOT / "mcts_v2.cpp"),
        ],
        include_dirs=[
            str(CPP_ROOT / "chess-library/include"),
            str(TORCH_INCLUDE),
            str(TORCH_INCLUDE_CSRC),
            str(CUDA_ROOT / "include"),
            str(CPP_ROOT),
            str(ROOT),
        ],
        library_dirs=[
            str(TORCH_LIB),
            str(CUDA_ROOT / "lib64"),
        ],
        libraries=["torch", "torch_cuda", "c10", "c10_cuda", "torch_cpu"],
        extra_compile_args=[
            "-O3",
            "-std=c++17",
            f"-D_GLIBCXX_USE_CXX11_ABI={CXX11_ABI}",
        ],
        extra_link_args=[
            f"-Wl,-rpath,{TORCH_LIB}",
        ],
    )
]

setup(
    name="v2_inference_cpp",
    version="0.1.0",
    description="V2 ChessDecoder libtorch inference + PUCT MCTS",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
