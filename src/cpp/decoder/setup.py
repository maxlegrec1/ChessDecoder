import os
from pathlib import Path

import torch
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).resolve().parent
CPP_ROOT = ROOT.parent  # src/cpp/

# libtorch paths (from the installed PyTorch package)
TORCH_DIR = Path(os.path.dirname(torch.__file__))
TORCH_LIB = TORCH_DIR / "lib"
TORCH_INCLUDE = TORCH_DIR / "include"
TORCH_INCLUDE_CSRC = TORCH_INCLUDE / "torch" / "csrc" / "api" / "include"
CUDA_ROOT = Path("/usr/local/cuda")

# Match PyTorch's C++ ABI setting
CXX11_ABI = "0"
try:
    CXX11_ABI = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"
except AttributeError:
    pass

ext_modules = [
    Pybind11Extension(
        "_decoder_inference_cpp",
        sources=[
            str(ROOT / "bindings.cpp"),
            str(ROOT / "vocab.cpp"),
            str(ROOT / "torch_backbone.cpp"),
            str(ROOT / "heads.cpp"),
            str(ROOT / "decoder_engine.cpp"),
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
    name="decoder_inference_cpp",
    version="0.2.0",
    description="Pybind11 bindings for the ChessDecoder libtorch thinking inference engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
