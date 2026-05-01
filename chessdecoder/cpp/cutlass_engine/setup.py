"""Build the cutlass_engine pybind11 extension.

Hand-rolls a build_ext that drives nvcc for .cu sources, since pybind11's
setup_helpers only knows about g++. The output is a single .so named
_cutlass_decoder_cpp with no libtorch on the link line.
"""

import os
import subprocess
import sys
from pathlib import Path

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

ROOT = Path(__file__).resolve().parent
CPP_ROOT = ROOT.parent  # chessdecoder/cpp/

CUDA_ROOT = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
NVCC = str(CUDA_ROOT / "bin" / "nvcc")

CUTLASS_ROOT = CPP_ROOT / "third_party" / "cutlass"
CUTLASS_INCLUDE = CUTLASS_ROOT / "include"
CUTLASS_TOOLS_INCLUDE = CUTLASS_ROOT / "tools" / "util" / "include"

ARCH = os.environ.get("CUTLASS_ENGINE_ARCH", "sm_100a")

DEBUG = os.environ.get("CUTLASS_ENGINE_DEBUG", "0") == "1"


def _src(name: str) -> str:
    # setuptools' editable-install (egg_info) refuses absolute paths, so we
    # use repo-root-relative paths.  setup.py runs with CWD == ROOT.
    return str(Path("src") / name)


CU_SOURCES = [
    _src("kernels/rmsnorm.cu"),
    _src("kernels/rope.cu"),
    _src("kernels/swiglu.cu"),
    _src("kernels/embedding.cu"),
    _src("kernels/sampler.cu"),
    _src("kernels/activations.cu"),
    _src("kernels/kv_write.cu"),
    _src("kernels/fmha_decode.cu"),
    _src("kernels/fmha_prefill.cu"),
    _src("kernels/misc.cu"),
    _src("layers/attention_block.cu"),
    _src("layers/mlp_block.cu"),
    _src("layers/transformer_layer.cu"),
    _src("model.cu"),
    _src("kv_cache.cu"),
    _src("state_machine.cu"),
    _src("scheduler.cu"),
    _src("engine.cu"),
    _src("allocator.cu"),
    _src("gemm.cu"),
]

CPP_SOURCES = [
    _src("weights.cpp"),
    _src("config.cpp"),
    _src("bindings.cpp"),
    # Reuse the existing vocab loader (no libtorch dep — depends only on
    # the header-only chess-library).
    "../decoder/vocab.cpp",
]


class build_ext(_build_ext):
    def build_extensions(self):
        # Register .cu as a known C++ extension so distutils' UnixCCompiler
        # routes the file through _compile (which we override below).
        self.compiler.src_extensions.append(".cu")
        # The default UnixCCompiler doesn't know .cu's language; tell it so
        # that the compiler-detection step doesn't reject the file.
        if hasattr(self.compiler, "language_map"):
            self.compiler.language_map[".cu"] = "c++"

        # Reroute the compile step to nvcc for .cu files; keep g++ for .cpp.
        original_compile = self.compiler._compile

        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith(".cu"):
                self._nvcc_compile(obj, src, cc_args, extra_postargs, pp_opts)
            else:
                original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = _compile
        super().build_extensions()

    def _nvcc_compile(self, obj, src, cc_args, extra_postargs, pp_opts):
        # Filter cc-only flags from extra_postargs (we keep our own list).
        nvcc_cmd = [
            NVCC,
            "-c",
            src,
            "-o",
            obj,
            f"-arch={ARCH}",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-Xcompiler",
            "-fPIC",
            "-Xcompiler",
            "-fvisibility=hidden",
            "-DCUTLASS_NVCC_ARCHS=100a",
        ]
        if DEBUG:
            nvcc_cmd += ["-O0", "-G", "-g", "-lineinfo"]
        else:
            nvcc_cmd += ["-O3", "--use_fast_math", "-lineinfo"]
        # Inherit include/define flags from pp_opts.
        nvcc_cmd += [a for a in pp_opts if a.startswith(("-I", "-D"))]
        # Make sure parent dir exists for the object.
        Path(obj).parent.mkdir(parents=True, exist_ok=True)
        print(" ".join(nvcc_cmd), flush=True)
        subprocess.check_call(nvcc_cmd)


ext = Extension(
    "_cutlass_decoder_cpp",
    sources=CPP_SOURCES + CU_SOURCES,
    include_dirs=[
        str(ROOT / "include"),
        str(CUTLASS_INCLUDE),
        str(CUTLASS_TOOLS_INCLUDE),
        str(CPP_ROOT / "decoder"),               # for vocab.hpp
        str(CPP_ROOT / "chess-library/include"), # for chess.hpp (header-only)
        str(CUDA_ROOT / "include"),
        pybind11.get_include(),
    ],
    library_dirs=[str(CUDA_ROOT / "lib64")],
    libraries=["cudart", "cublas"],
    extra_compile_args=[
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-fvisibility=hidden",
    ],
    extra_link_args=[
        f"-L{CUDA_ROOT}/lib64",
        f"-Wl,-rpath,{CUDA_ROOT}/lib64",
    ],
    language="c++",
)


setup(
    name="cutlass_decoder_cpp",
    version="0.1.0",
    description="CUTLASS-based ChessDecoder inference engine (FP16 + FP8) — pybind11 bindings",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.10",
)
