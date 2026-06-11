from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="puct-cpp",
    version="0.1.0",
    ext_modules=[Pybind11Extension(
        "_puct_cpp", ["puct.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3", "-march=native"],
    )],
    cmdclass={"build_ext": build_ext},
)
