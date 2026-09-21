"""Build script for the C++/Cython Ising simulator (module ``cising``).

Build in place with:  python setup.py build_ext --inplace
(or run ../condor/build.sh).  Requires Cython, numpy and a C++ compiler.
"""
import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "cising",
        sources=["cising.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
        language="c++",
    )
]

setup(
    name="cising",
    ext_modules=cythonize(extensions, language_level=3),
)
