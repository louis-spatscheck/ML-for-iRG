"""Build the C++/Cython Ising simulator as ``invrg.ising.cising``.

All package metadata lives in pyproject.toml; this file only declares the
compiled extension.  The extension is optional: if no working C++ compiler is
available, the build prints a warning and the rest of the package (RG
analysis, models, plotting) still installs.  Only ``invrg.ising`` and
``invrg.simulation`` are then unavailable.
"""
import sys

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Skip extensions that fail to build instead of aborting the install."""

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:  # compiler missing or compilation failed
            print(
                f"WARNING: could not build {ext.name} ({exc!r}). "
                "The C++ Ising simulator will be unavailable; "
                "install a C++ compiler and reinstall to enable it."
            )


extensions = [
    Extension(
        "invrg.ising.cising",
        sources=["src/invrg/ising/cising.pyx"],
        include_dirs=["src/invrg/ising"],  # for ising.hpp
        extra_compile_args=["/O2"] if sys.platform == "win32" else ["-O3"],
        language="c++",
    )
]

ext_modules = cythonize(extensions, language_level=3)
for ext in ext_modules:
    # cythonize() creates new Extension objects and drops this flag, so set it
    # afterwards. It makes setuptools skip a missing compiled file.
    ext.optional = True

setup(ext_modules=ext_modules, cmdclass={"build_ext": OptionalBuildExt})
