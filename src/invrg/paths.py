"""Locations of simulation data, intermediate results and plots.

The analysis scripts read and write files below three root directories.
They default to ``data``, ``scratch`` and ``plots`` in the current working
directory and can be changed with environment variables:

===================== ============================================================
``INVRG_DATA_DIR``    simulation output and trained models (``lattice_size<L>/``,
                      ``final_models/``, ``standard_renorm/``, ``inverse_renorm/``)
``INVRG_SCRATCH_DIR`` large intermediate files (``forward_renorm/``,
                      ``test_samples/``, ``complexUNet*/``)
``INVRG_PLOTS_DIR``   figures
===================== ============================================================
"""
import os
from pathlib import Path


def _root(variable: str, default: str) -> Path:
    return Path(os.environ.get(variable, default)).expanduser()


DATA_DIR = _root("INVRG_DATA_DIR", "data")
SCRATCH_DIR = _root("INVRG_SCRATCH_DIR", "scratch")
PLOTS_DIR = _root("INVRG_PLOTS_DIR", "plots")


def out_path(path) -> str:
    """Create the parent directory of ``path`` and return the path as a string."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
