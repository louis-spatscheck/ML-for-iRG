"""C++ Metropolis simulation of the 2D Ising model (compiled Cython wrapper)."""
try:
    from .cising import IsingModel
except ImportError as exc:  # extension not built
    raise ImportError(
        "The compiled Ising simulator (invrg.ising.cising) is not available. "
        "Install a C++ compiler and reinstall with `pip install -e .`."
    ) from exc

__all__ = ["IsingModel"]
