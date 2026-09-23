import importlib
import sys

import pytest


def test_helpful_error_when_the_simulator_is_not_built(monkeypatch):
    """Without a C++ compiler the package installs, but invrg.ising must say why it is missing."""
    monkeypatch.setitem(sys.modules, "invrg.ising.cising", None)  # makes the import fail
    monkeypatch.delitem(sys.modules, "invrg.ising", raising=False)
    with pytest.raises(ModuleNotFoundError, match="C\\+\\+ compiler"):
        importlib.import_module("invrg.ising")
    # the simulator tests use pytest.importorskip, which must turn this into a skip
    with pytest.raises(pytest.skip.Exception):
        pytest.importorskip("invrg.ising")
