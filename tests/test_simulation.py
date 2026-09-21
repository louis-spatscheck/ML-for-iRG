import numpy as np
import pytest

pytest.importorskip("invrg.ising", reason="C++ simulator not built")

from invrg import simulation  # noqa: E402
from invrg.ising import IsingModel  # noqa: E402


def energy_of(configuration):
    spins = np.asarray(configuration, dtype=float)
    return -np.sum(spins * np.roll(spins, 1, axis=0)) - np.sum(spins * np.roll(spins, 1, axis=1))


def test_ordered_and_disordered_phase():
    for beta, low, high in ((0.3, 0.0, 0.3), (0.6, 0.9, 1.0)):
        model = IsingModel(beta, 16)
        magnetization = []
        for i in range(3000):
            model.try_many_random_flips(16 * 16)
            if i > 1000:
                magnetization.append(abs(model.magnetization()))
        assert low <= np.mean(magnetization) <= high


def test_energy_matches_configuration():
    model = IsingModel(0.44, 8)
    model.try_many_random_flips(1000)
    assert model.energy() == energy_of(model.as_numpy())


def test_fixed_seed_is_reproducible():
    results = []
    for _ in range(2):
        model = IsingModel(0.44, 8)
        model.try_many_random_flips(1000)
        results.append(model.as_numpy())
    assert np.array_equal(*results)


def test_continue_simulation_from_file(tmp_path):
    file = str(tmp_path / "data.gz")
    simulation.run_simulation(file, 1 / 0.44, 8, 300)

    model, state = simulation.read_state(file)
    assert np.array_equal(model.as_numpy(), state["configuration"])
    assert model.energy() == energy_of(state["configuration"])
    assert model.energy() == state["energies"][-1]

    simulation.run_simulation(file, 1 / 0.44, 8, 200)
    _, state = simulation.read_state(file)
    assert len(state["energies"]) == 500
