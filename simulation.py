import scipy.constants
import cising
import numpy as np
import os.path
import pickle
import gzip
import tqdm
import multiprocessing
import argparse
import scipy



def initialize_simulation(
        file : str,
        temperature : float,
        length : int
    ) -> tuple[cising.IsingModel , dict]:
    if not os.path.exists(file):
        simulation = initialize_new_simulation(temperature, length)
        state = initialize_new_state(temperature, length)
    if os.path.exists(file):
        simulation, state = read_state(file)

    return simulation, state

def initialize_new_simulation(
        temperature : float,
        length : int
    ) -> cising.IsingModel:
    simulation = cising.IsingModel(1/temperature, length)
    initialize_energy_and_magnetization(simulation)
    return simulation

def initialize_energy_and_magnetization(simulation : cising.IsingModel):
    _ = simulation.energy()
    _ = simulation.magnetization()

def initialize_new_state(
        temperature : float,
        length : int
    ) -> dict:
    state = {
        'temperature' : temperature,
        'length' : length,
        'energies' : [],
        'magnetizations' : [],
        'configuration' : np.empty(shape = (length , length),dtype=np.int8),
        'configurations' : np.empty(shape = (int(1e6),length , length),dtype=np.int8) #dtype geandert
    }
    return state

def read_state(filepath : str) -> tuple[cising.IsingModel , dict]:
    with gzip.open(filepath, mode = 'rb') as file:
        state = pickle.load(file)
    simulation = cising.IsingModel(1/state['temperature'], state['length'])
    set_configuration(simulation, state['configuration'])
    initialize_energy_and_magnetization(simulation)
    return simulation, state

def set_configuration(
        simulation : cising.IsingModel,
        configuration : np.array
    ) -> None:
    length = simulation.as_numpy().shape()[0]
    for row in range(length):
        for column in range(length):
            simulation.set_spin(row, column, configuration[row , column])

def save_state(
        filepath : str,
        state : dict
    ) -> None:
    with gzip.open(filepath, mode = 'wb') as file:
        pickle.dump(state, file)


def run_simulation(
        file : str,
        temperature : float,
        length : int,
        measurements : int,
        steps_per_measurement : int = None
    ) -> None:

    if steps_per_measurement == None:
        steps_per_measurement = length**2
    simulation, state = initialize_simulation(file, temperature, length)
    length = state['length']
    energies = list(state['energies'])
    magnetizations = list(state['magnetizations'])

    k = 0
    for i in range(measurements):
        simulation.try_many_random_flips(steps_per_measurement)
        energies.append(simulation.energy())
        magnetizations.append(simulation.magnetization())
        if i > int(1e6) and i % 2 == 0:
            state['configurations'][k] = simulation.as_numpy().copy()
            k += 1



    state['energies'] = energies
    state['magnetizations'] = magnetizations
    state['configuration'] = simulation.as_numpy()
    save_state(file, state)

if __name__ == '__main__':
    def run_simulation_wrapper(args : tuple[str, float, int, int]) -> None:
        run_simulation(args[0], args[1], args[2], args[3])

    



    parser = argparse.ArgumentParser(description="Simulation Script")
    parser.add_argument("--lattice_size", type=int, help="Lattice Size")
    parser.add_argument("--betaJ", type=float, help="BetaJ Value")


    args = parser.parse_args()


    # how many measurements to make in each simulation
    measurements = int(1.2e6)
    # for which temperatures to simulate
    betaJ = args.betaJ
    # for which lattice sizes to simulate
    length = args.lattice_size

    temperature = 1/betaJ

    data = []

    data.append(
        (
            f'./data.gz',
            temperature,
            length,
            measurements
        )
    )
    
    pool = multiprocessing.Pool(maxtasksperchild=1)
    for _ in tqdm.tqdm(
        pool.imap_unordered(
            run_simulation_wrapper,
            data
            ),
        total=len(data)
    ):
        pass