"""Standard (forward) renormalization group transformation for 2D Ising configurations."""
import time

import numpy as np
from tqdm import tqdm


def majority_rule(configs, L):
    """Apply the majority rule (block-spin transformation, b = 2) to Ising configurations.

    Every 2x2 block of spins is replaced by the sign of its sum. Blocks with
    a vanishing sum (two up, two down) are assigned +1 or -1 at random.

    Parameters
    ----------
    configs : array-like, shape (num, L, L)
        Ising configurations with spins +-1.
    L : int
        Linear lattice size of the input configurations.

    Returns
    -------
    numpy.ndarray, shape (num, L//2, L//2)
        The renormalized configurations.
    """
    start_time = time.time()
    num_configs = len(configs)
    new_configs = np.empty((num_configs, L//2, L//2))

    # Split the configurations into 2x2 blocks and sum over the blocks
    configs = np.array(configs)

    for n in tqdm(range(num_configs), desc="Processing configurations"):
        config = configs[n]
        subgrids = config.reshape(L//2, 2, L//2, 2).sum(axis=(1, 3))

        # Apply the majority rule and handle zero sums
        spins = np.sign(subgrids)
        zero_spins = (spins == 0)
        spins[zero_spins] = np.random.choice([-1, 1], size=zero_spins.sum())

        new_configs[n] = spins

    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")

    return new_configs


def calc_magnetization(configs):
    magnetization = np.empty(len(configs))
    n = 0
    for config in configs:
        magnetization[n] = np.abs(np.sum(config))
        n += 1

    return magnetization
