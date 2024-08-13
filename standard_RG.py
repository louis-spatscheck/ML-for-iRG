import numpy as np
import pickle
import gzip
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as spip
import scipy.optimize as spopt
import autocorr
#import analysis_autocorr
import argparse
import scipy




import numpy as np
import time
from tqdm import tqdm

def majority_rule(configs, L):
    """
    Wendet die Mehrheitsregel auf eine Liste von Ising-Konfigurationen an.
    
    Parameter:
        configs (list): Die Liste der Ising-Konfigurationen als 3D-Arrays (num, L, L).
        
    Rückgabewert:
        new_configs (list): Die Liste der renormierten Ising-Konfigurationen als 3D-Arrays (num, L//2, L//2).
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


def renormalization():

    betaJ = 0.44
        


    data= pickle.load(
    open(
        f'/tikhome/lspatscheck/Documents/bsc/simulation_data/lattice_size64/betaJ{betaJ}/configs.pickle',
         'rb'
        )
    )
    config = data['L=128 configurations']

    renorm_config = majority_rule(config,128)
    print("here")
    pickle.dump(
        renorm_config,
        open(
            f'/data/lspatscheck/forward_renorm/128/config_renorm64.pickle',
            mode = 'wb'
        )
    )
    print("Done")
    renorm_config = majority_rule(renorm_config,64)

    pickle.dump(
        renorm_config,
        open(
            f'/data/lspatscheck/forward_renorm/128/config_renorm32.pickle',
            mode = 'wb'
        )
    )
    renorm_config = majority_rule(renorm_config,32)

    pickle.dump(
        renorm_config,
        open(
            f'/data/lspatscheck/forward_renorm/128/config_renorm16.pickle',
            mode = 'wb'
        )
    )
    renorm_config = majority_rule(renorm_config,16)

    pickle.dump(
        renorm_config,
        open(
            f'/data/lspatscheck/forward_renorm/128/config_renorm8.pickle',
            mode = 'wb'
        )
    )

    renorm_config = majority_rule(renorm_config,8)

    pickle.dump(
        renorm_config,
        open(
            f'/data/lspatscheck/forward_renorm/128/config_renorm4.pickle',
            mode = 'wb'
        )
    )


def get_configuration():
    betaJs= [0.4406867935]
    lengths = [128]
    data ={}

    for index, betaJ in enumerate(betaJs):
        for length in lengths:

            simulation_result = pickle.load(
                gzip.open(
                f'/tikhome/lspatscheck/Documents/bsc/simulation_data/lattice_size64/betaJ{betaJ}/data.gz',
                mode = 'rb'
                )
            )

            configuration = (simulation_result['configurations'])
            print(len(configuration))
            key = f'L={length} configurations'
            if key not in data:
                data[key] = {}  # Initialisiere den Schlüssel, wenn er nicht existiert
            data[key] = configuration


        
    pickle.dump(
        data,
        open(
            f'/tikhome/lspatscheck/Documents/bsc/simulation_data/lattice_size64/betaJ{betaJ}/configs.pickle',
            mode = 'wb'
        )
    )
        






get_configuration()
renormalization()


