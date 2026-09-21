"""Forward RG driver: majority-rule renormalization of L=128 Ising configurations.

Reads the simulated configurations, applies the majority rule repeatedly
(128 -> 64 -> 32 -> ... -> 4) and stores every level as a pickle file.
"""
import gzip
import pickle

from invrg.rg import majority_rule


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


if __name__ == "__main__":
    get_configuration()
    renormalization()
