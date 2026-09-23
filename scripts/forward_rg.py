"""Forward RG: apply the majority rule repeatedly to simulated Ising configurations.

Reads the configurations stored in a simulation result (``data.gz`` from
``python -m invrg.simulation``), halves the lattice size step by step
(e.g. 128 -> 64 -> 32 -> 16 -> 8 -> 4) and stores every level as
``config_renorm<L>.pickle``.

Example::

    python scripts/forward_rg.py --input data/lattice_size128/betaJ0.44/data.gz
"""
import argparse
import gzip
import pickle
from pathlib import Path

import numpy as np

from invrg.paths import SCRATCH_DIR
from invrg.rg import majority_rule


def load_configurations(path):
    """Load the stored configurations of a simulation result file."""
    with gzip.open(path, mode="rb") as file:
        state = pickle.load(file)
    configurations = state["configurations"]
    print(len(configurations), "configurations loaded")

    # The simulation preallocates the array; rows that were never filled stay 0
    # (a valid configuration only contains +-1).
    filled = np.count_nonzero(configurations.reshape(len(configurations), -1).any(axis=1))
    if filled < len(configurations):
        print(
            f"WARNING: only {filled} of {len(configurations)} configurations are filled "
            "(the others are all zero). They are renormalized like the rest."
        )
    return configurations


def renormalize(configurations, smallest_size=4):
    """Yield ``(size, configurations)`` for every RG step down to ``smallest_size``."""
    size = configurations.shape[-1]
    while size > smallest_size:
        configurations = majority_rule(configurations, size)
        size //= 2
        yield size, configurations


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", required=True, help="simulation result (data.gz)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="output directory (default: <scratch>/forward_renorm/<L>, see invrg.paths)",
    )
    parser.add_argument("--smallest-size", type=int, default=4, help="stop at this lattice size")
    args = parser.parse_args(argv)

    configurations = load_configurations(args.input)
    size = configurations.shape[-1]
    output_dir = Path(args.output_dir) if args.output_dir else SCRATCH_DIR / "forward_renorm" / str(size)
    output_dir.mkdir(parents=True, exist_ok=True)

    for size, renormalized in renormalize(configurations, args.smallest_size):
        with open(output_dir / f"config_renorm{size}.pickle", "wb") as file:
            pickle.dump(renormalized, file)
        print("Done", size)


if __name__ == "__main__":
    main()
