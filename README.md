# ML-for-iRG

Machine learning for the inverse renormalization group (RG) of the 2D Ising model.
Convolutional networks are trained to invert the majority-rule RG transformation
and are then applied iteratively to generate large lattices near the critical
point, from which critical exponents are extracted.

Code accompanying the bachelor thesis *Machine Learning for the Inverse
Renormalization Group* (University of Stuttgart, Institute for Computational
Physics, 2024).

> **Work in progress:** the repository is being restructured. Paths in the
> scripts and cluster job files are still specific to the original cluster
> setup and have to be adapted.

## Layout

```
src/invrg/            installable package
    ising/            C++ Metropolis simulator (Cython wrapper)
    simulation.py     run and store Ising simulations
    autocorr.py       autocorrelation-aware error estimation
    rg.py             forward RG (majority rule)
    models.py         CNN architectures (shallow, deep, U-Net)
scripts/              training, extrapolation and analysis scripts
condor/               HTCondor job files for the cluster
```

## Installation

```bash
pip install -e .          # core package (+ C++ simulator if a compiler is available)
pip install -e ".[ml]"    # additionally install PyTorch (models, training)
```

The C++ simulator needs a C++ compiler. Without one the installation still
succeeds, but `invrg.ising` and `invrg.simulation` are unavailable.

Run a simulation with `python -m invrg.simulation --lattice_size 64 --betaJ 0.44`.
