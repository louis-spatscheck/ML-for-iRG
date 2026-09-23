# ML-for-iRG

Machine learning for the inverse renormalization group (RG) of the 2D Ising model.
Convolutional networks are trained to invert the majority-rule RG transformation
and are then applied iteratively to generate large lattices near the critical
point, from which critical exponents are extracted.

Code accompanying the bachelor thesis *Machine Learning for the Inverse
Renormalization Group* (University of Stuttgart, Institute for Computational
Physics, 2024).

> **Work in progress:** the repository is being cleaned up step by step. The
> analysis scripts (`analyze_*.py`, `plot_inverse_crits.py`,
> `extrapolate_with_UNet.py`) still contain the original analysis code and are
> only configurable through the data directories described below.

## Layout

```
src/invrg/            installable package
    ising/            C++ Metropolis simulator (Cython wrapper)
    simulation.py     run and store Ising simulations
    autocorr.py       autocorrelation-aware error estimation
    rg.py             forward RG (majority rule)
    models.py         CNN architectures (shallow, deep, U-Net)
    constants.py      critical values (beta_c, exponents)
    paths.py          data / scratch / plot directories
scripts/              command-line tools and analysis scripts
examples/             worked example notebook of the whole workflow
condor/               HTCondor job files for the cluster
tests/                pytest suite
```

## Installation

```bash
pip install -e .              # core package (+ C++ simulator if a compiler is available)
pip install -e ".[ml]"        # additionally install PyTorch (models, training)
pip install -e ".[ml,test]"   # ... and pytest
pip install -e ".[ml,examples]"  # ... and JupyterLab for the notebook in examples/
```

The C++ simulator needs a C++ compiler. Without one the installation still
succeeds, but `invrg.ising` and `invrg.simulation` are unavailable.

## Workflow

A guided, runnable walk-through of all steps below is in
[`examples/inverse_rg_workflow.ipynb`](examples/inverse_rg_workflow.ipynb).

1. **Simulate** the Ising model (writes `data.gz` into the current directory):
   ```bash
   python -m invrg.simulation --lattice_size 64 --betaJ 0.44
   ```
2. **Forward RG**: apply the majority rule repeatedly (128 -> 64 -> ... -> 4):
   ```bash
   python scripts/forward_rg.py --input data.gz --output-dir forward_renorm/128
   ```
3. **Train** a network (`--model shallow|deep|unet`) to invert the transformation:
   ```bash
   python scripts/train.py --model unet --sample-size 5000 \
       --data-dir DATA --output-dir results/unet
   ```
   `DATA/train_data/` must contain `config.pickle` (a dict with the original
   L = 32 configurations under a key such as `"L=32 configurations"`) and
   `config_renorm.pickle` (the L = 16 configurations obtained with the majority
   rule, in the same order). The defaults reproduce the setup of the original
   training scripts (10 runs x 10 rounds x 100 epochs, Adam with lr 3e-4,
   batch size 1). See `python scripts/train.py --help`.
4. **Extrapolate and analyse** with `scripts/extrapolate_with_UNet.py`,
   `scripts/analyze_RG.py`, `scripts/analyze_inverseRG.py` and
   `scripts/plot_inverse_crits.py`.

**Not included in this repository:** the scripts that prepared the training data
(sampling configurations at uncorrelated intervals, rotation augmentation,
`config_renorm.pickle`) and the test samples (`test_samples/test_data*.pickle`).

### Directories used by the analysis scripts

The analysis scripts read and write below three directories, set by environment
variables (defaults are relative to the working directory):

| variable | default | contents |
|---|---|---|
| `INVRG_DATA_DIR` | `data` | `lattice_size<L>/betaJ<b>/final_result/data_2e6.gz`, `final_models/`, results in `standard_renorm/`, `inverse_renorm/` |
| `INVRG_SCRATCH_DIR` | `scratch` | `forward_renorm/`, `test_samples/`, extrapolation output `complexUNet*/` |
| `INVRG_PLOTS_DIR` | `plots` | figures |

## Cluster

`condor/condor_simulation.sh` and `condor/condor_training.sh` write (and, with the
argument `condor`, submit) HTCondor jobs; see the comments at the top of each
file for the settings. `bash condor/build.sh` installs the package.

## Tests

```bash
pytest
```
