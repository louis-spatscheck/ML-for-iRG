# Examples

## `inverse_rg_workflow.ipynb`

A worked example of the complete workflow, written as an annotated tutorial:

1. generate an ensemble of 2D Ising configurations with the C++ simulator (equilibration, autocorrelation time, decorrelated samples),
2. coarse-grain it with the majority rule (forward RG) to obtain training pairs,
3. train the U-Net (`invrg.models.UNet`) to invert the transformation,
4. apply the learned inverse RG iteratively (16 -> 32 -> 64 -> 128) to generate lattices that are never simulated,
5. compare with direct simulations and extract the finite-size-scaling exponent beta/nu = 1/8.

**Requirements:** the package with PyTorch and a notebook environment,

```bash
pip install -e ".[ml,examples]"
jupyter lab examples/inverse_rg_workflow.ipynb
```

**Runtime:** about 15 minutes on a laptop CPU with the default parameters, much less on a GPU (used automatically).
All parameters are collected in one cell (Sec. III B). The stored outputs come from one run with the default parameters.
Trained weights are cached in `examples/_cache/` (not tracked by git); set `use_pretrained = True` to reuse them.

The notebook is a scaled-down demonstration. The settings used in the thesis are listed in its last section.
