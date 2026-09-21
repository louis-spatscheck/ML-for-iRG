from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

NOTEBOOK = Path(__file__).resolve().parent.parent / "examples" / "inverse_rg_workflow.ipynb"

# very small settings, so that the whole notebook runs in about 15 seconds
FAST_PARAMETERS = """
beta0 = 0.44
L_fine = 16
n_configs = 80
n_val = 20
epochs = 1
batch_size = 16
learning_rate = 5e-4
use_pretrained = False
n_test = 20
n_steps = 3
seed = 2024
"""


def load_notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def test_notebook_is_valid_and_code_cells_compile():
    notebook = load_notebook()
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"cell {index}", "exec")


def test_notebook_has_a_parameters_cell():
    tagged = [c for c in load_notebook().cells if "parameters" in c.metadata.get("tags", [])]
    assert len(tagged) == 1


def test_notebook_runs_with_small_parameters(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("ipykernel")
    pytest.importorskip("invrg.ising")
    nbclient = pytest.importorskip("nbclient")
    from jupyter_client.kernelspec import KernelSpecManager

    if "python3" not in KernelSpecManager().find_kernel_specs():
        pytest.skip("no python3 Jupyter kernel available")

    notebook = load_notebook()
    for cell in notebook.cells:
        if "parameters" in cell.metadata.get("tags", []):
            cell.source = FAST_PARAMETERS.strip()
    client = nbclient.NotebookClient(
        notebook, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}
    )
    client.execute()
