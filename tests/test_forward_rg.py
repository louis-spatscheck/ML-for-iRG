import gzip
import pickle

import numpy as np


def test_forward_rg_levels_and_zero_row_warning(load_script, tmp_path, capsys):
    forward_rg = load_script("forward_rg")
    configurations = np.random.default_rng(0).choice([-1, 1], size=(10, 16, 16)).astype(np.int8)
    configurations[6:] = 0  # rows that were never filled by the simulation
    with gzip.open(tmp_path / "data.gz", "wb") as file:
        pickle.dump({"configurations": configurations}, file)

    loaded = forward_rg.load_configurations(tmp_path / "data.gz")
    assert "only 6 of 10" in capsys.readouterr().out

    levels = {size: config.shape for size, config in forward_rg.renormalize(loaded)}
    assert levels == {8: (10, 8, 8), 4: (10, 4, 4)}


def test_forward_rg_command_line(load_script, tmp_path):
    forward_rg = load_script("forward_rg")
    configurations = np.random.default_rng(1).choice([-1, 1], size=(5, 16, 16)).astype(np.int8)
    with gzip.open(tmp_path / "data.gz", "wb") as file:
        pickle.dump({"configurations": configurations}, file)

    forward_rg.main(["--input", str(tmp_path / "data.gz"), "--output-dir", str(tmp_path / "out")])
    for size in (8, 4):
        with open(tmp_path / "out" / f"config_renorm{size}.pickle", "rb") as file:
            assert pickle.load(file).shape == (5, size, size)
