import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from invrg.rg import majority_rule  # noqa: E402


@pytest.fixture
def train_data(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "train_data").mkdir(parents=True)
    fine = np.random.default_rng(0).choice([-1.0, 1.0], size=(12, 32, 32))
    with open(data_dir / "train_data" / "config.pickle", "wb") as file:
        pickle.dump({"L=32 configurations": fine}, file)
    with open(data_dir / "train_data" / "config_renorm.pickle", "wb") as file:
        pickle.dump(majority_rule(fine, 32), file)
    return data_dir


def test_split_sizes_and_overlap_warning(load_script, train_data, capsys):
    train = load_script("train")
    original, renormalized = train.load_data(train_data)
    assert original.shape == (12, 32, 32) and renormalized.shape == (12, 16, 16)
    assert set(np.unique(original)) == {0.0, 2.0}  # shifted by +1

    (train_x, _), (val_x, _) = train.split_data(original, renormalized, 8, seed=50)
    assert len(train_x) == 8 and len(val_x) == 2
    assert "overlap" not in capsys.readouterr().out

    train.split_data(original, renormalized, 12, seed=50)  # 12 + 3 > 12
    assert "overlap by 3 samples" in capsys.readouterr().out

    with pytest.raises(ValueError):
        train.split_data(original, renormalized, 13, seed=50)


def test_training_run_writes_all_outputs(load_script, train_data, tmp_path):
    train = load_script("train")
    output = tmp_path / "out"
    train.main([
        "--model", "shallow", "--data-dir", str(train_data), "--output-dir", str(output),
        "--sample-size", "8", "--repeats", "2", "--rounds", "2", "--epochs", "3",
    ])
    assert (output / "DONE").exists()
    for repeat in (0, 1):
        run = output / f"run_{repeat}"
        assert (run / "models" / "model_1.pth").exists()
        assert (run / "losses_data" / "val_losses_1.pickle").exists()
        assert (run / "losses_plot" / "training_loss_1.png").exists()
        assert (run / "pictures" / "pictures_val_1_0.png").exists()
        assert (run / "pictures" / "pictures_train_1_7.png").exists()
    state = torch.load(output / "run_0" / "models" / "model_1.pth")
    assert "Tconv.0.weight" in state


def test_torch_seed_makes_training_reproducible(load_script, train_data, tmp_path):
    train = load_script("train")
    states = []
    for name in ("a", "b"):
        output = tmp_path / name
        train.main([
            "--model", "shallow", "--data-dir", str(train_data), "--output-dir", str(output),
            "--sample-size", "8", "--repeats", "1", "--rounds", "1", "--epochs", "2", "--torch-seed", "1",
        ])
        states.append(torch.load(output / "run_0" / "models" / "model_0.pth"))
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
