"""Train a CNN to invert the majority-rule RG transformation of 2D Ising configurations.

The network gets a renormalized configuration (L/2 x L/2) as input and is
trained to reproduce the original configuration (L x L). Spins are shifted by
+1 (values 0 and 2) so that the ReLU activations stay effective.

Expected input data, below ``--data-dir``::

    train_data/config.pickle          dict with the original configurations
                                      (a key ending in "configurations",
                                      e.g. "L=32 configurations")
    train_data/config_renorm.pickle   the renormalized configurations (same order)

Results are written below ``--output-dir``::

    run_<r>/models/model_<k>.pth        checkpoint after training round k
    run_<r>/losses_data/*.pickle        training / validation / magnetization losses
    run_<r>/losses_plot/*.png           loss curves
    run_<r>/pictures/*.png              example reconstructions
    DONE                                written when everything has finished

Each of the ``--repeats`` runs trains a freshly initialised network for
``--rounds`` rounds of ``--epochs`` epochs; the optimizer state is kept across
the rounds of one run.

Example::

    python scripts/train.py --model unet --sample-size 5000 \\
        --data-dir data/final_training --output-dir results/unet
"""
import argparse
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are only written to file
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from invrg.models import DeepCNN, ShallowCNN, UNet

MODELS = {"shallow": ShallowCNN, "deep": DeepCNN, "unet": UNet}
VALIDATION_FRACTION = 0.25


def load_data(data_dir):
    """Load original and renormalized configurations, shifted by +1 to {0, 2}."""
    data_dir = Path(data_dir) / "train_data"
    with open(data_dir / "config.pickle", "rb") as file:
        raw = pickle.load(file)
    keys = [key for key in raw if str(key).endswith("configurations")]
    if len(keys) != 1:
        raise KeyError(f"expected exactly one '...configurations' key in config.pickle, found {keys}")
    original = np.array(raw[keys[0]]) + 1.0
    with open(data_dir / "config_renorm.pickle", "rb") as file:
        renormalized = np.array(pickle.load(file)) + 1.0
    if len(original) != len(renormalized):
        raise ValueError(
            f"{len(original)} original but {len(renormalized)} renormalized configurations"
        )
    return original, renormalized


def split_data(original, renormalized, sample_size, seed):
    """Shuffle both arrays identically; take ``sample_size`` samples for training
    (from the front) and 25% of that number for validation (from the back)."""
    n = len(original)
    val_size = int(sample_size * VALIDATION_FRACTION)
    if val_size < 1:
        raise ValueError("--sample-size must be at least 4")
    if sample_size > n:
        raise ValueError(f"--sample-size {sample_size} exceeds the {n} available configurations")
    if sample_size + val_size > n:
        print(
            f"WARNING: {n} configurations are available but training ({sample_size}) and "
            f"validation ({val_size}) sets together need {sample_size + val_size}: "
            f"they overlap by {sample_size + val_size - n} samples."
        )
    order = np.random.RandomState(seed).permutation(n)
    original, renormalized = original[order], renormalized[order]
    return (
        (original[:sample_size], renormalized[:sample_size]),
        (original[-val_size:], renormalized[-val_size:]),
    )


def make_loader(original, renormalized, batch_size):
    dataset = TensorDataset(
        torch.tensor(original, dtype=torch.float32).unsqueeze(1),  # targets
        torch.tensor(renormalized, dtype=torch.float32).unsqueeze(1),  # inputs
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def save_comparison(model, device, original, renormalized, index, filename):
    """Plot original, network output, input and squared error for one sample."""
    with torch.no_grad():
        targets = torch.tensor(original[index], dtype=torch.float32).reshape(1, 1, *original[index].shape)
        inputs = torch.tensor(renormalized[index], dtype=torch.float32).reshape(1, 1, *renormalized[index].shape)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        origin = targets.cpu().squeeze().numpy()
        out = outputs.cpu().squeeze().numpy()
        inp = inputs.cpu().squeeze().numpy()

        # common color range for the original and the output
        min_val = min(origin.min(), out.min())
        max_val = max(origin.max(), out.max())

        plt.figure(figsize=(20, 8))
        plt.subplot(1, 4, 1)
        plt.imshow(origin, cmap="gray", vmin=min_val, vmax=max_val)
        plt.title("Original")
        plt.subplot(1, 4, 2)
        plt.imshow(out, cmap="gray", vmin=min_val, vmax=max_val)
        plt.title("Output")
        plt.subplot(1, 4, 3)
        plt.imshow(inp, cmap="gray", vmin=min_val, vmax=max_val)
        plt.title("Input")
        plt.subplot(1, 4, 4)
        plt.imshow((out - origin) ** 2, cmap="seismic", interpolation="nearest")
        plt.colorbar(label="Error")
        plt.title("Error for Each Pixel")
        plt.savefig(filename)
        plt.close("all")


def train_round(model, criterion, optimizer, device, train_loader, val_loader, epochs):
    """Train for ``epochs`` epochs. Returns training, validation and magnetization losses."""
    losses = np.empty(epochs, dtype=np.float32)
    val_losses = np.empty(epochs, dtype=np.float32)
    mag_losses = []
    num_batches = len(train_loader)

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        if epoch == 0:
            start = time.time()
        if epoch == 6:
            seconds_per_epoch = (time.time() - start) / 6.0
            print("Estimated Time [h]:", seconds_per_epoch * epochs / 3600)

        for originals, inputs in train_loader:
            inputs, originals = inputs.to(device), originals.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, originals)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            mag_loss = 0.0
            for val_originals, val_inputs in val_loader:
                val_inputs, val_originals = val_inputs.to(device), val_originals.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_originals).item()

                if epoch % 10 == 0:  # additional monitoring of the magnetization
                    mean_originals = torch.mean(val_originals - 1.0)
                    mean_outputs = torch.mean(val_outputs - 1.0)
                    mag_loss += float(torch.abs(mean_originals) - torch.abs(mean_outputs))

        if epoch % 10 == 0:
            mag_losses.append(mag_loss / len(val_loader))
        val_losses[epoch] = val_loss / len(val_loader)
        losses[epoch] = running_loss / num_batches
        if epoch % 2 == 1:
            print(
                f"Epoch {epoch + 1}/{epochs}, Training Loss: {losses[epoch]}, "
                f"Validation Loss: {val_losses[epoch]}, Mean mag: {mag_losses[-1]}"
            )
    return losses, val_losses, mag_losses


def dump(obj, filename):
    with open(filename, "wb") as file:
        pickle.dump(obj, file)


def run_training(args, train_set, val_set, device):
    model_class = MODELS[args.model]
    train_loader = make_loader(*train_set, args.batch_size)
    val_loader = make_loader(*val_set, args.batch_size)
    sample_size = len(train_set[0])
    val_size = len(val_set[0])

    for repeat in range(args.repeats):
        run_dir = Path(args.output_dir) / f"run_{repeat}"
        for sub in ("models", "losses_data", "losses_plot", "pictures"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)

        model = model_class()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model.to(device)
        print(f"Start Training (run {repeat + 1}/{args.repeats}, model: {args.model})")

        for training in range(args.rounds):
            if training > 0:
                checkpoint = run_dir / "models" / f"model_{training - 1}.pth"
                model.load_state_dict(torch.load(checkpoint, map_location=device))
                model.to(device)

            losses, val_losses, mag_losses = train_round(
                model, criterion, optimizer, device, train_loader, val_loader, args.epochs
            )

            plt.figure()
            plt.plot(losses, label="loss")
            plt.plot(val_losses, label="val_loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Loss Function")
            plt.yscale("log")
            plt.legend()
            plt.savefig(run_dir / "losses_plot" / f"training_loss_{training}.png")
            plt.close("all")

            dump(losses, run_dir / "losses_data" / f"losses_{training}.pickle")
            dump(val_losses, run_dir / "losses_data" / f"val_losses_{training}.pickle")
            dump(mag_losses, run_dir / "losses_data" / f"loss_mag_{training}.pickle")
            print("Finished Training")
            torch.save(model.state_dict(), run_dir / "models" / f"model_{training}.pth")

            # example reconstructions: first, middle and last sample
            for name, data, size in (("val", val_set, val_size), ("train", train_set, sample_size)):
                for i in (0, size // 2 - 1, size - 1):
                    save_comparison(
                        model, device, *data, i, run_dir / "pictures" / f"pictures_{name}_{training}_{i}.png"
                    )
        print("Finished Training")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", choices=sorted(MODELS), required=True, help="network architecture")
    parser.add_argument("--data-dir", required=True, help="directory containing train_data/")
    parser.add_argument("--output-dir", required=True, help="directory for checkpoints, losses and plots")
    parser.add_argument("--sample-size", type=int, required=True, help="number of training configurations")
    parser.add_argument("--repeats", type=int, default=10, help="independent runs (fresh network each)")
    parser.add_argument("--rounds", type=int, default=10, help="training rounds per run (one checkpoint each)")
    parser.add_argument("--epochs", type=int, default=100, help="epochs per round")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=50, help="seed of the train/validation shuffling")
    parser.add_argument("--torch-seed", type=int, default=None, help="seed for weight initialisation (default: random)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original, renormalized = load_data(args.data_dir)
    train_set, val_set = split_data(original, renormalized, args.sample_size, args.seed)
    run_training(args, train_set, val_set, device)

    Path(args.output_dir, "DONE").write_text("finished\n")


if __name__ == "__main__":
    main()
