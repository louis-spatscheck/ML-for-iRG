import pytest

torch = pytest.importorskip("torch")

from invrg.models import DeepCNN, ShallowCNN, UNet  # noqa: E402

# number of parameters, thesis Table A.1
PARAMETERS = [(ShallowCNN, 2049), (DeepCNN, 3221249), (UNet, 8402369)]


@pytest.mark.parametrize("model_class, expected", PARAMETERS)
def test_parameter_count_matches_thesis(model_class, expected):
    assert sum(p.numel() for p in model_class().parameters()) == expected


@pytest.mark.parametrize("model_class, _", PARAMETERS)
def test_output_has_twice_the_lattice_size(model_class, _):
    model = model_class().eval()
    with torch.no_grad():
        output = model(torch.rand(2, 1, 16, 16) * 2)
    assert output.shape == (2, 1, 32, 32)


def test_layer_names_are_stable():
    """Checkpoints of the original training scripts must stay loadable."""
    keys = set(UNet().state_dict())
    for name in ("conv1.0.weight", "Tconv1.0.weight", "Tconv3.1.running_mean", "final_conv.0.bias"):
        assert name in keys
    assert "Tconv.0.weight" in ShallowCNN().state_dict()
