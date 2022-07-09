import torch

from models.mlp import MLP, HiddenLayer


def test_hidden_layer() -> None:
    layer = HiddenLayer(10, 20, 0.5)
    x = torch.rand((4, 10))

    y = layer(x)

    assert y.shape == (4, 20)


def test_mlp_skip_first_inference() -> None:
    inputs = [torch.rand((4, 100)), torch.rand((4, 50)), torch.rand((4, 20))]
    mlp = MLP(input_size=100, num_classes=2, hidden_sizes=[50, 20])

    for skip_idx, input_vector in enumerate(inputs):
        y = mlp(input_vector, skip_first=skip_idx)
        assert y.shape == (4, 2)


def test_mlp_skip_last_inference() -> None:
    input_vector = torch.rand((8, 50))
    hidden_sizes = [40, 30, 20]
    mlp = MLP(input_size=50, num_classes=5, hidden_sizes=hidden_sizes)

    for skip_idx, hidden_size in enumerate(reversed(hidden_sizes + [5])):
        y = mlp(input_vector, skip_last=skip_idx)
        assert y.shape == (8, hidden_size)


def test_get_features() -> None:
    input_vector = torch.rand((2, 100))
    mlp = MLP(input_size=100, num_classes=5, hidden_sizes=[50, 20, 10])

    y = mlp.get_features(input_vector)
    assert y.shape == (2, 10)
