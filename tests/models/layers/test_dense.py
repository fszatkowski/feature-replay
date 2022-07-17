import torch

from models.layers.dense import DenseLayer


def test_dense_layer() -> None:
    layer = DenseLayer(10, 20, 0.5)
    x = torch.rand((4, 10))

    y = layer(x)

    assert y.shape == (4, 20)
