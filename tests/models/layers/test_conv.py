import torch

from models.layers.conv import ConvLayer


def test_conv_layer_no_pooling() -> None:
    layer = ConvLayer(
        in_channels=3, out_channels=1, kernel_size=3, pooling=False, dropout_ratio=0.5
    )
    x = torch.rand((3, 32, 32))

    y = layer(x)

    assert y.shape == (1, 32, 32)


def test_conv_layer_pooling() -> None:
    layer = ConvLayer(
        in_channels=3, out_channels=1, kernel_size=3, pooling=True, dropout_ratio=0.5
    )
    x = torch.rand((3, 32, 32))

    y = layer(x)

    assert y.shape == (1, 16, 16)


def test_conv_layer_pooling_flatten() -> None:
    layer = ConvLayer(
        in_channels=3,
        out_channels=1,
        kernel_size=3,
        pooling=True,
        dropout_ratio=0.5,
        flatten=True,
    )
    x = torch.rand((3, 32, 32))

    y = layer(x)

    assert y.shape == (1, 256)
